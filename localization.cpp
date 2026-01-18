/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

#include <fins/node.hpp>
#include "Scancontext.hpp"

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct KeyFrame {
  int id;
  std::string timestamp_str;
  Eigen::Matrix4d pose;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
};

struct Config {
  std::string map_dir;
  std::string child_frame_id = "livox_frame";
  double sc_dist_thres = 0.20;
  double icp_max_dist = 1.0;
  double icp_score_thres = 0.2;
  bool use_icp = true;
};

class ScanContextNode : public fins::Node {
public:
  void define() override {
    set_name("ScanContextNode");
    set_description("Robust Global Localization using SC + Centroid-ICP");
    set_category("Navigation");

    register_input<0, pcl::PointCloud<pcl::PointXYZI>::Ptr>("cloud", &ScanContextNode::on_cloud);

    register_output<0, geometry_msgs::msg::TransformStamped>("tf");
    register_output<1, nav_msgs::msg::Path>("path");
    register_output<2, pcl::PointCloud<pcl::PointXYZRGB>::Ptr>("global_map");
    register_output<3, pcl::PointCloud<pcl::PointXYZI>::Ptr>("aligned_cloud");

    register_parameter<std::string>("map_dir", &ScanContextNode::update_map_dir, "/path/to/map");
    register_parameter<double>("dist_threshold", &ScanContextNode::update_dist_thres, 0.20);
    register_parameter<bool>("use_icp", &ScanContextNode::update_use_icp, true);
  }

  void initialize() override {
    std::lock_guard<std::mutex> lock(mutex_);
    sc_manager_ = std::make_unique<SCManager>();

    global_path_.header.frame_id = "map";
  }

  ~ScanContextNode() { stop_publish_thread(); }

  void run() override {
    if (is_running_)
      return;
    is_running_ = true;
    pub_thread_ = std::thread([this]() {
      while (is_running_) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (required<2>() && global_map_cloud_viz_ && !global_map_cloud_viz_->empty()) {
            send<2>(global_map_cloud_viz_, fins::now());
          }
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
      }
    });
  }

  void pause() override { stop_publish_thread(); }

  void reset() override {
    std::lock_guard<std::mutex> lock(mutex_);
    global_path_.poses.clear();
    last_localized_pose_ = Eigen::Vector3d::Zero();
  }

  void update_map_dir(const std::string &v) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config_.map_dir == v)
      return;
    config_.map_dir = v;
    logger->info("Map dir changed. Reloading from: {}", v);
    load_map_database(v);
  }
  void update_dist_thres(const double &v) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.sc_dist_thres = v;
  }
  void update_use_icp(const bool &v) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.use_icp = v;
  }

  void on_cloud(const fins::Msg<pcl::PointCloud<pcl::PointXYZI>::Ptr> &msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto current_cloud = *msg;

    if (!current_cloud || current_cloud->empty() || keyframes_.empty())
      return;

    // Step 1: Scan Context Coarse Matching (Global Retrieval)
    // Returns: <Index, Yaw_Rotation_Rad>
    auto [match_idx, yaw_diff] = match_scancontext(current_cloud);

    if (match_idx == -1) {
      logger->error("ScanContext failed to find a match.");
      return;
    }

    // Step 2: Pose Refinement (ICP with Centroid Alignment)
    Eigen::Matrix4d final_pose = Eigen::Matrix4d::Identity();
    bool localization_success = false;

    Eigen::Matrix4d T_world_keyframe = keyframes_[match_idx].pose;

    if (config_.use_icp) {
      Eigen::Matrix4d T_keyframe_current;
      double score = 0.0;

      bool icp_ok = refine_with_icp(current_cloud, // Source
                                    keyframes_[match_idx].cloud, // Target
                                    yaw_diff, // Initial Yaw Guess
                                    T_keyframe_current, // Output Transform
                                    score // Output Score
      );

      if (icp_ok) {
        final_pose = T_world_keyframe * T_keyframe_current;
        localization_success = true;
      }
    } else {
      Eigen::AngleAxisd rotation_sc(yaw_diff, Eigen::Vector3d::UnitZ());
      Eigen::Matrix4d T_sc_local = Eigen::Matrix4d::Identity();
      T_sc_local.block<3, 3>(0, 0) = rotation_sc.toRotationMatrix();

      final_pose = T_world_keyframe * T_sc_local;
      localization_success = true;
    }

    if (localization_success) {
      Eigen::Vector3d current_pos = final_pose.block<3, 1>(0, 3);
      if (last_localized_pose_.norm() > 0.1) {
        if ((current_pos - last_localized_pose_).norm() > 5.0) {
          logger->warn("Large pose jump detected! Ignored.");
          return;
        }
      }
      last_localized_pose_ = current_pos;

      publish_results(final_pose, current_cloud, msg.event_time);

      logger->info("Localized: Frame[{}] Pose[{:.2f}, {:.2f}, {:.2f}]", match_idx, current_pos.x(), current_pos.y(),
                   current_pos.z());
    }
  }

private:
  std::mutex mutex_;
  Config config_;

  std::unique_ptr<SCManager> sc_manager_;

  std::vector<KeyFrame> keyframes_;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_map_cloud_viz_;
  nav_msgs::msg::Path global_path_;
  Eigen::Vector3d last_localized_pose_ = Eigen::Vector3d::Zero();

  std::atomic<bool> is_running_{false};
  std::thread pub_thread_;

  bool load_map_database(const std::string &dir_path) {
    namespace fs = std::filesystem;

    keyframes_.clear();
    sc_manager_ = std::make_unique<SCManager>();
    global_map_cloud_viz_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

    if (!fs::exists(dir_path))
      return false;

    std::vector<std::string> pcd_files;
    for (const auto &entry: fs::directory_iterator(dir_path)) {
      if (entry.path().extension() == ".pcd") {
        pcd_files.push_back(entry.path().stem().string());
      }
    }
    std::sort(pcd_files.begin(), pcd_files.end());

    if (pcd_files.empty())
      return false;

    logger->info("Loading {} frames from {}...", pcd_files.size(), dir_path);

    for (size_t i = 0; i < pcd_files.size(); ++i) {
      std::string stem = pcd_files[i];
      std::string pcd_path = dir_path + "/" + stem + ".pcd";
      std::string odom_path = dir_path + "/" + stem + ".odom";

      KeyFrame kf;
      kf.id = i;
      kf.timestamp_str = stem;
      kf.cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);

      if (pcl::io::loadPCDFile(pcd_path, *kf.cloud) == -1)
        continue;

      kf.pose = Eigen::Matrix4d::Identity();
      if (fs::exists(odom_path)) {
        std::ifstream file(odom_path);
        for (int r = 0; r < 4; ++r)
          for (int c = 0; c < 4; ++c)
            file >> kf.pose(r, c);
      }

      sc_manager_->makeAndSaveScancontextAndKeys(*kf.cloud);

      pcl::PointCloud<pcl::PointXYZRGB> viz_cloud;
      pcl::copyPointCloud(*kf.cloud, viz_cloud);

      uint8_t r = (i * 30) % 255;
      uint8_t g = (i * 100) % 255;
      uint8_t b = 200;
      for (auto &p: viz_cloud.points) {
        p.r = r;
        p.g = g;
        p.b = b;
      }

      pcl::transformPointCloud(viz_cloud, viz_cloud, kf.pose.cast<float>());
      *global_map_cloud_viz_ += viz_cloud;

      keyframes_.push_back(kf);
    }

    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    vg.setLeafSize(0.02f, 0.02f, 0.02f);
    vg.setInputCloud(global_map_cloud_viz_);
    vg.filter(*global_map_cloud_viz_);
    global_map_cloud_viz_->header.frame_id = "map";

    if (!sc_manager_->polarcontext_invkeys_mat_.empty()) {
      sc_manager_->polarcontext_invkeys_to_search_ = sc_manager_->polarcontext_invkeys_mat_;
      sc_manager_->polarcontext_tree_ =
          std::make_unique<InvKeyTree>(sc_manager_->PC_NUM_RING, sc_manager_->polarcontext_invkeys_to_search_, 10);
    }

    logger->info("Map loaded. {} keyframes.", keyframes_.size());
    return true;
  }

  std::pair<int, float> match_scancontext(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
    Eigen::MatrixXd sc = sc_manager_->makeScancontext(*cloud);
    Eigen::MatrixXd ringkey = sc_manager_->makeRingkeyFromScancontext(sc);
    std::vector<float> query_key = eig2stdvec(ringkey);

    int candidates = std::min((int) keyframes_.size(), (int) sc_manager_->NUM_CANDIDATES_FROM_TREE);
    std::vector<size_t> indices(candidates);
    std::vector<float> dists(candidates);
    nanoflann::KNNResultSet<float> result(candidates);
    result.init(&indices[0], &dists[0]);
    sc_manager_->polarcontext_tree_->index->findNeighbors(result, &query_key[0], nanoflann::SearchParams(10));

    double min_dist = 1e9;
    int best_idx = -1;
    int best_align = 0;

    for (int i = 0; i < candidates; ++i) {
      int map_id = indices[i];
      auto [dist, align] = sc_manager_->distanceBtnScanContext(sc, sc_manager_->polarcontexts_[map_id]);

      if (dist < min_dist) {
        min_dist = dist;
        best_idx = map_id;
        best_align = align;
      }
    }

    if (best_idx != -1 && min_dist < config_.sc_dist_thres) {
      float yaw_rad = deg2rad(best_align * sc_manager_->PC_UNIT_SECTORANGLE);
      return {best_idx, yaw_rad};
    }
    return {-1, 0.0f};
  }

  bool refine_with_icp(pcl::PointCloud<pcl::PointXYZI>::Ptr src_cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr tgt_cloud,
                       float yaw_guess, Eigen::Matrix4d &result_pose, double &final_score) {
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::PointNormal> ne;
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
    ne.setSearchMethod(tree);

    pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;
    pcl::PointCloud<pcl::PointNormal> unused_res;

    Eigen::Matrix4f current_guess = Eigen::Matrix4f::Identity();
    Eigen::AngleAxisf init_rot(yaw_guess, Eigen::Vector3f::UnitZ());
    current_guess.block<3, 3>(0, 0) = init_rot.toRotationMatrix();

    // =======================================================
    // Stage 1: Coarse
    // Voxel: 0.5m | Radius: 2.0m | Dist: 2.5m
    // =======================================================
    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr src_s1(new pcl::PointCloud<pcl::PointXYZI>);
      pcl::PointCloud<pcl::PointXYZI>::Ptr tgt_s1(new pcl::PointCloud<pcl::PointXYZI>);

      vg.setLeafSize(0.5f, 0.5f, 0.5f);
      vg.setInputCloud(src_cloud);
      vg.filter(*src_s1);
      vg.setInputCloud(tgt_cloud);
      vg.filter(*tgt_s1);

      if (src_s1->size() < 10 || tgt_s1->size() < 10)
        return false;

      pcl::PointCloud<pcl::PointNormal>::Ptr src_n(new pcl::PointCloud<pcl::PointNormal>);
      pcl::PointCloud<pcl::PointNormal>::Ptr tgt_n(new pcl::PointCloud<pcl::PointNormal>);

      ne.setRadiusSearch(2.0);
      ne.setInputCloud(src_s1);
      ne.compute(*src_n);
      pcl::copyPointCloud(*src_s1, *src_n);
      ne.setInputCloud(tgt_s1);
      ne.compute(*tgt_n);
      pcl::copyPointCloud(*tgt_s1, *tgt_n);

      icp.setInputSource(src_n);
      icp.setInputTarget(tgt_n);
      icp.setMaxCorrespondenceDistance(2.5);
      icp.setMaximumIterations(20);
      icp.setTransformationEpsilon(1e-4);

      icp.align(unused_res, current_guess);
      if (icp.hasConverged())
        current_guess = icp.getFinalTransformation();
    }

    // =======================================================
    // Stage 2: Medium
    // Voxel: 0.2m | Radius: 0.8m | Dist: 0.5m
    // =======================================================
    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr src_s2(new pcl::PointCloud<pcl::PointXYZI>);
      pcl::PointCloud<pcl::PointXYZI>::Ptr tgt_s2(new pcl::PointCloud<pcl::PointXYZI>);

      vg.setLeafSize(0.2f, 0.2f, 0.2f);
      vg.setInputCloud(src_cloud);
      vg.filter(*src_s2);
      vg.setInputCloud(tgt_cloud);
      vg.filter(*tgt_s2);

      pcl::PointCloud<pcl::PointNormal>::Ptr src_n(new pcl::PointCloud<pcl::PointNormal>);
      pcl::PointCloud<pcl::PointNormal>::Ptr tgt_n(new pcl::PointCloud<pcl::PointNormal>);

      ne.setRadiusSearch(0.8);
      ne.setInputCloud(src_s2);
      ne.compute(*src_n);
      pcl::copyPointCloud(*src_s2, *src_n);
      ne.setInputCloud(tgt_s2);
      ne.compute(*tgt_n);
      pcl::copyPointCloud(*tgt_s2, *tgt_n);

      icp.setInputSource(src_n);
      icp.setInputTarget(tgt_n);
      icp.setMaxCorrespondenceDistance(0.5);
      icp.setMaximumIterations(30);
      icp.setTransformationEpsilon(1e-6);

      icp.align(unused_res, current_guess);
      if (icp.hasConverged())
        current_guess = icp.getFinalTransformation();
    }

    // =======================================================
    // Stage 3: Ultra-Fine
    // Voxel: 0.05m | Radius: 0.3m | Dist: 0.1m
    // =======================================================
    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr src_s3(new pcl::PointCloud<pcl::PointXYZI>);
      pcl::PointCloud<pcl::PointXYZI>::Ptr tgt_s3(new pcl::PointCloud<pcl::PointXYZI>);

      vg.setLeafSize(0.05f, 0.05f, 0.05f);
      vg.setInputCloud(src_cloud);
      vg.filter(*src_s3);
      vg.setInputCloud(tgt_cloud);
      vg.filter(*tgt_s3);

      pcl::PointCloud<pcl::PointNormal>::Ptr src_n(new pcl::PointCloud<pcl::PointNormal>);
      pcl::PointCloud<pcl::PointNormal>::Ptr tgt_n(new pcl::PointCloud<pcl::PointNormal>);

      ne.setRadiusSearch(0.2);
      ne.setInputCloud(src_s3);
      ne.compute(*src_n);
      pcl::copyPointCloud(*src_s3, *src_n);
      ne.setInputCloud(tgt_s3);
      ne.compute(*tgt_n);
      pcl::copyPointCloud(*tgt_s3, *tgt_n);

      icp.setInputSource(src_n);
      icp.setInputTarget(tgt_n);

      icp.setMaxCorrespondenceDistance(0.1);

      icp.setMaximumIterations(80);
      icp.setTransformationEpsilon(1e-9);
      icp.setEuclideanFitnessEpsilon(1e-9);

      icp.align(unused_res, current_guess);

      final_score = icp.getFitnessScore();
      result_pose = icp.getFinalTransformation().cast<double>();

      if (icp.hasConverged()) {
        Eigen::Matrix3d rot = result_pose.block<3, 3>(0, 0);
        Eigen::Vector3d euler = rot.eulerAngles(2, 1, 0);
        logger->info("[ICP Ultra] Trans: {:.3f}, {:.3f} | Score: {:.5f}", result_pose(0, 3), result_pose(1, 3),
                     final_score);
        return true;
      }
    }

    if (final_score < 0.5)
      return true;

    return false;
  }


  void publish_results(const Eigen::Matrix4d &pose, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
                       fins::time_stamp event_time) {
    Eigen::Vector3d t = pose.block<3, 1>(0, 3);
    Eigen::Quaterniond q(pose.block<3, 3>(0, 0));

    // 1. TF
    if (required<0>()) {
      geometry_msgs::msg::TransformStamped tf_msg;
      tf_msg.header.stamp = pcl_conversions::fromPCL(cloud->header.stamp);
      tf_msg.header.frame_id = "map";
      tf_msg.child_frame_id = config_.child_frame_id;
      tf_msg.transform.translation.x = t.x();
      tf_msg.transform.translation.y = t.y();
      tf_msg.transform.translation.z = t.z();
      tf_msg.transform.rotation.x = q.x();
      tf_msg.transform.rotation.y = q.y();
      tf_msg.transform.rotation.z = q.z();
      tf_msg.transform.rotation.w = q.w();
      send<0>(tf_msg, event_time);
    }

    // 2. Path
    if (required<1>()) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header.stamp = pcl_conversions::fromPCL(cloud->header.stamp);
      ps.header.frame_id = "map";
      ps.pose.position.x = t.x();
      ps.pose.position.y = t.y();
      ps.pose.position.z = t.z();
      ps.pose.orientation.x = q.x();
      ps.pose.orientation.y = q.y();
      ps.pose.orientation.z = q.z();
      ps.pose.orientation.w = q.w();
      global_path_.poses.push_back(ps);
      if (global_path_.poses.size() > 5000)
        global_path_.poses.erase(global_path_.poses.begin());

      global_path_.header.stamp = ps.header.stamp;
      send<1>(global_path_, event_time);
    }

    // 3. Aligned Cloud
    if (required<3>()) {
      pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>);
      pcl::transformPointCloud(*cloud, *aligned, pose);
      aligned->header.frame_id = "map";
      aligned->header.stamp = cloud->header.stamp;
      send<3>(aligned, event_time);
    }
  }

  void stop_publish_thread() {
    is_running_ = false;
    if (pub_thread_.joinable())
      pub_thread_.join();
  }
};

EXPORT_NODE(ScanContextNode)
DEFINE_PLUGIN_ENTRY()