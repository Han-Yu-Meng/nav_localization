/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

#include "Scancontext.hpp"
#include <fins/node.hpp>

// ROS Messages
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// PCL
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>

// Small GICP
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_registration.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/util/downsampling_omp.hpp>

// Eigen & Std
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct KeyFrame {
  int id;
  Eigen::Matrix4d pose;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
};

struct Config {
  std::string map_dir;
  double sc_dist_thres = 0.50;

  // GICP Parameters
  int num_threads = 4;
  int num_neighbors = 20;
  double global_leaf_size = 0.25;
  double registered_leaf_size = 0.5;
  double max_dist_sq = 1.0;
  double accumulation_time = 5.0;
};

enum class SystemState {
  IDLE,         // Waiting for map to load
  ACCUMULATING, // Accumulating point cloud for 5 seconds
  LOCALIZING,   // Running ScanContext + Initial Alignment
  TRACKING      // Tracking pose using Small-GICP
};

class RelocalizationNode : public fins::Node {
public:
  void define() override {
    set_name("RelocalizationNode");
    set_description("Relocalization: 5s Accumulation -> SC -> Small GICP");
    set_category("Navigation>Localization");

    // Input 0: Cloud Registered (from Fast-LIO, in ODOM/Camera_Init frame)
    register_input<pcl::PointCloud<pcl::PointXYZI>::Ptr>(
        "cloud_registered", &RelocalizationNode::on_cloud);
    // Input 1: Odometry (from Fast-LIO, Odom -> Body TF)
    register_input<nav_msgs::msg::Odometry>("fastlio_odom",
                                            &RelocalizationNode::on_odom);

    // Outputs
    register_output<geometry_msgs::msg::TransformStamped>("map_to_odom_tf");
    register_output<nav_msgs::msg::Path>("corrected_path");
    register_output<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>("global_map_viz");
    register_output<pcl::PointCloud<pcl::PointXYZI>::Ptr>(
        "debug_accumulated_cloud");

    // Parameters
    register_parameter<std::string>(
        "map_dir", &RelocalizationNode::update_map_dir, "/path/to/map");
    register_parameter<double>("sc_dist_threshold",
                               &RelocalizationNode::update_dist_thres, 0.40);
  }

  void initialize() override {
    std::lock_guard<std::mutex> lock(mutex_);

    // Init Modules
    sc_manager_ = std::make_unique<SCManager>();

    global_map_ = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    target_covariance_ =
        std::make_shared<pcl::PointCloud<pcl::PointCovariance>>();
    accumulated_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());

    // Init State
    state_ = SystemState::IDLE;
    T_map_odom_ = Eigen::Isometry3d::Identity();
    T_odom_body_ = Eigen::Isometry3d::Identity();

    corrected_path_.header.frame_id = "map";
  }

  ~RelocalizationNode() { stop_publish_thread(); }

  void run() override {
    if (is_running_)
      return;
    is_running_ = true;

    // Thread for publishing heavy visualization (Map)
    pub_thread_ = std::thread([this]() {
      while (is_running_) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (required("global_map_viz") && global_map_viz_ &&
              !global_map_viz_->empty()) {
            send("global_map_viz", global_map_viz_, fins::now());
          }
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
      }
    });
  }

  void pause() override { stop_publish_thread(); }

  void reset() override {
    std::lock_guard<std::mutex> lock(mutex_);
    corrected_path_.poses.clear();
    T_map_odom_ = Eigen::Isometry3d::Identity();
    T_odom_body_ = Eigen::Isometry3d::Identity();
    state_ = SystemState::IDLE;
    accumulated_cloud_->clear();
    accumulation_started_ = false;
  }

  void update_map_dir(const std::string &v) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config_.map_dir == v)
      return;
    config_.map_dir = v;
    load_map_database(v);
  }

  void update_dist_thres(const double &v) { config_.sc_dist_thres = v; }

  void on_odom(const fins::Msg<nav_msgs::msg::Odometry> &msg) {
    std::lock_guard<std::mutex> lock(mutex_);

    Eigen::Vector3d p(msg->pose.pose.position.x, msg->pose.pose.position.y,
                      msg->pose.pose.position.z);
    Eigen::Quaterniond q(
        msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    T_odom_body_ = Eigen::Isometry3d::Identity();
    T_odom_body_.translate(p);
    T_odom_body_.rotate(q);

    if (state_ == SystemState::TRACKING || state_ == SystemState::LOCALIZING) {
      publish_tf(msg.acq_time);
    }
  }

  void on_cloud(const fins::Msg<pcl::PointCloud<pcl::PointXYZI>::Ptr> &msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto input_cloud = *msg;

    if (!input_cloud || input_cloud->empty())
      return;

    double current_timestamp = fins::to_seconds(msg.acq_time);

    if (state_ == SystemState::ACCUMULATING) {
      if (!accumulation_started_) {
        start_accumulation_time_ = current_timestamp;
        accumulation_started_ = true;
        logger->info("Started 5s Accumulation for Relocalization...");
      }

      double elapsed = current_timestamp - start_accumulation_time_;

      if (elapsed < config_.accumulation_time) {
        static pcl::VoxelGrid<pcl::PointXYZI> vg;
        vg.setLeafSize(0.1f, 0.1f, 0.1f);
        vg.setInputCloud(input_cloud);
        pcl::PointCloud<pcl::PointXYZI> temp;
        vg.filter(temp);
        *accumulated_cloud_ += temp;
      } else {
        logger->info(
            "Accumulation done ({:.1f}s). Points: {}. Starting ScanContext...",
            elapsed, accumulated_cloud_->size());
        state_ = SystemState::LOCALIZING;
        perform_global_localization(msg.acq_time);
      }
    } else if (state_ == SystemState::TRACKING) {
      perform_tracking(input_cloud, msg.acq_time);
    }
  }

private:
  std::mutex mutex_;
  Config config_;
  SystemState state_;
  std::atomic<bool> is_running_{false};
  std::thread pub_thread_;

  // Accumulation Vars
  bool accumulation_started_ = false;
  double start_accumulation_time_ = 0.0;
  pcl::PointCloud<pcl::PointXYZI>::Ptr accumulated_cloud_;

  // SC & Map
  std::unique_ptr<SCManager> sc_manager_;
  std::vector<KeyFrame> keyframes_;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_map_viz_;

  // GICP Data
  pcl::PointCloud<pcl::PointXYZI>::Ptr global_map_; // Target
  pcl::PointCloud<pcl::PointCovariance>::Ptr target_covariance_;
  std::shared_ptr<small_gicp::KdTree<pcl::PointCloud<pcl::PointCovariance>>>
      target_tree_;

  // Transforms
  Eigen::Isometry3d T_map_odom_; // Correction Transform
  fins::AcqTime transform_time_;
  Eigen::Isometry3d T_odom_body_; // Fast-LIO Input

  // Visualization
  nav_msgs::msg::Path corrected_path_;

  void stop_publish_thread() {
    if (is_running_) {
      is_running_ = false;
      if (pub_thread_.joinable())
        pub_thread_.join();
    }
  }

  // =================================================================================
  // 1. Map Loading & Preprocessing
  // =================================================================================
  bool load_map_database(const std::string &dir_path) {
    namespace fs = std::filesystem;
    keyframes_.clear();
    sc_manager_ = std::make_unique<SCManager>();
    global_map_viz_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    global_map_->clear();

    if (!fs::exists(dir_path))
      return false;

    std::vector<std::string> pcd_files;
    for (const auto &entry : fs::directory_iterator(dir_path)) {
      if (entry.path().extension() == ".pcd")
        pcd_files.push_back(entry.path().stem().string());
    }
    std::sort(pcd_files.begin(), pcd_files.end());

    if (pcd_files.empty())
      return false;

    logger->info("Loading Map Database...");

    for (size_t i = 0; i < pcd_files.size(); ++i) {
      std::string pcd_path = dir_path + "/" + pcd_files[i] + ".pcd";
      std::string odom_path = dir_path + "/" + pcd_files[i] + ".odom";

      KeyFrame kf;
      kf.id = i;
      kf.cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
      if (pcl::io::loadPCDFile(pcd_path, *kf.cloud) == -1)
        continue;

      // Load pose
      kf.pose = Eigen::Matrix4d::Identity();
      if (fs::exists(odom_path)) {
        std::ifstream file(odom_path);
        for (int r = 0; r < 4; ++r)
          for (int c = 0; c < 4; ++c)
            file >> kf.pose(r, c);
      }

      // SC Database Construction
      sc_manager_->makeAndSaveScancontextAndKeys(*kf.cloud);

      // Global Map Construction (Transform to Map Frame)
      pcl::PointCloud<pcl::PointXYZI> cloud_in_map;
      pcl::transformPointCloud(*kf.cloud, cloud_in_map, kf.pose.cast<float>());
      *global_map_ += cloud_in_map;

      // Visualization
      pcl::PointCloud<pcl::PointXYZRGB> viz;
      pcl::copyPointCloud(cloud_in_map, viz);
      *global_map_viz_ += viz;

      keyframes_.push_back(kf);
    }

    // Build SC Tree
    if (!sc_manager_->polarcontext_invkeys_mat_.empty()) {
      sc_manager_->polarcontext_invkeys_to_search_ =
          sc_manager_->polarcontext_invkeys_mat_;
      sc_manager_->polarcontext_tree_ = std::make_unique<InvKeyTree>(
          sc_manager_->PC_NUM_RING,
          sc_manager_->polarcontext_invkeys_to_search_, 10);
    }

    // Preprocess GICP Target
    preprocess_gicp_target();

    logger->info("Map loaded. Ready to accumulate clouds.");
    state_ = SystemState::ACCUMULATING; // Set state to accumulation
    return true;
  }

  void preprocess_gicp_target() {
    logger->info("Preprocessing Global Map for GICP...");

    // Downsample Global Map
    auto downsampled_map =
        small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>,
                                           pcl::PointCloud<pcl::PointXYZI>>(
            *global_map_, config_.global_leaf_size);

    // Create Covariances
    target_covariance_ =
        std::make_shared<pcl::PointCloud<pcl::PointCovariance>>();

    target_covariance_->points.resize(downsampled_map->points.size());
    target_covariance_->width = downsampled_map->width;
    target_covariance_->height = downsampled_map->height;
    target_covariance_->is_dense = downsampled_map->is_dense;

    for (size_t i = 0; i < downsampled_map->points.size(); ++i) {
      const auto &src = downsampled_map->points[i];
      auto &dst = target_covariance_->points[i];
      dst.x = src.x;
      dst.y = src.y;
      dst.z = src.z;
    }

    small_gicp::estimate_covariances_omp(
        *target_covariance_, config_.num_neighbors, config_.num_threads);

    // Build KDTree
    target_tree_ = std::make_shared<
        small_gicp::KdTree<pcl::PointCloud<pcl::PointCovariance>>>(
        target_covariance_, small_gicp::KdTreeBuilderOMP(config_.num_threads));

    logger->info("GICP Target Ready. Size: {}", target_covariance_->size());
  }

  // =================================================================================
  // 2. Global Localization (ScanContext)
  // =================================================================================
  void perform_global_localization(fins::AcqTime timestamp) {
    if (accumulated_cloud_->empty()) {
      logger->error("Accumulation failed: Cloud empty. Resetting.");
      accumulated_cloud_->clear();
      accumulation_started_ = false;
      state_ = SystemState::ACCUMULATING;
      return;
    }

    Eigen::MatrixXd sc = sc_manager_->makeScancontext(*accumulated_cloud_);
    Eigen::MatrixXd ringkey = sc_manager_->makeRingkeyFromScancontext(sc);
    std::vector<float> query_key = eig2stdvec(ringkey);

    int candidates = std::min((int)keyframes_.size(),
                              (int)sc_manager_->NUM_CANDIDATES_FROM_TREE);
    std::vector<size_t> indices(candidates);
    std::vector<float> dists(candidates);
    nanoflann::KNNResultSet<float> result(candidates);
    result.init(&indices[0], &dists[0]);
    sc_manager_->polarcontext_tree_->index->findNeighbors(
        result, &query_key[0], nanoflann::SearchParams(10));

    double min_dist = 1e9;
    int best_idx = -1;
    int best_align = 0;

    for (int i = 0; i < candidates; ++i) {
      int map_id = indices[i];
      auto [dist, align] = sc_manager_->distanceBtnScanContext(
          sc, sc_manager_->polarcontexts_[map_id]);

      if (dist < min_dist) {
        min_dist = dist;
        best_idx = map_id;
        best_align = align;
      }
    }

    logger->info("SC Result: Frame {} | Dist {:.4f}", best_idx, min_dist);

    if (best_idx != -1 && min_dist < config_.sc_dist_thres) {
      float yaw_diff = deg2rad(best_align * sc_manager_->PC_UNIT_SECTORANGLE);

      Eigen::Matrix4d T_map_kf = keyframes_[best_idx].pose;
      Eigen::AngleAxisd rot_yaw(yaw_diff, Eigen::Vector3d::UnitZ());
      Eigen::Matrix4d T_kf_accum = Eigen::Matrix4d::Identity();
      T_kf_accum.block<3, 3>(0, 0) = rot_yaw.toRotationMatrix();
      Eigen::Matrix4d T_map_accum = T_map_kf * T_kf_accum;
      T_map_odom_ = Eigen::Isometry3d(T_map_accum);

      logger->info("Localization Success! Initializing Tracking.");
      state_ = SystemState::TRACKING;

      // Debug: Visualize the accumulated cloud in Map Frame
      if (required("accumulated_cloud")) {
        accumulated_cloud_->header.frame_id = "map";
        send("accumulated_cloud", accumulated_cloud_, timestamp);
      }
    } else {
      logger->warn(
          "Localization Failed (Dist > Threshold). Retrying accumulation...");
      accumulated_cloud_->clear();
      accumulation_started_ = false;
      state_ = SystemState::ACCUMULATING;
    }
  }

  float deg2rad(float deg) { return deg * M_PI / 180.0f; }

  // =================================================================================
  // 3. Tracking (Small GICP)
  // =================================================================================
  void perform_tracking(pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_odom,
                        fins::AcqTime time) {
    static int frame_count = 0;
    if (frame_count++ % 5 != 0)
      return;

    // 1. Prepare Source Cloud (Current Scan in Odom Frame)
    // Downsample
    auto source_down = small_gicp::voxelgrid_sampling_omp<
        pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointCovariance>>(
        *input_cloud_odom, config_.registered_leaf_size);

    // Covariance
    small_gicp::estimate_covariances_omp(*source_down, config_.num_neighbors,
                                         config_.num_threads);

    // Tree
    auto source_tree = std::make_shared<
        small_gicp::KdTree<pcl::PointCloud<pcl::PointCovariance>>>(
        source_down, small_gicp::KdTreeBuilderOMP(config_.num_threads));

    // 2. Configure GICP
    small_gicp::Registration<small_gicp::GICPFactor,
                             small_gicp::ParallelReductionOMP>
        registration;
    registration.reduction.num_threads = config_.num_threads;
    registration.rejector.max_dist_sq = config_.max_dist_sq;
    registration.optimizer.max_iterations = 15;

    // 3. Align
    auto result = registration.align(*target_covariance_, *source_down,
                                     *target_tree_, T_map_odom_);

    if (result.converged) {
      T_map_odom_ = result.T_target_source;
      transform_time_ = time;
    } else {
      logger->warn("GICP Tracking Warning: Not converged.");
    }
  }

  // =================================================================================
  // Output Publishing
  // =================================================================================
  void publish_tf(fins::AcqTime time) {
    // Current Global Pose = T_map_odom * T_odom_body
    Eigen::Isometry3d T_map_body = T_map_odom_ * T_odom_body_;

    // 1. Publish TF (Map -> Odom)
    if (required("tf")) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp = fins::to_ros_msg_time(time);
      tf.header.frame_id = "map";
      tf.child_frame_id = "camera_init";

      Eigen::Vector3d t = T_map_odom_.translation();
      Eigen::Quaterniond q(T_map_odom_.rotation());

      tf.transform.translation.x = t.x();
      tf.transform.translation.y = t.y();
      tf.transform.translation.z = t.z();
      tf.transform.rotation.w = q.w();
      tf.transform.rotation.x = q.x();
      tf.transform.rotation.y = q.y();
      tf.transform.rotation.z = q.z();

      send("tf", tf, time);
    }

    if (required("corrected_path")) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header.frame_id = "map";
      ps.pose.position.x = T_map_body.translation().x();
      ps.pose.position.y = T_map_body.translation().y();
      ps.pose.position.z = T_map_body.translation().z();
      Eigen::Quaterniond q(T_map_body.rotation());
      ps.pose.orientation.w = q.w();
      ps.pose.orientation.x = q.x();
      ps.pose.orientation.y = q.y();
      ps.pose.orientation.z = q.z();

      corrected_path_.poses.push_back(ps);
      if (corrected_path_.poses.size() > 2000)
        corrected_path_.poses.erase(corrected_path_.poses.begin());
      send("corrected_path", corrected_path_, time);
    }
  }
};

EXPORT_NODE(RelocalizationNode)
DEFINE_PLUGIN_ENTRY()