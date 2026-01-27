/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

#include "Scancontext.hpp"
#include <fins/node.hpp>

// ROS Messages
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// PCL
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
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
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iomanip>
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
  std::string sc_feature_dir;
  std::string global_map_path;
  double sc_dist_thres = 0.50;

  int num_threads = 4;
  int num_neighbors = 20;
  double global_leaf_size = 0.25;
  double registered_leaf_size = 0.1;
  double max_dist_sq = 5.0;
  double accumulation_time = 5.0;
  double tracking_interval = 1.0;
};

enum class SystemState { IDLE, ACCUMULATING, LOCALIZING, TRACKING };

struct GICPJob {
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_odom;
  fins::AcqTime timestamp;
  Eigen::Isometry3d T_map_odom_initial;
  Eigen::Isometry3d T_odom_lidar_snapshot;
};

class RelocalizationNode : public fins::Node {
public:
  void define() override {
    set_name("RelocalizationNode");
    set_description(
        "Relocalization: Dual Input (Odom Cloud for GICP, Lidar Cloud for SC)");
    set_category("Navigation>Localization");

    register_input<pcl::PointCloud<pcl::PointXYZI>::Ptr>(
        "cloud_odom", &RelocalizationNode::on_cloud_odom);
    register_input<pcl::PointCloud<pcl::PointXYZI>::Ptr>(
        "cloud_lidar", &RelocalizationNode::on_cloud_lidar);
    register_input<geometry_msgs::msg::TransformStamped>(
        "$T_{odom\\to lidar}$", &RelocalizationNode::on_transform);

    register_output<geometry_msgs::msg::TransformStamped>("$T_{map\\to odom}$");
    register_output<nav_msgs::msg::Path>("corrected_path");

    register_output<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>("global_map_viz");
    register_output<pcl::PointCloud<pcl::PointXYZI>::Ptr>(
        "debug_accumulated_cloud");
    register_output<pcl::PointCloud<pcl::PointXYZI>::Ptr>("debug_gicp_init");

    register_parameter<std::string>("sc_feature_map_dir",
                                    &RelocalizationNode::update_feature_map_dir,
                                    "/path/to/sc_features/");
    register_parameter<std::string>("global_map_dir",
                                    &RelocalizationNode::update_global_map_dir,
                                    "/path/to/global_map.pcd");
    register_parameter<double>("sc_dist_threshold",
                               &RelocalizationNode::update_dist_thres, 0.40);
  }

  void initialize() override {
    std::lock_guard<std::mutex> lock(mutex_);

    sc_manager_ = std::make_unique<SCManager>();
    global_map_target_ = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    target_covariance_ =
        std::make_shared<pcl::PointCloud<pcl::PointCovariance>>();

    accumulated_cloud_lidar_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    global_map_viz_.reset(new pcl::PointCloud<pcl::PointXYZRGB>());

    state_ = SystemState::IDLE;
    T_map_odom_ = Eigen::Isometry3d::Identity();
    T_odom_lidar_ = Eigen::Isometry3d::Identity();

    corrected_path_.header.frame_id = "map";

    is_running_ = true;
    gicp_worker_thread_ =
        std::thread(&RelocalizationNode::gicp_worker_loop, this);

    viz_thread_ = std::thread([this]() {
      while (is_running_) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (required("global_map_viz") && global_map_viz_ &&
              !global_map_viz_->empty()) {
            send("global_map_viz", global_map_viz_, fins::now());
          }
        }
        std::this_thread::sleep_for(std::chrono::seconds(5));
      }
    });
  }

  ~RelocalizationNode() { stop_threads(); }

  void run() override {

  }

  void pause() override {

  }

  void reset() override {
    std::lock_guard<std::mutex> lock(mutex_);
    corrected_path_.poses.clear();
    T_map_odom_ = Eigen::Isometry3d::Identity();
    T_odom_lidar_ = Eigen::Isometry3d::Identity();
    state_ = SystemState::IDLE;
    accumulated_cloud_lidar_->clear();
    accumulation_started_ = false;
  }

  void update_feature_map_dir(const std::string &v) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config_.sc_feature_dir == v)
      return;
    config_.sc_feature_dir = v;
    load_sc_database();
    check_state_ready();
  }

  void update_global_map_dir(const std::string &v) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config_.global_map_path == v)
      return;
    config_.global_map_path = v;
    load_global_map();
    check_state_ready();
  }

  void update_dist_thres(const double &v) { config_.sc_dist_thres = v; }

  void
  on_transform(const fins::Msg<geometry_msgs::msg::TransformStamped> &msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto &t = msg->transform.translation;
    const auto &r = msg->transform.rotation;
    Eigen::Vector3d p(t.x, t.y, t.z);
    Eigen::Quaterniond q(r.w, r.x, r.y, r.z);
    T_odom_lidar_ = Eigen::Isometry3d::Identity();
    T_odom_lidar_.translate(p);
    T_odom_lidar_.rotate(q);

    if (state_ == SystemState::TRACKING || state_ == SystemState::LOCALIZING) {
      publish_outputs(msg.acq_time);
    }
  }

  void
  on_cloud_lidar(const fins::Msg<pcl::PointCloud<pcl::PointXYZI>::Ptr> &msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != SystemState::ACCUMULATING)
      return;

    auto input_cloud = *msg;
    if (!input_cloud || input_cloud->empty())
      return;

    double current_timestamp = fins::to_seconds(msg.acq_time);

    if (!accumulation_started_) {
      start_accumulation_time_ = current_timestamp;
      accumulation_started_ = true;
      accumulated_cloud_lidar_->clear();
      logger->info("[State] ACCUMULATING started (Input: Cloud Lidar)...");
    }

    double elapsed = current_timestamp - start_accumulation_time_;
    if (elapsed < config_.accumulation_time) {
      static pcl::VoxelGrid<pcl::PointXYZI> vg;
      vg.setLeafSize(0.2f, 0.2f, 0.2f);
      vg.setInputCloud(input_cloud);
      pcl::PointCloud<pcl::PointXYZI> temp;
      vg.filter(temp);
      *accumulated_cloud_lidar_ += temp;
    } else {
      logger->info(
          "[State] Accumulation finished. Points: {}. Running LOCALIZING...",
          accumulated_cloud_lidar_->size());
      state_ = SystemState::LOCALIZING;
      perform_global_localization(msg.acq_time);
    }
  }

  // 回调 2: 处理 Odom 系注册点云 (用于 GICP 追踪)
  void
  on_cloud_odom(const fins::Msg<pcl::PointCloud<pcl::PointXYZI>::Ptr> &msg) {
    // 这里不持有主锁 mutex_，只在需要读取共享状态时短暂持有
    // 或者我们直接在这里做简单的判断

    if (state_ != SystemState::TRACKING)
      return;

    auto input_cloud = *msg;
    if (!input_cloud || input_cloud->empty())
      return;

    double current_timestamp = fins::to_seconds(msg.acq_time);

    // 1. 检查时间间隔 (> 5.0s)
    if (current_timestamp - last_tracking_request_time_ <
        config_.tracking_interval) {
      return; // 还没到时间，忽略
    }

    // 2. 检查工作线程是否空闲
    // 如果正在处理，直接跳过（Drop frame），保证实时性
    if (is_gicp_processing_) {
      // logger->debug("[GICP] Worker busy, skipping frame.");
      return;
    }

    // 3. 准备数据快照 (Snapshot)
    // 需要加锁来安全复制当前的位姿估计
    GICPJob new_job;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!map_ready_ || !target_tree_)
        return;

      new_job.T_map_odom_initial = T_map_odom_;
      new_job.T_odom_lidar_snapshot = T_odom_lidar_;
      new_job.cloud_odom.reset(new pcl::PointCloud<pcl::PointXYZI>(
          *input_cloud)); // 深拷贝或引用计数增加
      new_job.timestamp = msg.acq_time;
    }

    // 4. 发送给工作线程
    {
      std::lock_guard<std::mutex> lock(job_mutex_);
      next_job_ = new_job;
      has_new_job_ = true;
    }
    job_cv_.notify_one();

    last_tracking_request_time_ = current_timestamp;
    logger->info("[GICP] Triggered tracking task at time {:.2f}",
                 current_timestamp);
  }

private:
  std::mutex mutex_;
  Config config_;
  SystemState state_;
  std::atomic<bool> is_running_{false};
  std::thread viz_thread_;

  // GICP Threading
  std::thread gicp_worker_thread_;
  std::mutex job_mutex_;
  std::condition_variable job_cv_;
  bool has_new_job_ = false;
  GICPJob next_job_;
  std::atomic<bool> is_gicp_processing_{false};
  double last_tracking_request_time_ = 0.0;

  bool accumulation_started_ = false;
  double start_accumulation_time_ = 0.0;
  pcl::PointCloud<pcl::PointXYZI>::Ptr accumulated_cloud_lidar_;

  std::unique_ptr<SCManager> sc_manager_;
  std::vector<KeyFrame> keyframes_;
  bool sc_ready_ = false;

  pcl::PointCloud<pcl::PointXYZI>::Ptr global_map_target_;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_map_viz_;
  pcl::PointCloud<pcl::PointCovariance>::Ptr target_covariance_;
  std::shared_ptr<small_gicp::KdTree<pcl::PointCloud<pcl::PointCovariance>>>
      target_tree_;
  bool map_ready_ = false;

  Eigen::Isometry3d T_map_odom_;
  Eigen::Isometry3d T_odom_lidar_;

  nav_msgs::msg::Path corrected_path_;

  void stop_threads() {
    is_running_ = false;
    job_cv_.notify_all();
    if (viz_thread_.joinable())
      viz_thread_.join();
    if (gicp_worker_thread_.joinable())
      gicp_worker_thread_.join();
  }

  void check_state_ready() {
    if (sc_ready_ && map_ready_ && state_ == SystemState::IDLE) {
      state_ = SystemState::ACCUMULATING;
      logger->info("[System] All maps loaded. Ready.");
    }
  }

  // =================================================================================
  // Database Loading (Helpers)
  // =================================================================================
  bool load_sc_database() {
    // ... (保持原样)
    namespace fs = std::filesystem;
    keyframes_.clear();
    sc_manager_ = std::make_unique<SCManager>();
    sc_ready_ = false;

    if (!fs::exists(config_.sc_feature_dir)) {
      logger->warn("SC Feature dir not found: {}", config_.sc_feature_dir);
      return false;
    }

    std::vector<std::string> pcd_files;
    for (const auto &entry : fs::directory_iterator(config_.sc_feature_dir)) {
      if (entry.path().extension() == ".pcd")
        pcd_files.push_back(entry.path().stem().string());
    }
    std::sort(pcd_files.begin(), pcd_files.end());

    logger->info("Loading SC Features from {}...", config_.sc_feature_dir);
    for (size_t i = 0; i < pcd_files.size(); ++i) {
      std::string pcd_path =
          config_.sc_feature_dir + "/" + pcd_files[i] + ".pcd";
      std::string odom_path =
          config_.sc_feature_dir + "/" + pcd_files[i] + ".odom";

      KeyFrame kf;
      kf.id = i;
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
      kf.cloud->clear();
      keyframes_.push_back(kf);
    }
    if (!sc_manager_->polarcontext_invkeys_mat_.empty()) {
      sc_manager_->polarcontext_invkeys_to_search_ =
          sc_manager_->polarcontext_invkeys_mat_;
      sc_manager_->polarcontext_tree_ = std::make_unique<InvKeyTree>(
          sc_manager_->PC_NUM_RING,
          sc_manager_->polarcontext_invkeys_to_search_, 10);
    }
    sc_ready_ = true;
    return true;
  }

  bool load_global_map() {
    map_ready_ = false;
    global_map_target_->clear();
    global_map_viz_->clear();
    if (pcl::io::loadPCDFile(config_.global_map_path, *global_map_target_) ==
        -1) {
      logger->error("Failed to load global map");
      return false;
    }
    pcl::copyPointCloud(*global_map_target_, *global_map_viz_);

    logger->info("Processing Global Map for GICP...");
    auto downsampled_map =
        small_gicp::voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZI>,
                                           pcl::PointCloud<pcl::PointXYZI>>(
            *global_map_target_, config_.global_leaf_size);

    target_covariance_->clear();
    target_covariance_->points.resize(downsampled_map->points.size());
    for (size_t i = 0; i < downsampled_map->points.size(); ++i) {
      target_covariance_->points[i].getVector3fMap() =
          downsampled_map->points[i].getVector3fMap();
    }
    small_gicp::estimate_covariances_omp(
        *target_covariance_, config_.num_neighbors, config_.num_threads);
    target_tree_ = std::make_shared<
        small_gicp::KdTree<pcl::PointCloud<pcl::PointCovariance>>>(
        target_covariance_, small_gicp::KdTreeBuilderOMP(config_.num_threads));

    map_ready_ = true;
    logger->info("Global Map loaded. Points: {}", global_map_target_->size());
    return true;
  }

  void perform_global_localization(fins::AcqTime timestamp) {
    if (accumulated_cloud_lidar_->empty()) {
      logger->warn("[SC] Cloud empty, retrying...");
      state_ = SystemState::ACCUMULATING;
      accumulation_started_ = false;
      return;
    }

    Eigen::MatrixXd sc =
        sc_manager_->makeScancontext(*accumulated_cloud_lidar_);
    Eigen::MatrixXd ringkey = sc_manager_->makeRingkeyFromScancontext(sc);
    std::vector<float> query_key = eig2stdvec(ringkey);

    int candidates = std::min((int)keyframes_.size(),
                              (int)sc_manager_->NUM_CANDIDATES_FROM_TREE);
    std::vector<size_t> indices(candidates);
    std::vector<float> dists(candidates);
    nanoflann::KNNResultSet<float> result_knn(candidates);
    result_knn.init(&indices[0], &dists[0]);
    sc_manager_->polarcontext_tree_->index->findNeighbors(
        result_knn, &query_key[0], nanoflann::SearchParams(10));

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

    if (best_idx != -1 && min_dist < config_.sc_dist_thres) {
      float yaw_diff =
          (best_align * sc_manager_->PC_UNIT_SECTORANGLE) * M_PI / 180.0f;
      Eigen::Isometry3d T_map_kf = Eigen::Isometry3d(keyframes_[best_idx].pose);
      Eigen::Isometry3d T_kf_current = Eigen::Isometry3d::Identity();
      T_kf_current.rotate(
          Eigen::AngleAxisd(yaw_diff, Eigen::Vector3d::UnitZ()));
      Eigen::Isometry3d T_map_lidar_sc = T_map_kf * T_kf_current;
      T_map_odom_ = T_map_lidar_sc * T_odom_lidar_.inverse();

      Eigen::Vector3d t = T_map_odom_.translation();
      logger->info("[SC] SUCCESS | ID: {} | Dist: {:.4f} | T_map_odom: "
                   "[{:.2f}, {:.2f}, {:.2f}]",
                   best_idx, min_dist, t.x(), t.y(), t.z());

      state_ = SystemState::TRACKING;

      last_tracking_request_time_ = fins::to_seconds(timestamp);

      if (required("debug_accumulated_cloud")) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr debug_cloud(
            new pcl::PointCloud<pcl::PointXYZI>());
        pcl::transformPointCloud(*accumulated_cloud_lidar_, *debug_cloud,
                                 T_map_lidar_sc.matrix().cast<float>());
        debug_cloud->header.frame_id = "map";
        send("debug_accumulated_cloud", debug_cloud, timestamp);
      }
    } else {
      logger->warn("[SC] FAILED | Min Dist: {:.4f}", min_dist);
      accumulated_cloud_lidar_->clear();
      accumulation_started_ = false;
      state_ = SystemState::ACCUMULATING;
    }
  }

  void gicp_worker_loop() {
    while (is_running_) {
      GICPJob job;
      {
        std::unique_lock<std::mutex> lock(job_mutex_);
        job_cv_.wait(lock, [this] { return has_new_job_ || !is_running_; });

        if (!is_running_)
          break;

        job = next_job_;
        has_new_job_ = false;
        is_gicp_processing_ = true;
      }

      perform_tracking_internal(job);

      is_gicp_processing_ = false;
    }
  }

  void perform_tracking_internal(const GICPJob &job) {
    auto start_time = std::chrono::high_resolution_clock::now();

    pcl::PointCloud<pcl::PointCovariance>::Ptr target_cov_ptr;
    std::shared_ptr<small_gicp::KdTree<pcl::PointCloud<pcl::PointCovariance>>>
        target_tree_ptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      target_cov_ptr = target_covariance_;
      target_tree_ptr = target_tree_;
    }

    if (!target_cov_ptr || !target_tree_ptr || job.cloud_odom->empty()) {
      return;
    }

    Eigen::Isometry3d T_map_lidar =
        job.T_map_odom_initial * job.T_odom_lidar_snapshot;

    if (T_map_lidar.matrix().hasNaN()) {
      logger->error("[GICP] T_map_lidar contains NaN! Aborting GICP.");
      return;
    }

    logger->info("[GICP Debug] Starting GICP. Odom Cloud Size: {}",
                 job.cloud_odom->size());

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_lidar(
        new pcl::PointCloud<pcl::PointXYZI>());
    Eigen::Matrix4f T_lidar_odom =
        job.T_odom_lidar_snapshot.inverse().matrix().cast<float>();
    pcl::transformPointCloud(*job.cloud_odom, *cloud_lidar, T_lidar_odom);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in_map(
        new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(*cloud_lidar, *cloud_in_map,
                             T_map_lidar.matrix().cast<float>());

    std::vector<int> indices;
    if (!cloud_in_map->is_dense) {
      pcl::removeNaNFromPointCloud(*cloud_in_map, *cloud_in_map, indices);
    }

    if (cloud_in_map->empty()) {
      logger->warn("[GICP] Cloud empty after Transform/NaN removal. Skipping.");
      return;
    }

    auto source_down = small_gicp::voxelgrid_sampling_omp<
        pcl::PointCloud<pcl::PointXYZI>, pcl::PointCloud<pcl::PointCovariance>>(
        *cloud_in_map, config_.registered_leaf_size);

    logger->info("[GICP Debug] After downsampling: {} -> {} points",
                 job.cloud_odom->size(), source_down->size());

    if (source_down->size() < 200) {
      logger->warn(
          "[GICP] Too few points ({}) for reliable registration. Skipping.",
          source_down->size());
      return;
    }

    small_gicp::estimate_covariances_omp(*source_down, config_.num_neighbors,
                                         config_.num_threads);

    small_gicp::Registration<small_gicp::GICPFactor,
                             small_gicp::ParallelReductionOMP>
        registration;
    registration.reduction.num_threads = config_.num_threads;
    registration.rejector.max_dist_sq = config_.max_dist_sq;
    registration.optimizer.max_iterations = 30;

    Eigen::Isometry3d initial_guess = Eigen::Isometry3d::Identity();

    auto align_start = std::chrono::high_resolution_clock::now();
    auto result = registration.align(*target_cov_ptr, *source_down,
                                     *target_tree_ptr, initial_guess);
    auto align_end = std::chrono::high_resolution_clock::now();

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time)
            .count();
    double align_ms =
        std::chrono::duration<double, std::milli>(align_end - align_start)
            .count();

    logger->info("[GICP Stats] Total Time: {:.2f}ms (Align: {:.2f}ms) | "
                 "Converged: {} | Iter: {}",
                 total_ms, align_ms, result.converged, result.iterations);

    Eigen::Isometry3d T_correction_map = result.T_target_source;
    Eigen::Vector3d correction_trans = T_correction_map.translation();
    double correction_rot =
        Eigen::AngleAxisd(T_correction_map.rotation()).angle();

    const double MAX_CORRECTION_TRANS = 5.0;
    const double MAX_CORRECTION_ROT = 5.0;

    bool correction_reasonable =
        (correction_trans.norm() < MAX_CORRECTION_TRANS) &&
        (correction_rot < MAX_CORRECTION_ROT);

    if (result.converged && correction_reasonable) {
      std::lock_guard<std::mutex> lock(mutex_);

      // 更新 T_map_odom_
      // T_map_odom_new = T_correction * T_map_odom_old
      // 注意：这里我们使用 Job 开始时的 T_map_odom_initial 还是当前的
      // T_map_odom_? 理论上应该应用修正量到当前的 T_map_odom_
      // 上（累积漂移修正）

      Eigen::Isometry3d T_map_odom_prev = T_map_odom_;
      T_map_odom_ = T_correction_map * T_map_odom_;

      Eigen::Isometry3d T_delta = T_map_odom_prev.inverse() * T_map_odom_;
      logger->info("[GICP] Update Applied. Delta: Trans={:.3f}m, Rot={:.2f}deg",
                   T_delta.translation().norm(),
                   Eigen::AngleAxisd(T_delta.rotation()).angle() * 180.0 /
                       M_PI);

      publish_outputs(job.timestamp);

    } else {
      if (!result.converged) {
        logger->warn("[GICP] Not converged.");
      } else {
        logger->warn(
            "[GICP] Correction too large (Trans={:.2f}, Rot={:.2f}). Ignored.",
            correction_trans.norm(), correction_rot);
      }
    }
  }

  void publish_outputs(fins::AcqTime time) {
    if (required("$T_{map\\to odom}$")) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp = fins::to_ros_msg_time(time);
      tf.header.frame_id = "map";
      tf.child_frame_id = "odom";

      Eigen::Vector3d t = T_map_odom_.translation();
      Eigen::Quaterniond q(T_map_odom_.rotation());

      tf.transform.translation.x = t.x();
      tf.transform.translation.y = t.y();
      tf.transform.translation.z = t.z();
      tf.transform.rotation.w = q.w();
      tf.transform.rotation.x = q.x();
      tf.transform.rotation.y = q.y();
      tf.transform.rotation.z = q.z();
      send("$T_{map\\to odom}$", tf, time);
    }

    if (required("corrected_path")) {
      Eigen::Isometry3d T_map_lidar = T_map_odom_ * T_odom_lidar_;
      geometry_msgs::msg::PoseStamped ps;
      ps.header.frame_id = "map";
      ps.header.stamp = fins::to_ros_msg_time(time);
      ps.pose.position.x = T_map_lidar.translation().x();
      ps.pose.position.y = T_map_lidar.translation().y();
      ps.pose.position.z = T_map_lidar.translation().z();
      Eigen::Quaterniond q(T_map_lidar.rotation());
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