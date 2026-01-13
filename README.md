
# ScanContextNode (FINS Integration)

## Overview
This repository contains a robust global localization node integrated into the FINS framework.

## Credits & Attribution
This node is an implementation based on the original Scan Context algorithm. 
*   **Original Author:** Giseop Kim (KAIST)
*   **Original Paper:** [Scan Context: Egocentric Spatial Descriptor for Place Recognition within 3D Point Cloud Map](https://github.com/gisbi-kim/scancontext) (IROS 2018).
*   **Acknowledgment:** We thank the original authors for providing the core descriptor generation and matching logic.

## Prerequisites
*   **FINS SDK**
*   **PCL** (Point Cloud Library)
*   **Eigen 3**

## Node Configuration

### Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `map_dir` | string | `/path/to/map` | Directory containing `.pcd` and `.odom` files. |
| `dist_threshold` | double | `0.20` | Scan Context matching distance threshold (lower is stricter). |
| `use_icp` | bool | `true` | Whether to enable the multi-stage ICP refinement. |

### Input Ports
*   **Port 0 (`cloud`):** `pcl::PointCloud<pcl::PointXYZI>::Ptr` - Current LiDAR scan.

### Output Ports
*   **Port 0 (`tf`):** `geometry_msgs::msg::TransformStamped` - Global pose in `map` frame.
*   **Port 1 (`path`):** `nav_msgs::msg::Path` - Accumulated trajectory.
*   **Port 2 (`global_map`):** `pcl::PointCloud<pcl::PointXYZRGB>::Ptr` - Colored global map for Rviz.
*   **Port 3 (`aligned_cloud`):** `pcl::PointCloud<pcl::PointXYZI>::Ptr` - Current scan transformed to map frame.

## License
*   **Original Scan Context Logic:** Distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (see original repository).
*   **FINS Wrapper & ICP Refinement:** Copyright (c) 2025 IWIN-FINS Lab, Shanghai Jiao Tong University.
