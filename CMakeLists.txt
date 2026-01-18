cmake_minimum_required(VERSION 3.16)

project(nav_localization)

find_package(PCL QUIET)
find_package(OpenCV 4.7.0 REQUIRED)

fins_add_node(${PROJECT_NAME} SHARED localization.cpp)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_include_directories(${PROJECT_NAME} PRIVATE ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

fins_link_ros_dependencies(${PROJECT_NAME}
  nav_msgs
  geometry_msgs
  cv_bridge
  pcl_conversions
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  ${PCL_LIBRARIES} 
  ${OpenCV_LIBS}
)