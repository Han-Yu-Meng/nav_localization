cmake_minimum_required(VERSION 3.16)

find_package(nav_msgs QUIET)
find_package(geometry_msgs QUIET)
find_package(cv_bridge QUIET)
find_package(PCL QUIET)
find_package(pcl_conversions QUIET)

find_package(OpenCV 4.7.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(nav_localization SHARED 
  localization.cpp
)

target_compile_features(nav_localization PRIVATE cxx_std_17)
target_include_directories(nav_localization PRIVATE ${PCL_INCLUDE_DIRS})

if(TARGET nav_msgs::nav_msgs)
    target_link_libraries(nav_localization PRIVATE nav_msgs::nav_msgs)
else()
    target_include_directories(nav_localization PRIVATE ${nav_msgs_INCLUDE_DIRS})
    target_link_libraries(nav_localization PRIVATE ${nav_msgs_LIBRARIES})
endif()

if(TARGET geometry_msgs::geometry_msgs)
    target_link_libraries(nav_localization PRIVATE geometry_msgs::geometry_msgs)
else()
    target_include_directories(nav_localization PRIVATE ${geometry_msgs_INCLUDE_DIRS})
    target_link_libraries(nav_localization PRIVATE ${geometry_msgs_LIBRARIES})
endif()

if (TARGET cv_bridge::cv_bridge)
	target_link_libraries(nav_localization PRIVATE cv_bridge::cv_bridge)
else()
	target_include_directories(nav_localization PRIVATE ${cv_bridge_INCLUDE_DIRS})
	target_link_libraries(nav_localization PRIVATE ${cv_bridge_LIBRARIES})
endif()

if(TARGET pcl_conversions::pcl_conversions)
    target_link_libraries(nav_localization PRIVATE pcl_conversions::pcl_conversions)
else()
    target_include_directories(nav_localization PRIVATE ${pcl_conversions_INCLUDE_DIRS})
    target_link_libraries(nav_localization PRIVATE ${pcl_conversions_LIBRARIES})
endif()

target_link_libraries(nav_localization PRIVATE fins_sdk ${PCL_LIBRARIES})
target_link_libraries(nav_localization PRIVATE ${OpenCV_LIBS})