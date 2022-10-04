cmake_minimum_required(VERSION 2.8.3)
project(map_localization)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  tf
  roscpp
  rospy
  cv_bridge
  # pcl library
  pcl_conversions
  # msgs
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_generation
  visualization_msgs
  ndt_omp
  fast_gicp
)

find_package(Boost REQUIRED COMPONENTS timer)
find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)
find_package(GTSAM REQUIRED QUIET)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

generate_dynamic_reconfigure_options(
  cfg/DynamicParam.cfg
)

add_message_files(
  DIRECTORY msg
  FILES
  cloud_info.msg
  cloud_feature_info.msg
  GPS_info.msg
)

add_service_files(
  DIRECTORY srv
  FILES
  save_map.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  include/faster_lio_sam
  third_party/GeographicLib/include
  third_party
  DEPENDS PCL GTSAM

  CATKIN_DEPENDS
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  message_runtime
  message_generation
  visualization_msgs
)

# include directories
include_directories(
	include
  include/faster_lio_sam
  third_party/GeographicLib/include
  # third_party
  "/usr/include/eigen3"
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
)

add_subdirectory(third_party/GeographicLib)

# link directories
link_directories(
	include
  include/faster_lio_sam
  third_party/GeographicLib/include
	${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
  ${CERES_LIBRARY_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

###########
## Build ##
###########

# Range Image Projection
add_executable(${PROJECT_NAME}_imageProjection src/imageProjection.cpp)
add_dependencies(${PROJECT_NAME}_imageProjection ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_imageProjection ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} libGeographiccc)

# Feature Association
add_executable(${PROJECT_NAME}_featureExtraction src/featureExtraction.cpp src/ClosedPointsPlane.cpp)
add_dependencies(${PROJECT_NAME}_featureExtraction ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_featureExtraction ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} libGeographiccc ${CERES_LIBRARIES})

# # Mapping Optimization
# add_executable(${PROJECT_NAME}_mapOptmization 
#   src/mapOptmization.cpp
#   src/Scancontext.cpp
#   src/registrations.cpp)
# add_dependencies(${PROJECT_NAME}_mapOptmization ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp ${PROJECT_NAME}_gencfg)
# target_compile_options(${PROJECT_NAME}_mapOptmization PRIVATE ${OpenMP_CXX_FLAGS})
# target_link_libraries(${PROJECT_NAME}_mapOptmization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam libGeographiccc ${CERES_LIBRARIES})

# IMU Preintegration
add_executable(${PROJECT_NAME}_imuPreintegration src/imuPreintegration.cpp)
target_link_libraries(${PROJECT_NAME}_imuPreintegration ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam libGeographiccc)

# # Mapping Optimization feature
# add_executable(${PROJECT_NAME}_mapOptmization_feature 
#   src/mapOptmization_feature.cpp
#   src/Scancontext.cpp
#   src/ClosedPointsPlane.cpp
#   src/registrations.cpp)
#   # third_party/fast_gicp/gicp/lsq_registration.cpp
#   # third_party/fast_gicp/gicp/fast_gicp.cpp
#   # third_party/fast_gicp/gicp/fast_gicp_st.cpp
#   # third_party/fast_gicp/gicp/fast_vgicp.cpp)
# add_dependencies(${PROJECT_NAME}_mapOptmization_feature ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp ${PROJECT_NAME}_gencfg)
# target_compile_options(${PROJECT_NAME}_mapOptmization_feature PRIVATE ${OpenMP_CXX_FLAGS})
# target_link_libraries(${PROJECT_NAME}_mapOptmization_feature ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam libGeographiccc ${CERES_LIBRARIES})

# Feature Association
add_executable(${PROJECT_NAME}_featureExtraction_lessF src/feature_opt/featureExtraction_opt.cpp src/ClosedPointsPlane.cpp)
add_dependencies(${PROJECT_NAME}_featureExtraction_lessF ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_featureExtraction_lessF ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} libGeographiccc ${CERES_LIBRARIES})

# Mapping Optimization feature
add_executable(${PROJECT_NAME}_mapOptmization_lessF 
  src/feature_opt/mapOptmization_feature_opt.cpp
  src/Scancontext.cpp
  src/registrations.cpp)
add_dependencies(${PROJECT_NAME}_mapOptmization_lessF ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp ${PROJECT_NAME}_gencfg)
target_compile_options(${PROJECT_NAME}_mapOptmization_lessF PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_mapOptmization_lessF ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam libGeographiccc ${CERES_LIBRARIES})

# Mapping Optimization feature
add_executable(${PROJECT_NAME}_mapOptmization_filter
  src/Filter/mapOptmization_filter.cpp
  src/Scancontext.cpp
  src/registrations.cpp)
add_dependencies(${PROJECT_NAME}_mapOptmization_filter ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp ${PROJECT_NAME}_gencfg)
target_compile_options(${PROJECT_NAME}_mapOptmization_filter PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_mapOptmization_filter ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam libGeographiccc ${CERES_LIBRARIES})

# Mapping Optimization feature
add_executable(${PROJECT_NAME}_mapOptmization_filter_S2
  src/Filter/mapOptmization_filter_S2.cpp
  src/Scancontext.cpp
  src/registrations.cpp)
add_dependencies(${PROJECT_NAME}_mapOptmization_filter_S2 ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp ${PROJECT_NAME}_gencfg)
target_compile_options(${PROJECT_NAME}_mapOptmization_filter_S2 PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_mapOptmization_filter_S2 ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam libGeographiccc ${CERES_LIBRARIES})