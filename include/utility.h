#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv/cv.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pclomp/ndt_omp.h>
#include <pclomp/gicp_omp.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
 
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <LocalCartesian.hpp>
// #include <novatel_oem7_msgs/INSPVAX.h>

#include <ceres/ceres.h> 
#include <opencv2/core/core.hpp>

using namespace std;

//终端字体颜色
#define RESET "\033[0m"
#define BLACK "\033[30m"     /* Black */
#define RED "\033[1;31m"     /* Red */
#define GREEN "\033[1;32m"   /* Green */
#define YELLOW "\033[1;33m"  /* Yellow */
#define BLUE "\033[1;34m"    /* Blue */
#define MAGENTA "\033[1;35m" /* Magenta */
#define CYAN "\033[1;36m"    /* Cyan */
#define WHITE "\033[1;37m"   /* White */

struct PointPlaneXYZI
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float w1;
    float w2;
    float w3;
    float d;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (PointPlaneXYZI,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (float, w1, w1) (float, w2, w2) (float, w3, w3) (float, d, d)
)

typedef pcl::PointXYZI PointType;

enum class SensorType { VELODYNE, OUSTER, RSLIDAR, RSLIDAR32};

class ParamServer
{
public:

    ros::NodeHandle nh;

    std::string robot_id;

    //Topics
    string pointCloudTopic;
    string imuTopic;
    string odomTopic;
    string gpsTopic;
    string gpsRawTopic;
    string insTopic;
    

    //Frames
    string lidarFrame;
    string baselinkFrame;
    string odometryFrame;
    string mapFrame;

    // GPS Settings
    bool useGps;
    bool useImuHeadingInitialization;
    bool useGpsElevation;
    float gpsCovThreshold;
    float poseCovThreshold;

    // Save pcd
    bool savePCD;
    string savePCDDirectory;

    // Lidar Sensor Configuration
    SensorType sensor;
    int N_SCAN;
    int Horizon_SCAN;
    int downsampleRate;
    float lidarMinRange;
    float lidarMaxRange;
    bool USE_CORNER;

    // IMU
    float imuAccNoise;
    float imuGyrNoise;
    float imuAccBiasN;
    float imuGyrBiasN;
    float imuGravity;
    float imuRPYWeight;
    float imuRate;
    vector<double> extRotV;
    vector<double> extRPYV;
    vector<double> extTransV;
    Eigen::Matrix3d extRot;
    Eigen::Matrix3d extRPY;
    Eigen::Vector3d extTrans;
    Eigen::Quaterniond extQRPY;
    Eigen::Vector3d gyr_prev;
    Eigen::Vector3d gyr_prev2;
    int imuQueSize;
    float imuTimeOffset;

    // LOAM
    float edgeThreshold;
    float surfThreshold;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;

    // voxel filter paprams
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize ;

    float z_tollerance; 
    float rotation_tollerance;

    // CPU Params
    int numberOfCores;
    double mappingProcessInterval;

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold; 
    float surroundingkeyframeAddingAngleThreshold; 
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;
    
    // Loop closure
    bool  loopClosureEnableFlag;
    float loopClosureFrequency;
    int   surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int   historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;

    // global map visualization radius
    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    //ground segmentation
    double HeightThreshold;

    //FOR CHANGAN
    bool FOR_CHANGAN;
    std::string gps_CHANGAN_odom;
    int keyframenumber;
    int containersize;
    //END

    //lego_loam
    int groundScanInd;
    double GroundAngleThreshold;
    float segmentTheta; 
    int segmentThresholdNum;
    int segmentValidPointNum;
    int segmentValidLineNum;

    //new_feature
    float e_mean_thr;
    float e_max_thr;
    float occ_thr;
    float corner_thr;
    float surf_thr;

    // extract feature threshold
    int MaxCornerSize;
    int MaxSurfSize;
    int MaxoptIteration;
    //degenerate threshold
    float Degenerate_Thr;

    //map extrinsic
    vector<double> RenuV;
    vector<double> leverV;
    vector<double> LLA_MAPV;
    Eigen::Matrix3d Renu;
    Eigen::Vector3d lever_arm;
    Eigen::Vector3d LLA_MAP;

    //map params
    std::string map_path;
    int UsagePointNum;
    int SubMapNum;
    int KeySubMapNum;
    std::string SubMapInfo;
    bool USE_SUBMAP;

    //Odom Save Path
    std::string Odom_Path;

    double LIDAR_STD;
    bool USE_S2;

    ParamServer()
    {   
        //map
        nh.param<std::string>("lio_sam/SubMapInfo", SubMapInfo, "/home/a/driver_ws/src/calib_lidar_imu/ndt_map/approximate_ndt_mapping_.ods");
        nh.param<int>("lio_sam/SubMapNum", SubMapNum, 48);
        nh.param<int>("lio_sam/KeySubMapNum", KeySubMapNum, 1);
        nh.param<int>("lio_sam/UsagePointNum", UsagePointNum, 10000);
        nh.param<bool>("lio_sam/USE_SUBMAP", USE_SUBMAP, true);
        //map extrinsic param
        nh.param<std::string>("lio_sam/map_path", map_path, "/home/a/octomap_ws/src/octomap_show/map.pcd");
        nh.param<vector<double>>("lio_sam/Renu", RenuV, vector<double>());
        nh.param<vector<double>>("lio_sam/lever_arm", leverV, vector<double>());
        nh.param<vector<double>>("lio_sam/LLA_MAP", LLA_MAPV, vector<double>());
        Renu = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(RenuV.data(), 3, 3);
        lever_arm = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(leverV.data(), 3, 1);
        LLA_MAP = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(LLA_MAPV.data(), 3, 1);
        
        //Odom_Path
        nh.param<std::string>("lio_sam/Odom_Path", Odom_Path, "/home/a/");

        //new feature
        nh.param<float>("lio_sam/e_mean_thr", e_mean_thr, 0.8);
        nh.param<float>("lio_sam/e_max_thr", e_max_thr, 1.0472);
        nh.param<float>("lio_sam/occ_thr", occ_thr, 0.3);
        nh.param<float>("lio_sam/corner_thr", corner_thr, 0.5);
        nh.param<float>("lio_sam/surf_thr", surf_thr, 0.8);

        // extract feature threshold
        nh.param<int>("lio_sam/MaxCornerSize", MaxCornerSize, 2);
        nh.param<int>("lio_sam/MaxSurfSize", MaxSurfSize, 5);
        //degenerate threshold
        nh.param<float>("lio_sam/Degenerate_Thr", Degenerate_Thr, 100.0);
        //optimization iteration threshold
        nh.param<int>("lio_sam/MaxoptIteration", MaxoptIteration, 15);

        //lego_loam
        nh.param<int>("lio_sam/segmentThresholdNum", segmentThresholdNum, 30);
        nh.param<int>("lio_sam/segmentValidPointNum", segmentValidPointNum, 5);
        nh.param<int>("lio_sam/segmentValidLineNum", segmentValidLineNum, 3);
        nh.param<float>("lio_sam/segmentTheta", segmentTheta, 1.0472);  // segmentTheta=1.0472<==>60度,在imageProjection中用于判断平面
        nh.param<double>("lio_sam/GroundAngleThreshold", GroundAngleThreshold, 10);
        nh.param<int>("lio_sam/groundScanInd", groundScanInd, 7);

        nh.param<int>("lio_sam/containersize", containersize, 100);
        nh.param<std::string>("lio_sam/gps_CHANGAN_odom", gps_CHANGAN_odom, "gnss_odom2");
        nh.param<bool>("lio_sam/FOR_CHANGAN", FOR_CHANGAN, false);
        nh.param<double>("lio_sam/HeightThreshold", HeightThreshold, -2.0);
        nh.param<int>("lio_sam/keyframenumber", keyframenumber, 100);
        nh.param<float>("lio_sam/imuRate", imuRate, 100.0);
        
        nh.param<std::string>("/robot_id", robot_id, "roboat");

        nh.param<std::string>("lio_sam/pointCloudTopic", pointCloudTopic, "points_raw");
        nh.param<std::string>("lio_sam/imuTopic", imuTopic, "imu_correct");
        nh.param<std::string>("lio_sam/odomTopic", odomTopic, "odometry/imu");
        nh.param<std::string>("lio_sam/gpsTopic", gpsTopic, "odometry/gps");
        nh.param<std::string>("lio_sam/gpsRawTopic", gpsRawTopic, "gps/fix");
        // nh.param<std::string>("lio_sam/insTopic", insTopic, "/novatel/oem7/inspvax");


        nh.param<std::string>("lio_sam/lidarFrame", lidarFrame, "base_link");
        nh.param<std::string>("lio_sam/baselinkFrame", baselinkFrame, "base_link");
        nh.param<std::string>("lio_sam/odometryFrame", odometryFrame, "odom");
        nh.param<std::string>("lio_sam/mapFrame", mapFrame, "map");

        nh.param<bool>("lio_sam/useGps", useGps, false);
        nh.param<bool>("lio_sam/useImuHeadingInitialization", useImuHeadingInitialization, false);
        nh.param<bool>("lio_sam/useGpsElevation", useGpsElevation, false);
        nh.param<float>("lio_sam/gpsCovThreshold", gpsCovThreshold, 2.0);
        nh.param<float>("lio_sam/poseCovThreshold", poseCovThreshold, 25.0);

        nh.param<bool>("lio_sam/savePCD", savePCD, false);
        nh.param<std::string>("lio_sam/savePCDDirectory", savePCDDirectory, "/Downloads/LOAM/");

        nh.param<double>("lio_sam/LIDAR_STD", LIDAR_STD, 0.001);
        nh.param<bool>("lio_sam/USE_S2", USE_S2, false);

        std::string sensorStr;
        nh.param<std::string>("lio_sam/sensor", sensorStr, "");
        if (sensorStr == "velodyne")
        {
            sensor = SensorType::VELODYNE;
        }
        else if (sensorStr == "ouster")
        {
            sensor = SensorType::OUSTER;
        }
        else if (sensorStr == "rslidar"){
            sensor = SensorType::RSLIDAR;
        }
        else if (sensorStr == "rslidar32"){
            sensor = SensorType::RSLIDAR32;
        }
        else
        {
            ROS_ERROR_STREAM(
                "Invalid sensor type (must be either 'velodyne' or 'ouster'): " << sensorStr);
            ros::shutdown();
        }

        nh.param<int>("lio_sam/N_SCAN", N_SCAN, 16);
        nh.param<int>("lio_sam/Horizon_SCAN", Horizon_SCAN, 1800);
        nh.param<int>("lio_sam/downsampleRate", downsampleRate, 1);
        nh.param<float>("lio_sam/lidarMinRange", lidarMinRange, 1.0);
        nh.param<float>("lio_sam/lidarMaxRange", lidarMaxRange, 1000.0);
        nh.param<bool>("lio_sam/USE_CORNER", USE_CORNER, false);

        nh.param<float>("lio_sam/imuAccNoise", imuAccNoise, 0.01);
        nh.param<float>("lio_sam/imuGyrNoise", imuGyrNoise, 0.001);
        nh.param<float>("lio_sam/imuAccBiasN", imuAccBiasN, 0.0002);
        nh.param<float>("lio_sam/imuGyrBiasN", imuGyrBiasN, 0.00003);
        nh.param<float>("lio_sam/imuGravity", imuGravity, 9.80511);
        nh.param<float>("lio_sam/imuRPYWeight", imuRPYWeight, 0.01);
        nh.param<vector<double>>("lio_sam/extrinsicRot", extRotV, vector<double>());
        nh.param<vector<double>>("lio_sam/extrinsicRPY", extRPYV, vector<double>());
        nh.param<vector<double>>("lio_sam/extrinsicTrans", extTransV, vector<double>());
        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        extQRPY = Eigen::Quaterniond(extRPY);
        nh.param<int>("lio_sam/imuQueSize", imuQueSize, 100);
        nh.param<float>("lio_sam/imuTimeOffset", imuTimeOffset, 0.0);

        nh.param<float>("lio_sam/edgeThreshold", edgeThreshold, 0.1);
        nh.param<float>("lio_sam/surfThreshold", surfThreshold, 0.1);
        nh.param<int>("lio_sam/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
        nh.param<int>("lio_sam/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

        nh.param<float>("lio_sam/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
        nh.param<float>("lio_sam/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
        nh.param<float>("lio_sam/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

        nh.param<float>("lio_sam/z_tollerance", z_tollerance, FLT_MAX);
        nh.param<float>("lio_sam/rotation_tollerance", rotation_tollerance, FLT_MAX);

        nh.param<int>("lio_sam/numberOfCores", numberOfCores, 2);
        nh.param<double>("lio_sam/mappingProcessInterval", mappingProcessInterval, 0.15);

        nh.param<float>("lio_sam/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
        nh.param<float>("lio_sam/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
        nh.param<float>("lio_sam/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
        nh.param<float>("lio_sam/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

        nh.param<bool>("lio_sam/loopClosureEnableFlag", loopClosureEnableFlag, false);
        nh.param<float>("lio_sam/loopClosureFrequency", loopClosureFrequency, 1.0);
        nh.param<int>("lio_sam/surroundingKeyframeSize", surroundingKeyframeSize, 50);
        nh.param<float>("lio_sam/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
        nh.param<float>("lio_sam/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
        nh.param<int>("lio_sam/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
        nh.param<float>("lio_sam/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

        nh.param<float>("lio_sam/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
        nh.param<float>("lio_sam/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
        nh.param<float>("lio_sam/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

        usleep(100);
    }

    sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in)
    {
        sensor_msgs::Imu imu_out = imu_in;
        // Eigen::Quaterniond Q(extRot);
        // tf::Quaternion transformQuaternion(Q.x(), Q.y(), Q.z(), Q.w());
        // double roll, pitch, yaw;
        // tf::Matrix3x3(transformQuaternion).getRPY(roll, pitch, yaw);
        // cout<<roll<<" "<<pitch<<" "<<yaw<<endl;
        // rotate acceleration
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();
        // rotate gyroscope
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();

        //六轴不用
        if(useGps)
        {
            // rotate roll pitch yaw
            Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
            Eigen::Quaterniond q_final = q_from * extQRPY.inverse();
            imu_out.orientation.x = q_final.x();
            imu_out.orientation.y = q_final.y();
            imu_out.orientation.z = q_final.z();
            imu_out.orientation.w = q_final.w();

            if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
            {
                ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
                ros::shutdown();
            }
        }

        return imu_out;
    }
};


inline sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

template<typename T>
double ROS_TIME(T msg)
{
    return msg->header.stamp.toSec();
}


template<typename T>
void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}


template<typename T>
void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
{
    *acc_x = thisImuMsg->linear_acceleration.x;
    *acc_y = thisImuMsg->linear_acceleration.y;
    *acc_z = thisImuMsg->linear_acceleration.z;
}


template<typename T>
void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    double imuRoll, imuPitch, imuYaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}


inline float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


inline float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

inline void ConvertLLAToENU(const Eigen::Vector3d& init_lla, 
                            const Eigen::Vector3d& point_lla, 
                            Eigen::Vector3d* point_enu) {
    static GeographicLib::LocalCartesian local_cartesian;
    local_cartesian.Reset(init_lla(0), init_lla(1), init_lla(2));
    local_cartesian.Forward(point_lla(0), point_lla(1), point_lla(2), 
                            point_enu->data()[0], point_enu->data()[1], point_enu->data()[2]);
}

inline void ConvertENUToLLA(const Eigen::Vector3d& init_lla, 
                            const Eigen::Vector3d& point_enu,
                            Eigen::Vector3d* point_lla) {
    static GeographicLib::LocalCartesian local_cartesian;
    local_cartesian.Reset(init_lla(0), init_lla(1), init_lla(2));
    local_cartesian.Reverse(point_enu(0), point_enu(1), point_enu(2), 
                            point_lla->data()[0], point_lla->data()[1], point_lla->data()[2]);                            
}

template<typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 3> GetSkewMatrix(const Eigen::MatrixBase<Derived> &v){
    Eigen::Matrix<typename Derived::Scalar, 3, 3> w;
    w<<0., -v(2), v(1),
       v(2), 0., -v(0),
       -v(1), v(0), 0.;
    return w;
}

inline Eigen::Matrix3f Jfai(Eigen::Vector3f dfai){
    float fai = dfai.norm();
    Eigen::Vector3f alpha = dfai.normalized();
    Eigen::Matrix3f Jfai = sin(fai)/fai * Eigen::Matrix3f::Identity() + (1 - sin(fai)/fai) * alpha * alpha.transpose()
                           + (1 - cos(fai))/fai * GetSkewMatrix(dfai);
    return (fai>0.00001)? Jfai : Eigen::Matrix3f::Identity();
}

inline Eigen::Matrix3f invJfai(Eigen::Vector3f dfai){
    float fai = dfai.norm();
    Eigen::Vector3f alpha = dfai.normalized();
    Eigen::Matrix3f invJfai = 0.5 * fai/tan(0.5 * fai) * Eigen::Matrix3f::Identity() 
                           + (1 - 0.5 * fai/tan(0.5 * fai)) * alpha * alpha.transpose()
                           - 0.5 * fai * GetSkewMatrix(dfai);
    return (fai>0.00001)? invJfai : Eigen::Matrix3f::Identity();
}

inline Eigen::Matrix3f Rodrigues(Eigen::Vector3f r){
    float fai = r.norm();
    Eigen::Vector3f normd = r.normalized();
    Eigen::Matrix3f x = cos(fai) * Eigen::Matrix3f::Identity() 
                    + (1-cos(fai)) * normd * normd.transpose() 
                    + sin(fai) * GetSkewMatrix(normd);
    return x;
}

inline Eigen::Vector3f Rotation2Euler(Eigen::Matrix3f R){
    Eigen::Matrix3f Rtrans = R.transpose();
    Eigen::Vector3f rad;
    rad(0) = atan2(Rtrans(1,2), Rtrans(2,2));
    rad(1) = -asin(Rtrans(0,2));
    rad(2) = atan2(Rtrans(0,1), Rtrans(0,0));
    return rad;
}

template<typename T>
bool has_nan(T point) {

    // remove nan point, or the feature assocaion will crash, the surf point will containing nan points
    // pcl remove nan not work normally
    // ROS_ERROR("Containing nan point!");
    if (pcl_isnan(point.x) || pcl_isnan(point.y) || pcl_isnan(point.z)) {
        return true;
    } else {
        return false;
    }
}

inline Eigen::Quaterniond getInterpolatedAttitude(const Eigen::Quaterniond &q_start, const Eigen::Quaterniond &q_end, double scale)
{
    return q_start.slerp(scale, q_end);
}

inline Eigen::Vector3d getInterpolatedTrans(const Eigen::Vector3d t_start, const Eigen::Vector3d t_end, double scale)
{
    return t_start + (t_end - t_start) * scale;
}

inline std::vector<float> splitPoseLine(std::string _str_line, char _delimiter) {
    std::vector<float> parsed;
    std::stringstream ss(_str_line);
    std::string temp;
    while (getline(ss, temp, _delimiter)) {
        parsed.push_back(std::stof(temp)); // convert string to "float"
    }
    return parsed;
}

inline pair<Eigen::Vector3f, Eigen::Vector3f> getAccGyro(const sensor_msgs::Imu& msg)
{
    Eigen::Vector3f acc = Eigen::Vector3f(msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z);
    Eigen::Vector3f gyro = Eigen::Vector3f(msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z);
    return make_pair(acc, gyro);
}

//A() in kalman fileters on manifold
template<typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 3> Amatrix(const Eigen::MatrixBase<Derived> &v){
    Eigen::Matrix<typename Derived::Scalar, 3, 3> w = Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity();
    typename Derived::Scalar norm = v.norm();
    Eigen::Matrix<typename Derived::Scalar, 3, 3> skew = GetSkewMatrix(v / norm);

    if(norm < 10e-5)
    {
        return w;
    }
    
    return Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() 
               + (1-cos(norm)) / norm * skew + (1 - sin(norm) / norm) * skew * skew;
}

// template <typename T>
// T NormalizeAngle(const T& angle_degrees) {
//     if (angle_degrees > T(M_PI))
//         return angle_degrees - T(2*M_PI);
//     else if (angle_degrees < T(-M_PI))
//         return angle_degrees + T(2*M_PI);
//     else
//         return angle_degrees;
// };

template <typename T>
Eigen::Matrix<T, 3, 1> NormalizeAngle(Eigen::Matrix<T, 3, 1>& angle_degrees)
{
    for(int i=0; i<3; ++i)
    {
        if (angle_degrees(i, 0) > T(M_PI))
            angle_degrees(i, 0) = angle_degrees(i, 0) - T(2*M_PI);
        else if (angle_degrees(i, 0) < T(-M_PI))
            angle_degrees(i, 0) = angle_degrees(i, 0) + T(2*M_PI);
    }

    return angle_degrees;
}

#endif
