#include "utility.h"
#include "map_localization/cloud_info.h"
#include <sophus/so3.hpp>

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

struct RsPointXYZIRT
{
  PCL_ADD_POINT4D;
  uint8_t intensity;
  uint16_t ring = 0;
  double timestamp = 0;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(RsPointXYZIRT, (float, x, x)(float, y, y)(float, z, z)(uint8_t, intensity, intensity)(
                                                     uint16_t, ring, ring)(double, timestamp, timestamp))

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;
    std::mutex gpsLock;
    //GPS相关
    //原始GPS在ENU坐标系下的xyz
    ros::Publisher pubgpsodom;
    ros::Publisher pubENUgps;
    ros::Subscriber subGPSRAW;

    std::deque<nav_msgs::Odometry> gpsQueue;
    int gpsSequence = 0;

    double latitude, longitude, altitude;
    Eigen::Vector3f coordinate_offset;
    Eigen::Quaterniond Qw0_gps;
    Eigen::Quaterniond Qgps_imu;
    Eigen::Quaterniond Qw0_i0_;
    std::ofstream file_gps;
    //END

    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;
    //for test
    // ros::Publisher pubModifiedRawCloud;
    //END

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;

    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<RsPointXYZIRT>::Ptr tmpRslidarCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;

    float odomIncreRoll;
    float odomIncrePitch;
    float odomIncreYaw;

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    Eigen::Affine3f transBtSE;
    Eigen::Vector3f rad_BtSE;

    nav_msgs::Odometry startOdomMsg;
    nav_msgs::Odometry endOdomMsg;

    map_localization::cloud_info cloudInfo;
    double timeScanCur;
    double timeScanEnd;
    std_msgs::Header cloudHeader;

public:
    ImageProjection():
    deskewFlag(0)
    {
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subGPSRAW = nh.subscribe<sensor_msgs::NavSatFix>(gpsRawTopic, 200, &ImageProjection::gpsRawHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam/deskew/cloud_deskewed", 1);
        // pubModifiedRawCloud = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/ModefiedCloud", 1);
        pubLaserCloudInfo = nh.advertise<map_localization::cloud_info> ("lio_sam/deskew/cloud_info", 1);

        //看ENU下gps坐标
        pubgpsodom = nh.advertise<nav_msgs::Odometry>(gpsTopic, 10);
        pubENUgps = nh.advertise<nav_msgs::Odometry>("odometry/enu",1000);
        //END

        Qw0_i0_ = Renu;
        coordinate_offset = (Qw0_i0_.toRotationMatrix() * extRot.transpose() * extTrans).cast<float>();

        cout<<"ImageProjection中,Qw0_i0_: "<<Qw0_i0_.w()<<" "
                                            <<Qw0_i0_.x()<<" "
                                            <<Qw0_i0_.y()<<" "
                                            <<Qw0_i0_.z()<<endl;
        cout<<"ImageProjection中,coordinate_offset: "<<coordinate_offset.transpose()<<endl;

        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
        file_gps.open(Odom_Path+"gps_odom(true).csv", std::ios::app);
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        tmpRslidarCloudIn.reset(new pcl::PointCloud<RsPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        cloudInfo.mapQlk.assign(4,0);
        cloudInfo.maptlk.assign(3,0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection(){
        file_gps.close();
    }

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {   
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

        if(imuTimeOffset != 0.0)
        {
            ros::Time time(thisImu.header.stamp.toSec() + imuTimeOffset);
            thisImu.header.stamp = time;
        }

        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);
    }

    // void gpsRawHandler(const sensor_msgs::NavSatFix::ConstPtr& gpsRawMsg){
    //     nav_msgs::Odometry gps_odom;
    //     gps_odom.header.stamp = gpsRawMsg->header.stamp;
    //     gps_odom.header.frame_id = "odom";
    //     gps_odom.header.seq = gpsSequence++;
        
        
    //     Eigen::Vector3d ENU;
    //     ConvertLLAToENU(LLA_MAP, Eigen::Vector3d(gpsRawMsg->latitude, gpsRawMsg->longitude, gpsRawMsg->altitude), &ENU);
    //     gps_odom.pose.pose.position.x = ENU(0);
    //     gps_odom.pose.pose.position.y = ENU(1);
    //     gps_odom.pose.pose.position.z = ENU(2);
    //     gps_odom.pose.pose.orientation.w = gpsRawMsg->position_covariance[0];
    //     gps_odom.pose.pose.orientation.x = gpsRawMsg->position_covariance[1];
    //     gps_odom.pose.pose.orientation.y = gpsRawMsg->position_covariance[2];
    //     gps_odom.pose.pose.orientation.z = gpsRawMsg->position_covariance[3];
    //     gps_odom.pose.covariance = 
    //     {0.0, 0.0, 0.0, 0.0, 
    //     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    //     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    //     pubgpsodom.publish(gps_odom);

    //     //add noise
    //     // double w_sigma = 1.0;
    //     // cv::RNG rng;
    //     gps_odom.pose.pose.position.x = ENU(0)/*+rng.gaussian(w_sigma)*/;
    //     gps_odom.pose.pose.position.y = ENU(1)/*+rng.gaussian(w_sigma)*/;
    //     gps_odom.pose.pose.position.z = ENU(2)/*+rng.gaussian(w_sigma+5)*/;
    //     gpsQueue.push_back(gps_odom);
        
    //     Eigen::Quaternionf Qw0_ik_(gpsRawMsg->position_covariance[0], gpsRawMsg->position_covariance[1], gpsRawMsg->position_covariance[2], gpsRawMsg->position_covariance[3]);
    //     Eigen::Vector3f pose = - Qw0_ik_.toRotationMatrix() * extRot.transpose().cast<float>() * extTrans.cast<float>()
    //                     + Eigen::Vector3d(gps_odom.pose.pose.position.x, gps_odom.pose.pose.position.y, gps_odom.pose.pose.position.z).cast<float>() 
    //                     + coordinate_offset
    //                     + Qw0_i0_.cast<float>() * lever_arm.cast<float>() - Qw0_ik_ * lever_arm.cast<float>();
        
    //     Eigen::Quaternionf Qw0_lk_(Qw0_ik_.toRotationMatrix() * extRot.transpose().cast<float>());
    //     nav_msgs::Odometry gps_odom2;
    //     gps_odom2.header.stamp = gpsRawMsg->header.stamp;
    //     gps_odom2.header.frame_id = "odom";
    //     gps_odom2.pose.pose.orientation.w = Qw0_lk_.w();
    //     gps_odom2.pose.pose.orientation.x = Qw0_lk_.x();
    //     gps_odom2.pose.pose.orientation.y = Qw0_lk_.y();
    //     gps_odom2.pose.pose.orientation.z = Qw0_lk_.z();
    //     gps_odom2.pose.pose.position.x = pose(0);
    //     gps_odom2.pose.pose.position.y = pose(1);
    //     gps_odom2.pose.pose.position.z = pose(2);
    //     pubENUgps.publish(gps_odom2);


    //     file_gps.setf(std::ios::fixed, std::_S_floatfield);
    //     file_gps << gps_odom2.header.stamp.toSec() << " " 
    //             << gps_odom2.pose.pose.position.x << " "
    //             << gps_odom2.pose.pose.position.y << " "
    //             << gps_odom2.pose.pose.position.z << " "
    //             << gps_odom2.pose.pose.orientation.x << " "
    //             << gps_odom2.pose.pose.orientation.y << " "
    //             << gps_odom2.pose.pose.orientation.z << " "
    //             << gps_odom2.pose.pose.orientation.w << std::endl;

    //     std::lock_guard<std::mutex> lock3(gpsLock);

    //     // gpsQueue.push_back(gps_odom2);
    // }

    void gpsRawHandler(const sensor_msgs::NavSatFix::ConstPtr& gpsRawMsg){
        nav_msgs::Odometry gps_odom;
        sensor_msgs::NavSatFix Initial_lla;
        Initial_lla.header.stamp = gpsRawMsg->header.stamp;
        gps_odom.header.stamp = gpsRawMsg->header.stamp;
        gps_odom.header.frame_id = "odom";
        gps_odom.header.seq = gpsSequence++;
        if(gpsSequence  == 1){
            cout<<"CURRENT GPS TIME: "<<setprecision(20)<<gpsRawMsg->header.stamp.toSec()<<endl;
            gps_odom.pose.pose.position.x = 0;
            gps_odom.pose.pose.position.y = 0;
            gps_odom.pose.pose.position.z = 0;
            gps_odom.pose.pose.orientation.w = gpsRawMsg->position_covariance[0];
            gps_odom.pose.pose.orientation.x = gpsRawMsg->position_covariance[1];
            gps_odom.pose.pose.orientation.y = gpsRawMsg->position_covariance[2];
            gps_odom.pose.pose.orientation.z = gpsRawMsg->position_covariance[3];
            gps_odom.pose.covariance = 
            {0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            latitude = gpsRawMsg->latitude;
            longitude = gpsRawMsg->longitude;
            altitude = gpsRawMsg->altitude;
            pubgpsodom.publish(gps_odom);

            Qw0_i0_ = Eigen::Quaterniond(gps_odom.pose.pose.orientation.w, gps_odom.pose.pose.orientation.x,
                                       gps_odom.pose.pose.orientation.y, gps_odom.pose.pose.orientation.z);
            coordinate_offset = (Qw0_i0_.toRotationMatrix() * extRot.transpose() * extTrans).cast<float>();

            //for ev200
            Qw0_gps = Qw0_i0_;
            cout<<"ImageProjection中,Qw0_i0_: "<<Qw0_i0_.w()<<" "
                                               <<Qw0_i0_.x()<<" "
                                               <<Qw0_i0_.y()<<" "
                                               <<Qw0_i0_.z()<<endl;
            cout<<"ImageProjection中,coordinate_offset: "<<coordinate_offset.transpose()<<endl;
        }
        else{
            Eigen::Vector3d ENU;
            ConvertLLAToENU(Eigen::Vector3d(latitude, longitude, altitude), 
                                                    Eigen::Vector3d(gpsRawMsg->latitude, gpsRawMsg->longitude, gpsRawMsg->altitude), 
                                                    &ENU);
            gps_odom.pose.pose.position.x = ENU(0);
            gps_odom.pose.pose.position.y = ENU(1);
            gps_odom.pose.pose.position.z = ENU(2);
            gps_odom.pose.pose.orientation.w = gpsRawMsg->position_covariance[0];
            gps_odom.pose.pose.orientation.x = gpsRawMsg->position_covariance[1];
            gps_odom.pose.pose.orientation.y = gpsRawMsg->position_covariance[2];
            gps_odom.pose.pose.orientation.z = gpsRawMsg->position_covariance[3];
            gps_odom.pose.covariance = 
            {0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            pubgpsodom.publish(gps_odom);
        }

        Eigen::Quaternionf Qw0_ik_(gpsRawMsg->position_covariance[0], gpsRawMsg->position_covariance[1], gpsRawMsg->position_covariance[2], gpsRawMsg->position_covariance[3]);
        Eigen::Vector3f pose = - Qw0_ik_.toRotationMatrix() * extRot.transpose().cast<float>() * extTrans.cast<float>()
                        + Eigen::Vector3d(gps_odom.pose.pose.position.x, gps_odom.pose.pose.position.y, gps_odom.pose.pose.position.z).cast<float>() 
                        + coordinate_offset
                        + Qw0_i0_.cast<float>() * lever_arm.cast<float>() - Qw0_ik_ * lever_arm.cast<float>();
        // cout<<"In imageprojection: "<<pose.transpose()<<endl;
        Eigen::Quaternionf Qw0_lk_(Qw0_ik_.toRotationMatrix() * extRot.transpose().cast<float>());
        nav_msgs::Odometry gps_odom2;
        gps_odom2.header.stamp = gpsRawMsg->header.stamp;
        gps_odom2.header.frame_id = "odom";
        gps_odom2.pose.pose.orientation.w = Qw0_lk_.w();
        gps_odom2.pose.pose.orientation.x = Qw0_lk_.x();
        gps_odom2.pose.pose.orientation.y = Qw0_lk_.y();
        gps_odom2.pose.pose.orientation.z = Qw0_lk_.z();
        gps_odom2.pose.pose.position.x = pose(0);
        gps_odom2.pose.pose.position.y = pose(1);
        gps_odom2.pose.pose.position.z = pose(2);
        pubENUgps.publish(gps_odom2);

        file_gps.setf(std::ios::fixed, std::_S_floatfield);
        file_gps << gps_odom2.header.stamp.toSec() << " " 
                << gps_odom2.pose.pose.position.x << " "
                << gps_odom2.pose.pose.position.y << " "
                << gps_odom2.pose.pose.position.z << " "
                << gps_odom2.pose.pose.orientation.x << " "
                << gps_odom2.pose.pose.orientation.y << " "
                << gps_odom2.pose.pose.orientation.z << " "
                << gps_odom2.pose.pose.orientation.w << std::endl;

        //给点云提供全局位姿消息
        std::lock_guard<std::mutex> lock3(gpsLock);
        gpsQueue.push_back(gps_odom);
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        if (!cachePointCloud(laserCloudMsg))
            return;
        if (!deskewInfo())
            return;
        projectPointCloud();

        cloudExtraction();
        
        publishClouds();
        
        resetParameters();
    }

    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;

        // convert cloud
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();
        if (sensor == SensorType::VELODYNE)
        {   
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
        }
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else if (sensor == SensorType::RSLIDAR)
        {   
            // Convert to Rslidar format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpRslidarCloudIn);

            laserCloudIn->points.resize(tmpRslidarCloudIn->size());
            laserCloudIn->is_dense = tmpRslidarCloudIn->is_dense;
            for (int i = 0; i < tmpRslidarCloudIn->size(); i++)
            {
                auto &src = tmpRslidarCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.timestamp - tmpRslidarCloudIn->points[0].timestamp;
            }
        }
        else if (sensor == SensorType::RSLIDAR32)
        {   
            // Convert to Rslidar32 format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpRslidarCloudIn);
            laserCloudIn->points.resize(tmpRslidarCloudIn->size());
            laserCloudIn->is_dense = tmpRslidarCloudIn->is_dense;

            for(size_t i = 0; i < tmpRslidarCloudIn->size(); i++){
                auto &new_point = laserCloudIn->points[i];

                if(has_nan(tmpRslidarCloudIn->points[i])) {
                    new_point.x = 0;
                    new_point.y = 0;
                    new_point.y = 0;
                }

                new_point.x = tmpRslidarCloudIn->points[i].x;
                new_point.y = tmpRslidarCloudIn->points[i].y;
                new_point.z = tmpRslidarCloudIn->points[i].z;
                new_point.intensity = tmpRslidarCloudIn->points[i].intensity;
                new_point.ring = tmpRslidarCloudIn->points[i].ring;
                new_point.time = tmpRslidarCloudIn->points[i].timestamp - tmpRslidarCloudIn->points[0].timestamp;
            }
        }
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        // get timestamp
        cloudHeader = currentCloudMsg.header;
        timeScanCur = cloudHeader.stamp.toSec();
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

        // check dense flag
        if (laserCloudIn->is_dense == false)
        {   
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring" || sensor == SensorType::RSLIDAR32)
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // check point time
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t" || field.name == "timestamp")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);
        std::lock_guard<std::mutex> lock3(gpsLock);

        // make sure IMU data available for the scan
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        imuDeskewInfo();

        odomDeskewInfo();

        //有问题
        if(useGps)
        {
            if(!gpsENUInterpolate()) {
                ROS_WARN("Initial Clouds with no GPS align!"); 
                return false;
            }
        }

        return true;
    }

    bool gpsENUInterpolate(){
        nav_msgs::Odometry frontgpsOdom;
        while (!gpsQueue.empty())
        {
            if (gpsQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
            {
                // frontgpsOdom = gpsQueue.front();
                gpsQueue.pop_front();
            }
            else
                break;
        }
        // gpsQueue.push_front(frontgpsOdom);

        if (gpsQueue.empty())
            return false;

        if (gpsQueue.front().header.stamp.toSec() > timeScanCur)
            return false;

        // get start odometry at the beinning of the scan
        //nav_msgs::Odometry startOdomMsg;

        auto gps_ptr = gpsQueue.begin();
        for (; gps_ptr != gpsQueue.end(); gps_ptr++)
        {   
            if (gps_ptr+1 < gpsQueue.end()) 
                if (gps_ptr->header.stamp.toSec() <= timeScanCur && (gps_ptr+1)->header.stamp.toSec() >= timeScanCur)
                    break;
            else return false;
        }
        if (gps_ptr != gpsQueue.begin())
            gpsQueue.erase(gpsQueue.begin(), gps_ptr);

        Eigen::Quaterniond q_start = Eigen::Quaterniond(gpsQueue.front().pose.pose.orientation.w,
                                                        gpsQueue.front().pose.pose.orientation.x,
                                                        gpsQueue.front().pose.pose.orientation.y,
                                                        gpsQueue.front().pose.pose.orientation.z);
        Eigen::Quaterniond q_end = Eigen::Quaterniond(gpsQueue[1].pose.pose.orientation.w,
                                                        gpsQueue[1].pose.pose.orientation.x,
                                                        gpsQueue[1].pose.pose.orientation.y,
                                                        gpsQueue[1].pose.pose.orientation.z);
        Eigen::Vector3d t_start = Eigen::Vector3d(gpsQueue.front().pose.pose.position.x,
                                                  gpsQueue.front().pose.pose.position.y,
                                                  gpsQueue.front().pose.pose.position.z);
        Eigen::Vector3d t_end = Eigen::Vector3d(gpsQueue[1].pose.pose.position.x,
                                                  gpsQueue[1].pose.pose.position.y,
                                                  gpsQueue[1].pose.pose.position.z);
        double scale = (timeScanCur - gpsQueue.front().header.stamp.toSec()) / 
                       (gpsQueue[1].header.stamp.toSec() - gpsQueue.front().header.stamp.toSec());
        //q_inter是在lidar时刻的imu姿态
        Eigen::Quaterniond q_inter = getInterpolatedAttitude(q_start, q_end, scale);
        Eigen::Vector3d t_inter = getInterpolatedTrans(t_start, t_end, scale);
        t_inter += - q_inter.toRotationMatrix() * extRot.transpose() * extTrans
                   + coordinate_offset.cast<double>() + Qw0_i0_ * lever_arm - q_inter * lever_arm;

        // q_inter = q_start;
        // t_inter = t_start + 
        //           - q_inter.toRotationMatrix() * extRot.transpose() * extTrans
        //            + coordinate_offset.cast<double>() + Qw0_i0_ * lever_arm - q_inter * lever_arm;;

        cloudInfo.mapQlk[0] = q_inter.w();
        cloudInfo.mapQlk[1] = q_inter.x();
        cloudInfo.mapQlk[2] = q_inter.y();
        cloudInfo.mapQlk[3] = q_inter.z();

        // double w_sigma = 1.0;
        // cv::RNG rng;

        cloudInfo.maptlk[0] = t_inter[0]/*+rng.gaussian(w_sigma)*/;
        cloudInfo.maptlk[1] = t_inter[1]/*+rng.gaussian(w_sigma)*/;
        cloudInfo.maptlk[2] = t_inter[2];

        return true;
    }

    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;

        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();
            // cout<<"CURRENT IMU TIME: "<<setprecision(20)<<currentImuTime<<endl;

            // get roll, pitch, and yaw estimation for this scan
            if(useGps)
            {
                if (currentImuTime <= timeScanCur)
                    imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);
            }

            if (currentImuTime > timeScanEnd + 0.01)
                break;

            //only calculate relative time
            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }

    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        //nav_msgs::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];
            if(i+1<=(int)odomQueue.size()-1){
            if (ROS_TIME(&odomQueue[i]) < timeScanCur && ROS_TIME(&odomQueue[i+1]) < timeScanCur)
                continue;
            else
                break;}
            // if (ROS_TIME(&startOdomMsg) < timeScanCur)
            //     continue;
            // else
            //     break;
        }

        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;

        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;

        // nav_msgs::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }
        if(ROS_TIME(&endOdomMsg) < timeScanEnd) return;

        // if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
        if (int(round(startOdomMsg.pose.covariance[0])) == 1 && int(round(endOdomMsg.pose.covariance[0]))==1)
            return;

        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        transBtSE = transBegin.inverse() * transEnd;
        Eigen::AngleAxisf AngleAxis(transBtSE.rotation().matrix());
        rad_BtSE = AngleAxis.angle() * AngleAxis.axis();

        //float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBtSE, odomIncreX, odomIncreY, odomIncreZ, odomIncreRoll, odomIncrePitch, odomIncreYaw);     

        odomDeskewFlag = true;
    }

    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;
        //MINE
        if( endOdomMsg.header.stamp.toSec() - startOdomMsg.header.stamp.toSec()>0 &&
            cloudInfo.odomAvailable == true && odomDeskewFlag == true){
            // std::cout<<"USING CORRECTION NOW!"<<endl;
            tf::Quaternion startOrientaion;
            tf::Quaternion endOrientation;
            tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, startOrientaion);
            tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, endOrientation);
            // Eigen::Quaternionf qstart = Eigen::Quaternionf(startOrientaion.w(),startOrientaion.x(),
            //                                                startOrientaion.y(),startOrientaion.z());
            // Eigen::Quaternionf qend = Eigen::Quaternionf(endOrientation.w(),endOrientation.x(),
            //                                                endOrientation.y(),endOrientation.z());
            float alpha = (pointTime - startOdomMsg.header.stamp.toSec())
                        / (endOdomMsg.header.stamp.toSec() - startOdomMsg.header.stamp.toSec());
            // Eigen::Matrix3f Rk = Eigen::Matrix3f(qstart).transpose() * Rodrigues(alpha * rad_BtSE) * Eigen::Matrix3f(qstart);
            Eigen::Matrix3f Rk = Rodrigues(alpha * rad_BtSE);
            Eigen::Vector3f rad = Rotation2Euler(Rk);
            // std::cout<<"a is: "<<alpha<<std::endl;
            // std::cout<<"pointTime - startOdomMsg.header.stamp.toSec() is:"<<pointTime - startOdomMsg.header.stamp.toSec()<<endl;
            // std::cout<<"endOdomMsg.header.stamp.toSec() - startOdomMsg.header.stamp.toSec() is:"<<endOdomMsg.header.stamp.toSec() - startOdomMsg.header.stamp.toSec()<<endl;
            // std::cout<<"rad_BtSE is: "<<endl<<rad_BtSE<<std::endl;
            // std::cout<<Rodrigues(alpha * rad_BtSE)<<std::endl;
            *rotXCur = rad(0);
            *rotYCur = rad(1);
            *rotZCur = rad(2);
            // std::cout<<"the delta euler are: "<<rad(0)<<"  "<<rad(1)<<"  "<<rad(2)<<endl;
        }
        else{
        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            // std::cout<<"USING INITIALCORRECTION NOW!"<<endl;
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
            // std::cout<<"the delta euler are: "<<*rotXCur<<"  "<<*rotYCur<<"  "<<*rotZCur<<endl;
        }}
        //END

        // int imuPointerFront = 0;
        // while (imuPointerFront < imuPointerCur)
        // {
        //     if (pointTime < imuTime[imuPointerFront])
        //         break;
        //     ++imuPointerFront;
        // }

        // if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        // {
        //     *rotXCur = imuRotX[imuPointerFront];
        //     *rotYCur = imuRotY[imuPointerFront];
        //     *rotZCur = imuRotZ[imuPointerFront];
        // } else {
        //     int imuPointerBack = imuPointerFront - 1;
        //     double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        //     double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        //     *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
        //     *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
        //     *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        // }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.
        //the initial one get initial RPY guess and the later one get back RPY
        if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
            return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);
        float ratio = relTime / (endOdomMsg.header.stamp.toSec() - startOdomMsg.header.stamp.toSec());

        *posXCur = ratio * odomIncreX;
        *posYCur = ratio * odomIncreY;
        *posZCur = ratio * odomIncreZ;
    }

    PointType deskewPoint(PointType *point, double relTime)
    {
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        double pointTime = timeScanCur + relTime;

        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;
            
            int rowIdn;
             rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            if (rowIdn % downsampleRate != 0)
                continue;

            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            static float ang_res_x = 360.0/float(Horizon_SCAN);
            int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time + timeScanCur - startOdomMsg.header.stamp.toSec());

            //这里用rangeMat代替了lego_loam中用fullCloud的点的intensity代表range的功能
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }

    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i)
        {
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }
    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo.publish(cloudInfo);

        // //for test
        // sensor_msgs::PointCloud2 tempCloud;
        // pcl::toROSMsg(*laserCloudIn, tempCloud);
        // tempCloud.header.stamp = cloudHeader.stamp;
        // tempCloud.header.frame_id = lidarFrame;
        // pubModifiedRawCloud.publish(tempCloud);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    
    return 0;
}
