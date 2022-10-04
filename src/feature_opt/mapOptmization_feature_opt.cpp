#include "utility.h"

#include <dynamic_reconfigure/server.h>
#include "map_localization/MAP_LOCALIZATION_PARAMConfig.h"

#include "lidarFactor.hpp"
#include "map_localization/cloud_info.h"
#include "map_localization/save_map.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <LocalCartesian.hpp>

#include "Scancontext.h"
#include "registrations.hpp"
#include <omp.h>

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

#define USE_MAP true

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;

class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;


    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;
    //FOR CHANGAN
    int Key_point_count;
    std::ofstream file_lidar;
    std::ofstream file_time;
    //END

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;

    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subLoop;

    ros::ServiceServer srvSaveMap;

    std::deque<nav_msgs::Odometry> gpsQueue;
    map_localization::cloud_info cloudInfo;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudRawLast;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudRawLastDS;

    //less feature points
    pcl::PointCloud<PointType>::Ptr laserCloudCornerlessLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurflessLast;
    //END

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterSC;
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    
    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;
    std::mutex mtxLoopInfo;
    std::mutex gpsLock;

    bool isDegenerate = false;
    cv::Mat matP;
    //LM algorithm coefficients
    // double u, v;
    cv::Mat matAtB;
    cv::Mat matX;
    cv::Mat matD;
    // double residual1, residual2;
    // int valid_corner_num, valid_surf_num;
    // vector<pair<PointType, PointType>> CornerParam;
    // vector<Eigen::Vector4f> SurfParam;
    bool keep_opt = true;
    //END

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    // int gpsSequence = 0;
    // double latitude, longitude, altitude;

    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer; // from new to old
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::Float64MultiArray> loopInfoVec;

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

    SCManager scManager;

    gtsam::Pose3 exT_gt;
    Eigen::Quaternionf q_exrot;
    Eigen::Affine3f exT_affine;
    Eigen::Vector3f coordinate_offset;

    dynamic_reconfigure::Server<map_localization_dynamic_param::MAP_LOCALIZATION_PARAMConfig> server;
    dynamic_reconfigure::Server<map_localization_dynamic_param::MAP_LOCALIZATION_PARAMConfig>::CallbackType f;
    bool USEGPS = false;
    bool USEFULLFEATURE = true;

    //map
    pcl::PointCloud<PointType>::Ptr MapSurfClouds;
    pcl::PointCloud<PointType>::Ptr MapCornerClouds;
    pcl::PointCloud<PointType>::Ptr MapCloudsDS;
    pcl::VoxelGrid<PointType> downSizeFilterMapCloud;
    // pclomp::NormalDistributionsTransform<PointType, PointType>::Ptr ndt_omp;
    boost::shared_ptr<pcl::Registration<pcl::PointXYZI, pcl::PointXYZI>> reg;
    vector<pcl::PointCloud<PointType>::Ptr> RawCloudKeyFrames;
    //submap
    pcl::PointCloud<PointType>::Ptr SubMapOriginCloud;
    vector<pcl::PointCloud<PointType>::Ptr> SubMapSetCorner;
    vector<pcl::PointCloud<PointType>::Ptr> SubMapSetSurf;
    pcl::KdTreeFLANN<PointType>::Ptr SubMapOriginKdtree;
    //SC重定位
    vector<PointTypePose> KeyFramesPoseFromMap;
    vector<pcl::PointCloud<PointType>::Ptr> SCRawCloudKeyFrames;
    pcl::KdTreeFLANN<PointType>::Ptr KeyFramesPoseFromMapKdtree;
    bool first_lidar_frame = true;
    bool gps_usage = false;

    //feature test
    pcl::PointCloud<PointType>::Ptr CorCorrespondence;
    pcl::PointCloud<PointType>::Ptr SurCorrespondence;
    pcl::PointCloud<PointType>::Ptr CornerSelected;
    pcl::PointCloud<PointType>::Ptr SurfSelected;
    int CorCorresCount;
    int SurCorresCount;
    ros::Publisher pubCorCorres;
    ros::Publisher pubSurCorres;
    ros::Publisher pubCorSelected;
    ros::Publisher pubSurSelected;
    omp_lock_t lock1;
    omp_lock_t lock2;
    //END


    mapOptimization()
    {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        Key_point_count = 0;

        f = boost::bind(&mapOptimization::callback,this, _1); //绑定回调函数
        server.setCallback(f); //为服务器设置回调函数， 节点程序运行时会调用一次回调函数来输出当前的参数配置情况

        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1);
        //lio_sam/mapping/odometry is the odometry with pose correction, thus it has no drift.
        //lio_sam/mapping/odometry_incremental is pure lidar odometry without any correction, thus it has drifts.
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry", 1);
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry_incremental", 1);
        pubPath                     = nh.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1);

        subCloud = nh.subscribe<map_localization::cloud_info>("lio_sam/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        // subGPSRAW = nh.subscribe<sensor_msgs::NavSatFix>(gpsRawTopic, 200, &mapOptimization::gpsRawHandler, this, ros::TransportHints().tcpNoDelay());
        subGPS   = nh.subscribe<nav_msgs::Odometry> ("odometry/enu", 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        subLoop  = nh.subscribe<std_msgs::Float64MultiArray>("lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());
        
        srvSaveMap  = nh.advertiseService("lio_sam/save_map", &mapOptimization::saveMapService, this);

        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1);

        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_local", 1);
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

        //feature test
        pubCorCorres = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/CorCorres", 1);
        pubSurCorres = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/SurCorres", 1);
        pubCorSelected = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/CorSelected", 1);
        pubSurSelected = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/SurSelected", 1);
        omp_init_lock(&lock1);
        omp_init_lock(&lock2);
        //END

        //map
        reg = select_registration_method(nh);

        file_lidar.open(Odom_Path+"lidar_odom.csv", std::ios::app);
        file_time.open(Odom_Path+"operation_time.csv", std::ios::app);

        const float kSCFilerSize = 0.25;
        downSizeFilterSC.setLeafSize(kSCFilerSize, kSCFilerSize, kSCFilerSize);
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization
        downSizeFilterMapCloud.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

        exTinitialization();

        allocateMemory();

        if(USE_SUBMAP) SubMapLoad();
        else MapLoad();
    }

    ~mapOptimization(){
        file_lidar.close();
        file_time.close();
    }

    void allocateMemory()
    {   
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudRawLast.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization
        laserCloudRawLastDS.reset(new pcl::PointCloud<PointType>());

        //less feature points
        laserCloudCornerlessLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurflessLast.reset(new pcl::PointCloud<PointType>());
        //END

        //feature test
        CorCorrespondence.reset(new pcl::PointCloud<PointType>());
        SurCorrespondence.reset(new pcl::PointCloud<PointType>());
        CornerSelected.reset(new pcl::PointCloud<PointType>());
        SurfSelected.reset(new pcl::PointCloud<PointType>());
        //END

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        // //LM algorithm
        // CornerParam.reserve(N_SCAN * Horizon_SCAN);
        // SurfParam.reserve(N_SCAN * Horizon_SCAN);
        //END

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        //map
        MapCloudsDS.reset(new pcl::PointCloud<PointType>());
        MapSurfClouds.reset(new pcl::PointCloud<PointType>());
        MapCornerClouds.reset(new pcl::PointCloud<PointType>());
        SubMapOriginCloud.reset(new pcl::PointCloud<PointType>());
        SubMapOriginKdtree.reset(new pcl::KdTreeFLANN<PointType>());
        SubMapSetCorner.reserve(SubMapNum);
        SubMapSetSurf.reserve(SubMapNum);


        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    }

    void callback(map_localization_dynamic_param::MAP_LOCALIZATION_PARAMConfig &config)
    {
        ROS_INFO("Reconfigure Request GPS: %s", config.useGps?"True":"False");
        ROS_INFO("Reconfigure Request FeatureFull: %s", config.useGps?"True":"False");
        USEGPS = config.useGps;
        USEFULLFEATURE = config.useFullFeature;
    }

    void laserCloudInfoHandler(const map_localization::cloud_infoConstPtr& msgIn)
    {
        // extract time stamp
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = msgIn->header.stamp.toSec();
        // cout<<"time "<<std::setprecision(15)<<timeLaserInfoCur<<endl;

        // extract info and feature cloud
        cloudInfo = *msgIn;

        // TicToc t_corner;
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);
        // t_corner.toc("Corner written time");
        // TicToc t_surf;
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);
        
        // TicToc t_Raw;
        pcl::fromROSMsg(msgIn->cloud_deskewed, *laserCloudRawLast);
        // t_Raw.toc("Raw cloud written time");

        //less feature points
        pcl::fromROSMsg(msgIn->cloud_corner_Super, *laserCloudCornerlessLast);
        pcl::fromROSMsg(msgIn->cloud_surface_Super, *laserCloudSurflessLast);
        cout<<"Edge and Surf points are: "<<laserCloudCornerlessLast->size()
                                          <<" and "<<laserCloudSurflessLast->size()<<endl;
        //END

        std::lock_guard<std::mutex> lock(mtx);

        static double timeLastProcessing = -1;
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            timeLastProcessing = timeLaserInfoCur;

            TicToc t_All;
            updateInitialGuessMap();
            TicToc t_sample;
            cout<<endl;

            TicToc t_localmap;
            if(USE_SUBMAP)  extractSurroundingKeyFramesSubMap();
            t_localmap.toc("local map");
            cout<<endl;

            downsampleCurrentScan();
            t_sample.toc("sample map");
            cout<<endl;
            
            TicToc t_scan2map;
            scan2MapOptimization();
            t_scan2map.toc("palnar and edge opt");
            cout<<endl;

            TicToc t_gtsam;
            saveKeyFramesAndFactor();
            t_gtsam.toc("whole gtsam");
            cout<<endl;

            TicToc t_correct;
            correctPoses();
            t_gtsam.toc("correct pose");
            cout<<endl;

            publishOdometry();

            publishFrames();

            t_All.toc("All operation time");
            cout<<endl<<endl;

            file_time.setf(std::ios::fixed, std::_S_floatfield);
            file_time << timeLastProcessing << ","
                      << t_localmap.elapsed_ms << ","
                      << t_sample.elapsed_ms << ","
                      << t_scan2map.elapsed_ms << ","
                      << t_gtsam.elapsed_ms << ","
                      << t_All.elapsed_ms << ","
                      <<std::endl;
        }
    }

    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {   
        std::lock_guard<std::mutex> lock3(gpsLock);
        gpsQueue.push_back(*gpsMsg);
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    void MapLoad(){
        cout<<"map path = "<<map_path<<endl;
        string SurfMap_path = map_path + "SurfMap.pcd";
        string CornerMap_path = map_path + "CornerMap.pcd";
        TicToc t_mapload;
        pcl::io::loadPCDFile (SurfMap_path, *MapSurfClouds);
        pcl::io::loadPCDFile (CornerMap_path, *MapCornerClouds);
        cout<<SurfMap_path<<": "<<MapSurfClouds->size()<<endl;
        cout<<CornerMap_path<<": "<<MapCornerClouds->size()<<endl;
        t_mapload.toc("LOAD MAP Success");
        cout<<endl;

        downSizeFilterCorner.setInputCloud(MapCornerClouds);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(MapSurfClouds);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // downSizeFilterMapCloud.setInputCloud(MapClouds);
        // downSizeFilterMapCloud.filter(*MapCloudsDS);

        TicToc t_mapkdtree;
        kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);
        kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
        t_mapkdtree.toc("Set Map Kdtree");

        //record key frames pose
        // parsing pose info
        TicToc t_mapkSC;
        std::ifstream pose_file_handle (map_path + "pose.txt");
        int num_poses {0};
        std::string strOneLine;
        while (getline(pose_file_handle, strOneLine)) 
        {
            // str to vec
            std::vector<float> ith_pose_vec = splitPoseLine(strOneLine, ' ');
            if(ith_pose_vec.size() == 6) {
                PointXYZIRPYT pose;
                pose.x = ith_pose_vec[0];
                pose.y = ith_pose_vec[1];
                pose.z = ith_pose_vec[2];
                pose.roll = ith_pose_vec[3];
                pose.pitch = ith_pose_vec[4];
                pose.yaw = ith_pose_vec[5];
                KeyFramesPoseFromMap.push_back(pose);
            }
            num_poses++;
        }

        for(int i=0; i<num_poses; i++){
            stringstream inter;
            string s_temp = std::to_string(i);
            inter << setw(5) << setfill('0') << s_temp;
            inter >> s_temp;

            pcl::PointCloud<PointType>::Ptr points(new pcl::PointCloud<PointType>());
            pcl::io::loadPCDFile<PointType> (map_path+"RawSubMap/"+s_temp+".pcd", *points);
            SCRawCloudKeyFrames.push_back(points);
            scManager.makeAndSaveScancontextAndKeys(*points);
        }
        t_mapkSC.toc("Set SC Map");
    }

    void SubMapLoad(){

        TicToc t_submapload;
        //导入子地图坐标原点信息
        ifstream inFile(SubMapInfo, ios::in);
        if (!inFile)
        {
            cout << "打开 " << SubMapInfo << " 文件失败!" << endl;
            exit(1);
        }
        string line;
        string field;
        double tlx, tly, tlz, roll, pitch, yaw;
        while (getline(inFile, line))//getline(inFile, line)表示按行读取CSV文件中的数据
        {   
            string field;
            istringstream sin(line); //将整行字符串line读入到字符串流sin中

            //lidar位姿
            getline(sin, field, ' ');   tlx = atof(field.c_str());
            getline(sin, field, ' ');   tly = atof(field.c_str());
            getline(sin, field, ' ');   tlz = atof(field.c_str());
            getline(sin, field, ' ');   roll = atof(field.c_str());
            getline(sin, field, ' ');   pitch = atof(field.c_str());
            getline(sin, field, ' ');   yaw = atof(field.c_str());

            //将雷达的平移量作为point输入到kdtree中
            PointType point;
            point.x = tlx;  point.y = tly;  point.z = tlz;
            cout<<"雷达坐标: "<<point.x<<" "<<point.y<<" "<<point.z<<endl;
            SubMapOriginCloud->push_back(point);
        }
        inFile.close();
        SubMapOriginKdtree->setInputCloud(SubMapOriginCloud);

        //导入子地图
        for (int i = 0; i < SubMapNum; i++){
            std::string s = std::to_string(i);
            std::string pcd_filename_corner = map_path + "/CornerSubMapFinal/submap_" + s + ".pcd";
            std::string pcd_filename_surf = map_path + "/SurfSubMapFinal/submap_" + s + ".pcd";
            pcl::PointCloud<PointType>::Ptr cloud_temp_corner(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr cloud_temp_surf(new pcl::PointCloud<PointType>());
            pcl::io::loadPCDFile (pcd_filename_corner, *cloud_temp_corner);
            pcl::io::loadPCDFile (pcd_filename_surf, *cloud_temp_surf);
            cout<<pcd_filename_corner<<": "<<cloud_temp_corner->size()<<endl;
            cout<<pcd_filename_corner<<": "<<cloud_temp_surf->size()<<endl;
            SubMapSetCorner.push_back(cloud_temp_corner);
            SubMapSetSurf.push_back(cloud_temp_surf);
        }

        //SC
        std::ifstream pose_file_handle (map_path + "pose.txt");
        int num_poses {0};
        std::string strOneLine;
        while (getline(pose_file_handle, strOneLine)) 
        {
            // str to vec
            std::vector<float> ith_pose_vec = splitPoseLine(strOneLine, ' ');
            if(ith_pose_vec.size() == 6) {
                PointXYZIRPYT pose;
                pose.x = ith_pose_vec[0];
                pose.y = ith_pose_vec[1];
                pose.z = ith_pose_vec[2];
                pose.roll = ith_pose_vec[3];
                pose.pitch = ith_pose_vec[4];
                pose.yaw = ith_pose_vec[5];
                KeyFramesPoseFromMap.push_back(pose);
            }
            num_poses++;
        }

        for(int i=0; i<num_poses; i++){
            stringstream inter;
            string s_temp = std::to_string(i);
            inter << setw(5) << setfill('0') << s_temp;
            inter >> s_temp;

            pcl::PointCloud<PointType>::Ptr points(new pcl::PointCloud<PointType>());
            pcl::io::loadPCDFile<PointType> (map_path+"RawSubMap/"+s_temp+".pcd", *points);
            SCRawCloudKeyFrames.push_back(points);
            scManager.makeAndSaveScancontextAndKeys(*points);
        }

        t_submapload.toc("LOAD SUBMAP Success");
    }

    void exTinitialization(){
        Eigen::Quaterniond q = Eigen::Quaterniond(extRot);
        q_exrot = Eigen::Quaternionf(float(q.w()), float(q.x()), float(q.y()), float(q.z()));
        exT_gt = gtsam::Pose3(gtsam::Rot3(q_exrot.w(), q_exrot.x(), q_exrot.y(), q_exrot.z()), 
                              gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));
        Eigen::Matrix4f exT_matrix = Eigen::Matrix4f::Identity();
        exT_matrix.block<3,3>(0,0) = q_exrot.toRotationMatrix();
        exT_matrix(0,3) = static_cast<float>(extTrans(0));
        exT_matrix(1,3) = static_cast<float>(extTrans(1));
        exT_matrix(2,3) = static_cast<float>(extTrans(2));
        exT_affine = exT_matrix;
    }













    bool saveMapService(map_localization::save_mapRequest& req, map_localization::save_mapResponse& res)
    {
      string saveMapDirectory;

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files ..." << endl;
      if(req.destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
      else saveMapDirectory = std::getenv("HOME") + req.destination;
      cout << "Save destination: " << saveMapDirectory << endl;
      // create directory and remove old files;
      int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
      unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
      // save key frame transformations
      pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
      // extract global point cloud map
      pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
      for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
          *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
          *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
          cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
      }

      if(req.resolution != 0)
      {
        cout << "\n\nSave resolution: " << req.resolution << endl;

        // down-sample and save corner cloud
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        downSizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterCorner.filter(*globalCornerCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
        // down-sample and save surf cloud
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
      }
      else
      {
        // save corner cloud
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
        // save surf cloud
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
      }

      // save global point cloud map
      *globalMapCloud += *globalCornerCloud;
      *globalMapCloud += *globalSurfCloud;

      int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
      res.success = ret == 0;

      downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
      downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files completed\n" << endl;

      return true;
    }

    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }

        if (savePCD == false)
            return;

        map_localization::save_mapRequest  req;
        map_localization::save_mapResponse res;

        if(!saveMapService(req, res)){
            cout << "Fail to save map" << endl;
        }
    }

    void publishGlobalMap()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }











    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(loopClosureFrequency);
        while (ros::ok())
        {
            rate.sleep();
            TicToc t_loop;
            performLoopClosure();
            t_loop.toc("loop closure");
            cout<<endl;
            visualizeLoopClosure();
        }
    }

    void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2)
            return;

        loopInfoVec.push_back(*loopMsg);

        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }

    void performLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        int loopKeyCur;
        int loopKeyPre;
        // if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
                return;

        // auto detectResult = scManager.detectLoopClosureID();
        // loopKeyCur = copy_cloudKeyPoses3D->size()-1;
        // loopKeyPre = detectResult.first;
        // float yawDiffRad = detectResult.second;
        // if(loopKeyPre == -1) return;

        // std::cout << "SC loop found! between " << loopKeyCur << " and " << loopKeyPre << "." << std::endl;

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // publish corrected cloud
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();
        // transform from world origin to wrong pose
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        // Vector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        // add loop constriant
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        auto it = loopIndexContainer.find(loopKeyCur);
        //if has been found before, no loop will run again
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {   cout<<"loopclosuredistance time "<<std::setprecision(15)<<timeLaserInfoCur<<endl;
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty())
            return false;

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();

        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        if (loopKeyCur == loopKeyPre)
            return false;

        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void visualizeLoopClosure()
    {
        if (loopIndexContainer.empty())
            return;
        
        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }




    //SC重定位
    void LocalizeNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int poseSize = KeyFramesPoseFromMap.size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= poseSize )
                continue;
            *nearKeyframes += *transformPointCloud(SCRawCloudKeyFrames[keyNear], &KeyFramesPoseFromMap[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // transform points to "key" local frame downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        // cloud_temp->resize(nearKeyframes->size());
        // Eigen::Affine3f transCur = pcl::getTransformation(KeyFramesPoseFromMap[key].x, KeyFramesPoseFromMap[key].y, 
        //                                                   KeyFramesPoseFromMap[key].z, KeyFramesPoseFromMap[key].roll, 
        //                                                   KeyFramesPoseFromMap[key].pitch, KeyFramesPoseFromMap[key].yaw).inverse();
        
        // #pragma omp parallel for num_threads(numberOfCores)
        // for (int i = 0; i < nearKeyframes->size(); ++i)
        // {
        //     PointType pointFrom = nearKeyframes->points[i];
        //     nearKeyframes->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        //     nearKeyframes->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        //     nearKeyframes->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        //     nearKeyframes->points[i].intensity = pointFrom.intensity;
        // }

        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        nearKeyframes->clear();
        *nearKeyframes = *cloud_temp;
    }

    void updateInitialGuessMap(){
        // if(!first_lidar_frame) incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);
        if(!cloudInfo.odomAvailable) incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        //Use SC
        if(!cloudInfo.odomAvailable)
        {   
            TicToc t_SC_localization;
            ROS_INFO("***Using SC Mode***");
            scManager.makeAndSaveScancontextAndKeys(*laserCloudRawLast);
            // pcl::io::savePCDFileBinary(map_path + "RawClouds.pcd", *laserCloudRawLast);

            auto detectResult = scManager.detectLoopClosureID();
            int SCKeyPre = detectResult.first;
            float yawdiff = detectResult.second;
            if(SCKeyPre == -1) {
                gps_usage = true;
                ROS_WARN("Can't find SC Correspondence");
            }
            if(!gps_usage)
            {   
                pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
                LocalizeNearKeyframes(prevKeyframeCloud, SCKeyPre, 2);

                //对原始点云进行下采样
                pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
                downSizeFilterICP.setInputCloud(laserCloudRawLast);
                downSizeFilterICP.filter(*cloud_temp);
                laserCloudRawLast->clear();
                *laserCloudRawLast = *cloud_temp;
                // pcl::io::savePCDFileBinary(map_path + "SCpointcloudsSource.pcd", *laserCloudRawLast);
                // pcl::io::savePCDFileBinary(map_path + "SCpointcloudsTarget.pcd", *prevKeyframeCloud);

                if (laserCloudRawLast->size() < 300 || prevKeyframeCloud->size() < 1000) {
                    gps_usage = true;
                    ROS_WARN("SC PC points too few");
                    }

                if(!gps_usage)
                {
                    // ICP Settings and Align clouds
                    Eigen::Affine3f prePose = pclPointToAffine3f(KeyFramesPoseFromMap[SCKeyPre]);
                    Eigen::Matrix4f Tpre2W = Eigen::Matrix4f(prePose.matrix());
                    Eigen::Matrix4f Tdyaw = Eigen::Matrix4f::Identity();
                    Eigen::AngleAxisf yaw_change(yawdiff, Eigen::Vector3f::UnitZ());

                    Tdyaw.block<3,3>(0,0) = yaw_change.toRotationMatrix().inverse();
                    Eigen::Matrix4f Tcur2W = Tpre2W * Tdyaw;


                    reg->setInputSource(laserCloudRawLast);
                    reg->setInputTarget(prevKeyframeCloud);
                    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
                    reg->align(*unused_result, Tcur2W);
                    ROS_INFO("ICP score: %f ", reg->getFitnessScore());
                    // pcl::io::savePCDFileBinary(map_path + "SCpointclouds1.pcd", *unused_result);

                    // if (reg->hasConverged() == false || reg->getFitnessScore() > historyKeyframeFitnessScore){
                    //     gps_usage = true;
                    //     ROS_WARN("ICP can't converge, its score: %d ", reg->getFitnessScore());
                    //     Eigen::Affine3f correctionLidarFrame;
                    //     correctionLidarFrame = reg->getFinalTransformation();
                    //     cout<<"ICP Transition is: "<<correctionLidarFrame(0,3)<<" "
                    //                <<correctionLidarFrame(1,3)<<" "
                    //                <<correctionLidarFrame(2,3)<<" "<<endl<<endl;
                    //     }
                    

                    if(!gps_usage){
                        // Get pose transformation
                        float x, y, z, roll, pitch, yaw;

                        Eigen::Affine3f correctionLidarFrame;
                        correctionLidarFrame = reg->getFinalTransformation();
                        Eigen::Vector3f eulerAngle_cur = correctionLidarFrame.rotation().eulerAngles(2,1,0);
                        Eigen::Vector3f eulerAngle_SC = Eigen::Matrix3f(Tpre2W.block<3,3>(0,0)).eulerAngles(2,1,0);

                        cout<<"ICP POSITION: ["<<correctionLidarFrame(0,3)<<" "
                                   <<correctionLidarFrame(1,3)<<" "
                                   <<correctionLidarFrame(2,3)<<" "
                                   <<eulerAngle_cur[0]<<" "
                                   <<eulerAngle_cur[1]<<" "
                                   <<eulerAngle_cur[2]<<"]"                                  
                                   <<endl;
                        cout<<"SC Map POSITION: "<<Tpre2W(0,3)<<" "
                                   <<Tpre2W(1,3)<<" "
                                   <<Tpre2W(2,3)<<" "
                                   <<eulerAngle_SC[0]<<" "
                                   <<eulerAngle_SC[1]<<" "
                                   <<eulerAngle_SC[2]<<" "
                                   <<endl<<endl;

                         //保存匹配图片
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr PCDcloud(new pcl::PointCloud<pcl::PointXYZRGB>());
                        for(auto pt : prevKeyframeCloud->points){
                            pcl::PointXYZRGB p;
                            p.x = pt.x; p.y = pt.y; p.z = pt.z;
                            p.r = 255; p.g = 0; p.b = 0;
                            PCDcloud->push_back(p);
                        }
                        for(auto pt : laserCloudRawLast->points){
                            pcl::PointXYZRGB p;
                            p.x = correctionLidarFrame(0,0) * pt.x + correctionLidarFrame(0,1) * pt.y + 
                                                         correctionLidarFrame(0,2) * pt.z + correctionLidarFrame(0,3);
                            p.y = correctionLidarFrame(1,0) * pt.x + correctionLidarFrame(1,1) * pt.y + 
                                                         correctionLidarFrame(1,2) * pt.z + correctionLidarFrame(1,3);
                            p.z = correctionLidarFrame(2,0) * pt.x + correctionLidarFrame(2,1) * pt.y + 
                                                         correctionLidarFrame(2,2) * pt.z + correctionLidarFrame(2,3);
                            p.r = 255; p.g = 255; p.b = 255;
                            PCDcloud->push_back(p);
                        }

                        // pcl::io::savePCDFileBinary(map_path + "SCpointclouds.pcd", *PCDcloud);

                        
                        
                        // transform from world origin to wrong pose
                        Eigen::Affine3f Tk2W = pclPointToAffine3f(KeyFramesPoseFromMap[SCKeyPre]);
                        // transform from world origin to corrected pose
                        // Eigen::Affine3f Tnow2W =Tk2W * correctionLidarFrame;// pre-multiplying -> successive rotation about a fixed frame
                        pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
                        transformTobeMapped[0] = roll;
                        transformTobeMapped[1] = pitch;
                        transformTobeMapped[2] = yaw;
                        transformTobeMapped[3] = x;
                        transformTobeMapped[4] = y;
                        transformTobeMapped[5] = z;
                        cout<<"SC Transition is: "<<x<<" "<<y<<" "<<z<<endl;
                    }
                }

                // if(first_lidar_frame) incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);
                if(cloudInfo.odomAvailable) incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

                first_lidar_frame = false;
                t_SC_localization.toc("SC_localization");
            }
        }

        if (gps_usage && first_lidar_frame && !cloudInfo.odomAvailable){
            ROS_INFO("***Using GPS Signal Mode***");
            //Use GPS signal
            double roll_temp, pitch_temp, yaw_temp;
            tf::Quaternion mapQlk_initial = tf::Quaternion(cloudInfo.mapQlk[1], cloudInfo.mapQlk[2], 
                                                        cloudInfo.mapQlk[3], cloudInfo.mapQlk[0]);
            
            tf::Matrix3x3(mapQlk_initial).getRPY(roll_temp, pitch_temp, yaw_temp);
            transformTobeMapped[0] = roll_temp;
            transformTobeMapped[1] = pitch_temp;
            transformTobeMapped[2] = yaw_temp;
            transformTobeMapped[3] = cloudInfo.maptlk[0];
            transformTobeMapped[4] = cloudInfo.maptlk[1];
            transformTobeMapped[5] = cloudInfo.maptlk[2];
            first_lidar_frame = false;
            //End
        }

        if(cloudInfo.odomAvailable && !first_lidar_frame)
        {
            ROS_INFO("***Using IMU Integration Mode***");
            transformTobeMapped[0] = cloudInfo.initialGuessRoll;
            transformTobeMapped[1] = cloudInfo.initialGuessPitch;
            transformTobeMapped[2] = cloudInfo.initialGuessYaw;
            transformTobeMapped[3] = cloudInfo.initialGuessX;
            transformTobeMapped[4] = cloudInfo.initialGuessY;
            transformTobeMapped[5] = cloudInfo.initialGuessZ;
        }
        
    }

    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    void extractSurroundingKeyFramesSubMap(){
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        PointType pose;
        pose.x = cloudInfo.maptlk[0];
        pose.y = cloudInfo.maptlk[1];
        pose.z = cloudInfo.maptlk[2];
        
        SubMapOriginKdtree->nearestKSearch(pose, KeySubMapNum, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < KeySubMapNum; i++){
            cout<<"第"<<pointSearchInd[i]<<"个子地图原点和当前帧之间的初始距离为: "<<pointSearchSqDis[i]<<endl;
            *laserCloudCornerFromMap += *SubMapSetCorner[pointSearchInd[i]];
            *laserCloudSurfFromMap += *SubMapSetSurf[pointSearchInd[i]];
        }

        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();
    }

    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
        for(auto& pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        // also extract some latest key frames in case the robot rotates in one position
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(surroundingKeyPosesDS);
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // transformed cloud available
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;

                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
            
        }

        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        //CHANGED
        cout<<"container size is: "<<laserCloudMapContainer.size()<<endl;
        if (static_cast<int>(laserCloudMapContainer.size()) > containersize){
            laserCloudMapContainer.clear();
            cout<<"Deleting map cache!"<<endl;
        }
    }

    void extractSurroundingKeyFrames()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return; 

        extractNearby();
    }

    void downsampleCurrentScan()
    {
         //Downsample cloud from de-skewed scan
        laserCloudRawLastDS->clear();
        downSizeFilterSC.setInputCloud(laserCloudRawLast);
        downSizeFilterSC.filter(*laserCloudRawLastDS);


        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
        
        // cout<<"Corner number is "<<laserCloudCornerLastDSNum<<endl;
        // cout<<"Surf number is "<<laserCloudSurfLastDSNum<<endl;
    }

    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    void cornerOptimization()
    {   
        updatePointAssociateToMap();

        //Switch to Super Feature
        if(USEFULLFEATURE)
        {
            #pragma omp parallel for num_threads(numberOfCores)
            for (int i = 0; i < laserCloudCornerLastDSNum; i++)
            {
                {
                    PointType pointOri, pointSel, coeff;
                    std::vector<int> pointSearchInd;
                    std::vector<float> pointSearchSqDis;

                    //less feature points
                    pointOri = laserCloudCornerLastDS->points[i];
                    pointAssociateToMap(&pointOri, &pointSel);
                    kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

                    cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
                    cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
                    cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
                            
                    if (pointSearchSqDis[4] < 1.0) {
                        float cx = 0, cy = 0, cz = 0;
                        for (int j = 0; j < 5; j++) {
                            cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                            cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                            cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                        }
                        cx /= 5; cy /= 5;  cz /= 5;

                        float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                        for (int j = 0; j < 5; j++) {
                            float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                            float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                            float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                            a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                            a22 += ay * ay; a23 += ay * az;
                            a33 += az * az;
                        }
                        a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                        matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                        matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                        matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                        cv::eigen(matA1, matD1, matV1);

                        if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                            float x0 = pointSel.x;
                            float y0 = pointSel.y;
                            float z0 = pointSel.z;
                            float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                            float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                            float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                            float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                            float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                            float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                            float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                            + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                            + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                            float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                            float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                            float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                            float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                            float ld2 = a012 / l12;

                            float s = 1 - 0.9 * fabs(ld2);

                            coeff.x = s * la;
                            coeff.y = s * lb;
                            coeff.z = s * lc;
                            coeff.intensity = s * ld2;

                            if (s > 0.1) {

                                //feature test
                                omp_set_lock(&lock1);
                                PointType pointCor_temp = pointSel;
                                pointCor_temp.intensity = CorCorresCount;
                                CornerSelected->push_back(pointCor_temp);

                                
                                for(int j = 0; j<5; j++){
                                    PointType point_temp;
                                    point_temp.x = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                                    point_temp.y = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                                    point_temp.z = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                                    point_temp.intensity = CorCorresCount;
                                    CorCorrespondence->push_back(point_temp);
                                }
                                
                                CorCorresCount++;
                                omp_unset_lock(&lock1);
                                //END

                                laserCloudOriCornerVec[i] = pointOri;
                                coeffSelCornerVec[i] = coeff;
                                laserCloudOriCornerFlag[i] = true;
                            }
                        }
                    }
                }
            }
        }

        else{
            #pragma omp parallel for num_threads(numberOfCores)
            for (size_t i = 0; i < laserCloudCornerlessLast->size(); i++)
            {
                PointType pointOri, pointSel, coeff;
                std::vector<int> pointSearchInd;
                std::vector<float> pointSearchSqDis;

                //less feature points
                pointOri = laserCloudCornerlessLast->points[i];
                //END
                pointAssociateToMap(&pointOri, &pointSel);
                kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

                cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
                cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
                cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
                        
                if (pointSearchSqDis[4] < 1.0) {
                    float cx = 0, cy = 0, cz = 0;
                    for (int j = 0; j < 5; j++) {
                        cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                        cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                        cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                    }
                    cx /= 5; cy /= 5;  cz /= 5;

                    float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                    for (int j = 0; j < 5; j++) {
                        float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                        float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                        float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                        a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                        a22 += ay * ay; a23 += ay * az;
                        a33 += az * az;
                    }
                    a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                    matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                    matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                    matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                    cv::eigen(matA1, matD1, matV1);

                    if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                        float x0 = pointSel.x;
                        float y0 = pointSel.y;
                        float z0 = pointSel.z;
                        float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                        float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                        float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                        float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                        float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                        float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                        float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                        + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                        + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                        float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                        float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                        float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                        float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                        float ld2 = a012 / l12;

                        float s = 1 - 0.9 * fabs(ld2);

                        coeff.x = s * la;
                        coeff.y = s * lb;
                        coeff.z = s * lc;
                        coeff.intensity = s * ld2;

                        if (s > 0.1) {
                            //feature test
                            omp_set_lock(&lock1);
                            PointType pointCor_temp = pointSel;
                            pointCor_temp.intensity = CorCorresCount;
                            CornerSelected->push_back(pointCor_temp);

                            
                            for(int j = 0; j<5; j++){
                                PointType point_temp;
                                point_temp.x = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                                point_temp.y = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                                point_temp.z = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                                point_temp.intensity = CorCorresCount;
                                CorCorrespondence->push_back(point_temp);
                            }
                            
                            CorCorresCount++;
                            omp_unset_lock(&lock1);
                            //END

                            laserCloudOriCornerVec[i] = pointOri;
                            coeffSelCornerVec[i] = coeff;
                            laserCloudOriCornerFlag[i] = true;
                        }
                    }
                }
            }
        }
    }

    void surfOptimization()
    {
        updatePointAssociateToMap();

        if(USEFULLFEATURE)
        {
            #pragma omp parallel for num_threads(numberOfCores)
            for (int i = 0; i < laserCloudSurfLastDSNum; i++)
            {
                PointType pointOri, pointSel, coeff;
                Eigen::Vector4f Vcoeff;
                std::vector<int> pointSearchInd;
                std::vector<float> pointSearchSqDis;

                pointOri = laserCloudSurfLastDS->points[i];
                pointAssociateToMap(&pointOri, &pointSel); 
                kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

                Eigen::Matrix<float, 5, 3> matA0;
                Eigen::Matrix<float, 5, 1> matB0;
                Eigen::Vector3f matX0;

                matA0.setZero();
                matB0.fill(-1);
                matX0.setZero();

                if (pointSearchSqDis[4] < 1.0) {
                    for (int j = 0; j < 5; j++) {
                        matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                        matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                        matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                    }

                    matX0 = matA0.colPivHouseholderQr().solve(matB0);

                    float pa = matX0(0, 0);
                    float pb = matX0(1, 0);
                    float pc = matX0(2, 0);
                    float pd = 1;

                    float ps = sqrt(pa * pa + pb * pb + pc * pc);
                    pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                    bool planeValid = true;
                    for (int j = 0; j < 5; j++) {
                        if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                                pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                                pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                            planeValid = false;
                            break;
                        }
                    }
                    
                    if (planeValid) {
                        float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;
                        float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                                + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                        coeff.x = s * pa;
                        coeff.y = s * pb;
                        coeff.z = s * pc;
                        coeff.intensity = s * pd2;

                        if (s > 0.1) {
                            //feature test
                            omp_set_lock(&lock2);
                            PointType pointSur_temp = pointSel;
                            pointSur_temp.intensity = SurCorresCount;
                            SurfSelected->push_back(pointSur_temp);

                            for(int j = 0; j<5; j++){
                                PointType point_temp;
                                point_temp.x = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                                point_temp.y = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                                point_temp.z = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                                point_temp.intensity = SurCorresCount;
                                SurCorrespondence->push_back(point_temp);
                            }
                            
                            SurCorresCount++;
                            omp_unset_lock(&lock2);
                            //END
                            
                            laserCloudOriSurfVec[i] = pointOri;
                            coeffSelSurfVec[i] = coeff;
                            laserCloudOriSurfFlag[i] = true;
                        }
                    }
                }
            }
        }

        else
        {
            #pragma omp parallel for num_threads(numberOfCores)
            for (size_t i = 0; i < laserCloudSurflessLast->size(); i++)
            {
                PointType pointOri, pointSel, coeff;
                Eigen::Vector4f Vcoeff;
                std::vector<int> pointSearchInd;
                std::vector<float> pointSearchSqDis;

                //less feature points
                pointOri = laserCloudSurflessLast->points[i];
                //END
                pointAssociateToMap(&pointOri, &pointSel); 
                kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

                Eigen::Matrix<float, 5, 3> matA0;
                Eigen::Matrix<float, 5, 1> matB0;
                Eigen::Vector3f matX0;

                matA0.setZero();
                matB0.fill(-1);
                matX0.setZero();

                if (pointSearchSqDis[4] < 1.0) {
                    for (int j = 0; j < 5; j++) {
                        matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                        matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                        matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                    }

                    matX0 = matA0.colPivHouseholderQr().solve(matB0);

                    float pa = matX0(0, 0);
                    float pb = matX0(1, 0);
                    float pc = matX0(2, 0);
                    float pd = 1;

                    float ps = sqrt(pa * pa + pb * pb + pc * pc);
                    pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                    bool planeValid = true;
                    for (int j = 0; j < 5; j++) {
                        if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                                pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                                pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                            planeValid = false;
                            break;
                        }
                    }

                    if (planeValid) {
                        float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                        float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                                + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                        coeff.x = s * pa;
                        coeff.y = s * pb;
                        coeff.z = s * pc;
                        coeff.intensity = s * pd2;

                        if (s > 0.1) {

                            //feature test
                            omp_set_lock(&lock2);
                            PointType pointSur_temp = pointSel;
                            pointSur_temp.intensity = SurCorresCount;
                            SurfSelected->push_back(pointSur_temp);

                            for(int j = 0; j<5; j++){
                                PointType point_temp;
                                point_temp.x = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                                point_temp.y = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                                point_temp.z = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                                point_temp.intensity = SurCorresCount;
                                SurCorrespondence->push_back(point_temp);
                            }
                            
                            SurCorresCount++;
                            omp_unset_lock(&lock2);
                            //END
                            
                            laserCloudOriSurfVec[i] = pointOri;
                            coeffSelSurfVec[i] = coeff;
                            laserCloudOriSurfFlag[i] = true;
                        }
                    }
                }
            }
        }
    }

    void combineOptimizationCoeffs(int iterCount)
    {
        // combine corner coeffs
        if(USEFULLFEATURE)
        {
            for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
                if (laserCloudOriCornerFlag[i] == true){
                    laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                    coeffSel->push_back(coeffSelCornerVec[i]);
                }
            }

            for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
                if (laserCloudOriSurfFlag[i] == true){
                    laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                    coeffSel->push_back(coeffSelSurfVec[i]);
                }
            }
        }
        else{
            for (size_t i = 0; i < laserCloudCornerlessLast->size(); ++i){
                if (laserCloudOriCornerFlag[i] == true){
                    laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                    coeffSel->push_back(coeffSelCornerVec[i]);
                }
            }

            for (size_t i = 0; i < laserCloudSurflessLast->size(); ++i){
                if (laserCloudOriSurfFlag[i] == true){
                    laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                    coeffSel->push_back(coeffSelSurfVec[i]);
                }
            }
        }

        // reset flag for next iteration
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        // cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        // cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        // cv::Mat matD(6, 6, CV_32F, cv::Scalar::all(0));
        matAtB = cv::Mat(6, 1, CV_32F, cv::Scalar::all(0));
        matX = cv::Mat(6, 1, CV_32F, cv::Scalar::all(0));
        matD = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        for (int i = 0; i < laserCloudSelNum; i++) {
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            // lidar -> camera
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;

        //LM algorithm
        // for(int i = 0; i < 6; i++){
        //     // matD.at<float>(i, i) = matAtA.at<float>(i, i);
        //     matD.at<float>(i, i) = 1;
        // }
        // matAtA += u * matD;
        //END

        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {

            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {Degenerate_Thr, Degenerate_Thr, Degenerate_Thr, 
                                 Degenerate_Thr, Degenerate_Thr, Degenerate_Thr};
            for (int i = 5; i >= 0; i--) {
                cout<<"第"<<" "<<Key_point_count<<" "<<"帧: Eigen Value is ";
                cout<<matE.at<float>(0, i)<<endl;
                if (matE.at<float>(0, i) < eignThre[i]) {    
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        cout<<"第"<<iterCount<<"次优化： "<<matX.at<float>(0, 0)<<" "<<matX.at<float>(1, 0)<<" "<<matX.at<float>(3, 0)<<" "
            <<matX.at<float>(4, 0)<<" "<<matX.at<float>(5, 0)<<endl;

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {

            return true; // converged
        }
        return false; // keep optimizing
    }

    void scan2MapOptimization()
    {
        // if (cloudKeyPoses3D->points.empty())
        //     return;

        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            if(USE_SUBMAP)
            {
                kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
                kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);
            }
            
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {   
                //feature test
                CorCorrespondence->clear();
                SurCorrespondence->clear();
                CornerSelected->clear();
                SurfSelected->clear();

                CorCorresCount = 1;
                SurCorresCount = 1;
                //END

                if(keep_opt)
                {
                    laserCloudOri->clear();
                    coeffSel->clear();
                    cornerOptimization();
                    surfOptimization();
                    combineOptimizationCoeffs(iterCount);
                }
                if (LMOptimization(iterCount) == true)
                {   
                    //feature test
                    publishCloud(&pubCorCorres, CorCorrespondence, timeLaserInfoStamp, odometryFrame);
                    publishCloud(&pubSurCorres, SurCorrespondence, timeLaserInfoStamp, odometryFrame);
                    publishCloud(&pubCorSelected, CornerSelected, timeLaserInfoStamp, odometryFrame);
                    publishCloud(&pubSurSelected, SurfSelected, timeLaserInfoStamp, odometryFrame);
                    //END
                    cout<<"LM iteration times: "<<iterCount<<endl;
                    break;
                }          
            }
            transformUpdate();
        } else {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    void transformUpdate()
    {
        if (cloudInfo.imuAvailable == true)
        {
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                double imuWeight = imuRPYWeight;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);

        cout<<"REGISTRATION POSITION: "<<"["
            <<transformTobeMapped[0]<<", "
            <<transformTobeMapped[1]<<", "
            <<transformTobeMapped[2]<<", "
            <<transformTobeMapped[3]<<", "
            <<transformTobeMapped[4]<<", "
            <<transformTobeMapped[5]<<"]"<<endl<<endl;
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    void addOdomFactor()
    {   
        if (cloudKeyPoses3D->points.empty())
        {
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            // gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            gtSAMgraph.add(BetweenFactor<Pose3>(Key_point_count-1, Key_point_count, poseFrom.between(poseTo), odometryNoise));
            // initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
            initialEstimate.insert(Key_point_count, poseTo);
        }
    }

    void addGPSFactor()
    {   
        if(!USEGPS) return;

        std::lock_guard<std::mutex> lock3(gpsLock);
        if (gpsQueue.empty())
        {
            cout<<"gps empty!!"<<endl;
            return;
        }

        // wait for system initialized and settles down
        if (cloudKeyPoses3D->points.empty())
            return;
        //FOR CHANG'AN
        {
            // if(!FOR_CHANGAN){if(pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0) return;}
            // else {
            //     // if(pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 1.0) return;
            //     }
        }
        //END

        // pose covariance small, no need to correct
        std::cout<<"pose covariance are: "<< poseCovariance(3,3)<<" , "<<poseCovariance(4,4)<<" and "<<poseCovariance(5,5)<<endl<<endl;
        
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;


        while (!gpsQueue.empty())
        {   
            //cout<<"gps time "<<std::setprecision(15)<<timeLaserInfoCur<<endl;
            // if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            //For CHANGAN
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.05)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.05)
            {
                // message too new
                break;
            }
            else
            {
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();
                
                if(true){
                // Quaternionf Qw0_ik_(thisGPS.pose.pose.orientation.w, thisGPS.pose.pose.orientation.x,
                //                     thisGPS.pose.pose.orientation.y, thisGPS.pose.pose.orientation.z);
                // Qw0_ik_.normalized();

                // Quaternionf Qw0_lk_(Qw0_ik_.toRotationMatrix() * extRot.transpose().cast<float>());
                // thisGPS.pose.pose.orientation.w = Qw0_lk_.w();
                // thisGPS.pose.pose.orientation.x = Qw0_lk_.x();
                // thisGPS.pose.pose.orientation.y = Qw0_lk_.y();
                // thisGPS.pose.pose.orientation.z = Qw0_lk_.z();

                // //Method 1:
                // Vector3f pose = - Qw0_ik_.toRotationMatrix() * extRot.transpose().cast<float>() * extTrans.cast<float>()
                //                 + Vector3d(thisGPS.pose.pose.position.x, thisGPS.pose.pose.position.y, thisGPS.pose.pose.position.z).cast<float>();
                // thisGPS.pose.pose.position.x = pose(0);
                // thisGPS.pose.pose.position.y = pose(1);
                // thisGPS.pose.pose.position.z = pose(2);

                // //Method 2:
                // Vector3f pose2 = pose + coordinate_offset;
                // thisGPS.pose.pose.position.x = pose2(0);
                // thisGPS.pose.pose.position.y = pose2(1);
                // thisGPS.pose.pose.position.z = pose2(2);
                }

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;

                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation)
                {   
                    // cout<<"FALSE"<<endl<<endl;
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 1.0)
                // if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 0.02f), max(noise_y, 0.02f), max(noise_z, 0.02f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(Key_point_count, gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);
                cout<<"第"<<" "<<Key_point_count<<" "<<"帧"<<endl;
                cout<<"[ADD gps factor!]"<<endl;
                cout<<"GPS POSITION: ["
                    <<gps_x<<", "
                    <<gps_y<<", "
                    <<gps_z<<"]"<<endl<<endl;

                //check:
                aLoopIsClosed = false;
                break;
            }
        }
    }

    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    void saveKeyFramesAndFactor()
    {
        if (saveFrame() == false)
            return;

        // odom factor
        addOdomFactor();

        // gps factor
        addGPSFactor();

        // loop factor
        addLoopFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }

        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;
        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        
        // Key_point_count++;
        thisPose3D.intensity = Key_point_count++;
        // thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);
        cout<<"Key frame number: "<<Key_point_count<<endl;
        // cout<<"Queue number: "<<cloudKeyPoses3D->size()<<endl;
        if(static_cast<int>(cloudKeyPoses3D->size())>keyframenumber) cloudKeyPoses3D->erase(cloudKeyPoses3D->begin());

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        // cout<<"The optimized RPY are: "<<endl<<thisPose6D.roll<<endl<<thisPose6D.pitch<<endl<<thisPose6D.yaw<<endl<<endl;
        // cout<<"save time "<<std::setprecision(15)<<timeLaserInfoCur<<endl;
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);
        if(static_cast<int>(cloudKeyPoses6D->size())>keyframenumber) cloudKeyPoses6D->erase(cloudKeyPoses6D->begin());

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);
        // save updated transform
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();
        
        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // cout<<"Raw Corner Points are: "<<laserCloudCornerLastDSNum<<endl;
        // cout<<"Raw Surf Points are: "<<laserCloudSurfLastDSNum<<endl;
        // cout<<"Corner Points are: "<<thisCornerKeyFrame->size()<<endl;
        // cout<<"Surf Points are: "<<thisSurfKeyFrame->size()<<endl;

        // pcl::PointCloud<PointType>::Ptr thisCornerRawKeyFrame(new pcl::PointCloud<PointType>());
        // pcl::copyPointCloud(*laserCloudCornerLast, *thisCornerRawKeyFrame);

        // pcl::PointCloud<PointType>::Ptr thisCornerRawKeyFrame(new pcl::PointCloud<PointType>());
        // pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerRawKeyFrame);

        //将该帧所有点投影到地图系
        // laserCloudRawLastDS->clear();
        // downSizeFilterSC.setInputCloud(laserCloudRawLast);
        // downSizeFilterSC.filter(*laserCloudRawLastDS);
        pcl::PointCloud<PointType>::Ptr thisRawCloudKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudRawLastDS,  *thisRawCloudKeyFrame);
        //cout<<"The size of Raw Cloud is : "<<thisRawCloudKeyFrame->size()<<endl;
        // scManager.makeAndSaveScancontextAndKeys(*thisRawCloudKeyFrame);
        //scManager.makeAndSaveScancontextAndKeys(*thisCornerRawKeyFrame);
        // scManager.makeAndSaveScancontextAndKeys(*thisSurfKeyFrame);

        // save key frame cloud
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);
        RawCloudKeyFrames.push_back(thisRawCloudKeyFrame);

        if(static_cast<int>(cornerCloudKeyFrames.size())>keyframenumber) {
            cornerCloudKeyFrames.erase(cornerCloudKeyFrames.begin());
            surfCloudKeyFrames.erase(surfCloudKeyFrames.begin());
            RawCloudKeyFrames.erase(RawCloudKeyFrames.begin());
        }

        // save path for visualization
        updatePath(thisPose6D);
    }

    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // clear map cache
            laserCloudMapContainer.clear();
            // clear path
            globalPath.poses.clear();
            // update key poses
            int numPoses = isamCurrentEstimate.size();

            //CHANGE FOR POSE LIMITATION
            if(numPoses<=keyframenumber){
                for (int i = 0; i < numPoses; ++i)
                {
                    cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                    cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                    cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                    cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                    cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                    cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                    cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                    cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                    cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                    updatePath(cloudKeyPoses6D->points[i]);
                }
            }
            else{
                int numPoseleft = numPoses - keyframenumber;
                for (int i = 0; i < keyframenumber; ++i)
                {
                    cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i+numPoseleft).translation().x();
                    cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i+numPoseleft).translation().y();
                    cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i+numPoseleft).translation().z();

                    cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                    cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                    cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                    cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i+numPoseleft).rotation().roll();
                    cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i+numPoseleft).rotation().pitch();
                    cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i+numPoseleft).rotation().yaw();

                    updatePath(cloudKeyPoses6D->points[i]);
                }
            }

            aLoopIsClosed = false;
        }
    }

    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    void publishOdometry()
    {
        // Publish odometry for ROS (global)
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        pubLaserOdometryGlobal.publish(laserOdometryROS);

        cout<<"GLOBAL POSITION: "<<"["
            <<transformTobeMapped[0]<<", "
            <<transformTobeMapped[1]<<", "
            <<transformTobeMapped[2]<<", "
            <<transformTobeMapped[3]<<", "
            <<transformTobeMapped[4]<<", "
            <<transformTobeMapped[5]<<"]"<<endl<<endl;

        file_lidar.setf(std::ios::fixed, std::_S_floatfield);
        file_lidar << laserOdometryROS.header.stamp.toSec() << " " 
                << laserOdometryROS.pose.pose.position.x << " "
                << laserOdometryROS.pose.pose.position.y << " "
                << laserOdometryROS.pose.pose.position.z << " "
                << laserOdometryROS.pose.pose.orientation.x << " "
                << laserOdometryROS.pose.pose.orientation.y << " "
                << laserOdometryROS.pose.pose.orientation.z << " "
                << laserOdometryROS.pose.pose.orientation.w << std::endl;
        
        // Publish TF
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
        br.sendTransform(trans_odom_to_lidar);

        // Publish odometry for ROS (incremental)
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            if (cloudInfo.imuAvailable == true)
            {
                if (std::abs(cloudInfo.imuPitchInit) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf::Quaternion imuQuaternion;
                    tf::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
            cout<<"GLOBAL_INCREMENTAL POSITION: "<<"["
            <<roll<<", "
            <<pitch<<", "
            <<yaw<<", "
            <<x<<", "
            <<y<<", "
            <<z<<"]"<<endl;
            if (isDegenerate)
                {std::cout<<"[DEGENERATE!!]"<<std::endl;
                laserOdomIncremental.pose.covariance[0] = 1;}
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        pubLaserOdometryIncremental.publish(laserOdomIncremental);
    }

    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
        publishCloud(&pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // Publish surrounding key frames
        publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        // publish registered key frame
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            // pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            // PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            // *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            // *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            publishCloud(&pubRecentKeyFrame, MapSurfClouds, timeLaserInfoStamp, odometryFrame);
        }
        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }

        // publish path
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    mapOptimization MO;

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");
    
    // std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ROS_INFO("Spinning node");

    ros::spin();

    // loopthread.join();
    visualizeMapThread.join();

    return 0;
}
