#include "utility.h"

#include <dynamic_reconfigure/server.h>
#include "map_localization/MAP_LOCALIZATION_PARAMConfig.h"

#include "lidarFactor.hpp"
#include "map_localization/cloud_info.h"
#include "map_localization/save_map.h"

#include <LocalCartesian.hpp>
#include "Scancontext.h"
#include "registrations.hpp"
#include <omp.h>
#include <FilterState.h>
#include <sophus/so3.hpp>
#include <unordered_set>


//relative to filetr
#define POS_ 0
#define ROT_ 3
#define VEL_ 6
#define BIA_ 9
#define BIG_ 12
#define GW_ 15

enum SYS_STATUS{FIRST_SCAN, OTHER_SCAN};

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
    //relative to filter method
    //compatible for S2 and normal Gravity
    SYS_STATUS sysStatus;
    // prediction
    Eigen::Matrix<float, 18, 18> F_t;
    Eigen::Matrix<float, 18, 12> G_t;
    Eigen::Matrix<float, 18, 18> J_t;
    Eigen::Matrix<float, 18, 18> Jt_inv;
    Eigen::Matrix<float, 18, 18> P_t;
    Eigen::Matrix<float, 18, 18> I18;
    Eigen::Matrix<float, 18, 18> P_t_inv;
    Eigen::Matrix<float, 12, 12> noise_; //imuAccNoise, imuGyrNoise, imuAccBiasN, imuGyrBiasN
    // measurement
    Eigen::Matrix<float, Eigen::Dynamic, 1> residual_;
    Eigen::Matrix<float, Eigen::Dynamic, 18> H_k;
    Eigen::Matrix<float, 18, Eigen::Dynamic> H_k_T_R_inv;
    Eigen::Matrix<float, 18, Eigen::Dynamic> K_k;
    Eigen::MatrixXf updateVec_;
    Eigen::Matrix<float, 18, 1> errVec_;
    Eigen::Matrix<float, 18, 18> HRH;
    Eigen::Matrix<float, 18, 18> HRH_inv;
    Eigen::MatrixXf R;
    //states
    float transformInLidar[6];
    float transformTobeMapped[18];
    float transformTobeMappedLast[18];
    FilterState filterState;
    FilterState intermediateState;
    Eigen::Vector3d accBias;
    Eigen::Vector3d gyrBias;
    //Data from other cpp
    std::deque<sensor_msgs::Imu> imuQueue;
    std::vector<sensor_msgs::Imu> imuBucket;
    //param
    bool converge = false;
    bool hasDiverged = false;
    bool residualNorm = 10e6;
    bool imuAligned = false;
    Vector3d mean_acc;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;
    
    int Key_point_count;
    int frame_count;
    std::ofstream file_lidar;
    std::ofstream file_time;
    std::ofstream file_predict;
    std::ofstream file_update;
    std::ofstream file_imuInput;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;

    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subImu;
    ros::Subscriber subBias;

    ros::ServiceServer srvSaveMap;

    std::deque<nav_msgs::Odometry> gpsQueue;
    map_localization::cloud_info cloudInfo;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

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
    pcl::PointCloud<PointXYZIRPYT>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointXYZIRPYT> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointXYZIRPYT> coeffSelSurfVec;
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

    std::mutex mtx;
    std::mutex mtxLoopInfo;
    std::mutex gpsLock;
    std::mutex imuLock;
    std::mutex biasLock;

    bool isDegenerate = false;
    cv::Mat matP;
    cv::Mat matAtB;
    cv::Mat matX;
    cv::Mat matD;
    bool keep_opt = true;
    //END

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

    SCManager scManager;

    Eigen::Quaternionf q_exrot;
    Eigen::Affine3f exT_affine;
    Eigen::Affine3f exT_inv_affine;
    Eigen::Affine3f wTimuAffine;
    Eigen::Vector3f coordinate_offset;
    Eigen::Matrix4f exT_M4f;
    Eigen::Matrix4f exT_inv_M4f;

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
    bool gps_usage = true;

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
        //*****topics receive and advertise******//
        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1);
        //lio_sam/mapping/odometry is the odometry with pose correction, thus it has no drift.
        //lio_sam/mapping/odometry_incremental is pure lidar odometry without any correction, thus it has drifts.
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry", 1);
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry_incremental", 1);
        pubPath                     = nh.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1);

        subCloud = nh.subscribe<map_localization::cloud_info>("lio_sam/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subGPS   = nh.subscribe<nav_msgs::Odometry> ("odometry/enu", 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        subImu   = nh.subscribe<sensor_msgs::Imu> (imuTopic, 2000, &mapOptimization::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subBias  = nh.subscribe<std_msgs::Float64MultiArray> ("lio_sam/imu/bias", 5, &mapOptimization::imuBiasHandler, this, ros::TransportHints().tcpNoDelay());
        srvSaveMap  = nh.advertiseService("lio_sam/save_map", &mapOptimization::saveMapService, this);

        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);

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

        //*****initialize parameters*****//
        //filter states
        sysStatus = FIRST_SCAN;

        //keyframe count
        Key_point_count = 0;
        frame_count = 0;

        //gps and feature dynamic param subscribe
        f = boost::bind(&mapOptimization::callback,this, _1); //绑定回调函数
        server.setCallback(f); //为服务器设置回调函数， 节点程序运行时会调用一次回调函数来输出当前的参数配置情况

        //map registration it sc
        reg = select_registration_method(nh);
        
        //extrinsic param
        exTinitialization();

        const float kSCFilerSize = 0.25;
        downSizeFilterSC.setLeafSize(kSCFilerSize, kSCFilerSize, kSCFilerSize);
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization
        downSizeFilterMapCloud.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

        file_lidar.open(Odom_Path+"lidar_odom.csv", std::ios::app);
        file_time.open(Odom_Path+"operation_time.csv", std::ios::app);
        file_predict.open(Odom_Path+"predict_state.csv", std::ios::app);
        file_update.open(Odom_Path+"update_state.csv", std::ios::app);
        file_imuInput.open(Odom_Path+"imuMeasure.csv", std::ios::app);

        allocateMemory();

        //load maps
        // if(USE_SUBMAP) SubMapLoad();
        // else MapLoad();
    }

    ~mapOptimization(){
        file_lidar.close();
        file_time.close();
        file_predict.close();
        file_update.close();
        file_imuInput.close();
    }

    void resetState()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        first_lidar_frame = true;
        sysStatus = FIRST_SCAN;
        P_t.setZero();
        filterState.setIdentity();
        ROS_WARN("Filter has diverged, restart system");
    }

    void allocateMemory()
    {   
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

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
        coeffSel.reset(new pcl::PointCloud<PointXYZIRPYT>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

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

        //state initilization
        for (int i = 0; i < 18; ++i){
            if(i<6) transformInLidar[i] = 0;
            transformTobeMapped[i] = 0;
            transformTobeMappedLast[i] = 0;
        }
        transformTobeMapped[GW_+2] = -imuGravity;
        transformTobeMappedLast[GW_+2] = -imuGravity;
        filterState.GT = GravityType::Normal;
        filterState.gn_ = Eigen::Vector3f(0.0, 0.0, -imuGravity);

        F_t.setZero();
        J_t.setIdentity();
        Jt_inv.setIdentity();
        G_t.setZero();
        P_t.setZero();
        I18.setIdentity();
        noise_.setZero();
        updateVec_.resize(18, 1);
        updateVec_.setZero();
        errVec_.setZero();
        // asDiagonal()指将向量作为对角线构建对角矩阵
        noise_.block<3, 3>(0, 0) = Eigen::Vector3f(imuAccNoise, imuAccNoise, imuAccNoise).asDiagonal();
        noise_.block<3, 3>(3, 3) = Eigen::Vector3f(imuGyrNoise, imuGyrNoise, imuGyrNoise).asDiagonal();
        noise_.block<3, 3>(6, 6) = Eigen::Vector3f(imuAccBiasN, imuAccBiasN, imuAccBiasN).asDiagonal();
        noise_.block<3, 3>(9, 9) = Eigen::Vector3f(imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).asDiagonal();

        HRH.setZero();
        HRH_inv.setZero();

        accBias.setZero();
        gyrBias.setZero();

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

        //check whether imu has been initialized
        if(!imuAligned) return;

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
            // timeLastProcessing = timeLaserInfoCur;

            frame_count++;

            TicToc t_All;
            updateInitialGuessMap();
            cout<<endl;

            TicToc t_localmap;
            extractSurroundingKeyFrames();
            t_localmap.toc("local map");
            cout<<endl;

            TicToc t_sample;
            downsampleCurrentScan();
            t_sample.toc("sample map");
            cout<<endl;
            
            TicToc t_FilterProcess;
            if(sysStatus == FIRST_SCAN) 
            {
                removeOldImu();
                sysStatus = OTHER_SCAN;
            }
            else
            {
                if(removeOldImu()) 
                {
                    statePredict();
                    updateStatebyIeskf(timeLaserInfoCur - timeLastProcessing);

                    // Eigen::Affine3f T_transformTobeMapped;
                    // T_transformTobeMapped.setIdentity();
                    // T_transformTobeMapped.pretranslate(filterState.rn_);
                    // T_transformTobeMapped.rotate(filterState.qbn_);
                    // pcl::getTranslationAndEulerAngles(T_transformTobeMapped, 
                    //     transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                    //     transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
                    // transformUpdate();
                }
                // else scan2MapOptimization();
                else return;
            }
            t_FilterProcess.toc("Filter Process");
            cout<<endl;

            TicToc t_gtsam;
            //Only save frames and points if will be used
            saveKeyFrames();
            t_gtsam.toc("whole gtsam");
            cout<<endl;

            publishOdometry();

            publishFrames();

            t_All.toc("All operation time");
            cout<<endl<<endl;

            timeLastProcessing = timeLaserInfoCur;

            file_time.setf(std::ios::fixed, std::_S_floatfield);
            file_time << timeLastProcessing << ","
                      << t_localmap.elapsed_ms << ","
                      << t_sample.elapsed_ms << ","
                      << t_FilterProcess.elapsed_ms << ","
                      << t_gtsam.elapsed_ms << ","
                      << t_All.elapsed_ms << ","
                      <<std::endl;
        }
    }

    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {   
        if(!USEGPS) return;
        std::lock_guard<std::mutex> lock3(gpsLock);
        gpsQueue.push_back(*gpsMsg);
    }

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        std::lock_guard<std::mutex> lock4(imuLock);
        imuQueue.push_back(*imuMsg);
        if(imuTimeOffset != 0.0)
        {
            ros::Time time(imuQueue.back().header.stamp.toSec() + imuTimeOffset);
            imuQueue.back().header.stamp = time;
        }

        if(!imuAligned && imuQueue.size() >= size_t(imuQueSize))
        {
            while(imuQueue.size()>size_t(imuQueSize)) imuQueue.pop_front();
            if(imuAlignImp()) imuAligned = true;
        }
        file_imuInput.setf(std::ios::fixed, std::_S_floatfield);
        file_imuInput << imuMsg->header.stamp.toSec() << "," 
                << imuMsg->angular_velocity.x<< ","
                << imuMsg->angular_velocity.y << ","
                << imuMsg->angular_velocity.z << ","
                << imuMsg->linear_acceleration.x << ","
                << imuMsg->linear_acceleration.y << ","
                << imuMsg->linear_acceleration.z  << std::endl;
    }

    void imuBiasHandler(const std_msgs::Float64MultiArray::ConstPtr& imuBiasMsg)
    {
        std::lock_guard<std::mutex> lock5(biasLock);
        for(int i=0; i<3; i++)
        {
            accBias(i) = imuBiasMsg->data[1+i];
            gyrBias(i) = imuBiasMsg->data[3+i];
        }
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

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    void Affine2trans(float transformOut[], const Affine3f aff)
    {
        Eigen::Quaternionf Qe(aff.rotation());
        tf::Quaternion Qtf = tf::Quaternion(Qe.x(), Qe.y(), Qe.z(), Qe.w());
        double roll, pitch, yaw;
        tf::Matrix3x3(Qtf).getRPY(roll, pitch, yaw);
        transformOut[0] = roll;
        transformOut[1] = pitch;
        transformOut[2] = yaw;
        transformOut[3] = aff.translation()[0];
        transformOut[4] = aff.translation()[1];
        transformOut[5] = aff.translation()[2];    
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

        downSizeFilterSurf.setInputCloud(MapSurfClouds);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

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
        exT_M4f = Eigen::Matrix4f::Identity();
        exT_M4f.block<3,3>(0,0) = q_exrot.toRotationMatrix();
        exT_M4f(0,3) = static_cast<float>(extTrans(0));
        exT_M4f(1,3) = static_cast<float>(extTrans(1));
        exT_M4f(2,3) = static_cast<float>(extTrans(2));
        exT_affine = exT_M4f;
        exT_inv_affine = exT_affine.inverse();
        exT_inv_M4f.block<3, 3>(0, 0) = exT_inv_affine.rotation();
        exT_inv_M4f.block<3, 1>(0, 3) = exT_inv_affine.translation();
        exT_inv_M4f(3, 3) = 1;
    }

    bool imuAlignImp()
    {
        using namespace Eigen;

        double kAccStdLimit = 3.0;
        // Compute mean and std of the imu buffer.
        Vector3d sum_acc(0., 0., 0.);
        for (const auto imu_data : imuQueue) {
            sum_acc[0] += imu_data.linear_acceleration.x;
            sum_acc[1] += imu_data.linear_acceleration.y;
            sum_acc[2] += imu_data.linear_acceleration.z;
        }
        mean_acc = sum_acc / (double)imuQueue.size();
        ROS_INFO_STREAM("Cur Gw in Imu frame: "<<mean_acc.transpose()<<endl);

        Vector3d sum_err2(0., 0., 0.);
        for (const auto imu_data : imuQueue) {
            sum_err2 += (Vector3d(imu_data.linear_acceleration.x, imu_data.linear_acceleration.y, imu_data.linear_acceleration.z)
                         - mean_acc).cwiseAbs2();
        }
        const Vector3d std_acc = (sum_err2 / (double)imuQueue.size()).cwiseSqrt();//均方根

        if (std_acc.maxCoeff() > kAccStdLimit) {
            ROS_WARN_STREAM("[ComputeG_R_IFromImuData]: Too big acc std: " << std_acc.transpose());
            return false;
        }


        // Eigen::Matrix3d acc_cov;
        // acc_cov = Matrix3d::Identity() * imuAccNoise * imuAccNoise;
        // mean_acc.normalize();
        // Eigen::Vector3d gravity_acc(Eigen::Vector3d(0, 0, 1));
        // double cos_value = mean_acc.dot(gravity_acc);
        // double angle = acos(cos_value);
        // Eigen::Vector3d axis = GetSkewMatrix(mean_acc) * gravity_acc;
        // Eigen::Matrix3d Jac = Amatrix(angle * axis).transpose() 
        //                     * (1/(sqrt(1-cos_value*cos_value)) * -axis * gravity_acc.transpose() - angle * GetSkewMatrix(gravity_acc));
        // acc_cov = Jac * acc_cov * Jac.transpose();

        // stateInitialization(acc_cov.cast<float>());

        Matrix3d rotMatrix = Quaterniond::FromTwoVectors(mean_acc, Vector3d(0, 0, 1)).toRotationMatrix();
        Matrix3d temp;
        // double sai = 30.0/180 * M_PI;
        // temp << cos(sai), sin(sai), 0, -sin(sai), cos(sai), 0, 0, 0, 1;
        Quaterniond Qrot(/*temp*/ rotMatrix);
        tf::Quaternion Qtf = tf::Quaternion(Qrot.x(), Qrot.y(), Qrot.z(), Qrot.w());
        double roll, pitch, yaw;
        tf::Matrix3x3(Qtf).getRPY(roll, pitch, yaw);

        transformTobeMapped[0] = roll;
        transformTobeMapped[1] = pitch;
        transformTobeMapped[2] = yaw;

        float g_norm = static_cast<float>(mean_acc.norm());
        filterState.gn_ = Eigen::Vector3f(0.0, 0.0, -g_norm);
    
        return true;
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

    void Lidar2Imu ()
    {
        Eigen::Affine3f A = trans2Affine3f(transformInLidar) * exT_affine;
        Affine2trans(transformTobeMapped, A);
    }

    void Imu2Lidar()
    {
        Eigen::Affine3f A = trans2Affine3f(transformTobeMapped) * exT_inv_affine;
        Affine2trans(transformInLidar, A);
    }

    void updateInitialGuessMap(){
        //Use SC
        if(sysStatus == FIRST_SCAN && !gps_usage)
        {   
            TicToc t_SC_localization;
            ROS_INFO("***Using SC Mode***");
            scManager.makeAndSaveScancontextAndKeys(*laserCloudRawLast);
            // pcl::io::savePCDFileBinary(map_path + "RawClouds.pcd", *laserCloudRawLast);

            auto detectResult = scManager.detectLoopClosureID();
            int SCKeyPre = detectResult.first;
            float yawdiff = detectResult.second;
            double sigma;
            if(SCKeyPre == -1) {
                gps_usage = true;
                ROS_WARN("Can't find SC Correspondence");
            }
            if(!gps_usage)
            {   
                pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
                //因为lidar和imu的平移和旋转不会造成太大的位移，正常找
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
                    //得到的是当前帧在lidar系下的位置，因为prevKeyframeCloud是雷达系下的
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
                    sigma = reg->getFitnessScore();

                    if(!gps_usage){
                        // Get pose transformation
                        float x, y, z, roll, pitch, yaw;

                        Eigen::Affine3f correctionLidarFrame;
                        correctionLidarFrame = reg->getFinalTransformation();
                        Eigen::Vector3f eulerAngle_cur = correctionLidarFrame.rotation().eulerAngles(2,1,0);
                        Eigen::Vector3f eulerAngle_SC = Eigen::Matrix3f(Tpre2W.block<3,3>(0,0)).eulerAngles(2,1,0);

                        cout<<"ICP POSITION in Lidar: ["<<correctionLidarFrame(0,3)<<" "
                                   <<correctionLidarFrame(1,3)<<" "
                                   <<correctionLidarFrame(2,3)<<" "
                                   <<eulerAngle_cur[0]<<" "
                                   <<eulerAngle_cur[1]<<" "
                                   <<eulerAngle_cur[2]<<"]"                                  
                                   <<endl;
                        cout<<"SC Map POSITION in Lidar: "<<Tpre2W(0,3)<<" "
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
                        // Eigen::Affine3f Tk2W = pclPointToAffine3f(KeyFramesPoseFromMap[SCKeyPre]);
                        // transform from world origin to corrected pose
                        // Eigen::Affine3f Tnow2W =Tk2W * correctionLidarFrame;// pre-multiplying -> successive rotation about a fixed frame
                        pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
                        transformInLidar[0] = roll;
                        transformInLidar[1] = pitch;
                        transformInLidar[2] = yaw;
                        transformInLidar[3] = x;
                        transformInLidar[4] = y;
                        transformInLidar[5] = z;
                        cout<<"SC Initialization transition in Lidar: "<<x<<" "<<y<<" "<<z<<endl;

                        Lidar2Imu();                        
                        stateInitialization(sigma);
                        first_lidar_frame = false;
                        incrementalOdometryAffineFront = trans2Affine3f(transformInLidar); 
                        for(int i=0; i<18; ++i)
                        {
                            transformTobeMappedLast[i] = transformTobeMapped[i];
                        } 
                    }
                }

                
                t_SC_localization.toc("SC_localization");
            }
        }

        if (gps_usage && first_lidar_frame && sysStatus == FIRST_SCAN){
            ROS_INFO("***Using GPS Signal Mode***");
            //Use GPS signal
            // double roll_temp, pitch_temp, yaw_temp;
            // tf::Quaternion mapQlk_initial = tf::Quaternion(cloudInfo.mapQlk[1], cloudInfo.mapQlk[2], 
            //                                             cloudInfo.mapQlk[3], cloudInfo.mapQlk[0]);
            
            // tf::Matrix3x3(mapQlk_initial).getRPY(roll_temp, pitch_temp, yaw_temp);
            // transformInLidar[0] = roll_temp;
            // transformInLidar[1] = pitch_temp;
            // transformInLidar[2] = yaw_temp;
            // transformInLidar[3] = cloudInfo.maptlk[0];
            // transformInLidar[4] = cloudInfo.maptlk[1];
            // transformInLidar[5] = cloudInfo.maptlk[2];
            // cout<<"GNSS Initialization transition in Lidar: "<<cloudInfo.maptlk[0]<<" "
            //                                                  <<cloudInfo.maptlk[1]<<" "
            //                                                  <<cloudInfo.maptlk[2]<<endl;
            
            // Lidar2Imu();
            // stateInitialization();
            // first_lidar_frame = false;
            // incrementalOdometryAffineFront = trans2Affine3f(transformInLidar);  

            //整个修改
            //for imu align
            //没问题，但是始终无法知道和g对齐后的坐标系和ENU坐标系之间差的yaw是多少
            Imu2Lidar();
            stateInitialization();
            first_lidar_frame = false;
            incrementalOdometryAffineFront = trans2Affine3f(transformInLidar);  
        }

        // if(cloudInfo.odomAvailable && !first_lidar_frame)
        if(sysStatus == OTHER_SCAN)
        {
            ROS_INFO("***Using IMU Integration Mode***");
            // transformTobeMapped[0] = cloudInfo.initialGuessRoll;
            // transformTobeMapped[1] = cloudInfo.initialGuessPitch;
            // transformTobeMapped[2] = cloudInfo.initialGuessYaw;
            // transformTobeMapped[3] = cloudInfo.initialGuessX;
            // transformTobeMapped[4] = cloudInfo.initialGuessY;
            // transformTobeMapped[5] = cloudInfo.initialGuessZ;
            // float temp[6] = {cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw,
            //                  cloudInfo.initialGuessX, cloudInfo.initialGuessY, cloudInfo.initialGuessZ};
            // Eigen::Affine3f t = trans2Affine3f(temp) * exT_affine;
            // Affine2trans(temp, t);
            // ROS_INFO_STREAM("IMU preint module output: "<<endl<<temp[0]<<" "
            //                                                   <<temp[1]<<" "
            //                                                   <<temp[2]<<" "
            //                                                   <<temp[3]<<" "
            //                                                   <<temp[4]<<" "
            //                                                   <<temp[5]<<endl;);
            incrementalOdometryAffineFront = trans2Affine3f(transformInLidar);
        }
        
    }

    void stateInitialization(float sigma = 0.1)
    {   
        float cov_pos = sigma * sigma;
        float cov_angle = 0.5 * cov_pos;
        float cov_vel = 25;
        float cov_bias = 1e-4;
        P_t.block<3,3>(POS_,POS_) = Eigen::Vector3f(cov_pos, cov_pos, cov_pos).asDiagonal();
        P_t.block<3,3>(ROT_, ROT_) = Eigen::Vector3f(cov_angle, cov_angle, cov_angle).asDiagonal();
        P_t.block<3,3>(VEL_, VEL_) = Eigen::Vector3f(cov_vel, cov_vel, cov_vel).asDiagonal();
        P_t.block<6,6>(BIA_, BIA_) = Eigen::Matrix<float, 6, 6>::Identity()*cov_bias;
        if(!USE_S2)
            P_t.block<3,3>(GW_, GW_) = Eigen::Vector3f(0.01, 0.01, 0.01).asDiagonal();
        else 
            P_t.block<2,2>(GW_, GW_) = Eigen::Vector2f(0.01, 0.01).asDiagonal();
        // ROS_INFO_STREAM("P0 is:"<<endl<<P_t<<endl;);

        setFilerPose();
    }

    void stateInitialization(Eigen::Matrix3f mat)
    {   
        float cov_pos = 1;
        float cov_vel = 25;
        float cov_bias = 1e-3;
        P_t.block<3,3>(POS_,POS_) = Eigen::Vector3f(cov_pos, cov_pos, cov_pos).asDiagonal();
        P_t.block<3,3>(ROT_, ROT_) = mat;
        P_t.block<3,3>(VEL_, VEL_) = Eigen::Vector3f(cov_vel, cov_vel, cov_vel).asDiagonal();
        P_t.block<6,6>(VEL_, VEL_) = Eigen::Matrix<float, 6, 6>::Identity()*cov_bias;
        if(!USE_S2)
            P_t.block<3,3>(GW_, GW_) = Eigen::Vector3f(0.01, 0.01, 0.01).asDiagonal();
        else 
            P_t.block<2,2>(GW_, GW_) = Eigen::Vector2f(0.01, 0.01).asDiagonal();
        // ROS_INFO_STREAM("P0 is:"<<endl<<P_t<<endl;);

        setFilerPose();
    }

    //convert frame from LIDAR to IMU
    void setFilerPose()
    {
        // Affine3f wTlidar = trans2Affine3f(transformTobeMapped);
        // wTimuAffine = wTlidar * exT_affine;

        wTimuAffine = trans2Affine3f(transformTobeMapped);
        filterState.qbn_ = wTimuAffine.rotation();
        filterState.rn_ = wTimuAffine.translation();
        // sysStatus = OTHER_SCAN;

        //Euler state
        filterState.euler_ = Eigen::Vector3f(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        //End
    }

    void statePredict()
    {
        biasLock.lock();
        filterState.ba_ = accBias.cast<float>();
        filterState.bw_ = gyrBias.cast<float>();
        biasLock.unlock();
        //imu    last|---------------|---------------|cur
        //lidar   last|***************|***************|cur


        // 将transformTobeMapped转成矩阵形式
        Eigen::Affine3f T_transformTobeMapped = trans2Affine3f(transformTobeMapped);
        // rotation, position, velocity
        Eigen::Quaternionf R_transformTobeMapped(T_transformTobeMapped.rotation());
        Eigen::Vector3f P_transformTobeMapped = T_transformTobeMapped.translation();
        Eigen::Vector3f V_transformTobeMapped = Eigen::Vector3f(transformTobeMapped[VEL_+0],
            transformTobeMapped[VEL_+1], transformTobeMapped[VEL_+2]);
        Eigen::Quaternionf R_transformTobeMappedlast = R_transformTobeMapped;
        Eigen::Vector3f P_transformTobeMappedlast = P_transformTobeMapped;
        Eigen::Vector3f V_transformTobeMappedlast = V_transformTobeMapped;

        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(T_transformTobeMapped, x, y, z, roll, pitch, yaw);
        file_predict.setf(std::ios::fixed, std::_S_floatfield);
        file_predict << frame_count << "," 
                << roll<< ","<< pitch << ","<< yaw << ","
                << x << ","<< y << ","<< z<< ","
                << V_transformTobeMapped[0] << ","<< V_transformTobeMapped[1] << ","<< V_transformTobeMapped[2]<< ","
                << std::endl;
        
        Eigen::Vector3f un_acc_last, un_gyr_last, un_acc_next, un_gyr_next;

        auto thisImu = getAccGyro(imuBucket[0]);
        auto lastImu = thisImu;
        double imuTime_last = imuBucket[0].header.stamp.toSec();
        double dt = 0.0;
        // convert to w frame (remove the grarity)
        un_acc_last = R_transformTobeMapped*(lastImu.first-filterState.ba_)+filterState.gn_;
        un_gyr_last = lastImu.second-filterState.bw_;
        for(size_t i=1; i<imuBucket.size(); i++)
        {
            thisImu = getAccGyro(imuBucket[i]);
            dt = imuBucket[i].header.stamp.toSec() - imuTime_last;
            // Get Rk+1 for mid integration
            un_gyr_next = thisImu.second-filterState.bw_;
            Eigen::Vector3f un_gyr = 0.5*(un_gyr_last + un_gyr_next);
            Eigen::Vector3f delta_angle_axis = un_gyr*dt;
            Eigen::Quaternionf dq(Eigen::AngleAxisf(delta_angle_axis.norm(), 
                                                      delta_angle_axis.normalized()).toRotationMatrix());
            R_transformTobeMapped = (R_transformTobeMapped*dq).normalized(); 


            un_acc_next = R_transformTobeMapped*(thisImu.first-filterState.ba_)+filterState.gn_;
            Eigen::Vector3f un_acc = 0.5*(un_acc_last + un_acc_next); // world frame

            P_transformTobeMapped = P_transformTobeMapped+V_transformTobeMapped*dt+0.5*un_acc*dt*dt;
            V_transformTobeMapped = V_transformTobeMapped+un_acc*dt;  

            float x, y, z, roll, pitch, yaw;
            Eigen::Affine3f tem;
            tem.setIdentity();
            tem.rotate(R_transformTobeMapped);
            tem.pretranslate(P_transformTobeMapped);
            pcl::getTranslationAndEulerAngles(tem, x, y, z, roll, pitch, yaw);
            file_predict.setf(std::ios::fixed, std::_S_floatfield);
            file_predict << imuBucket[i].header.stamp.toSec() << "," 
                    << roll<< ","<< pitch << ","<< yaw << ","
                    << x << ","<< y << ","<< z<< ","
                    << V_transformTobeMapped[0] << ","<< V_transformTobeMapped[1] << ","<< V_transformTobeMapped[2]<< ","
                    << std::endl;

            // predict relative transformation
            filterState.rn_ = P_transformTobeMapped;
            filterState.vn_ = V_transformTobeMapped;
            filterState.qbn_ = R_transformTobeMapped;
            // filterState.ba_?
            // filterState.bw_? 
            // filterState.gn_?

            F_t.setIdentity();
            Eigen::Vector3f midAcc = 0.5*(thisImu.first + lastImu.first);
            Eigen::Vector3f midGyr = 0.5*(thisImu.second + lastImu.second);
            Eigen::Vector3f AccMinusBiaslast = dt*(lastImu.first-filterState.ba_);
            Eigen::Vector3f AccMinusBias = dt*(thisImu.first-filterState.ba_);
            Eigen::Vector3f midGyrMinusBias = dt*(midGyr-filterState.bw_);
            Eigen::Matrix3f Rix = R_transformTobeMappedlast*GetSkewMatrix(AccMinusBiaslast);
            Eigen::Matrix3f Rjx = R_transformTobeMapped*GetSkewMatrix(AccMinusBias);

            // //method 1
            // F_t.block<3, 3>(ROT_, ROT_) = Eigen::Matrix3f::Identity() + GetSkewMatrix(-midGyrMinusBias);
            // F_t.block<3, 3>(ROT_, BIG_) = -dt*Eigen::Matrix3f::Identity();

            // F_t.block<3, 3>(VEL_, ROT_) = -0.5 * (Rix + Rjx*(Eigen::Matrix3f::Identity() -GetSkewMatrix(midGyrMinusBias)));
            // F_t.block<3, 3>(VEL_, BIA_) = -0.5 * (R_transformTobeMappedlast.toRotationMatrix() 
            //                                     + R_transformTobeMapped.toRotationMatrix()) * dt;
            // F_t.block<3, 3>(VEL_, BIG_) = 0.5 * Rjx * dt;
            // if(!USE_S2)
            //     F_t.block<3, 3>(VEL_, GW_)  = dt * Eigen::Matrix3f::Identity();
            // else
            // {
            //     Eigen::Vector3f g_temp = filterState.Gs2.get_vect();
            //     Eigen::Matrix<float, 3, 2>Bx;
            //     filterState.Gs2.S2_Bx(Bx);
            //     F_t.block<3, 2>(VEL_, GW_)  = dt * -GetSkewMatrix(g_temp) * Bx;
            // }

            // F_t.block<3, 3>(POS_, ROT_) = 0.5 * F_t.block<3, 3>(VEL_, ROT_) * dt;
            // F_t.block<3, 3>(POS_, VEL_) = dt*Eigen::Matrix3f::Identity();
            // F_t.block<3, 3>(POS_, BIA_) = 0.5 * F_t.block<3, 3>(VEL_, BIA_) * dt;
            // F_t.block<3, 3>(POS_, BIG_) = 0.5 * F_t.block<3, 3>(VEL_, BIG_) * dt;
            // if(!USE_S2)
            //     F_t.block<3, 3>(POS_, GW_)  = 0.5 * F_t.block<3, 3>(VEL_, GW_)  * dt;
            // else
            //     F_t.block<3, 2>(POS_, GW_)  = 0.5 * F_t.block<3, 2>(VEL_, GW_)  * dt;
            
            // G_t.setZero();
            // G_t.block<3, 3>(ROT_, 3) = 0.5 * Eigen::Matrix3f::Identity() * dt * 2;

            // G_t.block<3, 3>(VEL_, 0) = 0.5 * (R_transformTobeMapped.toRotationMatrix() + 
            //                                   R_transformTobeMappedlast.toRotationMatrix()) * dt;
            // G_t.block<3, 3>(VEL_, 3) = -0.25 * Rjx * dt * 2;

            // G_t.block<3, 3>(POS_, 0) = 0.5 * dt * G_t.block<3, 3>(VEL_, 0);
            // G_t.block<3, 3>(POS_, 3) = 0.5 * dt * G_t.block<3, 3>(VEL_, 3);

            // G_t.block<3, 3>(BIA_, 6) = Eigen::Matrix<float, 3, 3>::Identity() * dt;
            // G_t.block<3, 3>(BIG_, 9) = Eigen::Matrix<float, 3, 3>::Identity() * dt;

            // //noise_ (imuAccNoise, imuGyrNoise, imuAccBiasN, imuGyrBiasN)
            // P_t = F_t * P_t * F_t.transpose() + G_t * noise_ * G_t.transpose();


            //method 2
            F_t.setIdentity();
            midAcc = 0.5*(thisImu.first + lastImu.first);
            midGyr = 0.5*(thisImu.second + lastImu.second);
            F_t.block<3, 3>(POS_, VEL_) = dt*Eigen::Matrix3f::Identity();
            F_t.block<3, 3>(ROT_, ROT_) = Eigen::Matrix3f::Identity() + GetSkewMatrix(-dt*(midGyr-filterState.bw_));
            F_t.block<3, 3>(ROT_, BIG_) = -dt*Eigen::Matrix3f::Identity();
            F_t.block<3, 3>(VEL_, ROT_) = -dt*filterState.qbn_.toRotationMatrix()*GetSkewMatrix(midAcc-filterState.ba_);
            F_t.block<3, 3>(VEL_, BIA_) = -dt*filterState.qbn_.toRotationMatrix();
            if(!USE_S2)
                F_t.block<3, 3>(VEL_, GW_)  = dt * Eigen::Matrix3f::Identity();
            else
            {
                Eigen::Vector3f g_temp = filterState.Gs2.get_vect();
                Eigen::Matrix<float, 3, 2>Bx;
                filterState.Gs2.S2_Bx(Bx);
                F_t.block<3, 2>(VEL_, GW_)  = dt * -GetSkewMatrix(g_temp) * Bx;
            }

            G_t.setZero();
            G_t.block<3, 3>(VEL_, 0) = -filterState.qbn_.toRotationMatrix();
            G_t.block<3, 3>(ROT_, 3) = -Eigen::Matrix<float, 3, 3>::Identity();
            G_t.block<3, 3>(BIA_, 6) = Eigen::Matrix<float, 3, 3>::Identity();
            G_t.block<3, 3>(BIG_, 9) = Eigen::Matrix<float, 3, 3>::Identity();

            P_t = F_t * P_t * F_t.transpose() + (dt*G_t) * noise_ * (dt*G_t).transpose();



            imuTime_last = imuBucket[i].header.stamp.toSec();
            un_acc_last = un_acc_next;
            un_gyr_last = un_gyr_next;
            lastImu = thisImu;

            R_transformTobeMappedlast = R_transformTobeMapped; 
            V_transformTobeMappedlast = V_transformTobeMapped;
            P_transformTobeMappedlast = P_transformTobeMapped;
        }

        P_t_inv = P_t.colPivHouseholderQr().inverse();

        // ROS_INFO("**********F, G, P and P_inv***********");
        // ROS_INFO_STREAM(F_t<<endl<<endl);
        // ROS_INFO_STREAM(G_t<<endl<<endl);
        // // ROS_INFO_STREAM(P_t<<endl);
        // // ROS_INFO_STREAM(P_t_inv);
        // ROS_INFO("**********F, G, P and P_inv***********");

        // remap to transformTobeMapped
        T_transformTobeMapped.setIdentity();
        T_transformTobeMapped.pretranslate(P_transformTobeMapped);
        T_transformTobeMapped.rotate(R_transformTobeMapped);
        pcl::getTranslationAndEulerAngles(T_transformTobeMapped, 
            transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        ROS_INFO_STREAM("precdict state transformTobeMapped: "<<endl<<transformTobeMapped[3]<<" "
                                                                    <<transformTobeMapped[4]<<" "
                                                                    <<transformTobeMapped[5]<<endl);
        // 更新速度
        for(int i=0; i<3; i++)
            transformTobeMapped[VEL_+i] = V_transformTobeMapped(i, 0);

        ROS_INFO_STREAM("precdict state in IMU: "<<endl<<filterState.qbn_.vec().transpose()<<" "
                                                <<filterState.rn_.transpose()<<" "
                                                <<filterState.vn_.transpose()<<endl);
        Eigen::Matrix4f t;
        t.block<3,3>(0,0) = filterState.qbn_.toRotationMatrix();
        t.block<3,1>(0,3) = filterState.rn_;
        t(3, 3) = 1;
        Eigen::Affine3f temp(t);
        temp *= exT_inv_M4f;
        ROS_INFO_STREAM("precdict state in Lidar: "<<endl<<temp.translation().transpose()<<endl);
    }

    void updatePointAssociateToImuMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped) * exT_affine;
    }

    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    bool updateTransformationIESKF()
    {
        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            ROS_WARN("Feature Correspondences are %d which is less than 50", laserCloudSelNum);
            return false;
        }

        using V3f = Eigen::Vector3f;
        using M3f = Eigen::Matrix3f;
        using M4f = Eigen::Matrix4f;
        
        residual_   = Eigen::Matrix<float, Eigen::Dynamic, 1>::Zero(laserCloudSelNum, 1);
        R           = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Zero(laserCloudSelNum, laserCloudSelNum);
        H_k         = Eigen::Matrix<float, Eigen::Dynamic, 18>::Zero(laserCloudSelNum, 18);
        H_k_T_R_inv = Eigen::Matrix<float, 18, Eigen::Dynamic>::Zero(18, laserCloudSelNum);
        K_k         = Eigen::Matrix<float, 18, Eigen::Dynamic>::Zero(18, laserCloudSelNum);
        
        cout<<"Total corres nums: "<<laserCloudSelNum<<endl<<endl;

        //Euler state
        for(int i=0; i< laserCloudSelNum; ++i)
        {
            PointType pt_orig = laserCloudOri->points[i];
            PointXYZIRPYT feature_param = coeffSel->points[i];
            V3f pt_orig_coor = V3f(pt_orig.x, pt_orig.y, pt_orig.z);

            
            float roll = intermediateState.euler_[0], pitch = intermediateState.euler_[1], yaw = intermediateState.euler_[2];
            float croll = cos(roll), sroll = sin(roll), 
                  cpitch = cos(pitch), spitch = sin(pitch), 
                  cyaw = cos(yaw), syaw = sin(yaw);
            V3f Jr_pt = V3f(feature_param.x, feature_param.y, feature_param.z);
            M3f Jpt_Aa = intermediateState.qbn_ * GetSkewMatrix(-pt_orig_coor);
            M3f Jpt_trans = M3f::Identity();
            M3f Jpt_rot = M3f::Zero();

            Jpt_rot(0, 0) = (croll * spitch * cyaw + sroll * syaw) * pt_orig_coor[1] 
                          + (croll * syaw - sroll * spitch * cyaw) * pt_orig_coor[2];
            Jpt_rot(0, 1) = (-spitch * cyaw) * pt_orig_coor[0]
                          + (sroll * cpitch * cyaw) * pt_orig_coor[1]
                          + (croll * cpitch * cyaw) * pt_orig_coor[2];
            Jpt_rot(0, 2) = (-cpitch * syaw) * pt_orig_coor[0]
                          + (-sroll * spitch * syaw - croll * cyaw) * pt_orig_coor[1]
                          + (sroll * cyaw - croll * spitch * syaw) * pt_orig_coor[2];
            
            Jpt_rot(1, 0) = (-sroll * cyaw + croll * spitch * syaw) * pt_orig_coor[1]
                          + (-sroll * spitch * syaw - croll * cyaw) * pt_orig_coor[2];
            Jpt_rot(1, 1) = (-spitch * syaw) * pt_orig_coor[0]
                          + (sroll * cpitch * syaw) * pt_orig_coor[1];
                          + (croll * cpitch * syaw) * pt_orig_coor[2];
            Jpt_rot(1, 2) = (cpitch * cyaw) * pt_orig_coor[0]
                          + (-croll * syaw + sroll * spitch * cyaw) * pt_orig_coor[1]
                          + (croll * spitch * cyaw + sroll * syaw) * pt_orig_coor[2];
            
            Jpt_rot(2, 0) = (croll * cpitch) * pt_orig_coor[1]
                          + (-sroll * cpitch) * pt_orig_coor[2];
            Jpt_rot(2, 1) = (-cpitch) * pt_orig_coor[0]
                          + (-sroll * spitch) * pt_orig_coor[1]
                          + (-croll * spitch) * pt_orig_coor[2];
            Jpt_rot(2, 2) = 0;


            H_k.block<1, 3>(i, POS_) = Jr_pt.transpose() * Jpt_trans / feature_param.roll;
            H_k.block<1, 3>(i, ROT_) = Jr_pt.transpose() * Jpt_rot / feature_param.roll;
            // H_k(i, ROT_) = 0;
            // H_k(i, ROT_+1) = 0;

            H_k_T_R_inv.block<6, 1>(0, i) = H_k.block<1, 6>(i, 0).transpose() * feature_param.roll;
            residual_(i, 0) = 0 - feature_param.intensity / feature_param.roll;
            R(i, i) = 1.0 / feature_param.roll;
        }
        
        errVec_ = move(intermediateState - filterState);
        // ROS_INFO_STREAM("errVec_: "<<endl<<errVec_.transpose()<<endl<<endl);

        //check LIDAR_STD
        Eigen::Matrix<float, 18, 18> P_tmp = LIDAR_STD * P_t_inv;
        
        HRH = H_k_T_R_inv*H_k+P_tmp;
        // ROS_INFO_STREAM("HRH: "<<endl<<HRH<<endl<<endl);
        HRH_inv = HRH.colPivHouseholderQr().inverse();
        K_k = HRH_inv * H_k_T_R_inv;
        // ROS_INFO_STREAM("K_k: "<<endl<<K_k<<endl<<endl);

        //calculate dxj0
        updateVec_ = - (errVec_) + K_k * (residual_ + H_k * errVec_);
        // ROS_INFO_STREAM("updateVec_: "<<endl<<updateVec_<<endl<<endl);

        // Divergence determination
        bool hasNaN = false;
        for (int i = 0; i < updateVec_.rows(); i++) {
            if (isnan(updateVec_(i, 0))) {
                updateVec_(i, 0) = 0;
                hasNaN = true;
            }
        }
        if (hasNaN == true) {
            ROS_WARN("System diverges Because of NaN...");
            hasDiverged = true;
            return false;
        }
        // // Check whether the filter converges
        // if (residual_.norm() > residualNorm * 10) {
        //     ROS_WARN("System diverges...");
        //     hasDiverged = true;
        //     return false;
        // }

        intermediateState += updateVec_;
        M4f tmpM = M4f::Identity();
        tmpM.block<3, 3>(0, 0) = intermediateState.qbn_.toRotationMatrix();
        tmpM.block<3, 1>(0, 3) = intermediateState.rn_;
        Affine2trans(transformTobeMapped, Eigen::Affine3f(tmpM));
        // ROS_INFO_STREAM("updated state: "<<endl<<intermediateState.qbn_.vec()<<" "
        //                                         <<intermediateState.rn_.transpose()<<" "
        //                                         <<intermediateState.vn_.transpose()<<endl);

        file_update.setf(std::ios::fixed, std::_S_floatfield);
        file_update << frame_count << "," 
                << updateVec_(0, 0)<< ","<< updateVec_(1, 0) << ","<< updateVec_(2, 0) << ","
                << updateVec_(3, 0)<< ","<< updateVec_(4, 0) << ","<< updateVec_(5, 0) << ","
                << updateVec_(6, 0)<< ","<< updateVec_(7, 0) << ","<< updateVec_(8, 0) << ","
                << updateVec_(9, 0)<< ","<< updateVec_(10, 0) << ","<< updateVec_(11, 0) << ","
                << updateVec_(12, 0)<< ","<< updateVec_(13, 0) << ","<< updateVec_(14, 0) << ","
                << std::endl;

        //check convegence
        //transition and rotation vector
        if(updateVec_.block<3, 1>(POS_, 0).norm()/*100*/ < 0.015 && updateVec_.block<3, 1>(ROT_, 0).norm() < 0.001)
        {
            return true;
        }
        else return false;
    }

    void updateState(double dt)
    {
        TicToc t_1;
        Eigen::MatrixXf partial = I18 - K_k * H_k;
        t_1.toc("Calculate I-KH");
        // //method 1 P
        // P_t = (I18 - K_k * H_k) * J_t * P_t * J_t.transpose();
        // //method 2 P positive and symmetric
        TicToc t_filter;
        P_t = partial * J_t * P_t * J_t.transpose() * partial.transpose() + K_k * R * K_k.transpose();
        t_filter.toc("Calculate P_t");
        // //method 3 P symmetric
        // Eigen::MatrixXf Ptangent = J_t * P_t * J_t.transpose();
        // P_t = Ptangent - K_k * (H_k * Ptangent * H_k.transpose() + R) * K_k.transpose();

        //set kth pose in lidar frame
        filterState = intermediateState;
        cout<<"更新后的states: "<<endl
                            <<filterState.euler_.transpose()<<" "<<endl
                            <<filterState.rn_.transpose()<<" "<<endl
                            <<filterState.vn_.transpose()<<" "<<endl
                            <<filterState.ba_.transpose()<<" "<<endl
                            <<filterState.bw_.transpose()<<" "<<endl
                            <<filterState.gn_.transpose()<<" "<<endl
                            <<endl;
                            

        transformTobeMapped[0] = filterState.euler_[0];
        transformTobeMapped[1] = filterState.euler_[1];
        transformTobeMapped[2] = filterState.euler_[2];

        transformTobeMapped[3] = filterState.rn_[0];
        transformTobeMapped[4] = filterState.rn_[1];
        transformTobeMapped[5] = filterState.rn_[2];

        for(int i=0; i<3; i++)
        {
            // 看作匀加速运动
            // transformTobeMapped[VEL_+i] = 2.0*(transformTobeMapped[POS_+i]-transformTobeMappedLast[POS_+i])/dt-transformTobeMappedLast[VEL_+i];
            // filterState.vn_(i) = transformTobeMapped[VEL_+i];

            // 看作匀速运动
            transformTobeMapped[VEL_+i] = (transformTobeMapped[3+i]-transformTobeMappedLast[3+i])/dt;
            filterState.vn_(i) = transformTobeMapped[VEL_+i];

            //速度直接从滤波结果获取
            // transformTobeMapped[VEL_+i] = filterState.vn_[i];
        }
        for(int i=0; i<18; ++i) transformTobeMappedLast[i] = transformTobeMapped[i];
    }

    void updateStatebyNotConverge(double dt)
    {
        ROS_WARN("Registration Update dosen't converge! Using predict states!");
        cout<<"预测的states: "<<endl
                            <<filterState.qbn_.vec().transpose()<<" "<<endl
                            <<filterState.rn_.transpose()<<" "<<endl
                            <<filterState.vn_.transpose()<<" "<<endl
                            <<filterState.ba_.transpose()<<" "<<endl
                            <<filterState.bw_.transpose()<<" "<<endl
                            <<filterState.gn_.transpose()<<" "<<endl
                            <<endl;
        
        
        transformTobeMapped[0] = filterState.euler_[0];
        transformTobeMapped[1] = filterState.euler_[1];
        transformTobeMapped[2] = filterState.euler_[2];

        transformTobeMapped[3] = filterState.rn_[0];
        transformTobeMapped[4] = filterState.rn_[1];
        transformTobeMapped[5] = filterState.rn_[2];

        for(int i=0; i<3; i++)
        {
            transformTobeMapped[VEL_+i] = filterState.vn_[i];
            // // 看作匀加速运动
            // transformTobeMapped[VEL_+i] = 2.0*(transformTobeMapped[POS_+i]-transformTobeMappedLast[POS_+i])/dt-transformTobeMappedLast[VEL_+i];
            // filterState.vn_(i) = transformTobeMapped[VEL_+i];
        }
        for(int i=0; i<18; ++i) transformTobeMappedLast[i] = transformTobeMapped[i];
    }

    void updateStatebyIeskf(double dt)
    {
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            converge = false;
            intermediateState = filterState;
            
            for (int iterCount = 0; iterCount < MaxoptIteration; iterCount++)
            {   
                cout<<"**********第"<<frame_count<<"的第"<<iterCount<<"次优化**********"<<endl;

                //feature test
                CorCorrespondence->clear();
                SurCorrespondence->clear();
                CornerSelected->clear();
                SurfSelected->clear();

                CorCorresCount = 1;
                SurCorresCount = 1;
                //END

                laserCloudOri->clear();
                coeffSel->clear();
                updatePointAssociateToMap();
                if(USE_CORNER)
                {
                    cornerOptimization();
                }
                surfOptimization();
                combineOptimizationCoeffs(iterCount);
                converge = updateTransformationIESKF();

                if (converge)
                {   
                    //feature test
                    publishCloud(&pubCorCorres, CorCorrespondence, timeLaserInfoStamp, odometryFrame);
                    publishCloud(&pubSurCorres, SurCorrespondence, timeLaserInfoStamp, odometryFrame);
                    publishCloud(&pubCorSelected, CornerSelected, timeLaserInfoStamp, odometryFrame);
                    publishCloud(&pubSurSelected, SurfSelected, timeLaserInfoStamp, odometryFrame);
                    //END
                    
                    //update States, covariance and transform pose to lidar frame
                    updateState(dt);
                    break;
                }
                else if(!converge && iterCount == MaxoptIteration-1)
                {
                    updateStatebyNotConverge(dt);
                }

                // if(hasDiverged)
                // {
                //     //还没写完
                //     resetState();
                //     break;
                // }          
            }
            transformUpdate();
        } else {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    bool removeOldImu()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        sensor_msgs::Imu frontImu;
        imuBucket.clear();

        while(!imuQueue.empty())
        {
            if(imuQueue.front().header.stamp.toSec()<timeLaserInfoCur)
            {
                frontImu = imuQueue.front();
                if(sysStatus == OTHER_SCAN) imuBucket.emplace_back(frontImu);
                imuQueue.pop_front();
            }
            else
                break;
        }
        imuQueue.push_front(frontImu); // Imu coverage lidar

        if(sysStatus == FIRST_SCAN || imuBucket.size() > 3) return true;
        else 
        {
            ROS_WARN("Extract IMU Bucket Fails!!");
            return false;
        }
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
            pcl::PointCloud<PointType>::Ptr CornerImuPC(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr SurfImuPC(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*SubMapSetCorner[pointSearchInd[i]], *CornerImuPC, exT_inv_M4f);
            pcl::transformPointCloud(*SubMapSetCorner[pointSearchInd[i]], *SurfImuPC, exT_inv_M4f);
            *laserCloudCornerFromMap += *CornerImuPC;
            *laserCloudSurfFromMap += *SurfImuPC;
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
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 5.0)
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

        std::unordered_set<int> KeyInd_sets;

        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {  
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            int modifiedKeyInd = Key_point_count<=keyframenumber ? (int)cloudToExtract->points[i].intensity : 
                                                                   (int)cloudToExtract->points[i].intensity - (Key_point_count - keyframenumber);
            int thisKeyInd = (int)cloudToExtract->points[i].intensity;

            if (KeyInd_sets.find(thisKeyInd) == KeyInd_sets.end()) KeyInd_sets.insert(thisKeyInd);
            else continue;

            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // transformed cloud available
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[modifiedKeyInd],  &cloudKeyPoses6D->points[modifiedKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[modifiedKeyInd],    &cloudKeyPoses6D->points[modifiedKeyInd]);
                
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
        if (laserCloudMapContainer.size() > size_t(containersize)){
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

        //Now the proposed Pose is Timu->map, so each scan in Lidar should to be transformed to Imu
        pcl::PointCloud<PointType>::Ptr CornerImuPC(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr SurfImuPC(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*laserCloudCornerLastDS, *CornerImuPC, exT_inv_M4f);
        pcl::transformPointCloud(*laserCloudSurfLastDS, *SurfImuPC, exT_inv_M4f);
        *laserCloudCornerLastDS = *CornerImuPC;
        *laserCloudSurfLastDS   = *SurfImuPC;

        CornerImuPC.reset(new pcl::PointCloud<PointType>());
        SurfImuPC.reset(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*laserCloudCornerlessLast, *CornerImuPC, exT_inv_M4f);
        pcl::transformPointCloud(*laserCloudSurflessLast, *SurfImuPC, exT_inv_M4f);
        *laserCloudCornerlessLast = *CornerImuPC;
        *laserCloudSurflessLast   = *SurfImuPC;
    }

    void cornerOptimization()
    {   
        //Switch to Super Feature
        if(USEFULLFEATURE)
        {
            #pragma omp parallel for num_threads(numberOfCores)
            for (int i = 0; i < laserCloudCornerLastDSNum; i++)
            {
                {
                    PointType pointOri, pointSel;
                    PointXYZIRPYT coeff;
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
                            coeff.roll = s;

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
                PointType pointOri, pointSel;
                PointXYZIRPYT coeff;
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
                        coeff.roll = s;

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
        if(USEFULLFEATURE)
        {
            #pragma omp parallel for num_threads(numberOfCores)
            for (int i = 0; i < laserCloudSurfLastDSNum; i++)
            {
                PointType pointOri, pointSel;
                PointXYZIRPYT coeff;
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

                    //px = x/d
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
                    // static int cnt = 0;
                    if (planeValid) {
                        float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;
                        // if(cnt++%20 == 0){
                        //     cout<<pd2<<endl;
                        //     // cout<<"旋转前的点: "<<pointOri.x<<" "<<pointOri.y<<" "<<pointOri.z<<endl;
                        //     // cout<<"旋转后的点: "<<pointSel.x<<" "<<pointSel.y<<" "<<pointSel.z<<endl;
                        //     // for(int i=0; i<5; ++i){
                        //     //     cout<<laserCloudSurfFromMapDS->points[pointSearchInd[i]].x<<" "
                        //     //         <<laserCloudSurfFromMapDS->points[pointSearchInd[i]].y<<" "
                        //     //         <<laserCloudSurfFromMapDS->points[pointSearchInd[i]].z<<" "<<endl;
                        //     // }
                        // }

                        float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                                + pointSel.y * pointSel.y + pointSel.z * pointSel.z));
                        
                        //这里是可以尝试用voxelmap的方法来估计的
                        coeff.x = s * pa;
                        coeff.y = s * pb;
                        coeff.z = s * pc;
                        coeff.intensity = s * pd2;
                        coeff.roll = s;

                        if (s > 0.9) {
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
                PointType pointOri, pointSel;
                PointXYZIRPYT coeff;
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
                        coeff.roll = s;

                        if (s > 0.9) {

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
            if(USE_CORNER)
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
            if (USE_CORNER)
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

                laserCloudOri->clear();
                coeffSel->clear();

                updatePointAssociateToMap();
                cornerOptimization();
                surfOptimization();
                combineOptimizationCoeffs(iterCount);


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
        Imu2Lidar();
        incrementalOdometryAffineBack = trans2Affine3f(transformInLidar);

        cout<<"REGISTRATION POSITION: "<<"["
            <<transformInLidar[0]<<", "
            <<transformInLidar[1]<<", "
            <<transformInLidar[2]<<", "
            <<transformInLidar[3]<<", "
            <<transformInLidar[4]<<", "
            <<transformInLidar[5]<<"]"<<endl<<endl;

        if(sysStatus == FIRST_SCAN) setFilerPose();
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
   

    void saveKeyFrames()
    {
        if (saveFrame() == false)
            return;

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;

        thisPose3D.x = transformTobeMapped[3];
        thisPose3D.y = transformTobeMapped[4];
        thisPose3D.z = transformTobeMapped[5];
        
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
        thisPose6D.roll  = transformTobeMapped[0];
        thisPose6D.pitch = transformTobeMapped[1];
        thisPose6D.yaw   = transformTobeMapped[2];
        // cout<<"The optimized RPY are: "<<endl<<thisPose6D.roll<<endl<<thisPose6D.pitch<<endl<<thisPose6D.yaw<<endl<<endl;
        // cout<<"save time "<<std::setprecision(15)<<timeLaserInfoCur<<endl;
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);
        if(static_cast<int>(cloudKeyPoses6D->size())>keyframenumber) cloudKeyPoses6D->erase(cloudKeyPoses6D->begin());
        
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
    }

    

    void publishOdometry()
    {
        //check
        Imu2Lidar();

        // Publish odometry for ROS (global)
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        // laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        // laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        // laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        // laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        laserOdometryROS.pose.pose.position.x = transformInLidar[3];
        laserOdometryROS.pose.pose.position.y = transformInLidar[4];
        laserOdometryROS.pose.pose.position.z = transformInLidar[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformInLidar[0], transformInLidar[1], transformInLidar[2]);
        
        pubLaserOdometryGlobal.publish(laserOdometryROS);

        cout<<"GLOBAL POSITION: "<<"["
            <<transformInLidar[0]<<", "
            <<transformInLidar[1]<<", "
            <<transformInLidar[2]<<", "
            <<transformInLidar[3]<<", "
            <<transformInLidar[4]<<", "
            <<transformInLidar[5]<<"]"<<endl<<endl;

        // file_lidar.setf(std::ios::fixed, std::_S_floatfield);
        // file_lidar << laserOdometryROS.header.stamp.toSec() << " " 
        //         << laserOdometryROS.pose.pose.position.x << " "
        //         << laserOdometryROS.pose.pose.position.y << " "
        //         << laserOdometryROS.pose.pose.position.z << " "
        //         << laserOdometryROS.pose.pose.orientation.x << " "
        //         << laserOdometryROS.pose.pose.orientation.y << " "
        //         << laserOdometryROS.pose.pose.orientation.z << " "
        //         << laserOdometryROS.pose.pose.orientation.w << std::endl;
        
        geometry_msgs::Quaternion q_temp = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        file_lidar.setf(std::ios::fixed, std::_S_floatfield);
        file_lidar << laserOdometryROS.header.stamp.toSec() << " " 
                << transformTobeMapped[3] << " "<< transformTobeMapped[4] << " "<< transformTobeMapped[5] << " "
                << q_temp.x << " "<< q_temp.y << " "<< q_temp.z << " "<< q_temp.w
                << std::endl;
        
        // Publish TF
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformInLidar[0], transformInLidar[1], transformInLidar[2]),
                                                      tf::Vector3(transformInLidar[3], transformInLidar[4], transformInLidar[5]));
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
            //check
            increOdomAffine = trans2Affine3f(transformInLidar);
        } else {
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            // if (cloudInfo.imuAvailable == true)
            // {
            //     if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            //     {
            //         double imuWeight = 0.1;
            //         tf::Quaternion imuQuaternion;
            //         tf::Quaternion transformQuaternion;
            //         double rollMid, pitchMid, yawMid;

            //         // slerp roll
            //         transformQuaternion.setRPY(roll, 0, 0);
            //         imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
            //         tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
            //         roll = rollMid;

            //         // slerp pitch
            //         transformQuaternion.setRPY(0, pitch, 0);
            //         imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
            //         tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
            //         pitch = pitchMid;
            //     }
            // }
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

    ROS_INFO("\033[1;32m----> Map Filter Optimization Started.\033[0m");
    
    // std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ROS_INFO("Spinning node");

    ros::spin();

    // loopthread.join();
    visualizeMapThread.join();

    return 0;
}
