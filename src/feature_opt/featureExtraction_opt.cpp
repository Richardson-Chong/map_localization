#include "utility.h"
#include "map_localization/cloud_info.h"
#include "ClosedPointsPlane.h"

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:

    ros::Subscriber subLaserCloudInfo;

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubCornerSuperSharpPoints;
    ros::Publisher pubSurfacePoints;
    ros::Publisher pubSurfaceSuperFlatPoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud_Super;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud_Super;

    pcl::VoxelGrid<PointType> downSizeFilter;

    map_localization::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

    //点云曲率集合
    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    //特征提取的标记，1表示遮挡、平行，或者已经进行过特征提取的点，0表示还未进行特征提取处理
    int *cloudNeighborPicked;
    //可以像lego_loam一样暂时就只添加一个noise集合，包括label为999999和聚类不超过30个点的那种，但是得考虑效率
    //1表示corner，2表示surf
    int *cloudLabel;

    FeatureExtraction()
    {
        subLaserCloudInfo = nh.subscribe<map_localization::cloud_info>("lio_sam/deskew/cloud_info", 1, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());

        pubLaserCloudInfo = nh.advertise<map_localization::cloud_info> ("lio_sam/feature/cloud_info", 1);
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_corner", 1);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_surface", 1);
        pubCornerSuperSharpPoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_corner_Super_sharp", 1);
        pubSurfaceSuperFlatPoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_surface_Super_flat", 1);
        
        initializationValue();
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud_Super.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud_Super.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }

    void laserCloudInfoHandler(const map_localization::cloud_infoConstPtr& msgIn)
    {
        cloudInfo = *msgIn; // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction
        
        calculateSmoothness();
        markOccludedPoints();
        extractFeatures();
        publishFeatureCloud();
    }

    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++)
        {   
            // 用当前激光点前后5个点计算当前点的曲率，平坦位置处曲率较小，角点处曲率较大；这个方法很简单但有效
            float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
                            + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
                            + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 10
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
                            + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
                            + cloudInfo.pointRange[i+5];            

            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];

            // 两个激光点之间的一维索引差值，如果在一条扫描线上，那么值为1；如果两个点之间有一些无效点被剔除了，可能会比1大，但不会特别大
            // 如果恰好前一个点在扫描一周的结束时刻，下一个点是另一条扫描线的起始时刻，那么值会很大
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));

            if (columnDiff < 10){
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));

            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    /**
     * 点云角点、平面点特征提取
     * 1、遍历扫描线，每根扫描线扫描一周的点云划分为6段，针对每段提取20个角点、不限数量的平面点，加入角点集合、平面点集合
     * 2、认为非角点的点都是平面点，加入平面点云集合，最后降采样
    */
    void extractFeatures()
    {
        cornerCloud->clear();
        cornerCloud_Super->clear();
        surfaceCloud->clear();
        surfaceCloud_Super->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        int super_surf_num = 0;
        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();

            for (int j = 0; j < 6; j++)
            {

                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                int largestPickedNumCorner = 0;
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && 
                        cloudCurvature[ind] > edgeThreshold 
                        )
                    {
                        largestPickedNumCorner++;
                        if (largestPickedNumCorner <= MaxCornerSize){
                            cloudLabel[ind] = 1;
                            //change
                            if(largestPickedNumCorner <= MaxCornerSize) 
                                cornerCloud_Super->push_back(extractedCloud->points[ind]);
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;

                        // 同一条扫描线上后5个点标记一下，不再处理，避免特征聚集
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                
                int largestPickedNumSurf = 0;
                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {
                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;

                        if(largestPickedNumSurf <= MaxSurfSize && cloudLabel[ind] < 0) 
                        {
                            largestPickedNumSurf++;
                            surfaceCloud_Super->push_back(extractedCloud->points[ind]);
                        }

                        for (int l = 1; l <= 5; l++) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0){
                        //smoothness大小测试
                        extractedCloud->points[k].intensity = cloudCurvature[cloudSmoothness[k].ind]*100;
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }

            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;
            
            // cout<<"the "<<i<<"th ring has "<<surfaceCloud_Super->size() - super_surf_num<<" super surf points"<<endl;
            // super_surf_num = surfaceCloud_Super->size();
        }
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        cloudInfo.cloud_corner  = publishCloud(&pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_corner_Super  = 
                    publishCloud(&pubCornerSuperSharpPoints,  cornerCloud_Super,  cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(&pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface_Super = 
                    publishCloud(&pubSurfaceSuperFlatPoints, surfaceCloud_Super, cloudHeader.stamp, lidarFrame);
        // cout<<"Super Points are "<<surfaceCloud_Super->size()<<endl;
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Feature Extraction Started.\033[0m");
   
    ros::spin();

    return 0;
}