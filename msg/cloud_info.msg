# Cloud Info
Header header 

int32[] startRingIndex
int32[] endRingIndex

int32[]  pointColInd # point column index in range image
float32[] pointRange # point range 
#lego
bool[]    segmentedCloudGroundFlag

int64 imuAvailable
int64 odomAvailable

# Attitude for LOAM initialization
float32 imuRollInit
float32 imuPitchInit
float32 imuYawInit

# Initial guess from imu pre-integration
float32 initialGuessX
float32 initialGuessY
float32 initialGuessZ
float32 initialGuessRoll
float32 initialGuessPitch
float32 initialGuessYaw

#Initial ground state
float32 ground_A
float32 ground_B
float32 ground_C

#gps odom
int32 GPS_INTER_AVA
float32[] mapQlk
float32[] maptlk

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed
sensor_msgs/PointCloud2 cloud_corner_Super    # extracted corner feature(Super)
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature(Less)
sensor_msgs/PointCloud2 cloud_surface_Super   # extracted surface feature(Super)
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature(Less)
sensor_msgs/PointCloud2 cloud_ground    # extracted ground points