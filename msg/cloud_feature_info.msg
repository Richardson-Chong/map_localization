# Cloud Info
Header header 

int32[] startRingIndex
int32[] endRingIndex

int32[]  pointColInd # point column index in range image
float32[] pointRange # point range 
float32[] pointAngle # point Angle
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

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed
sensor_msgs/PointCloud2 cloud_corner_outward
sensor_msgs/PointCloud2 cloud_corner_inward
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature

sensor_msgs/PointCloud2 cloud_ground    # extracted ground points