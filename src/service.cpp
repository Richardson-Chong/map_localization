#include <utility.h>

#include <dynamic_reconfigure/server.h>
#include "lio_sam/LIO_SAM_PARAMConfig.h"

void callback(lio_sam_dynamic_param::LIO_SAM_PARAMConfig &config)
{
  ROS_INFO("Reconfigure Request: %s",
           config.useGps?"True":"False");
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "custom_dynamic_server");

  dynamic_reconfigure::Server<lio_sam_dynamic_param::LIO_SAM_PARAMConfig> server;
  dynamic_reconfigure::Server<lio_sam_dynamic_param::LIO_SAM_PARAMConfig>::CallbackType f;

  f = boost::bind(&callback, _1); //绑定回调函数
  server.setCallback(f); //为服务器设置回调函数， 节点程序运行时会调用一次回调函数来输出当前的参数配置情况

  ROS_INFO("Spinning node");
  ros::spin(); //服务器循环监听重配置请求，当服务器收到重配置请求的时候，就会自动调用回调函数
  return 0;
}
