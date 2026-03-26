cmake_minimum_required(VERSION 3.5)
project(ros_driver_package)

find_package(catkin REQUIRED COMPONENTS
  rospy
  rospack
  sensor_msgs
  cv_bridge
  duckietown_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  src/driver_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY
    launch
    config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)