cmake_minimum_required(VERSION 3.0.2)

project(zed_yolo_ros)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  roslaunch
  zed_interfaces
  message_generation
  roscpp
  rospy
  std_msgs
)

add_message_files(
  DIRECTORY msg
  FILES
  Object.msg
  ObjectsStamped.msg
  Skeleton3D.msg
  Keypoint3D.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)
###################################
## catkin specific configuration ##
###################################

catkin_package(
  CATKIN_DEPENDS message_runtime roscpp rospy std_msgs

)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############
catkin_python_setup()

catkin_install_python(PROGRAMS scripts/detector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

