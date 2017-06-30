cmake_minimum_required(VERSION 2.8.3)
project(int_estimator)

find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
  std_msgs
  message_generation
  interruptibility_msgs
)

add_message_files(
    DIRECTORY msg
    FILES
    Estimate.msg
)

add_service_files(
  DIRECTORY srv
  FILES
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
  interruptibility_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
