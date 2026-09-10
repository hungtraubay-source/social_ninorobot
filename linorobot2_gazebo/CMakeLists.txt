cmake_minimum_required(VERSION 3.5)
project(linorobot2_gazebo)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(gazebo REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(social_perception REQUIRED)

add_library(animated_people_release SHARED
  src/animated_people_release.cpp)
target_include_directories(animated_people_release PRIVATE
  ${GAZEBO_INCLUDE_DIRS})
target_link_libraries(animated_people_release ${GAZEBO_LIBRARIES})
ament_target_dependencies(animated_people_release
  rclcpp std_msgs gazebo_ros social_perception)

execute_process(
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/map2gazebo.py
  --map_dir ${CMAKE_CURRENT_SOURCE_DIR}/../linorobot2_navigation/maps
  --model_dir ${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/models
  --world_dir ${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/worlds
  RESULT_VARIABLE SCRIPT_RESULT
  OUTPUT_VARIABLE SCRIPT_OUTPUT
)

if(NOT SCRIPT_RESULT EQUAL 0)
  message(WARNING "Python script execution failed with error code: ${SCRIPT_RESULT}")
  message(WARNING "Script output: ${SCRIPT_OUTPUT}")
endif()

install(
  DIRECTORY launch rviz worlds
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS animated_people_release
  LIBRARY DESTINATION lib)

install(PROGRAMS
  scripts/command_timeout.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/linorobot2_gazebo.sh.in")
ament_package()
