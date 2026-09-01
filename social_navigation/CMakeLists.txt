cmake_minimum_required(VERSION 3.8)
project(social_navigation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_util REQUIRED)
find_package(pluginlib REQUIRED)
find_package(social_perception REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Make model://person resolvable by a Gazebo process launched after sourcing
# this workspace. This also keeps the model independent of ~/.gazebo/models.
ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/social_navigation_gazebo.sh.in")

add_library(social_layer SHARED src/social_layer.cpp)
target_include_directories(social_layer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(social_layer PUBLIC cxx_std_17)
ament_target_dependencies(social_layer
  geometry_msgs nav2_costmap_2d nav_msgs nav2_util pluginlib rclcpp rclcpp_lifecycle social_perception
  tf2 tf2_geometry_msgs)
pluginlib_export_plugin_description_file(nav2_costmap_2d social_layer_plugin.xml)

install(TARGETS social_layer
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(FILES social_layer_plugin.xml DESTINATION share/${PROJECT_NAME})
install(FILES README.md RUN_SCENARIOS.txt DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch config models rviz DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)
install(PROGRAMS
  scripts/release_animated_people.py
  scripts/social_velocity_filter.py
  DESTINATION lib/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(social_layer)
ament_export_dependencies(social_perception)
ament_package()
