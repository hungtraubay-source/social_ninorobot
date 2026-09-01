cmake_minimum_required(VERSION 3.8)
project(social_perception)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Person.msg"
  "msg/People.msg"
  "msg/Keypoint2D.msg"
  "msg/Keypoint3D.msg"
  "msg/PersonObservation.msg"
  "msg/PeopleObservations.msg"
  "msg/PersonOrientation.msg"
  "msg/PeopleOrientations.msg"
  "msg/TalkingInteraction.msg"
  "msg/TalkingInteractions.msg"
  "msg/VlmPersonState.msg"
  "msg/VlmPersonStates.msg"
  "msg/SocialConstraintLayer.msg"
  "msg/SocialConstraintLayers.msg"
  DEPENDENCIES builtin_interfaces geometry_msgs nav_msgs std_msgs
)

set(SOCIAL_PERCEPTION_SCRIPTS
  scripts/social_vlm_perception.py
  scripts/social_vlm_interaction.py
  scripts/social_constraint_grounding.py)
# Keep the package buildable while optional recorder implementations are not
# present in this checkout. If either source is restored later it is installed
# automatically, so its existing launch file starts working without another
# CMake edit.
foreach(RECORDER_SCRIPT
    scripts/four_frame_vlm_recorder.py
    scripts/dataset_snapshot_button.py
    scripts/gazebo_collection_motion.py
    scripts/social_dataset_recorder.py
    scripts/latest_image_history_recorder.py)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${RECORDER_SCRIPT}")
    list(APPEND SOCIAL_PERCEPTION_SCRIPTS ${RECORDER_SCRIPT})
  endif()
endforeach()

install(PROGRAMS
  ${SOCIAL_PERCEPTION_SCRIPTS}
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)
install(FILES README.md requirements.txt DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)
ament_package()
