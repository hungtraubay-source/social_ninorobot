# terminal 1:
source install/setup.bash
ros2 launch linorobot2_gazebo gazebo.launch.py run_ekf:=true


source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch linorobot2_gazebo gazebo.launch.py \
  run_ekf:=true \
  spawn_x:=-3.0 \
  spawn_y:=0.0 \
  spawn_z:=0.0 \
  spawn_yaw:=0.0 \
  publish_odom_tf:=false


source /opt/ros/humble/setup.bash
source /home/hung/ninorobot2/install/setup.bash

ros2 run tf2_ros static_transform_publisher \
  --x 0 --y 0 --z 0 --roll 0 --pitch 0 --yaw 0 \
  --frame-id world --child-frame-id odom


# terminal 2:
source install/setup.bash
ros2 launch social_navigation social_bringup.launch.py \
  sim:=true \
  perception:=true \
  vlm_interaction:=true \
  vlm_interaction_config:=/home/hung/ninorobot2/social_perception/config/social_vlm_interaction.yaml \
  yolo_model_path:=/home/hung/ninorobot2/yolov8n.pt \
  rviz:=false


  source /opt/ros/humble/setup.bash
source install/setup.bash

source install/setup.bash
ros2 launch social_navigation social_bringup.launch.py yolo_model_path:=/home/hung/my_amr_thesis_test-main/src/my_amr_perception/weights/yolo26n-pose.pt

#### 
source /opt/ros/humble/setup.bash
source install/setup.bash


## terminal 3:

source install/setup.bash
ros2 launch social_navigation social_sim.launch.py \
  scenario:=talking \
  wait_for_perception:=true


source install/setup.bash
ros2 topic pub --once /animated_people/scenario std_msgs/msg/String "{data: talking}"

##### terminal 4:

source install/setup.bash
ros2 launch social_rl rl_agent.launch.py \
  model_path:=/home/hung/social_rl_runs/20260922_235954/checkpoints/recurrent_ppo_495352_steps.zip \
  sim:=true \
  localization:=true \
  rviz:=true \
  people:=perception \
  zones:=true





source install/setup.bash

ros2 run social_rl rl_agent --ros-args \
  -p use_sim_time:=true \
  -p model_path:=/home/hung/social_rl_runs/20260922_235954/checkpoints/recurrent_ppo_495352_steps.zip \
  -p people_topic_override:=/people/tracks \
  -p cmd_vel_topic:=/cmd_vel


  ros2 launch social_rl rl_agent.launch.py \
  model_path:=/home/hung/social_rl_runs/20260922_235954/checkpoints/recurrent_ppo_495352_steps.zip \
  people:=ground_truth \
  localization:=true \
  rviz:=true