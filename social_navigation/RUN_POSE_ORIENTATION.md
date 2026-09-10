===============================================================================
 LỆNH CHẠY YOLO POSE + KEYPOINT + ORIENTATION TRÊN RVIZ
===============================================================================

Mở từng terminal theo thứ tự dưới đây. Không bật RViz ở Terminal 4.

-------------------------------------------------------------------------------
 TERMINAL 1 - GAZEBO
-------------------------------------------------------------------------------
cd /home/hung/ninorobot2
source /opt/ros/humble/setup.bash
colcon build --packages-select linorobot2_gazebo --symlink-install
source install/setup.bash



## chạy này để test vùng : 
# terminal 1
cd /home/hung/ninorobot2
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch linorobot2_gazebo gazebo.launch.py \
  run_ekf:=false publish_odom_tf:=true


  cd /home/hung/ninorobot2
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch linorobot2_gazebo gazebo.launch.py \
  run_ekf:=false \
  spawn_x:=-3.0 \
  spawn_y:=-1.0 \
  spawn_z:=0.35 \
  spawn_yaw:=0.0 \
  run_ekf:=false publish_odom_tf:=true

# terminal 2:
source /opt/ros/humble/setup.bash
source /home/hung/ninorobot2/install/setup.bash

ros2 run tf2_ros static_transform_publisher \
  --x 0 --y 0 --z 0 --roll 0 --pitch 0 --yaw 0 \
  --frame-id world --child-frame-id odom

# terminal 3:
cd /home/hung/ninorobot2
source /opt/ros/humble/setup.bash
source install/setup.bash

source install/setup.bash
ros2 launch social_navigation social_bringup.launch.py yolo_model_path:=/home/hung/my_amr_thesis_test-main/src/my_amr_perception/weights/yolo26n-pose.pt

# terminal 4:
cd /home/hung/ninorobot2
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch social_navigation social_sim.launch.py scenario:=crossing

source install/setup.bash
ros2 launch social_navigation social_sim.launch.py scenario:=talking


source install/setup.bash
ros2 launch social_navigation social_sim.launch.py scenario:=standing

# talking 2 người

source install/setup.bash
ros2 launch social_navigation social_sim.launch.py scenario:=stationary

ros2 launch social_perception social_constraint_grounding.launch.py \
  expected_frame:=map
  use_sim_time:=true social_state:=stationary

# terminal 5:

source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch social_perception social_constraint_grounding.launch.py


cd /home/hung/ninorobot2
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch social_perception social_constraint_grounding.launch.py \
  use_sim_time:=true \
  social_state:=stationary \
  publish_cost_grid:=false

# terminal 6:
rviz2






-------------------------------------------------------------------------------
 TERMINAL 2 - YOLO POSE + DEPTH + KEYPOINT + ORIENTATION + RVIZ
-------------------------------------------------------------------------------
# chạy này
cd /home/hung/ninorobot2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch social_navigation social_bringup.launch.py sim:=true rviz:=true

# chạy 5 cái này: 
source /opt/ros/humble/setup.bash
source /home/hung/ninorobot2/install/setup.bash

ros2 run tf2_ros static_transform_publisher \
  --x 0 --y 0 --z 0 --roll 0 --pitch 0 --yaw 0 \
  --frame-id world --child-frame-id odom


cd /home/hung/ninorobot2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch social_navigation social_sim.launch.py scenario:=crossing


ros2 run social_perception social_vlm_interaction.py --ros-args \
  --params-file /home/hung/ninorobot2/social_perception/config/social_vlm_interaction.yaml \
  -p enable_vlm:=true \
  -p vlm_adapter_path:=/home/hung/ninorobot2/Saved_Model

# Terminal 4
source install/setup.bash
ros2 launch linorobot2_navigation navigation.launch.py sim:=true \
  map:=/home/hung/ninorobot2/linorobot2_navigation/maps/cafe_vlm.yaml \
  rviz:=true


-------------------------------------------------------------------------------
 TERMINAL 3 - pose + rviz
-------------------------------------------------------------------------------

cd /home/hung/ninorobot2
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch social_navigation social_bringup.launch.py sim:=true rviz:=true \
  yolo_model_path:=/home/hung/my_amr_thesis_test-main/src/my_amr_perception/weights/yolo26n-pose.pt

----------- 
# này là để chạy cái tét khối D
source install/setup.bash
ros2 launch social_navigation social_bringup.launch.py yolo_model_path:=/home/hung/my_amr_thesis_test-main/src/my_amr_perception/weights/yolo26n-pose.pt

# Muốn test người đi vào/nói chuyện/đi ra thì thay talking bằng gathering.
# Muốn chỉ một người đi ngang trước camera thì thay talking bằng crossing.

cd /home/hung/ninorobot2
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch social_navigation social_sim.launch.py scenario:=crossing 


#### này là khối D nó tạo ra được nhé

source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch social_perception social_constraint_grounding.launch.py


#### chạy lúc mà model_saved

ros2 run social_perception social_vlm_interaction.py --ros-args \
  --params-file /home/hung/ninorobot2/social_perception/config/social_vlm_interaction.yaml \
  -p enable_vlm:=true \
  -p vlm_adapter_path:=/home/hung/ninorobot2/Saved_Model




########

source /opt/ros/humble/setup.bash
source /home/hung/ninorobot2/install/setup.bash

ros2 run tf2_ros tf2_echo map camera_depth_link
ros2 topic echo /people --once
ros2 topic echo /people/orientation --once

ros2 topic echo /people/tracked_state_json std_msgs/msg/String --field data --once | sed -n '1p' | python3 -m json.tool

-------------------------------------------------------------------------------
 TERMINAL 4 - NAV2 + SOCIAL COSTMAP (KHÔNG MỞ RVIZ THỨ HAI)
-------------------------------------------------------------------------------

cd ~/ninorobot2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch linorobot2_navigation navigation.launch.py sim:=true

-------------------------------------------------------------------------------
 TERMINAL 5 - KIỂM TRA KEYPOINT VÀ BODY YAW
-------------------------------------------------------------------------------

cd ~/ninorobot2
source /opt/ros/humble/setup.bash
source install/setup.bash

# Kiểm tra node nhận camera và YOLO đã sẵn sàng.
ros2 topic echo /social_perception/ready --once

# Mỗi người: id, valid, body_yaw_rad. valid=true thì RViz phải có mũi tên đỏ.
ros2 topic echo /people/orientation --once

# Có bbox, keypoints_2d, keypoints_3d và body_yaw_rad để so với ảnh annotated.
ros2 topic echo /people_observations --once
