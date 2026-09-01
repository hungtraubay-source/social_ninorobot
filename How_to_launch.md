#### Bring up Lidar (Pi 4)

ros2 run sllidar_ros2 sllidar_node --ros-args -p serial_port:=/dev/rplidar -p serial_baudrate:=115200 -p scan_mode:=Standard

ros2 launch linorobot2_bringup sensors.launch.py

ros2 service call /stop_motor std_srvs/srv/Empty {}

ros2 service call /start_motor std_srvs/srv/Empty {}

## URDF

### 1. Define robot properties
Build the robot computer's workspace to load the new URDF:

    cd <robot_computer_ws>
    colcon build

The same changes must be made on the host machine's <robot_type>.properties.urdf.xacro if you're simulating the robot in Gazebo.

    cd <host_machine_ws>
    colcon build

### 2. Visualize the newly created URDF
#### Visualize the robot from the host machine:

  ros2 launch linorobot2_description description.launch.py rviz:=true

## Quickstart

### 1. Booting up the robot

#### 1.1a Using a real robot:

    export LINOROBOT2_LASER_SENSOR=""
    
    ros2 launch linorobot2_bringup bringup.launch.py base_serial_port:=/dev/ttyUSB0 lidar_serial_port:=/dev/ttyUSB1 micro_ros_baudrate:=921600

#### 1.1b Using Gazebo:
    
    # Mặc định chạy worlds/lirs_test.world (map trống, chưa có người).
    # Muốn dùng map cũ thì vẫn truyền world:=... như trước.
    # Robot mặc định xuất hiện tại x=-3.0, y=-2.0, z=0.35.
    # Khi chỉ thu dữ liệu RGB-D, chạy lệnh này để không bật EKF:
    ros2 launch linorobot2_gazebo gazebo.launch.py run_ekf:=false

    # Terminal thứ hai: CHỈ perception — nạp YOLO/depth/VLM và xử lý ảnh.
    # Xuất /people (mọi người nhìn thấy) và /people_groups (vùng hội thoại
    # do VLM xác nhận). Cả hai đều được SocialLayer đưa vào costmap.
    # Không thả người; đợi log "SẴN SÀNG: camera + YOLO hoạt động, VLM đã nạp xong".
    ros2 launch social_navigation social_bringup.launch.py

    # Terminal thứ ba: thả người vào map, chạy khi bạn muốn.
    # Ctrl+C ở đây gỡ người ra mà perception vẫn chạy, nên đổi kịch bản
    # không phải nạp lại VLM.
    ros2 launch social_navigation social_sim.launch.py scenario:=talking

    # Kịch bản động: hai người đi tới, đứng nói chuyện, rồi tản ra và lặp lại.
    # Dùng để kiểm tra costmap vừa TẠO được vùng xã hội vừa XOÁ được nó.
    ros2 launch social_navigation social_sim.launch.py scenario:=gathering

Chu kỳ mặc định của `scenario:=gathering` là 96 giây: đi tới 8s, nói chuyện 60s,
tản ra 8s, khuất camera 20s. Sửa trong `lirs_test.world` nếu VLM của bạn cần
lâu hơn để trả lời:

    <plugin name="animated_people_factory" filename="libanimated_people_release.so">
      <approach_duration>8</approach_duration>
      <talk_duration>60</talk_duration>
      <disperse_duration>8</disperse_duration>
      <away_duration>20</away_duration>
    </plugin>

#### 1.1c Chạy thật: camera + VLM trên máy trạm, robot chỉ điều hướng

Camera thật có thể cắm ở máy trạm và đứng cố định quan sát hiện trường. Khác
với mô phỏng (RGB-D gắn trên robot), khi đó robot không chạy driver camera này.

    # Máy trạm, terminal 1: driver camera RGB-D của bạn, ví dụ RealSense
    ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

    # Máy trạm, terminal 2: perception + TF vị trí camera.
    # 6 số dưới đây là vị trí/hướng thật của camera đo trong frame map.
    ros2 launch social_navigation social_bringup.launch.py sim:=false \
        camera_x:=-3.0 camera_y:=0.0 camera_z:=2.0 \
        camera_roll:=0.0 camera_pitch:=0.35 camera_yaw:=0.0

`sim:=false` tự động: chọn `social_vlm_perception_real.yaml`, bỏ qua actor,
và phát TF `map -> camera_link`. Đổi tên frame bằng `camera_frame:=...` nếu
driver của bạn đặt tên khác. Topic đầu ra vẫn là `/people` và `/people_groups`
y như mô phỏng, nên `social_navigation` không phải sửa gì.

Trên robot chỉ cần bringup phần cứng và Nav2:

    ros2 launch linorobot2_bringup bringup.launch.py
    ros2 launch linorobot2_navigation navigation.launch.py map:=<map.yaml>

    # Terminal thứ tư: Nav2 + social costmap layer.
    # KHÔNG cần rviz:=true, social_bringup đã mở RViz rồi.
    ros2 launch linorobot2_navigation navigation.launch.py sim:=true \
        map:=<đường dẫn tới map.yaml>

Muốn xem trực quan thì thêm `rviz:=true` vào `social_bringup` (mặc định tắt):

    ros2 launch social_navigation social_bringup.launch.py sim:=true rviz:=true

Đổi giao diện bằng `rviz_config:=<đường dẫn .rviz>`. Chỉ nên bật RViz ở **một**
chỗ — hoặc `social_bringup`, hoặc `navigation.launch.py`, không cả hai.

Các display có sẵn và ý nghĩa:

| Display | Topic | Thấy gì |
|---|---|---|
| Social O-P-R Regions | `/social_spaces` | Đĩa vàng mờ + 3 vòng O/P/R + nhãn chữ nổi |
| Tracked People | `/social_perception/person_markers` | Trụ người, keypoint 3D, skeleton và mũi tên hướng |
| Global Costmap | `/global_costmap/costmap` | Chi phí thật planner dùng (Color Scheme = costmap) |
| YOLO Annotated | `/social_perception/annotated_image` | Ảnh camera kèm bbox |
| Depth View | `/social_perception/depth_visualization` | Ảnh độ sâu (mặc định tắt) |

Fixed Frame là `map` cho cả hai môi trường. Trong Gazebo node nhận thức phát ở
frame `world`, và `gazebo.launch.py` đã phát TF tĩnh `world -> map` nên hiển thị
đúng; chạy thật thì nó phát thẳng ở `map`.

Bộ lọc vận tốc `social_velocity_filter` do `gazebo.launch.py` tự khởi động
(`social_safety:=false` để tắt), nên không cần terminal riêng.

### Kiểm tra vùng xã hội có vào costmap chưa

    ros2 param list /global_costmap/global_costmap | grep social   # plugin đã nạp
    ros2 topic echo /people_groups --once                          # VLM thấy hội thoại
    ros2 topic hz /people                                          # người được bám vết

Trong RViz: `Social O-P-R Spaces` vẽ 3 vòng tròn, `Tracked People` vẽ từng người,
`Global Costmap` (Color Scheme = costmap) hiện vùng chi phí thật mà planner dùng.

### 
    ros2 launch linorobot2_gazebo gazebo.launch.py world:=worlds/empty.world

        ros2 launch linorobot2_navigation navigation.launch.py sim:=true rviz:=true map:=/home/hung/ninorobot2/linorobot2_navigation/maps/hungmap.yaml
#####




    ros2 launch linorobot2_gazebo gazebo.launch.py world:=worlds/empty.world
rviz:=true
    ros2 launch linorobot2_gazebo gazebo.launch.py paused:=true rviz:=true world:=worlds/empty.world spawn_x:=1.1 spawn_y:=0.8 spawn_yaw:=0.0

    ros2 service call /reset_simulation std_srvs/srv/Empty {}

    ros2 service call /reset_world std_srvs/srv/Empty {}

    ros2 service call /set_pose gazebo_msgs/srv/SetEntityState '{state: {name: "linorobot2", pose: {position: {x: 0.0, y: 0.0, z: 0.1}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}'

    ros2 service call /set_pose robot_localization/srv/SetPose "{pose: {header: {frame_id: 'map'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}}"

    ros2 run tf2_tools view_frames

linorobot2_bringup.launch.py or gazebo.launch.py must always be run on a separate terminal before creating a map or robot navigation when working on a real robot or gazebo simulation respectively.

### 2. Controlling the robot

    ros2 run teleop_twist_keyboard teleop_twist_keyboard

### 3. Creating a map

#### 3.1 Run [SLAM Toolbox](https://github.com/SteveMacenski/slam_toolbox):

    ros2 launch linorobot2_navigation slam.launch.py rviz:=true sim:=true

- **sim** - Set to true for simulated robots on the host machine. Default value is false.
- **rviz** - Set to true to visualize the robot in RVIZ. Default value is false.

#### 3.2 Move the robot to start mapping

Drive the robot manually until the robot has fully covered its area of operation. Alternatively, you can use the `2D Goal Pose` tool in RVIZ to set an autonomous goal while mapping.

#### 3.3 Save the map

    cd ~/ninorobot2/linorobot2_navigation/maps
    ros2 run nav2_map_server map_saver_cli -f hungmap --ros-args -p save_map_timeout:=10000.0

### 4. Autonomous Navigation

#### 4.1 Load the map you created:

Open linorobot2/linorobot2_navigation/launch/navigation.launch.py and change *MAP_NAME* to the name of the newly created map. Build the robot computer's workspace once done:
    
    cd ~/linorobot2_ws
    colcon build

Alternatively, `map` argument can be used when launching Nav2 (next step) to dynamically load map files. For example:

    ros2 launch linorobot2_navigation navigation.launch.py map:=linorobot2_navigation/maps/hungmap.yaml


#### 4.2 Run [Nav2](https://navigation.ros.org/tutorials/docs/navigation2_on_real_turtlebot3.html) package:

    ros2 launch linorobot2_navigation navigation.launch.py sim:=true rviz:=true map:=/home/hung/ninorobot2/linorobot2_navigation/maps/hungmap.yaml

Optional parameter for loading maps:
- **map** - Path to newly created map <map_name.yaml>.

Optional parameters for simulation on host machine:
- **sim** - Set to true for simulated robots on the host machine. Default value is false.
- **rviz** - Set to true to visualize the robot in RVIZ. Default value is false.


-------------------------------------------------------------------------------------


# 1. Xóa các thư mục build cũ để tránh rác cấu hình
cd ~/linorobot2_ws
rm -rf build/ install/ log/

# 2. Build lại toàn bộ
cd ~/linorobot2_ws
colcon build
source install/setup.bash

killall -9 gzserver gzclient
killall -9 rviz2
pkill -f rviz

ros2 daemon stop
ros2 daemon start
