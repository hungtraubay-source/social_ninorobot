## kiem tra port
    ls -la /dev/ttyUSB*
    sudo usermod -a -G dialout $USER   ::cap quyen port cho he dieu hanh ---> sau do log out ra --> log in

    groups  :: neu thay dialout thi okeee

cp ~/ninorobot2/linorobot2_gazebo/worlds/map1.world ~/ninorobot2/linorobot2_gazebo/worlds/map1.world.backup

sed -i '/<model name=/a\    <static>1</static>' ~/ninorobot2/linorobot2_gazebo/worlds/map1.world

grep -A 2 "<model name=" ~/ninorobot2/linorobot2_gazebo/worlds/map_test.world | head -20


 scp -r /home/hung/ninorobot2 pi@10.0.0.67:/home/pi/
 
 lenh chay lidar
ros2 run sllidar_ros2 sllidar_node --ros-args -p serial_port:=/dev/ttyUSB0 -p serial_baudrate:=115200 -p frame_id:=laser

####  CHAY ROBOT THAT

# chay tren PI: Khởi động hệ thống (Bringup)
ssh...
ros2 launch linorobot2_bringup bringup.launch.py base_serial_port:=/dev/ttyUSB0  : *doi port neu laf ACM

ros2 launch linorobot2_bringup bringup.launch.py base_serial_port:=/dev/ttyUSB0 micro_ros_baudrate:=921600   thay toc do: neu 115200 

# Lưu ý: Chỉ khi thấy dòng session established thì robot mới thực sự "thông" phần điều khiển.

### 
Chi tiết: Nó sẽ chạy cùng lúc: micro-ROS Agent (để nói chuyện với mạch động cơ), 
Lidar Driver (để quét laser),  
Robot State Publisher (để định nghĩa khung xương robot).
###


# Muon check xem co micro ros chua: 
    ros2 node list

# Tác dụng: Kiểm tra xem máy tính đã thực sự "nhìn thấy" mạch điều khiển chưa. Nếu thấy node /micro_ros_agent là thành công.


##### TREEN LAPTOP #######
Terminal 1: Điều khiển thủ công (Teleop)
    cd linorobot2_ws
    ros2 run teleop_twist_keyboard teleop_twist_keyboard

###
Tác dụng: Cho phép bạn dùng bàn phím để lái robot đi tới, lui, xoay.
Mục đích: Dùng để kiểm tra xem bánh xe quay đúng chiều chưa và lái robot đi quanh phòng để vẽ bản đồ ở bước sau.
###

Terminal 2: chay nay la de quet map
    ros2 launch linorobot2_navigation slam.launch.py

--- Chi tiết: Robot sẽ dùng dữ liệu từ Lidar và Odom (bánh xe) để vẽ ra sơ đồ căn phòng.


Terminal 3: xem lai map : Hiển thị hình ảnh (RViz2)

    ros2 launch linorobot2_description description.launch.py rv2:=true

# Tác dụng: Mở phần mềm RViz2 để bạn "nhìn" thấy những gì robot đang thấy.

Terminal 4: Luu lai map vao may: Lưu bản đồ vào máy

    ros2 run nav2_map_server map_saver_cli -f ~/map_nha_toi : cos the doi ten theo y minh
# Tác dụng: Đóng gói bản đồ vừa vẽ thành file .yaml và .pgm để dùng lại mãi mãi.Lưu ý: Chỉ chạy lệnh này khi bạn đã lái robot đi đủ các góc trong phòng và thấy bản đồ trên RViz đã đẹp.

Terminal 5: Navigation tu dong robot

     ros2 launch linorobot2_navigation navigation.launch.py map:=~/map_nha_toi.yaml

# Lệnh này sẽ chạy AMCL (để robot tự định vị trên map đã lưu) và các Planner (để tìm đường đi ngắn nhất mà không đâm vào tường).



# bonus them
ros2 topic list              # Liệt kê toàn bộ topic đang hoạt động
ros2 topic info /imu/data    # Xem thông tin cấu trúc của topic IMU
ros2 topic echo /imu/data    # Xem dữ liệu thực tế từ IMU nhảy trên màn hình

ros2 run rqt_tf_tree rqt_tf_tree: 
Tác dụng: Mở sơ đồ dạng cây để kiểm tra xem các bộ phận của robot (bánh xe, lidar, khung xe) đã được nối với nhau đúng chưa.


ros2 bag record -a -o my_test  # Ghi lại toàn bộ dữ liệu để phân tích sau
ros2 bag play my_test          # Phát lại dữ liệu đã ghi để debug