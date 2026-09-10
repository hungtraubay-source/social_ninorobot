===============================================================================
===============================================================================


# Này là tải lại folder lên github
git add -A
git status
git commit -m "Update social navigation and Gazebo scenarios"
git push origin main


# Các lệnh tải code mới nhất về
git checkout main
git pull origin main

    • git checkout main: chuyển về branch chính.
    • git pull origin main: tải các commit mới trên GitHub và cập nhật main ở máy mày.


# Luồng hằng ngày phải nhớ 
git checkout main
git pull origin main
git checkout -b feature/fix-txt
# code
git add /home/hung/ninorobot2/social_navigation/GIthub.md     
git commit -m "Mo ta thay doi"
git push -u origin feature/fix-txt


git add linorobot2_gazebo/worlds/lirs_test.world
git add linorobot2_navigation/config/nav_sim.yaml
git commit -m "Update Gazebo scenario and navigation config"
git push -u origin feature/fix-gazebo-config



# Kiểm tra node nhận camera và YOLO đã sẵn sàng.
ros2 topic echo /social_perception/ready --once

# Mỗi người: id, valid, body_yaw_rad. valid=true thì RViz phải có mũi tên đỏ.
ros2 topic echo /people/orientation --once

# Có bbox, keypoints_2d, keypoints_3d và body_yaw_rad để so với ảnh annotated.
ros2 topic echo /people_observations --once
