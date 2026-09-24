===============================================================================
===============================================================================
# sửa file nào thì up lên 
git checkout -b (tên của branch)

git add .
git commit -m "Update code"
git push origin "tên"



# lần thứ 2 mà sửa code thì dùng này + thêm cả file code mới nhé
git add .
git commit --amend   :wq
git push origin (tên) -f


# kéo code mới nhất về ddi kkk
git pull origin main
git branch -D "tên branch"
git branch

# check 5 cái gần nhất
git log --online -n5

# dành cho người dùng chung
folk về của mình ---> clone về máy --->

git remote -v : kiểm tra xem origin là của ai 

git remote add "ten công ty" "link SSH"

# lấy code về
git pull trungquan17 main






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


##### này chạy để  ignore file 

git checkout -b fix/ignore-local-model

# 1. Thêm dòng ignore vào file
echo "duong_dan/toi/file_can_bo" >> .gitignore

# 2. Dừng tracking file đó (nếu nó đã từng commit trước đây)
git rm -r --cached duong_dan/toi/file_can_bo

# 3. Add + commit
git add .gitignore
git commit -m "Stop tracking local file"

# 4. Push
git push -u origin fix/ignore-local-model

với điều kiện mày phải ở branch fix











# Kiểm tra node nhận camera và YOLO đã sẵn sàng.
ros2 topic echo /social_perception/ready --once

# Mỗi người: id, valid, body_yaw_rad. valid=true thì RViz phải có mũi tên đỏ.
ros2 topic echo /people/orientation --once

# Có bbox, keypoints_2d, keypoints_3d và body_yaw_rad để so với ảnh annotated.
ros2 topic echo /people_observations --once
