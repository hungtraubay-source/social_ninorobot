# Tổng quan package `social_navigation`

> Lưu ý kiến trúc hiện tại: phần camera, tracking và nhận diện hội thoại đã
> được tách sang package `social_perception`. Tài liệu bên dưới giữ phần mô
> phỏng cũ để tham khảo; `social_navigation` hiện chỉ tiêu thụ kết quả nhận thức
> để tạo costmap.

## 1. Mục đích

Package `social_navigation` cung cấp một pipeline điều hướng xã hội cho ROS 2 Humble, Gazebo Classic và Nav2. Package có các chức năng chính:

- Tạo các model người trong Gazebo.
- Điều khiển người đứng yên hoặc di chuyển theo waypoint.
- Lấy pose và vận tốc ground truth của người từ Gazebo.
- Phát hiện nhóm người dựa trên khoảng cách.
- Tạo các vùng không gian xã hội O–P–R cho cá nhân và nhóm.
- Hiển thị người và vùng xã hội trong RViz.
- Ghi social cost vào global/local costmap để Nav2 lập đường tránh người.

Đây là pipeline mô phỏng. Vị trí người được lấy trực tiếp từ Gazebo, chưa phải kết quả phát hiện bằng camera hoặc LiDAR.

## 2. Luồng dữ liệu tổng thể

```text
people_paths.yaml
       │
       ▼
people_motion_controller.py
       │  /set_entity_state
       ▼
     Gazebo
       │  /get_entity_state
       ▼
gazebo_people_tracker.py
       │
       │  /people (social_perception/msg/People)
       ├───────────────────────────────┐
       ▼                               ▼
people_group_detector.py          SocialLayer
       │                               │
       ├─ /people_groups ──────────────┤
       │                               ▼
       └─ /social_spaces          Nav2 costmap
              │                        │
              ▼                        ▼
             RViz              Planner/Controller
                                        │
                                        ▼
                                Robot tránh người
```

Luồng có thể chia thành ba phần:

1. **Mô phỏng:** spawn và di chuyển các model người trong Gazebo.
2. **Nhận thức xã hội:** đọc trạng thái người, phân nhóm và tạo vùng O–P–R.
3. **Điều hướng:** chuyển các vùng xã hội thành cost để Nav2 tránh.

## 3. Cấu trúc package

```text
social_navigation/
├── CMakeLists.txt
├── package.xml
├── README.md
├── social_layer_plugin.xml
├── config/
│   ├── people_paths.yaml
│   └── social_navigation.yaml
├── include/social_navigation/
│   └── social_layer.hpp
├── launch/
│   └── social_sim.launch.py
├── models/person/
│   └── model.sdf
├── msg/
│   ├── Person.msg
│   ├── People.msg
│   ├── Group.msg
│   └── Groups.msg
├── scripts/
│   ├── gazebo_people_tracker.py
│   ├── people_group_detector.py
│   └── people_motion_controller.py
└── src/
    └── social_layer.cpp
```

## 4. Khởi động hệ thống

### `launch/social_sim.launch.py`

Đây là launch file chính của package. File thực hiện hai nhiệm vụ.

### 4.1. Spawn người

Bốn model người được tạo tại các vị trí ban đầu:

| Model | X | Y | Vai trò hiện tại |
|---|---:|---:|---|
| `person_1` | -0.6 | 4.5 | Thành viên nhóm |
| `person_2` | 0.6 | 4.5 | Thành viên nhóm |
| `person_3` | -2.2 | -2.5 | Cá nhân đứng riêng |
| `person_4` | 3.0 | -2.5 | Cá nhân đứng riêng |

Mỗi model được tạo bằng executable `gazebo_ros/spawn_entity.py`, sử dụng model `models/person/model.sdf`. Tâm model được đặt tại `z = 0.9 m` vì thân người cao `1.8 m`.

### 4.2. Chạy các node

Sau khi chờ ba giây để Gazebo tạo model, launch file chạy:

- `gazebo_people_tracker.py`
- `people_group_detector.py`
- `people_motion_controller.py`

Hai launch argument:

- `spawn_people:=true`: tạo model người trong Gazebo.
- `move_people:=true`: bật điều khiển waypoint.

Ví dụ:

```bash
ros2 launch social_navigation social_sim.launch.py
```

Không spawn lại nếu các model đã có:

```bash
ros2 launch social_navigation social_sim.launch.py spawn_people:=false
```

Không điều khiển vị trí người:

```bash
ros2 launch social_navigation social_sim.launch.py move_people:=false
```

## 5. Model người trong Gazebo

### `models/person/model.sdf`

Model người gồm:

- Thân hình cylinder, bán kính `0.28 m`, chiều cao `1.8 m`.
- Đầu hình sphere, bán kính `0.20 m`.
- Thân màu xanh và đầu màu da.

Các thuộc tính chính:

```xml
<static>false</static>
<gravity>false</gravity>
<kinematic>true</kinematic>
```

Model không phải vật thể tĩnh, không chịu trọng lực và được điều khiển trực tiếp bằng pose. Collision hiện chỉ được khai báo cho phần thân cylinder; phần đầu chỉ là visual.

## 6. Điều khiển chuyển động

### `config/people_paths.yaml`

File này định nghĩa waypoint trong hệ tọa độ `world` của Gazebo. Một waypoint gồm:

- `time`: thời gian tính từ đầu chu kỳ.
- `x`, `y`: vị trí trên mặt phẳng.
- `z`: độ cao tâm model.
- `yaw`: hướng nhìn, đơn vị radian.

Hiện tại mỗi người chỉ có một waypoint nên họ đứng yên. Ví dụ một quỹ đạo có chuyển động:

```yaml
person_3:
  waypoints:
    - {time: 0.0, x: -2.2, y: -2.5, z: 0.9, yaw: 0.0}
    - {time: 5.0, x:  0.0, y: -2.5, z: 0.9, yaw: 0.0}
    - {time: 10.0, x: 2.0, y: -1.0, z: 0.9, yaw: 0.8}
```

### `scripts/people_motion_controller.py`

Node đọc `people_paths.yaml` và gọi service:

```text
/set_entity_state
gazebo_msgs/srv/SetEntityState
```

Tại mỗi chu kỳ, node:

1. Tính thời gian đã chạy.
2. Tìm hai waypoint bao quanh thời điểm hiện tại.
3. Nội suy tuyến tính vị trí `x`, `y`, `z`.
4. Nội suy `yaw` theo chiều quay ngắn nhất.
5. Tính vận tốc tuyến tính giữa hai waypoint.
6. Gửi `EntityState` tới Gazebo với `reference_frame = world`.

Các tham số chính:

- `update_rate: 20.0`: cập nhật model 20 lần/giây.
- `loop: true`: lặp lại quỹ đạo khi đến waypoint cuối.

Nếu một người chỉ có một waypoint, vận tốc của họ bằng 0. Khi node này đang chạy, kéo model bằng chuột trong Gazebo chỉ có tác dụng tạm thời vì chu kỳ tiếp theo sẽ đặt model trở lại waypoint.

## 7. Theo dõi người trong Gazebo

### `scripts/gazebo_people_tracker.py`

Node này là nguồn dữ liệu nhận thức trong mô phỏng. Nó không xử lý ảnh mà đọc trực tiếp trạng thái ground truth bằng service:

```text
/get_entity_state
gazebo_msgs/srv/GetEntityState
```

Node gửi request bất đồng bộ cho từng model trong `people_names`. Hai cấu trúc nội bộ được sử dụng:

- `states`: lưu trạng thái mới nhất của từng người.
- `pending`: lưu các request chưa có kết quả, tránh gửi request trùng.

Cấu hình hiện tại:

```yaml
gazebo_reference_frame: linorobot2
output_frame: base_link
publish_rate: 8.0
```

Gazebo trả pose người tương đối với model robot `linorobot2`; node công bố dữ liệu với `frame_id = base_link`. Cách này tránh phụ thuộc trực tiếp vào phép hiệu chỉnh giữa `world` của Gazebo và `map` của ROS.

Topic đầu ra:

```text
/people
social_perception/msg/People
```

Header của message được để timestamp bằng 0, có nghĩa là các thành phần dùng TF sẽ lấy transform mới nhất. Điều này giúp tránh hiện tượng marker chớp tắt do Gazebo state và odometry TF lệch nhau một vài mili giây.

## 8. Custom messages

### `msg/Person.msg`

```text
string id
geometry_msgs/Pose pose
geometry_msgs/Twist velocity
```

Biểu diễn một người gồm ID, pose và vận tốc.

### `msg/People.msg`

```text
std_msgs/Header header
social_perception/Person[] people
```

Chứa danh sách người. `header.frame_id` cho biết pose của các phần tử đang thuộc hệ tọa độ nào.

### `msg/Group.msg`

```text
string id
string[] member_ids
geometry_msgs/Point center
float32 o_radius
float32 p_radius
float32 r_radius
```

Biểu diễn một nhóm, danh sách thành viên, tâm nhóm và bán kính ba vùng xã hội.

### `msg/Groups.msg`

```text
std_msgs/Header header
social_perception/Group[] groups
```

Chứa danh sách các nhóm được phát hiện.

## 9. Phát hiện nhóm và hiển thị O–P–R

### `scripts/people_group_detector.py`

Node subscribe:

```text
/people
```

và publish:

```text
/people_groups   social_perception/msg/Groups
/social_spaces   visualization_msgs/msg/MarkerArray
```

### 9.1. Phân nhóm

Hai người được nối với nhau nếu khoảng cách Euclidean nhỏ hơn hoặc bằng `group_distance`:

```text
sqrt((x1 - x2)^2 + (y1 - y2)^2) <= group_distance
```

Cấu hình hiện tại:

```yaml
group_distance: 1.5
minimum_group_size: 2
```

`person_1` và `person_2` cách nhau `1.2 m`, vì vậy được xem là một nhóm. `person_3` và `person_4` cách nhau `5.2 m`, vì vậy đứng riêng.

Thuật toán dùng connected components và có tính bắc cầu. Nếu A gần B, B gần C nhưng A không gần C thì cả ba vẫn có thể thuộc cùng một nhóm.

### 9.2. Tâm và bán kính nhóm

Tâm nhóm là trung bình tọa độ thành viên:

```text
center_x = tổng x / số thành viên
center_y = tổng y / số thành viên
```

`member_radius` là khoảng cách lớn nhất từ tâm tới một thành viên. Bán kính nhóm được tính:

```text
O radius = max(o_space_min_radius, member_radius * 0.5)
P radius = max(O radius + 0.15, member_radius + p_space_margin)
R radius = P radius + r_space_margin
```

Với `person_1 = (-0.6, 4.5)` và `person_2 = (0.6, 4.5)`:

- Tâm nhóm là `(0, 4.5)`.
- `member_radius = 0.6 m`.
- O-space có bán kính `0.45 m`.
- P-space có bán kính `1.05 m`.
- R-space có bán kính `2.05 m`.

### 9.3. Ý nghĩa O–P–R

- **O-space:** vùng hoạt động/giao tiếp trung tâm; robot không nên đi xuyên qua.
- **P-space:** vùng trực tiếp chứa người và hoạt động giao tiếp.
- **R-space:** vùng ngoài cùng; robot được khuyến khích tránh nếu có đường tốt hơn.

### 9.4. Không gian cá nhân

Người đứng riêng có các vùng bất đối xứng theo hướng nhìn:

| Vùng | Phía trước | Hai bên | Phía sau |
|---|---:|---:|---:|
| O | 0.38 m | 0.38 m | 0.38 m |
| P | 0.90 m | 0.65 m | 0.50 m |
| R | 1.50 m | 1.00 m | 0.75 m |

Phía trước rộng hơn phía sau, do đó robot được khuyến khích đi phía sau người thay vì cắt ngang phía trước mặt.

### 9.5. Marker RViz

Topic `/social_spaces` chứa:

- Cylinder màu xanh biểu diễn thân người.
- Đường O-space màu đỏ.
- Đường P-space màu cam.
- Đường R-space màu vàng.

Mỗi lần cập nhật, node gửi `Marker.DELETEALL` trước marker mới để RViz không giữ lại dữ liệu cũ. Người thuộc nhóm chỉ được vẽ vùng O–P–R chung của nhóm; người đứng riêng được vẽ vùng cá nhân.

## 10. SocialLayer cho Nav2

### `include/social_navigation/social_layer.hpp`

Header khai báo class:

```cpp
SocialLayer : public nav2_costmap_2d::CostmapLayer
```

Các hàm chính:

- `onInitialize()`: đọc tham số và tạo subscriber.
- `updateBounds()`: xác định vùng costmap cần cập nhật.
- `updateCosts()`: tính social cost cho từng cell.
- `reset()`: xóa dữ liệu người và nhóm.
- `isClearable()`: cho phép Nav2 clear layer.

Mutex bảo vệ dữ liệu vì callback subscriber và vòng cập nhật costmap có thể chạy đồng thời.

### `src/social_layer.cpp`

Plugin subscribe trực tiếp hai topic:

```text
/people
/people_groups
```

`/social_spaces` chỉ dành cho RViz và không tham gia tính cost.

### 10.1. Chuyển hệ tọa độ

Global costmap sử dụng frame `map`, local costmap sử dụng frame `odom`, trong khi `/people` hiện ở `base_link`. Plugin dùng TF để chuyển:

```text
base_link -> map     cho global costmap
base_link -> odom    cho local costmap
```

Nhờ đó cùng một dữ liệu `/people` có thể dùng cho cả hai costmap.

### 10.2. `updateBounds()`

Hàm xác định hình chữ nhật bao quanh:

- Vị trí hiện tại của từng người.
- Vị trí dự đoán trong tương lai.
- Toàn bộ R-space của nhóm.
- Bounds của chu kỳ trước.

Bounds cũ được đưa vào để Nav2 cập nhật và xóa social cost còn sót lại khi người di chuyển sang vị trí mới.

### 10.3. Dự đoán chuyển động

Vị trí tương lai được tính:

```text
future_x = current_x + speed * prediction_time * cos(yaw)
future_y = current_y + speed * prediction_time * sin(yaw)
```

Với cấu hình:

```yaml
prediction_time: 1.5
prediction_steps: 2
```

Plugin tính không gian xã hội tại ba vị trí: hiện tại, sau `0.75 giây` và sau `1.5 giây`. Kết quả tạo thành một hành lang social cost theo hướng chuyển động dự đoán.

### 10.4. Cost cá nhân

Với mỗi cell, vector từ người tới cell được đổi sang trục theo hướng của người:

- `forward`: khoảng cách trước/sau.
- `side`: khoảng cách trái/phải.

O-space là hình tròn. P-space và R-space là ellipse bất đối xứng:

```text
forward^2 / longitudinal_radius^2
  + side^2 / side_radius^2 <= 1
```

Bán kính dọc được chọn khác nhau tùy cell nằm phía trước hay phía sau người.

### 10.5. Cost nhóm

Nhóm dùng ba hình tròn đồng tâm tại `group.center`. Cell được gán cost theo khoảng cách tới tâm và các bán kính `o_radius`, `p_radius`, `r_radius`.

Giá trị cost hiện tại:

| Vùng | Cost |
|---|---:|
| O-space | 254 |
| P-space | 230 |
| R-space | 120 |
| Ngoài vùng | 0 |

O-space gần tương đương lethal obstacle. P-space có cost rất cao. R-space có cost trung bình để planner ưu tiên đường khác nhưng vẫn có thể đi qua khi cần thiết.

### 10.6. Ghép với master costmap

Plugin dùng `updateWithMax()`, vì vậy cost cuối cùng là giá trị lớn nhất giữa social layer và các layer khác. Social layer không thể xóa một vật cản thật đã được static, obstacle hoặc voxel layer đánh dấu.

Trong RViz, người thuộc nhóm chỉ được vẽ vùng chung. Trong costmap, plugin vẫn tính cả vùng cá nhân của từng thành viên và vùng chung của nhóm để bảo vệ từng người.

## 11. Đăng ký plugin

### `social_layer_plugin.xml`

File khai báo với Pluginlib:

```text
Tên plugin: social_navigation::SocialLayer
Class C++:  social_navigation::SocialLayer
Base class: nav2_costmap_2d::Layer
Library:    social_layer
```

Macro `PLUGINLIB_EXPORT_CLASS` ở cuối `social_layer.cpp` đăng ký class khi chạy. Nếu thiếu XML hoặc macro này, Nav2 không thể nạp plugin.

## 12. Cấu hình tham số

### `config/social_navigation.yaml`

File cung cấp tham số cho ba Python node.

#### `gazebo_people_tracker`

- `people_names`: danh sách model cần theo dõi.
- `gazebo_reference_frame`: frame dùng khi hỏi Gazebo.
- `output_frame`: frame ghi trong `/people`.
- `publish_rate`: tần số đọc và publish trạng thái.

#### `people_motion_controller`

- `update_rate`: tần số đặt trạng thái Gazebo.
- `loop`: có lặp quỹ đạo hay không.

#### `people_group_detector`

- Ngưỡng khoảng cách và số người tối thiểu để tạo nhóm.
- Margin dùng để tính O–P–R của nhóm.
- Kích thước vùng O–P–R của người đứng riêng.

Kích thước marker trong `social_navigation.yaml` và kích thước cost trong `nav_sim.yaml` được cấu hình riêng. Khi thay đổi bán kính, nên chỉnh đồng bộ hai file để hình RViz khớp với costmap thực tế.

## 13. Tích hợp với Nav2

Social layer được thêm vào cả global và local costmap trong:

```text
linorobot2_navigation/config/nav_sim.yaml
```

Global costmap:

```yaml
plugins: ["static_layer", "obstacle_layer", "voxel_layer",
          "social_layer", "inflation_layer"]
```

Local costmap:

```yaml
plugins: ["obstacle_layer", "voxel_layer",
          "social_layer", "inflation_layer"]
```

Global planner sử dụng social cost để chọn đường tổng thể. Local controller sử dụng cost để điều chỉnh chuyển động ngắn hạn khi đến gần người.

## 14. Topic và service

### Topic

| Topic | Kiểu message | Publisher | Subscriber | Mục đích |
|---|---|---|---|---|
| `/people` | `social_perception/msg/People` | `social_vlm_perception` | SocialLayer | Pose và vận tốc người |
| `/people_groups` | `social_perception/msg/Groups` | `social_vlm_perception` | SocialLayer | Tâm và vùng O–P–R được VLM xác nhận |
| `/social_spaces` | `visualization_msgs/msg/MarkerArray` | `social_vlm_perception` | RViz | Hiển thị vùng xã hội |

### Service Gazebo

| Service | Kiểu | Client | Mục đích |
|---|---|---|---|
| `/get_entity_state` | `gazebo_msgs/srv/GetEntityState` | People tracker | Đọc pose/vận tốc người |
| `/set_entity_state` | `gazebo_msgs/srv/SetEntityState` | Motion controller | Đặt pose/vận tốc người |

Hai service này yêu cầu Gazebo được chạy cùng plugin state tương ứng, thường là `libgazebo_ros_state.so`.

## 15. Tần số hoạt động

| Thành phần | Tần số hiện tại |
|---|---:|
| Điều khiển model người | 20 Hz |
| Đọc Gazebo và publish `/people` | 8 Hz |
| Phát hiện nhóm | Theo `/people`, khoảng 8 Hz |
| Global costmap update | 3 Hz |
| Global costmap publish | 2 Hz |
| Local costmap update | 6 Hz |
| Local costmap publish | 3 Hz |

Gazebo được cập nhật nhanh để chuyển động mượt, trong khi dữ liệu xã hội và costmap được cập nhật theo tốc độ cần thiết cho Nav2.

## 16. Build và dependency

### `CMakeLists.txt`

File build:

- Sinh interface C++/Python từ bốn file `.msg`.
- Biên dịch `social_layer.cpp` thành shared library.
- Liên kết với ROS 2, Nav2, TF2 và Pluginlib.
- Cài đặt header, plugin XML, launch, config và model.
- Cài ba Python script thành ROS 2 executable.

### `package.xml`

Các dependency chính:

- `rclpy`: các node Python.
- `rclcpp`: plugin C++.
- `gazebo_msgs`: đọc/ghi trạng thái Gazebo.
- `geometry_msgs`: pose, twist và point.
- `visualization_msgs`: marker RViz.
- `nav2_costmap_2d`: costmap plugin.
- `tf2`, `tf2_geometry_msgs`: chuyển hệ tọa độ.
- `rosidl_default_generators/runtime`: custom messages.
- `python3-yaml`: đọc waypoint YAML.

Build package:

```bash
cd ~/ninorobot2
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select social_navigation
source install/setup.bash
```

## 17. Kiểm tra khi chạy

Kiểm tra các node:

```bash
ros2 node list
```

Kiểm tra trạng thái người:

```bash
ros2 topic echo /people
```

Kiểm tra nhóm:

```bash
ros2 topic echo /people_groups
```

Kiểm tra tần số:

```bash
ros2 topic hz /people
ros2 topic hz /people_groups
```

Trong RViz, thêm display `MarkerArray` và chọn topic `/social_spaces`. Social cost thực tế nằm trong `/global_costmap/costmap` và `/local_costmap/costmap`.

## 18. Hạn chế hiện tại

1. Tracker dùng ground truth Gazebo, chưa có detector camera/LiDAR.
2. Group detector chỉ dùng khoảng cách, chưa xét hướng nhìn hoặc F-formation thực sự.
3. ID nhóm được tạo lại theo thứ tự mỗi callback, chưa ổn định theo thời gian.
4. Tâm nhóm là trung bình vị trí, chưa phải tâm tương tác được ước lượng từ hướng nhìn.
5. Dự đoán dùng độ lớn vận tốc nhưng dùng `yaw` làm hướng; người đi ngang hoặc đi lùi có thể bị dự đoán sai.
6. Timestamp `/people` bằng 0 giúp TF ổn định nhưng khiến `data_timeout` trong SocialLayer không loại dữ liệu cũ, vì timeout chỉ kiểm tra timestamp khác 0.
7. Bán kính hiển thị và bán kính costmap nằm ở hai file riêng nên có thể không đồng bộ.
8. Model cylinder/sphere phù hợp thử nghiệm navigation nhưng không phù hợp để đánh giá model nhận diện người bằng hình ảnh.

## 19. Kết luận

Pipeline hiện tại hoạt động theo chuỗi:

```text
Waypoint
  -> điều khiển model Gazebo
  -> đọc pose và vận tốc
  -> publish /people
  -> phát hiện nhóm
  -> publish /people_groups và /social_spaces
  -> SocialLayer tính cost
  -> Nav2 lập đường tránh vùng xã hội
```

Package đã có đầy đủ giao diện giữa mô phỏng, nhận thức xã hội, trực quan hóa và điều hướng. Khi chuyển sang robot thật, phần cần thay thế chủ yếu là `gazebo_people_tracker.py`: một detector/tracker thực tế chỉ cần tiếp tục publish đúng interface `/people`, còn group detector và SocialLayer có thể được giữ lại hoặc nâng cấp độc lập.
