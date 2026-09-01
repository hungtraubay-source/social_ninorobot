# Xây dựng pipeline Social Navigation sử dụng camera trong Gazebo

## 1. Mục tiêu

Pipeline `social_navigation` hiện tại lấy trực tiếp vị trí thật của người từ Gazebo qua service `/get_entity_state`. Cách này phù hợp để kiểm tra thuật toán social costmap, nhưng chưa mô phỏng đúng hệ thống thực tế sử dụng camera và model nhận diện người.

Pipeline cần chuyển thành:

```text
Người trong Gazebo
        ↓
Camera RGB-D mô phỏng
        ↓
Model phát hiện người
        ↓
Định vị người trong không gian 3D
        ↓
Tracking: ID, pose, velocity, orientation
        ↓
Model tính vùng O-P-R
        ↓
SocialLayer
        ↓
Global/local costmap của Nav2
        ↓
Robot điều hướng tránh người
```

Mục tiêu quan trọng là giữ `SocialLayer` độc lập với nguồn phát hiện. Plugin chỉ cần nhận đúng message cuối cùng, không cần biết dữ liệu đến từ Gazebo ground-truth hay camera.

---

## 2. Pipeline hiện tại

```text
people_paths.yaml
        ↓
people_motion_controller.py
        ↓ /set_entity_state
Gazebo
        ↓ /get_entity_state
gazebo_people_tracker.py
        ↓ /people
people_group_detector.py
        ├── /people_groups
        └── /social_spaces

/people + /people_groups
        ↓
SocialLayer
        ↓
Nav2 costmap
```

Điểm cần thay đổi là `gazebo_people_tracker.py`. Node này đang đọc pose chính xác của người từ Gazebo, bỏ qua toàn bộ camera và model perception.

---

## 3. Pipeline camera đề xuất

```text
/camera/color/image_raw ─────────────┐
/camera/depth/image_rect_raw ────────┤
/camera/color/camera_info ───────────┤
TF camera → base_link → odom/map ────┘
                    ↓
          camera_people_detector.py
                    ↓
          /people/detections_2d
                    ↓
            people_3d_localizer.py
                    ↓
          /people/detections_3d
                    ↓
              people_tracker.py
                    ↓
                  /people
                    ↓
             opr_inference_node.py
                    ├── /people_groups
                    └── /social_spaces
                    ↓
               SocialLayer
                    ↓
              Nav2 costmap
```

---

## 4. Bật camera RGB-D trong mô phỏng

Repository đã có mô tả camera tại:

```text
linorobot2_description/urdf/sensors/depth_sensor.urdf.xacro
```

Camera này có thể publish:

```text
/camera/color/image_raw
/camera/color/camera_info
/camera/depth/image_rect_raw
/camera/depth/color/points
```

Trong robot 2WD, phần include và khởi tạo depth camera hiện đang bị comment trong:

```text
linorobot2_description/urdf/robots/2wd.urdf.xacro
```

Cần bật lại:

```xml
<xacro:include
  filename="$(find linorobot2_description)/urdf/sensors/depth_sensor.urdf.xacro" />
```

và khối:

```xml
<xacro:depth_sensor>
  <xacro:insert_block name="depth_sensor_pose" />
</xacro:depth_sensor>
```

Sau khi build và chạy Gazebo, kiểm tra:

```bash
ros2 topic list | grep camera
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/depth/image_rect_raw
ros2 topic echo /camera/color/camera_info --once
ros2 run tf2_ros tf2_echo base_link camera_depth_link
```

Nếu thiếu TF từ camera tới `base_link`, detection không thể chuyển sang hệ tọa độ của costmap.

---

## 5. Thay model người hiện tại

Model hiện tại nằm tại:

```text
social_navigation/models/person/model.sdf
```

Nó chỉ gồm một cylinder làm thân và một sphere làm đầu. Detector pretrained với class `person`, chẳng hạn YOLO, nhiều khả năng không nhận hình này là người.

Nên thay bằng một trong các lựa chọn sau:

1. Gazebo actor có mesh, texture và animation người.
2. Model người 3D có hình dạng và texture thực tế.
3. Giữ model hiện tại nhưng huấn luyện detector bằng ảnh synthetic đúng loại model đó.

Nếu cần đánh giá model phát hiện người, nên sử dụng nhiều actor khác nhau về ngoại hình, tư thế, ánh sáng và mức độ che khuất.

---

## 6. Node phát hiện người từ ảnh

Cần thêm node, ví dụ:

```text
social_navigation/scripts/camera_people_detector.py
```

Input:

```text
/camera/color/image_raw
```

Node chạy model của nhóm và tạo:

- Bounding box `[xmin, ymin, xmax, ymax]`.
- Class `person`.
- Confidence.
- Timestamp và frame của ảnh gốc.

Output nên dùng `vision_msgs/msg/Detection2DArray`:

```text
/people/detections_2d
```

Không nên publish thẳng `/people` từ bounding box 2D vì costmap cần vị trí theo mét trong không gian 3D.

---

## 7. Chuyển detection 2D thành vị trí 3D

Cần thêm node:

```text
social_navigation/scripts/people_3d_localizer.py
```

Input:

```text
/people/detections_2d
/camera/depth/image_rect_raw
/camera/color/camera_info
```

Với mỗi bounding box:

1. Chọn vùng depth nằm trong phần thân người.
2. Loại bỏ depth bằng 0, NaN và các outlier.
3. Lấy median depth `Z`.
4. Chọn điểm ảnh đại diện `(u, v)`.
5. Dùng camera intrinsics để back-project:

```text
X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
Z = depth
```

Sau đó dùng TF2 để chuyển tọa độ:

```text
camera_depth_optical_frame → base_link
```

hoặc trực tiếp sang `odom`/`map`.

Không nên chỉ lấy depth đúng tại tâm bounding box vì điểm đó có thể nằm giữa hai chân hoặc trên vật thể phía sau. Median depth của một vùng nhỏ thường ổn định hơn.

Output đề xuất:

```text
/people/detections_3d
```

Nếu model của nhóm đã nhận RGB-D và xuất trực tiếp vị trí 3D thì detector và localizer có thể gộp thành một node.

---

## 8. Tracking người

`social_perception/msg/Person.msg` hiện cung cấp:

```text
string id
geometry_msgs/Pose pose
geometry_msgs/Twist velocity
```

Detector theo từng frame thường không có ID ổn định hoặc velocity. Vì vậy cần node:

```text
social_perception/scripts/social_vlm_perception.py
```

Node thực hiện:

- Ghép detection giữa các frame.
- Duy trì ID ổn định.
- Lọc nhiễu vị trí.
- Ước lượng vận tốc.
- Dự đoán ngắn hạn khi mất detection.

Có thể sử dụng Kalman filter kết hợp Hungarian matching, ByteTrack hoặc DeepSORT tùy model của nhóm.

Output:

```text
/people
```

với kiểu:

```text
social_perception/msg/People
```

Node này sẽ thay vai trò đầu vào điều hướng của `gazebo_people_tracker.py`.

---

## 9. Ước lượng hướng của người

`SocialLayer` cần orientation để phân biệt vùng phía trước, hai bên và phía sau người. Bounding box và depth chỉ cung cấp vị trí, không cung cấp hướng nhìn.

Có thể lấy orientation từ:

- Model body orientation.
- Pose estimation/keypoint vai, hông và đầu.
- Hướng vận tốc khi người đang di chuyển.
- Model OPR của nhóm nếu model đã dự đoán orientation.

Chiến lược fallback:

```text
Nếu orientation model đủ tin cậy:
    yaw = kết quả model
Ngược lại nếu tốc độ đủ lớn:
    yaw = atan2(vy, vx)
Ngược lại:
    giữ yaw hợp lệ gần nhất của track
```

Nếu yaw sai, ellipse O-P-R sẽ quay sai và robot có thể tránh phía sau nhưng lại đi cắt ngang trước mặt người.

---

## 10. Tích hợp model O-P-R của nhóm

### Phương án A: thay đổi ít nhất

Model nhận `/people` và xuất `/people_groups` theo message hiện tại:

```text
social_perception/msg/Groups
```

Mỗi group chứa:

```text
id
member_ids
center
o_radius
p_radius
r_radius
```

Khi đó thay `people_group_detector.py` bằng:

```text
opr_inference_node.py
```

`SocialLayer` có thể giữ nguyên nếu model chỉ tính O-P-R chung cho nhóm.

### Phương án B: model tính O-P-R riêng cho từng người

Hiện tại kích thước O-P-R cá nhân bị đặt cố định trong `SocialLayer`. Nếu model trả vùng khác nhau cho từng người, cần mở rộng `Person.msg`:

```text
string id
geometry_msgs/Pose pose
geometry_msgs/Twist velocity

float32 o_radius
float32 p_front_radius
float32 p_side_radius
float32 p_rear_radius
float32 r_front_radius
float32 r_side_radius
float32 r_rear_radius
float32 confidence
```

Sau đó sửa `SocialLayer` để đọc bán kính của từng `Person`, thay vì chỉ dùng các tham số cố định trong YAML.

Các tham số YAML nên được giữ làm giá trị mặc định khi output model không hợp lệ hoặc confidence thấp.

### Phương án C: model trả heatmap hoặc hình dạng phức tạp

Nếu model trả polygon, ellipse, Gaussian hoặc heatmap thay vì ba bán kính đơn giản, nên tạo interface tổng quát:

```text
SocialZone.msg
SocialZones.msg
```

Ví dụ:

```text
string id
uint8 shape_type
geometry_msgs/Pose pose
float32 front_radius
float32 side_radius
float32 rear_radius
uint8 cost
float32 confidence
```

Model publish `/social_zones`; `SocialLayer` chỉ chịu trách nhiệm rasterize các zone vào costmap.

---

## 11. Giữ ground-truth để đánh giá, không dùng để điều hướng

Không nên xóa `gazebo_people_tracker.py`. Nên đổi topic output của nó thành:

```text
/ground_truth/people
```

Quy ước:

```text
/people               = kết quả camera + model
/ground_truth/people  = vị trí thật từ Gazebo
```

Nhờ đó có thể đo:

- Precision và recall.
- Sai số vị trí 3D.
- Sai số vận tốc và orientation.
- ID switch của tracker.
- Sai số vùng O-P-R.
- Khoảng cách nhỏ nhất giữa robot và người.
- Tỷ lệ hoàn thành navigation.

Phải bảo đảm `SocialLayer` chỉ subscribe `/people`, không subscribe `/ground_truth/people` trong bài đánh giá perception.

---

## 12. Đồng bộ thời gian và TF

Tất cả node mô phỏng phải dùng:

```yaml
use_sim_time: true
```

Output perception phải giữ timestamp của ảnh gốc:

```python
output.header.stamp = image.header.stamp
output.header.frame_id = image.header.frame_id
```

Ví dụ ảnh được chụp tại `t0`, inference mất 80 ms thì detection được publish tại `t0 + 80 ms`, nhưng `header.stamp` vẫn phải bằng `t0`.

Nếu gắn timestamp lúc inference hoàn tất, TF sẽ coi một quan sát cũ như quan sát mới, gây sai vị trí khi robot hoặc người đang di chuyển.

`SocialLayer` hiện có `data_timeout=1.0`. Tổng độ trễ perception không nên vượt quá giới hạn này.

---

## 13. Xử lý khi camera mất dấu người

Người có thể ra khỏi camera, bị che khuất hoặc detector bỏ sót tạm thời. Không nên xóa track ngay lập tức vì cost xã hội sẽ biến mất đột ngột.

Tracker nên có các trạng thái:

```text
Có detection mới       → TRACKED
Mất detection ngắn     → PREDICTED
Mất detection quá lâu  → REMOVED
```

Ví dụ:

```text
0–0.5 giây: tiếp tục dự đoán bằng Kalman
0.5–1.0 giây: tăng uncertainty hoặc mở rộng vùng an toàn
trên 1.0 giây: xóa track
```

---

## 14. Quan hệ giữa VoxelLayer và SocialLayer

`nav_sim.yaml` đã cấu hình point cloud camera cho `VoxelLayer`:

```text
/camera/depth/color/points
```

Do đó người có thể xuất hiện đồng thời ở hai layer:

```text
VoxelLayer  → coi thân người là vật cản hình học
SocialLayer → tạo khoảng cách xã hội lớn hơn quanh người
```

Hai layer được ghép bằng cost lớn nhất. Đây là hành vi mong muốn:

- VoxelLayer ngăn va chạm vật lý.
- SocialLayer khiến planner chủ động đi cách xa người.

---

## 15. Các file cần giữ, sửa và bổ sung

### Giữ nguyên hoặc gần như giữ nguyên

```text
social_navigation/social_layer_plugin.xml
social_navigation/include/social_navigation/social_layer.hpp
social_navigation/src/social_layer.cpp
social_navigation/scripts/people_motion_controller.py
linorobot2_navigation/config/nav_sim.yaml
```

`SocialLayer` chỉ cần sửa đáng kể nếu model trả O-P-R riêng cho từng người hoặc trả hình dạng mới.

### Cần sửa

```text
linorobot2_description/urdf/robots/2wd.urdf.xacro
social_navigation/models/person/model.sdf
social_navigation/launch/social_sim.launch.py
social_navigation/config/social_navigation.yaml
social_navigation/CMakeLists.txt
social_navigation/package.xml
```

### Cần thay vai trò

```text
gazebo_people_tracker.py
```

Node này chỉ nên publish `/ground_truth/people` để đánh giá.

```text
people_group_detector.py
```

Node này có thể được thay bằng model O-P-R của nhóm.

### Nên bổ sung

```text
social_navigation/scripts/camera_people_detector.py
social_navigation/scripts/people_3d_localizer.py
social_navigation/scripts/people_tracker.py
social_navigation/scripts/opr_inference_node.py
social_navigation/config/perception.yaml
social_navigation/config/opr_model.yaml
social_navigation/launch/social_perception_sim.launch.py
```

---

## 16. Launch tổng thể đề xuất

```text
Gazebo
  ├── spawn robot có RGB-D camera
  ├── spawn actor người
  └── people_motion_controller

Perception
  ├── camera_people_detector
  ├── people_3d_localizer
  ├── people_tracker
  └── opr_inference_node

Evaluation
  └── gazebo_people_tracker → /ground_truth/people

Navigation
  └── Nav2 + SocialLayer
```

Launch mới không được để ground-truth tracker và camera tracker cùng publish `/people`, vì dữ liệu sẽ bị trộn lẫn.

---

## 17. Lộ trình triển khai

Nên kiểm tra từng tầng theo thứ tự:

1. Bật RGB-D camera và xem được RGB/depth trong RViz.
2. Kiểm tra đầy đủ TF camera → `base_link` → `odom` → `map`.
3. Thay cylinder bằng actor mà model nhận diện được.
4. Chạy detector và kiểm tra bounding box trên ảnh.
5. Chuyển bounding box sang vị trí 3D.
6. So sánh vị trí camera với Gazebo ground-truth.
7. Thêm tracking để có ID và velocity.
8. Thêm orientation estimation.
9. Publish `/people` từ pipeline camera.
10. Cho model O-P-R subscribe `/people`.
11. Publish `/people_groups` hoặc `/social_zones`.
12. Kiểm tra marker O-P-R trong RViz.
13. Cho `SocialLayer` nhận output model.
14. Kiểm tra social cost trong global/local costmap.
15. Đánh giá navigation và khoảng cách an toàn.

---

## 18. Phiên bản tối thiểu để chạy sớm

Nếu muốn thay đổi ít nhất, dùng pipeline:

```text
Camera RGB-D
    ↓
Detector + depth localization + tracker
    ↓
/people
    ↓
Model O-P-R của nhóm
    ↓
/people_groups
    ↓
SocialLayer hiện tại
    ↓
Nav2
```

Các việc bắt buộc:

1. Bật depth camera trong URDF.
2. Dùng Gazebo actor/model mà detector nhận diện được.
3. Thêm detector node.
4. Thêm định vị 3D từ depth.
5. Thêm tracking ID, velocity và orientation.
6. Đổi ground-truth tracker sang `/ground_truth/people`.
7. Thay group detector bằng node model O-P-R.
8. Sửa launch và khai báo dependency.

Chỉ cần thay đổi message và `SocialLayer` nếu model tính O-P-R động riêng cho từng người hoặc output không thể biểu diễn bằng `Groups.msg` hiện tại.

---

## 19. Checklist kiểm tra trước khi cho robot chạy

- [ ] Camera publish RGB, depth và camera info.
- [ ] Timestamp của RGB và depth đồng bộ.
- [ ] Có TF từ optical frame tới `base_link`, `odom` và `map`.
- [ ] Detector nhận đúng actor người trong Gazebo.
- [ ] Detection 3D có đơn vị mét và đúng trục tọa độ.
- [ ] Tracker duy trì ID ổn định.
- [ ] Velocity không nhiễu quá lớn.
- [ ] Orientation đúng hướng người nhìn hoặc di chuyển.
- [ ] `/people` chỉ đến từ pipeline camera.
- [ ] `/ground_truth/people` chỉ dùng để đánh giá.
- [ ] Model O-P-R publish đúng frame và timestamp.
- [ ] RViz hiển thị O-P-R đúng vị trí người.
- [ ] Social cost xuất hiện trong global và local costmap.
- [ ] Cost cũ được xóa khi người di chuyển hoặc biến mất.
- [ ] VoxelLayer vẫn ngăn va chạm với thân người.
- [ ] Planner ưu tiên tránh P/R-space và không đi vào O-space.
