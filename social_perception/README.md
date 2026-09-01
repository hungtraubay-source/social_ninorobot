# social_perception

Gói ROS 2 định vị và bám vết người từ RGB-D và YOLO Pose; **không dùng VLM**
ở runtime, không tạo vùng O/P/R.

Luồng xử lý:

1. RGB + depth + `CameraInfo` đi vào node.
2. YOLO Pose tìm nhiều người và 17 COCO keypoints mỗi người (cần pose checkpoint).
3. Median depth biến keypoint/torso 2-D thành điểm 3-D; TF biến chúng sang
   `target_frame` (mặc định `map`).
4. Tracker giữ ID ổn định. Khi profile mặt thấy rõ, horizontal `nose → shoulder`
   cho heading trực tiếp; còn lại yaw là weighted circular average hip, shoulder,
   nose-to-torso và được lọc theo `track_id` trong RViz.
5. Node xuất người đã định vị/tracking sang `/people`; `SocialLayer` của Nav2
   dùng từng người riêng lẻ.

VLM/Qwen cũ còn nằm trong source dưới nhãn `LEGACY / DISABLED` để tham khảo,
nhưng node ép `enable_vlm` thành `false`: không tải model, không tạo crop ảnh,
không chạy GPU inference. Các tham số `vlm_*` trong YAML chỉ được giữ để các
file cấu hình cũ không lỗi.

## Topic input

- `/camera/color/image_raw` (`sensor_msgs/Image`): ảnh RGB cho YOLO Pose.
- `/camera/depth/image_raw` (`sensor_msgs/Image`): depth theo pixel RGB.
- `/camera/color/camera_info` (`sensor_msgs/CameraInfo`): intrinsics RGB dùng
  để đổi pixel sang 3-D.
- `/camera/depth/camera_info` (`sensor_msgs/CameraInfo`): metadata depth riêng,
  optional. Depth vẫn phải được register/aligned với RGB trước khi đưa vào node.

Tên cụ thể là parameter `rgb_topic`, `depth_topic`, `camera_info_topic` và
`depth_camera_info_topic`; profile Gazebo giữ namespace camera mô phỏng hiện có.

Hai profile simulation và camera thật được giữ đồng bộ tại
`config/social_vlm_perception.yaml` và
`config/social_vlm_perception_real.yaml`.

## Cài dependency, build và chạy

```bash
python3 -m pip install -r social_perception/requirements.txt
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to social_perception social_navigation
source install/setup.bash
ros2 launch social_navigation social_bringup.launch.py rviz:=true
```

## Topic chính

- `/people`: mọi người đã được định vị 3-D trong frame navigation.
- `/people_observations`: bbox YOLO, confidence, 17 keypoints 2-D, các
  keypoint 3-D hợp lệ, depth torso và yaw của từng `track_id`.
- `/camera/color/image_raw`, `/camera/depth/image_raw`: RGB/depth thô để
  record. Trong Gazebo, RGB-D được gắn trên robot và phát trực tiếp hai topic
  này; camera thật cũng dùng cùng tên chuẩn.
- `/yolo/people` (`vision_msgs/Detection2DArray`): bbox, `id` track (khi có
  depth/TF) và confidence YOLO.
- `/people/orientation`: `body_yaw_rad` và trạng thái valid của từng `track_id`.
- `/people/tracks` (`social_perception/People`): pose, velocity và heading của
  mỗi người. `/people` vẫn được phát song song cho SocialLayer/Nav2 cũ.
- `/scenario/ground_truth` (`social_perception/People`): nhãn pose actor thật
  do Gazebo phát trong frame `world`; actor đang bị ẩn sẽ không có trong mảng.
- `/odom/unfiltered` (khi mô phỏng chạy `run_ekf:=false`), hoặc `/odom` (khi
  bật EKF), cùng `/tf`: pose robot do Gazebo/robot localization phát, không
  phải dữ liệu perception tạo ra.
- `/social_perception/person_markers`: người, keypoint 3-D, skeleton và mũi
  tên body-yaw khi hai vai có depth hợp lệ.
- `/social_perception/detections_2d`, `/social_perception/annotated_image`,
  `/social_perception/depth_visualization`: topic debug camera.

`/social_perception/vlm_person_states` (`VlmPersonStates`) là output optional
của VLM. Node lấy 4 RGB frame có overlay ByteTrack ID, rồi xuất state semantic
theo từng track, ví dụ `track_id: 12, state: crossing`. Hiện Nav2 không dùng
topic này để tạo costmap. `/social_perception/talking_interactions` chỉ còn là
topic legacy của nhánh VLM cũ đã tắt.

`/social_perception/vlm_input_image` là frame RGB mới nhất đã vẽ bbox và
`ID <track_id>` trước khi Qwen nhận nó. Mở topic này trong RViz để xác nhận
format ảnh runtime tương thích với format đã dùng khi fine-tune.

## Lưu dataset RGB-D và JSON semantic

Recorder chạy tách khỏi YOLO. Nó chỉ ghi một sample khi RGB, depth,
`/people_observations` và `/people/tracks` cùng timestamp đã đến đủ. Kết quả là
`rgb/<sample_id>.jpg`, `depth/<sample_id>.png` (16-bit mm) và
`metadata/<sample_id>.json`.

Schema `social_semantics_v1` được tối ưu cho semantic VLM: fine-tune loader chỉ
cần `rgb_path` làm image input và `vlm_target.scene_semantics` làm target.
Gazebo phát `/scenario/state` với `approaching`, `talking`, `dispersing`,
`away`, được map thành `people_approaching`, `two_people_conversing`,
`people_dispersing`, `no_people`. Vì vậy label không do recorder đoán từ ảnh.

`runtime_context` giữ track ID, vị trí tương đối robot, distance, body heading,
speed, motion direction, confidence, pair features và object tĩnh cho các thử
nghiệm prompt/social logic. `evaluation_only` giữ pose actor thật và pose robot;
fine-tune/inference không được đưa block này vào input VLM. `scene_objects_csv`
là metadata scene tĩnh do người chạy scenario khai báo.

`social_collection.launch.py` là collector cho Gazebo: robot chạy tới các pose
`capture_distances_m` và `capture_angles_deg` quanh một người đã track, quay
mặt về họ rồi mới mở recorder. Mỗi file metadata `social_semantics_v2` chứa
Block-B đồng bộ ở `runtime_context.block_b`. Collector mặc định chỉ lưu RGB +
metadata; depth vẫn dùng để perception định vị người nhưng không ghi ra đĩa.
Đặt `save_depth:=true` nếu cần xuất depth PNG. Ví dụ 4 pose gần/xa:

```bash
ros2 launch social_perception social_collection.launch.py \
  output_directory:=recordings/orbit_near_far \
  capture_distances_m:='1.0,1.5,2.0,1.5' \
  capture_angles_deg:='0,90,180,-90' \
  capture_duration_s:=1.2 sample_rate_hz:=2.0
```

```bash
# Camera thật, chạy sau khi perception đã chạy
ros2 launch social_perception social_dataset_recorder.launch.py \
  output_directory:=recordings/social_perception

# Gazebo: giữ empty frame để có class no_people.
ros2 launch social_perception social_dataset_recorder.launch.py \
  output_directory:=recordings/social_perception_sim \
  scenario_name:=gathering episode_id:=conversation_0042 \
  dataset_split:=train scene_objects_csv:=shelf_right \
  save_empty_frames:=true
```
