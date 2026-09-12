"""Vẽ K_soc của khối D ra RViz. Chỉ để NHÌN, không nằm trong vòng điều khiển.

    /social_gt/people, /animated_people/scenario  ->  compile_zones()  ->
        /social_rl/zone_markers   (MarkerArray, một ellipse 1-sigma + nhãn mỗi zone)
        /social_rl/social_costmap (OccupancyGrid, render_zones() y hệt CNN ăn)

11-09-2026, VIẾT LẠI HOÀN TOÀN. Bản cũ nghe message `social_perception/msg/
ConstraintField` do một node khác publish -- message đó đã bị xoá khỏi package
từ đợt cắt dự báo, và không có node train/agent nào từng publish nó (đã kiểm
tra: ros_interface.py chỉ tính K_soc trong tiến trình, không serialize ra
topic). Node cũ vì vậy import lỗi ngay lập tức và chưa từng chạy được.

Bản này TỰ tính K_soc, y hệt cách ros_interface.observe() làm lúc train:
    người: /social_gt/people (id, pose, velocity) -- animated_people_release.cpp
    scene_type: /animated_people/scenario, qua đúng bảng
                ground_truth._SCENARIO_SCENE_TYPE (không chép lại)
    công thức: constraint_field.compile_zones()/render_zones() (không chép lại)

CHỈ CHẠY ĐƯỢC TRONG SIM, cần /social_gt/people -- tức world có nạp plugin
animated_people_factory (lirs_test.world hoặc bookstore.world). Không cần robot
nào tồn tại: publish thẳng trong frame `world`, dựa vào cạnh TF world -> map mà
gazebo.launch.py đã publish sẵn để xem chồng lên map trong RViz (Fixed Frame:
map hoặc world đều được).

Gaussian không có biên cứng nên MarkerArray chỉ vẽ một ellipse ở bán kính
1-sigma để có hình dạng tham khảo (KHÔNG phải "trong đó ăn đủ tiền, ngoài đó
miễn phí" như bản ramp cũ) -- OccupancyGrid mới là bản đúng giá trị liên tục,
và cũng đúng là thứ CNN của policy ăn, chỉ khác độ phân giải.
"""
import math
import traceback

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from social_perception.msg import People
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from social_rl.constraint_field import ConstraintFieldConfig, compile_zones, render_zones
from social_rl.ground_truth import _SCENARIO_SCENE_TYPE, _yaw
from social_rl.observation import RelativeEntity

SEGMENTS = 72           # số điểm ellipse tham khảo mỗi zone
WEIGHT_COLOUR = {        # màu theo scene_type, chỉ để phân biệt bằng mắt
    'talking': (0.90, 0.15, 0.15),
    'waiting': (0.95, 0.55, 0.10),
    'passing': (0.20, 0.55, 0.95),
    'walking': (0.20, 0.55, 0.95),
    '': (0.55, 0.55, 0.55),
}


def sigma_ellipse(sample, radius_sigma: float = 1.0):
    """Đường viền tham khảo ở bán kính `radius_sigma` lần sigma, frame message.

    KHÔNG phải biên 0.0 -- Gaussian không có biên. sigma_h/sigma_r bất đối
    xứng trước/sau, đúng cấu trúc _gaussian_region trong constraint_field.py.
    """
    sigma_h, sigma_s, sigma_r = (float(v) for v in sample.size)
    cx, cy = float(sample.center[0]), float(sample.center[1])
    cos_o, sin_o = math.cos(sample.orientation), math.sin(sample.orientation)
    points = []
    for index in range(SEGMENTS + 1):
        angle = 2.0 * math.pi * index / SEGMENTS
        along = sigma_h if math.cos(angle) >= 0.0 else sigma_r
        local_x = radius_sigma * along * math.cos(angle)
        local_y = radius_sigma * sigma_s * math.sin(angle)
        points.append((cx + cos_o * local_x - sin_o * local_y,
                       cy + sin_o * local_x + cos_o * local_y))
    return points


class ZoneMarkers(Node):
    def __init__(self):
        super().__init__('zone_markers')
        self.declare_parameter('people_topic', '/social_gt/people')
        # 12-09-2026: đọc /animated_people/current_scenario (latched, do
        # animated_people_release.cpp phát lại), KHÔNG phải
        # /animated_people/scenario (--once, mất ngay sau khi gửi). Node khởi
        # động sau khi lệnh scenario đã gửi vẫn bắt kịp được, không cần ai
        # gửi lại.
        self.declare_parameter('scenario_topic',
                               '/animated_people/current_scenario')
        # Để trống thì đọc observation.constraint_field trong config gói kèm
        # theo social_rl, tức đúng tham số Gauss rl_train.yaml đang dùng.
        self.declare_parameter('env_config', '')
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('grid_center_x', 0.0)
        self.declare_parameter('grid_center_y', 0.0)
        self.declare_parameter('grid_range', 4.0)     # nửa cạnh hộp, mét
        self.declare_parameter('grid_resolution', 0.10)

        people_topic = str(self.get_parameter('people_topic').value)
        scenario_topic = str(self.get_parameter('scenario_topic').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        grid_range = float(self.get_parameter('grid_range').value)
        self.grid_resolution = float(self.get_parameter('grid_resolution').value)
        self.grid_cx = float(self.get_parameter('grid_center_x').value)
        self.grid_cy = float(self.get_parameter('grid_center_y').value)

        config_path = str(self.get_parameter('env_config').value)
        if not config_path:
            config_path = (get_package_share_directory('social_rl')
                           + '/config/rl_train.yaml')
        with open(config_path, 'r') as handle:
            saved = yaml.safe_load(handle) or {}
        self.cfg = ConstraintFieldConfig.from_dict(
            (saved.get('observation') or {}).get('constraint_field') or {})

        cells = int(round(2.0 * grid_range / self.grid_resolution))
        offsets = (-grid_range + self.grid_resolution
                  * (0.5 + np.arange(cells))).astype(np.float32)
        self.x_axis = self.grid_cx + offsets
        self.y_axis = self.grid_cy + offsets
        self.grid_cells = cells

        self._scene_type = ''
        latest = QoSProfile(
            depth=1, history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE)
        self.create_subscription(People, people_topic, self._on_people, latest)
        scenario_qos = QoSProfile(
            depth=1, history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            String, scenario_topic, self._on_scenario, scenario_qos)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/social_rl/zone_markers', 10)
        self.grid_pub = self.create_publisher(
            OccupancyGrid, '/social_rl/social_costmap', 10)
        self.get_logger().info(
            f'zone_markers: nghe {people_topic} + {scenario_topic}, vẽ K_soc '
            f'({self.cfg.d0=:.2f} d0, weights={self.cfg.weights}) quanh '
            f'({self.grid_cx:.1f}, {self.grid_cy:.1f}) frame `{self.frame_id}`, '
            f'lưới {self.grid_cells}x{self.grid_cells} ô {self.grid_resolution} m/ô')

    def _on_scenario(self, message: String):
        name = message.data.split()[0].strip().lower() if message.data else ''
        self._scene_type = _SCENARIO_SCENE_TYPE.get(name, self._scene_type)

    def _on_people(self, message: People):
        # 12-09-2026: bọc try/except CHỈ để LOG lỗi thật ra terminal, không che
        # đi. Trước đây một ngoại lệ ở đây làm callback câm lặng vĩnh viễn (executor
        # nuốt mất traceback) trong khi node vẫn sống và RViz vẫn hiển thị ảnh
        # MarkerArray/OccupancyGrid ĐÓNG BĂNG từ lần cuối chạy được - trông y hệt
        # "vùng vẽ sai" chứ không phải "node đã chết". Đo 12-09-2026: restart sau
        # đó publish lại ngay 17 Hz, không lỗi nào lặp lại - nguyên nhân gốc chưa
        # xác định được, log này để lần sau bắt được ngay tại chỗ.
        try:
            people = []
            for person in message.people:
                people.append(RelativeEntity(
                    x=person.pose.position.x, y=person.pose.position.y,
                    vx=person.velocity.linear.x, vy=person.velocity.linear.y,
                    facing=_yaw(person.pose.orientation),
                    scene_type=self._scene_type, track_id=person.id,
                    scene_confidence=1.0))
            field = compile_zones(people, self.cfg, frame=self.frame_id)
            self._publish_markers(field)
            self._publish_grid(field)
        except Exception:
            # rclpy's Logger.error() has no exc_info kwarg (that is Python
            # logging's, not rcutils') -- format the traceback into the
            # message string instead, or this handler masks the real error
            # with a TypeError about an unknown logging filter.
            self.get_logger().error(
                'zone_markers: lỗi khi xử lý /social_gt/people, BỎ QUA frame '
                'này (RViz sẽ đứng ở ảnh cũ cho tới frame kế tiếp thành '
                f'công):\n{traceback.format_exc()}',
                throttle_duration_sec=5.0)

    def _publish_markers(self, field):
        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.frame_id
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        marker_id = 0
        for zone in field.zones:
            colour = WEIGHT_COLOUR.get(zone.scene_type, WEIGHT_COLOUR[''])
            sample = zone.trajectory_of_zone[0]

            edge = Marker()
            edge.header.frame_id = self.frame_id
            edge.ns = f'{zone.zone_id}/sigma1'
            edge.id = marker_id
            marker_id += 1
            edge.type = Marker.LINE_STRIP
            edge.action = Marker.ADD
            edge.pose.orientation.w = 1.0
            edge.scale.x = 0.03
            edge.color.r, edge.color.g, edge.color.b = colour
            edge.color.a = 0.9
            for x, y in sigma_ellipse(sample):
                p = Point()
                p.x, p.y, p.z = float(x), float(y), 0.02
                edge.points.append(p)
            markers.markers.append(edge)

            label = Marker()
            label.header.frame_id = self.frame_id
            label.ns = f'{zone.zone_id}/label'
            label.id = marker_id
            marker_id += 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(sample.center[0])
            label.pose.position.y = float(sample.center[1])
            label.pose.position.z = 1.2
            label.pose.orientation.w = 1.0
            label.scale.z = 0.22
            label.color.r, label.color.g, label.color.b = colour
            label.color.a = 1.0
            label.text = (f'{zone.scene_type or "?"} [{zone.hardness}] wt='
                          f'{sample.weight:.2f} {",".join(zone.track_ids)}')
            markers.markers.append(label)
        self.marker_pub.publish(markers)

    def _publish_grid(self, field):
        values = render_zones(field, self.x_axis, self.y_axis,
                              self.cfg.prediction_times)[0]  # (nx, ny)
        grid = OccupancyGrid()
        grid.header.frame_id = self.frame_id
        grid.info.resolution = self.grid_resolution
        grid.info.width = self.grid_cells
        grid.info.height = self.grid_cells
        grid.info.origin.position.x = float(self.x_axis[0] - self.grid_resolution / 2.0)
        grid.info.origin.position.y = float(self.y_axis[0] - self.grid_resolution / 2.0)
        grid.info.origin.orientation.w = 1.0
        # OccupancyGrid đọc theo hàng = y trước, cột = x sau -> transpose so
        # với (x, y) mà render_zones trả về. 0..100 là thang hợp lệ; để 0 cho
        # ô rỗng chứ không phải -1 ("chưa biết", RViz tô khác hẳn).
        grid.data = (values.T * 100.0).astype(np.int8).ravel().tolist()
        self.grid_pub.publish(grid)


def main(args=None):
    rclpy.init(args=args)
    node = ZoneMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
