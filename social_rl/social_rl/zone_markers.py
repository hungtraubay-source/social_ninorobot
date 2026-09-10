"""Vẽ K_soc của khối D ra RViz. Chỉ để NHÌN, không nằm trong vòng điều khiển.

    /social_rl/constraint_field  ->  /social_rl/zone_markers   (MarkerArray)
                                 ->  /social_rl/social_costmap (OccupancyGrid)

Chỉ NGHE và vẽ, không đụng gì vào vòng train.

Mỗi zone vẽ:
    LÕI   mặt tô mờ  - chỗ giá trị bằng 1.0, đây là thứ khối F coi là tường
    NGOÀI nét mảnh   - chỗ giá trị về 0.0, ngoài nó không tốn gì

Màu theo `hardness`, KHÔNG theo scene_type: đỏ là o-space cuộc trò chuyện (chỗ
duy nhất không được đi qua), cam là vùng cá nhân (đắt nhưng đi được).

CHỈ VẼ t=0, tức vùng ở NGAY LÚC NÀY. Người đi thì vùng đi theo, đúng một hình.

Bốn mốc dự báo t+0.5 .. t+2.0 CÓ TỒN TẠI trong message và cùng với t=0 làm nên
năm kênh K_soc vào CNN, nhưng vẽ hết ra thì mỗi người thành năm vòng chồng lên
nhau - với người đang đi chúng còn lệch dần theo hướng đi và trông như lỗi.
Muốn xem thì truyền tay, ví dụ `zone_markers.py 0.0,1.0,2.0`.

OccupancyGrid là bản để xem TRÊN MAP: nó trải phẳng như một costmap, và vì là
ảnh xám nên nó cho thấy cả ĐỘ DỐC của trường - 1.0 trong lõi rồi giảm dần ra
biên - thứ mà hai đường viền không diễn tả được. Đó cũng đúng là hình mà policy
nhìn thấy, chỉ khác độ phân giải.

Grid mang frame_id của message (base_link), nên RViz tự kéo nó theo robot; đặt
Fixed Frame là `map` thì thấy nó trượt trên bản đồ.
"""
import math
import sys

import rclpy
import numpy as np
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from social_perception.msg import ConstraintField

SEGMENTS = 72          # số điểm mỗi đường viền
GRID_RANGE = 4.0       # nửa cạnh hộp OccupancyGrid, mét
GRID_RES = 0.10        # mét mỗi ô. Mịn gấp đôi lưới 0.2 m của observation, để
                       # nhìn cho rõ; policy vẫn ăn bản 0.2 m.
HARD = (0.90, 0.15, 0.15)
SOFT = (1.00, 0.55, 0.10)


def reach(angle, front, side, rear):
    """Bán kính ellipse lệch trước/sau theo hướng `angle`, trong frame của zone."""
    along = front if math.cos(angle) >= 0.0 else rear
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    denominator = (cos_a / along) ** 2 + (sin_a / side) ** 2
    return 1.0 / math.sqrt(max(denominator, 1e-12))


def contour(sample, use_core):
    """Điểm của một đường viền, trong frame của message."""
    front, side, rear = (float(v) for v in sample.size)
    cx, cy = float(sample.center[0]), float(sample.center[1])
    cos_o, sin_o = math.cos(sample.orientation), math.sin(sample.orientation)
    points = []
    for index in range(SEGMENTS + 1):
        angle = 2.0 * math.pi * index / SEGMENTS
        radius = reach(angle, front, side, rear)
        if use_core:
            # Biên đặc thật sự: hướng nào ellipse gần hơn lõi thì lấy ellipse.
            radius = min(float(sample.core), radius)
        local_x = radius * math.cos(angle)
        local_y = radius * math.sin(angle)
        points.append((cx + cos_o * local_x - sin_o * local_y,
                       cy + sin_o * local_x + cos_o * local_y))
    return points


def field_value(sample, x, y):
    """Giá trị K_soc của MỘT zone trên cả lưới. Cùng công thức khối D dùng.

    1.0 trong lõi, dốc tuyến tính ra 0.0 ở biên. Hướng nào biên gần hơn lõi thì
    đặc tới biên - đó là trục ngắn của vùng nhóm.
    """
    front, side, rear = (float(v) for v in sample.size)
    core = float(sample.core)
    delta_x = x - float(sample.center[0])
    delta_y = y - float(sample.center[1])
    cos_o, sin_o = math.cos(sample.orientation), math.sin(sample.orientation)
    forward = delta_x * cos_o + delta_y * sin_o
    sideways = -delta_x * sin_o + delta_y * cos_o
    along = np.where(forward >= 0.0, front, rear)
    distance = np.hypot(forward, sideways)
    safe = np.maximum(distance, 1e-6)
    denominator = (forward / safe / along) ** 2 + (sideways / safe / side) ** 2
    reach = 1.0 / np.sqrt(np.maximum(denominator, 1e-12))
    span = np.maximum(reach - core, 1e-6)
    return np.clip((reach - distance) / span, 0.0, 1.0)


class ZoneMarkers(Node):
    def __init__(self):
        super().__init__('zone_markers')
        # Mặc định chỉ t=0. Vẽ cả bốn mốc dự báo thì mỗi người thành năm vòng
        # chồng lên nhau, và với người đang đi chúng lệch dần theo hướng đi
        # nên trông y như lỗi. Đặt [0.0, 1.0, 2.0] khi thật sự muốn soi dự báo.
        self.declare_parameter('prediction_times', [0.0])
        self.horizons = [float(v) for v in
                         self.get_parameter('prediction_times').value]
        self.publisher = self.create_publisher(
            MarkerArray, '/social_rl/zone_markers', 10)
        self.grid_publisher = self.create_publisher(
            OccupancyGrid, '/social_rl/social_costmap', 10)
        cells = int(round(2.0 * GRID_RANGE / GRID_RES))
        axis = (-GRID_RANGE + GRID_RES * (0.5 + np.arange(cells))).astype(np.float32)
        # (hàng, cột) = (y, x) theo đúng thứ tự OccupancyGrid đọc dữ liệu.
        self.grid_x = axis[None, :]
        self.grid_y = axis[:, None]
        self.grid_cells = cells
        self.create_subscription(ConstraintField,
                                 '/social_rl/constraint_field', self._on_field, 10)
        self.get_logger().info(
            f'vẽ mốc t={self.horizons} ra /social_rl/zone_markers, và K_soc tại '
            f't=0 ra /social_rl/social_costmap '
            f'({cells}x{cells} ô, {GRID_RES} m/ô), frame theo message')

    def _on_field(self, field):
        markers = MarkerArray()
        clear = Marker()
        clear.header = field.header
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        marker_id = 0
        for zone in field.zones:
            colour = HARD if zone.hardness == 'hard' else SOFT
            for sample in zone.trajectory_of_zone:
                if not any(abs(sample.t - h) < 1e-6 for h in self.horizons):
                    continue
                newest = abs(sample.t) < 1e-6
                fade = 1.0 - 0.30 * sample.t

                # Lõi: TÔ MẶT. Một đường viền nữa chỉ thành thêm một vòng tròn
                # khó phân biệt; cái mắt cần thấy là một VÙNG có diện tích.
                fill = Marker()
                fill.header = field.header
                fill.ns = f'{zone.zone_id}/core'
                fill.id = marker_id
                marker_id += 1
                fill.type = Marker.TRIANGLE_LIST
                fill.action = Marker.ADD
                fill.pose.orientation.w = 1.0
                fill.scale.x = fill.scale.y = fill.scale.z = 1.0
                fill.color.r, fill.color.g, fill.color.b = colour
                fill.color.a = max(0.10, 0.35 * fade)
                points = contour(sample, True)
                centre = Point()
                centre.x = float(sample.center[0])
                centre.y = float(sample.center[1])
                centre.z = 0.02
                for first, second in zip(points, points[1:]):
                    for x, y in ((centre.x, centre.y), first, second):
                        p = Point()
                        p.x, p.y, p.z = float(x), float(y), 0.02
                        fill.points.append(p)
                markers.markers.append(fill)

                # Biên ngoài: chỉ một nét mảnh, đủ để biết vùng ảnh hưởng tới đâu.
                edge = Marker()
                edge.header = field.header
                edge.ns = f'{zone.zone_id}/outer'
                edge.id = marker_id
                marker_id += 1
                edge.type = Marker.LINE_STRIP
                edge.action = Marker.ADD
                edge.pose.orientation.w = 1.0
                edge.scale.x = 0.03 if newest else 0.015
                edge.color.r, edge.color.g, edge.color.b = colour
                edge.color.a = max(0.15, 0.7 * fade)
                for x, y in contour(sample, False):
                    p = Point()
                    p.x, p.y, p.z = x, y, 0.02
                    edge.points.append(p)
                markers.markers.append(edge)

            # Nhãn: ai, tình huống gì, cứng hay mềm, tin bao nhiêu.
            now = zone.trajectory_of_zone[0]
            label = Marker()
            label.header = field.header
            label.ns = f'{zone.zone_id}/label'
            label.id = marker_id
            marker_id += 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(now.center[0])
            label.pose.position.y = float(now.center[1])
            label.pose.position.z = 1.2
            label.pose.orientation.w = 1.0
            label.scale.z = 0.22
            label.color.r, label.color.g, label.color.b = colour
            label.color.a = 1.0
            label.text = (f'{zone.scene_type or "?"} [{zone.hardness}] '
                          f'{",".join(zone.track_ids)} c={zone.confidence:.2f}')
            markers.markers.append(label)
        self.publisher.publish(markers)
        self._publish_grid(field)

    def _publish_grid(self, field):
        """K_soc tại t=0 dưới dạng OccupancyGrid, để xem phẳng trên map.

        Dựng lại từ chính các zone trong message chứ không đọc lưới của
        observation: node này không có quyền vào tiến trình train, và dựng lại
        từ hình học là cách duy nhất thấy được ĐÚNG thứ khối D đã công bố.
        """
        values = np.zeros((self.grid_cells, self.grid_cells), dtype=np.float32)
        for zone in field.zones:
            for sample in zone.trajectory_of_zone:
                if abs(sample.t) > 1e-6:
                    continue
                np.maximum(values, field_value(sample, self.grid_x, self.grid_y),
                           out=values)

        grid = OccupancyGrid()
        grid.header = field.header
        grid.info.resolution = GRID_RES
        grid.info.width = self.grid_cells
        grid.info.height = self.grid_cells
        grid.info.origin.position.x = -GRID_RANGE
        grid.info.origin.position.y = -GRID_RANGE
        grid.info.origin.orientation.w = 1.0
        # 0..100 là thang OccupancyGrid; ô rỗng để 0 chứ không phải -1, vì -1 là
        # "chưa biết" và RViz tô nó khác hẳn.
        grid.data = (values * 100.0).astype(np.int8).ravel().tolist()
        self.grid_publisher.publish(grid)


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
