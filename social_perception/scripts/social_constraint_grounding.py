#!/usr/bin/env python3
"""Block D: turn Block-B tracks into current social Gaussian constraints.

Input: ``/people/tracked_state_json`` (``std_msgs/String``) from Block B.  Its
``position_m`` values must be metres in ``expected_frame``.  Output:
``/social_constraints/markers`` (``visualization_msgs/MarkerArray``) always,
plus optional current ``nav_msgs/OccupancyGrid`` debug topics.

The four explicit geometries are crossing, talking, standing, and stationary.
Standing uses the person-to-configured-object vector as the ellipse heading.
Stationary is intentionally separate: it creates a fixed circular 0.5 m
Gaussian around one person without using velocity or an object pose.

For the Gazebo RL bookshelf scene, the same MarkerArray also contains a static
RViz CUBE.  It is visualisation only: Gazebo collision remains the authority
for simulation physics, and this node intentionally has no Nav2 dependency.
"""

import json
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray


@dataclass(frozen=True)
class ConstraintZone:
    """Chứa thông tin vùng ràng buộc xã hội (Gaussian zone)."""
    track_id: int
    x_m: float
    y_m: float
    yaw_rad: float
    front_semi_axis_m: float
    rear_semi_axis_m: float
    left_semi_axis_m: float
    right_semi_axis_m: float
    social_state: str
    social_weight: float
    member_ids: Tuple[int, ...] = ()
    separation_m: float = 0.0
    # Lưu thêm vị trí gốc của các thành viên để vẽ trực quan lên RViz
    member_positions: Tuple[Tuple[float, float], ...] = ()


class SocialConstraintGrounding(Node):
    def __init__(self) -> None:
        super().__init__('social_constraint_grounding')

        # Topic nhận JSON từ Block B (YOLO + ByteTrack + Camera project xuống sàn)
        self.declare_parameter('input_topic', '/people/tracked_state_json')
        self.declare_parameter('current_grid_topic', '/social_constraints/current_grid')
        self.declare_parameter('combined_grid_topic', '/social_constraints/grid')
        self.declare_parameter('output_markers_topic', '/social_constraints/markers')
        self.declare_parameter('expected_frame', 'map')
        self.declare_parameter('publish_cost_grid', False)

        # Tham số lưới chi phí (Cost Grid)
        self.declare_parameter('grid_resolution_m', 0.05)
        self.declare_parameter('grid_origin_x_m', -10.0)
        self.declare_parameter('grid_origin_y_m', -10.0)
        self.declare_parameter('grid_width_m', 20.0)
        self.declare_parameter('grid_height_m', 20.0)
        self.declare_parameter('maximum_cost', 100)
        self.declare_parameter('minimum_published_cost', 1)
        self.declare_parameter('combine_mode', 'max')

        # Tham số hàm Gauss
        self.declare_parameter('gaussian_d0_m', 0.5)
        self.declare_parameter('gaussian_a', 0.5)
        self.declare_parameter('gaussian_c', 0.9)
        self.declare_parameter('gaussian_k_s', 0.5)
        # EMA chỉ làm mượt vị trí trước khi tạo vùng/vẽ marker. 0<alpha<=1;
        # alpha nhỏ mượt hơn nhưng tạo độ trễ khi người thực sự di chuyển.
        self.declare_parameter('position_ema_alpha', 0.20)
        self.declare_parameter('contour_levels', [0.90, 0.75, 0.60, 0.45, 0.30, 0.15])
        self.declare_parameter('social_state', 'auto')
        self.declare_parameter('social_weight_talking', 0.9)
        self.declare_parameter('social_weight_standing', 0.7)
        # Stationary is a manually selected, object-independent one-person
        # case.  Keep the radius tunable in YAML rather than baking a training
        # scenario constant into the geometry code.
        self.declare_parameter('stationary_sigma_m', 0.5)
        self.declare_parameter('social_weight_stationary', 0.5)
        self.declare_parameter('social_weight_crossing', 0.5)
        # Standing means a person looking at one configured object.  The
        # object coordinates use the same metres/frame as Block-B position_m.
        self.declare_parameter('standing_object_x_m', 0.0)
        self.declare_parameter('standing_object_y_m', 0.0)
        self.declare_parameter('standing_object_label', 'bookshelf')
        # The semantic point above is intentionally not the collision-volume
        # centre: in lirs_test.world the bookshelf's origin is its back-centre,
        # while the 0.90 x 0.40 x 1.20 m physical envelope is centred at
        # (0, -0.195, 0.60).  Keeping both configurable lets standing geometry
        # retain its intended target while RViz/RL debug sees the real obstacle.
        self.declare_parameter('static_obstacle_marker_enabled', True)
        self.declare_parameter('static_obstacle_marker_center_x_m', 0.0)
        self.declare_parameter('static_obstacle_marker_center_y_m', -0.195)
        self.declare_parameter('static_obstacle_marker_center_z_m', 0.60)
        self.declare_parameter('static_obstacle_marker_size_x_m', 0.90)
        self.declare_parameter('static_obstacle_marker_size_y_m', 0.40)
        self.declare_parameter('static_obstacle_marker_size_z_m', 1.20)
        self.declare_parameter('static_obstacle_marker_yaw_rad', 0.0)
        self.declare_parameter('marker_lifetime_s', 0.75)

        self.gaussian_d0 = float(self.get_parameter('gaussian_d0_m').value)
        self.gaussian_a = float(self.get_parameter('gaussian_a').value)
        self.gaussian_c = float(self.get_parameter('gaussian_c').value)
        self.gaussian_k = float(self.get_parameter('gaussian_k_s').value)
        self.position_ema_alpha = float(
            self.get_parameter('position_ema_alpha').value)
        if (not all(math.isfinite(v) for v in (
                self.gaussian_d0, self.gaussian_a, self.gaussian_c,
                self.gaussian_k, self.position_ema_alpha))
                or self.gaussian_d0 <= 0 or self.gaussian_a < 0
                or self.gaussian_k < 0 or not 0 <= self.gaussian_c <= 1
                or not 0.0 < self.position_ema_alpha <= 1.0):
            raise ValueError(
                'Gaussian requires d0>0, a/k>=0, 0<=c<=1 and 0<EMA alpha<=1')

        levels = [float(level) for level in self.get_parameter('contour_levels').value]
        if not levels or any(not math.isfinite(level) or not 0.0 < level < 1.0 for level in levels):
            raise ValueError('contour_levels must contain finite fractions between 0 and 1')
        self.contour_levels = sorted(set(levels), reverse=True)
        self.outer_contour_scale = math.sqrt(-2.0 * math.log(self.contour_levels[-1]))

        self.social_weights = {
            state: float(self.get_parameter(f'social_weight_{state}').value)
            for state in ('talking', 'standing', 'stationary', 'crossing')
        }
        self.stationary_sigma_m = float(
            self.get_parameter('stationary_sigma_m').value)
        if (not math.isfinite(self.stationary_sigma_m)
                or self.stationary_sigma_m <= 0.0):
            raise ValueError('stationary_sigma_m must be a finite value > 0 m')
        self.social_state = str(self.get_parameter('social_state').value).strip().lower()
        supported_states = {'auto', *self.social_weights}
        if self.social_state not in supported_states:
            raise ValueError(
                f'social_state must be one of {sorted(supported_states)}, got {self.social_state!r}')

        object_x_m = float(self.get_parameter('standing_object_x_m').value)
        object_y_m = float(self.get_parameter('standing_object_y_m').value)
        if not math.isfinite(object_x_m) or not math.isfinite(object_y_m):
            raise ValueError('standing_object_x_m/y_m must be finite metres')
        self.standing_object_position_m = (object_x_m, object_y_m)
        self.standing_object_label = str(
            self.get_parameter('standing_object_label').value).strip() or 'object'
        self.static_obstacle_marker_enabled = bool(self.get_parameter(
            'static_obstacle_marker_enabled').value)
        static_obstacle_values = tuple(float(self.get_parameter(name).value) for name in (
            'static_obstacle_marker_center_x_m',
            'static_obstacle_marker_center_y_m',
            'static_obstacle_marker_center_z_m',
            'static_obstacle_marker_size_x_m',
            'static_obstacle_marker_size_y_m',
            'static_obstacle_marker_size_z_m',
            'static_obstacle_marker_yaw_rad',
        ))
        if (not all(math.isfinite(value) for value in static_obstacle_values)
                or any(value <= 0.0 for value in static_obstacle_values[3:6])):
            raise ValueError(
                'static obstacle marker requires finite pose and positive size')
        self.static_obstacle_marker_pose_m = static_obstacle_values[:3]
        self.static_obstacle_marker_size_m = static_obstacle_values[3:6]
        self.static_obstacle_marker_yaw_rad = static_obstacle_values[6]

        self.input_topic = self.get_parameter('input_topic').value
        self.expected_frame = self.get_parameter('expected_frame').value
        self.publish_cost_grid = bool(self.get_parameter('publish_cost_grid').value)
        self.grid_resolution_m = max(0.01, float(self.get_parameter('grid_resolution_m').value))
        self.grid_origin_x_m = float(self.get_parameter('grid_origin_x_m').value)
        self.grid_origin_y_m = float(self.get_parameter('grid_origin_y_m').value)
        self.grid_width_m = max(self.grid_resolution_m, float(self.get_parameter('grid_width_m').value))
        self.grid_height_m = max(self.grid_resolution_m, float(self.get_parameter('grid_height_m').value))
        self.maximum_cost = max(1, min(100, int(self.get_parameter('maximum_cost').value)))
        self.minimum_published_cost = max(1, min(self.maximum_cost, int(self.get_parameter('minimum_published_cost').value)))
        self.combine_mode = self.get_parameter('combine_mode').value
        # Bộ nhớ theo ByteTrack ID; đơn vị là mét trong expected_frame. Không
        # phát topic mới và không thay đổi dữ liệu thô do Block B phát ra.
        self.filtered_positions_m: dict[int, Tuple[float, float]] = {}

        self.grid_width_cells = int(math.ceil(self.grid_width_m / self.grid_resolution_m))
        self.grid_height_cells = int(math.ceil(self.grid_height_m / self.grid_resolution_m))

        if self.publish_cost_grid:
            self.current_grid_pub = self.create_publisher(
                OccupancyGrid, self.get_parameter('current_grid_topic').value, 1)
            self.combined_grid_pub = self.create_publisher(
                OccupancyGrid, self.get_parameter('combined_grid_topic').value, 1)

        self.markers_pub = self.create_publisher(
            MarkerArray, self.get_parameter('output_markers_topic').value, 1)
        self.subscription = self.create_subscription(
            String, self.input_topic, self.block_b_callback, 10)

        self.get_logger().info('Block D: Da khoi tao thanh cong voi Marker vi tri nguoi va tam Gauss.')

    def block_b_callback(self, message: String) -> None:
        """Nhận std_msgs/String từ Block B, làm mượt (x,y), rồi phát D outputs.

        Input là JSON ``/people/tracked_state_json`` trong ``expected_frame``.
        EMA chỉ áp dụng cho vị trí dùng tính Gaussian, OccupancyGrid và
        MarkerArray; velocity/heading vẫn là phép đo tức thời của Block B.
        """
        try:
            state = json.loads(message.data)
            frame_id = self.validate_block_b(state)
            filtered_people = self.filter_people_positions(state.get('people', []))
            zones = self.build_constraint_zones(filtered_people)
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            self.get_logger().warn(f'Loi du lieu Block-B: {error}')
            return

        stamp_ns = int(state.get('header', {}).get('stamp_ns', 0))
        if self.publish_cost_grid:
            grid = self.make_grid(frame_id, stamp_ns, zones)
            self.current_grid_pub.publish(grid)
            self.combined_grid_pub.publish(grid)
        self.markers_pub.publish(self.make_markers(frame_id, stamp_ns, zones))

    def filter_people_positions(self, people: Iterable[dict]) -> List[dict]:
        """Return copies of people with EMA-filtered metres coordinates.

        A first observation starts the filter at the measured position, so a
        newly detected person is never delayed.  When an ID disappears from
        the current Block-B message its state is removed: ByteTrack may later
        reuse that ID for a different person and must not inherit old data.
        Malformed entries are deliberately returned unchanged for the normal
        geometry validator to reject and log.
        """
        filtered_people: List[dict] = []
        active_track_ids = set()
        alpha = self.position_ema_alpha

        for person in people:
            try:
                track_id = int(person['track_id'])
                position = person['position_m']
                measured_x_m = float(position['x'])
                measured_y_m = float(position['y'])
                if not math.isfinite(measured_x_m) or not math.isfinite(measured_y_m):
                    raise ValueError('position_m must be finite')
            except (TypeError, ValueError, KeyError):
                filtered_people.append(person)
                continue

            previous = self.filtered_positions_m.get(track_id)
            if previous is None:
                filtered_x_m, filtered_y_m = measured_x_m, measured_y_m
            else:
                filtered_x_m = alpha * measured_x_m + (1.0 - alpha) * previous[0]
                filtered_y_m = alpha * measured_y_m + (1.0 - alpha) * previous[1]

            self.filtered_positions_m[track_id] = (filtered_x_m, filtered_y_m)
            active_track_ids.add(track_id)
            filtered_person = dict(person)
            filtered_position = dict(position)
            filtered_position['x'] = filtered_x_m
            filtered_position['y'] = filtered_y_m
            filtered_person['position_m'] = filtered_position
            filtered_people.append(filtered_person)

        self.filtered_positions_m = {
            track_id: position_m
            for track_id, position_m in self.filtered_positions_m.items()
            if track_id in active_track_ids
        }
        return filtered_people

    def validate_block_b(self, state: dict) -> str:
        header = state.get('header')
        if not isinstance(header, dict):
            raise ValueError('Thieu header trong ban tin JSON')
        frame_id = header.get('frame_id')
        if frame_id != self.expected_frame:
            raise ValueError(f'Sai toa do: can {self.expected_frame}, nhan {frame_id}')
        people = state.get('people')
        if not isinstance(people, list):
            raise ValueError('people phai la danh sach (list)')
        return frame_id

    def build_constraint_zones(self, people: Iterable[dict]) -> List[ConstraintZone]:
        people = list(people)
        selected_state = self.social_state
        if selected_state == 'auto':
            if len(people) not in (1, 2):
                return []
            selected_state = 'talking' if len(people) == 2 else 'crossing'
        if selected_state == 'talking':
            if len(people) != 2:
                return []
            try:
                return [self.make_talking_zone(people[0], people[1])]
            except (TypeError, ValueError, KeyError) as error:
                self.get_logger().warn(f'Bo qua cap talking loi: {error}')
                return []
        if selected_state == 'standing':
            if len(people) != 1:
                return []
            try:
                return [self.make_standing_zone(people[0])]
            except (TypeError, ValueError, KeyError) as error:
                self.get_logger().warn(f'Bo qua nguoi dung yen loi: {error}')
                return []
        if selected_state == 'stationary':
            if len(people) != 1:
                return []
            try:
                return [self.make_stationary_zone(people[0])]
            except (TypeError, ValueError, KeyError) as error:
                self.get_logger().warn(f'Bo qua nguoi stationary loi: {error}')
                return []
        zones: List[ConstraintZone] = []
        for person in people:
            try:
                zones.extend(self.make_zones_for_person(person))
            except (TypeError, ValueError, KeyError) as error:
                self.get_logger().warn(f'Bo qua nguoi loi: {error}')
        return zones

    def make_talking_zone(self, first: dict, second: dict) -> ConstraintZone:
        first, second = sorted((first, second), key=lambda p: int(p['track_id']))
        first_id, second_id = int(first['track_id']), int(second['track_id'])
        x1, y1 = float(first['position_m']['x']), float(first['position_m']['y'])
        x2, y2 = float(second['position_m']['x']), float(second['position_m']['y'])

        dx, dy = x2 - x1, y2 - y1
        separation_m = math.hypot(dx, dy)
        if not math.isfinite(separation_m) or separation_m <= 1e-6:
            raise ValueError('Hai nguoi trung vi tri, khong tinh duoc truc yaw')

        factor = 1.0 + self.gaussian_a * (1.0 - self.gaussian_c)
        sigma_h = sigma_r = factor * (separation_m + self.gaussian_d0) / 4.0
        sigma_s = sigma_h / 3.0

        # Tam Gauss la trung diem cua hai nguoi: (x1+x2)/2, (y1+y2)/2
        return ConstraintZone(
            track_id=first_id,
            x_m=x1 / 2.0 + x2 / 2.0,
            y_m=y1 / 2.0 + y2 / 2.0,
            yaw_rad=math.atan2(dy, dx),
            front_semi_axis_m=sigma_h,
            rear_semi_axis_m=sigma_r,
            left_semi_axis_m=sigma_s,
            right_semi_axis_m=sigma_s,
            social_state='talking',
            social_weight=self.social_weights['talking'],
            member_ids=(first_id, second_id),
            separation_m=separation_m,
            member_positions=((x1, y1), (x2, y2))
        )

    def make_zones_for_person(self, person: dict) -> List[ConstraintZone]:
        """Create the current one-person social zone from a Block-B track.

        This is the crossing formulation.  The standing formulation is kept
        separate because it derives its heading and sigma from the configured
        object rather than noisy RGB-D velocity/body yaw.
        """
        track_id = int(person['track_id'])
        position = person['position_m']
        velocity = person.get('velocity_mps', {})
        x_m = float(position['x'])
        y_m = float(position['y'])
        vx_mps = float(velocity.get('vx', 0.0))
        vy_mps = float(velocity.get('vy', 0.0))
        speed_mps = math.hypot(vx_mps, vy_mps)
        stationary = speed_mps <= 1e-6

        if not stationary:
            yaw_rad = math.atan2(vy_mps, vx_mps)
        elif bool(person.get('body_orientation_valid', False)):
            yaw_rad = float(person['body_orientation_rad'])
        elif bool(person.get('motion_heading_valid', False)):
            yaw_rad = float(person['motion_heading_rad'])
        else:
            # A circle is invariant under yaw.  Use zero solely because the
            # marker/cost evaluator still carries a yaw field for all zones.
            yaw_rad = 0.0

        factor = 1.0 + self.gaussian_a * (1.0 - self.gaussian_c)
        selected_state = 'crossing' if self.social_state == 'auto' else self.social_state
        sigma_h = factor * (self.gaussian_d0 + self.gaussian_k * speed_mps)
        sigma_s = sigma_r = sigma_h / 3.0

        # Tam Gauss cua 1 nguoi trung voi toa do nguoi do
        return [ConstraintZone(
            track_id=track_id,
            x_m=x_m,
            y_m=y_m,
            yaw_rad=yaw_rad,
            front_semi_axis_m=sigma_h,
            rear_semi_axis_m=sigma_r,
            left_semi_axis_m=sigma_s,
            right_semi_axis_m=sigma_s,
            social_state=selected_state,
            social_weight=self.social_weights[selected_state],
            member_ids=(track_id,),
            separation_m=0.0,
            member_positions=((x_m, y_m),)
        )]

    def make_standing_zone(self, person: dict) -> ConstraintZone:
        """Create the one-person ellipse for ``standing`` near an object.

        The Gaussian centre is the EMA-filtered person position.  The yaw is
        the vector person -> object, which is the known person-facing direction
        in the bookshelf test.  With ``d_obj`` in metres, the supplied formula
        is sigma_h=sigma_r=[1+a*(1-c)]*(d_obj+d0)/2 and sigma_s=sigma_h/3.
        """
        track_id = int(person['track_id'])
        x_m = float(person['position_m']['x'])
        y_m = float(person['position_m']['y'])
        object_x_m, object_y_m = self.standing_object_position_m
        dx_m, dy_m = object_x_m - x_m, object_y_m - y_m
        object_distance_m = math.hypot(dx_m, dy_m)
        if (not all(math.isfinite(value) for value in (
                x_m, y_m, object_x_m, object_y_m, object_distance_m))
                or object_distance_m <= 1e-6):
            raise ValueError('Nguoi va vat phai co toa do huu han, cach nhau > 0 m')

        factor = 1.0 + self.gaussian_a * (1.0 - self.gaussian_c)
        sigma_h = factor * (object_distance_m + self.gaussian_d0) / 2.0
        sigma_s = sigma_r = sigma_h / 3.0
        return ConstraintZone(
            track_id=track_id,
            x_m=x_m,
            y_m=y_m,
            yaw_rad=math.atan2(dy_m, dx_m),
            front_semi_axis_m=sigma_h,
            rear_semi_axis_m=sigma_r,
            left_semi_axis_m=sigma_s,
            right_semi_axis_m=sigma_s,
            social_state='standing',
            social_weight=self.social_weights['standing'],
            member_ids=(track_id,),
            separation_m=object_distance_m,
            member_positions=((x_m, y_m),)
        )

    def make_stationary_zone(self, person: dict) -> ConstraintZone:
        """Create an object-independent circular Gaussian for one still person.

        The mode is selected manually with ``social_state:=stationary``.  It
        deliberately ignores velocity, body yaw, and the bookshelf so noisy
        RGB-D velocity cannot stretch or rotate the training/debug region.
        ``stationary_sigma_m`` is assigned to every semi-axis in metres.
        """
        track_id = int(person['track_id'])
        x_m = float(person['position_m']['x'])
        y_m = float(person['position_m']['y'])
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            raise ValueError('stationary person position must be finite')

        return ConstraintZone(
            track_id=track_id,
            x_m=x_m,
            y_m=y_m,
            # A circle has no preferred direction; zero avoids implying that
            # a stationary body-yaw estimate affects this explicitly fixed mode.
            yaw_rad=0.0,
            front_semi_axis_m=self.stationary_sigma_m,
            rear_semi_axis_m=self.stationary_sigma_m,
            left_semi_axis_m=self.stationary_sigma_m,
            right_semi_axis_m=self.stationary_sigma_m,
            social_state='stationary',
            social_weight=self.social_weights['stationary'],
            member_ids=(track_id,),
            separation_m=0.0,
            member_positions=((x_m, y_m),)
        )

    def zone_cost(self, x_m: float, y_m: float, zone: ConstraintZone) -> int:
        dx = x_m - zone.x_m
        dy = y_m - zone.y_m
        forward = math.cos(zone.yaw_rad) * dx + math.sin(zone.yaw_rad) * dy
        lateral = -math.sin(zone.yaw_rad) * dx + math.cos(zone.yaw_rad) * dy
        forward_axis = zone.front_semi_axis_m if forward >= 0.0 else zone.rear_semi_axis_m
        lateral_axis = zone.left_semi_axis_m if lateral >= 0.0 else zone.right_semi_axis_m
        normalized_distance_sq = ((forward / forward_axis) ** 2 + (lateral / lateral_axis) ** 2)
        return int(round(self.maximum_cost * zone.social_weight * math.exp(-0.5 * normalized_distance_sq)))

    def make_grid(self, frame_id: str, stamp_ns: int, zones: List[ConstraintZone]) -> OccupancyGrid:
        grid = OccupancyGrid()
        grid.header.frame_id = frame_id
        grid.header.stamp.sec = stamp_ns // 1_000_000_000
        grid.header.stamp.nanosec = stamp_ns % 1_000_000_000
        grid.info.resolution = self.grid_resolution_m
        grid.info.width = self.grid_width_cells
        grid.info.height = self.grid_height_cells
        grid.info.origin.position.x = self.grid_origin_x_m
        grid.info.origin.position.y = self.grid_origin_y_m
        grid.info.origin.orientation.w = 1.0
        grid.data = [0] * (self.grid_width_cells * self.grid_height_cells)

        for row in range(self.grid_height_cells):
            y_m = self.grid_origin_y_m + (row + 0.5) * self.grid_resolution_m
            for col in range(self.grid_width_cells):
                x_m = self.grid_origin_x_m + (col + 0.5) * self.grid_resolution_m
                costs = [self.zone_cost(x_m, y_m, zone) for zone in zones]
                if not costs:
                    continue
                cost = min(self.maximum_cost, sum(costs)) if self.combine_mode == 'sum_clamped' else max(costs)
                if cost >= self.minimum_published_cost:
                    grid.data[row * self.grid_width_cells + col] = cost
        return grid

    def make_markers(self, frame_id: str, stamp_ns: int, zones: List[ConstraintZone]) -> MarkerArray:
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        lifetime_s = float(self.get_parameter('marker_lifetime_s').value)
        sec = stamp_ns // 1_000_000_000
        nanosec = stamp_ns % 1_000_000_000

        for zone_id, zone in enumerate(zones):
            if zone.social_weight == 0.0:
                continue

            # -------------------------------------------------------------
            # 1. MARKER TÂM GAUSS (Hình cầu màu VÀNG)
            # -------------------------------------------------------------
            center_marker = Marker()
            center_marker.header.frame_id = frame_id
            center_marker.header.stamp.sec = sec
            center_marker.header.stamp.nanosec = nanosec
            center_marker.ns = 'social_zone_center'
            center_marker.id = zone_id
            center_marker.type = Marker.SPHERE
            center_marker.action = Marker.ADD
            center_marker.pose.position.x = zone.x_m
            center_marker.pose.position.y = zone.y_m
            center_marker.pose.position.z = 0.1
            center_marker.pose.orientation.w = 1.0
            center_marker.scale.x = 0.15  # Đường kính 15cm
            center_marker.scale.y = 0.15
            center_marker.scale.z = 0.15
            center_marker.color.r = 1.0
            center_marker.color.g = 0.9
            center_marker.color.b = 0.0
            center_marker.color.a = 0.9
            center_marker.lifetime.sec = int(lifetime_s)
            center_marker.lifetime.nanosec = int((lifetime_s % 1.0) * 1_000_000_000)
            markers.markers.append(center_marker)

            # -------------------------------------------------------------
            # 2. MARKER TỌA ĐỘ NGƯỜI (các chấm cầu VÀNG)
            # -------------------------------------------------------------
            # RViz vẽ Marker.POINTS thành ô vuông. Dùng SPHERE_LIST để mỗi
            # tọa độ (x, y), đơn vị mét trong ``frame_id``, là một chấm tròn
            # vàng. Đây chỉ là mốc vị trí, không phải mô hình cơ thể 3-D.
            if zone.member_positions:
                people_marker = Marker()
                people_marker.header.frame_id = frame_id
                people_marker.header.stamp.sec = sec
                people_marker.header.stamp.nanosec = nanosec
                people_marker.ns = 'person_position'
                people_marker.id = zone_id
                people_marker.type = Marker.SPHERE_LIST
                people_marker.action = Marker.ADD
                # Đường kính chấm bằng chấm tâm Gaussian, đơn vị mét.
                people_marker.scale.x = 0.15
                people_marker.scale.y = 0.15
                people_marker.scale.z = 0.15
                people_marker.color.r = 1.0
                people_marker.color.g = 0.9
                people_marker.color.b = 0.0
                people_marker.color.a = 0.95
                people_marker.lifetime = center_marker.lifetime
                for px, py in zone.member_positions:
                    point = Point()
                    point.x = px
                    point.y = py
                    point.z = 0.1
                    people_marker.points.append(point)
                markers.markers.append(people_marker)

            # -------------------------------------------------------------
            # 3. CÁC ĐƯỜNG CONTOUR GAUSS (Màu ĐỎ)
            # -------------------------------------------------------------
            for level_id, level in enumerate(self.contour_levels):
                contour_scale = math.sqrt(-2.0 * math.log(level))
                marker = Marker()
                marker.header.frame_id = frame_id
                marker.header.stamp.sec = sec
                marker.header.stamp.nanosec = nanosec
                marker.ns = 'social_constraint_contours'
                marker.id = zone_id * len(self.contour_levels) + level_id
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.03 if level_id == len(self.contour_levels) - 1 else 0.015
                marker.color.a = 0.85
                marker.color.r = 1.0
                marker.color.g = 0.05
                marker.color.b = 0.0
                marker.lifetime = center_marker.lifetime

                for index in range(73):
                    angle = 2.0 * math.pi * index / 72.0
                    forward_axis = zone.front_semi_axis_m if math.cos(angle) >= 0.0 else zone.rear_semi_axis_m
                    lateral_axis = zone.left_semi_axis_m if math.sin(angle) >= 0.0 else zone.right_semi_axis_m
                    forward = contour_scale * forward_axis * math.cos(angle)
                    lateral = contour_scale * lateral_axis * math.sin(angle)
                    point = Point()
                    point.x = zone.x_m + math.cos(zone.yaw_rad) * forward - math.sin(zone.yaw_rad) * lateral
                    point.y = zone.y_m + math.sin(zone.yaw_rad) * forward + math.cos(zone.yaw_rad) * lateral
                    point.z = 0.05
                    marker.points.append(point)
                markers.markers.append(marker)

            # Nhãn text thông tin
            label = Marker()
            label.header.frame_id = frame_id
            label.header.stamp.sec = sec
            label.header.stamp.nanosec = nanosec
            label.ns = 'social_constraint_info'
            label.id = zone_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = zone.x_m
            label.pose.position.y = zone.y_m
            label.pose.position.z = 1.5
            label.pose.orientation.w = 1.0
            label.scale.z = 0.18
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.lifetime = center_marker.lifetime
            label.text = f'TAM GAUSS ({zone.social_state})\n({zone.x_m:.2f}, {zone.y_m:.2f})'
            if zone.social_state == 'standing':
                label.text += (
                    f'\n{self.standing_object_label}: d_obj='
                    f'{zone.separation_m:.2f} m')
            markers.markers.append(label)

        static_obstacle = self.make_static_obstacle_marker(
            frame_id, sec, nanosec)
        if static_obstacle is not None:
            markers.markers.append(static_obstacle)
        return markers

    def make_static_obstacle_marker(
            self, frame_id: str, sec: int, nanosec: int) -> Optional[Marker]:
        """Return the fixed bookshelf collision envelope for RViz.

        Input frame is the validated Block-B frame (``odom`` in this Gazebo
        scenario; normally ``map`` in a real setup).  The marker has zero
        lifetime so an RL/debug viewer keeps seeing the static obstacle between
        person updates.  Each later MarkerArray starts with DELETEALL then
        re-adds this cube, preventing stale geometry after a parameter change.
        """
        if not self.static_obstacle_marker_enabled:
            return None

        center_x_m, center_y_m, center_z_m = self.static_obstacle_marker_pose_m
        size_x_m, size_y_m, size_z_m = self.static_obstacle_marker_size_m
        obstacle = Marker()
        obstacle.header.frame_id = frame_id
        obstacle.header.stamp.sec = sec
        obstacle.header.stamp.nanosec = nanosec
        obstacle.ns = 'static_obstacles'
        obstacle.id = 0
        obstacle.type = Marker.CUBE
        obstacle.action = Marker.ADD
        obstacle.pose.position.x = center_x_m
        obstacle.pose.position.y = center_y_m
        obstacle.pose.position.z = center_z_m
        obstacle.pose.orientation.z = math.sin(self.static_obstacle_marker_yaw_rad / 2.0)
        obstacle.pose.orientation.w = math.cos(self.static_obstacle_marker_yaw_rad / 2.0)
        obstacle.scale.x = size_x_m
        obstacle.scale.y = size_y_m
        obstacle.scale.z = size_z_m
        # Brown, semi-transparent fill distinguishes a physical static obstacle
        # from red social-cost contours and yellow person-centre markers.
        obstacle.color.r = 0.36
        obstacle.color.g = 0.18
        obstacle.color.b = 0.04
        obstacle.color.a = 0.85
        return obstacle


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = SocialConstraintGrounding()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
