#!/usr/bin/env python3
"""Block D: turn Block-B person state into a social constraint field.

The node deliberately has no Nav2 dependency.  It consumes the JSON exported
by Block B and publishes a visual/debug OccupancyGrid plus RViz markers.  This
keeps the social geometry testable before it is connected to a costmap layer
or a safety shield.

The only policy-specific functions are ``make_zones_for_person`` and
``zone_cost``.  Replace their baseline ellipse/Gaussian logic when the final
social-space-time formulation is ready; parsing, frame checks, prediction
handling, grid generation, and visualisation can stay unchanged.
"""

import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from social_perception.msg import SocialConstraintLayer, SocialConstraintLayers


@dataclass(frozen=True)
class ConstraintZone:
    """One time-indexed local social zone in the tracking/map frame.

    Front/rear/left/right dimensions are ellipse semi-axes. A stationary
    person uses the same value for all four and therefore has a circle.
    """

    track_id: int
    dt_s: float
    x_m: float
    y_m: float
    yaw_rad: float
    front_semi_axis_m: float
    rear_semi_axis_m: float
    left_semi_axis_m: float
    right_semi_axis_m: float
    is_prediction: bool


class SocialConstraintGrounding(Node):
    """Ground Block-B state and trajectory into a spatial social field."""

    def __init__(self) -> None:
        super().__init__('social_constraint_grounding')

        # I/O contract. Block B already publishes all positions in map.
        self.declare_parameter('input_topic', '/people/tracked_state_json')
        # current_grid is Figure-1-like K_soc(x, y, t_now).  The legacy
        # combined grid is visual/debug only; do not use it for a time-aware
        # controller because it deliberately loses dt_s.
        self.declare_parameter('current_grid_topic', '/social_constraints/current_grid')
        self.declare_parameter('combined_grid_topic', '/social_constraints/grid')
        self.declare_parameter('prediction_layers_topic', '/social_constraints/prediction_layers')
        self.declare_parameter('output_markers_topic', '/social_constraints/markers')
        self.declare_parameter('expected_frame', 'map')

        # Fixed grid in the map frame. Keep it fixed so RViz and a future
        # costmap consumer do not see a jumping origin.
        self.declare_parameter('grid_resolution_m', 0.05)
        self.declare_parameter('grid_origin_x_m', -10.0)
        self.declare_parameter('grid_origin_y_m', -10.0)
        self.declare_parameter('grid_width_m', 20.0)
        self.declare_parameter('grid_height_m', 20.0)
        self.declare_parameter('maximum_cost', 100)
        self.declare_parameter('minimum_published_cost', 1)
        self.declare_parameter('combine_mode', 'max')  # max | sum_clamped

        # ---- BASELINE geometry: tune or replace inside make_zones_for_person().
        self.declare_parameter('stationary_radius_m', 0.75)
        self.declare_parameter('front_semi_axis_m', 1.10)
        self.declare_parameter('rear_semi_axis_m', 0.65)
        self.declare_parameter('left_semi_axis_m', 0.70)
        self.declare_parameter('right_semi_axis_m', 0.70)
        self.declare_parameter('prediction_cost_decay_per_s', 0.25)
        self.declare_parameter('marker_lifetime_s', 0.75)

        self.input_topic = self.get_parameter('input_topic').value
        self.expected_frame = self.get_parameter('expected_frame').value
        self.grid_resolution_m = max(0.01, float(self.get_parameter('grid_resolution_m').value))
        self.grid_origin_x_m = float(self.get_parameter('grid_origin_x_m').value)
        self.grid_origin_y_m = float(self.get_parameter('grid_origin_y_m').value)
        self.grid_width_m = max(self.grid_resolution_m, float(self.get_parameter('grid_width_m').value))
        self.grid_height_m = max(self.grid_resolution_m, float(self.get_parameter('grid_height_m').value))
        self.maximum_cost = max(1, min(100, int(self.get_parameter('maximum_cost').value)))
        self.minimum_published_cost = max(1, min(self.maximum_cost,
                                                  int(self.get_parameter('minimum_published_cost').value)))
        self.combine_mode = self.get_parameter('combine_mode').value

        self.grid_width_cells = int(math.ceil(self.grid_width_m / self.grid_resolution_m))
        self.grid_height_cells = int(math.ceil(self.grid_height_m / self.grid_resolution_m))

        self.current_grid_pub = self.create_publisher(
            OccupancyGrid, self.get_parameter('current_grid_topic').value, 1)
        self.combined_grid_pub = self.create_publisher(
            OccupancyGrid, self.get_parameter('combined_grid_topic').value, 1)
        self.prediction_layers_pub = self.create_publisher(
            SocialConstraintLayers, self.get_parameter('prediction_layers_topic').value, 1)
        self.markers_pub = self.create_publisher(
            MarkerArray, self.get_parameter('output_markers_topic').value, 1)
        self.subscription = self.create_subscription(
            String, self.input_topic, self.block_b_callback, 10)

        self.get_logger().info(
            f'Block D ready: {self.input_topic} -> '
            f'{self.get_parameter("current_grid_topic").value} + '
            f'{self.get_parameter("prediction_layers_topic").value}, '
            f'expected_frame={self.expected_frame}')

    def block_b_callback(self, message: String) -> None:
        """Parse one Block-B JSON message, then publish its grounded field."""
        try:
            state = json.loads(message.data)
            frame_id = self.validate_block_b(state)
            zones = self.build_constraint_zones(state.get('people', []))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            self.get_logger().warn(f'Ignoring invalid Block-B state: {error}')
            return

        stamp_ns = int(state.get('header', {}).get('stamp_ns', 0))
        current_zones = [zone for zone in zones if not zone.is_prediction]
        prediction_zones = [zone for zone in zones if zone.is_prediction]
        self.current_grid_pub.publish(self.make_grid(frame_id, stamp_ns, current_zones))
        # Retained as a convenient visual safety envelope. It is not a single
        # time slice and therefore is not the field from Figure 1.
        self.combined_grid_pub.publish(self.make_grid(frame_id, stamp_ns, zones))
        self.prediction_layers_pub.publish(
            self.make_prediction_layers(frame_id, stamp_ns, prediction_zones))
        self.markers_pub.publish(self.make_markers(frame_id, stamp_ns, zones))

    def validate_block_b(self, state: dict) -> str:
        """Validate the small stable contract D needs from B."""
        header = state.get('header')
        if not isinstance(header, dict):
            raise ValueError('missing header')
        frame_id = header.get('frame_id')
        if frame_id != self.expected_frame:
            raise ValueError(
                f'expected frame {self.expected_frame!r}, received {frame_id!r}; '
                'transform in Block B before sending to D')
        people = state.get('people')
        if not isinstance(people, list):
            raise ValueError('people must be a list')
        return frame_id

    def build_constraint_zones(self, people: Iterable[dict]) -> List[ConstraintZone]:
        """Convert all valid B people into current and predicted zones."""
        zones: List[ConstraintZone] = []
        for person in people:
            try:
                zones.extend(self.make_zones_for_person(person))
            except (TypeError, ValueError, KeyError) as error:
                self.get_logger().warn(f'Skipping malformed person: {error}')
        return zones

    # ------------------------------------------------------------------
    # POLICY HOOK 1: replace this method with the team's D formulation.
    # Inputs are map-frame position/velocity/body yaw and B's trajectory.
    # Output stays a list of ConstraintZone, so grid/RViz need no rewiring.
    # ------------------------------------------------------------------
    def make_zones_for_person(self, person: dict) -> List[ConstraintZone]:
        track_id = int(person['track_id'])
        position = person['position_m']
        velocity = person.get('velocity_mps', {})
        x_m = float(position['x'])
        y_m = float(position['y'])
        vx_mps = float(velocity.get('vx', 0.0))
        vy_mps = float(velocity.get('vy', 0.0))
        # Proxemics follows body direction when keypoints support it. Movement
        # heading is the fallback. No reliable orientation means a circle.
        if bool(person.get('body_orientation_valid', False)):
            yaw_rad = float(person.get('body_orientation_rad', 0.0))
            oriented = True
        elif bool(person.get('motion_heading_valid', False)):
            yaw_rad = float(person.get('motion_heading_rad', math.atan2(vy_mps, vx_mps)))
            oriented = True
        else:
            yaw_rad = 0.0
            oriented = False

        zones = [self.make_baseline_zone(
            track_id, 0.0, x_m, y_m, yaw_rad, oriented, False)]

        prediction = person.get('prediction', {})
        trajectory = prediction.get('trajectory_m', []) if isinstance(prediction, dict) else []
        for point in trajectory:
            if not isinstance(point, dict):
                continue
            dt_s = max(0.0, float(point['dt_s']))
            zones.append(self.make_baseline_zone(
                track_id, dt_s, float(point['x']), float(point['y']), yaw_rad,
                oriented, True))
        return zones

    def make_baseline_zone(
            self, track_id: int, dt_s: float, x_m: float, y_m: float,
            yaw_rad: float, oriented: bool,
            is_prediction: bool) -> ConstraintZone:
        """Create a fixed-size social Gaussian; prediction does not expand it."""
        if not oriented:
            radius = float(self.get_parameter('stationary_radius_m').value)
            return ConstraintZone(track_id, dt_s, x_m, y_m, yaw_rad,
                                  radius, radius, radius, radius, is_prediction)

        forward = float(self.get_parameter('front_semi_axis_m').value)
        rear = float(self.get_parameter('rear_semi_axis_m').value)
        left = float(self.get_parameter('left_semi_axis_m').value)
        right = float(self.get_parameter('right_semi_axis_m').value)
        return ConstraintZone(track_id, dt_s, x_m, y_m, yaw_rad,
                              forward, rear, left, right, is_prediction)

    # ------------------------------------------------------------------
    # POLICY HOOK 2: replace this evaluator for a different K_soc(x, y, t).
    # It receives a cell and one time-indexed social zone and returns [0, 100].
    # ------------------------------------------------------------------
    def zone_cost(self, x_m: float, y_m: float, zone: ConstraintZone) -> int:
        dx = x_m - zone.x_m
        dy = y_m - zone.y_m
        forward = math.cos(zone.yaw_rad) * dx + math.sin(zone.yaw_rad) * dy
        lateral = -math.sin(zone.yaw_rad) * dx + math.cos(zone.yaw_rad) * dy
        forward_axis = zone.front_semi_axis_m if forward >= 0.0 else zone.rear_semi_axis_m
        lateral_axis = zone.left_semi_axis_m if lateral >= 0.0 else zone.right_semi_axis_m
        normalized_distance_sq = (
            (forward / forward_axis) ** 2 +
            (lateral / lateral_axis) ** 2)

        # Baseline smooth ellipse. Prediction reduces confidence with dt;
        # replace this line when defining temporal aggregation for K_soc.
        temporal_weight = math.exp(
            -float(self.get_parameter('prediction_cost_decay_per_s').value) * zone.dt_s)
        return int(round(self.maximum_cost * temporal_weight *
                         math.exp(-0.5 * normalized_distance_sq)))

    def make_grid(self, frame_id: str, stamp_ns: int,
                  zones: List[ConstraintZone]) -> OccupancyGrid:
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
                if self.combine_mode == 'sum_clamped':
                    cost = min(self.maximum_cost, sum(costs))
                else:
                    cost = max(costs)
                if cost >= self.minimum_published_cost:
                    grid.data[row * self.grid_width_cells + col] = cost
        return grid

    def make_prediction_layers(
            self, frame_id: str, stamp_ns: int,
            prediction_zones: List[ConstraintZone]) -> SocialConstraintLayers:
        """Keep one K_soc grid per dt_s instead of mixing time slices."""
        layers = SocialConstraintLayers()
        layers.header.frame_id = frame_id
        layers.header.stamp.sec = stamp_ns // 1_000_000_000
        layers.header.stamp.nanosec = stamp_ns % 1_000_000_000
        zones_by_dt: Dict[float, List[ConstraintZone]] = {}
        for zone in prediction_zones:
            # Block B normally has fixed 0.5-s samples. Rounding avoids an
            # accidental separate layer due to floating point representation.
            zones_by_dt.setdefault(round(zone.dt_s, 6), []).append(zone)
        for dt_s in sorted(zones_by_dt):
            layer = SocialConstraintLayer()
            layer.dt_s = dt_s
            future_stamp_ns = stamp_ns + int(round(dt_s * 1_000_000_000))
            layer.grid = self.make_grid(
                frame_id, future_stamp_ns, zones_by_dt[dt_s])
            layers.layers.append(layer)
        return layers

    def make_markers(self, frame_id: str, stamp_ns: int,
                     zones: List[ConstraintZone]) -> MarkerArray:
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for marker_id, zone in enumerate(zones):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp.sec = stamp_ns // 1_000_000_000
            marker.header.stamp.nanosec = stamp_ns % 1_000_000_000
            marker.ns = 'social_constraint_predictions' if zone.is_prediction else 'social_constraint_current'
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.035
            marker.color.a = 0.85
            marker.color.r = 1.0
            marker.color.g = 0.55 if zone.is_prediction else 0.05
            marker.color.b = 0.0
            marker.lifetime.sec = int(float(self.get_parameter('marker_lifetime_s').value))
            marker.lifetime.nanosec = int(
                (float(self.get_parameter('marker_lifetime_s').value) % 1.0) * 1_000_000_000)

            # Local ellipse -> map points. The final point closes the loop.
            for index in range(37):
                angle = 2.0 * math.pi * index / 36.0
                forward_axis = (zone.front_semi_axis_m if math.cos(angle) >= 0.0
                                else zone.rear_semi_axis_m)
                forward = forward_axis * math.cos(angle)
                lateral_axis = (zone.left_semi_axis_m if math.sin(angle) >= 0.0
                                else zone.right_semi_axis_m)
                lateral = lateral_axis * math.sin(angle)
                point = Point()
                point.x = zone.x_m + math.cos(zone.yaw_rad) * forward - math.sin(zone.yaw_rad) * lateral
                point.y = zone.y_m + math.sin(zone.yaw_rad) * forward + math.cos(zone.yaw_rad) * lateral
                point.z = 0.05 + 0.02 * zone.dt_s
                marker.points.append(point)
            markers.markers.append(marker)
        return markers


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
