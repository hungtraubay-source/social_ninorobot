"""Visualize the talking group zone that the VLM-to-RL path grounds.

The node deliberately consumes the same ``People`` and ``VlmPersonStates``
messages as the deployed RL bridge.  It does not control the robot: markers
are debug output only, while the agent continues to calculate its own field.
"""

import math
import os

import rclpy
from rclpy.node import Node
from social_perception.msg import People, VlmPersonStates
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from social_rl.constraint_field import ConstraintFieldConfig, compile_zones
from social_rl.observation import ObservationConfig, RelativeEntity
from social_rl.ros_interface import EnvConfig
from social_rl.semantic_fusion import VlmStateCache


OUTER_CONTOUR_LEVEL = 0.15
ELLIPSE_SEGMENTS = 72


class ZoneMarkerHold:
    """Keep the newest VLM-confirmed talking zone visible for debug only.

    The hold is intentionally local to RViz output. It never changes the
    People/VLM stream, RL observation, social costmap, or robot motion.
    """

    def __init__(self, hold_seconds: float) -> None:
        self._hold_ns = int(max(0.0, hold_seconds) * 1_000_000_000)
        self._expires_ns = -1
        self._message = None
        self._zones = ()

    def update(self, message, zones, now_ns: int) -> None:
        """Refresh the retained scene only when VLM currently confirms a zone."""
        if not zones:
            return
        self._message = message
        self._zones = tuple(zones)
        self._expires_ns = now_ns + self._hold_ns

    def active(self, now_ns: int):
        """Return the retained source scene until its debug hold expires."""
        if self._message is not None and now_ns <= self._expires_ns:
            return self._message, self._zones
        self._message = None
        self._zones = ()
        return None, ()


def quaternion_to_yaw(quaternion) -> float:
    """Return planar yaw in radians from a geometry_msgs quaternion."""
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2))


def resolve_env_config_path(model_path: str, explicit_path: str) -> str:
    """Use the same checkpoint-adjacent config lookup as ``rl_agent``."""
    if explicit_path:
        return os.path.expanduser(explicit_path)
    expanded_model = os.path.expanduser(model_path)
    direct = os.path.join(os.path.dirname(expanded_model), 'env_config.yaml')
    if os.path.exists(direct):
        return direct
    return os.path.join(
        os.path.dirname(os.path.dirname(expanded_model)), 'env_config.yaml')


def talking_group_zones(people_message, states: VlmStateCache,
                        field_config: ConstraintFieldConfig):
    """Build only confirmed talking group zones in the People message frame."""
    entities = []
    for person in people_message.people:
        state, confidence = states.lookup(person.id, people_message.header)
        if state != 'talking':
            continue
        entities.append(RelativeEntity(
            x=person.pose.position.x,
            y=person.pose.position.y,
            vx=person.velocity.linear.x,
            vy=person.velocity.linear.y,
            facing=quaternion_to_yaw(person.pose.orientation),
            scene_type=state,
            track_id=person.id,
            scene_confidence=confidence))

    field = compile_zones(
        entities, field_config,
        frame=people_message.header.frame_id or 'odom')
    return [zone for zone in field.zones
            if zone.scene_type == 'talking' and zone.hardness == 'hard']


class VlmZoneVisualizer(Node):
    """Publish RViz markers for VLM-confirmed two-person talking zones."""

    def __init__(self) -> None:
        super().__init__('vlm_zone_visualizer')
        self.declare_parameter('model_path', '')
        self.declare_parameter('env_config', '')
        self.declare_parameter('people_topic_override', '')
        self.declare_parameter('output_markers_topic', '/social_rl/zone_markers')
        # Seconds a VLM-confirmed zone remains in RViz after its label vanishes.
        # This affects visualization only; 0 disables the hold.
        self.declare_parameter('marker_lifetime_s', 3.0)

        model_path = str(self.get_parameter('model_path').value)
        config_path = resolve_env_config_path(
            model_path, str(self.get_parameter('env_config').value))
        if not os.path.exists(config_path):
            raise RuntimeError(
                f'no env_config.yaml found for VLM zone visualizer: {config_path!r}')
        with open(config_path, encoding='utf-8') as config_file:
            saved = yaml.safe_load(config_file) or {}
        env_config = EnvConfig.from_dict(saved.get('env', {}))
        override = str(self.get_parameter('people_topic_override').value)
        if override:
            env_config.people_topic = override
        self.field_config = ObservationConfig.from_dict(
            saved.get('observation', {})).constraint_field
        self.states = VlmStateCache(env_config.vlm_state_timeout)
        self.marker_lifetime_s = max(
            0.0, float(self.get_parameter('marker_lifetime_s').value))
        self.zone_marker_hold = ZoneMarkerHold(self.marker_lifetime_s)

        self.marker_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter('output_markers_topic').value),
            10)
        self.create_subscription(
            VlmPersonStates, env_config.vlm_states_topic,
            self.states.update, 10)
        self.create_subscription(People, env_config.people_topic,
                                 self.people_callback, 10)
        self.get_logger().info(
            f'VLM talking-zone visualizer: people={env_config.people_topic}, '
            f'states={env_config.vlm_states_topic}, '
            f'markers={self.get_parameter("output_markers_topic").value}')

    def people_callback(self, message: People) -> None:
        """Draw confirmed zones, retaining the newest one briefly for RViz."""
        zones = talking_group_zones(message, self.states, self.field_config)
        now_ns = self.get_clock().now().nanoseconds
        self.zone_marker_hold.update(message, zones, now_ns)
        marker_message, marker_zones = self.zone_marker_hold.active(now_ns)
        output = MarkerArray()
        # Clearing then re-adding retained markers prevents departed pairs from
        # lingering beyond the configured visualization-only hold.
        clear = Marker()
        clear.action = Marker.DELETEALL
        output.markers.append(clear)
        if marker_message is not None:
            for index, zone in enumerate(marker_zones):
                output.markers.extend(self.zone_markers(marker_message, zone, index))
        self.marker_pub.publish(output)

    def zone_markers(self, header, zone, index: int):
        """Render a 15%-peak Gaussian contour and a concise group label."""
        sample = zone.trajectory_of_zone[0]
        sigma_front, sigma_side, sigma_rear = sample.size
        contour_scale = math.sqrt(-2.0 * math.log(OUTER_CONTOUR_LEVEL))
        cosine = math.cos(sample.orientation)
        sine = math.sin(sample.orientation)

        outline = Marker()
        outline.header = header.header
        outline.ns = 'vlm_talking_social_zone'
        outline.id = 2 * index
        outline.type = Marker.LINE_STRIP
        outline.action = Marker.ADD
        outline.scale.x = 0.045
        outline.color.r = 0.95
        outline.color.g = 0.10
        outline.color.b = 0.15
        outline.color.a = 0.95
        self.set_lifetime(outline)
        for segment in range(ELLIPSE_SEGMENTS + 1):
            angle = 2.0 * math.pi * segment / ELLIPSE_SEGMENTS
            forward_sigma = sigma_front if math.cos(angle) >= 0.0 else sigma_rear
            forward = contour_scale * forward_sigma * math.cos(angle)
            lateral = contour_scale * sigma_side * math.sin(angle)
            point = self.make_point(
                sample.center[0] + cosine * forward - sine * lateral,
                sample.center[1] + sine * forward + cosine * lateral)
            outline.points.append(point)

        label = Marker()
        label.header = header.header
        label.ns = 'vlm_talking_social_zone_label'
        label.id = 2 * index + 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position = self.make_point(sample.center[0], sample.center[1], 0.25)
        label.scale.z = 0.22
        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        label.color.a = 1.0
        label.text = f'TALKING — hard zone\n{", ".join(zone.track_ids)}'
        self.set_lifetime(label)
        return outline, label

    def set_lifetime(self, marker: Marker) -> None:
        seconds = int(self.marker_lifetime_s)
        marker.lifetime.sec = seconds
        marker.lifetime.nanosec = int(
            (self.marker_lifetime_s - seconds) * 1_000_000_000)

    @staticmethod
    def make_point(x: float, y: float, z: float = 0.0):
        from geometry_msgs.msg import Point

        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)
        return point


def main(args=None):
    rclpy.init(args=args)
    node = VlmZoneVisualizer()
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
