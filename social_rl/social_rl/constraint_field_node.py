"""Continuously ground tracker + VLM output into the deploy ConstraintField."""

from dataclasses import replace
import math
import os

from builtin_interfaces.msg import Time as TimeMsg
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray
import yaml

from social_rl.field_markers import FieldMarkerRenderer
from social_rl.field_transport import field_to_json
from social_rl.observation import ObservationConfig
from social_rl.ros_interface import (EnvConfig, PerceptionBridge,
                                     inverse_transform_point,
                                     quaternion_to_yaw, transform_point)


LOCK_DURATION_SECONDS = 8.0


def _config_path(model_path: str, configured_path: str) -> str:
    path = os.path.expanduser(configured_path)
    if path:
        return path
    path = os.path.join(os.path.dirname(model_path), 'env_config.yaml')
    if os.path.exists(path):
        return path
    return os.path.join(
        os.path.dirname(os.path.dirname(model_path)), 'env_config.yaml')


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _transform_field(field, transform, target_frame: str, *, inverse: bool):
    """Move zone geometry between the robot and fixed odometry frames."""
    yaw = quaternion_to_yaw(transform.rotation)
    angle_delta = -yaw if inverse else yaw
    zones = []
    for zone in field.zones:
        samples = []
        for sample in zone.trajectory_of_zone:
            center = (inverse_transform_point(transform, *sample.center)
                      if inverse else
                      transform_point(transform, *sample.center))
            samples.append(replace(
                sample,
                center=center,
                orientation=_wrap_angle(sample.orientation + angle_delta)))
        zones.append(replace(zone, trajectory_of_zone=tuple(samples)))
    return replace(field, frame=target_frame, zones=tuple(zones))


class LockedTalkingField:
    """Latch one VLM-grounded talking zone for one eight-second cycle."""

    def __init__(self, goal_frame: str, robot_frame: str):
        self._goal_frame = goal_frame
        self._robot_frame = robot_frame
        self._locked = None
        self._locked_at = None

    @property
    def locked(self) -> bool:
        return self._locked is not None

    def try_lock(self, field, goal_to_robot, now: float) -> bool:
        """Store only the first hard talking zone in the fixed goal frame."""
        if self.locked:
            return False
        talking_zone = next((
            zone for zone in field.zones
            if zone.scene_type == 'talking' and zone.hardness == 'hard'), None)
        if talking_zone is None:
            return False
        talking_field = replace(field, zones=(talking_zone,))
        self._locked = _transform_field(
            talking_field, goal_to_robot, self._goal_frame, inverse=True)
        self._locked_at = float(now)
        return True

    def expire_if_due(self, now: float) -> bool:
        """Clear the locked zone after exactly one eight-second cycle."""
        if not self.locked:
            return False
        elapsed = float(now) - self._locked_at
        if 0.0 <= elapsed < LOCK_DURATION_SECONDS:
            return False
        self._locked = None
        self._locked_at = None
        return True

    def for_robot(self, goal_to_robot):
        """Express the fixed world zone in the moving robot frame for RL."""
        if not self.locked:
            raise RuntimeError('no talking constraint has been locked yet')
        return _transform_field(
            self._locked, goal_to_robot, self._robot_frame, inverse=False)

    def for_visualization(self):
        """Return the same locked zone in its stable odometry frame."""
        if not self.locked:
            raise RuntimeError('no talking constraint has been locked yet')
        return self._locked


class DeploymentConstraintFieldNode(Node):
    """Own the live deploy field independently of goal and policy state."""

    def __init__(self):
        super().__init__('social_rl_constraint_field')
        self.declare_parameter('model_path', '')
        self.declare_parameter('env_config', '')
        self.declare_parameter('people_topic_override', '')
        self.declare_parameter(
            'field_topic', '/social_rl/constraint_field')
        self.declare_parameter('publish_zone_markers', True)
        self.declare_parameter(
            'zone_markers_topic', '/social_rl/zone_markers')

        model_path = os.path.expanduser(
            str(self.get_parameter('model_path').value))
        configured_path = str(self.get_parameter('env_config').value)
        config_path = _config_path(model_path, configured_path)
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError(
                f'model_path "{model_path}" does not exist')
        if not os.path.exists(config_path):
            raise RuntimeError(
                f'no env_config.yaml found for model "{model_path}"')

        with open(config_path, 'r') as handle:
            saved = yaml.safe_load(handle) or {}
        env_config = EnvConfig.from_dict(saved.get('env', {}))
        people_topic_override = str(
            self.get_parameter('people_topic_override').value)
        if people_topic_override:
            env_config.people_topic = people_topic_override
        observation_config = ObservationConfig.from_dict(
            saved.get('observation', {}))

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._bridge = PerceptionBridge(
            self, self._tf_buffer, env_config, observation_config,
            subscribe_motion=False, subscribe_social=True)
        self._field_latch = LockedTalkingField(
            env_config.goal_frame, env_config.robot_frame)
        self._field_pub = self.create_publisher(
            String, str(self.get_parameter('field_topic').value), 10)
        self._marker_pub = None
        if bool(self.get_parameter('publish_zone_markers').value):
            self._marker_pub = self.create_publisher(
                MarkerArray,
                str(self.get_parameter('zone_markers_topic').value), 10)
        self._marker_renderer = FieldMarkerRenderer()
        self.create_timer(env_config.control_period, self._publish_field)
        self.get_logger().info(
            f'each valid talking ConstraintField is locked for '
            f'{LOCK_DURATION_SECONDS:.1f} s: '
            f'{env_config.people_topic} + {env_config.vlm_states_topic} -> '
            f'{self.get_parameter("field_topic").value}')

    def _publish_field(self):
        try:
            now = self.get_clock().now().nanoseconds * 1e-9
            if self._field_latch.expire_if_due(now):
                # Match a fresh node cycle: discard all evidence that existed
                # before reset, so only a later VLM message can create a zone.
                self._bridge.reset_deployment_grounding()
                self.get_logger().info(
                    'talking constraint expired after 8.0 s; waiting for a '
                    'new VLM result')
            if not self._field_latch.locked:
                candidate = self._bridge.deployment_constraint_field()
                goal_to_robot = self._bridge.goal_transform()
                if not self._field_latch.try_lock(
                        candidate, goal_to_robot, now):
                    # VLM has not produced a hard talking constraint yet.
                    field = replace(candidate, zones=())
                else:
                    self.get_logger().info(
                        'locked new talking constraint for 8.0 s; tracker/VLM '
                        'updates are ignored during this cycle')
                    field = self._field_latch.for_robot(goal_to_robot)
            else:
                # Only TF remains live: the zone stays fixed in odom while the
                # robot-frame coordinates change as the robot moves.
                field = self._field_latch.for_robot(
                    self._bridge.goal_transform())
        except RuntimeError as error:
            self.get_logger().warn(str(error), throttle_duration_sec=5.0)
            return

        message = String()
        message.data = field_to_json(field)
        self._field_pub.publish(message)
        if self._marker_pub is not None:
            marker_field = (
                self._field_latch.for_visualization()
                if self._field_latch.locked else field)
            # The locked geometry is already fixed in odom. A zero stamp asks
            # RViz for the latest map -> odom transform instead of requiring
            # an exact transform at this timer tick, which avoids extrapolation
            # drops while leaving the RL field and policy input unchanged.
            markers = self._marker_renderer.render(
                marker_field, TimeMsg())
            self._marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = DeploymentConstraintFieldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
