#!/usr/bin/env python3
"""Apply a non-oscillating last-line safety limit around tracked people."""

import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.node import Node
from social_perception.msg import People
from tf2_ros import Buffer, TransformException, TransformListener


@dataclass(frozen=True)
class RelativePerson:
    """A tracked person expressed in the robot command frame."""

    x: float
    y: float
    vx: float
    vy: float


class SocialVelocityFilter(Node):
    def __init__(self):
        super().__init__('social_velocity_filter')
        self.declare_parameter('slow_distance', 2.0)
        self.declare_parameter('stop_distance', 0.8)
        self.declare_parameter('resume_distance', 1.0)
        self.declare_parameter('minimum_speed_scale', 0.2)
        self.declare_parameter('direction_margin', 0.2)
        self.declare_parameter('collision_horizon', 2.0)
        self.declare_parameter('collision_clearance', 0.65)
        self.declare_parameter('predicted_stop_clearance', 0.48)
        self.declare_parameter('people_timeout', 0.5)
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('transform_timeout', 0.05)
        self.declare_parameter('escape_evaluation_time', 0.6)
        self.declare_parameter('escape_minimum_progress', 0.0)
        self.declare_parameter('escape_maximum_speed', 0.12)
        self.declare_parameter('input_command_timeout', 0.35)
        self.declare_parameter('watchdog_rate', 10.0)
        self.declare_parameter('stop_broadcast_time', 1.0)
        # Wiring is a deployment choice, not a tuning choice. Gazebo's diff-drive
        # plugin listens on /cmd_vel_safe, while micro-ROS firmware on the real
        # base listens on the fixed topic /cmd_vel. Only these two names differ
        # between the two targets.
        self.declare_parameter('people_topic', '/people')
        self.declare_parameter('input_topic', '/cmd_vel')
        self.declare_parameter('output_topic', '/cmd_vel_safe')
        self.slow_distance = float(self.get_parameter('slow_distance').value)
        self.stop_distance = float(self.get_parameter('stop_distance').value)
        self.resume_distance = max(
            self.stop_distance,
            float(self.get_parameter('resume_distance').value))
        self.minimum_scale = float(self.get_parameter('minimum_speed_scale').value)
        self.direction_margin = float(self.get_parameter('direction_margin').value)
        self.collision_horizon = max(
            0.0, float(self.get_parameter('collision_horizon').value))
        self.collision_clearance = max(
            0.0, float(self.get_parameter('collision_clearance').value))
        self.predicted_stop_clearance = max(
            0.0, float(self.get_parameter('predicted_stop_clearance').value))
        self.people_timeout = max(
            0.0, float(self.get_parameter('people_timeout').value))
        self.robot_frame = str(self.get_parameter('robot_frame').value)
        self.transform_timeout = max(
            0.0, float(self.get_parameter('transform_timeout').value))
        self.escape_evaluation_time = max(
            0.05, float(self.get_parameter('escape_evaluation_time').value))
        self.escape_minimum_progress = max(
            0.0, float(self.get_parameter('escape_minimum_progress').value))
        self.escape_maximum_speed = max(
            0.0, float(self.get_parameter('escape_maximum_speed').value))
        self.input_command_timeout = max(
            0.0, float(self.get_parameter('input_command_timeout').value))
        self.stop_broadcast_time = max(
            0.0, float(self.get_parameter('stop_broadcast_time').value))
        watchdog_rate = float(self.get_parameter('watchdog_rate').value)
        if watchdog_rate <= 0.0:
            raise ValueError('watchdog_rate must be greater than zero')
        self.people = []
        self.people_received_at = None
        self.command_received_at = None
        self.emergency_stop = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        people_topic = str(self.get_parameter('people_topic').value)
        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        if input_topic == output_topic:
            raise ValueError(
                f'input_topic and output_topic are both "{input_topic}". The '
                'filter would consume its own output and feed back forever.')
        self.people_sub = self.create_subscription(
            People, people_topic, self.people_callback, 10)
        self.cmd_sub = self.create_subscription(
            Twist, input_topic, self.cmd_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, output_topic, 10)
        self.create_timer(1.0 / watchdog_rate, self.command_watchdog)
        # Printed so a wrong deployment profile is visible immediately instead
        # of showing up as a robot that silently refuses to move.
        self.get_logger().info(
            f'Velocity filter active: {input_topic} -> {output_topic} '
            f'(people on {people_topic}, stop at {self.stop_distance:.2f} m)')

    def people_callback(self, message):
        source_frame = message.header.frame_id
        if not source_frame:
            self.get_logger().warn(
                'Ignoring /people message with an empty frame_id',
                throttle_duration_sec=2.0)
            return
        try:
            if source_frame == self.robot_frame:
                transform = None
            else:
                # The perception node publishes people in map, while cmd_vel is
                # expressed in base_link. Use the latest transform so distance
                # and "moving away" are evaluated around the robot, not around
                # the map origin.
                transform = self.tf_buffer.lookup_transform(
                    self.robot_frame, source_frame, rclpy.time.Time(),
                    timeout=Duration(seconds=self.transform_timeout))
        except TransformException as error:
            self.get_logger().warn(
                f'Cannot transform people from {source_frame} to '
                f'{self.robot_frame}: {error}', throttle_duration_sec=2.0)
            return

        self.people = [
            self.relative_person(person, transform) for person in message.people]
        self.people_received_at = self.get_clock().now()

    @staticmethod
    def relative_person(person, transform):
        if transform is None:
            return RelativePerson(
                person.pose.position.x, person.pose.position.y,
                person.velocity.linear.x, person.velocity.linear.y)

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z))
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        x = person.pose.position.x
        y = person.pose.position.y
        vx = person.velocity.linear.x
        vy = person.velocity.linear.y
        return RelativePerson(
            cosine * x - sine * y + translation.x,
            sine * x + cosine * y + translation.y,
            cosine * vx - sine * vy,
            sine * vx + cosine * vy)

    def current_people(self):
        """Never act forever on a tracker frame that has stopped updating."""
        if self.people_received_at is None:
            return []
        age = (self.get_clock().now() - self.people_received_at).nanoseconds / 1e9
        return self.people if age <= self.people_timeout else []

    @staticmethod
    def predicted_approach(person, command, horizon):
        """Return closest separation and its time over the command horizon."""
        x = person.x
        y = person.y
        relative_vx = person.vx - command.linear.x
        relative_vy = person.vy - command.linear.y
        speed_squared = relative_vx * relative_vx + relative_vy * relative_vy
        if speed_squared < 1e-6:
            return math.hypot(x, y), 0.0
        closest_time = -(x * relative_vx + y * relative_vy) / speed_squared
        closest_time = max(0.0, min(horizon, closest_time))
        return (math.hypot(
            x + relative_vx * closest_time,
            y + relative_vy * closest_time), closest_time)

    def is_escape_command(self, command, people):
        """Allow only translation that opens clearance to every close person.

        Rotation remains available separately, so a differential-drive robot
        can first turn toward the open side. A translation is admitted only if
        the predicted separation grows for all people inside the hysteresis
        radius. This prevents the escape exception from becoming permission to
        squeeze past one member of a conversation while avoiding the other.
        """
        if math.hypot(command.linear.x, command.linear.y) < 1e-4:
            return False

        checked_someone = False
        horizon = self.escape_evaluation_time
        for person in people:
            distance = math.hypot(person.x, person.y)
            if distance >= self.resume_distance:
                continue
            checked_someone = True
            future_x = person.x + (person.vx - command.linear.x) * horizon
            future_y = person.y + (person.vy - command.linear.y) * horizon
            future_distance = math.hypot(future_x, future_y)
            if future_distance < distance + self.escape_minimum_progress:
                return False
        return checked_someone

    def limited_escape(self, command):
        output = Twist()
        speed = math.hypot(command.linear.x, command.linear.y)
        scale = min(1.0, self.escape_maximum_speed / max(speed, 1e-6))
        output.linear.x = command.linear.x * scale
        output.linear.y = command.linear.y * scale
        output.linear.z = command.linear.z * scale
        output.angular.x = command.angular.x
        output.angular.y = command.angular.y
        output.angular.z = command.angular.z
        return output

    def cmd_callback(self, command):
        self.command_received_at = self.get_clock().now()
        people = self.current_people()
        nearest_distance = math.inf
        predicted_collision = False
        for person in people:
            distance = math.hypot(person.x, person.y)
            if distance < nearest_distance:
                nearest_distance = distance
            predicted_distance, closest_time = self.predicted_approach(
                person, command, self.collision_horizon)
            if closest_time > 0.0 and predicted_distance <= self.predicted_stop_clearance:
                predicted_collision = True

        # Separate enter/exit thresholds prevent noisy detections on one radius
        # from repeatedly toggling the robot. If a social region appears around
        # an already-overlapping robot, keep the stop latched but admit a slow
        # command that demonstrably increases every close-person clearance.
        if self.emergency_stop:
            self.emergency_stop = (
                nearest_distance < self.resume_distance or predicted_collision)
        else:
            self.emergency_stop = (
                nearest_distance <= self.stop_distance or predicted_collision)
        if self.emergency_stop:
            if not predicted_collision and self.is_escape_command(command, people):
                self.cmd_pub.publish(self.limited_escape(command))
            else:
                # Keep rotation so Nav2 can turn toward an escape direction.
                rotation_only = Twist()
                rotation_only.angular.x = command.angular.x
                rotation_only.angular.y = command.angular.y
                rotation_only.angular.z = command.angular.z
                self.cmd_pub.publish(rotation_only)
            return

        scale = 1.0
        direction = 1.0 if command.linear.x >= 0.0 else -1.0

        for person in people:
            x = person.x
            y = person.y
            distance = math.hypot(x, y)

            # Only limit translation for people lying in the commanded travel
            # direction. Once they pass behind the robot, motion resumes.
            if direction * x < -self.direction_margin or distance >= self.slow_distance:
                continue
            predicted_distance, _ = self.predicted_approach(
                person, command, self.collision_horizon)
            if predicted_distance >= self.collision_clearance:
                continue
            ratio = (distance - self.stop_distance) / max(
                1e-6,
                self.slow_distance - self.stop_distance)
            scale = min(scale, max(self.minimum_scale, min(1.0, ratio)))

        output = Twist()
        output.linear.x = command.linear.x * scale
        output.linear.y = command.linear.y * scale
        output.linear.z = command.linear.z * scale
        # Keep rotation available while translation is stopped so Nav2 can turn
        # toward a valid detour instead of becoming completely immobilized.
        output.angular.x = command.angular.x
        output.angular.y = command.angular.y
        output.angular.z = command.angular.z
        self.cmd_pub.publish(output)

    def command_watchdog(self):
        """Stop the base if the upstream mux/controller disappears, then let go.

        The zero burst is bounded on purpose. Republishing zero forever keeps
        this node an active publisher on output_topic for the rest of the
        session, so an upstream that comes and goes -- rl_agent started once
        and closed -- silently overwrites whatever writes that topic next, with
        no error printed on either side. stop_broadcast_time is long enough to
        stop the wheels and no longer; they stay stopped afterwards because the
        last command they received was zero.
        """
        if self.command_received_at is None:
            return
        age = (self.get_clock().now() - self.command_received_at).nanoseconds / 1e9
        if 0.0 <= age <= self.input_command_timeout:
            return
        if age > self.input_command_timeout + self.stop_broadcast_time:
            return
        self.cmd_pub.publish(Twist())
        self.get_logger().warn(
            'Velocity input timed out; publishing zero velocity',
            throttle_duration_sec=2.0)

    def stop(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = SocialVelocityFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl+C and a launch shutdown both reach here with the context already
        # torn down, and publishing then raises out of the handler instead of
        # stopping the wheels. The zero that matters was already sent by the
        # watchdog burst; this is the clean-exit case only.
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
