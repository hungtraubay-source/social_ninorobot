#!/usr/bin/env python3
"""Apply a non-oscillating last-line safety limit around tracked people."""

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from social_perception.msg import People


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
        self.people = []
        self.people_received_at = None
        self.emergency_stop = False
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
        # Printed so a wrong deployment profile is visible immediately instead
        # of showing up as a robot that silently refuses to move.
        self.get_logger().info(
            f'Velocity filter active: {input_topic} -> {output_topic} '
            f'(people on {people_topic}, stop at {self.stop_distance:.2f} m)')

    def people_callback(self, message):
        self.people = list(message.people)
        self.people_received_at = self.get_clock().now()

    def current_people(self):
        """Never act forever on a tracker frame that has stopped updating."""
        if self.people_received_at is None:
            return []
        age = (self.get_clock().now() - self.people_received_at).nanoseconds / 1e9
        return self.people if age <= self.people_timeout else []

    @staticmethod
    def predicted_approach(person, command, horizon):
        """Return closest separation and its time over the command horizon."""
        x = person.pose.position.x
        y = person.pose.position.y
        relative_vx = person.velocity.linear.x - command.linear.x
        relative_vy = person.velocity.linear.y - command.linear.y
        speed_squared = relative_vx * relative_vx + relative_vy * relative_vy
        if speed_squared < 1e-6:
            return math.hypot(x, y), 0.0
        closest_time = -(x * relative_vx + y * relative_vy) / speed_squared
        closest_time = max(0.0, min(horizon, closest_time))
        return (math.hypot(
            x + relative_vx * closest_time,
            y + relative_vy * closest_time), closest_time)

    def cmd_callback(self, command):
        people = self.current_people()
        nearest_distance = math.inf
        predicted_collision = False
        for person in people:
            distance = math.hypot(person.pose.position.x, person.pose.position.y)
            if distance < nearest_distance:
                nearest_distance = distance
            predicted_distance, closest_time = self.predicted_approach(
                person, command, self.collision_horizon)
            if closest_time > 0.0 and predicted_distance <= self.predicted_stop_clearance:
                predicted_collision = True

        # Separate enter/exit thresholds prevent noisy detections on one radius
        # from repeatedly toggling the robot. A full stop is deterministic and
        # lets Nav2's 2 Hz replanner choose the detour; this node must not invent
        # a turn direction, which previously made the robot circle a person.
        if self.emergency_stop:
            self.emergency_stop = (
                nearest_distance < self.resume_distance or predicted_collision)
        else:
            self.emergency_stop = (
                nearest_distance <= self.stop_distance or predicted_collision)
        if self.emergency_stop:
            self.cmd_pub.publish(Twist())
            return

        scale = 1.0
        direction = 1.0 if command.linear.x >= 0.0 else -1.0

        for person in people:
            x = person.pose.position.x
            y = person.pose.position.y
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


def main(args=None):
    rclpy.init(args=args)
    node = SocialVelocityFilter()
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
