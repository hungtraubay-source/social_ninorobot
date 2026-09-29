#!/usr/bin/env python3
"""Verify that the talking pair's *default* lateral offset is zero.

This is deliberately not an ``offset 0 0`` test: that token is an override.
For each of two clearly different world-frame routes, the probe sends only
``talking route ...``, reads the ground-truth pair, and projects its centre
onto that route.  A pass means the plugin accepted each route and its default
offset is zero (within 2 cm of pose/publication noise).

Run after rebuilding and fully restarting Gazebo::

    source ~/social_ninorobot/install/setup.bash
    python3 ~/social_ninorobot/social_rl/tools/check_talking_offset_zero.py
"""

import math

import rclpy
from rclpy.node import Node
from social_perception.msg import People
from std_msgs.msg import String


# World-frame routes. B differs greatly from the plugin's historical default.
ROUTES = (
    ('A', (-3.0, -2.0, 1.2, 0.3)),
    ('B', (-3.0, 2.0, 1.0, -2.5)),
)
SAMPLES_PER_ROUTE = 6
MAX_LATERAL_ERROR = 0.02  # metres
RATIO_RANGE = (0.44, 0.81)  # SpawnTalkingPeople samples [0.45, 0.80]


def route_error(route, message):
    """Return signed lateral error and along-route ratio of the pair centre."""
    x0, y0, x1, y1 = route
    people = message.people
    centre_x = sum(person.pose.position.x for person in people) / 2.0
    centre_y = sum(person.pose.position.y for person in people) / 2.0
    dx, dy = x1 - x0, y1 - y0
    length_squared = dx * dx + dy * dy
    ratio = ((centre_x - x0) * dx + (centre_y - y0) * dy) / length_squared
    # Dot with the route's unit left normal.
    lateral = (-(centre_x - x0) * dy + (centre_y - y0) * dx) / math.sqrt(
        length_squared)
    return lateral, ratio


class Probe(Node):
    def __init__(self):
        super().__init__('talking_offset_zero_probe')
        self.publisher = self.create_publisher(
            String, '/animated_people/scenario', 10)
        self.create_subscription(People, '/social_gt/people', self._on_people, 10)
        self.message = None
        self.pending = False

    def _on_people(self, message):
        if self.pending and len(message.people) == 2:
            self.message = message
            self.pending = False

    def draw(self, route):
        self.publisher.publish(String(data='none'))
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.05)
        self.message = None
        self.pending = True
        # Intentionally no `offset` token: this tests the plugin default.
        self.publisher.publish(String(
            data='talking route {:.3f} {:.3f} {:.3f} {:.3f}'.format(*route)))
        for _ in range(80):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.message is not None:
                return self.message
        self.pending = False
        return None


def main():
    rclpy.init()
    node = Probe()
    passed = True
    try:
        for name, route in ROUTES:
            readings = []
            for _ in range(SAMPLES_PER_ROUTE):
                message = node.draw(route)
                if message is None:
                    continue
                readings.append(route_error(route, message))
            if not readings:
                print(f'Route {name}: FAIL — no talking pair received')
                passed = False
                continue
            lateral = [value[0] for value in readings]
            ratios = [value[1] for value in readings]
            in_line = all(abs(value) <= MAX_LATERAL_ERROR for value in lateral)
            in_segment = all(RATIO_RANGE[0] <= value <= RATIO_RANGE[1]
                             for value in ratios)
            print(f'Route {name}: lateral {min(lateral):+.3f}..{max(lateral):+.3f} m; '
                  f'ratio {min(ratios):.3f}..{max(ratios):.3f}; '
                  f'{len(readings)}/{SAMPLES_PER_ROUTE} samples')
            passed = passed and in_line and in_segment
        print('\nPASS: default talking offset = 0.0 m and route is received'
              if passed else
              '\nFAIL: restart Gazebo after build; if it still fails, do not train')
    finally:
        node.publisher.publish(String(data='none'))
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
