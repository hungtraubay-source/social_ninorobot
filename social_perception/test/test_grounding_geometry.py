"""Regression checks for current talking/crossing geometry, without a ROS graph.

Run after sourcing ROS: python3 -m unittest discover -s social_perception/test.
Use actual ROS messages and production methods, but capture publishers locally
so synthetic people never enter the running Gazebo/perception pipeline.
"""

import importlib.util
import json
import math
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest

from std_msgs.msg import String
from visualization_msgs.msg import Marker


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'social_constraint_grounding.py'
SPEC = importlib.util.spec_from_file_location('grounding_under_test', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CapturedPublisher:
    """Capture the last ROS message without DDS or a running ROS node."""

    def publish(self, message):
        self.message = message


class GroundingGeometryTest(unittest.TestCase):
    """Verify group geometry, contour meaning and pair-loss cleanup."""

    def setUp(self):
        """Create an isolated harness around the production node methods."""
        self.node = SimpleNamespace(
            gaussian_d0=0.5, gaussian_a=0.5, gaussian_c=0.9, gaussian_k=0.5,
            social_state='talking',
            social_weights={'talking': 0.9, 'standing': 0.7,
                            'stationary': 0.5, 'crossing': 0.5},
            stationary_sigma_m=0.5,
            maximum_cost=100, expected_frame='odom', publish_cost_grid=False,
            position_ema_alpha=0.20, filtered_positions_m={},
            contour_levels=[.90, .75, .60, .45, .30, .15],
            outer_contour_scale=math.sqrt(-2*math.log(.15)),
            static_obstacle_marker_enabled=True,
            static_obstacle_marker_pose_m=(0.0, -0.195, 0.60),
            static_obstacle_marker_size_m=(0.90, 0.40, 1.20),
            static_obstacle_marker_yaw_rad=0.0,
            markers_pub=CapturedPublisher())
        for name in ('make_talking_zone', 'build_constraint_zones',
                     'make_zones_for_person', 'make_stationary_zone',
                     'filter_people_positions', 'zone_cost',
                     'make_markers', 'make_static_obstacle_marker',
                     'block_b_callback', 'validate_block_b'):
            setattr(self.node, name, MethodType(
                getattr(MODULE.SocialConstraintGrounding, name), self.node))
        self.node.get_parameter = lambda name: SimpleNamespace(value=0.75)
        self.node.get_logger = lambda: SimpleNamespace(warn=lambda message: None)
        # A stationary pair 1.5 m apart, rotated away from world axes.
        self.people = [
            {'track_id': 3, 'position_m': {'x': -0.6, 'y': -0.45}},
            {'track_id': 8, 'position_m': {'x': 0.6, 'y': 0.45}},
        ]

    def test_pair_geometry_and_order(self):
        """The pair is centred, symmetric, and independent of input order."""
        zones = self.node.build_constraint_zones(self.people)
        self.assertEqual(len(zones), 1)
        zone = zones[0]
        self.assertEqual(zone, self.node.build_constraint_zones(self.people[::-1])[0])
        self.assertEqual(zone.member_ids, (3, 8))
        self.assertAlmostEqual(zone.x_m, 0.0)
        self.assertAlmostEqual(zone.y_m, 0.0)
        self.assertAlmostEqual(zone.separation_m, 1.5)
        self.assertAlmostEqual(zone.yaw_rad, math.atan2(0.9, 1.2))
        self.assertAlmostEqual(zone.front_semi_axis_m, 0.525)
        self.assertAlmostEqual(zone.rear_semi_axis_m, 0.525)
        self.assertAlmostEqual(zone.left_semi_axis_m, 0.175)
        self.assertEqual(self.node.zone_cost(0.0, 0.0, zone), 90)

    def test_contours_follow_weighted_cost_for_pair_and_crossing(self):
        """Every contour has its declared peak fraction, including rotated axes."""
        pair = self.node.build_constraint_zones(self.people)[0]
        self.node.social_state = 'crossing'
        crossing = self.node.make_zones_for_person({
            'track_id': 9,
            'position_m': {'x': 2.0, 'y': -1.0},
            'velocity_mps': {'vx': math.cos(-.7), 'vy': math.sin(-.7)},
        })[0]
        for zone in (pair, crossing):
            markers = self.node.make_markers('odom', 1_000_000_000, [zone]).markers
            self.assertEqual(markers[0].action, Marker.DELETEALL)
            outlines = [m for m in markers if m.type == Marker.LINE_STRIP]
            self.assertEqual(len(outlines), 6)
            for level, outline in zip(self.node.contour_levels, outlines):
                self.assertEqual(outline.header.frame_id, 'odom')
                self.assertAlmostEqual(outline.points[0].x, outline.points[-1].x)
                self.assertAlmostEqual(outline.points[0].y, outline.points[-1].y)
                for point in outline.points:
                    dx, dy = point.x-zone.x_m, point.y-zone.y_m
                    forward = math.cos(zone.yaw_rad)*dx + math.sin(zone.yaw_rad)*dy
                    side = -math.sin(zone.yaw_rad)*dx + math.cos(zone.yaw_rad)*dy
                    sigma = zone.front_semi_axis_m if forward >= 0 else zone.rear_semi_axis_m
                    q = (forward/sigma)**2 + (side/zone.left_semi_axis_m)**2
                    self.assertAlmostEqual(math.exp(-q/2), level)
                    self.assertLessEqual(abs(self.node.zone_cost(point.x, point.y, zone)
                                             - 100*zone.social_weight*level), .50000001)
            info = [marker for marker in markers if marker.ns == 'social_constraint_info']
            self.assertIn('TAM GAUSS', info[-1].text)
        # Displaying 15% does not truncate the underlying Gaussian tail.
        distance = 3*pair.front_semi_axis_m
        self.assertEqual(self.node.zone_cost(
            distance*math.cos(pair.yaw_rad), distance*math.sin(pair.yaw_rad), pair), 1)

    def test_contour_ids_and_single_level_configuration(self):
        """Two people must not overwrite contours; [0.15] selects one line."""
        self.node.social_state = 'crossing'
        zones = [self.node.make_zones_for_person({
            'track_id': track_id,
            'position_m': {'x': float(track_id), 'y': 0.0},
            'velocity_mps': {'vx': 0.0, 'vy': 1.0},
        })[0] for track_id in (1, 2)]
        markers = self.node.make_markers('odom', 0, zones).markers[1:]
        self.assertEqual(len({(m.ns, m.id) for m in markers}), len(markers))
        self.node.contour_levels = [.15]
        markers = self.node.make_markers('odom', 0, zones[:1]).markers
        self.assertEqual(len([m for m in markers if m.type == Marker.LINE_STRIP]), 1)
        info = [marker for marker in markers if marker.ns == 'social_constraint_info']
        self.assertIn('TAM GAUSS', info[-1].text)

    def test_static_bookshelf_marker_matches_gazebo_collision_envelope(self):
        """RViz CUBE must show the same fixed obstacle used by the RL world."""
        markers = self.node.make_markers('odom', 1_000_000_000, []).markers
        shelf = next(marker for marker in markers if marker.ns == 'static_obstacles')
        self.assertEqual(shelf.type, Marker.CUBE)
        self.assertEqual(shelf.header.frame_id, 'odom')
        self.assertAlmostEqual(shelf.pose.position.x, 0.0)
        self.assertAlmostEqual(shelf.pose.position.y, -0.195)
        self.assertAlmostEqual(shelf.pose.position.z, 0.60)
        self.assertAlmostEqual(shelf.scale.x, 0.90)
        self.assertAlmostEqual(shelf.scale.y, 0.40)
        self.assertAlmostEqual(shelf.scale.z, 1.20)
        self.assertAlmostEqual(shelf.pose.orientation.z, 0.0)
        self.assertAlmostEqual(shelf.pose.orientation.w, 1.0)

    def test_stationary_mode_uses_fixed_half_metre_circle(self):
        """Manual stationary mode ignores object position and velocity noise."""
        self.node.social_state = 'stationary'
        person = {
            'track_id': 42,
            'position_m': {'x': 1.25, 'y': -0.75},
            'velocity_mps': {'vx': 3.0, 'vy': -2.0},
        }
        zone = self.node.build_constraint_zones([person])[0]
        self.assertEqual(zone.social_state, 'stationary')
        self.assertEqual(zone.member_ids, (42,))
        self.assertAlmostEqual(zone.x_m, 1.25)
        self.assertAlmostEqual(zone.y_m, -0.75)
        self.assertAlmostEqual(zone.yaw_rad, 0.0)
        self.assertEqual((zone.front_semi_axis_m, zone.rear_semi_axis_m,
                          zone.left_semi_axis_m, zone.right_semi_axis_m),
                         (0.5, 0.5, 0.5, 0.5))
        self.assertEqual(self.node.zone_cost(zone.x_m, zone.y_m, zone), 50)
        self.assertEqual(self.node.build_constraint_zones(self.people), [])

    def test_incomplete_ambiguous_or_invalid_pair(self):
        """Do not invent pairs from missing, duplicate or nonfinite tracks."""
        for people in ([], self.people[:1], self.people + [self.people[0]],
                       [self.people[0], self.people[0]],
                       [self.people[0], {'track_id': 8, 'position_m': {'x': float('nan'), 'y': 0}}],
                       [self.people[0], {'track_id': 8, 'position_m': self.people[0]['position_m']}],
                       [self.people[0], {}]):
            with self.subTest(people=people):
                self.assertEqual(self.node.build_constraint_zones(people), [])

    def test_outline_only_callback_clears_lost_pair(self):
        """Default callback does no raster work and clears an old group."""
        def unexpected_grid(*args):
            raise AssertionError('outline-only mode must not generate a grid')
        self.node.make_grid = unexpected_grid
        # Pair: DELETEALL + centre + people + six contours + info + shelf.
        # One person is invalid in explicit talking mode, so only DELETEALL
        # and the always-visible static shelf remain.
        for people, expected_count in ((self.people, 5 + len(self.node.contour_levels)),
                                       (self.people[:1], 2)):
            message = String(data=json.dumps({
                'header': {'frame_id': 'odom', 'stamp_ns': 1_000_000_000},
                'people': people}))
            self.node.block_b_callback(message)
            self.assertEqual(len(self.node.markers_pub.message.markers), expected_count)
            self.assertEqual(self.node.markers_pub.message.markers[0].action, Marker.DELETEALL)

    def test_crossing_remains_individual_current_geometry(self):
        """Adding a talking pair must not change crossing sigma or predictions."""
        self.node.social_state = 'crossing'
        for person in self.people:
            person['velocity_mps'] = {'vx': 0.8, 'vy': 0.6}
            person['prediction'] = {'trajectory_m': [{'dt_s': 1, 'x': 20, 'y': 20}]}
        zones = self.node.build_constraint_zones(self.people)
        self.assertEqual(len(zones), 2)
        for zone in zones:
            self.assertEqual(zone.member_ids, (zone.track_id,))
            self.assertAlmostEqual(zone.front_semi_axis_m, 1.05)
            self.assertAlmostEqual(zone.rear_semi_axis_m, 0.35)
            self.assertEqual(self.node.zone_cost(zone.x_m, zone.y_m, zone), 50)

    def test_auto_switches_without_restarting_or_changing_selector(self):
        """One -> two -> one -> empty removes obsolete marker namespaces."""
        self.node.social_state = 'auto'
        self.people[0]['velocity_mps'] = {'vx': 1.0, 'vy': 0.0}
        for people, state, peak in (
                (self.people[:1], 'crossing', 50),
                (self.people, 'talking', 90),
                (self.people[:1], 'crossing', 50)):
            zones = self.node.build_constraint_zones(people)
            self.assertEqual(len(zones), 1)
            self.assertEqual(self.node.zone_cost(zones[0].x_m, zones[0].y_m, zones[0]), peak)
            self.node.block_b_callback(String(data=json.dumps({
                'header': {'frame_id': 'odom', 'stamp_ns': 1_000_000_000},
                'people': people})))
            markers = self.node.markers_pub.message.markers
            self.assertEqual(markers[0].action, Marker.DELETEALL)
            info = [marker for marker in markers if marker.ns == 'social_constraint_info']
            self.assertIn(f'TAM GAUSS ({state})', info[0].text)
            self.assertEqual(markers[-1].ns, 'static_obstacles')
            self.assertEqual(self.node.social_state, 'auto')
        self.assertEqual(self.node.build_constraint_zones([]), [])
        self.assertEqual(self.node.build_constraint_zones(self.people + [self.people[0]]), [])


if __name__ == '__main__':
    unittest.main()
