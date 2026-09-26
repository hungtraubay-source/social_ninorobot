"""Render the exact ConstraintField consumed by the RL observation in RViz."""

import math

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


OUTER_CONTOUR_LEVEL = 0.15
ELLIPSE_SEGMENTS = 72
ZONE_COLOURS = {
    'talking': (0.95, 0.10, 0.15),
    'waiting': (0.95, 0.55, 0.10),
    'passing': (0.20, 0.55, 0.95),
    'walking': (0.20, 0.55, 0.95),
    '': (0.55, 0.55, 0.55),
}


def _ellipse_points(sample):
    """Return the 15%-peak contour of one Gaussian sample."""
    radius_sigma = math.sqrt(-2.0 * math.log(OUTER_CONTOUR_LEVEL))
    sigma_front, sigma_side, sigma_rear = sample.size
    cosine = math.cos(sample.orientation)
    sine = math.sin(sample.orientation)
    points = []
    for segment in range(ELLIPSE_SEGMENTS + 1):
        angle = 2.0 * math.pi * segment / ELLIPSE_SEGMENTS
        forward_sigma = (
            sigma_front if math.cos(angle) >= 0.0 else sigma_rear)
        forward = radius_sigma * forward_sigma * math.cos(angle)
        lateral = radius_sigma * sigma_side * math.sin(angle)
        point = Point()
        point.x = float(
            sample.center[0] + cosine * forward - sine * lateral)
        point.y = float(
            sample.center[1] + sine * forward + cosine * lateral)
        point.z = 0.02
        points.append(point)
    return points


class FieldMarkerRenderer:
    """Keep RViz markers aligned with the current field, without expiry."""

    def __init__(self):
        self._marker_ids = {}
        self._active_keys = set()
        self._next_marker_id = 0
        self._clear_on_first_render = True

    def _marker_id(self, key):
        marker_id = self._marker_ids.get(key)
        if marker_id is None:
            marker_id = self._next_marker_id
            self._next_marker_id += 1
            self._marker_ids[key] = marker_id
        return marker_id

    @staticmethod
    def _delete_marker(frame, stamp, namespace, marker_id):
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.action = Marker.DELETE
        return marker

    def render(self, field, stamp) -> MarkerArray:
        """Render current zones and explicitly delete only disappeared ones."""
        output = MarkerArray()
        if self._clear_on_first_render:
            clear = Marker()
            clear.header.frame_id = field.frame
            clear.header.stamp = stamp
            clear.action = Marker.DELETEALL
            output.markers.append(clear)
            self._clear_on_first_render = False

        entries = []
        current_keys = set()
        for zone in field.zones:
            for sample_index, sample in enumerate(zone.trajectory_of_zone):
                key = (zone.zone_id, sample_index)
                current_keys.add(key)
                entries.append((key, zone, sample))

        for key in self._active_keys - current_keys:
            marker_id = self._marker_id(key)
            output.markers.append(self._delete_marker(
                field.frame, stamp, 'rl_constraint_field', marker_id))
            output.markers.append(self._delete_marker(
                field.frame, stamp, 'rl_constraint_field_label', marker_id))

        for key, zone, sample in entries:
            marker_id = self._marker_id(key)
            colour = ZONE_COLOURS.get(
                zone.scene_type, ZONE_COLOURS[''])

            outline = Marker()
            outline.header.frame_id = field.frame
            outline.header.stamp = stamp
            outline.ns = 'rl_constraint_field'
            outline.id = marker_id
            outline.type = Marker.LINE_STRIP
            outline.action = Marker.ADD
            outline.pose.orientation.w = 1.0
            outline.scale.x = (
                0.04 if zone.hardness == 'hard' else 0.025)
            outline.color.r, outline.color.g, outline.color.b = colour
            outline.color.a = 0.95
            outline.points = _ellipse_points(sample)
            output.markers.append(outline)

            label = Marker()
            label.header.frame_id = field.frame
            label.header.stamp = stamp
            label.ns = 'rl_constraint_field_label'
            label.id = marker_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(sample.center[0])
            label.pose.position.y = float(sample.center[1])
            label.pose.position.z = 0.25
            label.pose.orientation.w = 1.0
            label.scale.z = 0.20
            label.color.r, label.color.g, label.color.b = colour
            label.color.a = 1.0
            label.text = (
                f'{zone.scene_type or "neutral"} [{zone.hardness}] '
                f'{",".join(str(track_id) for track_id in zone.track_ids)}')
            output.markers.append(label)

        self._active_keys = current_keys
        return output
