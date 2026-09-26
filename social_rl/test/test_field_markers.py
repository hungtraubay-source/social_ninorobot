"""Tests for rendering the exact field consumed by the RL policy."""

from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker

from social_rl.constraint_field import ConstraintField, Zone, ZoneSample
from social_rl.field_markers import FieldMarkerRenderer


def _zone(zone_id, track_ids, scene_type, hardness, center):
    sample = ZoneSample(
        t=0.0,
        shape='asymmetric_gaussian',
        center=center,
        size=(0.8, 0.3, 0.6),
        weight=0.9,
        orientation=0.25)
    return Zone(
        zone_id=zone_id,
        track_ids=track_ids,
        scene_type=scene_type,
        hardness=hardness,
        confidence=0.9,
        valid_from=1.0,
        valid_to=1.0,
        trajectory_of_zone=(sample,))


def test_markers_preserve_the_exact_constraint_field():
    field = ConstraintField(
        timestamp=1.0,
        frame='base_link',
        horizon_steps=1,
        dt=0.0,
        zones=(
            _zone('talking-1-2', ('1', '2'), 'talking', 'hard',
                  (1.2, -0.4)),
            _zone('person-3', ('3',), 'waiting', 'soft',
                  (-0.5, 0.8)),
        ))

    renderer = FieldMarkerRenderer()
    result = renderer.render(field, Time(sec=7))

    assert result.markers[0].action == Marker.DELETEALL
    assert len(result.markers) == 5
    assert all(marker.header.frame_id == 'base_link'
               for marker in result.markers)

    first_outline, first_label = result.markers[1:3]
    second_outline, second_label = result.markers[3:5]
    assert first_outline.type == Marker.LINE_STRIP
    assert second_outline.type == Marker.LINE_STRIP
    assert (first_label.pose.position.x, first_label.pose.position.y) == (
        1.2, -0.4)
    assert (second_label.pose.position.x, second_label.pose.position.y) == (
        -0.5, 0.8)
    assert first_label.text == 'talking [hard] 1,2'
    assert second_label.text == 'waiting [soft] 3'
    assert all(
        marker.lifetime.sec == 0 and marker.lifetime.nanosec == 0
        for marker in result.markers[1:])


def test_marker_is_updated_until_its_zone_disappears():
    active = ConstraintField(
        timestamp=2.0,
        frame='base_link',
        horizon_steps=1,
        dt=0.0,
        zones=(
            _zone('talking-1-2', ('1', '2'), 'talking', 'hard',
                  (1.2, -0.4)),
        ))
    empty = ConstraintField(
        timestamp=3.0, frame='base_link', horizon_steps=1, dt=0.0, zones=())
    renderer = FieldMarkerRenderer()

    first = renderer.render(active, Time(sec=8))
    refreshed = renderer.render(active, Time(sec=9))
    deleted = renderer.render(empty, Time(sec=10))

    assert first.markers[0].action == Marker.DELETEALL
    assert [marker.action for marker in refreshed.markers] == [
        Marker.ADD, Marker.ADD]
    assert [marker.id for marker in refreshed.markers] == [0, 0]
    assert [marker.action for marker in deleted.markers] == [
        Marker.DELETE, Marker.DELETE]
    assert {marker.ns for marker in deleted.markers} == {
        'rl_constraint_field', 'rl_constraint_field_label'}
