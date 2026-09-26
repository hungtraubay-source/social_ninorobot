"""Tests for lossless deploy ConstraintField transport."""

from social_rl.constraint_field import ConstraintField, Zone, ZoneSample
from social_rl.field_transport import field_from_json, field_to_json


def test_constraint_field_json_round_trip_is_exact():
    sample = ZoneSample(
        t=0.0,
        shape='gaussian',
        center=(1.234567890123, -0.333333333333),
        size=(0.75, 0.25, 0.75),
        weight=0.9,
        orientation=1.234567890123)
    zone = Zone(
        zone_id='z_gperson_1_person_2',
        track_ids=('person_1', 'person_2'),
        scene_type='talking',
        hardness='hard',
        confidence=0.987654321,
        valid_from=0.0,
        valid_to=0.0,
        trajectory_of_zone=(sample,))
    field = ConstraintField(
        timestamp=0.0,
        frame='base_link',
        horizon_steps=1,
        dt=0.0,
        zones=(zone,))

    restored = field_from_json(field_to_json(field))

    assert restored == field
