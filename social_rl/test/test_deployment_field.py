"""Tests for deploy field generation independent of a navigation goal."""

from types import SimpleNamespace
import math

from social_rl.constraint_field import (ConstraintField, ConstraintFieldConfig,
                                        Zone, ZoneSample, compile_zones)
from social_rl.constraint_field_node import LockedTalkingField
from social_rl.observation import RelativeEntity
from social_rl.ros_interface import PerceptionBridge


class _Memory:
    def __init__(self):
        self.received = None
        self.cleared = False

    def remember(self, people, transform, now):
        self.received = (people, transform, now)
        return people

    def clear(self):
        self.cleared = True


class _Cache:
    def __init__(self):
        self.cleared = False

    def clear(self):
        self.cleared = True


def test_deploy_field_matches_direct_compile_without_goal():
    people = [
        RelativeEntity(
            x=1.0, y=-0.4, scene_type='talking',
            track_id='person_1', scene_confidence=1.0),
        RelativeEntity(
            x=1.0, y=0.4, scene_type='talking',
            track_id='person_2', scene_confidence=1.0),
    ]
    config = ConstraintFieldConfig()
    memory = _Memory()
    bridge = object.__new__(PerceptionBridge)
    bridge._people_provider = None
    bridge._observation = SimpleNamespace(constraint_field=config)
    bridge._memory = memory
    bridge._node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=2_000_000_000)))
    bridge.people_transform_valid = True
    transform = object()
    bridge.goal_transform = lambda: transform
    bridge.relative_people = lambda: people

    field = bridge.deployment_constraint_field()

    assert field == compile_zones(people, config)
    assert memory.received == (people, transform, 2.0)


def _transform(translation_x, yaw=0.0):
    return SimpleNamespace(
        translation=SimpleNamespace(x=translation_x, y=0.0),
        rotation=SimpleNamespace(
            x=0.0, y=0.0,
            z=math.sin(0.5 * yaw), w=math.cos(0.5 * yaw)))


def _field(center, *, scene_type='talking', hardness='hard'):
    sample = ZoneSample(
        t=0.0, shape='gaussian', center=center,
        size=(0.5, 0.2, 0.5), weight=0.9, orientation=0.0)
    zone = Zone(
        zone_id='zone', track_ids=('one', 'two'),
        scene_type=scene_type, hardness=hardness, confidence=1.0,
        valid_from=0.0, valid_to=0.0, trajectory_of_zone=(sample,))
    return ConstraintField(
        timestamp=0.0, frame='base_link', horizon_steps=1, dt=0.0,
        zones=(zone,))


def test_first_talking_zone_is_fixed_in_odom_and_cannot_be_replaced():
    latch = LockedTalkingField('odom', 'base_link')

    # Robot is at odom x=2, so base_link x=1 is odom x=3.
    assert latch.try_lock(_field((1.0, 0.0)), _transform(-2.0), 10.0)
    # A later VLM result must not replace the first accepted constraint.
    assert not latch.try_lock(
        _field((50.0, 0.0)), _transform(-2.0), 12.0)

    fixed = latch.for_visualization()
    assert fixed.frame == 'odom'
    assert fixed.zones[0].trajectory_of_zone[0].center == (3.0, 0.0)

    # After the robot moves to odom x=2.5, the same world zone is 0.5 m ahead.
    output = latch.for_robot(_transform(-2.5))
    assert output.frame == 'base_link'
    assert len(output.zones) == 1
    assert output.zones[0].trajectory_of_zone[0].center == (0.5, 0.0)

    assert not latch.expire_if_due(17.999)
    assert latch.expire_if_due(18.0)
    assert not latch.locked
    assert latch.try_lock(_field((4.0, 0.0)), _transform(0.0), 19.0)


def test_non_talking_zone_does_not_lock_the_field():
    latch = LockedTalkingField('odom', 'base_link')

    assert not latch.try_lock(
        _field((1.0, 0.0), scene_type='', hardness='soft'),
        _transform(0.0), 0.0)
    assert not latch.locked


def test_deployment_reset_clears_people_memory_and_vlm_cache():
    bridge = object.__new__(PerceptionBridge)
    bridge._people_provider = None
    bridge._memory = _Memory()
    bridge._vlm_states = _Cache()
    bridge.constraint_field = object()

    bridge.reset_deployment_grounding()

    assert bridge._memory.cleared
    assert bridge._vlm_states.cleared
    assert bridge.constraint_field is None
