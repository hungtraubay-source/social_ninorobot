"""Regression tests for selecting the social intrusion charged by reward."""

from types import SimpleNamespace

import pytest

from social_rl.reward import RewardConfig, evaluate_step
from social_rl.ros_env import _reward_social_intrusion
from social_rl.ros_interface import PerceptionBridge


def test_ground_truth_uses_full_intrusion_at_full_weight():
    """Charge the unfiltered ground-truth depth with no attenuation."""
    state = {
        'social_intrusion': 0.0,
        'social_intrusion_hidden': 0.75,
    }

    intrusion = _reward_social_intrusion(state, has_ground_truth=True)
    outcome = evaluate_step(
        RewardConfig(), goal_distance=2.0, previous_goal_distance=2.0,
        goal_bearing=0.0, previous_goal_bearing=0.0,
        minimum_scan=2.0, social_intrusion=intrusion)

    assert outcome.components['social'] == pytest.approx(-6.0)


def test_empty_ground_truth_is_authoritative_zero():
    """Do not substitute a remembered value for a truly empty full scene."""
    state = {
        'social_intrusion': 0.8,
        'social_intrusion_hidden': 0.0,
    }

    assert _reward_social_intrusion(state, has_ground_truth=True) == 0.0


def test_perception_explicitly_falls_back_to_visible_intrusion():
    """Use tracker intrusion when simulator truth is unavailable."""
    state = {'social_intrusion': 0.4}

    assert _reward_social_intrusion(state, has_ground_truth=False) == 0.4


def test_ground_truth_never_silently_falls_back_when_full_value_is_missing():
    """Expose a broken bridge instead of silently reopening the loophole."""
    with pytest.raises(KeyError):
        _reward_social_intrusion(
            {'social_intrusion': 0.4}, has_ground_truth=True)


@pytest.mark.parametrize(
    ('full_people', 'expected_intrusion', 'expected_hidden'),
    [
        ([SimpleNamespace(distance=0.8, intrusion=0.75)], 0.75, 1),
        ([], 0.0, 0),
    ],
)
def test_bridge_always_emits_authoritative_full_scene(
        monkeypatch, full_people, expected_intrusion, expected_hidden):
    """Do not let the legacy diagnostic flag disable the reward input."""

    class Provider:
        def relative_people(self, apply_camera=True):
            return [] if apply_camera else full_people

    class Memory:
        @staticmethod
        def remember(people, _transform, _now):
            return people

    bridge = object.__new__(PerceptionBridge)
    bridge.scan = object()
    bridge._env = SimpleNamespace(
        scan_topic='/scan', people_occlusion=False, vlm_noise=False,
        measure_hidden_intrusion=False)
    bridge._observation = SimpleNamespace(constraint_field=object())
    bridge._people_provider = Provider()
    bridge._memory = Memory()
    bridge._node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)))
    bridge.people_transform_valid = False
    bridge.goal_transform = lambda: object()
    bridge.scan_returns = lambda: ([], [])
    bridge.robot_velocity = lambda: (0.0, 0.0)
    bridge.minimum_scan = lambda _ranges: 2.0
    bridge._publish_field = lambda _field: None

    monkeypatch.setattr(
        'social_rl.ros_interface.transform_point',
        lambda _transform, x, y: (x, y))
    monkeypatch.setattr(
        'social_rl.ros_interface.compile_zones',
        lambda people, _config: list(people))
    monkeypatch.setattr(
        'social_rl.ros_interface.intrusion_at_zones',
        lambda zones: max(
            (getattr(zone, 'intrusion', 0.0) for zone in zones), default=0.0))
    monkeypatch.setattr(
        'social_rl.ros_interface.build_observation',
        lambda _data, _config, _field: 'observation')

    observation, state = bridge.observe(1.0, 2.0)

    assert observation == 'observation'
    assert state['social_intrusion'] == 0.0
    assert state['social_intrusion_hidden'] == expected_intrusion
    assert state['hidden_people'] == expected_hidden
    assert state['person_distances_all'] == [
        person.distance for person in full_people]
