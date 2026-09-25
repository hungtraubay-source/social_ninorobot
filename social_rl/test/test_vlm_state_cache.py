"""Regression tests for joining delayed VLM states to tracked people."""

from types import SimpleNamespace

from social_rl.constraint_field import ConstraintFieldConfig, compile_zones
from social_rl.observation import RelativeEntity
from social_rl.semantic_fusion import VlmStateCache


def _header(seconds):
    whole = int(seconds)
    return SimpleNamespace(
        stamp=SimpleNamespace(sec=whole, nanosec=int((seconds - whole) * 1e9)))


def _result(seconds, *states):
    return SimpleNamespace(
        header=_header(seconds),
        states=[SimpleNamespace(person_id=person_id, state=state)
                for person_id, state in states])


def test_talking_track_states_create_one_group_zone():
    cache = VlmStateCache(timeout=3.0)
    cache.update(_result(10.0, ('person_4', 'talking'),
                         ('person_9', 'talking')))

    state_a, confidence_a = cache.lookup('person_4', _header(11.0))
    state_b, confidence_b = cache.lookup('person_9', _header(11.0))
    field = compile_zones([
        RelativeEntity(1.0, 0.0, scene_type=state_a, track_id='person_4',
                       scene_confidence=confidence_a),
        RelativeEntity(2.0, 0.0, scene_type=state_b, track_id='person_9',
                       scene_confidence=confidence_b),
    ], ConstraintFieldConfig())

    assert len(field.zones) == 1
    zone = field.zones[0]
    assert zone.scene_type == 'talking'
    assert zone.hardness == 'hard'
    assert zone.track_ids == ('person_4', 'person_9')
    assert zone.trajectory_of_zone[0].center == (1.5, 0.0)


def test_track_state_is_normalized_and_expires_by_source_stamp():
    cache = VlmStateCache(timeout=3.0)
    cache.update(_result(10.0, ('person_12', 'crossing'),
                         ('person_13', 'unsupported')))

    assert cache.lookup('person_12', _header(12.5)) == ('passing', 1.0)
    assert cache.lookup('person_13', _header(12.5)) == ('', 0.0)
    assert cache.lookup('person_12', _header(9.9)) == ('', 0.0)
    assert cache.lookup('person_12', _header(13.1)) == ('', 0.0)


def test_talking_result_survives_the_measured_qwen_latency():
    cache = VlmStateCache(timeout=25.0)
    cache.update(_result(100.0, ('person_1', 'talking')))

    assert cache.lookup('person_1', _header(120.0)) == ('talking', 1.0)
    assert cache.lookup('person_1', _header(125.1)) == ('', 0.0)
