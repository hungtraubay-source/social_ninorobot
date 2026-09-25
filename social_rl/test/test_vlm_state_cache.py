"""Regression tests for joining delayed VLM states to tracked people."""

from types import SimpleNamespace

from social_rl.constraint_field import ConstraintFieldConfig, compile_zones
from social_rl.observation import RelativeEntity
from social_rl.semantic_fusion import VlmStateCache


def _header(seconds):
    whole = int(seconds)
    return SimpleNamespace(
        stamp=SimpleNamespace(sec=whole, nanosec=int((seconds - whole) * 1e9)))


def _result(source_seconds, *states, inference_seconds=None):
    """Build a fake VlmPersonStates message.

    ``source_seconds`` is the RGB frame capture time (header.stamp).
    ``inference_seconds`` is the publish time (inference_stamp); defaults to
    ``source_seconds`` for tests that do not care about the distinction.
    """
    if inference_seconds is None:
        inference_seconds = source_seconds
    return SimpleNamespace(
        header=_header(source_seconds),
        inference_stamp=_header(inference_seconds).stamp,
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


def test_track_state_is_normalized_and_expires_by_inference_stamp():
    """States expire relative to inference_stamp, not the source frame stamp."""
    cache = VlmStateCache(timeout=3.0)
    # Source frame captured at t=10, inference finished at t=10 (same for simplicity).
    cache.update(_result(10.0, ('person_12', 'crossing'),
                         ('person_13', 'unsupported'),
                         inference_seconds=10.0))

    # t=12.5: age=2.5s < 3s timeout → valid
    assert cache.lookup('person_12', _header(12.5)) == ('passing', 1.0)
    # Unsupported state → empty
    assert cache.lookup('person_13', _header(12.5)) == ('', 0.0)
    # t=13.1: age=3.1s > 3s → expired
    assert cache.lookup('person_12', _header(13.1)) == ('', 0.0)


def test_talking_result_survives_the_measured_qwen_latency():
    """VLM results must NOT expire due to the ~27s inference latency.

    Bug: previously the source frame stamp was used for freshness, so a result
    published 27s after capture was already 27s > 25s timeout and discarded.
    Fix: inference_stamp (publish time) is now the freshness reference.
    """
    cache = VlmStateCache(timeout=25.0)
    # Source frame at t=100, inference finishes 27s later at t=127.
    cache.update(_result(100.0, ('person_1', 'talking'), inference_seconds=127.0))

    # Lookup at t=128 (1s after publish) → age=1s < 25s → valid
    assert cache.lookup('person_1', _header(128.0)) == ('talking', 1.0)
    # Lookup at t=152 (25s after publish) → age=25s = timeout → expired
    assert cache.lookup('person_1', _header(152.1)) == ('', 0.0)


def test_single_talking_person_creates_individual_zone():
    field = compile_zones([
        RelativeEntity(1.0, 0.0, scene_type='talking', track_id='person_1',
                       scene_confidence=1.0, facing=0.5),
    ], ConstraintFieldConfig())

    assert len(field.zones) == 1
    zone = field.zones[0]
    assert zone.scene_type == 'talking'
    assert zone.hardness == 'soft'
    assert zone.track_ids == ('person_1',)
    assert zone.trajectory_of_zone[0].center == (1.0, 0.0)
    assert zone.trajectory_of_zone[0].orientation == 0.5

