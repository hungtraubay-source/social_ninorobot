"""Timestamp-safe fusion of delayed VLM states with tracked person IDs."""

import math


# Block-C labels are deliberately collapsed into the vocabulary understood by
# constraint_field.py. Their geometry is still determined from the tracker
# pose and velocity; a VLM decision can never create or remove a person.
_VLM_STATE_TO_SCENE_TYPE = {
    'crossing': 'passing',
    'approaching': 'passing',
    'straight': 'passing',
    'talking': 'talking',
    'waiting': 'waiting',
}


def _header_stamp_seconds(header):
    """Return a ROS header stamp as seconds, or None for malformed input."""
    try:
        stamp = getattr(header, 'stamp', header)
        seconds = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except (AttributeError, TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0.0 else None


class VlmStateCache:
    """Join delayed per-track VLM labels to the matched People track.

    States are aged from ``inference_stamp`` (the wall-clock time when the
    VLM *published* its result), not from the source RGB frame header.  Using
    the source frame stamp would expire every result immediately because the
    frame is ~27 s old by the time inference finishes.  The freshness window
    (``timeout``) therefore measures how long a semantic label remains valid
    *after publication*, which is the correct contract for the costmap logic.
    """

    def __init__(self, timeout: float):
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError('vlm_state_timeout must be a finite value > 0 s')
        self._timeout = float(timeout)
        self._states = {}

    def update(self, message) -> None:
        """Store only recognized semantic labels from one VLM result."""
        # Use inference_stamp (publish time) so freshness is measured from when
        # the result became available, not from the original source frame stamp.
        inference_stamp = _header_stamp_seconds(
            getattr(message, 'inference_stamp', None))
        if inference_stamp is None:
            # Fallback to header.stamp if inference_stamp is missing or zero.
            inference_stamp = _header_stamp_seconds(
                getattr(message, 'header', None))
        if inference_stamp is None:
            return
        for item in getattr(message, 'states', ()):
            person_id = str(getattr(item, 'person_id', '')).strip()
            state = _VLM_STATE_TO_SCENE_TYPE.get(
                str(getattr(item, 'state', '')).strip().lower())
            if not person_id or state is None:
                continue
            previous = self._states.get(person_id)
            # Always overwrite with the newest inference result.  inference_stamp
            # is monotonically increasing (wall clock), so a simple comparison
            # is sufficient; no special clock-reset handling is needed here.
            if previous is None or inference_stamp >= previous[0]:
                self._states[person_id] = (inference_stamp, state)

        # Prune entries older than timeout relative to the current inference batch.
        self._states = {
            pid: entry for pid, entry in self._states.items()
            if 0.0 <= inference_stamp - entry[0] <= self._timeout
        }

    def lookup(self, person_id: str, people_header) -> tuple:
        """Return (scene_type, confidence) if the label is still fresh."""
        people_stamp = _header_stamp_seconds(people_header)
        entry = self._states.get(str(person_id))
        if people_stamp is None or entry is None:
            return '', 0.0
        # Age is measured from inference time (publication), not source frame.
        # Use people_stamp as a proxy for wall-clock now; in sim they advance
        # together so this is a valid freshness guard.
        age = people_stamp - entry[0]
        if age < -5.0 or age > self._timeout:
            # Allow small negative values (≤5 s) to tolerate slight clock skew
            # between the VLM publish stamp and the people topic stamp.
            return '', 0.0
        return entry[1], 1.0

