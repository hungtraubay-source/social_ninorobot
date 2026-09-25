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
        stamp = header.stamp
        seconds = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except (AttributeError, TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0.0 else None


class VlmStateCache:
    """Join delayed per-track VLM labels to the matching People sample.

    VLM output carries the header of the RGB/People frame it classified, not
    its completion time. Looking up by that source stamp prevents a result
    from a late inference or a reused ByteTrack ID from changing a newer track.
    """

    def __init__(self, timeout: float):
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError('vlm_state_timeout must be a finite value > 0 s')
        self._timeout = float(timeout)
        self._states = {}

    def update(self, message) -> None:
        """Store only recognized semantic labels from one VLM result."""
        stamp = _header_stamp_seconds(getattr(message, 'header', None))
        if stamp is None:
            return
        for item in getattr(message, 'states', ()):
            person_id = str(getattr(item, 'person_id', '')).strip()
            state = _VLM_STATE_TO_SCENE_TYPE.get(
                str(getattr(item, 'state', '')).strip().lower())
            if not person_id or state is None:
                continue
            previous = self._states.get(person_id)
            # A simulator clock reset is the one case where an older source
            # stamp must replace a cached value from the previous episode.
            if (previous is None or stamp >= previous[0]
                    or previous[0] - stamp > self._timeout):
                self._states[person_id] = (stamp, state)

        # Bound memory even if a tracker creates many short-lived IDs.
        self._states = {
            person_id: entry for person_id, entry in self._states.items()
            if 0.0 <= stamp - entry[0] <= self._timeout
        }

    def lookup(self, person_id: str, people_header) -> tuple:
        """Return (scene_type, confidence) valid for this People timestamp."""
        people_stamp = _header_stamp_seconds(people_header)
        entry = self._states.get(str(person_id))
        if people_stamp is None or entry is None:
            return '', 0.0
        age = people_stamp - entry[0]
        if age < 0.0 or age > self._timeout:
            return '', 0.0
        # VlmPersonState has no calibrated probability yet. A fresh typed
        # decision is therefore treated as confident; an absent/stale one uses
        # the neutral fallback above.
        return entry[1], 1.0
