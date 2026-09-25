"""Block D: grounding people into the social Gaussian constraint field.

This is the TRAINING-side implementation. It has to agree with the deployment
one -- social_perception/scripts/social_constraint_grounding.py, the Gaussian
grounding node the real robot runs -- because a policy trained against one
field and shielded against another is two different problems. The formulas here
are that node's, and the report they both come from:

    factor = 1 + a * (1 - c)          a: expansion coeff, c: situation confidence
    d0 = 0.5 m                        base personal distance

    passing   (one moving person)     sigma_h = factor * (d0 + k * speed)
                                      sigma_s = sigma_r = sigma_h / 3
                                      centre = person, orientation = velocity
    talking   (a face-to-face pair)   sigma_h = sigma_r = factor * (sep + d0)/3
                                      sigma_s = sigma_h / 3
                                      centre = midpoint, orientation = pair axis
                                      (15-09-2026: /4 -> /3, theo yêu cầu --
                                      xem RUN_RL.txt/lịch sử sửa ngày này cho
                                      lý do)
    waiting   (person facing object)  sigma_h = factor * (d_obj + d0)/2
                                      sigma_s = sigma_r = sigma_h / 3
                                      centre = person, orientation = person's
                                      own facing. d_obj is config.waiting_distance
                                      (11-09-2026) -- a tuned constant, not a
                                      measured distance to a real object; see
                                      compile_zones.
    base      (one still person)      sigma_h = sigma_s = sigma_r = d0/2,
                                      circular (15-09-2026: was d0, theo
                                      yêu cầu)

The field VALUE at a point is an anisotropic Gaussian, split front/rear:

    g(x, y) = wt * exp( -0.5 * [ (fwd / sigma_fwd)^2 + (lat / sigma_s)^2 ] )
    sigma_fwd = sigma_h if the point is in front of the centre, else sigma_r

`wt` is the per-situation weight (the Gaussian's peak): talking 0.9, waiting
0.7, passing 0.5, base 0.5. Overlap is resolved by maximum.

Two deliberate departures from the old ramp field, both signed off:

  * NO trajectory prediction. Block B is tracking only now; the field is one
    grid for the present instant, not five along a 2 s horizon.
  * NO separate group o-space. A conversation is ONE Gaussian at the midpoint,
    its spread set by the pair separation -- the node's behaviour, not the old
    "bounding ellipse laid over both personal zones" union.

Output is still a LIST OF ZONES (compile_zones) that render_zones rasterises,
so block F can read regions rather than pixels; each zone now just holds a
single present-time sample.

The static occupancy grid is NOT in here -- that is block H, on its own path
to block E.
"""

import math
from dataclasses import asdict, dataclass, field

import numpy as np

# One instant only: the present. The tuple length is still what sizes the CNN,
# so keeping it a tuple (rather than dropping the machinery) means observation.py
# and train.py need no special case for "no prediction".
DEFAULT_PREDICTION_TIMES = (0.0,)

# Gaussian peak per situation. An unknown label falls back to the base weight:
# a person is still a person even when block C cannot say what they are doing.
DEFAULT_WEIGHTS = {
    'talking': 0.9,
    'waiting': 0.7,
    'passing': 0.5,
    'walking': 0.5,
    '': 0.5,
}


@dataclass(frozen=True)
class ConstraintFieldConfig:
    """Parameters of the social Gaussian field. Distances in metres."""

    # Present only. Length drives the channel count and therefore the CNN, so a
    # checkpoint does not load into a config with a different length; train.py
    # refuses to resume across a change here.
    prediction_times: tuple = DEFAULT_PREDICTION_TIMES

    # Base personal distance d0, and the two shaping coefficients from the
    # report. `expansion_a` in (0, 1) keeps a low-confidence region from
    # ballooning: at c = 0.7 it grows the sigmas 15 % rather than 30 %.
    d0: float = 0.5
    expansion_a: float = 0.5
    # How fast the forward sigma grows with a person's speed, m per (m/s).
    speed_gain_k: float = 0.5

    # Gaussian peak per scene_type.
    weights: dict = field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS))

    # Two people whose scene_type is `talking` and who stand closer than this
    # share one conversation Gaussian at their midpoint.
    group_max_distance: float = 3.0

    # 11-09-2026: the `waiting` object (a shelf, a counter) used to be a real
    # point (config.waiting_object) the caller had to place every step. Given
    # up -- pinning it to an actual Gazebo mesh needs sighted placement this
    # package cannot do, and a wrong anchor put the actor INSIDE the shelf.
    # d_obj is now this fixed distance, not a measured one, and orientation
    # comes from the person's own `facing` instead of a bearing to the
    # object. A person spawns like any other `RoutePoint`-placed actor
    # (talking, backs_turned) and only needs to report which way they are
    # looking, which the plugin already does correctly.
    waiting_distance: float = 0.7

    # Speed below which a person counts as standing still: a still person with
    # no known situation gets the circular base region, not a passing one.
    still_speed: float = 0.1

    @property
    def channels(self) -> int:
        return len(self.prediction_times)

    def weight_for(self, scene_type: str) -> float:
        """The Gaussian peak for a situation, falling back to the base weight."""
        return float(self.weights.get(
            scene_type, self.weights.get('', 0.5)))

    def to_dict(self) -> dict:
        values = asdict(self)
        # YAML round-trips lists, not tuples, and a config that does not compare
        # equal after save/load makes train.py reject checkpoints it should
        # accept.
        values['prediction_times'] = list(self.prediction_times)
        values['weights'] = dict(self.weights)
        return values

    @classmethod
    def from_dict(cls, values: dict) -> 'ConstraintFieldConfig':
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                f'unknown constraint_field keys: {sorted(unknown)}. '
                f'Valid keys: {sorted(known)}')
        values = dict(values)
        if 'prediction_times' in values:
            values['prediction_times'] = tuple(values['prediction_times'])
        return cls(**values)


def _factor(confidence: float, config: ConstraintFieldConfig) -> float:
    """1 + a*(1-c). c is the situation confidence; ground truth is 1.0 -> 1.0."""
    c = min(1.0, max(0.0, float(confidence)))
    return 1.0 + config.expansion_a * (1.0 - c)


def _passing_sigmas(speed: float, factor: float, config: ConstraintFieldConfig):
    sigma_h = factor * (config.d0 + config.speed_gain_k * max(0.0, speed))
    side = sigma_h / 3.0
    return sigma_h, side, side


def _talking_sigmas(separation: float, factor: float,
                    config: ConstraintFieldConfig):
    # 15-09-2026: /4 -> /3, theo yêu cầu -- ellipse dọc trục cặp lớn hơn,
    # xem RUN_RL.txt.
    sigma_h = factor * ((separation + config.d0) / 3.0)
    return sigma_h, sigma_h / 3.0, sigma_h


def _waiting_sigmas(d_obj: float, factor: float,
                    config: ConstraintFieldConfig):
    sigma_h = factor * ((max(0.0, d_obj) + config.d0) / 2.0)
    side = sigma_h / 3.0
    return sigma_h, side, side


def _base_sigmas(config: ConstraintFieldConfig):
    # 15-09-2026: d0 -> d0/2, theo yêu cầu.
    half = config.d0 / 2.0
    return half, half, half


def _gaussian_region(x, y, centre_x, centre_y, orientation,
                     sigma_h, sigma_s, sigma_r, weight):
    """One zone's value on the cell grid: wt * exp(-0.5 * normalised dist^2).

    The ellipse is expressed in the zone's own frame -- +forward is
    `orientation` -- so the forward sigma is sigma_h ahead of the centre and
    sigma_r behind it, while sigma_s is the same on both sides.
    """
    delta_x = x - centre_x
    delta_y = y - centre_y
    cos_t, sin_t = math.cos(orientation), math.sin(orientation)
    forward = delta_x * cos_t + delta_y * sin_t
    lateral = -delta_x * sin_t + delta_y * cos_t
    sigma_forward = np.where(forward >= 0.0, sigma_h, sigma_r)
    # Sigmas are >= d0/3 > 0 for every case, so no divide-by-zero guard needed.
    quad = (forward / sigma_forward) ** 2 + (lateral / sigma_s) ** 2
    return weight * np.exp(-0.5 * quad)


# --------------------------------------------------------------- block D out
#
# K_soc leaves this module as a LIST OF ZONES, not an image. render_zones (block
# E's half) rasterises it. A zone still carries its track_ids, confidence and
# hardness so block F can take regions rather than a 0.2 m grid; it just holds
# one present-time sample now instead of a five-step trajectory.


@dataclass(frozen=True)
class ZoneSample:
    """One zone at one instant. With prediction gone there is exactly one.

    `size` is (sigma_h, sigma_s, sigma_r): forward, side, rear standard
    deviations of the Gaussian, in metres. `weight` is the peak value at the
    centre. `orientation` is the +forward direction in the field frame.
    """

    t: float
    shape: str
    center: tuple
    size: tuple
    weight: float
    orientation: float

    def to_dict(self) -> dict:
        return {
            't': round(self.t, 6),
            'shape': self.shape,
            'center': [round(value, 6) for value in self.center],
            'size': [round(value, 6) for value in self.size],
            'weight': round(self.weight, 6),
            'orientation': round(self.orientation, 6),
        }


@dataclass(frozen=True)
class Zone:
    """One social region.

    `hardness` is what the region means, not how it draws. `hard` marks the
    shared space of a conversation -- the one place the robot must not pass and
    the one block F should take as a CBF constraint. `soft` is one person's
    space: expensive to enter, sometimes worth it.
    """

    zone_id: str
    track_ids: tuple
    scene_type: str
    hardness: str
    confidence: float
    valid_from: float
    valid_to: float
    trajectory_of_zone: tuple

    def to_dict(self) -> dict:
        return {
            'zone_id': self.zone_id,
            'track_ids': list(self.track_ids),
            'scene_type': self.scene_type,
            'hardness': self.hardness,
            'confidence': round(self.confidence, 6),
            'valid_from': round(self.valid_from, 6),
            'valid_to': round(self.valid_to, 6),
            'trajectory_of_zone': [sample.to_dict()
                                   for sample in self.trajectory_of_zone],
        }


@dataclass(frozen=True)
class ConstraintField:
    """K_soc: everything block D says about this instant.

    `horizon_steps` is 1 and `dt` is 0.0 now -- there is no horizon. Both are
    kept so consumers that log or compare a field need no special case.
    """

    timestamp: float
    frame: str
    horizon_steps: int
    dt: float
    zones: tuple

    def to_dict(self) -> dict:
        return {
            'timestamp': round(self.timestamp, 6),
            'frame': self.frame,
            'horizon_steps': self.horizon_steps,
            'dt': round(self.dt, 6),
            'zones': [zone.to_dict() for zone in self.zones],
        }


def sample_times(config: ConstraintFieldConfig) -> tuple:
    """Every instant a zone is described at. Just the present now."""
    times = tuple(config.prediction_times)
    return times if times and times[0] == 0.0 else (0.0,) + times


def _pair_key(people, first: int, second: int) -> str:
    def name(index):
        track = getattr(people[index], 'track_id', '')
        return track if track else f'#{index}'
    return f'z_g{name(first)}_{name(second)}'


def _talking_pairs(people, config: ConstraintFieldConfig):
    """(first, second) index pairs of people holding a conversation.

    Pairwise, not clustered: three people in a circle give three pairs whose
    midpoint Gaussians already cover the centre, and a clustering step would be
    one more thing to get wrong at this crowd size.
    """
    talking = [index for index, person in enumerate(people)
               if getattr(person, 'scene_type', '') == 'talking']
    pairs = []
    for a_index in range(len(talking)):
        for b_index in range(a_index + 1, len(talking)):
            first, second = talking[a_index], talking[b_index]
            if math.hypot(people[first].x - people[second].x,
                          people[first].y - people[second].y) \
                    > config.group_max_distance:
                continue
            pairs.append((first, second))
    return pairs


def compile_zones(people, config: ConstraintFieldConfig, *,
                  timestamp: float = 0.0,
                  frame: str = 'base_link') -> ConstraintField:
    """Ground block B and block C into K_soc. The whole of block D.

    A conversation is ONE Gaussian at the pair midpoint (`hard`). Everybody not
    in a conversation gets one Gaussian of their own (`soft`): `passing` shape
    if they carry that label or are simply moving, `waiting` shape if they
    carry that label, the circular base otherwise. A person in a conversation
    gets NO personal zone -- the pair zone is the region.
    """
    times = sample_times(config)
    pairs = _talking_pairs(people, config)
    paired = {index for pair in pairs for index in pair}
    zones = []

    for first, second in pairs:
        one, other = people[first], people[second]
        separation = math.hypot(one.x - other.x, one.y - other.y)
        confidence = min(getattr(one, 'scene_confidence', 0.0),
                         getattr(other, 'scene_confidence', 0.0))
        sigma_h, sigma_s, sigma_r = _talking_sigmas(
            separation, _factor(confidence, config), config)
        sample = ZoneSample(
            t=0.0, shape='gaussian',
            center=(0.5 * (one.x + other.x), 0.5 * (one.y + other.y)),
            size=(sigma_h, sigma_s, sigma_r),
            weight=config.weight_for('talking'),
            orientation=math.atan2(other.y - one.y, other.x - one.x))
        zones.append(Zone(
            zone_id=_pair_key(people, first, second),
            track_ids=(getattr(one, 'track_id', ''),
                       getattr(other, 'track_id', '')),
            scene_type='talking', hardness='hard',
            confidence=confidence,
            valid_from=times[0], valid_to=times[0],
            trajectory_of_zone=(sample,)))

    for index, person in enumerate(people):
        if index in paired:
            continue
        scene_type = getattr(person, 'scene_type', '')
        confidence = getattr(person, 'scene_confidence', 0.0)
        factor = _factor(confidence, config)
        speed = math.hypot(person.vx, person.vy)
        moving = speed >= config.still_speed

        if scene_type == 'waiting':
            sigma_h, sigma_s, sigma_r = _waiting_sigmas(
                config.waiting_distance, factor, config)
            orientation = person.facing
        elif scene_type == 'talking':
            # Single talking person: orient forward along their facing direction (or
            # velocity vector if moving) to maintain an active interaction personal space.
            if moving:
                sigma_h, sigma_s, sigma_r = _passing_sigmas(speed, factor, config)
                orientation = math.atan2(person.vy, person.vx)
            else:
                sigma_h, sigma_s, sigma_r = _waiting_sigmas(
                    config.waiting_distance, factor, config)
                orientation = person.facing
        elif scene_type == 'passing' or moving:
            sigma_h, sigma_s, sigma_r = _passing_sigmas(speed, factor, config)
            orientation = (math.atan2(person.vy, person.vx) if moving
                           else person.facing)
        else:
            sigma_h, sigma_s, sigma_r = _base_sigmas(config)
            orientation = 0.0

        sample = ZoneSample(
            t=0.0, shape='gaussian',
            center=(person.x, person.y),
            size=(sigma_h, sigma_s, sigma_r),
            weight=config.weight_for(scene_type),
            orientation=orientation)
        track = getattr(person, 'track_id', '')
        zones.append(Zone(
            zone_id=f'z_p{track}' if track else f'z_p#{index}',
            track_ids=(track,),
            scene_type=scene_type, hardness='soft',
            confidence=confidence,
            valid_from=times[0], valid_to=times[0],
            trajectory_of_zone=(sample,)))

    return ConstraintField(timestamp=timestamp, frame=frame,
                           horizon_steps=1, dt=0.0, zones=tuple(zones))


def render_zones(field: ConstraintField, x_axis, y_axis, times) -> np.ndarray:
    """Rasterise K_soc into one grid per requested time. Block E's half.

    Overlap is resolved by maximum, not sum: standing where two Gaussians cross
    costs what the worse of them costs, which keeps the penalty bounded whatever
    the crowd size. `hardness` is not read here -- both kinds draw the same.
    """
    grid_shape = (x_axis.size, y_axis.size)
    channels = np.zeros((len(times),) + grid_shape, dtype=np.float32)
    if not field.zones:
        return channels

    x = x_axis[:, None]
    y = y_axis[None, :]
    for index, horizon_time in enumerate(times):
        layer = channels[index]
        for zone in field.zones:
            sample = _sample_at(zone, horizon_time)
            sigma_h, sigma_s, sigma_r = sample.size
            np.maximum(layer, _gaussian_region(
                x, y, sample.center[0], sample.center[1], sample.orientation,
                sigma_h=sigma_h, sigma_s=sigma_s, sigma_r=sigma_r,
                weight=sample.weight), out=layer)
    return channels


def _sample_at(zone: Zone, horizon_time: float) -> ZoneSample:
    """The zone's description at one instant. There is only t = 0.0 now."""
    for sample in zone.trajectory_of_zone:
        if abs(sample.t - horizon_time) < 1e-9:
            return sample
    available = [sample.t for sample in zone.trajectory_of_zone]
    raise ValueError(
        f'zone {zone.zone_id} was not compiled at t={horizon_time}; '
        f'it holds {available}.')


def intrusion_at_zones(field: ConstraintField) -> float:
    """The field value at the robot right now, in [0, max weight].

    Read at the robot's exact position (the origin of the field frame), not out
    of the rasterised grid: at 0.2 m per cell the robot's centre falls between
    four cells and none holds the value it deserves.
    """
    if not field.zones:
        return 0.0
    origin = np.zeros(1, dtype=np.float32)
    return float(render_zones(field, origin, origin, (0.0,))[0, 0, 0])


def intrusion_by_type(field: ConstraintField) -> dict:
    """Same reading as intrusion_at_zones(), split into C_talk/C_view/C_cross.

    12-09-2026, feeds reward.py's r_social,t = -lambda_s*(w_talk*C_talk +
    w_view*C_view + w_cross*C_cross) from the báo cáo. Each is the MAX
    weighted Gaussian value at the robot's exact position among zones of
    that bucket, 0.0 if none. `cross` is the catch-all -- passing, walking,
    an unrecognised label, AND a motionless unlabelled person all land there
    on purpose: a person no VLM call has classified yet is still a zone to
    avoid, not free space.
    """
    values = {'talking': 0.0, 'waiting': 0.0, 'cross': 0.0}
    for zone in field.zones:
        bucket = ('talking' if zone.scene_type == 'talking' else
                  'waiting' if zone.scene_type == 'waiting' else 'cross')
        sample = _sample_at(zone, 0.0)
        sigma_h, sigma_s, sigma_r = sample.size
        value = float(_gaussian_region(
            0.0, 0.0, sample.center[0], sample.center[1], sample.orientation,
            sigma_h=sigma_h, sigma_s=sigma_s, sigma_r=sigma_r,
            weight=sample.weight))
        values[bucket] = max(values[bucket], value)
    return values


def constraint_field(people, x_axis, y_axis,
                     config: ConstraintFieldConfig) -> np.ndarray:
    """Compile K_soc and rasterise it: one present-time grid.

    Kept as one call because that is what the observation builder wants; the
    halves are compile_zones and render_zones.
    """
    return render_zones(compile_zones(people, config),
                        x_axis, y_axis, config.prediction_times)


def intrusion_at_robot(people, config: ConstraintFieldConfig) -> float:
    """Field value at the robot, straight from the people list."""
    return intrusion_at_zones(compile_zones(people, config))
