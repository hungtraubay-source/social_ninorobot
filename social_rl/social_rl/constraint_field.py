"""Block D: grounding, and compiling the spatio-temporal social constraint field.

Input is what block B and block C produce -- where people are, where they are
going, and what social situation each one is in. Output is K_soc as a LIST OF
ZONES: geometry, not an image.

    people (robot frame, with scene_type)  ->  ConstraintField
                                               .zones[].trajectory_of_zone[]

    ConstraintField(timestamp, frame, horizon_steps=4, dt=0.5, zones=[
        Zone(zone_id, track_ids, scene_type, hardness, confidence,
             valid_from=0.0, valid_to=2.0, trajectory_of_zone=[
                 ZoneSample(t, shape, center, size, core, orientation), ...])])

The raster the CNN eats is produced from that by render_zones, which is block
E's first step rather than block D's last. Two things fall out of the split.
A zone can carry its track_ids, its confidence and its hardness -- none of
which a float per cell has anywhere to put -- so block F can take the same
regions as CBF constraints without inheriting a 0.2 m grid. And the instant the
reward is charged at, t = 0.0, lives in the same structure as the four
predicted ones, so what the policy SEES and what it is CHARGED for cannot drift
apart.

Five samples per zone for five channels: t = 0.0 is the present, and it IS
rasterised -- see render_zones for the measurement that put it back.
horizon_steps counts intervals, so 2.0 s at 0.5 s spacing is 4.

The static occupancy grid is deliberately NOT in here -- that belongs to block
H, which reaches block E by its own path. Mixing the two would make the field
say "a person is here" and "a wall is here" with the same number, and the
policy could no longer learn that one of them moves.

Why scene_type at all, rather than deriving everything from position and
velocity: two people standing 1.5 m apart are either holding a conversation --
in which case the space between them is the one place the robot must not go --
or waiting at a counter with their backs turned, in which case that same space
is the natural way through. No amount of geometry separates those two. That
judgement is what block C exists to make; during RL training the Gazebo
ground-truth publisher makes it instead, because it knows which scenario it was
asked to play.

The region shape is the O/P/R geometry already tuned for the Nav2 costmap in
social_navigation/src/social_layer.cpp, made continuous. There the three zones
are three discrete costs (254 / 250 / 100); a policy learns better from a
gradient it can slide down than from three plateaus, so here the value ramps
from 1.0 at the o-space boundary to 0.0 at the outer ellipse.

A conversation REPLACES its members' personal zones with one region covering
both of them, rather than adding a disc on top of them. See compile_zones for
the measurement that decided the shape of it.
"""

import math
from dataclasses import asdict, dataclass, field

import numpy as np

# One channel per instant, the present included. Kept as a module constant
# because the CNN input shape is built from it in three places.
DEFAULT_PREDICTION_TIMES = (0.0, 0.5, 1.0, 1.5, 2.0)

# Multipliers on (front, side, rear) reach, per scene_type. These are the whole
# of the "grounding" step: the same person, in the same place, moving at the
# same speed, is given a different region depending on what they are doing.
#
#   talking       Engaged with somebody. The region leans forward, towards the
#                 partner, and the pair also gets a shared o-space below.
#   backs_turned  Facing away. Passing behind is the socially correct move, so
#                 the rear reach is cut hard while the front is left alone.
#   passing       Somebody walking past, crossing the path or closing head on.
#                 Merged from `crossing` and `approaching` 10-09-2026; kept the
#                 crossing multipliers, so the region is widened sideways
#                 because the field is built before the direction is settled.
#   walking       Moving, nothing more known about it.
#   ''            Nothing known at all -- what a run without block C gets.
DEFAULT_SCENE_SCALES = {
    'talking': (1.25, 1.0, 0.9),
    'backs_turned': (1.0, 0.9, 0.45),
    'passing': (1.1, 1.35, 0.8),
    'walking': (1.15, 1.0, 0.85),
    '': (1.0, 1.0, 1.0),
}


@dataclass(frozen=True)
class ConstraintFieldConfig:
    """Geometry of the social constraint field. Distances in metres."""

    # The prediction horizon, as the plan specifies it: 2.0 s ahead, sampled
    # every 0.5 s. Changing the LENGTH of this tuple changes the number of
    # channels and therefore the CNN, so a checkpoint does not load into a
    # different one. Changing the VALUES keeps the shape and only re-times what
    # each channel means -- still a different observation, and train.py refuses
    # to resume across it.
    prediction_times: tuple = DEFAULT_PREDICTION_TIMES

    # Inner core, value 1.0 everywhere inside it. 0.45 m is individual_o_radius
    # in social_layer.cpp: the space a person occupies plus the room they need
    # to take a step.
    o_radius: float = 0.45
    # Outer ellipse, value 0.0 on the boundary. r_front / r_side / r_rear in
    # social_layer.cpp. Asymmetric because personal space is: people mind
    # somebody in front of them far more than somebody behind.
    r_front: float = 1.5
    r_side: float = 1.0
    r_rear: float = 0.75

    # Per-situation multipliers on the three radii above.
    scene_scales: dict = field(
        default_factory=lambda: dict(DEFAULT_SCENE_SCALES))

    # --- the group o-space, which is the point of the whole exercise ---
    # Two people whose scene_type is `talking` and who stand closer together
    # than this share an o-space: the disc between them, which the robot must
    # go around rather than through.
    group_max_distance: float = 3.0
    # Grown past the halfway point so that the region actually blocks the gap
    # rather than leaving a corridor down the middle of it.
    group_margin: float = 0.35

    # How much a predicted region grows per second of prediction, to stand in
    # for the fact that a constant-velocity guess gets worse the further ahead
    # it looks. At 0.15 the t+2.0 s slice is 0.30 m wider than the one at t.
    # Set to 0.0 to see the raw constant-velocity prediction.
    uncertainty_growth: float = 0.15

    # Speed below which a person counts as standing still, so that tracker
    # noise on a stationary person does not smear their region across two
    # metres of the t+2.0 s channel.
    still_speed: float = 0.1

    @property
    def channels(self) -> int:
        return len(self.prediction_times)

    def scales_for(self, scene_type: str):
        """(front, side, rear) multipliers, falling back to the neutral set."""
        return self.scene_scales.get(
            scene_type, self.scene_scales.get('', (1.0, 1.0, 1.0)))

    def to_dict(self) -> dict:
        values = asdict(self)
        # YAML round-trips a list, not a tuple, and a config that does not
        # compare equal after a save/load cycle makes the resume check in
        # train.py reject checkpoints it should accept.
        values['prediction_times'] = list(self.prediction_times)
        values['scene_scales'] = {
            key: list(value) for key, value in self.scene_scales.items()}
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
        if 'scene_scales' in values:
            values['scene_scales'] = {
                key: tuple(value)
                for key, value in values['scene_scales'].items()}
        return cls(**values)


def _elliptical_region(x, y, centre_x, centre_y, facing,
                       front, side, rear, core):
    """One person's region: 1.0 inside `core`, 0.0 on the ellipse, ramp between.

    `x` and `y` are the cell-centre grids. The ellipse is expressed in the
    person's own frame, which is why `facing` matters and the raw distance does
    not: the same 1.0 m is intimate in front of somebody and unremarkable
    behind them.
    """
    delta_x = x - centre_x
    delta_y = y - centre_y
    cos_facing, sin_facing = math.cos(facing), math.sin(facing)
    # Into the person's frame: +forward is where they are looking.
    forward = delta_x * cos_facing + delta_y * sin_facing
    sideways = -delta_x * sin_facing + delta_y * cos_facing

    # Semi-axis along the person's forward axis, which side depends on sign.
    along = np.where(forward >= 0.0, front, rear)
    distance = np.hypot(forward, sideways)
    # Reach of the ellipse in the direction of each cell. Guarding the
    # division keeps the person's own cell finite.
    safe = np.maximum(distance, 1e-6)
    cos_angle = forward / safe
    sin_angle = sideways / safe
    # Exactly at the centre there is no direction, so both cosines are zero and
    # this sum is too -- which made `reach` infinite and the ramp below inf/inf,
    # i.e. NaN. Harmless while every region was centred on a person, because
    # the robot is never standing inside somebody. It stopped being harmless
    # when a group zone put a centre in the gap BETWEEN two people, which is
    # the one cell this whole system exists to price: driving into the middle
    # of a conversation returned NaN, and NaN * social_penalty is a NaN reward
    # that takes the run down with it. The floor caps `reach` at 1e6 instead,
    # which lands the centre on 1.0 -- correct, since the centre is inside
    # every region by definition. Never binds elsewhere: away from the centre
    # cos^2 + sin^2 is 1, so the sum is at least 1/max(front, side, rear)^2.
    denominator = (cos_angle / along) ** 2 + (sin_angle / side) ** 2
    reach = 1.0 / np.sqrt(np.maximum(denominator, 1e-12))

    span = np.maximum(reach - core, 1e-6)
    return np.clip((reach - distance) / span, 0.0, 1.0)




# --------------------------------------------------------------- block D out
#
# K_soc leaves this module as a LIST OF ZONES, not as an image. The raster the
# CNN eats is produced from it by render_zones, which belongs to block E: a
# zone says where a region is and what it means, and rasterising is one way of
# reading that, not the thing itself. Keeping the two apart is what lets block
# F take the same regions as CBF constraints without inheriting a 0.2 m grid,
# and what lets a zone carry its track_ids, its confidence and its hardness --
# all of which a float per cell has nowhere to put.


@dataclass(frozen=True)
class ZoneSample:
    """One zone at one instant of the prediction horizon.

    `size` is (front, side, rear) rather than the two semi-axes an ellipse
    normally needs, because these regions are asymmetric front to back and that
    asymmetry is the entire point: a person minds somebody standing in front of
    them far more than somebody behind, and `backs_turned` cuts the rear reach
    to 0.45 of nominal precisely so that passing behind is affordable. Two
    numbers cannot hold that, and flattening it would make every scene_type
    decoration again.

    `core` is the inner radius the value is 1.0 inside, with a linear ramp from
    there to 0.0 on the boundary. Where the boundary in some direction is
    nearer than `core` -- which happens on the short axis of a group zone --
    the region is simply solid out to the boundary in that direction.
    """

    t: float
    shape: str
    center: tuple
    size: tuple
    core: float
    orientation: float

    def to_dict(self) -> dict:
        return {
            't': round(self.t, 6),
            'shape': self.shape,
            'center': [round(value, 6) for value in self.center],
            'size': [round(value, 6) for value in self.size],
            'core': round(self.core, 6),
            'orientation': round(self.orientation, 6),
        }


@dataclass(frozen=True)
class Zone:
    """One social region, tracked across the whole horizon.

    `hardness` is what the region means, not how it is drawn -- both kinds
    rasterise identically. `hard` marks the shared space of a conversation:
    the one place the robot must not pass through, and the one block F should
    take as a CBF constraint rather than a cost. `soft` is one person's
    personal space, which is expensive to enter and sometimes worth it.
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

    `horizon_steps` counts INTERVALS, not samples, so a 2.0 s horizon at 0.5 s
    spacing is 4 steps and 5 samples -- the extra one is t = 0.0, the present.
    That sample is not rasterised into a channel; it is what the reward is
    charged on, and having it in the same structure as the predicted ones is
    what stops the field the policy SEES from drifting away from the depth it
    is CHARGED for.
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
    """Every instant a zone is described at: the present, then the horizon."""
    times = tuple(config.prediction_times)
    return times if times and times[0] == 0.0 else (0.0,) + times


def _predict(person, horizon: float, config: ConstraintFieldConfig):
    """One person moved forward at constant velocity."""
    speed = math.hypot(person.vx, person.vy)
    moving = speed >= config.still_speed
    return _Predicted(
        x=person.x + (person.vx * horizon if moving else 0.0),
        y=person.y + (person.vy * horizon if moving else 0.0),
        # Facing follows the direction of travel once somebody is actually
        # walking; a standing person keeps the facing block C reported, which
        # for a conversation is the whole point.
        facing=(math.atan2(person.vy, person.vx) if moving else person.facing),
        vx=person.vx, vy=person.vy,
        scene_type=getattr(person, 'scene_type', ''))


def _pair_key(people, first: int, second: int) -> str:
    def name(index):
        track = getattr(people[index], 'track_id', '')
        return track if track else f'#{index}'
    return f'z_g{name(first)}_{name(second)}'


def _talking_pairs(people, config: ConstraintFieldConfig):
    """Indices of people holding a conversation, as (first, second) pairs.

    Pairwise rather than clustered: three people in a circle produce three
    pairs whose zones cover the middle between them anyway, and a proper
    clustering step would be a second thing to get wrong for no visible gain at
    this crowd size.
    """
    talking = [index for index, person in enumerate(people)
               if getattr(person, 'scene_type', '') == 'talking']
    pairs = []
    for first_index in range(len(talking)):
        for second_index in range(first_index + 1, len(talking)):
            first, second = talking[first_index], talking[second_index]
            if math.hypot(people[first].x - people[second].x,
                          people[first].y - people[second].y) > config.group_max_distance:
                continue
            pairs.append((first, second))
    return pairs


def compile_zones(people, config: ConstraintFieldConfig, *,
                  timestamp: float = 0.0,
                  frame: str = 'base_link') -> ConstraintField:
    """Ground block B and block C into K_soc. This is the whole of block D.

    Two kinds of zone come out, and a person in a conversation gets both:

      group      Two people talking within group_max_distance. One ellipse
                 covering BOTH of them and the space between, oriented along
                 the line joining them, solid out to the o-space and ramping
                 from there. Marked `hard`.
      personal   EVERYBODY, in a group or not. The asymmetric O/P/R ellipse,
                 scaled by their scene_type. Marked `soft`.

    A group is laid ON TOP OF its members' personal zones rather than replacing
    them, and render_zones takes the maximum, so the conversation adds to what
    each person already claims instead of standing in for it.

    It used to replace them, and that left a hole exactly where the pair stands
    furthest apart. The group ellipse reaches sep/2 + rear along the line
    joining the pair but only r_side * side_scale = 1.00 m ACROSS it, whatever
    the separation. Measured 30-08-2026 on a pair 2.77 m apart, the geometry the
    scenario actually produces: a robot passing 0.89 m off one person's flank
    scored 0.000, while the same robot scored 0.200 if block C had said nothing
    at all, and 0.511 if it had misread the pair as `crossing`. Recognising the
    conversation correctly bought the least protection of any answer -- and the
    better block C gets, the more often that hole is the one the robot drives
    through. The union keeps the o-space at 1.000 and puts the flank back to
    0.200; `backs_turned`, `passing`, `walking` and the empty label are
    bit-identical either way, since only `talking` ever forms a group.

    The group zone is a bounding ellipse rather than the bare o-space disc for
    a measured reason. The disc reaches sep/2 + group_margin from the midpoint,
    which for a pair 1.5 m apart is 1.10 m -- 0.35 m past each person. A robot
    standing 0.5 m behind one of them is 1.25 m from the midpoint, so a bare
    disc scores it 0.00 where the personal zone it replaced scored 0.78, and
    brushing past somebody's back mid-conversation would become free. The
    bounding ellipse reaches sep/2 + rear to 1.425 m along the line and keeps
    that point at 0.54.
    """
    times = sample_times(config)
    horizon = max(times)
    steps = max(1, len(times) - 1)
    field_dt = horizon / steps

    pairs = _talking_pairs(people, config)
    front_scale, side_scale, rear_scale = config.scales_for('talking')
    zones = []

    for first, second in pairs:
        samples = []
        for horizon_time in times:
            one = _predict(people[first], horizon_time, config)
            other = _predict(people[second], horizon_time, config)
            separation = math.hypot(one.x - other.x, one.y - other.y)
            growth = config.uncertainty_growth * horizon_time
            # Along the line joining them each person's REAR faces outwards --
            # they are looking at each other -- so the reach along that axis is
            # half the gap plus one rear reach, and the region is symmetric
            # about the midpoint.
            along = separation / 2.0 + config.r_rear * rear_scale + growth
            samples.append(ZoneSample(
                t=horizon_time,
                shape='ellipse',
                center=(0.5 * (one.x + other.x), 0.5 * (one.y + other.y)),
                size=(along, config.r_side * side_scale + growth, along),
                # The shared space of the conversation, solid. This is the one
                # region that exists BETWEEN people rather than around any of
                # them, and it is what stops the policy from treating the gap
                # in a face-to-face pair as a shortcut.
                core=separation / 2.0 + config.group_margin + growth,
                orientation=math.atan2(other.y - one.y, other.x - one.x)))
        zones.append(Zone(
            zone_id=_pair_key(people, first, second),
            track_ids=(getattr(people[first], 'track_id', ''),
                       getattr(people[second], 'track_id', '')),
            scene_type='talking',
            hardness='hard',
            # A region resting on two rulings is worth the weaker of them.
            confidence=min(getattr(people[first], 'scene_confidence', 0.0),
                           getattr(people[second], 'scene_confidence', 0.0)),
            valid_from=times[0],
            valid_to=horizon,
            trajectory_of_zone=tuple(samples)))

    for index, person in enumerate(people):
        scene_type = getattr(person, 'scene_type', '')
        person_front, person_side, person_rear = config.scales_for(scene_type)
        samples = []
        for horizon_time in times:
            predicted = _predict(person, horizon_time, config)
            # The further ahead, the less the prediction is worth, so the
            # region GROWS rather than its value dropping: a policy should keep
            # more clearance from where somebody might be than from where they
            # are.
            growth = config.uncertainty_growth * horizon_time
            samples.append(ZoneSample(
                t=horizon_time,
                shape='ellipse',
                center=(predicted.x, predicted.y),
                size=(config.r_front * person_front + growth,
                      config.r_side * person_side + growth,
                      config.r_rear * person_rear + growth),
                core=config.o_radius,
                orientation=predicted.facing))
        track = getattr(person, 'track_id', '')
        zones.append(Zone(
            zone_id=f'z_p{track}' if track else f'z_p#{index}',
            track_ids=(track,),
            scene_type=scene_type,
            hardness='soft',
            confidence=getattr(person, 'scene_confidence', 0.0),
            valid_from=times[0],
            valid_to=horizon,
            trajectory_of_zone=tuple(samples)))

    return ConstraintField(timestamp=timestamp, frame=frame,
                           horizon_steps=steps, dt=field_dt,
                           zones=tuple(zones))


def render_zones(field: ConstraintField, x_axis, y_axis, times) -> np.ndarray:
    """Rasterise K_soc into one grid per requested time. This is block E's half.

    Overlap is resolved by maximum, not by sum: standing where two regions
    cross costs what the worse of the two costs. That is what keeps the penalty
    bounded whatever the crowd size, and it is why a policy trained beside two
    people does not panic beside four.

    t = 0.0 is one of the times drawn, and it took a measurement to get there.
    It used to be excluded on the grounds that the t+0.5 s slice all but covers
    it -- true for somebody STANDING, false for somebody WALKING. r_rear is
    0.75 m scaled by 0.85 for `walking`, so the tail of the region reaches
    0.64 m behind a person; at 1.0 m/s that tail travels 0.5 m per slice and
    outruns a robot sitting behind them. Measured 30-08-2026, robot 0.3 m
    behind a person walking 1.0 m/s: the reward charged 1.000 while all four
    predicted channels read 0.000. Same hole for `crossing` at 1.2 m/s and
    `approaching` at 1.0 m/s, none for `talking`, whose people do not move.
    The robot following somebody down a corridor is exactly that geometry, so
    the policy was being charged for a situation it could not see.

    `hardness` is not read here. Both kinds draw the same way -- solid inside
    `core`, linear ramp to 0.0 on the boundary -- because the difference
    between them is what a region MEANS to block F, not how deep into it the
    robot is standing.
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
            front, side, rear = sample.size
            np.maximum(layer, _elliptical_region(
                x, y, sample.center[0], sample.center[1], sample.orientation,
                front=front, side=side, rear=rear, core=sample.core),
                out=layer)
    return channels


def _sample_at(zone: Zone, horizon_time: float) -> ZoneSample:
    """The zone's description at one instant of its own trajectory.

    Exact match, not interpolation: a caller asking for a time the field was
    not compiled at is asking about an instant block D never ruled on, and
    inventing one by interpolating would hide a config mismatch between the
    producer and the consumer behind a plausible-looking number.
    """
    for sample in zone.trajectory_of_zone:
        if abs(sample.t - horizon_time) < 1e-9:
            return sample
    available = [sample.t for sample in zone.trajectory_of_zone]
    raise ValueError(
        f'zone {zone.zone_id} was not compiled at t={horizon_time}; '
        f'it holds {available}. The prediction window of the producer and '
        f'the consumer disagree.')


def intrusion_at_zones(field: ConstraintField) -> float:
    """How deep into K_soc the robot is standing right now, in [0, 1].

    Evaluated at t = 0.0, not at any of the predicted instants: the reward
    charges for where the robot IS, while the channels show it where people are
    GOING. And evaluated at the robot's exact position rather than read out of
    the rasterised grid, because at 0.2 m per cell the robot's own centre falls
    on the corner of four cells and none of them holds the value it deserves.
    """
    if not field.zones:
        return 0.0
    origin = np.zeros(1, dtype=np.float32)
    return float(render_zones(field, origin, origin, (0.0,))[0, 0, 0])


def constraint_field(people, x_axis, y_axis,
                     config: ConstraintFieldConfig) -> np.ndarray:
    """Compile K_soc and rasterise it, one grid per prediction time.

    Kept as one call because that is what the observation builder wants; the
    two halves are compile_zones and render_zones, and anything that needs the
    regions themselves rather than a picture of them should use those.
    """
    return render_zones(compile_zones(people, config),
                        x_axis, y_axis, config.prediction_times)


def intrusion_at_robot(people, config: ConstraintFieldConfig) -> float:
    """Depth into K_soc at the robot, straight from the people list."""
    return intrusion_at_zones(compile_zones(people, config))


@dataclass
class _Predicted:
    """A person moved forward to one prediction time. Internal to this module."""

    x: float
    y: float
    facing: float
    vx: float
    vy: float
    scene_type: str
