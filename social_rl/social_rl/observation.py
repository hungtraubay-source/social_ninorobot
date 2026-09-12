"""Turn the live topics into the observation the recurrent policy sees.

Training and deployment share this module on purpose. The observation layout is
the contract between the two, and a policy fed a differently shaped observation
than it was trained on fails silently -- it still drives, just badly. Training
writes the configuration it used next to the checkpoint and the agent node
loads that file back, so the two cannot drift apart.

The observation is the two arrows that reach block E in the architecture, and
nothing else:

  grid   (6, N, N)  an egocentric top-down view, robot at the centre facing up
                    channel 0     block H: local occupancy, from the laser
                    channels 1-5  block D: the social constraint field K_soc,
                                  at t+0.0, t+0.5, t+1.0, t+1.5 and t+2.0 s
  vector (5,)       block H: goal distance and bearing, plus the robot's
                    measured v and w

Occupancy is kept as its own channel rather than folded into the constraint
field because the two mean different things: a wall will still be there in two
seconds and a person will not, and a policy given one number for both cannot
learn the difference. They travel in one tensor only so that a single small CNN
reads them together -- which is what lets it learn "this person is about to be
against that wall, there is no room to pass on that side".

Everything here is expressed in the robot frame (`base_link`), never in `map`:
that is what makes the same policy usable on hardware, where the map origin and
the goal come from somewhere else entirely.
"""

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from social_rl.constraint_field import (ConstraintFieldConfig, compile_zones,
                                        render_zones)

GOAL_FEATURES = 3       # distance, sin(bearing), cos(bearing)
ROBOT_FEATURES = 2      # measured linear, measured angular
VECTOR_FEATURES = GOAL_FEATURES + ROBOT_FEATURES

# Channel 0 is block H's occupancy; everything after it is block D's field.
OCCUPANCY_CHANNEL = 0
SOCIAL_CHANNEL_START = 1


@dataclass(frozen=True)
class ObservationConfig:
    """Shape and scaling of the observation."""

    # 32 x 32 cells at 0.2 m is a 6.4 x 6.4 m box centred on the robot: far
    # enough to see a person two seconds before reaching them at 0.5 m/s, small
    # enough that three strided convolutions reduce it to 4 x 4. It also has to
    # cover the prediction horizon -- at 2.0 s a person walking 1.0 m/s towards
    # the robot moves 2.0 m, and a box that cannot hold that has channels whose
    # content falls off the edge.
    grid_size: int = 32
    grid_resolution: float = 0.2
    # Beams longer than this are dropped before rasterising. Anything past the
    # edge of the box would be dropped anyway; this only saves the arithmetic.
    scan_max_range: float = 5.0
    # Returns landing this close to base_link are the robot seeing itself, and
    # are dropped before anything reads the scan. The base collision box is
    # 0.20 x 0.20 centred on base_link, so its corners sit at 0.1414 m; 0.15
    # covers them with 9 mm to spare and stays well under the 0.30 m collision
    # threshold, which keeps a real wall detectable.
    #
    # It is not a hypothetical: the lidar plane is only 39 mm above the top of
    # that box, so every teleport bounce at reset dips it into the robot's own
    # front edge and returns 0.10-0.30 m from a room that is empty.
    scan_self_hit_radius: float = 0.15
    goal_max_distance: float = 10.0
    max_linear_speed: float = 0.5
    max_angular_speed: float = 1.0
    robot_frame: str = 'base_link'

    # Block D. Its prediction_times decide how many channels the grid has, so
    # this is part of the network shape, not just of the semantics.
    constraint_field: ConstraintFieldConfig = field(
        default_factory=ConstraintFieldConfig)

    @property
    def grid_channels(self) -> int:
        """1 occupancy channel from block H plus block D's prediction slices."""
        return 1 + self.constraint_field.channels

    @property
    def grid_extent(self) -> float:
        """Side of the observed box in metres."""
        return self.grid_size * self.grid_resolution

    @property
    def dimension(self) -> int:
        """Total scalar count. For logging only -- the policy reads a dict."""
        return self.grid_channels * self.grid_size ** 2 + VECTOR_FEATURES

    def to_dict(self) -> dict:
        values = asdict(self)
        values['constraint_field'] = self.constraint_field.to_dict()
        return values

    @classmethod
    def from_dict(cls, values: dict) -> 'ObservationConfig':
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                f'unknown observation keys: {sorted(unknown)}. '
                f'Valid keys: {sorted(known)}')
        values = dict(values)
        values['constraint_field'] = ConstraintFieldConfig.from_dict(
            values.get('constraint_field') or {})
        return cls(**values)


@dataclass
class RelativeEntity:
    """A person in the robot frame, metres and metres per second.

    `facing` and `scene_type` are what block C contributes. Both have a
    defined meaning when they are absent: facing 0.0 is only read for somebody
    standing still, and an empty scene_type selects the neutral region shape.

    `track_id` is block B's, and it is what makes a zone in block D's output
    traceable back to the person it protects and stable from frame to frame.
    Empty is allowed -- a caller that builds these by hand for a test gets an
    index-based zone id instead, which is stable within one compile but not
    across them.
    """

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    facing: float = 0.0
    scene_type: str = ''
    track_id: str = ''
    # How much block C's ruling on scene_type is worth. Carried through to the
    # zone so a consumer can weigh a region built on a 0.55 verdict against one
    # the simulator handed over at 1.0.
    scene_confidence: float = 0.0

    @property
    def distance(self) -> float:
        return math.hypot(self.x, self.y)


@dataclass
class ObservationInput:
    """Everything one observation is built from, already in the robot frame."""

    # (N, 2) laser endpoints, x forward and y left from base_link.
    scan_points: np.ndarray
    goal_x: float
    goal_y: float
    people: list = field(default_factory=list)
    linear: float = 0.0
    angular: float = 0.0


def cell_centres(config: ObservationConfig):
    """Robot-frame coordinates of every row and column centre.

    Row 0 is the far edge in front of the robot and column 0 the far edge to
    its left, so the array reads like a photograph taken from above with the
    robot facing up the page.
    """
    half = config.grid_size / 2.0
    offsets = (half - 0.5 - np.arange(config.grid_size, dtype=np.float32))
    axis = offsets * config.grid_resolution
    return axis, axis          # x per row, y per column


def occupancy_channel(points: np.ndarray,
                      config: ObservationConfig) -> np.ndarray:
    """Rasterise laser endpoints into a binary hit map. This is block H.

    Hits only: an empty cell means "no return here", which covers both free
    space and the shadow behind an obstacle. Ray tracing the free space would
    triple the cost of every step to tell the policy something it can infer
    from the hits themselves.
    """
    grid = np.zeros((config.grid_size, config.grid_size), dtype=np.float32)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return grid
    half = config.grid_size / 2.0
    rows = np.floor(half - points[:, 0] / config.grid_resolution).astype(int)
    cols = np.floor(half - points[:, 1] / config.grid_resolution).astype(int)
    inside = ((rows >= 0) & (rows < config.grid_size)
              & (cols >= 0) & (cols < config.grid_size))
    grid[rows[inside], cols[inside]] = 1.0
    return grid


def social_channels(field, config: ObservationConfig) -> np.ndarray:
    """Rasterise block D's zones. This is block E's first step, not block D's last.

    The zone list is the interface; this turns it into something a CNN can
    read. Channel 0 of these is t = 0.0, the instant the reward is charged at,
    so what the policy SEES contains what it is CHARGED for. It was left out
    once, on the argument that the occupancy channel and the t+0.5 s slice
    between them already imply where people are standing; render_zones carries
    the measurement showing that is only true of people who are not moving.
    """
    x_axis, y_axis = cell_centres(config)
    return render_zones(field, x_axis, y_axis, config.constraint_field.prediction_times)


def build_observation(data: ObservationInput, config: ObservationConfig,
                      field=None) -> dict:
    """Flatten one snapshot into the policy input.

    `field` is block D's already-compiled zones. Passing it in rather than
    compiling here is what lets one caller charge the reward on the SAME object
    it drew the channels from: compiling twice would work until the two calls
    were handed people lists a control period apart, and then the policy would
    be paying for a situation it was never shown.
    """
    if field is None:
        field = compile_zones(data.people, config.constraint_field)
    grid = np.concatenate(
        [occupancy_channel(data.scan_points, config)[None, ...],
         social_channels(field, config)]).astype(np.float32)

    goal_distance = math.hypot(data.goal_x, data.goal_y)
    goal_bearing = math.atan2(data.goal_y, data.goal_x)
    vector = np.array(
        [min(goal_distance, config.goal_max_distance) / config.goal_max_distance,
         math.sin(goal_bearing),
         math.cos(goal_bearing),
         np.clip(data.linear / config.max_linear_speed, -1.0, 1.0),
         np.clip(data.angular / config.max_angular_speed, -1.0, 1.0)],
        dtype=np.float32)

    expected = (config.grid_channels, config.grid_size, config.grid_size)
    if grid.shape != expected:
        raise RuntimeError(
            f'grid shape {grid.shape} does not match the declared {expected}')
    return {'grid': grid, 'vector': vector}
