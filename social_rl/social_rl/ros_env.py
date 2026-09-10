"""Gymnasium environment wrapping the running ROS 2 / Gazebo stack.

The policy is given exactly what the real robot can give it -- /scan, /people,
/odom and a goal in the map frame -- and answers with a Twist. Nothing
here reads a Gazebo actor pose or any other quantity that exists only in
simulation: the one simulation-only piece, moving the base between episodes,
sits behind GazeboWorld and is skipped when randomize_start is false.

One environment, one Gazebo. Vectorised training is not available here (there
is a single simulator on this machine), so RecurrentPPO runs with n_envs=1 and
the LSTM carries the temporal information that frame stacking would otherwise
have to.
"""

import math
import os
import random
import subprocess
import time

import gymnasium as gym
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.parameter import Parameter
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from social_rl.gazebo_world import GazeboWorld
from social_rl.ground_truth import GroundTruthPeople
from social_rl.observation import VECTOR_FEATURES, ObservationConfig
from social_rl.reward import RewardConfig, evaluate_step
from social_rl.ros_interface import (EnvConfig, PerceptionBridge,
                                     scale_action)


def _reward_social_intrusion(state: dict, has_ground_truth: bool) -> float:
    """Choose the authoritative social depth for this environment step.

    Ground-truth mode must fail loudly if the bridge ever stops supplying the
    full scene; silently falling back there would reopen the turn-away
    loophole.
    Perception mode has no simulator-only list and deliberately uses its
    visible/remembered value.
    """
    key = ('social_intrusion_hidden'
           if has_ground_truth else 'social_intrusion')
    return state[key]


class FreeSpace:
    """Poses a robot centre may be dropped on, read from the map_server map.

    Both ends of an episode used to come from two lists of four: 16 routes,
    3.18-5.80 m long, bearing -30 to +57 degrees, and never once a goal behind
    the robot. A goal clicked in RViz is none of those. Drawing both ends out
    of the very map those clicks land on is what makes the two the same
    question.

    Only cells written as free (254) qualify. Unknown (205) passes map_server's
    trinary thresholds as free as well, and it is the unmapped outside of the
    room -- 9615 of this map's cells, next to 68249 that are genuinely free.
    """

    def __init__(self, yaml_path: str, clearance: float,
                 goal_clearance: float):
        with open(yaml_path) as handle:
            meta = yaml.safe_load(handle)
        resolution = float(meta['resolution'])
        origin_x, origin_y = (float(meta['origin'][0]), float(meta['origin'][1]))
        image = self._read_pgm(
            os.path.join(os.path.dirname(yaml_path), meta['image']))
        height = image.shape[0]

        def points_for(margin):
            rows, columns = self._standable(image, resolution, margin)
            points = np.column_stack((
                origin_x + columns * resolution,
                origin_y + (height - 1 - rows) * resolution)).astype(float)
            if not len(points):
                raise RuntimeError(
                    f'{yaml_path} has no cell left after eroding it by '
                    f'{margin:.2f} m; the robot would not fit anywhere.')
            return points

        # TWO SETS, and the difference is not a safety margin. Standing needs
        # room for the body. A GOAL additionally has to be reachable: the
        # episode ends the moment the robot is within reward.goal_distance of
        # it, so a goal that close to a wall is an episode that scores
        # hit_obstacle exactly where it was supposed to score goal.
        self._points = points_for(clearance)
        self._goal_points = points_for(goal_clearance)
        self.area = len(self._points) * resolution * resolution
        self.goal_area = len(self._goal_points) * resolution * resolution

    @staticmethod
    def _read_pgm(path: str) -> np.ndarray:
        with open(path, 'rb') as handle:
            data = handle.read()
        fields, offset = [], 0
        while len(fields) < 4:
            end = data.index(b'\n', offset)
            line, offset = data[offset:end], end + 1
            if not line.startswith(b'#'):
                fields.extend(line.split())
        if fields[0] != b'P5':
            raise RuntimeError(f'{path} is not a binary PGM (P5)')
        width, height = int(fields[1]), int(fields[2])
        return np.frombuffer(data, dtype=np.uint8, count=width * height,
                             offset=offset).reshape(height, width)

    @staticmethod
    def _standable(image, resolution: float, clearance: float):
        """Cells with no occupied cell within `clearance`, as (rows, columns).

        A summed-area table rather than a convolution: the erosion runs once at
        startup, and this keeps scipy out of the trainer's dependencies.
        """
        blocked = image != 254
        radius = max(0, int(round(clearance / resolution)))
        padded = np.pad(blocked, radius, constant_values=True)
        integral = np.pad(
            np.cumsum(np.cumsum(padded, axis=0, dtype=np.int64), axis=1),
            ((1, 0), (1, 0)))
        size = 2 * radius + 1
        height, width = blocked.shape
        occupied_in_window = (integral[size:size + height, size:size + width]
                              - integral[0:height, size:size + width]
                              - integral[size:size + height, 0:width]
                              + integral[0:height, 0:width])
        return np.nonzero(occupied_in_window == 0)

    def sample_pose(self):
        """A free (x, y) and a yaw drawn over the whole circle.

        The yaw is the half that fixes bearing coverage: a start pose facing a
        fixed direction can only ever produce goals in front of the robot.
        """
        x, y = self._points[random.randrange(len(self._points))]
        return float(x), float(y), random.uniform(-math.pi, math.pi)

    def sample_goal(self, x: float, y: float, minimum: float, maximum: float):
        offsets = self._goal_points - np.array([x, y])
        distance = np.hypot(offsets[:, 0], offsets[:, 1])
        candidates = np.nonzero((distance >= minimum) & (distance <= maximum))[0]
        if not candidates.size:
            raise RuntimeError(
                f'no free cell between {minimum:.1f} and {maximum:.1f} m of '
                f'({x:.2f}, {y:.2f}); widen the range or check the map.')
        goal = self._goal_points[
            int(candidates[random.randrange(candidates.size)])]
        return float(goal[0]), float(goal[1])


def resolve_map_path(reference: str) -> str:
    """'<package>/<relative path>' -> an absolute path in that package's share.

    Written this way so a world that moves machines keeps working; an absolute
    path in the YAML has broken the mesh lookup here before.
    """
    package, _, relative = reference.partition('/')
    return os.path.join(get_package_share_directory(package), relative)


def gzclient_running() -> bool:
    return subprocess.run(['pgrep', '-x', 'gzclient'],
                          stdout=subprocess.DEVNULL).returncode == 0


def start_gzclient():
    """Open the Gazebo 3D window on the gzserver that is already running.

    gzclient attaches to a live server, so this works at any point during a run
    and needs nothing from gazebo.launch.py -- start training headless
    (gui:=false) and turn the window on here when you want to watch.

    The PRIME offload variables are the same two gazebo.launch.py sets for the
    server: this machine is `prime-select on-demand`, so without them the
    window renders on the Intel iGPU and steals CPU from the simulation it is
    supposed to be showing.
    """
    return subprocess.Popen(
        ['gzclient'],
        env={**os.environ,
             '__NV_PRIME_RENDER_OFFLOAD': '1',
             '__GLX_VENDOR_LIBRARY_NAME': 'nvidia'},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class SocialAvoidEnv(gym.Env):
    """One episode = drive to a sampled goal without crowding anybody."""

    metadata = {'render_modes': ['human']}

    def __init__(self, env_config: EnvConfig,
                 observation_config: ObservationConfig,
                 reward_config: RewardConfig, node_name='social_rl_env'):
        super().__init__()
        self.env_config = env_config
        self.observation_config = observation_config
        self.reward_config = reward_config

        if not rclpy.ok():
            raise RuntimeError('rclpy.init() must be called before the env')
        self._node = Node(node_name, parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self._logger = self._node.get_logger()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self._node)
        # Blocks B and C. In simulation they are Gazebo itself; `perception`
        # puts the real YOLO + depth tracker back in the loop, which is what
        # you switch to when you want to measure how much the policy loses to
        # detector noise before putting it on the robot.
        if env_config.people_source == 'ground_truth':
            self._ground_truth = GroundTruthPeople(self._node, env_config)
        elif env_config.people_source == 'perception':
            self._ground_truth = None
        else:
            raise ValueError(
                f'env.people_source is "{env_config.people_source}"; it must '
                f'be "ground_truth" or "perception"')
        self._bridge = PerceptionBridge(self._node, self._tf_buffer,
                                        env_config, observation_config,
                                        people_provider=self._ground_truth)
        self._scenario_pub = self._node.create_publisher(
            String, env_config.scenario_topic, 10)
        self._cmd_pub = self._node.create_publisher(
            Twist, env_config.cmd_vel_topic, 10)

        self._world = GazeboWorld(
            self._node, self._tf_buffer,
            entity_name=env_config.entity_name,
            set_entity_state_service=env_config.set_entity_state_service,
            ekf_set_pose_service=env_config.ekf_set_pose_service,
            reseed_localization=env_config.reseed_localization)
        if env_config.randomize_start:
            self._world.wait_for_services()

        # Eroded by the radius at which a step is already scored as a
        # collision, plus a margin for the map being a SLAM scan of the world
        # rather than the world: a start pose that scores hit_obstacle on step
        # 0 is an episode the policy cannot answer for.
        self._free_space = None
        if env_config.free_space_map:
            self._free_space = FreeSpace(
                resolve_map_path(env_config.free_space_map),
                reward_config.obstacle_collision_distance + 0.15,
                reward_config.obstacle_collision_distance
                + reward_config.goal_distance + 0.05)
            self._logger.info(
                f'start and goal drawn from {env_config.free_space_map}: '
                f'{self._free_space.area:.1f} m2 to stand on, '
                f'{self._free_space.goal_area:.1f} m2 to aim at, goals '
                f'{env_config.minimum_goal_distance:.1f}-'
                f'{env_config.maximum_goal_distance:.1f} m away')

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # A dict rather than one flat Box: the grid goes through a CNN and the
        # five scalars go straight to the LSTM, and SocialGridExtractor needs
        # them kept apart to do that. See social_rl/policy.py.
        grid_shape = (observation_config.grid_channels,
                      observation_config.grid_size,
                      observation_config.grid_size)
        self.observation_space = gym.spaces.Dict({
            'grid': gym.spaces.Box(low=0.0, high=1.0, shape=grid_shape,
                                   dtype=np.float32),
            'vector': gym.spaces.Box(low=-1.0, high=1.0,
                                     shape=(VECTOR_FEATURES,),
                                     dtype=np.float32),
        })

        # Only the window this env opened is closed again on shutdown: a
        # gzclient you started yourself in another terminal is yours to keep.
        self.render_mode = 'human' if env_config.render else None
        self._gzclient = None
        if env_config.render:
            self.render()

        self._goal = (0.0, 0.0)
        self._previous_goal_distance = 0.0
        self._previous_goal_bearing = 0.0
        self._steps = 0
        self._episode_reward = 0.0
        self._minimum_person_distance = math.inf
        self._maximum_intrusion = 0.0
        # Full-scene twin of the visible peak above. Reward charges the full
        # per-step value; both peaks stay separate for the out-of-frame check.
        self._maximum_hidden_intrusion = 0.0
        self._hidden_person_steps = 0
        self._minimum_person_distance_all = math.inf
        # Sanity counter for simulated block C, not a metric: a noise model
        # that never fires reads exactly like a clean run.
        self._corrupted_steps = 0
        self._overshoot = 0.0
        self._respawns = 0
        self._scenario = 'none'
        # This episode's start -> goal line in the world frame, handed to the
        # actor plugin so it lays people out around the route the robot is
        # actually driving. None until the first reset has sampled a goal.
        self._route = None

    # ------------------------------------------------------------------ time

    def _spin_for(self, duration: float):
        """Let ROS run for `duration` of simulated time.

        Simulated, not wall clock: gzserver on this machine does not hold real
        time while perception is loaded, and a policy trained on wall-clock
        steps would see a different control period on every run. The wall-clock
        guard only exists to turn a paused simulator into an error, not a hang.
        """
        start = self._node.get_clock().now()
        limit = max(10.0, duration * 20.0)
        wall_deadline = time.monotonic() + limit
        while True:
            rclpy.spin_once(self._node, timeout_sec=0.02)
            elapsed = (self._node.get_clock().now() - start).nanoseconds * 1e-9
            # Elapsed time alone is not enough to say the step is over. A policy
            # update stops this node from spinning for seconds of wall time
            # while Gazebo keeps running, so when it resumes the very first
            # /clock message can satisfy `duration` by itself and leave the
            # sensor cache holding whatever arrived before the pause. Measured:
            # the first step after the first PPO update saw /odom 1.302 s old
            # and the run died. Waiting for both to be fresh is what makes one
            # step mean one step of current data rather than a clock jump.
            if (elapsed >= duration and self._bridge.has_scan()
                    and self._bridge.has_odom()):
                return
            if time.monotonic() > wall_deadline:
                if elapsed < duration:
                    raise RuntimeError(
                        f'simulated clock advanced {elapsed:.3f} s in '
                        f'{limit:.0f} s of wall time. Gazebo is paused or '
                        f'/clock is not being published.')
                stale = [name for name, fresh in
                         (('scan', self._bridge.has_scan()),
                          ('odom', self._bridge.has_odom())) if not fresh]
                raise RuntimeError(
                    f'no fresh {" and ".join(stale)} after {limit:.0f} s of '
                    f'wall time, though the clock advanced {elapsed:.3f} s. '
                    f'Check that Gazebo (terminal 1) is still publishing.')

    def _ground_truth_ready(self) -> bool:
        return self._ground_truth is None or self._ground_truth.ready()

    def _wait_for_data(self):
        deadline = time.monotonic() + self.env_config.startup_timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if (self._bridge.has_scan() and self._bridge.has_odom()
                    and self._bridge.can_transform_map()
                    and self._ground_truth_ready()):
                return
        if not self._bridge.has_scan():
            missing = self.env_config.scan_topic
        elif not self._bridge.has_odom():
            missing = self.env_config.odom_topic
        elif not self._bridge.can_transform_map():
            missing = (f'{self.env_config.goal_frame} -> '
                       f'{self.env_config.robot_frame} TF')
        else:
            missing = (f'{self.env_config.ground_truth_people_topic} and '
                       f'{self.env_config.model_states_topic} (the actor '
                       f'plugin and libgazebo_ros_state.so, both loaded by '
                       f'lirs_test.world)')
        raise RuntimeError(
            f'no {missing} after {self.env_config.startup_timeout:.0f} s. '
            f'Gazebo (terminal 1) has to be up before training starts.')

    # ------------------------------------------------------------------- gym

    def _publish(self, linear: float, angular: float):
        command = Twist()
        command.linear.x = float(linear)
        command.angular.z = float(angular)
        self._cmd_pub.publish(command)

    def _robot_position_in_map(self):
        transform = self._tf_buffer.lookup_transform(
            self.env_config.goal_frame, self.env_config.robot_frame,
            Time()).transform
        return transform.translation.x, transform.translation.y

    def _sample_goal(self, robot_x: float, robot_y: float):
        if self._free_space is not None:
            return self._free_space.sample_goal(
                robot_x, robot_y,
                self.env_config.minimum_goal_distance,
                self.env_config.maximum_goal_distance)
        candidates = [goal for goal in self.env_config.goals
                      if math.hypot(goal[0] - robot_x, goal[1] - robot_y)
                      >= self.env_config.minimum_goal_distance]
        if not candidates:
            raise RuntimeError(
                f'every goal in env.goals is closer than '
                f'{self.env_config.minimum_goal_distance} m to the start pose '
                f'({robot_x:.2f}, {robot_y:.2f}). Add a farther goal or lower '
                f'minimum_goal_distance.')
        return tuple(random.choice(candidates))

    def _refill_scenario(self):
        """Lay the scenario out again once its people have walked off.

        An episode is max_episode_steps * control_period long -- 50 s at the
        defaults -- but a walker crosses its line once and then parks under the
        floor. Measured 28-08-2026: `approaching` lasts 6.4-11.5 s and
        `crossing` 6.7-17.5 s, so two thirds of those episodes used to be an
        empty room while the policy kept collecting reward for driving through
        it. `talking` never emptied, so a uniform draw over the three still
        spent about two thirds of its PEOPLE-time on conversations.

        Re-sending the same name gets a fresh layout from the plugin, which is
        what keeps the situation alive without turning the walker into somebody
        who paces back and forth -- the one-pass behaviour is what a person
        crossing a corridor actually does.

        `none` is exempt: an empty scene is the whole point of that one, and
        refilling it would publish on every step forever.

        The new person keeps the actor name, so block B sees walker_1 jump
        across the room. The plugin already refuses to differentiate a jump
        that large (kMaximumStep), so it reports zero velocity for one publish
        period and then tracks normally.
        """
        # Hỏi CẢNH, không hỏi OBSERVATION.
        #
        # Trước đây chỗ này đọc state['person_distances'], tức danh sách người
        # đã lọc theo nón camera. Kết quả: người chỉ cần đi ra khỏi tầm nhìn là
        # env tưởng cảnh trống và thả lại kịch bản - đo được 157 lần thả lại
        # trong MỘT tập 228 bước, cảnh bị dựng lại liên tục và không tập nào
        # còn ý nghĩa. Người ngoài tầm nhìn vẫn đang ở đó; chỉ khi họ đã đi hết
        # đường và đỗ xuống dưới sàn thì cảnh mới thực sự rỗng.
        if self._scenario in ('', 'none'):
            return
        if self._ground_truth is not None:
            if self._ground_truth.person_positions().shape[0] > 0:
                return
        elif self._bridge.people is not None and self._bridge.people.people:
            return
        message = String()
        message.data = self._with_route(self._scenario)
        self._scenario_pub.publish(message)
        self._respawns += 1

    def _with_route(self, scenario: str) -> str:
        """Append this episode's own start -> goal to the scenario command.

        The plugin lays people out around a route. Its default is one fixed
        line from the world file, but a training episode draws its start pose
        and its goal fresh from 4 x 4 combinations, so people were being placed
        around a line the robot was often not driving.

        Measured 31-08-2026 over the 16 combinations: with the `talking` pair
        offset 0.4-1.4 m from the FIXED line, 30.6% of episodes still put it
        within 0.5 m of the line the robot actually drives -- dead centre,
        where the constraint field is a plateau and no gradient points out of
        it. The policy at checkpoint 197500 drove straight through in 31% of
        `talking` episodes and detoured cleanly in the other 69%. Same
        episodes.

        World frame, because that is what the plugin works in; map_to_world
        reads the world -> map edge from TF rather than copying the spawn
        offset, so changing spawn_x/spawn_y cannot silently misplace people.

        Returns the bare name when the route is not known yet, which is what
        the plugin has always accepted.
        """
        if self._route is None:
            return scenario
        return '{} route {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            scenario, *self._route)

    def _episode_route(self, robot_x: float, robot_y: float, goal) -> tuple:
        """This episode's start -> goal line, converted into the world frame."""
        start_x, start_y, _ = self._world.map_to_world(robot_x, robot_y, 0.0)
        end_x, end_y, _ = self._world.map_to_world(goal[0], goal[1], 0.0)
        return (start_x, start_y, end_x, end_y)

    def _select_scenario(self):
        """Lay a fresh set of people out for this episode.

        The trainer picks the name and the actor plugin picks the geometry, so
        the same four scene types come back with different placements every
        time. Without this the policy meets one conversation, always in the
        same place, and learns that spot rather than the situation.
        """
        if not self.env_config.scenarios:
            return 'none'
        scenario = random.choice(self.env_config.scenarios)
        message = String()
        message.data = self._with_route(scenario)
        self._scenario_pub.publish(message)
        return scenario

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        self._publish(0.0, 0.0)
        # Physics has to run for the teleport to settle, for the actors to
        # start walking and for the localizers to catch up.
        if self.env_config.pause_between_steps:
            self._world.unpause()

        if self.env_config.randomize_start:
            if self._free_space is not None:
                start = self._free_space.sample_pose()
            else:
                # Position from the fixed list; yaw is drawn separately (the
                # yaw written in start_poses is ignored).
                #
                # Full circle (-pi, pi) is what lets the goal land behind the
                # robot so it has to learn to turn around. 04-09-2026: dropped
                # to (-pi/2, pi/2) as a curriculum step -- none_goal plateaued
                # ~0.6 across LR, scenario-mix and heading_gain changes with
                # the full circle, and goals sit roughly towards +x/+y from
                # every start pose, so full-circle yaw was spending a lot of
                # episodes on a near-180 deg pivot (~16 control steps at
                # max_angular_speed 1.0) before any progress reward was
                # reachable. Narrower range still asks for real turning (up to
                # ~90 deg) without that worst case. Widen back towards the
                # full circle once none_goal recovers past 0.90.
                pose = random.choice(self.env_config.start_poses)
                start = (pose[0], pose[1],
                        random.uniform(-math.pi / 2, math.pi / 2))
            self._world.teleport_robot(float(start[0]), float(start[1]),
                                       float(start[2]))
            # The reseeded EKF and AMCL need a few cycles before TF reports the
            # new pose; sampling the goal any earlier measures the distance
            # from where the robot used to be.
            self._spin_for(0.5)

        self._wait_for_data()
        # ORDER MATTERS (31-08-2026). The goal is sampled BEFORE the scenario
        # is asked for, because the scenario command now carries this episode's
        # start -> goal line and the plugin places people around it. Publishing
        # the name first, as this used to, meant the layout was chosen against
        # the route of the PREVIOUS episode. See _with_route for the 30.6%.
        robot_x, robot_y = self._robot_position_in_map()
        self._goal = self._sample_goal(robot_x, robot_y)
        self._route = self._episode_route(robot_x, robot_y, self._goal)
        self._scenario = self._select_scenario()

        # Settle AFTER the spawn, which is what makes the first observation of
        # the episode show people who are already walking rather than a scene
        # frozen at its start pose with every velocity still zero.
        self._spin_for(self.env_config.scenario_settle_time)

        # People from the previous episode are not evidence about this one, and
        # a reset teleports the base -- a remembered person carried across
        # would be recalled at a position that means nothing here.
        self._bridge.forget_people()
        if self._ground_truth is not None:
            self._ground_truth.forget_rulings()

        observation, state = self._bridge.observe(self._goal[0], self._goal[1])
        if self.env_config.pause_between_steps:
            self._world.pause()
        self._previous_goal_distance = state['goal_distance']
        self._previous_goal_bearing = state['goal_bearing']
        self._steps = 0
        self._episode_reward = 0.0
        self._minimum_person_distance = math.inf
        self._maximum_intrusion = 0.0
        # Full-scene twin of the visible peak above. Reward charges the full
        # per-step value; both peaks stay separate for the out-of-frame check.
        self._maximum_hidden_intrusion = 0.0
        self._hidden_person_steps = 0
        self._minimum_person_distance_all = math.inf
        self._corrupted_steps = 0
        self._overshoot = 0.0
        self._respawns = 0
        self._logger.info(
            f'episode start -> scenario {self._scenario}, goal {self._goal}, '
            f'{state["goal_distance"]:.2f} m away')
        return observation, {'goal': self._goal, 'scenario': self._scenario}

    def step(self, action):
        linear, angular = scale_action(action, self.env_config)
        self._publish(linear, angular)

        # Advance the world by exactly one control period. The command is
        # published BEFORE unpausing so that the wheels are already carrying it
        # when time starts moving; publishing after would spend the first
        # milliseconds of the step executing the previous action.
        started = self._node.get_clock().now()
        if self.env_config.pause_between_steps:
            self._world.unpause()
            self._spin_for(self.env_config.control_period)
            self._world.pause()
        else:
            self._spin_for(self.env_config.control_period)
        advanced = (self._node.get_clock().now()
                    - started).nanoseconds * 1e-9
        # How much simulated time the step actually took beyond what was asked
        # for. Two ROS service round trips sit inside the window, so this is
        # never exactly zero; it is logged per episode instead of being
        # asserted away, because the number is what tells you whether the
        # control period is still meaningful.
        self._overshoot = max(
            self._overshoot, advanced - self.env_config.control_period)

        observation, state = self._bridge.observe(self._goal[0], self._goal[1])
        # Ground-truth training has an authoritative full scene, including an
        # authoritative empty list.  Charge it without attenuation so turning
        # away or hiding somebody behind an occluder cannot erase the social
        # penalty.  The perception path has no privileged list and therefore
        # falls back explicitly to what its tracker supplied.
        social_intrusion = _reward_social_intrusion(
            state, has_ground_truth=self._ground_truth is not None)
        outcome = evaluate_step(
            self.reward_config,
            goal_distance=state['goal_distance'],
            previous_goal_distance=self._previous_goal_distance,
            goal_bearing=state['goal_bearing'],
            previous_goal_bearing=self._previous_goal_bearing,
            minimum_scan=state['minimum_scan'],
            social_intrusion=social_intrusion)

        self._previous_goal_distance = state['goal_distance']
        self._previous_goal_bearing = state['goal_bearing']
        self._steps += 1
        self._episode_reward += outcome.reward
        self._minimum_person_distance = min(
            self._minimum_person_distance,
            min(state['person_distances'], default=math.inf))
        self._maximum_intrusion = max(
            self._maximum_intrusion, state['social_intrusion'])
        if state.get('corrupted_labels', 0) > 0:
            self._corrupted_steps += 1
        if 'social_intrusion_hidden' in state:
            self._maximum_hidden_intrusion = max(
                self._maximum_hidden_intrusion,
                state['social_intrusion_hidden'])
            self._minimum_person_distance_all = min(
                self._minimum_person_distance_all,
                min(state['person_distances_all'], default=math.inf))
            if state['hidden_people'] > 0:
                self._hidden_person_steps += 1
        self._refill_scenario()

        truncated = (not outcome.terminated
                     and self._steps >= self.env_config.max_episode_steps)
        if outcome.terminated or truncated:
            self._publish(0.0, 0.0)

        info = {'reward_components': outcome.components}
        if outcome.terminated or truncated:
            reason = outcome.reason or 'timeout'
            closest = (self._minimum_person_distance
                       if math.isfinite(self._minimum_person_distance) else -1.0)
            info['episode_outcome'] = reason
            info['episode_steps'] = self._steps
            info['minimum_person_distance'] = closest
            info['scenario'] = self._scenario
            # Visible/remembered peak depth into K_soc over the episode. Read it
            # with maximum_hidden_intrusion below: either one alone can hide
            # whether the trajectory genuinely kept clear of everybody.
            info['maximum_intrusion'] = self._maximum_intrusion
            # The peak measured against everybody in the scene, including the
            # people the camera cone and walls hid. This is also the per-step
            # field reward charges. The gap to the visible peak answers "is the
            # policy avoiding people, or relying on keeping them out of view".
            info['maximum_hidden_intrusion'] = self._maximum_hidden_intrusion
            info['hidden_person_steps'] = self._hidden_person_steps
            info['corrupted_label_steps'] = self._corrupted_steps
            info['minimum_person_distance_all'] = (
                self._minimum_person_distance_all
                if math.isfinite(self._minimum_person_distance_all) else -1.0)
            info['step_overshoot'] = self._overshoot
            # How many times the scene had to be laid out again because its
            # people had finished walking. 0 for `talking`, which never empties.
            info['scenario_respawns'] = self._respawns
            self._logger.info(
                f'episode ended: {reason} after {self._steps} steps '
                f'({self._scenario}), return {self._episode_reward:.1f}, '
                f'closest person {closest:.2f} m (thuc su '
                f'{self._minimum_person_distance_all:.2f} m), peak intrusion '
                f'{self._maximum_intrusion:.2f} (unseen '
                f'{self._maximum_hidden_intrusion:.2f} over '
                f'{self._hidden_person_steps} steps), '
                f'{self._corrupted_steps} steps with a misread label, '
                f'worst step overshoot '
                f'{self._overshoot * 1e3:.0f} ms, scene refilled '
                f'{self._respawns}x')
        return observation, outcome.reward, outcome.terminated, truncated, info

    def render(self):
        """Show the robot in Gazebo. Idempotent, and cheap to call again."""
        if self._gzclient is not None and self._gzclient.poll() is None:
            return
        if gzclient_running():
            self._logger.info('gzclient is already open, using that window')
            return
        self._gzclient = start_gzclient()
        self._logger.info('opened gzclient (pid '
                          f'{self._gzclient.pid}); expect a lower real-time '
                          'factor while the window is up')

    def close(self):
        # Leave the simulation unpaused. A run that stops with physics held
        # leaves gzserver frozen for whatever you start next, and the symptom
        # -- every topic silent, no error anywhere -- costs an hour to find.
        if self.env_config.pause_between_steps:
            try:
                self._world.unpause()
            except RuntimeError as error:
                self._logger.warn(f'could not unpause on close: {error}')
        self._publish(0.0, 0.0)
        if self._gzclient is not None and self._gzclient.poll() is None:
            self._gzclient.terminate()
            self._gzclient.wait(timeout=5.0)
        self._node.destroy_node()
