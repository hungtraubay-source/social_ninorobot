"""Run a trained Recurrent PPO policy as an ordinary ROS 2 velocity node.

Nothing simulation-specific is imported here: it subscribes to /scan, /people,
/odom and a goal topic and publishes a Twist, so the same node and the same
.zip run on the robot. Point it at a different config file (rl_agent_real.yaml) and the
only things that change are use_sim_time and the cmd_vel topic.

The policy is the only controller: it drives the whole route and avoids both
people and obstacles by itself. No Nav2 runs alongside it.

The LSTM state is carried between control cycles and reset only when a new goal
arrives. Dropping it every cycle would turn the recurrent policy back into a
memoryless one -- it would still drive, just without the memory of a person it
can no longer see, which is the part that was trained.
"""

import os

from geometry_msgs.msg import PoseStamped, Twist
import numpy as np
import rclpy
from rclpy.node import Node
from sb3_contrib import RecurrentPPO
from tf2_ros import Buffer, TransformException, TransformListener
import yaml

from social_rl.observation import (VECTOR_FEATURES,
                                   ObservationConfig)
from social_rl.reward import RewardConfig
from social_rl.ros_interface import (EnvConfig, PerceptionBridge,
                                     scale_action, transform_point)


class SocialRlAgent(Node):
    """Recurrent policy runner: goal in, velocity out."""

    def __init__(self):
        super().__init__('social_rl_agent')
        self.declare_parameter('model_path', '')
        # Written by train.py next to the checkpoint. Loading it rather than
        # the training YAML is what guarantees the observation layout matches
        # the weights.
        self.declare_parameter('env_config', '')
        self.declare_parameter('goal_topic', '/rl_goal_pose')
        # A goal clicked on the map arrives in `map` while the checkpoint was
        # trained with odom-based episode reset. The policy only sees a
        # robot-relative bearing, so carrying the goal across frames changes no
        # observation dimension or scaling.
        self.declare_parameter('goal_frame_override', '')
        # Which topic the people come in on, when it should not be the one the
        # run was configured with. The point of it is testing a trained policy
        # in simulation before block C exists: pointed at /social_gt/people the
        # node reads the SAME labels the policy trained on, so a bad run is the
        # policy's fault rather than a missing VLM's. See the `people` argument
        # in rl_agent.launch.py, which is the only thing meant to set this.
        #
        # Not a second people source -- it is the same subscription, the same
        # message type and the same filtering; only the topic name differs.
        # social_rl/ground_truth.py stays unimportable here, which is what
        # keeps this node runnable on the robot.
        self.declare_parameter('people_topic_override', '')
        # /cmd_vel, not /cmd_vel_safe: on the robot the firmware listens on
        # /cmd_vel, and in simulation this leaves social_velocity_filter in the
        # loop as a last line of defence behind the policy.
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        # The network is two 128-unit layers and one LSTM; on CPU a forward
        # pass is well under a millisecond, and it leaves the 4 GiB card to the
        # VLM. Set cuda:0 only if you have measured a reason to.
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('deterministic', True)

        model_path = os.path.expanduser(
            str(self.get_parameter('model_path').value))
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError(
                f'model_path "{model_path}" does not exist. Pass the .zip '
                f'saved by train_rl, e.g. '
                f'model_path:=~/social_rl_runs/<run>/final_model.zip')

        config_path = os.path.expanduser(
            str(self.get_parameter('env_config').value))
        if not config_path:
            config_path = os.path.join(os.path.dirname(model_path),
                                       'env_config.yaml')
            # Checkpoints live one directory deeper than the file train.py
            # writes, so look up one level before giving up.
            if not os.path.exists(config_path):
                config_path = os.path.join(
                    os.path.dirname(os.path.dirname(model_path)),
                    'env_config.yaml')
        if not os.path.exists(config_path):
            raise RuntimeError(
                f'no env_config.yaml found next to the model (looked at '
                f'"{config_path}"). It is written by train.py and defines the '
                f'observation layout these weights expect.')
        with open(config_path, 'r') as handle:
            saved = yaml.safe_load(handle) or {}
        self._env_config = EnvConfig.from_dict(saved.get('env', {}))
        goal_frame_override = str(
            self.get_parameter('goal_frame_override').value)
        if goal_frame_override:
            self._env_config.goal_frame = goal_frame_override
        people_topic_override = str(
            self.get_parameter('people_topic_override').value)
        if people_topic_override:
            self._env_config.people_topic = people_topic_override
        self._observation_config = ObservationConfig.from_dict(
            saved.get('observation', {}))
        # Only the arrival radius is read, not the whole reward block: the
        # reward shapes training and takes no part in the control loop.
        # Rebuilding the full RewardConfig here would tie every saved run
        # to the current set of reward terms, and a checkpoint written
        # before a term was added or dropped would stop loading.
        self._goal_distance = float(
            saved.get('reward', {}).get('goal_distance',
                                        RewardConfig.goal_distance))

        # Deployment wiring overrides whatever training used.
        self._cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self._deterministic = bool(self.get_parameter('deterministic').value)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        # No people_provider, ever. Training may have run on Gazebo ground
        # truth (env.people_source: ground_truth), but the only source a robot
        # has is the /people topic, and social_rl/ground_truth.py must not even
        # be importable here -- it reads /model_states. The observation the
        # policy sees is identical either way; what changes is how good the
        # numbers in it are.
        if people_topic_override:
            # Loud on purpose: a number measured on this wiring says nothing
            # about what the robot will do, and the two are easy to confuse
            # weeks later when only the log is left.
            self.get_logger().warn(
                f'people come from {self._env_config.people_topic}, NOT the '
                f'{EnvConfig.people_topic} this run was configured with. '
                'Simulation only -- that topic does not exist on the robot. '
                'What this measures is the policy, on the labels it trained '
                'on; it says nothing about how well block B and block C feed '
                'it.')
        elif self._env_config.people_source == 'ground_truth':
            self.get_logger().info(
                'this policy was trained on Gazebo ground truth; running it '
                f'on {self._env_config.people_topic} from block B instead. '
                'Expect worse tracking than it saw in training -- people '
                'behind the robot are invisible and scene_type is whatever '
                'block C reports, empty if no VLM is running.')
        self._bridge = PerceptionBridge(self, self._tf_buffer,
                                        self._env_config,
                                        self._observation_config)
        self._cmd_pub = self.create_publisher(Twist, self._cmd_vel_topic, 10)
        self._lstm_states = None
        self._episode_start = True
        self._goal = None
        self._goal_identity = None
        self._goal_reached = False

        goal_topic = str(self.get_parameter('goal_topic').value)
        self.create_subscription(PoseStamped, goal_topic, self._on_goal, 10)

        device = str(self.get_parameter('device').value)
        self._model = RecurrentPPO.load(model_path, device=device)

        self.create_timer(self._env_config.control_period, self._control_step)
        self.get_logger().info(
            f'policy {os.path.basename(model_path)} on {device}, '
            f'obs {self._observation_config.grid_channels}x{self._observation_config.grid_size}'
            f'x{self._observation_config.grid_size} grid '
            f'({self._observation_config.grid_extent:.1f} m box) + '
            f'{VECTOR_FEATURES} scalars, '
            f'{1.0 / self._env_config.control_period:.1f} Hz -> '
            f'{self._cmd_vel_topic}'
            f'; goal frame {self._env_config.goal_frame}. Waiting on '
            f'{goal_topic}.')

    def _on_goal(self, msg: PoseStamped):
        frame = msg.header.frame_id or self._env_config.goal_frame
        source_x, source_y = msg.pose.position.x, msg.pose.position.y
        identity = (frame, source_x, source_y)
        changed = identity != self._goal_identity

        goal_x, goal_y = source_x, source_y
        if frame != self._env_config.goal_frame:
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._env_config.goal_frame, frame,
                    rclpy.time.Time()).transform
            except TransformException as error:
                self.get_logger().error(
                    f'goal is in frame "{frame}" and it cannot be carried '
                    'into '
                    f'{self._env_config.goal_frame}: {error}')
                return
            goal_x, goal_y = transform_point(transform, goal_x, goal_y)
        self._goal = (goal_x, goal_y)
        self._goal_identity = identity
        if changed:
            # A new goal is a new episode: the memory of the last approach is
            # not about this one.
            self._goal_reached = False
            self._reset_policy_state()
            self.get_logger().info(
                f'new goal: ({goal_x:.2f}, {goal_y:.2f}) in '
                f'{self._env_config.goal_frame}')

    def _reset_policy_state(self):
        self._lstm_states = None
        self._episode_start = True

    def stop(self):
        if rclpy.ok():
            self._cmd_pub.publish(Twist())

    def _control_step(self):
        if self._goal is None:
            return
        if self._goal_reached:
            self.stop()
            return
        if not self._bridge.has_scan():
            self.get_logger().warn(
                f'no fresh {self._env_config.scan_topic}, holding still',
                throttle_duration_sec=5.0)
            self.stop()
            return
        if not self._bridge.has_odom():
            self.get_logger().warn(
                f'no fresh {self._env_config.odom_topic}, holding still',
                throttle_duration_sec=5.0)
            self.stop()
            return
        try:
            observation, state = self._bridge.observe(
                self._goal[0], self._goal[1])
        except RuntimeError as error:
            self.get_logger().warn(str(error), throttle_duration_sec=5.0)
            self.stop()
            return
        if not state['people_transform_valid']:
            self.get_logger().error(
                'tracked people cannot be transformed into the robot frame; '
                'holding still instead of treating them as absent',
                throttle_duration_sec=2.0)
            self.stop()
            return

        if state['goal_distance'] <= self._goal_distance:
            self.get_logger().info('goal reached')
            self._goal_reached = True
            self.stop()
            return

        action, self._lstm_states = self._model.predict(
            observation, state=self._lstm_states,
            episode_start=np.array([self._episode_start]),
            deterministic=self._deterministic)
        self._episode_start = False

        linear, angular = scale_action(action, self._env_config)
        command = Twist()
        command.linear.x = linear
        command.angular.z = angular
        self._cmd_pub.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = SocialRlAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
