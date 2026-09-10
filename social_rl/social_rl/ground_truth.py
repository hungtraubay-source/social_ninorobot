"""People straight from the simulator, standing in for blocks B and C.

During training the whole perception chain -- YOLO on the RGB-D stream, the
tracker, the video VLM that judges what people are doing -- is replaced by what
Gazebo already knows exactly. That is the point of the plan's step 2: the
policy is what is being trained, and feeding it detector noise while it is
still learning to drive means it never finds out whether a bad episode was its
own fault or the tracker's.

Two sources, both simulation only:

    /social_gt/people   pose, velocity, facing and scene_type of every person,
                        published by the animated-people plugin in the `world`
                        frame. It comes from the plugin rather than from
                        /model_states because /model_states reports zero
                        velocity for actors and knows nothing about scene_type.
    /model_states       the robot's own pose in `world`.

Both are converted into the robot frame here, so what leaves this module has
exactly the shape and the meaning that the deployment path produces from the
real /people topic. No TF and no localizer is involved -- which is also why RL
training needs neither AMCL nor Nav2.

This file must never be imported by the deployment agent. It is the one place
in the package that reads a quantity no robot has.
"""

import math
import random
from dataclasses import replace

import numpy as np
from gazebo_msgs.msg import ModelStates
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from rclpy.time import Time
from social_perception.msg import People

from social_rl.observation import RelativeEntity
from social_rl.ros_interface import visible_to_camera

# What block C will be able to say. '' is not in here: an abstention is drawn
# separately, because "I could not tell" is a different failure from "I read it
# wrong", and block D treats it differently -- '' selects the neutral (1,1,1)
# scales rather than another situation's shape.
VLM_LABELS = ('talking', 'backs_turned', 'passing', 'walking')


class SimulatedVLM:
    """Block C, behaving the way a real video-VLM will when it exists.

    Block C is not built yet, and the policy that will eventually be handed its
    output is being trained NOW, on a ground truth whose scene_type is always
    right and always instant. Training against that and then plugging in a real
    VLM is how you find out, at the worst possible moment, that the policy had
    learned to trust the label completely.

    THE LABEL IS CORRUPTED, THE REWARD IS NOT. observe() charges the reward on
    the true labels while the policy is shown these. That split is deliberate
    and it is the whole point of this class:

      * the camera cone and the occlusion test still limit the observation, but
        the training reward deliberately uses the complete truth list. Recently
        seen people are learnable through the LSTM; never-seen people teach the
        conservative speed and clearance wanted in blind space.
      * a misread label is different. The person is right there, at the right
        place, with the right velocity; only the ACTIVITY was misjudged. The
        situation on the ground is still a conversation. Charging by the noisy
        label would teach the policy that a conversation misread as `walking`
        really is cheap -- that is, it would teach it to TRUST the label, which
        is exactly the habit that will hurt when block C arrives.

    Charging by truth teaches the opposite: when the label or current view is
    unreliable, leave more room. The LSTM carries previously observed geometry
    and doubt across steps; blind space is handled by conservative motion.

    Held per track and only re-drawn every vlm_period seconds, which produces
    the latency and the flicker of a real 1-3 Hz video model without needing a
    separate parameter for either.
    """

    def __init__(self, env_config, logger=None):
        self._env = env_config
        # Seeded explicitly so a run can be repeated. 0 means "different every
        # time", which is what you want while training and not what you want
        # while chasing down one bad episode.
        seed = int(env_config.vlm_seed)
        self._rng = random.Random(seed if seed else None)
        if logger is not None and env_config.vlm_noise:
            logger.info(
                f'simulated block C is ON: every {env_config.vlm_period:.2f} s '
                f'a ruling is re-drawn per person, wrong with probability '
                f'{env_config.vlm_wrong_label_prob:.2f}, abstained with '
                f'{env_config.vlm_abstain_prob:.2f}. The REWARD still uses the '
                f'true labels -- see SimulatedVLM.')
        # track_id -> (stamp, label, facing_offset, facing_lost)
        self._held = {}
        self.corrupted = 0

    def clear(self):
        """Forget every ruling. Called on reset, like the people memory."""
        self._held.clear()

    def _draw(self, truth):
        roll = self._rng.random()
        if roll < self._env.vlm_abstain_prob:
            label = ''
        elif roll < self._env.vlm_abstain_prob + self._env.vlm_wrong_label_prob:
            others = [name for name in VLM_LABELS if name != truth]
            label = self._rng.choice(others)
        else:
            label = truth
        # A bias that persists for the life of the ruling rather than being
        # re-rolled every step: a model that is wrong about which way somebody
        # faces is wrong about it consistently, and a per-step jitter would
        # average out to the truth over a few frames.
        offset = self._rng.gauss(0.0, self._env.vlm_facing_sigma)
        lost = self._rng.random() < self._env.vlm_facing_lost_prob
        return label, offset, lost

    def corrupt(self, people, now, robot_yaw):
        """Return the list the POLICY sees. The caller keeps the true one.

        `robot_yaw` is the robot's heading in the world frame, needed for the
        lost-facing case: block B on the real robot reports orientation 0 in
        the `map` frame for anybody standing still, which is a fixed compass
        direction, not a symmetric region. Reproducing that exactly is the
        point -- a wrongly oriented region is worse than an absent one.
        """
        self.corrupted = 0
        if not self._env.vlm_noise:
            return people
        shown = []
        present = set()
        for person in people:
            present.add(person.track_id)
            entry = self._held.get(person.track_id)
            elapsed = now - entry[0] if entry else None
            if entry is None or elapsed >= self._env.vlm_period or elapsed < 0.0:
                entry = (now,) + self._draw(person.scene_type)
                self._held[person.track_id] = entry
            _, label, offset, lost = entry
            facing = -robot_yaw if lost else _wrap(person.facing + offset)
            if label != person.scene_type:
                self.corrupted += 1
            shown.append(replace(person, scene_type=label, facing=facing))
        for track_id in [key for key in self._held if key not in present]:
            del self._held[track_id]
        return shown


def _yaw(orientation) -> float:
    return math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y
                     + orientation.z * orientation.z))


class GroundTruthPeople:
    """Gazebo's own answer to "who is where, doing what"."""

    def __init__(self, node, env_config):
        self._node = node
        self._env = env_config
        self._logger = node.get_logger()

        # Keep only the newest of each: a step acts on now, and /model_states
        # arrives at about 93 Hz, so a queue would only hold history nobody
        # reads.
        latest = QoSProfile(
            depth=1, history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE)
        self.people = None
        self.states = None
        node.create_subscription(People, env_config.ground_truth_people_topic,
                                 self._on_people, 10)
        node.create_subscription(ModelStates, env_config.model_states_topic,
                                 self._on_states, latest)
        # Block C, simulated. Lives here rather than in ros_interface because
        # it is a simulation-only corruption of a simulation-only truth, and
        # the robot must never grow a copy of it: its block C will be wrong on
        # its own.
        self._vlm = SimulatedVLM(env_config, self._logger)

    def _on_people(self, message):
        self.people = message

    def _on_states(self, message):
        self.states = message

    def ready(self) -> bool:
        """Both feeds have arrived at least once.

        /social_gt/people is published whether or not anybody is in the scene,
        so waiting for it is not the same as waiting for people to appear.
        """
        return self.states is not None and self.people is not None

    def robot_pose(self):
        """The robot's (x, y, yaw) in the Gazebo world frame.

        ModelStates carries no header and therefore no stamp, so there is no
        freshness check to make here. It is published on every world update; if
        it has stopped, the simulation has stopped, and the step loop in
        ros_env.py is what notices that.
        """
        if self.states is None:
            raise RuntimeError(
                f'no message on {self._env.model_states_topic} yet. It comes '
                f'from libgazebo_ros_state.so, which lirs_test.world loads; '
                f'check that Gazebo is running.')
        name = self._env.entity_name
        if name not in self.states.name:
            raise RuntimeError(
                f'no entity "{name}" in {self._env.model_states_topic} '
                f'(present: {sorted(self.states.name)}). That is the -entity '
                f'argument spawn_entity.py was given in gazebo.launch.py.')
        pose = self.states.pose[self.states.name.index(name)]
        # The URDF root is base_footprint and base_link sits directly above it
        # (base_to_footprint is a pure 0.0325 m z offset), so in the plane this
        # pose is base_link and no further transform is needed.
        return (pose.position.x, pose.position.y, _yaw(pose.orientation))

    def relative_people(self, apply_camera=True):
        """Everybody the camera could see, in the robot frame.

        Filtered to the camera cone when env.people_camera_only is set,
        which is the default. Unfiltered ground truth hands the policy a sense
        the robot does not have: it would learn to walk round the back of
        somebody it can no longer see, which works perfectly while a simulator
        keeps telling it where they are and fails the moment it stops.

        What is NOT given up by filtering: for everybody still in the cone the
        position, velocity, facing and scene_type stay exact. A bad episode is
        still the policy's fault rather than a detector's, which is the whole
        reason for training on ground truth.

        apply_camera=False returns the unfiltered list. observe() uses it both
        for the full-list social reward and for the `peak_unseen` diagnostic.
        The policy never receives it. Occlusion is applied by the caller, not
        here, because it needs the scan.
        """
        message = self.people
        if message is None or not message.people:
            return []
        # Stale ground truth means the plugin stopped, not that the scene
        # emptied -- but treating it as an empty scene is still the safe read,
        # and _spin_for refuses to finish a step on stale data anyway.
        age = (self._node.get_clock().now()
               - Time.from_msg(message.header.stamp)).nanoseconds * 1e-9
        if age > self._env.people_timeout:
            return []

        robot_x, robot_y, robot_yaw = self.robot_pose()
        cos_yaw, sin_yaw = math.cos(robot_yaw), math.sin(robot_yaw)
        people = []
        for person in message.people:
            delta_x = person.pose.position.x - robot_x
            delta_y = person.pose.position.y - robot_y
            local_x = cos_yaw * delta_x + sin_yaw * delta_y
            local_y = -sin_yaw * delta_x + cos_yaw * delta_y
            if (apply_camera and self._env.people_camera_only
                    and not visible_to_camera(local_x, local_y, self._env)):
                continue
            people.append(RelativeEntity(
                x=local_x,
                y=local_y,
                vx=(cos_yaw * person.velocity.linear.x
                    + sin_yaw * person.velocity.linear.y),
                vy=(-sin_yaw * person.velocity.linear.x
                    + cos_yaw * person.velocity.linear.y),
                # Where they are looking, expressed relative to where the robot
                # is looking. The constraint field's asymmetric regions are
                # built in the robot frame, so this has to be too.
                facing=_wrap(_yaw(person.pose.orientation) - robot_yaw),
                scene_type=person.scene_type,
                track_id=person.id,
                scene_confidence=person.scene_confidence))
        return people

    def corrupt(self, people, now):
        """The list the policy is SHOWN, once simulated block C has had a go.

        Returns the input unchanged when env.vlm_noise is off. The caller keeps
        the true list; the reward ultimately uses the complete unfiltered truth
        list -- see SimulatedVLM for why those are deliberately different.
        """
        return self._vlm.corrupt(people, now, self.robot_pose()[2])

    @property
    def corrupted_people(self) -> int:
        """How many labels the last corrupt() actually changed.

        A sanity counter, not a metric: a misconfigured noise model that never
        fires is otherwise indistinguishable from a clean run.
        """
        return self._vlm.corrupted

    def forget_rulings(self):
        """Drop every held VLM ruling. Called on reset with the people memory."""
        self._vlm.clear()

    def person_positions(self):
        """(N, 2) world-frame positions. Used by the reset check, not the policy."""
        if self.people is None or not self.people.people:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(
            [[person.pose.position.x, person.pose.position.y]
             for person in self.people.people], dtype=np.float32)


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
