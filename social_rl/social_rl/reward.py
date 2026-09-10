"""Reward and episode termination for the person-avoidance task.

Three things end or score a step, and nothing else:

    reach the goal                  +goal_reward, episode over
    touch an obstacle               obstacle_collision_penalty, episode over
    stand inside K_soc              social_penalty, scaled by how deep in

Nothing here treats a person as solid. That is deliberate and it depends on
the world file: lirs_test.world keeps its actor collision proxies commented
out, so /scan passes through people and the obstacle rule cannot fire on one.
Uncomment them and the terminal person collision comes back through the lidar
under the name hit_obstacle.

Walking into a person no longer ends the episode. That rule -- -200 and the
episode over inside person_collision_distance -- turned every attempt to slip
round the back of the pair into a cliff: one bad step cost 200 plus everything
the rest of the episode could still have earned, so the policy stopped
approaching at all. Measured in run 20260827_013703: the goal was reached eight
times between hour 2 and hour 5 with the nearest person 1.15-2.19 m away, so
the detour was learned, and then lost -- the last 150 episodes were 100%
timeout, spinning in place with the goal 4 m off. Intrusion is still charged,
now only through the constraint field, which is bounded and does not cut the
episode short, so exploring near people is affordable again.

Two shaping terms keep that learnable. progress_gain is what makes the policy
move at all -- with only the four terminal rules, every reward except the last
step of an episode is zero, and a run that never stumbles onto the goal never
learns there is one. step_penalty is what makes loitering cost something, so
"drive nowhere safely" is not a winning strategy.

Deliberately absent: penalties on angular velocity and on command changes. Both
were in the earlier version and both fight the task -- going around a person
requires turning, and with an 86 degree camera the robot also has to turn to
keep that person in sight at all. Paying a fee for the manoeuvre being learned
is how a policy ends up preferring to stand still.

Kept separate from the environment so the weights can be changed and compared
without touching the ROS plumbing.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Weights and thresholds. All distances in metres, from base_link."""

    # --- terminal events ---
    goal_reward: float = 200.0
    goal_distance: float = 0.35

    # Contact ends an episode only for obstacles now. They come from the
    # laser, which sees tables and walls but passes straight through the
    # actors, so this cannot fire on a person.
    #
    # That last clause is a property of the world file, not of this module.
    # lirs_test.world briefly carried a collision cylinder per actor, dragged
    # along by animated_people_factory to make the lidar see people. With those
    # in place minimum_scan hit 0.30 while the person's centre was still 0.45 m
    # away, which reinstated the terminal person collision dropped below,
    # relabelled hit_obstacle. They are commented out for that reason; check
    # the world before assuming people are invisible to /scan.
    obstacle_collision_penalty: float = -200.0
    # 0.26 is robot_radius in nav_sim.yaml; the margin makes the episode end
    # just before the mesh actually touches, where recovery is hopeless anyway.
    obstacle_collision_distance: float = 0.30

    # --- the social constraint field ---
    # Charged per step, as social_penalty times how deep into K_soc the robot
    # is standing: 0.0 outside every region, 1.0 inside somebody's o-space or
    # inside the o-space of a conversation.
    #
    # This replaced a plain distance ramp (0 at 1.2 m, -8.0 at 0.20 m) on
    # 28-08-2026. The ramp measured Euclidean distance and nothing else, so it
    # charged the same for cutting between two people mid-conversation as for
    # walking past somebody's back at the same range -- which made every
    # scene_type in the observation decoration the policy had no reason to
    # read. What the policy is SHOWN and what it is CHARGED FOR now come out of
    # the same function in constraint_field.py, which is the only way the
    # difference between those two situations reaches the gradient. During
    # ground-truth training the caller evaluates that function on two lists:
    # filtered people for observation, everybody for this reward.
    #
    # -8.0 keeps the tuned break-even from the old ramp: at full speed a step
    # covers 0.1 m and earns +4.0 of progress against -0.8 of step penalty, so
    # driving through the middle of a region nets -4.8 a step. Still a loss,
    # and enough of one that going around wins, but bounded -- it does not end
    # the episode and take everything it could still have earned with it.
    social_penalty: float = -8.0

    # --- shaping ---
    # Paid per metre of progress towards the goal, so one metre gained is worth
    # roughly a fifth of reaching it and the policy cannot farm shaping instead
    # of finishing.
    progress_gain: float = 40.0
    # Paid per radian that |bearing to goal| shrinks in a step. progress_gain
    # is zero while the robot rotates in place -- distance to the goal does not
    # change -- so with a start yaw drawn over the whole circle (02-09-2026)
    # roughly half of all episodes open with a turn that earns nothing until it
    # is finished, and the policy never learns to make it. This term pays for
    # that turn: at 0.2 rad/step a rotation toward the goal earns +1.0 against
    # -0.8 of step penalty. It is symmetric, so wobbling to farm it nets zero,
    # and near zero bearing it is dwarfed by progress (a full-speed step earns
    # +4.0), so it does not distort the drive-straight behaviour once aligned.
    # Set to 0.0 to recover the pre-02-09 reward exactly.
    heading_gain: float = 5.0
    # Charged every step regardless, which is what makes loitering expensive.
    #
    # -0.8, not the -0.5 it was. At -0.5 a full 250 step episode of standing
    # still cost -125 while hitting a wall cost -200, so refusing to move was
    # the cheapest way to end an episode and the policy took it: run
    # 20260827_013703 finished with 100% timeouts and 71% of commands below
    # 0.05 m/s. -0.8 puts a whole idle episode at exactly -200, level with the
    # collision it was hiding from.
    #
    # Do not raise it past 0.8 without lowering obstacle_collision_penalty by
    # as much. Once idling costs more than crashing, driving into the nearest
    # wall becomes the cheap way out and the policy will find it.
    step_penalty: float = -0.8

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> 'RewardConfig':
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                f'unknown reward keys: {sorted(unknown)}. '
                f'Valid keys: {sorted(known)}')
        return cls(**values)


@dataclass
class StepOutcome:
    reward: float
    terminated: bool
    reason: str
    components: dict


def evaluate_step(config: RewardConfig, *, goal_distance: float,
                  previous_goal_distance: float, goal_bearing: float,
                  previous_goal_bearing: float, minimum_scan: float,
                  social_intrusion: float) -> StepOutcome:
    """Score one control step and decide whether the episode ends here."""
    components = {}

    # The obstacle check comes before the goal, so that arriving and clipping
    # a table on the same step counts as clipping it. This costs nothing at the
    # goals in rl_train.yaml: all four sit at least 1.07 m from the table leg,
    # so no goal is reachable only by first entering the collision radius.
    #
    # There is no matching check for people. Driving through somebody is
    # priced entirely by the constraint field: -8.0 per step at the bottom of
    # it. At full speed a step covers 0.1 m and earns +4.0 of progress, so
    # passing right through a region nets -4.8 a step -- still a loss, and
    # enough of one that going round wins, but no longer a cliff that ends the
    # episode and everything it could still have earned.
    if minimum_scan <= config.obstacle_collision_distance:
        return StepOutcome(config.obstacle_collision_penalty, True,
                           'hit_obstacle',
                           {'collision': config.obstacle_collision_penalty})

    if goal_distance <= config.goal_distance:
        return StepOutcome(config.goal_reward, True, 'goal',
                           {'goal': config.goal_reward})

    components['progress'] = config.progress_gain * (
        previous_goal_distance - goal_distance)
    # Reward shrinking the angle to the goal. abs() over a control step is
    # safe: bearing moves at most max_angular_speed * control_period per step
    # (~0.2 rad), far from the +-pi wrap, so consecutive |bearing| differ by
    # a small, well-defined amount.
    components['heading'] = config.heading_gain * (
        abs(previous_goal_bearing) - abs(goal_bearing))
    components['step'] = config.step_penalty

    components['social'] = config.social_penalty * min(
        max(social_intrusion, 0.0), 1.0)

    return StepOutcome(float(sum(components.values())), False, '', components)
