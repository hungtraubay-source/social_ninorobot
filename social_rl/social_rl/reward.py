"""Reward and episode termination for the person-avoidance task.

Three things end or score a step, and nothing else:

    reach the goal                  +goal_reward, episode over
    touch an obstacle               obstacle_collision_penalty, episode over
    stand inside K_soc              -lambda_s*(w_talk*C_talk + w_view*C_view
                                    + w_cross*C_cross), one term per
                                    scene_type bucket (12-09-2026)

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
    # 12-09-2026: R_g/R_c từ báo cáo Gaussian (link arXiv 2009.04770 cho vùng
    # xã hội, phần hàm thưởng là thiết kế riêng của dự án). Mọi hệ số shaping
    # bên dưới chia theo TỈ LỆ CŨ với R_g/R_c để giữ nguyên mọi điểm hoà vốn
    # đã tuning (200/-200 -> 20/-20 hôm 11-09, giờ -> 10/-10): progress_gain
    # vẫn đúng 1/5 goal_reward, |step_penalty|*max_episode_steps vẫn đúng
    # bằng |obstacle_collision_penalty|.
    goal_reward: float = 10.0
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
    obstacle_collision_penalty: float = -10.0
    # 0.26 is robot_radius in nav_sim.yaml; the margin makes the episode end
    # just before the mesh actually touches, where recovery is hopeless anyway.
    obstacle_collision_distance: float = 0.30

    # --- the social constraint field ---
    # 12-09-2026: replaced the single social_penalty with the report's
    # three-way split:
    #
    #   r_social,t = -lambda_s * (w_talk*C_talk + w_view*C_view + w_cross*C_cross)
    #
    # C_talk/C_view/C_cross are the weighted Gaussian value (each zone's own
    # weight already baked in) at the robot's EXACT position, split by
    # scene_type via constraint_field.intrusion_by_type(): 'talking' on its
    # own, 'waiting' on its own, everything else (passing, walking,
    # unrecognised, backs_turned) pooled into 'cross' so no person ever falls
    # through the reward for free. w_talk/w_view/w_cross is a SEPARATE axis
    # from each zone's own weight in constraint_field.py -- that one shapes
    # what the policy SEES, this one shapes what it is CHARGED, and they do
    # not have to agree.
    #
    # Previously (social_penalty=-8.0, one term, MAX over every zone): at the
    # centre of a talking pair (weight 0.9) that cost -7.2/step. With
    # lambda_s=3, w_talk=1.0: -3*1.0*0.9 = -2.7/step -- noticeably lighter,
    # because every situation type now carries its own weight (w_view=0.8,
    # w_cross=0.6) instead of one shared factor for all of them. NOT
    # retrained against yet -- check RUN_RL.txt before trusting a run built
    # on these numbers.
    lambda_s: float = 3.0
    w_talk: float = 1.0
    w_view: float = 0.8
    w_cross: float = 0.6

    # --- shaping ---
    # Paid per metre of progress towards the goal, so one metre gained is worth
    # roughly a fifth of reaching it and the policy cannot farm shaping instead
    # of finishing.
    progress_gain: float = 2.0
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
    heading_gain: float = 0.25
    # Charged every step regardless, which is what makes loitering expensive.
    #
    # -0.8, not the -0.5 it was. At -0.5 a full 250 step episode of standing
    # still cost -125 while hitting a wall cost -200, so refusing to move was
    # the cheapest way to end an episode and the policy took it: run
    # 20260827_013703 finished with 100% timeouts and 71% of commands below
    # 0.05 m/s. -0.8 puts a whole idle episode at exactly -200, level with the
    # collision it was hiding from.
    #
    # Do not raise it past |obstacle_collision_penalty| / max_episode_steps
    # (0.04 at 250 steps and -10.0 collision, 12-09-2026). Once idling costs
    # more than crashing, driving into the nearest wall becomes the cheap way
    # out and the policy will find it.
    step_penalty: float = -0.04

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
                  talk_intrusion: float, view_intrusion: float,
                  cross_intrusion: float) -> StepOutcome:
    """Score one control step and decide whether the episode ends here."""
    components = {}

    # The obstacle check comes before the goal, so that arriving and clipping
    # a table on the same step counts as clipping it. This costs nothing at the
    # goals in rl_train.yaml: all four sit at least 1.07 m from the table leg,
    # so no goal is reachable only by first entering the collision radius.
    #
    # There is no matching check for people. Driving through somebody is
    # priced entirely by the constraint field, split by scene_type (12-09-2026,
    # see RewardConfig.lambda_s). Talk_intrusion tops out at the 'talking'
    # zone's own weight (0.9): -lambda_s*w_talk*0.9 = -3*1.0*0.9 = -2.7 at the
    # worst point. At full speed a step covers 0.1 m and earns +0.2 of
    # progress, so passing right through the centre of a conversation nets
    # -2.54 a step -- still a loss, and enough of one that going round wins,
    # but no longer a cliff that ends the episode and everything it could
    # still have earned. See RewardConfig's field comments for the full
    # arithmetic and its history.
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

    components['social'] = -config.lambda_s * (
        config.w_talk * min(max(talk_intrusion, 0.0), 1.0) +
        config.w_view * min(max(view_intrusion, 0.0), 1.0) +
        config.w_cross * min(max(cross_intrusion, 0.0), 1.0))

    return StepOutcome(float(sum(components.values())), False, '', components)
