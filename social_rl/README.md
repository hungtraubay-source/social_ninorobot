# social_rl

Block **E** of the architecture: a **constraint-conditioned** recurrent policy
(RecurrentPPO + LSTM, `sb3-contrib`) that drives the base from start to goal
without violating the social constraint field. It also contains block **D**
(compiling that field) and block **H** (goal, robot state, local occupancy).

Nav2 is not in the loop. The policy owns the base for the whole route.

The task is not "avoid people". It is: cutting between two people **holding a
conversation** must cost more than passing behind two people **with their backs
turned**, even when the geometry is identical. Nothing about position and
velocity separates those two cases, which is why `scene_type` exists.

## Layout

| File | Block | Role |
|---|---|---|
| `social_rl/constraint_field.py` | **D** | grounding + compiling K_soc into 4 predicted grids |
| `social_rl/observation.py` | **D + H** | the 6-channel observation; the contract between training and deployment |
| `social_rl/ground_truth.py` | **B + C** | people straight from Gazebo. **Simulation only** — never imported by the agent node |
| `social_rl/ros_interface.py` | — | topics → robot-frame observation; shared by trainer and agent |
| `social_rl/ros_env.py` | — | `gymnasium.Env`: paused stepping, scenario reset |
| `social_rl/gazebo_world.py` | — | episode reset: teleport, re-seed EKF, pause/unpause |
| `social_rl/reward.py` | — | goal / obstacle / K_soc intrusion, plus two shaping terms |
| `social_rl/policy.py` | **E** | `SocialGridExtractor` — the CNN that reads the grid |
| `social_rl/train.py` | **E** | `train_rl` executable |
| `social_rl/agent_node.py` | **E** | `rl_agent` executable — runs a trained `.zip` as a `cmd_vel` node |
| `config/rl_train.yaml` | — | one file for env + observation + constraint field + reward + hyperparameters |

## The observation

```
grid   (6, 32, 32) float32   6.4 x 6.4 m box, robot at the centre facing up
    channel 0   block H: occupancy, 1.0 where a laser beam ends
    channel 1   block D: visible/remembered K_soc now, at t+0.0 s
    channel 2   block D: K_soc predicted at t+0.5 s
    channel 3   block D: K_soc predicted at t+1.0 s
    channel 4   block D: K_soc predicted at t+1.5 s
    channel 5   block D: K_soc predicted at t+2.0 s
vector (5,)      float32   block H
    [0] goal distance / goal_max_distance   (clipped at 10 m)
    [1] sin(goal bearing)
    [2] cos(goal bearing)
    [3] measured v / max_linear_speed       (from /odom, not the last command)
    [4] measured w / max_angular_speed
```

Five social channels: the present plus a 2.0 s horizon at 0.5 s spacing.
Occupancy is kept out of the field because a wall will still be there in two
seconds and a person will not.

Channel 1 is the instant field for the people the camera has seen or the memory
still carries. The social reward uses the same `intrusion_at_zones` geometry and
true activity weights, but evaluates it on the complete Gazebo person list: no
camera cone and no occlusion filter. This deliberate asymmetry prevents turning
away from a person from making the penalty disappear. The LSTM still receives
the information needed to carry recently seen people across a dropout.

`grid_size` and the length of `prediction_times` set the network shape, so a
checkpoint does not load into another configuration. Checkpoints from before
30-08-2026 (5-channel observation) cannot be resumed, nor can those from
before 28-08-2026 (2-channel).

## Why recurrent

A single frame cannot separate an approaching person from a receding one, and
the LSTM state carries "there was somebody on my left a moment ago" across a
perception dropout. Frame stacking would need a stack deep enough to span a
multi-second gap.

## Reward

`+200` reaching the goal, `-200` touching an obstacle, and `-8.0` per step times
how deep into K_soc the robot is standing. Two shaping terms: `+40` per metre of
progress and `-0.8` per step.

The social penalty and the channels the policy sees use the same constraint
compiler and `intrusion_at_zones()` function, so `scene_type` keeps its exact
meaning. Their person lists intentionally differ during ground-truth training:
the observation is camera-limited, while the reward covers everybody in the
scene at the full `social_penalty` weight. Charge for Euclidean distance instead
and every `scene_type` becomes decoration the policy has no reason to read.

Action: `Box(-1, 1)²` → linear `[0, 0.5] m/s` (forward only, there is no rear
sensor), angular `[-1, 1] rad/s`.

## Install and build

```bash
python3 -m pip install --user -r social_rl/requirements.txt   # sb3-contrib, gymnasium
cd ~/ninorobot2
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

`torch` is deliberately absent from `requirements.txt`: this machine already has
2.13.0+cu130 matching the installed driver, and letting pip resolve torch again
replaces it with a CPU wheel.

`social_perception/msg/Person.msg` gained a `scene_type` field, so build the
whole workspace rather than one package.

## Run

`RUN_RL.txt` is the source of truth, in Vietnamese. Two terminals:

```bash
ros2 launch linorobot2_gazebo gazebo.launch.py gui:=false   # 1
ros2 launch social_rl rl_train.launch.py                    # 2
tensorboard --logdir ~/social_rl_runs                       # any time
```

No perception terminal and no actor terminal: people come from Gazebo ground
truth, and the trainer sends a fresh scenario at the start of every episode.

`gz physics -u 0` on the running world raises rollout collection from 3.65 to
5.57 steps/s (measured 28-08-2026). Unlike the old camera-based pipeline,
nothing falls behind when you do this — the ground truth is computed inside the
world-update loop, so it cannot lag it.

End to end, including the PPO updates between rollouts, that is 3.9 steps/s
against the old pipeline's 4.2, so **this is not yet faster overall**: the
paused stepping costs about what the raised ceiling gains, and buys determinism
instead. `RUN_RL.txt` has the full table.

## What to watch

Read `outcome/goal`, `social/mean_peak_intrusion`, and
`social/mean_peak_intrusion_unseen` together. A policy that reaches every goal
by driving through the middle of every conversation scores perfectly on the
first and 1.0 on the social peaks. A rising visible-to-full gap exposes paths
that only keep people out of frame. `scenario/<name>_goal` splits success per
situation because `crossing` is usually learned before `talking`.

TensorBoard is only a training monitor. Select a concrete checkpoint by running
deterministic evaluation on its exact `.zip` path:

```bash
ros2 run social_rl train_rl \
  --eval ~/social_rl_runs/<run>/checkpoints/recurrent_ppo_<step>_steps.zip \
  --eval-episodes 40
```

The evaluation verdict passes only when all four fixed criteria pass:

- `peak_unseen - peak_intrusion < 0.10`
- `clear_episodes > 60%`
- mean closest person on the complete ground-truth list `> 0.60 m`
- goal rate in scenario `none > 90%`

Passing a run directory evaluates its `final_model.zip`; it does not search for
the best intermediate checkpoint.

## Not built yet

- **Block C** (video VLM). The seam is ready: `scene_type` in `Person.msg`. A
  real block C fills it and block D reads it, with no change to the network
  shape and no retraining. Without it the field falls back to a neutral region.
- **Block F** (CBF/QP safety shield). `social_navigation`'s
  `social_velocity_filter` plays a similar role today but is not a CBF: no
  barrier function, no QP, no safe-set invariance guarantee.
