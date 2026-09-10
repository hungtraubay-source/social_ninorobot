"""Train the person-avoidance policy with Recurrent PPO (LSTM).

Run it against a simulation that is already up (Gazebo, perception, actors,
Nav2 -- see RUN_RL.txt); this program never starts or stops the simulator, so a
crashed training run costs you the run, not the world and the loaded VLM.

Why RecurrentPPO and not plain PPO: a single lidar frame cannot tell an
approaching person from a receding one, and the depth camera drops out entirely
whenever the robot turns away. The LSTM state is what carries "there was
somebody on my left a second ago" across those gaps, which is exactly the
information avoidance needs. Frame stacking would need a stack deep enough to
span a multi-second perception dropout.
"""

import argparse
import json
import math
import os
import sys
import zipfile
from dataclasses import replace
from datetime import datetime

import numpy as np
import rclpy
import torch
import yaml
from ament_index_python.packages import get_package_share_directory
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from social_rl.observation import ObservationConfig
from social_rl.policy import SocialGridExtractor
from social_rl.reward import RewardConfig
from social_rl.ros_env import SocialAvoidEnv
from social_rl.ros_interface import EnvConfig

# Resolved through the ament index, not from __file__: with --symlink-install
# the module is imported from the source tree and a relative path would happen
# to work, then break the day the workspace is built without symlinks.
DEFAULT_CONFIG = os.path.join(
    get_package_share_directory('social_rl'), 'config', 'rl_train.yaml')


class OutcomeCallback(BaseCallback):
    """Log how episodes end, which is what says whether training works.

    Mean reward alone hides the difference between "reaches the goal" and
    "learned to stand still and pay only the step penalty".
    """

    def __init__(self, window=20):
        super().__init__()
        self._window = window
        self._outcomes = []
        self._clearances = []
        self._intrusions = []
        self._hidden_intrusions = []
        self._hidden_fractions = []
        self._scenarios = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode_outcome' not in info:
                continue
            self._outcomes.append(info['episode_outcome'])
            self._scenarios.append(info.get('scenario', 'none'))
            self._intrusions.append(info.get('maximum_intrusion', 0.0))
            self._hidden_intrusions.append(
                info.get('maximum_hidden_intrusion', 0.0))
            steps = max(info.get('episode_steps', 0), 1)
            self._hidden_fractions.append(
                info.get('hidden_person_steps', 0) / steps)
            clearance = info.get('minimum_person_distance', -1.0)
            if clearance >= 0.0:
                self._clearances.append(clearance)
            self._outcomes = self._outcomes[-self._window:]
            self._clearances = self._clearances[-self._window:]
            self._intrusions = self._intrusions[-self._window:]
            self._hidden_intrusions = self._hidden_intrusions[-self._window:]
            self._hidden_fractions = self._hidden_fractions[-self._window:]
            self._scenarios = self._scenarios[-self._window:]
            total = len(self._outcomes)
            for reason in ('goal', 'hit_obstacle', 'timeout'):
                self.logger.record(
                    f'outcome/{reason}',
                    self._outcomes.count(reason) / total)
            if self._clearances:
                self.logger.record(
                    'outcome/mean_closest_person',
                    sum(self._clearances) / len(self._clearances))
            # The social half of the task, which reward alone hides: a policy
            # that reaches every goal by driving straight through the middle
            # of every conversation scores well on outcome/goal and badly
            # here. Watch the two together or you are only measuring
            # navigation.
            self.logger.record(
                'social/mean_peak_intrusion',
                sum(self._intrusions) / len(self._intrusions))
            self.logger.record(
                'social/clear_episodes',
                sum(1 for value in self._intrusions if value <= 0.0) / total)
            # DIAGNOSTIC PAIR. The first peak comes from the people visible to
            # the policy; `unseen` is computed against everybody in the scene,
            # hidden ones included, and is now also what the reward charges.
            # Read them together: the two tracking each other means the policy
            # is genuinely keeping clear, while `unseen` climbing away from the
            # visible one means its path still relies on keeping people out of
            # frame. `hidden_person_fraction` says how much of the episode
            # there was anybody to hide from at all -- near zero makes the
            # comparison meaningless rather than reassuring.
            self.logger.record(
                'social/mean_peak_intrusion_unseen',
                sum(self._hidden_intrusions) / len(self._hidden_intrusions))
            self.logger.record(
                'social/hidden_person_fraction',
                sum(self._hidden_fractions) / len(self._hidden_fractions))
            # Per scenario, so a policy that has learned `crossing` but not
            # `talking` is visible as two curves rather than one average.
            for scenario in set(self._scenarios):
                indices = [index for index, name in enumerate(self._scenarios)
                           if name == scenario]
                self.logger.record(
                    f'scenario/{scenario}_goal',
                    sum(1 for index in indices
                        if self._outcomes[index] == 'goal') / len(indices))
        return True


def load_config(path: str) -> dict:
    with open(path, 'r') as handle:
        config = yaml.safe_load(handle) or {}
    unknown = set(config) - {'env', 'observation', 'reward', 'train'}
    if unknown:
        raise ValueError(f'unknown top-level sections in {path}: '
                         f'{sorted(unknown)}')
    return config


def resolve_resume(path: str) -> str:
    """Accept either a .zip or the run directory that holds one.

    Typing the run directory is what you actually remember after a crash; the
    exact checkpoint number is not.
    """
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise RuntimeError(f'--resume {path} is neither a file nor a directory')
    final = os.path.join(path, 'final_model.zip')
    if os.path.exists(final):
        return final
    checkpoints = os.path.join(path, 'checkpoints')
    saved = [os.path.join(checkpoints, name)
             for name in os.listdir(checkpoints)] if os.path.isdir(
                 checkpoints) else []
    if not saved:
        raise RuntimeError(
            f'{path} holds no final_model.zip and no checkpoints/*.zip')
    # By step count, not by name: recurrent_ppo_95000_steps sorts after
    # recurrent_ppo_100000_steps alphabetically.
    return max(saved, key=lambda name: int(
        ''.join(c for c in os.path.basename(name) if c.isdigit()) or 0))


def check_resume_layout(resume_path: str, observation_config):
    """Refuse to continue a checkpoint that was trained on another layout.

    Without this the mismatch surfaces as a shape error deep inside torch, or
    -- worse, when only the scaling changed and the shapes still line up --
    not at all.
    """
    directory = os.path.dirname(resume_path)
    saved_config = os.path.join(directory, 'env_config.yaml')
    if not os.path.exists(saved_config):
        saved_config = os.path.join(os.path.dirname(directory),
                                    'env_config.yaml')
    if not os.path.exists(saved_config):
        raise RuntimeError(
            f'no env_config.yaml next to {resume_path}; cannot tell whether '
            f'that checkpoint was trained on the current observation layout')
    with open(saved_config, 'r') as handle:
        previous = (yaml.safe_load(handle) or {}).get('observation', {})
    current = observation_config.to_dict()
    differences = {key: (previous.get(key), value)
                   for key, value in current.items()
                   if previous.get(key) != value}
    if differences:
        lines = '\n'.join(f'  {key}: was {was!r}, now {now!r}'
                          for key, (was, now) in sorted(differences.items()))
        raise RuntimeError(
            f'the observation in {saved_config} is not the one this run '
            f'would use:\n{lines}\nTrain a fresh run, or restore those '
            f'values in the training YAML.')
    return saved_config


RESUMABLE_KEYS = ('learning_rate', 'n_steps', 'batch_size', 'n_epochs',
                  'gamma', 'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef',
                  'max_grad_norm', 'target_kl')


def resume_overrides(resume_path: str, train_config) -> dict:
    """Hyperparameters from the YAML that a resume must not inherit from the
    checkpoint.

    Every one of these is stored inside the .zip, and SB3 prefers what it finds
    there, so a resumed run would otherwise train with the settings of the run
    that produced the checkpoint. Prints what it changes: an override that goes
    unnoticed is the same trap as no override at all.
    """
    overrides = {key: train_config[key]
                 for key in RESUMABLE_KEYS if key in train_config}
    with zipfile.ZipFile(resume_path) as archive:
        stored = json.loads(archive.read('data'))
    for key, value in sorted(overrides.items()):
        previous = stored.get(key)
        # learning_rate and clip_range are stored as a serialised schedule
        # object, not a number, so there is nothing to compare against. Forcing
        # them is still correct; announcing a change that may not be one is not.
        if isinstance(previous, dict):
            continue
        if previous != value:
            print(f'[social_rl] override {key}: {previous} -> {value}')
    return overrides


def resolve_device(requested: str) -> str:
    """Fail loudly rather than training 20x slower on the CPU by accident."""
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            'train.device asks for CUDA but torch.cuda.is_available() is '
            'False. Check nvidia-smi and that the torch build matches the '
            'driver, or set device: cpu deliberately.')
    return requested


def evaluate(model, env, episodes: int, deterministic: bool = True):
    """Drive a trained policy for `episodes` episodes without learning.

    This exists because there was no way to ask "is the POLICY any good".
    agent_node.py deliberately never reads ground truth, so testing a
    checkpoint meant testing it through block B, and a bad number there is
    equally well explained by a blind tracker. Running the same checkpoint in
    this env, against whatever people source the YAML names, is what separates
    the two.

    Read it as a ladder, one YAML edit apart:
        people_source: ground_truth, no simulated noise  -> is the policy good
        ground_truth + simulated block C noise           -> does it survive C
        people_source: perception                        -> what block B costs

    Nothing is written to disk. The numbers are meant to be compared across
    those three runs, so print them and keep the three outputs side by side.
    """
    rows = []
    for index in range(episodes):
        observation, _ = env.reset()
        # RecurrentPPO carries its own hidden state between steps, and it has
        # to be dropped at an episode boundary or the first steps of episode
        # n+1 are conditioned on the end of episode n.
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        total_reward = 0.0
        while True:
            action, lstm_states = model.predict(
                observation, state=lstm_states, episode_start=episode_start,
                deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_start = np.zeros((1,), dtype=bool)
            if terminated or truncated:
                break
        rows.append({
            'outcome': info.get('episode_outcome', 'timeout'),
            'scenario': info.get('scenario', 'none'),
            'steps': info.get('episode_steps', 0),
            'return': total_reward,
            'intrusion': info.get('maximum_intrusion', 0.0),
            'unseen': info.get('maximum_hidden_intrusion', 0.0),
            'closest': info.get('minimum_person_distance', -1.0),
            'closest_true': info.get('minimum_person_distance_all', -1.0),
            'misread': info.get('corrupted_label_steps', 0),
        })
        row = rows[-1]
        print(f'  [{index + 1:>3}/{episodes}] {row["outcome"]:<13} '
              f'{row["scenario"]:<12} {row["steps"]:>4} steps  '
              f'return {row["return"]:>8.1f}  intrusion {row["intrusion"]:.2f} '
              f'(unseen {row["unseen"]:.2f})  closest {row["closest"]:>5.2f} m '
              f'(true {row["closest_true"]:>5.2f} m)', flush=True)
    return rows


EVAL_MAX_UNSEEN_GAP = 0.10
EVAL_MIN_CLEAR_EPISODES = 0.60
EVAL_MIN_TRUE_CLEARANCE = 0.60
EVAL_MIN_NONE_GOAL = 0.90


def evaluation_acceptance(rows, *, hidden_intrusion_available=True):
    """Calculate the four checkpoint acceptance gates used by ``--eval``.

    The inequalities deliberately match the written acceptance criteria: the
    unseen gap is strictly below 0.10, while the other three scores are
    strictly above their thresholds. Missing evidence is not a zero score and
    cannot pass -- notably an evaluation with no ``none`` episode says nothing
    about whether avoidance destroyed ordinary navigation.

    ``intrusion_gap`` is signed. Only unseen intrusion pulling above visible
    intrusion demonstrates the cheap "turn people out of frame" behaviour;
    taking an absolute value would reject the opposite measurement as if it
    were the same failure.
    """
    total = len(rows)
    if total:
        mean_intrusion = sum(row['intrusion'] for row in rows) / total
        mean_unseen = sum(row['unseen'] for row in rows) / total
        intrusion_gap = (mean_unseen - mean_intrusion
                         if hidden_intrusion_available else None)
        clear_episodes = (
            sum(1 for row in rows if row['intrusion'] <= 0.0) / total)
    else:
        intrusion_gap = None
        clear_episodes = None

    true_rows = [row for row in rows if row['closest_true'] >= 0.0]
    true_clearance = (sum(row['closest_true'] for row in true_rows)
                      / len(true_rows) if true_rows else None)

    none_rows = [row for row in rows if row['scenario'] == 'none']
    none_goal = (sum(1 for row in none_rows if row['outcome'] == 'goal')
                 / len(none_rows) if none_rows else None)

    values = {
        'intrusion_gap': intrusion_gap,
        'clear_episodes': clear_episodes,
        'true_clearance': true_clearance,
        'none_goal': none_goal,
    }
    passed = {
        'intrusion_gap': (intrusion_gap is not None
                          and intrusion_gap < EVAL_MAX_UNSEEN_GAP),
        'clear_episodes': (clear_episodes is not None
                           and clear_episodes > EVAL_MIN_CLEAR_EPISODES),
        'true_clearance': (true_clearance is not None
                           and true_clearance > EVAL_MIN_TRUE_CLEARANCE),
        'none_goal': none_goal is not None and none_goal > EVAL_MIN_NONE_GOAL,
    }
    return {
        'values': values,
        'passed': passed,
        'all_passed': all(passed.values()),
    }


def report_evaluation(rows, env_config):
    """Print the summary. One block, so three modes can be diffed by eye."""
    if not rows:
        print('[social_rl] no episodes ran')
        print('  VERDICT: FAIL -- do not select this checkpoint')
        return
    total = len(rows)

    def mean(key, rows=rows):
        return sum(row[key] for row in rows) / len(rows) if rows else 0.0

    print()
    print('=' * 72)
    print(f'  people_source        {env_config.people_source}')
    print(f'  camera cone          {env_config.people_camera_only}'
          f'  ({math.degrees(env_config.camera_fov):.0f} deg)')
    print(f'  occlusion            {env_config.people_occlusion}')
    print(f'  people memory        {env_config.people_memory_time:.1f} s moving'
          f' / {env_config.people_memory_time_still:.1f} s still')
    if env_config.vlm_noise:
        print(f'  simulated block C    ON  (wrong '
              f'{env_config.vlm_wrong_label_prob:.2f}, abstain '
              f'{env_config.vlm_abstain_prob:.2f}, every '
              f'{env_config.vlm_period:.2f} s; REWARD USES TRUE LABELS)')
    else:
        print('  simulated block C    off')
    print('-' * 72)
    for reason in ('goal', 'hit_obstacle', 'timeout'):
        count = sum(1 for row in rows if row['outcome'] == reason)
        print(f'  outcome/{reason:<16} {count / total:>6.1%}  ({count}/{total})')
    print(f'  mean episode steps     {mean("steps"):>6.1f}')
    print(f'  mean return            {mean("return"):>6.1f}')
    print('-' * 72)
    # The social half. `intrusion` is measured on the people visible to the
    # policy; `unseen` includes everybody in the ground-truth scene and is the
    # value the reward charges. They agree when the policy genuinely keeps
    # clear. `unseen` pulling above the visible peak means the trajectory still
    # passes near people outside the camera frame.
    print(f'  social/peak_intrusion  {mean("intrusion"):>6.3f}')
    print(f'  social/peak_unseen     {mean("unseen"):>6.3f}')
    clear = sum(1 for row in rows if row['intrusion'] <= 0.0)
    print(f'  social/clear_episodes  {clear / total:>6.1%}')
    if env_config.vlm_noise:
        # Sanity, not a score: 0 here with the noise switched on means the
        # model never fired and the run is secretly a clean one.
        print(f'  steps w/ misread label {mean("misread"):>6.1f} '
              f'of {mean("steps"):.1f}')
    seen = [row for row in rows if row['closest'] >= 0.0]
    true = [row for row in rows if row['closest_true'] >= 0.0]
    if seen:
        print(f'  mean closest person    {mean("closest", seen):>6.2f} m')
    if true:
        print(f'  mean closest, TRUE     {mean("closest_true", true):>6.2f} m')
    print('-' * 72)
    for scenario in sorted({row['scenario'] for row in rows}):
        subset = [row for row in rows if row['scenario'] == scenario]
        goals = sum(1 for row in subset if row['outcome'] == 'goal')
        print(f'  {scenario:<14} goal {goals / len(subset):>6.1%}  '
              f'intrusion {mean("intrusion", subset):.3f}  '
              f'(unseen {mean("unseen", subset):.3f})  n={len(subset)}')
    acceptance = evaluation_acceptance(
        rows,
        hidden_intrusion_available=(
            env_config.people_source == 'ground_truth'))
    values = acceptance['values']
    passed = acceptance['passed']

    def status(key):
        return 'PASS' if passed[key] else 'FAIL'

    print('-' * 72)
    print('  CHECKPOINT ACCEPTANCE (all four must pass)')
    gap = values['intrusion_gap']
    if gap is None:
        print('  [FAIL] peak_unseen - peak_intrusion  N/A < 0.100')
    else:
        print(f'  [{status("intrusion_gap")}] '
              f'peak_unseen - peak_intrusion  '
              f'{gap:>6.3f} < {EVAL_MAX_UNSEEN_GAP:.3f}')
    clear_rate = values['clear_episodes']
    if clear_rate is None:
        print('  [FAIL] clear episodes             N/A > 60.0%')
    else:
        print(f'  [{status("clear_episodes")}] clear episodes            '
              f'{clear_rate:>6.1%} > {EVAL_MIN_CLEAR_EPISODES:.1%}')
    true_clearance = values['true_clearance']
    if true_clearance is None:
        print('  [FAIL] mean closest, TRUE          N/A > 0.60 m')
    else:
        print(f'  [{status("true_clearance")}] mean closest, TRUE         '
              f'{true_clearance:>6.2f} m > '
              f'{EVAL_MIN_TRUE_CLEARANCE:.2f} m')
    none_goal = values['none_goal']
    if none_goal is None:
        print('  [FAIL] none goal                   N/A > 90.0%')
    else:
        print(f'  [{status("none_goal")}] none goal                  '
              f'{none_goal:>6.1%} > {EVAL_MIN_NONE_GOAL:.1%}')
    verdict = 'PASS -- select this checkpoint' if acceptance[
        'all_passed'] else 'FAIL -- do not select this checkpoint'
    print(f'  VERDICT: {verdict}')
    print('=' * 72)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG,
                        help='training YAML (default: the packaged one)')
    parser.add_argument('--run-dir', default=None,
                        help='where checkpoints and logs go '
                             '(default: runs/<timestamp> under train.output_root)')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='override train.total_timesteps')
    parser.add_argument('--resume', default=None,
                        help='continue from a saved .zip, or from a run '
                             'directory (its final_model.zip, else the newest '
                             'checkpoint). The old run is never written to: '
                             'this run gets its own directory as usual')
    parser.add_argument('--render', default='',
                        help="'true' opens the Gazebo window and watches the "
                             "robot train; empty keeps env.render from the YAML")
    parser.add_argument('--eval', default=None,
                        help='run a saved .zip (or a run directory) for '
                             '--eval-episodes episodes WITHOUT learning, then '
                             'print metrics and the fixed checkpoint acceptance '
                             'verdict. Pass an exact checkpoints/*.zip to '
                             'select an intermediate checkpoint; a run '
                             'directory evaluates its final_model.zip. Nothing '
                             'is trained or written. The people source and the '
                             'rest come from --config')
    parser.add_argument('--eval-episodes', type=int, default=40,
                        help='how many episodes --eval drives (default 40)')
    parser.add_argument('--eval-stochastic', action='store_true',
                        help='sample actions instead of taking the mean. The '
                             'default is deterministic, which is what '
                             'rl_agent.yaml runs on the robot')
    args, ros_args = parser.parse_known_args()

    config = load_config(args.config)
    env_config = EnvConfig.from_dict(config.get('env', {}))
    observation_config = ObservationConfig.from_dict(config.get('observation', {}))
    reward_config = RewardConfig.from_dict(config.get('reward', {}))
    train_config = config.get('train', {})
    resume_path = resolve_resume(args.resume) if args.resume else None
    if resume_path:
        check_resume_layout(resume_path, observation_config)
    if args.eval:
        if resume_path:
            raise RuntimeError('--eval and --resume do the opposite things; '
                               'pass one of them')
        eval_path = resolve_resume(args.eval)
        # The same layout check a resume gets, and for the same reason: the
        # CNN is sized by the observation, so a config that changed grid_size
        # or prediction_times would fail deep inside torch instead of here.
        check_resume_layout(eval_path, observation_config)
    if args.render:
        env_config = replace(
            env_config, render=args.render.lower() in ('true', '1', 'yes', 'on'))

    device = resolve_device(str(train_config.get('device', 'cuda:0')))
    if args.eval:
        # An evaluation is something you read while it runs and usually pipe to
        # a file to compare three of them; block buffering would hold every
        # line until the process exits.
        sys.stdout.reconfigure(line_buffering=True)
        print(f'[social_rl] eval    : {eval_path}')
        print(f'[social_rl] episodes: {args.eval_episodes}, '
              f'{"stochastic" if args.eval_stochastic else "deterministic"}')
        rclpy.init(args=ros_args)
        env = SocialAvoidEnv(env_config, observation_config, reward_config)
        rows = []
        try:
            model = RecurrentPPO.load(eval_path, device=device)
            # Ctrl+C still reports: half an evaluation is worth reading, and
            # 20 episodes is several minutes of simulation to throw away.
            rows = evaluate(model, env, args.eval_episodes,
                            deterministic=not args.eval_stochastic)
        except KeyboardInterrupt:
            print('\n[social_rl] interrupted')
        finally:
            report_evaluation(rows, env_config)
            env.close()
            rclpy.shutdown()
        return

    output_root = os.path.expanduser(
        train_config.get('output_root', '~/social_rl_runs'))
    run_dir = args.run_dir or os.path.join(
        output_root, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)

    # The exact env/observation/reward this policy was trained with, saved next
    # to it. agent_node.py loads this file rather than the training YAML, so
    # editing the YAML afterwards cannot change what a finished policy is fed.
    saved = {'env': env_config.to_dict(),
             'observation': observation_config.to_dict(),
             'reward': reward_config.to_dict()}
    if resume_path:
        # Recorded so a chain of resumed runs can be traced back to the one
        # that started from scratch. agent_node.py ignores this key.
        saved['resumed_from'] = resume_path
    with open(os.path.join(run_dir, 'env_config.yaml'), 'w') as handle:
        yaml.safe_dump(saved, handle, sort_keys=False)

    print(f'[social_rl] run dir : {run_dir}')
    print(f'[social_rl] device  : {device}'
          + (f' ({torch.cuda.get_device_name(0)}, '
             f'{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB)'
             if device.startswith('cuda') else ''))
    horizons = observation_config.constraint_field.prediction_times
    print(f'[social_rl] obs     : grid {observation_config.grid_channels}x'
          f'{observation_config.grid_size}x{observation_config.grid_size} '
          f'({observation_config.grid_extent:.1f} m box, '
          f'{observation_config.grid_resolution:.2f} m/cell) + 5 scalars')
    print(f'[social_rl] channels: 0 occupancy (block H), 1-'
          f'{len(horizons)} social constraint field (block D) at '
          + ', '.join(f't+{value:.1f}s' for value in horizons))

    rclpy.init(args=ros_args)
    env = Monitor(SocialAvoidEnv(env_config, observation_config, reward_config),
                  filename=os.path.join(run_dir, 'monitor.csv'))
    try:
        policy_kwargs = {
            # The recurrent part of the network. 128 units is small enough that
            # the whole rollout fits comfortably on a 4 GiB card and large
            # enough to hold a few seconds of where people were.
            'lstm_hidden_size': int(train_config.get('lstm_hidden_size', 128)),
            'n_lstm_layers': int(train_config.get('n_lstm_layers', 1)),
            # Actor and critic each get their own LSTM. Sharing one saves
            # memory this card does not need to save, and the two have
            # genuinely different jobs.
            'shared_lstm': False,
            'enable_critic_lstm': True,
            'net_arch': dict(pi=list(train_config.get('pi_layers', [128, 128])),
                             vf=list(train_config.get('vf_layers', [128, 128]))),
            # Reads the occupancy and social channels with a small CNN and
            # passes the five scalars through untouched. Stored in the .zip by
            # import path, which is why it lives in its own module.
            'features_extractor_class': SocialGridExtractor,
            'features_extractor_kwargs': {
                'cnn_features': int(train_config.get('cnn_features', 128))},
        }
        if resume_path:
            print(f'[social_rl] resuming from {resume_path}')
            # SB3 restores every hyperparameter from the .zip, so without this
            # a resumed run silently ignores the `train:` section of the YAML:
            # edit ent_coef, resume, and the old value keeps training. Only
            # keys that leave the network SHAPE alone can be forced -- the
            # architecture lives in policy_kwargs and set_parameters() runs
            # with exact_match=True, so a changed layer size fails on shapes.
            # load() applies these before _setup_model(), which is what makes
            # a changed n_steps rebuild the rollout buffer correctly.
            model = RecurrentPPO.load(resume_path, env=env, device=device,
                                      tensorboard_log=run_dir,
                                      custom_objects=resume_overrides(
                                          resume_path, train_config))
        else:
            model = RecurrentPPO(
                'MultiInputLstmPolicy', env,
                learning_rate=float(train_config.get('learning_rate', 3.0e-4)),
                # One rollout is n_steps control periods of real simulation:
                # 512 x 0.2 s is about 100 s of driving between updates.
                n_steps=int(train_config.get('n_steps', 512)),
                batch_size=int(train_config.get('batch_size', 128)),
                n_epochs=int(train_config.get('n_epochs', 10)),
                gamma=float(train_config.get('gamma', 0.99)),
                gae_lambda=float(train_config.get('gae_lambda', 0.95)),
                clip_range=float(train_config.get('clip_range', 0.2)),
                ent_coef=float(train_config.get('ent_coef', 0.005)),
                vf_coef=float(train_config.get('vf_coef', 0.5)),
                max_grad_norm=float(train_config.get('max_grad_norm', 0.5)),
                # Stop the epoch loop as soon as the policy has moved this far
                # from the one that collected the rollout. Without it SB3 runs
                # all n_epochs whatever happens, which is how run
                # 20260831_145903 died: one update at approx_kl 0.218 and
                # clip_fraction 0.446, and the very next rollout scored 0% on
                # every scenario -- `none` included, so the driving itself was
                # gone, not just the social part.
                target_kl=float(train_config.get('target_kl', 0.05)),
                policy_kwargs=policy_kwargs,
                tensorboard_log=run_dir,
                device=device,
                verbose=1)

        checkpoint = CheckpointCallback(
            save_freq=int(train_config.get('checkpoint_every', 5000)),
            save_path=os.path.join(run_dir, 'checkpoints'),
            name_prefix='recurrent_ppo')
        total = args.timesteps or int(train_config.get('total_timesteps', 300000))
        # Ctrl+C stops the run but still saves: a training that dies with
        # nothing on disk after two hours of simulation is the expensive
        # failure here.
        try:
            # reset_num_timesteps=False on a resume: the step counter, the
            # episode counter and the tensorboard x-axis carry on from the
            # checkpoint instead of restarting at zero, so `total` reads as
            # "train this many MORE steps".
            model.learn(total_timesteps=total,
                        callback=[checkpoint, OutcomeCallback()],
                        reset_num_timesteps=resume_path is None,
                        progress_bar=False)
        except KeyboardInterrupt:
            print('\n[social_rl] interrupted, saving what has been learned')
        model.save(os.path.join(run_dir, 'final_model'))
        print(f'[social_rl] saved {os.path.join(run_dir, "final_model.zip")}')
    finally:
        env.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
