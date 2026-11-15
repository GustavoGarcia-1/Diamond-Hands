"""Bounded PPO using the opt-in Out form reward and explicit sensor context.

Run the zero-action form audit first. This trainer does not select a controller
from proxy reward: it archives paired physical evaluations for later review.
"""
from __future__ import annotations

import argparse
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from .aero_residual_rl import AeroResidualEnv, StopEpisode, ACTION_SIZE, CORRECTED_REWARD_SCHEMA
from .catch_form import FORM_SCHEMA
from .go_catch_form_rl import GoCatchFormAudit, FORM_OBSERVATION_SCHEMA, FORM_OBSERVATION_SIZE

REWARD_SCHEMA = CORRECTED_REWARD_SCHEMA + '+' + FORM_SCHEMA + '+visual-hold-hypothesis-v1'


def out_argv(profile, source, amp_source, output, duration=8.):
    """Exact selected Out CLI; controller/route/cameras remain fixed."""
    return ['--source', str(source), '--amp-source', str(amp_source), '--output', str(output),
            '--duration', str(duration), '--approach-handoff-depth-m', '2.1', '--cut-speed', '1.4',
            '--blend-s', '.5', '--turn-s', '.4', '--hold-straight-s', '0', '--depth-feedback-gain', '4',
            '--camera-start-s', '.8', '--stopping-lookahead-s', '.55', '--reverse-speed', '0',
            '--approach-speed', '.48', '--horizontal-pass-distance', '--pass-profile', str(profile),
            '--record-poses']


class OutCatchFormEnv(AeroResidualEnv):
    """Queue adapter; only the audit's 119 sensor/command features reach PPO."""

    def __init__(self, profile, source, amp_source, episode_log=None, evaluation_dir=None,
                 duration=8.,
                 sensor_retention_camera_position=None, sensor_retention_camera_target=None,
                 sensor_retention_pose_camera_position=None, sensor_retention_pose_camera_target=None):
        super().__init__(profile, source, amp_source, episode_log, evaluation_dir)
        self.duration = duration
        self.sensor_camera_position = sensor_retention_camera_position
        self.sensor_camera_target = sensor_retention_camera_target
        self.sensor_pose_camera_position = sensor_retention_pose_camera_position
        self.sensor_pose_camera_target = sensor_retention_pose_camera_target
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(FORM_OBSERVATION_SIZE,), dtype=np.float32)
        self.last_vector = np.zeros(FORM_OBSERVATION_SIZE, dtype=np.float32)

    def _worker(self):
        temporary = None
        try:
            script_root = Path(__file__).resolve().parents[2]/'scripts'
            if str(script_root) not in sys.path:
                sys.path.insert(0, str(script_root))
            import probe_aero_out as runner
            delivered_return = 0.
            audit = None

            def choose_action(vector):
                nonlocal delivered_return
                # The callback ledger has already separated the discarded reset
                # reward before this call. Do not reconstruct reward from scores.
                reward = audit.ledger.step_return - delivered_return
                delivered_return = audit.ledger.step_return
                self.states.put(('state', vector, reward, dict(
                    sensor_valid=audit.sensors.valid, visual_dwell_s=audit.sensors.dwell,
                    form_cost_including_reset=audit.total_form_cost)))
                action = self.actions.get(timeout=self.timeout_s)
                if action is None:
                    raise StopEpisode()
                return action

            audit = GoCatchFormAudit(json.loads(self.reference.read_text()), policy=choose_action)
            if self.evaluation_dir is None:
                temporary = tempfile.TemporaryDirectory(prefix='out-form-rollout-')
                output = Path(temporary.name)
            else:
                output = self.evaluation_dir/f'episode-{self.episode:04d}'
                output.mkdir(parents=True, exist_ok=False)
            runner.main(out_argv(self.reference, self.source, self.amp_source,
                                 output/'trial', self.duration), residual_callback=audit,
                        sensor_retention_camera=(self.sensor_camera_position, self.sensor_camera_target)
                            if self.sensor_camera_position is not None else None,
                        sensor_retention_pose_camera=(self.sensor_pose_camera_position, self.sensor_pose_camera_target)
                            if self.sensor_pose_camera_position is not None else None)
            # The runner's physical report is never read in this environment.
            terminal = audit.finish()
            if self.evaluation_dir is not None:
                audit.save(output, terminal)
            reward = audit.ledger.step_return - delivered_return
            info = dict(sensor_return=audit.ledger.step_return,
                        reset_sensor_reward=audit.ledger.reset_reward,
                        policy_steps=audit.steps, final_visual_dwell_s=audit.sensors.dwell,
                        valid_captures=audit.sensors.valid_count,
                        camera_captures=audit.sensors.capture_count,
                        max_visual_dwell_s=audit.sensors.max_dwell,
                        form_cost_including_reset=audit.total_form_cost)
            self.states.put(('done', self.last_vector.copy(), reward, info))
        except StopEpisode:
            pass
        except BaseException as error:
            self.states.put(('error', repr(error), 0., {}))
        finally:
            if temporary is not None:
                temporary.cleanup()


def evaluate_pair(args, model, output):
    results = []
    for label, lateral_shift in (('nominal', 0.), ('aim-minus-2cm', -.02), ('aim-plus-2cm', .02)):
        directory = output/label
        directory.mkdir(parents=True)
        profile = json.loads(args.profile.read_text())
        for key in ('quarterback_target_m', 'field_catch_window_m'):
            profile['play'][key][1] += lateral_shift
        profile_path = directory/'profile.json'
        profile_path.write_text(json.dumps(profile, indent=2)+'\n')
        for policy_name in ('zero', 'learned'):
            env = OutCatchFormEnv(profile_path, args.source, args.amp_source,
                                evaluation_dir=directory/policy_name, duration=args.duration,
                                sensor_retention_camera_position=args.sensor_retention_camera_position,
                                sensor_retention_camera_target=args.sensor_retention_camera_target,
                                sensor_retention_pose_camera_position=args.sensor_retention_pose_camera_position,
                                sensor_retention_pose_camera_target=args.sensor_retention_pose_camera_target)
            try:
                observation, _ = env.reset(seed=args.seed)
                total = 0.
                while True:
                    action = (np.zeros(ACTION_SIZE, dtype=np.float32) if policy_name == 'zero'
                              else model.predict(observation, deterministic=True)[0])
                    observation, reward, done, _, info = env.step(action)
                    total += reward
                    if done:
                        break
                results.append(dict(case=label, policy=policy_name, returned_reward=total, **info))
            finally:
                env.close()
    (output/'paired-sensor-results.json').write_text(json.dumps(results, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('profile', 'source', 'amp-source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8192)
    parser.add_argument('--n-envs', type=int, default=4)
    parser.add_argument('--wall-seconds', type=float, default=1200.)
    parser.add_argument('--duration', type=float, default=8.)
    parser.add_argument('--seed', type=int, default=173)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--sensor-retention-camera-position', type=float, nargs=3)
    parser.add_argument('--sensor-retention-camera-target', type=float, nargs=3)
    parser.add_argument('--sensor-retention-pose-camera-position', type=float, nargs=3)
    parser.add_argument('--sensor-retention-pose-camera-target', type=float, nargs=3)
    args = parser.parse_args()
    if not 1 <= args.n_envs <= 8 or not 0 < args.steps <= 262144 or not 0 < args.wall_seconds <= 6000:
        parser.error('require bounded steps, 1..8 environments, and <=6000 seconds')
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    metadata = dict(observation_schema=FORM_OBSERVATION_SCHEMA, observation_size=FORM_OBSERVATION_SIZE,
                    reward_schema=REWARD_SCHEMA, action_size=ACTION_SIZE, seed=args.seed,
                    profile_sha256=sha256(args.profile.read_bytes()).hexdigest(),
                    steps_requested=args.steps, n_envs=args.n_envs,
                    sensor_retention_camera_position=args.sensor_retention_camera_position,
                    sensor_retention_camera_target=args.sensor_retention_camera_target,
                    sensor_retention_pose_camera_position=args.sensor_retention_pose_camera_position,
                    sensor_retention_pose_camera_target=args.sensor_retention_pose_camera_target,
                    runtime_inputs=['RGB-D', 'IMU', 'joint encoders', 'commanded catch plane/type'],
                    learned_component='bounded arm/closure residual; baseline route and gait remain fixed',
                    paired_evaluation='nominal and fixed +/-2cm throw/catch-window lateral perturbations')
    for name in ('catch_form.py', 'go_catch_form_rl.py', 'out_catch_form_training.py'):
        metadata[name+'_sha256'] = sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    (args.output/'contract.json').write_text(json.dumps(metadata, indent=2)+'\n')
    env = None
    try:
        if args.smoke:
            env = OutCatchFormEnv(args.profile, args.source, args.amp_source,
                                evaluation_dir=args.output/'smoke', duration=args.duration,
                                sensor_retention_camera_position=args.sensor_retention_camera_position,
                                sensor_retention_camera_target=args.sensor_retention_camera_target,
                                sensor_retention_pose_camera_position=args.sensor_retention_pose_camera_position,
                                sensor_retention_pose_camera_target=args.sensor_retention_pose_camera_target)
            observation, _ = env.reset(seed=args.seed)
            total = 0.
            while True:
                observation, reward, done, _, info = env.step(np.zeros(ACTION_SIZE, dtype=np.float32))
                total += reward
                if done:
                    break
            metadata.update(smoke=True, actual_timesteps=info['policy_steps'], returned_reward=total,
                            **info)
            if not np.isclose(total, info['sensor_return'], atol=1e-9):
                raise RuntimeError('Gym rewards differ from audit ledger')
        else:
            import torch
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.vec_env import SubprocVecEnv
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required for requested training')
            torch.set_num_threads(1)
            metadata['cuda_device'] = torch.cuda.get_device_name(0)
            env = SubprocVecEnv([
                partial(OutCatchFormEnv, args.profile, args.source, args.amp_source,
                        episode_log=args.output/f'episodes-{i}.jsonl', duration=args.duration,
                        sensor_retention_camera_position=args.sensor_retention_camera_position,
                        sensor_retention_camera_target=args.sensor_retention_camera_target,
                        sensor_retention_pose_camera_position=args.sensor_retention_pose_camera_position,
                        sensor_retention_pose_camera_target=args.sensor_retention_pose_camera_target) for i in range(args.n_envs)
            ], start_method='spawn')

            class BoundedTraining(BaseCallback):
                def _on_step(self):
                    return time.monotonic()-start < args.wall_seconds

                last_save_s = 0.

                def _on_rollout_start(self):
                    with (args.output/'progress.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(timesteps=self.num_timesteps,
                            optimizer_epochs=int(self.model._n_updates), elapsed_s=time.monotonic()-start))+'\n')
                    elapsed = time.monotonic()-start
                    if elapsed-self.last_save_s >= 300.:
                        self.model.save(args.output/f'checkpoint-{self.num_timesteps:08d}')
                        self.last_save_s = elapsed

            model = PPO('MlpPolicy', env, device='cuda', seed=args.seed, n_steps=64,
                        batch_size=64, n_epochs=4, learning_rate=1e-4,
                        policy_kwargs=dict(net_arch=[128,128], log_std_init=-2.), verbose=1)
            torch.nn.init.zeros_(model.policy.action_net.weight)
            torch.nn.init.zeros_(model.policy.action_net.bias)
            model.receiver_sensor_schema = FORM_OBSERVATION_SCHEMA
            model.receiver_reward_schema = REWARD_SCHEMA
            model.save(args.output/'initial')
            model.learn(total_timesteps=args.steps, callback=BoundedTraining())
            model.save(args.output/'final')
            metadata.update(actual_timesteps=model.num_timesteps, optimizer_epochs=int(model._n_updates))
            env.close()
            env = None
            evaluate_pair(args, model, args.output/'evaluation')
    finally:
        if env is not None:
            env.close()
        metadata['elapsed_s'] = time.monotonic()-start
        (args.output/'receipt.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
