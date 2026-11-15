"""Bounded PPO for scalar late-reach rate using the capture-aligned sensor contract."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

from g1_receiver.late_reach_rl import LateReachEnv, OBS_SCHEMA, OBS_SIZE, REWARD_SCHEMA


def make_env(config, driver, output, index, seed):
    from stable_baselines3.common.monitor import Monitor
    env = LateReachEnv(config, driver, episode_log=output/f'worker-{index:02d}-episodes.jsonl')
    env.action_space.seed(seed+index)
    return Monitor(env)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--driver', required=True, help='Importable rollout module implementing the declared callback API')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wall-seconds', type=int, default=3600)
    p.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=2)
    p.add_argument('--seed', type=int, default=173)
    p.add_argument('--evaluate', type=Path, help='Evaluate this checkpoint instead of training')
    p.add_argument('--zero', action='store_true', help='Evaluate the matched zero-rate-residual heuristic')
    a = p.parse_args()
    if not 30 <= a.wall_seconds <= 5400:
        p.error('training must be bounded to30..5400seconds')
    if a.evaluate and a.zero:
        p.error('choose model or zero evaluation')
    a.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(a.config.read_text())
    if a.evaluate or a.zero:
        # Matched evaluation is explicitly nominal. Held-out sensor stresses
        # are evaluated separately, never randomly selected for a headline.
        config.pop('sensor_domains', None)
    # Spawned workers and the rollout thread must resolve the same driver.
    script_root = str(Path(__file__).resolve().parent)
    if script_root not in sys.path:
        sys.path.insert(0, script_root)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv
    import torch
    metadata = dict(observation_schema=OBS_SCHEMA, reward_schema=REWARD_SCHEMA,
                    learned_component='scalar reach-phase rate residual; visual cue, arm geometry, grip and locomotion remain separately scripted/pretrained',
                    action_size=1, observation_size=OBS_SIZE, seed=a.seed, config=config,
                    config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),
                    torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
                    no_physical_score_reward_or_selection=True)
    (a.output/'configuration.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if a.evaluate or a.zero:
        model = None if a.zero else PPO.load(a.evaluate, device='cuda')
        if model is not None and model.receiver_sensor_schema != OBS_SCHEMA:
            raise ValueError('matching capture-aligned late-reach model required')
        if model is not None:
            (a.output/'evaluated-model.json').write_text(json.dumps(dict(
                checkpoint_sha256=hashlib.sha256(a.evaluate.read_bytes()).hexdigest(),
                model_training_reward_schema=model.receiver_reward_schema,
                evaluation_reward_schema=REWARD_SCHEMA), indent=2)+'\n')
        env = LateReachEnv(config, a.driver, evaluation_dir=a.output/'evaluation')
        try:
            obs, _ = env.reset(seed=a.seed)
            total, rows = 0., []
            while True:
                action = np.zeros(1, dtype=np.float32) if model is None else model.predict(obs, deterministic=True)[0]
                rows.append((obs.copy(), action.copy()))
                obs, value, done, _, info = env.step(action)
                total += value
                if done:
                    break
            if not np.isclose(total, info['sensor_return'], atol=1e-9):
                raise RuntimeError('evaluation reward ledger mismatch')
            np.savez_compressed(a.output/'policy-trace.npz', observation=[x for x, _ in rows], action=[x for _, x in rows])
            (a.output/'evaluation.json').write_text(json.dumps(dict(returned_reward=total, **info), indent=2)+'\n')
        finally:
            env.close()
        return

    class BoundedTraining(BaseCallback):
        def __init__(self):
            super().__init__()
            self.started = time.monotonic()
            self.last_log = self.started
            self.last_save = 0
        def _on_step(self):
            now = time.monotonic()
            if self.num_timesteps-self.last_save >= 2048:
                self.model.save(a.output/f'checkpoint-{self.num_timesteps:08d}.zip')
                self.last_save = self.num_timesteps
            if now-self.last_log >= 10.:
                with (a.output/'progress.jsonl').open('a') as f:
                    f.write(json.dumps(dict(elapsed_s=now-self.started, timesteps=self.num_timesteps,
                                            mean_episode_reward=float(np.mean([x['r'] for x in self.model.ep_info_buffer]))
                                            if self.model.ep_info_buffer else None))+'\n')
                self.last_log = now
            return now-self.started < a.wall_seconds

    env = SubprocVecEnv([partial(make_env, config, a.driver, a.output, i, a.seed) for i in range(a.workers)],
                       start_method='spawn')
    started = time.monotonic()
    try:
        model = PPO('MlpPolicy', env, device='cuda', seed=a.seed,
                    n_steps=64, batch_size=64, n_epochs=4, learning_rate=2e-4,
                    gamma=.99, gae_lambda=.95, ent_coef=.01,
                    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])), verbose=0)
        model.receiver_sensor_schema = OBS_SCHEMA
        model.receiver_reward_schema = REWARD_SCHEMA
        model.save(a.output/'initial.zip')
        model.learn(total_timesteps=10_000_000, callback=BoundedTraining())
        model.save(a.output/'final.zip')
        (a.output/'receipt.json').write_text(json.dumps(dict(elapsed_s=time.monotonic()-started,
                timesteps=model.num_timesteps, workers=a.workers, wall_budget_s=a.wall_seconds,
                final_checkpoint_sha256=hashlib.sha256((a.output/'final.zip').read_bytes()).hexdigest(),
                **metadata), indent=2)+'\n')
    finally:
        env.close()


if __name__ == '__main__':
    main()
