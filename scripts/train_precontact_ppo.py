"""Bounded CUDA PPO for precontact arm residuals; physical evaluation is separate."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from g1_receiver.precontact_native_rl import ACTION_SIZE, OBS_SIZE, OBS_SCHEMA, REWARD_SCHEMA
from g1_receiver.precontact_training import PrecontactEnv, audit_definition


def make_env(config, driver, output, index, seed):
    from stable_baselines3.common.monitor import Monitor
    env = PrecontactEnv(config, driver, episode_log=output/f'episodes-{index:02d}.jsonl')
    env.action_space.seed(seed+index)
    return Monitor(env)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--driver', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wall-seconds', type=int, default=3600)
    p.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=2)
    p.add_argument('--seed', type=int, default=2026111517)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--zero', action='store_true')
    mode.add_argument('--evaluate', type=Path)
    a = p.parse_args()
    if not 30 <= a.wall_seconds <= 5400:
        p.error('bounded wall time 30..5400 required')
    a.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(a.config.read_text())
    _, observation_schema, reward_schema = audit_definition(config)
    config['policy_kind'] = 'zero' if a.zero else 'checkpoint' if a.evaluate else 'stochastic_train'
    metadata = dict(observation_schema=observation_schema, reward_schema=reward_schema,
                    action_size=ACTION_SIZE, observation_size=OBS_SIZE, seed=a.seed,
                    config=config, config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),
                    learned_component='14 bounded arm residuals before conservative sensor-predicted arrival; last residual held afterward',
                    fixed_components='native grip including 1.5mm encoder regulation, six thumb requests, gait, visual cue and nominal arm geometry',
                    physical_evaluation='separate full-play catch rescore; no simulator truth in actor, critic or reward')
    (a.output/'configuration.json').write_text(json.dumps(metadata, indent=2)+'\n')
    from stable_baselines3 import PPO
    import torch
    torch.set_num_threads(1)
    if a.zero or a.evaluate:
        config.pop('training_randomization', None)
        model = None if a.zero else PPO.load(a.evaluate, device='cuda')
        if model is not None and (model.receiver_sensor_schema != observation_schema
                                  or model.receiver_reward_schema != reward_schema):
            raise ValueError('checkpoint observation schema mismatch')
        env = PrecontactEnv(config, a.driver, evaluation_dir=a.output/'evaluation')
        try:
            obs, _ = env.reset(seed=a.seed)
            total = 0.
            while True:
                action = np.zeros(ACTION_SIZE, np.float32) if model is None else model.predict(obs, deterministic=True)[0]
                obs, reward, done, _, info = env.step(action)
                total += reward
                if done:
                    break
            if not np.isclose(total, info['sensor_return'], atol=1e-9):
                raise RuntimeError('Gym and sensor reward ledger mismatch')
            (a.output/'evaluation.json').write_text(json.dumps(dict(returned_reward=total, **info), indent=2)+'\n')
        finally:
            env.close()
        return
    if not torch.cuda.is_available():
        raise RuntimeError('requested PPO requires CUDA')
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    class BoundedTraining(BaseCallback):
        def __init__(self):
            super().__init__()
            self.started = time.monotonic()
            self.last_save = 0.

        def _on_step(self):
            return time.monotonic()-self.started < a.wall_seconds

        def _on_rollout_start(self):
            elapsed = time.monotonic()-self.started
            row = dict(elapsed_s=elapsed, timesteps=self.num_timesteps,
                       optimizer_epochs=int(self.model._n_updates))
            with (a.output/'progress.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            if elapsed-self.last_save >= 300.:
                self.model.save(a.output/f'checkpoint-{self.num_timesteps:08d}.zip')
                self.last_save = elapsed

    env = SubprocVecEnv([partial(make_env, config, a.driver, a.output, i, a.seed)
                         for i in range(a.workers)], start_method='spawn')
    started = time.monotonic()
    try:
        model = PPO('MlpPolicy', env, device='cuda', seed=a.seed, n_steps=64,
                    batch_size=64, n_epochs=4, learning_rate=1e-4, gamma=.99,
                    gae_lambda=.95, ent_coef=0., clip_range=.15,
                    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=-2.), verbose=0)
        # The deterministic initial actor is exactly the corrected scripted baseline.
        torch.nn.init.zeros_(model.policy.action_net.weight)
        torch.nn.init.zeros_(model.policy.action_net.bias)
        model.receiver_sensor_schema = observation_schema
        model.receiver_reward_schema = reward_schema
        model.save(a.output/'initial.zip')
        model.learn(total_timesteps=10_000_000, callback=BoundedTraining())
        model.save(a.output/'final.zip')
        (a.output/'receipt.json').write_text(json.dumps(dict(elapsed_s=time.monotonic()-started,
            timesteps=model.num_timesteps, optimizer_epochs=int(model._n_updates),
            cuda_device=torch.cuda.get_device_name(0), workers=a.workers, wall_budget_s=a.wall_seconds,
            final_checkpoint_sha256=hashlib.sha256((a.output/'final.zip').read_bytes()).hexdigest(),
            **metadata), indent=2)+'\n')
    finally:
        env.close()


if __name__ == '__main__':
    main()
