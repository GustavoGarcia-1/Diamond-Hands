"""Process-isolated PPO bridge for sensor-only precontact arm coordination.

Each route driver owns its frozen simulation and evaluation outputs. This
environment discards its return value and reads only the sensor audit ledger.
"""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from queue import Queue
import tempfile
from threading import Thread
import traceback

import gymnasium as gym
import numpy as np

from .precontact_native_rl import PrecontactNativeAudit, ACTION_SIZE, OBS_SIZE, OBS_SCHEMA, REWARD_SCHEMA


def audit_definition(config):
    variant = config.get('audit_variant', 'original')
    if variant == 'original':
        return PrecontactNativeAudit, OBS_SCHEMA, REWARD_SCHEMA
    if variant == 'camera_translation':
        from .transport_precontact_rl import TransportPrecontactAudit, OBS_SCHEMA as obs, REWARD_SCHEMA as reward
        return TransportPrecontactAudit, obs, reward
    raise ValueError('unknown precontact audit variant')


class StopPrecontactEpisode(Exception):
    pass


class PrecontactEnv(gym.Env):
    def __init__(self, config, driver, episode_log=None, evaluation_dir=None):
        self.config = deepcopy(config)
        self.driver = driver
        self.episode_log = None if episode_log is None else Path(episode_log)
        self.evaluation_dir = None if evaluation_dir is None else Path(evaluation_dir)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_SIZE,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (ACTION_SIZE,), np.float32)
        self.thread = None
        self.episode = 0
        self.last_vector = np.zeros(OBS_SIZE, np.float32)
        self.pending_terminal = None

    def _worker(self):
        temporary = None
        try:
            if self.evaluation_dir is None:
                temporary = tempfile.TemporaryDirectory(prefix='receiver-precontact-')
                output = Path(temporary.name)
            else:
                output = self.evaluation_dir / f'episode-{self.episode:04d}'
                output.mkdir(parents=True, exist_ok=False)
            config = deepcopy(self.episode_config)
            profile = json.loads(Path(config['profile']).read_text())
            scenario = config.pop('scenario', {})
            release_shift = float(scenario.get('release_s', 0.))
            profile['play']['release_s'] += release_shift
            if 'break_release_delay_s' in config:
                config['break_release_delay_s'] += release_shift
            lateral_axis = np.asarray(config.get('training_lateral_axis_xy', [0., 1.]), float)
            if lateral_axis.shape != (2,) or not np.isfinite(lateral_axis).all() or not np.isclose(np.linalg.norm(lateral_axis), 1.):
                raise ValueError('unit surveyed training lateral axis required')
            for index in (0, 1):
                profile['play']['quarterback_target_m'][index] += float(scenario.get('lateral_m', 0.))*float(lateral_axis[index])
            profile['play']['quarterback_target_m'][2] += float(scenario.get('height_m', 0.))
            profile_path = output / 'episode-profile.json'
            profile_path.write_text(json.dumps(profile, indent=2)+'\n')
            config['profile'] = str(profile_path)
            delivered = 0.
            audit = None

            def choose_action(vector):
                nonlocal delivered
                reward = audit.ledger.step_return - delivered
                delivered = audit.ledger.step_return
                self.states.put(('state', np.asarray(vector, np.float32), reward, {}))
                action = self.actions.get(timeout=180.)
                if action is None:
                    raise StopPrecontactEpisode()
                return action

            audit_class, _, _ = audit_definition(config)
            audit = audit_class(profile, policy=choose_action)
            driver = importlib.import_module(self.driver)
            # Evaluation reports contain physical truth. Deliberately discard.
            driver.run_training_rollout(config, audit, output / 'rollout')
            terminal = audit.finish()
            summary = audit.save(output / 'audit', terminal)
            info = dict(sensor_return=float(audit.ledger.step_return),
                        reset_sensor_reward=float(audit.ledger.reset_reward),
                        policy_steps=int(audit.steps), episode=self.episode,
                        scenario=scenario, sensor_summary=summary)
            if self.episode_log is not None:
                self.episode_log.parent.mkdir(parents=True, exist_ok=True)
                with self.episode_log.open('a') as stream:
                    stream.write(json.dumps(info)+'\n')
            self.states.put(('done', None, audit.ledger.step_return-delivered, info))
        except StopPrecontactEpisode:
            return
        except BaseException as error:
            self.states.put(('error', traceback.format_exc(), 0., {'error': repr(error)}))
        finally:
            if temporary is not None:
                temporary.cleanup()

    def _read(self):
        kind, vector, reward, info = self.states.get(timeout=240.)
        if kind == 'error':
            raise RuntimeError(vector)
        if kind == 'state':
            self.last_vector = vector
        return kind, self.last_vector.copy(), reward, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        self.episode += 1
        self.pending_terminal = None
        self.episode_config = deepcopy(self.config)
        variation = self.episode_config.pop('training_randomization', None)
        if variation:
            self.episode_config['scenario'] = {
                key: float(self.np_random.uniform(-float(bound), float(bound)))
                for key, bound in variation.items()
            }
        self.states, self.actions = Queue(), Queue()
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()
        kind, vector, reward, info = self._read()
        if kind == 'done':
            self.pending_terminal = (reward, info)
            self.last_vector = vector = np.zeros(OBS_SIZE, np.float32)
        return vector, info

    def step(self, action):
        action = np.asarray(action, float)
        if action.shape != (ACTION_SIZE,) or not np.isfinite(action).all() or np.any(abs(action)>1.+1e-8):
            raise ValueError('finite normalized arm actions required')
        if self.pending_terminal is not None:
            reward, info = self.pending_terminal
            self.pending_terminal = None
            return self.last_vector.copy(), reward, True, False, info
        self.actions.put(action.copy())
        kind, vector, reward, info = self._read()
        return vector, reward, kind == 'done', False, info

    def close(self):
        if self.thread is not None and self.thread.is_alive():
            self.actions.put(None)
            self.thread.join(timeout=30.)
            if self.thread.is_alive():
                raise RuntimeError('precontact rollout failed to stop')
        self.thread = None
