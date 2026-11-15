"""Sensor-only scalar reach-rate learning with exposure-age-aware observations."""
from __future__ import annotations
from queue import Queue
from threading import Thread
from pathlib import Path
import importlib
import json
import numpy as np
import gymnasium as gym

OBS_SCHEMA = 'late-reach-phase-65-capture-aligned-v2'
OBS_SIZE = 65
REWARD_SCHEMA = 'late-reach-sensor-tracking-visible-retention-v3'
VECTOR_SIZES = {'compact_error_rad': 14, 'receive_error_rad': 14, 'arm_velocity_rad_s': 14,
                'gravity_body': 3, 'imu_gyro_rad_s': 3, 'ball_cup_error_body_m': 3,
                'ball_cup_relative_velocity_body_mps': 3, 'ball_cup_sigma_m': 3}


class StopLateEpisode(Exception):
    pass


def observation(packet):
    for name, size in VECTOR_SIZES.items():
        a = np.asarray(packet[name], dtype=float)
        if a.shape != (size,) or not np.isfinite(a).all():
            raise ValueError(f'finite {size}-vector required: {name}')
    forecast, visible = bool(packet['forecast_valid']), bool(packet['visual_valid'])
    ttc = [packet['ttc_s'], packet['ttc_std_s']] if forecast else [0., 0.]
    visual = np.r_[packet['ball_cup_error_body_m'], packet['ball_cup_relative_velocity_body_mps'],
                   packet['ball_cup_sigma_m']] if visible else np.zeros(9)
    age = float(packet['time_s']-packet['ball_capture_s']) if visible else 0.
    if not -1e-8 <= age <= .08000001:
        raise ValueError('visible camera features require a causal fresh exposure timestamp')
    raw = np.r_[packet['compact_error_rad'], packet['receive_error_rad'], packet['arm_velocity_rad_s'],
                packet['gravity_body'], packet['imu_gyro_rad_s'], ttc, float(forecast),
                packet['phase'], packet['phase_rate_s_inv'], float(visible), visual,
                min(3., float(packet['visual_dwell_s'])) if visible else 0., max(0., age)]
    scale = np.r_[np.full(28, 2.), np.full(14, 6.), np.ones(3), np.full(3, 5.),
                  .7, .06, 1., 1., 4., 1., np.full(3, .2), np.full(3, 5.), np.full(3, .06), 3., .08]
    if raw.shape != (OBS_SIZE,) or scale.shape != (OBS_SIZE,) or not np.isfinite(raw).all():
        raise ValueError('invalid capture-aligned late-reach sensor observation')
    return (raw/scale).astype(np.float32)


class SensorReward:
    """Sensor proxy only; no physical-quality score or truth state is accepted.

    The archived fixed-phase ablation demonstrates false retention credit during
    excluded hand-mount support. Proximity, motion and visual dwell cannot prove
    bilateral load-bearing contact or a clean acquisition history. Keep v3 for
    exact reproduction; do not interpret its return as a catch-success score.
    """
    def __init__(self):
        self.last_time = self.last_capture = None
        self.paid = np.zeros(2, dtype=bool)
        self.total = 0.
        self.last_packet = None
        self.finished = False
        self.dwell = 0.
        self.last_qualified_capture = None
        self.coverage = dict(calls=0, forecast=0, forecast_over_half_second=0, visible=0,
                             retention_qualified=0, max_visual_dwell_s=0.)

    @staticmethod
    def visible(packet):
        capture = packet['ball_capture_s']
        sigma = np.asarray(packet['ball_cup_sigma_m'], dtype=float)
        error = np.asarray(packet['ball_cup_error_body_m'], dtype=float)
        velocity = np.asarray(packet['ball_cup_relative_velocity_body_mps'], dtype=float)
        return bool(packet['visual_valid'] and capture is not None
                    and np.isfinite([packet['time_s'], capture]).all()
                    and -1e-8 <= packet['time_s']-capture <= .08000001
                    and sigma.shape == (3,) and np.isfinite(sigma).all()
                    and error.shape == velocity.shape == (3,)
                    and np.isfinite(error).all() and np.isfinite(velocity).all()
                    and np.all((sigma >= 0.) & (sigma <= .06)))

    @staticmethod
    def receiving_form(packet):
        up = np.asarray(packet.get('palm_up_cos', []), dtype=float)
        height = np.asarray(packet.get('palm_height_above_shoulder_m', []), dtype=float)
        return bool(up.shape == height.shape == (2,) and np.isfinite(up).all()
                    and np.isfinite(height).all() and np.min(up) >= .6
                    and np.max(height) <= .15)

    def update(self, packet):
        if self.finished:
            raise RuntimeError('cannot reward an already finished episode')
        now = float(packet['time_s'])
        dt = .02 if self.last_time is None else now-self.last_time
        if not 0. < dt <= .041:
            raise ValueError('late-reach reward requires strictly causal20ms packets')
        self.last_time, self.last_packet = now, packet
        value = -dt*2.*max(0., .9-float(packet['upright_cos']))
        valid = bool(packet['forecast_valid'])
        ttc = float(packet['ttc_s']) if valid else None
        self.coverage['calls'] += 1
        self.coverage['forecast'] += int(valid)
        self.coverage['forecast_over_half_second'] += int(valid and ttc > .5)
        # Posture and angular motion are distinct. The early penalty is not
        # applied to an occluded holding phase after the reach has begun.
        early = (valid and ttc > .5) or (not valid and float(packet['phase']) == 0.)
        if early:
            value -= dt*(.12*np.mean(np.square(packet['compact_error_rad']))
                         + .025*np.mean(np.square(packet['arm_velocity_rad_s'])))
        if valid and 0. <= ttc <= .15:
            value -= dt*2.*float(np.mean(np.square(packet['receive_error_rad'])))
        capture = packet['ball_capture_s']
        new_capture = capture is not None and (self.last_capture is None or float(capture) > self.last_capture)
        visible = self.visible(packet)
        error = np.asarray(packet['ball_cup_error_body_m'], dtype=float)
        velocity = np.asarray(packet['ball_cup_relative_velocity_body_mps'], dtype=float)
        qualified = (visible and bool(packet['visual_velocity_valid']) and self.receiving_form(packet)
                     and np.isfinite(packet['upright_cos']) and packet['upright_cos'] >= .9
                     and error.shape == velocity.shape == (3,)
                     and np.isfinite(error).all() and np.isfinite(velocity).all()
                     and np.max(np.abs(error)) <= .12 and np.linalg.norm(velocity) <= .8)
        if not qualified:
            self.dwell, self.last_qualified_capture = 0., None
        if new_capture:
            capture = float(capture)
            gap = None if self.last_capture is None else capture-self.last_capture
            self.last_capture = capture
            self.coverage['visible'] += int(visible)
            if qualified:
                qualified_gap = None if self.last_qualified_capture is None else capture-self.last_qualified_capture
                self.dwell = (self.dwell+qualified_gap if qualified_gap is not None
                              and 0. < qualified_gap <= .08000001 else 0.)
                self.last_qualified_capture = capture
                self.coverage['retention_qualified'] += 1
                self.coverage['max_visual_dwell_s'] = max(self.coverage['max_visual_dwell_s'], self.dwell)
            if visible:
                phi = float(np.exp(-.5*np.sum((error/np.array([.15, .12, .15]))**2)))
                # A fresh close observation is useful evidence. Missing images
                # never pay; a newly visible ball must not incur a visibility
                # entry debit, as the earlier v2 shaping inadvertently did.
                capture_dt = min(dt, .04) if gap is None else min(.04, max(0., gap))
                value += .2*capture_dt*phi
                if bool(packet['visual_velocity_valid']) and gap is not None and 0. < gap <= .10000001:
                    value += gap*phi/(1.+float(velocity@velocity))
                    dwell = self.dwell
                    value += gap if dwell > 0. else 0.
                    for i, threshold in enumerate((.75, 2.)):
                        if dwell >= threshold and not self.paid[i]:
                            value += 2.
                            self.paid[i] = True
        self.total += value
        return float(value)

    def finish(self):
        if self.finished:
            raise RuntimeError('terminal reward must settle once')
        self.finished = True
        value = 0.
        # The driver may finish one control tick after its last sensor packet.
        # Continuous visual rewards already pay for sensor-proxy dwell; do not give
        # an endpoint bonus without an actual terminal sensor observation.
        self.total += value
        return float(value)


class LateReachEnv(gym.Env):
    """Threaded physics rollout; queues carry only the declared sensor packet."""
    def __init__(self, config, driver_module, episode_log=None, evaluation_dir=None):
        self.config, self.driver_module = dict(config), driver_module
        self.episode_log = None if episode_log is None else Path(episode_log)
        self.evaluation_dir = None if evaluation_dir is None else Path(evaluation_dir)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=np.float32)
        self.thread = None
        self.episode = 0
        self.last_vector = np.zeros(OBS_SIZE, dtype=np.float32)
        self.pending_terminal = None

    def _worker(self):
        reward = SensorReward()
        delivered, reset_debit, calls, started = 0., 0., 0, False
        output = None if self.evaluation_dir is None else self.evaluation_dir/f'episode-{self.episode:04d}'
        def policy(packet):
            nonlocal delivered, reset_debit, calls, started
            reward.update(packet)
            active = ((packet['forecast_valid'] and 0. <= packet['ttc_s'] <= .65)
                      or packet['phase'] > 0.)
            if not active and not started:
                return 0.
            if not started:
                started = True
                reset_debit = reward.total
                delivered = reward.total
            vector = observation(packet)
            increment = reward.total-delivered
            delivered = reward.total
            self.states.put(('state', vector, increment, {}))
            action = self.actions.get(timeout=120.)
            if action is None:
                raise StopLateEpisode()
            calls += 1
            return float(action)
        try:
            driver = importlib.import_module(self.driver_module)
            # Driver return values may include evaluation artifacts; none are
            # consumed for observation, reward, episode selection or critic.
            from g1_receiver.hand_route_plant import HandRoutePlant
            from g1_receiver.rgbd_impairments import RGBDImpairments
            impairments = RGBDImpairments(self.episode_config.get('sensor_impairments', {}))
            original_capture = HandRoutePlant.capture
            def capture(plant, camera='survey_rgbd'):
                frame, survey = original_capture(plant, camera)
                return impairments.apply(camera, frame, survey, frame.timestamp_s)
            try:
                HandRoutePlant.capture = capture
                driver.run_training_rollout(dict(self.episode_config), policy, output)
            finally:
                HandRoutePlant.capture = original_capture
            reward.finish()
            info = dict(sensor_return=reward.total-reset_debit, reset_debit=reset_debit,
                        policy_calls=calls, sensor_coverage=reward.coverage,
                        observation_schema=OBS_SCHEMA, reward_schema=REWARD_SCHEMA,
                        sensor_impairments=impairments.manifest())
            if self.episode_log is not None:
                self.episode_log.parent.mkdir(parents=True, exist_ok=True)
                with self.episode_log.open('a') as f:
                    f.write(json.dumps(dict(episode=self.episode, **info))+'\n')
            self.states.put(('done', None, reward.total-delivered, info))
        except StopLateEpisode:
            return
        except BaseException as error:
            import traceback
            self.states.put(('error', traceback.format_exc(), 0., {'error': repr(error)}))

    def _read(self):
        kind, vector, reward, info = self.states.get(timeout=180.)
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
        self.episode_config = dict(self.config)
        domains = self.episode_config.pop('sensor_domains', None)
        if domains:
            selected = dict(domains[int(self.np_random.integers(len(domains)))])
            selected['seed'] = int(self.np_random.integers(1, 2**31))
            self.episode_config['sensor_impairments'] = selected
        self.states, self.actions = Queue(), Queue()
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()
        kind, vector, reward, info = self._read()
        if kind == 'done':
            self.pending_terminal = (reward, info)
            vector = np.zeros(OBS_SIZE, dtype=np.float32)
            self.last_vector = vector
        return vector, info

    def step(self, action):
        value = np.asarray(action, dtype=float)
        if value.shape != (1,) or not np.isfinite(value).all() or abs(value[0]) > 1.+1e-8:
            raise ValueError('one normalized phase-rate action required')
        if self.pending_terminal is not None:
            reward, info = self.pending_terminal
            self.pending_terminal = None
            return self.last_vector.copy(), reward, True, False, info
        self.actions.put(float(value[0]))
        kind, vector, reward, info = self._read()
        return vector, reward, kind == 'done', False, info

    def close(self):
        if self.thread is not None and self.thread.is_alive():
            self.actions.put(None)
            self.thread.join(timeout=10.)
            if self.thread.is_alive():
                raise RuntimeError('Late-reach rollout failed to stop cleanly')
        self.thread = None
