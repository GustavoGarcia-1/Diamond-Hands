"""Unintegrated sensor/action contract for the next catch-training experiment.

This does not launch training, change deployed controllers, or define a reward.
The existing visual-retention reward remains a proxy, not proof of a catch.
"""
from dataclasses import dataclass

import numpy as np

from .late_reach_rl import observation as reach_observation
from .lower_cup_grip import OPEN

SCHEMA = 'catch-coordination-95-native-causal-v2'
OBS_SIZE = 95
NATIVE_NAMES = tuple(f'{side}_{name}' for side in ('left', 'right')
                     for name in ('len_if', 'len_mf', 'len_rf', 'len_pf',
                                  'len_th_abd', 'len_th1', 'len_th2'))
POSITION_SCALE = np.tile([.05, .05, .05, .05, 2., .02, .05], 2)
VELOCITY_SCALE = np.tile([.12, .12, .12, .12, 4.2, .12, .12], 2)


@dataclass(frozen=True)
class NativeSample:
    timestamp_s: float
    position: np.ndarray
    velocity: np.ndarray


class NativeEncoderHistory:
    """Finite differences of actual 50 Hz native encoder packets, not plant qpos."""
    def __init__(self):
        self.previous = None
        self.filtered_velocity = None

    def update(self, timestamp_s, position):
        now = float(timestamp_s)
        q = np.asarray(position, dtype=float)
        if not np.isfinite(now) or q.shape != (14,) or not np.isfinite(q).all():
            raise ValueError('Finite timestamp and14 native encoder values required')
        if self.previous is None:
            self.previous = (now, q.copy())
            return None  # Do not invent a zero velocity for the first packet.
        stamp, old = self.previous
        dt = now - stamp
        if not .01 - 1e-9 <= dt <= .041:
            raise ValueError('Native encoder packets require causal10..41ms spacing')
        velocity = (q - old) / dt
        if self.filtered_velocity is not None:
            velocity = .5 * self.filtered_velocity + .5 * velocity
        self.filtered_velocity = velocity.copy()
        self.previous = (now, q.copy())
        return NativeSample(now, q.copy(), velocity.copy())


def observation(packet, native):
    """Append encoder features with their own age; never retimestamp a sample."""
    if not isinstance(native, NativeSample):
        raise ValueError('A measured native position/velocity sample is required')
    age = float(packet['time_s']) - native.timestamp_s
    q, velocity = np.asarray(native.position), np.asarray(native.velocity)
    if not np.isfinite(age) or not -1e-9 <= age <= .041:
        raise ValueError('Native sample is future-dated or stale')
    if q.shape != (14,) or velocity.shape != (14,) or not np.isfinite(np.r_[q, velocity]).all():
        raise ValueError('Invalid native encoder sample')
    velocity_valid = bool(packet['visual_valid'] and packet['visual_velocity_valid'])
    visual_packet = dict(packet)
    if not velocity_valid:
        visual_packet['ball_cup_relative_velocity_body_mps'] = np.zeros(3)
    result = np.r_[reach_observation(visual_packet), float(velocity_valid),
                   (q - np.tile(OPEN, 2)) / POSITION_SCALE,
                   velocity / VELOCITY_SCALE, max(0., age) / .041]
    if result.shape != (OBS_SIZE,) or not np.isfinite(result).all():
        raise ValueError('Invalid catch-coordination observation')
    return result.astype(np.float32)


def decode_action(action):
    """Return reach-rate residual and a bounded finger-reference shift in metres.

    These are requests to existing bounded controllers, never direct torques or
    simulator state edits. Zero means no added residual. Driver integration and
    an exact zero-action rollout are mandatory before training this contract.
    """
    action = np.asarray(action, dtype=float)
    if action.shape != (2,) or not np.isfinite(action).all() or np.any(abs(action) > 1.):
        raise ValueError('Two finite actions in[-1,1] required')
    return float(action[0]), .002 * float(action[1])
