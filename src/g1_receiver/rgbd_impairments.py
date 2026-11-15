"""Causal RGB-D transport and measurement stress; no simulator state input.

These are declared engineering test conditions, not a calibrated sensor model.
RGB/depth and their surveyed extrinsics retain the original exposure timestamp.
Unavailable transport returns the last delivered capture, or invalid depth if
no capture has arrived. It never fills an occlusion from simulator geometry.
"""
from collections import deque
from dataclasses import asdict, dataclass, replace
import hashlib
import numpy as np
from .perception import RGBDFrame


@dataclass(frozen=True)
class RGBDImpairmentConfig:
    seed: int = 9101
    latency_s: float = 0.
    latency_jitter_s: float = 0.
    frame_drop_probability: float = 0.
    rate_hz: float = 0.
    depth_noise_floor_m: float = 0.
    depth_noise_quadratic_per_m: float = 0.
    depth_quantization_m: float = 0.
    invalid_pixel_probability: float = 0.
    maximum_depth_m: float = 0.
    rgb_noise_std: float = 0.
    rgb_gain: float = 1.
    survey_position_std_m: float = 0.
    survey_angle_std_rad: float = 0.

    def __post_init__(self):
        values = asdict(self)
        if not np.isfinite(list(values.values())).all():
            raise ValueError('finite impairment parameters required')
        if self.seed < 0 or int(self.seed) != self.seed:
            raise ValueError('nonnegative integer seed required')
        if any(v < 0 for k, v in values.items() if k != 'seed'):
            raise ValueError('nonnegative impairment parameters required')
        if not 0 <= self.frame_drop_probability <= 1 or not 0 <= self.invalid_pixel_probability <= 1:
            raise ValueError('probabilities must be in [0, 1]')
        if self.latency_s+self.latency_jitter_s > .5 or self.rate_hz > 200:
            raise ValueError('bounded transport stress required')

    @property
    def identity(self):
        return all(v == (1. if k == 'rgb_gain' else 0.)
                   for k, v in asdict(self).items() if k != 'seed')


def rotation_vector_matrix(vector):
    angle = float(np.linalg.norm(vector))
    if angle == 0.:
        return np.eye(3)
    x, y, z = np.asarray(vector)/angle
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+np.sin(angle)*skew+(1.-np.cos(angle))*(skew@skew)


class RGBDImpairments:
    def __init__(self, config):
        self.config = config if isinstance(config, RGBDImpairmentConfig) else RGBDImpairmentConfig(**config)
        self.cameras = {}
        self.stats = {}

    def _camera(self, name):
        if name not in self.cameras:
            camera_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little')
            rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, camera_seed]))
            self.cameras[name] = dict(rng=rng, pending=deque(), delivered=None,
                                     last_exposure=None, last_now=None,
                                     translation=rng.normal(0., self.config.survey_position_std_m, 3),
                                     rotation=rotation_vector_matrix(rng.normal(0., self.config.survey_angle_std_rad, 3)))
            self.stats[name] = dict(calls=0, exposures=0, dropped=0, delivered=0,
                                    invalid_before_first_delivery=0, repeated=0, max_delivered_age_s=0.)
        return self.cameras[name], self.stats[name]

    def _measure(self, frame, survey, state):
        c, rng = self.config, state['rng']
        depth = frame.depth_m.copy()
        valid = np.isfinite(depth) & (depth > 0.)
        if c.maximum_depth_m:
            valid &= depth <= c.maximum_depth_m
        depth[~valid] = 0.
        if c.depth_noise_floor_m or c.depth_noise_quadratic_per_m:
            sigma = c.depth_noise_floor_m+c.depth_noise_quadratic_per_m*depth[valid]**2
            depth[valid] += rng.normal(0., sigma)
        if c.depth_quantization_m:
            depth[valid] = np.round(depth[valid]/c.depth_quantization_m)*c.depth_quantization_m
        if c.invalid_pixel_probability:
            depth[rng.random(depth.shape) < c.invalid_pixel_probability] = 0.
        depth[depth <= 0.] = 0.
        rgb = frame.rgb
        if c.rgb_noise_std or c.rgb_gain != 1.:
            rgb = np.clip(rgb.astype(float)*c.rgb_gain+rng.normal(0., c.rgb_noise_std, rgb.shape), 0., 255.).astype(np.uint8)
        observed = RGBDFrame(frame.timestamp_s, frame.frame_id, rgb, depth, frame.intrinsics)
        calibration = replace(survey, rotation=state['rotation']@survey.rotation,
                              translation=survey.translation+state['translation'],
                              position_std_m=float(np.hypot(survey.position_std_m, c.survey_position_std_m)),
                              angle_std_rad=float(np.hypot(survey.angle_std_rad, c.survey_angle_std_rad)))
        return observed, calibration

    def apply(self, name, frame, survey, now_s):
        if not np.isfinite(now_s) or frame.timestamp_s > now_s+1e-8:
            raise ValueError('cannot deliver a future exposure')
        if abs(frame.timestamp_s-survey.timestamp_s) > 1e-8:
            raise ValueError('RGB-D and survey must share exposure time')
        if self.config.identity:
            return frame, survey
        state, stats = self._camera(name)
        if state['last_now'] is not None and now_s < state['last_now']-1e-8:
            raise ValueError('transport clock must be monotonic')
        state['last_now'] = now_s
        stats['calls'] += 1
        period = 0. if not self.config.rate_hz else 1./self.config.rate_hz
        last = state['last_exposure']
        if last is None or (frame.timestamp_s > last+1e-8 and frame.timestamp_s-last >= period-1e-8):
            state['last_exposure'] = frame.timestamp_s
            stats['exposures'] += 1
            if state['rng'].random() < self.config.frame_drop_probability:
                stats['dropped'] += 1
            else:
                pair = self._measure(frame, survey, state)
                delay = max(0., self.config.latency_s+state['rng'].uniform(-self.config.latency_jitter_s, self.config.latency_jitter_s))
                state['pending'].append((frame.timestamp_s+delay, pair))
        available = [pair for due, pair in state['pending'] if due <= now_s+1e-8]
        state['pending'] = deque((due, pair) for due, pair in state['pending'] if due > now_s+1e-8)
        candidate = max(available, key=lambda pair: pair[0].timestamp_s) if available else None
        if candidate is not None and (state['delivered'] is None or candidate[0].timestamp_s > state['delivered'][0].timestamp_s+1e-8):
            state['delivered'] = candidate
            stats['delivered'] += 1
        else:
            stats['repeated'] += 1
        if state['delivered'] is None:
            stats['invalid_before_first_delivery'] += 1
            return replace(frame, rgb=np.zeros_like(frame.rgb), depth_m=np.zeros_like(frame.depth_m)), survey
        stats['max_delivered_age_s'] = max(stats['max_delivered_age_s'], now_s-state['delivered'][0].timestamp_s)
        return state['delivered']

    def manifest(self):
        return dict(config=asdict(self.config), calibrated_hardware_model=False,
                    runtime_inputs=['RGB image', 'registered axial depth', 'surveyed optical extrinsics', 'sensor clock'],
                    original_exposure_timestamps_preserved=True, no_geometry_or_state_fill=True,
                    cameras={name: dict(**self.stats[name],
                                        fixed_survey_translation_error_m=state['translation'].tolist(),
                                        fixed_survey_rotation_error=state['rotation'].tolist())
                             for name, state in self.cameras.items()})
