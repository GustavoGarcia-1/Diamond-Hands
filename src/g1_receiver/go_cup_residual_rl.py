"""Sensor-only residual diagnostics on the physically retained Go baseline.

No training entry point is provided. Actor/critic features and reward reuse the
fixed normalized 111D CupSensors schema and corrected potential-debit reward.
Physics traces are read only after the episode for an independent comparison.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np

from .aero_residual_rl import (
    ACTION_SIZE, OBS_SIZE, NORMALIZED_SCHEMA, CORRECTED_REWARD_SCHEMA,
    CupSensors, EpisodeReturnLedger, bounded_residual, scale_observation,
)
from .observable import ARM_JOINTS


class GoCupSensorAudit:
    """Zero-action callback with explicit sensor packets and separate FK state."""
    def __init__(self, policy=None):
        self.policy = policy
        self.sensors = None
        self.action = np.zeros(ACTION_SIZE)
        self.actual = np.zeros(ACTION_SIZE)
        self.ticks = self.steps = self.calls = self.disabled_ticks = 0
        self.ledger = EpisodeReturnLedger()
        self.rows = []
        self.camera_samples = []
        self.last_camera = None
        self.first_enabled_s = None
        self.finished = False

    def bind(self, model, joint_names):
        # Runner supplies an independently constructed static calibration model.
        self.sensors = CupSensors(model, joint_names, corrected_potential_reward=True)
        self.arms = [joint_names.index(name) for name in ARM_JOINTS]
        self.ranges = np.array([model.joint(name).range.copy() for name in ARM_JOINTS])

    def __call__(self, obs, pose, frame, survey, target, closure, enabled):
        if self.sensors is None:
            raise RuntimeError('callback requires static-model binding')
        self.calls += 1
        if frame is not None and survey is not None and frame.timestamp_s != self.last_camera:
            self.last_camera = frame.timestamp_s
            ball = self.sensors.detector.detect(frame, survey)
            self.camera_samples.append({
                'time_s': float(obs.timestamp), 'capture_s': float(frame.timestamp_s),
                'enabled': bool(enabled), 'ball_detected': ball is not None,
                'ball_rgbd_m': None if ball is None else ball.position_m.tolist(),
                'ball_covariance_m2': None if ball is None else ball.covariance_m2.tolist(),
                'ball_camera_position_m': survey.translation.tolist(),
                'pose_time_s': None if pose is None else float(pose.timestamp_s),
                'pose_z_m': None if pose is None else pose.z_m,
                'pose_z_std_m': None if pose is None else pose.z_std_m,
                'pose_covariance': None if pose is None else pose.covariance.tolist(),
            })
        if not enabled:
            self.disabled_ticks += 1
            return target, closure
        if self.first_enabled_s is None:
            self.first_enabled_s = float(obs.timestamp)
        self.sensors.update(obs, pose, frame, survey)
        if self.ticks % 2 == 0:
            raw = self.sensors.vector(obs, self.action, self.actual)
            normalized = scale_observation(raw)
            reward = float(self.sensors.reward)
            self.sensors.reward = 0.
            self.ledger.record(reward, reset=self.steps == 0)
            requested = np.zeros(ACTION_SIZE) if self.policy is None else np.asarray(self.policy(normalized.copy()), dtype=float)
            if requested.shape != (ACTION_SIZE,) or not np.isfinite(requested).all() or np.any(np.abs(requested) > 1. + 1e-8):
                raise ValueError('policy must return sixteen finite normalized actions in [-1,1]')
            self.action = requested.copy()
            self.rows.append(dict(
                time_s=float(obs.timestamp), valid=bool(self.sensors.valid),
                velocity_valid=bool(self.sensors.velocity_valid),
                cup_error_m=self.sensors.e.tolist(), cup_velocity_mps=self.sensors.vel.tolist(),
                covariance_diag_m2=(self.sensors.sigma**2).tolist(),
                sigma_m=self.sensors.sigma.tolist(), visual_dwell_s=float(self.sensors.dwell),
                upright_cosine=float(self.sensors.upright), reward=reward,
                reset_reward=self.steps == 0, raw=raw, normalized=normalized,
                action=self.action.copy(), previous_residual=self.actual.copy(),
            ))
            self.steps += 1
        old = self.actual.copy()
        self.actual = bounded_residual(self.action, self.actual)
        self.sensors.reward -= .02 * (.05 * float(np.mean(self.action**2))
                                     + .02 * float(np.mean(((self.actual-old)/.02)**2)))
        self.ticks += 1
        result = np.asarray(target).copy()
        result[self.arms] += self.actual[:14]
        result[self.arms] = np.clip(result[self.arms], self.ranges[:, 0], self.ranges[:, 1])
        curls = np.broadcast_to(np.asarray(closure, dtype=float), (2,)).copy()
        return result, np.clip(curls + self.actual[14:], 0., 1.)

    def finish(self):
        if self.finished:
            raise RuntimeError('terminal reward must be settled once')
        self.finished = True
        self.sensors.finish()
        terminal = 2. if self.sensors.valid and self.sensors.dwell >= .75 and self.sensors.upright >= .90 else 0.
        self.ledger.record(float(self.sensors.reward + terminal))
        self.sensors.reward = 0.
        return terminal

    def save(self, directory, terminal_bonus):
        directory = Path(directory)
        rows = self.rows
        def values(key, size):
            return np.asarray([r[key] for r in rows], dtype=float).reshape(-1, size)
        np.savez_compressed(directory/'sensor-trace.npz',
            timestamp_s=np.asarray([r['time_s'] for r in rows]),
            raw_observation=values('raw', OBS_SIZE), normalized_observation=values('normalized', OBS_SIZE),
            action=values('action', ACTION_SIZE), previous_residual=values('previous_residual', ACTION_SIZE))
        serial = [{k: v for k, v in r.items() if k not in ('raw', 'normalized', 'action', 'previous_residual')} for r in rows]
        (directory/'sensor-samples.json').write_text(json.dumps(dict(policy_samples=serial, camera_samples=self.camera_samples), indent=2)+'\n')
        valid = [r for r in rows if r['valid']]
        def quantiles(key):
            if not valid:
                return None
            a = np.asarray([r[key] for r in valid])
            return dict(p05=np.quantile(a,.05,axis=0).tolist(), median=np.quantile(a,.5,axis=0).tolist(), p95=np.quantile(a,.95,axis=0).tolist())
        summary = dict(
            mode='zero_action_sensor_audit' if self.policy is None else 'sensor_policy_callback',
            training_performed=False, observation_schema=NORMALIZED_SCHEMA,
            reward_schema=CORRECTED_REWARD_SCHEMA, observation_size=OBS_SIZE, action_size=ACTION_SIZE,
            action_arm_order=list(ARM_JOINTS), action_arm_limit_rad=.18, action_closure_limit=.15,
            first_enabled_s=self.first_enabled_s, callback_calls=self.calls,
            disabled_ticks=self.disabled_ticks, policy_steps=self.steps,
            camera_frames_total=len(self.camera_samples), camera_captures=self.sensors.capture_count,
            valid_captures=self.sensors.valid_count, valid_policy_frames=len(valid),
            max_visual_dwell_s=float(self.sensors.max_dwell), final_visual_dwell_s=float(self.sensors.dwell),
            min_visible_cup_distance_m=self.sensors.min_cup_distance,
            cup_error_m=quantiles('cup_error_m'), covariance_diag_m2=quantiles('covariance_diag_m2'),
            cup_velocity_mps=quantiles('cup_velocity_mps'),
            reset_sensor_reward=self.ledger.reset_reward, sensor_return=self.ledger.step_return,
            sensor_return_including_reset=self.ledger.step_return+self.ledger.reset_reward,
            terminal_sensor_bonus=terminal_bonus,
            runtime_inputs=['surveyed RGB-D ball/receiver estimates', 'IMU', '29 joint encoders', 'action history'],
            excluded_from_actor_critic_reward=['simulation root/ball poses', 'contacts', 'replay qpos', 'physics catch scores'],
        )
        (directory/'sensor-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        return summary


def compare_physics(current, reference):
    """Independent post-episode equality check; never used by actor or reward."""
    current, reference = Path(current), Path(reference)
    with np.load(current, allow_pickle=False) as a, np.load(reference, allow_pickle=False) as b:
        keys_match = set(a.files) == set(b.files)
        arrays = {}
        for key in sorted(set(a.files) | set(b.files)):
            if key not in a or key not in b:
                arrays[key] = dict(exact=False, reason='missing key')
                continue
            x, y = a[key], b[key]
            exact = x.shape == y.shape and np.array_equal(x, y, equal_nan=True) if x.dtype.kind not in 'SU' and y.dtype.kind not in 'SU' else np.array_equal(x,y)
            detail = dict(exact=bool(exact), shape_current=list(x.shape), shape_reference=list(y.shape))
            if x.shape == y.shape and x.dtype.kind in 'iufb' and y.dtype.kind in 'iufb':
                finite = np.isfinite(x) & np.isfinite(y)
                detail['maximum_absolute_difference'] = float(np.max(np.abs(x[finite].astype(float)-y[finite].astype(float)), initial=0.))
            arrays[key] = detail
    return dict(keys_match=keys_match, all_arrays_exact=keys_match and all(v['exact'] for v in arrays.values()),
                current_sha256=sha256(current.read_bytes()).hexdigest(), reference_sha256=sha256(reference.read_bytes()).hexdigest(), arrays=arrays)


def main():
    script_root = Path(__file__).resolve().parents[2]/'scripts'
    if str(script_root) not in sys.path:
        sys.path.insert(0, str(script_root))
    import probe_go_cup as runner
    parser = runner.build_parser()
    parser.description = __doc__
    parser.add_argument('--sensor-retention-camera-position', type=float, nargs=3)
    parser.add_argument('--sensor-retention-camera-target', type=float, nargs=3)
    parser.add_argument('--baseline-trace', type=Path, help='Independent equality reference, read only after rollout')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    args.output = output/'trial'
    audit = GoCupSensorAudit()
    runner.run(args, residual_callback=audit)
    terminal = audit.finish()
    summary = audit.save(output, terminal)
    if args.baseline_trace is not None:
        equality = compare_physics(args.output/'trace.npz', args.baseline_trace)
        (output/'physics-comparison.json').write_text(json.dumps(equality, indent=2)+'\n')
        summary['physics_arrays_exact'] = equality['all_arrays_exact']
        summary['physics_comparison_file'] = 'physics-comparison.json'
    summary['retention_camera_position_m'] = args.sensor_retention_camera_position
    summary['retention_camera_target_m'] = args.sensor_retention_camera_target
    summary['runner_sha256'] = sha256(Path(runner.__file__).read_bytes()).hexdigest()
    summary['diagnostic_module_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    (output/'sensor-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
