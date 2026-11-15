"""Evaluator-only held-out rendered throws. Never imported by a controller."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from g1_receiver.perception import BallTracker, CameraIntrinsics, ColorBallDetector, RGBDFrame, RigidPose
from g1_receiver.sim import simulate


def predict(sensor, delay_s, dropout, rng):
    """Only sensor arrays enter the perception path; truth is unavailable here."""
    detector = ColorBallDetector()
    tracker = BallTracker(allow_synthetic_fixture=True)
    intrinsics = CameraIntrinsics(*sensor['intrinsics'])
    records = []
    detection_count = 0
    dropped = 0
    for i, timestamp in enumerate(sensor['timestamps_s']):
        now = float(timestamp + delay_s)
        skipped = rng.random() < dropout
        observation = None
        if skipped:
            dropped += 1
        else:
            depth = sensor['depth_m'][i].astype(float).copy()
            depth += rng.normal(0, 0.005, depth.shape)
            frame = RGBDFrame(float(timestamp), 'camera_optical', sensor['rgb'][i], depth, intrinsics)
            pose = RigidPose(float(sensor['camera_pose_timestamps_s'][i]), 'camera_optical', 'field',
                             sensor['camera_rotation'][i], sensor['camera_translation_m'][i],
                             'synthetic_fixture')
            observation = detector.detect(frame, pose)
        if observation is not None:
            detection_count += 1
            tracker.add(observation)
        result = tracker.intercept(now, np.array([1., 0., 0.]), np.array([1., 0., 0.]), 'field')
        records.append(None if result is None else {
            'now_s': now, 'crossing_timestamp_s': float(result.timestamp_s),
            'position_m': result.position_m.tolist(),
        })
    return records, detection_count, dropped


def crossing(truth):
    xyz = truth['ball_position_m']
    indices = np.flatnonzero((xyz[:-1, 0] >= 1.) & (xyz[1:, 0] < 1.))
    if len(indices) == 0:
        return None
    i = int(indices[0])
    fraction = (xyz[i, 0] - 1.) / (xyz[i, 0] - xyz[i+1, 0])
    return (float(np.interp(fraction, [0., 1.], truth['timestamps_s'][i:i+2])),
            xyz[i] * (1-fraction) + xyz[i+1] * fraction)


def summarize(values):
    return None if not values else {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)), 'max': float(max(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/perception-validation.json'))
    parser.add_argument('--throws', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261111)
    args = parser.parse_args()
    if args.throws <= 0:
        parser.error('--throws must be positive')
    rng = np.random.default_rng(args.seed)
    rows = []
    conditions = [('nominal', 0., 0.), ('delay_50ms_dropout_30pct', .05, .3), ('stale_120ms', .12, 0.)]
    for trial in range(args.throws):
        release = (float(rng.uniform(4.2, 5.8)), float(rng.uniform(-.3, .3)), float(rng.uniform(1.2, 1.6)))
        velocity = (float(rng.uniform(-6.5, -4.8)), float(-release[1]/.8), float(rng.uniform(2.8, 4.0)))
        camera = (.1, float(rng.uniform(-.05, .05)), float(rng.uniform(1.18, 1.26)))
        sensor, truth, _ = simulate(45, release_position_m=release, release_velocity_mps=velocity, camera_position_m=camera)
        for name, delay, dropout in conditions:
            # Finish inference before evaluator truth is inspected.
            records, detections, dropped = predict(sensor, delay, dropout, rng)
            target = crossing(truth)
            errors, time_errors = [], []
            late = 0
            for record in records:
                if record is None or target is None:
                    continue
                target_time, target_position = target
                if record['now_s'] >= target_time:
                    late += 1
                    continue
                errors.append(float(np.linalg.norm(np.asarray(record['position_m'])-target_position)))
                time_errors.append(abs(record['crossing_timestamp_s']-target_time))
            rows.append({'trial': trial, 'condition': name, 'release_position_m': release,
                         'release_velocity_mps': velocity, 'camera_position_m': camera,
                         'frames': len(records), 'detections': detections, 'dropped_frames': dropped,
                         'no_prediction_frames': sum(x is None for x in records),
                         'predictions_before_true_crossing': len(errors), 'predictions_after_true_crossing': late,
                         'true_crossing_available': target is not None,
                         'position_error_m': summarize(errors), 'time_error_s': summarize(time_errors)})
        print(f'Validated rendered throw {trial+1}/{args.throws}', file=sys.stderr)
    report = {'schema': 'receiver-perception-validation-v1', 'evidence_class': 'rendered_simulation',
              'seed': args.seed, 'held_out_definition': 'Random release/camera variations not used to tune detector or fit parameters; no policy training.',
              'noise': {'depth_std_m': .005}, 'trials': rows, 'catch_demonstrated': False,
              'limitations': ['Ideal pinhole images and fixed calibrated camera', 'Unvalidated ellipsoid football physics; no aerodynamic drag',
                              'Visible-surface predictions compared with evaluator-only center truth',
                              'Delay is capture-to-query age, not a simulated concurrent communications system',
                              'No real camera, locomotion, or contact validation',
                              'Renderer may report limited depth precision on this platform']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
