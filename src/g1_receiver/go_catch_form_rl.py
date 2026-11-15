"""Opt-in human-informed catch form reward on the sensor-only Go callback.

The original 111D cup observation/reward and archived controllers are unchanged.
This version adds eight explicit context features and debits measured form costs.
Its CLI is a zero-action audit; passing a policy to GoCatchFormAudit supports the
same bounded residual interface without reading the simulation's live state.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

from .aero_residual_rl import ACTION_SIZE, OBS_SIZE, CORRECTED_REWARD_SCHEMA
from .catch_form import CatchFormContext, CatchFormEvaluator
from .go_cup_residual_rl import GoCupSensorAudit, compare_physics
from .perception import BallTracker, ColorBallDetector

FORM_OBSERVATION_SCHEMA = 'go-cup-catch-form-context-v2-hold-hypothesis'
FORM_OBSERVATION_SIZE = OBS_SIZE + 8
CONTEXT_FEATURES = (
    'approach', 'receive', 'hold', 'causal_intercept_valid',
    'intercept_height_above_shoulder_m', 'intercept_height_std_over_0p1m',
    'arrival_delay_over_1p5s', 'arrival_std_over_0p06s',
)


class VisualReceivePhase:
    """Form-only hold hypothesis; never latches positive retention reward.

The Go callback supplies the preceding control tick's visual state, at most
20ms old. Require a fresh, distinct accepted capture for either transition.
Occlusion preserves a form hypothesis but CupSensors still invalidates its
actual dwell/reward and terminal bonus normally.
"""
    def __init__(self):
        self.hold_hypothesis = False
        self.last_capture = None

    def update(self, now, sensors, enabled):
        if not enabled:
            return 'approach'
        stamp = sensors.last_valid
        fresh = (sensors.valid and stamp is not None and -1e-8 <= now-stamp <= .08+1e-8
                 and (self.last_capture is None or stamp > self.last_capture))
        if fresh:
            self.last_capture = stamp
            if np.linalg.norm(sensors.e) > .35:
                self.hold_hypothesis = False
            elif sensors.dwell >= .15:
                self.hold_hypothesis = True
        return 'hold' if self.hold_hypothesis else 'receive'


def context_vector(context):
    """Fixed scales; invalid trajectory estimates are zero with an explicit flag."""
    valid = bool(context.causal_fit_valid)
    return np.asarray([
        float(context.phase == phase) for phase in ('approach', 'receive', 'hold')
    ] + [float(valid)] + ([
        context.predicted_intercept_height_above_shoulder_m,
        context.intercept_height_std_m / .1,
        context.time_to_arrival_s / 1.5,
        context.arrival_std_s / .06,
    ] if valid else [0.] * 4), dtype=np.float32)


class CausalCatchContext:
    """Predict a descending pass at a commanded, surveyed receiving plane.

The target is a controller command, never an exact ball/pelvis position. Ball
flight comes from distinct RGB-D captures. Pelvis height comes from RGB-D;
shoulder height comes from encoder/IMU FK. The extra 3cm standard deviation is
an explicit engineering allowance for FK and vertical receiver motion, not a
fitted confidence claim. No known release or contact time enters this object.
"""

    def __init__(self, plane_point, plane_normal):
        self.plane_point = np.asarray(plane_point, dtype=float)
        self.plane_normal = np.asarray(plane_normal, dtype=float)
        if (self.plane_point.shape != (3,) or self.plane_normal.shape != (3,)
                or not np.isfinite([self.plane_point, self.plane_normal]).all()
                or np.linalg.norm(self.plane_normal) < 1e-9):
            raise ValueError('finite surveyed catch plane required')
        self.detector = ColorBallDetector(max_depth_m=20., ball_center_offset_m=.075)
        self.tracker = BallTracker(max_age_s=.08 + 1e-8, max_horizon_s=1.5,
                                   allow_synthetic_fixture=True)
        self.last_capture = None
        self.last_ball = None

    def update(self, now, pose, frame, survey, shoulder_height, phase):
        fallback = CatchFormContext(over_shoulder=True, phase=phase)
        if frame is None or survey is None:
            return fallback
        capture = float(frame.timestamp_s)
        if not -1e-8 <= now - capture <= .08 + 1e-8:
            return fallback
        if self.last_capture is None or capture > self.last_capture:
            self.last_capture = capture
            ball = self.detector.detect(frame, survey)
            self.last_ball = ball
            if ball is not None:
                self.tracker.add(ball)
        if (pose is None or pose.z_m is None or pose.z_std_m is None
                or not pose.fresh(now, .08 + 1e-8)
                or abs(capture - pose.timestamp_s) > .04 + 1e-8
                or pose.frame_id != 'field' or self.last_ball is None
                or pose.pose_provenance != self.last_ball.pose_provenance
                or not np.isfinite(shoulder_height)
                or not np.isfinite([pose.z_m, pose.z_std_m]).all() or pose.z_std_m < 0):
            return fallback
        crossing = self.tracker.intercept(now, self.plane_point, self.plane_normal, 'field')
        if (crossing is None or crossing.last_inlier_timestamp_s != capture
                or crossing.velocity_mps is None or crossing.velocity_mps[2] >= 0.
                or crossing.time_std_s > .06):
            return fallback
        variance = float(crossing.covariance_m2[2, 2]) + pose.z_std_m**2 + .03**2
        if not np.isfinite(variance) or variance < 0.:
            return fallback
        return CatchFormContext(
            over_shoulder=True, phase=phase, perception_timestamp_s=capture,
            causal_fit_valid=True,
            predicted_intercept_height_above_shoulder_m=float(crossing.position_m[2] - pose.z_m - shoulder_height),
            intercept_height_std_m=float(np.sqrt(variance)),
            time_to_arrival_s=float(crossing.timestamp_s - capture),
            arrival_std_s=float(crossing.time_std_s),
        )


class GoCatchFormAudit(GoCupSensorAudit):
    """Apply form cost only when the residual has control; log earlier form too."""

    def __init__(self, profile, policy=None):
        self.form_policy = policy
        self.form_rows = []
        self.form_observations = []
        self.form_actions = []
        self.current_context = CatchFormContext(over_shoulder=True, phase='inactive')
        self.context_features = context_vector(self.current_context)
        play = profile['play']
        self.context_tracker = CausalCatchContext(
            play['field_catch_window_m'], (*play['field_approach_axis_xy'], 0.))
        self.form_evaluator = None
        self.last_form_time = None
        self.total_form_cost = 0.
        self.form_version = None
        self.visual_phase = VisualReceivePhase()
        super().__init__(policy=self._choose_action)

    def _choose_action(self, cup_observation):
        observation = np.r_[cup_observation, self.context_features].astype(np.float32)
        if observation.shape != (FORM_OBSERVATION_SIZE,) or not np.isfinite(observation).all():
            raise ValueError('invalid form observation')
        action = (np.zeros(ACTION_SIZE) if self.form_policy is None
                  else np.asarray(self.form_policy(observation.copy()), dtype=float))
        self.form_observations.append(observation)
        self.form_actions.append(action.copy())
        return action

    def bind(self, model, joint_names):
        super().bind(model, joint_names)
        self.form_evaluator = CatchFormEvaluator(model, joint_names)

    def __call__(self, obs, pose, frame, survey, target, closure, enabled):
        now = float(obs.timestamp)
        if self.last_form_time is not None and now < self.last_form_time:
            raise ValueError('form samples must be causal')
        dt = 0. if self.last_form_time is None else now - self.last_form_time
        self.last_form_time = now
        geometry = self.form_evaluator.evaluate(
            obs, CatchFormContext(over_shoulder=True, phase='inactive'))
        # A hold phase requires a fresh visual dwell; simulator contact never
        # declares this transition. Post-catch tuck is allowed its own shape.
        phase = self.visual_phase.update(now, self.sensors, enabled)
        self.current_context = self.context_tracker.update(
            now, pose, frame, survey,
            geometry['shoulder_midpoint_height_above_pelvis_m'], phase)
        self.context_features = context_vector(self.current_context)
        metrics = self.form_evaluator.evaluate(obs, self.current_context)
        self.form_version = metrics['version']
        cost = dt * float(metrics['penalty_per_s']) if enabled else 0.
        self.sensors.reward -= cost
        self.total_form_cost += cost
        self.form_rows.append(dict(time_s=now, residual_enabled=bool(enabled),
                                   reward_debit=cost, context=asdict(self.current_context), **metrics))
        return super().__call__(obs, pose, frame, survey, target, closure, enabled)

    def save(self, directory, terminal_bonus):
        directory = Path(directory)
        summary = super().save(directory, terminal_bonus)
        summary.update(
            mode='zero_action_catch_form_audit' if self.form_policy is None else 'catch_form_policy_callback',
            observation_schema=FORM_OBSERVATION_SCHEMA, observation_size=FORM_OBSERVATION_SIZE,
            reward_schema=CORRECTED_REWARD_SCHEMA + '+' + str(self.form_version) + '+visual-hold-hypothesis-v1',
            total_form_cost_including_reset=self.total_form_cost,
            form_context_features=list(CONTEXT_FEATURES),
            earlier_arm_preparation='diagnosed only; residual cannot control pre-enable staged arms',
            hold_form_phase='visual hypothesis only, one control tick lag, no latched positive catch reward',
            form_runtime_inputs=['surveyed RGB-D', 'IMU', 'joint encoders', 'commanded catch plane/type'],
        )
        (directory/'form-samples.json').write_text(json.dumps(self.form_rows, indent=2, allow_nan=False)+'\n')
        np.savez_compressed(directory/'form-policy-trace.npz',
                            observation=np.asarray(self.form_observations), action=np.asarray(self.form_actions))
        (directory/'sensor-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        return summary


def main():
    script_root = Path(__file__).resolve().parents[2]/'scripts'
    if str(script_root) not in sys.path:
        sys.path.insert(0, str(script_root))
    import probe_go_cup as runner
    parser = runner.build_parser()
    parser.description = __doc__
    parser.add_argument('--baseline-trace', type=Path)
    parser.add_argument('--sensor-retention-camera-position', type=float, nargs=3)
    parser.add_argument('--sensor-retention-camera-target', type=float, nargs=3)
    parser.add_argument('--sensor-retention-pose-camera-position', type=float, nargs=3)
    parser.add_argument('--sensor-retention-pose-camera-target', type=float, nargs=3)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    args.output = output/'trial'
    audit = GoCatchFormAudit(json.loads(args.profile.read_text()))
    runner.run(args, residual_callback=audit)
    summary = audit.save(output, audit.finish())
    if args.baseline_trace is not None:
        equality = compare_physics(args.output/'trace.npz', args.baseline_trace)
        (output/'physics-comparison.json').write_text(json.dumps(equality, indent=2)+'\n')
        summary['physics_arrays_exact'] = equality['all_arrays_exact']
    summary['diagnostic_source_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    summary['retention_camera_position_m'] = args.sensor_retention_camera_position
    summary['retention_camera_target_m'] = args.sensor_retention_camera_target
    summary['retention_pose_camera_position_m'] = args.sensor_retention_pose_camera_position
    summary['retention_pose_camera_target_m'] = args.sensor_retention_pose_camera_target
    (output/'sensor-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
