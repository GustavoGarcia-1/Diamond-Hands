"""Sensor-only route-learning contract; no ball contact or catch reward.

PPO actor and critic consume exactly the same vector. SGS observes a measured
route-tracking proxy, never evaluator contact, hidden root state, or catch truth.
The environment remains responsible for constructing these values from its
camera/encoder/IMU boundary; provenance labels cannot authenticate an arbitrary
caller's data.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .localization import ReceiverPose

OBSERVATION_SCHEMA = "receiver-route-sensors-v1"
OBSERVATION_SIZE = 81
ACTION_SIZE = 2


def _vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return array


@dataclass(frozen=True)
class RouteSensorState:
    timestamp_s: float
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    imu_quat_wxyz: np.ndarray
    imu_gyro: np.ndarray
    pose: ReceiverPose | None
    target_xy_m: np.ndarray
    command_field_velocity_mps: np.ndarray
    progress_m: float
    cross_track_error_m: float
    previous_action: np.ndarray

    def __post_init__(self):
        for name, size in (("joint_position", 29), ("joint_velocity", 29),
                           ("imu_quat_wxyz", 4), ("imu_gyro", 3),
                           ("target_xy_m", 2), ("command_field_velocity_mps", 2),
                           ("previous_action", 2)):
            value = _vector(getattr(self, name), size, name).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not np.isfinite([self.timestamp_s, self.progress_m, self.cross_track_error_m]).all():
            raise ValueError("sensor time and route measurements must be finite")
        if self.progress_m < 0 or self.cross_track_error_m < 0:
            raise ValueError("route progress and cross-track distance must be nonnegative")
        if not np.isclose(np.linalg.norm(self.imu_quat_wxyz), 1., atol=1e-3):
            raise ValueError("IMU quaternion must be normalized")
        if np.any(np.abs(self.previous_action) > 1.):
            raise ValueError("previous residual action must be normalized to [-1,1]")
        if self.pose is not None:
            if self.pose.frame_id != "field" or self.pose.pose_provenance not in {
                "fixed_calibration", "live_estimate", "synthetic_fixture"
            }:
                raise ValueError("receiver pose must be a declared sensor estimate in field")
            _vector(self.pose.xy_m, 2, "pose xy")
            covariance = np.asarray(self.pose.covariance)
            if (covariance.shape != (3, 3) or not np.isfinite(covariance).all()
                    or not np.allclose(covariance, covariance.T)
                    or np.linalg.eigvalsh(covariance).min() < -1e-10
                    or not np.isfinite([self.pose.timestamp_s, self.pose.yaw_rad]).all()):
                raise ValueError("invalid receiver-pose covariance or timestamp")


def pose_is_fresh(state: RouteSensorState, max_pose_age_s=.06):
    if not math.isfinite(max_pose_age_s) or max_pose_age_s <= 0:
        raise ValueError("pose age limit must be positive")
    return state.pose is not None and state.pose.fresh(state.timestamp_s, max_pose_age_s)


def build_observation(state: RouteSensorState, max_pose_age_s=.06):
    """Return the shared actor/critic input; stale camera features are masked."""
    valid = pose_is_fresh(state, max_pose_age_s)
    if valid:
        pose = state.pose
        c, s = np.cos(pose.yaw_rad), np.sin(pose.yaw_rad)
        field_to_body = np.array([[c, s], [-s, c]])
        camera = np.r_[pose.xy_m, s, c, np.diag(pose.covariance),
                       field_to_body @ (state.target_xy_m-pose.xy_m),
                       field_to_body @ state.command_field_velocity_mps,
                       state.progress_m, state.timestamp_s-pose.timestamp_s, 1.]
    else:
        camera = np.zeros(14)
        camera[-2] = max_pose_age_s
    values = np.r_[state.joint_position, state.joint_velocity,
                   state.imu_quat_wxyz, state.imu_gyro, camera, state.previous_action]
    if values.shape != (OBSERVATION_SIZE,):
        raise RuntimeError("observation schema mismatch")
    return values.astype(np.float32)


def upright_cosine(state: RouteSensorState):
    # Pelvis IMU orientation maps local into field; yaw does not affect upright.
    _, x, y, _ = state.imu_quat_wxyz
    return float(1 - 2 * (x*x + y*y))


def route_reward(previous: RouteSensorState | None, current: RouteSensorState,
                 residual, max_pose_age_s=.06):
    """A route tracking proxy, explicitly not catch/retention/competition success."""
    action = _vector(residual, ACTION_SIZE, "residual")
    if np.any(np.abs(action) > 1.):
        raise ValueError("residual action must be normalized to [-1,1]")
    valid = pose_is_fresh(current, max_pose_age_s)
    upright = upright_cosine(current)
    progress = 0.
    if previous is not None:
        dt = current.timestamp_s-previous.timestamp_s
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("reward requires increasing sensor timestamps")
        if valid and pose_is_fresh(previous, max_pose_age_s) and dt <= .2:
            # Bound localization jumps instead of rewarding a teleport-like estimate.
            progress = float(np.clip(current.progress_m-previous.progress_m, -dt, dt))
    tracking = math.exp(-(current.cross_track_error_m/.15)**2) if valid else 0.
    prior_action = previous.previous_action if previous is not None else current.previous_action
    smoothness = float(np.sum((action-prior_action)**2))
    effort_proxy = float(np.mean(action**2))
    reward = 10*progress + .05*tracking - .01*effort_proxy - .01*smoothness
    reward -= 0.1 if not valid else 0.
    reward -= max(0., .9-upright)
    return {
        "reward": float(reward), "pose_valid": bool(valid),
        "tracking_proxy_success": bool(valid and current.cross_track_error_m <= .15 and upright >= .9),
        "measured_progress_delta_m": progress,
        "upright_cosine": upright,
        "reward_basis": "camera_route_tracking_and_imu_proxy_not_catch",
    }


class SensorSGS:
    """Smoothed failure-weighted sampling from sensor tracking outcomes only."""

    def __init__(self, scenario_ids, minimum_weight=.1):
        self.scenario_ids = tuple(scenario_ids)
        if not self.scenario_ids or len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise ValueError("SGS requires unique scenario identifiers")
        if not math.isfinite(minimum_weight) or not 0 < minimum_weight <= 1:
            raise ValueError("minimum sampling weight must be in (0,1]")
        self.minimum_weight = minimum_weight
        self.attempts = np.zeros(len(self.scenario_ids), dtype=int)
        self.successes = np.zeros(len(self.scenario_ids), dtype=int)

    def update(self, scenario_id, tracking_proxy_success):
        if type(tracking_proxy_success) not in (bool, np.bool_):
            raise ValueError("SGS requires a boolean sensor tracking outcome")
        index = self.scenario_ids.index(scenario_id)
        self.attempts[index] += 1
        self.successes[index] += int(tracking_proxy_success)

    def probabilities(self):
        failure_rate = (self.attempts-self.successes+1)/(self.attempts+2)
        weights = np.maximum(self.minimum_weight, failure_rate)
        return weights/weights.sum()

    def snapshot(self):
        return {"basis": "sensor_route_tracking_proxy_not_catch",
                "scenario_ids": list(self.scenario_ids), "attempts": self.attempts.tolist(),
                "successes": self.successes.tolist(), "probabilities": self.probabilities().tolist()}
