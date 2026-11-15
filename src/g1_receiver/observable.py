"""Causal, sensor-estimated generalized coordinates for receiver learning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .localization import ReceiverPose
from .perception import BallObservation

SCHEMA = "receiver-observable-gencoord-v1"
JOINT_COUNT = 29
ARM_JOINTS = tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                  "wrist_roll", "wrist_pitch", "wrist_yaw")
)
ACTION_SIZE = 3 + len(ARM_JOINTS)
ACTION_LIMITS = np.array([.35, .22, .35] + [.90, .36, .22, .40, .20, .20, .20] * 2)
_FIELDS = (
    ("gravity_body", 3), ("gyro_body_rad_s", 3),
    ("joint_position_rad", JOINT_COUNT), ("joint_velocity_rad_s", JOINT_COUNT),
    ("receiver_field_xyz_m", 3), ("receiver_yaw_cos_sin", 2),
    ("receiver_field_velocity_mps", 3), ("receiver_yaw_rate_rad_s", 1),
    ("receiver_xyz_yaw_std", 4), ("receiver_age_s", 1), ("receiver_valid", 1),
    ("receiver_velocity_valid", 1),
    ("ball_relative_xyz_m", 3), ("ball_relative_velocity_mps", 3),
    ("ball_relative_xyz_std_m", 3), ("ball_age_s", 1), ("ball_valid", 1),
    ("ball_velocity_valid", 1), ("goal_body_xy_m", 2), ("goal_valid", 1),
    ("previous_action", ACTION_SIZE),
    ("previous_joint_targets_rad", JOINT_COUNT),
)
OBS_SLICES = {}
offset = 0
for name, width in _FIELDS:
    OBS_SLICES[name] = slice(offset, offset + width)
    offset += width
OBS_SIZE = offset
OBS_NAMES = tuple(
    name if width == 1 else f"{name}[{index}]"
    for name, width in _FIELDS for index in range(width)
)


def _vector(value, width: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (width,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite {width}-vector")
    return vector


def project_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quat = _vector(quaternion_wxyz, 4, "IMU quaternion")
    norm = np.linalg.norm(quat)
    if norm <= 1e-9:
        raise ValueError("IMU quaternion has zero norm")
    w, x, y, z = quat / norm
    return np.array([2 * (w*y - x*z), -2 * (w*x + y*z), 2 * (x*x + y*y) - 1])


def bounded_command(
    action: np.ndarray, nominal: np.ndarray, joint_names: list[str], joint_ranges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalized actor output to bounded WBC drive and joint targets."""
    values = _vector(action, ACTION_SIZE, "receiver action")
    command = np.clip(values, -1., 1.) * ACTION_LIMITS
    indices = [joint_names.index(name) for name in ARM_JOINTS]
    upper = np.array(nominal, dtype=float, copy=True)
    upper[indices] += command[3:]
    np.clip(upper, joint_ranges[:, 0], joint_ranges[:, 1], out=upper)
    return command[:3], upper


@dataclass(frozen=True)
class ObservableState:
    timestamp_s: float
    vector: np.ndarray
    receiver: ReceiverPose | None
    ball: BallObservation | None


class ObservableGenCoord:
    """Consume arriving RGB-D observations and synchronous encoder/IMU samples.

    A missing camera result invalidates that channel. A causal short history
    estimates rates only after enough distinct captures, without physics truth.
    """

    def __init__(
        self, *, allow_synthetic_fixture: bool = False, max_age_s: float = .12,
        velocity_window_s: float = .22,
    ) -> None:
        if not np.isfinite([max_age_s, velocity_window_s]).all() or min(max_age_s, velocity_window_s) <= 0:
            raise ValueError("observable timing limits must be finite and positive")
        self.allow_synthetic_fixture = allow_synthetic_fixture
        self.max_age_s = max_age_s
        self.velocity_window_s = velocity_window_s
        self.reset()

    def reset(self) -> None:
        self.receiver: ReceiverPose | None = None
        self.ball: BallObservation | None = None
        self.capture_s = float("-inf")
        self.receiver_samples: deque = deque()
        self.ball_samples: deque = deque()

    def update_camera(
        self, capture_s: float, receiver: ReceiverPose | None, ball: BallObservation | None,
    ) -> bool:
        if not np.isfinite(capture_s):
            raise ValueError("camera capture time must be finite")
        if capture_s <= self.capture_s:
            return False
        for item in (receiver, ball):
            if item is None:
                continue
            if item.frame_id != "field" or item.timestamp_s != capture_s:
                raise ValueError("camera observations must share field frame and capture timestamp")
            if item.pose_provenance == "synthetic_fixture":
                if not self.allow_synthetic_fixture:
                    raise ValueError("synthetic measurements require fixture opt-in")
            elif item.pose_provenance not in {"fixed_calibration", "live_estimate"}:
                raise ValueError("unknown observable provenance")
        if receiver is not None and ball is not None and receiver.pose_provenance != ball.pose_provenance:
            raise ValueError("camera observation provenance differs")
        if receiver is not None and receiver.z_m is not None and receiver.z_std_m is not None:
            covariance = np.asarray(receiver.covariance, dtype=float)
            if (
                covariance.shape != (3, 3) or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.T)
                or np.linalg.eigvalsh(covariance).min() < -1e-10
                or not np.isfinite(receiver.z_std_m) or receiver.z_std_m < 0
            ):
                raise ValueError("receiver uncertainty must be finite and positive semidefinite")
        self.capture_s = capture_s
        self.receiver = (receiver if receiver is not None and receiver.z_m is not None
                         and receiver.z_std_m is not None else None)
        self.ball = ball
        receiver_vector = (
            np.r_[self.receiver.xy_m, self.receiver.z_m, self.receiver.yaw_rad]
            if self.receiver is not None else None
        )
        self._append(self.receiver_samples, capture_s, receiver_vector)
        self._append(self.ball_samples, capture_s, ball.position_m if ball is not None else None)
        return True

    def _append(self, samples: deque, timestamp_s: float, vector: np.ndarray | None) -> None:
        if vector is None or (samples and timestamp_s - samples[-1][0] > self.max_age_s):
            samples.clear()
        if vector is not None:
            if not np.isfinite(vector).all():
                raise ValueError("camera coordinate must be finite")
            samples.append((timestamp_s, np.array(vector, copy=True)))
        while samples and timestamp_s - samples[0][0] > self.velocity_window_s:
            samples.popleft()

    @staticmethod
    def _velocity(samples: deque, width: int, *, yaw: bool = False) -> tuple[np.ndarray, bool]:
        if len(samples) < 3 or samples[-1][0] - samples[0][0] < .055:
            return np.zeros(width), False
        times = np.array([stamp for stamp, _ in samples])
        times -= times.mean()
        values = np.stack([vector for _, vector in samples])
        if yaw:
            values[:, 3] = np.unwrap(values[:, 3])
        return (times @ values) / (times @ times), True

    def observe(
        self, now_s: float, proprioception: tuple[np.ndarray, ...], goal_xy_m: np.ndarray,
        previous_action: np.ndarray, previous_joint_targets: np.ndarray,
    ) -> ObservableState:
        if not np.isfinite(now_s):
            raise ValueError("observable time must be finite")
        q, dq, quat, gyro = proprioception
        vector = np.zeros(OBS_SIZE, dtype=np.float32)
        vector[OBS_SLICES["gravity_body"]] = project_gravity(quat)
        vector[OBS_SLICES["gyro_body_rad_s"]] = _vector(gyro, 3, "IMU gyro")
        vector[OBS_SLICES["joint_position_rad"]] = _vector(q, JOINT_COUNT, "joint encoders")
        vector[OBS_SLICES["joint_velocity_rad_s"]] = _vector(dq, JOINT_COUNT, "joint velocities")
        vector[OBS_SLICES["previous_action"]] = np.clip(
            _vector(previous_action, ACTION_SIZE, "previous action"), -1., 1.
        )
        vector[OBS_SLICES["previous_joint_targets_rad"]] = _vector(
            previous_joint_targets, JOINT_COUNT, "previous joint targets"
        )
        goal = _vector(goal_xy_m, 2, "route goal")
        age = now_s - self.capture_s
        age_value = float(np.clip(age, 0., self.max_age_s)) if age >= 0 else self.max_age_s
        vector[OBS_SLICES["receiver_age_s"]] = age_value
        vector[OBS_SLICES["ball_age_s"]] = age_value
        if not 0 <= age <= self.max_age_s or self.receiver is None:
            return ObservableState(now_s, vector, None, None)

        receiver = self.receiver
        origin = np.r_[receiver.xy_m, receiver.z_m]
        c, s = np.cos(receiver.yaw_rad), np.sin(receiver.yaw_rad)
        to_body = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
        receiver_velocity, receiver_velocity_valid = self._velocity(self.receiver_samples, 4, yaw=True)
        xy_yaw_std = np.sqrt(np.maximum(0., np.diag(receiver.covariance)))
        receiver_std = np.r_[xy_yaw_std[:2], receiver.z_std_m, xy_yaw_std[2]]
        vector[OBS_SLICES["receiver_field_xyz_m"]] = origin
        vector[OBS_SLICES["receiver_yaw_cos_sin"]] = [c, s]
        vector[OBS_SLICES["receiver_field_velocity_mps"]] = receiver_velocity[:3]
        vector[OBS_SLICES["receiver_yaw_rate_rad_s"]] = receiver_velocity[3]
        vector[OBS_SLICES["receiver_xyz_yaw_std"]] = receiver_std
        vector[OBS_SLICES["receiver_valid"]] = 1
        vector[OBS_SLICES["receiver_velocity_valid"]] = receiver_velocity_valid
        vector[OBS_SLICES["goal_body_xy_m"]] = to_body[:2, :2] @ (goal - receiver.xy_m)
        vector[OBS_SLICES["goal_valid"]] = 1

        ball = self.ball
        if ball is not None:
            relative = to_body @ (ball.position_m - origin)
            ball_velocity, ball_velocity_valid = self._velocity(self.ball_samples, 3)
            relative_cov = to_body @ (ball.covariance_m2 + np.diag(receiver_std[:3]**2)) @ to_body.T
            yaw_jacobian = np.array([relative[1], -relative[0], 0.])
            relative_cov += receiver_std[3]**2 * np.outer(yaw_jacobian, yaw_jacobian)
            vector[OBS_SLICES["ball_relative_xyz_m"]] = relative
            vector[OBS_SLICES["ball_relative_xyz_std_m"]] = np.sqrt(
                np.maximum(0., np.diag(relative_cov))
            )
            vector[OBS_SLICES["ball_valid"]] = 1
            if ball_velocity_valid and receiver_velocity_valid:
                velocity = to_body @ (ball_velocity - receiver_velocity[:3])
                velocity += yaw_jacobian * receiver_velocity[3]
                vector[OBS_SLICES["ball_relative_velocity_mps"]] = velocity
                vector[OBS_SLICES["ball_velocity_valid"]] = 1
        if not np.isfinite(vector).all():
            raise ValueError("nonfinite observable feature")
        return ObservableState(now_s, vector, receiver, ball)
