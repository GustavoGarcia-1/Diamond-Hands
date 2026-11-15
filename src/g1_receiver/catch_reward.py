"""Visual-retention proxy from camera ball/receiver estimates, never contacts."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .localization import ReceiverPose
from .perception import BallObservation


@dataclass(frozen=True)
class RetentionEstimate:
    status: str
    timestamp_s: float
    visible: bool
    relative_position_m: np.ndarray | None = None
    relative_velocity_mps: np.ndarray | None = None
    covariance_m2: np.ndarray | None = None
    stable_duration_s: float = 0.
    retained_proxy: bool = False


class VisualRetentionProxy:
    """Require sustained observed low relative motion inside a receiving volume.

    This is a visible-surface proxy and cannot establish contact, grip, or a
    competition catch. A missing, stale, or out-of-order observation resets its
    dwell. No hidden state is used to bridge an occlusion.
    """

    def __init__(self, *, allow_synthetic_fixture=False, required_duration_s=2.,
                 max_age_s=.08, max_gap_s=.10, max_relative_speed_mps=.35):
        values = [required_duration_s, max_age_s, max_gap_s, max_relative_speed_mps]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('retention limits must be finite and positive')
        self.allow_synthetic_fixture = allow_synthetic_fixture
        self.required_duration_s = required_duration_s
        self.max_age_s = max_age_s
        self.max_gap_s = max_gap_s
        self.max_relative_speed_mps = max_relative_speed_mps
        self.reset()

    def reset(self):
        self._history = deque(maxlen=12)
        self._stable_since = None
        self._last_capture_s = None

    def _reject(self, status, now_s, visible=False):
        self.reset()
        return RetentionEstimate(status, now_s, visible)

    def update(self, ball: BallObservation | None, receiver: ReceiverPose | None,
               now_s: float) -> RetentionEstimate:
        if not np.isfinite(now_s):
            raise ValueError('retention query time must be finite')
        if ball is None or receiver is None:
            return self._reject('missing_measurement', now_s, ball is not None)
        if ball.frame_id != 'field' or receiver.frame_id != 'field':
            raise ValueError('retention measurements must use the field frame')
        if ball.pose_provenance != receiver.pose_provenance:
            raise ValueError('ball and receiver provenance must match')
        if ball.pose_provenance == 'synthetic_fixture' and not self.allow_synthetic_fixture:
            raise ValueError('synthetic retention observations require explicit fixture opt-in')
        if not (0 <= now_s-ball.timestamp_s <= self.max_age_s and receiver.fresh(now_s, self.max_age_s)):
            return self._reject('stale_measurement', now_s, True)
        if abs(ball.timestamp_s-receiver.timestamp_s) > .04:
            return self._reject('measurement_time_skew', now_s, True)
        if receiver.z_m is None or receiver.z_std_m is None:
            return self._reject('missing_receiver_height', now_s, True)
        if self._last_capture_s is not None:
            gap = ball.timestamp_s-self._last_capture_s
            if gap <= 0:
                return self._reject('out_of_order_measurement', now_s, True)
            if gap > self.max_gap_s:
                self.reset()
        self._last_capture_s = ball.timestamp_s
        if (not np.isfinite([*receiver.xy_m, receiver.z_m, receiver.yaw_rad, receiver.z_std_m]).all()
                or receiver.z_std_m < 0):
            raise ValueError('receiver pose must be finite with nonnegative uncertainty')
        pose_cov = np.asarray(receiver.covariance)
        if (pose_cov.shape != (3, 3) or not np.isfinite(pose_cov).all()
                or not np.allclose(pose_cov, pose_cov.T)
                or np.linalg.eigvalsh(pose_cov).min() < -1e-10):
            raise ValueError('receiver covariance must be positive semidefinite')
        c, s = np.cos(receiver.yaw_rad), np.sin(receiver.yaw_rad)
        rotation = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
        relative = rotation @ (ball.position_m-np.r_[receiver.xy_m, receiver.z_m])
        position_covariance = np.zeros((3, 3))
        position_covariance[:2, :2] = pose_cov[:2, :2]
        position_covariance[2, 2] = receiver.z_std_m**2
        covariance = rotation @ (ball.covariance_m2+position_covariance) @ rotation.T
        yaw_jacobian = np.array([relative[1], -relative[0], 0.])
        covariance += pose_cov[2, 2]*np.outer(yaw_jacobian, yaw_jacobian)
        sigma = np.sqrt(np.maximum(0., np.diag(covariance)))
        if max(sigma) > .12:
            return self._reject('uncertain_measurement', now_s, True)
        # Heuristic half-sigma envelope, not calibrated confidence coverage.
        margin = .5*sigma
        lower = np.array([.02, -.20, .10])
        upper = np.array([.45, .20, .50])
        if np.any(relative-margin < lower) or np.any(relative+margin > upper):
            self._history.clear()
            self._stable_since = None
            return RetentionEstimate('outside_receiving_volume', ball.timestamp_s, True,
                                     relative, None, covariance)
        self._history.append((ball.timestamp_s, relative))
        while self._history and ball.timestamp_s-self._history[0][0] > .25:
            self._history.popleft()
        times = np.array([item[0] for item in self._history])
        if len(times) < 3 or np.ptp(times) < .055:
            self._stable_since = None
            return RetentionEstimate('insufficient_motion_history', ball.timestamp_s, True,
                                     relative, None, covariance)
        design = np.column_stack((np.ones(len(times)), times-times[-1]))
        velocity = np.linalg.lstsq(design, np.stack([x[1] for x in self._history]), rcond=None)[0][1]
        if np.linalg.norm(velocity) > self.max_relative_speed_mps:
            self._stable_since = None
            return RetentionEstimate('relative_motion_too_fast', ball.timestamp_s, True,
                                     relative, velocity, covariance)
        if self._stable_since is None:
            self._stable_since = ball.timestamp_s
        duration = float(ball.timestamp_s-self._stable_since)
        retained = bool(duration + 1e-9 >= self.required_duration_s)
        return RetentionEstimate('retained_visual_proxy' if retained else 'stable_visible_ball',
                                 ball.timestamp_s, True, relative, velocity, covariance,
                                 duration, retained)
