"""Camera-timed request for one bounded stationary cradle motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .localization import ReceiverPose
from .perception import Array, BallTracker, ColorBallDetector, Intercept, RGBDFrame, RigidPose


@dataclass(frozen=True)
class CatchWindow:
    """Provisional reach and timing envelope for a stationary bare-wrist cradle."""

    plane_x_m: float = 0.28
    max_lateral_m: float = 0.16
    min_height_m: float = 0.12
    max_height_m: float = 0.52
    min_lead_s: float = 0.12
    max_lead_s: float = 0.42
    min_incoming_mps: float = 0.15
    max_time_std_s: float = 0.12
    max_crossing_std_m: float = 0.16
    min_inliers: int = 4
    max_frame_age_s: float = 0.08
    max_receiver_age_s: float = 0.10
    confidence_sigma: float = 0.5

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("catch window settings must be positive and finite")
        if not 0 < self.min_lead_s < self.max_lead_s <= 1:
            raise ValueError("catch lead limits must be ordered within one second")
        if self.min_height_m >= self.max_height_m or self.min_inliers < 3:
            raise ValueError("catch height limits or minimum inliers are invalid")


@dataclass(frozen=True)
class CatchDecision:
    status: str
    capture_timestamp_s: float
    ball_visible: bool
    triggered: bool = False
    latched: bool = False
    prediction: Intercept | None = None
    time_to_crossing_s: float | None = None
    receiver_relative_crossing_m: Array | None = None
    velocity_mps: Array | None = None


class VisualCatchGate:
    """Use RGB-D and measured pelvis pose for a one-shot stationary arm request.

    The catch plane is frozen at the latest pelvis pose. Crossing uncertainty is
    provisional, and the window must be characterized for each arm trajectory.
    """

    def __init__(
        self,
        window: CatchWindow = CatchWindow(),
        *,
        frame_id: str = "field",
        allow_synthetic_fixture: bool = False,
        ball_center_offset_m: float = 0.0,
    ) -> None:
        self.window = window
        self.frame_id = frame_id
        self.allow_synthetic_fixture = allow_synthetic_fixture
        self.detector = ColorBallDetector(ball_center_offset_m=ball_center_offset_m)
        self.reset()

    def reset(self) -> None:
        self.tracker = BallTracker(
            frame_id=self.frame_id,
            max_age_s=max(self.window.max_receiver_age_s, self.window.max_frame_age_s),
            max_horizon_s=1.0,
            allow_synthetic_fixture=self.allow_synthetic_fixture,
        )
        self._last_capture_s = float("-inf")
        self._latched = False

    def update(
        self,
        frame: RGBDFrame,
        survey: RigidPose,
        receiver_pose: ReceiverPose | None,
        now_s: float,
    ) -> CatchDecision:
        stamp = frame.timestamp_s
        if self._latched:
            return CatchDecision("latched", stamp, False, latched=True)
        if survey.target_frame != self.frame_id or survey.source_frame != frame.frame_id:
            raise ValueError("survey must transform the camera frame into the catch field")
        if survey.provenance == "synthetic_fixture":
            if not self.allow_synthetic_fixture:
                raise ValueError("synthetic camera survey requires explicit fixture opt-in")
        elif survey.provenance != "fixed_calibration":
            raise ValueError("visual catch gate requires a fixed camera survey")
        if not 0 <= now_s - stamp <= self.window.max_frame_age_s:
            return CatchDecision("stale_frame", stamp, False)
        if stamp <= self._last_capture_s:
            return CatchDecision("out_of_order_frame", stamp, False)
        self._last_capture_s = stamp

        observation = self.detector.detect(frame, survey)
        if observation is None:
            return CatchDecision("ball_not_visible", stamp, False)
        self.tracker.add(observation)
        if receiver_pose is None or not receiver_pose.fresh(now_s, self.window.max_receiver_age_s):
            return CatchDecision("no_fresh_receiver_pose", stamp, True)
        pelvis = self._pelvis_transform(receiver_pose, survey)
        if pelvis is None:
            return CatchDecision("no_receiver_height", stamp, True)
        prediction = self.tracker.intercept(
            now_s, np.array([self.window.plane_x_m, 0., 0.]), np.array([1., 0., 0.]),
            pelvis.source_frame, receiver_pose=pelvis,
        )
        if prediction is None or prediction.inliers < self.window.min_inliers:
            return CatchDecision("no_stable_crossing", stamp, True)
        if prediction.last_inlier_timestamp_s != stamp:
            return CatchDecision("latest_detection_outlier", stamp, True, prediction=prediction)
        relative = pelvis.rotation.T @ (prediction.position_m - pelvis.translation)
        velocity = pelvis.rotation.T @ prediction.velocity_mps
        delay = prediction.timestamp_s - now_s
        status = self._check_crossing(pelvis, prediction, relative, velocity, delay)
        triggered = status == "triggered"
        self._latched = triggered
        return CatchDecision(status, stamp, True, triggered, triggered, prediction, delay, relative, velocity)

    def _pelvis_transform(self, pose: ReceiverPose, survey: RigidPose) -> RigidPose | None:
        if pose.frame_id != self.frame_id or pose.pose_provenance != survey.provenance:
            raise ValueError("receiver pose must match the surveyed field and provenance")
        if pose.z_m is None or pose.z_std_m is None:
            return None
        if (
            pose.covariance.shape != (3, 3)
            or not np.isfinite(pose.covariance).all()
            or min(np.diag(pose.covariance)) < 0
            or not np.isfinite(pose.z_std_m)
            or pose.z_std_m < 0
        ):
            raise ValueError("invalid receiver pose uncertainty")
        c, s = np.cos(pose.yaw_rad), np.sin(pose.yaw_rad)
        return RigidPose(
            pose.timestamp_s,
            "receiver_pelvis",
            self.frame_id,
            np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]),
            np.r_[pose.xy_m, pose.z_m],
            pose.pose_provenance,
            position_std_m=max(float(np.sqrt(max(np.diag(pose.covariance)[:2]))), pose.z_std_m),
            angle_std_rad=float(np.sqrt(pose.covariance[2, 2])),
        )

    def _check_crossing(
        self, pelvis: RigidPose, prediction: Intercept, relative: Array,
        velocity: Array, delay: float,
    ) -> str:
        w = self.window
        if velocity[0] >= -w.min_incoming_mps or velocity[2] >= 0.:
            return "not_incoming_descending"
        if delay < w.min_lead_s:
            return "too_late"
        if delay > w.max_lead_s:
            return "too_early"
        covariance = pelvis.rotation.T @ prediction.covariance_m2 @ pelvis.rotation
        transverse_std = np.sqrt(
            np.maximum(0., np.diag(covariance)[1:]) + pelvis.position_std_m**2
        )
        if prediction.time_std_s > w.max_time_std_s or max(transverse_std) > w.max_crossing_std_m:
            return "uncertain_crossing"
        time_margin = w.confidence_sigma * prediction.time_std_s
        if delay - time_margin < w.min_lead_s or delay + time_margin > w.max_lead_s:
            return "uncertain_lead"
        margin = w.confidence_sigma * transverse_std
        if (
            abs(relative[1]) + margin[0] > w.max_lateral_m
            or relative[2] - margin[1] < w.min_height_m
            or relative[2] + margin[1] > w.max_height_m
        ):
            return "outside_reach"
        return "triggered"
