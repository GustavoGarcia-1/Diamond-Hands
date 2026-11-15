"""Causal ball acquisition and hand closure using surveyed RGB-D."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .localization import ReceiverPose
from .perception import BallTracker, ColorBallDetector, RGBDFrame, RigidPose
from .short_throw import GripTimingWindow, predict_grip_timing


@dataclass(frozen=True)
class PassGrip:
    plane_m: float = .47
    lead_s: float = .10
    height_m: float = .33
    mode: str = "predict"
    lateral_limit_m: float = .13
    predict_beyond_last_frame: bool = False
    crossing_direction: int = -1

    def __post_init__(self):
        if not np.isfinite((self.plane_m, self.lead_s, self.height_m, self.lateral_limit_m)).all():
            raise ValueError("grip geometry and anticipation must be finite")
        if not .20 <= self.plane_m <= .75 or not .02 <= self.lead_s <= .24 or not .15 <= self.height_m <= .75:
            raise ValueError("grip is outside the supported reach or anticipation window")
        if self.mode not in ("predict", "descending") or self.crossing_direction not in (-1, 1):
            raise ValueError("unknown grip timing mode")
        if not .05 <= self.lateral_limit_m <= .25:
            raise ValueError("lateral grip limit is outside the available reach")


def observed_velocity(poses):
    """Fit velocity to recent, distinct camera measurements."""
    rows = np.asarray(poses, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 3 or len(rows) < 3:
        return None
    times = rows[:, 0] - rows[:, 0].mean()
    if np.ptp(times) < .075:
        return None
    return times @ rows[:, 1:] / (times @ times)


class VisualPassGrip:
    def __init__(self, grip, *, allow_synthetic_fixture=False, ball_center_offset_m=.075):
        self.grip = grip
        self.allow_synthetic_fixture = allow_synthetic_fixture
        self.detector = ColorBallDetector(ball_center_offset_m=ball_center_offset_m)
        self.window = GripTimingWindow(
            plane_forward_m=grip.plane_m, min_lead_s=.02, max_lead_s=grip.lead_s,
            min_height_m=max(.08, grip.height_m - .15), max_height_m=grip.height_m + .14,
            max_lateral_m=grip.lateral_limit_m,
            crossing_direction=grip.crossing_direction,
        )
        self.history = deque(maxlen=8)
        self.closed_at = None
        self.ready_at = None
        self.crossing_yz = None
        self.measured_hand_offset = None
        self.last_capture_s = -np.inf
        self.trace = []

    def update(self, frame: RGBDFrame, survey: RigidPose, pose: ReceiverPose | None,
               estimated_velocity, now_s: float):
        if self.closed_at is not None:
            return True
        if survey.provenance == "synthetic_fixture" and not self.allow_synthetic_fixture:
            raise ValueError("synthetic captures require explicit fixture opt-in")
        if not 0 <= now_s - frame.timestamp_s <= .08 or frame.timestamp_s <= self.last_capture_s:
            return False
        self.last_capture_s = frame.timestamp_s
        obs = self.detector.detect(frame, survey)
        if pose is None or pose.z_m is None or not pose.fresh(now_s, .10):
            return False
        if obs is None:
            if (self.grip.mode == "predict" and self.grip.predict_beyond_last_frame
                    and estimated_velocity is not None and self.history):
                return self._predict_with_occlusion(pose, estimated_velocity, now_s)
            return False
        if pose.frame_id != obs.frame_id or pose.pose_provenance != obs.pose_provenance:
            raise ValueError("receiver and ball observations must share the calibrated field")
        old = self.history[-1] if self.history else None
        if old is not None and obs.timestamp_s - old.timestamp_s > .12:
            self.history.clear()
            old = None
        self.history.append(obs)
        heading = np.array([np.cos(pose.yaw_rad), np.sin(pose.yaw_rad)])
        offset = obs.position_m[:2] - pose.xy_m
        forward = float(offset @ heading)
        lateral = float(offset @ np.array([-heading[1], heading[0]]))
        height = float(obs.position_m[2] - pose.z_m)
        close, status, prediction = False, "waiting", None
        if self.grip.mode == "predict" and estimated_velocity is not None:
            pred = predict_grip_timing(
                self.history, pose, estimated_velocity, now_s, window=self.window,
                allow_synthetic_fixture=self.allow_synthetic_fixture,
            )
            close, status, prediction = pred.close, pred.status, pred.seconds_to_crossing
            if (prediction is not None and .025 <= prediction <= .30
                    and pred.crossing_relative_xyz_m is not None
                    and pred.crossing_time_std_s is not None and pred.crossing_time_std_s <= .12):
                y, z = pred.crossing_relative_xyz_m[1:]
                if abs(y) <= .20 and .12 <= z <= .60:
                    self.crossing_yz = np.array([y, z])
            if (self.ready_at is None and prediction is not None and .02 <= prediction <= .30
                    and pred.crossing_relative_xyz_m is not None
                    and abs(pred.crossing_relative_xyz_m[1]) <= .20
                    and .12 <= pred.crossing_relative_xyz_m[2] <= .60):
                self.ready_at = now_s
        in_reach = (abs(forward - self.grip.plane_m) <= .12 and abs(lateral) <= self.grip.lateral_limit_m
                    and self.grip.height_m - .13 <= height <= self.grip.height_m)
        if (not close and in_reach and old is not None and obs.position_m[2] < old.position_m[2] - .001
                and self.grip.crossing_direction == -1):
            close, status = True, "descending_in_reach"
        self.trace.append((now_s, *obs.position_m.tolist(), forward, lateral, height, status, prediction))
        if close:
            self.closed_at = now_s
            if self.ready_at is None:
                self.ready_at = now_s
        return close

    def _predict_with_occlusion(self, pose, velocity, now_s):
        prediction = predict_grip_timing(
            self.history, pose, velocity, now_s, window=self.window,
            allow_synthetic_fixture=self.allow_synthetic_fixture,
        )
        if not prediction.close:
            return False
        self.closed_at = now_s
        if self.ready_at is None:
            self.ready_at = now_s
        self.crossing_yz = np.array(prediction.crossing_relative_xyz_m[1:])
        return True

    def observe_after_close(self, frame, survey, pose, now_s, *, max_delta_s=.55):
        """Track a partially slipping ball through the nominal receiving window."""
        if (self.closed_at is None or now_s > self.closed_at + max_delta_s
                or pose is None or pose.z_m is None or not pose.fresh(now_s, .10)):
            return
        obs = self.detector.detect(frame, survey)
        if obs is None or not 0 <= now_s - obs.timestamp_s <= .08:
            return
        if obs.frame_id != pose.frame_id or obs.pose_provenance != pose.pose_provenance:
            return
        heading = np.array([np.cos(pose.yaw_rad), np.sin(pose.yaw_rad)])
        xy = obs.position_m[:2] - pose.xy_m
        x, y = xy @ heading, xy @ np.array([-heading[1], heading[0]])
        z = obs.position_m[2] - pose.z_m
        if self.grip.plane_m - .23 <= x <= self.grip.plane_m + .14 and abs(y) <= .18 and .10 <= z <= .55:
            raw = np.array([x - self.grip.plane_m, y, z - self.grip.height_m])
            if self.measured_hand_offset is None:
                self.measured_hand_offset = raw
            else:
                self.measured_hand_offset += .35 * (raw - self.measured_hand_offset)


class FieldPassGrip(VisualPassGrip):
    """Latch a catch from RGB-D flight through a surveyed route catch window."""

    def __init__(self, grip, target, *, allow_synthetic_fixture=False,
                 approach_axis_xy=(1., 0.), minimum_approach_speed_mps=2.5):
        super().__init__(grip, allow_synthetic_fixture=allow_synthetic_fixture)
        self.target = np.asarray(target, dtype=float)
        if self.target.shape != (3,) or not np.isfinite(self.target).all():
            raise ValueError("field catch window requires a surveyed XYZ target")
        axis = np.asarray(approach_axis_xy, dtype=float)
        if (axis.shape != (2,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8
                or not np.isfinite(minimum_approach_speed_mps) or minimum_approach_speed_mps <= 0):
            raise ValueError("catch approach needs a finite nonzero surveyed axis and positive speed gate")
        self.approach_axis = axis / np.linalg.norm(axis)
        self.lateral_axis = np.array([-self.approach_axis[1], self.approach_axis[0]])
        self.minimum_approach_speed_mps = minimum_approach_speed_mps
        self.field_tracker = BallTracker(max_age_s=.08, max_horizon_s=.55,
                                        allow_synthetic_fixture=allow_synthetic_fixture)

    def update(self, frame, survey, pose, estimated_velocity, now_s):
        if self.closed_at is not None:
            return True
        if not 0 <= now_s - frame.timestamp_s <= .08 or frame.timestamp_s <= self.last_capture_s:
            return False
        self.last_capture_s = frame.timestamp_s
        obs = self.detector.detect(frame, survey)
        if obs is None or not self.field_tracker.add(obs):
            return False
        crossing = self.field_tracker.intercept(now_s, self.target, (*self.approach_axis, 0.), obs.frame_id)
        if crossing is None or crossing.last_inlier_timestamp_s != obs.timestamp_s:
            self.trace.append((now_s, *obs.position_m, 0., 0., 0., "no_stable_crossing", None))
            return False
        lead = crossing.timestamp_s - now_s
        offset = crossing.position_m - self.target
        lateral, height = float(offset[:2] @ self.lateral_axis), float(offset[2])
        if crossing.velocity_mps[:2] @ self.approach_axis < self.minimum_approach_speed_mps or crossing.velocity_mps[2] >= 0.:
            status = "not_a_descending_downfield_throw"
        elif crossing.time_std_s > .06 or abs(lateral) > self.grip.lateral_limit_m or abs(height) > .30:
            status = "outside_field_catch_window"
        else:
            self.ready_at = now_s if self.ready_at is None else self.ready_at
            if lead > self.grip.lead_s:
                status = "too_early"
            elif lead < .02:
                status = "too_late"
            else:
                status = "close"
                self.closed_at = now_s
        self.trace.append((now_s, *obs.position_m, float((obs.position_m[:2] - self.target[:2]) @ self.approach_axis),
                           float(lateral), float(height), status, lead))
        return self.closed_at is not None
