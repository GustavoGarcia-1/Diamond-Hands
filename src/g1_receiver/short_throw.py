"""Ballistic fixture deliveries and a causal RGB-D grip timing prediction."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from .localization import ReceiverPose
from .perception import BallObservation, BallTracker, GRAVITY_M_S2, RigidPose


def _vector(values, length: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite {length}-vector")
    return vector


def _yaw_rotation(yaw: float) -> np.ndarray:
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])


def _receiver_state(pose: ReceiverPose, velocity_xy_mps, now_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.r_[_vector(velocity_xy_mps, 2, "measured receiver velocity"), 0.]
    if pose.z_m is None or not np.isfinite([pose.z_m, pose.yaw_rad, pose.timestamp_s, now_s]).all():
        raise ValueError("throw planning requires a finite receiver pose with height")
    origin = np.r_[_vector(pose.xy_m, 2, "measured receiver position"), pose.z_m]
    return origin + (now_s - pose.timestamp_s) * velocity, velocity, _yaw_rotation(pose.yaw_rad)


@dataclass(frozen=True)
class UnderhandThrow:
    release_time_s: float
    flight_time_s: float
    release_xyz_m: tuple[float, float, float]
    release_velocity_mps: tuple[float, float, float]
    release_quaternion_wxyz: tuple[float, float, float, float]
    intercept_xyz_m: tuple[float, float, float]
    intercept_velocity_mps: tuple[float, float, float]
    toward_receiver_travel_m: float


def plan_underhand_throw(
    receiver_pose: ReceiverPose,
    receiver_velocity_xy_mps: Sequence[float],
    *,
    release_time_s: float,
    flight_time_s: float,
    intercept_body_xyz_m: Sequence[float],
    incoming_relative_body_mps: Sequence[float],
    gravity_mps2: float = GRAVITY_M_S2,
    min_release_height_m: float = .12,
    min_toward_receiver_travel_m: float = 0.,
    max_pose_age_s: float = .10,
    allow_downfield_lob: bool = False,
) -> UnderhandThrow:
    """Back-solve a rising launch to a descending contact-time target.

    The target translates with the measured receiver XY velocity and keeps its
    measured yaw and height. Body X points forward. A negative relative X speed
    approaches the receiver. Choose a faster incoming X or a longer flight if
    the receiver's forward motion makes the field-frame throw travel too short.
    """
    settings = (release_time_s, flight_time_s, gravity_mps2, min_release_height_m,
                min_toward_receiver_travel_m, max_pose_age_s)
    if (not np.isfinite(settings).all() or min(flight_time_s, gravity_mps2, max_pose_age_s) <= 0
            or min(min_release_height_m, min_toward_receiver_travel_m) < 0):
        raise ValueError("invalid ballistic flight, gravity, clearance, or pose-age settings")
    if not receiver_pose.fresh(release_time_s, max_pose_age_s):
        raise ValueError("throw requires a fresh, causal receiver pose")
    receiver, receiver_velocity, rotation = _receiver_state(receiver_pose, receiver_velocity_xy_mps, release_time_s)
    offset = _vector(intercept_body_xyz_m, 3, "intercept offset")
    incoming = _vector(incoming_relative_body_mps, 3, "incoming relative velocity")
    if offset[0] <= 0 or incoming[0] >= 0 or incoming[2] >= 0:
        raise ValueError("the target must be ahead and the ball incoming and descending")
    intercept = receiver + receiver_velocity * flight_time_s + rotation @ offset
    intercept_velocity = receiver_velocity + rotation @ incoming
    gravity = np.array([0., 0., -gravity_mps2])
    release_velocity = intercept_velocity - gravity * flight_time_s
    release = intercept - intercept_velocity * flight_time_s + .5 * gravity * flight_time_s**2
    toward_travel = float((release - intercept) @ rotation[:, 0])
    if release_velocity[2] <= 0 or not min_release_height_m <= release[2] < intercept[2]:
        raise ValueError("requested flight does not launch upward from below the target with floor clearance")
    if (not allow_downfield_lob and toward_travel <= 1e-9
            or abs(toward_travel) < min_toward_receiver_travel_m):
        raise ValueError("requested flight does not travel far enough toward the receiver in the field")
    yaw = .5 * receiver_pose.yaw_rad
    return UnderhandThrow(
        release_time_s, flight_time_s, tuple(release), tuple(release_velocity),
        (float(np.cos(yaw)), 0., 0., float(np.sin(yaw))), tuple(intercept), tuple(intercept_velocity), toward_travel,
    )


class FreeBallRelease:
    """Write one free ball's initial state once. Construct a new release per trial."""

    def __init__(self, model, data, *, joint_name: str = "football_free"):
        import mujoco

        joint = model.joint(joint_name)
        if int(joint.type[0]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError("delivery requires a free ball joint")
        self._model = model
        self._data = data
        self._position = int(joint.qposadr[0])
        self._velocity = int(joint.dofadr[0])
        self._used = False

    def apply(self, throw: UnderhandThrow) -> None:
        import mujoco

        if self._used:
            raise RuntimeError("the ball was already released for this trial")
        if not np.isclose(self._data.time, throw.release_time_s, rtol=0., atol=1e-7):
            raise ValueError("release must be applied at the planned simulation time")
        position = _vector(throw.release_xyz_m, 3, "ball launch position")
        velocity = _vector(throw.release_velocity_mps, 3, "ball launch velocity")
        orientation = _vector(throw.release_quaternion_wxyz, 4, "ball launch orientation")
        if not np.isclose(np.linalg.norm(orientation), 1.):
            raise ValueError("ball launch orientation must be a unit quaternion")
        self._used = True
        self._data.qpos[self._position:self._position + 7] = (*position, *orientation)
        self._data.qvel[self._velocity:self._velocity + 6] = (*velocity, 0., 0., 0.)
        mujoco.mj_forward(self._model, self._data)


@dataclass(frozen=True)
class GripTimingWindow:
    plane_forward_m: float = .44
    min_lead_s: float = .04
    max_lead_s: float = .16
    max_age_s: float = .08
    min_incoming_mps: float = .08
    max_lateral_m: float = .13
    min_height_m: float = .14
    max_height_m: float = .53
    max_time_std_s: float = .20
    confidence_sigma: float = .50
    crossing_direction: int = -1

    def __post_init__(self) -> None:
        values = tuple(value for name, value in vars(self).items() if name != "crossing_direction")
        if (not np.isfinite(values).all() or min(values) <= 0 or self.min_lead_s >= self.max_lead_s
                or self.min_height_m >= self.max_height_m or self.crossing_direction not in (-1, 1)):
            raise ValueError("invalid grip timing window")


@dataclass(frozen=True)
class GripTimingPrediction:
    status: str
    seconds_to_crossing: float | None = None
    crossing_relative_xyz_m: tuple[float, float, float] | None = None
    crossing_relative_velocity_mps: tuple[float, float, float] | None = None
    crossing_time_std_s: float | None = None

    @property
    def close(self) -> bool:
        return self.status == "close"


def predict_grip_timing(
    observations: Iterable[BallObservation],
    receiver_pose: ReceiverPose | None,
    receiver_velocity_xy_mps: Sequence[float],
    now_s: float,
    *,
    window: GripTimingWindow = GripTimingWindow(),
    allow_synthetic_fixture: bool = False,
) -> GripTimingPrediction:
    """Fit RGB-D samples in a translating frame and predict a forward-plane crossing.

    The caller owns the recent sample history and the closure latch. Receiver
    velocity is a causal camera estimate held constant over the fit horizon.
    """
    if receiver_pose is None or receiver_pose.z_m is None or not receiver_pose.fresh(now_s, window.max_age_s):
        return GripTimingPrediction("no_fresh_receiver_pose")
    receiver, receiver_velocity, rotation = _receiver_state(receiver_pose, receiver_velocity_xy_mps, now_s)
    covariance = np.asarray(receiver_pose.covariance, dtype=float)
    if (covariance.shape != (3, 3) or not np.isfinite(covariance).all()
            or np.any(np.diag(covariance) < 0)):
        raise ValueError("invalid camera-estimated receiver covariance")
    height_std = receiver_pose.z_std_m if receiver_pose.z_std_m is not None else 0.
    if not np.isfinite(height_std) or height_std < 0:
        raise ValueError("invalid camera-estimated receiver height uncertainty")
    tracker = BallTracker(
        frame_id=receiver_pose.frame_id, max_age_s=window.max_age_s,
        max_horizon_s=max(.5, window.max_lead_s), allow_synthetic_fixture=allow_synthetic_fixture,
    )
    accepted_latest = None
    for observation in deque(observations, maxlen=16):
        if observation.timestamp_s > now_s:
            return GripTimingPrediction("future_ball_observation")
        if observation.pose_provenance != receiver_pose.pose_provenance:
            raise ValueError("ball and receiver observations must share survey provenance")
        # Use field axes, with a horizontal origin translating at receiver speed.
        shift = receiver_velocity * (observation.timestamp_s - now_s)
        if tracker.add(replace(observation, position_m=observation.position_m - shift)):
            accepted_latest = observation.timestamp_s
    pose = RigidPose(
        now_s, "receiver", receiver_pose.frame_id, rotation, receiver, receiver_pose.pose_provenance,
        position_std_m=float(np.sqrt(max(covariance[0, 0], covariance[1, 1]))),
        angle_std_rad=float(np.sqrt(covariance[2, 2])),
    )
    intercept = tracker.intercept(now_s, np.array([window.plane_forward_m, 0., 0.]),
                                  np.array([1., 0., 0.]), "receiver", receiver_pose=pose)
    if intercept is None or intercept.last_inlier_timestamp_s != accepted_latest:
        return GripTimingPrediction("no_stable_crossing")
    relative = rotation.T @ (intercept.position_m - receiver)
    velocity = rotation.T @ intercept.velocity_mps
    lead = intercept.timestamp_s - now_s
    relative_cov = rotation.T @ intercept.covariance_m2 @ rotation
    lateral_axis = rotation[:2, 1]
    receiver_lateral_variance = lateral_axis @ covariance[:2, :2] @ lateral_axis
    transverse_variance = np.diag(relative_cov)[1:] + [receiver_lateral_variance, height_std**2]
    margin_y, margin_z = window.confidence_sigma * np.sqrt(np.maximum(0., transverse_variance))
    if window.crossing_direction * velocity[0] < window.min_incoming_mps or velocity[2] >= 0:
        status = "not_incoming_descending"
    elif intercept.time_std_s > window.max_time_std_s:
        status = "uncertain_crossing"
    elif lead > window.max_lead_s:
        status = "too_early"
    elif lead < window.min_lead_s:
        status = "too_late"
    elif (abs(relative[1]) + margin_y > window.max_lateral_m
          or relative[2] - margin_z < window.min_height_m or relative[2] + margin_z > window.max_height_m):
        status = "outside_grip_window"
    else:
        status = "close"
    return GripTimingPrediction(status, lead, tuple(relative), tuple(velocity), intercept.time_std_s)
