"""Ball detection and interception from calibrated, capture-timestamped RGB-D."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
GRAVITY_M_S2 = 9.80665
# Capture clocks commonly accumulate binary floating-point integration error.
# This only admits numerical boundary noise, not another sensor/control tick.
TIMESTAMP_EPS_S = 1e-8
POSE_PROVENANCE = {"fixed_calibration", "live_estimate", "synthetic_fixture"}


@dataclass(frozen=True)
class CameraIntrinsics:
    """Intrinsics for a rectified image with registered axial depth in meters."""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all():
            raise ValueError("intrinsics must be finite")
        if min(self.fx, self.fy) <= 0:
            raise ValueError("focal lengths must be positive")

    def ray(self, u: float, v: float) -> Array:
        return np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])


@dataclass(frozen=True)
class RigidPose:
    """Transform from source into target, measured at timestamp_s."""

    timestamp_s: float
    source_frame: str
    target_frame: str
    rotation: Array
    translation: Array
    provenance: str
    position_std_m: float = 0.0
    angle_std_rad: float = 0.0

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation, dtype=float)
        scales = [self.timestamp_s, self.position_std_m, self.angle_std_rad]
        if not np.isfinite(scales).all() or min(scales[1:]) < 0:
            raise ValueError("invalid pose time or uncertainty")
        if self.provenance not in POSE_PROVENANCE:
            raise ValueError("pose must declare calibration, live estimate, or synthetic provenance")
        if not self.source_frame or not self.target_frame:
            raise ValueError("pose requires named frames")
        if (
            rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1)
        ):
            raise ValueError("pose rotation must be a proper 3D rotation")
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("pose translation must be a finite 3-vector")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def point(self, point_m: Array) -> Array:
        return self.rotation @ point_m + self.translation


@dataclass(frozen=True)
class RGBDFrame:
    timestamp_s: float
    frame_id: str
    rgb: NDArray[np.uint8]
    depth_m: Array
    intrinsics: CameraIntrinsics

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp_s) or not self.frame_id:
            raise ValueError("frame requires a finite time and named optical frame")
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or self.rgb.dtype != np.uint8:
            raise ValueError("RGB must have shape (height, width, 3) and uint8 channels")
        if self.depth_m.shape != self.rgb.shape[:2]:
            raise ValueError("depth must be registered to RGB")


@dataclass(frozen=True)
class BallObservation:
    """3D ball observation from the visible surface, optionally center-corrected."""

    timestamp_s: float
    frame_id: str
    position_m: Array
    covariance_m2: Array
    pose_provenance: str

    def __post_init__(self) -> None:
        if self.pose_provenance not in POSE_PROVENANCE:
            raise ValueError("observation must declare valid pose provenance")
        position = np.asarray(self.position_m, dtype=float)
        covariance = np.asarray(self.covariance_m2, dtype=float)
        if not np.isfinite(self.timestamp_s) or not self.frame_id:
            raise ValueError("observation requires a finite time and named frame")
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("observation position must be a finite 3-vector")
        if (
            covariance.shape != (3, 3)
            or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T)
            or np.linalg.eigvalsh(covariance).min() <= 0
        ):
            raise ValueError("observation covariance must be positive definite")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "covariance_m2", covariance)


def _components(mask: NDArray[np.bool_]):
    """Yield 4-connected foreground components as arrays of row/column pixels."""
    height, width = mask.shape
    foreground = set(np.flatnonzero(mask).tolist())
    while foreground:
        seed = foreground.pop()
        pending = [seed]
        pixels = []
        while pending:
            index = pending.pop()
            row, col = divmod(index, width)
            pixels.append((row, col))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                y, x = row + dy, col + dx
                if 0 <= y < height and 0 <= x < width:
                    neighbor = y * width + x
                    if neighbor in foreground:
                        foreground.remove(neighbor)
                        pending.append(neighbor)
        yield np.asarray(pixels)


@dataclass(frozen=True)
class ColorBallDetector:
    """Classical detector for a high-contrast orange/brown practice football."""

    min_pixels: int = 12
    max_pixels: int = 8_000
    min_depth_m: float = 0.2
    max_depth_m: float = 6.0
    max_pose_skew_s: float = 0.01
    ball_center_offset_m: float = 0.0

    def __post_init__(self) -> None:
        scales = [
            self.min_depth_m, self.max_depth_m, self.max_pose_skew_s, self.ball_center_offset_m,
        ]
        if (
            self.min_pixels < 1
            or self.max_pixels < self.min_pixels
            or not np.isfinite(scales).all()
            or not 0 < self.min_depth_m < self.max_depth_m
            or self.max_pose_skew_s < 0
            or self.ball_center_offset_m < 0
        ):
            raise ValueError("invalid detector settings")

    def detect(self, frame: RGBDFrame, camera_pose: RigidPose) -> BallObservation | None:
        if camera_pose.source_frame != frame.frame_id:
            raise ValueError("camera pose source does not match optical frame")
        if abs(camera_pose.timestamp_s - frame.timestamp_s) > self.max_pose_skew_s:
            return None

        red, green, blue = np.moveaxis(frame.rgb.astype(np.int16), -1, 0)
        mask = (red >= 90) & (red - green >= 22) & (green - blue >= 12)
        if np.count_nonzero(mask) > max(10 * self.max_pixels, mask.size // 3):
            return None

        candidates = []
        for pixels in _components(mask):
            if self.min_pixels <= len(pixels) <= self.max_pixels:
                candidate = self._candidate(frame, pixels, camera_pose)
                if candidate is not None:
                    candidates.append(candidate)
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        if len(candidates) > 1 and candidates[1][0] >= 0.8 * candidates[0][0]:
            return None
        return candidates[0][1]

    def _candidate(
        self, frame: RGBDFrame, pixels: NDArray[np.int_], pose: RigidPose
    ) -> tuple[float, BallObservation] | None:
        rows, cols = pixels.T
        height, width = np.ptp(pixels, axis=0) + 1
        fill = len(pixels) / (height * width)
        if min(height, width) < 3 or max(height, width) / min(height, width) > 3 or fill < 0.35:
            return None
        depths = frame.depth_m[rows, cols]
        valid = np.isfinite(depths) & (depths >= self.min_depth_m) & (depths <= self.max_depth_m)
        if np.count_nonzero(valid) < 0.6 * len(depths):
            return None
        axial_depth = float(np.median(depths[valid]))
        inliers = valid & (np.abs(depths - axial_depth) <= max(0.05, 0.035 * axial_depth))
        if np.count_nonzero(inliers) < 0.6 * len(depths):
            return None
        u, v = float(np.median(cols[inliers])), float(np.median(rows[inliers]))
        depth = float(np.median(depths[inliers]))
        ray = frame.intrinsics.ray(u, v)
        camera_point = depth * ray + self.ball_center_offset_m * ray / np.linalg.norm(ray)

        # Include a depth floor and pose uncertainty so a smooth patch is not overconfident.
        depth_mad = float(np.median(np.abs(depths[inliers] - depth)))
        sigma_z = max(0.015, 0.01 * depth, 1.4826 * depth_mad)
        sigma_pixels = max(1.0, 0.08 * min(height, width))
        jacobian = np.column_stack((
            [depth / frame.intrinsics.fx, 0.0, 0.0],
            [0.0, depth / frame.intrinsics.fy, 0.0],
            ray,
        ))
        covariance = jacobian @ np.diag([sigma_pixels**2, sigma_pixels**2, sigma_z**2]) @ jacobian.T
        pose_variance = pose.position_std_m**2 + (np.linalg.norm(camera_point) * pose.angle_std_rad)**2
        covariance = pose.rotation @ covariance @ pose.rotation.T + pose_variance * np.eye(3)
        observation = BallObservation(
            frame.timestamp_s, pose.target_frame, pose.point(camera_point), covariance, pose.provenance
        )
        return (fill * np.count_nonzero(inliers), observation)


@dataclass(frozen=True)
class Intercept:
    """Provisional covariance from the fit and plane pose, excluding model mismatch."""

    timestamp_s: float
    frame_id: str
    position_m: Array
    covariance_m2: Array
    time_std_s: float
    inliers: int
    velocity_mps: Array | None = None
    last_inlier_timestamp_s: float | None = None


class BallTracker:
    """Short-window ballistic fit in one fixed z-up frame with outlier rejection."""

    def __init__(
        self,
        frame_id: str = "field",
        *,
        history_s: float = 0.45,
        max_age_s: float = 0.10,
        max_horizon_s: float = 1.0,
        allow_synthetic_fixture: bool = False,
    ) -> None:
        limits = [history_s, max_age_s, max_horizon_s]
        if not frame_id or not np.isfinite(limits).all() or min(limits) <= 0:
            raise ValueError("tracking frame and time limits must be valid")
        self.frame_id = frame_id
        self.history_s = history_s
        self.max_age_s = max_age_s
        self.max_horizon_s = max_horizon_s
        self.allow_synthetic_fixture = allow_synthetic_fixture
        self.gravity = np.array([0.0, 0.0, -GRAVITY_M_S2])
        self._observations: list[BallObservation] = []

    def add(self, observation: BallObservation) -> bool:
        if observation.pose_provenance == "synthetic_fixture" and not self.allow_synthetic_fixture:
            raise ValueError("synthetic observation requires explicit fixture opt-in")
        if observation.frame_id != self.frame_id:
            raise ValueError("ball observation must be in the tracking frame")
        if self._observations:
            gap = observation.timestamp_s - self._observations[-1].timestamp_s
            if gap <= 0:
                return False
            if gap > self.max_age_s + TIMESTAMP_EPS_S:
                self._observations.clear()
        self._observations.append(observation)
        cutoff = observation.timestamp_s - self.history_s
        self._observations = [item for item in self._observations[-16:] if item.timestamp_s >= cutoff]
        return True

    def _fit(self, now_s: float) -> tuple[Array, Array, int, float] | None:
        if not self._observations or not np.isfinite(now_s):
            return None
        age = now_s - self._observations[-1].timestamp_s
        if not -TIMESTAMP_EPS_S <= age <= self.max_age_s + TIMESTAMP_EPS_S:
            return None
        times = np.array([item.timestamp_s - now_s for item in self._observations])
        if len(times) < 3 or np.ptp(times) < 0.055:
            return None
        corrected = (
            np.stack([item.position_m for item in self._observations])
            - 0.5 * times[:, None] ** 2 * self.gravity
        )
        covariances = np.stack([item.covariance_m2 for item in self._observations])
        weights = np.linalg.inv(covariances)
        best = None
        best_score = (0, -np.inf)
        for i, j in combinations(range(len(times)), 2):
            if times[j] - times[i] < 0.04:
                continue
            velocity = (corrected[j] - corrected[i]) / (times[j] - times[i])
            residuals = corrected - (corrected[i] + (times - times[i])[:, None] * velocity)
            distance_squared = np.einsum("ni,nij,nj->n", residuals, weights, residuals)
            inliers = distance_squared <= 16
            score = (np.count_nonzero(inliers), -float(distance_squared[inliers].sum()))
            if score > best_score:
                best, best_score = inliers, score
        if best is None or best_score[0] < max(3, (len(times) + 1) // 2):
            return None
        if -times[best][-1] > self.max_age_s + TIMESTAMP_EPS_S or np.ptp(times[best]) < 0.055:
            return None

        identity = np.eye(3)
        design = np.stack([np.hstack((identity, t * identity)) for t in times[best]])
        precision = np.einsum("nij,nik,nkl->jl", design, weights[best], design)
        rhs = np.einsum("nij,nik,nk->j", design, weights[best], corrected[best])
        covariance = np.linalg.inv(precision)
        last_inlier = self._observations[np.flatnonzero(best)[-1]].timestamp_s
        return covariance @ rhs, covariance, int(best_score[0]), last_inlier

    def intercept(
        self,
        now_s: float,
        plane_point_m: Array,
        plane_normal: Array,
        frame_id: str,
        *,
        receiver_pose: RigidPose | None = None,
    ) -> Intercept | None:
        """Predict the crossing of a plane fixed in the field or receiver pose at now_s."""
        point = np.asarray(plane_point_m, dtype=float)
        normal = np.asarray(plane_normal, dtype=float)
        if point.shape != (3,) or normal.shape != (3,) or not np.isfinite([point, normal]).all():
            raise ValueError("catch plane requires finite 3-vectors")
        magnitude = np.linalg.norm(normal)
        if magnitude <= 1e-12:
            raise ValueError("catch plane normal must be nonzero")
        normal = normal / magnitude
        if receiver_pose is None:
            if frame_id != self.frame_id:
                raise ValueError("catch plane must be in the tracking frame")
        else:
            if receiver_pose.provenance == "synthetic_fixture" and not self.allow_synthetic_fixture:
                raise ValueError("synthetic receiver pose requires explicit fixture opt-in")
            if receiver_pose.source_frame != frame_id or receiver_pose.target_frame != self.frame_id:
                raise ValueError("receiver pose must transform the catch plane into the tracking frame")
            if not 0 <= now_s - receiver_pose.timestamp_s <= self.max_age_s:
                return None
            point = receiver_pose.point(point)
            normal = receiver_pose.rotation @ normal
        fit = self._fit(now_s)
        if fit is None:
            return None
        state, covariance, inliers, last_inlier = fit
        position, velocity = state[:3], state[3:]
        coefficients = [0.5 * normal @ self.gravity, normal @ velocity, normal @ (position - point)]
        roots = np.roots(np.trim_zeros(np.array(coefficients), "f"))
        future_roots = [
            float(root.real) for root in roots
            if abs(root.imag) < 1e-9 and 0 < root.real <= self.max_horizon_s
        ]
        if not future_roots:
            return None
        delay = min(future_roots)
        crossing_velocity = velocity + delay * self.gravity
        normal_speed = normal @ crossing_velocity
        if abs(normal_speed) < 1e-3:
            return None
        intercept_position = position + delay * velocity + 0.5 * delay**2 * self.gravity
        position_jacobian = np.hstack((np.eye(3), delay * np.eye(3)))
        time_jacobian = -(normal @ position_jacobian) / normal_speed
        intercept_jacobian = position_jacobian + np.outer(crossing_velocity, time_jacobian)
        plane_time_variance = 0.0
        if receiver_pose is not None:
            lever_arm = np.linalg.norm(np.cross(intercept_position - receiver_pose.translation, normal))
            plane_time_variance = (
                receiver_pose.position_std_m**2 + (lever_arm * receiver_pose.angle_std_rad)**2
            ) / normal_speed**2
        intercept_covariance = (
            intercept_jacobian @ covariance @ intercept_jacobian.T
            + plane_time_variance * np.outer(crossing_velocity, crossing_velocity)
        )
        return Intercept(
            now_s + delay,
            self.frame_id,
            intercept_position,
            intercept_covariance,
            float(np.sqrt(time_jacobian @ covariance @ time_jacobian + plane_time_variance)),
            inliers,
            crossing_velocity,
            last_inlier,
        )


def read_capture(
    capture: Path, *, allow_synthetic_fixture: bool = False
) -> Iterator[tuple[RGBDFrame, RigidPose]]:
    """Yield calibrated RGB-D samples in capture order from a replay archive."""
    camera_frame = "camera_optical"
    with np.load(capture, allow_pickle=False) as data:
        times = data["timestamps_s"]
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("capture timestamps must be finite and strictly increasing")
        provenance = str(data["pose_provenance"].item())
        if provenance not in POSE_PROVENANCE:
            raise ValueError("capture must declare valid camera pose provenance")
        if provenance == "synthetic_fixture" and not allow_synthetic_fixture:
            raise ValueError("synthetic capture requires --allow-synthetic-fixture")
        pose_times = data["camera_pose_timestamps_s"]
        if pose_times.shape != times.shape or not np.isfinite(pose_times).all():
            raise ValueError("camera pose timestamps must be finite and match the capture length")
        intrinsics = CameraIntrinsics(*data["intrinsics"])
        fields = ["rgb", "depth_m", "camera_rotation", "camera_translation_m"]
        arrays = [data[field] for field in fields]
        position_std = data.get("camera_position_std_m", np.zeros(len(times)))
        angle_std = data.get("camera_angle_std_rad", np.zeros(len(times)))
        streams = [*arrays, position_std, angle_std]
        if any(stream.ndim == 0 or len(stream) != len(times) for stream in streams):
            raise ValueError("image and pose arrays must match the capture length")
        frames = zip(times, pose_times, *streams)
        for timestamp, pose_time, rgb, depth, rotation, translation, pos_std, ang_std in frames:
            frame = RGBDFrame(timestamp, camera_frame, rgb, depth, intrinsics)
            pose = RigidPose(
                pose_time, camera_frame, "field", rotation, translation,
                provenance, pos_std, ang_std,
            )
            yield frame, pose


def replay(
    capture: Path,
    plane_point_m: Array,
    plane_normal: Array,
    *,
    allow_synthetic_fixture: bool = False,
) -> None:
    """Replay camera observations and emit one JSON record per predicted crossing."""
    detector = ColorBallDetector()
    tracker = BallTracker(allow_synthetic_fixture=allow_synthetic_fixture)
    for frame, pose in read_capture(capture, allow_synthetic_fixture=allow_synthetic_fixture):
        timestamp = frame.timestamp_s
        observation = detector.detect(frame, pose)
        if observation is not None:
            tracker.add(observation)
        prediction = tracker.intercept(timestamp, plane_point_m, plane_normal, tracker.frame_id)
        if prediction is not None:
            print(json.dumps({
                "frame_timestamp_s": float(timestamp),
                "crossing_timestamp_s": float(prediction.timestamp_s),
                "frame_id": prediction.frame_id,
                "pose_provenance": pose.provenance,
                "position_basis": "visible_surface_proxy",
                "uncertainty_basis": "provisional_fit_covariance",
                "position_m": prediction.position_m.tolist(),
                "covariance_m2": prediction.covariance_m2.tolist(),
                "time_std_s": prediction.time_std_s,
                "inliers": prediction.inliers,
            }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="timestamped RGB-D NPZ archive")
    parser.add_argument("--plane-point", type=float, nargs=3, default=(2.0, 0.0, 0.0))
    parser.add_argument("--plane-normal", type=float, nargs=3, default=(1.0, 0.0, 0.0))
    parser.add_argument("--allow-synthetic-fixture", action="store_true")
    args = parser.parse_args()
    try:
        replay(
            args.capture, np.asarray(args.plane_point), np.asarray(args.plane_normal),
            allow_synthetic_fixture=args.allow_synthetic_fixture,
        )
    except (KeyError, OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
