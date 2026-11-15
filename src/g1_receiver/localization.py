"""Field localization from two colored spheres rigidly mounted on the pelvis."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .perception import Array, RGBDFrame, RigidPose, _components, read_capture

LEFT_MARKER_M = (0.15, 0.11, 0.08)
RIGHT_MARKER_M = (0.15, -0.11, 0.08)
MARKER_RADIUS_M = 0.035


@dataclass(frozen=True)
class ReceiverPose:
    """Field position and yaw with provisional uncertainty at image capture time."""

    timestamp_s: float
    frame_id: str
    xy_m: Array
    yaw_rad: float
    covariance: Array
    pose_provenance: str
    z_m: float | None = None
    z_std_m: float | None = None

    def fresh(self, now_s: float, max_age_s: float) -> bool:
        return 0 <= now_s - self.timestamp_s <= max_age_s


class MarkerLocalizer:
    """Estimate a nearly level pelvis pose from a fixed calibrated RGB-D camera."""

    def __init__(
        self,
        *,
        allow_synthetic_fixture: bool = False,
        max_pose_skew_s: float = 0.010,
        max_range_m: float = 6.0,
    ) -> None:
        limits = [max_pose_skew_s, max_range_m]
        if not np.isfinite(limits).all() or min(limits) <= 0:
            raise ValueError("localizer time and depth limits must be positive")
        self.allow_synthetic_fixture = allow_synthetic_fixture
        self.max_pose_skew_s = max_pose_skew_s
        self.max_range_m = max_range_m

    def estimate(self, frame: RGBDFrame, camera_pose: RigidPose) -> ReceiverPose | None:
        if camera_pose.source_frame != frame.frame_id:
            raise ValueError("camera survey source must match the RGB-D optical frame")
        if camera_pose.provenance == "synthetic_fixture":
            if not self.allow_synthetic_fixture:
                raise ValueError("synthetic calibration requires explicit fixture opt-in")
        elif camera_pose.provenance != "fixed_calibration":
            raise ValueError("marker localizer requires a fixed camera survey")
        if abs(frame.timestamp_s - camera_pose.timestamp_s) > self.max_pose_skew_s:
            return None

        red, green, blue = np.moveaxis(frame.rgb.astype(np.int16), -1, 0)
        cyan = (green >= 70) & (blue >= 70) & (green - red >= 30) & (blue - red >= 30)
        magenta = (red >= 70) & (blue >= 70) & (red - green >= 30) & (blue - green >= 30)
        left = self._center(frame, cyan)
        right = self._center(frame, magenta)
        if left is None or right is None:
            return None

        left_field, right_field = camera_pose.point(left), camera_pose.point(right)
        baseline = left_field - right_field
        span = LEFT_MARKER_M[1] - RIGHT_MARKER_M[1]
        horizontal_span = np.linalg.norm(baseline[:2])
        if abs(np.linalg.norm(baseline) - span) > 0.045 or horizontal_span < 0.85 * span:
            return None

        yaw = float(np.arctan2(-baseline[0], baseline[1]))
        midpoint = (left_field + right_field) / 2
        offset = LEFT_MARKER_M[0]
        xy = midpoint[:2] - offset * np.array([np.cos(yaw), np.sin(yaw)])

        # Approximate uncertainty combines point resolution, survey error, and
        # midpoint displacement when the pelvis departs from the level assumption.
        distance = (np.linalg.norm(left) + np.linalg.norm(right)) / 2
        point_std = max(0.01, 0.0075 * distance)
        xy_std = np.sqrt(
            point_std**2 / 2
            + camera_pose.position_std_m**2
            + (distance * camera_pose.angle_std_rad)**2
            + 0.025**2
        )
        yaw_std = np.hypot(np.sqrt(2) * point_std / horizontal_span, camera_pose.angle_std_rad)
        pose_jacobian = np.array([offset * np.sin(yaw), -offset * np.cos(yaw), 1.0])
        covariance = np.diag([xy_std**2, xy_std**2, 0.0])
        covariance += yaw_std**2 * np.outer(pose_jacobian, pose_jacobian)
        return ReceiverPose(
            frame.timestamp_s, camera_pose.target_frame, xy, yaw, covariance, camera_pose.provenance,
            z_m=float(midpoint[2] - LEFT_MARKER_M[2]), z_std_m=float(xy_std),
        )

    def _center(self, frame: RGBDFrame, mask: np.ndarray) -> Array | None:
        if np.count_nonzero(mask) > mask.size // 10:
            return None
        centers = []
        image_height, image_width = mask.shape
        for pixels in _components(mask):
            if len(pixels) < 5:
                continue
            rows, cols = pixels.T
            if (
                rows.min() == 0 or cols.min() == 0
                or rows.max() == image_height - 1 or cols.max() == image_width - 1
            ):
                continue
            heights, widths = np.ptp(pixels, axis=0) + 1
            if min(heights, widths) < 2 or max(heights, widths) > 2.5 * min(heights, widths):
                continue
            depth = frame.depth_m[rows, cols]
            valid = np.isfinite(depth) & (depth > 0.20) & (depth <= self.max_range_m)
            if np.count_nonzero(valid) < 0.6 * len(depth):
                continue
            median_depth = np.median(depth[valid])
            expected_diameter = 2 * MARKER_RADIUS_M * np.array(
                [frame.intrinsics.fy, frame.intrinsics.fx]
            ) / median_depth
            size_ratio = np.array([heights, widths]) / expected_diameter
            if np.any(size_ratio < 0.4) or np.any(size_ratio > 1.9):
                continue
            valid &= np.abs(depth - median_depth) < 1.5 * MARKER_RADIUS_M
            if np.count_nonzero(valid) < 0.6 * len(depth):
                continue
            center_ray = frame.intrinsics.ray(float(np.median(cols)), float(np.median(rows)))
            center_ray /= np.linalg.norm(center_ray)

            samples = np.column_stack((
                (cols[valid] - frame.intrinsics.cx) / frame.intrinsics.fx,
                (rows[valid] - frame.intrinsics.cy) / frame.intrinsics.fy,
                np.ones(np.count_nonzero(valid)),
            )) * depth[valid, None]
            ray_distance = samples @ center_ray
            perpendicular_sq = np.sum((samples - ray_distance[:, None] * center_ray) ** 2, axis=1)
            on_marker = perpendicular_sq <= (1.25 * MARKER_RADIUS_M) ** 2
            if np.count_nonzero(on_marker) < 0.6 * len(depth):
                continue
            # Visible samples are on the near hemisphere. Recover the center
            # along the silhouette's central ray using the known sphere radius.
            offsets = np.sqrt(np.maximum(0, MARKER_RADIUS_M**2 - perpendicular_sq[on_marker]))
            center_distance = np.median(ray_distance[on_marker] + offsets)
            centers.append(center_distance * center_ray)

        if len(centers) != 1:
            return None
        return centers[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="timestamped RGB-D NPZ archive")
    parser.add_argument("--allow-synthetic-fixture", action="store_true")
    args = parser.parse_args()
    localizer = MarkerLocalizer(allow_synthetic_fixture=args.allow_synthetic_fixture)
    try:
        for frame, survey in read_capture(
            args.capture, allow_synthetic_fixture=args.allow_synthetic_fixture
        ):
            pose = localizer.estimate(frame, survey)
            record = {
                "frame_timestamp_s": float(frame.timestamp_s),
                "frame_id": survey.target_frame,
                "pose_provenance": survey.provenance,
                "status": "missing_pair" if pose is None else "localized",
                "uncertainty_basis": "provisional_marker_and_survey_covariance",
            }
            if pose is not None:
                record.update(
                    xy_m=pose.xy_m.tolist(),
                    yaw_rad=pose.yaw_rad,
                    z_m=pose.z_m,
                    z_std_m=pose.z_std_m,
                    covariance=pose.covariance.tolist(),
                )
            print(json.dumps(record))
    except (KeyError, OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
