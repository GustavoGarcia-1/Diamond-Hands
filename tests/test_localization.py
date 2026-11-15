"""Geometry tests using synthetic RGB-D samples of a surveyed marker rig."""

import numpy as np
import pytest

from g1_receiver.localization import (
    LEFT_MARKER_M,
    MARKER_RADIUS_M,
    RIGHT_MARKER_M,
    MarkerLocalizer,
)
from g1_receiver.perception import CameraIntrinsics, RGBDFrame, RigidPose


INTRINSICS = CameraIntrinsics(160, 160, 100, 75)
ROTATION = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
CAMERA_M = np.array([-1.1, 0, 1.05])
PELVIS_M = np.array([1.1, 0.19, 0.78])


def fixture(yaw=0.52, *, roll=0.0, right_visible=True, right_offset=None):
    width, height = 201, 151
    rows, cols = np.mgrid[:height, :width]
    rays = np.stack([
        (cols - INTRINSICS.cx) / INTRINSICS.fx,
        (rows - INTRINSICS.cy) / INTRINSICS.fy,
        np.ones_like(rows),
    ], axis=-1)
    c, s = np.cos(yaw), np.sin(yaw)
    yaw_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    c_roll, s_roll = np.cos(roll), np.sin(roll)
    roll_rotation = np.array([[1, 0, 0], [0, c_roll, -s_roll], [0, s_roll, c_roll]])
    pelvis_rotation = yaw_rotation @ roll_rotation
    rgb = np.full((height, width, 3), 60, dtype=np.uint8)
    depth = np.full((height, width), 5.0)
    markers = [(LEFT_MARKER_M, [13, 210, 210])]
    if right_visible:
        right = np.asarray(RIGHT_MARKER_M)
        if right_offset is not None:
            right = right + right_offset
        markers.append((right, [240, 20, 205]))
    for body_position, color in markers:
        field_position = PELVIS_M + pelvis_rotation @ body_position
        center = ROTATION.T @ (field_position - CAMERA_M)
        ray_sq = (rays**2).sum(axis=-1)
        along_ray = rays @ center
        discriminant = along_ray**2 - ray_sq * (center @ center - MARKER_RADIUS_M**2)
        hit = discriminant >= 0
        depth[hit] = (along_ray[hit] - np.sqrt(discriminant[hit])) / ray_sq[hit]
        rgb[hit] = color
    return RGBDFrame(0.2, "camera_optical", rgb, depth, INTRINSICS)


def survey(t=0.2, source="camera_optical", provenance="synthetic_fixture"):
    return RigidPose(t, source, "field", ROTATION, CAMERA_M, provenance, 0.02, 0.01)


@pytest.mark.parametrize("yaw", [-0.63, 0.0, 0.52, np.pi - 0.11])
def test_recovers_pelvis_xy_and_directed_heading(yaw):
    pose = MarkerLocalizer(allow_synthetic_fixture=True).estimate(fixture(yaw), survey())
    assert pose is not None
    assert pose.frame_id == "field"
    assert pose.pose_provenance == "synthetic_fixture"
    assert pose.timestamp_s == 0.2
    assert pose.fresh(0.22, 0.05)
    assert not pose.fresh(0.26, 0.05)
    assert not pose.fresh(0.1, 0.05)
    np.testing.assert_allclose(pose.xy_m, PELVIS_M[:2], atol=0.02)
    assert abs(np.arctan2(np.sin(pose.yaw_rad - yaw), np.cos(pose.yaw_rad - yaw))) < 0.05
    assert np.linalg.eigvalsh(pose.covariance).min() > 0


def test_rejects_missing_or_inconsistent_markers_and_stale_survey():
    localizer = MarkerLocalizer(allow_synthetic_fixture=True)
    assert localizer.estimate(fixture(right_visible=False), survey()) is None
    assert localizer.estimate(fixture(right_offset=[0, -0.15, 0]), survey()) is None
    assert localizer.estimate(fixture(roll=np.deg2rad(45)), survey()) is None
    assert localizer.estimate(fixture(), survey(t=0.15)) is None
    bad_depth = fixture()
    bad_depth.depth_m[bad_depth.rgb[..., 0] == 240] = 0.0
    assert localizer.estimate(bad_depth, survey()) is None


def test_rejects_ambiguous_marker_blob():
    localizer = MarkerLocalizer(allow_synthetic_fixture=True)
    frame = fixture()
    cyan = frame.rgb[..., 1] == 210
    frame.rgb[np.roll(cyan, 30, axis=0)] = frame.rgb[cyan]
    frame.depth_m[np.roll(cyan, 30, axis=0)] = frame.depth_m[cyan]
    assert localizer.estimate(frame, survey()) is None


def test_checks_frame_and_survey_provenance():
    localizer = MarkerLocalizer()
    frame = fixture()
    with pytest.raises(ValueError, match="source"):
        localizer.estimate(frame, survey(source="unrelated"))
    with pytest.raises(ValueError, match="explicit"):
        localizer.estimate(frame, survey())
    with pytest.raises(ValueError, match="fixed camera"):
        localizer.estimate(frame, survey(provenance="live_estimate"))
    assert localizer.estimate(frame, survey(provenance="fixed_calibration")) is not None
