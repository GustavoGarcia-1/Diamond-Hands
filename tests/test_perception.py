"""Sensor-style unit fixtures with synthetic truth used only for assertions."""

import tempfile
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import unittest

import numpy as np

from g1_receiver.perception import (
    GRAVITY_M_S2,
    BallObservation,
    BallTracker,
    CameraIntrinsics,
    ColorBallDetector,
    RGBDFrame,
    RigidPose,
    replay,
)


ROTATION = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
INTRINSICS = CameraIntrinsics(100, 100, 20, 20)


def image_fixture():
    rgb = np.full((40, 40, 3), 70, dtype=np.uint8)
    depth = np.full((40, 40), 4.0)
    rows, cols = np.ogrid[:40, :40]
    ball = ((cols - 25) / 5) ** 2 + ((rows - 18) / 3) ** 2 <= 1
    rgb[ball] = [205, 100, 40]
    depth[ball] = 2.0
    depth[18, 25] = 0
    return rgb, depth


def observation(t, offset=None):
    position = np.array([1.0 + 4.0 * t, -0.3 + 0.25 * t, 1.1 + 2.2 * t - 0.5 * GRAVITY_M_S2 * t**2])
    if offset is not None:
        position += offset
    return BallObservation(t, "field", position, np.eye(3) * 0.02**2, "synthetic_fixture")


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.rgb, self.depth = image_fixture()
        self.frame = RGBDFrame(1.0, "camera_optical", self.rgb, self.depth, INTRINSICS)
        self.detector = ColorBallDetector()

    def pose(self, t=1.0, source="camera_optical", uncertainty=0):
        return RigidPose(
            t, source, "field", ROTATION, np.array([1, 2, 3]),
            "synthetic_fixture", uncertainty,
        )

    def test_detection_transforms_capture_timestamp_and_uncertainty(self):
        detected = self.detector.detect(self.frame, self.pose(uncertainty=0.03))
        self.assertIsNotNone(detected)
        self.assertEqual(detected.timestamp_s, 1.0)
        self.assertEqual(detected.frame_id, "field")
        np.testing.assert_allclose(detected.position_m, [3, 1.9, 3.04], atol=0.01)
        self.assertGreater(np.linalg.eigvalsh(detected.covariance_m2).min(), 0.03**2)

    def test_rejects_stale_pose_frame_mismatch_and_invalid_depth(self):
        self.assertIsNone(self.detector.detect(self.frame, self.pose(0.9)))
        with self.assertRaisesRegex(ValueError, "source"):
            self.detector.detect(self.frame, self.pose(source="wrong"))
        self.depth[:] = np.nan
        self.assertIsNone(self.detector.detect(self.frame, self.pose()))

    def test_rejects_ambiguous_blobs(self):
        self.rgb[:, :20] = self.rgb[:, 20:]
        self.depth[:, :20] = self.depth[:, 20:]
        self.assertIsNone(self.detector.detect(self.frame, self.pose()))

    def test_rejects_invalid_calibration_and_privileged_pose_label(self):
        with self.assertRaisesRegex(ValueError, "focal"):
            CameraIntrinsics(0, 100, 20, 20)
        with self.assertRaisesRegex(ValueError, "registered"):
            RGBDFrame(1.0, "camera_optical", self.rgb, self.depth[:20], INTRINSICS)
        with self.assertRaisesRegex(ValueError, "proper"):
            RigidPose(1, "camera_optical", "field", -ROTATION, np.zeros(3), "live_estimate")
        with self.assertRaisesRegex(ValueError, "declare"):
            RigidPose(1, "camera_optical", "field", ROTATION, np.zeros(3), "sim_truth")


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = BallTracker(allow_synthetic_fixture=True)
        self.point = np.array([3.0, 0, 0])
        self.normal = np.array([1.0, 0, 0])

    def intercept(self, now, frame="field", **kwargs):
        return self.tracker.intercept(now, self.point, self.normal, frame, **kwargs)

    def test_ballistic_crossing_rejects_single_large_outlier(self):
        for i, t in enumerate([0.00, 0.037, 0.07, 0.103, 0.14, 0.172, 0.201]):
            self.tracker.add(observation(t, np.array([0.5, 0.6, -0.2]) if i == 3 else None))
        predicted = self.intercept(0.21)
        self.assertIsNotNone(predicted)
        self.assertEqual(predicted.inliers, 6)
        self.assertAlmostEqual(predicted.timestamp_s, 0.5)
        np.testing.assert_allclose(predicted.position_m, observation(0.5).position_m)
        self.assertGreater(predicted.time_std_s, 0)
        self.assertGreater(predicted.covariance_m2[1, 1], 0)
        self.assertAlmostEqual(predicted.covariance_m2[0, 0], 0)

    def test_requires_explicit_fixture_opt_in(self):
        with self.assertRaisesRegex(ValueError, "fixture opt-in"):
            BallTracker().add(observation(0))

    def test_stale_dropout_and_out_of_order(self):
        for t in [0, 0.035, 0.07]:
            self.assertTrue(self.tracker.add(observation(t)))
        self.assertIsNotNone(self.intercept(0.071))
        self.assertFalse(self.tracker.add(observation(0.07)))
        self.assertFalse(self.tracker.add(observation(0.06)))
        self.assertIsNone(self.intercept(0.05))
        self.assertIsNone(self.intercept(0.18))
        self.assertTrue(self.tracker.add(observation(0.19)))
        self.assertIsNone(self.intercept(0.19))
        with self.assertRaisesRegex(ValueError, "catch plane"):
            self.intercept(0.19, frame="camera_optical")

    def test_measured_robot_pose_transforms_plane_and_adds_uncertainty(self):
        for t in [0, 0.035, 0.07]:
            self.tracker.add(observation(t))
        pose = RigidPose(
            0.07, "receiver", "field", np.eye(3), np.array([0.5, 0, 0]),
            "synthetic_fixture", 0.06, 0.01,
        )
        predicted = self.intercept(0.071, frame="receiver", receiver_pose=pose)
        self.assertIsNotNone(predicted)
        self.assertAlmostEqual(predicted.timestamp_s, 0.625)
        self.assertAlmostEqual(predicted.position_m[0], 3.5)
        self.assertGreaterEqual(predicted.time_std_s, 0.06 / 4)
        self.tracker.add(observation(0.12))
        self.tracker.add(observation(0.17))
        self.assertIsNone(self.intercept(0.18, frame="receiver", receiver_pose=pose))

    def test_capture_gap_at_age_boundary_keeps_history_despite_clock_roundoff(self):
        tracker = BallTracker(max_age_s=.08, allow_synthetic_fixture=True)
        # Shift the clock to reproduce the observed 5.08 -> 5.16 boundary.
        for timestamp in [5.00, 5.04, 5.08, 5.160000000000027]:
            sample = observation(timestamp-5.)
            tracker.add(BallObservation(timestamp, sample.frame_id, sample.position_m,
                                        sample.covariance_m2, sample.pose_provenance))
        predicted = tracker.intercept(5.160000000000027, self.point, self.normal, 'field')
        self.assertIsNotNone(predicted)
        self.assertEqual(predicted.inliers, 4)
        self.assertIsNotNone(tracker.intercept(5.240000000000054, self.point, self.normal, 'field'))
        self.assertIsNone(tracker.intercept(5.240001, self.point, self.normal, 'field'))
        sample = observation(.240001)
        tracker.add(BallObservation(5.240001, sample.frame_id, sample.position_m,
                                    sample.covariance_m2, sample.pose_provenance))
        self.assertIsNone(tracker.intercept(5.240001, self.point, self.normal, 'field'))

    def test_past_parallel_and_invalid_plane_are_rejected(self):
        for t in [0, 0.035, 0.07]:
            self.tracker.add(observation(t))
        with self.assertRaisesRegex(ValueError, "nonzero"):
            self.tracker.intercept(0.07, self.point, np.zeros(3), "field")
        self.assertIsNone(self.tracker.intercept(0.07, np.zeros(3), self.normal, "field"))
        self.assertIsNone(
            self.tracker.intercept(0.07, np.array([0, 0, 10]), np.array([0, 0, 1]), "field")
        )


class ReplayTests(unittest.TestCase):
    def test_fixture_opt_in_and_timestamp_validation(self):
        rgb, depth = image_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.npz"
            arrays = {
                "timestamps_s": np.array([1.0]),
                "camera_pose_timestamps_s": np.array([1.0]),
                "pose_provenance": "synthetic_fixture",
                "intrinsics": [100, 100, 20, 20],
                "rgb": rgb[None, ...],
                "depth_m": depth[None, ...],
                "camera_rotation": ROTATION[None, ...],
                "camera_translation_m": np.zeros((1, 3)),
            }
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "allow-synthetic-fixture"):
                replay(path, np.ones(3), np.ones(3))
            output = StringIO()
            with redirect_stdout(output):
                replay(path, np.ones(3), np.ones(3), allow_synthetic_fixture=True)
            self.assertEqual(output.getvalue(), "")
            arrays["timestamps_s"] = np.array([1.0, 1.0])
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                replay(path, np.ones(3), np.ones(3), allow_synthetic_fixture=True)


if __name__ == "__main__":
    unittest.main()
