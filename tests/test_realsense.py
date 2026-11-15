"""Small SDK interface fixtures for RealSense recording contracts."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from g1_receiver.perception import CameraIntrinsics, RigidPose
from g1_receiver.realsense import _collect, _color_axial_factors, _survey


class Frame:
    def __init__(self, timestamp, data, domain="hardware_clock"):
        self.timestamp = timestamp
        self.data = data
        self.domain = domain

    def get_frame_timestamp_domain(self):
        return self.domain

    def get_timestamp(self):
        return self.timestamp

    def get_data(self):
        return self.data

    def get_units(self):
        return 0.001


class Frameset:
    def __init__(self, color_time, depth_time, color_domain="hardware_clock"):
        self.color = Frame(color_time, np.full((2, 2, 3), 175, dtype=np.uint8), color_domain)
        self.depth = Frame(depth_time, np.array([[3000, 0], [1000, 2000]], dtype=np.uint16))

    def get_color_frame(self):
        return self.color

    def get_depth_frame(self):
        return self.depth


def survey_fixture():
    return RigidPose(
        0.0, "camera_optical", "field", np.eye(3), np.zeros(3), "fixed_calibration",
        position_std_m=0.03, angle_std_rad=0.01,
    )


class RealSenseContracts(unittest.TestCase):
    def collect(self, frames, count=2):
        input_frames = iter(frames)
        pipeline = SimpleNamespace(wait_for_frames=lambda: next(input_frames))
        align = SimpleNamespace(process=lambda frameset: frameset)
        return _collect(
            pipeline, align, None, np.full((2, 2), 0.5), 0.1, CameraIntrinsics(100, 100, 1, 1),
            survey_fixture(), {"serial": "unit-fixture"}, count, 5.0,
        )

    def test_capture_preserves_clock_and_rejects_bad_pairs(self):
        capture = self.collect([
            Frameset(1000.0, 980.0),
            Frameset(1000.0, 1000.0, color_domain="system_time"),
            Frameset(1000.0, 1001.0),
            Frameset(1033.0, 1001.0),
            Frameset(1033.0, 1034.0),
        ])
        self.assertEqual(capture["timestamp_domain"], "hardware_clock")
        self.assertEqual(capture["pose_provenance"], "fixed_calibration")
        np.testing.assert_allclose(capture["timestamps_s"], [1.001, 1.034])
        np.testing.assert_allclose(capture["camera_pose_timestamps_s"], [1.001, 1.034])
        np.testing.assert_allclose(capture["color_timestamps_s"], [1.0, 1.033])
        np.testing.assert_allclose(capture["depth_m"][0], [[5.8, 0], [1.8, 3.8]], atol=1e-6)
        self.assertEqual(capture["rgb"].dtype, np.uint8)
        self.assertGreater(capture["host_receive_monotonic_s"][1], capture["host_receive_monotonic_s"][0])

    def test_capture_rejects_camera_clock_reset(self):
        with self.assertRaisesRegex(RuntimeError, "backwards"):
            self.collect([Frameset(1000.0, 1000.0), Frameset(500.0, 500.0)])

    def test_axial_transform_uses_color_to_depth_extrinsics(self):
        intrinsics = CameraIntrinsics(1, 1, 0, 0)
        extrinsics = SimpleNamespace(translation=np.array([0, 0, 0.1]))
        rs = SimpleNamespace(rs2_transform_point_to_point=lambda e, v: np.asarray(v) + e.translation)
        factors, offset = _color_axial_factors(rs, extrinsics, intrinsics, 2, 2)
        np.testing.assert_array_equal(factors, np.ones((2, 2)))
        self.assertAlmostEqual(offset, 0.1)

    def test_survey_binds_pose_to_serial(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "survey.json"
            path.write_text(json.dumps({
                "serial": "surveyed",
                "rotation": np.eye(3).tolist(),
                "translation_m": [0, 0, 1],
                "position_std_m": 0.02,
                "angle_std_rad": 0.03,
            }))
            self.assertEqual(_survey(path, "surveyed").provenance, "fixed_calibration")
            with self.assertRaisesRegex(ValueError, "serial"):
                _survey(path, "other")


if __name__ == "__main__":
    unittest.main()
