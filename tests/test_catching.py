"""Catch gating from synthetic sensor observations and actual RGB-D pixels."""

from dataclasses import replace

import numpy as np
import pytest

from g1_receiver.catching import CatchWindow, VisualCatchGate
from g1_receiver.localization import ReceiverPose
from g1_receiver.perception import (
    BallObservation, CameraIntrinsics, GRAVITY_M_S2, RGBDFrame, RigidPose,
)


OPTICAL_TO_FIELD = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
INTRINSICS = CameraIntrinsics(135., 135., 80., 60.)
STAMPS = (0.10, 0.14, 0.18, 0.22)
CROSSING_S = 0.50
PELVIS_Z_M = 0.8


def point_at(t, *, target_y=0., target_z=0.25, vx=-1.2, vz=-1.):
    elapsed = t - CROSSING_S
    return np.array([
        0.28 + vx * elapsed, target_y,
        PELVIS_Z_M + target_z + vz * elapsed - 0.5 * GRAVITY_M_S2 * elapsed**2,
    ])


class ObservationDetector:
    def __init__(self, **flight):
        self.flight = flight

    def detect(self, frame, survey):
        return BallObservation(
            frame.timestamp_s, survey.target_frame, point_at(frame.timestamp_s, **self.flight),
            np.eye(3) * 0.00002, survey.provenance,
        )


def sample(t, *, draw_ball=False, **flight):
    survey = RigidPose(
        t, "camera_optical", "field", OPTICAL_TO_FIELD, np.array([-1.75, 0., 1.]),
        "synthetic_fixture",
    )
    rgb = np.full((121, 161, 3), 70, dtype=np.uint8)
    depth = np.zeros(rgb.shape[:2], dtype=float)
    if draw_ball:
        camera = survey.rotation.T @ (point_at(t, **flight) - survey.translation)
        u = round(INTRINSICS.cx + INTRINSICS.fx * camera[0] / camera[2])
        v = round(INTRINSICS.cy + INTRINSICS.fy * camera[1] / camera[2])
        yy, xx = np.ogrid[:rgb.shape[0], :rgb.shape[1]]
        mask = ((xx - u) / 7)**2 + ((yy - v) / 4)**2 <= 1
        rgb[mask] = [210, 105, 40]
        depth[mask] = camera[2]
    frame = RGBDFrame(t, "camera_optical", rgb, depth, INTRINSICS)
    receiver = ReceiverPose(
        t, "field", np.zeros(2), 0., np.eye(3) * 0.000025,
        "synthetic_fixture", z_m=PELVIS_Z_M, z_std_m=0.005,
    )
    return frame, survey, receiver


def run(gate, stamps=STAMPS, *, draw_ball=False, **flight):
    result = None
    for t in stamps:
        result = gate.update(*sample(t, draw_ball=draw_ball, **flight), t)
    return result


def test_sensor_trigger_has_one_edge_and_resets_for_next_trial():
    gate = VisualCatchGate(allow_synthetic_fixture=True)
    gate.detector = ObservationDetector()
    decision = run(gate)
    assert decision.status == "triggered"
    assert decision.triggered and decision.latched and decision.ball_visible
    assert decision.prediction.inliers == 4
    assert decision.time_to_crossing_s == pytest.approx(0.28)
    np.testing.assert_allclose(decision.receiver_relative_crossing_m, [0.28, 0., 0.25])
    np.testing.assert_allclose(decision.velocity_mps, [-1.2, 0., -1.])
    latched = run(gate, [0.26])
    assert latched.status == "latched"
    assert latched.latched and not latched.triggered
    gate.reset()
    assert run(gate, [STAMPS[0]]).status == "no_stable_crossing"


def test_gate_can_trigger_from_pixels_without_an_oracle_observation():
    gate = VisualCatchGate(
        CatchWindow(max_crossing_std_m=0.35, max_time_std_s=0.25, min_lead_s=0.10),
        allow_synthetic_fixture=True,
    )
    decision = run(gate, draw_ball=True)
    assert decision.triggered
    assert decision.prediction.inliers == 4
    assert decision.time_to_crossing_s == pytest.approx(0.28, abs=0.02)


@pytest.mark.parametrize(
    ("flight", "window", "stamps", "expected"),
    [
        ({"target_y": 0.22}, CatchWindow(), STAMPS, "outside_reach"),
        ({"target_z": 0.6}, CatchWindow(), STAMPS, "outside_reach"),
        ({"vx": 1.2}, CatchWindow(), STAMPS, "not_incoming_descending"),
        ({"vz": 1.0}, CatchWindow(), STAMPS, "not_incoming_descending"),
        ({}, CatchWindow(max_lead_s=0.20), STAMPS, "too_early"),
        ({}, CatchWindow(), (0.29, 0.33, 0.37, 0.41), "too_late"),
        ({}, CatchWindow(max_time_std_s=0.001), STAMPS, "uncertain_crossing"),
    ],
)
def test_rejects_crossings_outside_window(flight, window, stamps, expected):
    gate = VisualCatchGate(window, allow_synthetic_fixture=True)
    gate.detector = ObservationDetector(**flight)
    decision = run(gate, stamps)
    assert decision.status == expected
    assert not decision.triggered and not decision.latched


def test_rejects_stale_duplicate_missing_and_untrusted_input():
    gate = VisualCatchGate(allow_synthetic_fixture=True)
    f, s, p = sample(0.10)
    assert gate.update(f, s, p, 0.20).status == "stale_frame"
    assert gate.update(f, s, p, 0.09).status == "stale_frame"
    assert gate.update(f, s, p, 0.10).status == "ball_not_visible"
    assert gate.update(f, s, p, 0.10).status == "out_of_order_frame"
    with pytest.raises(ValueError, match="explicit fixture opt-in"):
        VisualCatchGate().update(f, s, p, 0.10)
    gate.detector = ObservationDetector()
    f, s, p = sample(0.22)
    assert gate.update(f, s, replace(p, timestamp_s=0.05), 0.22).status == "no_fresh_receiver_pose"
    f, s, p = sample(0.26)
    assert gate.update(f, s, replace(p, z_m=None), 0.26).status == "no_receiver_height"
    f, s, p = sample(0.30)
    with pytest.raises(ValueError, match="match the surveyed field"):
        gate.update(f, s, replace(p, frame_id="other"), 0.30)


def test_current_detection_must_support_fitted_crossing():
    gate = VisualCatchGate(
        CatchWindow(min_lead_s=0.12, max_lead_s=0.25),
        allow_synthetic_fixture=True,
    )
    gate.detector = ObservationDetector()
    assert run(gate, STAMPS).status == "too_early"

    class OutlierDetector(ObservationDetector):
        def detect(self, frame, survey):
            observation = super().detect(frame, survey)
            return replace(observation, position_m=observation.position_m + [0.5, 0.4, 0.3])

    gate.detector = OutlierDetector()
    decision = run(gate, [0.26])
    assert decision.status == "latest_detection_outlier"
    assert decision.ball_visible and not decision.triggered
