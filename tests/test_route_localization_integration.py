"""Closed-loop slant with an injected optical dropout and off-controller scoring."""

import numpy as np
import pytest

from g1_receiver.localization import MarkerLocalizer
from g1_receiver.locomotion import run
from g1_receiver.routes import make_route
from g1_receiver.runtime import DEFAULT_RUNTIME_ROOT


def test_slant_uses_rgbd_localization_and_holds_after_dropout(monkeypatch):
    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    estimate = MarkerLocalizer.estimate

    def interrupt(self, frame, survey):
        if 1.10 <= frame.timestamp_s < 1.28:
            return None
        return estimate(self, frame, survey)

    monkeypatch.setattr(MarkerLocalizer, "estimate", interrupt)
    truth, _, sensor, commands = run(DEFAULT_RUNTIME_ROOT, "route", 6.0, route_name="slant", route_scale=0.45)

    timestamps = commands["timestamps_s"]
    status = commands["status"]
    velocity = commands["command_base_velocity"]
    after_hold = (timestamps >= 1.18) & (timestamps <= 1.26)
    assert np.count_nonzero(after_hold) >= 3
    assert np.all(status[after_hold] == "no_fresh_pose")
    np.testing.assert_array_equal(velocity[after_hold], np.zeros_like(velocity[after_hold]))
    assert np.any((timestamps < 1.10) & (status == "follow_route"))
    assert np.any((timestamps > 1.28) & (status == "follow_route") & (velocity[:, 1] > 0.10))

    fresh = np.isfinite(commands["estimate_timestamps_s"])
    assert np.count_nonzero(fresh) >= 0.85 * len(timestamps)
    assert np.max(timestamps[fresh] - commands["estimate_timestamps_s"][fresh]) <= 0.061
    assert sensor["pose_provenance"] == "synthetic_fixture"
    assert "robot_base_position_m" not in sensor
    assert "robot_base_position_m" not in commands

    # Physical state is used only after control has finished, to score the result.
    xy = truth["robot_base_position_m"][:, :2]
    pose_error = np.linalg.norm(commands["estimated_xy_m"][fresh] - xy[fresh], axis=1)
    assert np.quantile(pose_error, 0.95) < 0.06
    slant = make_route("slant", scale=0.45, speed_mps=0.35)
    assert slant.project(xy[-1])[1] > 0.85 * slant.length_m
    assert np.linalg.norm(xy[-1] - slant.waypoints[-1]) < 0.15
    assert truth["robot_upright"].min() > 0.9
