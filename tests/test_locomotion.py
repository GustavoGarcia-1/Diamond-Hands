import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("onnxruntime")

def test_free_base_bare_wrist_cradle_against_neutral(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    for mode in ("cradle", "neutral"):
        subprocess.run(
            [sys.executable, "-m", "g1_receiver.locomotion", "--mode", mode, "--truth", str(tmp_path / f"{mode}.npz"), "--seconds", "1.5"],
            check=True,
        )
    with np.load(tmp_path / "cradle.npz") as cradle, np.load(tmp_path / "neutral.npz") as neutral:
        np.testing.assert_array_equal(cradle["ball_position_m"][0], neutral["ball_position_m"][0])
        np.testing.assert_array_equal(cradle["ball_velocity_mps"][0], neutral["ball_velocity_mps"][0])
        assert cradle["robot_upright"].min() > .9
        assert not cradle["ball_ground_contact"].any()
        assert cradle["ball_robot_contact"][-10:].all()
        assert cradle["ball_position_m"][-1, 2] > .8
        assert np.linalg.norm(cradle["ball_velocity_mps"][-1]) < .1
        assert neutral["ball_ground_contact"].any()
        assert not neutral["ball_robot_contact"].any()
        assert not any("_hand_" in body for body in cradle["robot_contact_bodies"])


def test_neutral_start_visual_catch_against_disabled_vision():
    from g1_receiver.locomotion import run
    from g1_receiver.runtime import DEFAULT_RUNTIME_ROOT

    release = {
        "release_position_m": (.84, 0., .38),
        "release_velocity_mps": (-1., 0., 4.36),
    }
    vision, _, _, active = run(DEFAULT_RUNTIME_ROOT, "visual-catch", 1.5, **release)
    neutral, _, _, disabled = run(
        DEFAULT_RUNTIME_ROOT, "visual-catch", 1.5, ball_vision=False, **release
    )
    edges = np.flatnonzero(active["catch_triggered"])
    assert len(edges) == 1
    trigger = edges[0]
    assert active["catch_capture_timestamp_s"][trigger] == active["timestamps_s"][trigger]
    assert active["catch_ball_visible"][trigger] and active["catch_inliers"][trigger] >= 4
    np.testing.assert_array_equal(
        active["upper_target_rad"][:trigger], disabled["upper_target_rad"][:trigger]
    )
    np.testing.assert_array_equal(
        vision["ball_position_m"][:trigger + 1], neutral["ball_position_m"][:trigger + 1]
    )
    arm_rate = np.abs(np.diff(active["upper_target_rad"], axis=0)) / np.diff(active["timestamps_s"])[:, None]
    assert arm_rate.max() <= active["arm_target_rate_rad_s"] + 1e-9
    assert not vision["ball_ground_contact"].any()
    assert vision["ball_robot_contact"][-10:].all()
    assert vision["ball_position_m"][-1, 2] > .8
    assert not disabled["catch_triggered"].any()
    assert neutral["ball_ground_contact"].any()
