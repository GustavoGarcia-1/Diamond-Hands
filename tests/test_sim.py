import hashlib
import json

import numpy as np
import pytest

from g1_receiver.runtime import DEFAULT_ROBOT_XML, DEFAULT_RUNTIME_ROOT
from g1_receiver.sim import FPS, simulate

pytest.importorskip("mujoco")


def test_ballistic_render_keeps_truth_separate():
    release = (4.8, .12, 1.35)
    velocity = (-4.2, .05, 2.8)
    camera = (.12, -.03, 1.25)
    sensor, truth, overview = simulate(
        12, overview=True, release_position_m=release, release_velocity_mps=velocity,
        camera_position_m=camera,
    )
    assert sensor["pose_provenance"] == "synthetic_fixture"
    assert sensor["evidence_class"] == "rendered_simulation"
    assert "ball_position_m" not in sensor
    np.testing.assert_array_equal(sensor["timestamps_s"], sensor["camera_pose_timestamps_s"])
    np.testing.assert_array_equal(sensor["timestamps_s"], truth["timestamps_s"])
    np.testing.assert_allclose(np.diff(sensor["timestamps_s"]), 1 / FPS)
    np.testing.assert_allclose(sensor["camera_translation_m"][0], camera)
    assert sensor["rgb"].dtype == np.uint8
    assert sensor["rgb"].shape == (*sensor["depth_m"].shape, 3)
    assert overview.shape == sensor["rgb"].shape
    assert np.isfinite(sensor["depth_m"]).all()
    assert not np.array_equal(sensor["rgb"][0], sensor["rgb"][-1])
    assert not np.any(truth["ball_ground_contact"])
    expected = np.asarray(release) + truth["timestamps_s"][:, None] * np.asarray(velocity)
    expected[:, 2] -= 9.81 / 2 * truth["timestamps_s"] ** 2
    np.testing.assert_allclose(truth["ball_position_m"], expected, atol=1e-8)


def test_invalid_fixture_vector_is_rejected():
    with pytest.raises(ValueError, match="finite XYZ"):
        simulate(2, release_velocity_mps=(np.inf, 0, 0))


def test_bundled_g1_render_and_wrist_model():
    _, truth, _ = simulate(2, robot_xml=DEFAULT_ROBOT_XML)
    assert truth["robot_model_present"]
    assert truth["robot_motion"] == "static collision-free visual reference"
    np.testing.assert_array_equal(truth["robot_base_position_m"][0], truth["robot_base_position_m"][1])


def test_bundled_upstream_asset_manifest():
    manifest = json.loads((DEFAULT_RUNTIME_ROOT / "ASSET-MANIFEST.json").read_text())
    assert len(manifest["files"]) == 56
    for record in manifest["files"]:
        contents = (DEFAULT_RUNTIME_ROOT / record["path"]).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == record["sha256"], record["path"]
