"""Sensor adapter checks without a simulator-state input."""

from dataclasses import replace

import numpy as np
import pytest

from g1_receiver.localization import ReceiverPose
from g1_receiver.observable import (
    ACTION_LIMITS, ACTION_SIZE, ARM_JOINTS, OBS_SLICES, ObservableGenCoord, bounded_command,
)
from g1_receiver.perception import BallObservation


PROPRIOCEPTION = (np.zeros(29), np.zeros(29), np.array([1., 0., 0., 0.]), np.zeros(3))


def samples(t):
    receiver = ReceiverPose(
        t, "field", np.array([t, 0.]), 0., np.eye(3) * .01,
        "synthetic_fixture", z_m=.8, z_std_m=.02,
    )
    ball = BallObservation(
        t, "field", np.array([1. - 2 * t, .2, 1.1]), np.eye(3) * .01,
        "synthetic_fixture",
    )
    return receiver, ball


def observe(adapter, now):
    return adapter.observe(now, PROPRIOCEPTION, np.array([2., 0.]), np.zeros(ACTION_SIZE), np.zeros(29))


def test_causal_relative_velocity_and_masks():
    adapter = ObservableGenCoord(allow_synthetic_fixture=True)
    for t in (.0, .04, .08):
        assert adapter.update_camera(t, *samples(t))
    state = observe(adapter, .10)
    np.testing.assert_allclose(state.vector[OBS_SLICES["ball_relative_xyz_m"]], [.76, .2, .3])
    np.testing.assert_allclose(state.vector[OBS_SLICES["ball_relative_velocity_mps"]], [-3., 0., 0.], atol=1e-7)
    assert state.vector[OBS_SLICES["ball_age_s"]][0] == pytest.approx(.02)
    assert state.vector[OBS_SLICES["ball_velocity_valid"]][0] == 1
    assert not adapter.update_camera(.06, *samples(.06))
    assert observe(adapter, .07).receiver is None
    assert observe(adapter, .21).ball is None


def test_missing_frame_clears_velocity_and_required_uncertainty():
    adapter = ObservableGenCoord(allow_synthetic_fixture=True)
    for t in (.0, .04, .08):
        adapter.update_camera(t, *samples(t))
    receiver, ball = samples(.12)
    adapter.update_camera(.12, receiver, None)
    assert observe(adapter, .12).vector[OBS_SLICES["ball_valid"]][0] == 0
    receiver, ball = samples(.16)
    adapter.update_camera(.16, receiver, ball)
    assert observe(adapter, .16).vector[OBS_SLICES["ball_velocity_valid"]][0] == 0
    receiver, ball = samples(.20)
    adapter.update_camera(.20, replace(receiver, z_std_m=None), ball)
    assert observe(adapter, .20).receiver is None


@pytest.mark.parametrize(
    "changes",
    [
        {"z_std_m": -0.1},
        {"z_std_m": np.nan},
        {"covariance": np.diag([.01, -.01, .01])},
        {"covariance": np.array([[.01, .02, 0.], [.03, .01, 0.], [0., 0., .01]])},
    ],
)
def test_rejects_malformed_receiver_uncertainty(changes):
    adapter = ObservableGenCoord(allow_synthetic_fixture=True)
    receiver, ball = samples(0.)
    with pytest.raises(ValueError, match="uncertainty"):
        adapter.update_camera(0., replace(receiver, **changes), ball)


def test_fixture_opt_in_and_measured_capture_time():
    receiver, ball = samples(0.)
    with pytest.raises(ValueError, match="fixture"):
        ObservableGenCoord().update_camera(0., receiver, ball)
    with pytest.raises(ValueError, match="timestamp"):
        ObservableGenCoord(allow_synthetic_fixture=True).update_camera(.04, receiver, ball)


def test_action_envelope_reconstructs_existing_cradle_and_limits_drive():
    from g1_receiver.locomotion import cradle_joint_targets

    names = list(ARM_JOINTS)
    nominal = np.zeros(len(names))
    nominal[names.index("left_shoulder_roll_joint")] = .20
    nominal[names.index("right_shoulder_roll_joint")] = -.20
    expected = cradle_joint_targets(nominal, names)
    action = np.r_[[2., -2., 2.], (expected - nominal) / ACTION_LIMITS[3:]]
    assert np.all(np.abs(action[3:]) <= 1)
    drive, upper = bounded_command(action, nominal, names, np.tile([-2., 2.], (len(names), 1)))
    np.testing.assert_allclose(drive, [.35, -.22, .35])
    np.testing.assert_allclose(upper, expected)
