from dataclasses import replace

import numpy as np
import pytest

from g1_receiver.catch_coordination_contract import (
    NativeEncoderHistory, decode_action, observation,
)
from g1_receiver.lower_cup_grip import OPEN


def packet():
    return dict(compact_error_rad=np.zeros(14), receive_error_rad=np.zeros(14),
                arm_velocity_rad_s=np.zeros(14), gravity_body=[0., 0., -1.],
                imu_gyro_rad_s=np.zeros(3), ttc_s=.3, ttc_std_s=.01,
                forecast_valid=True, phase=.5, phase_rate_s_inv=1.,
                visual_valid=False, visual_velocity_valid=False,
                ball_cup_error_body_m=np.zeros(3),
                ball_cup_relative_velocity_body_mps=np.zeros(3),
                ball_cup_sigma_m=np.ones(3), visual_dwell_s=0.,
                time_s=5., ball_capture_s=None)


def sample():
    history = NativeEncoderHistory()
    q = np.tile(OPEN, 2)
    assert history.update(4.96, q) is None
    q[0] -= .001
    return history.update(4.98, q)


def test_encoder_motion_is_measured_and_its_age_is_preserved():
    native = sample()
    assert native.velocity[0] == pytest.approx(-.05)
    p = packet()
    features = observation(p, native)
    assert features.shape == (95,)
    assert features[-1] == pytest.approx(.02 / .041)
    # Clock origin does not supply a release-time shortcut.
    p['time_s'] += 100.
    shifted = replace(native, timestamp_s=native.timestamp_s + 100.)
    np.testing.assert_allclose(features, observation(p, shifted), atol=1e-7)


def test_simulator_truth_cannot_change_the_observation():
    p, native = packet(), sample()
    before = observation(p, native)
    p.update(exact_ball_position=[1e6, 1e6, 1e6], contact_force=1e9,
             physical_catch_score=True, fixture_release_time=4.2)
    assert np.array_equal(before, observation(p, native))


def test_invalid_visual_velocity_cannot_masquerade_as_measured_stationarity():
    p, native = packet(), sample()
    p.update(visual_valid=True, ball_capture_s=4.98)
    p['ball_cup_relative_velocity_body_mps'] = np.ones(3) * 100.
    invalid = observation(p, native)
    p['ball_cup_relative_velocity_body_mps'] = np.zeros(3)
    assert np.array_equal(invalid, observation(p, native))
    p['visual_velocity_valid'] = True
    measured_stationary = observation(p, native)
    assert not np.array_equal(invalid, measured_stationary)
    assert invalid[65] == 0. and measured_stationary[65] == 1.


@pytest.mark.parametrize('stamp', [5.01, 4.9])
def test_future_or_stale_native_samples_are_rejected(stamp):
    with pytest.raises(ValueError):
        observation(packet(), replace(sample(), timestamp_s=stamp))


def test_repeated_encoder_timestamp_does_not_create_a_velocity():
    history = NativeEncoderHistory()
    history.update(1., np.tile(OPEN, 2))
    with pytest.raises(ValueError):
        history.update(1., np.tile(OPEN, 2))


def test_zero_and_bounded_actions_cannot_request_unbounded_grip_changes():
    assert decode_action([0., 0.]) == (0., 0.)
    assert decode_action([-1., 1.]) == (-1., .002)
    for bad in ([0.], [0., 1.01], [0., np.nan]):
        with pytest.raises(ValueError):
            decode_action(bad)
