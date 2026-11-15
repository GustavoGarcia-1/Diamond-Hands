from dataclasses import replace
import numpy as np
import pytest
from g1_receiver.perception import CameraIntrinsics, RGBDFrame, RigidPose
from g1_receiver.rgbd_impairments import RGBDImpairments


def capture(t):
    frame = RGBDFrame(t, 'camera', np.full((5, 7, 3), 100, np.uint8),
                      np.full((5, 7), 2.), CameraIntrinsics(100., 100., 3., 2.))
    pose = RigidPose(t, 'camera', 'field', np.eye(3), np.zeros(3), 'synthetic_fixture')
    return frame, pose


def test_nominal_is_exact_identity_and_inputs_are_not_mutated():
    f, p = capture(0.)
    clean = RGBDImpairments({})
    assert clean.apply('pass', f, p, 0.) == (f, p)
    noisy = RGBDImpairments(dict(depth_noise_floor_m=.01, rgb_noise_std=3., survey_position_std_m=.005))
    observed, survey = noisy.apply('pass', f, p, 0.)
    assert not np.array_equal(observed.depth_m, f.depth_m)
    assert np.array_equal(f.depth_m, np.full((5, 7), 2.))
    assert np.array_equal(f.rgb, np.full((5, 7, 3), 100, np.uint8))
    assert survey.provenance == p.provenance
    assert survey.position_std_m == .005


def test_latency_preserves_original_exposure_and_never_returns_future_frame():
    observer = RGBDImpairments(dict(latency_s=.04))
    for t in (0., .02):
        f, p = observer.apply('pass', *capture(t), t)
        assert not f.depth_m.any()
    f, p = observer.apply('pass', *capture(.04), .04)
    assert f.timestamp_s == p.timestamp_s == 0.
    f, p = observer.apply('pass', *capture(.06), .06)
    assert f.timestamp_s == p.timestamp_s == .02
    with pytest.raises(ValueError):
        observer.apply('pass', *capture(.08), .06)


def test_camera_rng_is_independent_of_other_camera_call_order():
    config = dict(depth_noise_floor_m=.01, frame_drop_probability=.2,
                  latency_s=.02, latency_jitter_s=.01, survey_angle_std_rad=.005)
    one, two = RGBDImpairments(config), RGBDImpairments(config)
    for t in np.arange(10)*.02:
        expected = one.apply('pass', *capture(t), t)
        two.apply('route', *capture(t), t)
        actual = two.apply('pass', *capture(t), t)
        assert expected[0].timestamp_s == actual[0].timestamp_s
        assert np.array_equal(expected[0].depth_m, actual[0].depth_m)
        assert np.array_equal(expected[1].rotation, actual[1].rotation)


def test_dropout_and_rate_reduction_do_not_invent_new_exposures():
    dropped = RGBDImpairments(dict(frame_drop_probability=1.))
    f, _ = dropped.apply('pass', *capture(1.), 1.)
    assert not f.depth_m.any()
    observer = RGBDImpairments(dict(rate_hz=25.))
    stamps = [observer.apply('pass', *capture(t), t)[0].timestamp_s for t in (0., .02, .04, .06)]
    assert stamps == [0., 0., .04, .04]
