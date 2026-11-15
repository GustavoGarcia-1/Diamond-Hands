from types import SimpleNamespace

import numpy as np
import pytest

from g1_receiver.transport_precontact_rl import CameraTranslation, transport_error, TransportPrecontactAudit
from test_precontact_native_rl import audit, packet


def pose_packet(t, xy=None, camera='one'):
    obs, pose, frame, survey = packet(t)
    pose.xy_m = np.array([t, -.5*t]) if xy is None else np.asarray(xy)
    frame.frame_id = camera
    return obs, pose, frame, survey


def test_translation_uses_only_distinct_past_single_camera_poses():
    fit = CameraTranslation()
    for t in [0., .04, .08, .12]:
        _, pose, frame, _ = pose_packet(t)
        assert fit.update(t, pose, frame) is None
    _, pose, frame, _ = pose_packet(.16)
    result = fit.update(.16, pose, frame)
    assert np.allclose(result['velocity_xy_mps'], [1., -.5])
    assert result['sample_count'] == 5
    assert fit.update(.17, pose, frame)['sample_count'] == 5
    assert fit.update(.1, pose, frame) is None
    assert fit.update(.30, pose, frame) is None
    frame.frame_id = 'different_camera'
    assert fit.update(.16, pose, frame) is None


def test_velocity_rejects_implausible_speed_and_invalid_covariance():
    fit = CameraTranslation()
    for t in [0., .04, .08, .12, .16]:
        _, pose, frame, _ = pose_packet(t, [10*t, 0.])
        assert fit.update(t, pose, frame) is None
    pose.covariance = -np.eye(2)
    assert fit.update(.16, pose, frame) is None


def test_translation_removes_future_motion_error_without_shrinking_uncertainty():
    error, sigma = transport_error([.4, -.2, .03], [.02, .03, .04], np.eye(3),
                                   [1., -.5], [.05, .06], .4, .01)
    assert np.allclose(error, [0., 0., .03])
    assert np.all(sigma >= [.02, .03, .04])
    assert sigma[2] == .04
    with pytest.raises(ValueError):
        transport_error([0., 0., 0.], [0., 0., 0.], np.eye(3), [1., 0.], [0., 0.], -.1, 0.)


def test_missing_translation_never_pays_old_reward_and_deadline_stays_absorbing():
    base = audit()
    a = TransportPrecontactAudit(base.profile)
    for key, value in base.__dict__.items():
        setattr(a, key, value)
    a.tracker.arrival = 1.
    for t in [0., .02, .04, .06, .08, .10, .12, .14]:
        obs, pose, frame, survey = pose_packet(t)
        target, native = a(obs, pose, frame, survey, np.zeros(29), np.zeros(14), np.zeros(14), True)
        assert np.array_equal(target, np.zeros(29))
        assert np.array_equal(native, np.zeros(14))
    assert a.positive_credit == 0.
    for t in [.16, .18, .20]:
        obs, pose, frame, survey = pose_packet(t)
        a(obs, pose, frame, survey, np.zeros(29), np.zeros(14), np.zeros(14), True)
    assert a.positive_credit > 0.
    before = a.positive_credit
    obs, pose, frame, survey = pose_packet(.20)
    obs.timestamp = .22
    a(obs, pose, frame, survey, np.zeros(29), np.zeros(14), np.zeros(14), True)
    assert a.positive_credit == before
    a.arrival.closed = True
    obs, pose, frame, survey = pose_packet(.24)
    a(obs, pose, frame, survey, np.zeros(29), np.zeros(14), np.zeros(14), True)
    assert a.positive_credit == before and a.positive_after_deadline == 0.
