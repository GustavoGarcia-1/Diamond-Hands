"""Causal form-context tests; supplied trajectories exist only in this fixture."""
from types import SimpleNamespace

import numpy as np
import pytest

from g1_receiver.catch_form import CatchFormContext
from g1_receiver.go_catch_form_rl import CausalCatchContext, VisualReceivePhase, context_vector
from g1_receiver.perception import BallObservation, GRAVITY_M_S2


def packet(t, *, variance=1e-5, vertical_velocity=4.):
    ball = BallObservation(t, 'field', np.array([2. + 8.*t, 0., 1.7 + vertical_velocity*t - .5*GRAVITY_M_S2*t*t]),
                           np.eye(3)*variance, 'synthetic_fixture')
    pose = SimpleNamespace(timestamp_s=t, z_m=.8, z_std_m=.01,
                           frame_id='field', pose_provenance='synthetic_fixture',
                           fresh=lambda now, age: 0. <= now-t <= age)
    return pose, SimpleNamespace(timestamp_s=t), ball


def fit_context(**kwargs):
    context = CausalCatchContext((10., 0., 1.), (1., 0., 0.))
    result = None
    for t in (.40, .44, .48, .52, .56, .60):
        pose, frame, ball = packet(t, **kwargs)
        context.detector = SimpleNamespace(detect=lambda *_, value=ball: value)
        result = context.update(t, pose, frame, object(), .25, 'receive')
    return context, result, pose, frame


def test_context_predicts_future_crossing_height_not_current_ball_height():
    _, result, _, _ = fit_context()
    assert result.causal_fit_valid
    assert result.time_to_arrival_s == pytest.approx(.4)
    assert result.predicted_intercept_height_above_shoulder_m == pytest.approx(1.7+4.-.5*GRAVITY_M_S2-.8-.25)
    assert result.intercept_height_std_m >= np.sqrt(.01**2+.03**2)
    assert result.perception_timestamp_s == .60


def test_missing_stale_or_mismatched_pose_abstains_without_invented_height():
    context, _, pose, frame = fit_context()
    for now, receiver in ((.60, None), (.80, pose)):
        result = context.update(now, receiver, frame, object(), .25, 'receive')
        assert not result.causal_fit_valid
        assert result.predicted_intercept_height_above_shoulder_m is None
    pose.pose_provenance = 'live_estimate'
    assert not context.update(.60, pose, frame, object(), .25, 'receive').causal_fit_valid


def test_ascending_future_crossing_is_not_over_shoulder_descent():
    _, result, _, _ = fit_context(vertical_velocity=12.)
    assert not result.causal_fit_valid


def test_invalid_context_features_have_explicit_mask_and_no_stale_coordinates():
    result = context_vector(CatchFormContext(over_shoulder=True, phase='receive'))
    np.testing.assert_array_equal(result, [0., 1., 0., 0., 0., 0., 0., 0.])
    _, context, _, _ = fit_context()
    result = context_vector(context)
    assert result.shape == (8,) and np.isfinite(result).all()
    assert result[3] == 1.
    assert result[6] == pytest.approx(.4/1.5)


def test_duplicate_frame_does_not_make_a_ballistic_fit_from_one_measurement():
    context = CausalCatchContext((10., 0., 1.), (1., 0., 0.))
    pose, frame, ball = packet(.4)
    context.detector = SimpleNamespace(detect=lambda *_: ball)
    for now in (.4, .42, .44, .46):
        assert not context.update(now, pose, frame, object(), .25, 'receive').causal_fit_valid


def test_form_hold_hypothesis_survives_occlusion_but_not_fresh_visible_escape():
    phase = VisualReceivePhase()
    sensor = SimpleNamespace(valid=True, last_valid=1., dwell=.16, e=np.zeros(3))
    assert phase.update(1.02, sensor, False) == 'approach'
    assert phase.update(1.02, sensor, True) == 'hold'
    sensor.valid = False
    sensor.dwell = 0.
    sensor.e[:] = 99.
    assert phase.update(1.2, sensor, True) == 'hold'
    # No synthetic dwell, validity, or terminal reward is written back.
    assert not sensor.valid and sensor.dwell == 0.
    sensor.valid = True
    sensor.last_valid = 1.2
    assert phase.update(1.22, sensor, True) == 'receive'


def test_stale_or_repeated_visual_sample_cannot_change_form_phase():
    phase = VisualReceivePhase()
    sensor = SimpleNamespace(valid=True, last_valid=1., dwell=.16, e=np.zeros(3))
    assert phase.update(1.2, sensor, True) == 'receive'
    sensor.last_valid = 1.2
    assert phase.update(1.22, sensor, True) == 'hold'
    sensor.e[:] = 1.  # corrupt repeated sample must not advance a transition
    assert phase.update(1.24, sensor, True) == 'hold'
