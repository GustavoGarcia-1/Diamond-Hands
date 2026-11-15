import numpy as np
import pytest

from g1_receiver.catch_reward import VisualRetentionProxy
from g1_receiver.localization import ReceiverPose
from g1_receiver.perception import BallObservation


def measurements(t, relative=(.23, 0., .26), speed=.4):
    xy = np.array([speed*t, .1*t])
    pose = ReceiverPose(t, 'field', xy, 0., np.eye(3)*.0001,
                        'synthetic_fixture', .75, .01)
    ball = BallObservation(t, 'field', np.r_[xy, .75]+relative,
                           np.eye(3)*.0001, 'synthetic_fixture')
    return ball, pose


def feed(proxy, times, relative=(.23, 0., .26)):
    for t in times:
        result = proxy.update(*measurements(t, relative), t)
    return result


def test_retention_is_relative_to_measured_moving_receiver_and_requires_dwell():
    proxy = VisualRetentionProxy(allow_synthetic_fixture=True)
    early = feed(proxy, np.arange(0, 1.01, .04))
    assert not early.retained_proxy
    late = feed(proxy, np.arange(1.04, 2.41, .04))
    assert late.retained_proxy
    np.testing.assert_allclose(late.relative_velocity_mps, 0., atol=1e-10)
    assert late.stable_duration_s >= 2


def test_missing_stale_duplicate_and_large_gaps_reset_retention():
    for failure in ('missing', 'stale', 'duplicate', 'gap'):
        proxy = VisualRetentionProxy(allow_synthetic_fixture=True)
        assert feed(proxy, np.arange(0, 2.41, .04)).retained_proxy
        if failure == 'missing':
            result = proxy.update(None, None, 2.44)
        elif failure == 'stale':
            result = proxy.update(*measurements(2.44), 2.6)
        elif failure == 'duplicate':
            result = proxy.update(*measurements(2.4), 2.4)
        else:
            result = proxy.update(*measurements(2.6), 2.6)
        assert not result.retained_proxy
        assert result.stable_duration_s == 0
        reacquired = feed(proxy, [2.64, 2.68, 2.72, 2.76])
        assert not reacquired.retained_proxy


def test_volume_and_high_relative_speed_cannot_become_retention():
    proxy = VisualRetentionProxy(allow_synthetic_fixture=True)
    assert feed(proxy, np.arange(0, .5, .04), (.23, 0, -.6)).status == 'outside_receiving_volume'
    proxy.reset()
    for t in np.arange(0, .28, .04):
        result = proxy.update(*measurements(t, (.10+.8*t, 0, .26)), t)
    assert result.status == 'relative_motion_too_fast'
    assert not result.retained_proxy


def test_synthetic_inputs_require_opt_in_and_uncertainty_is_not_ignored():
    with pytest.raises(ValueError, match='opt-in'):
        VisualRetentionProxy().update(*measurements(0), 0)
    ball, pose = measurements(0)
    ball = BallObservation(0, 'field', ball.position_m, np.eye(3)*.1, 'synthetic_fixture')
    result = VisualRetentionProxy(allow_synthetic_fixture=True).update(ball, pose, 0)
    assert result.status == 'uncertain_measurement'
