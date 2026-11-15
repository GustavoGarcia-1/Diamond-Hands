from copy import deepcopy
import numpy as np
import pytest
from g1_receiver.late_reach_rl import observation, SensorReward, LateReachEnv, OBS_SIZE


def packet():
    return dict(compact_error_rad=np.zeros(14), receive_error_rad=np.zeros(14),
                arm_velocity_rad_s=np.zeros(14), gravity_body=np.array([0., 0., -1.]),
                imu_gyro_rad_s=np.zeros(3), ttc_s=.6, ttc_std_s=.01, forecast_valid=True,
                phase=0., phase_rate_s_inv=0., visual_valid=False, visual_velocity_valid=False,
                ball_cup_error_body_m=np.zeros(3), ball_cup_relative_velocity_body_mps=np.zeros(3),
                ball_cup_sigma_m=np.full(3, .01), visual_dwell_s=0., upright_cos=1.,
                time_s=5., ball_capture_s=None, palm_up_cos=[.8, .8],
                palm_height_above_shoulder_m=[0., 0.])


def test_actor_has_no_absolute_clock_or_physical_score_channel():
    a = packet()
    b = deepcopy(a)
    b.update(time_s=99., ball_capture_s=98.99, exact_ball_position=[100., 0., 0.],
             physical_catch_score=True, contact_force=1000.)
    assert observation(a).shape == (OBS_SIZE,)
    assert np.array_equal(observation(a), observation(b))


def test_quiet_but_early_extension_is_penalized_separately_from_angular_speed():
    a, b = packet(), packet()
    b['compact_error_rad'] = np.ones(14)
    assert SensorReward().update(b) < SensorReward().update(a)
    b.update(ttc_s=.1, phase=.7, arm_velocity_rad_s=np.full(14, 5.))
    a.update(ttc_s=.1, phase=.7)
    assert SensorReward().update(b) == SensorReward().update(a)


def test_repeated_or_stale_image_cannot_keep_paying_visible_retention():
    p = packet()
    p.update(phase=1., forecast_valid=False, visual_valid=True, visual_velocity_valid=True,
             ball_capture_s=5., visual_dwell_s=.3)
    reward = SensorReward()
    reward.update(p)
    p.update(time_s=5.04, ball_capture_s=5.04, visual_dwell_s=.34)
    assert reward.update(p) > 0.
    p['time_s'] = 5.06
    assert reward.update(p) == 0.
    for t in (5.08, 5.10, 5.12, 5.14):
        p['time_s'] = t
        reward.update(p)
    p['visual_dwell_s'] = 3.
    assert reward.finish() <= 0.
    with pytest.raises(RuntimeError):
        reward.finish()


def test_threaded_episode_keeps_action_and_reward_ledger_causal(monkeypatch):
    import sys
    from types import SimpleNamespace
    actions = []
    def rollout(config, policy, output):
        for i in range(12):
            p = packet()
            p.update(time_s=5.+.02*i, ttc_s=.8-.04*i,
                     compact_error_rad=np.full(14, .1))
            actions.append(policy(p))
        return {'physical_catch_score': 1e100}
    monkeypatch.setitem(sys.modules, '_test_late_driver', SimpleNamespace(run_training_rollout=rollout))
    env = LateReachEnv({}, '_test_late_driver')
    try:
        obs, _ = env.reset()
        assert obs.shape == (OBS_SIZE,)
        total = 0.
        while True:
            _, reward, done, truncated, info = env.step(np.array([.25]))
            total += reward
            assert not truncated
            if done:
                break
        assert np.isclose(total, info['sensor_return'])
        assert info['reset_debit'] < 0.
        assert actions[:4] == [0.]*4
        assert actions[4:] == [.25]*8
        assert info['policy_calls'] == 8
    finally:
        env.close()


@pytest.mark.parametrize('sigma,velocity_valid', [([-1., 0., 0.], True),
                                               ([np.nan, 0., 0.], True),
                                               ([.01]*3, False)])
def test_terminal_retention_requires_valid_uncertainty_and_velocity(sigma, velocity_valid):
    p = packet()
    p.update(phase=1., forecast_valid=False, visual_valid=True,
             visual_velocity_valid=velocity_valid, ball_capture_s=5.,
             visual_dwell_s=3., ball_cup_sigma_m=sigma)
    reward = SensorReward()
    reward.update(p)
    assert reward.finish() <= 0.


def test_uncertain_history_cannot_transfer_dwell_into_one_good_frame():
    p = packet()
    p.update(phase=1., forecast_valid=False, visual_valid=True,
             visual_velocity_valid=True, visual_dwell_s=3., ball_cup_sigma_m=[.1]*3)
    reward = SensorReward()
    for i in range(30):
        p.update(time_s=5.+.04*i, ball_capture_s=5.+.04*i)
        reward.update(p)
    p.update(time_s=6.2, ball_capture_s=6.2, ball_cup_sigma_m=[.01]*3)
    assert reward.update(p) < .1
    assert reward.dwell == 0.
    assert not reward.paid.any()
    assert reward.finish() <= 0.


def test_fresh_visible_near_ball_is_not_worse_than_never_visible():
    visible, blind = SensorReward(), SensorReward()
    p = packet()
    p.update(phase=1., forecast_valid=False, visual_velocity_valid=False,
             visual_valid=True, ball_capture_s=5.)
    visible.update(p)
    p['visual_valid'] = False
    blind.update(p)
    visible.finish()
    blind.finish()
    assert visible.total > blind.total


def test_wrong_palm_form_cannot_accumulate_retention_milestone():
    reward = SensorReward()
    p = packet()
    p.update(phase=1., forecast_valid=False, visual_velocity_valid=True,
             visual_valid=True, palm_up_cos=[-.8, -.8])
    for i in range(50):
        p.update(time_s=5.+.02*i, ball_capture_s=5.+.02*i)
        reward.update(p)
    assert reward.dwell == 0.
    assert not reward.paid.any()


def test_actor_uses_camera_age_but_not_absolute_time():
    a = packet()
    a.update(visual_valid=True, ball_capture_s=4.96)
    b = deepcopy(a)
    b.update(time_s=105., ball_capture_s=104.96)
    assert np.allclose(observation(a), observation(b))
    b['ball_capture_s'] = 105.
    assert observation(a)[-1] > observation(b)[-1]


def test_fallen_robot_cannot_earn_retention_milestones():
    reward = SensorReward()
    p = packet()
    p.update(phase=1., forecast_valid=False, visual_velocity_valid=True,
             visual_valid=True, upright_cos=.7)
    for i in range(50):
        p.update(time_s=5.+.02*i, ball_capture_s=5.+.02*i)
        reward.update(p)
    assert reward.dwell == 0.
    assert not reward.paid.any()
