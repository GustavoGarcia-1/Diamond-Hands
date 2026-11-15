"""Ballistic delivery and closure lead use surveyed estimates only."""

from collections import deque
from dataclasses import replace

import numpy as np
import pytest

from g1_receiver.localization import ReceiverPose
from g1_receiver.perception import BallObservation, GRAVITY_M_S2
from g1_receiver.short_throw import FreeBallRelease, plan_underhand_throw, predict_grip_timing

GRAVITY = np.array([0., 0., -GRAVITY_M_S2])
VELOCITY = np.array([.14, .02])
OFFSET = np.array([.44, -.02, .36])
INCOMING = np.array([-.48, .01, -.15])


def pose(timestamp_s=0., *, yaw=0.):
    return ReceiverPose(timestamp_s, "field", np.array([.10, -.03]) + timestamp_s * VELOCITY,
                        yaw, np.diag([.003**2, .003**2, .01**2]), "synthetic_fixture", z_m=.75)


def plan(**kwargs):
    return plan_underhand_throw(pose(yaw=kwargs.pop("yaw", 0.)), VELOCITY, release_time_s=0.,
                               flight_time_s=.30, intercept_body_xyz_m=OFFSET,
                               incoming_relative_body_mps=kwargs.pop("incoming", INCOMING), **kwargs)


def trajectory(throw, t):
    return np.array(throw.release_xyz_m) + t * np.array(throw.release_velocity_mps) + .5 * t**2 * GRAVITY


def samples(throw, stamps):
    return deque(BallObservation(t, "field", trajectory(throw, t), np.eye(3) * .002**2,
                                 "synthetic_fixture") for t in stamps)


def test_underhand_targets_position_and_incoming_velocity_after_receiver_translation():
    yaw = .37
    throw = plan(yaw=yaw, min_toward_receiver_travel_m=.09)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.], [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
    expected = np.r_[pose(.30).xy_m, .75] + rotation @ OFFSET
    assert trajectory(throw, .30) == pytest.approx(expected)
    assert np.array(throw.release_velocity_mps) + .30 * GRAVITY == pytest.approx(np.r_[VELOCITY, 0.] + rotation @ INCOMING)
    assert throw.release_velocity_mps[2] > 0
    assert throw.release_xyz_m[2] < throw.intercept_xyz_m[2]
    with pytest.raises(ValueError, match="far enough"):
        plan(incoming=(-.20, 0., -.15), min_toward_receiver_travel_m=.08)


def test_camera_estimate_predicts_relative_crossing_and_rejects_future_sample():
    throw = plan()
    observations = samples(throw, np.linspace(.0, .20, 6))
    prediction = predict_grip_timing(observations, pose(.20), VELOCITY, .20, allow_synthetic_fixture=True)
    assert prediction.close
    assert prediction.seconds_to_crossing == pytest.approx(.10)
    assert prediction.crossing_relative_xyz_m == pytest.approx(OFFSET)
    assert prediction.crossing_relative_velocity_mps == pytest.approx(INCOMING)
    assert predict_grip_timing(observations, pose(0.), VELOCITY, .20,
                               allow_synthetic_fixture=True).status == "no_fresh_receiver_pose"
    assert predict_grip_timing(observations, pose(.18), VELOCITY, .18,
                               allow_synthetic_fixture=True).status == "future_ball_observation"
    observations[-1] = replace(observations[-1], position_m=observations[-1].position_m + [0., .3, 0.])
    assert predict_grip_timing(observations, pose(.20), VELOCITY, .20,
                               allow_synthetic_fixture=True).status == "no_stable_crossing"


def test_release_uses_one_state_write_and_ball_flies_under_gravity():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(f'''
<mujoco>
  <option timestep=".001" gravity="0 0 {-GRAVITY_M_S2}" integrator="RK4"/>
  <worldbody><body name="football"><freejoint name="football_free"/>
    <geom type="ellipsoid" size=".142 .085 .085" mass=".42"/>
  </body></worldbody>
</mujoco>
''')
    data = mujoco.MjData(model)
    throw = plan()
    release = FreeBallRelease(model, data)
    release.apply(throw)
    for _ in range(300):
        mujoco.mj_step(model, data)
    assert data.qpos[:3] == pytest.approx(throw.intercept_xyz_m)
    assert data.qvel[:3] == pytest.approx(throw.intercept_velocity_mps)
    with pytest.raises(RuntimeError, match="already released"):
        release.apply(throw)
