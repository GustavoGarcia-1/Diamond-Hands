from dataclasses import replace

import numpy as np
import pytest

from g1_receiver.learning import (
    OBSERVATION_SIZE, RouteSensorState, SensorSGS, build_observation, route_reward,
)
from g1_receiver.localization import ReceiverPose


def state(t=1., progress=.2, pose_time=None, cross_track=.05):
    pose = ReceiverPose(t if pose_time is None else pose_time, "field", np.array([.2, .1]),
                        0., np.eye(3)*.001, "synthetic_fixture")
    return RouteSensorState(t, np.zeros(29), np.zeros(29), np.array([1., 0, 0, 0]),
                            np.zeros(3), pose, np.array([1., .5]), np.array([.3, 0]),
                            progress, cross_track, np.zeros(2))


def test_shared_actor_critic_vector_masks_stale_camera_without_losing_encoders():
    current = replace(state(), joint_position=np.full(29, .1))
    fresh = build_observation(current)
    assert fresh.shape == (OBSERVATION_SIZE,)
    assert fresh.dtype == np.float32
    stale = build_observation(replace(current, timestamp_s=1.2))
    np.testing.assert_allclose(stale[:29], .1)
    assert fresh[78] == 1
    assert stale[78] == 0
    np.testing.assert_array_equal(stale[65:77], 0)
    assert stale[77] == pytest.approx(.06)
    # Actor and critic call exactly the same public builder; no privileged builder exists.
    np.testing.assert_array_equal(build_observation(current), fresh)


def test_reward_uses_measured_progress_and_withholds_on_loss_or_reacquisition():
    before = state(t=1., progress=.1)
    after = state(t=1.1, progress=.13)
    result = route_reward(before, after, np.zeros(2), max_pose_age_s=.12)
    assert result["measured_progress_delta_m"] == pytest.approx(.03)
    assert result["tracking_proxy_success"]
    assert "not_catch" in result["reward_basis"]
    lost = replace(after, pose=None, progress_m=10.)
    missing = route_reward(before, lost, np.zeros(2))
    assert missing["measured_progress_delta_m"] == 0
    assert not missing["tracking_proxy_success"]
    assert missing["reward"] < 0
    reacquired = state(t=1.2, progress=11.)
    assert route_reward(lost, reacquired, np.zeros(2))["measured_progress_delta_m"] == 0


def test_reward_rejects_time_and_action_errors_and_does_not_reward_large_pose_jump():
    before = state(t=1., progress=.1)
    after = state(t=1.1, progress=100.)
    assert route_reward(before, after, np.zeros(2))["measured_progress_delta_m"] <= .100001
    with pytest.raises(ValueError, match="increasing"):
        route_reward(after, before, np.zeros(2))
    with pytest.raises(ValueError, match="normalized"):
        route_reward(before, after, [1.1, 0])
    tilted = replace(after, imu_quat_wxyz=np.array([np.sqrt(.5), np.sqrt(.5), 0, 0]))
    assert not route_reward(before, tilted, np.zeros(2))["tracking_proxy_success"]


def test_action_smoothness_uses_pre_transition_history_not_new_observation():
    before = state(t=1.)
    action = np.array([.5, -.5])
    after = replace(state(t=1.1), previous_action=action)
    changed = route_reward(before, after, action)["reward"]
    unchanged = route_reward(replace(before, previous_action=action), after, action)["reward"]
    assert unchanged-changed == pytest.approx(.005)


def test_contract_rejects_nonfinite_measurements_and_unknown_truth_fields():
    current = state()
    with pytest.raises(ValueError, match="finite"):
        replace(current, joint_velocity=np.full(29, np.nan))
    with pytest.raises(ValueError, match="normalized"):
        replace(current, imu_quat_wxyz=np.zeros(4))
    with pytest.raises(TypeError):
        replace(current, simulator_contacts=True)
    with pytest.raises(ValueError, match="declared sensor"):
        replace(current, pose=replace(current.pose, pose_provenance="sim_truth"))


def test_sgs_prior_and_failure_weighting_are_sensor_outcome_only():
    sampler = SensorSGS(["slant", "out"])
    np.testing.assert_allclose(sampler.probabilities(), [.5, .5])
    for _ in range(10):
        sampler.update("slant", True)
        sampler.update("out", False)
    probabilities = sampler.probabilities()
    assert 0 < probabilities[0] < probabilities[1] < 1
    assert probabilities.sum() == pytest.approx(1)
    assert sampler.snapshot()["basis"] == "sensor_route_tracking_proxy_not_catch"
    with pytest.raises(ValueError, match="boolean"):
        sampler.update("out", 1)
    with pytest.raises(ValueError):
        sampler.update("unknown", True)
