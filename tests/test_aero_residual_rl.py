"""Behavioral checks for causal residual learning, independent of contact truth."""
from types import SimpleNamespace
import numpy as np
import pytest
from g1_receiver.aero_residual_rl import CupSensors, bounded_residual, OBS_SIZE, scale_observation

def test_unit_scaling_preserves_flags_and_returns_a_copy():
    raw=np.zeros(OBS_SIZE,dtype=np.float32)
    raw[29]=10.;raw[64]=.25;raw[74:76]=1.;raw[93]=.18
    scaled=scale_observation(raw)
    np.testing.assert_allclose(scaled[[29,64,74,75,93]],1.)
    assert raw[29]==10. and not np.shares_memory(raw,scaled)

@pytest.fixture
def sensors():
    import mujoco
    body='<body name="pelvis"><freejoint name="floating_base_joint"/><geom size=".04" mass="1"/>'
    for i in range(29):body+=f'<body><joint name="j{i}"/><geom size=".01" mass=".01"/>'
    body+='</body>'*29
    body+='<body name="aero_left_palm" pos=".4 .08 .3"/><body name="aero_right_palm" pos=".4 -.08 .3"/></body>'
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+body+'</worldbody></mujoco>')
    obj=CupSensors(model,[f'j{i}' for i in range(29)])
    return obj

def packet(t, ball=None):
    obs=SimpleNamespace(timestamp=t,joint_position=np.zeros(29),joint_velocity=np.zeros(29),imu_quat_wxyz=np.array([1.,0,0,0]),imu_gyro=np.zeros(3))
    pose=SimpleNamespace(timestamp_s=t,z_m=0.,z_std_m=.001,xy_m=np.zeros(2),covariance=np.eye(3)*1e-6,pose_provenance='synthetic_fixture',frame_id='field',fresh=lambda now,age:0<=now-t<=age)
    frame=SimpleNamespace(timestamp_s=t)
    if ball is None:ball=SimpleNamespace(position_m=np.array([.5,0,.38]),covariance_m2=np.eye(3)*1e-6,pose_provenance='synthetic_fixture',frame_id='field')
    return obs,pose,frame,SimpleNamespace(),ball

def deliver(sensors,t,ball=None):
    obs,pose,frame,survey,b=packet(t,ball)
    sensors.detector=SimpleNamespace(detect=lambda *_:b)
    sensors.update(obs,pose,frame,survey)
    return obs

def test_action_rate_bounds_and_invalid_values():
    value=bounded_residual(np.ones(16),np.zeros(16))
    np.testing.assert_allclose(value[:14],.03);np.testing.assert_allclose(value[14:],.02)
    for _ in range(100):value=bounded_residual(np.ones(16),value)
    np.testing.assert_allclose(value[:14],.18);np.testing.assert_allclose(value[14:],.15)
    with pytest.raises(ValueError):bounded_residual(np.full(16,np.nan),value)

def test_duplicate_capture_does_not_pay_or_advance_dwell(sensors):
    obs=None
    for t in np.arange(0.,.44,.04):obs=deliver(sensors,float(t))
    assert sensors.dwell>.2
    old=(sensors.dwell,sensors.reward,len(sensors.history))
    obs,pose,frame,survey,b=packet(.40);obs.timestamp=.42
    sensors.update(obs,pose,frame,survey)
    assert (sensors.dwell,sensors.reward,len(sensors.history))==old
    assert sensors.vector(obs,np.zeros(16),np.zeros(16)).shape==(OBS_SIZE,)

def test_occlusion_and_high_speed_reset_hold(sensors):
    for t in np.arange(0.,.44,.04):deliver(sensors,float(t))
    assert sensors.dwell>.2
    obs,pose,frame,survey,_=packet(.44)
    sensors.detector=SimpleNamespace(detect=lambda *_:None)
    sensors.update(obs,pose,frame,survey)
    assert sensors.dwell==0 and not sensors.valid
    for i,t in enumerate(np.arange(.48,.68,.04)):
        b=SimpleNamespace(position_m=np.array([.5+float(i)*.08,0,.38]),covariance_m2=np.eye(3)*1e-6,pose_provenance='synthetic_fixture',frame_id='field')
        deliver(sensors,float(t),b)
    assert sensors.dwell==0

def test_fk_state_is_reset_and_uncertainty_does_not_enlarge_gate(sensors):
    deliver(sensors,0.)
    baseline=sensors.e.copy()
    sensors.data.qpos[:]=99. # not a valid sensor source; update must discard it
    deliver(sensors,.04)
    np.testing.assert_allclose(sensors.e,baseline)
    b=SimpleNamespace(position_m=np.array([.5,0,.38]),covariance_m2=np.eye(3)*.1,pose_provenance='synthetic_fixture',frame_id='field')
    deliver(sensors,.08,b)
    assert not sensors.valid and sensors.dwell==0

def test_milestones_paid_only_once_and_endpoint_is_current(sensors):
    for t in np.arange(0.,1.04,.04):deliver(sensors,float(t))
    assert sensors.paid[0]==1
    sensors.invalidate()
    assert sensors.dwell==0 and sensors.paid[0]==1
    for t in np.arange(1.04,2.12,.04):deliver(sensors,float(t))
    assert sensors.paid[0]==1 and sensors.paid[1]==0

def test_corrected_reward_visibility_loss_settles_once_without_windfall(sensors):
    sensors.corrected_potential_reward=True
    for i,x in enumerate((.9,.7,.5)):
        b=SimpleNamespace(position_m=np.array([x,0,.38]),covariance_m2=np.eye(3)*1e-6,pose_provenance='synthetic_fixture',frame_id='field')
        deliver(sensors,i*.04,b)
    assert sensors.reward>0  # approach shaping has accrued, not a hold bonus
    phi=sensors.potential;before=sensors.reward
    obs,pose,frame,survey,_=packet(.12)
    sensors.detector=SimpleNamespace(detect=lambda *_:None)
    sensors.update(obs,pose,frame,survey)
    assert sensors.reward==pytest.approx(before-.02-.15*phi)
    assert sensors.reward<0  # hiding the close ball cannot preserve approach profit
    after=sensors.reward
    obs,pose,frame,survey,_=packet(.16)
    sensors.update(obs,pose,frame,survey)
    assert sensors.reward==pytest.approx(after-.02)  # visibility cost, no second potential debit
    sensors.finish()
    assert sensors.reward==pytest.approx(after-.02)

def test_corrected_reward_stale_and_terminal_boundaries_settle_once(sensors):
    sensors.corrected_potential_reward=True
    obs=deliver(sensors,0.)
    assert sensors.potential>0
    before=sensors.reward;phi=sensors.potential
    obs.timestamp=.12
    sensors.update(obs,None,None,None)  # no new capture; old measurement expires
    assert sensors.reward==pytest.approx(before-.15*phi)
    after=sensors.reward
    obs.timestamp=.14;sensors.update(obs,None,None,None);sensors.finish()
    assert sensors.reward==pytest.approx(after)
    deliver(sensors,.16)  # a fresh segment initializes rather than paying reappearance
    before=sensors.reward;phi=sensors.potential
    sensors.finish()
    assert sensors.reward==pytest.approx(before-.15*phi)
    after=sensors.reward;sensors.finish()
    assert sensors.reward==after

def test_legacy_reward_remains_default_and_terminal_does_not_change_it(sensors):
    assert not sensors.corrected_potential_reward
    deliver(sensors,0.);before=sensors.reward
    sensors.finish()
    assert sensors.reward==before
    sensors.invalidate()
    assert sensors.reward==pytest.approx(before-.02)

def test_reset_reward_is_separate_from_sum_of_gym_step_rewards():
    from g1_receiver.aero_residual_rl import EpisodeReturnLedger
    ledger=EpisodeReturnLedger()
    ledger.record(-.02,reset=True)
    delivered=(.1,-.04,.025)
    for reward in delivered:ledger.record(reward)
    assert ledger.step_return==pytest.approx(sum(delivered))
    assert ledger.reset_reward==pytest.approx(-.02)
    assert ledger.step_return+ledger.reset_reward==pytest.approx(sum(delivered)-.02)
