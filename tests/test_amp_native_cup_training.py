from types import SimpleNamespace
import numpy as np
import pytest
from g1_receiver.amp_native_cup_training import leg_observation, MarkerSensors, sensor_reward
from g1_receiver.observable import project_gravity
from g1_receiver.football_routes import imu_yaw

def test_canonical_vector_matches_preextraction_math_and_ignores_truth_fields():
    rng=np.random.default_rng(173)
    for valid in (False,True):
        obs=SimpleNamespace(timestamp=3.14,joint_position=rng.normal(size=29),joint_velocity=rng.normal(size=29),imu_quat_wxyz=np.array([np.cos(.3),0.,0.,np.sin(.3)]),imu_gyro=rng.normal(size=3))
        sensors=MarkerSensors();sensors.pose_valid=valid;sensors.velocity_valid=valid
        if valid:sensors.pose=SimpleNamespace(timestamp_s=3.10,xy_m=np.array([5.,.2]))
        sensors.velocity=np.array([1.8,-.1]);offset=rng.uniform(-.25,.25,15);drive=np.array([2.,0.,-.3]);stage=.72
        xy,velocity,age=sensors.fields(obs.timestamp);yaw=imu_yaw(obs.imu_quat_wxyz)
        legacy=np.r_[obs.joint_position/2.,obs.joint_velocity/10.,project_gravity(obs.imu_quat_wxyz),obs.imu_gyro/5.,np.sin(yaw),np.cos(yaw),xy/[16.,2.],velocity/4.,age/.14,float(valid),float(valid),drive/[2.,.5,.75],obs.timestamp/7.,stage,offset/.25]
        assert np.array_equal(leg_observation(obs,sensors,offset,drive,stage),np.clip(legacy,-10.,10.).astype(np.float32))
        obs.base_pose_wxyz=np.full(7,999.);obs.ball_position_world=np.full(3,999.)
        assert np.array_equal(leg_observation(obs,sensors,offset,drive,stage),np.clip(legacy,-10.,10.).astype(np.float32))

def test_invalid_pose_cannot_earn_motion_or_path_reward():
    obs=SimpleNamespace(imu_quat_wxyz=np.array([1.,0.,0.,0.]))
    sensors=MarkerSensors();sensors.velocity=np.array([2.,0.]);sensors.pose=SimpleNamespace(xy_m=np.zeros(2))
    reward,parts=sensor_reward(obs,sensors,np.zeros(15),np.zeros(15))
    assert np.isfinite(reward)
    assert parts['speed_reward']==parts['stem_reward']==0.

def test_default_action_math_exact_legacy():
    from g1_receiver.amp_native_cup_training import bounded_leg_action
    rng=np.random.default_rng(173);previous=np.zeros(15)
    for step in range(100):
        action=rng.uniform(-2.,2.,15)
        expected=np.clip(np.clip(action,-1.,1.)*.25,previous-.04,previous+.04)
        actual=bounded_leg_action(action,previous,step*.02)
        assert np.array_equal(actual,expected)
        previous=actual

def test_gate_limit_and_rate_boundary():
    from g1_receiver.amp_native_cup_training import bounded_leg_action
    config=dict(residual_start_s=1.6,limit_rad=.12,rate_rad_s=1.)
    previous=np.zeros(15)
    for t in (0.,.8,1.58):
        previous=bounded_leg_action(np.ones(15),previous,t,**config)
        assert np.array_equal(previous,np.zeros(15))
    first=bounded_leg_action(np.ones(15),previous,1.6,**config)
    assert np.array_equal(first,np.full(15,.02))
    previous=first
    for i in range(20):
        current=bounded_leg_action(np.ones(15),previous,1.62+i*.02,**config)
        assert abs(current).max()<=.12
        assert abs(current-previous).max()<=.020000000000001
        previous=current
    assert np.array_equal(previous,np.full(15,.12))
    reverse=bounded_leg_action(-np.ones(15),previous,2.1,**config)
    assert np.allclose(reverse,.10,rtol=0,atol=1e-15)

def test_model_bounds_legacy_and_new_metadata_validated():
    from g1_receiver.amp_native_cup_training import model_residual_config
    assert model_residual_config(SimpleNamespace())==dict(residual_start_s=0.,limit_rad=.25,rate_rad_s=2.)
    model=SimpleNamespace(receiver_residual_start_s=1.6,receiver_limit_rad=.12,receiver_rate_rad_s=1.)
    assert model_residual_config(model)==dict(residual_start_s=1.6,limit_rad=.12,rate_rad_s=1.)
    for name,value in [('receiver_limit_rad',.3),('receiver_rate_rad_s',2.1),('receiver_residual_start_s',float('nan'))]:
        model=SimpleNamespace(**{name:value})
        with pytest.raises(ValueError):model_residual_config(model)

def test_runtime_adapter_enforces_same_gate_rate_and_preserves_upper_targets():
    from g1_receiver.amp_native_leg_residual import NativeCupLegResidual
    from g1_receiver._vendor.g1_runtime.pool.contracts import ActuatorCommand
    adapter=NativeCupLegResidual.__new__(NativeCupLegResidual)
    adapter.sensors=MarkerSensors();adapter.offset=np.zeros(15);adapter.lower=np.arange(15)
    adapter.ranges=np.tile([-10.,10.],(29,1));adapter.rows=[];adapter.observations=[]
    adapter.residual_config=dict(residual_start_s=1.6,limit_rad=.12,rate_rad_s=1.)
    adapter.policy=SimpleNamespace(predict=lambda vector,deterministic:(np.ones(15),None))
    command=ActuatorCommand(np.full(29,.1),np.ones(29),np.ones(29),np.zeros(29),'test')
    obs=SimpleNamespace(timestamp=1.58,joint_position=np.zeros(29),joint_velocity=np.zeros(29),imu_quat_wxyz=np.array([1.,0.,0.,0.]),imu_gyro=np.zeros(3))
    before=adapter.apply(command,obs,None,np.array([2.,0.,0.]),0.)
    assert np.array_equal(before.target_position,command.target_position)
    obs.timestamp=1.6
    after=adapter.apply(command,obs,None,np.array([2.,0.,0.]),0.)
    assert np.allclose(after.target_position[:15],.12,rtol=0,atol=1e-15)
    assert np.array_equal(after.target_position[15:],command.target_position[15:])
    assert np.array_equal(after.kp,command.kp)
