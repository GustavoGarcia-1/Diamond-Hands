"""Sensor/frame invariants on the actual static Aero calibration model."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import numpy as np
import pytest
import mujoco
from g1_receiver.aero import make_aero_scene
from g1_receiver.runtime import DEFAULT_RUNTIME_ROOT
from g1_receiver._vendor.g1_runtime.pool.wbc import BODY_NAMES
from g1_receiver.native_cup_residual_rl import NativeCupSensors,scale_observation,bounded_native_residual
from g1_receiver.lower_cup_grip import NativeHandCalibration,OPEN
from g1_receiver.perception import CameraIntrinsics,RGBDFrame,RigidPose
from g1_receiver.localization import ReceiverPose

@pytest.fixture
def fixture():
 source=Path(os.environ.get('AERO_SOURCE','/tmp/g1-aero-source'));pose=Path(__file__).parent/'fixtures/native_cup_calibration.json'
 if not source.exists()or not pose.exists():pytest.skip('official static Aero calibration assets required')
 model=mujoco.MjModel.from_xml_string(make_aero_scene(DEFAULT_RUNTIME_ROOT,source));q=np.array([model.qpos0[model.joint(n).qposadr[0]]for n in BODY_NAMES]);row=json.loads(pose.read_text())
 for name,value in zip(row['arm_order'],row['arm_targets_rad']):q[list(BODY_NAMES).index(name)]=value
 return model,q

def packet(sensor,q,stamp,yaw):
 imu0=np.array([.99,.03,.08,.05]);imu0/=np.linalg.norm(imu0);rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)]));rotation=rotation.reshape(3,3);imu=np.zeros(4);mujoco.mju_mulQuat(imu,np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)]),imu0)
 g=sensor.geometry(q,imu0);base=np.array([.2,-.1,.75]);target=base+g['world_to_receiver'].T@(g['cup_receiver']+[.01,-.005,.005]);optical=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]]);camera_position=target-optical@np.array([0,0,3.075]);pixels=np.zeros((96,128,3),np.uint8);depth=np.full((96,128),np.inf);yy,xx=np.indices(depth.shape);mask=(xx-64)**2+(yy-48)**2<=9;pixels[mask]=[184,74,23];depth[mask]=3.;frame=RGBDFrame(stamp,'optical',pixels,depth,CameraIntrinsics(100,100,64,48));survey=RigidPose(stamp,'optical','field',rotation@optical,rotation@camera_position,'synthetic_fixture');cov=np.diag([.0001,.0002,.0003]);transform=np.eye(3);transform[:2,:2]=rotation[:2,:2];pose=ReceiverPose(stamp,'field',(rotation@base)[:2],yaw,transform@cov@transform.T,'synthetic_fixture',base[2],.01);obs=SimpleNamespace(timestamp=stamp,joint_position=q,joint_velocity=np.zeros(29),imu_quat_wxyz=imu,imu_gyro=np.zeros(3));return obs,pose,frame,survey

def test_rgbd_full_observation_heading_equivariance_and_invalid_debit(fixture):
 model,q=fixture;a=NativeCupSensors(model,BODY_NAMES);b=NativeCupSensors(model,BODY_NAMES);cal=NativeHandCalibration.from_model(model);native=np.tile(OPEN,2)
 for i in range(12):
  t=1.+i*.04;pa=packet(a,q,t,0);pb=packet(b,q,t,.73);a.update(*pa);b.update(*pb)
  va=a.vector(pa[0],np.zeros(20),np.zeros(20),native,native,cal,np.zeros(8));vb=b.vector(pb[0],np.zeros(20),np.zeros(20),native,native,cal,np.zeros(8));np.testing.assert_allclose(va,vb,atol=2e-6,rtol=0);np.testing.assert_allclose(scale_observation(va),scale_observation(vb),atol=2e-6,rtol=0)
 assert a.valid and b.valid
 assert a.reward==pytest.approx(b.reward,abs=1e-10)
 before=(a.dwell,a.reward,a.capture_count);a.update(*pa);assert (a.dwell,a.reward,a.capture_count)==before
 obs,pose,frame,survey=packet(a,q,1.5,0);bad=RGBDFrame(frame.timestamp_s,frame.frame_id,np.zeros_like(frame.rgb),frame.depth_m,frame.intrinsics);reward=a.reward;a.update(obs,pose,bad,survey);assert not a.valid and a.dwell==0 and a.potential is None and a.reward<=reward-.02
 before=a.reward;a.update(obs,pose,bad,survey);assert a.reward==before

def test_native_twenty_action_rate_and_zero():
 np.testing.assert_array_equal(bounded_native_residual(np.zeros(20),np.zeros(20)),np.zeros(20));x=bounded_native_residual(np.ones(20),np.zeros(20));np.testing.assert_allclose(x[:14],.03);np.testing.assert_allclose(x[14:],.02)
 with pytest.raises(ValueError):bounded_native_residual(np.zeros(16),np.zeros(20))

def test_native_callback_applies_all_six_groups_and_preserves_zero(fixture):
 from g1_receiver.native_cup_residual_rl import NativeCupAudit
 from g1_receiver.observable import ARM_JOINTS
 model,q=fixture
 profile={'play':{'field_catch_window_m':[6.5,0,1.2],'field_approach_axis_xy':[1,0]}}
 native=np.tile(OPEN,2)
 zero=NativeCupAudit(profile);zero.bind(model,BODY_NAMES)
 unchanged_q,unchanged_hand=zero(*packet(zero.sensors,q,1.,0),q,native,native,True)
 np.testing.assert_array_equal(unchanged_q,q);np.testing.assert_array_equal(unchanged_hand,native)
 active=NativeCupAudit(profile,policy=lambda obs:np.ones(20));active.bind(model,BODY_NAMES)
 changed_q,changed_hand=active(*packet(active.sensors,q,1.,0),q,native,native,True)
 indices=np.array([list(BODY_NAMES).index(n)for n in ARM_JOINTS]);other=np.setdiff1d(np.arange(29),indices)
 np.testing.assert_array_equal(changed_q[other],q[other]);np.testing.assert_allclose(changed_q[indices]-q[indices],.03,atol=1e-12)
 np.testing.assert_allclose(active.calibration.fractions(changed_hand),np.full(14,.02),atol=1e-12)
 assert np.all(changed_hand!=native)
