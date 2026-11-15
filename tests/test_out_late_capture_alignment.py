from collections import deque
from types import SimpleNamespace
import numpy as np
import mujoco
from g1_receiver.out_late_cup_capture import OutLateCupReach


def sample(t,angle=0.):
    return SimpleNamespace(timestamp=t,joint_position=np.zeros(0),imu_quat_wxyz=np.array([np.cos(angle/2),0.,0.,np.sin(angle/2)]))


def test_encoder_imu_interpolation_uses_only_observed_bracket():
    cue=OutLateCupReach.__new__(OutLateCupReach)
    cue.sensor_history=deque([sample(1.,0.),sample(1.02,.2)])
    aligned=cue.sampled_observation(1.01)
    assert np.allclose(aligned.imu_quat_wxyz,[np.cos(.05),0,0,np.sin(.05)])
    assert cue.sampled_observation(.99)is None
    assert cue.sampled_observation(1.03)is None
    cue.sensor_history=deque([sample(1.),sample(1.10)])
    assert cue.sampled_observation(1.05)is None


def test_recent_velocity_recovers_after_old_camera_gap():
    history=[(0.,0.,0.,0.),(1.,0.,0.,0.),(1.02,.04,0.,0.),(1.04,.08,0.,0.),(1.06,.12,0.,0.)]
    assert len(OutLateCupReach.recent(history))==4
    assert np.allclose(OutLateCupReach.velocity(history),[2.,0,0])


def test_slerp_antipodal_quaternions_do_not_invent_rotation():
    q=np.array([.5,.5,.5,.5])
    assert np.allclose(OutLateCupReach.slerp(q,-q,.5),q)


def test_fk_uses_capture_pose_age_and_separates_marker_yaw_variance():
    xml='''<mujoco><worldbody><body name="pelvis"><freejoint name="floating_base_joint"/><geom type="sphere" size=".05" mass="1"/>
    <body name="left_shoulder_pitch_link" pos="0 .1 .3"><body name="aero_left_palm" pos=".2 0 0"><geom name="aero_left_palm_collision_7" type="box" size=".007 .04 .013"/></body></body>
    <body name="right_shoulder_pitch_link" pos="0 -.1 .3"><body name="aero_right_palm" pos=".2 0 0"><geom name="aero_right_palm_collision_7" type="box" size=".007 .04 .013"/></body></body>
    </body></worldbody></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml);cue=OutLateCupReach.__new__(OutLateCupReach)
    cue.model=m;cue.mj=mujoco;cue.data=mujoco.MjData(m);cue.base=0;cue.qadr=np.array([],dtype=int);cue.offset=np.zeros(3)
    cue.sensor_history=deque([sample(1.,0.),sample(1.02,.2)])
    pose=lambda t,z:SimpleNamespace(timestamp_s=t,xy_m=np.zeros(2),z_m=z,z_std_m=.02,covariance=np.diag([.0004,.0009,.25]))
    cue.pose_samples=deque([pose(1.,.75),pose(1.015,9.)])
    cue.visual_capture_time=1.01;cue.cup_history=deque();cue.last_cup_capture=None
    center,R,cov,up=cue.fk(None,None,1.08)
    assert np.allclose(center,[(.2+.007)*np.cos(.1),(.2+.007)*np.sin(.1),1.05])
    assert np.allclose(np.diag(cov),[.000525,.001025,.000525])
    assert cue.fk_pose_timestamp==1.
    assert cue.cup_history[-1][0]==1.01
    assert np.allclose(cue.fk_heights,[0.,0.])
