"""Contracts for the new reward boundary, not physical catch certificates."""
from types import SimpleNamespace
import numpy as np
import pytest
from g1_receiver.precontact_native_rl import ArrivalDeadline,PrecontactNativeAudit,project_residual,expected_alignment


def test_arrival_deadline_cannot_extend_or_reopen():
    deadline=ArrivalDeadline()
    assert deadline.update(1.,2.,.05)
    assert deadline.deadline==1.9
    deadline.update(1.2,4.,.01)
    assert deadline.deadline==1.9
    deadline.update(1.3,1.8,.05)
    assert deadline.deadline==pytest.approx(1.7)
    assert not deadline.update(1.7)
    assert not deadline.update(1.8,10.,.001)
    assert deadline.deadline==pytest.approx(1.7)


def test_delivered_residual_bounds_and_moving_range_infeasibility_are_explicit():
    r=project_residual(np.ones(14),np.zeros(14),np.zeros(14),np.tile([-1.,1.],(14,1)))
    assert np.allclose(r,.03)
    ranges=np.tile([-1.,1.],(14,1));nominal=np.full(14,.99)
    r=project_residual(np.ones(14),np.zeros(14),nominal,ranges)
    assert np.allclose(r,.01)
    with pytest.raises(ValueError,match='infeasible'):
        project_residual(np.zeros(14),np.full(14,.18),np.ones(14),ranges)


class Sensors:
    def __init__(self):
        self.reward=0.;self.dwell=0.;self.paid=np.zeros(2);self.potential=None
        self.form_acquired=False;self.valid=False;self.last_valid=None;self.upright=1.
        self.current_geometry={'world_to_receiver':np.eye(3),'cup_receiver':np.zeros(3)}
        self.detector=SimpleNamespace(detect=lambda frame,survey:SimpleNamespace(
            timestamp_s=frame.timestamp_s,frame_id='field',pose_provenance='synthetic_fixture',
            position_m=np.zeros(3),covariance_m2=np.eye(3)*1e-5))
    def update(self,obs,pose,frame,survey):
        # Deliberately emulate enormous obsolete possession credit.
        self.reward=1e9;self.dwell=100.;self.paid[:]=1.;self.form_acquired=True
        self.valid=frame is not None and pose is not None
        if self.valid:self.last_valid=frame.timestamp_s
    def vector(self,*args):
        x=np.zeros(169,dtype=np.float32);x[76]=100.;x[159:161]=1.;x[-8:]=args[-1];return x


class Tracker:
    def __init__(self):self.stamp=None;self.arrival=.20
    def add(self,ball):self.stamp=ball.timestamp_s;return True
    def intercept(self,*args):
        return SimpleNamespace(timestamp_s=self.arrival,time_std_s=.01,last_inlier_timestamp_s=self.stamp,
            velocity_mps=np.array([3.,0.,-1.]),position_m=np.zeros(3),covariance_m2=np.eye(3)*1e-5)


def audit(policy=None):
    a=PrecontactNativeAudit({'play':{'field_catch_window_m':[1.,0.,1.],
                                    'field_approach_axis_xy':[1.,0.]}},policy)
    a.sensors=Sensors();a.tracker=Tracker();a.calibration=None;a.arms=np.arange(14)
    a.ranges=np.tile([-1.,1.],(14,1))
    a.form=SimpleNamespace(evaluate=lambda *args:{'penalty_per_s':0.,
                           'shoulder_midpoint_height_above_pelvis_m':0.})
    return a


def packet(now,capture=None):
    obs=SimpleNamespace(timestamp=now,joint_position=np.zeros(29),joint_velocity=np.zeros(29),
                        imu_quat_wxyz=np.array([1.,0.,0.,0.]),imu_gyro=np.zeros(3))
    stamp=now if capture is None else capture
    frame=SimpleNamespace(timestamp_s=stamp)
    pose=SimpleNamespace(timestamp_s=stamp,z_m=0.,z_std_m=.001,xy_m=np.zeros(2),
         covariance=np.eye(2)*1e-6,frame_id='field',pose_provenance='synthetic_fixture',
         fresh=lambda now,age:-1e-8<=now-stamp<=age)
    return obs,pose,frame,SimpleNamespace()


def call(a,now,capture=None):
    native=np.linspace(.01,.14,14);target=np.zeros(29)
    upper,result=a(*packet(now,capture),target,native,np.zeros(14),True)
    assert np.array_equal(result,native)
    assert np.array_equal(upper[14:],target[14:])
    return upper


def test_old_possession_reward_never_enters_ledger_and_zero_is_exact(tmp_path):
    a=audit()
    for t in [0.,.02,.04]:
        obs=packet(t)[0];target=np.zeros(29);native=np.arange(14,dtype=float)
        upper,result=a(obs,None,None,None,target,native,np.zeros(14),True)
        assert np.array_equal(upper,target)and np.array_equal(result,native)
    assert a.finish()==0.
    summary=a.save(tmp_path,0.)
    assert summary['sensor_return']==summary['reset_sensor_reward']==0.
    assert summary['terminal_sensor_bonus']==0.
    for row in a.rows:
        assert row['raw'][76]==0 and np.all(row['raw'][159:161]==0)


def test_unique_capture_credit_cached_actor_signal_and_absorbing_cutoff():
    decisions=[]
    a=audit(lambda vector:decisions.append(vector.copy())or np.full(14,.2))
    call(a,0.);call(a,.02);credit=a.positive_credit
    assert credit>0
    call(a,.04,capture=.02)
    assert a.positive_credit==credit
    assert a.rows[-1]['intercept_valid']  # opposite camera/policy phase remains usable
    assert a.rows[-1]['raw'][161+2]==1
    call(a,.06);call(a,.08);call(a,.10);call(a,.12);call(a,.14);call(a,.16)
    before=a.actual.copy();count=len(decisions)
    call(a,.18)
    assert a.arrival.closed and np.array_equal(a.actual,before)
    a.tracker.arrival=2.0
    call(a,.20);call(a,.22)
    assert len(decisions)==count
    assert np.array_equal(a.actual,before)
    assert a.positive_after_deadline==0.
    assert all(row['positive_alignment_credit']==0 for row in a.camera_rows if row.get('deadline_closed'))
    assert a.finish()==0.
    with pytest.raises(RuntimeError):a.finish()


def test_future_frames_receive_no_positive_credit():
    a=audit();call(a,0.,capture=.02);call(a,.02,capture=.04)
    assert a.positive_credit==0. and a.arrival.deadline is None


def test_missing_frame_keeps_only_fresh_cached_actor_signal_and_never_repays():
    a=audit();call(a,0.);call(a,.02);credit=a.positive_credit
    original=a.sensors.update
    def cached_update(obs,pose,frame,survey):
        if frame is not None:original(obs,pose,frame,survey)
    a.sensors.update=cached_update
    obs=packet(.04)[0]
    upper,native=a(obs,None,None,None,np.zeros(29),np.zeros(14),np.zeros(14),True)
    assert a.positive_credit==credit
    assert a.rows[-1]['intercept_valid']
    obs=packet(.12)[0]
    # Continue in20ms increments to exercise freshness expiry without clock gaps.
    for t in [.06,.08,.10,.12]:
        a(packet(t)[0],None,None,None,np.zeros(29),np.zeros(14),np.zeros(14),True)
    assert not a.rows[-1]['intercept_valid']


def test_tracker_warms_before_enabled_and_old_possession_covariance_cannot_block_reward():
    a=audit();original=a.sensors.update
    def invalid_old_cup(*args):
        original(*args);a.sensors.valid=False
    a.sensors.update=invalid_old_cup
    for t in [0.,.02]:
        obs,pose,frame,survey=packet(t);pose.covariance=np.eye(2)*.01
        a(obs,pose,frame,survey,np.zeros(29),np.zeros(14),np.zeros(14),False)
    assert a.tracker.stamp==.02
    assert a.positive_credit==0.
    for t in [.04,.06]:
        obs,pose,frame,survey=packet(t);pose.covariance=np.eye(2)*.01
        a(obs,pose,frame,survey,np.zeros(29),np.zeros(14),np.zeros(14),True)
    assert a.positive_credit>0.
    assert a.arrival.deadline is not None
    assert a.camera_rows[-1]['sigma_m'][0]>.06
    assert a.camera_rows[-1]['alignment_proxy']<1.


def test_expected_alignment_is_bounded_and_uncertainty_attenuates_exact_alignment():
    assert expected_alignment(np.zeros(3),np.zeros(3))==1.
    assert 0.<expected_alignment(np.zeros(3),np.ones(3)*.1)<1.
    assert expected_alignment(np.ones(3)*10,np.zeros(3))<1e-10


def test_declared_alignment_weight_scales_only_positive_precontact_credit(monkeypatch):
    import g1_receiver.precontact_native_rl as module
    monkeypatch.setattr(module,'ALIGNMENT_WEIGHT',1.)
    a=audit();call(a,0.);call(a,.02);unweighted=a.positive_credit
    monkeypatch.setattr(module,'ALIGNMENT_WEIGHT',25.)
    b=audit();call(b,0.);call(b,.02)
    assert b.positive_credit==pytest.approx(25*unweighted)
    assert b.camera_rows[-1]['positive_alignment_credit']==pytest.approx(
        25*b.camera_rows[-1]['unweighted_alignment_credit'])
