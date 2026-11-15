import numpy as np
import pytest
from g1_receiver.lower_cup_absorption import bounded_joint_step

def test_joint_step_enforces_amplitude_and_rate_without_component_direction_change():
 desired=np.linspace(-1,1,14);previous=np.zeros(14);out=bounded_joint_step(desired,previous,.02)
 assert abs(out).max()<=.0300000001
 assert np.allclose(out/desired,.03,atol=1e-12)
 for _ in range(30):out=bounded_joint_step(desired,out,.02)
 assert abs(out).max()<=.1800000001
 assert np.allclose(out/desired,.18)
@pytest.mark.parametrize('dt',[0.,-.02,.1,np.nan])
def test_joint_step_rejects_bad_clock(dt):
 with pytest.raises(ValueError):bounded_joint_step(np.zeros(14),np.zeros(14),dt)
def test_recovery_remains_rate_limited():
 p=np.full(14,.18);out=bounded_joint_step(np.zeros(14),p,.02);assert np.allclose(out,.15)

def _armed_pulse(last_capture=5.08):
 from g1_receiver.lower_cup_absorption import AnticipatoryCupGive
 c=AnticipatoryCupGive({'play':{'field_catch_window_m':[6,0,1.2],'field_approach_axis_xy':[1,0]}},gain=.1,lead_s=.15,pulse_lead_s=.025)
 c._fk=lambda obs:(np.array([0.,0.,1.]),np.eye(12,14))
 c.arms=np.arange(14);c.ranges=np.tile([-3.,3.],(14,1))
 c.plan=dict(contact_s=5.2,speed=.2,normal=np.array([0.,0.,1.]),forecast_s=5.08,last_capture_s=last_capture,time_std_s=.008)
 return c

def test_latched_rgbd_forecast_waits_until_near_arrival_despite_occlusion():
 from types import SimpleNamespace
 c=_armed_pulse()
 for t in np.arange(5.08,5.161,.02):
  target,_=c(SimpleNamespace(timestamp=t),None,None,None,np.zeros(29),0.,True)
  assert np.array_equal(target,np.zeros(29))
 target,_=c(SimpleNamespace(timestamp=5.18),None,None,None,np.zeros(29),0.,True)
 assert np.linalg.norm(target)>0
 assert c.rows[-1]['phase']=='anticipatory_give'
 assert abs(target).max()<=.03

def test_stale_forecast_cancels_before_scheduled_pulse():
 from types import SimpleNamespace
 c=_armed_pulse(last_capture=5.)
 target,_=c(SimpleNamespace(timestamp=5.18),None,None,None,np.zeros(29),0.,True)
 assert c.plan is None
 assert np.array_equal(target,np.zeros(29))
