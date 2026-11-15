import numpy as np
import pytest
from g1_receiver.lower_cup_grip import NativeHandCalibration,NativeCupGrip,legacy_targets,OPEN,LEGACY_CLOSE

def calibration():
 return NativeHandCalibration(np.tile([[.05852,.110387]]*4+[[0,1.7453],[.026152,.038389],[.081568,.112138]],(2,1)),np.arange(14))
def test_legacy_targets_preserve_bilateral_scalar_mapping():
 for closure in (0.,.2,1.,np.array([.65,.8])):
  expected=(OPEN+np.broadcast_to(np.clip(closure,0,1),(2,))[:,None]*(LEGACY_CLOSE-OPEN)).ravel()
  assert np.array_equal(legacy_targets(closure),expected)
def test_native_fraction_zero_and_one_respect_native_endpoint_units():
 c=calibration();assert np.array_equal(c.targets(np.zeros(7)),np.tile(OPEN,2))
 t=c.targets(np.ones(7));assert np.all(t>=c.ranges[:,0])and np.all(t<=c.ranges[:,1]);assert t[0]==.05852 and t[4]==1.7453
@pytest.mark.parametrize('fraction',[[-.01]*7,[1.01]*7,[np.nan]*7,[0.]*6])
def test_reject_invalid_native_fraction(fraction):
 with pytest.raises(ValueError):calibration().targets(fraction)
def test_visual_gate_delays_digits_independently_without_contact_input():
 c=calibration();g=NativeCupGrip(c,dict(preload=[0]*7,capture=[1]*7,digit_delay_s=[.1,0,0,0,0,0,0],ramp_s=.2));e=np.tile(OPEN,2)
 assert np.array_equal(g.targets(1.,.2,None,None,e),legacy_targets(.2))
 assert np.array_equal(g.targets(2.,.2,1.,None,e),c.targets(np.zeros(7)))
 t=g.targets(3.1,.2,1.,3.,e);f=c.fractions(t);assert abs(f[0])<1e-12 and np.isclose(f[1],.5)
def test_encoder_packet_is_not_a_finger_posture_or_contact_proxy():
 g=NativeCupGrip(calibration(),dict(preload=[.1]*7,capture=[.8]*7));a=g.targets(3.1,.2,1.,3.,np.zeros(14));b=g.targets(3.1,.2,1.,3.,np.ones(14));assert np.array_equal(a,b)
 with pytest.raises(ValueError):g.targets(3.1,.2,1.,3.,np.zeros(29))

def test_observed_flight_requires_three_fresh_consistent_moving_captures():
 from g1_receiver.lower_cup_grip_observed import ObservedNativeCupGrip
 g=ObservedNativeCupGrip(calibration(),dict(preload=[.2]*7,capture=[.8]*7,preload_source='rgbd_flight'));e=np.zeros(14)
 g.observed_targets(1.04,None,None,None,[[1.,0,0,0],[1.02,.06,0,0]],e);assert g.activation_at is None
 g.observed_targets(1.04,None,None,None,[[1.,0,0,0],[1.02,.06,0,0],[1.04,.12,0,0]],e);assert g.activation_at==1.04
@pytest.mark.parametrize('history',[
 [[1.,0,0,0],[1.02,0,0,0],[1.04,0,0,0]],
 [[0.,0,0,0],[.02,.06,0,0],[.04,.12,0,0]],
 [[1.,0,0,0],[1.02,.06,0,0],[1.04,0,0,0]],
])
def test_observed_activation_rejects_stationary_stale_or_reversing_ball(history):
 from g1_receiver.lower_cup_grip_observed import ObservedNativeCupGrip
 g=ObservedNativeCupGrip(calibration(),dict(preload_source='rgbd_flight'));g.observed_targets(1.04,None,None,None,history,np.zeros(14));assert g.activation_at is None

def test_route_ready_activation_needs_no_machine_event():
 from g1_receiver.lower_cup_grip_observed import ObservedNativeCupGrip
 g=ObservedNativeCupGrip(calibration(),dict(preload_source='route_ready'));g.observed_targets(2.4,None,None,2.38,[],np.zeros(14));assert g.activation_at==2.4
