"""A surveyed throw must produce the same gate decision when field axes rotate."""
from types import SimpleNamespace
import numpy as np
import pytest
from g1_receiver.pass_control import FieldPassGrip, PassGrip

@pytest.mark.parametrize('angle',[0.,np.pi/2,2.2])
def test_camera_gate_rotates_with_surveyed_throw(angle):
    axis=np.array([np.cos(angle),np.sin(angle)])
    target=np.r_[6*axis,1.]
    grip=FieldPassGrip(PassGrip(lead_s=.15),target,approach_axis_xy=axis)
    obs=SimpleNamespace(position_m=target-np.r_[.4*axis,0.],frame_id='field',timestamp_s=1.)
    crossing=SimpleNamespace(position_m=target,velocity_mps=np.r_[4*axis,-2.],
        timestamp_s=1.1,last_inlier_timestamp_s=1.,time_std_s=.01)
    grip.detector=SimpleNamespace(detect=lambda *a:obs)
    def intercept(now,point,normal,frame):
        np.testing.assert_allclose(normal,np.r_[axis,0.],atol=1e-12)
        return crossing
    grip.field_tracker=SimpleNamespace(add=lambda o:True,intercept=intercept)
    assert grip.update(SimpleNamespace(timestamp_s=1.),None,None,None,1.)
    assert grip.closed_at==1.

@pytest.mark.parametrize('axis',[(0.,0.),(np.nan,1.),(1.,)])
def test_invalid_survey_axis_rejected(axis):
    with pytest.raises(ValueError):
        FieldPassGrip(PassGrip(),(6,0,1),approach_axis_xy=axis)
