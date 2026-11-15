from types import SimpleNamespace as NS
import numpy as np
from g1_receiver.catch_view_selection import CatchView,select_catch_view

def view(name,t=1.,variance=.001,provenance='synthetic_fixture'):
    return CatchView(name,NS(timestamp_s=t),NS(),NS(timestamp_s=t,frame_id='field',pose_provenance=provenance,covariance_m2=np.eye(3)*variance))

def pose(t=1.,variance=.001,provenance='synthetic_fixture'):
    return NS(timestamp_s=t,frame_id='field',pose_provenance=provenance,z_m=.75,z_std_m=np.sqrt(variance),covariance=np.eye(3)*variance)

def choose(v,p,now=1.):return select_catch_view(v,p,now_s=now,imu_quat_wxyz=(1,0,0,0))

def test_best_pair_preserves_measured_uncertainty_without_multicamera_discount():
    a,b=view('far',variance=.004),view('near')
    v,p,d=choose([a,b],[('old',pose(variance=.003)),('close',pose())])
    assert v is b and d['pose_camera']=='close'
    np.testing.assert_allclose(d['sigma_body_m'],np.sqrt(.0021))
    _,_,duplicate=choose([b,b,b],[('close',pose()),('copy',pose())])
    np.testing.assert_allclose(duplicate['sigma_body_m'],d['sigma_body_m'])

def test_stale_future_skew_and_provenance_mismatch_cannot_supply_pose():
    for bad in (pose(.8),pose(1.1),pose(.95),pose(provenance='live_estimate')):
        assert choose([view('ball')],[('bad',bad)])[0] is None
    assert choose([view('future',t=1.1)],[('pose',pose())])[0] is None

def test_covariance_ranking_uses_worst_axis_not_trace():
    thin=pose();thin.covariance=np.diag([.0035,.0001,.001])
    balanced=pose(variance=.002)
    _,p,d=choose([view('ball')],[('thin',thin),('balanced',balanced)])
    assert p is balanced
    assert d['pose_camera']=='balanced'

def test_invalid_covariance_and_missing_ball_abstain():
    bad=view('bad');bad.ball.covariance_m2[0,0]=-.01
    assert choose([bad],[('pose',pose())])[0] is None
    assert choose([CatchView('occluded',NS(timestamp_s=1.),NS(),None)],[('pose',pose())])[0] is None
