"""Physical image ambiguities, with unchanged single-pair covariance."""
from dataclasses import replace
import numpy as np
import pytest
from g1_receiver.associated_markers import AssociatedRearMarkerLocalizer
from g1_receiver.localization import MarkerLocalizer
from test_localization import fixture, survey


def rear_frame():
    frame=fixture()
    rgb=np.full_like(frame.rgb,60)
    rgb[frame.rgb[...,1]==210]=(20,20,240)
    rgb[frame.rgb[...,0]==240]=(240,240,20)
    return replace(frame,rgb=rgb)


def legacy_rear(frame):
    rgb=np.zeros_like(frame.rgb)
    rgb[frame.rgb[...,2]==240]=(0,255,255)
    rgb[frame.rgb[...,0]==240]=(255,0,255)
    p=MarkerLocalizer(allow_synthetic_fixture=True).estimate(replace(frame,rgb=rgb),survey())
    if p is None:return None
    R=np.diag([1.,1.,-1.])
    return replace(p,xy_m=p.xy_m+.30*np.array([np.cos(p.yaw_rad),np.sin(p.yaw_rad)]),covariance=R@p.covariance@R)


def test_unique_pair_is_identical_to_legacy_pose_and_covariance():
    frame=rear_frame()
    old=legacy_rear(frame)
    new=AssociatedRearMarkerLocalizer(allow_synthetic_fixture=True).estimate(frame,survey())
    assert old is not None and new is not None
    np.testing.assert_array_equal(new.xy_m,old.xy_m)
    np.testing.assert_array_equal(new.covariance,old.covariance)
    assert new.yaw_rad==old.yaw_rad and new.z_m==old.z_m and new.z_std_m==old.z_std_m


def test_incompatible_yellow_distractor_is_rejected_as_an_identity():
    frame=rear_frame();clean=legacy_rear(frame)
    yellow=frame.rgb[...,0]==240
    distractor=np.roll(yellow,30,axis=0)
    frame.rgb[distractor]=frame.rgb[yellow]
    frame.depth_m[distractor]=frame.depth_m[yellow]
    assert legacy_rear(frame) is None
    localizer=AssociatedRearMarkerLocalizer(allow_synthetic_fixture=True)
    pose=localizer.estimate(frame,survey())
    assert pose is not None
    assert localizer.last_diagnostics['right_candidates']==2
    assert localizer.last_diagnostics['compatible_pairs']==1
    np.testing.assert_array_equal(pose.xy_m,clean.xy_m)
    np.testing.assert_array_equal(pose.covariance,clean.covariance)


def test_two_geometrically_valid_rigs_abstain_instead_of_choosing_nearest():
    frame=rear_frame()
    colored=(frame.rgb[...,0]==240)|(frame.rgb[...,2]==240)
    shifted=np.roll(colored,35,axis=1)
    frame.rgb[shifted]=frame.rgb[colored]
    frame.depth_m[shifted]=frame.depth_m[colored]
    localizer=AssociatedRearMarkerLocalizer(allow_synthetic_fixture=True)
    assert localizer.estimate(frame,survey()) is None
    assert localizer.last_diagnostics['compatible_pairs']>=2


def test_missing_marker_stale_frame_and_wrong_provenance_still_fail_closed():
    frame=rear_frame();frame.rgb[frame.rgb[...,0]==240]=60
    localizer=AssociatedRearMarkerLocalizer(allow_synthetic_fixture=True)
    assert localizer.estimate(frame,survey()) is None
    assert localizer.estimate(rear_frame(),survey(t=.15)) is None
    with pytest.raises(ValueError,match='fixed camera'):
        localizer.estimate(rear_frame(),survey(provenance='live_estimate'))
    with pytest.raises(ValueError,match='explicit'):
        AssociatedRearMarkerLocalizer().estimate(rear_frame(),survey())
