"""Choose a fresh surveyed RGB-D ball/receiver pair without reducing uncertainty.

This selects existing measurements; it does not fuse correlated cameras, invent
poses during occlusion, change confidence thresholds, or consume simulator state.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CatchView:
    name: str
    frame: object
    survey: object
    ball: object


def body_yaw_rotation(imu_quat_wxyz):
    q=np.asarray(imu_quat_wxyz,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-9:
        raise ValueError('finite nonzero IMU quaternion required')
    w,x,y,z=q/np.linalg.norm(q)
    yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z));c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])


def select_catch_view(views, poses, *, now_s, imu_quat_wxyz,
                      max_age_s=.08, max_skew_s=.04, fk_std_m=.01):
    """Return (view, pose, diagnostics), or (None,None,diagnostics).

    Receiver covariance is XY/yaw, with vertical variance reported separately.
    Ranking uses the largest body-frame coordinate standard deviation, matching
    CupSensors' existing confidence gate. No new acceptance rule is substituted.
    """
    if not np.isfinite([now_s,max_age_s,max_skew_s,fk_std_m]).all() or min(max_age_s,max_skew_s,fk_std_m)<=0:
        raise ValueError('finite positive sensor uncertainty/time limits required')
    rotation=body_yaw_rotation(imu_quat_wxyz)
    choices=[]
    for view in views:
        ball=view.ball
        if ball is None or ball.frame_id!='field':continue
        capture=float(view.frame.timestamp_s)
        if not -1e-8<=now_s-capture<=max_age_s+1e-8 or abs(ball.timestamp_s-capture)>1e-8:continue
        b=np.asarray(ball.covariance_m2,dtype=float)
        if b.shape!=(3,3) or not np.isfinite(b).all() or not np.allclose(b,b.T) or np.linalg.eigvalsh(b).min() < -1e-10:continue
        for pose_name,pose in poses:
            if (pose is None or pose.z_m is None or pose.z_std_m is None
                    or pose.frame_id!=ball.frame_id or pose.pose_provenance!=ball.pose_provenance
                    or not -1e-8<=now_s-pose.timestamp_s<=max_age_s+1e-8
                    or abs(pose.timestamp_s-capture)>max_skew_s+1e-8
                    or not np.isfinite([pose.z_m,pose.z_std_m]).all() or pose.z_std_m<0):continue
            p=np.asarray(pose.covariance,dtype=float)
            if p.shape!=(3,3) or not np.isfinite(p).all() or not np.allclose(p,p.T) or np.linalg.eigvalsh(p).min() < -1e-10:continue
            cov=b.copy();cov[:2,:2]+=p[:2,:2];cov[2,2]+=pose.z_std_m**2
            cov=rotation@cov@rotation.T+np.eye(3)*fk_std_m**2
            sigma=np.sqrt(np.maximum(0.,np.diag(cov)))
            score=(float(sigma.max()),abs(capture-pose.timestamp_s),now_s-capture,view.name,pose_name)
            choices.append((score,view,pose,dict(ball_camera=view.name,pose_camera=pose_name,
                           capture_s=capture,pose_timestamp_s=float(pose.timestamp_s),sigma_body_m=sigma.tolist())))
    diagnostics=dict(candidate_pairs=len(choices),uncertainty_reduced_by_fusion=False)
    if not choices:return None,None,diagnostics
    _,view,pose,chosen=min(choices,key=lambda row:row[0])
    diagnostics.update(chosen)
    return view,pose,diagnostics
