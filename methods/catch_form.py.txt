"""Sensor-only Aero catch-form diagnostics and opt-in versioned penalties.

A private FK state consumes only29 body encoders and IMU orientation. No plant
MjData, root position, contact, ball truth, or simulator pose is accepted.
Engineering tolerances below are not measured human biomechanics.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import math
import numpy as np

FORM_SCHEMA='aero-over-shoulder-form-v1'

@dataclass(frozen=True)
class CatchFormContext:
    over_shoulder: bool=False
    phase: str='inactive'  # approach / receive / hold / inactive
    perception_timestamp_s: float|None=None
    causal_fit_valid: bool=False  # caller certifies causal descending-pass fit
    predicted_intercept_height_above_shoulder_m: float|None=None
    intercept_height_std_m: float|None=None
    time_to_arrival_s: float|None=None  # at perception_timestamp_s, not now
    arrival_std_s: float|None=None

@dataclass(frozen=True)
class CatchFormTolerances:
    min_receive_palm_up_cos: float=.60
    nominal_max_palm_above_shoulder_m: float=.12
    intercept_height_margin_m: float=.06
    height_scale_m: float=.15
    max_intercept_height_std_m: float=.08
    max_perception_age_s: float=.12
    max_arrival_std_s: float=.06
    confidently_distant_arrival_s: float=.65
    high_ball_early_abstain_height_m: float=.25
    early_hand_height_above_shoulder_m: float=-.05
    palm_up_weight_per_s: float=2.
    excessive_height_weight_per_s: float=1.
    early_hands_weight_per_s: float=.5


def form_penalty(metrics,context,now,tolerances=None):
    """Pure JSONable penalty computation. Caller supplies causally gated context.

    Palm orientation is relevant only in commanded over-shoulder receive phase.
    Height/early-hand terms abstain without fresh confident intercept prediction.
    Hold permits reorientation/tucking; frontal catches incur none of these costs.
    """
    t=tolerances or CatchFormTolerances()
    if context.phase not in ('inactive','approach','receive','hold'):raise ValueError('unknown catch-form phase')
    active=bool(context.over_shoulder and context.phase!='inactive')
    orientation=active and context.phase=='receive'
    age=None if context.perception_timestamp_s is None else float(now-context.perception_timestamp_s)
    fresh=bool(age is not None and math.isfinite(age) and -1e-8<=age<=t.max_perception_age_s+1e-8)
    h=context.predicted_intercept_height_above_shoulder_m;std=context.intercept_height_std_m
    known_height=bool(context.causal_fit_valid and fresh and h is not None and std is not None and math.isfinite(h)and math.isfinite(std)and 0.<=std<=t.max_intercept_height_std_m)
    height_upper=None if not known_height else float(h+2.*std)
    height_gate=bool(active and known_height)
    allowed_height=None if not height_gate else max(t.nominal_max_palm_above_shoulder_m,height_upper+t.intercept_height_margin_m)
    eta=context.time_to_arrival_s;eta_std=context.arrival_std_s
    known_eta=bool(known_height and eta is not None and eta_std is not None and math.isfinite(eta)and math.isfinite(eta_std)and 0.<=eta_std<=t.max_arrival_std_s)
    lower_eta=None if not known_eta else float(eta-max(0.,age)-2.*eta_std)
    early=bool(active and context.phase=='approach'and known_eta and lower_eta>t.confidently_distant_arrival_s and height_upper<=t.high_ball_early_abstain_height_m)
    palm_cos=np.asarray(metrics['palm_up_cos'],float);height=np.asarray(metrics['palm_face_height_above_shoulders_m'],float)
    if palm_cos.shape!=(2,)or height.shape!=(2,)or not np.isfinite(palm_cos).all()or not np.isfinite(height).all():raise ValueError('finite bilateral form metrics required')
    palm=t.palm_up_weight_per_s*float(np.mean((np.maximum(0.,t.min_receive_palm_up_cos-palm_cos)/(1.+t.min_receive_palm_up_cos))**2))if orientation else 0.
    raised=t.excessive_height_weight_per_s*float(np.mean(np.minimum(2.,np.maximum(0.,height-allowed_height)/t.height_scale_m)**2))if height_gate else 0.
    early_cost=t.early_hands_weight_per_s*float(np.mean(np.minimum(2.,np.maximum(0.,height-t.early_hand_height_above_shoulder_m)/t.height_scale_m)**2))if early else 0.
    reasons=[]
    if not active:reasons.append('not_commanded_over_shoulder_or_inactive')
    if not orientation:reasons.append('orientation_only_applies_in_receive_phase')
    if not known_height:reasons.append('height_abstains_without_fresh_confident_causal_intercept_height')
    if not known_eta:reasons.append('early_hands_abstains_without_confident_causal_arrival')
    if height_upper is not None and height_upper>t.high_ball_early_abstain_height_m:reasons.append('early_hands_abstains_for_high_intercept')
    return {'schema':FORM_SCHEMA,'version':FORM_SCHEMA,'penalty_per_s':float(palm+raised+early_cost),'penalty_terms_per_s':{'palm_up':palm,'excessive_raise':raised,'early_hands':early_cost},'gates':{'active_over_shoulder':active,'receive_orientation':orientation,'fresh_causal_intercept_height':known_height,'height_penalty':height_gate,'early_hands_penalty':early},'allowed_palm_height_above_shoulder_m':allowed_height,'confident_arrival_lower_bound_s':lower_eta,'abstention_reasons':reasons,'engineering_tolerances':asdict(t)}

class CatchFormEvaluator:
    """Independent FK from model template plus measured body encoders/IMU."""
    def __init__(self,model,joint_names,tolerances=None):
        import mujoco
        self._mj=mujoco;self._model=model;self._data=mujoco.MjData(model);self._names=tuple(joint_names)
        if len(self._names)!=29 or len(set(self._names))!=29:raise ValueError('29 unique body encoder names required')
        self._enc=np.array([int(model.joint(n).qposadr[0])for n in self._names]);self._base=int(model.joint('floating_base_joint').qposadr[0]);self.tolerances=tolerances or CatchFormTolerances()
        self._palms=[int(model.body(f'aero_{s}_palm').id)for s in ('left','right')];self._shoulders=[int(model.body(f'{s}_shoulder_pitch_link').id)for s in ('left','right')];self._elbows=[int(model.body(f'{s}_elbow_link').id)for s in ('left','right')]
        self._head=None;self._head_vertices=None;self._head_source='unavailable'
        try:
            self._head=int(model.geom('head_collision').id);self._head_source='head_collision capsule world-z bound'
        except KeyError:
            try:
                mid=int(model.mesh('head_link').id);self._head=int(np.flatnonzero(model.geom_dataid==mid)[0]);a=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);self._head_vertices=model.mesh_vert[a:a+n].copy();self._head_source='head_link static mesh maximum world-z'
            except (KeyError,IndexError):pass
        self.calibration=self._calibrate()
    def _calibrate(self):
        result={}
        for side,bid in zip(('left','right'),self._palms):
            ids=[int(self._model.site(f'aero_{side}_palm_collision_{i}').id)for i in range(1,5)]
            if any(int(self._model.site_bodyid[i])!=bid for i in ids):raise ValueError('palm face sites must be fixed to calibrated palm body')
            points=self._model.site_pos[ids].copy();center=points.mean(0);_,_,vh=np.linalg.svd(points-center);normal=vh[-1];back=self._model.geom(f'aero_{side}_palm_collision_6').pos
            if normal@(center-back)<0:normal=-normal
            expected=np.array([0.,1. if side=='left'else -1.,0.])
            if not np.allclose(normal,expected,atol=1e-7):raise ValueError('Aero palm geometry does not match calibrated chiral gripping face')
            result[side]={'normal_local':normal.tolist(),'face_center_local_m':center.tolist(),'site_plane_max_error_m':float(abs((points-center)@normal).max()),'normal_sign_evidence':'front face sites1..4 relative to back palm collision6','finger_longitudinal_axis_local_approx':[0.,0.,1.]}
        return result
    def evaluate(self,observation,context=None):
        q=np.asarray(observation.joint_position,float);imu=np.asarray(observation.imu_quat_wxyz,float);now=float(observation.timestamp)
        if q.shape!=(29,)or imu.shape!=(4,)or not np.isfinite(q).all()or not np.isfinite(imu).all()or np.linalg.norm(imu)<1e-9 or not math.isfinite(now):raise ValueError('finite29 encoders, IMU quaternion and sensor clock required')
        # This is private FK state, not the plant. Unknown fingers stay at model rest.
        self._data.qpos[:]=self._model.qpos0;self._data.qvel[:]=0.;self._data.qpos[self._base:self._base+7]=np.r_[np.zeros(3),imu/np.linalg.norm(imu)];self._data.qpos[self._enc]=q
        self._mj.mj_kinematics(self._model,self._data)
        shoulder=self._data.xpos[self._shoulders].mean(0);elbows=self._data.xpos[self._elbows];face=[];normals=[]
        for side,bid in zip(('left','right'),self._palms):
            R=self._data.xmat[bid].reshape(3,3);c=self.calibration[side];face.append(self._data.xpos[bid]+R@c['face_center_local_m']);normals.append(R@c['normal_local'])
        face=np.array(face);normals=np.array(normals);g=self._head;R=np.eye(3)if g is None else self._data.geom_xmat[g].reshape(3,3);size=np.zeros(3)if g is None else self._model.geom_size[g]
        # Model head reference is the collision capsule's conservative world-z top.
        head_top=None if g is None else float((self._head_vertices@R.T+self._data.geom_xpos[g])[:,2].max())if self._head_vertices is not None else float(self._data.geom_xpos[g,2]+abs(R[2,2])*size[1]+size[0])
        metrics={'palm_up_cos':normals[:,2].tolist(),'palm_normals_world':normals.tolist(),'palm_face_centers_relative_pelvis_m':face.tolist(),'palm_face_height_above_shoulders_m':(face[:,2]-shoulder[2]).tolist(),'palm_face_height_above_head_top_m':None if head_top is None else (face[:,2]-head_top).tolist(),'shoulder_midpoint_height_above_pelvis_m':float(shoulder[2]),'head_top_height_above_pelvis_m':head_top,'head_top_reference':self._head_source,'palm_normal_pair_cos':float(normals[0]@normals[1]),'both_palms_up_at_least_060':bool((normals[:,2]>=.60).all()),'both_palms_up_preferred_0707':bool((normals[:,2]>=.707).all()),'elbow_height_above_shoulders_m':(elbows[:,2]-shoulder[2]).tolist(),'elbow_span_m':float(np.linalg.norm(elbows[0]-elbows[1])),'palm_face_separation_m':float(np.linalg.norm(face[0]-face[1])),'elbow_joint_encoder_rad':[float(q[self._names.index(f'{s}_elbow_joint')])for s in ('left','right')],'finger_closure_measured':False}
        result=form_penalty(metrics,context or CatchFormContext(),now,self.tolerances);result['metrics']=metrics
        # Stable top-level aliases used by the integration's causal-height context.
        result['shoulder_midpoint_height_above_pelvis_m']=metrics['shoulder_midpoint_height_above_pelvis_m']
        result['head_top_height_above_pelvis_m']=metrics['head_top_height_above_pelvis_m']
        return result
