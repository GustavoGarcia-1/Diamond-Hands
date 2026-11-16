"""Versioned evaluation-only anatomical Aero catch evidence.

All native finger/thumb collision surfaces count; mounts and back/palmar-side
contacts are distinguished at the actual contact point. Never use truth arrays
from this module in online control, a training actor/critic/reward, or curriculum.
Independent offline selection/promotion is precisely what these criteria serve.
Legacy strict scores remain unchanged and must be reported alongside this one.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
from hashlib import sha256
import json
import numpy as np
import mujoco

SCHEMA='aero-anatomical-catch-v1-predeclared-20261114'
GROUPS=('left_finger','left_thumb','left_palmar','right_finger','right_thumb','right_palmar','excluded_hand','torso','floor','other')
COLUMNS=('timestamp_s','ball_x_m','ball_y_m','ball_z_m','ball_floor_contact','applied_ball_load',*(g+'_normal_n'for g in GROUPS),*(g+'_vertical_n'for g in GROUPS),'interhand_self_normal_n','minimum_interhand_distance_m','minimum_ball_contact_distance_m')
INDEX={name:i for i,name in enumerate(COLUMNS)}

@dataclass(frozen=True)
class AnatomicalCatchCriteria:
    min_initial_free_flight_s:float=.020
    min_normal_per_hand_n:float=.10
    min_continuous_retention_s:float=.75
    final_window_s:float=.25
    final_bilateral_fraction:float=.90
    max_final_gap_s:float=.025
    support_force_tolerance_n:float=1e-9
    min_mean_vertical_hand_weight_fraction:float=.80
    form_before_contact_s:float=.020
    form_after_contact_s:float=.200
    min_palm_up_cos:float=.60
    max_face_height_above_shoulder_m:float=.15
    min_form_fraction:float=.80
    max_form_sample_gap_s:float=.050
    max_interhand_penetration_m:float=.005
    max_ball_penetration_m:float=.010
    min_upright_cosine:float=.90
    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all()or min(asdict(self).values())<=0:raise ValueError('positive finite anatomical criteria required')
        if self.final_bilateral_fraction>1 or self.min_form_fraction>1 or self.min_palm_up_cos>1:raise ValueError('invalid fraction/cosine criteria')

def _ancestor(model,body,root):
    while body:
        if body==root:return True
        body=int(model.body_parentid[body])
    return False

def physics_fingerprint(model):
    """Physical model fingerprint; cameras/lighting/noncollidable visuals omitted.

    Host PD gains/torque clamps outside MJCF require their own source/config hash.
    No runtime pose or contact is part of the model fingerprint.
    """
    def value(x):return np.asarray(x).tolist()
    payload={'schema':'anatomical-physics-model-v1','timestep_s':float(model.opt.timestep),'gravity':value(model.opt.gravity),'integrator':int(model.opt.integrator),'solver':int(model.opt.solver),'iterations':int(model.opt.iterations),'tolerance':float(model.opt.tolerance),'cone':int(model.opt.cone),'body_mass':value(model.body_mass),'body_inertia':value(model.body_inertia),'joint_range':value(model.jnt_range),'joint_stiffness':value(model.jnt_stiffness),'dof_damping':value(model.dof_damping),'dof_armature':value(model.dof_armature),'actuator_gainprm':value(model.actuator_gainprm),'actuator_biasprm':value(model.actuator_biasprm),'actuator_gear':value(model.actuator_gear),'actuator_forcerange':value(model.actuator_forcerange),'actuator_ctrlrange':value(model.actuator_ctrlrange),'tendon_stiffness':value(model.tendon_stiffness),'tendon_damping':value(model.tendon_damping),'tendon_range':value(model.tendon_range),'collision_geoms':[]}
    for i in range(model.ngeom):
        if not(model.geom_contype[i]or model.geom_conaffinity[i]):continue
        payload['collision_geoms'].append({'body':model.body(int(model.geom_bodyid[i])).name,'name':model.geom(i).name,'type':int(model.geom_type[i]),'size':value(model.geom_size[i]),'pos':value(model.geom_pos[i]),'quat':value(model.geom_quat[i]),'friction':value(model.geom_friction[i]),'solref':value(model.geom_solref[i]),'solimp':value(model.geom_solimp[i]),'margin':float(model.geom_margin[i]),'gap':float(model.geom_gap[i]),'contype':int(model.geom_contype[i]),'conaffinity':int(model.geom_conaffinity[i])})
    for attr in ('body_pos','body_quat','body_ipos','body_iquat','jnt_pos','jnt_axis','dof_frictionloss','geom_condim','geom_priority','geom_solmix','pair_solref','pair_solimp','pair_friction','eq_type','eq_solref','eq_solimp','tendon_frictionloss','actuator_dynprm','actuator_trntype','actuator_dyntype','actuator_gaintype','actuator_biastype'):
        if hasattr(model,attr):payload[attr]=value(getattr(model,attr))
    payload['collision_mesh_binary_geometry']=[]
    ids=sorted({int(model.geom_dataid[i])for i in range(model.ngeom)if (model.geom_contype[i]or model.geom_conaffinity[i])and int(model.geom_type[i])==int(mujoco.mjtGeom.mjGEOM_MESH)})
    for mid in ids:
        va=int(model.mesh_vertadr[mid]);vn=int(model.mesh_vertnum[mid]);fa=int(model.mesh_faceadr[mid]);fn=int(model.mesh_facenum[mid]);binary=np.ascontiguousarray(model.mesh_vert[va:va+vn]).tobytes()+np.ascontiguousarray(model.mesh_face[fa:fa+fn]).tobytes()
        payload['collision_mesh_binary_geometry'].append({'name':model.mesh(mid).name,'vertices':vn,'faces':fn,'sha256':sha256(binary).hexdigest()})
    digest=sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'sha256':digest,'parameters':payload,'host_gains_and_torque_source_hash_required_separately':True}

class AnatomicalContactAudit:
    """Call after every physics step, outside the controller. No plant mutation."""
    def __init__(self,model,*,reference_physics_sha256=None):
        self.model=model;self.dt=float(model.opt.timestep);self.ball=int(model.geom('football_surface').id);self.ball_body=int(model.geom_bodyid[self.ball]);self.floor=int(model.geom('floor').id)
        joint=model.joint('football_free')
        if int(joint.type[0])!=int(mujoco.mjtJoint.mjJNT_FREE)or int(joint.bodyid[0])!=self.ball_body:raise ValueError('ball must have its own free joint')
        for i,kind in enumerate(model.eq_type):
            if int(kind)not in (int(mujoco.mjtEq.mjEQ_CONNECT),int(mujoco.mjtEq.mjEQ_WELD)):continue
            objects=(int(model.eq_obj1id[i]),int(model.eq_obj2id[i]))
            if int(model.eq_objtype[i])==int(mujoco.mjtObj.mjOBJ_SITE):objects=tuple(int(model.site_bodyid[j])for j in objects if j>=0)
            if self.ball_body in objects:raise ValueError('ball cannot be attached by equality')
        adr=int(joint.dofadr[0]);self.ball_dofs=slice(adr,adr+6);self.ball_weight_n=float(model.body_mass[self.ball_body]*np.linalg.norm(model.opt.gravity))
        if self.ball_weight_n<=0:raise ValueError('ball weight must be positive')
        self.palms=[int(model.body('aero_'+s+'_palm').id)for s in ('left','right')];self.side=np.full(model.ngeom,-1,int);self.kind=['other']*model.ngeom;self.manifest=[]
        for g in range(model.ngeom):
            body=int(model.geom_bodyid[g]);name=model.geom(g).name or'';bodyname=model.body(body).name or''
            for i,side in enumerate(('left','right')):
                if bodyname.startswith('aero_'+side+'_'):self.side[g]=i
                if body==self.palms[i]:self.kind[g]='palmar_candidate'
                elif _ancestor(model,body,self.palms[i]):
                    if any(word in bodyname for word in ('index','middle','ring','pinky')):self.kind[g]='finger'
                    elif 'thumb'in bodyname or name.startswith('aero_'+side+'_th_'):self.kind[g]='thumb'
                    else:self.kind[g]='excluded_hand'
                elif self.side[g]==i:self.kind[g]='excluded_hand'
            if model.geom_contype[g]or model.geom_conaffinity[g]:self.manifest.append({'geom_id':g,'geom':name,'body':bodyname,'side':None if self.side[g]<0 else ('left','right')[self.side[g]],'category':self.kind[g]})
        self._force=np.zeros(6);self._rows=[];self.initial_physics=physics_fingerprint(model);self.reference=reference_physics_sha256
    def classify_surface(self,geom_id,point_world,normal_on_ball_world,data):
        """Actual palm contact must be on its receiving half, not a back clamp."""
        g=int(geom_id);side=int(self.side[g]);kind=self.kind[g]
        if side<0:
            if g==self.floor:return 'floor'
            return 'torso'if self.model.body(int(self.model.geom_bodyid[g])).name in ('pelvis','waist_yaw_link','waist_roll_link','torso_link')else'other'
        if kind=='palmar_candidate':
            bid=self.palms[side];R=data.xmat[bid].reshape(3,3);axis=np.array([0.,1. if side==0 else -1.,0.]);normal=R@axis;front=float((np.asarray(point_world)-data.xpos[bid])@normal)
            # 2mm plane slack and near-tangential edges are explicit engineering tolerances.
            kind='palmar'if front>=-.002 and float(np.asarray(normal_on_ball_world)@normal)>=-.10 else'excluded_hand'
        return ('left_'if side==0 else'right_')+kind if kind in ('finger','thumb','palmar')else'excluded_hand'
    def record(self,data):
        normal=np.zeros(len(GROUPS));vertical=np.zeros(len(GROUPS));floor=False;self_force=0.;min_distance=0.;ball_distance=0.
        for i,c in enumerate(data.contact):
            a,b=map(int,c.geom)
            if a==self.ball:other=b;sign=-1.
            elif b==self.ball:other=a;sign=1.
            else:
                if self.side[a]>=0 and self.side[b]>=0 and self.side[a]!=self.side[b]:
                    mujoco.mj_contactForce(self.model,data,i,self._force);self_force+=max(0.,float(self._force[0]));min_distance=min(min_distance,float(c.dist))
                continue
            ball_distance=min(ball_distance,float(c.dist))
            mujoco.mj_contactForce(self.model,data,i,self._force);force=max(0.,float(self._force[0]));world_normal=sign*np.asarray(c.frame).reshape(3,3)[0];group=self.classify_surface(other,c.pos,world_normal,data);j=GROUPS.index(group)
            floor|=other==self.floor and(c.dist<=0 or force>0)
            if force>0:normal[j]+=force;vertical[j]+=sign*float(np.asarray(c.frame).reshape(3,3)[:,2]@self._force[:3])
        applied=bool(np.any(data.xfrc_applied[self.ball_body])or np.any(data.qfrc_applied[self.ball_dofs]))
        self._rows.append(np.r_[data.time,data.geom_xpos[self.ball],floor,applied,normal,vertical,self_force,min_distance,ball_distance])
    def arrays(self):
        return {'anatomical_contact':np.asarray(self._rows,float).reshape(-1,len(COLUMNS)),'anatomical_contact_columns':np.asarray(COLUMNS),'anatomical_free_joint_verified':np.asarray(True)}
    def metadata(self):
        current=physics_fingerprint(self.model)
        return {'version':SCHEMA,'free_ball_joint_and_no_attachment_verified':True,'physics_timestep_s':self.dt,'ball_weight_n':self.ball_weight_n,'physics_sha256':self.initial_physics['sha256'],'physics_unchanged_during_capture':current['sha256']==self.initial_physics['sha256'],'reference_physics_sha256':self.reference,'physics_reference_match':None if self.reference is None else self.reference==current['sha256'],'physics_parameters':self.initial_physics['parameters'],'geometry_classification':self.manifest,'palmar_contact_rule':{'local_normal_left':[0,1,0],'local_normal_right':[0,-1,0],'front_plane_slack_m':.002,'min_outward_normal_cos':-.10},'native_finger_thumb_all_collision_surfaces_included':True,'mounts_back_palm_excluded':True,'host_gains_and_torque_source_hash_required_separately':True,'legacy_strict_score_unchanged':True,'truth_evaluation_only':True}

def _runs(mask):
    e=np.flatnonzero(np.diff(np.r_[False,mask,False]));return list(zip(e[::2],e[1::2]))

def score_anatomical_catch(arrays,*,release_time_s,finish_time_s,ball_weight_n,form_rows=None,motion=None,required_intercept_height_above_shoulder_m=None,height_exception_basis=None,physics_reference_match=None,verified_free_ball=None,criteria=None):
    c=criteria or AnatomicalCatchCriteria();verified=bool(arrays.get('anatomical_free_joint_verified',False))if verified_free_ball is None else bool(verified_free_ball);rows=np.asarray(arrays['anatomical_contact'],float)
    if rows.ndim!=2 or rows.shape[1]!=len(COLUMNS)or not len(rows)or not np.isfinite(rows).all():raise ValueError('finite anatomical physics trace required')
    if not np.isfinite([release_time_s,finish_time_s,ball_weight_n]).all()or ball_weight_n<=0:raise ValueError('finite release/end and positive weight required')
    t=rows[:,0];dt=float(np.median(np.diff(t)))if len(t)>1 else .001;clock=bool(dt>0 and np.allclose(np.diff(t),dt,rtol=0,atol=1e-7)and abs(t[0]-release_time_s-dt)<=1e-7 and abs(t[-1]-finish_time_s)<=1e-7)
    n=rows[:,6:6+len(GROUPS)];v=rows[:,6+len(GROUPS):6+2*len(GROUPS)];left=n[:,:3].sum(1);right=n[:,3:6].sum(1);bilateral=(left>=c.min_normal_per_hand_n)&(right>=c.min_normal_per_hand_n)
    bad=(n[:,6:].sum(1)>c.support_force_tolerance_n)|rows[:,4].astype(bool)|rows[:,5].astype(bool);first_bad=np.flatnonzero(bad);cutoff=int(first_bad[0])if len(first_bad)else len(t);clean=bilateral.copy();clean[cutoff:]=False
    a,b=max(_runs(clean),key=lambda ab:ab[1]-ab[0],default=(0,0));duration=float((b-a)*dt);initial=max(1,round(c.min_initial_free_flight_s/dt));free=bool(len(t)>=initial and not rows[:initial,4:6].any()and n[:initial].sum(1).max()<=c.support_force_tolerance_n and rows[:initial,3].min()>=.15)
    hits=np.flatnonzero((left+right)>=c.min_normal_per_hand_n);first_contact=float(t[hits[0]])if len(hits)else None;last_count=max(1,round(c.final_window_s/dt));tail=clean[-last_count:];gaps=_runs(~tail);maxgap=max((j-i)*dt for i,j in gaps)if gaps else 0.;fraction=float(tail.mean());final=bool(len(t)>=last_count and tail[-1]and fraction>=c.final_bilateral_fraction and maxgap<=c.max_final_gap_s+1e-8)
    vertical_fraction=float(v[a:b,:6].sum(1).mean()/ball_weight_n)if b>a else None;deep_self=bool(np.min(rows[:,INDEX['minimum_interhand_distance_m']]) < -c.max_interhand_penetration_m);deep_ball=bool(np.min(rows[:,INDEX['minimum_ball_contact_distance_m']]) < -c.max_ball_penetration_m);form={'available':False,'pass':False};height_limit=c.max_face_height_above_shoulder_m
    if required_intercept_height_above_shoulder_m is not None:
        if not height_exception_basis or not np.isfinite(required_intercept_height_above_shoulder_m):raise ValueError('height exception requires explicit predeclared trajectory basis')
        height_limit=max(height_limit,float(required_intercept_height_above_shoulder_m)+.05)
    if form_rows is not None and first_contact is not None:
        selected=[r for r in form_rows if first_contact-c.form_before_contact_s-1e-8<=r['time_s']<=first_contact+c.form_after_contact_s+1e-8]
        if selected:
            ft=np.array([r['time_s']for r in selected]);metrics=[r.get('metrics',r)for r in selected];up=np.array([r['palm_up_cos']for r in metrics]);height=np.array([r['palm_face_height_above_shoulders_m']for r in metrics]);coverage=bool(len(ft)>=3 and ft[0]<=first_contact+.005 and ft[-1]>=first_contact+c.form_after_contact_s-c.max_form_sample_gap_s and np.diff(ft).max(initial=0)<=c.max_form_sample_gap_s)
            finite=bool(np.isfinite(up).all()and np.isfinite(height).all());proper=(up.min(1)>=c.min_palm_up_cos)&(height.max(1)<=height_limit);form={'available':True,'sample_count':len(ft),'clock_coverage_ok':coverage,'finite':finite,'both_up_fraction':float((up.min(1)>=c.min_palm_up_cos).mean()),'both_height_fraction':float((height.max(1)<=height_limit).mean()),'joint_form_fraction':float(proper.mean()),'height_limit_above_shoulder_m':height_limit,'height_exception_basis':height_exception_basis,'pass':bool(coverage and finite and proper.mean()>=c.min_form_fraction)}
    upright=None
    if motion is not None and first_contact is not None:
        m=np.asarray(motion,float);window=m[(m[:,0]>=first_contact-1e-8)&(m[:,0]<=finish_time_s+1e-8)];upright=bool(len(window)and np.isfinite(window).all()and window[:,4].min()>=c.min_upright_cosine)
    physics_checks={'physics_clock_and_endpoint':clock,'initial_free_flight':free,'verified_free_joint_no_attachment':verified,'continuous_anatomical_bilateral_075s':duration+1e-8>=c.min_continuous_retention_s,'final_anatomical_hand_hold':final,'no_excluded_floor_body_or_applied_support':not bool(bad.any()),'load_bearing_hands':vertical_fraction is not None and vertical_fraction>=c.min_mean_vertical_hand_weight_fraction,'no_deep_interhand_penetration':not deep_self,'no_deep_ball_penetration':not deep_ball}
    control=all(physics_checks.values());quality_checks={**physics_checks,'acquisition_lower_palms_up_form':form['pass'],'upright_through_finish':upright is True,'reference_physics_verified':physics_reference_match is True}
    return {'version':SCHEMA,'intended_form':'over_shoulder_lower_upward_cradle','criteria':asdict(c),'anatomical_hand_control':bool(control),'realistic_lower_catch':bool(all(quality_checks.values())),'physics_checks':physics_checks,'quality_checks':quality_checks,'first_anatomical_contact_s':first_contact,'first_excluded_support_s':None if not len(first_bad)else float(t[cutoff]),'longest_clean_bilateral_s':duration,'longest_clean_start_s':float(t[a])if b>a else None,'longest_clean_end_s':float(t[b-1])if b>a else None,'final_bilateral_fraction':fraction,'final_max_gap_s':float(maxgap),'mean_vertical_hand_weight_fraction_in_long_hold':vertical_fraction,'form':form,'upright':upright,'physics_reference_match':physics_reference_match,'peak_interhand_self_normal_n':float(rows[:,INDEX['interhand_self_normal_n']].max()),'minimum_interhand_distance_m':float(rows[:,INDEX['minimum_interhand_distance_m']].min()),'minimum_ball_contact_distance_m':float(rows[:,INDEX['minimum_ball_contact_distance_m']].min()),'peak_ball_normal_n_by_group':dict(zip(GROUPS,n.max(0).tolist())),'gait_evaluated_separately':True,'legacy_strict_score_unchanged':True}
