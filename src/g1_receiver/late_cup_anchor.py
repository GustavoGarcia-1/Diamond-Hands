"""Opt-in surveyed world-cup hold using nominal-command FK and body sensors.
No active simulator state/contact/ball truth is accepted. Weak wrists are
regularized, gains and torque limits are unchanged. Correction freezes for carry.
"""
import numpy as np
from collections import deque
from .out_late_cup_capture import AXES

class WorldCupAnchor:
    def __init__(self,config):
        self.goal=np.asarray(config['world_center_m'],float);self.gain=float(config.get('gain',0.));self.limit=float(config.get('max_translation_m',.12));self.q_limit=.18;self.rate=1.5;self.previous=np.zeros(14);self.rows=[];self.history=deque(maxlen=6);self.last_pose=None;self.last_now=None;self.freeze_at=None;self.frozen=False
        if self.goal.shape!=(3,)or not np.isfinite(self.goal).all()or not 0<=self.gain<=1 or not 0<self.limit<=.12:raise ValueError('bounded declared world-cup command required')
    def bind(self,model,names):
        import mujoco
        self.mj=mujoco;self.m=model;self.d=mujoco.MjData(model);self.names=tuple(names);self.arms=np.array([names.index(f'{s}_{a}_joint')for s in ['left','right']for a in AXES]);self.qadr=np.array([int(model.joint(n).qposadr[0])for n in names]);self.base=int(model.joint('floating_base_joint').qposadr[0]);self.dofs=np.array([int(model.joint(names[i]).dofadr[0])for i in self.arms]);self.ranges=np.array([model.joint(names[i]).range for i in self.arms]);self.pads=[int(model.geom('aero_'+s+'_palm_collision_7').id)for s in ['left','right']];prox=[g for g in range(model.ngeom)if any(model.body(int(model.geom_bodyid[g])).name.startswith(s+'_'+p)for s in ['left','right']for p in ['shoulder_roll','shoulder_yaw','elbow','wrist'])and(model.geom_contype[g]or model.geom_conaffinity[g])];torso=[g for g in range(model.ngeom)if model.body(int(model.geom_bodyid[g])).name=='torso_link'and(model.geom_contype[g]or model.geom_conaffinity[g])];self.pairs=[(g,h)for g in prox for h in torso]
    def geometry(self,q,with_jac=False):
        self.d.qpos[self.qadr[self.arms]]=q;self.mj.mj_kinematics(self.m,self.d);self.mj.mj_comPos(self.m,self.d);P=[];R=[];J=[]
        for g in self.pads:
            rotation=self.d.geom_xmat[g].reshape(3,3).copy();point=self.d.geom_xpos[g]+self.m.geom_size[g,0]*rotation[:,0];P.append(point);R.append(rotation)
            if with_jac:
                a=np.zeros((3,self.m.nv));b=np.zeros_like(a);self.mj.mj_jac(self.m,self.d,a,b,point,int(self.m.geom_bodyid[g]));J.append(np.vstack([a[:,self.dofs],.20*b[:,self.dofs]]))
        return np.array(P),R,J
    def clearance(self,q):
        self.geometry(q);worst=.03
        for g,h in self.pairs:
            if np.linalg.norm(self.d.geom_xpos[g]-self.d.geom_xpos[h])-self.m.geom_rbound[g]-self.m.geom_rbound[h]>.003:continue
            worst=min(worst,float(self.mj.mj_geomDistance(self.m,self.d,g,h,.03,None)))
        return worst
    def apply(self,obs,pose,target,phase,forecast):
        now=float(obs.timestamp);dt=.02 if self.last_now is None else now-self.last_now;self.last_now=now
        if not 0<dt<=.041:raise ValueError('anchor requires50Hzcausalcontrol')
        result=np.asarray(target,float).copy();diag={'time_s':now,'phase':float(phase),'gain':self.gain,'pose_present':pose is not None,'frozen':self.frozen};valid=pose is not None and pose.z_m is not None and pose.z_std_m is not None and pose.fresh(now,.08+1e-8)
        if valid:
            std=float(np.sqrt(np.linalg.eigvalsh(np.asarray(pose.covariance)[:2,:2]).max()));valid=std<=.06 and pose.z_std_m<=.06;diag.update(pose_age_s=now-pose.timestamp_s,xy_std_m=std,z_std_m=pose.z_std_m)
        if valid and (self.last_pose is None or pose.timestamp_s>self.last_pose+1e-8):self.history.append((pose.timestamp_s,*pose.xy_m,pose.z_m));self.last_pose=pose.timestamp_s
        if self.freeze_at is not None and now>=self.freeze_at:self.frozen=True
        if self.frozen:
            if self.gain:result[self.arms]+=self.previous
            diag.update(correction_rad=self.previous.tolist(),status='frozen_carry');self.rows.append(diag);return result
        desired=self.previous.copy()
        if valid and phase>.7 and forecast is not None:
            if self.freeze_at is None:self.freeze_at=float(forecast)+.15
            rows=np.asarray(self.history);velocity=np.zeros(3)
            if len(rows)>=3 and np.ptp(rows[:,0])>=.055:
                t=rows[:,0]-rows[:,0].mean();velocity=t@rows[:,1:]/(t@t)
            xyz=np.r_[pose.xy_m,pose.z_m]+velocity*max(0.,now-pose.timestamp_s);self.d.qpos[:]=self.m.qpos0;self.d.qpos[self.qadr]=obs.joint_position;quat=np.asarray(obs.imu_quat_wxyz);self.d.qpos[self.base:self.base+7]=np.r_[xyz,quat/np.linalg.norm(quat)];nominal=result[self.arms].copy();P0,R0,_=self.geometry(nominal);shift=self.goal-P0.mean(0);shift*=min(1.,self.limit/max(np.linalg.norm(shift),1e-9));goal=P0+self.gain*shift;lo=np.maximum(nominal-self.q_limit,self.ranges[:,0]+.03);hi=np.minimum(nominal+self.q_limit,self.ranges[:,1]-.03);lo[0]=max(lo[0],0.);hi[7]=min(hi[7],0.);q=np.clip(nominal+self.previous,lo,hi)
            for _ in range(4):
                P,R,J=self.geometry(q,True);rhs=[]
                for i in range(2):rhs.extend([*(goal[i]-P[i]),*(.20*.5*sum(np.cross(R[i][:,k],R0[i][:,k])for k in range(3)))])
                A=np.vstack(J);regularizer=np.full(14,.015**2);regularizer[[5,6,12,13]]*=100.;step=np.linalg.solve(A.T@A+np.diag(regularizer),A.T@np.asarray(rhs));trial=np.clip(q+step,lo,hi)
                for scale in [1.,.5,.25,0.]:
                    candidate=q+scale*(trial-q)
                    if self.clearance(candidate)>=.002:q=candidate;break
            desired=q-nominal;diag.update(status='world_hold_near_arrival',nominal_command_center_m=P0.mean(0).tolist(),declared_world_center_m=self.goal.tolist(),translation_command_m=(self.gain*shift).tolist(),target_geometry_clearance_m=self.clearance(q),receiver_pose_timestamp_s=pose.timestamp_s,freeze_at_s=self.freeze_at)
        else:diag['status']='abstain_no_fresh_pose_or_phase'
        updated=self.previous+np.clip(desired-self.previous,-self.rate*dt,self.rate*dt);updated=np.clip(updated,-self.q_limit,self.q_limit)
        if self.gain:self.previous=updated;result[self.arms]=np.clip(result[self.arms]+self.previous,*self.ranges.T)
        diag.update(correction_rad=self.previous.tolist());self.rows.append(diag);return result
