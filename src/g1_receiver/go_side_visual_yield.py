"""Experimental bounded cup yield from fresh RGB-D forecast + encoder/IMU FK.

No live plant data, contact, force, ball truth, fixture timing or reward is accepted.
A valid visual trigger latches one short open-loop yield, then holds that command.
"""
import numpy as np
from .late_cup_anchor import WorldCupAnchor

class VisualCupYield(WorldCupAnchor):
    def __init__(self,gain=0.,max_shift_m=.08,duration_s=.20,min_command_up=.6):
        super().__init__({'world_center_m':[0,0,0],'gain':gain,'max_translation_m':max_shift_m})
        self.min_command_up=float(min_command_up);self.duration=duration_s;self.start=None;self.shift=None;self.packet=None;self.trigger=None
        assert 0<=gain<=1 and 0<duration_s<=.3
    def bind(self,model,names):
        super().bind(model,names)
        groups=[]
        for side in ['left','right']:
            groups.append([g for g in range(model.ngeom)if(model.geom_contype[g]or model.geom_conaffinity[g])and(model.body(int(model.geom_bodyid[g])).name.startswith('aero_'+side)or model.body(int(model.geom_bodyid[g])).name.startswith(side+'_wrist'))])
        self.interhand_pairs=[(g,h)for g in groups[0]for h in groups[1]]
    def clearance(self,q):
        worst=super().clearance(q)
        for g,h in self.interhand_pairs:
            if np.linalg.norm(self.d.geom_xpos[g]-self.d.geom_xpos[h])-self.m.geom_rbound[g]-self.m.geom_rbound[h]>.005:continue
            worst=min(worst,float(self.mj.mj_geomDistance(self.m,self.d,g,h,.03,None)))
        return worst
    def observe(self,obs,pose,prediction,phase):
        self.packet=obs;now=float(obs.timestamp)
        if pose is not None and pose.z_m is not None and pose.fresh(now,.08+1e-8):
            if self.last_pose is None or pose.timestamp_s>self.last_pose+1e-8:self.history.append((pose.timestamp_s,*pose.xy_m,pose.z_m));self.last_pose=pose.timestamp_s
        if self.start is not None or prediction is None:return
        valid=pose is not None and pose.z_std_m is not None and pose.z_std_m<=.06 and pose.fresh(now,.08+1e-8)and np.sqrt(np.linalg.eigvalsh(np.asarray(pose.covariance)[:2,:2]).max())<=.06
        valid=valid and prediction.last_inlier_timestamp_s is not None and -1e-8<=now-prediction.last_inlier_timestamp_s<=.08+1e-8 and prediction.time_std_s<=.06
        ttc=float(prediction.timestamp_s-now)
        if not(valid and phase>.95 and .02<=ttc<=.10):return
        h=np.asarray(self.history);velocity=np.zeros(3)
        if len(h)>=3 and np.ptp(h[:,0])>=.055:
            t=h[:,0]-h[:,0].mean();velocity=t@h[:,1:]/(t@t)
        relative=np.asarray(prediction.velocity_mps)-velocity
        if relative[2]>=0:return
        direction=np.array([np.clip(relative[0],0,.8),np.clip(relative[1],-.25,.25),np.clip(relative[2],-1.2,0)])
        if np.linalg.norm(direction)<1e-6:return
        self.start=now;self.shift=self.limit*direction/np.linalg.norm(direction);self.trigger=dict(time_s=now,ttc_s=ttc,forecast_age_s=now-prediction.last_inlier_timestamp_s,forecast_std_s=float(prediction.time_std_s),relative_rgbd_velocity_mps=relative.tolist(),receiver_velocity_rgbd_mps=velocity.tolist(),shift_world_m=self.shift.tolist(),pose_timestamp_s=pose.timestamp_s)
    def apply(self,target,now,phase):
        result=np.array(target,float).copy()
        if self.start is None or self.packet is None:return result
        obs=self.packet;dt=.02 if self.last_now is None else now-self.last_now;self.last_now=now
        if not 0<dt<=.041:raise ValueError('50Hz target updates required')
        u=float(np.clip((now-self.start)/self.duration,0,1));ramp=u**3*(10-15*u+6*u*u);nominal=result[self.arms].copy();self.d.qpos[:]=self.m.qpos0;self.d.qpos[self.qadr]=obs.joint_position;quat=np.array(obs.imu_quat_wxyz);self.d.qpos[self.base:self.base+7]=np.r_[0.,0.,0.,quat/np.linalg.norm(quat)];P0,R0,_=self.geometry(nominal);goal=P0+self.gain*ramp*self.shift;lo=np.maximum(nominal-self.q_limit,self.ranges[:,0]+.04);hi=np.minimum(nominal+self.q_limit,self.ranges[:,1]-.04);q=np.clip(nominal+self.previous,lo,hi)
        for _ in range(4):
            P,R,J=self.geometry(q,True);rhs=np.concatenate([np.r_[goal[i]-P[i],.20*.5*sum(np.cross(R[i][:,k],R0[i][:,k])for k in range(3))]for i in range(2)]);A=np.vstack(J);reg=np.full(14,.02**2);reg[[5,6,12,13]]*=25;step=np.linalg.solve(A.T@A+np.diag(reg),A.T@rhs)
            for scale in[1.,.5,.25,0.]:
                candidate=np.clip(q+scale*step,lo,hi);points,_,_=self.geometry(candidate);ups=[float(self.d.xmat[int(self.m.geom_bodyid[g])].reshape(3,3)[2,1]*(1 if i==0 else -1))for i,g in enumerate(self.pads)];heights=[float(points[i,2]-self.d.body(s+'_shoulder_pitch_link').xpos[2])for i,s in enumerate(['left','right'])]
                if self.clearance(candidate)>=.005 and min(ups)>=self.min_command_up and max(heights)<=.15:q=candidate;break
        desired=q-nominal;self.previous+=np.clip(desired-self.previous,-self.rate*dt,self.rate*dt);result[self.arms]=np.clip(nominal+self.previous,*self.ranges.T);self.rows.append(dict(time_s=now,ramp=ramp,shift_world_m=(self.gain*ramp*self.shift).tolist(),correction_rad=self.previous.tolist(),commanded_facet_shift_m=(self.geometry(result[self.arms])[0]-P0).tolist(),gain=self.gain));return result
