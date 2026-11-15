"""Opt-in post-arrival RGB-D settling correction; no contact/fixture-state input."""
import numpy as np
from .late_cup_relative_servo import WorldCupAnchor as RelativeCup

class WorldCupAnchor(RelativeCup):
    def __init__(self,config):
        super().__init__(config)
        self.carry_gain=float(config.get('carry_gain',0.));assert 0<=self.carry_gain<=.25
        self.carry_envelope=float(config.get('carry_proximal_envelope_rad',.18));assert .18<=self.carry_envelope<=.30
        self.carry_limits=np.full(14,self.q_limit);self.carry_limits[[0,1,2,3,7,8,9,10]]=self.carry_envelope;self.dry_run=bool(config.get('carry_dry_run',False))
        self.capture_packet=None;self.capture_observation=None;self.carry=np.zeros(14);self.carry_stamp=None
    def apply(self,obs,pose,target,phase,prediction):
        result=super().apply(obs,pose,target,phase,prediction)
        row=self.rows[-1];packet=self.capture_packet;sample=self.capture_observation
        nominal=np.asarray(target)[self.arms];current=result[self.arms].copy()
        active=self.freeze_at is not None and self.freeze_at+.28<=obs.timestamp<=self.freeze_at+1.48
        if not self.carry_gain:return result
        if active and packet is not None and sample is not None and packet.get('visual_valid') and packet.get('visual_velocity_valid'):
            age=float(obs.timestamp-packet['ball_capture_s']);sigma=np.asarray(packet['ball_cup_sigma_m']);fresh=0<=age<=.04+1e-8 and np.max(sigma)+.2*age<=.06
            valid_pose=pose is not None and pose.z_m is not None and pose.fresh(obs.timestamp,.08+1e-8)
            if fresh and valid_pose and(self.carry_stamp is None or packet['ball_capture_s']>self.carry_stamp+1e-8):
                self.carry_stamp=float(packet['ball_capture_s']);self.d.qpos[:]=self.m.qpos0;self.d.qpos[self.qadr]=obs.joint_position;self.d.qpos[self.base:self.base+7]=np.r_[pose.xy_m,pose.z_m,np.asarray(obs.imu_quat_wxyz)];_,_,J=self.geometry(np.asarray(obs.joint_position)[self.arms],True)
                q=np.asarray(sample.imu_quat_wxyz,float);R=np.zeros(9);self.mj.mju_quat2Mat(R,q/np.linalg.norm(q));error=R.reshape(3,3)@(np.asarray(packet['ball_cup_error_body_m'])+age*np.asarray(packet['ball_cup_relative_velocity_body_mps']));error*=min(1.,.06/max(np.linalg.norm(error),1e-8));A=np.vstack(J);rhs=np.tile(np.r_[self.carry_gain*error,np.zeros(3)],2);reg=np.full(14,.02**2);reg[[5,6,12,13]]*=100.;increment=np.linalg.solve(A.T@A+np.diag(reg),A.T@rhs);increment[[5,6,12,13]]=0.;desired=np.clip(current+self.carry+increment,nominal-self.carry_limits,nominal+self.carry_limits);desired=np.clip(desired,self.ranges[:,0]+.03,self.ranges[:,1]-.03);desired[0]=max(0.,desired[0]);desired[7]=min(0.,desired[7])
                for scale in [1.,.5,.25,0.]:
                    candidate=current+self.carry+scale*(desired-current-self.carry)
                    if self.clearance(candidate)>=.002:
                        self.carry+=np.clip(candidate-current-self.carry,-.03,.03);break
                row.update(carry_status='fresh_capture_settling_correction',carry_capture_s=self.carry_stamp,carry_age_s=age,carry_sigma_m=sigma.tolist(),carry_error_world_m=error.tolist())
        row.update(carry_gain=self.carry_gain,carry_correction_rad=(np.zeros(14)if self.dry_run else self.carry).tolist(),carry_proposed_correction_rad=self.carry.tolist(),carry_proposed_target_rad=(current+self.carry).tolist(),carry_proximal_envelope_rad=self.carry_envelope,carry_dry_run=self.dry_run)
        if not self.dry_run:result[self.arms]=np.clip(current+self.carry,self.ranges[:,0],self.ranges[:,1])
        return result
