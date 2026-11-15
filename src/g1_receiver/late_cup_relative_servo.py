"""Short-horizon sensor/proprioception cup interception; optional and bounded."""
import numpy as np
from .late_cup_anchor import WorldCupAnchor as AnchorGeometry

def command_form_diagnostic(up_target, ups, heights):
    measured = bool(min(ups)>=.65 and max(heights)<=.14)
    return dict(command_form_proxy_pass=measured,
                command_form_hard_rejection_enabled=not bool(up_target),
                command_form_constraint_pass=None if up_target else measured)

class WorldCupAnchor(AnchorGeometry):
    def __init__(self,config):
        super().__init__(config);self.offset=np.asarray(config.get('ball_center_offset_body_m',[0,0,.06]),float);self.last_prediction=None;self.orientation_sides=config.get("orientation_sides",[0,1]);self.up_target=float(config.get("receiving_up_target",0.));self.orientation_gain=float(config.get("absolute_orientation_gain",0.));assert 0<=self.orientation_gain<=1
    def apply(self,obs,pose,target,phase,prediction):
        now=float(obs.timestamp);dt=.02 if self.last_now is None else now-self.last_now;self.last_now=now;result=np.asarray(target,float).copy();diag={'time_s':now,'gain':self.gain,'phase':float(phase),'mode':'capture_forecast_relative_integral_servo'}
        if prediction is not None and prediction.last_inlier_timestamp_s is not None and -1e-8<=now-prediction.last_inlier_timestamp_s<=.08+1e-8 and prediction.time_std_s<=.06:self.last_prediction=prediction
        pred=self.last_prediction
        if self.freeze_at is not None and now>=self.freeze_at:self.frozen=True
        if self.frozen:
            if self.gain:result[self.arms]+=self.previous
            diag.update(status='frozen_carry',correction_rad=self.previous.tolist());self.rows.append(diag);return result
        valid=pose is not None and pose.z_m is not None and pose.z_std_m is not None and pose.fresh(now,.08+1e-8)and np.sqrt(np.linalg.eigvalsh(np.asarray(pose.covariance)[:2,:2]).max())<=.06 and pose.z_std_m<=.06
        if valid and(self.last_pose is None or pose.timestamp_s>self.last_pose+1e-8):self.history.append((pose.timestamp_s,*pose.xy_m,pose.z_m));self.last_pose=pose.timestamp_s
        ttc=None if pred is None else float(pred.timestamp_s-now);forecast_valid=pred is not None and pred.last_inlier_timestamp_s is not None and -1e-8<=now-pred.last_inlier_timestamp_s<=.12+1e-8 and pred.time_std_s+.05*max(0.,now-pred.last_inlier_timestamp_s)<=.06
        desired=self.previous.copy()
        if valid and forecast_valid and phase>.95 and -.02<=ttc<=.14:
            if self.freeze_at is None:self.freeze_at=float(pred.timestamp_s)+.02
            horizon=max(0.,ttc);history=np.asarray(self.history);velocity=np.zeros(3)
            if len(history)>=3 and np.ptp(history[:,0])>=.055:
                t=history[:,0]-history[:,0].mean();velocity=t@history[:,1:]/(t@t);velocity[:2]*=min(1.,2./max(np.linalg.norm(velocity[:2]),1e-8));velocity[2]=np.clip(velocity[2],-.25,.25)
            xyz=np.r_[pose.xy_m,pose.z_m]+velocity*(horizon+max(0.,now-pose.timestamp_s));quat=np.asarray(obs.imu_quat_wxyz,float).copy();quat/=np.linalg.norm(quat);omega=np.asarray(obs.imu_gyro,float);angle=min(.12,float(np.linalg.norm(omega))*horizon);delta=np.r_[np.cos(angle/2),np.zeros(3)if np.linalg.norm(omega)<1e-8 else np.sin(angle/2)*omega/np.linalg.norm(omega)];future_quat=np.zeros(4);self.mj.mju_mulQuat(future_quat,quat,delta)
            self.d.qpos[:]=self.m.qpos0;self.d.qpos[self.qadr]=obs.joint_position;self.d.qpos[self.base:self.base+7]=np.r_[xyz,future_quat];nominal=result[self.arms].copy();actual=np.asarray(obs.joint_position)[self.arms];dq=np.asarray(obs.joint_velocity)[self.arms];toward=nominal-actual;predicted=actual+np.clip(dq*horizon,np.minimum(0.,toward),np.maximum(0.,toward));_,reference_R,_=self.geometry(nominal)
            if self.up_target:
                for i,g in enumerate(self.pads):
                    if i not in self.orientation_sides:continue
                    body_R=self.d.xmat[int(self.m.geom_bodyid[g])].reshape(3,3);n=body_R[:,1]*(1 if i==0 else -1);desired_n=n.copy();desired_n[:2]*=np.sqrt(max(0.,1-self.up_target**2))/max(np.linalg.norm(n[:2]),1e-8);desired_n[2]=self.up_target;axis=np.cross(n,desired_n);a=np.linalg.norm(axis)
                    if a>1e-8:
                        axis/=a;angle=np.arctan2(a,np.dot(n,desired_n));K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]]);reference_R[i]=(np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K))@reference_R[i]
            P,R,J=self.geometry(predicted,True);pelvisR=self.d.body('pelvis').xmat.reshape(3,3);cup=P.mean(0)+pelvisR@self.offset;error=np.asarray(pred.position_m)-cup;error*=min(1.,self.limit/max(np.linalg.norm(error),1e-8));A=np.vstack(J);rhs=np.concatenate([np.r_[self.gain*error,.20*self.orientation_gain*(1 if i in self.orientation_sides else 0)*.5*sum(np.cross(R[i][:,k],reference_R[i][:,k])for k in range(3))]for i in range(2)]);reg=np.full(14,.02**2);reg[[5,6,12,13]]*=100.;increment=np.linalg.solve(A.T@A+np.diag(reg),A.T@rhs);increment[[5,6,12,13]]=0.;proposed=np.clip(self.previous+increment,-self.q_limit,self.q_limit);lo=np.maximum(nominal-self.q_limit,self.ranges[:,0]+.03);hi=np.minimum(nominal+self.q_limit,self.ranges[:,1]-.03);lo[0]=max(lo[0],0.);hi[7]=min(hi[7],0.)
            for scale in [1.,.5,.25,0.]:
                candidate=np.clip(nominal+self.previous+scale*(proposed-self.previous),lo,hi)
                clear=self.clearance(candidate)
                form_ok=True
                if self.orientation_gain:
                    points,_,_=self.geometry(candidate);ups=[float(self.d.xmat[int(self.m.geom_bodyid[g])].reshape(3,3)[2,1]*(1 if i==0 else -1))for i,g in enumerate(self.pads)];heights=[float(points[i,2]-self.d.body(side+'_shoulder_pitch_link').xpos[2])for i,side in enumerate(['left','right'])];measured_form_ok=min(ups)>=.65 and max(heights)<=.14;form_ok=True if self.up_target else measured_form_ok
                    diag.update(command_palm_up_cos=ups,command_height_above_shoulder_m=heights,**command_form_diagnostic(self.up_target,ups,heights))
                if clear>=.002 and form_ok:desired=candidate-nominal;break
            diag.update(status='relative_forecast_correction',ttc_s=ttc,forecast_time_s=float(pred.timestamp_s),forecast_age_s=now-pred.last_inlier_timestamp_s,forecast_std_s=float(pred.time_std_s),ball_intercept_rgbd_m=np.asarray(pred.position_m).tolist(),predicted_cup_center_m=cup.tolist(),forecast_error_m=error.tolist(),predicted_arm_q=predicted.tolist(),observed_arm_q=actual.tolist(),nominal_arm_target=nominal.tolist(),freeze_at_s=self.freeze_at,absolute_orientation_gain=self.orientation_gain)
        else:diag.update(status='abstain_no_fresh_pair_or_latephase',ttc_s=ttc,pose_valid=bool(valid),forecast_valid=bool(forecast_valid))
        if self.gain:
            self.previous+=np.clip(desired-self.previous,-self.rate*dt,self.rate*dt);result[self.arms]=np.clip(result[self.arms]+self.previous,*self.ranges.T)
        diag['correction_rad']=self.previous.tolist();self.rows.append(diag);return result
