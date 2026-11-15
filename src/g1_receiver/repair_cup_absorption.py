"""Experimental larger vertical cup stroke from surveyed RGB-D and encoder/IMU FK only.

No fixture time, contact, ball truth, active MjData, force or reward input.
Isolated repair branch: world-Z motion, <=180mm stroke, <=0.60rad residual,
<=4.2rad/s residual slew; official host/actuator gains and limits unchanged.
The declared catch plane is a command. A private FK/Jacobian translates both
palms together while penalizing orientation changes; official gains stay fixed.
"""
from collections import deque
import numpy as np
from .perception import ColorBallDetector,BallTracker

ARM_AXES=('shoulder_roll','shoulder_yaw','shoulder_pitch','elbow','wrist_roll','wrist_pitch','wrist_yaw')

def bounded_joint_step(desired,previous,dt):
    desired=np.asarray(desired,float);previous=np.asarray(previous,float)
    if desired.shape!=(14,)or previous.shape!=(14,)or not np.isfinite([desired,previous]).all()or not 0.<dt<=.041:
        raise ValueError('finite14D and causal control interval required')
    norm=float(abs(desired).max());desired=desired*min(1.,.60/max(1e-12,norm));delta=desired-previous;delta*=min(1.,4.2*dt/max(1e-12,float(abs(delta).max())))
    return previous+delta

class VerticalCupStroke:
    def __init__(self,profile,gain=.35,max_give_m=.035,lead_s=.08,deceleration_s=.12,recovery_s=.35,pulse_lead_s=None,actuator_latency_s=0.):
        if not 0<=gain<=1 or not 0<max_give_m<=.18 or not .02<=lead_s<=.15 or not .04<=deceleration_s<=.25 or not .1<=recovery_s<=.8:
            raise ValueError('bounded give geometry and timing required')
        if pulse_lead_s is not None and not .01<=pulse_lead_s<=.03:raise ValueError("future pulse lead must be10..30ms")
        if not 0.<=actuator_latency_s<=.12:raise ValueError('declared actuator lead must be0..120ms')
        self.actuator_latency=float(actuator_latency_s)
        self.pulse_lead=pulse_lead_s
        self.gain=gain;self.max_give=max_give_m;self.lead=lead_s;self.deceleration=deceleration_s;self.recovery=recovery_s;play=profile['play'];self.plane=np.asarray(play['field_catch_window_m'],float);axis=np.asarray(play['field_approach_axis_xy'],float);self.axis=np.r_[axis/np.linalg.norm(axis),0.];self.detector=ColorBallDetector(max_depth_m=20.,ball_center_offset_m=.075);self.tracker=BallTracker(max_age_s=.08,max_horizon_s=.55,allow_synthetic_fixture=True);self.pose_history=deque(maxlen=6);self.last_capture=None;self.last_now=None;self.previous=np.zeros(14);self.plan=None;self.rows=[];self.model=None
    def bind(self,model,joint_names):
        import mujoco
        self.mj=mujoco;self.model=model;self.data=mujoco.MjData(model);self.names=tuple(joint_names);self.qadr=np.array([int(model.joint(n).qposadr[0])for n in self.names]);self.base=int(model.joint('floating_base_joint').qposadr[0]);self.arms=np.array([self.names.index(f'{s}_{a}_joint')for s in ('left','right')for a in ARM_AXES]);self.dofs=np.array([int(model.joint(self.names[i]).dofadr[0])for i in self.arms]);self.ranges=np.array([model.joint(self.names[i]).range.copy()for i in self.arms]);self.palms=[int(model.body('aero_'+s+'_palm').id)for s in ('left','right')]
    def _fk(self,obs):
        d=self.data;d.qpos[:]=self.model.qpos0;d.qvel[:]=0.;imu=np.asarray(obs.imu_quat_wxyz,float);d.qpos[self.base:self.base+7]=np.r_[np.zeros(3),imu/np.linalg.norm(imu)];d.qpos[self.qadr]=obs.joint_position;self.mj.mj_kinematics(self.model,d);self.mj.mj_comPos(self.model,d);normals=[];Jp=[];Jr=[]
        for bid,sign,side in zip(self.palms,(1.,-1.),('left','right')):
            R=d.xmat[bid].reshape(3,3);normal=sign*R[:,1];point=d.xpos[bid]+R@np.array([-.0025,sign*.017,.07225]);a=np.zeros((3,self.model.nv));b=np.zeros_like(a);self.mj.mj_jac(self.model,d,a,b,point,bid);Jp.append(a[:,self.dofs]);Jr.append(b[:,self.dofs]);normals.append(normal)
        n=np.mean(normals,axis=0);n/=max(1e-12,np.linalg.norm(n));return n,np.vstack([*Jp,*[.12*x for x in Jr]])
    def __call__(self,obs,pose,frame,survey,target,closure,enabled):
        now=float(obs.timestamp);dt=.02 if self.last_now is None else now-self.last_now;self.last_now=now
        if not 0<dt<=.041:raise ValueError('give requires causal50Hz control')
        normal,J=self._fk(obs);normal=np.array([0.,0.,1.]);fresh=False;prediction=None;vn=None;diagnostic=dict(frame=frame is not None,survey=survey is not None,pose=pose is not None)
        if pose is not None:
            diagnostic.update(pose_frame=pose.frame_id,pose_age_s=now-pose.timestamp_s,z_std_m=pose.z_std_m,xy_std_m=float(np.sqrt(np.linalg.eigvalsh(pose.covariance).max())))
        if frame is not None:diagnostic["ball_frame_age_s"]=now-frame.timestamp_s
        if (frame is not None and survey is not None and pose is not None and pose.z_m is not None and pose.z_std_m is not None and pose.frame_id=='field'and pose.fresh(now,.08+1e-8)and -1e-8<=now-frame.timestamp_s<=.08+1e-8 and abs(frame.timestamp_s-pose.timestamp_s)<=.04+1e-8 and pose.z_std_m<=.06 and np.sqrt(np.linalg.eigvalsh(pose.covariance).max())<=.12):
            diagnostic["pose_gate_pass"]=True
            if self.last_capture is None or frame.timestamp_s>self.last_capture:
                self.last_capture=float(frame.timestamp_s);ball=self.detector.detect(frame,survey)
                diagnostic["ball_detected"]=ball is not None
                diagnostic["matching_provenance"]=ball is not None and ball.pose_provenance==pose.pose_provenance
                if ball is not None and ball.pose_provenance==pose.pose_provenance:
                    self.tracker.add(ball)
                    self.pose_history.append((float(pose.timestamp_s),*pose.xy_m,float(pose.z_m)))
            prediction=self.tracker.intercept(now,self.plane,self.axis,'field')
            diagnostic['prediction_exists']=prediction is not None
            if prediction is not None:diagnostic.update(inliers=prediction.inliers,time_std_s=prediction.time_std_s)
            if prediction is not None and prediction.velocity_mps is not None and prediction.last_inlier_timestamp_s is not None and prediction.time_std_s<=.06 and prediction.inliers>=3 and -1e-8<=now-prediction.last_inlier_timestamp_s<=.08+1e-8:
                fresh=True;body_v=np.zeros(3);history=np.asarray(self.pose_history)
                if len(history)>=3 and np.ptp(history[:,0])>=.055:
                    t=history[:,0]-history[:,0].mean();body_v=t@history[:,1:]/(t@t)
                vn=float((prediction.velocity_mps-body_v)@normal)
        if self.plan is None and enabled and fresh and vn<-.15 and .015<=prediction.timestamp_s-now<=self.lead+self.actuator_latency:
            speed=min(self.gain*(-vn),2.,2*self.max_give/((self.pulse_lead or self.lead)+self.deceleration));self.plan=dict(contact_s=float(prediction.timestamp_s)-self.actuator_latency,predicted_ball_arrival_s=float(prediction.timestamp_s),actuator_latency_s=self.actuator_latency,speed=float(speed),normal=normal.copy(),forecast_s=now,last_capture_s=float(prediction.last_inlier_timestamp_s),time_std_s=float(prediction.time_std_s))
        if self.pulse_lead is not None and self.plan is not None and now<self.plan['contact_s']-self.pulse_lead and fresh:
            if abs(prediction.timestamp_s-self.plan['contact_s'])>max(.02,2*prediction.time_std_s) or vn>=0:self.plan=None
            else:self.plan.update(last_capture_s=float(prediction.last_inlier_timestamp_s),time_std_s=float(prediction.time_std_s))
        if self.pulse_lead is not None and self.plan is not None and self.plan['contact_s']-self.pulse_lead<=now<self.plan['contact_s']:
            age=now-self.plan['last_capture_s'];propagated=self.plan['time_std_s']+.05*age
            if age>.14+1e-8 or propagated>.06:self.plan=None
        offset=np.zeros(3);phase='inactive'
        if self.plan is not None:
            contact=self.plan['contact_s'];speed=self.plan['speed'];t=now-contact;lead=self.pulse_lead or self.lead;u=(t+lead)/lead
            if t< -lead:distance=0.
            elif t<0:distance=.5*speed*lead*u*u;phase='anticipatory_give'
            elif t<self.deceleration:distance=.5*speed*lead+speed*(t-.5*t*t/self.deceleration);phase='decelerate'
            else:
                distance=.5*speed*(lead+self.deceleration);distance*=float(np.clip(1.-(t-self.deceleration-.08)/self.recovery,0,1));phase='recover'
            offset=-self.plan['normal']*min(self.max_give,distance)
        desired=np.zeros(14)
        if enabled and self.gain and self.plan is not None:
            rhs=np.r_[offset,offset,np.zeros(6)];desired=J.T@np.linalg.solve(J@J.T+np.eye(12)*.015**2,rhs)
        self.previous=bounded_joint_step(desired,self.previous,dt);adjusted=np.asarray(target).copy()
        if self.gain:
            adjusted[self.arms]=np.clip(adjusted[self.arms]+self.previous,*self.ranges.T)
        self.rows.append(dict(time_s=now,pulse_lead_s=self.pulse_lead,latched_prediction=None if self.plan is None else {k:v for k,v in self.plan.items() if k!="normal"},diagnostic=diagnostic,enabled=bool(enabled),fresh_rgbd_pose_and_velocity=fresh,arrival_s=None if prediction is None else float(prediction.timestamp_s),normal_velocity_mps=vn,phase=phase,commanded_cup_offset_m=offset.tolist(),actual_joint_residual_rad=self.previous.tolist(),normal_world=normal.tolist()))
        return adjusted,np.broadcast_to(np.asarray(closure,float),(2,)).copy()
