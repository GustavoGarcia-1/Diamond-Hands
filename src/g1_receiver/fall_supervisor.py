"""Simulation-only IMU protective supervision, without pose/contact oracles.

Balance abort stops the commanded walk and retains the existing WBC. Crouch/tuck
is a separate failed/successful research candidate, not a learned get-up skill.
"""
from dataclasses import dataclass, replace
import numpy as np

@dataclass(frozen=True)
class FallConfig:
    mode: str = 'balance_abort'
    severe_upright_cos: float = .95
    warning_upright_cos: float = .985
    warning_gyro_rad_s: float = .35
    confirm_samples: int = 2
    max_target_velocity_rad_s: float = 2.5
    max_target_acceleration_rad_s2: float = 10.

class FallSupervisor:
    def __init__(self,names,joint_ranges,nominal,config=FallConfig()):
        if config.mode not in ('balance_abort','crouch_tuck'):raise ValueError('unknown supervision mode')
        self.names=tuple(names);self.ranges=np.asarray(joint_ranges,float);self.config=config;self.protective=np.asarray(nominal,float).copy()
        for side,sign in [('left',1),('right',-1)]:
            for joint,value in [('hip_pitch',-.65),('hip_roll',0.),('hip_yaw',0.),('knee',1.3),('ankle_pitch',-.65),('ankle_roll',0.),('shoulder_pitch',-.30),('shoulder_roll',sign*.20),('shoulder_yaw',0.),('elbow',1.25),('wrist_roll',0.),('wrist_pitch',0.),('wrist_yaw',0.)]:self.protective[self.names.index(side+'_'+joint+'_joint')]=value
        for n in ['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']:self.protective[self.names.index(n)]=0.
        self.protective=np.clip(self.protective,self.ranges[:,0],self.ranges[:,1]);self.last_time=None;self.count=0;self.triggered_at=None;self.previous=None;self.start=None;self.transition_s=None;self.rows=[];self.latest=None
    def observe(self,obs):
        now=float(obs.timestamp)
        if self.last_time is not None and abs(now-self.last_time)<1e-10:return self.triggered_at is not None
        dt=.02 if self.last_time is None else now-self.last_time
        if not 0<dt<=.041:raise ValueError('causal50Hz supervision required')
        q=np.asarray(obs.imu_quat_wxyz,float).copy();g=np.asarray(obs.imu_gyro,float)
        if q.shape!=(4,)or g.shape!=(3,)or not np.isfinite(np.r_[now,q,g]).all()or np.linalg.norm(q)<1e-8:raise ValueError('finite timestamped IMU required')
        q/=np.linalg.norm(q);up=float(1.-2.*(q[1]**2+q[2]**2));rate=float(np.linalg.norm(g[:2]));warning=up<self.config.severe_upright_cos or(up<self.config.warning_upright_cos and rate>self.config.warning_gyro_rad_s);self.count=self.count+1 if warning else 0
        if self.triggered_at is None and self.count>=self.config.confirm_samples:self.triggered_at=now
        self.last_time=now;self.latest=(now,up,rate,warning);return self.triggered_at is not None
    def apply(self,obs,command):
        self.observe(obs);target=np.asarray(command.target_position,float)
        if self.triggered_at is not None and self.config.mode=='crouch_tuck':
            if self.start is None:
                self.start=(target if self.previous is None else self.previous).copy();distance=float(abs(self.protective-self.start).max());self.transition_s=max(.35,1.875*distance/self.config.max_target_velocity_rad_s,np.sqrt(5.773503*distance/self.config.max_target_acceleration_rad_s2))
            u=float(np.clip((obs.timestamp-self.triggered_at)/self.transition_s,0.,1.));alpha=u**3*(10.-15.*u+6.*u*u);target=self.start+alpha*(self.protective-self.start);command=replace(command,target_position=np.clip(target,self.ranges[:,0],self.ranges[:,1]),status='imu_crouch_tuck_abort')
        elif self.triggered_at is not None:command=replace(command,status='imu_balance_abort')
        self.previous=np.asarray(command.target_position).copy();now,up,rate,warning=self.latest;self.rows.append(dict(time_s=now,imu_upright_cos=up,imu_tilt_rate_rad_s=rate,warning=warning,triggered_at_s=self.triggered_at,mode=self.config.mode,target_rad=self.previous.tolist()));return command

@dataclass(frozen=True)
class StylizedFallConfig:
    """New simulation research mode; existing FallConfig/defaults are untouched."""
    near_cup_m: float = .55
    escaped_cup_m: float = .60
    below_cup_m: float = -.12
    max_sensor_sigma_m: float = .06
    miss_confirm_frames: int = 2
    severe_upright_cos: float = .95
    tilt_rate_rad_s: float = 1.2
    max_target_velocity_rad_s: float = 3.
    max_target_acceleration_rad_s2: float = 12.
    max_target_jerk_rad_s3: float = 100.
    tuck_s: float = .50
    roll_s: float = .85
    seat_s: float = 1.15


class VisualMissGate:
    """RGB-D escape estimate, not a contact or possession oracle."""
    def __init__(self, model, joint_names, config=StylizedFallConfig()):
        from .native_cup_residual_rl import NativeCupSensors
        from .perception import ColorBallDetector
        self.fk=NativeCupSensors(model,joint_names)
        self.detector=ColorBallDetector(max_depth_m=20.,ball_center_offset_m=.075)
        self.config=config;self.last_capture=None;self.previous=None;self.seen_near=False
        self.count=0;self.armed_at=None;self.rows=[]

    def update(self,obs,pose,frame,survey,enabled):
        if frame is None or survey is None:return self.armed_at is not None
        capture=float(frame.timestamp_s)
        if self.last_capture is not None and capture<=self.last_capture+1e-10:return self.armed_at is not None
        self.last_capture=capture;now=float(obs.timestamp);ball=self.detector.detect(frame,survey)
        valid=(bool(enabled)and ball is not None and pose is not None and pose.z_m is not None
               and pose.z_std_m is not None and pose.fresh(now,.08+1e-8)
               and -1e-8<=now-capture<=.08+1e-8 and abs(capture-pose.timestamp_s)<=.04+1e-8
               and ball.frame_id==pose.frame_id=='field'and ball.pose_provenance==pose.pose_provenance)
        row=dict(time_s=now,capture_s=capture,valid=False,seen_near=self.seen_near,miss_armed_at_s=self.armed_at)
        if valid:
            g=self.fk.geometry(obs.joint_position,obs.imu_quat_wxyz);Y=g['world_to_receiver']
            cov=np.asarray(ball.covariance_m2).copy();cov[:2,:2]+=np.asarray(pose.covariance)[:2,:2]
            cov[2,2]+=pose.z_std_m**2;cov=Y@cov@Y.T+np.eye(3)*.01**2
            sigma=np.sqrt(np.maximum(0.,np.diag(cov)))
            valid=bool(np.isfinite(cov).all()and np.linalg.eigvalsh(cov).min()>=-1e-10 and sigma.max()<=self.config.max_sensor_sigma_m)
            if valid:
                relative=Y@(ball.position_m-np.r_[pose.xy_m,pose.z_m]);cup=g['faces_receiver'].mean(0);error=relative-cup
                distance=float(np.linalg.norm(error));self.seen_near|=distance<=self.config.near_cup_m
                outward=False;velocity=np.zeros(3)
                if self.previous is not None:
                    dt=capture-self.previous[0]
                    if 0<dt<=.10:
                        velocity=(error-self.previous[1])/dt;outward=bool(float(error@velocity)>.10)
                escaped=bool(self.seen_near and distance>=self.config.escaped_cup_m and error[2]<=self.config.below_cup_m and outward)
                self.count=self.count+1 if escaped else 0
                if self.armed_at is None and self.count>=self.config.miss_confirm_frames:self.armed_at=now
                self.previous=(capture,error.copy())
                row.update(valid=True,error_to_measured_palm_midpoint_m=error.tolist(),velocity_mps=velocity.tolist(),sigma_m=sigma.tolist(),distance_m=distance,outward=outward,escaped=escaped,seen_near=self.seen_near,miss_armed_at_s=self.armed_at)
        if not valid:self.count=0;self.previous=None
        self.rows.append(row);return self.armed_at is not None


class StylizedFallSupervisor:
    """Hand-authored tuck/side-roll/seated target requests after visual miss + IMU.

    Only body encoders, IMU and a separate VisualMissGate arm this controller.
    Root poses, contacts and forces are not accepted. End poses are objectives,
    never written into simulator state. All actuator gains/caps remain external.
    """
    def __init__(self,names,joint_ranges,nominal,config=StylizedFallConfig()):
        self.names=tuple(names);self.ranges=np.asarray(joint_ranges,float);self.config=config
        nominal=np.asarray(nominal,float);self.goals=[]
        settings=[
            dict(hip_pitch=-.40,knee=.80,ankle_pitch=-.40,hip_roll=0.,waist_roll=0.,waist_pitch=.10,left_shoulder_roll=.40,right_shoulder_roll=-.40,left_shoulder_pitch=-.65,right_shoulder_pitch=-.65,left_elbow=1.4,right_elbow=1.4),
            dict(hip_pitch=-.65,knee=1.25,ankle_pitch=-.45,hip_roll=-.18,waist_roll=-.20,waist_pitch=.15,left_shoulder_roll=.65,right_shoulder_roll=-.30,left_shoulder_pitch=-.35,right_shoulder_pitch=-.65,left_elbow=1.2,right_elbow=1.4),
            dict(hip_pitch=-1.40,knee=2.30,ankle_pitch=-.60,hip_roll=0.,waist_roll=-.15,waist_pitch=.15,left_shoulder_roll=.55,right_shoulder_roll=-.25,left_shoulder_pitch=.15,right_shoulder_pitch=-.40,left_elbow=.70,right_elbow=1.3)]
        for values in settings:
            q=nominal.copy()
            for key,value in values.items():
                joints=[key+'_joint']if key.startswith(('left_','right_','waist_'))else [s+'_'+key+'_joint'for s in('left','right')]
                for joint in joints:q[self.names.index(joint)]=value
            self.goals.append(np.clip(q,self.ranges[:,0]+.02,self.ranges[:,1]-.02))
        self.miss_armed_at=None;self.triggered_at=None;self.count=0;self.rows=[]
        self.previous=None;self.velocity=np.zeros(29);self.acceleration=np.zeros(29);self.last_time=None;self.start=None

    def arm_after_visual_miss(self,stamp):
        if self.miss_armed_at is None:self.miss_armed_at=float(stamp)

    def apply(self,obs,command):
        now=float(obs.timestamp);dt=.02 if self.last_time is None else now-self.last_time
        if not 0<dt<=.041:raise ValueError('causal50Hz stylized supervision required')
        quat=np.asarray(obs.imu_quat_wxyz,float);gyro=np.asarray(obs.imu_gyro,float)
        if quat.shape!=(4,)or gyro.shape!=(3,)or not np.isfinite(np.r_[quat,gyro]).all():raise ValueError('finite measuredIMU required')
        quat=quat/np.linalg.norm(quat);up=float(1-2*(quat[1]**2+quat[2]**2));warning=up<self.config.severe_upright_cos or np.linalg.norm(gyro[:2])>self.config.tilt_rate_rad_s
        self.count=self.count+1 if self.miss_armed_at is not None and warning else 0
        if self.triggered_at is None and self.count>=2:self.triggered_at=now;self.start=np.asarray(command.target_position).copy()
        scripted=np.asarray(command.target_position,float);target=scripted.copy();phase='waiting_for_observed_miss_and_tilt'
        if self.previous is None:self.previous=scripted.copy()
        if self.triggered_at is None:
            velocity=(target-self.previous)/dt;self.acceleration=(velocity-self.velocity)/dt;self.velocity=velocity
        else:
            elapsed=now-self.triggered_at;durations=[self.config.tuck_s,self.config.roll_s,self.config.seat_s];begin=self.start.copy();offset=0.;desired=self.goals[-1].copy();phase='braced_seated_request'
            for k,duration in enumerate(durations):
                if elapsed<=offset+duration:
                    u=float(np.clip((elapsed-offset)/duration,0,1));alpha=u*u*u*(10+u*(-15+6*u));desired=begin+alpha*(self.goals[k]-begin);phase=['tuck_request','left_side_roll_request','braced_seated_request'][k];break
                begin=self.goals[k];offset+=duration
            error=desired-self.previous;vmax=self.config.max_target_velocity_rad_s;amax=self.config.max_target_acceleration_rad_s2
            wanted_v=np.sign(error)*np.minimum(vmax,np.sqrt(2*amax*np.abs(error)));wanted_v=np.clip(wanted_v,-np.abs(error)/dt,np.abs(error)/dt)
            wanted_a=np.clip((wanted_v-self.velocity)/dt,-amax,amax)
            acceleration=self.acceleration+np.clip(wanted_a-self.acceleration,-self.config.max_target_jerk_rad_s3*dt,self.config.max_target_jerk_rad_s3*dt)
            acceleration=np.clip(acceleration,-amax,amax);velocity=np.clip(self.velocity+acceleration*dt,-vmax,vmax)
            old_velocity=self.velocity.copy()
            target=np.clip(self.previous+velocity*dt,self.ranges[:,0]+.01,self.ranges[:,1]-.01)
            self.velocity=(target-self.previous)/dt;self.acceleration=(self.velocity-old_velocity)/dt
            command=replace(command,target_position=target,status='visual_miss_stylized_'+phase)
        self.previous=target.copy();self.last_time=now;self.rows.append(dict(time_s=now,miss_armed_at_s=self.miss_armed_at,triggered_at_s=self.triggered_at,imu_upright_cos=up,phase=phase,target_rad=target.tolist(),requested_goals_not_forced=True))
        return command
