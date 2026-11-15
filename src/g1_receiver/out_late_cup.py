"""Camera-caused late reach with a preoriented compact waypoint.

Actor inputs are timestamped RGB-D, surveyed receiver marker pose, body encoders
and IMU only. A separate static model is used for FK. No simulator state,
contacts, fixture release clock or hand-ready event is accepted.
"""
from collections import deque
import math
import numpy as np
from .perception import ColorBallDetector, BallTracker

AXES=('shoulder_roll','shoulder_yaw','shoulder_pitch','elbow','wrist_roll','wrist_pitch','wrist_yaw')
PACKET_KEYS=('compact_error_rad','receive_error_rad','arm_velocity_rad_s','gravity_body','imu_gyro_rad_s','ttc_s','ttc_std_s','forecast_valid','phase','phase_rate_s_inv','visual_valid','ball_cup_error_body_m','ball_cup_relative_velocity_body_mps','ball_cup_sigma_m','visual_dwell_s')

def quintic(x):
    x=float(np.clip(x,0.,1.));return x*x*x*(10.-15.*x+6.*x*x)

class OutLateCupReach:
    def __init__(self,config,phase_policy=None):
        self.config=dict(config);self.enabled=bool(config.get('enabled',False));self.policy=phase_policy
        self.compact=np.asarray(config['compact_arm_targets_rad'],float);self.receive=np.asarray(config['receive_arm_targets_rad'],float)
        if self.compact.shape!=(14,)or self.receive.shape!=(14,)or not np.isfinite([self.compact,self.receive]).all():raise ValueError('finite14D compact/receive waypoints required')
        if np.max(abs(self.compact[[4,5,6,11,12,13]]-self.receive[[4,5,6,11,12,13]]))>.30000001:raise ValueError('receiving wrists must be preoriented within.30rad')
        self.duration=float(config.get('duration_s',.5));self.cue=float(config.get('cue_ttc_s',.65));self.vmax=float(config.get('max_command_velocity_rad_s',4.));self.amax=float(config.get('max_command_acceleration_rad_s2',30.));self.rate_accel=2.;self.delta=float(abs(self.receive-self.compact).max())
        if not .35<=self.duration<=.8 or not .4<=self.cue<=.75 or not 0<self.delta<=1. or not 0<self.vmax<=4. or not 0<self.amax<=30.:raise ValueError('bounded reach duration/cue/joint motion required')
        self.rate_max=min(self.vmax/(1.875*self.delta),math.sqrt(max(0.,self.amax/self.delta-1.875*self.rate_accel)/5.773503))
        if 1./self.duration>self.rate_max+1e-8:raise ValueError('nominal quintic violates declared joint velocity/acceleration bounds')
        self.plane=np.asarray(config['catch_plane_point_m'],float);axis=np.asarray(config['catch_plane_normal_xy'],float);self.axis=np.r_[axis/np.linalg.norm(axis),0.]
        self.offset=np.asarray(config.get('ball_center_offset_body_m',[0.,0.,.06]),float)
        if self.plane.shape!=(3,)or self.offset.shape!=(3,)or not np.isfinite([self.plane,self.offset]).all():raise ValueError('finite declared aperture command required')
        self.detector=ColorBallDetector(max_depth_m=20.,ball_center_offset_m=.075)
        # Numerical timestamp tolerance is local to this observer; the scripted grip tracker is unchanged.
        self.tracker=BallTracker(max_age_s=.08+1e-8,max_horizon_s=.9,allow_synthetic_fixture=True)
        self.last_ball=None;self.motion_seen=False;self.ball_history=deque(maxlen=6);self.pose_history=deque(maxlen=6);self.cup_history=deque(maxlen=6);self.last_pose_stamp=None;self.last_capture=None;self.phase=0.;self.phase_rate=0.;self.triggered_at=None;self.forecast_at_trigger=None;self.last_now=None;self.last_dwell_stamp=None;self.dwell=0.;self.rows=[];self.last_packet=None;self.latest_forecast=None;self.preparation_fraction=0.;self.preparation_reference=None
    def bind(self,model,joint_names):
        import mujoco
        self.mj=mujoco;self.model=model;self.data=mujoco.MjData(model);self.names=tuple(joint_names);self.qadr=np.array([int(model.joint(n).qposadr[0])for n in self.names]);self.base=int(model.joint('floating_base_joint').qposadr[0]);self.arms=np.array([self.names.index(f'{s}_{a}_joint')for s in ['left','right']for a in AXES]);self.ranges=np.array([model.joint(self.names[i]).range for i in self.arms]);
        if np.any(self.compact<self.ranges[:,0])or np.any(self.compact>self.ranges[:,1])or np.any(self.receive<self.ranges[:,0])or np.any(self.receive>self.ranges[:,1]):raise ValueError('waypoint outside mechanical joint ranges')
    def initial_upper(self,reference):
        result=np.asarray(reference,float).copy()
        if self.enabled and self.config.get('compact_at_reset',False):result[self.arms]=self.compact
        return result
    @staticmethod
    def velocity(history):
        rows=np.asarray(history,float)
        if len(rows)<3 or np.ptp(rows[:,0])<.055:return np.zeros(3)
        t=rows[:,0]-rows[:,0].mean();return t@rows[:,1:]/(t@t)
    def fk(self,obs,pose,now):
        if pose is None or pose.z_m is None or pose.z_std_m is None or not pose.fresh(now,.08+1e-8):return None
        xy_std=float(np.sqrt(np.linalg.eigvalsh(np.asarray(pose.covariance)[:2,:2]).max()))
        if xy_std>.12 or pose.z_std_m>.06:return None
        if self.last_pose_stamp is None or pose.timestamp_s>self.last_pose_stamp+1e-8:
            self.pose_history.append((float(pose.timestamp_s),*pose.xy_m,float(pose.z_m)));self.last_pose_stamp=float(pose.timestamp_s)
        xyz=np.r_[pose.xy_m,pose.z_m]+self.velocity(self.pose_history)*max(0.,now-pose.timestamp_s)
        d=self.data;d.qpos[:]=self.model.qpos0;d.qpos[self.qadr]=obs.joint_position;quat=np.asarray(obs.imu_quat_wxyz,float);quat/=np.linalg.norm(quat);d.qpos[self.base:self.base+7]=np.r_[xyz,quat];self.mj.mj_kinematics(self.model,d);R=d.body('pelvis').xmat.reshape(3,3).copy();points=[];up=[]
        for side,sign in [('left',1),('right',-1)]:
            g=int(self.model.geom('aero_'+side+'_palm_collision_7').id);r=d.geom_xmat[g].reshape(3,3);points.append(d.geom_xpos[g]+self.model.geom_size[g,0]*r[:,0]);up.append(float(sign*d.body('aero_'+side+'_palm').xmat.reshape(3,3)[2,1]))
        cup=np.mean(points,axis=0)+R@self.offset;self.cup_history.append((now,*cup));source_cov=np.asarray(pose.covariance);cov=np.zeros((3,3));cov[:2,:2]=source_cov[:2,:2];cov[2,2]=pose.z_std_m**2;cov+=np.eye(3)*(.005**2+(1.*max(0.,now-pose.timestamp_s))**2);return cup,R,cov,up
    def step(self,obs,pose,frame,survey,reference_target,alternate_capture=None):
        now=float(obs.timestamp);dt=.02 if self.last_now is None else now-self.last_now;self.last_now=now
        if not 0<dt<=.041:raise ValueError('late reach requires causal50Hz sensor/control clock')
        ball=None;captures=[(frame,survey)]
        if alternate_capture is not None:captures.append(alternate_capture)
        for f,s in captures:
            if f is None or s is None or not -1e-8<=now-f.timestamp_s<=.08+1e-8:continue
            detected=self.detector.detect(f,s)
            if detected is not None:
                ball=detected;break
        if ball is not None and (self.last_capture is None or ball.timestamp_s>self.last_capture+1e-8):
            self.last_capture=float(ball.timestamp_s)
            if not self.motion_seen and self.last_ball is not None:
                delta=ball.timestamp_s-self.last_ball.timestamp_s
                if .005<=delta<=.081 and np.linalg.norm(ball.position_m-self.last_ball.position_m)/delta>=2.:
                    self.motion_seen=True;self.tracker.add(self.last_ball)
            if self.motion_seen:self.tracker.add(ball)
            self.last_ball=ball;self.ball_history.append((float(ball.timestamp_s),*ball.position_m))
        prediction=self.tracker.intercept(now,self.plane,self.axis,'field')if self.motion_seen else None
        valid=bool(prediction is not None and prediction.inliers>=3 and prediction.last_inlier_timestamp_s is not None and -1e-8<=now-prediction.last_inlier_timestamp_s<=.08+1e-8 and prediction.time_std_s<=.06 and np.sqrt(np.linalg.eigvalsh(prediction.covariance_m2).max())<=.15)
        if valid:self.latest_forecast=float(prediction.timestamp_s)
        ttc=None if not valid else float(prediction.timestamp_s-now);fk=self.fk(obs,pose,now);visual=False;velocity_valid=False;error=np.zeros(3);relative=np.zeros(3);sigma=np.ones(3);palm_up=[None,None]
        if fk is not None and ball is not None and abs(now-ball.timestamp_s)<=1e-8:
            cup,R,pose_cov,palm_up=fk;error=R.T@(ball.position_m-cup);relative=R.T@(self.velocity(self.ball_history)-self.velocity(self.cup_history));combined_cov=pose_cov+ball.covariance_m2;sigma=np.full(3,np.sqrt(max(0.,float(np.linalg.eigvalsh(combined_cov).max()))));visual=bool(np.max(sigma)<=.06)
            velocity_valid=all(len(h)>=3 and np.ptp(np.asarray(h)[:,0])>=.055 and np.diff(np.asarray(h)[:,0]).max()<=.08+1e-8 and -1e-8<=now-h[-1][0]<=.08+1e-8 for h in [self.ball_history,self.cup_history])
            in_cup=visual and velocity_valid and np.max(abs(error))<=.12 and np.linalg.norm(relative)<=.8
            if in_cup and self.last_dwell_stamp is not None and ball.timestamp_s-self.last_dwell_stamp<=.08+1e-8:self.dwell+=max(0.,ball.timestamp_s-self.last_dwell_stamp)
            else:self.dwell=0.
            self.last_dwell_stamp=float(ball.timestamp_s)if in_cup else None
        else:self.dwell=0.;self.last_dwell_stamp=None
        quat=np.asarray(obs.imu_quat_wxyz,float);from .observable import project_gravity
        armq=np.asarray(obs.joint_position)[self.arms];armdq=np.asarray(obs.joint_velocity)[self.arms]
        packet={'time_s':now,'compact_error_rad':(self.compact-armq).tolist(),'receive_error_rad':(self.receive-armq).tolist(),'arm_velocity_rad_s':armdq.tolist(),'gravity_body':project_gravity(quat).tolist(),'imu_gyro_rad_s':np.asarray(obs.imu_gyro).tolist(),'ttc_s':ttc,'ttc_std_s':None if not valid else float(prediction.time_std_s),'forecast_valid':valid,'phase':float(self.phase),'phase_rate_s_inv':float(self.phase_rate),'visual_valid':visual,'ball_cup_error_body_m':error.tolist(),'ball_cup_relative_velocity_body_mps':relative.tolist(),'ball_cup_sigma_m':sigma.tolist(),'visual_dwell_s':float(self.dwell),'visual_velocity_valid':bool(visual and velocity_valid),'upright_cos':float(1.-2.*(quat[1]**2+quat[2]**2)),'ball_capture_s':None if ball is None else float(ball.timestamp_s),'receiver_pose_timestamp_s':None if pose is None else float(pose.timestamp_s),'receiver_xy_yaw_covariance_mixed_units':None if pose is None else np.asarray(pose.covariance).tolist(),'receiver_xy_covariance_m2':None if pose is None else np.asarray(pose.covariance)[:2,:2].tolist(),'receiver_z_std_m':None if pose is None else pose.z_std_m,'ball_inliers':0 if prediction is None else int(prediction.inliers),'palm_up_cos':palm_up,'actual_target_rad':None,'compact_arm_targets_rad':self.compact.tolist(),'receive_arm_targets_rad':self.receive.tolist()}
        action=0. if self.policy is None else float(self.policy(dict(packet)))
        if not np.isfinite(action)or not -1.000001<=action<=1.000001:raise ValueError('phase policy returns one finite scalar in[-1,1]')
        if self.enabled and self.triggered_at is None and valid and 0<ttc<=self.cue:
            self.triggered_at=now;self.forecast_at_trigger=float(prediction.timestamp_s);self.phase_rate=1./self.duration
        if self.enabled and self.triggered_at is not None and self.phase<1:
            desired=min(self.rate_max,(1.+.35*float(np.clip(action,-1,1)))/self.duration);self.phase_rate+=float(np.clip(desired-self.phase_rate,-self.rate_accel*dt,self.rate_accel*dt));self.phase=min(1.,self.phase+dt*self.phase_rate)
        alpha=quintic(self.phase);target=np.asarray(reference_target,float).copy()
        if self.enabled:
            compact_target=self.compact+alpha*(self.receive-self.compact)
            target[self.arms]=(np.asarray(self.preparation_reference)[self.arms] if self.preparation_reference is not None else np.asarray(reference_target)[self.arms])+self.preparation_fraction*(compact_target-(np.asarray(self.preparation_reference)[self.arms] if self.preparation_reference is not None else np.asarray(reference_target)[self.arms])) if self.triggered_at is None else compact_target
        packet.update(phase_after=float(self.phase),phase_rate_after=float(self.phase_rate),alpha=alpha,triggered_at_s=self.triggered_at,forecast_at_trigger_s=self.forecast_at_trigger,phase_policy_action=action,actual_target_rad=target[self.arms].tolist(),enabled=self.enabled,preparation_fraction=float(self.preparation_fraction))
        self.rows.append(packet);self.last_packet=packet;return target
