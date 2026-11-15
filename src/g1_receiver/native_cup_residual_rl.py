"""Native20D residual sensor audit: body/IMU, RGB-D, fourteen actuator encoders.

No contact, exact ball/root state or unobserved finger-joint state enters runtime
observation, critic or reward. Palmar face FK uses only29 body encoders. Old111D,
119D and scalar-closure controllers/rewards remain unchanged.
"""
from collections import deque
from pathlib import Path
import json
import numpy as np
from .observable import ARM_JOINTS,project_gravity
from .perception import ColorBallDetector
from .lower_cup_grip import NativeHandCalibration,OPEN
from .aero_residual_rl import EpisodeReturnLedger
from .go_catch_form_rl import CausalCatchContext,VisualReceivePhase,context_vector
from .catch_form import CatchFormEvaluator,CatchFormContext

ACTION_SIZE=20
OBS_SIZE=169
OBS_SCHEMA='native-cup-face-normal-169-v1'
REWARD_SCHEMA='native-cup-visual-v3-potential-debit-form-acquisition'
GROUP_MAP=np.array([0,0,0,0,1,2,2,3,3,3,3,4,5,5])
ACTION_LIMIT=np.r_[np.full(14,.18),np.full(6,.15)]
ACTION_RATE=np.r_[np.full(14,1.5),np.ones(6)]

def bounded_native_residual(action,previous,dt=.02):
    action=np.asarray(action,dtype=float);previous=np.asarray(previous,dtype=float)
    if action.shape!=(20,) or previous.shape!=(20,) or not np.isfinite(action).all() or not np.isfinite(previous).all() or np.any(abs(action)>1.+1e-8):raise ValueError('twenty finite normalized actions required')
    desired=action*ACTION_LIMIT
    return previous+np.clip(desired-previous,-ACTION_RATE*dt,ACTION_RATE*dt)

def scale_observation(vector):
    scales=np.r_[np.full(29,2.),np.full(29,10.),np.ones(3),np.full(3,5.),np.full(3,.25),np.full(3,5.),np.full(3,.06),.08,1.,1.,3.,np.ones(28),np.full(6,.5),np.ones(8),np.ones(20),ACTION_LIMIT,np.ones(2),np.ones(8)]
    if scales.shape!=(169,):raise RuntimeError('native schema scale mismatch')
    return (np.asarray(vector,dtype=np.float32)/scales).astype(np.float32)

class NativeCupSensors:
    """Only explicit sensor packets reach this object; FK has separate state."""
    def __init__(self, model, joint_names, *, radius_m=.1135):
        import mujoco
        self.mj = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.enc = [int(model.joint(n).qposadr[0]) for n in joint_names]
        self.base = int(model.joint('floating_base_joint').qposadr[0])
        self.palms = [model.body('aero_'+s+'_palm').id for s in ('left','right')]
        self.radius_m = float(radius_m)
        if not .085 <= self.radius_m <= .142:
            raise ValueError('declared football radius proxy must lie within known semiaxes')
        self.corrected_potential_reward = True
        self.shoulders = [model.body(s+'_shoulder_pitch_link').id for s in ('left','right')]
        self.face_offsets = [np.array([-.0025,.017,.07225]),np.array([-.0025,-.017,.07225])]
        self.current_geometry = None
        self.form_acquired = False
        self.detector = ColorBallDetector(max_depth_m=20.,ball_center_offset_m=.075)
        self.encoder_history = deque(maxlen=32)
        self.capture_count=0; self.valid_count=0; self.max_dwell=0.; self.min_cup_distance=None
        self.history = deque(maxlen=8)
        self.last_capture = None
        self.last_valid = None
        self.dwell = 0.
        self.paid = np.zeros(2)
        self.potential = None
        self.e = np.zeros(3); self.vel = np.zeros(3); self.sigma = np.zeros(3)
        self.valid = False; self.velocity_valid = False
        self.reward = 0.
        self.last_now = None
        self.upright = 1.

    def update(self, obs, pose, frame, survey):
        now = float(obs.timestamp)
        self.current_geometry = self.geometry(obs.joint_position, obs.imu_quat_wxyz)
        self.encoder_history.append((now,np.asarray(obs.joint_position).copy(),np.asarray(obs.imu_quat_wxyz).copy()))
        gravity = project_gravity(obs.imu_quat_wxyz)
        upright = -float(gravity[2])
        self.upright = upright
        dt = .02 if self.last_now is None else max(0.,now-self.last_now)
        self.last_now = now
        self.reward -= dt*2.*max(0.,.9-upright)
        self.valid = self.last_valid is not None and -1e-8<=now-self.last_valid<=.08+1e-8
        if not self.valid:
            self.dwell=0.; self.history.clear(); self._clear_potential()
        if frame is None or survey is None:
            return
        capture = float(frame.timestamp_s)
        if self.last_capture is not None and capture == self.last_capture:
            return  # a repeated image must not reset, advance or repay dwell
        self.capture_count+=1
        previous_capture = self.last_capture
        self.last_capture = capture
        ball = self.detector.detect(frame,survey)
        ok = (ball is not None and pose is not None and pose.fresh(now,.08+1e-8)
              and -1e-8<=now-capture<=.08+1e-8 and abs(capture-pose.timestamp_s)<=.04+1e-8
              and ball.frame_id=='field' and pose.frame_id=='field'
              and ball.pose_provenance==pose.pose_provenance and pose.z_m is not None
              and pose.z_std_m is not None and (previous_capture is None or capture>previous_capture))
        if not ok:
            self.invalidate(); return
        # All runtime inputs below are measured joints/IMU and RGB-D estimates.
        samples=list(self.encoder_history)
        measured_q=np.asarray(obs.joint_position); measured_imu=np.asarray(obs.imu_quat_wxyz)
        for left,right in zip(samples[:-1],samples[1:]):
            if left[0]<=capture<=right[0]:
                u=(capture-left[0])/max(1e-12,right[0]-left[0])
                measured_q=(1-u)*left[1]+u*right[1]
                rq=right[2] if np.dot(left[2],right[2])>=0 else -right[2]
                measured_imu=(1-u)*left[2]+u*rq
                break
        geometry = self.geometry(measured_q, measured_imu)
        R = geometry['rotation']
        Y = geometry['world_to_receiver']
        palms = geometry['faces_receiver']
        cup = geometry['cup_receiver']
        relative = Y @ (ball.position_m-np.r_[pose.xy_m,pose.z_m])
        cov=np.asarray(ball.covariance_m2).copy()
        cov[:2,:2]+=np.asarray(pose.covariance)[:2,:2]
        cov[2,2]+=pose.z_std_m**2
        cov=Y@cov@Y.T+np.eye(3)*.01**2  # declared 1cm FK calibration uncertainty
        if (not np.isfinite(cov).all() or not np.allclose(cov,cov.T)
                or np.linalg.eigvalsh(cov).min()< -1e-10):
            self.invalidate(); return
        self.sigma=np.sqrt(np.maximum(0.,np.diag(cov)))
        if self.sigma.max()>.06:
            self.invalidate(); return
        self.e=relative-cup
        if np.linalg.norm(self.e)>.35:self.form_acquired=False
        gap=None if previous_capture is None else capture-previous_capture
        if gap is not None and gap>.10:
            self.history.clear();self.dwell=0.;self._clear_potential()
        self.history.append((capture,self.e.copy()))
        while self.history and capture-self.history[0][0]>.20:self.history.popleft()
        self.velocity_valid=len(self.history)>=3 and capture-self.history[0][0]>=.055
        self.vel=np.zeros(3)
        if self.velocity_valid:
            times=np.array([a for a,_ in self.history]);design=np.c_[np.ones(len(times)),times-times[-1]]
            self.vel=np.linalg.lstsq(design,np.array([b for _,b in self.history]),rcond=None)[0][1]
        phi=float(np.exp(-.5*np.sum((self.e/np.array([.12,.08,.10]))**2)))
        if self.potential is not None and gap is not None and gap<=.10:
            self.reward+=.15*((.99**(gap/.04))*phi-self.potential)
        self.potential=phi
        # A bounded dense signal helps discover interception before any dwell
        # milestone is reached. It pays only fresh visible captures, weighted
        # down by measured ball-to-cup motion; physical support remains an audit.
        if self.velocity_valid and gap is not None and 0 < gap <= .10:
            self.reward += gap * phi / (1. + float(self.vel @ self.vel))
        stable=(self.velocity_valid and np.linalg.norm(self.vel)<=.35 and upright>=.9
                and np.all(abs(self.e)+2*self.sigma<=np.array([.16,.12,.16]))
                and .10<=np.linalg.norm(palms[0]-palms[1])<=.28
                and geometry['normal_bisector_valid']
                and (self.form_acquired or (geometry['up_cos'].min()>=.60 and geometry['height_above_shoulders_m'].max()<=.15)))
        if stable and gap is not None and 0<gap<=.10:
            old=self.dwell;self.dwell+=gap
            if self.dwell>=.20:self.form_acquired=True
            self.reward+=max(0.,min(3.,self.dwell)-min(3.,old))
            for i,threshold in enumerate((.75,2.)):
                if self.dwell>=threshold and not self.paid[i]:self.reward+=2.;self.paid[i]=1.
        else:self.dwell=0.
        self.valid=True;self.last_valid=capture
        self.valid_count+=1;self.max_dwell=max(self.max_dwell,self.dwell)
        distance=float(np.linalg.norm(self.e))
        self.min_cup_distance=distance if self.min_cup_distance is None else min(self.min_cup_distance,distance)

    def _clear_potential(self):
        # An absorbing invalid/terminal state has zero potential. Clearing it
        # without this debit lets proximity shaping survive an occlusion/drop.
        if self.corrected_potential_reward and self.potential is not None:
            self.reward -= .15 * self.potential
        self.potential = None  # a second boundary cannot debit the same bank

    def finish(self):
        """Settle potential at actual episode end; legacy reward is unchanged."""
        if self.corrected_potential_reward:
            self._clear_potential()

    def invalidate(self):
        self.reward -= .02
        self.valid=False;self.velocity_valid=False;self.last_valid=None
        self.history.clear();self.dwell=0.;self._clear_potential()
        self.e[:]=0.;self.vel[:]=0.;self.sigma[:]=0.

    def geometry(self, q, imu):
        q=np.asarray(q,dtype=float);imu=np.asarray(imu,dtype=float)
        if q.shape!=(29,) or imu.shape!=(4,) or not np.isfinite(q).all() or not np.isfinite(imu).all() or np.linalg.norm(imu)<1e-9:
            raise ValueError('finite measured body encoders and IMU required')
        self.data.qpos[:]=self.model.qpos0
        self.data.qpos[self.base:self.base+7]=(0.,0.,0.,1.,0.,0.,0.)
        self.data.qpos[self.enc]=q
        self.mj.mj_kinematics(self.model,self.data)
        w,x,y,z=imu/np.linalg.norm(imu)
        R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        faces=[];normals=[]
        for i,bid in enumerate(self.palms):
            rotation=self.data.xmat[bid].reshape(3,3)
            faces.append(self.data.xpos[bid]+rotation@self.face_offsets[i])
            normals.append(rotation@np.array([0.,1. if i==0 else -1.,0.]))
        faces=np.array(faces);normals=np.array(normals);bisector=normals.mean(0);length=np.linalg.norm(bisector)
        valid=bool(length>=.25)
        bisector=bisector/max(length,1e-12)
        shoulder=self.data.xpos[self.shoulders].mean(0)
        yaw=np.arctan2(R[1,0],R[0,0]);c,s=np.cos(yaw),np.sin(yaw);Y=np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])
        framed=Y@R
        return dict(rotation=R,world_to_receiver=Y,faces_receiver=(framed@faces.T).T,normals_receiver=(framed@normals.T).T,up_cos=(R@normals.T)[2],cup_receiver=framed@(faces.mean(0)+self.radius_m*bisector),normal_bisector_valid=valid,height_above_shoulders_m=(R@(faces-shoulder).T)[2])

    def vector(self, obs, previous, rate_limited, native_encoders, native_nominal, calibration, context):
        age=.08 if self.last_valid is None else np.clip(obs.timestamp-self.last_valid,0.,1.)
        e=self.e if self.valid else np.zeros(3);vel=self.vel if self.valid and self.velocity_valid else np.zeros(3);g=self.current_geometry
        result=np.r_[obs.joint_position,obs.joint_velocity,project_gravity(obs.imu_quat_wxyz),obs.imu_gyro,e,vel,self.sigma if self.valid else np.zeros(3),age,float(self.valid),float(self.valid and self.velocity_valid),min(self.dwell,3.),calibration.fractions(native_encoders),calibration.fractions(native_nominal),g['faces_receiver'].ravel(),g['normals_receiver'].ravel(),g['up_cos'],previous,rate_limited,self.paid,context]
        if result.shape!=(OBS_SIZE,) or not np.isfinite(result).all():raise ValueError('invalid native sensor observation')
        return result.astype(np.float32)

class NativeCupAudit:
    """Twenty-action callback; only actual sensor/command packets reach policy."""
    def __init__(self,profile,policy=None,radius_m=.1135):
        self.profile=profile;self.policy=policy;self.radius_m=radius_m
        self.sensors=None;self.calibration=None;self.action=np.zeros(20);self.actual=np.zeros(20)
        self.calls=self.ticks=self.steps=self.disabled_ticks=0;self.first_enabled_s=None
        self.ledger=EpisodeReturnLedger();self.rows=[];self.camera_rows=[];self.form_rows=[]
        self.last_camera=self.last_form_time=None;self.total_form_cost=0.;self.finished=False
        self.context_tracker=CausalCatchContext(profile['play']['field_catch_window_m'],(*profile['play']['field_approach_axis_xy'],0.))
        self.phase=VisualReceivePhase();self.context=np.zeros(8)

    def bind(self,model,joint_names):
        self.sensors=NativeCupSensors(model,joint_names,radius_m=self.radius_m)
        self.calibration=NativeHandCalibration.from_model(model)
        self.arms=np.array([joint_names.index(n)for n in ARM_JOINTS]);self.ranges=np.array([model.joint(n).range.copy()for n in ARM_JOINTS])
        self.form=CatchFormEvaluator(model,joint_names)

    def __call__(self,obs,pose,frame,survey,target,native_target,native_encoders,enabled):
        if self.sensors is None:raise RuntimeError('static-model bind required')
        now=float(obs.timestamp);self.calls+=1
        native_target=np.asarray(native_target,dtype=float);native_encoders=np.asarray(native_encoders,dtype=float)
        if native_target.shape!=(14,)or native_encoders.shape!=(14,)or not np.isfinite(native_target).all()or not np.isfinite(native_encoders).all():raise ValueError('finite fourteen native command/encoder values required')
        if frame is not None and survey is not None and frame.timestamp_s!=self.last_camera:
            self.last_camera=frame.timestamp_s;ball=self.sensors.detector.detect(frame,survey)
            self.camera_rows.append(dict(time_s=now,capture_s=float(frame.timestamp_s),enabled=bool(enabled),ball_detected=ball is not None,ball_position_m=None if ball is None else ball.position_m.tolist(),ball_covariance_m2=None if ball is None else ball.covariance_m2.tolist(),camera_position_m=survey.translation.tolist(),pose_time_s=None if pose is None else float(pose.timestamp_s),pose_covariance=None if pose is None else pose.covariance.tolist(),pose_z_m=None if pose is None else pose.z_m,pose_z_std_m=None if pose is None else pose.z_std_m))
        dt=0. if self.last_form_time is None else now-self.last_form_time
        if dt<0:raise ValueError('causal sensor clock required')
        self.last_form_time=now
        geometry=self.form.evaluate(obs,CatchFormContext(over_shoulder=True,phase='inactive'))
        phase=self.phase.update(now,self.sensors,enabled)
        context=self.context_tracker.update(now,pose,frame,survey,geometry['shoulder_midpoint_height_above_pelvis_m'],phase)
        self.context=context_vector(context);form=self.form.evaluate(obs,context)
        cost=dt*float(form['penalty_per_s'])if enabled else 0.;self.total_form_cost+=cost
        self.form_rows.append(dict(time_s=now,residual_enabled=bool(enabled),reward_debit=cost,**form))
        if not enabled:
            self.disabled_ticks+=1
            return np.asarray(target).copy(),native_target.copy()
        if self.first_enabled_s is None:self.first_enabled_s=now
        self.sensors.update(obs,pose,frame,survey);self.sensors.reward-=cost
        if self.ticks%2==0:
            raw=self.sensors.vector(obs,self.action,self.actual,native_encoders,native_target,self.calibration,self.context);vector=scale_observation(raw);reward=float(self.sensors.reward);self.sensors.reward=0.;self.ledger.record(reward,reset=self.steps==0)
            requested=np.zeros(20)if self.policy is None else np.asarray(self.policy(vector.copy()),dtype=float)
            if requested.shape!=(20,)or not np.isfinite(requested).all()or np.any(abs(requested)>1.+1e-8):raise ValueError('policy requires20 finite normalized actions')
            self.action=requested.copy();g=self.sensors.current_geometry
            self.rows.append(dict(time_s=now,valid=bool(self.sensors.valid),velocity_valid=bool(self.sensors.velocity_valid),cup_error_m=self.sensors.e.tolist(),cup_velocity_mps=self.sensors.vel.tolist(),sigma_m=self.sensors.sigma.tolist(),visual_dwell_s=self.sensors.dwell,form_acquired=self.sensors.form_acquired,up_cos=g['up_cos'].tolist(),face_centres_receiver_m=g['faces_receiver'].tolist(),normals_receiver=g['normals_receiver'].tolist(),cup_target_receiver_m=g['cup_receiver'].tolist(),native_encoder=native_encoders.tolist(),native_nominal=native_target.tolist(),reward=reward,reset_reward=self.steps==0,raw=raw,normalized=vector,action=self.action.copy(),rate_limited_residual=self.actual.copy()))
            self.steps+=1
        old=self.actual.copy();self.actual=bounded_native_residual(self.action,self.actual)
        self.sensors.reward-=.02*(.05*float(np.mean(self.action**2))+.02*float(np.mean(((self.actual-old)/.02)**2)))
        self.ticks+=1;result=np.asarray(target).copy()
        if np.any(self.actual[:14]):result[self.arms]=np.clip(result[self.arms]+self.actual[:14],self.ranges[:,0],self.ranges[:,1])
        native=native_target.copy();fractions=self.calibration.fractions(native_target);opened=np.tile(OPEN,2);closed=self.calibration.ranges[:,0].copy();closed[[4,11]]=self.calibration.ranges[[4,11],1]
        for i,group in enumerate(GROUP_MAP):
            bias=self.actual[14+group]
            if bias!=0.:native[i]=opened[i]+np.clip(fractions[i]+bias,0.,1.)*(closed[i]-opened[i])
        return result,native

    def finish(self):
        if self.finished:raise RuntimeError('terminal ledger must settle once')
        self.finished=True;self.sensors.finish();bonus=2. if self.sensors.valid and self.sensors.dwell>=.75 and self.sensors.upright>=.9 and self.sensors.form_acquired else 0.
        self.ledger.record(float(self.sensors.reward+bonus));self.sensors.reward=0.;return bonus

    def save(self,directory,bonus):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        def matrix(key,n):return np.asarray([r[key]for r in self.rows],dtype=float).reshape(-1,n)
        np.savez_compressed(directory/'native-policy-trace.npz',timestamp_s=np.array([r['time_s']for r in self.rows]),raw_observation=matrix('raw',169),normalized_observation=matrix('normalized',169),action=matrix('action',20),rate_limited_residual=matrix('rate_limited_residual',20))
        serial=[{k:v for k,v in r.items()if k not in('raw','normalized','action','rate_limited_residual')}for r in self.rows]
        (directory/'sensor-samples.json').write_text(json.dumps(dict(policy_samples=serial,camera_samples=self.camera_rows),indent=2)+'\n');(directory/'form-samples.json').write_text(json.dumps(self.form_rows,indent=2)+'\n')
        summary=dict(observation_schema=OBS_SCHEMA,reward_schema=REWARD_SCHEMA,observation_size=169,action_size=20,arm_action_order=list(ARM_JOINTS),native_action_order=['left_four_fingers','left_thumb_opposition','left_both_thumb_flexors','right_four_fingers','right_thumb_opposition','right_both_thumb_flexors'],native_group_map=GROUP_MAP.tolist(),action_limits=ACTION_LIMIT.tolist(),action_rates=ACTION_RATE.tolist(),cup_radius_m=self.radius_m,radius_basis='fixedmidpointknownfootballminor/majorsemiaxes(.085+.142)/2; goalproxy, not measured ballorientation',cup_face_offsets_m=[x.tolist()for x in self.sensors.face_offsets],cup_target='mean calibrated palmar FACE +declaredradius*normalized mean palmarNORMAL; yaw-aligned receiverframe/worldZup; IMUfullrotation thenyaw removal',first_enabled_s=self.first_enabled_s,callback_calls=self.calls,disabled_ticks=self.disabled_ticks,policy_steps=self.steps,camera_captures=self.sensors.capture_count,valid_captures=self.sensors.valid_count,max_visual_dwell_s=self.sensors.max_dwell,final_visual_dwell_s=self.sensors.dwell,form_acquired=self.sensors.form_acquired,total_form_cost=self.total_form_cost,reset_sensor_reward=self.ledger.reset_reward,sensor_return=self.ledger.step_return,terminal_bonus=bonus,native_encoder_scope='six tendon position and one thumb opposition encoder perhand; no unobserved fingerjointFK',acquisition_gate='fresh visual stable dwell requiresboth up>=.60 and face<=shoulder+.15 until.20s; formhypothesis maypersistocclusion but positive dwell/reward NEVER latched',runtime_inputs=['RGBD+explicitpose/frame/survey','29body encoders+IMU','14nativeactuator encoders','commandednative targets/catchplane'],excluded=['exactball/root state','contacts/forces','replayqpos','unobservedfingerjointpositions','physicalqualityscores'])
        (directory/'sensor-summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary

def compare_reference(current,reference):
    """Postepisode only: all protectedreference arrays, plus explicitnew diagnostics."""
    current,reference=Path(current),Path(reference)
    with np.load(current,allow_pickle=False)as a,np.load(reference,allow_pickle=False)as b:
        rows={}
        for key in b.files:
            if key not in a:rows[key]=dict(exact=False,reason='missing');continue
            x,y=a[key],b[key];equal=bool(x.shape==y.shape and np.array_equal(x,y,equal_nan=True))if x.dtype.kind not in 'OUS'else bool(np.array_equal(x,y));rows[key]=dict(exact=equal,shape_current=list(x.shape),shape_reference=list(y.shape))
            if x.shape==y.shape and x.dtype.kind in 'ifub':
                finite=np.isfinite(x)&np.isfinite(y);rows[key]['maximum_absolute_difference']=float(np.max(abs(x[finite].astype(float)-y[finite].astype(float)),initial=0.))
        additional=sorted(set(a.files)-set(b.files))
        applied=None
        if 'native_applied_commands'in a and'native_hand'in b:
            applied=bool(np.array_equal(a['native_applied_commands'][:,0],b['native_hand'][:,0])and np.array_equal(a['native_applied_commands'][:,1:],b['native_hand'][:,15:29]))
    return dict(protected_reference_arrays_exact=bool(all(x['exact']for x in rows.values())),additional_current_diagnostics=additional,actual_native_command_exact=applied,arrays=rows)

def main():
    import sys
    scripts=Path(__file__).resolve().parents[2]/'scripts'
    if str(scripts)not in sys.path:sys.path.insert(0,str(scripts))
    import probe_native_cup_residual as runner
    parser=runner.build_parser();parser.add_argument('--baseline-trace',type=Path);parser.add_argument('--baseline-view',type=Path);parser.add_argument('--cup-radius',type=float,default=.1135);args=parser.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=False);args.output=output/'trial';audit=NativeCupAudit(json.loads(args.profile.read_text()),radius_m=args.cup_radius);runner.run(args,native_residual_callback=audit);bonus=audit.finish();summary=audit.save(output,bonus)
    if args.baseline_trace:
        eq=compare_reference(args.output/'trace.npz',args.baseline_trace)
        if args.baseline_view:eq['view']=compare_reference(args.output/'view.npz',args.baseline_view)
        (output/'zero-physical-equality.json').write_text(json.dumps(eq,indent=2)+'\n');summary['zero_physical_exact']=eq['protected_reference_arrays_exact']and eq.get('actual_native_command_exact',True)is not False and eq.get('view',{}).get('protected_reference_arrays_exact',True)
    (output/'sensor-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
