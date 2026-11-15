"""Go QB-side cup with camera-TTC-only late reach; original body route and native-rate limits.

All qpos/contact reads below are passive evaluation recorders. No teacher, critic or reward.
"""
import argparse,sys,json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import replace
import numpy as np,mujoco
p=argparse.ArgumentParser();
for k in ['frozen-root','profile','compact','cup','source','amp-source','output']:p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--timeconstant',type=float,choices=[.02,.0025],default=.02);p.add_argument('--cue-lead',type=float,default=.65);p.add_argument('--reach-duration',type=float,default=.45);a=p.parse_args();sys.path[:0]=[str(a.frozen_root/'src'),str(a.frozen_root/'scripts')]
from types import SimpleNamespace
from g1_receiver.hand_route_plant import HandRoutePlant
from g1_receiver.anatomical_catch_eval import AnatomicalContactAudit,physics_fingerprint
from g1_receiver._vendor.g1_runtime.pool.wbc import WholeBodyController
from g1_receiver.perception import BallTracker,CameraIntrinsics
from g1_receiver.scrimmage_throw import ScrimmagePass
from g1_receiver.sim import WIDTH,HEIGHT
from probe_ball_hand_contact_model import add_pairs
import probe_native_cup_residual as runner
compact=json.load(open(a.compact));cup=json.load(open(a.cup));profile=json.load(open(a.profile));q0=np.array(compact['arm_targets_rad']);q1=np.array(cup['arm_targets_rad']);state=dict(handoff=None,delivery=None);commands=[];poses=[];velocities=[];controls=[];self_contacts=[];ball_contacts=[];holder={}
original_delivery=runner.OpticalPass;original_init=HandRoutePlant.__init__;original_step=HandRoutePlant.step;original_capture=HandRoutePlant.capture;original_wbc=WholeBodyController.act;original_plan=ScrimmagePass.plan;original_record=AnatomicalContactAudit.record
cam_position=np.array([3.,-4.,3.]);cam_target=np.array([3.5,-.4,2.]);forward=cam_target-cam_position;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1.]);right/=np.linalg.norm(right);down=np.cross(forward,right);cam_rotation=np.column_stack((right,down,forward));focal=.5*HEIGHT/np.tan(np.deg2rad(80.)/2);intrinsics=CameraIntrinsics(focal,focal,(WIDTH-1)/2,(HEIGHT-1)/2)
class CausalSideCup(original_delivery):
 def __init__(self,plant,p,shift):
  super().__init__(plant,p,shift);self.initial=None;self.cue_at=None;self.cue_ttc=None;self.cue_trace=[];self.last_capture=-np.inf;self.cue_tracker=BallTracker(max_age_s=.08,max_horizon_s=1.,allow_synthetic_fixture=True);state['delivery']=self
 def observe(self,plant,now,pose,history):
  super().observe(plant,now,pose,history)
  if state['handoff']is None and pose is not None and pose.fresh(now,.14) and pose.xy_m[0]>=1.3:state['handoff']=now
  frame,survey=plant.capture('late_cue')
  if frame.timestamp_s<=self.last_capture+1e-8:return
  self.last_capture=frame.timestamp_s;obs=self.hands.detector.detect(frame,survey);row=dict(t=now,frame_time_s=frame.timestamp_s,status='no_ball')
  if obs is not None and self.cue_tracker.add(obs):
   crossing=self.cue_tracker.intercept(now,self.hands.target,(*self.hands.approach_axis,0.),obs.frame_id)
   row.update(status='no_fresh_crossing',rgbd_ball_m=obs.position_m.tolist())
   if crossing is not None and abs(crossing.last_inlier_timestamp_s-obs.timestamp_s)<=1e-8:
    lead=crossing.timestamp_s-now;offset=crossing.position_m-self.hands.target;row.update(ttc_s=float(lead),time_std_s=float(crossing.time_std_s),predicted_m=crossing.position_m.tolist(),status='too_early')
    incoming=crossing.velocity_mps[:2]@self.hands.approach_axis>=2.5 and crossing.velocity_mps[2]<0
    fresh=pose is not None and pose.fresh(now,.14)
    inside=abs(offset[:2]@self.hands.lateral_axis)<=self.hands.grip.lateral_limit_m and abs(offset[2])<=.30
    if not fresh:row['status']='receiver_pose_stale'
    elif not incoming:row['status']='not_incoming'
    elif crossing.time_std_s>.06 or not inside:row['status']='uncertain_or_outside'
    elif lead<a.reach_duration+.10:row['status']='too_late_no_clock_fallback'
    elif self.cue_at is None and lead<=a.cue_lead:
     self.cue_at=now;self.cue_ttc=float(lead);row['status']='causal_visual_reach_cue'
  self.cue_trace.append(row)
 def targets(self,now,upper):
  _,closure=super().targets(now,upper)
  if self.initial is None:self.initial=upper[self.arm_indices].copy()
  target=upper.copy();phase=0.;prep=0.
  if state['handoff']is None:target[self.arm_indices]=self.initial
  else:
   u=float(np.clip((now-state['handoff']-.7)/1.,0,1));prep=u**3*(10-15*u+6*u*u);target[self.arm_indices]=self.initial+prep*(q0-self.initial)
   if self.cue_at is not None:
    u=float(np.clip((now-self.cue_at)/a.reach_duration,0,1));phase=u**3*(10-15*u+6*u*u);target[self.arm_indices]+=phase*(q1-q0)
  commands.append(np.r_[now,target[self.arm_indices],phase,prep]);return target,closure

def init(self,*args,**kwargs):
 setup=kwargs.get('scene_setup')
 def scene_setup(scene):
  if setup:setup(scene)
  ET.SubElement(scene.find('worldbody'),'camera',name='late_cue',pos=' '.join(map(str,cam_position)),xyaxes=' '.join(map(str,(*right,*-down))),fovy='80')
  if a.timeconstant!=.02:holder['pair_declaration']=add_pairs(scene,a.timeconstant)
 kwargs['scene_setup']=scene_setup;original_init(self,*args,**kwargs);self.cameras['late_cue']=(cam_rotation,cam_position);a.output.mkdir(parents=True,exist_ok=False);(a.output/'scene.xml').write_text(self.scene_xml);holder.update(model=self.model,ball=self.model.geom('football_surface').id,physics=physics_fingerprint(self.model)['sha256'])
def capture(self,camera='survey_rgbd'):
 if camera in['pass','late_cue']:
  now=self.timestamp_s
  if holder.get('capture_time')!=now:
   frame,survey=original_capture(self,'late_cue');holder['capture_value']=(replace(frame,intrinsics=intrinsics),survey);holder['capture_time']=now
  return holder['capture_value']
 return original_capture(self,camera)
def step(self,command,*,closure=0.,hand_target_rad=None,post_step=None):
 def observe(data):
  if post_step:post_step(data)
  poses.append(np.r_[data.time,data.qpos.copy()]);velocities.append(np.r_[data.time,data.qvel.copy()]);controls.append(np.r_[data.time,data.ctrl.copy()])
  for i,c in enumerate(data.contact):
   g,h=map(int,c.geom)
   if self.model.geom_bodyid[g]>0 and self.model.geom_bodyid[h]>0 and holder['ball']not in(g,h):
    f=np.zeros(6);mujoco.mj_contactForce(self.model,data,i,f);self_contacts.append(np.r_[data.time,g,h,c.dist,f[0]])
 return original_step(self,command,closure=closure,hand_target_rad=hand_target_rad,post_step=observe)
def wbc(self,obs,goal,upper_target=None):
 start=state['handoff'];u=0. if start is None else float(np.clip(obs.timestamp-start-1.7,0,1));phase=u**3*(10-15*u+6*u*u);return original_wbc(self,obs,replace(goal,torso_rpy=np.array([0.,0.,-.65*phase])),upper_target)
def plan(self,*args,**kwargs):return replace(original_plan(self,*args,**kwargs),release_quaternion_wxyz=tuple(profile['initial_ball_quaternion_wxyz']))
def record(self,data):
 original_record(self,data)
 for i,c in enumerate(data.contact):
  g,h=map(int,c.geom)
  if g==self.ball:other=h;sign=-1.
  elif h==self.ball:other=g;sign=1.
  else:continue
  f=np.zeros(6);mujoco.mj_contactForce(self.model,data,i,f);F=np.asarray(c.frame).reshape(3,3);ball_contacts.append(np.r_[data.time,other,int(self.model.geom_bodyid[other]),c.dist,f[0],sign*(F.T@f[:3])[2],c.pos,sign*F[0]])
runner.OpticalPass=CausalSideCup;HandRoutePlant.__init__=init;HandRoutePlant.capture=capture;HandRoutePlant.step=step;WholeBodyController.act=wbc;ScrimmagePass.plan=plan;AnatomicalContactAudit.record=record
args=SimpleNamespace(source=a.source,amp_source=a.amp_source,profile=a.profile,output=a.output/'trial',no_pass=False,walk_speed=1.,handoff_depth=1.3,initial_yaw=-.7,duration=9.,record_anatomical=True,reference_physics_sha256='173bf57168f5f97d0471d7bfa0a05e067a00f1e27a385994f57f21998a30af15',sensor_retention_camera_position=None,sensor_retention_camera_target=None,sensor_retention_pose_camera_position=None,sensor_retention_pose_camera_target=None)
try:report=runner.run(args,native_residual_callback=None)
finally:runner.OpticalPass=original_delivery;HandRoutePlant.__init__=original_init;HandRoutePlant.capture=original_capture;HandRoutePlant.step=original_step;WholeBodyController.act=original_wbc;ScrimmagePass.plan=original_plan;AnatomicalContactAudit.record=original_record
m=holder['model'];names=dict(geom_names=np.array([m.geom(i).name or f'geom-{i}'for i in range(m.ngeom)]),body_names=np.array([m.body(i).name for i in range(m.nbody)]),geom_bodyid=m.geom_bodyid);np.savez_compressed(a.output/'actual-state.npz',time_qpos=np.asarray(poses),time_qvel=np.asarray(velocities),time_control=np.asarray(controls),time_arm_target_phase=np.asarray(commands));np.savez_compressed(a.output/'ball-contact-witness.npz',ball_contact_witness=np.asarray(ball_contacts),**names);np.savez_compressed(a.output/'self-contact-witness.npz',contacts=np.asarray(self_contacts),**names);delivery=state['delivery'];(a.output/'causal-cue-trace.json').write_text(json.dumps(delivery.cue_trace,indent=2)+'\n');extended=dict(physics_sha256=holder['physics'],pair_geom1=m.pair_geom1.tolist(),pair_geom2=m.pair_geom2.tolist(),pair_solref=m.pair_solref.tolist(),pair_solimp=m.pair_solimp.tolist(),pair_friction=m.pair_friction.tolist(),pair_dim=m.pair_dim.tolist(),pair_margin=m.pair_margin.tolist(),pair_gap=m.pair_gap.tolist(),ccd_iterations=int(m.opt.ccd_iterations),ccd_tolerance=float(m.opt.ccd_tolerance),enableflags=int(m.opt.enableflags),disableflags=int(m.opt.disableflags));(a.output/'controller-receipt.json').write_text(json.dumps(dict(actor_inputs=['RGBDball+survey','RGBDreceiverpose','IMU','29bodyencoders','14nativeencoders','fixedcommandedcuppose'],no_privileged_teacher_reward=True,no_training=True,arm_cue_only_fresh_rgbd_ttc=True,no_fixture_clock_arm_cue=True,actual_cue_s=delivery.cue_at,actual_cue_ttc_s=delivery.cue_ttc,handoff_s=state['handoff'],native_preload_dependency='declaredcommandedmachine-releaseevent',reach_duration_s=a.reach_duration,cue_lead_s=a.cue_lead,camera=dict(position=cam_position.tolist(),target=cam_target.tolist(),vertical_fov_deg=80,actual_intrinsics=vars(intrinsics)),model_assumption=dict(timeconstant_s=a.timeconstant,original_model=a.timeconstant==.02,calibrated_material=False,extended_identity=extended,extended_sha256=hashlib.sha256(json.dumps(extended,sort_keys=True,separators=(',',':')).encode()).hexdigest()),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),perception_sha256=hashlib.sha256((a.frozen_root/'src/g1_receiver/perception.py').read_bytes()).hexdigest(),profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest()),indent=2)+'\n')
