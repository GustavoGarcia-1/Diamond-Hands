"""Isolated explicit ball-hand contact assumption; immutable controller, unchanged quality gates."""
import argparse,hashlib,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

def add_pairs(scene,timeconstant):
 import mujoco
 before=ET.tostring(scene,encoding='unicode');model=mujoco.MjModel.from_xml_string(before)
 if model.npair:raise ValueError('expected no original explicit pairs')
 nodes={'world':scene.find('worldbody')};nodes.update({b.get('name'):b for b in scene.findall('.//worldbody//body')})
 hand=[];renamed=[]
 for body in range(model.nbody):
  if not model.body(body).name.startswith('aero_'):continue
  elements=nodes[model.body(body).name].findall('geom');ids=list(range(int(model.body_geomadr[body]),int(model.body_geomadr[body]+model.body_geomnum[body])))
  if len(elements)!=len(ids):raise ValueError('body geom correspondence mismatch')
  for g,element in zip(ids,elements,strict=True):
   if (element.get('name')or'')!=(model.geom(g).name or''):raise ValueError('geom order mismatch')
   if not(model.geom_contype[g]or model.geom_conaffinity[g]):continue
   name=element.get('name')
   if not name:name=f'contact_model_native_geom_{g}';element.set('name',name);renamed.append(dict(geom_id=g,name=name,body=model.body(body).name))
   hand.append((g,name))
 ball=model.geom('football_surface').id;section=scene.find('contact')
 if section is None:section=ET.SubElement(scene,'contact')
 pairs=[]
 for g,name in hand:
  if model.geom_priority[g]!=model.geom_priority[ball]:raise ValueError('unexpected unequal priority')
  mix=model.geom_solmix[ball]/(model.geom_solmix[g]+model.geom_solmix[ball]);sr=mix*model.geom_solref[ball]+(1-mix)*model.geom_solref[g];si=mix*model.geom_solimp[ball]+(1-mix)*model.geom_solimp[g];fr=np.maximum(model.geom_friction[g],model.geom_friction[ball]);fr5=[fr[0],fr[0],fr[1],fr[2],fr[2]];original=sr.copy();sr[0]=timeconstant
  attrs=dict(name=f'contact_model_ball_hand_{g}',geom1='football_surface',geom2=name,condim=str(max(model.geom_condim[ball],model.geom_condim[g])),solref=' '.join(map(str,sr)),solimp=' '.join(map(str,si)),friction=' '.join(map(str,fr5)),margin=str(max(model.geom_margin[ball],model.geom_margin[g])),gap=str(max(model.geom_gap[ball],model.geom_gap[g])))
  ET.SubElement(section,'pair',**attrs);pairs.append(dict(**attrs,original_effective_solref=original.tolist()))
 return dict(renamed=renamed,pairs=pairs,original_xml_sha256=hashlib.sha256(before.encode()).hexdigest())

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('frozen-root','profile','model','source','amp-source','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--timeconstant',type=float,choices=[.02,.01,.005,.0025],required=True);p.add_argument('--unmodified',action='store_true');p.add_argument('--gain',type=float,choices=[0.,.5],required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 sys.path[:0]=[str(a.frozen_root/'src'),str(a.frozen_root/'scripts')]
 import mujoco
 from stable_baselines3 import PPO
 from g1_receiver.hand_route_plant import HandRoutePlant
 from g1_receiver.anatomical_catch_eval import AnatomicalContactAudit,physics_fingerprint
 from g1_receiver.native_cup_training import NativeCatchEnv
 original_init=HandRoutePlant.__init__;original_record=AnatomicalContactAudit.record;metadata={};contacts=[];poses=[];controls=[];names={};work=np.zeros(6)
 def init(self,*args,**kwargs):
  setup=kwargs.get('scene_setup')
  def scene_setup(scene):
   if setup:setup(scene)
   if not a.unmodified:metadata.update(add_pairs(scene,a.timeconstant))
  kwargs['scene_setup']=scene_setup;original_init(self,*args,**kwargs)
  if a.timeconstant<2*self.model.opt.timestep:raise ValueError('timeconstant below2dt')
  fp=physics_fingerprint(self.model);extended=dict(base_physics_sha256=fp['sha256'],pair_geom1=self.model.pair_geom1.tolist(),pair_geom2=self.model.pair_geom2.tolist(),pair_dim=self.model.pair_dim.tolist(),pair_margin=self.model.pair_margin.tolist(),pair_gap=self.model.pair_gap.tolist(),pair_solref=self.model.pair_solref.tolist(),pair_solimp=self.model.pair_solimp.tolist(),pair_friction=self.model.pair_friction.tolist(),enableflags=int(self.model.opt.enableflags),disableflags=int(self.model.opt.disableflags))
  metadata.update(mujoco_version=mujoco.__version__,timestep_s=float(self.model.opt.timestep),physics_sha256=fp['sha256'],extended_physics=extended,extended_physics_sha256=hashlib.sha256(json.dumps(extended,sort_keys=True,separators=(',',':')).encode()).hexdigest(),scene_xml_sha256=hashlib.sha256(self.scene_xml.encode()).hexdigest());(a.output/'scene.xml').write_text(self.scene_xml)
 def record(self,data):
  original_record(self,data)
  if not names:names.update(geom_names=[self.model.geom(i).name or f'geom-{i}'for i in range(self.model.ngeom)],body_names=[self.model.body(i).name for i in range(self.model.nbody)])
  if 4.9<=data.time<=5.5:poses.append(np.r_[data.time,data.qpos.copy()]);controls.append(np.r_[data.time,data.ctrl.copy()])
  for i,c in enumerate(data.contact):
   g1,g2=map(int,c.geom)
   if g1==self.ball:g=g2;sign=-1.
   elif g2==self.ball:g=g1;sign=1.
   else:continue
   mujoco.mj_contactForce(self.model,data,i,work);F=np.asarray(c.frame).reshape(3,3);contacts.append(np.r_[data.time,g,int(self.model.geom_bodyid[g]),c.dist,work[0],sign*(F.T@work[:3])[2],c.pos,sign*F[0],c.solref,c.solimp])
 HandRoutePlant.__init__=init;AnatomicalContactAudit.record=record
 model=PPO.load(a.model,device=a.device) if a.gain else None
 env=NativeCatchEnv(a.profile,a.source,a.amp_source,evaluation_dir=a.output/'rollout',duration=9.,sensor_retention_camera_position=(6,-2.5,2.4),sensor_retention_camera_target=(7,0,1),sensor_retention_pose_camera_position=(3.5,0,1.4),sensor_retention_pose_camera_target=(6.5,0,.9))
 try:
  obs,_=env.reset(seed=173);total=0
  while True:
   action=np.zeros(20,dtype=np.float32)if model is None else model.predict(obs,deterministic=True)[0]*a.gain
   obs,r,done,_,info=env.step(action);total+=r
   if done:break
 finally:env.close();HandRoutePlant.__init__=original_init;AnatomicalContactAudit.record=original_record
 np.savez_compressed(a.output/'ball-contact-witness.npz',ball_contact_witness=np.asarray(contacts),**names)
 np.savez_compressed(a.output/'offline-robot-qpos.npz',timestamps_s=np.asarray(poses)[:,0],qpos=np.asarray(poses)[:,1:]);np.savez_compressed(a.output/'actual-control.npz',time_control=np.asarray(controls))
 metadata.update(arguments={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()},sensor_return=total,terminal_info=info,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),model_changed=not a.unmodified,controller_changed=False,calibrated_material=False,quality_gates_unchanged=True,witness_columns=['time','geom','body','distance','normal_n','vertical_n','x','y','z','nx','ny','nz','solref0','solref1','solimp0','solimp1','solimp2','solimp3','solimp4'])
 (a.output/'contact-model-receipt.json').write_text(json.dumps(metadata,indent=2)+'\n')
if __name__=='__main__':main()
