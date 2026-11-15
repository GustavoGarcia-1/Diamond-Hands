"""Fixed-gain native PPO continuation with independent throw-only fixture RNG.

The protected commanded-release native preload is preserved. Random aim values
are fixture inputs, never actor/critic observations, reward inputs or curriculum.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import time
import numpy as np
from .native_cup_training import NativeCatchEnv
from .native_cup_residual_rl import OBS_SCHEMA, REWARD_SCHEMA, compare_reference

FIXTURE_SCHEMA='native-fixed-catchplane-random-throw-yz-v1'
CAMERAS=dict(sensor_retention_camera_position=(6,-2.5,2.4),sensor_retention_camera_target=(7,0,1),sensor_retention_pose_camera_position=(3.5,0,1.4),sensor_retention_pose_camera_target=(6.5,0,.9))

def mutate_throw_only(donor,delta_yz):
 delta=np.asarray(delta_yz,dtype=float)
 if delta.shape!=(2,)or not np.isfinite(delta).all()or np.any(abs(delta)>.020000001):raise ValueError('bounded two-axis throw perturbation required')
 result=deepcopy(donor);target=list(result['play']['quarterback_target_m']);target[1]+=float(delta[0]);target[2]+=float(delta[1]);result['play']['quarterback_target_m']=target
 before=deepcopy(donor);after=deepcopy(result);before['play'].pop('quarterback_target_m');after['play'].pop('quarterback_target_m')
 if before!=after or target[0]!=donor['play']['quarterback_target_m'][0]:raise RuntimeError('fixture changed anything except throw aim Y/Z')
 return result

class NativeContinuationEnv(NativeCatchEnv):
 def __init__(self,profile,source,amp_source,*,fixture_root,fixture_seed=173,random_aim_yz=(.02,.02),action_gain=.5,episode_log=None,evaluation_dir=None):
  super().__init__(profile,source,amp_source,episode_log=episode_log,evaluation_dir=evaluation_dir,duration=9.,**CAMERAS)
  self.donor=json.loads(Path(profile).read_text());self.donor_sha=sha256(Path(profile).read_bytes()).hexdigest();self.fixture_root=Path(fixture_root);self.fixture_root.mkdir(parents=True,exist_ok=True)
  self.fixture_rng=np.random.default_rng(fixture_seed);self.amplitude=np.asarray(random_aim_yz,dtype=float);self.action_gain=float(action_gain)
  if self.amplitude.shape!=(2,)or not np.isfinite(self.amplitude).all()or np.any(self.amplitude<0)or np.any(self.amplitude>.02)or not 0<self.action_gain<=1:raise ValueError('bounded fixtures and action gain required')

 def reset(self,*,seed=None,options=None):
  self.close()
  if seed is not None:self.fixture_rng=np.random.default_rng(seed)
  delta=self.fixture_rng.uniform(-self.amplitude,self.amplitude);profile=mutate_throw_only(self.donor,delta);path=self.fixture_root/f'episode-{self.episode+1:04d}.json';text=json.dumps(profile,indent=2)+'\n';path.write_text(text);self.reference=path
  with(self.fixture_root/'fixture-ledger.jsonl').open('a')as stream:stream.write(json.dumps(dict(episode=self.episode+1,seed_reset=seed,delta_yz_m=delta.tolist(),donor_sha256=self.donor_sha,profile_sha256=sha256(text.encode()).hexdigest(),mutated_fields=['play.quarterback_target_m[1]','play.quarterback_target_m[2]'],actor_critic_reward_receive_fixture_parameters=False))+'\n')
  return super().reset(seed=seed,options=options)

 def step(self,action):
  raw=np.asarray(action,dtype=np.float32)
  if not self.action_space.contains(raw):raise ValueError('twenty normalized raw actions required')
  # Exactly the validated deterministic gain-ablation mapping. The audit stores
  # scaled actual action history; PPO likelihoods remain those of raw actions.
  return super().step(raw*np.float32(self.action_gain))

def episode(env,model=None,seed=173):
 obs,_=env.reset(seed=seed);total=0.
 try:
  while True:
   action=np.zeros(20,np.float32)if model is None else model.predict(obs,deterministic=True)[0]
   obs,reward,done,_,info=env.step(action);total+=reward
   if done:break
  if not np.isclose(total,info['sensor_return'],atol=1e-9):raise RuntimeError('returned reward differs from sensor ledger')
  return dict(returned_reward=total,**info)
 finally:env.close()

def optimizer_digest(model):
 digest=sha256();state=model.policy.optimizer.state_dict()
 digest.update(json.dumps(state['param_groups'],sort_keys=True,default=str).encode());steps=[]
 for index,item in sorted(state['state'].items()):
  digest.update(str(index).encode())
  for key,value in sorted(item.items()):
   digest.update(key.encode());array=value.detach().cpu().numpy()if hasattr(value,'detach')else np.asarray(value);digest.update(str(array.dtype).encode());digest.update(array.tobytes())
   if key=='step':steps.append(float(array))
 return dict(sha256=digest.hexdigest(),state_entries=len(state['state']),minimum_step=min(steps,default=0),maximum_step=max(steps,default=0))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['profile','source','amp-source','output','resume']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--action-gain',type=float,default=.5);p.add_argument('--random-aim-y-m',type=float,default=.02);p.add_argument('--random-aim-z-m',type=float,default=.02)
 p.add_argument('--wall-seconds',type=float,default=2700);p.add_argument('--steps',type=int,default=262144);p.add_argument('--n-envs',type=int,default=2);p.add_argument('--seed',type=int,default=173)
 p.add_argument('--preflight',action='store_true');p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--zero-reference-trace',type=Path);p.add_argument('--zero-reference-view',type=Path);p.add_argument('--model-reference-trace',type=Path);p.add_argument('--model-reference-view',type=Path)
 a=p.parse_args()
 if not 0<a.wall_seconds<=2700 or not 0<a.steps<=262144 or not 1<=a.n_envs<=2 or not 0<a.action_gain<=1 or not 0<=a.random_aim_y_m<=.02 or not 0<=a.random_aim_z_m<=.02:p.error('bounded continuation required')
 a.output.mkdir(parents=True,exist_ok=False)
 import torch
 from stable_baselines3 import PPO
 from stable_baselines3.common.callbacks import BaseCallback
 from stable_baselines3.common.vec_env import SubprocVecEnv
 torch.set_num_threads(1);start=time.monotonic();model=PPO.load(a.resume,device=a.device)
 if model.observation_space.shape!=(169,)or model.action_space.shape!=(20,)or getattr(model,'receiver_sensor_schema',None)!=OBS_SCHEMA:raise RuntimeError('native169D/20D checkpoint schema required')
 metadata=dict(observation_schema=OBS_SCHEMA,reward_schema=REWARD_SCHEMA,fixture_schema=FIXTURE_SCHEMA,resume_sha256=sha256(a.resume.read_bytes()).hexdigest(),initial_num_timesteps=model.num_timesteps,initial_optimizer_epochs=model._n_updates,initial_optimizer=optimizer_digest(model),action_gain=a.action_gain,effective_arm_limit_rad=.18*a.action_gain,effective_native_group_limit=.15*a.action_gain,random_aim_amplitude_yz_m=[a.random_aim_y_m,a.random_aim_z_m],seed=a.seed,preload_clock='protected commanded-release event, external machine command integration unverified; unchanged',actor_critic_reward_fixture_parameters=False,device=a.device,cameras=CAMERAS)
 (a.output/'contract.json').write_text(json.dumps(metadata,indent=2)+'\n');env=None
 try:
  if a.preflight:
   results=[]
   for label,policy,trace_ref,view_ref in [('zero',None,a.zero_reference_trace,a.zero_reference_view),('model',model,a.model_reference_trace,a.model_reference_view)]:
    out=a.output/label;env=NativeContinuationEnv(a.profile,a.source,a.amp_source,fixture_root=out/'fixtures',random_aim_yz=(0,0),action_gain=a.action_gain,evaluation_dir=out/'rollout');row=episode(env,policy,a.seed);trial=out/'rollout/episode-0001/trial'
    if trace_ref:
     row['trace_equality']=compare_reference(trial/'trace.npz',trace_ref)
     if not row['trace_equality']['protected_reference_arrays_exact']:raise RuntimeError(label+' trace mismatch')
    if view_ref:
     row['view_equality']=compare_reference(trial/'view.npz',view_ref)
     if not row['view_equality']['protected_reference_arrays_exact']:raise RuntimeError(label+' view mismatch')
    results.append(dict(label=label,**row))
   (a.output/'preflight.json').write_text(json.dumps(results,indent=2)+'\n');metadata['preflight_pass']=True
  else:
   if a.device!='cuda'or not torch.cuda.is_available():raise RuntimeError('CUDA required for training')
   env=SubprocVecEnv([partial(NativeContinuationEnv,a.profile,a.source,a.amp_source,fixture_root=a.output/'fixtures'/f'env-{i}',fixture_seed=a.seed+i,random_aim_yz=(a.random_aim_y_m,a.random_aim_z_m),action_gain=a.action_gain,episode_log=a.output/f'episodes-{i}.jsonl')for i in range(a.n_envs)],start_method='spawn')
   model.set_env(env,force_reset=True);model.seed=a.seed;model.set_random_seed(a.seed)
   metadata['optimizer_after_set_env']=optimizer_digest(model)
   if metadata['optimizer_after_set_env']!=metadata['initial_optimizer']:raise RuntimeError('optimizer changed during continuation setup')
   model.receiver_action_gain=a.action_gain;model.receiver_fixture_schema=FIXTURE_SCHEMA;model.save(a.output/'initial-warmstart')
   class Bounded(BaseCallback):
    last_save_s=0.
    def _on_step(self):return time.monotonic()-start<a.wall_seconds
    def _on_rollout_start(self):
     elapsed=time.monotonic()-start
     with(a.output/'progress.jsonl').open('a')as f:f.write(json.dumps(dict(timesteps=self.num_timesteps,new_timesteps=self.num_timesteps-metadata['initial_num_timesteps'],optimizer_epochs=int(self.model._n_updates),elapsed_s=elapsed))+'\n')
     if elapsed-self.last_save_s>=300:self.model.save(a.output/f'checkpoint-{self.num_timesteps:08d}');self.last_save_s=elapsed
   model.learn(total_timesteps=max(1,a.steps-model.num_timesteps),reset_num_timesteps=False,callback=Bounded());model.save(a.output/'final');metadata.update(actual_timesteps=model.num_timesteps,optimizer_epochs=model._n_updates,final_optimizer=optimizer_digest(model));env.close();env=None
 finally:
  if env is not None:env.close()
  metadata['elapsed_s']=time.monotonic()-start;(a.output/'receipt.json').write_text(json.dumps(metadata,indent=2)+'\n')
if __name__=='__main__':main()
