"""Sensor-only lower-body residual PPO for frozen AMP with a fixed high cup.

No ball, contact, root truth, or body truth enters observations or rewards.
TruthRecorder is instantiated only for independent evaluation, never training.
"""
from __future__ import annotations
from collections import deque
from dataclasses import replace
from hashlib import sha256
import argparse,json,time
from pathlib import Path
import numpy as np
import gymnasium as gym
from .observable import ARM_JOINTS,project_gravity

SCHEMA='amp-high-cup-lower-residual-sensors-v1-fixed-scaled-93'
REWARD_SCHEMA='amp-high-cup-upright-rgbd-speed-stem-v1-additive'
OBS_SIZE=93
DT=.02
CAMERAS={'track':((-1.,0.,1.5),(3.,0.,.9)), 'pass':((5.,0.,1.5),(10.,0.,.9)), 'finish':((10.,0.,1.5),(15.,0.,.9))}

class MarkerSensors:
    """Only rendered RGB-D estimates, timestamps, encoders and IMU."""
    def __init__(self):
        self.history=deque(maxlen=7);self.pose=None;self.velocity=np.zeros(2)
        self.pose_valid=False;self.velocity_valid=False;self.capture_count=0;self.valid_count=0
    def update(self,now,pose=None):
        if pose is not None and pose.fresh(now,.14) and np.isfinite(pose.xy_m).all():
            covariance=np.asarray(pose.covariance)
            if np.isfinite(covariance).all() and np.max(np.diag(covariance)[:2])<=.04:
                if not self.history or pose.timestamp_s>self.history[-1][0]+1e-9:
                    if self.history and pose.timestamp_s-self.history[-1][0]>.14:self.history.clear()
                    self.history.append((float(pose.timestamp_s),np.asarray(pose.xy_m).copy()))
                self.pose=pose
        self.pose_valid=self.pose is not None and self.pose.fresh(now,.14)
        self.velocity_valid=False;self.velocity[:]=0.
        if self.pose_valid and len(self.history)>=3:
            t=np.array([p[0]for p in self.history]);xy=np.array([p[1]for p in self.history]);t=t-t.mean()
            if np.ptp(t)>=.079:
                self.velocity=(t[:,None]*(xy-xy.mean(0))).sum(0)/(t@t);self.velocity_valid=bool(np.isfinite(self.velocity).all())
        if not self.pose_valid:self.history.clear()
    def fields(self,now):
        xy=np.asarray(self.pose.xy_m)if self.pose_valid else np.zeros(2)
        age=min(1.,max(0.,now-self.pose.timestamp_s))if self.pose is not None else 1.
        return xy,self.velocity.copy(),age

def sensor_reward(obs,sensors,offset,delta):
    """Additive sensor reward; invalid/stale pose earns no motion/path reward."""
    upright=-float(project_gravity(obs.imu_quat_wxyz)[2]);valid=sensors.pose_valid and sensors.velocity_valid
    up=float(np.clip((upright-.5)/.5,0.,1.));speed=stem=0.
    if valid:
        speed=2.*float(np.exp(-((sensors.velocity[0]-2.)/.7)**2-(sensors.velocity[1]/.6)**2))
        stem=.5*float(np.exp(-(float(sensors.pose.xy_m[1])/.4)**2))
    cost=.02*float(np.mean((offset/.25)**2))+.002*float(np.mean((delta/.04)**2))
    missing=0. if valid else .5
    reward=DT*(up+speed+stem-cost-missing-3.*max(0.,.9-upright))
    return float(reward),{'upright_sensor':upright,'pose_valid':bool(sensors.pose_valid),'velocity_valid':bool(sensors.velocity_valid),'speed_reward':speed,'stem_reward':stem,'action_cost':cost}

class TruthRecorder:
    """Offline evaluator. Never constructed in a training environment."""
    def __init__(self,plant):
        from .running_catch_eval import FootContactRecorder
        self.plant=plant;self.feet=FootContactRecorder(plant.model);self.motion=[];self.poses=[]
    def record_tick(self,tick,now):
        b=self.plant.data.body('pelvis');q=self.plant.data.qpos
        self.motion.append((now,*b.xpos,float(b.xmat[8])))
        if tick%2==0:self.poses.append((now,q.copy()))
    def save(self,output):
        output.mkdir(parents=True,exist_ok=False);m=np.array(self.motion);foot=self.feet.arrays()
        np.savez_compressed(output/'trace.npz',motion=m,**foot)
        t,q=zip(*self.poses);np.savez_compressed(output/'view.npz',timestamps_s=t,qpos=q)
        report={'duration_s':float(m[-1,0]),'minimum_upright_cosine':float(m[:,4].min()),'max_stem_error_m':float(abs(m[:,2]).max()),'final_xyz_m':m[-1,1:4].tolist()}
        (output/'physical-report.json').write_text(json.dumps(report,indent=2)+'\n');return report

class AmpCupEnv(gym.Env):
    metadata={}
    def __init__(self,source,amp_source,profile,*,initial_yaw=-.7,evaluation_dir=None,raise_cup=True):
        super().__init__();self.source=Path(source);self.amp_source=Path(amp_source);self.profile=Path(profile)
        self.initial_yaw=float(initial_yaw);self.evaluation_dir=Path(evaluation_dir)if evaluation_dir else None;self.raise_cup=raise_cup
        self.action_space=gym.spaces.Box(-1.,1.,(15,),dtype=np.float32);self.observation_space=gym.spaces.Box(-10.,10.,(OBS_SIZE,),dtype=np.float32)
        self.plant=None;self.episode=0
    def _build(self):
        from evaluate_aero_receiver import make_plant
        from evaluate_dextra_in_stride import RearMarkerLocalizer
        from .running_policy import AmpRunningPolicy
        from .aero import aero_ready_joint_targets
        from .routes import Route
        self.plant=make_plant(self.source,Route('go',((0.,0.),(16.,0.)),2.),tracking_camera=CAMERAS['track'],pass_camera=CAMERAS['pass'],finish_camera=CAMERAS['finish'])
        self.policy=AmpRunningPolicy(self.plant.joint_names,self.plant.joint_ranges,self.amp_source)
        self.ready=aero_ready_joint_targets(self.policy.nominal,self.plant.joint_names)
        self.arms=np.array([self.plant.joint_names.index(f'{s}_{axis}_joint')for s in ('left','right')for axis in ('shoulder_roll','shoulder_yaw','shoulder_pitch','elbow','wrist_roll','wrist_pitch','wrist_yaw')])
        self.lower=np.array([i for i,n in enumerate(self.plant.joint_names)if n not in ARM_JOINTS]);assert len(self.lower)==15
        self.offsets=np.asarray(json.loads(self.profile.read_text())['out_staged_arm_offsets_rad'],float)
        if self.offsets.shape!=(14,)or not np.isfinite(self.offsets).all():raise ValueError('finite14-arm cup offsets required')
        self.localizer=RearMarkerLocalizer(allow_synthetic_fixture=True,max_range_m=8.)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if self.plant is None:self._build()
        import mujoco
        self.plant.reset(joint_positions=self.ready,release_position_m=(-10.,-10.,.085))
        self.plant.data.qpos[self.plant.base+3:self.plant.base+7]=(np.cos(self.initial_yaw/2),0.,0.,np.sin(self.initial_yaw/2));mujoco.mj_forward(self.plant.model,self.plant.data)
        self.policy.reset();self.tick=0;self.offset=np.zeros(15);self.sensors=MarkerSensors();self.last_camera='track';self.return_sum=0.;self.sensor_log=[];self.reward_log=[];self.episode+=1
        self.truth=TruthRecorder(self.plant)if self.evaluation_dir is not None else None
        if self.truth:self.truth.record_tick(0,self.plant.timestamp_s)
        return self._observe(),{}
    def _packet(self):
        from .runtime import RobotObservation
        return RobotObservation(self.plant.timestamp_s,*self.plant.observe(),np.zeros(29),None)
    def _stage(self,now):
        u=float(np.clip((now-1.6)/1.,0.,1.))if self.raise_cup else 0.
        return u,u*u*(3.-2.*u)
    def _drive(self,obs):
        from .football_routes import imu_yaw,heading_error
        return np.array([min(2.,1.8*max(0.,obs.timestamp-.4)),0.,np.clip(1.8*heading_error(0.,imu_yaw(obs.imu_quat_wxyz)),-.75,.75)])
    def _observe(self):
        from .football_routes import imu_yaw
        obs=self._packet();now=float(obs.timestamp)
        if self.tick%2==0 and now>=.8:
            order=[self.last_camera]+[n for n in CAMERAS if n!=self.last_camera];pose=None
            for name in order:
                frame,survey=self.plant.capture(name);self.sensors.capture_count+=1
                pose=self.localizer.estimate(frame,survey)
                if pose is not None:self.last_camera=name;self.sensors.valid_count+=1;break
            self.sensors.update(now,pose)
        else:self.sensors.update(now)
        xy,velocity,age=self.sensors.fields(now);yaw=imu_yaw(obs.imu_quat_wxyz);u,stage=self._stage(now)
        raw=np.r_[np.asarray(obs.joint_position)/2.,np.asarray(obs.joint_velocity)/10.,project_gravity(obs.imu_quat_wxyz),np.asarray(obs.imu_gyro)/5.,np.sin(yaw),np.cos(yaw),xy/[16.,2.],velocity/4.,age/.14,float(self.sensors.pose_valid),float(self.sensors.velocity_valid),self._drive(obs)/[2.,.5,.75],now/7.,stage,self.offset/.25]
        if raw.shape!=(OBS_SIZE,)or not np.isfinite(raw).all():raise ValueError('invalid sensor-only observation')
        self.obs=obs;self.vector=np.clip(raw,-10.,10.).astype(np.float32)
        self.sensor_log.append({'time_s':now,'pose_valid':self.sensors.pose_valid,'velocity_valid':self.sensors.velocity_valid,'xy_m':xy.tolist(),'velocity_mps':velocity.tolist(),'pose_age_s':age,'camera':self.last_camera})
        return self.vector
    def step(self,action):
        from .runtime import SkillGoal
        action=np.asarray(action,float)
        if action.shape!=(15,)or not np.isfinite(action).all():raise ValueError('finite15-residual action required')
        old=self.offset.copy();self.offset=np.clip(np.clip(action,-1.,1.)*.25,old-.04,old+.04)
        now=float(self.obs.timestamp);u,stage=self._stage(now);target=self.ready.copy();target[self.arms]+=stage*self.offsets
        command=self.policy.act(self.obs,SkillGoal(base_velocity=self._drive(self.obs)),upper_target=target)
        if np.any(self.offset):
            positions=command.target_position.copy();positions[self.lower]=np.clip(positions[self.lower]+self.offset,self.plant.joint_ranges[self.lower,0],self.plant.joint_ranges[self.lower,1]);command=replace(command,target_position=positions)
        self.plant.step(command,closure=.2,post_step=self.truth.feet.record if self.truth else None)
        self.tick+=1
        if self.truth:self.truth.record_tick(self.tick,self.plant.timestamp_s)
        vector=self._observe();reward,parts=sensor_reward(self.obs,self.sensors,self.offset,self.offset-old)
        w,x,y,z=self.obs.imu_quat_wxyz;terminated=bool(1.-2.*(x*x+y*y)<.54);truncated=self.tick>=350
        if terminated:reward-=5.
        self.return_sum+=reward;self.reward_log.append({'time_s':float(self.obs.timestamp),'reward':reward,**parts})
        info={}
        if terminated or truncated:
            row={'episode':self.episode,'return':self.return_sum,'steps':self.tick,'sensor_capture_count':self.sensors.capture_count,'valid_capture_count':self.sensors.valid_count,'sensor_schema':SCHEMA,'reward_schema':REWARD_SCHEMA}
            info['episode_sensor']=row
            if self.truth:
                folder=self.evaluation_dir/f'episode-{self.episode:03d}';self.truth.save(folder)
                (folder/'sensors.json').write_text(json.dumps(self.sensor_log,indent=2)+'\n');(folder/'rewards.json').write_text(json.dumps(self.reward_log,indent=2)+'\n');(folder/'sensor-report.json').write_text(json.dumps(row,indent=2)+'\n')
        return vector,float(reward),terminated,bool(truncated),info
    def close(self):
        if self.plant is not None:self.plant.close();self.plant=None

def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from functools import partial
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    class SaveProgress(BaseCallback):
        def _on_step(self):
            if self.num_timesteps%256==0:
                row={'timesteps':self.num_timesteps,'optimizer_updates':self.model._n_updates,'elapsed_s':time.monotonic()-start}
                with (args.output/'progress.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
                self.model.save(args.output/'latest.zip')
            for info in self.locals['infos']:
                if 'episode_sensor'in info:
                    with (args.output/'episodes.jsonl').open('a')as f:f.write(json.dumps(info['episode_sensor'])+'\n')
            return time.monotonic()-start<args.wall_seconds
    env=SubprocVecEnv([partial(AmpCupEnv,args.source,args.amp_source,args.profile)for _ in range(args.envs)],start_method='spawn')
    try:
        model=PPO('MlpPolicy',env,device=args.device,seed=args.seed,n_steps=64,batch_size=128,n_epochs=4,learning_rate=1e-4,gamma=.99,gae_lambda=.95,clip_range=.15,policy_kwargs={'net_arch':dict(pi=[128,128],vf=[128,128]),'log_std_init':-2.},verbose=0)
        torch.nn.init.zeros_(model.policy.action_net.weight);torch.nn.init.zeros_(model.policy.action_net.bias)
        model.receiver_sensor_schema=SCHEMA;model.receiver_reward_schema=REWARD_SCHEMA
        model.learn(total_timesteps=args.steps,callback=SaveProgress());model.save(args.output/'final.zip')
        receipt={'requested_steps':args.steps,'actual_steps':model.num_timesteps,'optimizer_updates':model._n_updates,'elapsed_s':time.monotonic()-start,'cuda_device_name':torch.cuda.get_device_name(0)if args.device.startswith('cuda')else None,'sensor_schema':SCHEMA,'reward_schema':REWARD_SCHEMA,'source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'profile_sha256':sha256(args.profile.read_bytes()).hexdigest(),'checkpoint_sha256':sha256((args.output/'final.zip').read_bytes()).hexdigest(),'no_truth_in_actor_critic_reward':True,'camera_surveys':CAMERAS,'physical_multicamera_integration_verified':False}
        (args.output/'training-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    finally:env.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--amp-source',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=16384);p.add_argument('--envs',type=int,default=8);p.add_argument('--wall-seconds',type=float,default=1800.);p.add_argument('--seed',type=int,default=173);p.add_argument('--device',default='cuda');args=p.parse_args();train(args)
if __name__=='__main__':main()
