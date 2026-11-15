"""Multistep Aero residual PPO with sensor-only actor, critic and reward.

World truth is confined to run_play's independent final evaluation, never sent
through the training queues. The worker's callback pauses every two body ticks.
"""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path
from queue import Queue
import sys
import threading
import time
import numpy as np
import gymnasium as gym
from .observable import ARM_JOINTS, project_gravity
from .perception import ColorBallDetector

SCHEMA = 'aero-residual-rgbd-fkcup-v1'
NORMALIZED_SCHEMA = 'aero-residual-rgbd-fkcup-v1-scaled-observation'
REWARD_SCHEMA = 'aero-visual-cup-reward-v1'
CORRECTED_REWARD_SCHEMA = 'aero-visual-cup-reward-v2-potential-debit'
ACTION_SIZE = 16
OBS_SIZE = 111

def scale_observation(vector):
    """Fixed physical-unit scales, with no fitted or privileged statistics."""
    scales = np.r_[np.full(29, 2.), np.full(29, 10.), np.ones(3), np.full(3, 5.),
                   np.full(3, .25), np.full(3, 5.), np.full(3, .06), .08,
                   1., 1., 3., np.ones(16), np.full(14, .18), .15, .15, 1., 1.]
    return (np.asarray(vector, dtype=np.float32) / scales).astype(np.float32)

class CupSensors:
    """Only explicit sensor packets reach this object; FK has separate state."""
    def __init__(self, model, joint_names, cup_offset=(.10, 0., .08), *, corrected_potential_reward=False):
        import mujoco
        self.mj = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.enc = [int(model.joint(n).qposadr[0]) for n in joint_names]
        self.base = int(model.joint('floating_base_joint').qposadr[0])
        self.palms = [model.body('aero_'+s+'_palm').id for s in ('left','right')]
        self.offset = np.asarray(cup_offset, dtype=float)
        self.corrected_potential_reward = corrected_potential_reward
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
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.base:self.base+7] = (0.,0.,0.,1.,0.,0.,0.)
        samples=list(self.encoder_history)
        measured_q=np.asarray(obs.joint_position); measured_imu=np.asarray(obs.imu_quat_wxyz)
        for left,right in zip(samples[:-1],samples[1:]):
            if left[0]<=capture<=right[0]:
                u=(capture-left[0])/max(1e-12,right[0]-left[0])
                measured_q=(1-u)*left[1]+u*right[1]
                rq=right[2] if np.dot(left[2],right[2])>=0 else -right[2]
                measured_imu=(1-u)*left[2]+u*rq
                break
        self.data.qpos[self.enc] = measured_q
        self.mj.mj_kinematics(self.model,self.data)
        w,x,y,z = np.asarray(measured_imu,dtype=float)/np.linalg.norm(measured_imu)
        R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                    [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                    [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        yaw=np.arctan2(R[1,0],R[0,0]);c,s=np.cos(yaw),np.sin(yaw)
        Y=np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])
        palms=(Y@R@self.data.xpos[self.palms].T).T
        cup=palms.mean(0)+(Y@R@self.offset)
        relative=Y@(ball.position_m-np.r_[pose.xy_m,pose.z_m])
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
                and .10<=np.linalg.norm(palms[0]-palms[1])<=.28)
        if stable and gap is not None and 0<gap<=.10:
            old=self.dwell;self.dwell+=gap
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

    def vector(self,obs,previous,actual):
        age=.08 if self.last_valid is None else np.clip(obs.timestamp-self.last_valid,0.,1.)
        e=self.e if self.valid else np.zeros(3)
        vel=self.vel if self.valid and self.velocity_valid else np.zeros(3)
        result=np.r_[obs.joint_position,obs.joint_velocity,project_gravity(obs.imu_quat_wxyz),obs.imu_gyro,
                     e,vel,self.sigma if self.valid else np.zeros(3),age,float(self.valid),
                     float(self.valid and self.velocity_valid),min(self.dwell,3.),previous,actual,self.paid]
        if result.shape!=(OBS_SIZE,) or not np.isfinite(result).all():raise ValueError('invalid sensor observation')
        return result.astype(np.float32)


def bounded_residual(action, previous, dt=.02):
    action=np.asarray(action,dtype=float)
    if action.shape!=(16,) or not np.isfinite(action).all():raise ValueError('expected finite 16D action')
    desired=np.clip(action,-1.,1.)*np.r_[np.full(14,.18),.15,.15]
    rate=np.r_[np.full(14,1.5),1.,1.]*dt
    return np.asarray(previous)+np.clip(desired-previous,-rate,rate)

class StopEpisode(Exception):pass

class EpisodeReturnLedger:
    """Gym reset emits no reward: log its discarded amount separately."""
    def __init__(self):
        self.reset_reward = 0.
        self.step_return = 0.

    def record(self, reward, *, reset=False):
        if reset:
            self.reset_reward += float(reward)
        else:
            self.step_return += float(reward)

class AeroResidualEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,reference,source,amp_source,episode_log=None,evaluation_dir=None,timeout_s=180.,normalized_observations=False,corrected_potential_reward=False):
        self.reference=Path(reference);self.source=Path(source);self.amp_source=Path(amp_source)
        self.episode_log=None if episode_log is None else Path(episode_log)
        self.evaluation_dir=None if evaluation_dir is None else Path(evaluation_dir)
        self.timeout_s=timeout_s;self.worker=None;self.episode=0
        self.normalized_observations=normalized_observations
        self.corrected_potential_reward=corrected_potential_reward
        self.action_space=gym.spaces.Box(-1.,1.,shape=(16,),dtype=np.float32)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(OBS_SIZE,),dtype=np.float32)
        self.last_vector=np.zeros(OBS_SIZE,dtype=np.float32)

    def _worker(self):
        plant=None
        try:
            script_root=Path(__file__).resolve().parents[2]/'scripts'
            if str(script_root) not in sys.path:sys.path.insert(0,str(script_root))
            import evaluate_aero_receiver as runner
            from .pass_control import PassGrip
            from .running_policy import AmpRunningPolicy
            from .routes import Route
            ref=json.loads(self.reference.read_text())
            config={k:v for k,v in ref['play'].items() if k in {f.name for f in fields(runner.Play)}}
            config['grip']=PassGrip(**config['grip']);play=runner.Play(**config)
            route=Route(play.route,tuple(map(tuple,ref['waypoints_m'])),play.speed)
            cameras={}
            for name,arg in [('pass','pass_camera'),('track','tracking_camera'),('finish','finish_camera')]:
                sensor=ref['camera_surveys'][name];p=np.asarray(sensor['position_m']);cameras[arg]=(p,p+np.asarray(sensor['rotation'])[:,2])
            for name in ('retention','retention_pose'):
                if name in ref['camera_surveys']:
                    sensor=ref['camera_surveys'][name];p=np.asarray(sensor['position_m'])
                    cameras[name+'_camera']=(p,p+np.asarray(sensor['rotation'])[:,2])
            plant=runner.make_plant(self.source,route,**cameras)
            policy=AmpRunningPolicy(plant.joint_names,plant.joint_ranges,self.amp_source)
            sensors=CupSensors(plant.model,plant.joint_names,corrected_potential_reward=self.corrected_potential_reward)
            indices=[plant.joint_names.index(n) for n in ARM_JOINTS]
            previous=np.zeros(16);actual=np.zeros(16);action=np.zeros(16);ticks=0;steps=0
            ledger=EpisodeReturnLedger()
            def callback(obs,pose,frame,survey,target,closure,enabled):
                nonlocal ticks,action,previous,actual,steps
                if not enabled:return target,closure
                sensors.update(obs,pose,frame,survey)
                if ticks%2==0:
                    vector=sensors.vector(obs,action,actual)
                    reward=float(sensors.reward);sensors.reward=0.
                    ledger.record(reward,reset=steps==0)
                    self.states.put(('state',vector,reward,{'visual_dwell_s':sensors.dwell,'sensor_valid':sensors.valid,'time_s':obs.timestamp}))
                    item=self.actions.get(timeout=self.timeout_s)
                    if item is None:raise StopEpisode()
                    previous=action.copy();action=np.asarray(item,dtype=float);steps+=1
                old=actual.copy();actual=bounded_residual(action,actual)
                sensors.reward-=.02*(.05*float(np.mean(action**2))+.02*float(np.mean(((actual-old)/.02)**2)))
                ticks+=1
                result=np.asarray(target).copy();result[indices]+=actual[:14]
                result[indices]=np.clip(result[indices],*plant.joint_ranges[indices].T)
                close=np.broadcast_to(np.asarray(closure,dtype=float),(2,)).copy()
                close=np.clip(close+actual[14:],0.,1.)
                return result,close
            output=None if self.evaluation_dir is None else self.evaluation_dir/f'episode-{self.episode:04d}'
            report=runner.run_play(plant,policy,play,route,output=output,record=False,record_poses=output is not None,residual_controller=callback)
            # report is deliberately never read for observation/reward/curriculum.
            sensors.finish()
            terminal_bonus=2. if sensors.valid and sensors.dwell>=.75 and sensors.upright>=.90 else 0.
            reward=float(sensors.reward+terminal_bonus)
            ledger.record(reward)
            self.states.put(('done',self.last_vector.copy(),reward,{'sensor_return':ledger.step_return,'reset_sensor_reward':ledger.reset_reward,'sensor_return_including_reset':ledger.step_return+ledger.reset_reward,'policy_steps':steps,'final_visual_dwell_s':sensors.dwell,'camera_captures':sensors.capture_count,'valid_captures':sensors.valid_count,'max_visual_dwell_s':sensors.max_dwell,'min_visible_cup_distance_m':sensors.min_cup_distance}))
        except StopEpisode:pass
        except BaseException as error:self.states.put(('error',repr(error),0.,{}))
        finally:
            if plant is not None:plant.close()

    def _read(self):
        kind,vector,reward,info=self.states.get(timeout=self.timeout_s)
        if kind=='error':raise RuntimeError(vector)
        if kind=='state':self.last_vector=vector
        else:vector=self.last_vector.copy()
        if self.normalized_observations:vector=scale_observation(vector)
        return kind,vector,reward,info

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.close();self.episode+=1
        self.states=Queue();self.actions=Queue();self.worker=threading.Thread(target=self._worker,daemon=True);self.worker.start()
        kind,vector,_,info=self._read()
        if kind!='state':raise RuntimeError('episode ended before visual activation')
        return vector,info

    def step(self,action):
        action=np.asarray(action,dtype=np.float32)
        if not self.action_space.contains(action):raise ValueError('action outside normalized 16D bounds')
        self.actions.put(action.copy());kind,vector,reward,info=self._read();done=kind=='done'
        if done and self.episode_log is not None:
            self.episode_log.parent.mkdir(parents=True,exist_ok=True)
            with self.episode_log.open('a') as f:f.write(json.dumps({'episode':self.episode,**info})+'\n')
        return vector,reward,done,False,info

    def close(self):
        if self.worker is not None and self.worker.is_alive():
            self.actions.put(None);self.worker.join(timeout=self.timeout_s)
            if self.worker.is_alive():raise RuntimeError('simulation worker did not close')
        self.worker=None


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference',type=Path,default=Path('reports/quarterback-pass/offset-fixed-controlled/report.json'))
    cli.add_argument('--source',type=Path,default=Path('/tmp/g1-aero-source'))
    cli.add_argument('--amp-source',type=Path,default=Path('/tmp/g1-pretrained-run-legged'))
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--steps',type=int,default=256);cli.add_argument('--wall-seconds',type=float,default=900.)
    cli.add_argument('--device',default='cuda');cli.add_argument('--seed',type=int,default=173)
    cli.add_argument('--n-envs',type=int,default=1)
    cli.add_argument('--normalized-observations',action='store_true',help='Fixed unit scaling; same sensors, actions and reward')
    cli.add_argument('--corrected-potential-reward',action='store_true',help='Opt-in v2 reward: settle potential once on invalid/stale/terminal transitions')
    cli.add_argument('--resume',type=Path)
    cli.add_argument('--eval',type=Path,help='Evaluate checkpoint against zero residual on matched reference')
    cli.add_argument('--smoke',action='store_true',help='One multistep zero-residual episode without PPO')
    args=cli.parse_args()
    if not 1 <= args.n_envs <= 8: cli.error('n-envs must be between one and eight')
    args.output.mkdir(parents=True,exist_ok=False)
    selected_schema=NORMALIZED_SCHEMA if args.normalized_observations else SCHEMA
    selected_reward_schema=CORRECTED_REWARD_SCHEMA if args.corrected_potential_reward else REWARD_SCHEMA
    metadata={'schema':selected_schema,'reward_schema':selected_reward_schema,'observation_size':OBS_SIZE,'action_size':16,'seed':args.seed,'device':args.device,
              'reference':str(args.reference),'reference_sha256':sha256(args.reference.read_bytes()).hexdigest(),
              'source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'policy_hz':25,
              'reward_basis':'distinct RGB-D captures, measured joints/IMU and separate encoder-only cup FK; no contacts',
              'cup_offset_m':[.10,0.,.08],'arm_limit_rad':.18,'closure_limit':.15,
              'steps_requested':args.steps,'n_envs':args.n_envs}
    (args.output/'contract.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if args.n_envs > 1 and not (args.smoke or args.eval):
        from functools import partial
        from stable_baselines3.common.vec_env import SubprocVecEnv
        env=SubprocVecEnv([partial(AeroResidualEnv,args.reference,args.source,args.amp_source,
                                  args.output/f'episodes-worker-{i}.jsonl',normalized_observations=args.normalized_observations,corrected_potential_reward=args.corrected_potential_reward) for i in range(args.n_envs)],
                         start_method='spawn')
    else:
        env=AeroResidualEnv(args.reference,args.source,args.amp_source,args.output/'episodes.jsonl',args.output/'evaluation' if args.eval else None,normalized_observations=args.normalized_observations,corrected_potential_reward=args.corrected_potential_reward)
    start=time.monotonic()
    try:
        if args.smoke:
            obs,info=env.reset(seed=args.seed);steps=0;total=0.
            while True:
                obs,reward,done,_,info=env.step(np.zeros(16,dtype=np.float32));steps+=1;total+=reward
                if done:break
            print(json.dumps({'steps':steps,'sensor_return':total,'final':info}),flush=True)
        else:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import BaseCallback
            if args.device.startswith('cuda'):
                import torch
                if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
                metadata['cuda_device_name']=torch.cuda.get_device_name(0)
                metadata['torch_version']=torch.__version__
            if args.eval:
                model=PPO.load(args.eval,device=args.device);results=[]
                if getattr(model,'receiver_sensor_schema',SCHEMA)!=selected_schema:
                    raise ValueError('checkpoint observation schema does not match requested environment')
                if getattr(model,'receiver_reward_schema',REWARD_SCHEMA)!=selected_reward_schema:
                    raise ValueError('checkpoint reward schema does not match requested environment')
                for name in ('zero','learned'):
                    env.evaluation_dir=args.output/'evaluation'/name
                    obs,_=env.reset(seed=args.seed);total=0.
                    while True:
                        action=np.zeros(16,dtype=np.float32) if name=='zero' else model.predict(obs,deterministic=True)[0]
                        obs,reward,done,_,info=env.step(action);total+=reward
                        if done:break
                    results.append({'policy':name,'sensor_return':total,**info})
                (args.output/'paired-evaluation.json').write_text(json.dumps(results,indent=2)+'\n')
            else:
                class BoundedSave(BaseCallback):
                    def _progress(self):
                        entry={'timesteps':self.num_timesteps,'optimizer_epochs_completed':int(self.model._n_updates),'elapsed_s':time.monotonic()-start}
                        with (args.output/'progress.jsonl').open('a') as stream:
                            stream.write(json.dumps(entry)+'\n');stream.flush()
                        print(json.dumps(entry),flush=True)
                    def _on_rollout_start(self):
                        if self.num_timesteps and self.num_timesteps%64==0:self._progress()
                    def _on_training_end(self):self._progress()
                    def _on_step(self):
                        if self.num_timesteps%256==0:self.model.save(args.output/f'checkpoint-{self.num_timesteps:06d}')
                        return time.monotonic()-start<args.wall_seconds
                import torch
                if args.resume:
                    model=PPO.load(args.resume,env=env,device=args.device)
                    if getattr(model,'receiver_sensor_schema',SCHEMA)!=selected_schema:
                        raise ValueError('resumed checkpoint observation schema mismatch')
                    if getattr(model,'receiver_reward_schema',REWARD_SCHEMA)!=selected_reward_schema:
                        raise ValueError('resumed checkpoint reward schema mismatch')
                    metadata['resumed_checkpoint_sha256']=sha256(args.resume.read_bytes()).hexdigest()
                else:
                    model=PPO('MlpPolicy',env,device=args.device,seed=args.seed,n_steps=64,batch_size=32,n_epochs=4,
                              learning_rate=3e-4,gamma=.99,policy_kwargs={'net_arch':[64,64],'log_std_init':-2.},verbose=1)
                    torch.nn.init.zeros_(model.policy.action_net.weight)
                    torch.nn.init.zeros_(model.policy.action_net.bias)
                model.receiver_sensor_schema=selected_schema
                model.receiver_reward_schema=selected_reward_schema
                model.save(args.output/'initial')
                model.learn(total_timesteps=args.steps,callback=BoundedSave())
                model.save(args.output/'final')
                metadata['actual_timesteps']=model.num_timesteps
                metadata['optimizer_epochs_completed']=int(model._n_updates)
    finally:
        env.close();metadata['elapsed_s']=time.monotonic()-start
        (args.output/'receipt.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':main()
