"""Fixture isolation, deterministic seed replay, and validated gain mapping."""
import json
import numpy as np
import pytest
from g1_receiver.native_cup_continuation import NativeContinuationEnv,mutate_throw_only
from g1_receiver.native_cup_training import NativeCatchEnv

def donor():return {'play':{'quarterback_target_m':[6.6,0,1.2],'field_catch_window_m':[6.6,0,1.2],'field_approach_axis_xy':[1,0]},'pass_camera':{'fixed':True},'native_grip':{'mode':'staged'}}

def test_throw_mutation_only_and_seed_replay(tmp_path,monkeypatch):
 profile=tmp_path/'profile.json';profile.write_text(json.dumps(donor()))
 def no_worker_reset(env,**kwargs):env.episode+=1;return np.zeros(169,np.float32),{}
 monkeypatch.setattr(NativeCatchEnv,'reset',no_worker_reset)
 envs=[NativeContinuationEnv(profile,tmp_path,tmp_path,fixture_root=tmp_path/str(i))for i in range(2)]
 histories=[]
 for env in envs:
  rows=[]
  for i in range(4):
   observation,info=env.reset(seed=173 if i==0 else None);assert not info;assert observation.shape==(169,)
   profile_now=json.loads(env.reference.read_text());delta=np.array(profile_now['play']['quarterback_target_m'])[1:]-np.array(donor()['play']['quarterback_target_m'])[1:];assert np.all(abs(delta)<=.02);assert profile_now==mutate_throw_only(donor(),delta);rows.append(profile_now)
  histories.append(rows)
 assert histories[0]==histories[1] and histories[0][0]!=histories[0][1]
 for delta in [[.021,0],[np.nan,0],[0,0,0]]:
  with pytest.raises(ValueError):mutate_throw_only(donor(),delta)
 assert mutate_throw_only(donor(),[0,0])==donor()

def test_effective_action_gain_and_zero_exact(tmp_path,monkeypatch):
 profile=tmp_path/'profile.json';profile.write_text(json.dumps(donor()));env=NativeContinuationEnv(profile,tmp_path,tmp_path,fixture_root=tmp_path/'fixtures',action_gain=.5)
 monkeypatch.setattr(NativeCatchEnv,'step',lambda self,action:action.copy())
 action=np.linspace(-1,1,20,dtype=np.float32);np.testing.assert_array_equal(env.step(action),action*np.float32(.5));np.testing.assert_array_equal(env.step(np.zeros(20,np.float32)),np.zeros(20,np.float32))
 with pytest.raises(ValueError):env.step(np.full(20,1.1,np.float32))
