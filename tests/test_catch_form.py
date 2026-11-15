"""Behavioral catch-form/context and independently oriented FK geometry tests."""
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest
from g1_receiver.catch_form import CatchFormContext,CatchFormEvaluator,form_penalty

def metric(up=(1.,1.),height=(.02,.02)):
 return {'palm_up_cos':list(up),'palm_face_height_above_shoulders_m':list(height)}

def fitted(**kwargs):
 return replace(CatchFormContext(over_shoulder=True,phase='receive',perception_timestamp_s=1.,causal_fit_valid=True,predicted_intercept_height_above_shoulder_m=.02,intercept_height_std_m=.02,time_to_arrival_s=.3,arrival_std_s=.02),**kwargs)

def test_contextual_palms_do_not_punish_frontal_catches_or_tucking():
 bad=metric(up=(-1.,-1.))
 assert form_penalty(bad,CatchFormContext(phase='receive'),1.)['penalty_per_s']==0
 assert form_penalty(bad,CatchFormContext(over_shoulder=True,phase='hold'),1.)['penalty_per_s']==0
 assert form_penalty(bad,CatchFormContext(over_shoulder=True,phase='approach'),1.)['penalty_per_s']==0
 assert form_penalty(bad,CatchFormContext(over_shoulder=True,phase='receive'),1.)['penalty_terms_per_s']['palm_up']==2.
 assert form_penalty(metric(up=(.358,.358)),CatchFormContext(over_shoulder=True,phase='receive'),1.)['penalty_per_s']>0
 assert form_penalty(metric(up=(.8,.8)),CatchFormContext(over_shoulder=True,phase='receive'),1.)['penalty_per_s']==0

def test_high_intercept_is_not_blanket_penalized_as_excessive_raise():
 hands=metric(height=(.50,.50));low=form_penalty(hands,fitted(),1.)
 high=form_penalty(hands,fitted(predicted_intercept_height_above_shoulder_m=.50),1.)
 assert low['penalty_terms_per_s']['excessive_raise']>0
 assert high['penalty_terms_per_s']['excessive_raise']==0
 assert not form_penalty(hands,fitted(phase='approach',time_to_arrival_s=2.,predicted_intercept_height_above_shoulder_m=.50),1.)['gates']['early_hands_penalty']

def test_height_and_early_hands_abstain_for_stale_or_uncertain_fit():
 for context in [fitted(causal_fit_valid=False),fitted(intercept_height_std_m=.2),fitted(perception_timestamp_s=.5),fitted(predicted_intercept_height_above_shoulder_m=None)]:
  r=form_penalty(metric(height=(.6,.6)),context,1.)
  assert r['penalty_terms_per_s']['excessive_raise']==r['penalty_terms_per_s']['early_hands']==0
  assert not r['gates']['fresh_causal_intercept_height']

def test_early_hands_requires_confident_distant_arrival_and_ages_eta():
 hands=metric(height=(.1,.1));r=form_penalty(hands,fitted(phase='approach',time_to_arrival_s=1.),1.04)
 assert r['penalty_terms_per_s']['early_hands']>0
 assert r['confident_arrival_lower_bound_s']==pytest.approx(.92)
 assert form_penalty(hands,fitted(phase='approach',time_to_arrival_s=.7),1.04)['penalty_terms_per_s']['early_hands']==0
 assert form_penalty(hands,fitted(phase='approach',time_to_arrival_s=2.,arrival_std_s=.2),1.)['penalty_terms_per_s']['early_hands']==0

@pytest.fixture
def rig():
 import mujoco
 body='<body name="pelvis"><freejoint name="floating_base_joint"/><geom size=".04" mass="1"/>'
 for side,sign in [('left',1),('right',-1)]:
  body+=f'<body name="{side}_shoulder_pitch_link" pos="0 {sign*.15} .3"><joint name="{side}_shoulder_pitch_joint" axis="0 1 0"/><geom size=".01" mass=".01"/><body name="{side}_elbow_link" pos=".2 0 0"><joint name="{side}_elbow_joint" axis="0 1 0"/><geom size=".01" mass=".01"/><body name="aero_{side}_palm" pos=".2 0 0" quat=".7071067812 {sign*.7071067812} 0 0">'
  for i,(x,z)in enumerate([(-.03,.07),(-.01,.07),(.01,.07),(.03,.055)],1):body+=f'<site name="aero_{side}_palm_collision_{i}" pos="{x} {sign*.017} {z}" size=".001"/>'
  body+=f'<geom name="aero_{side}_palm_collision_6" pos="0 {-sign*.007} .07" type="box" size=".01 .01 .01" mass=".1"/></body></body></body>'
 for i in range(25):body+=f'<body><joint name="extra{i}"/><geom size=".01" mass=".01"/></body>'
 body+='</body>';model=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+body+'</worldbody></mujoco>');names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i)for i in range(1,30)]
 return model,names

def packet():
 return SimpleNamespace(timestamp=1.,joint_position=np.zeros(29),imu_quat_wxyz=np.array([1.,0.,0.,0.]))

def test_chiral_face_normals_not_finger_longitudinal_axes(rig):
 model,names=rig;e=CatchFormEvaluator(model,names);r=e.evaluate(packet(),CatchFormContext(over_shoulder=True,phase='receive'))
 np.testing.assert_allclose(e.calibration['left']['normal_local'],[0,1,0],atol=1e-12)
 np.testing.assert_allclose(e.calibration['right']['normal_local'],[0,-1,0],atol=1e-12)
 np.testing.assert_allclose(r['metrics']['palm_up_cos'],[1,1],atol=1e-12)
 assert r['penalty_per_s']==0 and r['head_top_height_above_pelvis_m']is None
 json.dumps(r,allow_nan=False)

def test_private_fk_uses_encoders_imu_not_external_plant_state(rig):
 import mujoco
 model,names=rig;e=CatchFormEvaluator(model,names);obs=packet();baseline=e.evaluate(obs)
 external=mujoco.MjData(model);external.qpos[:3]=[50.,-80.,100.];mujoco.mj_forward(model,external)
 obs.ball_truth=np.array([np.nan]*3);obs.root_truth=external.qpos.copy()
 assert e.evaluate(obs)==baseline
 obs.joint_position[names.index('left_shoulder_pitch_joint')]=.4;r=e.evaluate(obs)
 assert r['metrics']['palm_up_cos'][0]<baseline['metrics']['palm_up_cos'][0]
 obs=packet();obs.imu_quat_wxyz=np.array([0.,1.,0.,0.]);upside=e.evaluate(obs,CatchFormContext(over_shoulder=True,phase='receive'))
 np.testing.assert_allclose(upside['metrics']['palm_up_cos'],[-1,-1],atol=1e-12)
 assert upside['penalty_per_s']==2.
 assert np.array_equal(obs.joint_position,np.zeros(29))
