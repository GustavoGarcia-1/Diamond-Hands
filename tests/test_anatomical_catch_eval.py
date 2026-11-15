"""Predeclared anatomical receiving-surface, support and form gate behavior."""
import numpy as np
import pytest
from g1_receiver.anatomical_catch_eval import AnatomicalContactAudit,AnatomicalCatchCriteria,COLUMNS,GROUPS,INDEX,physics_fingerprint,score_anatomical_catch

WEIGHT=4.1202

def evidence():
 t=np.arange(1,1101)*.001;r=np.zeros((len(t),len(COLUMNS)));r[:,0]=t;r[:,3]=1.;touch=t>=.101
 r[touch,INDEX['left_finger_normal_n']]=3.;r[touch,INDEX['right_thumb_normal_n']]=3.
 r[touch,INDEX['left_finger_vertical_n']]=WEIGHT/2;r[touch,INDEX['right_thumb_vertical_n']]=WEIGHT/2
 form=[{'time_s':float(t),'metrics':{'palm_up_cos':[.9,.9],'palm_face_height_above_shoulders_m':[.1,.1]}}for t in np.arange(.08,.321,.02)]
 motion=np.c_[np.arange(0,1.101,.02),np.zeros((56,3)),np.ones(56)]
 return {'anatomical_contact':r,'anatomical_free_joint_verified':np.asarray(True)},form,motion

def score(a,form,motion,**kw):return score_anatomical_catch(a,release_time_s=0.,finish_time_s=1.1,ball_weight_n=WEIGHT,form_rows=form,motion=motion,physics_reference_match=True,**kw)

def test_all_native_finger_thumb_surfaces_count_without_legacy_tip_palm_omission():
 a,f,m=evidence();r=score(a,f,m)
 assert r['anatomical_hand_control']and r['realistic_lower_catch']
 assert r['longest_clean_bilateral_s']==pytest.approx(1.)
 assert r['first_anatomical_contact_s']==pytest.approx(.101)

def test_first_excluded_support_permanently_cuts_off_later_recovery():
 a,f,m=evidence();a['anatomical_contact'][299,INDEX['excluded_hand_normal_n']]=1.
 r=score(a,f,m);assert not r['anatomical_hand_control'];assert r['first_excluded_support_s']==pytest.approx(.3)
 assert r['longest_clean_bilateral_s']==pytest.approx(.199)

def test_self_contact_is_not_ball_contact_and_deep_penetration_rejects():
 a,f,m=evidence();rows=a['anatomical_contact'];rows[:,6:26]=0.;rows[:,INDEX['interhand_self_normal_n']]=10000.
 assert not score(a,f,m)['anatomical_hand_control']
 a,f,m=evidence();a['anatomical_contact'][400,INDEX['minimum_interhand_distance_m']]=-.006
 assert not score(a,f,m)['physics_checks']['no_deep_interhand_penetration']
 a,f,m=evidence();a['anatomical_contact'][400,INDEX['minimum_ball_contact_distance_m']]=-.011
 assert not score(a,f,m)['physics_checks']['no_deep_ball_penetration']

def test_missing_form_or_unverified_physics_cannot_pass_realistic_quality():
 a,f,m=evidence();r=score(a,None,m);assert r['anatomical_hand_control']and not r['realistic_lower_catch']
 r=score_anatomical_catch(a,release_time_s=0.,finish_time_s=1.1,ball_weight_n=WEIGHT,form_rows=f,motion=m)
 assert not r['realistic_lower_catch']
 a.pop('anatomical_free_joint_verified');assert not score(a,f,m)['anatomical_hand_control']

def test_height_exception_must_be_declared_and_misoriented_palms_still_fail():
 a,f,m=evidence()
 for row in f:row['metrics']['palm_face_height_above_shoulders_m']=[.4,.4]
 assert not score(a,f,m)['realistic_lower_catch']
 with pytest.raises(ValueError):score(a,f,m,required_intercept_height_above_shoulder_m=.4)
 assert score(a,f,m,required_intercept_height_above_shoulder_m=.4,height_exception_basis='predeclared commanded high throw')['realistic_lower_catch']
 for row in f:row['metrics']['palm_up_cos']=[.35,.35]
 assert not score(a,f,m,required_intercept_height_above_shoulder_m=.4,height_exception_basis='predeclared commanded high throw')['realistic_lower_catch']

def test_floor_flag_or_applied_load_rejects_even_without_contact_force():
 for key in ['ball_floor_contact','applied_ball_load']:
  a,f,m=evidence();a['anatomical_contact'][500,INDEX[key]]=1
  assert not score(a,f,m)['anatomical_hand_control']

def model_xml(camera='',friction='.2 .005 .0001',weld=''):
 return f'''<mujoco><option timestep=".001"/><worldbody>{camera}<geom name="floor" type="plane" size="2 2 .1"/>
 <body name="aero_left_palm" pos="0 .1 .8"><geom name="aero_left_palm_collision_1" size=".02"/><body name="aero_left_left_index_proximal_link"><geom name="left_finger_skin" size=".01"/></body><body name="aero_left_left_thumb_proximal_link"><geom name="left_thumb_skin" size=".01"/></body></body>
 <body name="aero_left_mount" pos="0 .1 .7"><geom name="mount" size=".01"/></body>
 <body name="aero_right_palm" pos="0 -.1 .8"><geom name="aero_right_palm_collision_1" size=".02"/></body>
 <body name="football" pos="0 0 1"><freejoint name="football_free"/><geom name="football_surface" size=".075" mass=".42" friction="{friction}"/></body></worldbody>{weld}</mujoco>'''

def test_surface_classification_excludes_mount_and_palmar_back_but_accepts_native_skin():
 import mujoco
 model=mujoco.MjModel.from_xml_string(model_xml());data=mujoco.MjData(model);mujoco.mj_kinematics(model,data);a=AnatomicalContactAudit(model)
 assert a.classify_surface(model.geom('left_finger_skin').id,[0,0,0],[0,0,1],data)=='left_finger'
 assert a.classify_surface(model.geom('left_thumb_skin').id,[0,0,0],[0,0,1],data)=='left_thumb'
 assert a.classify_surface(model.geom('mount').id,[0,0,0],[0,0,1],data)=='excluded_hand'
 palm=model.geom('aero_left_palm_collision_1').id
 assert a.classify_surface(palm,[0,.12,.8],[0,1,0],data)=='left_palmar'
 assert a.classify_surface(palm,[0,.08,.8],[0,-1,0],data)=='excluded_hand'
 assert a.classify_surface(palm,[0,.12,.8],[0,-1,0],data)=='excluded_hand'

def test_physics_fingerprint_ignores_camera_but_detects_contact_changes_and_ball_weld():
 import mujoco
 base=mujoco.MjModel.from_xml_string(model_xml());camera=mujoco.MjModel.from_xml_string(model_xml('<camera name="observer" pos="1 2 3"/>'));changed=mujoco.MjModel.from_xml_string(model_xml(friction='.9 .005 .0001'))
 assert physics_fingerprint(base)['sha256']==physics_fingerprint(camera)['sha256']
 assert physics_fingerprint(base)['sha256']!=physics_fingerprint(changed)['sha256']
 welded=mujoco.MjModel.from_xml_string(model_xml(weld='<equality><weld body1="football" body2="aero_left_palm"/></equality>'))
 with pytest.raises(ValueError,match='attached'):AnatomicalContactAudit(welded)
