"""Data-integrity gates for scientific comparison export; no simulated outcome fixtures."""
import copy,hashlib,importlib.util,json,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np,pytest
spec=importlib.util.spec_from_file_location('comparison_plot',Path(__file__).parents[1]/'scripts/plot_receiver_comparison.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
@pytest.fixture
def batch(tmp_path):
 scenarios={i:dict(trial_id=i,lateral_m=0.,height_m=0.,release_s=0.)for i in range(50)};protocol=dict(scenarios_sha256='s'*64,horizons_s={'go':11.});sha='p'*64;source='a'*64;rows=[]
 for method in mod.METHODS:
  for i in range(50):
   score=dict(version=mod.SCHEMA,criteria=asdict(mod.AnatomicalCatchCriteria()),physics_checks={k:True for k in mod.CHECKS},form={'pass':True},upright=True,longest_clean_bilateral_s=1.,final_bilateral_fraction=1.,physics_reference_match=False,realistic_lower_catch=False,first_excluded_support_s=None,minimum_ball_contact_distance_m=-.006)
   report=dict(anatomical_score=score,warning_count=0,actual_finish_s=11.);folder=tmp_path/method/str(i);folder.mkdir(parents=True);(folder/'report.json').write_text(json.dumps(report));np.savez(folder/'trace.npz',motion=np.array([[0,0,0,.7,1.],[11,0,0,.7,1.]]));rows.append(dict(route='go',method=method,trial_id=i,status='completed',exit_code=0,report_path=f'{method}/{i}/report.json',protocol_sha256=sha,scenarios_sha256='s'*64,source_archive_sha256=source,profile_sha256='f'*64,scenario=scenarios[i],anatomical_score=score,warning_count=0,actual_finish_s=11.,primary_success=True,physics_identity_sha256='m'*64))
 manifest=dict(route='go',protocol_sha256=sha,scenarios_sha256='s'*64,source_archive_sha256=source,methods=list(mod.METHODS));path=tmp_path/'route-manifest.json';path.write_text(json.dumps(manifest));(tmp_path/'results.json').write_text(json.dumps(rows));return path,protocol,scenarios,sha,rows

def validate(batch,rows=None):
 path,protocol,scenarios,sha,original=batch
 if rows is not None:(path.parent/'results.json').write_text(json.dumps(rows))
 return mod.validate_route('go',path,protocol,scenarios,sha,verify_traces=False)
def test_wilson_zero_and_all_success_are_not_false_certainty():
 assert mod.wilson(0,50)[1]==pytest.approx(.0713476,rel=1e-5)
 assert mod.wilson(50,50)[0]==pytest.approx(.9286524,rel=1e-5)
 assert mod.wilson(0,50)[0]==0 and mod.wilson(50,50)[1]==1

def test_complete_pairs_preserve_reference_mismatch_separately(batch):
 result=mod.summarize(validate(batch));assert result['methods']['current']['successes']==50
 assert result['methods']['current']['original_reference_matches']==0
 assert result['methods']['current']['legacy_realistic_lower_catch_count']==0
 assert result['paired']['current-minus-earlier']['both_success']==50
@pytest.mark.parametrize('kind',['missing','duplicate','infrastructure','mixed_model','changed_scenario','changed_threshold'])
def test_rejects_incomplete_or_unmatched_scientific_data(batch,kind):
 rows=copy.deepcopy(batch[-1])
 if kind=='missing':rows.pop()
 elif kind=='duplicate':rows[-1]=copy.deepcopy(rows[0])
 elif kind=='infrastructure':rows[0]['status']='timeout'
 elif kind=='mixed_model':rows[0]['physics_identity_sha256']='z'*64
 elif kind=='changed_scenario':rows[0]['scenario']['height_m']=.01
 else:rows[0]['anatomical_score']['criteria']['max_ball_penetration_m']=.02
 with pytest.raises(mod.InvalidBatch):validate(batch,rows)
def test_warning_failure_is_counted_not_discarded(batch):
 rows=copy.deepcopy(batch[-1]);rows[0]['warning_count']=1;rows[0]['primary_success']=False;report=batch[0].parent/rows[0]['report_path'];d=json.loads(report.read_text());d['warning_count']=1;report.write_text(json.dumps(d));result=mod.summarize(validate(batch,rows));assert result['methods']['current']['successes']==49;assert result['methods']['current']['failed_check_counts']['zero_warnings']==1

def test_free_ball_check_uses_scene_joint_layout_and_all_parked_frames(tmp_path):
 (tmp_path/'scene.xml').write_text('<mujoco><worldbody><body><joint name="body"/><body><freejoint name="football_free"/></body></body></worldbody></mujoco>')
 q=np.zeros((3,8));q[:,1:4]=[1,2,1.4];np.savez(tmp_path/'view.npz',timestamps_s=[0,.5,1.1],qpos=q)
 np.savez(tmp_path/'anatomical-contact.npz',anatomical_contact=np.array([[1.001,0],[1.002,0]]),anatomical_contact_columns=['timestamp_s','applied_ball_load'])
 report={'release':{'release_time_s':1.,'release_xyz_m':[1,2,1.4]}}
 assert mod.verify_free_ball_evidence(tmp_path,report)['prerelease_frames']==2
 q[1,1]+=.01;np.savez(tmp_path/'view.npz',timestamps_s=[0,.5,1.1],qpos=q)
 with pytest.raises(mod.InvalidBatch,match='moves before release'):mod.verify_free_ball_evidence(tmp_path,report)

def test_free_ball_rejects_external_load_after_release(tmp_path):
 (tmp_path/'scene.xml').write_text('<mujoco><worldbody><body><freejoint name="football_free"/></body></worldbody></mujoco>')
 q=np.zeros((2,7));np.savez(tmp_path/'view.npz',timestamps_s=[0,1.1],qpos=q);np.savez(tmp_path/'anatomical-contact.npz',anatomical_contact=np.array([[1.001,1]]),anatomical_contact_columns=['timestamp_s','applied_ball_load'])
 with pytest.raises(mod.InvalidBatch,match='postrelease applied'):mod.verify_free_ball_evidence(tmp_path,{'release':{'release_time_s':1.,'release_xyz_m':[0,0,0]}})

def test_appendix_cardinality_is_explicit_and_does_not_loosen_main_batch(batch):
 path,protocol,scenarios,sha,rows=batch;rows=[r for r in rows if r['method']=='current']
 for r in rows:r['appendix_protocol_sha256']='x'*64
 (path.parent/'results.json').write_text(json.dumps(rows));manifest=json.loads(path.read_text());manifest['methods']=['current'];manifest['appendix_protocol_sha256']='x'*64;path.write_text(json.dumps(manifest))
 with pytest.raises(mod.InvalidBatch):mod.validate_route('go',path,protocol,scenarios,sha,verify_traces=False)
 assert len(mod.validate_route('go',path,protocol,scenarios,sha,verify_traces=False,methods=('current',),appendix_sha='x'*64)['rows'])==50
 with pytest.raises(mod.InvalidBatch,match='appendix protocol'):mod.validate_route('go',path,protocol,scenarios,sha,verify_traces=False,methods=('current',),appendix_sha='z'*64)
