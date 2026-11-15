"""Validate complete paired receiver batches and export scientific static figures.

Usage: PYTHONPATH=src python scripts/plot_receiver_comparison.py --protocol PROTOCOL
 --manifest go=GO/route-manifest.json --manifest out=OUT/route-manifest.json
 --manifest slant=SLANT/route-manifest.json --output FRESH
Use --only-routes go explicitly for a completed single-route interim artifact.
Missing/infrastructure-failed trials never become invented physical failures.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys,textwrap,platform
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'src')not in sys.path:sys.path.insert(0,str(ROOT/'src'))
from g1_receiver.anatomical_catch_eval import SCHEMA,AnatomicalCatchCriteria,score_anatomical_catch,INDEX,GROUPS
METHODS=('current','earlier','exact_ball_diagnostic')
CHECKS={'physics_clock_and_endpoint','initial_free_flight','verified_free_joint_no_attachment','continuous_anatomical_bilateral_075s','final_anatomical_hand_hold','no_excluded_floor_body_or_applied_support','load_bearing_hands','no_deep_interhand_penetration','no_deep_ball_penetration'}
class InvalidBatch(ValueError):pass
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(value,message):
 if not value:raise InvalidBatch(message)
def wilson(successes,total,z=1.959963984540054):
 require(total>0 and 0<=successes<=total,'invalid binomial count');p=successes/total;den=1+z*z/total;center=(p+z*z/(2*total))/den;half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den;return (0. if successes==0 else max(0.,center-half)),(1. if successes==total else min(1.,center+half))
def analysis_environment():
 info=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,matplotlib=None,font_family=None)
 try:
  import matplotlib
  from matplotlib import font_manager
  info['matplotlib']=matplotlib.__version__;font=font_manager.findfont(font_manager.FontProperties(family=['Arial','Helvetica','DejaVu Sans']));info['font_family']=font_manager.get_font(font).family_name;info['font_file']=font
 except ImportError:pass
 return info
def verify_free_ball_evidence(trial,report):
 view=np.load(trial/'view.npz',allow_pickle=False);q=view['qpos'];time=view['timestamps_s'];release=float(report['release']['release_time_s']);scene=next((p for p in [trial/'scene.xml',trial.parent/'scene.xml',trial.parent/'compiled-scene.xml',trial.parent.parent/'compiled-scene.xml']if p.is_file()),trial/'scene.xml');require(scene.is_file(),'missing archived scene for independent ball address check');joints=[e for e in ET.parse(scene).getroot().find('worldbody').iter()if e.tag in('joint','freejoint')];require(joints[-1].get('name')=='football_free','football freejoint no longer final; explicit address audit required');nq=sum(7 if e.tag=='freejoint'or e.get('type')=='free'else 4 if e.get('type')=='ball'else 1 for e in joints);require(nq==q.shape[1],'scene/view qpos layout differs');before=time<release-1e-9;require(bool(before.any()),'no prerelease frames');origin=np.asarray(report['release']['release_xyz_m']);drift=float(np.max(np.linalg.norm(q[before,nq-7:nq-4]-origin,axis=1)));require(drift<1e-9,'ball moves before release');archive=np.load(trial/'anatomical-contact.npz',allow_pickle=False);a=archive['anatomical_contact'];columns=archive['anatomical_contact_columns'].tolist();post=a[:,0]>=release-1e-9;loads=int(np.count_nonzero(a[post,columns.index('applied_ball_load')]));require(loads==0,'postrelease applied ball load');return dict(prerelease_frames=int(before.sum()),max_prerelease_position_drift_m=drift,postrelease_applied_load_frames=loads,scene_sha256=digest(scene))
def primary(score,finish,warnings,horizon):
 require(score.get('version')==SCHEMA,'unrecognized anatomical score version')
 require(score.get('criteria')==asdict(AnatomicalCatchCriteria()),'anatomical criteria differ from frozen v1')
 checks=score.get('physics_checks',{});require(set(checks)==CHECKS and all(type(v)is bool for v in checks.values()),'missing/nonboolean physical gate')
 require(type(score.get('form',{}).get('pass'))is bool,'missing form gate');require(warnings is not None and int(warnings)==warnings and warnings>=0,'missing/noninteger warnings');require(np.isfinite(finish)and finish>=0,'invalid actual finish')
 return bool(all(checks.values())and score['form']['pass']and score.get('upright')is True and finish>=horizon-1e-7 and warnings==0)
def canonical_protocol(path):
 protocol=read(path);require(protocol.get('frozen')is True,'protocol is not frozen');require(protocol.get('scenario_count')==50 and protocol.get('method_count')==3,'expected frozen50×3 protocol');require(tuple(protocol.get('methods',[]))==METHODS,'method identity/order differs');scenarios=read(Path(path).parent/'scenarios.json');require(digest(Path(path).parent/'scenarios.json')==protocol['scenarios_sha256'],'scenario file SHA mismatch');require(scenarios==protocol['scenarios'],'embedded scenarios differ');require(len(scenarios)==50 and {r['trial_id']for r in scenarios}==set(range(50)),'invalid scenario IDs')
 for s in scenarios:
  require(all(np.isfinite(s[k])and abs(s[k])<=.020000000001 for k in['lateral_m','height_m','release_s']),'scenario outside predeclared engineering box')
 return protocol,{s['trial_id']:s for s in scenarios}
def validate_route(route,path,protocol,scenario_map,protocol_sha,verify_traces=True,methods=METHODS,appendix_sha=None):
 path=Path(path);base=path.parent;manifest=read(path);require(manifest.get('route')==route,'route manifest label mismatch');require(manifest.get('protocol_sha256')==protocol_sha,'route protocol hash differs');require(manifest.get('scenarios_sha256')==protocol['scenarios_sha256'],'route scenario hash differs');require(set(manifest.get('methods',[]))==set(methods),'route methods differ');source=manifest.get('source_archive_sha256');require(isinstance(source,str)and len(source)==64,'missing source archive identity');rows=read(base/manifest.get('results_file','results.json'));require(isinstance(rows,list)and len(rows)==50*len(methods),f'{route}: incomplete batch: expected{50*len(methods)} rows');keys=[(r.get('method'),r.get('trial_id'))for r in rows];require(len(set(keys))==50*len(methods) and set(keys)=={(m,i)for m in methods for i in range(50)},f'{route}: duplicates/missing/misidentified pairs');require(all(r.get('status')=='completed'and r.get('exit_code',0)==0 for r in rows),f'{route}: infrastructure failure/incomplete outcomes; no finished graphs');identities=set();validated=[]
 if appendix_sha is not None:require(manifest.get('appendix_protocol_sha256')==appendix_sha,'appendix protocol hash differs')
 for row in rows:
  if appendix_sha is not None:require(row.get('appendix_protocol_sha256')==appendix_sha,'row appendix protocol hash differs')
  name=f'{route}/{row["method"]}/{row["trial_id"]}';require(row.get('route')==route,name+': route mismatch');require(row.get('protocol_sha256')==protocol_sha and row.get('scenarios_sha256')==protocol['scenarios_sha256'],name+': input hashes differ');require(row.get('source_archive_sha256')==source,name+': source identity differs');require(row.get('scenario')==scenario_map[row['trial_id']],name+': paired perturbation differs');require(isinstance(row.get('profile_sha256'),str)and len(row['profile_sha256'])==64,name+': profile identity missing');identity=row.get('physics_identity_sha256');require(isinstance(identity,str)and len(identity)==64,name+': physics identity missing');identities.add(identity)
  score=row.get('anatomical_score');require(isinstance(score,dict),name+': full anatomical score missing');horizon=float(protocol['horizons_s'][route]);finish=float(row['actual_finish_s']);warnings=row['warning_count'];passed=primary(score,finish,warnings,horizon);require(type(row.get('primary_success'))is bool and row['primary_success']==passed,name+': exported primary differs from frozen predicate');report_path=Path(row['report_path']);report_path=report_path if report_path.is_absolute()else base/report_path;require(report_path.is_file(),name+': report file missing');report=read(report_path);require(report.get('anatomical_score')==score,name+': embedded score differs from report');require(report.get('warning_count')==warnings,name+': warning count differs');motion=np.load(report_path.parent/'trace.npz',allow_pickle=False)['motion'];require(abs(float(motion[-1,0])-finish)<1e-7,name+': finish differs from authoritative trace');reported_finish=report.get('duration_s',report.get('actual_finish_s'));require(reported_finish is None or abs(float(reported_finish)-finish)<1e-7,name+': finish differs from report')
  hold=float(score['longest_clean_bilateral_s']);require(np.isfinite(hold)and 0<=hold<=horizon+1e-7,name+': invalid clean hold duration');raw=None
  if verify_traces:
   trial=report_path.parent;meta=read(trial/'anatomical-metadata.json');contact=dict(np.load(trial/'anatomical-contact.npz',allow_pickle=False));rescored=score_anatomical_catch(contact,release_time_s=report['release']['release_time_s'],finish_time_s=finish,ball_weight_n=meta['ball_weight_n'],form_rows=read(trial/'anatomical-form.json'),motion=motion[:,:5],physics_reference_match=meta['physics_reference_match'],verified_free_ball=meta['free_ball_joint_and_no_attachment_verified']);require(rescored==score,name+': full trace rescore mismatch');a=contact['anatomical_contact'];left=[INDEX[g+'_normal_n']for g in GROUPS[:3]];right=[INDEX[g+'_normal_n']for g in GROUPS[3:6]];threshold=AnatomicalCatchCriteria().min_normal_per_hand_n;b=(a[:,left].sum(1)>=threshold)&(a[:,right].sum(1)>=threshold);ed=np.diff(np.r_[False,b,False].astype(int));start=np.flatnonzero(ed==1);end=np.flatnonzero(ed==-1);raw=float(max(end-start,default=0)*np.median(np.diff(a[:,0])))
  failed=[k for k,v in score['physics_checks'].items()if not v]
  if not score['form']['pass']:failed.append('form.pass')
  if score.get('upright')is not True:failed.append('upright')
  if finish<horizon-1e-7:failed.append('full_horizon')
  if warnings!=0:failed.append('zero_warnings')
  free_ball_evidence=verify_free_ball_evidence(report_path.parent,report)if verify_traces else None
  validated.append(dict(free_ball_evidence=free_ball_evidence,donor_source_archive_sha256=row.get("donor_source_archive_sha256"),failed_primary_checks=failed,route=route,method=row['method'],trial_id=row['trial_id'],primary_success=passed,clean_hold_s=hold,raw_native_bilateral_s=raw,actual_finish_s=finish,warning_count=warnings,final_bilateral_fraction=score['final_bilateral_fraction'],form_pass=score['form']['pass'],upright=score.get('upright'),original_reference_match=score.get('physics_reference_match'),realistic_lower_catch=score['realistic_lower_catch'],first_excluded_support_s=score['first_excluded_support_s'],overlap_mm=-1000*score['minimum_ball_contact_distance_m'],actual_release_s=float(report['release']['release_time_s'])if 'release'in report else None,requested_release_offset_s=row['scenario']['release_s'],report_sha256=digest(report_path),profile_sha256=row['profile_sha256'],physics_identity_sha256=identity))
 require(len(identities)==1,route+': physical models differ across scenarios/methods');expected=manifest.get('physics_identity_sha256');require(expected is None or expected==next(iter(identities)),route+': declared model identity differs');validated.sort(key=lambda r:(methods.index(r['method']),r['trial_id']));return dict(route=route,manifest_sha256=digest(path),results_sha256=digest(base/manifest.get('results_file','results.json')),source_archive_sha256=source,physics_identity_sha256=next(iter(identities)),labels=manifest.get('labels',{}),comparison=manifest.get('comparison','Route-specific frozen controller bundles'),trace_rescores_verified=verify_traces,prerelease_view_and_postrelease_applied_load_verified=verify_traces,rows=validated)
def summarize(route):
 result={k:v for k,v in route.items()if k!='rows'};result['methods']={};paired={r['method']:{x['trial_id']:x for x in route['rows']if x['method']==r['method']}for r in route['rows']}
 for method in METHODS:
  rows=list(paired[method].values());success=sum(r['primary_success']for r in rows);hold=np.array([r['clean_hold_s']for r in rows]);low,high=wilson(success,50);failures={k:sum(k in r['failed_primary_checks']for r in rows)for k in sorted(CHECKS|{'form.pass','upright','full_horizon','zero_warnings'})};result['methods'][method]=dict(failed_check_counts=failures,failure_counts_overlap_not_causal_partition=True,successes=success,total=50,rate=success/50,wilson95=[low,high],actual_release_range_s=[min(r['actual_release_s']for r in rows),max(r['actual_release_s']for r in rows)]if all(r.get('actual_release_s')is not None for r in rows)else None,mean_clean_hold_s=float(hold.mean()),median_clean_hold_s=float(np.median(hold)),clean_hold_quantiles_s=np.quantile(hold,[0,.25,.5,.75,1]).tolist(),original_reference_matches=sum(r['original_reference_match']is True for r in rows),legacy_realistic_lower_catch_count=sum(r['realistic_lower_catch']for r in rows))
 result['paired']={}
 for candidate,reference in [('current','earlier'),('exact_ball_diagnostic','current')]:
  c=[paired[candidate][i]for i in range(50)];r=[paired[reference][i]for i in range(50)];delta=[x['clean_hold_s']-y['clean_hold_s']for x,y in zip(c,r)];wins=sum(x['primary_success']and not y['primary_success']for x,y in zip(c,r));losses=sum(not x['primary_success']and y['primary_success']for x,y in zip(c,r));both=sum(x['primary_success']and y['primary_success']for x,y in zip(c,r));result['paired'][candidate+'-minus-'+reference]=dict(candidate_only_success=wins,reference_only_success=losses,both_success=both,both_failure=50-wins-losses-both,success_rate_difference=(wins-losses)/50,mean_clean_hold_difference_s=float(np.mean(delta)),median_clean_hold_difference_s=float(np.median(delta)),clean_hold_differences_by_trial_s=delta,no_independent_samples_assumption=True)
 return result
COLORS={'current':'#202020','earlier':'#929292','exact_ball_diagnostic':'#376d7b'}
def figures(routes,summaries,protocol,output,prefix='',web=False):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':10,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
 footer='50 paired seeded throws per route · Wilson95% for the declared simulation sampler · Exact ball bypasses visibility/confidence; not deployable.\nDifferent route models; no pooled success rate or hardware reliability claim. Hold duration alone does not enforce every gate.'
 count=len(routes)
 for kind in ['success-rates','clean-hold-distributions','paired-hold-differences']:
  fig,axes=plt.subplots(1,count,figsize=(5*count,4.4),squeeze=False)
  for ax,route,summary in zip(axes[0],routes,summaries):
   labels=([{'current':'Current','earlier':'Delayed grip'if route['route']=='slant'else'Earlier','exact_ball_diagnostic':'Exact ball'}[m]for m in METHODS]if web else[route['labels'].get(m,{'current':'Current','earlier':'Earlier','exact_ball_diagnostic':'Exact ball diagnostic'}[m])for m in METHODS]);ax.set_title(route['route'].capitalize());ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
   if kind=='success-rates':
    rates=np.array([summary['methods'][m]['rate']for m in METHODS]);ci=np.array([summary['methods'][m]['wilson95']for m in METHODS]);bars=ax.bar(np.arange(3),rates*100,color=[COLORS[m]for m in METHODS],width=.6);bars[2].set_hatch('//');ax.errorbar(np.arange(3),rates*100,yerr=np.vstack((rates-ci[:,0],ci[:,1]-rates))*100,fmt='none',color='#202020',capsize=4);ax.set_xticks(np.arange(3),labels,rotation=15,ha='right');ax.set_ylim(0,112);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Retained catch under\ndeclared model (%)')
    for i,m in enumerate(METHODS):ax.text(i,max(rates[i]*100,ci[i,1]*100)+3,f"{summary['methods'][m]['successes']}/50",ha='center')
   elif kind=='clean-hold-distributions':
    for m,label,style in zip(METHODS,labels,['-','--',':']):
     values=np.sort([r['clean_hold_s']for r in route['rows']if r['method']==m]);x=np.r_[0.,values,protocol['horizons_s'][route['route']]];y=np.r_[0.,np.arange(1,51)/50,1.];ax.step(x,y*100,where='post',label=label,color=COLORS[m],linestyle=style,linewidth=2)
    maximum=max(r['clean_hold_s']for r in route['rows']);limit=max(.05,1.1*maximum);ax.set_xlim(0,limit);ax.text(.02,.97,('Route-specific zoom; 0.75 s minimum outside view'if limit<.75 else'Route-specific duration axis; minimum 0.75 s'),transform=ax.transAxes,va='top',fontsize=8);ax.set_ylim(0,103);ax.set_xlabel('Longest clean bilateral hold (s)');ax.set_ylabel('Empirical fraction with hold ≤ x (%)');ax.legend(fontsize=8,frameon=False,loc='lower right')
   else:
    delta=summary['paired']['current-minus-earlier']['clean_hold_differences_by_trial_s'];ax.axhline(0,color='#999',linewidth=1);ax.scatter(range(50),delta,s=15,color=COLORS['current']);ax.set_xlabel('Paired trial ID');ax.set_ylabel('Current − earlier clean hold (s)');extent=max(.1,max(abs(x)for x in delta)*1.15);ax.set_ylim(-extent,extent);ax.text(.02,.97,f"Mean {np.mean(delta):+.3f}s · median {np.median(delta):+.3f}s",transform=ax.transAxes,va='top',fontsize=9)
  if not web:fig.text(.03,.015,'\n'.join(textwrap.fill(line,width=65*count)for line in footer.split('\n')),fontsize=8,color='#666')
  fig.tight_layout(rect=(0,0 if web else .25 if count==1 else .16,1,1))
  for extension in ['png','svg']:fig.savefig(output/(prefix+kind+'.'+extension),dpi=180)
  plt.close(fig)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--manifest',action='append',required=True,help='route=/path/route-manifest.json');p.add_argument('--only-routes',nargs='+',choices=['go','out','slant']);p.add_argument('--output',type=Path,required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args();require(not a.output.exists(),'output must be fresh');a.output.mkdir(parents=True)
 try:
  protocol,scenario_map=canonical_protocol(a.protocol);inputs={}
  for value in a.manifest:
   route,sep,path=value.partition('=');require(bool(sep)and route not in inputs,'invalid/duplicate route input');inputs[route]=Path(path)
  expected=a.only_routes or protocol['routes'];require(set(inputs)==set(expected),'missing/unexpected route manifests; use --only-routes explicitly for an interim route');routes=[validate_route(route,inputs[route],protocol,scenario_map,digest(a.protocol),verify_traces=True)for route in expected];summaries=[summarize(r)for r in routes];result=dict(analysis_environment=analysis_environment(),schema='receiver-comparison-analysis-v1',protocol_sha256=digest(a.protocol),scenarios_sha256=protocol['scenarios_sha256'],requested_routes=expected,all_requested_routes_complete=True,full_protocol_complete=set(expected)==set(protocol['routes']),interval='Wilson95%; per-route seeded engineering sampler, not hardware reliability',no_pooled_route_rate=True,original_reference_flags_preserved_separately=True,source_script_sha256=digest(__file__),routes=summaries);(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
  rows=[r for route in routes for r in route['rows']]
  with(a.output/'trials.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  if not a.validate_only:
   figures(routes,summaries,protocol,a.output)
   for route,summary in zip(routes,summaries):
    figures([route],[summary],protocol,a.output,prefix=route['route']+'-')
    figures([route],[summary],protocol,a.output,prefix=route['route']+'-web-',web=True)
  generated={p.name:dict(sha256=digest(p),bytes=p.stat().st_size)for p in a.output.iterdir()if p.is_file()};(a.output/'artifacts.json').write_text(json.dumps(generated,indent=2)+'\n')
  print(json.dumps({'routes':expected,'completed_trials':len(rows),'full_protocol_complete':result['full_protocol_complete']}))
 except Exception as error:
  (a.output/'validation-error.json').write_text(json.dumps({'valid':False,'error':str(error),'no_finished_graphs_generated':True},indent=2)+'\n');raise
if __name__=='__main__':main()
