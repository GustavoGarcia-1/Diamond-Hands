"""Print included frozen outcomes; this does not rescore omitted dense traces."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def main():
 study=json.loads((ROOT/'evidence/study-600/summary-450.json').read_text())
 print('Frozen 450: seeded simulation diagnostics, 50 trials per route/method; not hardware reliability')
 for route in study['routes']:
  print(route['route'],', '.join(f"{name}: {method['successes']}/50"for name,method in route['methods'].items()))
 appendix=json.loads((ROOT/'evidence/study-600/summary-150-appendix.json').read_text())
 print('Posthoc 150: exact ball with 1 mm uncertainty floor; separate from original 450')
 for route in appendix['routes']:print(route['route'],route['methods']['exact_ball_uncertainty_floor']['successes'],'/50')
 training=json.loads((ROOT/'evidence/training/three-route-hour-summary.json').read_text())
 print('Later one-hour PPO: fixed heterogeneous diagnostic counts, not success rates')
 for name,route in training['routes'].items():print(name,route['primary_counts_fixed_diagnostics'])
 latest=json.loads((ROOT/'evidence/nov15/results.json').read_text())
 print('November15 Go fresh paired confirmation:',latest['go_confirmation']['counts'])
 print('November15 Slant fixed seven factorial, not a population estimate:')
 for name,row in latest['slant_factorial']['aggregate'].items():
  print(name,row['retained_with_form_and_upright'],'/7 retained')
 if 'slant_confirmation' in latest:
  stats=latest['slant_confirmation']['statistics']
  print('November15 Slant fresh confirmation, both with the same closing limit:',
        {name:stats[name]['successes'] for name in ['zero','learned']})
  print('Learned minus scripted paired estimate and 95% interval:',
        stats['paired']['learned_minus_zero'],stats['paired']['paired_percentile_bootstrap95'])
  print('No conditional learning gain established; confirmation data excluded from further tuning.')
if __name__=='__main__':main()
