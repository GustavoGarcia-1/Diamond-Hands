import numpy as np
from g1_receiver.lower_cup_grip import NativeHandCalibration,NativeCupGrip,OPEN
from g1_receiver.native_grip_compliance import CompliantNativeCupGrip

def calibration():
 return NativeHandCalibration(np.tile([[.05852,.110387]]*4+[[0,1.7453],[.026152,.038389],[.081568,.112138]],(2,1)),np.arange(14))

def test_disabled_cap_matches_original_commands_and_native_trace_exactly():
 c=calibration();cfg=dict(preload=[.2]*7,capture=[.85]*7,ramp_s=.2)
 a=NativeCupGrip(c,cfg);b=CompliantNativeCupGrip(c,cfg)
 for t in (0.,1.,2.,2.1,2.3):
  e=c.targets(np.full(7,min(.8,t/4)))
  assert np.array_equal(a.targets(t,.2,1.,2.,e),b.targets(t,.2,1.,2.,e))
 assert np.array_equal(a.rows,b.rows)

def test_visual_delay_then_sampled_encoder_cap_preserves_opposition_and_logs_actual():
 c=calibration();cfg=dict(preload=[0]*7,capture=[1]*7,ramp_s=.1,compliance=dict(maximum_closing_error_m=.001,after_visual_gate_s=.12))
 g=CompliantNativeCupGrip(c,cfg);e=np.tile(OPEN,2)
 before=g.targets(2.1,0.,1.,2.,e)
 assert before[0]<e[0]-.001
 after=g.targets(2.12,0.,1.,2.,e)
 assert np.allclose(after[g.tendons],e[g.tendons]-.001)
 assert after[4]==c.ranges[4,1] and after[11]==c.ranges[11,1]
 assert np.array_equal(np.asarray(g.rows[-1][15:29]),after)
 shifted=e.copy();shifted[g.tendons]-=.002
 next_target=g.targets(2.14,0.,1.,2.,shifted)
 assert np.allclose(next_target[g.tendons],shifted[g.tendons]-.001)
 assert np.all(next_target>=c.ranges[:,0])and np.all(next_target<=c.ranges[:,1])
