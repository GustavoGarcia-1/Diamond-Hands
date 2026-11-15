import numpy as np
from g1_receiver._vendor.g1_runtime.pool.contracts import ActuatorCommand
from g1_receiver.visual_arm_compliance import modulate_arm_gains

def test_only_commanded_arm_stiffness_changes_inside_visual_window():
 command=ActuatorCommand(np.zeros(29),np.arange(29,dtype=float)+10.,np.ones(29),np.zeros(29),'test')
 arms=np.arange(15,29);cfg=dict(kp_scale=.25,start_after_gate_s=.12,ramp_s=.04,recover_after_gate_s=.45,recovery_s=.2)
 for time,gate in ((5.,None),(5.1,5.),(5.7,5.)):
  result,scale=modulate_arm_gains(command,arms,time,gate,cfg);assert result is command and scale==1.
 result,scale=modulate_arm_gains(command,arms,5.3,5.,cfg)
 assert scale==.25 and np.array_equal(result.kp[:15],command.kp[:15])
 assert np.array_equal(result.kp[arms],command.kp[arms]*.25)
 assert np.array_equal(result.kd,command.kd)and np.array_equal(result.target_position,command.target_position)
 assert np.array_equal(command.kp,np.arange(29)+10.)
