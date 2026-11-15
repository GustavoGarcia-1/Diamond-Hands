"""Explicit gain modulation commanded relative to a causal RGB-D grasp event."""
from dataclasses import replace
import numpy as np

def modulate_arm_gains(command,arms,now,visual_gate_at,config):
    if config is None or visual_gate_at is None:return command,1.
    scale=float(config['kp_scale']);start=float(config.get('start_after_gate_s',.12));ramp=float(config.get('ramp_s',.04));hold=float(config.get('recover_after_gate_s',.45));recovery=float(config.get('recovery_s',.2))
    if not np.isfinite([scale,start,ramp,hold,recovery]).all()or not .1<=scale<=1. or not 0<=start<hold<=1. or not .02<=ramp<=.2 or not .04<=recovery<=.5:
        raise ValueError('bounded gain scale and visual-relative timing required')
    elapsed=now-visual_gate_at
    smooth=lambda x:float(np.clip(x,0.,1.))**2*(3.-2.*float(np.clip(x,0.,1.)))
    envelope=smooth((elapsed-start)/ramp)*(1.-smooth((elapsed-hold)/recovery))
    multiplier=1.-(1.-scale)*envelope
    if multiplier==1.:return command,multiplier
    kp=command.kp.copy();kp[np.asarray(arms)]*=multiplier
    return replace(command,kp=kp),multiplier
