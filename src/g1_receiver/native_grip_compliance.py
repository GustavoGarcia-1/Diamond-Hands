"""Opt-in sampled tendon target-error cap using native actuator encoders only.

This is not a contact detector or a continuous force limit. Official model gains,
force limits and plant rate limits are unchanged. Between controller updates,
external motion can still increase actuator error and contact force.
"""
import numpy as np
from .lower_cup_grip import NativeCupGrip

class CompliantNativeCupGrip(NativeCupGrip):
    def __init__(self,calibration,config):
        super().__init__(calibration,config)
        settings=config.get('compliance')
        self.cap=None if settings is None else float(settings['maximum_closing_error_m'])
        self.delay=0. if settings is None else float(settings.get('after_visual_gate_s',.12))
        if self.cap is not None and(not np.isfinite(self.cap)or not .0001<=self.cap<=.005):
            raise ValueError('sampled tendon error cap must be between0.1and5mm')
        if not np.isfinite(self.delay)or not 0.<=self.delay<=.5:raise ValueError('bounded visual-relative cap delay required')
        self.tendons=np.array([0,1,2,3,5,6,7,8,9,10,12,13]);self.compliance_rows=[]
    def targets(self,now,legacy_closure,release_at,visual_gate_at,encoders):
        requested=super().targets(now,legacy_closure,release_at,visual_gate_at,encoders)
        result=requested.copy();active=self.cap is not None and visual_gate_at is not None and now>=visual_gate_at+self.delay-1e-8
        if active:
            sensed=np.asarray(encoders,float)
            result[self.tendons]=np.maximum(result[self.tendons],sensed[self.tendons]-self.cap)
            result[self.tendons]=np.minimum(result[self.tendons],self.calibration.ranges[self.tendons,1])
            # Existing native trace records the command actually sent to the plant.
            previous=self.rows[-1];self.rows[-1]=(*previous[:15],*result,previous[-1])
        self.compliance_rows.append((float(now),float(active),*requested,*result))
        return result
