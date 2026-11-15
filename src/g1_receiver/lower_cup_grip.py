"""Opt-in native Aero digit commands using visual events and actuator encoders.

No contact, ball truth, full finger-joint state, or force input is accepted.
Native fractions span each actuator's declared control range. Official plant
rate limits remain authoritative. Encoders are logged/validated, not treated as
contact sensors or as a complete measurement of underactuated finger posture.
"""
from dataclasses import dataclass
import numpy as np

OPEN = np.array([.110387]*4+[0.,.038389,.112138])
LEGACY_CLOSE = np.array([.080]*4+[1.30,.032,.092])
SENSOR_ORDER = ('len_if','len_mf','len_rf','len_pf','len_th_abd','len_th1','len_th2')

def legacy_targets(closure):
    closure = np.asarray(closure, dtype=float)
    if closure.shape not in ((),(2,)) or not np.isfinite(closure).all():
        raise ValueError('finite scalar or bilateral legacy closure required')
    return (OPEN + np.broadcast_to(np.clip(closure,0.,1.),(2,))[:,None]*(LEGACY_CLOSE-OPEN)).ravel()

@dataclass(frozen=True)
class NativeHandCalibration:
    ranges: np.ndarray
    sensor_addresses: np.ndarray

    @classmethod
    def from_model(cls, model):
        ids=[i for s in ('left','right') for i in range(model.nu)
             if model.actuator(i).name.startswith('aero_'+s+'_')]
        addresses=[int(model.sensor('aero_'+s+'_'+n).adr[0])
                   for s in ('left','right') for n in SENSOR_ORDER]
        if len(ids)!=14 or len(set(addresses))!=14:
            raise ValueError('seven native actuators and encoders per hand required')
        ranges=model.actuator_ctrlrange[ids].copy()
        for offset in (4,11):
            joint=int(model.actuator_trnid[ids[offset],0])
            ranges[offset,0]=max(ranges[offset,0],float(model.jnt_range[joint,0]))
            ranges[offset,1]=min(ranges[offset,1],float(model.jnt_range[joint,1]))
        return cls(ranges,np.asarray(addresses))

    def targets(self, fractions):
        f=np.asarray(fractions,dtype=float)
        if f.shape==(7,):f=np.tile(f,2)
        if f.shape!=(14,) or not np.isfinite(f).all() or np.any((f<0.)|(f>1.)):
            raise ValueError('seven or fourteen finite normalized fractions in [0,1] required')
        opened=np.tile(OPEN,2);closed=self.ranges[:,0].copy()
        closed[[4,11]]=self.ranges[[4,11],1]
        return opened+f*(closed-opened)

    def fractions(self, targets):
        opened=np.tile(OPEN,2);closed=self.ranges[:,0].copy();closed[[4,11]]=self.ranges[[4,11],1]
        return (np.asarray(targets)-opened)/(closed-opened)

class NativeCupGrip:
    def __init__(self,calibration,config):
        self.calibration=calibration;self.mode=config.get('mode','staged')
        if self.mode not in ('legacy','staged'):raise ValueError('unknown native grip mode')
        self.preload=np.asarray(config.get('preload',[0.]*7),dtype=float)
        self.capture=np.asarray(config.get('capture',[1.]*7),dtype=float)
        if self.preload.shape==(7,):self.preload=np.tile(self.preload,2)
        if self.capture.shape==(7,):self.capture=np.tile(self.capture,2)
        calibration.targets(self.preload);calibration.targets(self.capture)
        delays=np.asarray(config.get('digit_delay_s',0.),dtype=float)
        if delays.shape==(7,):delays=np.tile(delays,2)
        self.delays=np.broadcast_to(delays,(14,)).copy()
        self.ramp=float(config.get('ramp_s',.2))
        if not np.isfinite(self.delays).all() or np.any(self.delays<0.) or not np.isfinite(self.ramp) or self.ramp<=0.:
            raise ValueError('nonnegative digit delays and positive ramp required')
        self.rows=[]

    def targets(self,now,legacy_closure,release_at,visual_gate_at,encoders):
        sensed=np.asarray(encoders,dtype=float)
        if sensed.shape!=(14,) or not np.isfinite(sensed).all():
            raise ValueError('fourteen finite native actuator encoder measurements required')
        phase=0
        if self.mode=='legacy' or release_at is None or now<release_at:
            target=legacy_targets(legacy_closure)
        else:
            f=self.preload.copy();phase=1
            if visual_gate_at is not None:
                u=np.clip((now-visual_gate_at-self.delays)/self.ramp,0.,1.)
                f=self.preload+u*(self.capture-self.preload);phase=2 if np.any(u<1.) else 3
            target=self.calibration.targets(f)
        self.rows.append((float(now),*sensed,*target,float(phase)))
        return target
