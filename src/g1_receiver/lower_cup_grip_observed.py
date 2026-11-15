"""Native grip activation from RGB-D flight or declared route preparation.

No passing-machine event, fixture release timestamp, contacts, or ball truth
enters this adapter. RGB-D history is admitted by the existing FieldPassGrip.
"""
import numpy as np
from .lower_cup_grip import NativeCupGrip

class ObservedNativeCupGrip(NativeCupGrip):
    def __init__(self,calibration,config):
        super().__init__(calibration,config)
        self.source=config.get('preload_source','rgbd_flight')
        if self.source not in ('rgbd_flight','visual_ready','route_ready'):
            raise ValueError('observed grip requires RGB-D or route-ready activation')
        self.activation_at=None;self.activation_rows=[]

    def observed_targets(self,now,visual_gate_at,visual_ready_at,route_ready_at,ball_history,encoders):
        speed=None
        if self.activation_at is None:
            if self.source=='route_ready':
                active=route_ready_at is not None and now>=route_ready_at-1e-8
            elif self.source=='visual_ready':
                active=visual_ready_at is not None and now>=visual_ready_at-1e-8
            else:
                samples=np.asarray([r[:4]for r in ball_history[-3:]],dtype=float)
                active=False
                if samples.shape==(3,4) and np.isfinite(samples).all():
                    times=samples[:,0];dt=np.diff(times)
                    if (dt>1e-8).all()and .039<=np.ptp(times)<=.121 and -1e-8<=now-times[-1]<=.08:
                        t=times-times.mean();velocity=t@samples[:,1:]/(t@t);speed=float(np.linalg.norm(velocity));segments=np.diff(samples[:,1:],axis=0)/dt[:,None];cos=float(segments[0]@segments[1]/max(1e-12,np.linalg.norm(segments[0])*np.linalg.norm(segments[1])))
                        active=bool(2.<speed<20. and cos>.9)
            if active:self.activation_at=float(now)
        self.activation_rows.append(dict(time_s=float(now),source=self.source,activation_at_s=self.activation_at,observed_flight_speed_mps=speed))
        # Before perception/route activation the hand stays open. The legacy
        # fixture-dependent scalar pregrip is deliberately not an input.
        return super().targets(now,0.,self.activation_at,visual_gate_at,encoders)
