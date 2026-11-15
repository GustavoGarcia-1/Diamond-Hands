"""Explicit bounded right-cup carry waypoint triggered by causal RGB-D forecast."""
import numpy as np
from .late_cup_relative_servo import WorldCupAnchor as RelativeCup

class WorldCupAnchor(RelativeCup):
    def __init__(self,config):
        super().__init__(config);self.cradle=np.asarray(config.get('cradle_right_delta_rad',[0.]*7),float);self.cradle_delay=float(config.get('cradle_delay_s',.45));self.cradle_duration=float(config.get('cradle_duration_s',.50));self.cradle_gain=float(config.get('cradle_gain',0.))
        if self.cradle.shape!=(7,)or not np.isfinite(self.cradle).all()or np.max(abs(self.cradle))>.30 or not 0<=self.cradle_gain<=1 or self.cradle_delay<.25 or self.cradle_duration<.50:raise ValueError('bounded delayed cradle waypoint required')
        if np.max(abs(self.cradle))*1.875/self.cradle_duration>1.5 or np.max(abs(self.cradle))*5.774/self.cradle_duration**2>30:raise ValueError('cradle slew bounds exceeded')
    def apply(self,obs,pose,target,phase,prediction):
        result=super().apply(obs,pose,target,phase,prediction);row=self.rows[-1]
        if not self.cradle_gain or self.freeze_at is None:return result
        cue=self.freeze_at-.02+self.cradle_delay;u=float(np.clip((obs.timestamp-cue)/self.cradle_duration,0,1));fraction=u*u*u*(10-15*u+6*u*u);delta=self.cradle_gain*fraction*self.cradle;result[self.arms[7:]]=np.clip(result[self.arms[7:]]+delta,self.ranges[7:,0]+.03,self.ranges[7:,1]-.03);row.update(cradle_forecast_based_start_s=cue,cradle_fraction=fraction,cradle_right_delta_rad=delta.tolist(),cradle_target_right_rad=result[self.arms[7:]].tolist(),cradle_scope='newexplicitpostentryorientationwaypoint; originalmotorcapsunchanged');return result
