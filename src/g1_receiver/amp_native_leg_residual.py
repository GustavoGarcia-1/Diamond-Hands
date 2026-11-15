"""Deploy the sensor-only 93D leg residual beside the frozen AMP controller."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import numpy as np
from .amp_native_cup_training import MarkerSensors,leg_observation,SCHEMA,bounded_leg_action,model_residual_config
from .observable import ARM_JOINTS

class NativeCupLegResidual:
    def __init__(self,checkpoint,profile,joint_names,joint_ranges,device='cuda'):
        from stable_baselines3 import PPO
        self.path=Path(checkpoint);self.digest=sha256(self.path.read_bytes()).hexdigest()
        self.policy=PPO.load(self.path,device=device)
        if getattr(self.policy,'receiver_sensor_schema',None)!=SCHEMA:
            raise ValueError('native leg residual sensor schema mismatch')
        if not np.array_equal(getattr(self.policy,'receiver_arm_offsets',None),profile['out_staged_arm_offsets_rad']):
            raise ValueError('leg residual was trained with different staged arm targets')
        if getattr(self.policy,'receiver_native_grip',None)!=profile['native_grip']:
            raise ValueError('leg residual was trained with a different native grip schedule')
        self.residual_config=model_residual_config(self.policy)
        self.lower=np.array([i for i,n in enumerate(joint_names)if n not in ARM_JOINTS])
        self.ranges=np.asarray(joint_ranges);self.sensors=MarkerSensors();self.offset=np.zeros(15)
        self.rows=[];self.observations=[];self.device=device
    def apply(self,command,obs,pose_measurement,drive,stage):
        self.sensors.update(float(obs.timestamp),pose_measurement)
        vector=leg_observation(obs,self.sensors,self.offset,drive,stage)
        action=np.asarray(self.policy.predict(vector,deterministic=True)[0],float)
        if action.shape!=(15,)or not np.isfinite(action).all():raise ValueError('invalid learned leg action')
        self.offset=bounded_leg_action(action,self.offset,float(obs.timestamp),**self.residual_config)
        self.observations.append(vector.copy());self.rows.append((float(obs.timestamp),*action,*self.offset))
        positions=command.target_position.copy()
        positions[self.lower]=np.clip(positions[self.lower]+self.offset,self.ranges[self.lower,0],self.ranges[self.lower,1])
        return replace(command,target_position=positions)
