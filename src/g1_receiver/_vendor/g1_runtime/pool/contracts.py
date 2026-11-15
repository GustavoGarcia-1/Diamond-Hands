"""Observation/action contract. Base pose must come from a declared sensor estimator."""
from dataclasses import dataclass,field
import numpy as np
@dataclass(frozen=True)
class RobotObservation:
    timestamp: float
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    imu_quat_wxyz: np.ndarray
    imu_gyro: np.ndarray
    motor_torque: np.ndarray
    base_pose_wxyz: np.ndarray | None
    ball_position_world: np.ndarray | None = None
    ball_velocity_world: np.ndarray | None = None
    ball_timestamp: float | None = None
@dataclass
class SkillGoal:
    base_velocity: np.ndarray=field(default_factory=lambda:np.zeros(3))
    base_height: float=.74
    torso_rpy: np.ndarray=field(default_factory=lambda:np.zeros(3))
@dataclass(frozen=True)
class ActuatorCommand:
    target_position: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    feedforward_torque: np.ndarray
    status: str
    def validate(self,n):
        for v in [self.target_position,self.kp,self.kd,self.feedforward_torque]:
            if v.shape!=(n,) or not np.isfinite(v).all():raise ValueError('Invalid actuator command')
        if (self.kp<0).any() or (self.kd<0).any():raise ValueError('Negative gains')
