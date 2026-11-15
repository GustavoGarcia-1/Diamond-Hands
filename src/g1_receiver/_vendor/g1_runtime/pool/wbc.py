"""Numpy/ONNX adapter for the pinned NVIDIA decoupled G1 lower-body policies.

Observation layout mirrors g1_gear_wbc_policy.py at the recorded upstream commit:
86 values x 6 frames, including measured arm state. No privileged velocity/pose.
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
import numpy as np
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation
from .contracts import ActuatorCommand, RobotObservation, SkillGoal

ROOT = Path(__file__).resolve().parents[1]
BODY_NAMES = ([f'{s}_{j}_joint' for s in ['left','right'] for j in
               ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]
              + ['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']
              + [f'{s}_{j}_joint' for s in ['left','right'] for j in
                 ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']])

class WholeBodyController:
    def __init__(self, joint_names: list[str], joint_ranges: np.ndarray, *, asset_root: Path = ROOT):
        self.names = joint_names
        self.ranges = joint_ranges
        self.body_ids = np.array([joint_names.index(n) for n in BODY_NAMES])
        self.lower_ids = self.body_ids[:15]
        self.cfg = yaml.safe_load((asset_root / 'assets/upstream/wbc.yaml').read_text())
        motors = yaml.safe_load((asset_root / 'assets/upstream/motor.yaml').read_text())
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.balance = ort.InferenceSession(str(asset_root/'checkpoints/balance.onnx'), opts, providers=['CPUExecutionProvider'])
        self.walk = ort.InferenceSession(str(asset_root/'checkpoints/walk.onnx'), opts, providers=['CPUExecutionProvider'])
        for session in [self.balance, self.walk]:
            if session.get_inputs()[0].shape[-1] != self.cfg['num_obs']:
                raise ValueError('Checkpoint observation mismatch')
        self.kp = np.full(len(joint_names), 3.)
        self.kd = np.full(len(joint_names), .1)
        self.kp[self.body_ids] = motors['MOTOR_KP']
        self.kd[self.body_ids] = motors['MOTOR_KD']
        self.kp[self.lower_ids] = self.cfg['kps']
        self.kd[self.lower_ids] = self.cfg['kds']
        self.nominal = np.zeros(len(joint_names))
        self.nominal[self.lower_ids] = self.cfg['default_angles']
        for side, sign in [('left',1), ('right',-1)]:
            self.nominal[joint_names.index(side+'_shoulder_roll_joint')] = sign * .2
        self.reset()

    def reset(self):
        self.previous_action = np.zeros(15)
        self.history = deque(maxlen=self.cfg['obs_history_len'])
        self.calls = 0
        self.last_q = self.nominal.copy()

    def act(self, obs: RobotObservation, goal: SkillGoal, upper_target: np.ndarray | None = None):
        q = obs.joint_position[self.body_ids]
        dq = obs.joint_velocity[self.body_ids]
        defaults = np.pad(np.array(self.cfg['default_angles']), (0,14))
        qw = obs.imu_quat_wxyz
        rot = Rotation.from_quat(qw[[1,2,3,0]])
        gravity = rot.inv().apply([0,0,-1])
        frame = np.concatenate([
            np.asarray(goal.base_velocity) * self.cfg['cmd_scale'],
            [goal.base_height], goal.torso_rpy,
            obs.imu_gyro * self.cfg['ang_vel_scale'], gravity,
            (q-defaults)*self.cfg['dof_pos_scale'], dq*self.cfg['dof_vel_scale'],
            self.previous_action
        ]).astype(np.float32)
        if frame.shape != (86,):
            raise ValueError('Invalid complete body observation')
        self.history.append(frame)
        while len(self.history) < self.history.maxlen:
            self.history.appendleft(np.zeros(86, dtype=np.float32))
        network_input = np.concatenate(self.history)[None,:]
        session = self.balance if np.linalg.norm(goal.base_velocity) < .05 else self.walk
        self.previous_action = session.run(None, {session.get_inputs()[0].name: network_input})[0].reshape(15)
        target = self.nominal.copy() if upper_target is None else upper_target.copy()
        target[self.lower_ids] = self.previous_action*self.cfg['action_scale'] + self.cfg['default_angles']
        target = np.clip(target, self.ranges[:,0], self.ranges[:,1])
        self.calls += 1
        self.last_q = target
        command = ActuatorCommand(target, self.kp.copy(), self.kd.copy(), np.zeros(len(target)), 'balance' if session is self.balance else 'walk')
        command.validate(len(self.names))
        return command
