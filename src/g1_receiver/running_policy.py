"""Opt-in adapter for the public legged_rl_lab G1 AMP running export."""

from __future__ import annotations

from collections import deque
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from .observable import ARM_JOINTS, project_gravity
from ._vendor.g1_runtime.pool.contracts import ActuatorCommand, RobotObservation, SkillGoal

UPSTREAM_REVISION = "386ee24bc62c8033641d65b940dc2cf1dec56545"
CHECKPOINT_SHA256 = "17019141d56e57f53059c671c81e22d63c1dd2f7cc69a8b322c06790c00cca6f"
CONFIG_SHA256 = "3d45b9641f460eed486e2caa92feaedbfbf62d41b2434a2d0b38bb829a5319c8"
CONTROL_PERIOD_S = .02
GROUP_SLICES = (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 38), slice(38, 67), slice(67, 96))


class AmpRunningPolicy:
    """29-axis 50 Hz velocity policy with optional bounded arm-target replacement."""

    def __init__(self, joint_names: list[str], joint_ranges: np.ndarray, upstream_checkout: Path):
        root = upstream_checkout / "deploy/g1_deploy"
        config_path = root / "config/g1_amp.yaml"
        checkpoint_path = root / "exported_policy/g1_run.onnx"
        config_bytes = config_path.read_bytes()
        checkpoint_bytes = checkpoint_path.read_bytes()
        if sha256(config_bytes).hexdigest() != CONFIG_SHA256 or sha256(checkpoint_bytes).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("AMP running weights or deployment config differ from the pinned export")
        config = yaml.safe_load(config_bytes)
        canonical = config["joint_names_mujoco"]
        if len(joint_names) != 29 or set(canonical) != set(joint_names):
            raise ValueError("running policy requires calibrated G1 29-joint encoder and actuator names")
        self.joint_ranges = np.asarray(joint_ranges, dtype=float)
        if (self.joint_ranges.shape != (29, 2) or not np.isfinite(self.joint_ranges).all()
                or np.any(self.joint_ranges[:, 0] >= self.joint_ranges[:, 1])):
            raise ValueError("invalid G1 joint limits")
        policy_names = [canonical[index] for index in config["mujoco_to_isaac_map"]]
        self.policy_indices = np.array([joint_names.index(name) for name in policy_names])
        self.arm_indices = np.array([joint_names.index(name) for name in ARM_JOINTS])
        self.default = np.asarray(config["default_joint_pos"], dtype=np.float32)
        self.action_scale = np.asarray(config["action_scale"], dtype=np.float32)
        self.kp, self.kd, self.nominal = (self._body_order(config[key]) for key in ("kps", "kds", "default_joint_pos"))
        self.velocity_scale = float(config["dof_vel_scale"])
        self.position_scale = float(config["dof_pos_scale"])
        self.gyro_scale = float(config["ang_vel_scale"])
        self.command_ranges = np.array([config["command_range"][key] for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(checkpoint_bytes, options, providers=["CPUExecutionProvider"])
        (model_input,), (model_output,) = self.session.get_inputs(), self.session.get_outputs()
        if model_input.shape[-1] != 384 or model_output.shape[-1] != 29 or config["control_dt"] != CONTROL_PERIOD_S:
            raise ValueError("unexpected running observation, action or timing contract")
        self.input_name = model_input.name
        self.reset()

    def _body_order(self, values) -> np.ndarray:
        mapped = np.empty(29)
        mapped[self.policy_indices] = values
        return mapped

    def reset(self) -> None:
        self.previous_action = np.zeros(29, dtype=np.float32)
        self.history = deque([np.zeros(96, dtype=np.float32) for _ in range(4)], maxlen=4)

    def act(self, obs: RobotObservation, goal: SkillGoal, upper_target: np.ndarray | None = None,
            *, arm_override_fraction: float = 1.) -> ActuatorCommand:
        q = np.asarray(obs.joint_position, dtype=float)
        dq = np.asarray(obs.joint_velocity, dtype=float)
        gyro = np.asarray(obs.imu_gyro, dtype=float)
        drive = np.asarray(goal.base_velocity, dtype=float)
        if any(value.shape != (size,) or not np.isfinite(value).all() for value, size in ((q, 29), (dq, 29), (gyro, 3), (drive, 3))):
            raise ValueError("running observation requires finite G1 encoder, gyro and velocity-command values")
        if not 0 <= arm_override_fraction <= 1:
            raise ValueError("arm override fraction must be from zero to one")
        gravity = project_gravity(obs.imu_quat_wxyz)
        drive = np.clip(drive, self.command_ranges[:, 0], self.command_ranges[:, 1])
        frame = np.r_[gyro * self.gyro_scale, gravity, drive,
                      (q[self.policy_indices] - self.default) * self.position_scale,
                      dq[self.policy_indices] * self.velocity_scale, self.previous_action].astype(np.float32)
        self.history.append(frame)
        frames = np.stack(self.history)
        policy_input = np.concatenate([frames[:, group].ravel() for group in GROUP_SLICES])[None, :]
        (output,) = self.session.run(None, {self.input_name: policy_input})
        action = output[0]
        if action.shape != (29,) or not np.isfinite(action).all():
            raise RuntimeError("running policy returned invalid residual actions")
        self.previous_action = action.copy()
        target = self._body_order(self.default + self.action_scale * action)
        if upper_target is not None:
            override = np.asarray(upper_target, dtype=float)
            if override.shape != (29,) or not np.isfinite(override).all():
                raise ValueError("arm override requires a finite 29-axis target")
            target[self.arm_indices] += arm_override_fraction * (override[self.arm_indices] - target[self.arm_indices])
        command = ActuatorCommand(np.clip(target, *self.joint_ranges.T), self.kp, self.kd, np.zeros(29), "amp_running")
        command.validate(29)
        return command
