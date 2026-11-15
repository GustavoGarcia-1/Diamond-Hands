"""Compute compact-hand reach targets using encoder and IMU measurements."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .aero import PALM_BODIES
from .observable import ARM_JOINTS


class AeroReach:
    """Damped palm-pose inverse kinematics on a separate kinematic model state."""

    def __init__(self, model, joint_names, ready):
        self.model = model
        self.work = mujoco.MjData(model)
        self.pelvis = int(model.joint("floating_base_joint").qposadr[0])
        joints = [model.joint(name).id for name in joint_names]
        self.positions = model.jnt_qposadr[joints]
        self.dofs = model.jnt_dofadr[joints]
        self.arm_indices = np.array([joint_names.index(name) for name in ARM_JOINTS]).reshape(2, 7)
        self.waist = [joint_names.index(f"waist_{name}_joint") for name in ("yaw", "roll", "pitch")]
        self.palms = [model.body(name).id for name in PALM_BODIES]
        self.ranges = model.jnt_range[joints]
        self.ready = np.array(ready, dtype=float, copy=True)
        self.work.qpos[self.pelvis:self.pelvis + 7] = [0., 0., 0., 1., 0., 0., 0.]
        self.work.qpos[self.positions] = self.ready
        mujoco.mj_fwdPosition(model, self.work)
        self.default_xyz = self.work.xpos[self.palms].copy()
        self.default_rotation = self.work.xmat[self.palms].reshape(2, 3, 3).copy()
        self.jac_position = np.empty((3, model.nv))
        self.jac_orientation = np.empty_like(self.jac_position)
        self.previous = self.ready.copy()

    def target(self, observation, ready, *, blend, xyz_offset=(0., 0., 0.), reference_from_ready=False):
        if blend <= 0:
            self.previous[:] = ready
            return ready
        w, x, y, z = observation.imu_quat_wxyz
        yaw = np.arctan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z))
        neutral_yaw = Rotation.from_rotvec([0., 0., -yaw])
        orientation = neutral_yaw * Rotation.from_quat([x, y, z, w])
        reference_xyz = self.default_xyz
        reference_rotation = self.default_rotation
        if reference_from_ready:
            # Stabilize the requested grasp itself, rather than the open ready
            # pose. This kinematic state uses only commanded joints, IMU and
            # encoders; it never reads the simulation's robot or ball state.
            self.work.qpos[self.pelvis:self.pelvis + 7] = [0., 0., 0., 1., 0., 0., 0.]
            self.work.qpos[self.positions] = ready
            mujoco.mj_fwdPosition(self.model, self.work)
            reference_xyz = self.work.xpos[self.palms].copy()
            reference_rotation = self.work.xmat[self.palms].reshape(2, 3, 3).copy()
        self.work.qpos[self.pelvis:self.pelvis + 7] = [0., 0., 0., *orientation.as_quat(scalar_first=True)]
        self.work.qpos[self.positions] = ready
        self.work.qpos[self.positions[self.waist]] = observation.joint_position[self.waist]

        target = ready.copy()
        requested_xyz = reference_xyz + np.asarray(xyz_offset, dtype=float)
        for _ in range(3):
            mujoco.mj_fwdPosition(self.model, self.work)
            for side, palm in enumerate(self.palms):
                indices = self.arm_indices[side]
                dofs = self.dofs[indices]
                mujoco.mj_jacBody(self.model, self.work, self.jac_position, self.jac_orientation, palm)
                rotation = self.work.xmat[palm].reshape(3, 3)
                rotation_error = Rotation.from_matrix(reference_rotation[side] @ rotation.T).as_rotvec()
                error = np.r_[requested_xyz[side] - self.work.xpos[palm], .13 * rotation_error]
                jacobian = np.vstack((self.jac_position[:, dofs], .13 * self.jac_orientation[:, dofs]))
                delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .0012 * np.eye(6), error)
                target[indices] = np.clip(target[indices] + np.clip(delta, -.12, .12),
                                          self.ranges[indices, 0], self.ranges[indices, 1])
            self.work.qpos[self.positions[self.arm_indices]] = target[self.arm_indices]
        command = ready + blend * (target - ready)
        limit = .075
        self.previous += np.clip(command - self.previous, -limit, limit)
        return self.previous.copy()
