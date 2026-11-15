"""Free-base simulated G1 with articulated Dextra or Aero hands."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ._vendor.g1_runtime.pool.wbc import BODY_NAMES, WholeBodyController
from .locomotion import (CONTROL_DT, SURVEY_CAMERA_FOV_Y_DEG, SURVEY_CAMERA_POSITION_M,
                         SURVEY_CAMERA_TARGET_M, survey_camera_rotation)
from .perception import CameraIntrinsics, RGBDFrame, RigidPose
from .runtime import DEFAULT_RUNTIME_ROOT
from .sim import HEIGHT, WIDTH


class HandRoutePlant:
    """Body PD and articulated hand control on a free-base model."""

    def __init__(self, codesign_source: Path, *, asset_root: Path = DEFAULT_RUNTIME_ROOT,
                 survey_camera: bool = True, route_camera_position=None, rear_markers=False,
                 hand_model: str = "dextra", scene_setup=None):
        if hand_model == "dextra":
            from .dextra import DextraActuation as HandActuation, make_dextra_scene as make_hand_scene
        elif hand_model == "aero":
            from .aero import AeroActuation as HandActuation, make_aero_scene as make_hand_scene
        else:
            raise ValueError(f"unknown hand model {hand_model!r}")
        self.hand_model = hand_model
        xml = make_hand_scene(asset_root, codesign_source, survey_camera=survey_camera)
        scene = ET.fromstring(xml)
        self.cameras = {"survey_rgbd": (survey_camera_rotation(), np.asarray(SURVEY_CAMERA_POSITION_M))}
        if rear_markers:
            pelvis = scene.find(".//body[@name='pelvis']")
            for color, lateral, rgba in (
                ("blue", .11, "0 .10 1 1"), ("yellow", -.11, "1 1 0 1")
            ):
                ET.SubElement(pelvis, "geom", name=f"rear_marker_{color}", type="sphere", size=".035",
                              pos=f"-.15 {lateral} .08", rgba=rgba, contype="0", conaffinity="0",
                              mass="0")
        if route_camera_position is not None:
            position = np.asarray(route_camera_position, dtype=float)
            forward = np.asarray(SURVEY_CAMERA_TARGET_M) - position
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, (0., 0., 1.))
            right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            self.cameras["route_rgbd"] = (np.column_stack((right, down, forward)), position)
            ET.SubElement(scene.find("worldbody"), "camera", name="route_rgbd",
                          pos=" ".join(map(str, position)), xyaxes=" ".join(map(str, (*right, *-down))),
                          fovy=str(SURVEY_CAMERA_FOV_Y_DEG))
        if scene_setup is not None:
            scene_setup(scene)
        sensors = scene.find("sensor")
        for index, name in enumerate(BODY_NAMES):
            ET.SubElement(sensors, "jointpos", name=f"hand_trial_encoder_q_{index}", joint=name)
            ET.SubElement(sensors, "jointvel", name=f"hand_trial_encoder_dq_{index}", joint=name)
        self.scene_xml = ET.tostring(scene, encoding="unicode")
        self.model = mujoco.MjModel.from_xml_string(self.scene_xml)
        self.data = mujoco.MjData(self.model)
        self.actuation = HandActuation.from_model(self.model)
        self.joint_names = list(BODY_NAMES)
        actuator_ids = [self.model.actuator(name).id for name in BODY_NAMES]
        joint_ids = self.model.actuator_trnid[actuator_ids, 0]
        self.body_actuators = np.asarray(actuator_ids)
        self.joint_ranges = self.model.jnt_range[joint_ids].copy()
        self.joint_qpos = self.model.jnt_qposadr[joint_ids]
        self.wbc = WholeBodyController(self.joint_names, self.joint_ranges, asset_root=asset_root)
        self.nominal = self.wbc.nominal.copy()
        self._q = self._encoder_addresses("q")
        self._dq = self._encoder_addresses("dq")
        self._quat = self._sensor_slice("receiver_orientation")
        self._gyro = self._sensor_slice("receiver_gyro")
        self._effort_min, self._effort_max = self.model.actuator_ctrlrange[self.body_actuators].T
        self.base = self.model.joint("floating_base_joint").qposadr[0]
        ball = self.model.joint("football_free")
        self.ball_qpos = ball.qposadr[0]
        self.ball_dof = ball.dofadr[0]
        self.substeps = round(CONTROL_DT / self.model.opt.timestep)
        if not np.isclose(self.substeps * self.model.opt.timestep, CONTROL_DT):
            raise ValueError("model physics step must divide the body control period")
        self.renderer = None
        self.survey_camera = survey_camera
        focal = HEIGHT / (2 * np.tan(np.deg2rad(SURVEY_CAMERA_FOV_Y_DEG / 2)))
        self.intrinsics = CameraIntrinsics(focal, focal, (WIDTH - 1) / 2, (HEIGHT - 1) / 2)

    def _encoder_addresses(self, quantity):
        return np.array([self.model.sensor(f"hand_trial_encoder_{quantity}_{i}").adr[0] for i in range(29)])

    def _sensor_slice(self, name):
        sensor = self.model.sensor(name)
        return slice(sensor.adr[0], sensor.adr[0] + sensor.dim[0])

    @property
    def timestamp_s(self):
        return float(self.data.time)

    def reset(self, *, release_position_m=(8., 4., .085), release_velocity_mps=(0., 0., 0.),
              joint_positions=None, closure=0.):
        positions = self.nominal if joint_positions is None else np.asarray(joint_positions, dtype=float)
        if (positions.shape != (29,) or not np.isfinite(positions).all()
                or (positions < self.joint_ranges[:, 0]).any() or (positions > self.joint_ranges[:, 1]).any()):
            raise ValueError("invalid body reset positions")
        mujoco.mj_resetData(self.model, self.data)
        self.wbc.reset()
        self.data.qpos[self.joint_qpos] = positions
        self.data.qpos[self.base:self.base + 7] = (0., 0., .79, 1., 0., 0., 0.)
        self.data.qpos[self.ball_qpos:self.ball_qpos + 3] = release_position_m
        self.data.qvel[self.ball_dof:self.ball_dof + 3] = release_velocity_mps
        self.actuation.set_closure(self.data, closure)
        mujoco.mj_forward(self.model, self.data)

    def observe(self):
        sensors = self.data.sensordata
        return tuple(sensors[index].copy() for index in (self._q, self._dq, self._quat, self._gyro))

    def capture(self, camera="survey_rgbd"):
        if not self.survey_camera:
            raise RuntimeError("this model was built without the survey camera")
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=HEIGHT, width=WIDTH)
        rotation, position = self.cameras[camera]
        self.renderer.update_scene(self.data, camera=camera)
        rgb = self.renderer.render()
        self.renderer.enable_depth_rendering()
        try:
            depth = self.renderer.render()
        finally:
            self.renderer.disable_depth_rendering()
        now = self.timestamp_s
        return (RGBDFrame(now, "camera_optical", rgb, depth, self.intrinsics),
                RigidPose(now, "camera_optical", "field", rotation, position,
                          "synthetic_fixture"))

    def step(self, command, *, closure=0., hand_target_rad=None, post_step=None):
        command.validate(29)
        if ((command.target_position < self.joint_ranges[:, 0]).any()
                or (command.target_position > self.joint_ranges[:, 1]).any()):
            raise ValueError("body command exceeds a joint range")
        if hand_target_rad is None:
            self.actuation.set_closure(self.data, closure, dt_s=CONTROL_DT)
        else:
            self.actuation.set_targets(self.data, hand_target_rad, dt_s=CONTROL_DT)
        for _ in range(self.substeps):
            sensed = self.data.sensordata
            effort = (command.kp * (command.target_position - sensed[self._q])
                      - command.kd * sensed[self._dq] + command.feedforward_torque)
            self.data.ctrl[self.body_actuators] = np.clip(effort, self._effort_min, self._effort_max)
            mujoco.mj_step(self.model, self.data)
            if post_step is not None:
                post_step(self.data)
        mujoco.mj_forward(self.model, self.data)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
