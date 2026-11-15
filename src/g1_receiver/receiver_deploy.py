"""Receiver inference from encoder, IMU and surveyed RGB-D measurements."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ._vendor.g1_runtime.pool.wbc import BODY_NAMES, WholeBodyController
from .localization import MarkerLocalizer
from .observable import ACTION_LIMITS, ACTION_SIZE, OBS_SIZE, ObservableGenCoord, bounded_command
from .perception import ColorBallDetector, RGBDFrame, RigidPose
from .routes import Route, RouteFollower
from .runtime import DEFAULT_RUNTIME_ROOT, RobotObservation, SkillGoal

CONTROL_DT = .02
HISTORY_LENGTH = 4
POLICY_TICKS = 2
BODY_SLEW_PER_S = np.array([1.1, 1., 1.6])
ARM_SLEW_RAD_S = 3.
ACTOR_SHAPES = ((128, HISTORY_LENGTH * OBS_SIZE), (128, 128), (ACTION_SIZE, 128))


def robot_limits(asset_root: Path) -> np.ndarray:
    world = ET.parse(asset_root / "assets/robot/g1_body.xml").getroot().find("worldbody")
    joints = {joint.attrib["name"]: joint for joint in world.iter("joint") if "name" in joint.attrib}
    return np.array([list(map(float, joints[name].attrib["range"].split())) for name in BODY_NAMES])


class NumpyReceiverActor:
    """Run the exported deterministic SB3 actor without training dependencies."""

    def __init__(self, weights: Path):
        with np.load(weights, allow_pickle=False) as data:
            if data["schema"].item() != "receiver-ppo-tanh-actor-v1":
                raise ValueError("unsupported receiver actor schema")
            self.layers = []
            for index, shape in enumerate(ACTOR_SHAPES):
                weight, bias = data[f"weight_{index}"], data[f"bias_{index}"]
                if (weight.shape != shape or bias.shape != (shape[0],)
                        or weight.dtype != np.float32 or bias.dtype != np.float32
                        or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
                    raise ValueError(f"invalid receiver actor layer {index}")
                self.layers.append((weight, bias))

    def __call__(self, observation):
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (HISTORY_LENGTH * OBS_SIZE,) or not np.isfinite(values).all():
            raise ValueError("invalid actor observation")
        for index, (weight, bias) in enumerate(self.layers):
            values = weight @ values + bias
            if index < len(self.layers) - 1:
                values = np.tanh(values)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite actor action")
        return np.clip(values, -1., 1.)


class ReceiverController:
    """50 Hz WBC targets from a 25 Hz actor and surveyed-camera route follower.

    This class does not connect to a robot or a motor transport. The caller
    supplies encoder/IMU samples and RGB-D frames in a shared clock domain.
    """

    def __init__(self, actor: NumpyReceiverActor, route: Route, *,
                 asset_root: Path = DEFAULT_RUNTIME_ROOT, allow_synthetic_fixture: bool = False):
        self.actor = actor
        self.route = route
        self.wbc = WholeBodyController(BODY_NAMES, robot_limits(asset_root), asset_root=asset_root)
        self.observable = ObservableGenCoord(allow_synthetic_fixture=allow_synthetic_fixture)
        self.localizer = MarkerLocalizer(allow_synthetic_fixture=allow_synthetic_fixture)
        self.detector = ColorBallDetector(ball_center_offset_m=.075)
        self.reset()

    def reset(self):
        self.wbc.reset()
        self.observable.reset()
        self.follower = RouteFollower(self.route)
        self.goal = np.array(self.route.waypoints[1])
        self.field_command = np.zeros(2)
        self.drive = np.zeros(3)
        self.drive_target = np.zeros(3)
        self.upper = self.wbc.nominal.copy()
        self.upper_target = self.upper.copy()
        self.previous_action = np.zeros(ACTION_SIZE)
        self.history = deque((np.zeros(OBS_SIZE, dtype=np.float32) for _ in range(HISTORY_LENGTH - 1)),
                             maxlen=HISTORY_LENGTH)
        self.last_time = -np.inf
        self.last_localized_time = -np.inf
        self.ticks = 0

    def ingest_rgbd(self, frame: RGBDFrame, survey: RigidPose, *, now_s: float) -> bool:
        if not np.isfinite(now_s) or not 0 <= now_s - frame.timestamp_s <= self.observable.max_age_s:
            return False
        if frame.timestamp_s <= self.observable.capture_s:
            return False
        receiver = self.localizer.estimate(frame, survey)
        ball = self.detector.detect(frame, survey)
        self.observable.update_camera(frame.timestamp_s, receiver, ball)
        if receiver is not None:
            self.last_localized_time = frame.timestamp_s
            self.field_command = np.array(self.follower.command(receiver.xy_m))
            self.goal = np.array(self.route.waypoints[min(self.follower.next_waypoint,
                                                          len(self.route.waypoints) - 1)])
        return True

    def tick(self, now_s: float, proprioception):
        if not np.isfinite(now_s) or now_s <= self.last_time:
            raise ValueError("proprioception time must increase")
        if self.ticks and now_s - self.last_time > .060:
            raise ValueError("proprioception gap exceeds 60 ms; reset under supervision")
        q, dq, quat, gyro = (np.asarray(item, dtype=float) for item in proprioception)
        state = self.observable.observe(now_s, (q, dq, quat, gyro), self.goal,
                                        self.previous_action, self.wbc.last_q)
        if self.ticks % POLICY_TICKS == 0:
            self.history.append(state.vector)
            action = self.actor(np.concatenate(self.history))
            if state.receiver is not None:
                c, s = np.cos(state.receiver.yaw_rad), np.sin(state.receiver.yaw_rad)
                drive_xy = np.array([[c, s], [-s, c]]) @ self.field_command
                action[:3] = np.clip(np.r_[drive_xy, 0.] / ACTION_LIMITS[:3], -1., 1.)
                self.drive_target, self.upper_target = bounded_command(
                    action, self.wbc.nominal, self.wbc.names, self.wbc.ranges)
                self.previous_action = action
        if state.receiver is None:
            self.drive_target.fill(0.)
            self.previous_action[:3] = 0.
            # Finish an already issued, rate-limited arm target through a
            # brief marker miss. Stop arm motion if localization stays lost.
            if now_s - self.last_localized_time > self.observable.max_age_s:
                self.upper_target = self.upper.copy()
        self.drive += np.clip(self.drive_target - self.drive, -BODY_SLEW_PER_S * CONTROL_DT,
                              BODY_SLEW_PER_S * CONTROL_DT)
        limit = ARM_SLEW_RAD_S * CONTROL_DT
        self.upper += np.clip(self.upper_target - self.upper, -limit, limit)
        observation = RobotObservation(now_s, q, dq, quat, gyro, np.zeros(len(q)), None)
        command = self.wbc.act(observation, SkillGoal(base_velocity=self.drive), upper_target=self.upper)
        self.ticks += 1
        self.last_time = now_s
        return command
