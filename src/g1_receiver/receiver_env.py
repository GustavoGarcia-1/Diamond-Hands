"""Multi-step sensor-only receiver control over the frozen G1 WBC."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from .catch_reward import VisualRetentionProxy
from .catching import VisualCatchGate
from .locomotion import (
    CATCH_BALL_CENTER_OFFSET_M, CONTROL_DT, CRADLE_TARGET_RATE_RAD_S, cradle_joint_targets,
)
from .observable import (
    ACTION_LIMITS, ACTION_SIZE, ARM_JOINTS, OBS_SIZE, OBS_SLICES, ObservableGenCoord,
    bounded_command,
)
from .perception import ColorBallDetector
from .routes import RouteFollower, make_route
from .runtime import DEFAULT_RUNTIME_ROOT
from .training import SensorPlant

POLICY_HZ = 25
POLICY_TICKS = round(1 / (CONTROL_DT * POLICY_HZ))
HISTORY_LENGTH = 4
EPISODE_S = 3.4
BODY_SLEW_PER_S = np.array([1.1, 1.0, 1.6])
FAMILIES = ("centered", "offset", "moving")
OBSERVATION_SCHEMA = "receiver-multistep-observable-history-v1"
LOBS = {
    "centered": ((.84, 0., .38), (-1., 0., 4.36)),
    "offset": ((.92, -.02, .38), (-1.08, .035, 4.42)),
    "moving": ((1.45, 0., .40), (-1.62, 0., 4.44)),
}


@dataclass(frozen=True)
class ReceiverTrial:
    family: str
    release_position_m: tuple[float, float, float]
    release_velocity_mps: tuple[float, float, float]
    route: str | None = None
    route_scale: float = .24
    floor_friction_scale: float = 1.
    depth_noise_m: float = 0.
    dropout_probability: float = 0.
    camera_delay_ticks: int = 0
    suite: str = "training"


def sample_trial(rng: np.random.Generator, family: str, *, nominal: bool = False) -> ReceiverTrial:
    position, velocity = LOBS[family]
    route = "go" if family == "moving" else None
    if nominal:
        return ReceiverTrial(family, position, velocity, route)
    position = np.asarray(position) + rng.uniform(-1., 1., 3) * [.035, .045, .017]
    velocity = np.asarray(velocity) + rng.uniform(-1., 1., 3) * [.055, .060, .060]
    return ReceiverTrial(
        family, tuple(position), tuple(velocity), route,
        route_scale=float(rng.uniform(.20, .28)),
        floor_friction_scale=float(rng.uniform(.86, 1.14)),
        depth_noise_m=float(rng.uniform(0., .003)),
        dropout_probability=float(rng.uniform(0., .005)),
        camera_delay_ticks=int(rng.integers(0, 3)),
    )


def heldout_trials() -> tuple[ReceiverTrial, ...]:
    """Frozen independent paired benchmark, shared across all learner seeds."""
    rng = np.random.default_rng(100_173)
    trials = [
        replace(sample_trial(rng, family, nominal=True), suite="nominal_regression")
        for family in FAMILIES
    ]
    trials += [
        replace(
            sample_trial(rng, family),
            floor_friction_scale=1., depth_noise_m=0., dropout_probability=0., camera_delay_ticks=0,
            suite="throw_variation",
        )
        for family in FAMILIES
    ]
    trials += [
        replace(
            sample_trial(rng, family), floor_friction_scale=friction, depth_noise_m=.003,
            dropout_probability=.01, camera_delay_ticks=2, suite="sensor_and_friction_robustness",
        )
        for family, friction in zip(FAMILIES, (.86, 1.14, .86))
    ]
    return tuple(trials)


class EncoderReach:
    """Kinematics on a separate model state populated only from encoder samples."""

    def __init__(self, plant: SensorPlant):
        self.mujoco = plant.mujoco
        self.model = plant.model
        self.data = self.mujoco.MjData(self.model)
        self.encoder_addresses = plant.joint_qpos
        self.base_address = plant.base_qpos
        self.wrist_ids = [self.model.body(f"{side}_wrist_yaw_link").id for side in ("left", "right")]

    def wrist_midpoint(self, measured_angles: np.ndarray, imu_quat_wxyz: np.ndarray) -> np.ndarray:
        self.data.qpos[self.base_address:self.base_address + 7] = (0., 0., 0., 1., 0., 0., 0.)
        self.data.qpos[self.encoder_addresses] = measured_angles
        self.mujoco.mj_kinematics(self.model, self.data)
        midpoint = self.data.xpos[self.wrist_ids].mean(axis=0)
        attitude = Rotation.from_quat(imu_quat_wxyz, scalar_first=True).as_matrix()
        yaw = np.arctan2(attitude[1, 0], attitude[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        yaw_removed = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]]) @ attitude
        return yaw_removed @ midpoint


class ReceiverEnv(gym.Env):
    """Action is body vx/vy/yaw plus 14 upper targets. No gate runs in this env."""

    metadata = {"render_modes": []}

    def __init__(self, asset_root: Path = DEFAULT_RUNTIME_ROOT):
        self.plant = SensorPlant(asset_root)
        self.reach = EncoderReach(self.plant)
        self.observable = ObservableGenCoord(allow_synthetic_fixture=True)
        self.detector = ColorBallDetector(ball_center_offset_m=CATCH_BALL_CENTER_OFFSET_M)
        self.retention = VisualRetentionProxy(allow_synthetic_fixture=True)
        self.action_space = gym.spaces.Box(-1., 1., (ACTION_SIZE,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (HISTORY_LENGTH * OBS_SIZE,), dtype=np.float32,
        )
        self.family_probabilities = np.full(len(FAMILIES), 1. / len(FAMILIES))
        self.floor = self.plant.model.geom("floor").id
        self.floor_friction = self.plant.model.geom_friction[self.floor].copy()
        self._arm_indices = [self.plant.joint_names.index(name) for name in ARM_JOINTS]

    def set_family_probabilities(self, probabilities) -> None:
        p = np.asarray(probabilities, dtype=float)
        if p.shape != (len(FAMILIES),) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
            raise ValueError("invalid receiver trial probabilities")
        self.family_probabilities = p

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "trial" in options:
            self.trial = options["trial"]
        else:
            family = str(self.np_random.choice(FAMILIES, p=self.family_probabilities))
            self.trial = sample_trial(self.np_random, family)
        self.plant.model.geom_friction[self.floor] = self.floor_friction * self.trial.floor_friction_scale
        self.plant.reset(
            0., 0., 0., release_position_m=self.trial.release_position_m,
            release_velocity_mps=self.trial.release_velocity_mps,
        )
        self.observable.reset()
        self.retention.reset()
        self.route = (
            make_route(self.trial.route, scale=self.trial.route_scale, speed_mps=.30)
            if self.trial.route is not None else None
        )
        self.follower = RouteFollower(self.route) if self.route is not None else None
        self.goal_xy = np.array(self.route.waypoints[1] if self.route else [0., 0.])
        self.field_command = np.zeros(2)
        self.previous_action = np.zeros(ACTION_SIZE)
        self.upper = self.plant.wbc.nominal.copy()
        self.body = np.zeros(3)
        self.steps = 0
        self._pending: deque = deque()
        self._last_camera = None
        self._last_retention = None
        self._max_dwell = 0.
        self._previous_dwell = 0.
        self._visible_frames = 0
        self._camera_frames = 0
        self._progress = 0.
        self._had_receiver = False
        self._sensor_sample()
        self.state = self._state()
        self.history = deque(
            [np.zeros_like(self.state.vector) for _ in range(HISTORY_LENGTH - 1)]
            + [self.state.vector], maxlen=HISTORY_LENGTH,
        )
        return self._observation(), {}

    def _sensor_sample(self) -> None:
        frame, survey = self.plant.capture()
        if self.np_random.random() < self.trial.dropout_probability:
            frame = replace(frame, depth_m=np.zeros_like(frame.depth_m))
        elif self.trial.depth_noise_m > 0:
            # Correlated axial depth bias for this capture.
            noise = self.np_random.normal(0., self.trial.depth_noise_m)
            depth = np.where(frame.depth_m > 0., np.maximum(0., frame.depth_m + noise), 0.)
            frame = replace(frame, depth_m=depth)
        arrival = frame.timestamp_s + self.trial.camera_delay_ticks * CONTROL_DT
        self._pending.append((arrival, frame, survey))
        now = self.plant.timestamp_s
        while self._pending and self._pending[0][0] <= now + 1e-8:
            _, arrived_frame, arrived_survey = self._pending.popleft()
            receiver = self.plant.localizer.estimate(arrived_frame, arrived_survey)
            ball = self.detector.detect(arrived_frame, arrived_survey)
            self.observable.update_camera(arrived_frame.timestamp_s, receiver, ball)
            self._last_camera = (arrived_frame, arrived_survey, receiver)
            self._last_retention = self.retention.update(ball, receiver, now)
            self._camera_frames += 1
            self._visible_frames += int(ball is not None)
            self._max_dwell = max(self._max_dwell, self._last_retention.stable_duration_s)
            if self.follower is not None and receiver is not None:
                self.field_command = np.asarray(self.follower.command(receiver.xy_m))
                index = min(self.follower.next_waypoint, len(self.route.waypoints) - 1)
                self.goal_xy = np.array(self.route.waypoints[index])

    def _state(self):
        return self.observable.observe(
            self.plant.timestamp_s, self.plant.proprioception(), self.goal_xy,
            self.previous_action, self.plant.last_joint_targets,
        )

    def _observation(self):
        return np.concatenate(tuple(self.history))

    def _advance(self):
        self.plant.advance(self.body, ticks=1, upper_target=self.upper)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (ACTION_SIZE,) or not np.isfinite(action).all():
            raise ValueError("invalid receiver action")
        action = np.clip(action, -1., 1.)
        drive, upper_target = bounded_command(
            action, self.plant.wbc.nominal, self.plant.joint_names, self.plant.wbc.ranges,
        )
        if self.state.receiver is None:
            drive = np.zeros(3)
        for _ in range(POLICY_TICKS):
            self.body += np.clip(
                drive - self.body, -BODY_SLEW_PER_S * CONTROL_DT, BODY_SLEW_PER_S * CONTROL_DT,
            )
            arm_delta = CRADLE_TARGET_RATE_RAD_S * CONTROL_DT
            self.upper += np.clip(upper_target - self.upper, -arm_delta, arm_delta)
            self._advance()
        self.steps += 1
        previous = self.previous_action
        self.previous_action = action
        self._sensor_sample()
        self.state = self._state()
        self.history.append(self.state.vector)
        reward, info = self._reward(action, previous)
        fallen = info["imu_upright"] < .55
        truncated = self.steps >= round(EPISODE_S * POLICY_HZ) and not fallen
        if (fallen or truncated) and info["retained_visual_proxy"] and not fallen:
            reward += 2.
        return self._observation(), float(reward), bool(fallen), bool(truncated), info

    def _reward(self, action: np.ndarray, previous: np.ndarray) -> tuple[float, dict]:
        vector = self.state.vector
        upright = -float(vector[OBS_SLICES["gravity_body"]][2])
        reward = -.05 * max(0., .94 - upright) - .004 * float(np.mean((action - previous)**2))
        reward -= .001 * float(np.mean(action**2))
        receiver = self.state.receiver
        endpoint_error = None
        if receiver is not None and self.route is not None:
            path_error, progress = self.route.project(receiver.xy_m)
            if self._had_receiver:
                reward += 3. * np.clip(progress - self._progress, -.03, .03)
            reward -= .012 * min(path_error, .5)
            self._progress = progress
            endpoint_error = float(np.linalg.norm(receiver.xy_m - self.route.waypoints[-1]))
        self._had_receiver = receiver is not None
        reach_error = None
        if vector[OBS_SLICES["ball_valid"]][0]:
            ball = vector[OBS_SLICES["ball_relative_xyz_m"]]
            if 0. <= ball[0] <= .65 and -.10 <= ball[2] <= .62:
                q = vector[OBS_SLICES["joint_position_rad"]]
                imu = self.plant.proprioception()[2]
                reach_error = float(np.linalg.norm(ball - self.reach.wrist_midpoint(q, imu)))
                reward += .018 * np.exp(-(reach_error / .16)**2)
        latest = self._last_retention
        recent = latest is not None and 0 <= self.plant.timestamp_s - latest.timestamp_s <= self.retention.max_age_s
        dwell = latest.stable_duration_s if recent else 0.
        reward += .45 * (min(dwell, 2.) - min(self._previous_dwell, 2.))
        self._previous_dwell = dwell
        info = {
            "family": self.trial.family,
            "imu_upright": upright,
            "retained_visual_proxy": bool(recent and latest.retained_proxy),
            "final_visual_dwell_s": float(dwell),
            "max_visual_dwell_s": float(self._max_dwell),
            "visible_ball_frames": self._visible_frames,
            "camera_frames": self._camera_frames,
            "measured_route_progress_m": float(self._progress),
            "measured_endpoint_error_m": endpoint_error,
            "encoder_fk_reach_error_m": reach_error,
        }
        return reward, info

    def close(self):
        self.plant.close()


class VisualTeacher:
    """Optional BC/data-collection teacher. Kept out of learned rollout control."""

    def __init__(self, env: ReceiverEnv):
        self.env = env
        self.gate = VisualCatchGate(
            allow_synthetic_fixture=True, ball_center_offset_m=CATCH_BALL_CENTER_OFFSET_M,
        )
        nominal = env.plant.wbc.nominal
        cradle = cradle_joint_targets(nominal, env.plant.joint_names)
        self.arms = (cradle - nominal)[env._arm_indices] / ACTION_LIMITS[3:]
        if np.any(np.abs(self.arms) > 1.):
            raise ValueError("demonstrated cradle exceeds the learned arm action envelope")
        self.latched = False
        self.last_stamp = float("-inf")

    def action(self) -> np.ndarray:
        env = self.env
        sample = env._last_camera
        if sample is not None and sample[0].timestamp_s > self.last_stamp:
            self.last_stamp = sample[0].timestamp_s
            self.latched = self.gate.update(*sample, env.plant.timestamp_s).latched
        action = np.zeros(ACTION_SIZE, dtype=np.float32)
        receiver = env.state.receiver
        if receiver is not None and env.route is not None:
            c, s = np.cos(receiver.yaw_rad), np.sin(receiver.yaw_rad)
            action[:2] = np.array([[c, s], [-s, c]]) @ env.field_command / ACTION_LIMITS[:2]
        if self.latched:
            action[3:] = self.arms
        return np.clip(action, -1., 1.)
