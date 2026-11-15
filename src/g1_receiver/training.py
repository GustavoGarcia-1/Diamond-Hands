"""Bounded PPO pilot for a two-axis route residual over the existing G1 WBC."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
from typing import Callable
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np

from .learning import (
    OBSERVATION_SCHEMA, OBSERVATION_SIZE, RouteSensorState, SensorSGS, build_observation, route_reward,
)
from .localization import MarkerLocalizer
from .locomotion import (
    CONTROL_DT,
    SUBSTEPS,
    SURVEY_CAMERA_FOV_Y_DEG,
    SURVEY_CAMERA_POSITION_M,
    make_dynamic_scene,
    survey_camera_rotation,
)
from .perception import CameraIntrinsics, RGBDFrame, RigidPose
from .routes import RouteFollower, make_route
from .runtime import DEFAULT_RUNTIME_ROOT, RobotObservation, SkillGoal
from .sim import HEIGHT, WIDTH

TICKS_PER_ACTION = 5
ACTION_PERIOD_S = TICKS_PER_ACTION * CONTROL_DT
MAX_POSE_AGE_S = ACTION_PERIOD_S + 0.02
RESIDUAL_LIMIT_MPS = 0.08
SETTLE_S = 0.4
EPISODE_STEPS = 42
ENDPOINT_TOLERANCE_M = 0.07
SCENARIOS = ("centered", "short_offset", "long_offset")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    route_scale: float
    start_x_m: float
    start_y_m: float
    start_yaw_rad: float


def sample_scenario(rng: np.random.Generator, scenario_id: str) -> Scenario:
    if scenario_id == "centered":
        return Scenario(scenario_id, rng.uniform(.22, .25), 0., rng.uniform(-.015, .015), 0.)
    if scenario_id == "short_offset":
        return Scenario(
            scenario_id, rng.uniform(.23, .27), rng.uniform(-.03, .03),
            rng.uniform(-.08, .08), rng.uniform(-.08, .08),
        )
    if scenario_id == "long_offset":
        return Scenario(
            scenario_id, rng.uniform(.27, .31), rng.uniform(-.04, .04),
            rng.uniform(-.12, .12), rng.uniform(-.16, .16),
        )
    raise ValueError(f"unknown scenario {scenario_id!r}")


def _plant_xml(pool_root: Path) -> str:
    model = ET.fromstring(make_dynamic_scene(pool_root, survey_camera=True))
    actuators = model.find("actuator")
    sensors = model.find("sensor")
    for i, motor in enumerate(actuators):
        ET.SubElement(sensors, "jointpos", name=f"pilot_encoder_q_{i}", joint=motor.get("joint"))
        ET.SubElement(sensors, "jointvel", name=f"pilot_encoder_dq_{i}", joint=motor.get("joint"))
        ET.SubElement(sensors, "actuatorfrc", name=f"pilot_effort_{i}", actuator=motor.get("name"))
    return ET.tostring(model, encoding="unicode")


class SensorPlant:
    """Physics adapter. Base state is written for reset, and observed via RGB-D/IMU."""

    def __init__(self, pool_root: Path):
        import mujoco
        from ._vendor.g1_runtime.pool.wbc import WholeBodyController

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(_plant_xml(pool_root))
        self.data = mujoco.MjData(self.model)
        joints = self.model.actuator_trnid[:, 0]
        self.joint_names = [self.model.joint(int(joint)).name for joint in joints]
        names = self.joint_names
        if len(names) != 29:
            raise ValueError("pilot expects the 29-actuator bare-wrist G1")
        self.wbc = WholeBodyController(names, self.model.jnt_range[joints], asset_root=pool_root)
        self.joint_qpos = self.model.jnt_qposadr[joints]
        self.base_qpos = int(self.model.joint("floating_base_joint").qposadr[0])
        ball_joint = self.model.joint("football_free")
        self.ball_qpos = int(ball_joint.qposadr[0])
        self.ball_dof = int(ball_joint.dofadr[0])
        self.q_addr = self._actuator_sensor_addresses("encoder_q", len(names))
        self.dq_addr = self._actuator_sensor_addresses("encoder_dq", len(names))
        self.effort_addr = self._actuator_sensor_addresses("effort", len(names))
        self.imu_quat_slice = self._sensor_slice("receiver_orientation")
        self.imu_gyro_slice = self._sensor_slice("receiver_gyro")
        self.renderer = mujoco.Renderer(self.model, height=HEIGHT, width=WIDTH)
        self.localizer = MarkerLocalizer(allow_synthetic_fixture=True)
        focal = HEIGHT / (2 * np.tan(np.deg2rad(SURVEY_CAMERA_FOV_Y_DEG / 2)))
        self.intrinsics = CameraIntrinsics(focal, focal, (WIDTH - 1) / 2, (HEIGHT - 1) / 2)
        self.rotation = survey_camera_rotation()
        self.translation = np.array(SURVEY_CAMERA_POSITION_M)

    def _actuator_sensor_addresses(self, prefix: str, count: int) -> np.ndarray:
        return np.array([self.model.sensor(f"pilot_{prefix}_{i}").adr[0] for i in range(count)])

    def _sensor_slice(self, name: str) -> slice:
        sensor = self.model.sensor(name)
        return slice(sensor.adr[0], sensor.adr[0] + sensor.dim[0])

    @property
    def timestamp_s(self) -> float:
        return float(self.data.time)

    def reset(
        self,
        start_x_m: float,
        start_y_m: float,
        start_yaw_rad: float,
        *,
        release_position_m=(8., 4., .085),
        release_velocity_mps=(0., 0., 0.),
    ) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        self.wbc.reset()
        self.data.qpos[self.joint_qpos] = self.wbc.nominal
        start = self.base_qpos
        self.data.qpos[start:start + 3] = (start_x_m, start_y_m, .79)
        angle = start_yaw_rad / 2
        self.data.qpos[start + 3:start + 7] = (np.cos(angle), 0., 0., np.sin(angle))
        self.data.qpos[self.ball_qpos:self.ball_qpos + 3] = release_position_m
        self.data.qvel[self.ball_dof:self.ball_dof + 3] = release_velocity_mps
        self.mujoco.mj_forward(self.model, self.data)
        self.last_joint_targets = self.wbc.nominal.copy()

    def proprioception(self) -> tuple[np.ndarray, ...]:
        measured = self.data.sensordata
        return (
            measured[self.q_addr],
            measured[self.dq_addr],
            measured[self.imu_quat_slice].copy(),
            measured[self.imu_gyro_slice].copy(),
        )

    def capture(self) -> tuple[RGBDFrame, RigidPose]:
        self.renderer.update_scene(self.data, camera="survey_rgbd")
        rgb = self.renderer.render()
        self.renderer.enable_depth_rendering()
        try:
            depth = self.renderer.render()
        finally:
            self.renderer.disable_depth_rendering()
        stamp = self.timestamp_s
        pose = RigidPose(
            stamp, "camera_optical", "field", self.rotation, self.translation, "synthetic_fixture"
        )
        frame = RGBDFrame(stamp, "camera_optical", rgb, depth, self.intrinsics)
        return frame, pose

    def camera_pose(self):
        return self.localizer.estimate(*self.capture())

    def advance(
        self, command_xy: np.ndarray, *, ticks: int = TICKS_PER_ACTION,
        upper_target: np.ndarray | None = None,
    ) -> None:
        drive = np.asarray(command_xy, dtype=float)
        if drive.shape == (2,):
            drive = np.r_[drive, 0.]
        if drive.shape != (3,) or not np.isfinite(drive).all():
            raise ValueError("body drive must contain finite forward, lateral and optional yaw commands")
        goal = SkillGoal(base_velocity=drive)
        low, high = self.model.actuator_ctrlrange.T
        for _ in range(ticks):
            obs = RobotObservation(
                self.timestamp_s, *self.proprioception(), self.data.sensordata[self.effort_addr], None,
            )
            command = self.wbc.act(obs, goal, upper_target=upper_target)
            self.last_joint_targets = command.target_position.copy()
            for _ in range(SUBSTEPS):
                measured = self.data.sensordata
                torque = command.kp * (command.target_position - measured[self.q_addr])
                torque -= command.kd * measured[self.dq_addr]
                self.data.ctrl[:] = np.clip(torque, low, high)
                self.mujoco.mj_step(self.model, self.data)
        self.mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        self.renderer.close()


ScenarioSampler = Callable[[np.random.Generator], Scenario]
EpisodeRecorder = Callable[[str, bool], None]


class SensorRouteEnv(gym.Env):
    """Gymnasium env. PPO actor and critic receive the same sensor observation."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        pool_root: Path,
        choose_scenario: ScenarioSampler,
        record_episode: EpisodeRecorder | None = None,
    ):
        self.plant = SensorPlant(pool_root)
        self.choose_scenario = choose_scenario
        self.record_episode = record_episode
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.scenario = options["scenario"] if options and "scenario" in options else self.choose_scenario(self.np_random)
        self.plant.reset(
            self.scenario.start_x_m, self.scenario.start_y_m, self.scenario.start_yaw_rad
        )
        self.route = make_route("slant", scale=self.scenario.route_scale, speed_mps=.35)
        self.follower = RouteFollower(self.route)
        self.steps = 0
        self.progress_m = 0.
        self.cross_track_error_m = 0.
        self.missing_frames = 0
        self.proxy_success_frames = 0
        self.state = self._measure(np.zeros(2))
        return self._observation(), {}

    def _measure(self, prior_action: np.ndarray) -> RouteSensorState:
        pose = self.plant.camera_pose()
        command = np.zeros(2)
        if pose is None:
            self.missing_frames += 1
        else:
            self.cross_track_error_m, self.progress_m = self.route.project(pose.xy_m)
            command = np.asarray(self.follower.command(pose.xy_m))
        target_index = min(self.follower.next_waypoint, len(self.route.waypoints) - 1)
        return RouteSensorState(
            self.plant.timestamp_s, *self.plant.proprioception(), pose,
            np.asarray(self.route.waypoints[target_index]), command,
            self.progress_m, self.cross_track_error_m, prior_action,
        )

    def _observation(self):
        return build_observation(self.state, max_pose_age_s=MAX_POSE_AGE_S)

    def step(self, action):
        residual = np.asarray(action, dtype=float)
        if residual.shape != (2,) or not np.isfinite(residual).all():
            raise ValueError("PPO residual must be a finite 2-vector")
        residual = np.clip(residual, -1.0, 1.0)
        previous = self.state
        command = np.zeros(2)
        applied = np.zeros(2)
        pose = previous.pose
        if (
            previous.timestamp_s >= SETTLE_S
            and pose is not None
            and pose.fresh(previous.timestamp_s, MAX_POSE_AGE_S)
            and np.linalg.norm(previous.command_field_velocity_mps) > 0.
        ):
            c, s = np.cos(pose.yaw_rad), np.sin(pose.yaw_rad)
            body_from_field = np.array([[c, s], [-s, c]])
            applied = residual
            command = body_from_field @ previous.command_field_velocity_mps + RESIDUAL_LIMIT_MPS * applied
            speed = np.linalg.norm(command)
            if speed > 0.43:
                command *= .43 / speed

        self.plant.advance(command)
        self.steps += 1
        self.state = self._measure(applied)
        result = route_reward(previous, self.state, applied, max_pose_age_s=MAX_POSE_AGE_S)
        self.proxy_success_frames += int(result["tracking_proxy_success"])
        current_pose = self.state.pose
        endpoint_error = (
            float(np.linalg.norm(current_pose.xy_m - self.route.waypoints[-1]))
            if current_pose is not None else None
        )
        quat = self.state.imu_quat_wxyz
        upright = float(1 - 2 * (quat[1]**2 + quat[2]**2) / (quat @ quat))
        fallen = upright < .65
        reached = endpoint_error is not None and endpoint_error <= ENDPOINT_TOLERANCE_M and not fallen
        terminated = reached or fallen
        truncated = self.steps >= EPISODE_STEPS and not terminated
        proxy_fraction = self.proxy_success_frames / self.steps
        if (terminated or truncated) and self.record_episode is not None:
            self.record_episode(self.scenario.scenario_id, bool(reached and proxy_fraction >= 0.8))
        info = {
            "scenario_id": self.scenario.scenario_id,
            "measured_success": bool(reached),
            "imu_upright": upright,
            "measured_endpoint_error_m": endpoint_error,
            "measured_progress_fraction": self.progress_m / self.route.length_m,
            "missing_camera_frames": self.missing_frames,
            "tracking_proxy_fraction": proxy_fraction,
        }
        return self._observation(), float(result["reward"]), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.plant.close()


def _evaluate(env: SensorRouteEnv, predict, heldout: list[Scenario]) -> list[dict]:
    results = []
    for scenario in heldout:
        observation, _ = env.reset(options={"scenario": scenario})
        total_reward = 0.
        while True:
            observation, reward, terminated, truncated, info = env.step(predict(observation))
            total_reward += reward
            if terminated or truncated:
                results.append({
                    **info,
                    "scenario": asdict(scenario),
                    "sensor_return": total_reward,
                    "decision_steps": env.steps,
                })
                break
    return results


def _source_digests(asset_root: Path) -> dict[str, str]:
    module_root = Path(__file__).resolve().parent
    paths = {
        f"g1_receiver/{name}.py": module_root / f"{name}.py"
        for name in ("learning", "training", "localization", "locomotion", "routes", "perception", "runtime")
    }
    paths["g1_receiver/_vendor/g1_runtime/pool/wbc.py"] = DEFAULT_RUNTIME_ROOT / "pool/wbc.py"
    for relative in ("assets/robot/g1_body.xml", "assets/upstream/wbc.yaml",
                     "assets/upstream/motor.yaml", "checkpoints/walk.onnx", "checkpoints/balance.onnx"):
        paths[f"runtime_assets/{relative}"] = asset_root / relative
    return {name: sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def run_pilot(
    pool_root: Path,
    output: Path,
    *,
    seed: int,
    steps_per_mode: int,
    modes: tuple[str, ...],
    eval_episodes: int,
) -> dict:
    import stable_baselines3
    import torch

    torch.set_num_threads(1)
    start = monotonic()
    source_digests = _source_digests(pool_root)
    heldout_rng = np.random.default_rng(seed + 100_000)
    heldout = [
        sample_scenario(heldout_rng, SCENARIOS[i % len(SCENARIOS)])
        for i in range(eval_episodes)
    ]
    training = {}
    evaluated = {}
    models = {}
    for mode in modes:
        curriculum = SensorSGS(SCENARIOS) if mode == "sgs" else None
        episodes = []

        def choose(rng):
            probabilities = curriculum.probabilities() if curriculum is not None else None
            key = rng.choice(SCENARIOS, p=probabilities)
            return sample_scenario(rng, str(key))

        def record(scenario_id, success):
            episodes.append({"scenario_id": scenario_id, "sensor_curriculum_success": success})
            if curriculum is not None:
                curriculum.update(scenario_id, success)

        print(f"Training {mode} PPO for {steps_per_mode} sensor-driven decisions", flush=True)
        env = SensorRouteEnv(pool_root, choose, record)
        try:
            model = stable_baselines3.PPO(
                "MlpPolicy", env, device="cpu", seed=seed, n_steps=64, batch_size=32, n_epochs=4,
                gamma=.98, learning_rate=3e-4, policy_kwargs={"net_arch": [32, 32]}, verbose=1,
            )
            model.learn(total_timesteps=steps_per_mode)
            model.save(str(output / f"{mode}.zip"))
            models[mode] = model
            training[mode] = {
                "environment_steps": model.num_timesteps,
                "episodes": episodes,
                "final_sampling_probabilities": (
                    dict(zip(SCENARIOS, map(float, curriculum.probabilities())))
                    if curriculum is not None else dict.fromkeys(SCENARIOS, 1.0 / len(SCENARIOS))
                ),
            }
            (output / f"{mode}-training.json").write_text(json.dumps(training[mode], indent=2) + "\n")
        finally:
            env.close()

    print(f"Evaluating each controller on {len(heldout)} matched held-out scenarios", flush=True)
    env = SensorRouteEnv(pool_root, lambda rng: sample_scenario(rng, SCENARIOS[0]))
    try:
        evaluated["zero_residual"] = _evaluate(env, lambda _: np.zeros(2), heldout)
        for mode, model in models.items():
            evaluated[mode] = _evaluate(env, lambda obs: model.predict(obs, deterministic=True)[0], heldout)
    finally:
        env.close()

    return {
        "evidence_class": "rendered_simulation_sensor_route_pilot",
        "seed": seed,
        "sb3_version": stable_baselines3.__version__,
        "torch_version": str(torch.__version__),
        "actor_observation_schema": OBSERVATION_SCHEMA,
        "critic_observation_schema": OBSERVATION_SCHEMA,
        "source_sha256_at_launch": source_digests,
        "camera_period_s": ACTION_PERIOD_S,
        "actor_period_s": ACTION_PERIOD_S,
        "evaluation_basis": "measured_rgbd_marker_and_imu_proxy",
        "ppo_rollout_steps": 64,
        "sgs_episode_outcome": "measured_endpoint_and_80_percent_sensor_tracking_proxy",
        "wbc_period_s": CONTROL_DT,
        "residual_limit_mps_per_axis": RESIDUAL_LIMIT_MPS,
        "elapsed_s": monotonic() - start,
        "training": training,
        "paired_heldout_sensor_evaluation": evaluated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps-per-mode", type=int, default=192)
    parser.add_argument("--mode", choices=("uniform", "sgs", "both"), default="both")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=173)
    args = parser.parse_args()
    if args.steps_per_mode < 64 or args.steps_per_mode % 64:
        parser.error("--steps-per-mode must be a positive multiple of 64")
    if args.eval_episodes < 1 or args.seed < 0:
        parser.error("--eval-episodes must be positive and --seed nonnegative")
    modes = ("uniform", "sgs") if args.mode == "both" else (args.mode,)
    if len(modes) * args.steps_per_mode > 2048:
        parser.error("this pilot allows at most 2048 total PPO decisions")
    args.output.mkdir(parents=True, exist_ok=False)
    report = run_pilot(
        args.pool_root, args.output, seed=args.seed, steps_per_mode=args.steps_per_mode,
        modes=modes, eval_episodes=args.eval_episodes,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, results in report["paired_heldout_sensor_evaluation"].items():
        print(f"{name}: {sum(item['measured_success'] for item in results)}/{len(results)} measured completions")


if __name__ == "__main__":
    main()
