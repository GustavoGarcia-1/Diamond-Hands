"""Brake the Aero G1 from AMP running using surveyed pose and a second controller."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import mujoco
import numpy as np

from evaluate_aero_receiver import fixed_camera
from evaluate_dextra_in_stride import RearMarkerLocalizer
from g1_receiver._vendor.g1_runtime.pool.contracts import ActuatorCommand, RobotObservation, SkillGoal
from g1_receiver.aero import aero_ready_joint_targets
from g1_receiver.hand_route_plant import HandRoutePlant
from g1_receiver.locomotion import CONTROL_DT
from g1_receiver.pass_control import observed_velocity
from g1_receiver.running_catch_eval import FootContactRecorder
from g1_receiver.running_policy import AmpRunningPolicy, CHECKPOINT_SHA256, CONFIG_SHA256


@dataclass(frozen=True)
class Plan:
    brake_x: float = 8.34
    min_brake_time_s: float = 4.24
    endpoint_x: float = 13.33
    endpoint_y: float = 3.89
    blend_s: float = .65
    absorb_speed: float = .75
    absorb_s: float = .65
    ramp_down_s: float = .80
    stop_radius: float = .035
    terminal_speed: float = .33
    horizon_s: float = 10.


def blend_pd(amp: ActuatorCommand, walk: ActuatorCommand, fraction: float) -> ActuatorCommand:
    """Blend unsaturated torques before the plant applies the motor torque limits."""
    ka = (1. - fraction) * amp.kp
    kw = fraction * walk.kp
    kp = ka + kw
    q = (ka * amp.target_position + kw * walk.target_position) / kp
    q = np.clip(q, np.minimum(amp.target_position, walk.target_position),
                np.maximum(amp.target_position, walk.target_position))
    return ActuatorCommand(q, kp, (1. - fraction) * amp.kd + fraction * walk.kd,
                           (1. - fraction) * amp.feedforward_torque + fraction * walk.feedforward_torque,
                           "amp_to_wbc_torque_blend")


def run(plant, policy, plan, *, viewer=None):
    upper = aero_ready_joint_targets(policy.nominal, plant.joint_names)
    plant.reset(joint_positions=upper, release_position_m=(18., 10., .085))
    policy.reset()
    pose = None
    poses = []
    brake_at = None
    brake_start = None
    settled = False
    feet = FootContactRecorder(plant.model)
    localizer = RearMarkerLocalizer(allow_synthetic_fixture=True, max_range_m=8.)
    body = plant.data.body("pelvis")
    endpoint = np.array([plan.endpoint_x, plan.endpoint_y])
    motion, sensed, images, stamps = [], [], [], []
    if viewer is not None:
        spectator = mujoco.MjvCamera()
        spectator.azimuth, spectator.elevation, spectator.distance = 123., -12., 3.5
    for tick in range(round(plan.horizon_s / CONTROL_DT) + 1):
        now = plant.timestamp_s
        observation = RobotObservation(now, *plant.observe(), np.zeros(29), None)
        w, x, y, z = observation.imu_quat_wxyz
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        if tick % 2 == 0:
            if now >= 2.6:
                order = ("finish", "entry") if pose is not None and pose.xy_m[0] >= 9.5 else ("entry", "finish")
                for camera in order:
                    candidate = localizer.estimate(*plant.capture(camera))
                    if candidate is not None:
                        pose = candidate
                        poses.append((now, *pose.xy_m))
                        poses = poses[-5:]
                        break
            if viewer is not None:
                spectator.lookat[:] = (*body.xpos[:2] + np.array([.40, 0.]), .78)
                viewer.update_scene(plant.data, camera=spectator)
                images.append(viewer.render())
                stamps.append(now)
        # Once the braking monitor is armed, stale pose triggers a conservative handoff.
        fresh = pose is not None and pose.fresh(now, .14)
        if (brake_at is None and now >= plan.min_brake_time_s - 1e-8
                and ((fresh and pose.xy_m[0] >= plan.brake_x) or (not fresh and now >= 4.40))):
            brake_at = now
            brake_start = {"time_s": now, "basis": "camera_progress" if fresh else "camera_timeout",
                           "estimated_xy_m": pose.xy_m.tolist() if fresh else None}
        elapsed = 0. if brake_at is None else now - brake_at
        u = float(np.clip(elapsed / plan.blend_s, 0., 1.))
        fraction = u * u * (3. - 2. * u)
        vx = min(2., 1.8 * max(0., now - .4))
        amp_goal = SkillGoal(base_velocity=np.array([vx * (1. - .4 * u), 0., np.clip(-1.8 * yaw, -.75, .75)]))
        absorb = plan.absorb_speed * np.clip(1. - max(0., elapsed - plan.absorb_s) / plan.ramp_down_s, 0., 1.)
        walk_drive = np.array([absorb, 0., np.clip(-.8 * yaw, -.28, .28) if absorb > .05 else 0.])
        if elapsed >= plan.absorb_s + plan.ramp_down_s:
            walk_drive[:] = 0.
            if fresh:
                error = endpoint - pose.xy_m
                distance = np.linalg.norm(error)
                speed_measurement = observed_velocity(poses)
                if distance <= plan.stop_radius and speed_measurement is not None and np.linalg.norm(speed_measurement) < .15:
                    settled = True
                if not settled:
                    field = .85 * error
                    magnitude = np.linalg.norm(field)
                    if magnitude >= 1e-6:
                        field *= np.clip(magnitude, .055, plan.terminal_speed) / magnitude
                    c, s = np.cos(pose.yaw_rad), np.sin(pose.yaw_rad)
                    walk_drive[:2] = np.array([[c, s], [-s, c]]) @ field
        motion.append((now, *body.xpos, float(body.xmat[8])))
        sensed.append((now, *(pose.xy_m if pose is not None else [np.nan, np.nan]), fresh,
                       *amp_goal.base_velocity, *walk_drive, fraction, settled))
        if tick == round(plan.horizon_s / CONTROL_DT) or 1. - 2. * (x*x + y*y) < .5:
            break
        amp = policy.act(observation, amp_goal, upper)
        walk = plant.wbc.act(observation, SkillGoal(base_velocity=walk_drive), upper_target=upper)
        plant.step(blend_pd(amp, walk, fraction), post_step=feet.record)
    motion, sensed = np.array(motion), np.array(sensed)
    tail = motion[motion[:, 0] >= motion[-1, 0] - .5 - 1e-8]
    tail_speed = float(np.linalg.norm(np.diff(tail[:, 1:3], axis=0), axis=1).sum() / (tail[-1, 0] - tail[0, 0]))
    error = np.linalg.norm(motion[:, 1:3] - endpoint, axis=1)
    in_bounds = np.flatnonzero(error <= .05)
    warnings = sum(warning.number for warning in plant.data.warning)
    report = {"plan": asdict(plan), "brake_start": brake_start, "actual_finish_s": float(motion[-1, 0]),
              "target_entry_s": float(motion[in_bounds[0], 0]) if len(in_bounds) else None,
              "endpoint_error_m": float(error[-1]), "closest_endpoint_m": float(error.min()),
              "pelvis_final_xy_m": motion[-1, 1:3].tolist(), "final_mean_speed_mps": tail_speed,
              "minimum_upright_cosine": float(motion[:, 4].min()), "warning_count": warnings,
              "terminal_settled": bool(motion[-1, 0] >= plan.horizon_s - 1e-7 and not warnings
                                       and motion[:, 4].min() >= .90 and tail_speed <= .075),
              "final_camera_pose_accepted": bool(sensed[-1, 3])}
    report["route_finish"] = bool(report["terminal_settled"] and error[-1] <= .05)
    return report, motion, sensed, feet.arrays(), images, stamps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--amp-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name, value in asdict(Plan()).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=value)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    plan = Plan(**{name: getattr(args, name) for name in asdict(Plan())})
    if (not 6. < plan.brake_x < plan.endpoint_x or not 0. < plan.blend_s <= 2.
            or not 3. <= plan.min_brake_time_s <= 4.4 or not 6. <= plan.horizon_s <= 15.):
        parser.error("invalid target, transition duration or horizon")
    args.output.mkdir(parents=True, exist_ok=False)
    cameras = {}

    def setup(scene):
        cameras["entry"] = fixed_camera(scene, "entry", (4.3, 1.50, 1.43), (8., 3.1, .81))
        cameras["finish"] = fixed_camera(scene, "finish", (8.8, 3.0, 1.37), (12.8, 3.7, .81))

    plant = HandRoutePlant(args.source, hand_model="aero", rear_markers=True, scene_setup=setup)
    plant.cameras.update(cameras)
    backdrop = plant.model.geom("backdrop").id
    if plant.model.geom_contype[backdrop] or plant.model.geom_conaffinity[backdrop]:
        raise ValueError("the backdrop must be visual only")
    plant.model.geom_rgba[backdrop, 3] = 0.
    plant.model.geom_group[backdrop] = 5
    policy = AmpRunningPolicy(plant.joint_names, plant.joint_ranges, args.amp_source)
    viewer = None
    try:
        if args.record:
            plant.model.vis.global_.offwidth = 640
            plant.model.vis.global_.offheight = 360
            viewer = mujoco.Renderer(plant.model, height=360, width=640)
        report, motion, sensors, foot_trace, rgb, times = run(plant, policy, plan, viewer=viewer)
        trace = args.output / "trace.npz"
        np.savez_compressed(trace, motion=motion, sensors=sensors, **foot_trace)
        report.update({"evidence_class": "simulation_only_aero_run_braking_without_pass", "trace_sha256": sha256(trace.read_bytes()).hexdigest(),
                       "checkpoint_sha256": CHECKPOINT_SHA256, "config_sha256": CONFIG_SHA256,
                       "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest()})
        if rgb:
            np.savez_compressed(args.output / "view.npz", rgb=rgb, timestamps_s=times)
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if viewer is not None:
            viewer.close()
        plant.close()


if __name__ == "__main__":
    main()
