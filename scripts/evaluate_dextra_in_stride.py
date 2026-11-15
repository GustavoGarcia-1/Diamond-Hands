"""Catch a free delivery during a vision-guided route and carry it to the finish."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

import mujoco
import numpy as np

from audit_dextra_moving_release import initialize_delivery
from audit_receiver_all_routes import vision_drive
from audit_receiver_walk_terminal_cap import TerminalCapFollower
from g1_receiver.dextra import dextra_ready_joint_targets
from g1_receiver.dextra_physical_eval import DextraContactAudit, TRACE_COLUMNS
from g1_receiver.hand_route_plant import HandRoutePlant
from g1_receiver.in_stride_eval import MOTION_COLUMNS, score_in_stride_catch
from g1_receiver.localization import MarkerLocalizer
from g1_receiver.locomotion import CATCH_BALL_CENTER_OFFSET_M, CONTROL_DT
from g1_receiver.perception import ColorBallDetector
from g1_receiver.routes import make_route
from g1_receiver.runtime import RobotObservation, SkillGoal
from g1_receiver.short_throw import FreeBallRelease, plan_underhand_throw

ROUTES = ("go", "slant", "out", "hitch", "post")
REAR_CAMERA_M = (-1.6, 0., 1.02)


@dataclass(frozen=True)
class Trial:
    route: str
    release_s: float = 2.2
    drop_forward_m: float = .48
    drop_height_m: float = .49
    close_below_release_m: float = .085
    squeeze_rad: float = .04
    curl_rad: float = .60
    speed_mps: float = .20
    horizon_s: float = 12.
    wrist_down_rad: float = 0.
    flight_s: float = 0.
    incoming_mps: float = .48


class RearMarkerLocalizer(MarkerLocalizer):
    """Isolate the rear optical markers and reuse the sphere pose estimator."""

    def estimate(self, frame, camera_pose):
        red, green, blue = np.moveaxis(frame.rgb.astype(np.int16), -1, 0)
        left = (blue >= 100) & (blue - red >= 60) & (blue - green >= 60)
        right = (red >= 110) & (green >= 110) & (red - blue >= 55) & (green - blue >= 55)
        markers = np.zeros_like(frame.rgb)
        markers[left] = (0, 255, 255)
        markers[right] = (255, 0, 255)
        pose = super().estimate(replace(frame, rgb=markers), camera_pose)
        if pose is None:
            return None
        heading = np.array([np.cos(pose.yaw_rad), np.sin(pose.yaw_rad)])
        reflect = np.diag([1., 1., -1.])
        return replace(pose, xy_m=pose.xy_m + .30 * heading,
                       covariance=reflect @ pose.covariance @ reflect)


def make_play(trial):
    route = make_route(trial.route, scale=.24, speed_mps=trial.speed_mps)
    if trial.route == "hitch":
        turn_x, turn_y = route.waypoints[-2]
        route = replace(route, waypoints=(*route.waypoints[:-1], (turn_x - .18, turn_y)))
    return route


def camera_velocity(poses):
    samples = np.asarray(poses)
    times = samples[:, 0] - samples[:, 0].mean()
    return times @ samples[:, 1:3] / (times @ times)


def run(plant, trial, recording=None, *, delivery_enabled=True):
    if recording is None:
        return _run(plant, trial, delivery_enabled=delivery_enabled)
    plant.model.vis.global_.offwidth = 640
    plant.model.vis.global_.offheight = 360
    with mujoco.Renderer(plant.model, height=360, width=640) as renderer:
        return _run(plant, trial, recording, renderer, delivery_enabled=delivery_enabled)


def _run(plant, trial, recording=None, renderer=None, *, delivery_enabled=True):
    route = make_play(trial)
    follower = TerminalCapFollower(route, terminal_cap_m=.025)
    upper = dextra_ready_joint_targets(plant.nominal, plant.joint_names)
    plant.reset(joint_positions=upper)
    localizer = RearMarkerLocalizer(allow_synthetic_fixture=True)
    detector = ColorBallDetector(ball_center_offset_m=CATCH_BALL_CENTER_OFFSET_M)
    poses = deque(maxlen=5)
    drive = np.zeros(3)
    shoulder = [plant.joint_names.index(f"{side}_shoulder_roll_joint") for side in ("left", "right")]
    wrist_roll = [plant.joint_names.index(f"{side}_wrist_roll_joint") for side in ("left", "right")]
    pelvis = plant.data.body("pelvis")
    pose = release = audit = close_s = previous_height = None
    thrower = FreeBallRelease(plant.model, plant.data) if trial.flight_s else None
    motion, detections, waypoint_times = [], [], []
    overview, camera_frames, frame_times = [], [], []
    if renderer is not None:
        spectator = mujoco.MjvCamera()
        midpoint = np.mean(np.asarray(route.waypoints), axis=0)
        spectator.lookat[:] = (*midpoint, .74)
        spectator.azimuth = 126.
        spectator.elevation = -8.
        spectator.distance = 1.90
    localized = 0
    next_waypoint = 1
    shoulder_directions = np.array([1., -1.])
    for tick in range(round(trial.horizon_s / CONTROL_DT) + 1):
        now = plant.timestamp_s
        if tick % 2 == 0:
            frame, survey = plant.capture("route_rgbd")
            measured = localizer.estimate(frame, survey)
            if measured is not None:
                pose = measured
                localized += 1
                poses.append((now, *pose.xy_m))
            if (delivery_enabled and release is None and now >= trial.release_s - 1e-8 and measured is not None
                    and pose.z_m is not None and len(poses) >= 3):
                if thrower is None:
                    position, velocity, estimated = initialize_delivery(
                        plant, pose, poses, trial.drop_forward_m, trial.drop_height_m)
                    release = {"time_s": now, "position_m": position.tolist(), "velocity_mps": velocity.tolist(),
                               "estimated_receiver_velocity_mps": estimated.tolist(), "mode": "controlled_drop"}
                else:
                    estimated = camera_velocity(poses)
                    flight = plan_underhand_throw(
                        pose, estimated, release_time_s=now, flight_time_s=trial.flight_s,
                        intercept_body_xyz_m=(trial.drop_forward_m, 0., trial.drop_height_m),
                        incoming_relative_body_mps=(-trial.incoming_mps, 0., -.15),
                    )
                    thrower.apply(flight)
                    release = {"time_s": now, **asdict(flight), "mode": "underhand_lob",
                               "estimated_receiver_velocity_mps": estimated.tolist()}
                audit = DextraContactAudit(plant.model)
            if renderer is not None:
                renderer.update_scene(plant.data, camera=spectator)
                overview.append(renderer.render().copy())
                camera_frames.append(frame.rgb.copy())
                frame_times.append(now)
        if release is not None and close_s is None and now <= release["time_s"] + trial.flight_s + .50:
            ball = detector.detect(*plant.capture())
            if ball is not None:
                position = ball.position_m
                height = float(position[2])
                if thrower is None:
                    threshold = release["position_m"][2] - trial.close_below_release_m
                    near = np.linalg.norm(position[:2] - release["position_m"][:2]) < .15
                else:
                    # Gate from the surveyed receiver pose and configured reach envelope.
                    heading = np.array([np.cos(pose.yaw_rad), np.sin(pose.yaw_rad)])
                    threshold = pose.z_m + trial.drop_height_m - trial.close_below_release_m
                    aim = pose.xy_m + trial.drop_forward_m * heading
                    near = pose.fresh(now, .10) and np.linalg.norm(position[:2] - aim) < .15
                if near and previous_height is not None and height < previous_height and threshold - .14 < height <= threshold:
                    close_s = now
                previous_height = height
                detections.append((now, *position))
        xyz = pelvis.xpos.copy()
        upright = float(pelvis.xmat[8])
        motion.append((now, *xyz, upright))
        if next_waypoint < len(route.waypoints):
            start, end = np.asarray(route.waypoints[next_waypoint - 1:next_waypoint + 1])
            if np.linalg.norm(xyz[:2] - end) <= min(.05, .1 * np.linalg.norm(end - start)):
                waypoint_times.append(now)
                next_waypoint += 1
        if tick == round(trial.horizon_s / CONTROL_DT) or upright < .55:
            break
        target = vision_drive(follower, pose, now, max_age_s=.10, minimum_speed_mps=.055) if now >= .25 else np.zeros(3)
        drive += np.clip(target - drive, -.55 * CONTROL_DT, .55 * CONTROL_DT)
        arm_target = upper.copy()
        reinforcement = 0. if close_s is None else np.clip((now - close_s - .30) / .35, 0., 1.)
        arm_target[shoulder] -= shoulder_directions * trial.squeeze_rad * reinforcement
        wrist_ramp = 0. if close_s is None else np.clip((now - close_s - .55) / .50, 0., 1.)
        arm_target[wrist_roll] += shoulder_directions * trial.wrist_down_rad * wrist_ramp
        obs = RobotObservation(now, *plant.observe(), np.zeros(29), None)
        command = plant.wbc.act(obs, SkillGoal(base_velocity=drive), upper_target=arm_target)
        fingers = 0. if close_s is None else .60 + (trial.curl_rad - .60) * reinforcement
        plant.step(command, hand_target_rad=fingers * plant.actuation.flexion,
                   post_step=audit.record if audit is not None else None)
    motion = np.asarray(motion)
    endpoint = float(np.linalg.norm(motion[-1, 1:3] - route.waypoints[-1]))
    tolerance = min(.05, .1 * np.linalg.norm(np.subtract(*route.waypoints[-2:])))
    final_speed = float(np.linalg.norm(np.diff(motion[-26:, 1:3], axis=0), axis=1).sum() / .50)
    route_complete = bool(next_waypoint == len(route.waypoints) and endpoint <= tolerance
                          and motion[:, 4].min() >= .90 and final_speed <= .075
                          and np.isclose(motion[-1, 0], trial.horizon_s))
    trace = audit.trace() if audit else np.empty((0, len(TRACE_COLUMNS)))
    contact = audit.summary() if len(trace) else None
    tip_indices = [TRACE_COLUMNS.index(f"{side}_tip_normal_n") for side in ("left", "right")]
    tip_hits = np.flatnonzero(trace[:, tip_indices].max(axis=1) >= .10)
    first_tip = float(trace[tip_hits[0], 0]) if tip_hits.size else None
    pre_speed = early_travel = total_travel = None
    if first_tip is not None:
        prior = motion[(motion[:, 0] >= first_tip - .14) & (motion[:, 0] < first_tip)]
        if len(prior) > 1:
            pre_speed = float(np.linalg.norm(prior[-1, 1:3] - prior[0, 1:3]) / (prior[-1, 0] - prior[0, 0]))
        at_contact = motion[np.searchsorted(motion[:, 0], first_tip), 1:3]
        at_half = motion[min(len(motion) - 1, np.searchsorted(motion[:, 0], first_tip + .5)), 1:3]
        _, progress = route.project(at_contact)
        _, later = route.project(at_half, min_progress_m=progress)
        early_travel = later - progress
        total_travel = route.length_m - progress if route_complete else None
    grounded = np.flatnonzero(trace[:, TRACE_COLUMNS.index("ball_floor_contact")])
    first_floor = float(trace[grounded[0], 0]) if grounded.size else None
    warning_count = sum(int(warning.number) for warning in plant.data.warning)
    independent = score_in_stride_catch(
        trace, motion, route, release_time_s=release["time_s"] if release else None,
        finish_time_s=trial.horizon_s, ball_weight_n=audit.ball_weight_n if audit else .42 * 9.81,
        verified_free_ball=audit is not None, physics_step_s=float(plant.model.opt.timestep),
        simulator_warning_count=warning_count,
    )
    if recording is not None:
        recording.mkdir(parents=True, exist_ok=False)
        for name, images in (("overview", overview), ("camera-rgb", camera_frames)):
            np.savez_compressed(recording / f"{name}.npz", rgb=images, timestamps_s=frame_times)
    result = {
        **asdict(trial), "waypoints_m": route.waypoints, "waypoint_hit_times_s": waypoint_times,
        "release": release, "close_s": close_s, "first_tip_contact_s": first_tip,
        "pre_contact_speed_mps": pre_speed, "route_progress_next_half_second_m": early_travel,
        "post_contact_route_progress_m": total_travel, "first_floor_contact_s": first_floor,
        "localized_frames": localized, "elapsed_s": float(motion[-1, 0]),
        "endpoint_error_m": endpoint, "endpoint_tolerance_m": tolerance, "settled_speed_mps": final_speed,
        "min_upright_cosine": float(motion[:, 4].min()), "route_complete": route_complete,
        "hands_only_contact": contact,
        "warning_count": warning_count, "ball_weight_n": audit.ball_weight_n if audit else None,
        "in_stride_catch_and_route": bool(close_s is not None and independent["in_stride_catch_and_route"]),
        "independent_score": independent,
        "ball_detections_until_closure": detections,
    }
    return result, motion, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codesign-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", action="append", choices=ROUTES)
    parser.add_argument("--release", type=float, action="append")
    parser.add_argument("--drop-forward", type=float, default=.48)
    parser.add_argument("--drop-height", type=float, default=.49)
    parser.add_argument("--close-below-release", type=float, default=.085)
    parser.add_argument("--squeeze", type=float, default=.04)
    parser.add_argument("--curl", type=float, default=.60)
    parser.add_argument("--wrist-down", type=float, default=0., help="mirrored wrist roll after a visual grasp in radians")
    parser.add_argument("--speed", type=float, default=.20)
    parser.add_argument("--horizon", type=float, default=12.)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--flight", type=float, default=0., help="Plan an underhand lob of this duration")
    parser.add_argument("--incoming", type=float, default=.48, help="Relative approach speed for the lob")
    args = parser.parse_args()
    if (not .40 <= args.drop_forward <= .60 or not .44 <= args.drop_height <= .60
            or not .035 <= args.close_below_release <= .15 or not 0 <= args.squeeze <= .09
            or not .60 <= args.curl <= .82 or not .12 <= args.speed <= .35 or not 6 <= args.horizon <= 18
            or not 0 <= args.wrist_down <= .20):
        parser.error("delivery, grip, speed, or horizon is outside the fixture range")
    releases = args.release or [2.2]
    if (args.flight and not .10 <= args.flight <= .40) or not .35 <= args.incoming <= 1.20:
        parser.error("lob duration must be 0.10–0.40 seconds and approach speed 0.35–1.20 m/s")
    if not all(.8 <= release <= args.horizon - 2.5 for release in releases):
        parser.error("release must leave at least 2.5 seconds for the catch and carry")
    args.output.mkdir(parents=True, exist_ok=False)
    plant = HandRoutePlant(args.codesign_source, route_camera_position=REAR_CAMERA_M, rear_markers=True)
    report = {"evidence_class": "controlled_free_ball_catch_during_route_simulation",
              "rear_camera_m": REAR_CAMERA_M, "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "results": []}
    try:
        for route in args.route or ["go"]:
            for release in releases:
                trial = Trial(route, release, args.drop_forward, args.drop_height, args.close_below_release,
                              args.squeeze, args.curl, args.speed, args.horizon, wrist_down_rad=args.wrist_down,
                              flight_s=args.flight, incoming_mps=args.incoming)
                folder = args.output / f"{route}-{release:.2f}"
                result, motion, trace = run(plant, trial, folder if args.record else None)
                trace_path = folder.with_name(f"{folder.name}.npz")
                np.savez_compressed(trace_path, contact=trace, contact_columns=TRACE_COLUMNS,
                                    motion=motion, motion_columns=MOTION_COLUMNS)
                result["trace_file"] = trace_path.name
                result["trace_sha256"] = sha256(trace_path.read_bytes()).hexdigest()
                report["results"].append(result)
                print(route, release, result["close_s"], result["pre_contact_speed_mps"],
                      result["first_floor_contact_s"], round(result["endpoint_error_m"], 3),
                      result["in_stride_catch_and_route"], flush=True)
    finally:
        plant.close()
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
