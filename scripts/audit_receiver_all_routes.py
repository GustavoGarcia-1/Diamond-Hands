"""Run no-throw vision/WBC locomotion on five route shapes and three mirrors."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import math
import os
from pathlib import Path

import numpy as np

from audit_receiver_checkpoints import ROOT, digest, read_json, route_result
from g1_receiver.locomotion import CONTROL_DT
from g1_receiver.routes import Route, RouteFollower, make_route
from g1_receiver.runtime import DEFAULT_RUNTIME_ROOT
from g1_receiver.train_receiver import _source_snapshot
from g1_receiver.training import SensorPlant

PLAN_NAME = "receiver-all-route-walk-plan.json"


def vision_drive(
    follower: RouteFollower, pose, now_s: float, *, max_age_s: float, minimum_speed_mps: float = 0.,
) -> np.ndarray:
    """Use surveyed pose only, with an optional floor above the WBC walk deadband."""
    if pose is None or not pose.fresh(now_s, max_age_s):
        return np.zeros(3)
    field = np.asarray(follower.command(pose.xy_m))
    magnitude = np.linalg.norm(field)
    if 0. < magnitude < minimum_speed_mps:
        field *= minimum_speed_mps / magnitude
    c, s = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
    return np.r_[np.array([[c, s], [-s, c]]) @ field, 0.]


def run_route(plant: SensorPlant, route: Route, plan: dict, minimum_speed_mps: float) -> dict:
    plant.reset(0., 0., 0.)
    follower = RouteFollower(route)
    data, model = plant.data, plant.model
    pelvis = model.body("pelvis").id
    football = model.body("football").id
    floor = model.geom("floor").id
    foot_bodies = tuple(model.body(f"{side}_ankle_roll_link").id for side in ("left", "right"))
    next_capture_s = 0.
    pose = None
    trace = []
    commands = []
    vision_errors = []
    vision_captures = 0
    stale_ticks = 0
    foot_contact_counts = np.zeros(3, dtype=int)
    min_ball_separation = math.inf
    announced_arrival_s = None
    planned_steps = round(plan["horizon_s"] / CONTROL_DT)
    for step in range(planned_steps + 1):
        now_s = plant.timestamp_s
        truth = data.xpos[pelvis].copy()
        drive = np.zeros(3)
        if now_s >= next_capture_s - 1e-8:
            next_capture_s += 1 / plan["camera_hz"]
            vision_captures += 1
            candidate = plant.camera_pose()
            if candidate is not None:
                pose = candidate
                vision_errors.append(float(np.linalg.norm(pose.xy_m - truth[:2])))
        if now_s >= plan["startup_balance_s"] - 1e-8:
            stale_ticks += int(pose is None or not pose.fresh(now_s, plan["max_pose_age_s"]))
            drive = vision_drive(
                follower, pose, now_s, max_age_s=plan["max_pose_age_s"], minimum_speed_mps=minimum_speed_mps,
            )
        if follower.next_waypoint >= len(route.waypoints) and announced_arrival_s is None:
            announced_arrival_s = now_s

        feet = set()
        for contact in data.contact:
            if contact.efc_address < 0 or floor not in (contact.geom1, contact.geom2):
                continue
            other_geom = contact.geom2 if contact.geom1 == floor else contact.geom1
            body = model.geom_bodyid[other_geom]
            if body in foot_bodies:
                feet.add(body)
        if np.linalg.norm(drive) >= .05:
            foot_contact_counts[len(feet)] += 1
        min_ball_separation = min(min_ball_separation, float(np.linalg.norm(data.xpos[football] - truth)))
        trace.append((now_s, truth[0], truth[1], float(data.xmat[pelvis, 8])))
        commands.append(drive)
        if step < planned_steps:
            plant.advance(drive, ticks=1)

    trace = np.asarray(trace)
    commands = np.asarray(commands)
    time, positions, upright = trace[:, 0], trace[:, 1:3], trace[:, 3]
    valid = bool(
        np.isfinite(trace).all()
        and np.allclose(time, CONTROL_DT * np.arange(len(time)), atol=1e-7, rtol=0.)
    )
    audit = np.column_stack((time, np.zeros((len(trace), 4)), positions))
    scored = route_result(route, audit, plan, valid)
    ordered = scored["endpoint_reached"]
    endpoint_error = float(np.linalg.norm(positions[-1] - route.waypoints[-1]))
    max_error = max(route.project(position)[0] for position in positions)
    tail = time >= time[-1] - plan["final_speed_window_s"] - 1e-8
    total_path_m = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    tail_speed = float(np.linalg.norm(np.diff(positions[tail], axis=0), axis=1).sum() / np.ptp(time[tail]))
    window_steps = round(.20 / CONTROL_DT)
    window_speed = np.linalg.norm(positions[window_steps:] - positions[:-window_steps], axis=1) / (.20)
    walk_mask = np.linalg.norm(commands[:-window_steps], axis=1) >= .05
    moving_speed = window_speed[walk_mask]
    final_hit_s = scored["ordered_waypoint_hit_s"][-1] if len(scored["ordered_waypoint_hit_s"]) == len(route.waypoints) - 1 else None
    reverse_m = None
    if route.name == "hitch" and len(scored["ordered_waypoint_hit_s"]) >= 1:
        turn_index = int(np.searchsorted(time, scored["ordered_waypoint_hit_s"][0]))
        reverse_m = float(np.max(positions[turn_index:, 0]) - positions[-1, 0])
    passed = (
        ordered and float(np.min(upright)) >= plan["minimum_upright_cosine"]
        and max_error <= plan["max_cross_track_m"]
        and tail_speed <= plan["max_final_mean_speed_mps"]
        and min_ball_separation > 3.
    )
    return {
        "passed": passed,
        "ordered_endpoint_reached": ordered,
        "timebase_valid": valid,
        "route_waypoints_m": route.waypoints,
        "route_length_m": route.length_m,
        "waypoint_hit_times_s": scored["ordered_waypoint_hit_s"],
        "controller_arrival_s": announced_arrival_s,
        "final_ordered_waypoint_first_hit_s": final_hit_s,
        "hitch_backward_travel_m": reverse_m,
        "endpoint_tolerance_m": scored["endpoint_tolerance_m"],
        "endpoint_error_m": endpoint_error,
        "max_cross_track_error_m": max_error,
        "min_upright_cosine": float(np.min(upright)),
        "final_mean_speed_mps": tail_speed,
        "total_pelvis_path_m": total_path_m,
        "mean_moving_speed_mps_200ms": float(np.mean(moving_speed)) if moving_speed.size else None,
        "p95_moving_speed_mps_200ms": float(np.percentile(moving_speed, 95)) if moving_speed.size else None,
        "peak_moving_speed_mps_200ms": float(np.max(moving_speed)) if moving_speed.size else None,
        "walking_command_samples_with_0_1_2_feet_on_floor": foot_contact_counts.tolist(),
        "camera_frames": vision_captures,
        "localized_frames": len(vision_errors),
        "localization_p95_error_m": float(np.percentile(vision_errors, 95)) if vision_errors else None,
        "stale_pose_control_ticks": stale_ticks,
        "closest_parked_ball_distance_m": min_ball_separation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "reports" / PLAN_NAME)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("stock", "walk_deadband"), default="stock")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    plan = read_json(args.plan)
    minimum_speed = plan["minimum_nonzero_command_mps"][args.mode]
    cases = [(name, False) for name in plan["routes"]] + [(name, True) for name in plan["mirrored_routes"]]
    plant = SensorPlant(DEFAULT_RUNTIME_ROOT)
    try:
        results = {}
        for name, mirror in cases:
            route = make_route(name, scale=plan["scale"], speed_mps=plan["command_speed_mps"], mirror=mirror)
            case_id = f"{name}_mirror" if mirror else name
            result = run_route(plant, route, plan, minimum_speed)
            results[case_id] = result
            print(
                case_id, result["passed"], result["waypoint_hit_times_s"],
                round(result["endpoint_error_m"], 4), result["controller_arrival_s"], flush=True,
            )
    finally:
        plant.close()
    report = {
        "evidence_class": "no_throw_rendered_vision_wbc_walk",
        "plan_sha256": digest(args.plan),
        "audit_script_sha256": digest(Path(__file__)),
        "source_sha256": _source_snapshot(None, DEFAULT_RUNTIME_ROOT),
        "mode": args.mode,
        "minimum_nonzero_command_mps": minimum_speed,
        "configuration": plan,
        "runtime": {"mujoco": version("mujoco"), "mujoco_gl": os.environ.get("MUJOCO_GL"), "controller_hz": 1 / CONTROL_DT},
        "passed": sum(row["passed"] for row in results.values()),
        "rollouts": len(results),
        "results": results,
        "limitations": [
            "Fixed surveyed RGB-D camera and rendered markers. The bare-wrist G1 walks without a throw. The scene football is parked outside the course.",
            "Only vendor balance and walk policies are available. Foot contact is sampled at 50 Hz and does not establish running or exclude short unsampled flight phases.",
            "The camera survey and simulator truth are used for separate purposes. Route commands use measured pose. Physical truth is read only for the audit.",
        ],
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
