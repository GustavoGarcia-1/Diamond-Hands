"""Run a vision-timed controlled drop after a hand-equipped G1 is already walking."""

from __future__ import annotations

import argparse
from collections import deque
from hashlib import sha256
import json
from pathlib import Path

import mujoco
import numpy as np

from audit_receiver_all_routes import vision_drive
from audit_receiver_walk_terminal_cap import TerminalCapFollower
from g1_receiver.dextra import UPSTREAM_URDF, dextra_ready_joint_targets, verify_source_urdf
from g1_receiver.dextra_physical_eval import DextraContactAudit, TRACE_COLUMNS
from g1_receiver.hand_route_plant import HandRoutePlant
from g1_receiver.localization import MarkerLocalizer
from g1_receiver.locomotion import CATCH_BALL_CENTER_OFFSET_M, CONTROL_DT
from g1_receiver.perception import ColorBallDetector
from g1_receiver.routes import make_route
from g1_receiver.runtime import RobotObservation, SkillGoal

ROUTE_CAMERA_M = (2.5, -.95, .65)
DROP_BODY_M = np.array([.47, 0., .51])
DROP_RELATIVE_VELOCITY_MPS = np.array([-.20, 0., -.15])
CLOSE_ABOVE_RELEASE_PELVIS_M = .425
WALK_START_S = .25
HORIZON_S = 8.
SPEED_MPS = .20
RELEASE_TIMES_S = (1.0, 1.3, 1.6, 1.9)


def initialize_delivery(plant, pose, positions, drop_forward_m, drop_height_m):
    """Place the free ball once, from the current surveyed pose and velocity."""
    samples = np.array(positions)
    times = samples[:, 0] - samples[:, 0].mean()
    velocity = times @ samples[:, 1:3] / (times @ times)
    cosine, sine = np.cos(pose.yaw_rad), np.sin(pose.yaw_rad)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    position = np.r_[pose.xy_m + rotation @ np.array([drop_forward_m, DROP_BODY_M[1]]), pose.z_m + drop_height_m]
    speed = np.r_[velocity + rotation @ DROP_RELATIVE_VELOCITY_MPS[:2], DROP_RELATIVE_VELOCITY_MPS[2]]
    plant.data.qpos[plant.ball_qpos:plant.ball_qpos + 7] = [*position, 1., 0., 0., 0.]
    plant.data.qvel[plant.ball_dof:plant.ball_dof + 6] = [*speed, 0., 0., 0.]
    mujoco.mj_forward(plant.model, plant.data)
    return position, speed, velocity


def evaluate(plant: HandRoutePlant, release_s: float, drop_forward_m: float, drop_height_m: float):
    upper = dextra_ready_joint_targets(plant.nominal, plant.joint_names)
    plant.reset(joint_positions=upper)
    route = make_route("go", scale=.24, speed_mps=SPEED_MPS)
    follower = TerminalCapFollower(route, terminal_cap_m=.025)
    localizer = MarkerLocalizer(allow_synthetic_fixture=True)
    detector = ColorBallDetector(ball_center_offset_m=CATCH_BALL_CENTER_OFFSET_M)
    recent_poses = deque(maxlen=5)
    drive = np.zeros(3)
    body_effort = np.zeros(29)
    shoulder_roll = [plant.joint_names.index(f"{side}_shoulder_roll_joint") for side in ("left", "right")]
    pelvis = plant.data.body("pelvis")
    current_pose = None
    previous_height = None
    first_pose_xy = None
    release_info = None
    close_s = None
    audit = None
    checkpoint_audit = None
    motion = []
    pelvis_estimates = []
    detections = []
    for tick in range(round(HORIZON_S / CONTROL_DT) + 1):
        now = plant.timestamp_s
        if tick % 2 == 0:
            estimate = localizer.estimate(*plant.capture("route_rgbd"))
            if estimate is not None:
                current_pose = estimate
                xy = estimate.xy_m.copy()
                recent_poses.append((now, *xy))
                pelvis_estimates.append((now, *xy))
                if first_pose_xy is None:
                    first_pose_xy = xy
            if (release_info is None and now >= release_s - 1e-8 and estimate is not None
                    and estimate.z_m is not None and len(recent_poses) >= 3):
                position, velocity, estimated_speed = initialize_delivery(
                    plant, estimate, recent_poses, drop_forward_m, drop_height_m,
                )
                audit = DextraContactAudit(plant.model)
                release_info = {
                    "time_s": now, "ball_initial_position_m": position.tolist(),
                    "ball_initial_velocity_mps": velocity.tolist(), "receiver_estimated_velocity_mps": estimated_speed.tolist(),
                    "receiver_estimated_xy_m": estimate.xy_m.tolist(), "receiver_estimated_z_m": estimate.z_m,
                    "camera_estimated_pre_release_displacement_m": float(np.linalg.norm(estimate.xy_m - first_pose_xy)),
                }
        if release_info is not None and close_s is None and now <= release_info["time_s"] + .50:
            ball = detector.detect(*plant.capture())
            if ball is not None:
                x, y, z = map(float, ball.position_m)
                detections.append((now, x, y, z))
                threshold = release_info["ball_initial_position_m"][2] - (DROP_BODY_M[2] - CLOSE_ABOVE_RELEASE_PELVIS_M)
                near = np.linalg.norm(ball.position_m[:2] - release_info["ball_initial_position_m"][:2]) < .15
                if near and previous_height is not None and z < previous_height and threshold - .14 < z <= threshold:
                    close_s = now
                previous_height = z
        if audit is not None and checkpoint_audit is None and now >= release_info["time_s"] + 2.50 - 1e-8:
            checkpoint_audit = {"time_s": now, **audit.summary()}
        motion.append((now, *pelvis.xpos, float(pelvis.xmat[8])))
        observation = RobotObservation(now, *plant.observe(), body_effort, None)
        tilt = observation.imu_quat_wxyz[1:3]
        if tick == round(HORIZON_S / CONTROL_DT) or 1. - 2. * float(tilt @ tilt) < .55:
            break
        target_drive = (vision_drive(follower, current_pose, now, max_age_s=.10, minimum_speed_mps=.055)
                        if now >= WALK_START_S else np.zeros(3))
        drive += np.clip(target_drive - drive, -.55 * CONTROL_DT, .55 * CONTROL_DT)
        arm_target = upper.copy()
        if close_s is not None:
            reinforcement = np.clip((now - close_s - .30) / .35, 0., 1.)
            arm_target[shoulder_roll] -= np.array([1., -1.]) * .04 * reinforcement
        command = plant.wbc.act(observation, SkillGoal(base_velocity=drive), upper_target=arm_target)
        plant.step(command, closure=float(close_s is not None), post_step=None if audit is None else audit.record)

    motion = np.asarray(motion)
    trace = audit.trace() if audit is not None else np.empty((0, len(TRACE_COLUMNS)))
    side_normal = [trace[:, [TRACE_COLUMNS.index(f"{side}_tip_normal_n"), TRACE_COLUMNS.index(f"{side}_palm_normal_n")]].sum(axis=1)
                   for side in ("left", "right")]
    contact_hits = np.flatnonzero(np.max(side_normal, axis=0) >= .10)
    first_contact_s = float(trace[contact_hits[0], 0]) if len(contact_hits) else None
    recent_speed = None
    estimated_pre_contact_displacement = None
    camera_recent_speed = None
    if first_contact_s is not None:
        recent = motion[(motion[:, 0] < first_contact_s) & (motion[:, 0] >= first_contact_s - .14)]
        if len(recent) > 1:
            recent_speed = float(np.linalg.norm(recent[-1, 1:3] - recent[0, 1:3]) / (recent[-1, 0] - recent[0, 0]))
        estimates = np.asarray(pelvis_estimates)
        prior = estimates[estimates[:, 0] < first_contact_s]
        if len(prior) > 1:
            estimated_pre_contact_displacement = float(np.linalg.norm(prior[-1, 1:] - prior[0, 1:]))
        recent_estimates = prior[prior[:, 0] >= first_contact_s - .24]
        if len(recent_estimates) > 1:
            camera_recent_speed = float(np.linalg.norm(recent_estimates[-1, 1:] - recent_estimates[0, 1:])
                                        / (recent_estimates[-1, 0] - recent_estimates[0, 0]))
    endpoint_errors = np.linalg.norm(motion[:, 1:3] - route.waypoints[-1], axis=1)
    tail_speed = float(np.linalg.norm(np.diff(motion[-26:, 1:3], axis=0), axis=1).sum() / .50)
    full_horizon = bool(np.isclose(plant.timestamp_s, HORIZON_S))
    warning_count = sum(w.number for w in plant.data.warning)
    route_pass = bool(full_horizon and endpoint_errors[-1] <= .05 and tail_speed <= .075
                      and motion[:, 4].min() >= .90 and warning_count == 0)
    final = audit.summary() if audit is not None else None
    moving_catch = bool(checkpoint_audit is not None and checkpoint_audit["hands_only_catch"]
                        and recent_speed is not None and recent_speed >= .05
                        and camera_recent_speed is not None and camera_recent_speed >= .05)
    result = {
        "requested_release_s": release_s, "drop_forward_m": drop_forward_m,
        "drop_height_above_measured_pelvis_m": drop_height_m, "release": release_info,
        "close_s": close_s, "first_palm_tip_force_s": first_contact_s,
        "true_pre_contact_pelvis_speed_mps": recent_speed,
        "camera_estimated_pre_contact_pelvis_displacement_m": estimated_pre_contact_displacement,
        "camera_recent_pre_contact_speed_mps": camera_recent_speed,
        "pretrigger_ball_detections": detections,
        "checkpoint_contact": checkpoint_audit, "final_contact": final,
        "endpoint_error_m": float(endpoint_errors[-1]), "final_mean_speed_mps": tail_speed,
        "minimum_upright_cosine": float(motion[:, 4].min()), "warning_count": warning_count,
        "full_horizon": full_horizon, "route_complete": route_pass,
        "moving_hands_only_catch": moving_catch,
        "moving_catch_and_route": bool(moving_catch and route_pass and final["hands_only_catch"]),
    }
    return result, trace, motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("codesign_source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-times", type=float, nargs="+", default=RELEASE_TIMES_S)
    parser.add_argument("--drop-forward", type=float, default=float(DROP_BODY_M[0]))
    parser.add_argument("--drop-height", type=float, default=float(DROP_BODY_M[2]))
    args = parser.parse_args()
    if not all(.8 <= release <= 2.5 for release in args.release_times) or not .40 <= args.drop_forward <= .55:
        parser.error("release times must be 0.8–2.5 s and forward placement 0.40–0.55 m")
    if not .45 <= args.drop_height <= .55:
        parser.error("drop-height must be 0.45–0.55 m above the measured pelvis")
    args.output.mkdir(parents=True, exist_ok=False)
    plant = HandRoutePlant(args.codesign_source, route_camera_position=ROUTE_CAMERA_M)
    report = {
        "evidence_class": "controlled_moving_release_development_simulation",
        "route": "go", "target_speed_mps": SPEED_MPS, "route_camera_m": ROUTE_CAMERA_M,
        "walk_start_s": WALK_START_S, "horizon_s": HORIZON_S,
        "close_height_above_release_pelvis_m": args.drop_height - (DROP_BODY_M[2] - CLOSE_ABOVE_RELEASE_PELVIS_M),
        "source_urdf_sha256": verify_source_urdf(args.codesign_source / UPSTREAM_URDF),
        "mujoco_version": mujoco.__version__, "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": [],
    }
    try:
        for index, release in enumerate(args.release_times):
            result, contact, motion = evaluate(plant, release, args.drop_forward, args.drop_height)
            trace_path = args.output / f"case-{index:02d}.npz"
            np.savez_compressed(trace_path, contact=contact, contact_columns=TRACE_COLUMNS,
                                motion=motion, motion_columns=("timestamp_s", "pelvis_x_m", "pelvis_y_m", "pelvis_z_m", "upright_cosine"))
            result["trace_sha256"] = sha256(trace_path.read_bytes()).hexdigest()
            report["results"].append(result)
            print(release, result["close_s"], result["true_pre_contact_pelvis_speed_mps"],
                  result["moving_hands_only_catch"], result["moving_catch_and_route"], flush=True)
    finally:
        plant.close()
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
