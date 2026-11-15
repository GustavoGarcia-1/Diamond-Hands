"""Run a sensor-guided receiver with a freely launched football and compact hands."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from evaluate_dextra_in_stride import RearMarkerLocalizer
from g1_receiver._vendor.g1_runtime.pool.contracts import ActuatorCommand
from g1_receiver.aero_reach import AeroReach
from g1_receiver.dextra_physical_eval import DextraContactAudit, TRACE_COLUMNS
from g1_receiver.hand_route_plant import HandRoutePlant
from g1_receiver.in_stride_eval import MOTION_COLUMNS, score_in_stride_catch
from g1_receiver.locomotion import CONTROL_DT, SURVEY_CAMERA_FOV_Y_DEG
from g1_receiver.observable import ARM_JOINTS
from g1_receiver.pass_control import FieldPassGrip, PassGrip, VisualPassGrip, observed_velocity
from g1_receiver.routes import RouteFollower, make_route
from g1_receiver.run_braking import RunningBrake, blend_pd
from g1_receiver.run_arm_receive import RunArmReceive
from g1_receiver.scrimmage_throw import FixedQuarterback, ScrimmagePass
from g1_receiver.running_catch_eval import FootContactRecorder, score_running_long_pass
from g1_receiver.runtime import RobotObservation, SkillGoal
from g1_receiver.short_throw import FreeBallRelease, plan_underhand_throw
from g1_receiver.throw_eval import score_throw_receive


@dataclass(frozen=True)
class Play:
    route: str = "go"
    scale: float = 1.0
    speed: float = 1.25
    finish_s: float = 9.
    release_s: float | None = 2.0
    flight_s: float = .38
    aim_x: float = .42
    aim_y: float = 0.
    aim_z: float = .37
    approach_mps: float = 2.5
    descent_mps: float = .50
    closure: float = .8
    squeeze: float = .025
    settle_s: float = .40
    acceleration: float = 2.
    shoulder_pitch_adjustment: float = 0.
    elbow_adjustment: float = 0.
    upper_fraction: float = 1.
    lateral_gain: float = .40
    yaw_gain: float = .75
    lateral_trim: float = 0.
    yaw_trim: float = 0.
    imu_stem_gain: float = 0.
    imu_stem_limit: float = .35
    imu_velocity_heading_gain: float = 0.
    pregrip: float = 0.
    terminal_blend: float = 0.
    terminal_radius_m: float = .75
    allow_downfield_lob: bool = False
    receive_wrist_pitch: float = 0.
    receive_wrist_roll: float = 0.
    receive_yaw_bias: float = 0.
    receive_wrist_pitch_left: float = 0.
    grip_retract_m: float = 0.
    torso_yaw_hold: float = 0.
    receive_elbow: float = 0.
    receive_shoulder_pitch: float = 0.
    arm_lead: bool = False
    arm_pump: bool = False
    reach_adjustment: float = 0.
    stabilize_palms: float = 0.
    stabilize_grasp: float = 0.
    stabilize_grasp_delay_s: float = .18
    follow_ball_gain: float = 0.
    delivery: str = "lob"
    runup: str = "route"
    launch_xyz: tuple[float, float, float] | None = None
    launch_velocity: tuple[float, float, float] | None = None
    quarterback_origin_m: tuple[float, float, float] | None = None
    quarterback_target_m: tuple[float, float, float] | None = None
    quarterback_level_ball: bool = False
    field_catch_window_m: tuple[float, float, float] | None = None
    grasp_offsets_rad: tuple[float, ...] | None = None
    grasp_ramp_s: float = .15
    grasp_joint_lead_s: float = .025
    grip_actuation_delay_s: float = 0.
    hold_offsets_rad: tuple[float, ...] | None = None
    hold_curls: tuple[float, float] | None = None
    hold_delay_s: float = .20
    hold_ramp_s: float = .18
    brake_x_m: float = 0.
    brake_timeout_s: float = 4.44
    brake_blend_s: float = .65
    brake_absorb_mps: float = .75
    brake_absorb_s: float = .65
    brake_ramp_down_s: float = .80
    brake_terminal_delay_s: float = 0.
    brake_no_terminal_correction: bool = False
    brake_predict_progress: bool = False
    grip: PassGrip = PassGrip()


def fixed_camera(scene, name, position, target):
    position = np.asarray(position, dtype=float)
    forward = np.asarray(target, dtype=float) - position
    if not np.isfinite(np.r_[position, forward]).all() or np.linalg.norm(forward[:2]) <= 1e-7:
        raise ValueError("a fixed camera needs finite, distinct horizontal position and aim")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0., 0., 1.))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    ET.SubElement(
        scene.find("worldbody"), "camera", name=name, pos=" ".join(map(str, position)),
        xyaxes=" ".join(map(str, (*right, *-down))), fovy=str(SURVEY_CAMERA_FOV_Y_DEG),
    )
    return np.column_stack((right, down, forward)), position


def make_plant(source, route, *, pass_camera=None, tracking_camera=None, finish_camera=None,
               retention_camera=None, retention_pose_camera=None, extra_cameras=None):
    pose = {}
    end_x = max(point[0] for point in route.waypoints)

    def add_cameras(scene):
        center = (end_x / 2, 0., .92)
        backdrop = scene.find("./worldbody/geom[@name='backdrop']")
        if backdrop is not None:
            backdrop.set("pos", f"{max(14., end_x + 4.)} 0 2")
        start_view = tracking_camera or ((-2.1, -.12, 1.28), center)
        end_view = finish_camera or ((max(0., end_x - 3.1), .10, 1.3), (end_x - .15, .05, .86))
        pose["track"] = fixed_camera(scene, "track", *start_view)
        pose["finish"] = fixed_camera(scene, "finish", *end_view)
        position, target = pass_camera or ((max(4., end_x + .6), -2.1, 1.65), center)
        pose["pass"] = fixed_camera(scene, "pass", position, target)
        if retention_camera is not None:
            pose["retention"] = fixed_camera(scene, "retention", *retention_camera)
        if retention_pose_camera is not None:
            pose["retention_pose"] = fixed_camera(scene, "retention_pose", *retention_pose_camera)
        for name, camera in (extra_cameras or {}).items():
            if name in pose or not name.replace('_', '').isalnum():
                raise ValueError('extra camera names must be unique alphanumeric labels')
            coordinates = np.asarray(camera, dtype=float)
            if coordinates.shape != (2, 3) or not np.isfinite(coordinates).all() or np.linalg.norm(coordinates[1]-coordinates[0]) < 1e-6:
                raise ValueError('extra camera requires finite distinct position and aim')
            pose[name] = fixed_camera(scene, name, *coordinates)

    plant = HandRoutePlant(source, hand_model="aero", rear_markers=True, scene_setup=add_cameras)
    plant.cameras.update(pose)
    return plant


def run_play(plant, policy, play, route, *, output=None, record=False, stadium=False,
             record_poses=False, residual_controller=None):
    from g1_receiver.aero import FINGERTIP_GEOMS, PALM_BODIES, aero_ready_joint_targets
    from g1_receiver.short_throw import UnderhandThrow

    nominal = policy.nominal
    upper = aero_ready_joint_targets(nominal, plant.joint_names)
    pose = None
    positions = deque(maxlen=7)
    velocity = None
    drive = np.zeros(3)
    follower = RouteFollower(route, lookahead_m=1.15, arrival_tolerance_m=.035)
    hands = (FieldPassGrip(play.grip, play.field_catch_window_m, allow_synthetic_fixture=True)
             if play.field_catch_window_m is not None else
             VisualPassGrip(play.grip, allow_synthetic_fixture=True))
    tracker = RearMarkerLocalizer(allow_synthetic_fixture=True, max_range_m=8.)
    shoulder = [plant.joint_names.index(f"{side}_shoulder_roll_joint") for side in ("left", "right")]
    arm_joints = tuple(f"{side}_{axis}_joint" for side in ("left", "right") for axis in (
        "shoulder_roll", "shoulder_yaw", "shoulder_pitch", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"))
    if set(arm_joints) != set(ARM_JOINTS):
        raise ValueError("grasp adjustments must address the fourteen arm joints")
    arms = [plant.joint_names.index(name) for name in arm_joints]
    wrist_pitch = [plant.joint_names.index(f"{side}_wrist_pitch_joint") for side in ("left", "right")]
    wrist_roll = [plant.joint_names.index(f"{side}_wrist_roll_joint") for side in ("left", "right")]
    shoulder_yaw = [plant.joint_names.index(f"{side}_shoulder_yaw_joint") for side in ("left", "right")]
    elbow = [plant.joint_names.index(f"{side}_elbow_joint") for side in ("left", "right")]
    shoulder_pitch = [plant.joint_names.index(f"{side}_shoulder_pitch_joint") for side in ("left", "right")]
    waist_yaw = plant.joint_names.index("waist_yaw_joint")
    for pair, offset in (("shoulder_pitch", play.shoulder_pitch_adjustment), ("elbow", play.elbow_adjustment)):
        indices = [plant.joint_names.index(f"{side}_{pair}_joint") for side in ("left", "right")]
        upper[indices] += offset
    reach = AeroReach(plant.model, plant.joint_names, upper) if play.stabilize_palms else None
    grasp_reach = AeroReach(plant.model, plant.joint_names, upper) if play.stabilize_grasp else None
    is_running_policy = hasattr(policy, "arm_indices")
    braking = RunningBrake(play.brake_x_m, play.brake_absorb_mps, play.brake_blend_s,
                           play.brake_absorb_s, play.brake_ramp_down_s,
                           play.brake_terminal_delay_s, not play.brake_no_terminal_correction) if play.brake_x_m else None
    if braking is not None and not is_running_policy:
        raise ValueError("the running brake requires an AMP policy")
    initial = upper if not is_running_policy else nominal + play.upper_fraction * (upper - nominal)
    arm_profile = RunArmReceive() if play.arm_pump else None
    if arm_profile is not None:
        if not is_running_policy:
            raise ValueError("the arm swing profile requires the AMP running policy")
        initial = arm_profile.initial(nominal, upper, plant.joint_names)
    quarterback = None
    reset_options = {}
    if play.quarterback_origin_m is not None:
        if np.linalg.norm(np.asarray(play.quarterback_origin_m[:2])) < .85:
            raise ValueError("line the quarterback up at least .85m from the receiver's starting position")
        reset_options["release_position_m"] = play.quarterback_origin_m
        quarterback = FixedQuarterback(plant.model, plant.data, ScrimmagePass(
            play.quarterback_origin_m, play.quarterback_target_m, play.flight_s,
            align_ball_to_launch=not play.quarterback_level_ball))
    plant.reset(joint_positions=np.clip(initial, *plant.joint_ranges.T), closure=play.pregrip,
                **reset_options)
    if quarterback is not None:
        quarterback.ready()
    policy.reset()
    contact = DextraContactAudit(plant.model, palms=PALM_BODIES, fingertip_geoms=FINGERTIP_GEOMS)
    feet = FootContactRecorder(plant.model)
    thrower = FreeBallRelease(plant.model, plant.data)
    release = None
    motion, readings = [], []
    localized_frames = 0
    body = plant.data.body("pelvis")
    final_point = np.asarray(route.waypoints[-1])
    final_leg = final_point - np.asarray(route.waypoints[-2])
    initial_leg = np.subtract(route.waypoints[1], route.waypoints[0])
    forward = initial_leg / np.linalg.norm(initial_leg)
    final_direction = final_leg / np.linalg.norm(final_leg)
    transition = 0.
    timestamps, images, spectator_states = [], [], []
    reach_trim = np.zeros(2)
    spectator = mujoco.MjvCamera()
    spectator.azimuth = 123.
    spectator.elevation = -8.
    spectator.distance = 3.1
    viewer = None
    if record:
        if stadium:
            from g1_receiver.stadium import StadiumRenderer

            heading = float(np.arctan2(forward[1], forward[0]))
            spectator.azimuth = 109. + np.degrees(heading)
            spectator.elevation = -10.
            spectator.distance = 3.2
            viewer = StadiumRenderer(plant, height=448, width=800,
                                     field_heading_rad=heading)
        else:
            plant.model.vis.global_.offwidth = 800
            plant.model.vis.global_.offheight = 448
            viewer = mujoco.Renderer(plant.model, height=448, width=800)

    def audit(data):
        feet.record(data)
        if release is not None:
            contact.record(data)

    planned_ticks = round(play.finish_s / CONTROL_DT)
    residual_frame = residual_survey = None
    residual_pose = None
    residual_tracker = RearMarkerLocalizer(allow_synthetic_fixture=True, max_range_m=8.)
    try:
        for tick in range(planned_ticks + 1):
            now = plant.timestamp_s
            observations = RobotObservation(now, *plant.observe(), np.zeros(29), None)
            if (quarterback is not None and play.release_s is not None and release is None
                    and now >= play.release_s - 1e-8):
                release = asdict(quarterback.release(now))
            if (play.launch_xyz is not None and play.release_s is not None
                    and release is None and now >= play.release_s - 1e-8):
                launch = UnderhandThrow(now, play.flight_s, play.launch_xyz, play.launch_velocity,
                                        (1., 0., 0., 0.), play.launch_xyz, play.launch_velocity, 0.)
                thrower.apply(launch)
                release = asdict(launch)
            if tick % 2 == 0:
                use_finish = pose is not None and (
                    pose.xy_m[0] >= braking.brake_x_m + 1.15 if braking is not None else
                    (final_point - pose.xy_m) @ final_direction < 1.8)
                camera = "finish" if use_finish else "track"
                reading = tracker.estimate(*plant.capture(camera))
                if reading is None:
                    other = "track" if use_finish else "finish"
                    reading = tracker.estimate(*plant.capture(other))
                if reading is not None:
                    pose = reading
                    localized_frames += 1
                    positions.append((now, *pose.xy_m))
                    velocity = observed_velocity(positions)
                throw_ready = (quarterback is None and play.launch_xyz is None
                               and reading is not None and velocity is not None)
                if play.release_s is not None and release is None and now >= play.release_s - 1e-8 and throw_ready:
                    if play.delivery == "lob":
                        launch = plan_underhand_throw(
                            pose, velocity, release_time_s=now, flight_time_s=play.flight_s,
                            intercept_body_xyz_m=(play.aim_x, play.aim_y, play.aim_z),
                            incoming_relative_body_mps=(-play.approach_mps, 0., -play.descent_mps),
                            allow_downfield_lob=play.allow_downfield_lob,
                        )
                    else:
                        heading = np.array([np.cos(pose.yaw_rad), np.sin(pose.yaw_rad)])
                        left = np.array([-heading[1], heading[0]])
                        origin = np.r_[pose.xy_m + play.aim_x * heading + play.aim_y * left,
                                       pose.z_m + play.aim_z]
                        speed = np.r_[velocity - play.approach_mps * heading, -play.descent_mps]
                        yaw = .5 * pose.yaw_rad
                        launch = UnderhandThrow(now, 0., tuple(origin), tuple(speed),
                                                (float(np.cos(yaw)), 0., 0., float(np.sin(yaw))),
                                                tuple(origin), tuple(speed), 0.)
                    thrower.apply(launch)
                    release = asdict(launch)
                if viewer is not None:
                    # Spectator framing is recorded separately from sensor inputs.
                    spectator.lookat[:] = (*body.xpos[:2] + np.array([.38, 0.]), .78)
                    viewer.update_scene(plant.data, camera=spectator)
                    images.append(viewer.render())
                if viewer is not None or record_poses:
                    timestamps.append(now)
                    spectator_states.append(plant.data.qpos.copy())
            if play.release_s is not None and hands.closed_at is None and (
                    quarterback is None or release is not None):
                hands.update(*plant.capture("pass"), pose, velocity, now)
            elif (play.follow_ball_gain and hands.closed_at is not None
                  and now <= hands.closed_at + .55):
                hands.observe_after_close(*plant.capture("pass"), pose, now)
            motion.append((now, *body.xpos, float(body.xmat[8])))
            _, imu_x, imu_y, _ = observations.imu_quat_wxyz
            if tick == planned_ticks or 1. - 2. * (imu_x ** 2 + imu_y ** 2) < .54:
                break

            target = np.zeros(3)
            if now >= play.settle_s - 1e-8 and pose is not None and pose.fresh(now, .12):
                desired = np.asarray(follower.command(pose.xy_m))
                heading = np.array([np.cos(pose.yaw_rad), np.sin(pose.yaw_rad)])
                if is_running_policy and np.linalg.norm(desired) > 0:
                    local = np.vstack((heading, [-heading[1], heading[0]])) @ desired
                    activity = min(1., np.linalg.norm(desired) / .65)
                    target[:2] = (max(0., local[0]), np.clip(play.lateral_gain * local[1], -.40, .40)
                                   + play.lateral_trim * activity)
                    target[2] = np.clip(-play.yaw_gain * pose.yaw_rad + play.yaw_trim * activity, -.65, .65)
                else:
                    target[:2] = np.vstack((heading, [-heading[1], heading[0]])) @ desired
                    target[2] = np.clip(-play.yaw_gain * pose.yaw_rad, -.65, .65)
                if is_running_policy and play.terminal_blend and follower.next_waypoint >= len(route.waypoints) - 1:
                    remaining = (final_point - pose.xy_m) @ final_direction
                    transition = max(transition, np.clip(
                        (play.terminal_radius_m - remaining) / (play.terminal_radius_m * .80), 0., 1.))
                    terminal = transition * play.terminal_blend
                    local = np.vstack((heading, [-heading[1], heading[0]])) @ desired
                    target[:2] = (1. - terminal) * target[:2] + terminal * np.clip(local, -.30, .30)
                    target[2] *= 1. - terminal
            terminal = transition * play.terminal_blend
            if play.runup == "imu" and terminal == 0.:
                w, x, y, z = observations.imu_quat_wxyz
                yaw = np.arctan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z))
                stem_heading = 0.
                if play.imu_stem_gain and pose is not None and pose.fresh(now, .12):
                    stem_heading = np.clip(-play.imu_stem_gain * pose.xy_m[1],
                                           -play.imu_stem_limit, play.imu_stem_limit)
                if (play.imu_velocity_heading_gain and pose is not None and pose.fresh(now, .12)
                        and velocity is not None and np.linalg.norm(velocity) >= .4):
                    travel_heading = np.arctan2(velocity[1], velocity[0])
                    stem_heading = yaw - play.imu_velocity_heading_gain * travel_heading + stem_heading
                speed = min(play.speed, play.acceleration * max(0., now - play.settle_s))
                drive[:] = (speed, play.lateral_trim * speed / play.speed,
                            np.clip(play.yaw_gain * (stem_heading - yaw), -.75, .75))
            else:
                drive += np.clip(target - drive, -play.acceleration * CONTROL_DT, play.acceleration * CONTROL_DT)
            readings.append((now, *(pose.xy_m if pose is not None else [np.nan, np.nan]), *drive))
            catch_target = upper.copy()
            if play.reach_adjustment and hands.crossing_yz is not None:
                y, z = hands.crossing_yz
                offset = np.array([y / .29, (play.grip.height_m - z) / .39])
                offset = np.clip(offset * play.reach_adjustment, [-.20, -.17], [.20, .17])
                reach_trim += np.clip(offset - reach_trim, -.035, .035)
                catch_target[shoulder_yaw] += reach_trim[0]
                catch_target[shoulder_pitch] += reach_trim[1]
            if reach is not None and hands.ready_at is not None:
                reach_blend = play.stabilize_palms * np.clip((now - hands.ready_at) / .10, 0., 1.)
                offset = np.r_[0., reach_trim[0] * .29, -reach_trim[1] * .39]
                if play.follow_ball_gain and hands.measured_hand_offset is not None:
                    residual = np.clip(hands.measured_hand_offset * play.follow_ball_gain, -.09, .09)
                    offset = .4 * offset + .6 * residual
                if hands.closed_at is not None:
                    elapsed = now - hands.closed_at
                    offset[0] += play.grip_retract_m * (1. - 2. * np.clip((elapsed - .17) / .24, 0., 1.))
                catch_target = reach.target(observations, upper, blend=reach_blend, xyz_offset=offset)
            closure = play.pregrip
            if hands.closed_at is not None:
                elapsed = now - hands.closed_at - play.grip_actuation_delay_s
                reinforcement = np.clip((elapsed - .12) / .30, 0., 1.)
                if play.grasp_offsets_rad is not None:
                    amount = np.clip((elapsed + play.grasp_joint_lead_s) / play.grasp_ramp_s, 0., 1.)
                    catch_target[arms] += amount * np.asarray(play.grasp_offsets_rad)
                catch_target[shoulder] -= np.array([1., -1.]) * play.squeeze * reinforcement
                catch_target[wrist_pitch] += play.receive_wrist_pitch * reinforcement
                wrist_ramp = np.clip((elapsed - .04) / .18, 0., 1.)
                catch_target[wrist_roll] += np.array([1., -1.]) * play.receive_wrist_roll * wrist_ramp
                catch_target[shoulder_yaw[0]] += play.receive_yaw_bias * wrist_ramp
                catch_target[wrist_pitch[0]] += play.receive_wrist_pitch_left * wrist_ramp
                catch_target[elbow] += play.receive_elbow * reinforcement
                catch_target[shoulder_pitch] += play.receive_shoulder_pitch * reinforcement
                if elapsed >= -1e-8:
                    closure = play.closure
                hold = np.clip((elapsed - play.hold_delay_s) / play.hold_ramp_s, 0., 1.)
                if hold > 0.:
                    if play.hold_offsets_rad is not None:
                        catch_target[arms] += hold * np.asarray(play.hold_offsets_rad)
                    if play.hold_curls is not None:
                        closure = play.closure + hold * (np.asarray(play.hold_curls) - play.closure)
            if grasp_reach is not None and hands.closed_at is not None:
                fraction = play.stabilize_grasp * np.clip(
                    (now - hands.closed_at - play.stabilize_grasp_delay_s) / .12, 0., 1.)
                catch_target = grasp_reach.target(observations, catch_target, blend=fraction,
                                                  reference_from_ready=True)
            if residual_controller is not None:
                if tick % 2 == 0:
                    camera = "retention" if "retention" in plant.cameras and hands.closed_at is not None else "pass"
                    residual_frame, residual_survey = plant.capture(camera)
                    measured_residual_pose = residual_tracker.estimate(residual_frame, residual_survey)
                    if measured_residual_pose is None and "retention_pose" in plant.cameras and camera == "retention":
                        measured_residual_pose = residual_tracker.estimate(*plant.capture("retention_pose"))
                    candidates = [p for p in (measured_residual_pose, residual_pose, pose)
                                  if p is not None and p.z_m is not None and p.fresh(now, .08 + 1e-8)]
                    residual_pose = max(candidates, key=lambda p: p.timestamp_s) if candidates else None
                adjusted, curls = residual_controller(
                    observations, residual_pose, residual_frame, residual_survey, catch_target.copy(),
                    np.broadcast_to(closure, (2,)).copy(), hands.ready_at is not None)
                adjusted, curls = np.asarray(adjusted), np.asarray(curls)
                if (adjusted.shape != (29,) or curls.shape != (2,)
                        or not np.isfinite(adjusted).all() or not np.isfinite(curls).all()
                        or np.any(curls < 0.) or np.any(curls > 1.)
                        or not np.allclose(np.delete(adjusted, arms), np.delete(catch_target, arms))
                        or np.max(np.abs(adjusted[arms] - catch_target[arms])) > .200001):
                    raise ValueError("grasp residual must only adjust bounded arm targets and hand closure")
                catch_target[arms] = np.clip(adjusted[arms], *plant.joint_ranges[arms].T)
                closure = curls
            if is_running_policy:
                reach_at = hands.ready_at if play.arm_lead else hands.closed_at
                fraction = play.upper_fraction if reach_at is None else (
                    play.upper_fraction + (1. - play.upper_fraction) * np.clip((now - reach_at) / .12, 0., 1.))
                if arm_profile is not None:
                    catch_target, fraction = arm_profile.targets(
                        catch_target, plant.joint_names, observations.joint_position, now, hands.ready_at)
                run_drive = drive
                if braking is not None:
                    w, x, y, z = observations.imu_quat_wxyz
                    yaw = np.arctan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z))
                    speed_scale, brake_fraction, walk_drive = braking.commands(
                        now, pose, velocity, yaw, final_point, timeout_s=play.brake_timeout_s,
                        predict_progress=play.brake_predict_progress)
                    run_drive = drive * np.array([speed_scale, 1., 1.])
                command = policy.act(observations, SkillGoal(base_velocity=run_drive), upper_target=catch_target,
                                     arm_override_fraction=fraction)
                if play.torso_yaw_hold:
                    targets = command.target_position.copy()
                    targets[waist_yaw] *= 1. - play.torso_yaw_hold
                    command = replace(command, target_position=targets)
                if braking is not None:
                    walk = plant.wbc.act(observations, SkillGoal(base_velocity=walk_drive), upper_target=catch_target)
                    command = blend_pd(command, walk, brake_fraction)
                elif play.terminal_blend:
                    walk = plant.wbc.act(observations, SkillGoal(base_velocity=drive), upper_target=catch_target)
                    if terminal:
                        command = ActuatorCommand(
                            (1. - terminal) * command.target_position + terminal * walk.target_position,
                            (1. - terminal) * command.kp + terminal * walk.kp,
                            (1. - terminal) * command.kd + terminal * walk.kd,
                            (1. - terminal) * command.feedforward_torque + terminal * walk.feedforward_torque,
                            "running_to_walking",
                        )
            else:
                command = policy.act(observations, SkillGoal(base_velocity=drive), upper_target=catch_target)
            if reach is not None:
                limited = np.clip(command.target_position, *plant.joint_ranges.T)
                command = replace(command, target_position=limited)
            plant.step(command, closure=closure, post_step=audit)
    finally:
        if quarterback is not None:
            quarterback.close()
        if viewer is not None:
            viewer.close()

    motion = np.asarray(motion)
    trace = contact.trace()
    foot_trace = feet.arrays()
    release_time = None if release is None else release["release_time_s"]
    warning_count = sum(int(w.number) for w in plant.data.warning)
    scored = score_in_stride_catch(
        trace, motion, route, release_time_s=release_time, finish_time_s=planned_ticks * CONTROL_DT,
        ball_weight_n=contact.ball_weight_n, verified_free_ball=True,
        physics_step_s=float(plant.model.opt.timestep), simulator_warning_count=warning_count,
    )
    running = score_running_long_pass(
        trace, motion, **foot_trace, strict_catch_score=scored, release_time_s=release_time,
        forward_xy=forward, gait_start_s=play.settle_s,
        pass_toward_receiver_xy=-forward if play.allow_downfield_lob else None,
    )
    result = {
        "evidence_class": "aero_receiver_physics_simulation",
        "play": asdict(play), "waypoints_m": route.waypoints, "release": release,
        "camera_surveys": {name: {"position_m": position.tolist(), "rotation": axes.tolist()}
                           for name, (axes, position) in plant.cameras.items()},
        "ball_weight_n": contact.ball_weight_n, "physics_step_s": float(plant.model.opt.timestep),
        "warning_count": warning_count, "actual_finish_s": float(motion[-1, 0]),
        "fixed_quarterback": quarterback is not None,
        "run_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "visual_reach_s": hands.ready_at, "visual_closure_s": hands.closed_at, "observed_ball": hands.trace,
        "finger_actuation_delay_s": play.grip_actuation_delay_s,
        "brake_at_s": None if braking is None else braking.brake_at_s,
        "brake_trigger": None if braking is None else braking.trigger,
        "localized_frames": localized_frames,
        "strict_score": scored, "running_score": running,
    }
    if quarterback is not None:
        result["quarterback_pass_score"] = score_throw_receive(
            trace, motion, release, scored["first_tip_contact_s"],
            physics_step_s=float(plant.model.opt.timestep))
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        trace_path = output / "trace.npz"
        np.savez_compressed(trace_path, motion=motion, motion_columns=MOTION_COLUMNS, contact=trace,
                            contact_columns=TRACE_COLUMNS, route_camera_and_drive=readings, **foot_trace)
        result["trace_sha256"] = sha256(trace_path.read_bytes()).hexdigest()
        (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if images:
            np.savez_compressed(output / "view.npz", rgb=images, timestamps_s=timestamps,
                                qpos=spectator_states)
        elif spectator_states:
            np.savez_compressed(output / "view.npz", timestamps_s=timestamps, qpos=spectator_states)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amp-source", type=Path)
    parser.add_argument("--route", choices=("go", "slant", "out", "hitch", "post"), default="go")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--stadium", action="store_true", help="record with a high-school spectator scene")
    parser.add_argument("--no-pass", action="store_true")
    parser.add_argument("--allow-downfield-lob", action="store_true")
    parser.add_argument("--arm-lead", action="store_true")
    parser.add_argument("--arm-pump", action="store_true")
    parser.add_argument("--brake-no-terminal-correction", action="store_true")
    parser.add_argument("--brake-predict-progress", action="store_true")
    parser.add_argument("--delivery", choices=("lob", "short"), default="lob")
    parser.add_argument("--runup", choices=("route", "imu"), default="route")
    parser.add_argument("--launch-xyz", type=float, nargs=3)
    parser.add_argument("--launch-velocity", type=float, nargs=3)
    parser.add_argument("--quarterback-origin-m", type=float, nargs=3)
    parser.add_argument("--quarterback-target-m", type=float, nargs=3)
    parser.add_argument("--quarterback-level-ball", action="store_true")
    parser.add_argument("--field-catch-window-m", type=float, nargs=3)
    parser.add_argument("--grasp-offsets-rad", type=float, nargs=14)
    parser.add_argument("--hold-offsets-rad", type=float, nargs=14)
    parser.add_argument("--hold-curls", type=float, nargs=2)
    for name in ("pass-camera", "tracking-camera", "finish-camera"):
        parser.add_argument("--" + name, type=float, nargs=6, metavar=("X", "Y", "Z", "AIM_X", "AIM_Y", "AIM_Z"))
    parser.add_argument("--grip-mode", choices=("predict", "descending"), default="predict")
    parser.add_argument("--grip-occlusion", action="store_true")
    parser.add_argument("--grip-crossing-direction", type=int, choices=(-1, 1), default=-1)
    grip = PassGrip()
    for name in ("plane_m", "lead_s", "height_m", "lateral_limit_m"):
        parser.add_argument("--grip-" + name.replace("_", "-"), type=float, default=getattr(grip, name))
    for name, value in vars(Play()).items():
        if isinstance(value, float):
            parser.add_argument("--" + name.replace("_", "-"), type=float, default=value)
    parser.add_argument("--endpoint-xy", type=float, nargs=2,
                        help="optional surveyed endpoint for a straight go route")
    args = parser.parse_args()
    if args.stadium and not args.record:
        parser.error("--stadium requires --record")
    if bool(args.launch_xyz) != bool(args.launch_velocity):
        parser.error("set both launch position and velocity for a fixed field launch")
    if bool(args.quarterback_origin_m) != bool(args.quarterback_target_m):
        parser.error("set both the quarterback release point and the downfield target")
    if args.quarterback_origin_m and (args.no_pass or args.launch_xyz is not None):
        parser.error("a quarterback pass is its own delivery configuration")
    if args.quarterback_origin_m:
        if np.linalg.norm(args.quarterback_origin_m[:2]) < .85:
            parser.error("line the quarterback up away from the receiver's starting position")
        ScrimmagePass(args.quarterback_origin_m, args.quarterback_target_m, args.flight_s).plan(args.release_s)
    settings = {name: getattr(args, name) for name in vars(Play()) if name != "grip"}
    if args.no_pass:
        settings["release_s"] = None
    grip = PassGrip(
        args.grip_plane_m, args.grip_lead_s, args.grip_height_m, args.grip_mode,
        args.grip_lateral_limit_m, args.grip_occlusion,
        args.grip_crossing_direction,
    )
    play = Play(**settings, grip=grip)
    if (not .10 <= play.speed <= 2.5 or not 1.0 <= play.finish_s <= 22.
            or not 0. <= play.pregrip <= 1. or not 0. <= play.closure <= 1.
            or not 0. <= play.upper_fraction <= 1. or not 0. <= play.terminal_blend <= 1.
            or not .15 <= play.terminal_radius_m <= 4. or play.acceleration <= 0.
            or not 0. <= play.torso_yaw_hold <= 1.
            or not 0. <= play.stabilize_palms <= 1. or not 0. <= play.reach_adjustment <= 1.5
            or not 0. <= play.stabilize_grasp <= 1. or not 0. <= play.stabilize_grasp_delay_s <= 1.):
        parser.error("invalid play duration, pace, or bounded controller setting")
    if play.release_s is not None and not 0 <= play.release_s < play.finish_s:
        parser.error("release time must be within the trial")
    max_grasp_offset = .75 if play.quarterback_origin_m is not None else .45
    if (not .06 <= play.grasp_ramp_s <= .4 or not 0. <= play.grasp_joint_lead_s <= .1
            or not 0. <= play.grip_actuation_delay_s <= .15) or (
            play.grasp_offsets_rad is not None and
            not all(abs(value) <= max_grasp_offset for value in play.grasp_offsets_rad)):
        parser.error("grasp trajectory is outside the validated joint offset bounds")
    if (not .06 <= play.hold_ramp_s <= .4 or not 0. <= play.hold_delay_s <= 1.
            or (play.hold_offsets_rad is not None and not all(-.45 <= value <= .45 for value in play.hold_offsets_rad))
            or (play.hold_curls is not None and not all(0. <= value <= 1. for value in play.hold_curls))):
        parser.error("hold trajectory is outside the validated joint and hand command bounds")
    if play.brake_x_m and (
            args.amp_source is None or play.runup != "imu" or play.terminal_blend or play.route != "go"
            or not 1. <= play.brake_x_m <= 30. or not .2 <= play.brake_blend_s <= 1.5
            or not .1 <= play.brake_absorb_mps <= 1. or not .2 <= play.brake_absorb_s <= 2.
            or not .2 <= play.brake_ramp_down_s <= 2. or not 1. <= play.brake_timeout_s < play.finish_s
            or not 0. <= play.brake_terminal_delay_s <= 3.):
        parser.error("sensor braking requires a straight AMP run and a bounded handoff")
    route = make_route(play.route, scale=play.scale, speed_mps=play.speed)
    if args.endpoint_xy is not None:
        if play.route != "go":
            parser.error("a custom endpoint requires a straight go route")
        route = replace(route, waypoints=(route.waypoints[0], tuple(args.endpoint_xy)))
    cameras = {}
    for name in ("pass_camera", "tracking_camera", "finish_camera"):
        spec = getattr(args, name)
        if spec is not None:
            cameras[name] = (spec[:3], spec[3:])
    plant = make_plant(args.source, route, **cameras)
    try:
        policy = plant.wbc
        if args.amp_source:
            from g1_receiver.running_policy import AmpRunningPolicy
            policy = AmpRunningPolicy(plant.joint_names, plant.joint_ranges, args.amp_source)
        result = run_play(plant, policy, play, route, output=args.output, record=args.record, stadium=args.stadium)
    finally:
        plant.close()
    score = result["strict_score"]
    running = result["running_score"]
    print(json.dumps({
        "finish_s": result["actual_finish_s"], "visual_closure_s": result["visual_closure_s"],
        "catch_and_route": score["in_stride_catch_and_route"], "failed_checks": score["failed_checks"],
        "endpoint_error_m": score["endpoint_error_m"], "running_long_pass_and_route": running["running_long_pass_and_route"],
        "ball_toward_receiver_travel_m": running["ball_toward_receiver_travel_m"],
    }))


if __name__ == "__main__":
    main()
