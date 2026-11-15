"""Evaluate a five-yard camera-triggered Out route and a surveyed throw."""

import argparse
from dataclasses import asdict, replace
from copy import deepcopy

import mujoco
import json
from pathlib import Path

import numpy as np

from evaluate_aero_receiver import fixed_camera
from evaluate_dextra_in_stride import RearMarkerLocalizer
from probe_aero_amp_slant import OpticalPass
from g1_receiver.aero import aero_ready_joint_targets
from g1_receiver.football_routes import YARD_M, imu_yaw
from g1_receiver.hand_route_plant import HandRoutePlant
from g1_receiver.locomotion import CONTROL_DT
from g1_receiver.in_stride_eval import score_in_stride_catch
from g1_receiver.out_course import OutCut, upper_lookback_fraction, interpolate_arm_path
from g1_receiver.pass_control import observed_velocity
from g1_receiver.run_braking import blend_pd
from g1_receiver.running_catch_eval import FootContactRecorder
from g1_receiver.running_policy import AmpRunningPolicy
from g1_receiver.routes import Route
from g1_receiver.runtime import RobotObservation, SkillGoal
from g1_receiver.throw_eval import score_throw_receive


def run(plant, policy, cut, *, duration=7.5, screening_break_s=None, screening_handoff_s=None,
        delivery=None, finish_outside_m=None, spectator_states=None, camera_start_s=1.8, finish_localization_fallback=None, finish_grip=None, upper_lookback_yaw_rad=None, staged_arm_offsets=None, carry_arm_path=None, carry_after_grip_s=.6, carry_ramp_s=1.2, residual_callback=None):
    upper = aero_ready_joint_targets(policy.nominal, plant.joint_names)
    arm_indices = [plant.joint_names.index(f"{side}_{axis}_joint")
                   for side in ("left", "right") for axis in
                   ("shoulder_roll", "shoulder_yaw", "shoulder_pitch", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
    if staged_arm_offsets is not None:
        staged_arm_offsets = np.asarray(staged_arm_offsets, dtype=float)
        if staged_arm_offsets.shape != (14,) or not np.isfinite(staged_arm_offsets).all():
            raise ValueError("staged interception requires14finitearmoffsets")
    if carry_arm_path is not None:
        carry_arm_path = np.asarray(carry_arm_path, dtype=float)
        interpolate_arm_path(carry_arm_path, 0.)
        if (not np.isfinite([carry_after_grip_s, carry_ramp_s]).all()
                or carry_after_grip_s < 0. or carry_ramp_s <= 0.
                or np.any(carry_arm_path < plant.joint_ranges[arm_indices, 0] - 1e-8)
                or np.any(carry_arm_path > plant.joint_ranges[arm_indices, 1] + 1e-8)):
            raise ValueError("carry path requires joint-limit-safe targets and positive timing")
    if isinstance(delivery, OpticalPass):
        plant.reset(joint_positions=upper,
                    release_position_m=delivery.initial_ball_position((8., 4., .085)))
        delivery.ready()
    else:
        plant.reset(joint_positions=upper)
    policy.reset()
    residual_pose = residual_frame = residual_survey = None
    if residual_callback is not None:
        if not callable(residual_callback):
            raise TypeError("residual callback must be callable")
        residual_tracker = RearMarkerLocalizer(allow_synthetic_fixture=True, max_range_m=8.)
        from g1_receiver.perception import ColorBallDetector
        residual_detector = ColorBallDetector(max_depth_m=20., ball_center_offset_m=.075)
        if hasattr(residual_callback, "bind"):
            residual_callback.bind(mujoco.MjModel.from_xml_string(plant.scene_xml), tuple(plant.joint_names))
    estimator = RearMarkerLocalizer(allow_synthetic_fixture=True, max_range_m=8.)
    feet = FootContactRecorder(plant.model)
    pelvis = plant.data.body("pelvis")
    pose = None
    rows = []
    poses = []
    finish = None
    fallback_samples = []
    upper_lookback_targets = []
    carry_arm_trace = []
    waist_index = plant.joint_names.index("waist_yaw_joint")

    def audit(data):
        feet.record(data)
        if delivery is not None and delivery.release is not None:
            delivery.audit.record(data)

    for tick in range(round(duration / CONTROL_DT) + 1):
        now = plant.timestamp_s
        obs = RobotObservation(now, *plant.observe(), np.zeros(29), None)
        yaw = imu_yaw(obs.imu_quat_wxyz)
        if screening_break_s is not None and cut.break_at_s is None and now >= screening_break_s - 1e-8:
            cut.break_at_s = now
        if screening_handoff_s is not None and cut.handoff_at_s is None and now >= screening_handoff_s - 1e-8:
            cut.handoff_at_s = now
        if (screening_break_s is None and tick % 2 == 0 and now >= camera_start_s
                and (cut.break_at_s is None or delivery is not None or finish_outside_m is not None
                     or cut.depth_feedback_gain)):
            cameras = ("outside", "break") if cut.break_at_s is not None else ("break",)
            measured = None
            for camera in cameras:
                measured = estimator.estimate(*plant.capture(camera))
                if measured is not None:
                    break
            # Opt-in surveyed finish recovery starts after the optical grip window.
            # It uses the same RGB-D marker estimator, never simulator pose/contact.
            if (measured is None and finish_localization_fallback is not None
                    and finish_outside_m is not None and delivery is not None
                    and delivery.hands.closed_at is not None
                    and now >= delivery.hands.closed_at + .7):
                measured = estimator.estimate(*plant.capture(finish_localization_fallback))
                if measured is not None:
                    fallback_samples.append((now, finish_localization_fallback, *measured.xy_m))
            if measured is not None:
                pose = measured
                poses = [*poses[-6:], (now, *pose.xy_m)]
        if delivery is not None:
            delivery.observe(plant, now, pose, poses)
        run_speed = min(2., 1.8 * max(0., now - .4))
        run_goal, walk_goal, fraction = cut.update(now, pose, yaw, run_speed,
                                                   field_velocity=observed_velocity(poses))
        if (finish_outside_m is not None and cut.break_at_s is not None and pose is not None
                and pose.fresh(now, .14) and pose.xy_m @ cut.outside >= finish_outside_m and finish is None):
            finish = now
        if finish is not None:
            walk_goal *= np.clip(1. - (now - finish) / .50, 0., 1.)
        rows.append((now, *pelvis.xpos, float(pelvis.xmat[8]), yaw, *run_goal, *walk_goal, fraction))
        if spectator_states is not None and tick % 2 == 0:
            spectator_states.append((now, plant.data.qpos.copy()))
        _, x, y, _ = obs.imu_quat_wxyz
        if tick == round(duration / CONTROL_DT) or 1. - 2. * (x*x + y*y) < .54:
            break
        prepared_upper = upper
        if staged_arm_offsets is not None and cut.break_at_s is not None:
            stage = float(np.clip((now - cut.break_at_s - .2) / .7, 0., 1.))
            stage = stage * stage * (3. - 2. * stage)
            prepared_upper = upper.copy()
            prepared_upper[arm_indices] += stage * staged_arm_offsets
        target, closure = (prepared_upper, 0.) if delivery is None else delivery.targets(now, prepared_upper)
        if (carry_arm_path is not None and delivery is not None and delivery.hands.closed_at is not None
                and now >= delivery.hands.closed_at + carry_after_grip_s):
            carry_fraction = float(np.clip((now - delivery.hands.closed_at - carry_after_grip_s) / carry_ramp_s, 0., 1.))
            target = target.copy()
            target[arm_indices] = interpolate_arm_path(carry_arm_path, carry_fraction)
            carry_arm_trace.append((now, carry_fraction))
        if finish_grip is not None and finish is not None and now >= finish + .15:
            grip_fraction = float(np.clip((now - finish - .15) / .30, 0., 1.))
            closure = ((1. - grip_fraction) * np.asarray(closure) + grip_fraction * finish_grip)
        if residual_callback is not None:
            if tick % 2 == 0:
                camera = "retention" if "retention" in plant.cameras and delivery is not None and delivery.hands.closed_at is not None else "pass"
                residual_frame, residual_survey = plant.capture(camera)
                if camera == "retention" and residual_detector.detect(residual_frame, residual_survey) is None:
                    residual_frame, residual_survey = plant.capture("pass")
                measured_residual_pose = residual_tracker.estimate(residual_frame, residual_survey)
                separate_pose = (residual_tracker.estimate(*plant.capture("retention_pose"))
                                 if "retention_pose" in plant.cameras else None)
                candidates = [p for p in (measured_residual_pose, separate_pose, residual_pose, pose)
                              if p is not None and p.z_m is not None and p.fresh(now, .08 + 1e-8)]
                if candidates:
                    latest = max(p.timestamp_s for p in candidates)
                    simultaneous = [p for p in candidates if p.timestamp_s >= latest - 1e-8]
                    residual_pose = min(simultaneous, key=lambda p: float(np.trace(np.asarray(p.covariance)[:2,:2])) + (np.inf if p.z_std_m is None else p.z_std_m**2))
                else:
                    residual_pose = None
            scripted = np.asarray(target).copy()
            scripted_closure = np.broadcast_to(np.asarray(closure, dtype=float), (2,)).copy()
            enabled = delivery is not None and delivery.hands.ready_at is not None
            adjusted, curls = residual_callback(deepcopy(obs), deepcopy(residual_pose),
                deepcopy(residual_frame), deepcopy(residual_survey), scripted.copy(), scripted_closure.copy(), enabled)
            adjusted, curls = np.asarray(adjusted, dtype=float), np.asarray(curls, dtype=float)
            if (adjusted.shape != (29,) or curls.shape != (2,) or not np.isfinite(adjusted).all()
                    or not np.isfinite(curls).all()
                    or not np.array_equal(np.delete(adjusted, arm_indices), np.delete(scripted, arm_indices))
                    or np.max(np.abs(adjusted[arm_indices] - scripted[arm_indices])) > .18000001
                    or np.max(np.abs(curls - scripted_closure)) > .15000001
                    or np.any(curls < 0.) or np.any(curls > 1.)):
                raise ValueError("residual may only change fourteen arms by .18rad and paired closure by .15")
            target = adjusted.copy()
            target[arm_indices] = np.clip(target[arm_indices], *plant.joint_ranges[arm_indices].T)
            closure = curls
        if fraction < 1. or cut.outside_run_fraction:
            amp = policy.act(obs, SkillGoal(base_velocity=run_goal), upper_target=target)
        walk = plant.wbc.act(obs, SkillGoal(base_velocity=walk_goal), upper_target=target)
        command = blend_pd(amp, walk, fraction) if fraction < 1. else walk
        if upper_lookback_yaw_rad is not None and cut.break_at_s is not None:
            upper_fraction = upper_lookback_fraction(now - cut.break_at_s)
            if upper_fraction:
                goal = (1. - upper_fraction) * command.target_position[waist_index] + upper_fraction * upper_lookback_yaw_rad
                command.target_position[waist_index] = np.clip(goal, *plant.joint_ranges[waist_index])
                upper_lookback_targets.append((now, float(command.target_position[waist_index])))
        plant.step(command, closure=closure, post_step=audit)
    trace = np.array(rows)
    if cut.break_at_s is not None:
        at_break = trace[np.argmin(abs(trace[:, 0] - cut.break_at_s))]
        before = trace[np.argmin(abs(trace[:, 0] - (cut.break_at_s - .28)))]
        start_velocity = (at_break[1:3] - before[1:3]) / (at_break[0] - before[0])
        tail = trace[trace[:, 0] >= trace[-1, 0] - .5]
        end_delta = tail[-1, 1:3] - tail[0, 1:3]
        heading_deg = float(np.degrees(np.arctan2(end_delta @ cut.outside, end_delta @ cut.downfield)))
        actual_break_depth = float(at_break[1:3] @ cut.downfield)
        measured_break_depth = (float(np.asarray(cut.break_pose_m) @ cut.downfield)
                                if cut.break_pose_m is not None else None)
    else:
        start_velocity = heading_deg = actual_break_depth = measured_break_depth = None
    report = {
        "evidence_class": ("scheduled_out_physics_screen" if screening_break_s is not None
                           else "camera_out_route_without_pass"),
        "requested_break_m": 5 * YARD_M, "field_heading_rad": cut.field_heading_rad,
        "outside_sign": cut.outside_sign, "break_at_s": cut.break_at_s,
        "outside_run_fraction": cut.outside_run_fraction,
        "outside_run_at_s": cut.outside_run_at_s,
        "receiving_heading_rad": cut.receiving_heading_rad,
        "receiving_heading_ramp_s": cut.receiving_heading_ramp_s,
        "approach_handoff_at_s": cut.handoff_at_s,
        "camera_break_depth_m": measured_break_depth, "measured_break_depth_m": actual_break_depth,
        "prebreak_speed_mps": None if start_velocity is None else float(np.linalg.norm(start_velocity)),
        "terminal_outside_heading_deg": heading_deg, "duration_s": float(trace[-1, 0]),
        "minimum_upright_cosine": float(trace[:, 4].min()),
        "final_xy_m": trace[-1, 1:3].tolist(),
        "finish_brake_at_s": finish,
        "finish_grip_target": finish_grip,
        "carry_arm_trace": carry_arm_trace,
        "carry_after_grip_s": carry_after_grip_s if carry_arm_path is not None else None,
        "carry_ramp_s": carry_ramp_s if carry_arm_path is not None else None,
        "staged_arm_offsets_rad": None if staged_arm_offsets is None else staged_arm_offsets.tolist(),
        "upper_lookback_yaw_rad": upper_lookback_yaw_rad,
        "upper_lookback_targets": upper_lookback_targets,
        "finish_localization_fallback": finish_localization_fallback,
        "finish_fallback_samples": fallback_samples,
        "warning_count": sum(int(w.number) for w in plant.data.warning),
    }
    if cut.break_at_s is not None:
        stem = trace[trace[:, 0] <= cut.break_at_s + 1e-8]
        report["stem_max_abs_lateral_deviation_m"] = float(np.max(np.abs(stem[:, 1:3] @ cut.outside)))
        report["stem_lateral_at_break_m"] = float(stem[-1, 1:3] @ cut.outside)
        report["prebreak_velocity_heading_relative_stem_deg"] = float(np.degrees(np.arctan2(
            start_velocity @ cut.outside, start_velocity @ cut.downfield)))
        after = trace[trace[:, 0] >= cut.break_at_s]
        exit_rows = trace[trace[:, 0] >= cut.break_at_s + cut.turn_s + cut.hold_straight_s]
        report["maximum_downfield_overshoot_m"] = float(max(after[:, 1:3] @ cut.downfield) - 5 * YARD_M)
        if len(exit_rows) > 1:
            velocity = np.diff(exit_rows[:, 1:3], axis=0) / np.diff(exit_rows[:, 0])[:, None]
            report["exit_mean_outside_speed_mps"] = float(np.mean(velocity @ cut.outside))
            report["exit_mean_downfield_speed_mps"] = float(np.mean(velocity @ cut.downfield))
        report["route_frame"] = {"scrimmage_origin_xy_m": [0., 0.],
                                 "downfield_unit_xy": cut.downfield.tolist(),
                                 "outside_unit_xy": cut.outside.tolist()}
    if delivery is not None:
        if delivery.quarterback_origin is not None:
            origin = np.asarray(delivery.quarterback_origin)
            report["quarterback_route_depth_m"] = float(origin[:2] @ cut.downfield)
        report["visual_closure_s"] = delivery.hands.closed_at
        report["observed_ball"] = delivery.hands.trace
        report["visual_aperture_trace"] = getattr(delivery, "aperture_trace", [])
        report["visual_aperture_gain"] = delivery.profile.get("visual_aperture_gain", 0.)
    return report, trace, feet.arrays()


def main(argv=None, residual_callback=None, sensor_retention_camera=None, sensor_retention_pose_camera=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--amp-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--outside", type=int, choices=(-1, 1), default=1)
    p.add_argument("--field-heading", type=float, default=.415)
    p.add_argument("--upper-lookback-yaw", type=float,
                   help="opt-in brief waist-only turn after camera break; pelvis heading unchanged")
    p.add_argument("--outside-run-fraction", type=float, default=0.,
                   help="opt-in AMP re-entry after IMU confirms outside heading alignment")
    p.add_argument("--outside-run-ramp-s", type=float, default=.6)
    p.add_argument("--finish-grip", type=float, help="normalized0..1 actuator closure after finish+.15s with .30s ramp")
    p.add_argument("--finish-localization-fallback", choices=("pass", "finish"),
                   help="opt-in surveyed RGB-D fallback after visual grip closure +0.7s")
    p.add_argument("--receiving-heading", type=float)
    p.add_argument("--receiving-heading-ramp-s", type=float, default=1.)
    p.add_argument("--approach-speed", type=float)
    p.add_argument("--camera-start-s", type=float, default=1.8)
    p.add_argument("--stopping-lookahead-s", type=float, default=.28)
    p.add_argument("--reverse-speed", type=float, default=.25)
    p.add_argument("--max-walk-fraction", type=float, default=1.)
    p.add_argument("--running-turn", action="store_true")
    p.add_argument("--horizontal-pass-distance", action="store_true",
                   help="require >=5m horizontal throw travel in the surveyed Out frame")
    p.add_argument("--align-quarterback", action="store_true",
                   help="project QB onto the scrimmage line normal to the declared stem")
    p.add_argument("--turn-s", type=float, default=.9)
    p.add_argument("--duration", type=float, default=7.5)
    p.add_argument("--cut-speed", type=float, default=.48)
    p.add_argument("--blend-s", type=float, default=.65)
    p.add_argument("--hold-straight-s", type=float, default=.25)
    p.add_argument("--approach-handoff-depth-m", type=float)
    p.add_argument("--depth-feedback-gain", type=float, default=0.)
    p.add_argument("--screening-break-s", type=float,
                   help="offline physics screening only: schedule the start of the cut")
    p.add_argument("--pass-profile", type=Path, help="JSON containing launch and camera-grasp parameters")
    p.add_argument("--finish-outside-m", type=float)
    p.add_argument("--record-poses", action="store_true", help="record states for the separate stadium renderer")
    args = p.parse_args(argv)
    if not 0. <= args.max_walk_fraction <= 1.:
        p.error("--max-walk-fraction must be between zero and one")
    if args.upper_lookback_yaw is not None and not -2.6 <= args.upper_lookback_yaw <= 2.6:
        p.error("upper lookback yaw must be in [-2.6,2.6] radians")
    if not 0. <= args.outside_run_fraction <= 1. or args.outside_run_ramp_s <= 0.:
        p.error("outside run fraction must be [0,1] and ramp must be positive")
    if args.finish_grip is not None and not 0. <= args.finish_grip <= 1.:
        p.error("--finish-grip must be in [0, 1]")
    if args.receiving_heading_ramp_s <= 0.:
        p.error("--receiving-heading-ramp-s must be positive")
    cameras = {}
    profile = json.loads(args.pass_profile.read_text()) if args.pass_profile is not None else None

    if profile is not None and args.align_quarterback:
        forward = np.array([np.cos(args.field_heading), np.sin(args.field_heading)])
        origin = np.array(profile["play"]["quarterback_origin_m"], dtype=float)
        origin[:2] -= forward * (origin[:2] @ forward)
        profile["play"]["quarterback_origin_m"] = origin.tolist()

    def setup(scene):
        if residual_callback is not None and sensor_retention_pose_camera is not None:
            position, target = sensor_retention_pose_camera
            if not np.isfinite((*position, *target)).all():
                raise ValueError("finite surveyed residual pose camera required")
            cameras["retention_pose"] = fixed_camera(scene, "retention_pose", position, target)
        if residual_callback is not None and sensor_retention_camera is not None:
            position, target = sensor_retention_camera
            if not np.isfinite((*position, *target)).all():
                raise ValueError("finite surveyed residual camera required")
            cameras["retention"] = fixed_camera(scene, "retention", position, target)
        cameras["break"] = fixed_camera(scene, "break", (.65, .15, 1.37), (4.2, 2.15, .8))
        cameras["outside"] = fixed_camera(scene, "outside", (8.4, 1.3, 1.46), (5.6, 4.05, .81))
        if args.finish_localization_fallback == "finish":
            cameras["finish"] = fixed_camera(scene, "finish", (2., 8.5, 1.8), (3.2, 4.5, .8))
        if profile is not None:
            position, target = profile["pass_camera"]
            cameras["pass"] = fixed_camera(scene, "pass", position, target)

    plant = HandRoutePlant(args.source, hand_model="aero", rear_markers=True, scene_setup=setup)
    plant.cameras.update(cameras)
    try:
        running = AmpRunningPolicy(plant.joint_names, plant.joint_ranges, args.amp_source)
        cut = OutCut(args.field_heading, args.outside, args.turn_s, args.cut_speed,
                     args.blend_s, args.hold_straight_s, args.approach_handoff_depth_m,
                     args.depth_feedback_gain, args.max_walk_fraction, args.running_turn,
                     args.stopping_lookahead_s, args.reverse_speed, args.approach_speed,
                     args.receiving_heading, args.receiving_heading_ramp_s,
                     args.outside_run_fraction, args.outside_run_ramp_s)
        delivery = OpticalPass(plant, profile["play"], (0., 0., 0.)) if profile is not None else None
        if args.horizontal_pass_distance and delivery is not None and delivery.quarterback is not None:
            delivery.quarterback.plan = replace(delivery.quarterback.plan, horizontal_distance_only=True)
        recording = [] if args.record_poses else None
        report, trace, feet = run(plant, running, cut, duration=args.duration, delivery=delivery,
                                  screening_break_s=args.screening_break_s,
                                  finish_outside_m=args.finish_outside_m,
                                  spectator_states=recording, camera_start_s=args.camera_start_s,
                                  finish_localization_fallback=args.finish_localization_fallback,
                                  finish_grip=args.finish_grip, upper_lookback_yaw_rad=args.upper_lookback_yaw,
                                  staged_arm_offsets=None if profile is None else profile.get("out_staged_arm_offsets_rad"),
                                  carry_arm_path=None if profile is None else profile.get("out_carry_arm_path_rad"),
                                  carry_after_grip_s=.6 if profile is None else profile.get("out_carry_after_grip_s", .6),
                                  carry_ramp_s=1.2 if profile is None else profile.get("out_carry_ramp_s", 1.2),
                                  residual_callback=residual_callback)
        if args.finish_localization_fallback is not None:
            rotation, position = cameras[args.finish_localization_fallback]
            report["finish_localization_contract"] = {
                "sensor": "surveyed RGB-D pelvis markers",
                "camera": args.finish_localization_fallback,
                "camera_position_m": np.asarray(position).tolist(),
                "camera_rotation": np.asarray(rotation).tolist(),
                "gate": "visual grip closed_at + 0.7 s and no primary camera measurement",
                "simulator_pose_or_contact_input": False,
            }
        if delivery is not None and delivery.release is not None:
            contact = delivery.audit.trace()
            route = Route("out", ((0., 0.), tuple(profile["break_xy_m"]), tuple(profile["finish_xy_m"])))
            report["catch_score"] = score_in_stride_catch(
                contact, trace[:, :5], route, release_time_s=delivery.release.release_time_s,
                finish_time_s=round(args.duration / CONTROL_DT) * CONTROL_DT,
                ball_weight_n=delivery.audit.ball_weight_n, verified_free_ball=True,
                simulator_warning_count=report["warning_count"],
            )
            report["evidence_class"] = "camera_out_route_with_free_pass"
            report["release"] = asdict(delivery.release)
            report["throw_distance_guard"] = "horizontal_5m" if args.horizontal_pass_distance else "world_x_5m"
            if delivery.quarterback is not None:
                report["quarterback_pass_score"] = score_throw_receive(
                    contact, trace[:, :5], report["release"], report["catch_score"]["first_tip_contact_s"],
                    physics_step_s=plant.model.opt.timestep)
            feet["contact"] = contact
    finally:
        plant.close()
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "trace.npz", motion=trace, **feet)
    if recording:
        stamps, poses = zip(*recording, strict=True)
        np.savez_compressed(args.output / "view.npz", timestamps_s=stamps, qpos=poses)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        key: report.get(key) for key in ("evidence_class", "measured_break_depth_m",
                                         "visual_closure_s", "quarterback_pass_score")
    }), flush=True)


if __name__ == "__main__":
    main()
