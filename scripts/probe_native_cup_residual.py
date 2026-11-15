"""Native20D residual hooks on a surveyed Go lower-cup controller.

This experimental controller is deliberately separate from the verified Go.
Control reads camera poses, IMU and encoders; exact poses and palms are logged
only for evaluation and offline throw calibration. No running-catch claim.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import mujoco

from evaluate_aero_receiver import make_plant
from evaluate_dextra_in_stride import RearMarkerLocalizer
from probe_aero_amp_slant import OpticalPass
from g1_receiver.aero import aero_ready_joint_targets, PALM_BODIES
from g1_receiver.football_routes import imu_yaw, heading_error
from g1_receiver.in_stride_eval import score_in_stride_catch
from g1_receiver.locomotion import CONTROL_DT
from g1_receiver.run_braking import blend_pd
from g1_receiver.running_catch_eval import FootContactRecorder, score_running_long_pass
from g1_receiver.running_policy import AmpRunningPolicy
from g1_receiver.routes import Route
from g1_receiver.runtime import RobotObservation, SkillGoal
from g1_receiver.throw_eval import score_throw_receive


def build_parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, required=True)
    cli.add_argument('--amp-source', type=Path, required=True)
    cli.add_argument('--profile', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--no-pass', action='store_true')
    cli.add_argument('--walk-speed', type=float, default=1.)
    cli.add_argument('--handoff-depth', type=float, default=2.1)
    cli.add_argument('--initial-yaw', type=float, default=0., help='Physical stance heading at reset only; never repositions an active play')
    cli.add_argument('--sensor-retention-camera-position',type=float,nargs=3)
    cli.add_argument('--sensor-retention-camera-target',type=float,nargs=3)
    cli.add_argument('--sensor-retention-pose-camera-position',type=float,nargs=3)
    cli.add_argument('--sensor-retention-pose-camera-target',type=float,nargs=3)
    cli.add_argument('--duration', type=float, default=14.)
    cli.add_argument('--record-anatomical', action='store_true')
    cli.add_argument('--reference-physics-sha256')
    return cli


def run(args, native_residual_callback=None):
    """Run the scripted controller; optional callback receives sensor packets only.

    Callback receives29body encoders/IMU, explicitRGBDpose/frame/survey, nominal
    upper29/native14 targets, and14named nativeencoders. It returns upper29
    plus ACTUALLY APPLIED native14; no scalarclosure residual exists.
    A callback bind(model,joint_names) hook receives a separate static FK model.
    """
    cli = build_parser()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not .3 <= args.walk_speed <= 1.5 or not 1. <= args.handoff_depth <= 8.:
        cli.error('bounded walking speed and handoff depth required')
    if not np.isfinite(args.initial_yaw) or abs(args.initial_yaw) > .8:
        cli.error('initial stance yaw must be in [-.8,.8] radians')
    profile = json.loads(args.profile.read_text())
    if not args.no_pass and not np.allclose(
            profile['play']['quarterback_origin_m'], (0., -1.8288, 1.4), atol=1e-10, rtol=0.):
        raise ValueError('Go throw profile must place the fixed machine at the declared two-yard formation')
    route = Route('go', ((0., 0.), (9.144, 0.), (12., 0.)), 2.)
    sensor_position = getattr(args, 'sensor_retention_camera_position', None)
    sensor_target = getattr(args, 'sensor_retention_camera_target', None)
    camera_options = {}
    if native_residual_callback is not None and (sensor_position is not None or sensor_target is not None):
        if sensor_position is None or sensor_target is None or not np.isfinite((*sensor_position, *sensor_target)).all():
            raise ValueError('a surveyed retention view requires finite position and target')
        camera_options['retention_camera'] = (sensor_position, sensor_target)
    pose_position = getattr(args, 'sensor_retention_pose_camera_position', None)
    pose_target = getattr(args, 'sensor_retention_pose_camera_target', None)
    if native_residual_callback is not None and (pose_position is not None or pose_target is not None):
        if pose_position is None or pose_target is None or not np.isfinite((*pose_position, *pose_target)).all():
            raise ValueError('a surveyed retention pose view requires finite position and target')
        camera_options['retention_pose_camera'] = (pose_position, pose_target)
    plant = make_plant(args.source, route,
                       tracking_camera=((-1., 0., 1.5), (3., 0., .9)),
                       finish_camera=((5., 0., 1.5), (10., 0., .9)),
                       pass_camera=profile.get('pass_camera', ((6., -4., 3.8), (9., 0., 1.5))), **camera_options)
    try:
        from g1_receiver.lower_cup_grip import NativeHandCalibration, NativeCupGrip
        native_calibration = NativeHandCalibration.from_model(plant.model)
        if 'native_grip' not in profile:raise ValueError('native residual runner requires an explicit native controller')
        native_grip = NativeCupGrip(native_calibration, profile['native_grip'])
        policy = AmpRunningPolicy(plant.joint_names, plant.joint_ranges, args.amp_source)
        upper = aero_ready_joint_targets(policy.nominal, plant.joint_names)
        arms = [plant.joint_names.index(f'{side}_{axis}_joint')
                for side in ('left', 'right') for axis in
                ('shoulder_roll', 'shoulder_yaw', 'shoulder_pitch', 'elbow',
                 'wrist_roll', 'wrist_pitch', 'wrist_yaw')]
        offsets = np.asarray(profile['out_staged_arm_offsets_rad'])
        if offsets.shape != (14,) or not np.isfinite(offsets).all():
            raise ValueError('fourteen finite staged arm offsets required')
        delivery = None if args.no_pass else OpticalPass(plant, profile['play'], (0., 0., 0.))
        plant.reset(joint_positions=upper, release_position_m=(0., -1.8288, 1.4))
        if args.initial_yaw:
            plant.data.qpos[plant.base+3:plant.base+7] = (np.cos(args.initial_yaw/2),0.,0.,np.sin(args.initial_yaw/2))
            mujoco.mj_forward(plant.model, plant.data)
        if delivery is not None:
            delivery.ready()
        policy.reset()
        anatomical = anatomical_form = interhand_geometry = None
        anatomical_form_rows = []
        if getattr(args, 'record_anatomical', False):
            from g1_receiver.anatomical_catch_eval import AnatomicalContactAudit, score_anatomical_catch
            from g1_receiver.catch_form import CatchFormEvaluator, CatchFormContext
            from g1_receiver.native_hand_eval import NativeInterhandGeometryAudit
            interhand_geometry = NativeInterhandGeometryAudit(plant.model)
            anatomical = AnatomicalContactAudit(plant.model, reference_physics_sha256=getattr(args,'reference_physics_sha256',None))
            anatomical_form = CatchFormEvaluator(mujoco.MjModel.from_xml_string(plant.scene_xml),tuple(plant.joint_names))
        native_frame=native_survey=native_pose=None
        native_view_rows=[];applied_native_rows=[]
        if native_residual_callback is not None:
            from g1_receiver.perception import ColorBallDetector
            from g1_receiver.catch_view_selection import CatchView,select_catch_view
            native_detector=ColorBallDetector(max_depth_m=20.,ball_center_offset_m=.075)
            native_localizer=RearMarkerLocalizer(allow_synthetic_fixture=True,max_range_m=8.)
            if hasattr(native_residual_callback,'bind'):
                native_residual_callback.bind(mujoco.MjModel.from_xml_string(plant.scene_xml),tuple(plant.joint_names))
        estimator = RearMarkerLocalizer(allow_synthetic_fixture=True, max_range_m=8.)
        feet = FootContactRecorder(plant.model)
        pose = None
        history, motion, poses, palms, commands = [], [], [], [], []
        handoff = None
        def audit(data):
            feet.record(data)
            if delivery is not None and delivery.release is not None:
                delivery.audit.record(data)
                if anatomical is not None:
                    anatomical.record(data)
                    interhand_geometry.record(data)
            elif getattr(args,'no_pass',False) and anatomical is not None:
                anatomical.record(data)
                interhand_geometry.record(data)
        for tick in range(round(args.duration/CONTROL_DT)+1):
            now = plant.timestamp_s
            obs = RobotObservation(now, *plant.observe(), np.zeros(29), None)
            if anatomical_form is not None:
                anatomical_form_rows.append(dict(time_s=now, **anatomical_form.evaluate(obs,CatchFormContext(over_shoulder=True,phase='receive'))))
            yaw = imu_yaw(obs.imu_quat_wxyz)
            if tick % 2 == 0 and now >= .8:
                for camera in ('finish', 'track') if handoff is not None else ('track',):
                    measured = estimator.estimate(*plant.capture(camera))
                    if measured is not None:
                        pose = measured
                        history = [*history[-6:], (now, *pose.xy_m)]
                        break
            fresh = pose is not None and pose.fresh(now, .14)
            if handoff is None and fresh and pose.xy_m[0] >= args.handoff_depth:
                handoff = now
            if delivery is not None:
                delivery.observe(plant, now, pose, history)
            elapsed = 0. if handoff is None else now-handoff
            u = float(np.clip(elapsed/.6, 0., 1.))
            blend = u*u*(3.-2.*u)
            stage = float(np.clip((elapsed-1.1)/.8, 0., 1.))
            stage = stage*stage*(3.-2.*stage)
            target = upper.copy()
            target[arms] += stage*offsets
            target, closure = (target, .2) if delivery is None else delivery.targets(now, target)
            _,imu_x,imu_y,_=obs.imu_quat_wxyz
            will_step=tick<round(args.duration/CONTROL_DT) and 1.-2.*(imu_x*imu_x+imu_y*imu_y)>=.54
            native_command=None
            if will_step:
                native_encoders=plant.data.sensordata[native_calibration.sensor_addresses].copy()
                native_nominal=native_grip.targets(now,closure,
                    None if delivery is None or delivery.release is None else delivery.release.release_time_s,
                    None if delivery is None else delivery.hands.closed_at,native_encoders)
                native_command=native_nominal.copy()
                if native_residual_callback is not None:
                    if tick%2==0:
                        views=[];pose_candidates=[('route',pose),('cached',native_pose)];frames={}
                        for camera in ('pass','retention','retention_pose'):
                            if camera not in plant.cameras:continue
                            frame,survey=plant.capture(camera);frames[camera]=(frame,survey)
                            detected=native_detector.detect(frame,survey)
                            views.append(CatchView(camera,frame,survey,detected))
                            pose_candidates.append((camera,native_localizer.estimate(frame,survey)))
                        selected,chosen_pose,diagnostics=select_catch_view(views,pose_candidates,now_s=now,imu_quat_wxyz=obs.imu_quat_wxyz)
                        if selected is None:
                            # Preserve actual failed-image timestamp and invalid debit.
                            native_frame,native_survey=frames['pass'];native_pose=None
                        else:
                            native_frame,native_survey,native_pose=selected.frame,selected.survey,chosen_pose
                        native_view_rows.append(dict(time_s=now,**diagnostics))
                    scripted=np.asarray(target).copy()
                    enabled=delivery is not None and delivery.hands.ready_at is not None
                    adjusted,changed=native_residual_callback(deepcopy(obs),deepcopy(native_pose),deepcopy(native_frame),deepcopy(native_survey),scripted.copy(),native_nominal.copy(),native_encoders.copy(),enabled)
                    adjusted,changed=np.asarray(adjusted,dtype=float),np.asarray(changed,dtype=float)
                    nominal_fraction=native_calibration.fractions(native_nominal);changed_fraction=native_calibration.fractions(changed)
                    if (adjusted.shape!=(29,) or changed.shape!=(14,) or not np.isfinite(adjusted).all() or not np.isfinite(changed).all()
                        or not np.array_equal(np.delete(adjusted,arms),np.delete(scripted,arms))
                        or np.max(abs(adjusted[arms]-scripted[arms]))>.18000001
                        or np.max(abs(changed_fraction-nominal_fraction))>.15000001
                        or np.any(changed_fraction < -1e-8)or np.any(changed_fraction>1.+1e-8)):
                        raise ValueError('native callback may change14arms by.18rad and14native normalizedtargets by.15 only')
                    target=adjusted.copy();native_command=changed.copy()
            run_drive = np.array([min(2., 1.8*max(0., now-.4)), 0.,
                                  np.clip(1.8*heading_error(0., yaw), -.75, .75)])
            lateral = 0. if not fresh else float(np.clip(-.8*pose.xy_m[1], -.35, .35))
            c, s = np.cos(yaw), np.sin(yaw)
            walk_drive = np.r_[np.array([[c,s],[-s,c]]) @ [args.walk_speed, lateral],
                               np.clip(1.15*heading_error(0., yaw), -.65, .65)]
            # These exact-state reads never enter either controller or grip.
            pelvis = plant.data.body('pelvis')
            motion.append((now, *pelvis.xpos, float(pelvis.xmat[8]), yaw))
            commands.append((now, *run_drive, *walk_drive, blend, stage,
                             *(pose.xy_m if fresh else (np.nan, np.nan))))
            if tick % 2 == 0:
                poses.append((now, plant.data.qpos.copy()))
                palms.append((now, *np.mean([plant.data.body(n).xpos for n in PALM_BODIES], axis=0)))
            _, x, y, _ = obs.imu_quat_wxyz
            if tick == round(args.duration/CONTROL_DT) or 1.-2.*(x*x+y*y) < .54:
                break
            walk = plant.wbc.act(obs, SkillGoal(base_velocity=walk_drive), upper_target=target)
            if blend < 1.:
                amp = policy.act(obs, SkillGoal(base_velocity=run_drive), upper_target=target)
                command = blend_pd(amp, walk, blend)
            else:
                command = walk
            applied_native_rows.append((now,*native_command))
            plant.step(command,hand_target_rad=native_command,post_step=audit)
        motion = np.asarray(motion)
        foot = feet.arrays()
        report = {'evidence_class': 'experimental_go_running_approach_walking_receive',
                  'controller_inputs': ['surveyed RGB-D', 'IMU', 'joint encoders'],
                  'handoff_at_s': handoff, 'handoff_depth_m': args.handoff_depth,
                  'initial_stance_yaw_rad': args.initial_yaw,
                  'walk_speed_command_mps': args.walk_speed,
                  'duration_s': float(motion[-1,0]),
                  'minimum_upright_cosine': float(motion[:,4].min()),
                  'max_stem_error_m': float(abs(motion[:,2]).max()),
                  'final_xyz_m': motion[-1,1:4].tolist(),
                  'formation': {'quarterback_origin_m': [0.,-1.8288,1.4], 'receiver_start_xy_m': [0.,0.]},
                  'camera_surveys': {n: {'rotation':r.tolist(), 'position_m':p.tolist()} for n,(r,p) in plant.cameras.items()},
                  'warning_count': sum(int(w.number) for w in plant.data.warning),
                  'physics_step_s': float(plant.model.opt.timestep)}
        if delivery is not None and delivery.release is not None:
            contact = delivery.audit.trace()
            release = asdict(delivery.release)
            strict = score_in_stride_catch(contact, motion[:,:5], route,
                       release_time_s=release['release_time_s'], finish_time_s=args.duration,
                       ball_weight_n=delivery.audit.ball_weight_n, verified_free_ball=True,
                       simulator_warning_count=report['warning_count'])
            report.update(play=profile['play'], release=release, strict_score=strict,
                          visual_closure_s=delivery.hands.closed_at, observed_ball=delivery.hands.trace,
                          quarterback_pass_score=score_throw_receive(contact, motion[:,:5], release,
                              strict['first_tip_contact_s'], physics_step_s=plant.model.opt.timestep))
            report['running_score'] = score_running_long_pass(
                contact, motion[:,:5], **foot, strict_catch_score=strict,
                release_time_s=release['release_time_s'], gait_start_s=.4,
                pass_toward_receiver_xy=(-1.,0.))
            foot['contact'] = contact
        if native_grip is not None:
            report['native_hand_contract'] = {'inputs':['commanded release event','visual grasp gate','six tendon position encoders and thumb opposition encoder per hand'], 'individual_finger_joint_state':False,'contacts_or_ball_truth':False,'encoder_use':'finite packet validation and logging only; not tactile feedback','native_control_ranges':native_calibration.ranges.tolist(),'official_rate_limits_unchanged':True,'grip_config':profile['native_grip']}
            foot['native_hand'] = np.asarray(native_grip.rows)
        foot['native_applied_commands']=np.asarray(applied_native_rows)
        args.output.mkdir(parents=True)
        (args.output/'native-view-selection.json').write_text(json.dumps(native_view_rows,indent=2)+'\n')
        if anatomical is not None:
            aa=anatomical.arrays();metadata=anatomical.metadata()
            np.savez_compressed(args.output/'interhand-geometry.npz',**interhand_geometry.arrays())
            report['interhand_geometry_audit']=interhand_geometry.summary()
            np.savez_compressed(args.output/'anatomical-contact.npz',**aa)
            (args.output/'anatomical-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
            (args.output/'anatomical-form.json').write_text(json.dumps(anatomical_form_rows,indent=2)+'\n')
            if delivery is not None and delivery.release is not None:
                report['anatomical_score']=score_anatomical_catch(aa,release_time_s=release['release_time_s'],finish_time_s=float(motion[-1,0]),ball_weight_n=anatomical.ball_weight_n,form_rows=anatomical_form_rows,motion=motion,physics_reference_match=metadata['physics_reference_match'],verified_free_ball=metadata['free_ball_joint_and_no_attachment_verified'])
        np.savez_compressed(args.output/'trace.npz', motion=motion, commands=commands, palms=palms, **foot)
        stamps, qpos = zip(*poses)
        np.savez_compressed(args.output/'view.npz', timestamps_s=stamps, qpos=qpos)
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({k:report[k] for k in ('handoff_at_s','duration_s','minimum_upright_cosine','max_stem_error_m','final_xyz_m')}))
        return report
    finally:
        plant.close()


def main():
    return run(build_parser().parse_args())


if __name__ == '__main__':
    main()
