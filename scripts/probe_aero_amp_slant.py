"""Run a two-leg Aero course with surveyed turn and braking markers."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import mujoco
import numpy as np

from evaluate_aero_receiver import fixed_camera
from evaluate_dextra_in_stride import RearMarkerLocalizer
from probe_aero_amp_braking import blend_pd
from g1_receiver.aero import FINGERTIP_GEOMS, PALM_BODIES, aero_ready_joint_targets
from g1_receiver.dextra_physical_eval import DextraContactAudit, DextraRetentionCriteria, TRACE_COLUMNS
from g1_receiver.hand_route_plant import HandRoutePlant
from g1_receiver.in_stride_eval import score_in_stride_catch
from g1_receiver.locomotion import CONTROL_DT
from g1_receiver.pass_control import FieldPassGrip, PassGrip, VisualPassGrip, observed_velocity
from g1_receiver.routes import Route
from g1_receiver.running_catch_eval import FootContactRecorder, score_running_long_pass
from g1_receiver.running_policy import AmpRunningPolicy
from g1_receiver.runtime import RobotObservation, SkillGoal
from g1_receiver.short_throw import FreeBallRelease, UnderhandThrow
from g1_receiver.scrimmage_throw import FixedQuarterback, ScrimmagePass
from g1_receiver.throw_eval import score_throw_receive

TURN = np.array([4.98, 2.51])
DEFAULT_FINISH = (13.33, 4.20)


class OpticalPass:
    """Fixed delivery with camera-only grip timing and two calibrated arm phases."""

    def __init__(self, plant, profile, shift):
        self.profile = profile
        self.quarterback_origin = profile.get('quarterback_origin_m')
        grip = PassGrip(**profile['grip'])
        self.launch_xyz = None
        self.quarterback = None
        if self.quarterback_origin is None:
            self.launch_xyz = tuple(np.asarray(profile['launch_xyz']) + shift)
            self.hands = VisualPassGrip(grip, allow_synthetic_fixture=True)
        else:
            self.quarterback_origin = tuple(self.quarterback_origin)
            if np.linalg.norm(self.quarterback_origin[:2]) < .85:
                raise ValueError('the quarterback and receiver need separate starting positions')
            aim = tuple(np.asarray(profile['quarterback_target_m']) + shift)
            window = profile.get('field_catch_window_m') or profile['quarterback_target_m']
            self.hands = FieldPassGrip(
                grip, np.asarray(window) + shift, allow_synthetic_fixture=True,
                approach_axis_xy=profile.get('field_approach_axis_xy', (1., 0.)),
                minimum_approach_speed_mps=profile.get('minimum_approach_speed_mps', 2.5))
            plan = ScrimmagePass(self.quarterback_origin, aim, profile['flight_s'],
                                 align_ball_to_launch=not profile.get('quarterback_level_ball', False),
                                 horizontal_distance_only=profile.get('quarterback_horizontal_distance_only', False))
            self.quarterback = FixedQuarterback(plant.model, plant.data, plan)
        self.arm_indices = [plant.joint_names.index(f'{side}_{axis}_joint')
                            for side in ('left', 'right') for axis in (
                                'shoulder_roll', 'shoulder_yaw', 'shoulder_pitch', 'elbow',
                                'wrist_roll', 'wrist_pitch', 'wrist_yaw')]
        self.audit = DextraContactAudit(plant.model, palms=PALM_BODIES, fingertip_geoms=FINGERTIP_GEOMS)
        self.thrower = FreeBallRelease(plant.model, plant.data)
        self.release = None
        self.aperture_gain = float(profile.get('visual_aperture_gain', 0.))
        if not np.isfinite(self.aperture_gain) or not 0. <= self.aperture_gain <= 3.:
            raise ValueError('visual aperture gain must be finite and in [0, 3]')
        self.aperture_start_after_grip_s = float(profile.get('visual_aperture_start_after_grip_s', 0.))
        if not np.isfinite(self.aperture_start_after_grip_s) or not 0. <= self.aperture_start_after_grip_s <= 1.:
            raise ValueError('visual aperture start delay must be in [0, 1] seconds')
        self.aperture_correction = np.zeros(14)
        self.aperture_trace = []
        self.aperture_history = []
        self.aperture_fk = mujoco.MjData(plant.model) if self.aperture_gain else None
        self.aperture_dofs = np.array([plant.model.joint(plant.joint_names[i]).dofadr[0] for i in self.arm_indices])

    def ready(self):
        if self.quarterback is None:
            return
        self.quarterback.ready()

    def initial_ball_position(self, parked):
        return parked if self.quarterback_origin is None else self.quarterback_origin

    def observe(self, plant, now, pose, poses):
        settings = self.profile
        if self.release is None and now >= settings['release_s'] - 1e-8:
            if self.quarterback is not None:
                self.release = self.quarterback.release(now)
            else:
                velocity = np.asarray(settings['launch_velocity'])
                flight = settings['flight_s']
                gravity = np.array([0., 0., -9.81])
                intercept = np.asarray(self.launch_xyz) + flight * velocity + .5 * flight**2 * gravity
                self.release = UnderhandThrow(now, flight, self.launch_xyz, tuple(velocity),
                                              (1., 0., 0., 0.), tuple(intercept),
                                              tuple(velocity + flight * gravity), float(flight * velocity[0]))
                self.thrower.apply(self.release)
        wait_s = max(1., settings['flight_s'] + .12) if self.quarterback is not None else 1.
        capture = None
        if self.release is not None and self.hands.closed_at is None and now < settings['release_s'] + wait_s:
            capture = plant.capture('pass')
            self.hands.update(*capture, pose, observed_velocity(poses), now)
        if (self.aperture_gain and self.hands.closed_at is not None
                and self.hands.closed_at + self.aperture_start_after_grip_s <= now <= self.hands.closed_at + 1.5):
            self._update_aperture(plant, now, pose, poses, capture)

    def _update_aperture(self, plant, now, pose, poses, capture):
        """Surveyed RGB-D plus encoder/IMU FK only; no simulation state/contact reads."""
        if pose is None or pose.z_m is None or not pose.fresh(now, .12):
            return
        frame, survey = plant.capture('pass') if capture is None else capture
        ball = self.hands.detector.detect(frame, survey)
        if ball is None or not 0. <= now - ball.timestamp_s <= .08:
            return
        if not self.aperture_history:
            self.aperture_history = [(row[0], *row[1:4]) for row in self.hands.trace[-5:]]
        if not self.aperture_history or ball.timestamp_s > self.aperture_history[-1][0] + 1e-8:
            self.aperture_history = [*self.aperture_history[-5:], (ball.timestamp_s, *ball.position_m)]
        rows = np.asarray(self.aperture_history, dtype=float)
        velocity = np.zeros(3)
        if len(rows) >= 3 and np.ptp(rows[:, 0]) >= .04:
            times = rows[:, 0] - rows[:, 0].mean()
            velocity = times @ rows[:, 1:] / (times @ times)
        q, dq, quat, gyro = plant.observe()
        fk = self.aperture_fk
        fk.qpos[:] = plant.model.qpos0
        fk.qpos[plant.joint_qpos] = q
        fk.qpos[plant.base:plant.base + 7] = (*pose.xy_m, pose.z_m, *quat)
        mujoco.mj_forward(plant.model, fk)
        pelvis_rotation = fk.body('pelvis').xmat.reshape(3, 3)
        palms = [fk.body(name) for name in PALM_BODIES]
        midpoint = np.mean([p.xpos for p in palms], axis=0)
        base_velocity = observed_velocity(poses)
        base_velocity = np.zeros(2) if base_velocity is None else base_velocity
        lead = float(self.profile.get('visual_aperture_lead_s', .04))
        offset = np.asarray(self.profile.get('visual_aperture_offset_body_m', (.08, 0., .04)), dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all() or not 0. <= lead <= .15:
            raise ValueError('invalid visual aperture target or prediction lead')
        relative_velocity = velocity - np.r_[base_velocity, 0.]
        error = ball.position_m - midpoint - pelvis_rotation @ offset + lead * relative_velocity
        error[2] *= float(self.profile.get('visual_aperture_z_gain', 0.))
        limit = float(self.profile.get('visual_aperture_max_shift_m', .10))
        if not 0. < limit <= .25:
            raise ValueError('visual aperture shift must be in (0, .25] metres')
        error *= min(1., limit / max(np.linalg.norm(error), 1e-9))
        jacobians = []
        for palm in palms:
            jacp, jacr = np.zeros((3, plant.model.nv)), np.zeros((3, plant.model.nv))
            mujoco.mj_jacBody(plant.model, fk, jacp, jacr, palm.id)
            jacobians.append(np.vstack((jacp[:, self.aperture_dofs], .15 * jacr[:, self.aperture_dofs])))
        jac = np.vstack(jacobians)
        goal = np.tile(np.r_[self.aperture_gain * error, np.zeros(3)], 2)
        correction = np.linalg.solve(jac.T @ jac + .001 * np.eye(14), jac.T @ goal)
        joint_limit = float(self.profile.get('visual_aperture_joint_limit_rad', .30))
        if not 0. < joint_limit <= .6:
            raise ValueError('visual aperture joint limit must be in (0, .6] radians')
        correction = np.clip(correction, -joint_limit, joint_limit)
        self.aperture_correction += np.clip(correction - self.aperture_correction, -.08, .08)
        self.aperture_trace.append(dict(time_s=now, ball_rgbd_m=ball.position_m.tolist(),
                                       palm_fk_m=midpoint.tolist(), error_m=error.tolist(),
                                       correction_rad=self.aperture_correction.tolist(),
                                       receiver_pose_time_s=pose.timestamp_s))

    def targets(self, now, ready):
        if self.hands.closed_at is None:
            pregrip = float(self.profile.get('pregrip_closure', 0.))
            if not np.isfinite(pregrip) or not 0. <= pregrip <= 1.:
                raise ValueError('pregrip closure must be in [0, 1]')
            return ready, pregrip if self.release is not None else 0.
        p = self.profile
        elapsed = now - self.hands.closed_at - p['grip_actuation_delay_s']
        grasp = np.clip((elapsed + p['grasp_joint_lead_s']) / p['grasp_ramp_s'], 0., 1.)
        hold = np.clip((elapsed - p['hold_delay_s']) / p['hold_ramp_s'], 0., 1.)
        target = ready.copy()
        if p['grasp_offsets_rad'] is not None:
            target[self.arm_indices] += grasp * np.asarray(p['grasp_offsets_rad'])
        if p['hold_offsets_rad'] is not None:
            target[self.arm_indices] += hold * np.asarray(p['hold_offsets_rad'])
        target[self.arm_indices] += self.aperture_correction
        closure = 0.
        if elapsed >= -1e-8:
            closure = p['closure']
            if p['hold_curls'] is not None:
                closure += hold * (np.asarray(p['hold_curls']) - closure)
        return target, closure


def run(plant, policy, heading, endpoint, viewer=None, delivery=None, horizon=11.,
        turn_xy=TURN, turn_ramp=.65, two_step=False, brake_x=8.33, brake_blend=.65,
        cut_lateral=0., cut_walk_s=0., cut_walk_vy=-.45, cut_walk_weight=1., cut_walk_vx=.65, cut_arm_override_fraction=1., cut_arm_recovery_s=0., disable_braking=False, post_cut_heading=None, spectator_states=None, staged_arm_offsets=None,
        arm_stage_after_turn_s=None, arm_stage_ramp_s=.7,
        receive_walk_weight=0., receive_walk_vx=1.5, receive_after_turn_s=3.2, residual_callback=None):
    upper = aero_ready_joint_targets(policy.nominal, plant.joint_names)
    arm_indices = [plant.joint_names.index(f'{side}_{axis}_joint') for side in ('left', 'right')
                   for axis in ('shoulder_roll', 'shoulder_yaw', 'shoulder_pitch', 'elbow',
                                'wrist_roll', 'wrist_pitch', 'wrist_yaw')]
    if staged_arm_offsets is not None:
        staged_arm_offsets = np.asarray(staged_arm_offsets, dtype=float)
        if staged_arm_offsets.shape != (14,) or not np.isfinite(staged_arm_offsets).all():
            raise ValueError('staged aperture requires fourteen finite arm offsets')
    parked = (18., 11., .085)
    ball_position = parked if delivery is None else delivery.initial_ball_position(parked)
    plant.reset(joint_positions=upper, release_position_m=ball_position)
    if delivery is not None:
        delivery.ready()
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
    camera = RearMarkerLocalizer(allow_synthetic_fixture=True, max_range_m=8.)
    foot = FootContactRecorder(plant.model)
    body = plant.data.body("pelvis")
    turn = brake = None
    brake_basis = None
    pose = None
    poses = []
    settled = False
    trace, frames, stamps = [], [], []
    def record(data):
        foot.record(data)
        if delivery is not None and delivery.release is not None:
            delivery.audit.record(data)

    if viewer is not None:
        spectator = mujoco.MjvCamera()
        spectator.azimuth, spectator.elevation, spectator.distance = 123., -12., 3.25
    final_tick = round(horizon / CONTROL_DT)
    for tick in range(final_tick + 1):
        now = plant.timestamp_s
        obs = RobotObservation(now, *plant.observe(), np.zeros(29), None)
        w, x, y, z = obs.imu_quat_wxyz
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        if spectator_states is not None and tick % 2 == 0:
            spectator_states.append((now, plant.data.qpos.copy()))
        if tick % 2 == 0:
            if two_step and .50 <= now < 2.08:
                cameras = ('break',)
            elif now >= 2.08:
                cameras = ("finish", "middle", "entry") if brake is not None else ("middle", "entry") if turn is not None else ("entry", "middle")
            else:
                cameras = ()
            for view in cameras:
                estimate = camera.estimate(*plant.capture(view))
                if estimate is not None:
                    pose = estimate
                    poses = [*poses[-(6 if delivery is not None else 4):], (now, *pose.xy_m)]
                    break
            if viewer is not None:
                spectator.lookat[:] = (*body.xpos[:2] + [.4, 0.], .78)
                viewer.update_scene(plant.data, camera=spectator)
                frames.append(viewer.render())
                stamps.append(now)
        if delivery is not None:
            delivery.observe(plant, now, pose, poses)
        fresh = pose is not None and pose.fresh(now, .14)
        if turn is None and fresh and np.linalg.norm(pose.xy_m - turn_xy) < (.075 if two_step else .15):
            turn = now
        steer = heading * np.clip(0. if turn is None else (now + CONTROL_DT - turn if two_step else now - turn) / turn_ramp, 0., 1.)
        if post_cut_heading is not None and turn is not None:
            post_u = np.clip((now - turn - cut_walk_s) / .45, 0., 1.)
            steer += (post_cut_heading - steer) * post_u * post_u * (3. - 2. * post_u)
        if not disable_braking and brake is None and now >= 4.24 and ((fresh and pose.xy_m[0] >= brake_x) or now >= 4.48):
            brake = now
            brake_basis = "camera" if fresh and pose.xy_m[0] >= brake_x else "timeout"
        elapsed = 0. if brake is None else now - brake
        u = float(np.clip(elapsed / brake_blend, 0., 1.))
        blend = u*u*(3.-2.*u)
        vx = min(2., 1.8*max(0.,now-.4))
        pulse = 0. if turn is None else np.clip((now + CONTROL_DT - turn) / .18, 0., 1.) * np.clip((turn + 1.25 - now) / .30, 0., 1.)
        amp_drive = np.array([vx*(1.-.4*u),cut_lateral*pulse,np.clip(1.8*(steer-yaw),-.75,.75)])
        absorb = .75*np.clip(1.-max(0.,elapsed-.65)/.8,0.,1.)
        walk = np.array([absorb,0.,np.clip(.8*(steer-yaw),-.28,.28) if absorb>.05 else 0.])
        cut_blend = 0.
        if cut_walk_s > 0. and turn is not None and brake is None:
            cut_elapsed = now + CONTROL_DT - turn
            cut_u = np.clip(min(cut_elapsed / .24, (cut_walk_s - cut_elapsed) / .34), 0., 1.)
            cut_blend = cut_walk_weight * cut_u * cut_u * (3. - 2. * cut_u)
            if cut_blend > 0.:
                walk = np.array([cut_walk_vx, cut_walk_vy, np.clip(1.4 * (steer - yaw), -.6, .6)])
        if elapsed >= 1.45:
            walk[:] = 0.
            if fresh:
                delta = endpoint-pose.xy_m
                velocity = observed_velocity(poses)
                if np.linalg.norm(delta)<.035 and velocity is not None and np.linalg.norm(velocity)<.15:
                    settled = True
                if not settled:
                    field = .85*delta
                    mag = np.linalg.norm(field)
                    if mag > 1e-8:
                        field *= np.clip(mag,.055,.33)/mag
                    c,s = np.cos(pose.yaw_rad),np.sin(pose.yaw_rad)
                    walk[:2] = np.array([[c,s],[-s,c]])@field
        receive_blend = 0.
        if receive_walk_weight > 0. and turn is not None and brake is None:
            receive_u = float(np.clip((now - turn - receive_after_turn_s) / .4, 0., 1.))
            receive_blend = receive_walk_weight * receive_u * receive_u * (3. - 2. * receive_u)
            if receive_blend > 0.:
                walk = np.array([receive_walk_vx, 0., np.clip(1.4 * (steer - yaw), -.6, .6)])
        blend = max(blend, cut_blend, receive_blend)
        trace.append((now,*body.xpos,body.xmat[8],*(pose.xy_m if fresh else (np.nan,np.nan)),steer,blend,yaw))
        if tick == final_tick or 1.-2.*(x*x+y*y)<.5:
            break
        prepared_upper = upper
        if staged_arm_offsets is not None and turn is not None:
            stage_start = cut_walk_s + cut_arm_recovery_s + .2 if arm_stage_after_turn_s is None else arm_stage_after_turn_s
            stage = float(np.clip((now - turn - stage_start) / arm_stage_ramp_s, 0., 1.))
            stage = stage * stage * (3. - 2. * stage)
            prepared_upper = upper.copy()
            prepared_upper[arm_indices] += stage * staged_arm_offsets
        target, closure = (prepared_upper, 0.) if delivery is None else delivery.targets(now, prepared_upper)
        if residual_callback is not None:
            if tick % 2 == 0:
                residual_view = "retention" if "retention" in plant.cameras and delivery is not None and delivery.hands.closed_at is not None else "pass"
                residual_frame, residual_survey = plant.capture(residual_view)
                if residual_view == "retention" and residual_detector.detect(residual_frame, residual_survey) is None:
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
        arm_fraction = 1. if turn is None else cut_arm_override_fraction
        if turn is not None and cut_arm_recovery_s > 0.:
            recover = np.clip((now - turn - cut_walk_s) / cut_arm_recovery_s, 0., 1.)
            recover = recover * recover * (3. - 2. * recover)
            arm_fraction += (1. - arm_fraction) * recover
        amp = policy.act(obs,SkillGoal(base_velocity=amp_drive),target, arm_override_fraction=arm_fraction)
        wbc = plant.wbc.act(obs,SkillGoal(base_velocity=walk),upper_target=target)
        plant.step(blend_pd(amp,wbc,blend),closure=closure,post_step=record)
    return np.array(trace),foot.arrays(),turn,brake,brake_basis,frames,stamps


def sustained_footfalls(time, contact, until):
    events = []
    for side, raw in zip(('left', 'right'), np.asarray(contact).T, strict=True):
        down = raw.copy()
        gaps = np.flatnonzero(np.diff(np.r_[True, down, True]))
        for start, stop in zip(gaps[::2], gaps[1::2], strict=True):
            if start > 0 and stop < len(time) and time[stop] - time[start] <= .045:
                down[start:stop] = True
        transitions = np.flatnonzero(np.diff(np.r_[False, down, False]))
        for start, stop in zip(transitions[::2], transitions[1::2], strict=True):
            if 0 < start < stop < len(time) and time[start] >= .5 and time[start] <= until + 1e-7 and time[stop] - time[start] >= .060:
                events.append((float(time[start]), side, float(time[stop] - time[start])))
    return sorted(events)


def score(trace, feet, turn, brake, brake_basis, endpoint, turn_xy=TURN):
    t=trace[:,0];xy=trace[:,1:3]; pre=np.linalg.norm(xy-turn_xy,axis=1); final=np.linalg.norm(xy[-1]-endpoint)
    tail=xy[t>=t[-1]-.5-1e-8]; stop=float(np.linalg.norm(np.diff(tail,axis=0),axis=1).sum()/.5)
    ftime=feet['foot_time_s']; fground=feet['foot_floor_contact']; rground=feet['robot_floor_contact']; clearance=(feet['sole_clearance_m']>=.005).all(axis=1)
    changes=np.flatnonzero(np.diff(np.r_[False,~rground,False])); phases=[]
    last_running_s = t[-1] if brake is None else brake
    for start,end in zip(changes[::2],changes[1::2],strict=True):
        if start==0 or end==len(ftime) or not 2. <= ftime[start] < ftime[end] < last_running_s:
            continue
        before,after=fground[start-1],fground[end]
        if before.sum()!=1 or after.sum()!=1 or np.array_equal(before,after):
            continue
        intervals=np.flatnonzero(np.diff(np.r_[False,clearance[start:end],False])); duration=(intervals[1::2]-intervals[::2]).max(initial=0)*.001
        if duration>=.025:
            phases.append((float(ftime[start]),float(duration)))
    before_heading = after_heading = before_after = None
    if turn is not None:
        def heading(start,end):
            displacement=np.array([np.interp(end,t,xy[:,axis])-np.interp(start,t,xy[:,axis]) for axis in (0,1)])
            return np.arctan2(displacement[1],displacement[0])
        before_heading = heading(turn-.70,turn-.12)
        after_heading = heading(turn+.55,turn+1.10)
        before_after = np.degrees(np.arctan2(np.sin(after_heading - before_heading), np.cos(after_heading - before_heading)))
    centerline_errors = []
    for start, finish in ((np.zeros(2), turn_xy), (turn_xy, endpoint)):
        direction = finish - start
        fraction = np.clip((xy - start) @ direction / (direction @ direction), 0., 1.)
        centerline_errors.append(np.linalg.norm(xy - start - fraction[:, None] * direction, axis=1))
    max_centerline = float(np.minimum(*centerline_errors).max())
    return dict(turn_time_s=turn,brake_time_s=brake,brake_basis=brake_basis,turn_xy_m=turn_xy.tolist(),closest_turn_m=float(pre.min()),
                pelvis_at_turn_m=None if turn is None else xy[np.argmin(abs(t-turn))].tolist(),
                sustained_footfalls_before_turn=[] if turn is None else sustained_footfalls(ftime, fground, turn),
                actual_finish_s=float(t[-1]),endpoint_xy_m=endpoint.tolist(),endpoint_error_m=float(final),
                final_pelvis_xyz_m=trace[-1,1:4].tolist(),final_speed_mps=stop,minimum_upright_cosine=float(trace[:,4].min()),
                max_centerline_error_m=max_centerline,
                travel_heading_before_deg=None if before_heading is None else float(np.degrees(before_heading)),
                travel_heading_after_deg=None if after_heading is None else float(np.degrees(after_heading)),
                motion_heading_change_deg=None if before_after is None else float(before_after),
                flights_s=phases, flights_after_turn=sum(turn is not None and time>turn for time,duration in phases),
                pass_bounded_course=bool(turn is not None and brake_basis=="camera" and pre.min()<=.15 and final<=.05
                                         and stop<=.075 and trace[:,4].min()>=.90 and t[-1]>=11.-1e-7
                                         and sum(time>turn for time,duration in phases)>=2))


def main(argv=None, residual_callback=None, sensor_retention_camera=None, sensor_retention_pose_camera=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path);parser.add_argument('--amp-source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--heading',type=float,default=.25)
    parser.add_argument('--endpoint',type=float,nargs=2,default=DEFAULT_FINISH);parser.add_argument('--record',action='store_true')
    parser.add_argument('--staged-arm-offsets', type=float, nargs=14, help='prepare an open aperture after cut recovery, fourteen offsets in L/R arm order')
    parser.add_argument('--arm-stage-after-turn-s', type=float)
    parser.add_argument('--arm-stage-ramp-s', type=float)
    parser.add_argument('--receive-walk-weight', type=float)
    parser.add_argument('--receive-walk-vx', type=float)
    parser.add_argument('--receive-after-turn-s', type=float)
    parser.add_argument('--record-poses', action='store_true', help='save exact physics states for separate stadium rendering')
    parser.add_argument('--pass-profile', type=Path, help='Go report containing the optical grip and grasp play')
    parser.add_argument('--launch-shift', type=float, nargs=3, default=(.006, -.018, -.003))
    parser.add_argument('--horizon', type=float, default=11.)
    parser.add_argument('--two-step', action='store_true', help='survey the break at the second sustained plant')
    parser.add_argument('--turn-ramp', type=float, default=.65)
    parser.add_argument('--brake-x', type=float, default=8.33)
    parser.add_argument('--brake-blend', type=float, default=.65)
    parser.add_argument('--cut-lateral', type=float, default=0., help='bounded lateral command during the two-step cut')
    parser.add_argument('--cut-walk-s', type=float, default=0., help='brief WBC cut before resuming AMP')
    parser.add_argument('--cut-walk-vy', type=float, default=-.45)
    parser.add_argument('--cut-walk-weight', type=float, default=1.)
    parser.add_argument('--disable-braking', action='store_true', help='continue running without the endpoint braking phase')
    parser.add_argument('--post-cut-heading', type=float, help='IMU yaw target after WBC cut, ramped over .45 seconds')
    parser.add_argument('--cut-arm-recovery-s', type=float, default=0., help='ramp catch-ready AMP arms back after WBC cut; zero keeps the selected fraction')
    parser.add_argument('--cut-arm-override-fraction', type=float, default=1., help='AMP arm replacement fraction after cut; zero restores learned arm motion')
    parser.add_argument('--cut-walk-vx', type=float, default=.65, help='WBC forward velocity during cut blend')
    args=parser.parse_args(argv)
    geometry = (args.horizon, args.turn_ramp, args.brake_x, args.brake_blend, *args.launch_shift)
    if min(geometry[:4]) <= 0 or not np.isfinite(geometry).all():
        parser.error('horizon must be positive and scenario geometry must be finite')
    if not -1.5 <= args.cut_lateral <= 1.5:
        parser.error('lateral cut command must be within ±1.5 m/s')
    if args.post_cut_heading is not None and not np.isfinite(args.post_cut_heading):
        parser.error('post-cut heading must be finite')
    if not 0. <= args.cut_arm_recovery_s <= 2.:
        parser.error('cut arm recovery duration must be in [0, 2]')
    if not 0. <= args.cut_arm_override_fraction <= 1.:
        parser.error('cut arm replacement fraction must be in [0, 1]')
    if not 0. <= args.cut_walk_vx <= 2.5:
        parser.error('cut WBC forward speed must be in [0, 2.5]')
    if not (0. <= args.cut_walk_s <= 2.5 and -1.5 <= args.cut_walk_vy <= 1.5 and 0. <= args.cut_walk_weight <= 1.):
        parser.error('cut walk duration and lateral speed exceed limits')
    profile = json.loads(args.pass_profile.read_text())['play'] if args.pass_profile else None
    settings = profile or {}
    staged = args.staged_arm_offsets if args.staged_arm_offsets is not None else settings.get('slant_staged_arm_offsets_rad')
    stage_after = args.arm_stage_after_turn_s if args.arm_stage_after_turn_s is not None else settings.get('slant_arm_stage_after_turn_s')
    stage_ramp = args.arm_stage_ramp_s if args.arm_stage_ramp_s is not None else settings.get('slant_arm_stage_ramp_s', .7)
    receive_weight = args.receive_walk_weight if args.receive_walk_weight is not None else settings.get('slant_receive_walk_weight', 0.)
    receive_vx = args.receive_walk_vx if args.receive_walk_vx is not None else settings.get('slant_receive_walk_vx', 1.5)
    receive_after = args.receive_after_turn_s if args.receive_after_turn_s is not None else settings.get('slant_receive_after_turn_s', 3.2)
    if not np.isfinite((stage_ramp, receive_weight, receive_vx, receive_after)).all() or not (.1 <= stage_ramp <= 2. and 0. <= receive_weight <= 1. and 0. <= receive_vx <= 2.5 and 0. <= receive_after <= 8.):
        parser.error('invalid staged-aperture or receive-handoff parameters')
    if stage_after is not None and (not np.isfinite(stage_after) or not 0. <= stage_after <= 8.):
        parser.error('staged-arm delay must be finite and in [0, 8]')
    if profile is not None and args.horizon < profile['release_s'] + CONTROL_DT:
        parser.error('pass run must continue past release')
    args.output.mkdir(parents=True,exist_ok=False); poses={}
    turn_xy = np.array([.87, .83]) if args.two_step else TURN
    def setup(scene):
        if residual_callback is not None and sensor_retention_pose_camera is not None:
            position, target = sensor_retention_pose_camera
            if not np.isfinite((*position, *target)).all():
                raise ValueError("finite surveyed residual pose camera required")
            poses["retention_pose"] = fixed_camera(scene, "retention_pose", position, target)
        if residual_callback is not None and sensor_retention_camera is not None:
            position, target = sensor_retention_camera
            if not np.isfinite((*position, *target)).all():
                raise ValueError("finite surveyed residual camera required")
            poses["retention"] = fixed_camera(scene, "retention", position, target)
        for name,pos,target in [('entry',(1.1,.45,1.28),(4.85,2.45,.85)),('middle',(4.2,1.35,1.36),(8.,3.2,.84)),('finish',(8.8,3.1,1.4),(13.1,args.endpoint[1],.81))]:
            poses[name]=fixed_camera(scene,name,pos,target)
        if profile is not None:
            shift = np.array([*args.launch_shift[:2], 0.])
            if profile.get('quarterback_origin_m') is None:
                position, aim = np.array([10.5, 3.65, 1.30]), np.array([7.85, 3., .91])
            else:
                position, aim = np.array([10.2, 5.5, 2.15]), np.array([7.25, 3.5, 1.48])
            position = np.asarray(profile.get('pass_camera_position_m', position))
            aim = np.asarray(profile.get('pass_camera_target_m', aim))
            poses['pass'] = fixed_camera(scene, 'pass', position + shift, aim + shift)
        if args.two_step:
            poses['break'] = fixed_camera(scene, 'break', (-1.4, -.60, 1.35), (.9, .82, .79))
    plant=HandRoutePlant(args.source,hand_model='aero',rear_markers=True,scene_setup=setup)
    plant.cameras.update(poses); backdrop=plant.model.geom('backdrop').id
    if plant.model.geom_contype[backdrop] or plant.model.geom_conaffinity[backdrop]:
        raise ValueError('backdrop must be visual only')
    plant.model.geom_rgba[backdrop,3]=0.;plant.model.geom_group[backdrop]=5
    policy=AmpRunningPolicy(plant.joint_names,plant.joint_ranges,args.amp_source);viewer=None
    delivery = OpticalPass(plant, profile, args.launch_shift) if profile is not None else None
    spectator_states = [] if args.record_poses else None
    try:
        if args.record:
            plant.model.vis.global_.offwidth=640;plant.model.vis.global_.offheight=360
            viewer=mujoco.Renderer(plant.model,height=360,width=640)
        result,foot,turn,brake,brake_basis,frames,stamps=run(plant,policy,args.heading,np.asarray(args.endpoint),
                                                         viewer=viewer,delivery=delivery,horizon=args.horizon,
                                                         turn_xy=turn_xy,turn_ramp=args.turn_ramp,two_step=args.two_step,
                                                         brake_x=args.brake_x,brake_blend=args.brake_blend,
                                                         cut_lateral=args.cut_lateral, cut_walk_s=args.cut_walk_s,
                                                         cut_walk_vy=args.cut_walk_vy, cut_walk_weight=args.cut_walk_weight, cut_walk_vx=args.cut_walk_vx,
                                                         cut_arm_override_fraction=args.cut_arm_override_fraction, cut_arm_recovery_s=args.cut_arm_recovery_s,
                                                         disable_braking=args.disable_braking, post_cut_heading=args.post_cut_heading,
                                                         spectator_states=spectator_states,
                                                         staged_arm_offsets=staged, arm_stage_after_turn_s=stage_after, arm_stage_ramp_s=stage_ramp,
                                                         receive_walk_weight=receive_weight, receive_walk_vx=receive_vx, receive_after_turn_s=receive_after, residual_callback=residual_callback)
        report=score(result,foot,turn,brake,brake_basis,np.asarray(args.endpoint),turn_xy);report['heading_target_rad']=args.heading
        report.update(staged_arm_offsets_rad=staged, arm_stage_after_turn_s=stage_after, arm_stage_ramp_s=stage_ramp,
                      receive_walk_weight=receive_weight, receive_walk_vx_mps=receive_vx, receive_after_turn_s=receive_after)
        report.update(turn_ramp_s=args.turn_ramp, brake_line_x_m=args.brake_x, brake_blend_s=args.brake_blend,
                      two_step_fixture=args.two_step, cut_lateral_mps=args.cut_lateral,
                      cut_walk_s=args.cut_walk_s, cut_walk_vy_mps=args.cut_walk_vy, cut_walk_weight=args.cut_walk_weight, cut_walk_vx_mps=args.cut_walk_vx,
                      cut_arm_override_fraction=args.cut_arm_override_fraction, cut_arm_recovery_s=args.cut_arm_recovery_s,
                      disable_braking=args.disable_braking, post_cut_heading_rad=args.post_cut_heading)
        report['warning_count']=sum(warning.number for warning in plant.data.warning)
        report['pass_bounded_course'] &= report['warning_count']==0
        traces = dict(motion=result, **foot)
        if delivery is not None:
            contact = delivery.audit.trace()
            traces.update(contact=contact, contact_columns=TRACE_COLUMNS)
            route = Route('slant', ((0., 0.), tuple(turn_xy), tuple(args.endpoint)), 2.)
            release_s = delivery.release.release_time_s if delivery.release else None
            strict = score_in_stride_catch(contact, result[:, :5], route, release_time_s=release_s,
                                          finish_time_s=round(args.horizon / CONTROL_DT) * CONTROL_DT,
                                          ball_weight_n=delivery.audit.ball_weight_n, verified_free_ball=True,
                                          physics_step_s=plant.model.opt.timestep,
                                          simulator_warning_count=report['warning_count'])
            running = score_running_long_pass(contact, result[:, :5], **foot, strict_catch_score=strict,
                                              release_time_s=release_s, forward_xy=(1., 0.), gait_start_s=.4,
                                              pass_toward_receiver_xy=(-1., 0.))
            minimum_hand_force = DextraRetentionCriteria().min_normal_per_hand_n
            contact_index = {name: index for index, name in enumerate(TRACE_COLUMNS)}
            left, right = (contact[:, [contact_index[f'{side}_{region}_normal_n'] for region in ('tip', 'palm')]].sum(axis=1)
                           for side in ('left', 'right'))
            bilateral = (left >= minimum_hand_force) & (right >= minimum_hand_force)
            intervals = np.flatnonzero(np.diff(np.r_[False, bilateral, False]))
            longest_bilateral_s = (intervals[1::2] - intervals[::2]).max(initial=0) * plant.model.opt.timestep
            report.update(profile_source=str(args.pass_profile), profile_sha256=sha256(args.pass_profile.read_bytes()).hexdigest(),
                          launch_shift_m=args.launch_shift, release=asdict(delivery.release) if delivery.release else None,
                          visual_closure_s=delivery.hands.closed_at, observed_ball=delivery.hands.trace,
                          visual_aperture_trace=delivery.aperture_trace,
                          longest_bilateral_palm_tip_contact_s=float(longest_bilateral_s),
                          strict_score=strict, running_score=running)
            if delivery.quarterback is not None:
                report['quarterback_pass_score'] = score_throw_receive(
                    contact, result[:, :5], report['release'], strict['first_tip_contact_s'],
                    physics_step_s=plant.model.opt.timestep)
        np.savez_compressed(args.output/'trace.npz',**traces)
        report['trace_sha256']=sha256((args.output/'trace.npz').read_bytes()).hexdigest()
        report['runner_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
        if spectator_states:
            pose_stamps, pose_qpos = zip(*spectator_states, strict=True)
            np.savez_compressed(args.output/'view.npz', timestamps_s=pose_stamps, qpos=pose_qpos,
                                **({'rgb': np.asarray(frames)} if frames else {}))
        elif frames:np.savez_compressed(args.output/'view.npz',rgb=frames,timestamps_s=stamps)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report),flush=True)
    finally:
        if viewer is not None:viewer.close()
        plant.close()

if __name__=='__main__':main()
