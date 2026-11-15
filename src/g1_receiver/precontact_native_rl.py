"""Opt-in precontact alignment PPO. No possession reward or privileged input.

The 169D vector reuses sensor scales, but slots 64:73 describe the predicted
intercept and its covariance, slot 76 and 159:161 are zero, and 161:169 contain
explicit precontact context. Native commands are never changed. Ball radius is
an alignment proxy, not measured orientation or proof of finite finger support.
"""
from pathlib import Path
import json
import numpy as np
from .native_cup_residual_rl import NativeCupSensors, scale_observation
from .lower_cup_grip import NativeHandCalibration
from .observable import ARM_JOINTS
from .aero_residual_rl import EpisodeReturnLedger
from .perception import BallTracker
from .catch_form import CatchFormContext, CatchFormEvaluator

ALIGNMENT_WEIGHT = 25.0  # Engineering reward scale, fixed before PPO training.
ACTION_SIZE = 14
OBS_SIZE = 169
OBS_SCHEMA = 'native-precontact-intercept-169-v2-uncertainty'
REWARD_SCHEMA = 'precontact-expected-alignment-no-possession-v2'
CONTEXT_FEATURES = ('enabled', 'deadline_closed', 'intercept_valid', 'deadline_known',
                    'arrival_delay_over_1p5s', 'arrival_std_over_0p06s',
                    'deadline_remaining_over_1p5s', 'alignment_proxy')

def expected_alignment(error, sigma):
    """Gaussian expected alignment, including uncertainty attenuation; not confidence calibration."""
    error,sigma=np.asarray(error,dtype=float),np.asarray(sigma,dtype=float)
    if error.shape!=(3,)or sigma.shape!=(3,)or not np.isfinite(np.r_[error,sigma]).all()or np.any(sigma<0):
        raise ValueError('finite position error and nonnegative standard deviation required')
    scale=np.array([.12,.08,.10]);variance=scale**2+sigma**2
    return float(np.prod(scale/np.sqrt(variance))*np.exp(-.5*np.sum(error**2/variance)))

class ArrivalDeadline:
    """An absorbing conservative estimate. Later fits cannot move it forward."""
    def __init__(self):
        self.deadline = None
        self.closed = False
    def update(self, now, arrival=None, std=None):
        if not np.isfinite(now):
            raise ValueError('finite control clock required')
        if not self.closed and arrival is not None:
            if not np.isfinite([arrival, std]).all() or std < 0:
                raise ValueError('finite arrival and nonnegative uncertainty required')
            candidate = float(arrival - 2.*std)
            self.deadline = candidate if self.deadline is None else min(self.deadline, candidate)
        if self.deadline is not None and now >= self.deadline - 1e-10:
            self.closed = True
        return not self.closed


def project_residual(action, previous, nominal, ranges, dt=.02):
    """Enforce bounds on the delivered residual, without a hidden final clamp."""
    action, previous, nominal = [np.asarray(x, dtype=float) for x in (action, previous, nominal)]
    ranges = np.asarray(ranges, dtype=float)
    if any(x.shape != (14,) for x in (action, previous, nominal)) or ranges.shape != (14, 2):
        raise ValueError('fourteen arm channels required')
    if not np.isfinite(np.r_[action, previous, nominal, ranges.ravel(), dt]).all() or dt <= 0 or np.any(abs(action) > 1.+1e-8):
        raise ValueError('finite causal bounded arm request required')
    low = np.maximum.reduce([np.full(14, -.18), ranges[:, 0]-nominal, previous-1.5*dt])
    high = np.minimum.reduce([np.full(14, .18), ranges[:, 1]-nominal, previous+1.5*dt])
    if np.any(low > high + 1e-10):
        raise ValueError('moving nominal makes delivered residual constraints infeasible')
    return np.minimum(np.maximum(action*.18, low), high)


class PrecontactNativeAudit:
    def __init__(self, profile, policy=None, radius_m=.1135):
        self.profile, self.policy, self.radius_m = profile, policy, radius_m
        play = profile['play']; settings = profile.get('precontact', {})
        point = settings.get('plane_point_m', play.get('field_catch_window_m', play.get('quarterback_target_m')))
        if point is None:
            raise ValueError('precontact requires a declared surveyed plane_point_m')
        self.plane_point = np.asarray(point, dtype=float)
        axis = settings.get('plane_normal', play.get('field_approach_axis_xy'))
        if axis is None:
            origin = play.get('quarterback_origin_m')
            axis = np.asarray(point)[:2]-np.asarray(origin)[:2] if origin is not None else (1., 0.)
        axis = np.asarray(axis, dtype=float)
        self.plane_normal = np.r_[axis, 0.] if axis.shape == (2,) else axis
        if self.plane_point.shape != (3,) or self.plane_normal.shape != (3,) or not np.isfinite(np.r_[self.plane_point,self.plane_normal]).all() or np.linalg.norm(self.plane_normal) < 1e-9:
            raise ValueError('finite declared plane required')
        self.plane_normal /= np.linalg.norm(self.plane_normal)
        self.tracker = BallTracker(max_age_s=.08+1e-8, max_horizon_s=1.5, allow_synthetic_fixture=True)
        self.arrival = ArrivalDeadline(); self.sensors = None; self.calibration = None
        self.action = np.zeros(14); self.actual = np.zeros(14)
        self.ledger = EpisodeReturnLedger(); self.pending_reward = 0.
        self.calls = self.ticks = self.steps = self.disabled_ticks = 0
        self.first_enabled_s = self.last_time = self.last_camera = None
        self.rows = []; self.camera_rows = []; self.form_rows = []
        self.total_form_cost = self.positive_credit = self.positive_after_deadline = 0.
        self.finished = False; self.intercept_valid = False
        self.predicted_error = np.zeros(3); self.predicted_velocity = np.zeros(3); self.predicted_sigma = np.zeros(3)
        self.crossing = None
        self.cached_crossing = None; self.cached_capture = None

    def bind(self, model, joint_names):
        self.sensors = NativeCupSensors(model, joint_names, radius_m=self.radius_m)
        self.calibration = NativeHandCalibration.from_model(model)
        self.arms = np.array([joint_names.index(n) for n in ARM_JOINTS])
        self.ranges = np.array([model.joint(n).range.copy() for n in ARM_JOINTS])
        self.form = CatchFormEvaluator(model, joint_names)

    def _prediction(self, obs, pose, frame, survey, enabled):
        now = float(obs.timestamp); self.intercept_valid = False; self.crossing = None
        if frame is None or survey is None:
            self._reuse_prediction(now)
            return 0.
        stamp = float(frame.timestamp_s)
        if self.last_camera is not None and stamp <= self.last_camera + 1e-10:
            self._reuse_prediction(now)
            return 0.
        previous = self.last_camera; self.last_camera = stamp
        self.cached_crossing = None; self.cached_capture = None
        row = dict(time_s=now, capture_s=stamp, enabled=bool(enabled), accepted=False,
                   ball_detected=False, tracker_added=False, positive_alignment_credit=0.,
                   rejection_reason=None)
        self.camera_rows.append(row)
        def reject(reason):
            row['rejection_reason'] = reason
            return 0.
        if not -1e-8 <= now-stamp <= .08+1e-8:
            return reject('stale_or_future_capture')
        ball = self.sensors.detector.detect(frame, survey)
        if ball is None:
            return reject('no_rgbd_ball')
        row.update(ball_detected=True, ball_position_m=np.asarray(ball.position_m).tolist(),
                   ball_covariance_m2=np.asarray(ball.covariance_m2).tolist())
        if ball.frame_id != 'field':
            return reject('ball_frame_not_field')
        # Forecast history must not depend on old possession/cup covariance gates.
        if not self.tracker.add(ball):
            return reject('tracker_rejected_ball')
        row['tracker_added'] = True
        crossing = self.tracker.intercept(now, self.plane_point, self.plane_normal, 'field')
        if crossing is None:
            return reject('no_intercept_fit')
        row.update(arrival_s=float(crossing.timestamp_s), arrival_std_s=float(crossing.time_std_s),
                   intercept_position_m=np.asarray(crossing.position_m).tolist(),
                   intercept_covariance_m2=np.asarray(crossing.covariance_m2).tolist())
        if crossing.last_inlier_timestamp_s != stamp:
            return reject('current_capture_not_inlier')
        if crossing.velocity_mps is None or crossing.velocity_mps[2] >= 0 or crossing.velocity_mps @ self.plane_normal < 2.5:
            return reject('not_descending_incoming')
        if not np.isfinite(crossing.time_std_s) or not 0 <= crossing.time_std_s <= .06:
            return reject('arrival_uncertainty')
        self.arrival.update(now, crossing.timestamp_s, crossing.time_std_s)
        row.update(deadline_s=self.arrival.deadline, deadline_closed=self.arrival.closed)
        if pose is None or pose.z_m is None or pose.z_std_m is None:
            return reject('receiver_pose_missing')
        row.update(pose_time_s=float(pose.timestamp_s), pose_covariance=np.asarray(pose.covariance).tolist(),
                   pose_z_m=float(pose.z_m), pose_z_std_m=float(pose.z_std_m))
        if (not pose.fresh(now,.08+1e-8) or abs(stamp-pose.timestamp_s)>.04+1e-8
                or not np.isfinite([pose.z_m,pose.z_std_m]).all() or pose.z_std_m<0):
            return reject('receiver_pose_stale_skew_or_invalid')
        if pose.frame_id!='field' or ball.pose_provenance!=pose.pose_provenance:
            return reject('receiver_frame_or_provenance_mismatch')
        cov=np.array(crossing.covariance_m2,copy=True)
        cov[:2,:2]+=np.asarray(pose.covariance)[:2,:2];cov[2,2]+=pose.z_std_m**2
        cov+=np.eye(3)*.01**2
        if not np.isfinite(cov).all() or not np.allclose(cov,cov.T) or np.linalg.eigvalsh(cov).min() < -1e-10:
            return reject('invalid_covariance')
        geometry=self.sensors.current_geometry;Y=geometry['world_to_receiver']
        self.predicted_sigma=np.sqrt(np.maximum(0.,np.diag(Y@cov@Y.T)))
        self.predicted_error=Y@(crossing.position_m-np.r_[pose.xy_m,pose.z_m])-geometry['cup_receiver']
        self.predicted_velocity=Y@crossing.velocity_mps
        self.intercept_valid=True;self.crossing=crossing
        self.cached_crossing=crossing;self.cached_capture=stamp
        phi=expected_alignment(self.predicted_error,self.predicted_sigma)
        gap=0. if previous is None else min(.08,max(0.,stamp-previous))
        unweighted=gap*phi if enabled and not self.arrival.closed else 0.
        credit=ALIGNMENT_WEIGHT*unweighted
        row.update(accepted=True,rejection_reason=None if enabled and not self.arrival.closed else
                   'deadline_closed' if self.arrival.closed else 'residual_disabled',
                   predicted_error_m=self.predicted_error.tolist(),sigma_m=self.predicted_sigma.tolist(),
                   alignment_proxy=phi,unweighted_alignment_credit=unweighted,positive_alignment_credit=credit)
        self.positive_credit+=credit
        if self.arrival.closed:self.positive_after_deadline+=credit
        return credit

    def _reuse_prediction(self, now):
        if (self.cached_crossing is not None
                and -1e-8 <= now-self.cached_capture <= .08+1e-8):
            self.crossing = self.cached_crossing
            self.intercept_valid = True

    def _vector(self, obs, native_encoders, native_target, enabled):
        now = float(obs.timestamp); crossing = self.crossing
        lead = 0. if crossing is None else crossing.timestamp_s-now
        std = 0. if crossing is None else crossing.time_std_s
        remaining = 0. if self.arrival.deadline is None else self.arrival.deadline-now
        phi = 0. if not self.intercept_valid else expected_alignment(self.predicted_error,self.predicted_sigma)
        context = np.array([float(enabled),float(self.arrival.closed),float(self.intercept_valid),
                            float(self.arrival.deadline is not None),np.clip(lead/1.5,-1,1),
                            std/.06,np.clip(remaining/1.5,-1,1),phi])
        raw = self.sensors.vector(obs,np.r_[self.action,np.zeros(6)],np.r_[self.actual,np.zeros(6)],
                                  native_encoders,native_target,self.calibration,context)
        raw[64:67] = self.predicted_error if self.intercept_valid else 0.
        raw[67:70] = self.predicted_velocity if self.intercept_valid else 0.
        raw[70:73] = self.predicted_sigma if self.intercept_valid else 0.
        raw[73] = 1. if self.cached_capture is None else np.clip(now-self.cached_capture,0.,1.)
        raw[74:76] = float(self.intercept_valid); raw[76] = 0.; raw[159:161] = 0.
        return raw, scale_observation(raw)

    def __call__(self, obs, pose, frame, survey, target, native_target, native_encoders, enabled):
        if self.sensors is None: raise RuntimeError('static model bind required')
        now = float(obs.timestamp); dt = .02 if self.last_time is None else now-self.last_time
        if not np.isfinite(dt) or not 0 < dt <= .041: raise ValueError('causal 50 Hz callback required')
        self.last_time = now; self.calls += 1
        target = np.asarray(target,dtype=float); native = np.asarray(native_target,dtype=float)
        enc = np.asarray(native_encoders,dtype=float)
        if target.shape != (29,) or native.shape != (14,) or enc.shape != (14,) or not np.isfinite(np.r_[target,native,enc]).all():
            raise ValueError('finite body/native sensor and command packets required')
        self.sensors.update(obs,pose,frame,survey)
        # Legacy preprocessing may compute its own reward: never consume it.
        self.sensors.reward = 0.; self.sensors.dwell = 0.; self.sensors.paid[:] = 0.
        self.sensors.form_acquired = False; self.sensors.potential = None
        self.arrival.update(now)
        credit = self._prediction(obs,pose,frame,survey,bool(enabled))
        context = CatchFormContext(over_shoulder=True,phase='receive' if enabled else 'inactive')
        if self.intercept_valid and frame is not None and pose is not None:
            c=self.crossing; initial=self.form.evaluate(obs,CatchFormContext())
            context=CatchFormContext(over_shoulder=True,phase='receive',perception_timestamp_s=float(frame.timestamp_s),
                causal_fit_valid=True,predicted_intercept_height_above_shoulder_m=float(c.position_m[2]-pose.z_m-initial['shoulder_midpoint_height_above_pelvis_m']),
                intercept_height_std_m=float(np.sqrt(c.covariance_m2[2,2]+pose.z_std_m**2+.03**2)),
                time_to_arrival_s=float(c.timestamp_s-frame.timestamp_s),arrival_std_s=float(c.time_std_s))
        form = self.form.evaluate(obs,context); cost = dt*float(form['penalty_per_s']) if enabled else 0.
        self.total_form_cost += cost
        self.form_rows.append(dict(time_s=now,residual_enabled=bool(enabled),reward_debit=cost,**form))
        if not enabled:
            self.disabled_ticks += 1
            return target.copy(),native.copy()
        if self.first_enabled_s is None: self.first_enabled_s=now
        self.pending_reward += credit-cost-dt*2.*max(0.,.9-self.sensors.upright)
        if self.ticks%2 == 0:
            raw, vector = self._vector(obs,enc,native,enabled)
            reward = float(self.pending_reward); self.pending_reward = 0.
            reset = len(self.rows)==0; self.ledger.record(reward,reset=reset)
            request = not self.arrival.closed or self.steps==0
            if request:
                action = np.zeros(14) if self.policy is None else np.asarray(self.policy(vector.copy()),dtype=float)
                if action.shape != (14,) or not np.isfinite(action).all() or np.any(abs(action)>1.+1e-8):
                    raise ValueError('policy must return fourteen normalized arm actions')
                if not self.arrival.closed: self.action = action.copy()
                self.steps += 1
            self.rows.append(dict(time_s=now,reward=reward,reset_reward=reset,raw=raw,normalized=vector,
                                  action=self.action.copy(),rate_limited_residual=self.actual.copy(),
                                  deadline_s=self.arrival.deadline,deadline_closed=self.arrival.closed,
                                  intercept_valid=self.intercept_valid,policy_requested=request,
                                  positive_alignment_credit=credit))
        old = self.actual.copy()
        if not self.arrival.closed:
            self.actual = project_residual(self.action,self.actual,target[self.arms],self.ranges,dt)
        elif np.any(target[self.arms]+self.actual < self.ranges[:,0]-1e-10) or np.any(target[self.arms]+self.actual > self.ranges[:,1]+1e-10):
            raise ValueError('held postarrival residual violates moving nominal range')
        result = target.copy(); result[self.arms] += self.actual
        self.pending_reward -= dt*(.05*float(np.mean(self.action**2))+.02*float(np.mean(((self.actual-old)/dt)**2)))
        self.ticks += 1
        return result,native.copy()

    def finish(self):
        if self.finished: raise RuntimeError('finish must be settled once')
        self.finished=True; self.ledger.record(float(self.pending_reward)); self.pending_reward=0.
        return 0.

    def save(self, directory, terminal_bonus=0.0):
        if terminal_bonus != 0.: raise ValueError('precontact has no terminal possession bonus')
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
        def matrix(key,n): return np.asarray([r[key] for r in self.rows],dtype=float).reshape(-1,n)
        np.savez_compressed(directory/'native-policy-trace.npz',timestamp_s=np.array([r['time_s'] for r in self.rows]),
            raw_observation=matrix('raw',169),normalized_observation=matrix('normalized',169),
            action=matrix('action',14),rate_limited_residual=matrix('rate_limited_residual',14))
        serial=[{k:v for k,v in r.items() if k not in ('raw','normalized','action','rate_limited_residual')} for r in self.rows]
        (directory/'sensor-samples.json').write_text(json.dumps(dict(policy_samples=serial,camera_samples=self.camera_rows),indent=2)+'\n')
        (directory/'form-samples.json').write_text(json.dumps(self.form_rows,indent=2)+'\n')
        summary=dict(observation_schema=OBS_SCHEMA,reward_schema=REWARD_SCHEMA,observation_size=169,action_size=14,
            arm_action_order=list(ARM_JOINTS),context_features=CONTEXT_FEATURES,policy_steps=self.steps,
            logged_samples=len(self.rows),callback_calls=self.calls,first_enabled_s=self.first_enabled_s,
            sensor_return=self.ledger.step_return,reset_sensor_reward=self.ledger.reset_reward,
            sensor_return_including_reset=self.ledger.step_return+self.ledger.reset_reward,
            terminal_sensor_bonus=0.,alignment_weight=ALIGNMENT_WEIGHT,
            unweighted_positive_alignment_credit=self.positive_credit/ALIGNMENT_WEIGHT,positive_alignment_credit=self.positive_credit,
            positive_credit_after_deadline=self.positive_after_deadline,deadline_s=self.arrival.deadline,
            deadline_closed=self.arrival.closed,total_form_cost=self.total_form_cost,
            camera_capture_count=len(self.camera_rows),tracker_capture_count=sum(r['tracker_added'] for r in self.camera_rows),
            accepted_alignment_captures=sum(r['accepted'] for r in self.camera_rows),
            rejection_counts={reason:sum(r['rejection_reason']==reason for r in self.camera_rows) for reason in sorted({r['rejection_reason'] for r in self.camera_rows if r['rejection_reason'] is not None})},
            uncertainty_method='Expected Gaussian alignment with full logged sensor covariance, goal scales .12/.08/.10 m; no inherited possession-specific 60 mm position cutoff. Arrival std remains capped at .06 s.',
            native_targets_unchanged=True,postarrival_action='hold last delivered arm residual',
            objective='Precontact intercept alignment proxy, not possession or physical catch certification',
            runtime_inputs=['RGBD ball and receiver estimates','IMU','29 body encoders','14 native encoders','known commands'],
            excluded=['plant root or ball state','contact','unobserved finger angles','physical success scores'])
        (directory/'sensor-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        return summary
