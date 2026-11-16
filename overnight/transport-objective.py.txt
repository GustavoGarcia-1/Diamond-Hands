"""Causal receiver translation for the precontact alignment experiment.

This is a separately versioned reward hypothesis. It estimates only horizontal
receiver motion from past camera poses. It does not predict future arm motion,
football orientation, contact, possession, or successful completion.
"""
from collections import deque
import json
from pathlib import Path

import numpy as np

from .precontact_native_rl import PrecontactNativeAudit, ALIGNMENT_WEIGHT, expected_alignment

OBS_SCHEMA = 'native-transported-intercept-169-v1'
REWARD_SCHEMA = 'transported-precontact-alignment-no-possession-v1'


class CameraTranslation:
    """Endpoint velocity from distinct causal camera poses in a surveyed field.

    The current adapter can reuse an optical frame label across physical views.
    This key separates frame labels, not guaranteed physical camera identities.
    Frozen November 15 jobs and their original metadata remain archived with
    the camera provenance erratum; this description corrects that metadata.

    A sum of endpoint standard deviations avoids assuming independent camera
    errors. The acceleration allowance is an engineering dispersion margin,
    not a calibrated probability or a verified hardware acceleration bound.
    """
    def __init__(self):
        self.histories = {}

    def update(self, now, pose, frame):
        if pose is None or frame is None:
            return None
        stamp = float(pose.timestamp_s)
        if (not -1e-8 <= now-stamp <= .08+1e-8
                or abs(stamp-float(frame.timestamp_s)) > .04+1e-8
                or pose.frame_id != 'field'):
            return None
        xy = np.asarray(pose.xy_m, float)
        covariance = np.asarray(pose.covariance, float)[:2, :2]
        if (xy.shape != (2,) or covariance.shape != (2, 2)
                or not np.isfinite(np.r_[xy, covariance.ravel()]).all()
                or not np.allclose(covariance, covariance.T)
                or np.linalg.eigvalsh(covariance).min() < -1e-10):
            return None
        key = (frame.frame_id, pose.frame_id, pose.pose_provenance)
        history = self.histories.setdefault(key, deque())
        if history and stamp < history[-1][0]-1e-10:
            return None
        if not history or stamp > history[-1][0]+1e-10:
            history.append((stamp, xy.copy(), covariance.copy()))
        while history and stamp-history[0][0] > .40+1e-8:
            history.popleft()
        if len(history) < 4:
            return None
        first = history[0]
        span = stamp-first[0]
        if span < .16-1e-8:
            return None
        velocity = (xy-first[1])/span
        if np.linalg.norm(velocity) > 3.:
            return None
        sigma = (np.sqrt(np.maximum(0, np.diag(covariance)))
                 + np.sqrt(np.maximum(0, np.diag(first[2]))))/span
        return dict(velocity_xy_mps=velocity, sigma_velocity_xy_mps=sigma,
                    span_s=span, sample_count=len(history), camera_frame=frame.frame_id,
                    first_capture_s=first[0], last_capture_s=stamp,
                    first_pose_xy_m=first[1].copy(), last_pose_xy_m=xy.copy(),
                    first_covariance_xy_m2=first[2].copy(), last_covariance_xy_m2=covariance.copy())


def transport_error(error, sigma, rotation, velocity, velocity_sigma, lead, time_std):
    """Move the observed receiver to arrival and conservatively enlarge dispersion."""
    error, sigma, rotation, velocity, velocity_sigma = [np.asarray(x, float)
        for x in (error, sigma, rotation, velocity, velocity_sigma)]
    if (error.shape != (3,) or sigma.shape != (3,) or rotation.shape != (3, 3)
            or velocity.shape != (2,) or velocity_sigma.shape != (2,)
            or not np.isfinite(np.r_[error, sigma, rotation.ravel(), velocity,
                                      velocity_sigma, lead, time_std]).all()
            or min(lead, time_std) < 0 or np.any(sigma < 0) or np.any(velocity_sigma < 0)):
        raise ValueError('finite causal transport quantities required')
    # Fixed before the experiment: 2 m/s² horizontal acceleration dispersion.
    translation_sigma = lead*velocity_sigma + lead*lead + np.abs(velocity)*time_std
    return (error-rotation[:, :2]@(velocity*lead),
            sigma+np.abs(rotation[:, :2])@translation_sigma)


class TransportPrecontactAudit(PrecontactNativeAudit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.translation = CameraTranslation()

    def _prediction(self, obs, pose, frame, survey, enabled):
        motion = self.translation.update(float(obs.timestamp), pose, frame)
        previous_count = len(self.camera_rows)
        previous_capture = self.last_camera
        old_credit = super()._prediction(obs, pose, frame, survey, enabled)
        if len(self.camera_rows) == previous_count or not self.camera_rows[-1]['accepted']:
            return old_credit
        row = self.camera_rows[-1]
        self.positive_credit -= old_credit
        row['untransported_error_m'] = self.predicted_error.tolist()
        row['untransported_sigma_m'] = self.predicted_sigma.tolist()
        row['untransported_alignment_proxy'] = row['alignment_proxy']
        row['positive_alignment_credit'] = row['unweighted_alignment_credit'] = 0.
        if motion is None:
            row.update(accepted=False, rejection_reason='receiver_translation_unavailable')
            self.intercept_valid = False
            self.cached_crossing = self.cached_capture = self.crossing = None
            return 0.
        crossing = self.crossing
        lead = max(0., float(crossing.timestamp_s-pose.timestamp_s))
        rotation = self.sensors.current_geometry['world_to_receiver']
        self.predicted_error, self.predicted_sigma = transport_error(
            self.predicted_error, self.predicted_sigma, rotation,
            motion['velocity_xy_mps'], motion['sigma_velocity_xy_mps'], lead,
            float(crossing.time_std_s))
        self.predicted_velocity -= rotation[:, :2]@motion['velocity_xy_mps']
        phi = expected_alignment(self.predicted_error, self.predicted_sigma)
        # Use the identical distinct capture weighting and absorbing deadline.
        gap = (0. if previous_capture is None else
               min(.08, max(0., float(frame.timestamp_s)-previous_capture)))
        if not enabled or self.arrival.closed:
            gap = 0.
        credit = ALIGNMENT_WEIGHT*gap*phi
        self.positive_credit += credit
        row.update(transport_lead_s=lead,
                   receiver_translation={k: v.tolist() if isinstance(v, np.ndarray) else v
                                         for k, v in motion.items()},
                   predicted_error_m=self.predicted_error.tolist(),
                   sigma_m=self.predicted_sigma.tolist(), alignment_proxy=phi,
                   unweighted_alignment_credit=gap*phi, positive_alignment_credit=credit)
        return credit

    def save(self, directory, terminal_bonus=0.):
        summary = super().save(directory, terminal_bonus)
        summary.update(observation_schema=OBS_SCHEMA, reward_schema=REWARD_SCHEMA,
            objective='Precontact relative alignment after camera estimated receiver translation; no possession certificate',
            uncertainty_method='Rotated marginal Gaussian alignment with conservative additive translation dispersion; no full covariance or calibrated confidence claim',
            receiver_translation='Past camera poses in a common surveyed field over0.16to0.40s; physical camera identity not enforced by generic frame labels; horizontal endpoint velocity; no future truth or command velocity substitute',
            translation_acceleration_dispersion_mps2=2.,
            future_arm_motion_predicted=False)
        (Path(directory)/'sensor-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        return summary
