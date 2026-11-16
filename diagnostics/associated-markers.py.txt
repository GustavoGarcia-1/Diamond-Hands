"""Versioned RGB-D rear-marker association using the surveyed rigid rig.

Color is a proposal, not an identity: football highlights can pass the yellow
sphere detector. Only a unique pair consistent with the existing 22 cm rig is
accepted. This module consumes images and a fixed camera survey, never plant
state. The original localizer and its historical behavior remain unchanged.
"""
from dataclasses import replace
import numpy as np
from .localization import MarkerLocalizer, LEFT_MARKER_M, RIGHT_MARKER_M
from .perception import _components

ASSOCIATION_SCHEMA = 'rear-marker-rigid-pair-v1'


class AssociatedRearMarkerLocalizer(MarkerLocalizer):
    """Associate first, then reuse the original pose and covariance estimator."""

    def estimate(self, frame, camera_pose):
        if camera_pose.source_frame != frame.frame_id:
            raise ValueError('camera survey source must match the RGB-D optical frame')
        if camera_pose.provenance == 'synthetic_fixture':
            if not self.allow_synthetic_fixture:
                raise ValueError('synthetic calibration requires explicit fixture opt-in')
        elif camera_pose.provenance != 'fixed_calibration':
            raise ValueError('marker localizer requires a fixed camera survey')
        self.last_diagnostics = dict(schema=ASSOCIATION_SCHEMA, accepted=False,
                                     unique_compatible_pair_required=True)
        if abs(frame.timestamp_s-camera_pose.timestamp_s) > self.max_pose_skew_s:
            self.last_diagnostics['reason'] = 'stale_survey'
            return None
        red, green, blue = np.moveaxis(frame.rgb.astype(np.int16), -1, 0)
        left = (blue >= 100) & (blue-red >= 60) & (blue-green >= 60)
        right = (red >= 110) & (green >= 110) & (red-blue >= 55) & (green-blue >= 55)
        if any(np.count_nonzero(mask) > mask.size//10 for mask in (left, right)):
            self.last_diagnostics['reason'] = 'excess_color_area'
            return None
        candidates=[]
        for mask in (left, right):
            side=[]
            for pixels in _components(mask):
                # Reuse ALL original blob shape, border, depth, size and sphere
                # tests on each component, without changing their thresholds.
                isolated=np.zeros_like(mask)
                isolated[tuple(pixels.T)]=True
                center=self._center(frame,isolated)
                if center is not None:
                    side.append((isolated,camera_pose.point(center)))
            candidates.append(side)
        span=LEFT_MARKER_M[1]-RIGHT_MARKER_M[1]
        pairs=[]; measurements=[]
        for left_index, (left_mask,left_field) in enumerate(candidates[0]):
            for right_index, (right_mask,right_field) in enumerate(candidates[1]):
                baseline=left_field-right_field
                distance=float(np.linalg.norm(baseline))
                horizontal=float(np.linalg.norm(baseline[:2]))
                compatible=abs(distance-span)<=.045 and horizontal>=.85*span
                measurements.append(dict(left=left_index,right=right_index,
                                         span_m=distance,horizontal_span_m=horizontal,
                                         compatible=bool(compatible)))
                if compatible:pairs.append((left_mask,right_mask))
        self.last_diagnostics.update(left_candidates=len(candidates[0]),
                                     right_candidates=len(candidates[1]),
                                     compatible_pairs=len(pairs),pairs=measurements)
        if len(pairs)!=1:
            self.last_diagnostics['reason']='missing_or_ambiguous_rigid_pair'
            return None
        markers=np.zeros_like(frame.rgb)
        markers[pairs[0][0]]=(0,255,255)
        markers[pairs[0][1]]=(255,0,255)
        pose=super().estimate(replace(frame,rgb=markers),camera_pose)
        if pose is None:
            self.last_diagnostics['reason']='original_pose_estimator_rejected_pair'
            return None
        heading=np.array([np.cos(pose.yaw_rad),np.sin(pose.yaw_rad)])
        reflection=np.diag([1.,1.,-1.])
        self.last_diagnostics.update(accepted=True,reason='unique_rigid_pair')
        return replace(pose,xy_m=pose.xy_m+.30*heading,
                       covariance=reflection@pose.covariance@reflection)
