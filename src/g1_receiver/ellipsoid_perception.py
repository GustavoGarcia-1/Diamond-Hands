"""Experimental image-only football center fit; existing detector stays unchanged."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from .perception import ColorBallDetector, BallObservation

# Capture the original implementation so an isolated evaluator can install this
# candidate method without recursion or any simulator-state input.
ORIGINAL_COLOR_CANDIDATE = ColorBallDetector._candidate


@dataclass(frozen=True)
class EllipsoidFit:
    center_m: np.ndarray
    major_axis: np.ndarray
    rms_surface_error_m: float
    jacobian_condition: float
    point_count: int


def fit_visible_ellipsoid(points, initial_center, *, major_radius=.142, minor_radius=.085):
    """Fit the known axisymmetric ball to measured visible 3D surface points.

    The axis is sign-ambiguous. No orientation, center or contact truth is read.
    Fit residuals do not replace the sensor's declared uncertainty floor.
    """
    points, initial = np.asarray(points, float), np.asarray(initial_center, float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 40:
        return None
    if initial.shape != (3,) or not np.isfinite(points).all() or not np.isfinite(initial).all():
        return None
    if np.linalg.norm(initial) <= major_radius:
        return None
    if not 0 < minor_radius < major_radius:
        raise ValueError('positive ordered semiaxes required')
    points = points[np.linspace(0, len(points)-1, min(len(points), 180)).astype(int)]
    _, _, vectors = np.linalg.svd(points-points.mean(0), full_matrices=False)
    def axis(x):
        theta, phi = x[3:]
        return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    def residual(x):
        delta = points-x[:3]
        along = delta@axis(x)
        radius = np.sqrt(np.maximum(1e-12, np.sum(delta*delta, axis=1)/minor_radius**2
            + (1/major_radius**2-1/minor_radius**2)*along**2))
        return minor_radius*(radius-1.)
    best = None
    for seed in (vectors[0], vectors[1], initial/np.linalg.norm(initial)):
        start = np.r_[initial, np.arccos(np.clip(seed[2], -1, 1)), np.arctan2(seed[1], seed[0])]
        fit = least_squares(residual, start, bounds=(np.r_[initial-.12, 0., -2*np.pi],
            np.r_[initial+.12, np.pi, 2*np.pi]), max_nfev=60, loss='soft_l1', f_scale=.002,
            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        rms = float(np.sqrt(np.mean(residual(fit.x)**2)))
        singular = np.linalg.svd(fit.jac, compute_uv=False)
        condition = float(singular[0]/max(singular[-1], 1e-15))
        if fit.success and rms <= .006 and condition <= 1e6 and fit.x[2] > major_radius:
            result = EllipsoidFit(fit.x[:3].copy(), axis(fit.x), rms, condition, len(points))
            if best is None or result.rms_surface_error_m < best.rms_surface_error_m:
                best = result
    return best


class EllipsoidBallDetector(ColorBallDetector):
    """Optional center correction, preserving rejection and uncertainty floors.

    If the finite surface fit is not observable, retain the original estimate.
    No claim of reduced uncertainty or real-time hardware readiness is made.
    """
    def _candidate(self, frame, pixels, pose):
        original = ORIGINAL_COLOR_CANDIDATE(self, frame, pixels, pose)
        if original is None:
            return None
        score, observation = original
        rows, cols = pixels.T
        depths = frame.depth_m[rows, cols]
        valid = np.isfinite(depths) & (depths >= self.min_depth_m) & (depths <= self.max_depth_m)
        # Avoid edge pixels whose registered depth can belong to the background.
        locations = set(map(tuple, pixels.tolist()))
        interior = np.array([all((int(r)+dr, int(c)+dc) in locations for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)))
            for r,c in pixels])
        valid &= interior
        depth = depths[valid]
        points = np.c_[(cols[valid]-frame.intrinsics.cx)*depth/frame.intrinsics.fx,
                       (rows[valid]-frame.intrinsics.cy)*depth/frame.intrinsics.fy, depth]
        initial = pose.rotation.T@(observation.position_m-pose.translation)
        fit = fit_visible_ellipsoid(points, initial)
        if fit is None:
            return original
        position = pose.point(fit.center_m)
        correction = position-observation.position_m
        # Conservative engineering inflation, not a calibrated confidence claim.
        # A larger correction cannot make the measurement more certain.
        covariance = observation.covariance_m2 + np.outer(correction, correction)
        covariance += np.eye(3)*fit.rms_surface_error_m**2
        return score, BallObservation(frame.timestamp_s, pose.target_frame, position,
            covariance, pose.provenance)
