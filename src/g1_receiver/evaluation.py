"""Evaluator-only truth metrics. Never import into a policy or reward pipeline.

Retention is an explicitly geometric test requiring physical contact evidence;
it is not a declaration of real grasp stability or hardware qualification.
"""
from dataclasses import dataclass
import math
from .routes import Route


@dataclass(frozen=True)
class EvaluationFrame:
    time_s: float
    receiver_xy: tuple[float, float]
    ball_in_capture_region: bool
    ball_ground_contact: bool
    ball_robot_contact: bool
    fallen: bool
    completed_steps: int
    # Populated by the runner integrity audit, never inferred from appearance.
    invalid_state_write: bool = False


def wilson_interval(successes, attempts, z=1.96):
    if attempts <= 0 or not 0 <= successes <= attempts:
        raise ValueError('Require 0 <= successes <= attempts and attempts > 0')
    p = successes/attempts
    denominator = 1+z*z/attempts
    center = (p+z*z/(2*attempts))/denominator
    margin = z*math.sqrt(p*(1-p)/attempts+z*z/(4*attempts*attempts))/denominator
    return max(0., center-margin), min(1., center+margin)


def evaluate_episode(frames: list[EvaluationFrame], route: Route, *, retention_s=2., subsequent_steps=3, finish_tolerance_m=0.35, max_frame_gap_s=0.1):
    if not frames:
        raise ValueError('Empty episode cannot be evaluated')
    if not all(math.isfinite(x) and x > 0 for x in (retention_s, finish_tolerance_m, max_frame_gap_s)) or subsequent_steps < 0:
        raise ValueError('Invalid evaluation thresholds')
    for i, f in enumerate(frames):
        if not math.isfinite(f.time_s) or f.time_s < 0 or type(f.completed_steps) is not int or f.completed_steps < 0:
            raise ValueError('Invalid timestamp or step count')
        if any(type(getattr(f, key)) is not bool for key in ('ball_in_capture_region', 'ball_ground_contact', 'ball_robot_contact', 'fallen', 'invalid_state_write')):
            raise ValueError('Contact and integrity flags must be booleans')
        if i and (f.time_s <= frames[i-1].time_s or f.completed_steps < frames[i-1].completed_steps):
            raise ValueError('Timestamps and step counter must be monotonic')
    distances = [route.project(f.receiver_xy)[0] for f in frames]
    max_gap = max((b.time_s-a.time_s for a, b in zip(frames, frames[1:])), default=0.)
    candidate = None
    for f in frames:
        if not f.ball_in_capture_region or f.ball_ground_contact or f.fallen:
            candidate = None
        elif candidate is None and f.ball_robot_contact:
            candidate = f
    retained = candidate is not None and frames[-1].time_s-candidate.time_s >= retention_s and frames[-1].completed_steps-candidate.completed_steps >= subsequent_steps
    fell = any(f.fallen for f in frames)
    integrity_passed = not any(f.invalid_state_write for f in frames) and max_gap <= max_frame_gap_s
    next_waypoint = 1
    for f in frames:
        if next_waypoint < len(route.waypoints):
            waypoint_tolerance = min(.1, math.dist(route.waypoints[next_waypoint-1], route.waypoints[next_waypoint])*.1)
            if math.dist(f.receiver_xy, route.waypoints[next_waypoint]) <= waypoint_tolerance:
                next_waypoint += 1
    route_passed = (math.dist(frames[0].receiver_xy, route.waypoints[0]) <= min(.1, finish_tolerance_m)
                    and next_waypoint == len(route.waypoints)
                    and max(distances) <= route.corridor_m
                    and math.dist(frames[-1].receiver_xy, route.waypoints[-1]) <= finish_tolerance_m)
    return {
        'schema': 'receiver-evaluation-v1',
        'success': retained and route_passed and not fell and integrity_passed,
        'physical_contact_observed': any(f.ball_robot_contact for f in frames),
        'retention_and_steps_passed': retained,
        'route_passed': route_passed,
        'fell': fell,
        'integrity_passed': integrity_passed,
        'max_route_error_m': max(distances),
        'rms_route_error_m': math.sqrt(sum(d*d for d in distances)/len(distances)),
        'max_frame_gap_s': max_gap,
        'duration_s': frames[-1].time_s-frames[0].time_s,
        'criteria': {'retention_s': retention_s, 'subsequent_steps': subsequent_steps, 'corridor_m': route.corridor_m, 'finish_tolerance_m': finish_tolerance_m, 'max_frame_gap_s': max_frame_gap_s},
    }
