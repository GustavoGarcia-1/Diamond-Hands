"""Offline scoring of physics contact and pelvis traces for an in-stride catch."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .dextra_physical_eval import DextraRetentionCriteria, GROUPS, TRACE_COLUMNS
from .routes import Route

MOTION_COLUMNS = ("timestamp_s", "pelvis_x_m", "pelvis_y_m", "pelvis_z_m", "upright_cosine")
CONTACT_INDEX = {column: index for index, column in enumerate(TRACE_COLUMNS)}
NORMAL_COLUMNS = [CONTACT_INDEX[f"{group}_normal_n"] for group in GROUPS]
BODY_CONTACT_GROUPS = [GROUPS.index("torso"), GROUPS.index("other")]
FLOOR_GROUP = GROUPS.index("floor")


@dataclass(frozen=True)
class InStrideCriteria:
    pre_contact_window_s: float = .14
    min_pre_contact_speed_mps: float = .10
    post_contact_window_s: float = .50
    min_immediate_route_displacement_m: float = .05
    min_total_route_displacement_m: float = .10
    waypoint_tolerance_cap_m: float = .05
    waypoint_tolerance_fraction: float = .10
    min_upright_cosine: float = .90
    settling_window_s: float = .50
    max_settled_speed_mps: float = .075
    max_motion_gap_s: float = .04
    min_initial_ball_height_m: float = .15
    force_zero_tolerance_n: float = 1e-9

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("scoring thresholds must be positive and finite")
        if self.min_upright_cosine > 1 or self.waypoint_tolerance_fraction >= .5:
            raise ValueError("invalid upright or waypoint threshold")


def _trace(rows, columns: tuple[str, ...], label: str) -> np.ndarray:
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != len(columns) or not np.isfinite(rows).all():
        raise ValueError(f"{label} trace must have finite values and the expected columns")
    if np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError(f"{label} timestamps must strictly increase")
    return rows


def _route_trace(route: Route, motion: np.ndarray, criteria: InStrideCriteria):
    points = np.asarray(route.waypoints)
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    directions = vectors / lengths[:, None]
    radii = np.minimum(criteria.waypoint_tolerance_cap_m, criteria.waypoint_tolerance_fraction * lengths)
    leg = 0
    hits = []
    progress = np.zeros(len(motion))
    xy = motion[:, 1:3]
    for index, now_s in enumerate(motion[:, 0]):
        if index:
            direction = directions[min(leg, len(lengths) - 1)]
            progress[index] = progress[index - 1] + (xy[index] - xy[index - 1]) @ direction
        if leg < len(lengths) and np.linalg.norm(xy[index] - points[leg + 1]) <= radii[leg]:
            leg += 1
            hits.append(float(now_s))
    return radii, hits, progress


def _position_at(motion: np.ndarray, at_s: float) -> np.ndarray:
    return np.array([np.interp(at_s, motion[:, 0], motion[:, axis]) for axis in (1, 2)])


def _retention(contact: np.ndarray, physics_step_s: float, ball_weight_n: float,
               criteria: InStrideCriteria, retention: DextraRetentionCriteria) -> dict:
    window_samples = round(retention.window_s / physics_step_s)
    if window_samples < 1:
        raise ValueError("physics step must be shorter than the retention window")
    rows = contact[-window_samples:]
    if not len(rows):
        return {
            "bilateral_palm_tip_fraction": None, "max_bilateral_gap_s": None,
            "mean_vertical_hand_force_as_ball_weight": None, "bilateral_at_finish": False,
            "hands_only_retention_at_finish": False,
        }
    normals = rows[:, NORMAL_COLUMNS]
    bilateral = (normals[:, :2].sum(axis=1) >= retention.min_normal_per_hand_n) & (
        normals[:, 3:5].sum(axis=1) >= retention.min_normal_per_hand_n
    )
    edges = np.flatnonzero(np.diff(np.r_[False, ~bilateral, False]))
    max_gap = float(np.max(edges[1::2] - edges[::2], initial=0) * physics_step_s)
    hand_vertical = rows[:, [CONTACT_INDEX[f"{group}_vertical_n"] for group in GROUPS[:6]]]
    vertical_fraction = float(hand_vertical.sum(axis=1).mean() / ball_weight_n)
    hand_only = (
        len(rows) == window_samples and bool(bilateral[-1])
        and bilateral.mean() >= retention.min_bilateral_fraction
        and max_gap <= retention.max_bilateral_gap_s + 1e-8
        and vertical_fraction >= retention.min_mean_hand_weight_fraction
        and not np.any(rows[:, [CONTACT_INDEX["ball_floor_contact"], CONTACT_INDEX["applied_ball_load"]]])
        and np.max(normals[:, 6:].sum(axis=1)) <= criteria.force_zero_tolerance_n
    )
    return {
        "bilateral_palm_tip_fraction": float(bilateral.mean()),
        "max_bilateral_gap_s": max_gap,
        "mean_vertical_hand_force_as_ball_weight": vertical_fraction,
        "bilateral_at_finish": bool(bilateral[-1]),
        "hands_only_retention_at_finish": bool(hand_only),
    }


def score_in_stride_catch(
    contact,
    motion,
    route: Route,
    *,
    release_time_s: float | None,
    finish_time_s: float,
    ball_weight_n: float,
    verified_free_ball: bool,
    physics_step_s: float = .001,
    simulator_warning_count: int = 0,
    criteria: InStrideCriteria = InStrideCriteria(),
    retention: DextraRetentionCriteria = DextraRetentionCriteria(),
) -> dict:
    """Score a complete release-to-finish contact trace and reset-to-finish pelvis trace.

    Use the named columns in TRACE_COLUMNS and MOTION_COLUMNS. The producer must
    verify the free ball joint and absence of a ball attachment. Neither trace
    provides control input.
    """
    contact = _trace(contact, TRACE_COLUMNS, "contact")
    motion = _trace(motion, MOTION_COLUMNS, "motion")
    if len(motion) < 2 or not np.isfinite((finish_time_s, ball_weight_n, physics_step_s)).all():
        raise ValueError("need a pelvis trace, a finite finish, ball weight and physics step")
    if min(ball_weight_n, physics_step_s) <= 0 or simulator_warning_count < 0:
        raise ValueError("invalid ball weight, physics step or warning count")
    if release_time_s is not None and (not np.isfinite(release_time_s) or not 0 <= release_time_s < finish_time_s):
        raise ValueError("release must precede the declared finish")
    time = motion[:, 0]
    radii, hits, progress = _route_trace(route, motion, criteria)
    final_error = float(np.linalg.norm(motion[-1, 1:3] - route.waypoints[-1]))
    initial_error = float(np.linalg.norm(motion[0, 1:3] - route.waypoints[0]))
    tail_start = time[-1] - criteria.settling_window_s
    tail_speed = None
    if tail_start >= time[0] - 1e-8:
        tail = np.vstack((_position_at(motion, tail_start), motion[time > tail_start, 1:3]))
        tail_speed = float(np.linalg.norm(np.diff(tail, axis=0), axis=1).sum() / criteria.settling_window_s)
    motion_complete = bool(
        abs(time[0]) <= 1e-7 and np.isclose(time[-1], finish_time_s, atol=1e-7, rtol=0)
        and np.max(np.diff(time)) <= criteria.max_motion_gap_s + 1e-8
    )
    route_pass = bool(
        motion_complete and initial_error <= radii[0] and len(hits) == len(radii)
        and final_error <= radii[-1] and tail_speed is not None
        and tail_speed <= criteria.max_settled_speed_mps and np.min(motion[:, 4]) >= criteria.min_upright_cosine
        and simulator_warning_count == 0
    )

    clocks_align = bool(
        len(contact) and release_time_s is not None
        and abs(contact[0, 0] - release_time_s - physics_step_s) <= 1e-7
        and abs(contact[-1, 0] - finish_time_s) <= 1e-7
        and np.allclose(np.diff(contact[:, 0]), physics_step_s, rtol=0, atol=1e-7)
    )
    normals = contact[:, NORMAL_COLUMNS]
    flags = contact[:, [CONTACT_INDEX["ball_floor_contact"], CONTACT_INDEX["applied_ball_load"]]]
    if np.any(normals < 0) or np.any((flags != 0) & (flags != 1)):
        raise ValueError("normal forces must be nonnegative and contact flags must be binary")
    floor_hits = (flags[:, 0] == 1) | (normals[:, FLOOR_GROUP] > criteria.force_zero_tolerance_n)
    external_load = bool(flags[:, 1].any())
    nonhand_body_peak = float(np.max(normals[:, BODY_CONTACT_GROUPS].sum(axis=1), initial=0))
    samples = max(1, round(retention.min_initial_free_flight_s / physics_step_s))
    airborne = bool(
        verified_free_ball and len(contact) >= samples and release_time_s is not None
        and abs(contact[0, 0] - release_time_s - physics_step_s) <= 1e-7
        and np.allclose(np.diff(contact[:samples, 0]), physics_step_s, rtol=0, atol=1e-7)
        and not np.any(flags[:samples]) and np.max(normals[:samples].sum(axis=1)) <= criteria.force_zero_tolerance_n
        and np.min(contact[:samples, CONTACT_INDEX["ball_z_m"]]) >= criteria.min_initial_ball_height_m
    )
    tip_hits = np.flatnonzero(np.max(normals[:, [0, 3]], axis=1) >= retention.min_normal_per_hand_n)
    tip_s = float(contact[tip_hits[0], 0]) if len(tip_hits) else None
    pre_speed, immediate_progress, total_progress = None, None, None
    if tip_s is not None and time[0] <= tip_s <= time[-1]:
        pre_index = np.searchsorted(time, tip_s - 1e-8, side="left") - 1
        pre_end = time[max(0, pre_index)]
        pre_start = pre_end - criteria.pre_contact_window_s
        if pre_start >= time[0] - 1e-8:
            pre_speed = float(np.linalg.norm(_position_at(motion, pre_end) - _position_at(motion, pre_start))
                              / criteria.pre_contact_window_s)
        at_tip = float(np.interp(tip_s, time, progress))
        total_progress = float(progress[-1] - at_tip)
        post_end = tip_s + criteria.post_contact_window_s
        if post_end <= time[-1] + 1e-8:
            immediate_progress = float(np.interp(post_end, time, progress) - at_tip)

    retention_result = _retention(contact, physics_step_s, ball_weight_n, criteria, retention)
    checks = {
        "complete_synchronized_traces": clocks_align and motion_complete,
        "initial_free_flight": airborne,
        "first_tip_contact": tip_s is not None,
        "pre_contact_speed": pre_speed is not None and pre_speed + 1e-8 >= criteria.min_pre_contact_speed_mps,
        "immediate_post_contact_progress": (
            immediate_progress is not None and immediate_progress + 1e-8 >= criteria.min_immediate_route_displacement_m
        ),
        "total_post_contact_progress": (
            total_progress is not None and total_progress + 1e-8 >= criteria.min_total_route_displacement_m
        ),
        "ordered_route_finish": route_pass,
        "hands_only_retention_at_finish": retention_result["hands_only_retention_at_finish"] and clocks_align,
        "no_floor_contact": not bool(floor_hits.any()),
        "no_nonhand_body_contact_force": nonhand_body_peak <= criteria.force_zero_tolerance_n,
        "no_applied_ball_load": not external_load,
    }
    first_floor = np.flatnonzero(floor_hits)
    return {
        "in_stride_catch_and_route": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "criteria": asdict(criteria),
        "retention_criteria": asdict(retention),
        "first_tip_contact_s": tip_s,
        "pre_tip_speed_mps": pre_speed,
        "immediate_post_tip_route_displacement_m": immediate_progress,
        "total_post_tip_route_displacement_m": total_progress,
        "ordered_waypoint_hit_times_s": hits,
        "endpoint_error_m": final_error,
        "endpoint_tolerance_m": float(radii[-1]),
        "final_mean_speed_mps": tail_speed,
        "first_floor_contact_s": float(contact[first_floor[0], 0]) if len(first_floor) else None,
        "peak_nonhand_body_normal_force_n": nonhand_body_peak,
        "retention": retention_result,
    }
