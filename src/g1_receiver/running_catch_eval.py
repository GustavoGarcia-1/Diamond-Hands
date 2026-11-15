"""Optional running and longer-pass checks for an independently scored catch."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .in_stride_eval import CONTACT_INDEX, MOTION_COLUMNS, NORMAL_COLUMNS, TRACE_COLUMNS, _trace


class FootContactRecorder:
    """Sample sole sphere clearance and floor constraints after each physics step."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.floor = model.geom("floor").id
        if model.geom_type[self.floor] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError("sole clearance requires a planar floor")
        self.ball_body = model.body("football").id
        feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
        self.soles = [
            np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype != 0))
            for body in feet
        ]
        if any(
            not len(group) or np.any(model.geom_type[group] != mujoco.mjtGeom.mjGEOM_SPHERE)
            for group in self.soles
        ):
            raise ValueError("each ankle must have identifiable spherical sole colliders")
        self.foot_side = {
            int(geom): side for side, group in enumerate(self.soles) for geom in group
        }
        self._rows = []

    def record(self, data: mujoco.MjData) -> None:
        """Record the solver configuration at the start of the completed step."""
        feet = np.zeros(2, dtype=bool)
        robot_ground = False
        for contact in data.contact:
            one, two = contact.geom
            if contact.efc_address < 0 or self.floor not in (one, two):
                continue
            other = two if one == self.floor else one
            if self.model.geom_bodyid[other] in (0, self.ball_body):
                continue
            robot_ground = True
            side = self.foot_side.get(other)
            if side is not None:
                feet[side] = True
        normal = data.geom_xmat[self.floor].reshape(3, 3)[:, 2]
        floor_origin = data.geom_xpos[self.floor]
        clearance = [
            np.min((data.geom_xpos[group] - floor_origin) @ normal - self.model.geom_size[group, 0])
            for group in self.soles
        ]
        self._rows.append((data.time - self.model.opt.timestep, *feet, robot_ground, *clearance))

    def arrays(self) -> dict:
        rows = np.asarray(self._rows, dtype=float).reshape(-1, 6)
        return {
            "foot_time_s": rows[:, 0],
            "foot_floor_contact": rows[:, 1:3].astype(bool),
            "robot_floor_contact": rows[:, 3].astype(bool),
            "sole_clearance_m": rows[:, 4:6],
        }


@dataclass(frozen=True)
class RunningCatchCriteria:
    min_forward_speed_mps: float = .80
    min_flight_s: float = .025
    min_sole_clearance_m: float = .005
    min_ground_interval_s: float = .035
    min_ground_fraction: float = .80
    min_recurrent_flights: int = 2
    max_neighbor_flight_spacing_s: float = .72
    gait_window_s: float = .75
    startup_exclusion_s: float = .35
    terminal_exclusion_s: float = .40
    max_foot_sample_gap_s: float = .005
    max_motion_gap_s: float = .04
    min_ball_flight_s: float = .18
    min_ball_forward_travel_m: float = .25
    min_ball_rise_m: float = .03
    min_ball_descent_m: float = .015
    ball_descent_window_s: float = .04
    max_ball_sample_gap_s: float = .005

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("running and pass thresholds must be positive and finite")
        if not isinstance(self.min_recurrent_flights, int) or self.min_recurrent_flights < 2:
            raise ValueError("recurrent running requires at least two flights")
        if self.min_ground_fraction > 1:
            raise ValueError("grounded fraction must be at most one")


def _runs(mask):
    changes = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(changes[::2], changes[1::2], strict=True)


def _binary(values, shape, name):
    array = np.asarray(values)
    if array.shape != shape or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain one Boolean per required sample")
    return array.astype(bool)


def _flight_phases(time, foot_ground, robot_ground, sole, motion, forward, tip_s, gait_start_s, criteria):
    dt = float(np.median(np.diff(time)))
    both_clear = np.isfinite(sole).all(axis=1) & (sole >= criteria.min_sole_clearance_m).all(axis=1)
    floor_free = ~robot_ground & ~foot_ground.any(axis=1)
    grounded = foot_ground.any(axis=1)
    overall_start = max(gait_start_s + criteria.startup_exclusion_s, motion[0, 0])
    overall_end = motion[-1, 0] - criteria.terminal_exclusion_s
    phases = []
    for start, stop in _runs(floor_free):
        if start == 0 or stop == len(time) or not grounded[start - 1] or not grounded[stop]:
            continue
        if time[start] < overall_start or time[stop] > overall_end:
            continue
        if time[start] < tip_s - criteria.gait_window_s or time[stop] > tip_s + criteria.gait_window_s:
            continue
        sustained_clearance = max(
            (time[start + b - 1] - time[start + a] + dt for a, b in _runs(both_clear[start:stop])),
            default=0.,
        )
        if sustained_clearance + 1e-8 < criteria.min_flight_s:
            continue
        on_entry = int(np.searchsorted(time, time[start] - criteria.min_ground_interval_s - 1e-8))
        on_exit = int(np.searchsorted(time, time[stop] + criteria.min_ground_interval_s - 1e-8))
        if on_entry == start or on_exit >= len(time) or time[start] - time[on_entry] + dt < criteria.min_ground_interval_s:
            continue
        # Contact constraints may pulse for a few milliseconds at toe-off.
        if min(grounded[on_entry:start].mean(), grounded[stop:on_exit + 1].mean()) < criteria.min_ground_fraction:
            continue
        span_start = time[on_entry]
        span_end = time[on_exit]
        progress = motion[:, 1:3] @ forward
        speed = (np.interp(span_end, motion[:, 0], progress) - np.interp(span_start, motion[:, 0], progress)) / (
            span_end - span_start
        )
        if speed >= criteria.min_forward_speed_mps:
            phases.append({
                "start_s": float(time[start]), "land_s": float(time[stop]),
                "both_soles_clear_s": float(sustained_clearance), "forward_speed_mps": float(speed),
            })
    return phases


def _pass_metrics(contact, release_time_s, forward, criteria):
    normals = contact[:, NORMAL_COLUMNS]
    load = contact[:, [CONTACT_INDEX["ball_floor_contact"], CONTACT_INDEX["applied_ball_load"]]]
    contacted = (normals.sum(axis=1) > 1e-9) | np.any(load, axis=1)
    hits = np.flatnonzero(contacted)
    empty = {
        "first_ball_contact_s": None, "unassisted_ball_flight_s": None,
        "ball_toward_receiver_travel_m": None, "ball_rise_m": None,
        "ball_descent_into_contact_m": None, "longer_airborne_pass": False,
    }
    if release_time_s is None or not len(hits) or contact[0, 0] <= release_time_s:
        return empty
    hit = int(hits[0])
    time = contact[:, 0]
    if (time[0] - release_time_s > criteria.max_ball_sample_gap_s + 1e-8
            or np.max(np.diff(time[:hit + 1]), initial=0.) > criteria.max_ball_sample_gap_s + 1e-8):
        return empty
    xyz = contact[:, 1:4]
    travel = float((xyz[0, :2] - xyz[hit, :2]) @ forward)
    duration = float(time[hit] - release_time_s)
    rise = float(np.max(xyz[:hit + 1, 2]) - xyz[0, 2])
    descent_start = max(time[0], time[hit] - criteria.ball_descent_window_s)
    descent = float(np.interp(descent_start, time, xyz[:, 2]) - xyz[hit, 2])
    return {
        "first_ball_contact_s": float(time[hit]), "unassisted_ball_flight_s": duration,
        "ball_toward_receiver_travel_m": travel, "ball_rise_m": rise,
        "ball_descent_into_contact_m": descent,
        "longer_airborne_pass": (
            duration >= criteria.min_ball_flight_s and travel >= criteria.min_ball_forward_travel_m
            and rise >= criteria.min_ball_rise_m and descent >= criteria.min_ball_descent_m
            and not bool(load[hit].any())
        ),
    }


def score_running_long_pass(
    contact, motion, foot_time_s, foot_floor_contact, robot_floor_contact, sole_clearance_m,
    *,
    strict_catch_score: dict,
    release_time_s: float | None,
    forward_xy=(1., 0.),
    pass_toward_receiver_xy=None,
    gait_start_s: float = 0.,
    criteria: RunningCatchCriteria = RunningCatchCriteria(),
) -> dict:
    """Add elective gates to a strict catch result without changing that result.

    Foot samples contain left/right active floor constraints and minimum
    collider-to-floor distances. Any active robot floor constraint blocks flight.
    Clearance can be NaN for planted feet. Field forward is the direction of the
    run at the catch, so a ball traveling toward the receiver has negative
    displacement along it. Use actual traces from the same rollout.
    """
    contact = _trace(contact, TRACE_COLUMNS, "contact")
    motion = _trace(motion, MOTION_COLUMNS, "motion")
    time = np.asarray(foot_time_s, dtype=float)
    feet = _binary(foot_floor_contact, (len(time), 2), "foot floor contact")
    grounded = _binary(robot_floor_contact, (len(time),), "robot floor contact")
    sole = np.asarray(sole_clearance_m, dtype=float)
    forward = np.array(forward_xy, dtype=float)
    if time.ndim != 1 or sole.shape != (len(time), 2):
        raise ValueError("foot time and sole clearance must align")
    if forward.shape != (2,) or not np.isfinite(forward).all() or not np.linalg.norm(forward):
        raise ValueError("forward direction must be a finite nonzero XY vector")
    if not np.isfinite(gait_start_s) or gait_start_s < 0 or (release_time_s is not None and not np.isfinite(release_time_s)):
        raise ValueError("invalid release or gait start time")
    forward /= np.linalg.norm(forward)
    pass_axis = forward
    if pass_toward_receiver_xy is not None:
        pass_axis = np.asarray(pass_toward_receiver_xy, dtype=float)
        if pass_axis.shape != (2,) or not np.isfinite(pass_axis).all() or not np.linalg.norm(pass_axis):
            raise ValueError("pass direction must be a finite nonzero XY vector")
        pass_axis = pass_axis / np.linalg.norm(pass_axis)
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0) or np.any(feet.any(axis=1) & ~grounded):
        raise ValueError("invalid foot clock or robot-floor contact flags")
    foot_clock_ok = bool(
        len(time) >= 2 and len(motion) >= 2
        and np.max(np.diff(time)) <= criteria.max_foot_sample_gap_s + 1e-8
        and np.max(np.diff(motion[:, 0])) <= criteria.max_motion_gap_s + 1e-8
        and time[0] <= motion[0, 0] + criteria.max_foot_sample_gap_s
        and time[-1] >= motion[-1, 0] - criteria.max_foot_sample_gap_s
    )
    tip_s = strict_catch_score["first_tip_contact_s"]
    phases = []
    pre_tip_speed = None
    if tip_s is not None and foot_clock_ok:
        phases = _flight_phases(time, feet, grounded, sole, motion, forward, tip_s, gait_start_s, criteria)
        window = strict_catch_score["criteria"]["pre_contact_window_s"]
        last = np.searchsorted(motion[:, 0], tip_s - 1e-8) - 1
        if last >= 0 and motion[last, 0] - window >= motion[0, 0] - 1e-8:
            progress = motion[:, 1:3] @ forward
            pre_tip_speed = float((progress[last] - np.interp(motion[last, 0] - window, motion[:, 0], progress)) / window)
    sequences = []
    for phase in phases:
        if not sequences or phase["start_s"] - sequences[-1][-1]["start_s"] > criteria.max_neighbor_flight_spacing_s:
            sequences.append([])
        sequences[-1].append(phase)
    recurrent = max((len(sequence) for sequence in sequences), default=0)
    spans_catch = any(
        len(sequence) >= criteria.min_recurrent_flights
        and sequence[0]["start_s"] <= tip_s <= sequence[-1]["land_s"]
        for sequence in sequences
    )
    running = (
        spans_catch and pre_tip_speed is not None
        and pre_tip_speed >= criteria.min_forward_speed_mps
    )
    airborne_pass = _pass_metrics(contact, release_time_s, pass_axis, criteria)
    return {
        "strict_in_stride_catch_and_route": bool(strict_catch_score["in_stride_catch_and_route"]),
        "running_long_pass_and_route": bool(
            strict_catch_score["in_stride_catch_and_route"] and running and airborne_pass["longer_airborne_pass"]
        ),
        "running_at_catch": bool(running),
        "foot_sample_clock_ok": foot_clock_ok,
        "pre_tip_forward_speed_mps": pre_tip_speed,
        "qualified_flight_phases": phases,
        "max_recurrent_flights": recurrent,
        "recurrent_flights_span_catch": spans_catch,
        **airborne_pass,
        "criteria": asdict(criteria),
    }
