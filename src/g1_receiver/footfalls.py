"""Score forward footfalls from evaluator-only MuJoCo foot truth."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_SWING_S = .06
MIN_STANCE_S = .03
MIN_FORWARD_STEP_M = .05


@dataclass(frozen=True)
class Footfall:
    body: str
    lifted_s: float
    landed_s: float
    forward_displacement_m: float


def forward_footfalls(
    timestamps_s: np.ndarray,
    foot_position_m: np.ndarray,
    foot_floor_contact: np.ndarray,
    foot_body_names: np.ndarray,
    *,
    forward_xy: tuple[float, float] = (1., 0.),
) -> list[Footfall]:
    """Count established stance -> airborne swing -> forward recontact events."""
    stamps = np.asarray(timestamps_s, dtype=float)
    positions = np.asarray(foot_position_m, dtype=float)
    contacts = np.asarray(foot_floor_contact)
    body_names = np.asarray(foot_body_names)
    direction = np.asarray(forward_xy, dtype=float)
    if stamps.ndim != 1 or not len(stamps) or not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
        raise ValueError("foot timestamps must be finite and strictly increasing")
    if body_names.ndim != 1 or len(body_names) != 2 or len(set(body_names)) != 2:
        raise ValueError("foot truth must name two distinct bodies")
    if positions.shape != (len(stamps), 2, 3) or not np.isfinite(positions).all():
        raise ValueError("foot positions must contain finite XYZ for each timestamp and body")
    if contacts.shape != positions.shape[:2] or contacts.dtype != np.bool_:
        raise ValueError("floor contacts must contain a Boolean for each timestamp and body")
    if direction.shape != (2,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError("forward direction must be a finite, nonzero XY vector")
    direction = direction / np.linalg.norm(direction)

    events = []
    for foot, body in enumerate(body_names):
        stance = contacts[:, foot]
        transitions = np.r_[0, np.flatnonzero(stance[1:] != stance[:-1]) + 1, len(stamps)]
        for planted, lift, land, end in zip(transitions, transitions[1:], transitions[2:], transitions[3:]):
            if not stance[planted]:
                continue
            if (stamps[lift] - stamps[planted] < MIN_STANCE_S
                    or stamps[land] - stamps[lift] < MIN_SWING_S
                    or stamps[end - 1] - stamps[land] < MIN_STANCE_S):
                continue
            forward = float((positions[land, foot, :2] - positions[lift - 1, foot, :2]) @ direction)
            if forward >= MIN_FORWARD_STEP_M:
                events.append(Footfall(str(body), float(stamps[lift]), float(stamps[land]), forward))
    return sorted(events, key=lambda event: event.landed_s)
