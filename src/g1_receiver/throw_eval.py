"""Audit the observed flight and receive window independently of a terminal stop."""

import numpy as np

from .in_stride_eval import CONTACT_INDEX, NORMAL_COLUMNS


def score_throw_receive(contact, motion, release, first_tip_s, *, physics_step_s):
    trace = np.asarray(contact, dtype=float)
    motion = np.asarray(motion, dtype=float)
    if not len(trace) or release is None:
        return None
    normals = trace[:, NORMAL_COLUMNS]
    ground_or_body = (normals[:, 6:].sum(axis=1) > 1e-9) | (
        trace[:, CONTACT_INDEX["ball_floor_contact"]].astype(bool)
        | trace[:, CONTACT_INDEX["applied_ball_load"]].astype(bool))
    times = trace[:, 0]
    origin = np.asarray(release["release_xyz_m"])
    touched = np.flatnonzero((normals.sum(axis=1) > 1e-9) | ground_or_body)
    contact_time = None if not len(touched) else float(times[touched[0]])
    duration = None if contact_time is None else contact_time - release["release_time_s"]
    displacement = None if not len(touched) else trace[touched[0], 1:4] - origin
    both = (normals[:, :2].sum(axis=1) >= .1) & (normals[:, 3:5].sum(axis=1) >= .1)
    external = np.flatnonzero(ground_or_body)
    if first_tip_s is not None:
        both &= times >= first_tip_s - 1e-8
        both &= times <= first_tip_s + .50
    else:
        both[:] = False
    if len(external):
        both[external[0]:] = False
    edges = np.flatnonzero(np.diff(np.r_[False, both, False]))
    interval_lengths = edges[1::2] - edges[::2]
    longest = float(interval_lengths.max(initial=0) * physics_step_s)
    rel_speed = None
    if len(interval_lengths):
        a = edges[2 * int(np.argmax(interval_lengths))]
        b = a + interval_lengths.max() - 1
        if times[b] - times[a] >= .08:
            # Estimate whether the football and the receiver are traveling together.
            pelvis_xy = np.array([np.interp(times[[a, b]], motion[:, 0], motion[:, i]) for i in (1, 2)]).T
            delta = np.diff(trace[[a, b], 1:3] - pelvis_xy, axis=0)[0]
            rel_speed = float(np.linalg.norm(delta) / (times[b] - times[a]))
    return {
        "flight_s": duration,
        "downfield_flight_m": None if displacement is None else float(displacement[0]),
        "horizontal_flight_m": None if displacement is None else float(np.linalg.norm(displacement[:2])),
        "receiver_depth_at_fingertip_m": None if first_tip_s is None else float(np.interp(
            first_tip_s, motion[:, 0], motion[:, 1])),
        "continuous_two_hand_contact_s": longest,
        "horizontal_ball_to_body_speed_during_contact_mps": rel_speed,
        "first_floor_or_body_support_s": None if not len(external) else float(times[external[0]]),
        "controlled_in_stride_receive": bool(longest >= .15 and rel_speed is not None and rel_speed <= 1.0),
    }
