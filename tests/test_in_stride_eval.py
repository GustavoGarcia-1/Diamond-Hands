import numpy as np
import pytest

pytest.importorskip("mujoco")

from g1_receiver.in_stride_eval import CONTACT_INDEX, TRACE_COLUMNS, score_in_stride_catch
from g1_receiver.routes import Route

RELEASE_S = .4
FINISH_S = 3.5
PHYSICS_STEP_S = .002
BALL_WEIGHT_N = 4.
GO = Route("go", ((0., 0.), (.4, 0.)))
HITCH = Route("hitch", ((0., 0.), (.6, 0.), (.4, 0.)))


def traces(path=lambda t: np.minimum(.20 * t, .4)):
    motion_time = np.linspace(0., FINISH_S, round(FINISH_S / .02) + 1)
    motion = np.column_stack((motion_time, path(motion_time), np.zeros_like(motion_time),
                              np.full_like(motion_time, .8), np.ones_like(motion_time)))
    contact_time = RELEASE_S + PHYSICS_STEP_S * np.arange(1, round((FINISH_S - RELEASE_S) / PHYSICS_STEP_S) + 1)
    contact = np.zeros((len(contact_time), len(TRACE_COLUMNS)))
    contact[:, CONTACT_INDEX["timestamp_s"]] = contact_time
    contact[:, CONTACT_INDEX["ball_z_m"]] = 1.2
    caught = contact_time >= .60 - 1e-9
    for side in ("left", "right"):
        contact[caught, CONTACT_INDEX[f"{side}_tip_normal_n"]] = 3.
        contact[caught, CONTACT_INDEX[f"{side}_tip_vertical_n"]] = 2.
    return contact, motion


def score(contact, motion, route=GO, *, verified_free_ball=True):
    return score_in_stride_catch(
        contact, motion, route, release_time_s=RELEASE_S, finish_time_s=FINISH_S,
        ball_weight_n=BALL_WEIGHT_N, verified_free_ball=verified_free_ball, physics_step_s=PHYSICS_STEP_S,
    )


def test_in_stride_grasp_requires_movement_and_retention_through_route_finish():
    contact, motion = traces()
    result = score(contact, motion)
    assert result["in_stride_catch_and_route"]
    assert result["first_tip_contact_s"] == pytest.approx(.60)
    assert result["pre_tip_speed_mps"] == pytest.approx(.20)
    assert result["immediate_post_tip_route_displacement_m"] == pytest.approx(.10)
    assert result["total_post_tip_route_displacement_m"] == pytest.approx(.28)
    contact[-1, CONTACT_INDEX["left_tip_normal_n"]] = 0.
    assert score(contact, motion)["failed_checks"] == ["hands_only_retention_at_finish"]


def test_pause_at_catch_is_detected_even_if_robot_finishes_route():
    contact, motion = traces(lambda t: np.interp(t, [0., .52, 1.30, 2.80], [0., .104, .104, .4]))
    result = score(contact, motion)
    assert result["checks"]["pre_contact_speed"]
    assert result["checks"]["ordered_route_finish"]
    assert result["failed_checks"] == ["immediate_post_contact_progress"]


def test_hitch_must_visit_turn_before_return_and_reverse_progress_counts():
    contact, motion = traces()
    assert score(contact, motion, HITCH)["failed_checks"] == ["ordered_route_finish"]
    contact, motion = traces(lambda t: np.interp(t, [0., 2., 3.], [0., .6, .4]))
    result = score(contact, motion, HITCH)
    assert result["in_stride_catch_and_route"]
    assert len(result["ordered_waypoint_hit_times_s"]) == 2
    assert result["total_post_tip_route_displacement_m"] > .5


def test_brief_floor_or_body_contact_before_retention_window_still_fails():
    contact, motion = traces()
    contact[200, CONTACT_INDEX["ball_floor_contact"]] = 1.
    contact[220, CONTACT_INDEX["torso_normal_n"]] = .05
    result = score(contact, motion)
    assert result["retention"]["hands_only_retention_at_finish"]
    assert set(result["failed_checks"]) == {"no_floor_contact", "no_nonhand_body_contact_force"}


def test_missing_physics_sample_and_unverified_free_ball_fail():
    contact, motion = traces()
    result = score(np.delete(contact, 200, axis=0), motion, verified_free_ball=False)
    assert "initial_free_flight" in result["failed_checks"]
    assert "complete_synchronized_traces" in result["failed_checks"]
    assert not result["in_stride_catch_and_route"]
