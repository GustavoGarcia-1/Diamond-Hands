import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from g1_receiver.in_stride_eval import CONTACT_INDEX, TRACE_COLUMNS
from g1_receiver.running_catch_eval import FootContactRecorder, score_running_long_pass


def traces():
    motion_time = np.linspace(0., 3., 151)
    motion = np.column_stack((motion_time, motion_time, motion_time * 0, motion_time * 0 + .8, motion_time * 0 + 1))
    contact_time = 1.1 + np.arange(1, 1901) * .001
    contact = np.zeros((len(contact_time), len(TRACE_COLUMNS)))
    contact[:, 0] = contact_time
    flight_time = np.minimum(contact_time - 1.1, .35)
    contact[:, 1] = 2. - flight_time
    contact[:, 3] = 1. + 1.7 * flight_time - 4.9 * flight_time**2
    contact[contact_time >= 1.45 - 1e-8, CONTACT_INDEX["left_tip_normal_n"]] = 3.
    foot_time = np.arange(3000) * .001
    foot_floor = np.ones((len(foot_time), 2), dtype=bool)
    for begin in (.1, 1.05, 1.42):
        flight = (foot_time >= begin) & (foot_time < begin + .06)
        foot_floor[flight] = False
    robot_floor = foot_floor.any(axis=1)
    clearance = np.where(foot_floor, 0., .012)
    return contact, motion, foot_time, foot_floor, robot_floor, clearance


def score(*traces, strict_pass=True):
    return score_running_long_pass(
        *traces, release_time_s=1.1,
        strict_catch_score={
            "in_stride_catch_and_route": strict_pass,
            "first_tip_contact_s": 1.45,
            "criteria": {"pre_contact_window_s": .14},
        },
    )


def test_requires_recurrent_running_actual_long_pass_and_strict_catch():
    samples = traces()
    result = score(*samples)
    assert result["running_long_pass_and_route"]
    assert result["running_at_catch"] and result["max_recurrent_flights"] == 2
    assert result["ball_toward_receiver_travel_m"] == pytest.approx(.349)
    assert not score(*samples, strict_pass=False)["running_long_pass_and_route"]


def test_launch_transient_solo_hop_and_other_grounding_do_not_count():
    contact, motion, time, feet, grounded, clearance = traces()
    grounded[1420:1480] = True
    result = score(contact, motion, time, feet, grounded, clearance)
    assert not result["running_at_catch"]
    assert result["max_recurrent_flights"] == 1
    assert result["longer_airborne_pass"]


def test_short_travel_and_early_non_tip_contact_fail_pass():
    contact, motion, time, feet, grounded, clearance = traces()
    contact[:, 1] = 2. + (contact[:, 1] - 2.) * .50
    assert not score(contact, motion, time, feet, grounded, clearance)["longer_airborne_pass"]
    contact[:, 1] = 2. + (contact[:, 1] - 2.) / .50
    contact[20, CONTACT_INDEX["right_palm_normal_n"]] = .05
    assert not score(contact, motion, time, feet, grounded, clearance)["longer_airborne_pass"]


def test_ground_contact_recorder_uses_sole_clearance_and_excludes_ball():
    model = mujoco.MjModel.from_xml_string("""
<mujoco><option timestep=".001"/>
<worldbody>
  <geom name="floor" type="plane" size="1 1 .1"/>
  <body name="football" pos=".7 0 .04"><freejoint/><geom size=".05"/></body>
  <body name="left_ankle_roll_link" pos="-.2 0 .04"><freejoint/><geom size=".05"/></body>
  <body name="right_ankle_roll_link" pos=".2 0 .11"><freejoint/><geom size=".05"/></body>
</worldbody></mujoco>
""")
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    recorder = FootContactRecorder(model)
    recorder.record(data)
    arrays = recorder.arrays()
    assert arrays["foot_time_s"].tolist() == [0.]
    assert arrays["foot_floor_contact"].tolist() == [[True, False]]
    assert arrays["robot_floor_contact"].tolist() == [True]
    assert arrays["sole_clearance_m"][0] == pytest.approx([-.01, .06])
