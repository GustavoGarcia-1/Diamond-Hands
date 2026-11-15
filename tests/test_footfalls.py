import numpy as np

from g1_receiver.footfalls import forward_footfalls


def test_forward_footfall_requires_lift_recontact_and_displacement():
    time = np.arange(201) / 1000
    pos = np.zeros((len(time), 2, 3))
    contact = np.ones((len(time), 2), dtype=bool)
    names = np.array(["left", "right"])
    contact[50:140, 0] = False
    pos[140:, 0, 0] = .08
    # A rapid contact flicker does not count as a step even with forward travel.
    contact[55:65, 1] = False
    pos[65:, 1, 0] = .1
    events = forward_footfalls(time, pos, contact, names)
    assert len(events) == 1
    assert events[0].body == "left"
    assert events[0].lifted_s == .05
    assert events[0].landed_s == .14
    assert events[0].forward_displacement_m == .08

    # In-place lifts, backward motion and initial placement are excluded.
    pos[140:, 0, 0] = -.08
    assert forward_footfalls(time, pos, contact, names) == []
    pos[140:, 0, 0] = .08
    contact[:140, 0] = False
    assert forward_footfalls(time, pos, contact, names) == []
