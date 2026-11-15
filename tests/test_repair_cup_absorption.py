from types import SimpleNamespace
import numpy as np
import pytest
from g1_receiver.repair_cup_absorption import VerticalCupStroke, bounded_joint_step


def make_controller(gain):
    controller = VerticalCupStroke({'play': {'field_catch_window_m': [6, 0, 1.2],
                                           'field_approach_axis_xy': [1, 0]}},
                                   gain=gain, max_give_m=.18, lead_s=.12)
    controller._fk = lambda obs: (np.array([1., 0., 0.]), np.eye(12, 14))
    controller.arms = np.arange(14)
    controller.ranges = np.tile([-3., 3.], (14, 1))
    return controller


def test_no_camera_cannot_trigger_stroke_even_when_enabled():
    controller = make_controller(.5)
    target = np.linspace(-.2, .2, 29)
    for timestamp in np.arange(0., 1., .02):
        changed, _ = controller(SimpleNamespace(timestamp=timestamp), None, None, None,
                                target, np.zeros(2), True)
        assert np.array_equal(changed, target)
        assert controller.plan is None


def test_isolated_extended_stroke_preserves_declared_target_and_slew_limits():
    previous = np.zeros(14)
    for _ in range(40):
        result = bounded_joint_step(np.ones(14), previous, .02)
        assert np.max(abs(result-previous)) <= .08400000001
        assert np.max(abs(result)) <= .60000000001
        previous = result
    assert np.allclose(previous, .6)
    with pytest.raises(ValueError):
        bounded_joint_step(previous, previous, .1)
