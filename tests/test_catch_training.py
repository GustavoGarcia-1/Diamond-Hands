"""Action and observation contracts for the optional PPO timing pilot."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from g1_receiver.catch_training import OBSERVATION_SIZE, bounded_delay_s, catch_observation


def test_requested_ticks_cannot_exceed_current_gate_slack():
    for max_delay in np.linspace(0., 0.08, 81):
        for requested_ticks in range(5):
            applied = bounded_delay_s(requested_ticks, max_delay)
            assert 0 <= applied <= max_delay + 1e-10
            assert applied == pytest.approx(round(applied / 0.02) * 0.02)
    assert bounded_delay_s(0, 0.08) == 0.
    assert bounded_delay_s(4, 0.035) == pytest.approx(0.02)
    assert bounded_delay_s(4, 0.08) == pytest.approx(0.08)
    for invalid in (-1, 1.5, 5):
        with pytest.raises(ValueError):
            bounded_delay_s(invalid, 0.08)


def test_ineligible_observation_contains_only_proprioception_and_empty_gate_features():
    measured = (np.zeros(29), np.ones(29), np.array([1, 0, 0, 0]), np.zeros(3))
    observation = catch_observation(measured, None, 0.)
    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    np.testing.assert_array_equal(observation[:65], np.concatenate(measured))
    assert not observation[65:].any()
