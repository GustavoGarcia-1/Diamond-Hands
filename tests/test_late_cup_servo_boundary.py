from types import SimpleNamespace
import numpy as np
import pytest
from g1_receiver.late_cup_carry_servo import WorldCupAnchor

def test_zero_servo_needs_no_simulator_state_and_preserves_command():
    servo=WorldCupAnchor({'world_center_m':[0.,0.,0.],'gain':0.,'carry_gain':0.})
    servo.arms=np.arange(14)
    command=np.arange(29,dtype=float)/20
    before=command.copy()
    result=servo.apply(SimpleNamespace(timestamp=1.),None,command,0.,None)
    np.testing.assert_array_equal(result,before)
    np.testing.assert_array_equal(command,before)
    assert result is not command
    assert servo.rows[-1]['status']=='abstain_no_fresh_pair_or_latephase'

def test_carry_controller_refuses_unbounded_gain():
    with pytest.raises(AssertionError):
        WorldCupAnchor({'world_center_m':[0.,0.,0.],'gain':.25,'carry_gain':.26})

def test_soft_up_target_never_claims_hard_form_pass():
    from g1_receiver.late_cup_relative_servo import command_form_diagnostic
    bad=command_form_diagnostic(.7,[.4,.8],[.2,.1])
    assert bad['command_form_constraint_pass'] is None
    assert bad['command_form_proxy_pass'] is False
    assert bad['command_form_hard_rejection_enabled'] is False
    good=command_form_diagnostic(.7,[.8,.8],[.1,.1])
    assert good['command_form_constraint_pass'] is None
    assert good['command_form_proxy_pass'] is True
