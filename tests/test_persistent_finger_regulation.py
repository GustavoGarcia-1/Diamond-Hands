"""Regression against archived physical replay packets, without simulation."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from g1_receiver.lower_cup_grip import NativeCupGrip, NativeHandCalibration


def test_opt_in_integration_reproduces_frozen_replay_requests():
    fixture=json.loads((Path(__file__).parent/'fixtures/persistent_finger_regulation.json').read_text())
    calibration=NativeHandCalibration(np.array(fixture['ranges']),np.arange(14))
    config=deepcopy(fixture['native_grip'])
    config['finger_encoder_regulation']={'closing_error_m':.0015}
    integrated=NativeCupGrip(calibration,config)
    original=NativeCupGrip(calibration,fixture['native_grip'])
    changed=False
    for row in fixture['samples']:
        now=row['time_s']
        release=fixture['release_at_s'] if now>=fixture['release_at_s'] else None
        arguments=(now,0.,release,row['visual_gate_s'],np.array(row['native_encoder']))
        nominal=original.targets(*arguments)
        applied=integrated.targets(*arguments)
        assert np.array_equal(nominal,row['nominal'])
        assert np.array_equal(applied,row['requested'])
        assert np.array_equal(applied[[4,5,6,11,12,13]],nominal[[4,5,6,11,12,13]])
        changed=changed or not np.array_equal(applied,nominal)
    assert changed
    assert np.array_equal(original.rows,integrated.rows)
    assert integrated.finger_regulation.rows[-1]['blend']==1.


def test_feedback_is_not_enabled_without_explicit_configuration():
    fixture=json.loads((Path(__file__).parent/'fixtures/persistent_finger_regulation.json').read_text())
    calibration=NativeHandCalibration(np.array(fixture['ranges']),np.arange(14))
    grip=NativeCupGrip(calibration,fixture['native_grip'])
    assert grip.finger_regulation is None
    assert 'finger_encoder_regulation' not in fixture['native_grip']
