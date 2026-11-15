"""Encoder-phased arm swing with a causal transition to an Aero receiving pose."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunArmReceive:
    """Shape arm targets while leaving the locomotion actor and its inputs intact."""

    running_fraction: float = .85
    pump_rad: float = .13
    lowered_rad: float = .12
    raise_duration_s: float = .36
    swing_start_s: float = 0.
    swing_ramp_s: float = .24

    def __post_init__(self):
        if not (0. <= self.running_fraction <= 1. and -.35 <= self.pump_rad <= .35
                and 0. <= self.lowered_rad <= .5 and .12 <= self.raise_duration_s <= .9
                and 0. <= self.swing_start_s <= 8. and .12 <= self.swing_ramp_s <= .9):
            raise ValueError('invalid bounded running arm settings')

    def initial(self, nominal, ready, names):
        """Select the matching startup posture before the AMP actor begins."""
        if self.swing_start_s > 0.:
            return ready.copy()
        target = nominal + self.running_fraction * (ready - nominal)
        target[self._shoulders(names)] += self.lowered_rad
        return target

    def targets(self, ready, names, measured_joint_rad, now_s, ready_at_s=None):
        """Return the arm target and override fraction after a visual ready cue.

        Store the first camera cue and pass that same timestamp on subsequent ticks.
        A no-ball physics experiment may supply its scheduled cue once reached.
        """
        elapsed = 0. if ready_at_s is None else now_s - ready_at_s
        raise_fraction = self._smoothstep(elapsed / self.raise_duration_s)
        engagement = 1. if self.swing_start_s == 0. else self._smoothstep((now_s - self.swing_start_s) / self.swing_ramp_s)
        swing_fraction = engagement * (1. - raise_fraction)
        left_hip, right_hip = (names.index(f'{side}_hip_pitch_joint') for side in ('left', 'right'))
        swing = self.pump_rad * np.tanh((measured_joint_rad[right_hip] - measured_joint_rad[left_hip]) / .40)
        target = ready.copy()
        target[self._shoulders(names)] += swing_fraction * (self.lowered_rad + np.array([swing, -swing]))
        if self.swing_start_s == 0.:
            override = self.running_fraction + (1. - self.running_fraction) * raise_fraction
        else:
            override = 1. - (1. - self.running_fraction) * swing_fraction
        return target, float(override)

    @staticmethod
    def _smoothstep(value):
        u = np.clip(value, 0., 1.)
        return u * u * (3. - 2. * u)

    @staticmethod
    def _shoulders(names):
        return [names.index(f'{side}_shoulder_pitch_joint') for side in ('left', 'right')]
