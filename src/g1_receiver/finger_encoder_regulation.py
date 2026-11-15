"""Experimental native finger compliance using encoders and known commands.

The ceiling bounds requested closing position error at the sensor sample, not
actual force between samples. It accepts no simulator state, contact, or load.
Existing native control range and slew limits remain authoritative downstream.
"""
from dataclasses import dataclass

import numpy as np


FLEXORS = np.array([0, 1, 2, 3, 7, 8, 9, 10], dtype=int)


@dataclass(frozen=True)
class ComplianceConfig:
    closing_error_m: float = 0.0015
    window_s: float = 0.60
    restore_s: float = 0.20
    persistent: bool = True

    def __post_init__(self):
        if not 0.0001 <= self.closing_error_m <= 0.003:
            raise ValueError("closing error must be between 0.1 and 3 millimeters")
        if not 0.1 <= self.window_s <= 1.0 or not 0.05 <= self.restore_s <= 0.5:
            raise ValueError("bounded visual event relative windows required")


class EncoderGripCompliance:
    def __init__(self, ranges, config=ComplianceConfig(), enabled=True):
        self.ranges = np.asarray(ranges, dtype=float).copy()
        if self.ranges.shape != (14, 2) or not np.isfinite(self.ranges).all():
            raise ValueError("fourteen finite native control ranges required")
        self.config, self.enabled = config, bool(enabled)
        self.last_time = None
        self.rows = []

    def apply(self, now_s, nominal, encoders, visual_gate_s):
        now = float(now_s)
        q = np.asarray(nominal, dtype=float)
        sensed = np.asarray(encoders, dtype=float)
        if q.shape != (14,) or sensed.shape != (14,) or not np.isfinite([q, sensed]).all():
            raise ValueError("finite native target and encoder packets required")
        if not np.isfinite(now) or (self.last_time is not None and now <= self.last_time):
            raise ValueError("strictly causal control timestamps required")
        self.last_time = now
        phase = None if visual_gate_s is None else now - float(visual_gate_s)
        blend = 0.0
        if self.enabled and phase is not None and 0.0 <= phase:
            if self.config.persistent or phase <= self.config.window_s:
                blend = 1.0
            elif phase < self.config.window_s + self.config.restore_s:
                u = (phase - self.config.window_s) / self.config.restore_s
                blend = 1.0 - u**3 * (10.0 - 15.0*u + 6.0*u*u)
        result = q.copy()
        if blend:
            limited = np.maximum(q[FLEXORS], sensed[FLEXORS] - self.config.closing_error_m)
            limited = np.clip(limited, *self.ranges[FLEXORS].T)
            # Never command additional closure or modify a thumb channel.
            result[FLEXORS] = q[FLEXORS] + blend * np.maximum(0.0, limited-q[FLEXORS])
        self.rows.append(dict(time_s=now, visual_gate_s=visual_gate_s,
                              phase_s=phase, blend=blend,
                              nominal=q.tolist(), native_encoder=sensed.tolist(),
                              requested=result.tolist(),
                              sampled_closing_error_m=(sensed[FLEXORS]-result[FLEXORS]).tolist()))
        return result
