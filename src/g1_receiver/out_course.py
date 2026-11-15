"""An opt-in five-yard Out maneuver driven by the camera and IMU."""

from dataclasses import dataclass
from math import pi

import numpy as np

from .football_routes import YARD_M, heading_error


@dataclass
class OutCut:
    field_heading_rad: float
    outside_sign: int
    turn_s: float = .9
    cut_speed_mps: float = .48
    blend_s: float = .65
    hold_straight_s: float = .25
    approach_handoff_depth_m: float | None = None
    depth_feedback_gain: float = 0.
    max_walk_fraction: float = 1.
    running_turn: bool = False
    stopping_lookahead_s: float = .28
    reverse_speed_mps: float = .25
    approach_speed_mps: float | None = None
    receiving_heading_rad: float | None = None
    receiving_heading_ramp_s: float = 1.
    outside_run_fraction: float = 0.
    outside_run_ramp_s: float = .6
    outside_run_at_s: float | None = None
    break_at_s: float | None = None
    break_pose_m: tuple[float, float] | None = None
    handoff_at_s: float | None = None

    @property
    def downfield(self):
        return np.array([np.cos(self.field_heading_rad), np.sin(self.field_heading_rad)])

    @property
    def outside(self):
        x, y = self.downfield
        return self.outside_sign * np.array([-y, x])

    def update(self, now, pose, yaw, run_speed, *, field_velocity=None):
        """Wait for the camera estimate to cross five yards from the start line."""
        if pose is not None and pose.fresh(now, .14):
            depth = pose.xy_m @ self.downfield
            if (self.handoff_at_s is None and self.approach_handoff_depth_m is not None
                    and depth >= self.approach_handoff_depth_m):
                self.handoff_at_s = now
            if self.break_at_s is None and depth >= 5. * YARD_M:
                self.break_at_s = now
                self.break_pose_m = tuple(float(value) for value in pose.xy_m)
        elapsed = max(0., now - self.break_at_s) if self.break_at_s is not None else 0.
        # Start the torque handoff as the receiver enters the outside cut.
        handoff = self.handoff_at_s if self.handoff_at_s is not None else self.break_at_s
        blend = float(np.clip((now - handoff) / self.blend_s, 0., 1.)) if handoff is not None else 0.
        blend = self.max_walk_fraction * blend * blend * (3. - 2. * blend)
        change = float(np.clip((elapsed - self.hold_straight_s) / self.turn_s, 0., 1.))
        bend = self.outside_sign * pi / 2. * change
        intended_heading = self.field_heading_rad + bend
        c, s = np.cos(yaw), np.sin(yaw)
        # Walking policy accepts full local XY commands and can turn or sidestep.
        speed_limit = (self.approach_speed_mps if self.break_at_s is None and self.approach_speed_mps is not None
                       else self.cut_speed_mps)
        world = speed_limit * np.array([np.cos(intended_heading), np.sin(intended_heading)])
        if (self.depth_feedback_gain and handoff is not None and pose is not None and pose.fresh(now, .14)):
            depth = pose.xy_m @ self.downfield
            error = 5. * YARD_M - depth
            speed = 0. if field_velocity is None else float(field_velocity @ self.downfield)
            # Command less forward motion as the measured stopping distance reaches the yard line.
            correction = self.depth_feedback_gain * (error - self.stopping_lookahead_s * max(0., speed))
            desired = float(np.clip(correction, -self.reverse_speed_mps, speed_limit))
            world += self.downfield * (desired - world @ self.downfield)
        local = np.array([[c, s], [-s, c]]) @ world
        body_heading = intended_heading
        if self.receiving_heading_rad is not None and self.break_at_s is not None:
            facing_fraction = float(np.clip(elapsed / self.receiving_heading_ramp_s, 0., 1.))
            body_heading = self.field_heading_rad + facing_fraction * heading_error(
                self.receiving_heading_rad, self.field_heading_rad)
        walk_drive = np.r_[local, np.clip(1.15 * heading_error(body_heading, yaw), -.65, .65)]
        if self.outside_run_fraction and self.break_at_s is not None:
            exit_heading = self.field_heading_rad + self.outside_sign * pi / 2.
            if (self.outside_run_at_s is None and elapsed >= 1.
                    and abs(heading_error(exit_heading, yaw)) <= .35):
                self.outside_run_at_s = now
            if self.outside_run_at_s is not None:
                reentry = float(np.clip((now - self.outside_run_at_s) / self.outside_run_ramp_s, 0., 1.))
                blend *= 1. - self.outside_run_fraction * reentry
        run_heading = intended_heading if (self.running_turn or self.outside_run_fraction) and self.break_at_s is not None else 0.
        run_drive = np.array([run_speed * (1. - .45 * blend), 0.,
                              np.clip(1.8 * heading_error(run_heading, yaw), -.75, .75)])
        return run_drive, walk_drive, blend


def upper_lookback_fraction(elapsed_s, *, ramp_s=.9, hold_s=1.2):
    """Brief camera-break-relative upper-body turn; pelvis goal is unaffected."""
    if ramp_s <= 0. or hold_s < 0.:
        raise ValueError("lookback ramp must be positive and hold nonnegative")
    elapsed_s -= .1
    up = float(np.clip(elapsed_s / ramp_s, 0., 1.))
    down = float(np.clip((elapsed_s - ramp_s - hold_s) / ramp_s, 0., 1.))
    return up * up * (3. - 2. * up) * (1. - down * down * (3. - 2. * down))


def interpolate_arm_path(path, fraction):
    """Interpolate a bounded, offline-calibrated paired-arm motor path."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 14 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError("carry path requires at least two finite14-axis arm targets")
    if not np.isfinite(fraction):
        raise ValueError("carry phase must be finite")
    index = float(np.clip(fraction, 0., 1.)) * (len(path) - 1)
    lower = min(int(index), len(path) - 2)
    blend = index - lower
    return (1. - blend) * path[lower] + blend * path[lower + 1]
