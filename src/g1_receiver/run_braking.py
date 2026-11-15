"""Sensor-commanded handoff from the running policy to the walking controller."""

from dataclasses import dataclass

import numpy as np

from ._vendor.g1_runtime.pool.contracts import ActuatorCommand


def blend_pd(run, walk, fraction):
    if fraction <= 0.:
        return run
    if fraction >= 1.:
        return walk
    run_kp, walk_kp = (1. - fraction) * run.kp, fraction * walk.kp
    kp = run_kp + walk_kp
    targets = np.divide(run_kp * run.target_position + walk_kp * walk.target_position, kp,
                        out=walk.target_position.copy(), where=kp > 0.)
    targets = np.clip(targets, np.minimum(run.target_position, walk.target_position),
                      np.maximum(run.target_position, walk.target_position))
    return ActuatorCommand(targets, kp, (1. - fraction) * run.kd + fraction * walk.kd,
                           (1. - fraction) * run.feedforward_torque + fraction * walk.feedforward_torque,
                           "running_to_walking")


@dataclass
class RunningBrake:
    brake_x_m: float
    absorb_speed_mps: float = .75
    blend_s: float = .65
    absorb_s: float = .65
    ramp_down_s: float = .80
    terminal_delay_s: float = 0.
    terminal_correction: bool = True
    brake_at_s: float | None = None
    trigger: str | None = None
    settled: bool = False

    def commands(self, now, pose, velocity, heading_yaw, endpoint, *, timeout_s, predict_progress=False):
        """Use a surveyed field line to start braking; a time limit guards camera loss."""
        fresh = pose is not None and pose.fresh(now, .14)
        if self.brake_at_s is None:
            depth = -np.inf if not fresh else pose.xy_m[0]
            if predict_progress and fresh and velocity is not None:
                age = min(now - pose.timestamp_s, .04)
                depth += float(np.clip(velocity[0], -3.5, 3.5) * age)
            if depth >= self.brake_x_m:
                self.brake_at_s, self.trigger = now, "camera_progress"
            elif now >= timeout_s - 1e-8:
                self.brake_at_s, self.trigger = now, "run_time_limit"
        elapsed = 0. if self.brake_at_s is None else now - self.brake_at_s
        u = float(np.clip(elapsed / self.blend_s, 0., 1.))
        speed = self.absorb_speed_mps * np.clip(1. - max(0., elapsed - self.absorb_s) / self.ramp_down_s, 0., 1.)
        drive = np.array([speed, 0., np.clip(-.8 * heading_yaw, -.28, .28) if speed > .05 else 0.])
        if (self.terminal_correction and elapsed >= self.absorb_s + self.ramp_down_s + self.terminal_delay_s
                and fresh and not self.settled):
            error = endpoint - pose.xy_m
            distance = np.linalg.norm(error)
            if distance <= .035 and velocity is not None and np.linalg.norm(velocity) < .15:
                self.settled = True
            elif distance > 1e-6:
                field = error * (np.clip(.85 * distance, .055, .33) / distance)
                c, s = np.cos(pose.yaw_rad), np.sin(pose.yaw_rad)
                drive[:2] = np.array([[c, s], [-s, c]]) @ field
        return 1. - .4 * u, u * u * (3. - 2. * u), drive
