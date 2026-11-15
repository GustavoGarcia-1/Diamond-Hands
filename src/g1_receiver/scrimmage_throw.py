"""A fixed-origin practice throw whose football is free from release onward."""

from dataclasses import dataclass

import mujoco
import numpy as np

from .short_throw import FreeBallRelease, UnderhandThrow


@dataclass(frozen=True)
class ScrimmagePass:
    origin_m: tuple[float, float, float]
    aim_m: tuple[float, float, float]
    duration_s: float
    align_ball_to_launch: bool = True
    horizontal_distance_only: bool = False

    def plan(self, release_time_s: float, gravity_mps2: float = 9.81) -> UnderhandThrow:
        origin, aim = np.asarray(self.origin_m), np.asarray(self.aim_m)
        duration = self.duration_s
        if (origin.shape != (3,) or aim.shape != (3,) or not np.isfinite([*origin, *aim, duration,
                                                                       release_time_s, gravity_mps2]).all()
                or not .4 <= duration <= 1.6 or gravity_mps2 <= 0.
                or not 1. <= origin[2] <= 2.2
                or (np.linalg.norm(aim[:2] - origin[:2]) if self.horizontal_distance_only
                    else aim[0] - origin[0]) < 5.
                or aim[2] <= .12 or release_time_s < 0.):
            raise ValueError("a scrimmage pass needs a fixed release and a target at least 5 m away in the declared distance metric")
        acceleration = np.array([0., 0., -gravity_mps2])
        initial = (aim - origin) / duration - .5 * acceleration * duration
        terminal = initial + acceleration * duration
        if initial[2] <= 0. or terminal[2] >= 0. or np.linalg.norm(initial) > 30.:
            raise ValueError("throw must rise, descend at its target, and stay within launch speed limits")
        yaw = .5 * np.arctan2(initial[1], initial[0])
        pitch = -.5 * np.arctan2(initial[2], np.linalg.norm(initial[:2]))
        sy, cy, sp, cp = np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch)
        orientation = ((cy * cp, -sy * sp, cy * sp, sy * cp)
                       if self.align_ball_to_launch else (1., 0., 0., 0.))
        return UnderhandThrow(release_time_s, duration, tuple(origin), tuple(initial), orientation,
                              tuple(aim), tuple(terminal), float(np.linalg.norm((aim - origin)[:2])))


class FixedQuarterback:
    """Hold the ball in a fixed simulated hand until its single ballistic release."""

    def __init__(self, model, data, plan):
        self.model, self.data, self.plan = model, data, plan
        self._release = FreeBallRelease(model, data)
        self._body_id = model.body("football").id
        self.released = False

    def ready(self):
        # A stationary launch hand supports the ball's weight until release.
        self.data.xfrc_applied[self._body_id, :3] = (
            -self.model.body_mass[self._body_id] * self.model.opt.gravity)
        mujoco.mj_forward(self.model, self.data)

    def release(self, now):
        if self.released:
            raise RuntimeError("the quarterback may throw only once")
        self.data.xfrc_applied[self._body_id] = 0.
        throw = self.plan.plan(now, -float(self.model.opt.gravity[2]))
        self._release.apply(throw)
        self.released = True
        return throw

    def close(self):
        self.data.xfrc_applied[self._body_id] = 0.
