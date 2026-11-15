"""Field targets for the full-size running receiver courses."""

from dataclasses import dataclass
import math

import numpy as np


YARD_M = .9144


@dataclass(frozen=True)
class FootballRouteSpec:
    name: str
    break_after_steps: int | None
    break_depth_m: float | None
    break_heading_rad: float | None
    catch_depth_m: float | None


def football_route(name: str, *, outside_sign: int = 1) -> FootballRouteSpec:
    """Field X runs downfield; field Y's outside side depends on receiver alignment."""
    if outside_sign not in (-1, 1):
        raise ValueError("choose the receiver's outside as field +Y or -Y")
    if name == "go":
        return FootballRouteSpec(name, None, None, None, 10 * YARD_M)
    if name == "slant":
        return FootballRouteSpec(name, 2, None, -outside_sign * math.pi / 4., None)
    if name == "out":
        return FootballRouteSpec(name, None, 5 * YARD_M, outside_sign * math.pi / 2., None)
    raise ValueError(f"unsupported full-size football route: {name!r}")


def imu_yaw(quaternion) -> float:
    w, x, y, z = quaternion
    return float(np.arctan2(2. * (w*z + x*y), 1. - 2. * (y*y + z*z)))


def heading_error(target: float, measured: float) -> float:
    delta = target - measured
    return math.atan2(math.sin(delta), math.cos(delta))
