"""Spectator-only quarterback and camera framing for a recorded scrimmage pass."""

from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .stadium import GOLD, NAVY, PAINT, _geom, _numbers, _pole, _yaw

SKIN = ".45 .26 .16 1"
VISOR = ".13 .20 .25 1"
PANTS = ".84 .80 .70 1"

ARM_LINKS = {
    "throw_upper": (.088, .15, NAVY),
    "throw_forearm": (.058, .145, SKIN),
    "throw_glove": (.052, .065, PAINT),
    "guide_upper": (.086, .15, NAVY),
    "guide_forearm": (.057, .145, SKIN),
    "guide_glove": (.050, .063, PAINT),
}


def add_quarterback(world, origin, heading):
    """Add a planted passer with a visual-only throwing arm."""
    body = ET.SubElement(world, "body", name="stadium_quarterback",
                         pos=_numbers((origin[0], origin[1], 0)), quat=_yaw(heading))
    _geom(body, "ellipsoid", PANTS, pos="-.51 .19 .91", size=".16 .185 .15")
    _geom(body, "ellipsoid", NAVY, pos="-.52 .19 1.25", size=".16 .24 .29")
    _geom(body, "ellipsoid", NAVY, pos="-.52 .19 1.48", size=".15 .28 .105")
    _geom(body, "ellipsoid", SKIN, pos="-.51 .19 1.52", size=".075 .069 .14")
    _geom(body, "ellipsoid", GOLD, pos="-.506 .19 1.735", size=".141 .135 .163")
    _geom(body, "ellipsoid", SKIN, pos="-.459 .19 1.643", size=".102 .095 .086")
    _geom(body, "ellipsoid", VISOR, pos="-.383 .19 1.675", size=".038 .089 .056")
    for z in (1.612, 1.645):
        _pole(body, (-.365, .115, z), (-.365, .265, z), .012, PAINT)
    for lateral in (.115, .265):
        _pole(body, (-.514, lateral, 1.65), (-.365, lateral, 1.615), .010, PAINT)
    for hip_y, knee_x, knee_y, shoe_x, shoe_y in (
        (.075, -.42, .03, -.33, -.02), (.30, -.63, .34, -.66, .40)
    ):
        _pole(body, (-.51, hip_y, .92), (knee_x, knee_y, .54), .101, PANTS)
        _geom(body, "ellipsoid", PANTS, pos=_numbers((knee_x, knee_y, .54)), size=".11 .10 .11")
        _pole(body, (knee_x, knee_y, .53), (shoe_x, shoe_y, .17), .067, PAINT)
        _pole(body, (shoe_x, shoe_y, .24), (shoe_x, shoe_y, .32), .070, NAVY)
        _geom(body, "ellipsoid", VISOR, pos=_numbers((shoe_x + .04, shoe_y, .069)), size=".141 .072 .067")
    # Jersey number 12, painted on the side seen from the sideline.
    for a, b in (
        ((-.61, 1.40), (-.57, 1.44)), ((-.57, 1.44), (-.57, 1.28)),
        ((-.60, 1.28), (-.54, 1.28)), ((-.49, 1.42), (-.42, 1.44)),
        ((-.42, 1.44), (-.405, 1.38)), ((-.405, 1.38), (-.49, 1.28)),
        ((-.49, 1.28), (-.40, 1.28)),
    ):
        _pole(body, (a[0], -.027, a[1]), (b[0], -.027, b[1]), .012, PAINT)
    for name, (radius, half_length, color) in ARM_LINKS.items():
        link = ET.SubElement(world, "body", name=f"stadium_qb_{name}", mocap="true",
                             pos=_numbers(origin))
        _geom(link, "capsule", color, size=_numbers((radius, half_length)))


def _ease(value):
    t = np.clip(value, 0., 1.)
    return t * t * (3 - 2 * t)


class QuarterbackPose:
    """Position the visual glove at the recorded ball before the release."""

    def __init__(self, model, origin, heading):
        self.origin = np.array(origin)
        self.transform = np.array(((math.cos(heading), -math.sin(heading), 0.),
                                   (math.sin(heading), math.cos(heading), 0.),
                                   (0., 0., 1.)))
        self.ids = {name: model.body(f"stadium_qb_{name}").mocapid[0] for name in ARM_LINKS}

    def update(self, data, release_time):
        t = data.time - release_time
        cocking = _ease((t + .30) / .22) * (1 - _ease((t + .09) / .09))
        follow = _ease(t / .22)
        recover = _ease((t - .31) / .70)
        shoulder = np.array((-.50, -.011, 1.45))
        elbow = np.array((-.25 - .23 * cocking + .13 * follow - .20 * recover,
                          -.11 - .08 * cocking, 1.32 + .28 * cocking - .22 * recover))
        glove = np.array((-.11 + .28 * follow - .45 * recover, -.009 - .15 * follow + .10 * recover,
                          self.origin[2] + .05 * follow - .36 * recover))
        self._arm(data, "throw", shoulder, elbow, glove)
        shoulder = np.array((-.50, .39, 1.46))
        elbow = np.array((-.37 - .03 * follow, .45, 1.21))
        glove = np.array((-.05 - .24 * follow, .109 + .32 * follow, self.origin[2] - .05 - .35 * follow))
        self._arm(data, "guide", shoulder, elbow, glove)

    def _arm(self, data, side, shoulder, elbow, glove):
        offset = np.r_[self.origin[:2], 0.]
        for part, start, finish in (
            ("upper", shoulder, elbow), ("forearm", elbow, glove),
            ("glove", glove - .04 * (glove - elbow) / np.linalg.norm(glove - elbow), glove),
        ):
            address = self.ids[f"{side}_{part}"]
            direction = self.transform @ (finish - start)
            data.mocap_pos[address] = offset + self.transform @ ((start + finish) / 2)
            mujoco.mju_quatZ2Vec(data.mocap_quat[address], direction / np.linalg.norm(direction))


@dataclass(frozen=True)
class PassCamera:
    release_time: float
    flight_end_time: float
    start_time: float
    origin: np.ndarray
    heading_deg: float

    def frame(self, camera, t, receiver_xy, ball_xyz):
        after_release = t - self.release_time
        if t - self.start_time < 1.10 or -.62 <= after_release < .13:
            camera.lookat[:] = self.origin + np.array((0., 0., -.47))
            camera.azimuth, camera.elevation, camera.distance = self.heading_deg + 125., -8., 2.85
        elif t < self.release_time:
            center = self.origin[:2] * .53 + receiver_xy * .47
            camera.lookat[:] = (*center, .82)
            camera.azimuth, camera.elevation = self.heading_deg + 102., -24.
            camera.distance = max(4.3, np.linalg.norm(receiver_xy - self.origin[:2]) * 1.13)
        elif t <= self.flight_end_time + .10:
            center = ball_xyz[:2] * .52 + receiver_xy * .48
            camera.lookat[:] = (*center, np.clip(.48 + ball_xyz[2] * .34, 1.10, 2.12))
            camera.azimuth, camera.elevation = self.heading_deg + 102., -13.
            camera.distance = max(3.8, np.linalg.norm(receiver_xy - ball_xyz[:2]) * 1.23)
