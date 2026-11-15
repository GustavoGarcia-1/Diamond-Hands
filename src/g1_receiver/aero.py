"""Official tendon-driven Aero Hand Open, mounted on the simulated G1 wrists."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .dextra import SIDES, WristMount, _set_bare_wrist_inertia
from .locomotion import cradle_joint_targets, make_dynamic_scene

UPSTREAM_REVISION = "fb36886b3023db08c198264e1e9956ab202c3e3b"
MODEL_DIR = "tetheria_aero_hand_open"
XML_SHA256 = {
    "left": "6e9ffa81c44e46ffb7df7196dec9f00db7e4592a1b5dc0df9444b4b6f24f53b6",
    "right": "139a72c991cd0d8b32b6544af8d220af2d5b8fb71c70042bd7eed98cbbc362b7",
}
MESH_SET_SHA256 = "8daf22951b4538f38d25f131e615a4495a17d1a60341e1c80eaa9838764dde37"
PALM_BODIES = ("aero_left_palm", "aero_right_palm")
FINGERTIP_GEOMS = tuple(
    tuple(f"aero_{side}_{digit}_tip" for digit in ("if", "mf", "rf", "pf", "th")) for side in SIDES
)
# Simulated, fixed wrist-to-official-mount transforms, using the upstream native chiral models.
_QUARTER = np.sqrt(0.5)
DEFAULT_MOUNTS = tuple(
    WristMount((0.067, 0., 0.), (_QUARTER, sign * _QUARTER, 0., 0.)) for sign in (1, -1)
)
REFERENCE_ATTRIBUTES = (
    "name", "class", "childclass", "body1", "body2", "geom", "joint", "joint1", "joint2",
    "material", "mesh", "site", "sidesite", "tendon",
)
# Order is four finger tendons, thumb opposition joint, thumb CMC and flexor tendons.
_OPEN = np.array([.110387, .110387, .110387, .110387, .0, .038389, .112138])
_CLOSE = np.array([.080, .080, .080, .080, 1.30, .032, .092])
_TARGET_RATE = np.array([.12, .12, .12, .12, 4.2, .12, .12])


def _load_source(source: Path) -> dict[str, ET.Element]:
    directory = source.resolve()
    if directory.name != MODEL_DIR:
        directory /= MODEL_DIR
    templates = {}
    mesh_files = set()
    for side, expected in XML_SHA256.items():
        path = directory / f"{side}_hand.xml"
        xml = path.read_bytes()
        if sha256(xml).hexdigest() != expected:
            raise ValueError(f"expected {side} Aero MJCF from Menagerie {UPSTREAM_REVISION}")
        template = ET.fromstring(xml)
        mesh_files.update(mesh.attrib["file"] for mesh in template.findall("./asset/mesh"))
        templates[side] = template
    digest = sha256()
    for filename in sorted(mesh_files):
        digest.update(filename.encode() + b"\0" + sha256((directory / "assets" / filename).read_bytes()).digest())
    if digest.hexdigest() != MESH_SET_SHA256:
        raise ValueError(f"expected Aero mesh set from Menagerie {UPSTREAM_REVISION}")
    for template in templates.values():
        for mesh in template.findall("./asset/mesh"):
            mesh.set("file", str(directory / "assets" / mesh.attrib["file"]))
    return templates


def _mount_hand(model: ET.Element, template: ET.Element, side: str, mount: WristMount) -> None:
    prefix = f"aero_{side}_"
    for element in template.iter():
        for key in REFERENCE_ATTRIBUTES:
            if key in element.attrib:
                element.set(key, prefix + element.attrib[key])
    world = model.find("worldbody")
    wrist = world.find(f".//body[@name='{side}_wrist_yaw_link']")
    _set_bare_wrist_inertia(wrist, side)
    root = template.find("./worldbody/body")
    root.set("pos", " ".join(map(str, mount.position_m)))
    root.set("quat", " ".join(map(str, mount.quaternion_wxyz)))
    wrist.append(root)
    for tag in ("asset", "default", "contact", "tendon", "actuator", "sensor", "equality"):
        target = model.find(tag)
        if target is None:
            target = ET.SubElement(model, tag)
        target.extend(template.find(tag))


def make_aero_scene(
    asset_root: Path,
    source: Path,
    *,
    survey_camera: bool = True,
    mounts: tuple[WristMount, WristMount] = DEFAULT_MOUNTS,
) -> str:
    """Build a free-base G1 scene with both pinned Aero hands and all cable couplings."""
    templates = _load_source(source)
    model = ET.fromstring(make_dynamic_scene(asset_root, survey_camera=survey_camera))
    model.set("model", "g1_free_base_with_aero")
    for side, mount in zip(SIDES, mounts, strict=True):
        _mount_hand(model, templates[side], side, mount)
    return ET.tostring(model, encoding="unicode")


def aero_ready_joint_targets(nominal: np.ndarray, names: list[str]) -> np.ndarray:
    """Put the grip faces on either side of the receiver's interception window."""
    targets = cradle_joint_targets(nominal, names)
    for side, sign in zip(SIDES, (1, -1), strict=True):
        targets[names.index(f"{side}_shoulder_roll_joint")] = sign * .04
        targets[names.index(f"{side}_wrist_yaw_joint")] = sign * .16
    return targets


@dataclass(frozen=True)
class AeroActuation:
    """Command fourteen native tendon lengths and thumb angles independently."""

    body_actuators: np.ndarray
    hand_actuators: np.ndarray
    ranges: np.ndarray

    @classmethod
    def from_model(cls, model) -> AeroActuation:
        names = [model.actuator(i).name for i in range(model.nu)]
        side_indices = [[i for i, name in enumerate(names) if name.startswith(f"aero_{side}_")] for side in SIDES]
        if model.nu != 43 or any(len(indices) != 7 for indices in side_indices):
            raise ValueError("expected 29 body motors and seven Aero actuators per hand")
        hand = np.array(side_indices).ravel()
        body = np.flatnonzero(~np.isin(np.arange(model.nu), hand))
        return cls(body, hand, model.actuator_ctrlrange[hand].copy())

    def set_targets(self, data, targets, *, dt_s: float | None = None) -> None:
        """Set native targets in metres for cables and radians for thumb opposition."""
        values = np.asarray(targets, dtype=float)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError("expected 14 finite Aero cable length and angle targets")
        values = np.clip(values, self.ranges[:, 0], self.ranges[:, 1])
        if dt_s is not None:
            if not np.isfinite(dt_s) or dt_s <= 0:
                raise ValueError("Aero target update interval must be finite and positive")
            previous = data.ctrl[self.hand_actuators]
            step = np.tile(_TARGET_RATE * dt_s, 2)
            values = previous + np.clip(values - previous, -step, step)
        data.ctrl[self.hand_actuators] = values

    def set_closure(self, data, fraction, *, dt_s: float | None = None) -> None:
        closure = np.asarray(fraction, dtype=float)
        if closure.shape not in ((), (2,)) or not np.isfinite(closure).all():
            raise ValueError("expected a finite closure fraction or a left/right pair")
        closure = np.broadcast_to(np.clip(closure, 0.0, 1.0), (2,))[:, None]
        self.set_targets(data, (_OPEN + closure * (_CLOSE - _OPEN)).ravel(), dt_s=dt_s)
