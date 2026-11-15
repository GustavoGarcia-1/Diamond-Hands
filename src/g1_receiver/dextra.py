"""Optional articulated House of Dextra hands for the free-base G1 simulation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as ET

import numpy as np

from .locomotion import cradle_joint_targets, make_dynamic_scene
from .sim import remove_dexterous_meshes

UPSTREAM_URDF = Path(
    "IsaacLab/source/codesign/codesign/assets/lego_hand_v2/"
    "hand_f1_110_f2_110_f3_110_f4_110_f5_110.urdf"
)
UPSTREAM_REVISION = "0f4d0f2ae97420817a05e83955bdc33981ec5146"
UPSTREAM_URDF_SHA256 = "2acf3b7bfb6c3965821aea6c7964a7e9ccc59a57052fb76d6a856cea5eb15ee5"
UPSTREAM_MESH_SET_SHA256 = "1b0c5182a4f57431f5be27f7b00c743b9907720b86b3d4ab596b525203c87395"
COMMAND_SPEED_RAD_S = 7.2
EFFORT_LIMIT_NM = 0.35
FLEXION_TARGET_RAD = 0.60
JOINTS_PER_HAND = 20
BARE_WRIST_MASS_KG = 0.08457647
ADAPTER_MASS_KG = 0.045
SIDES = ("left", "right")


@dataclass(frozen=True)
class WristMount:
    """Simulation-only pose of the palm in the G1 wrist-yaw body frame."""

    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if np.shape(self.position_m) != (3,) or not np.isfinite(self.position_m).all():
            raise ValueError("mount position must be a finite XYZ vector")
        if (
            np.shape(self.quaternion_wxyz) != (4,)
            or not np.isfinite(self.quaternion_wxyz).all()
            or not np.isclose(np.linalg.norm(self.quaternion_wxyz), 1.0)
        ):
            raise ValueError("mount quaternion must be a finite unit WXYZ vector")


# The reference wrist mesh ends at X=41.5 mm. The collars bridge to a palm
# origin 10.5 mm farther forward. Opposing palm faces point toward the ball.
_QUARTER_TURN = np.sqrt(0.5)
DEFAULT_MOUNTS = (
    WristMount((0.052, 0.012, 0.0), (_QUARTER_TURN, -_QUARTER_TURN, 0.0, 0.0)),
    WristMount((0.052, -0.012, 0.0), (_QUARTER_TURN, _QUARTER_TURN, 0.0, 0.0)),
)


def verify_source_urdf(urdf: Path) -> str:
    digest = hashlib.sha256(urdf.read_bytes()).hexdigest()
    if digest != UPSTREAM_URDF_SHA256:
        raise ValueError(f"expected the Dextra URDF from revision {UPSTREAM_REVISION}")
    return digest


def import_dextra_urdf(urdf: Path) -> ET.Element:
    """Import the pinned URDF with separate visual and collision meshes."""
    import mujoco

    verify_source_urdf(urdf)
    source = ET.parse(urdf).getroot()
    mesh_files = sorted({mesh.attrib["filename"] for mesh in source.findall(".//mesh")})
    digest = hashlib.sha256()
    for mesh_file in mesh_files:
        digest.update(mesh_file.encode() + b"\0")
        digest.update(hashlib.sha256((urdf.parent / mesh_file).read_bytes()).digest())
    if digest.hexdigest() != UPSTREAM_MESH_SET_SHA256:
        raise ValueError(f"expected the Dextra mesh files from revision {UPSTREAM_REVISION}")
    ET.SubElement(
        ET.SubElement(source, "mujoco"), "compiler",
        strippath="false", discardvisual="false", fusestatic="false",
        meshdir=str(urdf.resolve().parent),
    )
    spec = mujoco.MjSpec.from_string(ET.tostring(source, encoding="unicode"))
    return ET.fromstring(spec.to_xml())


def add_position_actuator(actuators: ET.Element, joint: ET.Element) -> None:
    name = joint.attrib["name"]
    joint.attrib.pop("actuatorfrcrange", None)
    # Keep the imported hand dynamics independent of the G1 joint defaults.
    joint.set("armature", "0")
    ET.SubElement(
        actuators, "position", name=name, joint=name, kp="2.8",
        forcerange=f"{-EFFORT_LIMIT_NM} {EFFORT_LIMIT_NM}", ctrlrange=joint.attrib["range"],
    )


def _text(values: Sequence[float]) -> str:
    return " ".join(map(str, values))


def _reflect(element: ET.Element, attribute: str, signs: Sequence[int]) -> None:
    values = np.fromstring(element.attrib[attribute], sep=" ")
    element.set(attribute, _text(values * signs))


def _set_bare_wrist_inertia(wrist: ET.Element, side: str) -> None:
    # Unitree unitree_ros rev_1_0 bare wrist, revision 725b59b6. The vendored
    # GR00T wrist's inertia includes the stock palm even with its meshes removed.
    mirror = 1 if side == "left" else -1
    wrist.remove(wrist.find("inertial"))
    inertia = ET.Element(
        "inertial", mass=str(BARE_WRIST_MASS_KG),
        pos=_text((0.02200381568, mirror * 0.00049485096, 0.00053861123)),
        fullinertia=_text((
            0.00004929128828, 0.00005973338134, 0.00003928083826,
            -mirror * 0.00000045735494, 0.00000445867591, mirror * 0.00000043217198,
        )),
    )
    wrist.insert(0, inertia)


def _add_hand(
    scene: ET.Element,
    template: ET.Element,
    mesh_directory: Path,
    side: str,
    mount: WristMount,
    compliant_tips: bool,
) -> None:
    prefix = f"dextra_{side}_"
    mirrored = side == "right"
    assets = scene.find("asset")
    for source_mesh in template.findall("./asset/mesh"):
        mesh = ET.SubElement(assets, "mesh", source_mesh.attrib)
        mesh.set("name", prefix + source_mesh.attrib["name"])
        mesh.set("file", str(mesh_directory / source_mesh.attrib["file"]))
        if mirrored:
            mesh.set("scale", "1 -1 1")

    palm = ET.fromstring(ET.tostring(template.find("./worldbody/body[@name='palm']")))
    contact = scene.find("contact")
    for element in palm.iter():
        if mirrored:
            if "pos" in element.attrib:
                _reflect(element, "pos", (1, -1, 1))
            if "quat" in element.attrib:
                _reflect(element, "quat", (1, -1, 1, -1))
            if element.tag == "joint":
                element.attrib.setdefault("axis", "0 0 1")
                _reflect(element, "axis", (-1, 1, -1))
        if "name" in element.attrib:
            element.set("name", prefix + element.attrib["name"])
        if element.tag == "geom":
            element.set("mesh", prefix + element.attrib["mesh"])
    for body in palm.iter("body"):
        for index, geom in enumerate(body.findall("geom")):
            geom.set("name", f"{body.attrib['name']}_geom_{index}")
            if geom.get("contype") == "0":
                geom.set("rgba", ".20 .23 .26 1")
            elif compliant_tips and body.attrib["name"].endswith("_finger_tip"):
                ET.SubElement(
                    contact, "pair", geom1="football_surface", geom2=geom.attrib["name"],
                    friction="0.95 0.95 0.005 0.0001 0.0001",
                    solref="0.030 1.10", solimp="0.85 0.95 0.004",
                )

    wrist = scene.find(f".//body[@name='{side}_wrist_yaw_link']")
    _set_bare_wrist_inertia(wrist, side)
    adapter = ET.SubElement(wrist, "body", name=f"{prefix}adapter")
    ET.SubElement(
        adapter, "geom", name=f"{prefix}adapter_geom", type="capsule", size="0.017",
        fromto=_text((0.034, 0.0, 0.0, *mount.position_m)), mass=str(ADAPTER_MASS_KG),
        rgba=".28 .32 .35 1",
    )
    adapter.append(palm)
    palm.attrib.update(pos=_text(mount.position_m), quat=_text(mount.quaternion_wxyz))

    actuators = scene.find("actuator")
    sensors = scene.find("sensor")
    start = 0 if side == "left" else JOINTS_PER_HAND
    joints = list(palm.iter("joint"))
    if len(joints) != JOINTS_PER_HAND:
        raise ValueError(f"expected {JOINTS_PER_HAND} Dextra joints, found {len(joints)}")
    for index, joint in enumerate(joints, start=start):
        add_position_actuator(actuators, joint)
        name = joint.attrib["name"]
        ET.SubElement(sensors, "jointpos", name=f"dextra_encoder_q_{index}", joint=name)
        ET.SubElement(sensors, "jointvel", name=f"dextra_encoder_dq_{index}", joint=name)
        ET.SubElement(sensors, "actuatorfrc", name=f"dextra_effort_{index}", actuator=name)


def attach_dextra_hands(
    xml: str,
    codesign_source: Path,
    *,
    mounts: tuple[WristMount, WristMount] = DEFAULT_MOUNTS,
    compliant_tips: bool = True,
) -> str:
    """Augment the receiver scene while preserving body actuator and sensor names."""
    scene = ET.fromstring(xml)
    world = scene.find("worldbody")
    if world is None:
        raise ValueError("expected a G1 worldbody")
    has_floating_base = any(
        world.find(f".//{tag}[@name='floating_base_joint']") is not None
        for tag in ("joint", "freejoint")
    )
    if (
        not has_floating_base
        or any(world.find(f".//body[@name='{side}_wrist_yaw_link']") is None for side in SIDES)
        or any(world.find(f".//body[@name='dextra_{side}_palm']") is not None for side in SIDES)
        or world.find(".//geom[@name='football_surface']") is None
        or scene.find("sensor") is None or scene.find("actuator") is None
    ):
        raise ValueError("expected a free-base G1 receiver scene before Dextra attachment")
    remove_dexterous_meshes(world)
    urdf = (codesign_source / UPSTREAM_URDF).resolve()
    template = import_dextra_urdf(urdf)
    scene.set("model", "g1_free_base_with_dextra")
    if scene.find("contact") is None:
        ET.SubElement(scene, "contact")
    for side, mount in zip(SIDES, mounts, strict=True):
        _add_hand(scene, template, urdf.parent, side, mount, compliant_tips)
    return ET.tostring(scene, encoding="unicode")


def make_dextra_scene(
    asset_root: Path,
    codesign_source: Path,
    *,
    survey_camera: bool = True,
    mounts: tuple[WristMount, WristMount] = DEFAULT_MOUNTS,
    compliant_tips: bool = True,
) -> str:
    """Build the opt-in free-base G1, football and two 20-DoF Dextra hands."""
    return attach_dextra_hands(
        make_dynamic_scene(asset_root, survey_camera), codesign_source,
        mounts=mounts, compliant_tips=compliant_tips,
    )


def dextra_ready_joint_targets(nominal: np.ndarray, names: list[str]) -> np.ndarray:
    """Set a provisional arm posture with space between the opposed palms."""
    target = cradle_joint_targets(nominal, names)
    for side, sign in (("left", 1), ("right", -1)):
        target[names.index(f"{side}_shoulder_roll_joint")] = sign * 0.15
        target[names.index(f"{side}_wrist_yaw_joint")] = sign * 0.20
    return target


@dataclass(frozen=True)
class DextraActuation:
    """Preserve the WBC torque slice and independently address the hand drives."""

    body_actuators: np.ndarray
    hand_actuators: np.ndarray
    hand_joint_qpos: np.ndarray
    hand_joint_dof: np.ndarray
    hand_ranges_rad: np.ndarray
    flexion: np.ndarray

    @classmethod
    def from_model(cls, model) -> DextraActuation:
        names = [model.actuator(index).name for index in range(model.nu)]
        left = [index for index, name in enumerate(names) if name.startswith("dextra_left_")]
        right = [index for index, name in enumerate(names) if name.startswith("dextra_right_")]
        if len(left) != JOINTS_PER_HAND or len(right) != JOINTS_PER_HAND or model.nu != 69:
            raise ValueError("expected 29 body torque motors and 20 Dextra drives per side")
        hand = np.array(left + right)
        body = np.flatnonzero(~np.isin(np.arange(model.nu), hand))
        joints = model.actuator_trnid[hand, 0]
        flexion = np.array([not names[index].endswith("_palm_joint") for index in hand])
        return cls(body, hand, model.jnt_qposadr[joints], model.jnt_dofadr[joints], model.actuator_ctrlrange[hand], flexion)

    def set_targets(self, data, target_rad: Sequence[float], *, dt_s: float | None = None) -> None:
        target = np.asarray(target_rad, dtype=float)
        if target.shape != self.hand_actuators.shape or not np.isfinite(target).all():
            raise ValueError("expected 40 finite Dextra position targets in radians")
        bounded = np.clip(target, self.hand_ranges_rad[:, 0], self.hand_ranges_rad[:, 1])
        if dt_s is not None:
            if not np.isfinite(dt_s) or dt_s <= 0:
                raise ValueError("hand command period must be positive and finite")
            previous = data.ctrl[self.hand_actuators]
            maximum_step = COMMAND_SPEED_RAD_S * dt_s
            bounded = np.clip(bounded, previous - maximum_step, previous + maximum_step)
        data.ctrl[self.hand_actuators] = bounded

    def set_closure(self, data, fraction: float | Sequence[float], *, dt_s: float | None = None) -> None:
        """Apply symmetric flexion or separate left/right closure fractions."""
        closure = np.asarray(fraction, dtype=float)
        if closure.shape not in ((), (2,)) or not np.isfinite(closure).all():
            raise ValueError("expected a finite closure fraction or a left/right pair")
        levels = np.repeat(np.broadcast_to(np.clip(closure, 0.0, 1.0), (2,)), JOINTS_PER_HAND)
        self.set_targets(data, FLEXION_TARGET_RAD * levels * self.flexion, dt_s=dt_s)
