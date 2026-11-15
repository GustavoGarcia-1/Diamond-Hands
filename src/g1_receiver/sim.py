"""Generate a MuJoCo football flight and an independent RGB-D replay."""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .runtime import DEFAULT_ROBOT_XML

CAMERA_NAME = "receiver_rgbd"
CAMERA_POSITION_M = (0.10, 0.0, 1.22)
CAMERA_ROTATION = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
FOV_Y_DEG = 53.0
BALL_POSITION_M = (5.0, 0.15, 1.38)
BALL_VELOCITY_MPS = (-5.5, -0.2, 3.4)
WIDTH = 320
HEIGHT = 180
FPS = 30
PHYSICS_STEPS_PER_FRAME = 20


def make_scene(
    robot_xml: Path | None,
    release_position_m: tuple[float, float, float] = BALL_POSITION_M,
    camera_position_m: tuple[float, float, float] = CAMERA_POSITION_M,
) -> str:
    """Add a static, collision-free G1 visual reference when supplied."""
    if robot_xml is None:
        model = ET.Element("mujoco", model="receiver_sensor_baseline")
        world = ET.SubElement(model, "worldbody")
    else:
        model = ET.parse(robot_xml).getroot()
        compiler = model.find("compiler")
        world = model.find("worldbody")
        if compiler is None or world is None or world.find("body[@name='pelvis']") is None:
            raise ValueError("expected the GR00T G1 MJCF with compiler and pelvis")
        compiler.set("meshdir", str(robot_xml.resolve().parent / compiler.get("meshdir", "")))
        for tag in ("actuator", "sensor"):
            section = model.find(tag)
            if section is not None:
                model.remove(section)
        for body in world.iter("body"):
            for child in list(body):
                if child.tag in ("joint", "freejoint", "site"):
                    body.remove(child)
        remove_dexterous_meshes(world)
        for geom in world.iter("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

    timestep = 1.0 / (FPS * PHYSICS_STEPS_PER_FRAME)
    option = model.find("option")
    if option is None:
        option = ET.SubElement(model, "option")
    option.attrib.update(timestep=str(timestep), integrator="RK4")
    add_flight_scene(model, world, release_position_m, camera_position_m)
    return ET.tostring(model, encoding="unicode")


def remove_dexterous_meshes(world: ET.Element) -> None:
    """Remove inherited articulated hand geometry from the bare-wrist model."""
    for body in world.iter("body"):
        for child in list(body):
            if (child.tag == "body" and "_hand_" in child.get("name", "")) or (
                child.tag == "geom" and "_hand_palm_" in child.get("mesh", "")
            ):
                body.remove(child)


def add_flight_scene(
    model: ET.Element,
    world: ET.Element,
    release_position_m: tuple[float, float, float] = BALL_POSITION_M,
    camera_position_m: tuple[float, float, float] = CAMERA_POSITION_M,
) -> None:
    """Append the shared football, field, lights and fixed cameras."""
    for section in model.findall("worldbody"):
        for geom in section.findall("geom"):
            if geom.get("type") == "plane":
                section.remove(geom)
    visual = model.find("visual")
    if visual is not None:
        model.remove(visual)
    visual = ET.SubElement(model, "visual")
    ET.SubElement(visual, "global", offwidth=str(WIDTH), offheight=str(HEIGHT))
    ET.SubElement(visual, "headlight", ambient=".35 .35 .35", diffuse=".7 .7 .7")
    ET.SubElement(world, "light", pos="0 -3 6", dir="0 .4 -1")
    ET.SubElement(world, "geom", name="floor", type="plane", size="20 20 .05",
                  rgba=".18 .37 .21 1", friction=".7 .04 .01")
    ET.SubElement(world, "geom", name="backdrop", type="box", pos="9 0 2", size=".05 6 2",
                  rgba=".27 .31 .34 1", contype="0", conaffinity="0")
    ET.SubElement(world, "camera", name=CAMERA_NAME, pos=" ".join(map(str, camera_position_m)),
                  xyaxes="0 -1 0 0 0 1", fovy=str(FOV_Y_DEG))
    ET.SubElement(world, "camera", name="overview", pos="5 -6 3",
                  xyaxes=".8944 .4472 0 -.13 .26 .96", fovy="48")
    ball = ET.SubElement(world, "body", name="football", pos=" ".join(map(str, release_position_m)))
    ET.SubElement(ball, "freejoint", name="football_free")
    ET.SubElement(ball, "geom", name="football_surface", type="ellipsoid",
                  size=".142 .085 .085", mass=".42", rgba=".72 .29 .09 1")


def simulate(
    frames: int,
    robot_xml: Path | None = None,
    overview: bool = False,
    *,
    release_position_m: tuple[float, float, float] = BALL_POSITION_M,
    release_velocity_mps: tuple[float, float, float] = BALL_VELOCITY_MPS,
    camera_position_m: tuple[float, float, float] = CAMERA_POSITION_M,
):
    """Render synchronized RGB-D, with sample-aligned physical truth returned separately."""
    import mujoco

    if frames <= 0:
        raise ValueError("frames must be positive")
    vectors = (release_position_m, release_velocity_mps, camera_position_m)
    if any(np.shape(vector) != (3,) or not np.isfinite(vector).all() for vector in vectors):
        raise ValueError("release position, velocity and camera position must be finite XYZ vectors")
    model = mujoco.MjModel.from_xml_string(make_scene(robot_xml, release_position_m, camera_position_m))
    data = mujoco.MjData(model)
    football = model.body("football").id
    football_geom = model.geom("football_surface").id
    floor_geom = model.geom("floor").id
    velocity_start = model.joint("football_free").dofadr[0]
    data.qvel[velocity_start:velocity_start + 6] = (*release_velocity_mps, 30.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)

    focal_length = HEIGHT / (2 * np.tan(np.deg2rad(FOV_Y_DEG / 2)))
    rgb = np.empty((frames, HEIGHT, WIDTH, 3), dtype=np.uint8)
    depth = np.empty((frames, HEIGHT, WIDTH), dtype=np.float32)
    sensor = {
        "timestamps_s": np.empty(frames),
        "rgb": rgb,
        "depth_m": depth,
        "intrinsics": np.array([focal_length, focal_length, (WIDTH - 1) / 2, (HEIGHT - 1) / 2]),
        "camera_rotation": np.broadcast_to(CAMERA_ROTATION, (frames, 3, 3)),
        "camera_translation_m": np.broadcast_to(camera_position_m, (frames, 3)),
    }
    sensor["evidence_class"] = np.array("rendered_simulation")
    sensor["pose_provenance"] = np.array("synthetic_fixture")
    sensor["camera_pose_timestamps_s"] = sensor["timestamps_s"]
    truth = {
        "timestamps_s": sensor["timestamps_s"],
        "ball_position_m": np.empty((frames, 3)),
        "ball_velocity_mps": np.empty((frames, 3)),
        "ball_quaternion_wxyz": np.empty((frames, 4)),
        "ball_ground_contact": np.zeros(frames, dtype=bool),
        "robot_model_present": np.array(robot_xml is not None),
        "evidence_class": np.array("rendered_simulation"),
        "physics_assumptions": np.array("rigid ellipsoid with gravity and floor contact; no aerodynamic model"),
        "robot_motion": np.array("static collision-free visual reference" if robot_xml else "absent"),
    }
    if robot_xml is not None:
        pelvis = model.body("pelvis").id
        truth["robot_base_position_m"] = np.broadcast_to(data.xpos[pelvis].copy(), (frames, 3))
        truth["robot_base_quaternion_wxyz"] = np.broadcast_to(data.xquat[pelvis].copy(), (frames, 4))
    overview_rgb = np.empty_like(rgb) if overview else None
    contact_ids = {football_geom, floor_geom}
    contacted_ground = False
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        for index in range(frames):
            sensor["timestamps_s"][index] = data.time
            truth["ball_position_m"][index] = data.xpos[football]
            truth["ball_velocity_mps"][index] = data.qvel[velocity_start:velocity_start + 3]
            truth["ball_quaternion_wxyz"][index] = data.xquat[football]
            truth["ball_ground_contact"][index] = contacted_ground

            renderer.update_scene(data, camera=CAMERA_NAME)
            rgb[index] = renderer.render()
            renderer.enable_depth_rendering()
            depth[index] = renderer.render()
            renderer.disable_depth_rendering()
            if overview_rgb is not None:
                renderer.update_scene(data, camera="overview")
                overview_rgb[index] = renderer.render()

            contacted_ground = False
            for _ in range(PHYSICS_STEPS_PER_FRAME):
                mujoco.mj_step(model, data)
                contacted_ground |= any(
                    {contact.geom1, contact.geom2} == contact_ids for contact in data.contact
                )
            # mj_step integrates qpos; refresh Cartesian quantities for the sampled state.
            mujoco.mj_forward(model, data)
    return sensor, truth, overview_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--overview", type=Path, help="Optional RGB-only spectator replay")
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML,
                        help="GR00T G1 MJCF for the static visual reference")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--release-position", type=float, nargs=3, default=BALL_POSITION_M)
    parser.add_argument("--release-velocity", type=float, nargs=3, default=BALL_VELOCITY_MPS)
    parser.add_argument("--camera-position", type=float, nargs=3, default=CAMERA_POSITION_M)
    args = parser.parse_args()

    outputs = [path for path in (args.sensor, args.truth, args.overview) if path is not None]
    if any(path.suffix != ".npz" for path in outputs):
        parser.error("outputs must use the .npz extension")
    if len({path.resolve() for path in outputs}) != len(outputs):
        parser.error("sensor, truth and overview outputs must have different paths")
    if args.frames <= 0:
        parser.error("--frames must be positive")

    sensor, truth, overview = simulate(
        args.frames, args.robot_xml, args.overview is not None,
        release_position_m=args.release_position, release_velocity_mps=args.release_velocity,
        camera_position_m=args.camera_position,
    )
    np.savez_compressed(args.sensor, **sensor)
    np.savez_compressed(args.truth, **truth)
    if args.overview is not None:
        np.savez_compressed(args.overview, rgb=overview, timestamps_s=sensor["timestamps_s"])
    print(f"Rendered {args.frames} MuJoCo RGB-D frames at {FPS} Hz")


if __name__ == "__main__":
    main()
