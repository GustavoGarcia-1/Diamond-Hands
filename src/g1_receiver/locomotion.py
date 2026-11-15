"""Run a free-base G1 walking, surveyed-route, or stationary contact trial."""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .localization import LEFT_MARKER_M, MARKER_RADIUS_M, RIGHT_MARKER_M, MarkerLocalizer
from .perception import CameraIntrinsics, RGBDFrame, RigidPose
from .routes import RouteFollower, make_route
from .runtime import DEFAULT_RUNTIME_ROOT, RobotObservation, SkillGoal
from .sim import HEIGHT, WIDTH, add_flight_scene, remove_dexterous_meshes

CONTROL_DT = 0.02
PHYSICS_DT = 0.001
SUBSTEPS = round(CONTROL_DT / PHYSICS_DT)
SURVEY_CAMERA_POSITION_M = (3.0, -1.3, 1.55)
SURVEY_CAMERA_TARGET_M = (.7, .2, .85)
SURVEY_CAMERA_FOV_Y_DEG = 48.0
MOVING_RELEASE_POSITION_M = (.84, 0., 1.17)
MOVING_RELEASE_VELOCITY_MPS = (-.79, 0., 3.5)
CRADLE_TARGET_RATE_RAD_S = 3.0
CATCH_BALL_CENTER_OFFSET_M = .075
PELVIS_MARKERS = {
    "cyan": (LEFT_MARKER_M, (.05, .90, .90)),
    "magenta": (RIGHT_MARKER_M, (.95, .05, .85)),
}


def survey_camera_rotation() -> np.ndarray:
    """Rotation from optical x-right/y-down/z-forward to field x/y/z-up."""
    forward = np.subtract(SURVEY_CAMERA_TARGET_M, SURVEY_CAMERA_POSITION_M)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0, 0, 1))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.column_stack((right, down, forward))


def add_survey_fixture(world: ET.Element, pelvis: ET.Element) -> None:
    right, down, _ = survey_camera_rotation().T
    ET.SubElement(world, "camera", name="survey_rgbd",
                  pos=" ".join(map(str, SURVEY_CAMERA_POSITION_M)),
                  xyaxes=" ".join(map(str, (*right, *-down))),
                  fovy=str(SURVEY_CAMERA_FOV_Y_DEG))
    for name, (position, color) in PELVIS_MARKERS.items():
        ET.SubElement(pelvis, "geom", name=f"localization_{name}", type="sphere",
                      pos=" ".join(map(str, position)), size=str(MARKER_RADIUS_M),
                      rgba=" ".join(map(str, (*color, 1))), contype="0", conaffinity="0",
                      mass="0")


def make_dynamic_scene(pool_root: Path, survey_camera: bool = False) -> str:
    model = ET.parse(pool_root / "assets/robot/g1_body.xml").getroot()
    model.find("compiler").set("meshdir", str(pool_root.resolve() / "assets/robot/meshes"))
    model.find("option").attrib.update(
        timestep=str(PHYSICS_DT), integrator="implicitfast", iterations="80", cone="elliptic"
    )
    world = model.find("worldbody")
    remove_dexterous_meshes(world)
    add_flight_scene(model, world)
    pelvis = world.find("body[@name='pelvis']")
    if survey_camera:
        add_survey_fixture(world, pelvis)
    imu = ET.SubElement(pelvis, "site",
                        name="receiver_imu", size=".001", rgba="0 0 0 0")
    sensors = model.find("sensor")
    ET.SubElement(sensors, "framequat", name="receiver_orientation",
                  objtype="site", objname=imu.get("name"))
    ET.SubElement(sensors, "gyro", name="receiver_gyro", site=imu.get("name"))
    for motor in model.find("actuator"):
        joint = world.find(f".//joint[@name='{motor.get('joint')}']")
        effort = min(abs(float(value)) for value in joint.get("actuatorfrcrange").split())
        motor.attrib.update(ctrllimited="true", ctrlrange=f"{-effort} {effort}")
    return ET.tostring(model, encoding="unicode")


def cradle_joint_targets(nominal: np.ndarray, names: list[str]) -> np.ndarray:
    """Mirror the nominal FK posture that places the wrists over the elbows."""
    target = nominal.copy()
    for side, sign in (("left", 1), ("right", -1)):
        target[names.index(f"{side}_shoulder_pitch_joint")] = -.816
        target[names.index(f"{side}_shoulder_roll_joint")] = -sign * .102
        target[names.index(f"{side}_shoulder_yaw_joint")] = sign * .018
        target[names.index(f"{side}_elbow_joint")] = .332
    return target


def run(
    pool_root: Path, mode: str, seconds: float, overview: bool = False, survey_camera: bool = False,
    *, route_name: str = "slant", route_scale: float = .45, camera_hz: float = 30.,
    release_position_m: tuple[float, float, float] = MOVING_RELEASE_POSITION_M,
    release_velocity_mps: tuple[float, float, float] = MOVING_RELEASE_VELOCITY_MPS,
    ball_vision: bool = True,
):
    import mujoco
    from ._vendor.g1_runtime.pool.wbc import WholeBodyController

    visual_mode = mode in ("visual-catch", "visual-moving-catch")
    route_mode = mode in ("route", "moving-catch", "visual-moving-catch")
    record_survey = survey_camera or mode == "route"
    survey_camera = survey_camera or route_mode or visual_mode
    gate = None
    if visual_mode and ball_vision:
        from .catching import VisualCatchGate

        gate = VisualCatchGate(
            allow_synthetic_fixture=True, ball_center_offset_m=CATCH_BALL_CENTER_OFFSET_M
        )
    catch_decision = None
    trigger_time_s = np.nan
    if not np.isfinite(camera_hz) or not 0 < camera_hz <= 1 / CONTROL_DT:
        raise ValueError("camera frequency must be in (0, 50] Hz")
    next_capture_s = 0.0
    if any(
        np.shape(vector) != (3,) or not np.isfinite(vector).all()
        for vector in (release_position_m, release_velocity_mps)
    ):
        raise ValueError("release position and velocity must be finite XYZ vectors")
    route = make_route(route_name, scale=route_scale, speed_mps=.35) if route_mode else None
    follower = RouteFollower(route) if route is not None else None
    localizer = MarkerLocalizer(allow_synthetic_fixture=True) if route_mode or visual_mode else None
    progress_m = 0.0
    last_pose = None
    route_start_s = 0.0 if mode in ("moving-catch", "visual-moving-catch") else .4
    rotation = survey_camera_rotation()
    translation = np.array(SURVEY_CAMERA_POSITION_M)
    focal = HEIGHT / (2 * np.tan(np.deg2rad(SURVEY_CAMERA_FOV_Y_DEG / 2)))
    intrinsics = CameraIntrinsics(focal, focal, (WIDTH - 1) / 2, (HEIGHT - 1) / 2)

    model = mujoco.MjModel.from_xml_string(make_dynamic_scene(pool_root, survey_camera))
    data = mujoco.MjData(model)
    joint_ids = model.actuator_trnid[:, 0]
    names = [model.joint(int(joint)).name for joint in joint_ids]
    qids = model.jnt_qposadr[joint_ids]
    vids = model.jnt_dofadr[joint_ids]
    controller = WholeBodyController(names, model.jnt_range[joint_ids], asset_root=pool_root)
    pelvis = model.body("pelvis").id
    ball = model.body("football").id
    ball_joint = model.joint("football_free")
    ball_qid = ball_joint.qposadr[0]
    ball_vid = ball_joint.dofadr[0]
    ball_geom = model.geom("football_surface").id
    floor_geom = model.geom("floor").id
    foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    foot_bodies = np.array([model.body(name).id for name in foot_names])
    foot_for_geom = {
        int(geom_id): side
        for side, body_id in enumerate(foot_bodies)
        for geom_id in np.flatnonzero((model.geom_bodyid == body_id) & (model.geom_contype != 0))
    }
    data.qpos[qids] = controller.nominal
    data.qpos[2] = .79
    upper = controller.nominal.copy()
    cradle = cradle_joint_targets(controller.nominal, names)
    if mode == "contact":
        for side, sign in (("left", 1), ("right", -1)):
            upper[names.index(f"{side}_shoulder_pitch_joint")] = -.55
            upper[names.index(f"{side}_shoulder_roll_joint")] = sign * .08
            upper[names.index(f"{side}_elbow_joint")] = .75
        data.qpos[qids] = upper
        data.qpos[ball_qid:ball_qid + 3] = (1.1, 0, 1.18)
        data.qvel[ball_vid:ball_vid + 3] = (-1.7, 0, 2.7)
    elif mode in ("cradle", "neutral", "moving-catch"):
        if mode != "neutral":
            upper = cradle.copy()
            data.qpos[qids] = upper
        data.qpos[ball_qid:ball_qid + 3] = (.24, 0, 1.23)
        data.qvel[ball_vid:ball_vid + 3] = (-.10, 0, -.15)
    if mode == "moving-catch" or visual_mode:
        data.qpos[ball_qid:ball_qid + 3] = release_position_m
        data.qvel[ball_vid:ball_vid + 3] = release_velocity_mps
    elif mode in ("walk", "route"):
        data.qpos[ball_qid:ball_qid + 3] = (8, 4, .085)
    mujoco.mj_forward(model, data)

    def read_sensor(name):
        sensor = model.sensor(name)
        return data.sensordata[sensor.adr[0]:sensor.adr[0] + sensor.dim[0]].copy()

    trace = {key: [] for key in (
        "timestamps_s", "ball_position_m", "ball_velocity_mps", "robot_base_position_m",
        "robot_base_quaternion_wxyz", "ball_ground_contact", "ball_robot_contact",
        "robot_upright", "command_forward_mps", "policy",
    )}
    contact_times = []
    contact_geoms = []
    foot_stamps = []
    foot_positions = []
    foot_floor_contacts = []
    frames = []
    survey_stamps = []
    survey_rgb = []
    survey_depth = []
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH) if overview or survey_camera else None
    commands = {key: [] for key in (
        "timestamps_s", "estimate_timestamps_s", "estimated_xy_m", "estimated_yaw_rad",
        "status", "command_base_velocity", "estimated_progress_m", "next_waypoint",
    )}
    catch_log = {key: [] for key in (
        "catch_status", "catch_latched", "catch_triggered", "catch_time_to_crossing_s",
        "catch_relative_crossing_m", "catch_velocity_mps", "upper_target_rad",
        "catch_capture_timestamp_s", "catch_ball_visible", "catch_crossing_time_std_s",
        "catch_inliers",
    )}
    robot_contact = False
    ground_contact = False
    try:
        for tick in range(round(seconds / CONTROL_DT) + 1):
            velocity = np.zeros(3)
            if mode == "walk" and .4 <= data.time < seconds - .4:
                velocity[0] = .35
            stamp = float(data.time)
            capture = survey_camera and stamp >= next_capture_s - 1e-8
            if capture:
                next_capture_s += 1 / camera_hz
                renderer.update_scene(data, camera="survey_rgbd")
                rgb = renderer.render()
                renderer.enable_depth_rendering()
                depth = renderer.render()
                renderer.disable_depth_rendering()
                if record_survey:
                    survey_stamps.append(stamp)
                    survey_rgb.append(rgb)
                    survey_depth.append(depth)
            triggered_now = False
            if capture and localizer is not None:
                survey = RigidPose(stamp, "camera_optical", "field", rotation, translation,
                                   "synthetic_fixture")
                frame = RGBDFrame(stamp, "camera_optical", rgb, depth, intrinsics)
                pose = localizer.estimate(frame, survey)
                if pose is not None:
                    last_pose = pose
                if gate is not None:
                    catch_decision = gate.update(frame, survey, pose, stamp)
                    triggered_now = catch_decision.triggered
                    if triggered_now:
                        trigger_time_s = stamp

            if visual_mode:
                if catch_decision is not None and catch_decision.latched:
                    max_delta = CRADLE_TARGET_RATE_RAD_S * CONTROL_DT
                    upper += np.clip(cradle - upper, -max_delta, max_delta)
                catch_log["catch_status"].append(
                    catch_decision.status if catch_decision is not None else
                    ("waiting_frame" if ball_vision else "ball_vision_disabled")
                )
                catch_log["catch_latched"].append(bool(catch_decision and catch_decision.latched))
                catch_log["catch_triggered"].append(triggered_now)
                catch_log["catch_capture_timestamp_s"].append(
                    catch_decision.capture_timestamp_s if catch_decision else np.nan
                )
                catch_log["catch_ball_visible"].append(
                    bool(catch_decision and catch_decision.ball_visible)
                )
                prediction = catch_decision.prediction if catch_decision else None
                catch_log["catch_crossing_time_std_s"].append(
                    prediction.time_std_s if prediction else np.nan
                )
                catch_log["catch_inliers"].append(prediction.inliers if prediction else 0)
                time_to_crossing = catch_decision.time_to_crossing_s if catch_decision else None
                catch_log["catch_time_to_crossing_s"].append(
                    time_to_crossing if time_to_crossing is not None else np.nan
                )
                for key, value in (
                    ("catch_relative_crossing_m",
                     catch_decision.receiver_relative_crossing_m if catch_decision else None),
                    ("catch_velocity_mps", catch_decision.velocity_mps if catch_decision else None),
                ):
                    catch_log[key].append(value if value is not None else (np.nan, np.nan, np.nan))
                catch_log["upper_target_rad"].append(upper.copy())

            if route_mode or visual_mode:
                fresh = last_pose is not None and last_pose.fresh(stamp, .06)
                status = "no_fresh_pose"
                if fresh:
                    status = "settle"
                    if route is not None:
                        _, progress_m = route.project(last_pose.xy_m, progress_m)
                    if route is not None and route_start_s <= stamp < seconds - .4:
                        field_v = follower.command(last_pose.xy_m)
                        cos_yaw, sin_yaw = np.cos(last_pose.yaw_rad), np.sin(last_pose.yaw_rad)
                        velocity[:2] = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]]) @ field_v
                        status = "arrived" if follower.next_waypoint == len(route.waypoints) else "follow_route"
                commands["timestamps_s"].append(stamp)
                commands["estimate_timestamps_s"].append(last_pose.timestamp_s if fresh else np.nan)
                commands["estimated_xy_m"].append(last_pose.xy_m if fresh else (np.nan, np.nan))
                commands["estimated_yaw_rad"].append(last_pose.yaw_rad if fresh else np.nan)
                commands["status"].append(status)
                commands["command_base_velocity"].append(velocity.copy())
                commands["estimated_progress_m"].append(progress_m)
                commands["next_waypoint"].append(follower.next_waypoint if follower else 0)
            observation = RobotObservation(
                float(data.time), data.qpos[qids].copy(), data.qvel[vids].copy(),
                read_sensor("receiver_orientation"), read_sensor("receiver_gyro"),
                data.actuator_force.copy(), None,
            )
            goal = SkillGoal(base_velocity=velocity)
            command = controller.act(observation, goal, upper_target=upper)
            trace["timestamps_s"].append(data.time)
            trace["ball_position_m"].append(data.xpos[ball].copy())
            trace["ball_velocity_mps"].append(data.qvel[ball_vid:ball_vid + 3].copy())
            trace["robot_base_position_m"].append(data.xpos[pelvis].copy())
            trace["robot_base_quaternion_wxyz"].append(data.xquat[pelvis].copy())
            trace["robot_upright"].append(data.xmat[pelvis][8])
            trace["ball_ground_contact"].append(ground_contact)
            trace["ball_robot_contact"].append(robot_contact)
            trace["command_forward_mps"].append(velocity[0])
            trace["policy"].append(command.status)
            if overview:
                renderer.update_scene(data, camera="overview")
                frames.append(renderer.render())
            if tick == round(seconds / CONTROL_DT):
                break

            ground_contact = False
            robot_contact = False
            for _ in range(SUBSTEPS):
                torque = command.kp * (command.target_position - data.qpos[qids])
                torque -= command.kd * data.qvel[vids]
                data.ctrl[:] = np.clip(torque, *model.actuator_ctrlrange.T)
                mujoco.mj_step(model, data)
                # Contacts and Cartesian positions describe the state used for this
                # solver step, before integration advances the clock by PHYSICS_DT.
                foot_on_floor = np.zeros(2, dtype=bool)
                for contact in data.contact:
                    if floor_geom in (contact.geom1, contact.geom2) and contact.efc_address >= 0:
                        other_geom = contact.geom2 if contact.geom1 == floor_geom else contact.geom1
                        side = foot_for_geom.get(other_geom)
                        if side is not None:
                            foot_on_floor[side] = True
                    if ball_geom not in (contact.geom1, contact.geom2):
                        continue
                    other = contact.geom2 if contact.geom1 == ball_geom else contact.geom1
                    if other == floor_geom:
                        ground_contact = True
                    elif model.geom_bodyid[other] != 0:
                        robot_contact = True
                        contact_times.append(data.time - PHYSICS_DT)
                        contact_geoms.append(model.body(int(model.geom_bodyid[other])).name)
                foot_stamps.append(data.time - PHYSICS_DT)
                foot_positions.append(data.xpos[foot_bodies].copy())
                foot_floor_contacts.append(foot_on_floor)
            mujoco.mj_forward(model, data)
    finally:
        if renderer is not None:
            renderer.close()
    trace["evidence_class"] = "physical_simulation"
    trace["mode"] = mode
    trace["robot_contact_timestamps_s"] = contact_times
    trace["robot_contact_bodies"] = np.asarray(contact_geoms, dtype=str)
    trace["foot_body_names"] = foot_names
    trace["foot_timestamps_s"] = foot_stamps
    trace["foot_position_m"] = foot_positions
    trace["foot_floor_contact"] = foot_floor_contacts
    sensor = {}
    if record_survey:
        stamps = np.asarray(survey_stamps)
        sensor = {
            "timestamps_s": stamps,
            "rgb": np.asarray(survey_rgb),
            "depth_m": np.asarray(survey_depth),
            "intrinsics": np.array([intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]),
            "camera_rotation": np.broadcast_to(rotation, (len(stamps), 3, 3)),
            "camera_translation_m": np.broadcast_to(translation, (len(stamps), 3)),
            "camera_pose_timestamps_s": stamps,
            "pose_provenance": np.array("synthetic_fixture"),
            "evidence_class": np.array("rendered_simulation"),
            "nominal_camera_hz": np.array(camera_hz),
        }
    if route_mode or visual_mode:
        commands["pose_provenance"] = "synthetic_fixture"
        commands["nominal_camera_hz"] = camera_hz
    if route is not None:
        commands["route_name"] = route.name
        commands["route_waypoints_m"] = route.waypoints
    if visual_mode:
        commands.update(catch_log)
        commands["catch_trigger_time_s"] = trigger_time_s
        commands["ball_vision_enabled"] = ball_vision
        commands["arm_target_rate_rad_s"] = CRADLE_TARGET_RATE_RAD_S
        commands["ball_center_offset_m"] = CATCH_BALL_CENTER_OFFSET_M
        commands["actuator_joint_names"] = names
    return (
        {key: np.asarray(values) for key, values in trace.items()},
        np.asarray(frames), sensor, {key: np.asarray(values) for key, values in commands.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", type=Path, default=DEFAULT_RUNTIME_ROOT,
                        help="Override the bundled G1 runtime asset directory")
    parser.add_argument("--mode", choices=("walk", "contact", "cradle", "neutral", "route", "moving-catch", "visual-catch", "visual-moving-catch"), required=True)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--overview", type=Path)
    parser.add_argument("--sensor", type=Path, help="Optional surveyed external RGB-D replay with pelvis markers")
    parser.add_argument("--controller", type=Path, help="Route estimates and commands, separate from truth")
    parser.add_argument("--route", choices=("slant", "hitch", "out", "post", "go"), default="slant")
    parser.add_argument("--route-scale", type=float, default=.45)
    parser.add_argument("--camera-hz", type=float, default=30.)
    parser.add_argument("--release-position", type=float, nargs=3, default=MOVING_RELEASE_POSITION_M)
    parser.add_argument("--release-velocity", type=float, nargs=3, default=MOVING_RELEASE_VELOCITY_MPS)
    parser.add_argument("--disable-ball-vision", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < CONTROL_DT:
        parser.error("--seconds must be finite and at least 0.02")
    if args.controller is not None and args.mode not in ("route", "moving-catch", "visual-catch", "visual-moving-catch"):
        parser.error("--controller is available only in camera-driven modes")
    if args.disable_ball_vision and args.mode not in ("visual-catch", "visual-moving-catch"):
        parser.error("--disable-ball-vision requires a visual catch mode")
    outputs = [path for path in (args.truth, args.overview, args.sensor, args.controller) if path is not None]
    if any(path.suffix != ".npz" for path in outputs) or len({p.resolve() for p in outputs}) != len(outputs):
        parser.error("outputs must use distinct .npz paths")
    trace, frames, sensor, commands = run(
        args.pool_root, args.mode, args.seconds, args.overview is not None, args.sensor is not None,
        route_name=args.route, route_scale=args.route_scale, camera_hz=args.camera_hz,
        release_position_m=args.release_position, release_velocity_mps=args.release_velocity,
        ball_vision=not args.disable_ball_vision,
    )
    np.savez_compressed(args.truth, **trace)
    if args.overview is not None:
        np.savez_compressed(args.overview, rgb=frames, timestamps_s=trace["timestamps_s"])
    if args.sensor is not None:
        np.savez_compressed(args.sensor, **sensor)
    if args.controller is not None:
        np.savez_compressed(args.controller, **commands)
    positions = trace["robot_base_position_m"]
    print(f"{args.mode}: x displacement {positions[-1, 0] - positions[0, 0]:.3f} m, "
          f"min height {positions[:, 2].min():.3f} m, "
          f"robot-contact sample count {trace['ball_robot_contact'].sum()}")


if __name__ == "__main__":
    main()
