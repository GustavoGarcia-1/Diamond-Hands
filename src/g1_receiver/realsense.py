"""Discover an attached RealSense or record a fixed, surveyed RGB-D capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np

from g1_receiver.perception import CameraIntrinsics, RigidPose


def _load_sdk():
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise RuntimeError(
            "pyrealsense2 is unavailable. Install the RealSense SDK Python bindings on "
            "the computer connected to the camera."
        ) from error
    return rs


def _device_info(device, rs) -> dict[str, str]:
    keys = {
        "name": rs.camera_info.name,
        "serial": rs.camera_info.serial_number,
        "firmware": rs.camera_info.firmware_version,
        "usb": rs.camera_info.usb_type_descriptor,
    }
    return {name: device.get_info(key) for name, key in keys.items() if device.supports(key)}


def _probe(context, rs) -> list[dict]:
    devices = []
    for device in context.query_devices():
        streams = []
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() not in (rs.stream.color, rs.stream.depth):
                    continue
                video = profile.as_video_stream_profile()
                streams.append({
                    "stream": str(profile.stream_type()),
                    "format": str(profile.format()),
                    "width": video.width(),
                    "height": video.height(),
                    "fps": profile.fps(),
                })
        devices.append({**_device_info(device, rs), "streams": streams})
    return devices


def _survey(path: Path, serial: str) -> RigidPose:
    data = json.loads(path.read_text())
    if data["serial"] != serial:
        raise ValueError("survey serial does not match the requested camera")
    return RigidPose(
        0.0, "camera_optical", "field", data["rotation"], data["translation_m"],
        "fixed_calibration", data["position_std_m"], data["angle_std_rad"],
    )


def _rectified_pixel_map(rs, intrinsics, projection: CameraIntrinsics):
    """Look up each rectified output pixel in the original color image."""
    if intrinsics.model == rs.distortion.none or not any(intrinsics.coeffs):
        return None
    width, height = intrinsics.width, intrinsics.height
    normalized_x = (np.arange(width) - projection.cx) / projection.fx
    normalized_y = (np.arange(height) - projection.cy) / projection.fy
    pixels = np.asarray([
        rs.rs2_project_point_to_pixel(intrinsics, [x, y, 1.0])
        for y in normalized_y for x in normalized_x
    ]).reshape(height, width, 2)
    if not np.isfinite(pixels).all():
        raise ValueError("camera distortion projection returned nonfinite pixels")
    cols, rows = np.moveaxis(np.rint(pixels).astype(int), -1, 0)
    valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    return np.clip(rows, 0, height - 1), np.clip(cols, 0, width - 1), valid


def _color_axial_factors(rs, extrinsics, intrinsics, width, height):
    # SDK alignment resamples the Z16 value, which is measured along the depth
    # imager's axis. Convert it into axial meters along the rectified color ray.
    origin = rs.rs2_transform_point_to_point(extrinsics, [0.0, 0.0, 0.0])
    depth_axis = [
        rs.rs2_transform_point_to_point(extrinsics, axis.tolist())[2] - origin[2]
        for axis in np.eye(3)
    ]
    rows, cols = np.ogrid[:height, :width]
    ray_factor = (
        depth_axis[0] * (cols - intrinsics.cx) / intrinsics.fx
        + depth_axis[1] * (rows - intrinsics.cy) / intrinsics.fy
        + depth_axis[2]
    )
    if not np.isfinite(ray_factor).all() or np.any(ray_factor <= 0) or not np.isfinite(origin[2]):
        raise ValueError("color-to-depth calibration produces invalid axial factors")
    return ray_factor.astype(np.float32), float(origin[2])


def _profile_metadata(profile) -> dict:
    calibration = profile.get_intrinsics()
    return {
        "width": calibration.width,
        "height": calibration.height,
        "fps": profile.fps(),
        "format": str(profile.format()),
        "intrinsics": [calibration.fx, calibration.fy, calibration.ppx, calibration.ppy],
        "distortion_model": str(calibration.model),
        "distortion_coefficients": list(calibration.coeffs),
    }


def _capture(context, rs, args: argparse.Namespace) -> dict:
    survey = _survey(args.survey, args.serial)
    serials = [_device_info(device, rs).get("serial") for device in context.query_devices()]
    if args.serial not in serials:
        raise RuntimeError(f"No connected RealSense matches serial {args.serial!r}. Run probe.")
    if min(*args.color_size, *args.depth_size, args.fps, args.frames) <= 0:
        raise ValueError("stream dimensions, frame rate, and frame count must be positive")
    if not np.isfinite(args.max_skew_ms) or args.max_skew_ms < 0:
        raise ValueError("maximum color-depth timestamp skew must be finite and nonnegative")

    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, *args.color_size, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, *args.depth_size, rs.format.z16, args.fps)
    pipeline = rs.pipeline(context)
    profile = pipeline.start(config)
    try:
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        camera_intrinsics = color_profile.get_intrinsics()
        intrinsics = CameraIntrinsics(
            camera_intrinsics.fx, camera_intrinsics.fy, camera_intrinsics.ppx, camera_intrinsics.ppy
        )
        pixel_map = _rectified_pixel_map(rs, camera_intrinsics, intrinsics)
        extrinsics = color_profile.get_extrinsics_to(depth_profile)
        ray_factor, depth_offset = _color_axial_factors(
            rs, extrinsics, intrinsics,
            camera_intrinsics.width, camera_intrinsics.height,
        )
        align = rs.align(rs.stream.color)
        capture = _collect(
            pipeline, align, pixel_map, ray_factor, depth_offset, intrinsics, survey,
            _device_info(profile.get_device(), rs), args.frames, args.max_skew_ms,
        )
        capture["camera_stream_calibration_json"] = json.dumps({
            "color": _profile_metadata(color_profile),
            "depth": _profile_metadata(depth_profile),
            "color_to_depth_rotation_column_major": list(extrinsics.rotation),
            "color_to_depth_translation_m": list(extrinsics.translation),
        })
        return capture
    finally:
        pipeline.stop()


def _collect(
    pipeline, align, pixel_map, ray_factor, depth_offset, intrinsics, survey,
    device, frame_count, max_skew_ms,
) -> dict:
    color_times = []
    depth_times = []
    receive_times = []
    rgb_frames = []
    depth_frames = []
    clock = None
    for _ in range(max(30, frame_count * 3)):
        frames = pipeline.wait_for_frames()
        host_receive_s = monotonic()
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            continue
        color_clock = color.get_frame_timestamp_domain()
        if color_clock != depth.get_frame_timestamp_domain():
            continue
        if clock is not None and color_clock != clock:
            raise RuntimeError("RealSense timestamp domain changed during capture")
        color_ms, depth_ms = color.get_timestamp(), depth.get_timestamp()
        if not np.isfinite([color_ms, depth_ms]).all() or abs(color_ms - depth_ms) > max_skew_ms:
            continue
        if color_times and (color_ms < color_times[-1] or depth_ms < depth_times[-1]):
            raise RuntimeError("RealSense timestamps moved backwards")
        if color_times and (color_ms == color_times[-1] or depth_ms == depth_times[-1]):
            continue

        aligned_depth = align.process(frames).get_depth_frame()
        if not aligned_depth:
            continue
        rgb = np.asanyarray(color.get_data())
        raw_depth = np.asanyarray(aligned_depth.get_data())
        if rgb.shape[:2] != raw_depth.shape or raw_depth.shape != ray_factor.shape:
            raise RuntimeError("RealSense RGB and aligned depth dimensions differ")
        if pixel_map is not None:
            rows, cols, valid = pixel_map
            rgb = np.where(valid[..., None], rgb[rows, cols], 0)
            raw_depth = np.where(valid, raw_depth[rows, cols], 0)
        depth_units = float(depth.get_units())
        if not np.isfinite(depth_units) or depth_units <= 0:
            raise RuntimeError("RealSense reported invalid depth units")
        axial_depth = (raw_depth.astype(np.float32) * depth_units - depth_offset) / ray_factor
        valid_depth = (raw_depth > 0) & (axial_depth > 0)
        rgb_frames.append(rgb.copy())
        depth_frames.append(np.where(valid_depth, axial_depth, 0.0))
        color_times.append(color_ms)
        depth_times.append(depth_ms)
        receive_times.append(host_receive_s)
        clock = color_clock
        if len(color_times) == frame_count:
            break
    if len(color_times) != frame_count:
        raise RuntimeError(
            f"Only {len(color_times)}/{frame_count} usable RGB-D pairs. "
            "Check stream availability and the configured color-depth timestamp skew."
        )

    # Use depth capture as the 3D measurement timestamp. Retain both timestamps.
    depth_timestamps = np.asarray(depth_times) * 0.001
    return {
        "timestamps_s": depth_timestamps,
        "camera_pose_timestamps_s": depth_timestamps,
        "color_timestamps_s": np.asarray(color_times) * 0.001,
        # Host receive and camera time are different domains until independently mapped.
        "host_receive_monotonic_s": np.asarray(receive_times),
        "timestamp_domain": str(clock),
        "pose_provenance": survey.provenance,
        "camera_device_json": json.dumps(device),
        "intrinsics": [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy],
        "rgb": np.stack(rgb_frames),
        "depth_m": np.stack(depth_frames),
        "camera_rotation": np.repeat(survey.rotation[None, ...], frame_count, axis=0),
        "camera_translation_m": np.repeat(survey.translation[None, ...], frame_count, axis=0),
        "camera_position_std_m": np.full(frame_count, survey.position_std_m),
        "camera_angle_std_rad": np.full(frame_count, survey.angle_std_rad),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("probe", help="list connected devices and advertised RGB/depth profiles")
    capture = modes.add_parser("capture", help="record RGB-D NPZ with fixed camera calibration")
    capture.add_argument("output", type=Path)
    capture.add_argument("--serial", required=True)
    capture.add_argument("--survey", type=Path, required=True)
    capture.add_argument("--color-size", type=int, nargs=2, required=True, metavar=("WIDTH", "HEIGHT"))
    capture.add_argument("--depth-size", type=int, nargs=2, required=True, metavar=("WIDTH", "HEIGHT"))
    capture.add_argument("--fps", type=int, required=True)
    capture.add_argument("--frames", type=int, required=True)
    capture.add_argument("--max-skew-ms", type=float, default=10.0)
    args = parser.parse_args()

    try:
        rs = _load_sdk()
        context = rs.context()
        if args.mode == "probe":
            devices = _probe(context, rs)
            if not devices:
                raise RuntimeError("No RealSense camera detected by the SDK.")
            print(json.dumps(devices, indent=2))
        else:
            if args.output.exists():
                raise FileExistsError(f"Capture file already exists: {args.output}")
            data = _capture(context, rs, args)
            with args.output.open("xb") as output:
                np.savez_compressed(output, **data)
            print(f"Saved {len(data['timestamps_s'])} RGB-D pairs to {args.output}")
    except (ImportError, OSError, RuntimeError, TypeError, KeyError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
