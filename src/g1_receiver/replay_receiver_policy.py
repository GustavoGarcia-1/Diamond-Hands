"""Export deterministic learned-policy MuJoCo replay and separate physics evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .receiver_env import ReceiverTrial, heldout_trials
from .receiver_physical_eval import PhysicalAuditReceiverEnv
from .runtime import DEFAULT_RUNTIME_ROOT
from .train_receiver import _source_snapshot


def _trial_from_manifest(manifest: dict, case: int) -> ReceiverTrial:
    trial = manifest["config"]["heldout"][case]
    return ReceiverTrial(
        **{**trial, "release_position_m": tuple(trial["release_position_m"]),
           "release_velocity_mps": tuple(trial["release_velocity_mps"])}
    )


def export_checkpoint(
    checkpoint: Path, output: Path, case: int, asset_root: Path,
    *, manifest_path: Path | None = None, device: str = "cpu",
) -> dict:
    from stable_baselines3 import PPO
    import torch

    if case < 0:
        raise ValueError("replay case must be nonnegative")
    manifest_path = manifest_path or checkpoint.parent / "manifest.json"
    source_verified = manifest_path.is_file()
    if source_verified:
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_sha256"] != _source_snapshot(None, asset_root):
            raise ValueError("replay source or runtime differs from checkpoint run manifest")
        trial = _trial_from_manifest(manifest, case)
    else:
        trial = heldout_trials()[case]

    torch.set_num_threads(1)
    model = PPO.load(str(checkpoint), device=device)
    env = PhysicalAuditReceiverEnv(asset_root)
    overview_frames = []
    overview_times = []
    sensor_rgb = []
    sensor_depth = []
    sensor_times = []
    sensor_arrival_times = []
    sensor_pose_times = []
    sensor_rotations = []
    sensor_translations = []
    control_times = []
    actions = []
    joint_targets = []
    total = 0.

    def record_frame():
        stamp = env.plant.timestamp_s
        renderer = env.plant.renderer
        renderer.update_scene(env.plant.data, camera="overview")
        overview_frames.append(renderer.render().copy())
        overview_times.append(stamp)
        captured = env._last_camera
        if captured is not None:
            frame, pose, _ = captured
            if not sensor_times or frame.timestamp_s > sensor_times[-1]:
                sensor_rgb.append(frame.rgb.copy())
                sensor_depth.append(frame.depth_m.copy())
                sensor_times.append(frame.timestamp_s)
                sensor_arrival_times.append(stamp)
                sensor_pose_times.append(pose.timestamp_s)
                sensor_rotations.append(pose.rotation)
                sensor_translations.append(pose.translation)

    try:
        observation, _ = env.reset(seed=100_173 + case, options={"trial": trial})
        record_frame()
        while True:
            action, _ = model.predict(observation, deterministic=True)
            control_times.append(env.plant.timestamp_s)
            actions.append(action.copy())
            observation, reward, terminated, truncated, sensor_info = env.step(action)
            joint_targets.append(env.plant.last_joint_targets.copy())
            total += reward
            record_frame()
            if terminated or truncated:
                break
        if len(overview_times) < 2 or not sensor_times:
            raise RuntimeError("trial ended before there were replayable sensor and overview frames")
        intr = env.plant.intrinsics
        physical_summary = env.physical_summary()
        physical_rows = env.sampled_truth()
    finally:
        env.close()

    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "overview.npz", rgb=np.stack(overview_frames), timestamps_s=overview_times)
    np.savez_compressed(
        output / "sensor.npz",
        rgb=np.stack(sensor_rgb), depth_m=np.stack(sensor_depth), timestamps_s=sensor_times,
        camera_arrival_timestamps_s=sensor_arrival_times,
        camera_pose_timestamps_s=sensor_pose_times, camera_rotation=np.stack(sensor_rotations),
        camera_translation_m=np.stack(sensor_translations),
        intrinsics=[intr.fx, intr.fy, intr.cx, intr.cy],
        pose_provenance="synthetic_fixture", evidence_class="rendered_simulation",
    )
    np.savez_compressed(
        output / "controller.npz",
        decision_timestamps_s=control_times, normalized_actions=np.stack(actions),
        executed_joint_targets_rad=np.stack(joint_targets),
    )
    np.savez_compressed(
        output / "physical.npz",
        samples=physical_rows,
        columns=["timestamp_s", "robot_support", "ground_contact", "ball_z_m",
                 "pelvis_upright_cosine", "pelvis_x_m", "pelvis_y_m"],
    )
    report = {
        "evidence_class": "deterministic_learned_policy_rendered_simulation",
        "case": case, "trial": asdict(trial), "policy_decisions": len(actions),
        "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "source_verified_against_manifest": source_verified,
        "sensor_result": {**sensor_info, "return": total},
        "physical_evaluation_only": physical_summary,
        "recordings": {
            name: sha256((output / name).read_bytes()).hexdigest()
            for name in ("overview.npz", "sensor.npz", "controller.npz", "physical.npz")
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=int, default=0, help="zero-based index in the run evaluation manifest")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        report = export_checkpoint(
            args.checkpoint, args.output, args.case, args.asset_root,
            manifest_path=args.manifest, device=args.device,
        )
    except (OSError, IndexError, KeyError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(
        f"Exported {report['policy_decisions']} deterministic learned-policy decisions "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
