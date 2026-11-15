"""Run the frozen fresh comparison on the three recovered uniform-173 checkpoints."""

from __future__ import annotations

import argparse
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tarfile

import numpy as np
from stable_baselines3 import PPO
import torch

from g1_receiver.locomotion import CONTROL_DT
from g1_receiver.receiver_env import EPISODE_S, LOBS, POLICY_HZ, ReceiverTrial
from g1_receiver.receiver_physical_eval import PhysicalAuditReceiverEnv
from g1_receiver.runtime import DEFAULT_RUNTIME_ROOT
from g1_receiver.train_receiver import _source_snapshot

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "antioch-final-uniform173"
RUN_ARCHIVE_SHA256 = "4f0dcd6c50ca9d5bf726db26c87a44b75379034526f7485faa9c88e49a75c26c"
SOURCE_ARCHIVE_SHA256 = "f046805ebde6d09b91a4f87829c98f02a1b49a8debdb22e2026f7e97fa755e7a"
CASES_NAME = "receiver-fresh-paired-cases.json"
CASES_SHA256 = "fdadace53ff7acd9e15b15d1aafe9591b46da972779fb5d74248f85c36821885"
CHECKPOINTS = {
    "bc": ("bc.zip", 0),
    "ppo_12288": ("ppo-000012288.zip", 12288),
    "final_uniform173": ("ppo-001757184.zip", 1757184),
}
# Per-axis supports from sample_trial. Include all families for each novelty check.
POSITION_JITTER_M = np.array([.035, .045, .017])
VELOCITY_JITTER_MPS = np.array([.055, .060, .060])


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_recovery(reports: Path) -> tuple[Path, dict, dict]:
    receipt_path = reports / f"{RUN_NAME}-receipt.json"
    receipt = read_json(receipt_path)
    archive_path = reports / f"{RUN_NAME}.tar.gz"
    if digest(archive_path) != RUN_ARCHIVE_SHA256 or receipt["archive_sha256"] != RUN_ARCHIVE_SHA256:
        raise ValueError("recovered archive differs from the pinned archive digest")
    recovery = reports / RUN_NAME
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        expected = receipt["files_sha256"]
        if len(members) != len(archive.getmembers()) or set(members) != set(expected):
            raise ValueError("recovered archive members differ from receipt")
        for relative, checksum in expected.items():
            with archive.extractfile(members[relative]) as stream:
                archived_hash = file_digest(stream, "sha256").hexdigest()
            if archived_hash != checksum or digest(recovery / relative) != checksum:
                raise ValueError(f"recovered file differs from archive or receipt: {relative}")

    source_path = reports / "antioch-receiver-uniform173.source.tar.gz"
    status = read_json(recovery / "status.json")
    if digest(source_path) != SOURCE_ARCHIVE_SHA256 or status["snapshot_sha256"] != SOURCE_ARCHIVE_SHA256:
        raise ValueError("source snapshot differs from the pinned run launch snapshot")
    with tarfile.open(source_path, "r:gz") as archive:
        source_receipt = json.load(archive.extractfile("snapshot-sha256.json"))
    source_files = {name: checksum for name, checksum in source_receipt.items() if name.startswith("src/")}
    for relative, checksum in source_files.items():
        if digest(ROOT / relative) != checksum:
            raise ValueError(f"local simulation source or asset differs from source archive: {relative}")

    run = recovery / "run"
    manifest = read_json(run / "manifest.json")
    if manifest["source_sha256"] != _source_snapshot(None, DEFAULT_RUNTIME_ROOT):
        raise ValueError("local source and runtime differ from recovered run manifest")
    config = manifest["config"]
    if (config["seed"], config["mode"]) != (173, "uniform"):
        raise ValueError("recovered run is not uniform seed 173")
    if (
        config["bc_input_sha256"] != expected["run/bc-dataset.npz"]
        or source_receipt["datasets/bc-seed173.npz"] != expected["run/bc-dataset.npz"]
    ):
        raise ValueError("BC dataset differs from the launch snapshot or run manifest")
    if read_json(run / "heldout-manifest.json") != config["heldout"]:
        raise ValueError("held-out case definitions differ from the run manifest")
    if read_json(run / "heldout-sha256.json")["sha256"] != expected["run/heldout-manifest.json"]:
        raise ValueError("held-out manifest digest differs from the run receipt")
    latest = read_json(run / "latest.json")
    final_name, final_steps = CHECKPOINTS["final_uniform173"]
    if (
        latest["checkpoint"] != final_name or latest["timesteps"] != final_steps
        or latest["checkpoint_sha256"] != expected[f"run/{final_name}"]
        or read_json(run / "run-summary.json")["timesteps"] != final_steps
    ):
        raise ValueError("final checkpoint differs from the recovered run's latest pointer")
    return run, config, {
        "run_archive_sha256": RUN_ARCHIVE_SHA256,
        "run_receipt_sha256": digest(receipt_path),
        "recovered_files_verified": len(expected),
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_snapshot_local_files_verified": len(source_files),
        "run_manifest_sha256": expected["run/manifest.json"],
        "bc_dataset_sha256": expected["run/bc-dataset.npz"],
        "original_heldout_sha256": expected["run/heldout-manifest.json"],
        "checkpoints": {
            label: {"filename": name, "sha256": expected[f"run/{name}"], "timesteps": steps}
            for label, (name, steps) in CHECKPOINTS.items()
        },
    }


def build_cases(plan: dict, previous: list[dict]) -> tuple[list[tuple], dict]:
    cases = []
    novelty = {}
    for name, throw in plan["throws"].items():
        position = np.array(throw["release_position_m"])
        velocity = np.array(throw["release_velocity_mps"])
        if any(
            np.all(np.abs(position - original_position) <= POSITION_JITTER_M)
            and np.all(np.abs(velocity - original_velocity) <= VELOCITY_JITTER_MPS)
            for original_position, original_velocity in LOBS.values()
        ):
            raise ValueError(f"{name} lies in the original training throw support")
        if any(
            throw["release_position_m"] == row["release_position_m"]
            and throw["release_velocity_mps"] == row["release_velocity_mps"]
            for row in previous
        ):
            raise ValueError(f"{name} repeats an original held-out release")
        novelty[name] = {
            "outside_all_declared_training_throw_supports": True,
            "matches_original_heldout_release": False,
        }
    for index, fixture in enumerate(plan["fixture_order"]):
        category, variant = fixture.split("-")
        route = None if category == "stationary" else category
        throw_name = fixture if route is None else f"moving-{variant}"
        scale = .24 if route is None else plan["route_scales"][route][variant]
        throw = plan["throws"][throw_name]
        for profile, sensor_settings in plan["profiles"].items():
            trial = ReceiverTrial(**throw, route=route, route_scale=scale, **sensor_settings, suite=profile)
            cases.append((f"{fixture}/{profile}", category, plan["sensor_seed_base"] + index, trial))
    return cases, novelty


def route_result(route, rows: np.ndarray, plan: dict, full_episode: bool) -> dict:
    if route is None:
        return {"endpoint_reached": None, "endpoint_tolerance_m": None, "ordered_waypoint_hit_s": []}
    waypoints = np.asarray(route.waypoints)
    lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    tolerances = np.minimum(
        plan["endpoint_tolerance_cap_m"], plan["waypoint_tolerance_fraction"] * lengths,
    )
    hits = []
    cursor = 0
    for waypoint, tolerance in zip(waypoints[1:], tolerances):
        distance = np.linalg.norm(rows[cursor:, 5:7] - waypoint, axis=1)
        indices = np.flatnonzero(distance <= tolerance)
        if not len(indices):
            break
        cursor += int(indices[0])
        hits.append(float(rows[cursor, 0]))
        cursor += 1
    final_error = np.linalg.norm(rows[-1, 5:7] - waypoints[-1])
    return {
        "route_length_m": route.length_m,
        "endpoint_reached": bool(full_episode and len(hits) == len(tolerances) and final_error <= tolerances[-1]),
        "endpoint_tolerance_m": float(tolerances[-1]),
        "ordered_waypoint_hit_s": hits,
    }


def evaluate(env, model, label, case, plan):
    case_id, category, seed, trial = case
    observation, _ = env.reset(seed=seed, options={"trial": trial})
    start_xy = env.plant.data.xpos[env._audit_pelvis, :2].copy()
    total = 0.
    for decisions in range(1, round(EPISODE_S * POLICY_HZ) + 1):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, sensor = env.step(action)
        if not np.isfinite(observation).all() or not np.isfinite(reward):
            raise RuntimeError(f"nonfinite rollout: {label} {case_id}")
        total += reward
        if terminated or truncated:
            break
    else:
        raise RuntimeError(f"missing episode termination: {label} {case_id}")

    rows = env.sampled_truth()
    # MuJoCo may reset its simulation clock following numeric instability.
    clock_ok = bool(
        np.isfinite(rows).all()
        and np.allclose(rows[:, 0], CONTROL_DT * np.arange(1, len(rows) + 1), rtol=0., atol=1e-7)
        and np.isclose(rows[-1, 0], decisions / POLICY_HZ, rtol=0., atol=1e-7)
    )
    physical = env.physical_summary()
    route = route_result(env.route, rows, plan, bool(truncated and clock_ok))
    support_indices = np.flatnonzero(rows[:, 1])
    first_support = rows[support_indices[0]] if len(support_indices) else None
    retained = bool(truncated and clock_ok and physical["retained_sim_support_2s"])
    return {
        "checkpoint": label,
        "case": case_id,
        "category": category,
        "profile": trial.suite,
        "seed": seed,
        "policy_decisions": decisions,
        "elapsed_sim_s": float(rows[-1, 0]),
        "episode_end": "clock_fault" if not clock_ok else ("fall" if terminated else "time_limit"),
        "sensor_result": {**sensor, "return": total},
        "physical_audit": {
            **physical,
            **route,
            "full_horizon_retained_2s": retained,
            "retention_and_endpoint": None if env.route is None else bool(retained and route["endpoint_reached"]),
            "first_support_s": None if first_support is None else float(first_support[0]),
            "pelvis_displacement_before_first_support_m": (
                None if first_support is None else float(np.linalg.norm(first_support[5:7] - start_xy))
            ),
            "pelvis_displacement_after_first_support_m": (
                None if first_support is None else float(np.linalg.norm(rows[-1, 5:7] - first_support[5:7]))
            ),
            "pelvis_displacement_final_m": float(np.linalg.norm(rows[-1, 5:7] - start_xy)),
        },
    }


def counts(rows: list[dict]) -> dict:
    physical = [row["physical_audit"] for row in rows]
    moving = [row for row in physical if row["endpoint_reached"] is not None]
    visual = [row["sensor_result"]["retained_visual_proxy"] for row in rows]
    errors = [row["final_route_endpoint_error_m"] for row in moving]
    return {
        "completed_rollouts": len(rows),
        "full_horizon": sum(row["episode_end"] == "time_limit" for row in rows),
        "falls": sum(row["episode_end"] == "fall" for row in rows),
        "clock_faults": sum(row["episode_end"] == "clock_fault" for row in rows),
        "visual_proxy_retention": sum(visual),
        "physical_support_2s": sum(row["full_horizon_retained_2s"] for row in physical),
        "visual_physical_disagreements": sum(
            proxy != row["full_horizon_retained_2s"] for proxy, row in zip(visual, physical)
        ),
        "ball_floor_contacts": sum(row["any_ball_floor_contact"] for row in physical),
        "moving_rollouts": len(moving),
        "endpoint_reached": sum(row["endpoint_reached"] for row in moving),
        "retention_and_endpoint": sum(row["retention_and_endpoint"] for row in moving),
        "median_endpoint_error_m": float(np.median(errors)) if errors else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only and args.output.exists():
        parser.error("output already exists")

    run, config, provenance = verify_recovery(args.reports)
    cases_path = args.reports / CASES_NAME
    if digest(cases_path) != CASES_SHA256:
        raise ValueError("challenge plan differs from the predeclared case manifest")
    plan = read_json(cases_path)
    cases, novelty = build_cases(plan, config["heldout"])
    print(f"Verified recovered checkpoints and {len(cases)} predeclared paired cases.", flush=True)
    if args.verify_only:
        return

    torch.set_num_threads(1)
    rows = []
    summary = {}
    for label, (name, steps) in CHECKPOINTS.items():
        model = PPO.load(str(run / name), device="cpu")
        if model.num_timesteps != steps:
            raise ValueError(f"checkpoint step count differs from its expected value: {name}")
        env = PhysicalAuditReceiverEnv(DEFAULT_RUNTIME_ROOT)
        try:
            group = []
            for case in cases:
                row = evaluate(env, model, label, case, plan)
                group.append(row)
                audit = row["physical_audit"]
                print(
                    label, row["case"], row["episode_end"],
                    audit["full_horizon_retained_2s"], audit["endpoint_reached"],
                    flush=True,
                )
        finally:
            env.close()
        rows.extend(group)
        summary[label] = {
            "all": counts(group),
            "by_category": {
                category: counts([row for row in group if row["category"] == category])
                for category in ("stationary", "go", "slant")
            },
            "by_profile": {
                profile: counts([row for row in group if row["profile"] == profile])
                for profile in plan["profiles"]
            },
        }

    report = {
        "evidence_class": "fresh_paired_rendered_simulation_cpu_policy_inference",
        "provenance": provenance,
        "case_manifest_sha256": CASES_SHA256,
        "audit_script_sha256": digest(Path(__file__)),
        "cases_per_policy": len(cases),
        "rollout_count": len(rows),
        "throw_novelty": novelty,
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "policy_device": "cpu",
            "dependencies": {
                name: version(name) for name in ("mujoco", "stable-baselines3", "gymnasium", "torch", "onnxruntime", "numpy")
            },
            "policy_hz": POLICY_HZ,
            "episode_horizon_s": EPISODE_S,
            "camera_delay_ms_per_control_tick": 1000 * CONTROL_DT,
        },
        "metric_definitions": {
            "physical_support_2s": (
                "Full episode horizon, at least 2 s contiguous final robot contact sampled at 50 Hz, "
                "no sampled floor contact, final ball height and upright constraints from PhysicalAuditReceiverEnv."
            ),
            "endpoint_reached": (
                "Full episode horizon, physical pelvis enters each waypoint region in order and ends "
                "inside the final one. Radius is the smaller of 5 cm and 10% of the incoming leg length."
            ),
            "retention_and_endpoint": "Both physical_support_2s and endpoint_reached in the same moving trial.",
        },
        "limitations": [
            "Six distinct engineered releases shared across profiles and moving routes. The correlated counts describe these cases.",
            "Out-of-training-support throws are near the training lob envelope. Slant and shortened go geometry are route extrapolations.",
            "Fixed synthetic RGB-D view, 3 mm per-frame axial noise or 40 ms frame delivery delay, zero dropout, and nominal floor friction.",
            "Contact is sampled at 50 Hz and includes torso support. This does not resolve between-sample contact or quantify footfalls.",
            "An endpoint pass measures ordered waypoint-region visits and final position. It does not require stopping, route-corridor adherence, or a post-arrival hold.",
            "Short indoor MuJoCo walks with upward lobs at 3.4 s horizon. Hardware, thrown ball diversity, and running transfer remain unverified.",
            "The checkpoints belong to one uniform run. This is not a curriculum or multi-seed comparison.",
        ],
        "summary": summary,
        "rollouts": rows,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(f"Wrote {len(rows)} rollouts to {args.output}", flush=True)


if __name__ == "__main__":
    main()
