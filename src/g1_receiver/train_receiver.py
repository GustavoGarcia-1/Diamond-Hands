"""Train and evaluate a multi-step RGB-D receiver policy."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile
from time import monotonic

import numpy as np

from .learning import SensorSGS
from .observable import ACTION_LIMITS, ACTION_SIZE, OBS_NAMES, OBS_SIZE, SCHEMA
from .receiver_env import (
    FAMILIES, HISTORY_LENGTH, OBSERVATION_SCHEMA, POLICY_HZ, ReceiverEnv, VisualTeacher,
    heldout_trials, sample_trial,
)
from .runtime import DEFAULT_RUNTIME_ROOT
from .training import _source_digests


def _json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _record(path: Path, value) -> None:
    with path.open("a") as file:
        file.write(json.dumps(value, allow_nan=False) + "\n")


def _source_snapshot(output: Path | None, asset_root: Path) -> dict:
    manifest = _source_digests(asset_root)
    manifest["g1_receiver/_vendor/g1_runtime/pool/contracts.py"] = sha256(
        (DEFAULT_RUNTIME_ROOT / "pool/contracts.py").read_bytes()
    ).hexdigest()
    snapshot = output / "source" if output is not None else None
    if snapshot is not None:
        snapshot.mkdir(exist_ok=True)
    module_root = Path(__file__).resolve().parent
    names = (
        "train_receiver", "receiver_env", "observable", "catching", "catch_reward", "sim",
        "training", "localization", "perception", "learning", "locomotion", "runtime", "routes",
        "receiver_physical_eval",
    )
    for name in names:
        source = (module_root / f"{name}.py").read_bytes()
        if snapshot is not None:
            (snapshot / f"{name}.py").write_bytes(source)
        manifest[f"g1_receiver/{name}.py"] = sha256(source).hexdigest()
    return manifest


def collect_demonstrations(asset_root: Path, output: Path, count: int, seed: int, deadline: float):
    env = ReceiverEnv(asset_root)
    rows = {key: [] for key in ("observation", "action", "next_observation", "reward", "done", "episode")}
    summary = []
    try:
        rng = np.random.default_rng(seed + 3_000_000)
        for index in range(count):
            if monotonic() >= deadline:
                break
            family = FAMILIES[index % len(FAMILIES)]
            trial = sample_trial(rng, family, nominal=index < len(FAMILIES))
            observation, _ = env.reset(seed=seed + index, options={"trial": trial})
            teacher = VisualTeacher(env)
            total = 0.
            while True:
                action = teacher.action()
                successor, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                values = (observation, action, successor, reward, done, index)
                for key, value in zip(rows, values):
                    rows[key].append(value)
                total += reward
                observation = successor
                if done:
                    item = {
                        **info, "episode": index, "trial": asdict(trial),
                        "return": total, "teacher_triggered": teacher.latched,
                    }
                    summary.append(item)
                    _record(output / "teacher-episodes.jsonl", item)
                    break
    finally:
        env.close()
    if rows["observation"]:
        np.savez_compressed(
            output / "bc-dataset.npz", **{key: np.asarray(value) for key, value in rows.items()},
            observable_schema=SCHEMA, history_length=HISTORY_LENGTH,
            provenance="synthetic_fixture_rendered_rgbd_and_mujoco_encoders",
            observation_names=np.asarray(OBS_NAMES),
            source_sha256_json=json.dumps(_source_snapshot(None, asset_root), sort_keys=True),
        )
    return rows, summary



def _load_demonstrations(path: Path, asset_root: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if (
            data["observable_schema"].item() != SCHEMA
            or data["history_length"].item() != HISTORY_LENGTH
            or data["provenance"].item() != "synthetic_fixture_rendered_rgbd_and_mujoco_encoders"
            or tuple(data["observation_names"]) != OBS_NAMES
        ):
            raise ValueError("BC archive provenance or observation schema differs")
        if json.loads(data["source_sha256_json"].item()) != _source_snapshot(None, asset_root):
            raise ValueError("BC archive source or runtime assets differ")
        observations, actions = data["observation"], data["action"]
    if (
        observations.ndim != 2 or observations.shape[1] != OBS_SIZE * HISTORY_LENGTH
        or not len(observations) or actions.shape != (len(observations), ACTION_SIZE)
        or not np.isfinite(observations).all() or not np.isfinite(actions).all()
        or np.any(np.abs(actions) > 1)
    ):
        raise ValueError("BC archive observation/action arrays are invalid")
    return {"observation": observations, "action": actions}


def fit_teacher(model, rows, epochs: int, seed: int) -> dict:
    import torch

    if not len(rows["observation"]):
        return {"examples": 0}
    rng = np.random.default_rng(seed + 4_000_000)
    observation = np.asarray(rows["observation"], dtype=np.float32)
    action = np.asarray(rows["action"], dtype=np.float32)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=.001)
    model.policy.set_training_mode(True)
    losses = []
    for _ in range(epochs):
        permutation = rng.permutation(len(observation))
        for start in range(0, len(permutation), 128):
            indices = permutation[start:start + 128]
            batch = torch.as_tensor(observation[indices], device=model.device)
            target = torch.as_tensor(action[indices], device=model.device)
            estimate = model.policy.get_distribution(batch).distribution.mean
            weights = torch.ones_like(target)
            weights[:, :3] = 1.5
            loss = ((estimate - target).square() * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return {"examples": len(observation), "epochs": epochs, "last_minibatch_loss": losses[-1]}


def evaluate(asset_root, output, label, model, trials, deadline, *, teacher=False):
    from .receiver_physical_eval import PhysicalAuditReceiverEnv

    env = PhysicalAuditReceiverEnv(asset_root)
    rows = []
    physical_rows = []
    try:
        for index, trial in enumerate(trials):
            if monotonic() >= deadline:
                break
            observation, _ = env.reset(seed=100_173 + index, options={"trial": trial})
            scripted = VisualTeacher(env) if teacher else None
            total = 0.
            while True:
                action = scripted.action() if scripted is not None else model.predict(observation, deterministic=True)[0]
                observation, reward, terminated, truncated, info = env.step(action)
                total += reward
                if terminated or truncated:
                    row = {**info, "case": index, "trial": asdict(trial), "label": label, "return": total}
                    rows.append(row)
                    _record(output / "heldout-sensor.jsonl", row)
                    truth_file = output / f"physical-{label}-{index:02d}.npz"
                    np.savez_compressed(
                        truth_file, samples=env.sampled_truth(),
                        columns=["timestamp_s", "robot_support", "ground_contact", "ball_z_m",
                                 "pelvis_upright_cosine", "pelvis_x_m", "pelvis_y_m"],
                    )
                    physical = {
                        **env.physical_summary(), "case": index, "trial": asdict(trial), "label": label,
                        "sampled_truth_file": truth_file.name,
                        "sampled_truth_sha256": sha256(truth_file.read_bytes()).hexdigest(),
                    }
                    physical_rows.append(physical)
                    _record(output / "heldout-physical.jsonl", physical)
                    break
    finally:
        env.close()
    suites = {}
    for suite in dict.fromkeys(trial.suite for trial in trials):
        sensor = [row for row in rows if row["trial"]["suite"] == suite]
        physical = [row for row in physical_rows if row["trial"]["suite"] == suite]
        suites[suite] = {
            "expected": sum(trial.suite == suite for trial in trials),
            "completed": len(sensor),
            "visual_retention": sum(row["retained_visual_proxy"] for row in sensor),
            "physical_support_2s": sum(row["retained_sim_support_2s"] for row in physical),
            "sensor_physical_disagreements": sum(
                a["retained_visual_proxy"] != b["retained_sim_support_2s"]
                for a, b in zip(sensor, physical)
            ),
        }
    _record(output / "heldout-status.jsonl", {
        "label": label, "suites": suites,
        "stop": "complete" if len(rows) == len(trials) else "wall_time",
    })
    return rows


def _checkpoint(model, output, sgs, index):
    name = f"ppo-{model.num_timesteps:09d}.zip"
    pending = output / f".{name}"
    model.save(str(pending))
    pending.replace(output / name)
    _json(output / "latest.json", {
        "checkpoint": name,
        "checkpoint_sha256": sha256((output / name).read_bytes()).hexdigest(),
        "timesteps": model.num_timesteps,
        "checkpoint_index": index,
        "sgs": sgs.snapshot() if sgs is not None else None,
    })


def run(args) -> None:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    torch.set_num_threads(1)
    deadline = monotonic() + args.wall_seconds
    output = args.output
    trials = heldout_trials()[:args.eval_cases]
    bc_input_sha = sha256(args.bc_data.read_bytes()).hexdigest() if args.bc_data else None
    config = {
        "schema": OBSERVATION_SCHEMA, "observable_schema": SCHEMA,
        "seed": args.seed, "mode": args.mode, "workers": args.workers,
        "rollout_steps": args.rollout_steps, "batch_size": args.batch_size,
        "policy_hz": POLICY_HZ, "history_length": HISTORY_LENGTH,
        "action_limits": ACTION_LIMITS.tolist(),
        "bc_input_sha256": bc_input_sha,
        "heldout": [asdict(trial) for trial in trials],
    }
    if args.resume:
        if not (output / "manifest.json").is_file() or not (output / "latest.json").is_file():
            raise ValueError("resume requires a saved manifest and checkpoint")
        manifest = json.loads((output / "manifest.json").read_text())
        # Compare through JSON because declared XYZ tuples are persisted as lists.
        if manifest["config"] != json.loads(json.dumps(config)):
            raise ValueError("resume configuration differs from the original run")
        current = _source_snapshot(None, args.asset_root)
        if manifest["source_sha256"] != current:
            raise ValueError("resume source or runtime assets differ from the original run")
        latest = json.loads((output / "latest.json").read_text())
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = {"config": config, "source_sha256": _source_snapshot(output, args.asset_root)}
        _json(output / "manifest.json", manifest)
        _json(output / "heldout-manifest.json", config["heldout"])
        _json(output / "heldout-sha256.json", {
            "sha256": sha256((output / "heldout-manifest.json").read_bytes()).hexdigest(),
        })
        latest = None

    sgs = SensorSGS(FAMILIES) if args.mode == "sgs" else None
    if latest is not None and sgs is not None:
        sgs.attempts[:] = latest["sgs"]["attempts"]
        sgs.successes[:] = latest["sgs"]["successes"]

    make_env = partial(ReceiverEnv, args.asset_root)
    vector = (
        DummyVecEnv([make_env]) if args.workers == 1
        else SubprocVecEnv([make_env] * args.workers, start_method="spawn")
    )

    class Progress(BaseCallback):
        def _on_step(self):
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done:
                    retained = bool(info["retained_visual_proxy"])
                    if sgs is not None:
                        sgs.update(info["family"], retained)
                    row = {
                        key: value for key, value in info.items()
                        if key not in {"terminal_observation", "TimeLimit.truncated"}
                    }
                    _record(output / "training-episodes.jsonl", {"step": self.num_timesteps, **row})
            return monotonic() < deadline

    try:
        if latest is not None:
            checkpoint = output / latest["checkpoint"]
            if sha256(checkpoint.read_bytes()).hexdigest() != latest["checkpoint_sha256"]:
                raise ValueError("checkpoint hash differs from saved manifest")
            model = PPO.load(str(checkpoint), env=vector, device=args.device)
            index = latest["checkpoint_index"]
            model.set_random_seed(args.seed + model.num_timesteps)
        else:
            model = PPO(
                "MlpPolicy", vector, device=args.device, seed=args.seed, n_steps=args.rollout_steps,
                batch_size=args.batch_size, n_epochs=5, gamma=.99, learning_rate=2.5e-4,
                policy_kwargs={"net_arch": [128, 128], "log_std_init": -2.3}, verbose=1,
            )
            index = 0
            if args.bc_data or args.bc_episodes:
                if args.bc_data:
                    rows, summary = _load_demonstrations(args.bc_data, args.asset_root), []
                    copyfile(args.bc_data, output / "bc-dataset.npz")
                else:
                    rows, summary = collect_demonstrations(
                        args.asset_root, output, args.bc_episodes, args.seed, deadline,
                    )
                bc = fit_teacher(model, rows, args.bc_epochs, args.seed)
                dataset = output / "bc-dataset.npz"
                _json(output / "bc-summary.json", {
                    **bc, "teacher_trials": summary,
                    "dataset_sha256": sha256(dataset.read_bytes()).hexdigest() if dataset.is_file() else None,
                })
                model.save(str(output / "bc.zip"))
            _checkpoint(model, output, sgs, index)
            if args.eval_cases:
                evaluate(args.asset_root, output, "scripted_visual_teacher", None, trials, deadline, teacher=True)
                if args.bc_data or args.bc_episodes:
                    evaluate(args.asset_root, output, "bc_before_ppo", model, trials, deadline)

        rollout_size = args.workers * args.rollout_steps
        while monotonic() < deadline and model.num_timesteps + rollout_size <= args.steps:
            if sgs is not None:
                vector.env_method("set_family_probabilities", sgs.probabilities())
            remaining_rollouts = (args.steps - model.num_timesteps) // rollout_size
            chunk = rollout_size * min(args.checkpoint_rollouts, remaining_rollouts)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False, callback=Progress())
            index += 1
            _checkpoint(model, output, sgs, index)
            if args.eval_cases and index % args.eval_every == 0:
                evaluate(args.asset_root, output, f"ppo_{model.num_timesteps}", model, trials, deadline)
        if args.eval_cases and monotonic() < deadline and index % args.eval_every:
            evaluate(args.asset_root, output, f"ppo_{model.num_timesteps}", model, trials, deadline)
        _json(output / "run-summary.json", {
            "evidence_class": "rendered_simulation_sensor_only_multistep_receiver",
            "timesteps": model.num_timesteps, "checkpoint_index": index,
            "stop": "wall_time" if monotonic() >= deadline else "step_budget",
            "sampling": sgs.snapshot() if sgs is not None else None,
        })
    finally:
        vector.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--mode", choices=("uniform", "sgs"), default="uniform")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--wall-seconds", type=float, default=7080.)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-rollouts", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--eval-cases", type=int, default=9)
    parser.add_argument("--bc-episodes", type=int, default=12)
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--bc-data", type=Path)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rollout_size = args.rollout_steps * args.workers
    if (
        args.seed < 0 or min(args.workers, args.rollout_steps, args.batch_size, args.steps,
                            args.checkpoint_rollouts, args.eval_every, args.bc_epochs) < 1
        or args.batch_size < 2 or rollout_size % args.batch_size
        or not np.isfinite(args.wall_seconds) or args.wall_seconds <= 0
        or not 0 <= args.eval_cases <= len(heldout_trials()) or args.bc_episodes < 0
    ):
        parser.error("invalid rollout, evaluation, wall-time, or BC bounds")
    if args.collect_only:
        if args.resume or args.bc_data or args.bc_episodes < 1:
            parser.error("--collect-only requires new teacher episodes and output")
        args.output.mkdir(parents=True, exist_ok=False)
        sources = _source_snapshot(args.output, args.asset_root)
        _json(args.output / "heldout-manifest.json", [
            asdict(trial) for trial in heldout_trials()[:args.eval_cases]
        ])
        _, summary = collect_demonstrations(
            args.asset_root, args.output, args.bc_episodes, args.seed, monotonic() + args.wall_seconds,
        )
        dataset = args.output / "bc-dataset.npz"
        _json(args.output / "bc-summary.json", {
            "teacher_trials": summary, "seed": args.seed, "observable_schema": SCHEMA,
            "dataset_sha256": sha256(dataset.read_bytes()).hexdigest() if dataset.is_file() else None,
            "heldout_sha256": sha256((args.output / "heldout-manifest.json").read_bytes()).hexdigest(),
            "source_sha256": sources,
        })
    else:
        run(args)


if __name__ == "__main__":
    main()
