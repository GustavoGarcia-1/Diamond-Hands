"""Compare a bounded receiver PPO continuation with and without an actor replay guard."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import partial
from hashlib import sha256
from pathlib import Path
from time import monotonic

import gymnasium
import mujoco
import numpy as np
import stable_baselines3
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .receiver_env import FAMILIES, ReceiverEnv, sample_trial
from .receiver_ppo_guard import ActorReplayGuard, ReceiverGuardPPO
from .runtime import DEFAULT_RUNTIME_ROOT
from .train_receiver import _json, _load_demonstrations, _source_snapshot


class Deadline(BaseCallback):
    def __init__(self, deadline: float):
        super().__init__()
        self.deadline = deadline

    def _on_step(self) -> bool:
        return monotonic() < self.deadline


def development_trials(seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [replace(sample_trial(rng, family), suite="ppo_guard_development") for family in FAMILIES]


def evaluate_development(model, asset_root, trials, seed, deadline) -> list[dict]:
    from .receiver_physical_eval import PhysicalAuditReceiverEnv

    rows = []
    env = PhysicalAuditReceiverEnv(asset_root)
    try:
        for case, trial in enumerate(trials):
            if monotonic() >= deadline:
                break
            observation, _ = env.reset(seed=seed + case, options={"trial": trial})
            total = 0.
            while monotonic() < deadline:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total += reward
                if terminated or truncated:
                    physical = env.physical_summary()
                    physical["evidence_class"] = "development_simulator_truth_evaluation_only"
                    rows.append({"case": case, "sensor_return": total, "sensor": info, "physical": physical})
                    break
    finally:
        env.close()
    return rows


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run(args) -> None:
    started = monotonic()
    deadline = started + args.wall_seconds
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = _source_snapshot(args.output, args.asset_root)
    for name in ("receiver_ppo_guard.py", "receiver_ppo_smoke.py"):
        local = Path(__file__).with_name(name)
        (args.output / "source" / name).write_bytes(local.read_bytes())
        sources[f"g1_receiver/{name}"] = file_hash(local)
    observations = _load_demonstrations(args.bc_data, args.asset_root)["observation"]
    trials = development_trials(args.dev_seed)
    report = {
        "evidence_class": "development_paired_simulation_smoke",
        "config": {
            "initialize_sha256": file_hash(args.initialize),
            "bc_data_sha256": file_hash(args.bc_data),
            "train_seed": args.seed, "development_seed": args.dev_seed,
            "train_sampling": "uniform", "workers": 1,
            "requested_steps_per_arm": args.steps, "actor_kl_budget_per_action": args.max_kl,
            "wall_seconds": args.wall_seconds, "replay_observations": len(observations),
        },
        "runtime": {
            "mujoco": mujoco.__version__, "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__,
            "device": "cpu",
        },
        "source_sha256": sources,
        "development_cases": [asdict(trial) for trial in trials],
        "arms": {},
    }
    report_path = args.output / "report.json"
    _json(report_path, report)
    vector = DummyVecEnv([partial(ReceiverEnv, args.asset_root)])
    try:
        for label, max_kl in (("unanchored", None), ("anchored", args.max_kl)):
            model = ReceiverGuardPPO.load(args.initialize, env=vector, device="cpu")
            if args.steps % model.n_steps:
                raise ValueError(f"steps must be a multiple of the checkpoint rollout size ({model.n_steps})")
            model.set_random_seed(args.seed)
            initial_steps = model.num_timesteps
            model.replay_guard = ActorReplayGuard(model, observations, max_kl)
            model.replay_log = []
            if label == "unanchored":
                report["config"].update({"initial_steps": initial_steps, "rollout_steps": model.n_steps})
                report["initial_development"] = evaluate_development(
                    model, args.asset_root, trials, args.dev_seed, deadline,
                )
            if monotonic() < deadline:
                model.learn(total_timesteps=args.steps, reset_num_timesteps=False, callback=Deadline(deadline))
            checkpoint = args.output / f"{label}.zip"
            model.save(checkpoint)
            report["arms"][label] = {
                "checkpoint_sha256": file_hash(checkpoint),
                "collected_decisions": model.num_timesteps - initial_steps,
                "updated_decisions": len(model.replay_log) * model.n_steps,
                "replay_log": model.replay_log,
                "development": evaluate_development(model, args.asset_root, trials, args.dev_seed, deadline),
            }
            _json(report_path, report)
    finally:
        vector.close()
    report["elapsed_seconds"] = monotonic() - started
    report["complete"] = (
        len(report["initial_development"]) == len(trials)
        and all(
            arm["updated_decisions"] == args.steps and len(arm["development"]) == len(trials)
            for arm in report["arms"].values()
        )
    )
    _json(report_path, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", required=True, type=Path)
    parser.add_argument("--bc-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42_173)
    parser.add_argument("--dev-seed", type=int, default=58_173)
    parser.add_argument("--max-kl", type=float, default=.025)
    parser.add_argument("--wall-seconds", type=float, default=240.)
    args = parser.parse_args()
    if min(args.seed, args.dev_seed) < 0 or args.seed == args.dev_seed:
        parser.error("training and development require distinct nonnegative seeds")
    if args.steps <= 0 or not np.isfinite(args.max_kl) or args.max_kl <= 0:
        parser.error("steps and max-kl must be positive")
    if not np.isfinite(args.wall_seconds) or args.wall_seconds <= 0:
        parser.error("wall-seconds must be positive and finite")
    run(args)


if __name__ == "__main__":
    main()
