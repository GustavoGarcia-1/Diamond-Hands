"""Run a bounded PPO pilot for one post-gate cradle delay decision per lob."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

import gymnasium as gym
import numpy as np

from .catch_reward import VisualRetentionProxy
from .catching import CatchDecision, VisualCatchGate
from .learning import SensorSGS
from .locomotion import (
    CATCH_BALL_CENTER_OFFSET_M, CONTROL_DT, CRADLE_TARGET_RATE_RAD_S, cradle_joint_targets,
)
from .runtime import DEFAULT_RUNTIME_ROOT
from .training import SensorPlant, _source_digests

OBSERVATION_SCHEMA = "receiver-stationary-catch-timing-v1"
OBSERVATION_SIZE = 76
CAMERA_PERIOD_S = 1 / 30
PRE_DECISION_LIMIT_S = 0.9
TRIAL_DURATION_S = 3.4
MAX_DELAY_TICKS = 4
MAX_DELAY_S = MAX_DELAY_TICKS * CONTROL_DT
ZERO_DRIVE = np.zeros(2)
FAMILIES = {
    "slow_center": ((0.84, 0., 0.38), (-1., 0., 4.36)),
    "slow_offset": ((0.92, -0.02, 0.38), (-1.08, 0.035, 4.42)),
    "long_offset": ((1.2, 0.02, 0.39), (-1.5, -0.03, 4.48)),
}


@dataclass(frozen=True)
class Lob:
    family: str
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]


def sample_lob(rng: np.random.Generator, family: str) -> Lob:
    position, velocity = FAMILIES[family]
    # These are reset distributions, never observation or reward inputs.
    position = position + rng.uniform(-1., 1., 3) * [0.025, 0.035, 0.015]
    velocity = velocity + rng.uniform(-1., 1., 3) * [0.035, 0.045, 0.045]
    return Lob(family, tuple(map(float, position)), tuple(map(float, velocity)))


def catch_observation(
    proprioception: tuple[np.ndarray, ...], decision: CatchDecision | None, max_delay_s: float
) -> np.ndarray:
    features = np.zeros(11)
    if decision is not None:
        features = np.r_[
            1., decision.time_to_crossing_s, decision.prediction.time_std_s,
            decision.receiver_relative_crossing_m, decision.velocity_mps,
            decision.prediction.inliers / 16., max_delay_s,
        ]
    vector = np.concatenate([*proprioception, features]).astype(np.float32)
    if vector.shape != (OBSERVATION_SIZE,) or not np.isfinite(vector).all():
        raise ValueError("invalid catch observation")
    return vector


def bounded_delay_s(requested_ticks: int, max_delay_s: float) -> float:
    if (
        not isinstance(requested_ticks, (int, np.integer))
        or not 0 <= requested_ticks <= MAX_DELAY_TICKS
        or not 0 <= max_delay_s <= MAX_DELAY_S
    ):
        raise ValueError("invalid cradle delay")
    available_ticks = int(np.floor(max_delay_s / CONTROL_DT + 1e-9))
    return min(requested_ticks, available_ticks) * CONTROL_DT


class CatchTimingEnv(gym.Env):
    """A one-step contextual bandit with full simulated physics per trial."""

    metadata = {"render_modes": []}

    def __init__(self, asset_root: Path, sampler, recorder=None):
        self.plant = SensorPlant(asset_root)
        self.sampler = sampler
        self.recorder = recorder
        self.gate = VisualCatchGate(
            allow_synthetic_fixture=True, ball_center_offset_m=CATCH_BALL_CENTER_OFFSET_M,
        )
        self.proxy = VisualRetentionProxy(allow_synthetic_fixture=True)
        self.cradle = cradle_joint_targets(self.plant.wbc.nominal, self.plant.joint_names)
        self.action_space = gym.spaces.Discrete(MAX_DELAY_TICKS + 1)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )

    def _capture(self):
        if self.plant.timestamp_s + 1e-8 < self.next_capture_s:
            return None
        self.next_capture_s += CAMERA_PERIOD_S
        frame, survey = self.plant.capture()
        receiver = self.plant.localizer.estimate(frame, survey)
        return frame, survey, receiver

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.lob = options["lob"] if options and "lob" in options else self.sampler(self.np_random)
        self.plant.reset(
            0., 0., 0., release_position_m=self.lob.position_m,
            release_velocity_mps=self.lob.velocity_mps,
        )
        self.gate.reset()
        self.proxy.reset()
        self.upper = self.plant.wbc.nominal.copy()
        self.next_capture_s = 0.
        self.decision = None
        self.max_delay_s = 0.
        while self.plant.timestamp_s <= PRE_DECISION_LIMIT_S + 1e-8:
            capture = self._capture()
            if capture is not None:
                decision = self.gate.update(*capture, self.plant.timestamp_s)
                if decision.triggered:
                    self.decision = decision
                    window = self.gate.window
                    self.max_delay_s = min(MAX_DELAY_S, max(
                        0., decision.time_to_crossing_s - window.min_lead_s
                        - window.confidence_sigma * decision.prediction.time_std_s,
                    ))
                    break
            self.plant.advance(ZERO_DRIVE, ticks=1, upper_target=self.upper)
        self.observation = catch_observation(
            self.plant.proprioception(), self.decision, self.max_delay_s
        )
        self.pending_step = True
        return self.observation.copy(), {}

    def step(self, action):
        if not self.pending_step:
            raise RuntimeError("reset is required after a completed catch trial")
        if not self.action_space.contains(action):
            raise ValueError("catch timing action must be a valid delay tick")
        requested_ticks = int(action)
        delay = bounded_delay_s(requested_ticks, self.max_delay_s)
        start_s = self.plant.timestamp_s + delay if self.decision is not None else np.inf
        latest = None
        max_dwell_s = 0.
        visible_count = 0
        measured_frames = 0
        while True:
            capture = self._capture()
            now_s = self.plant.timestamp_s
            if capture is not None:
                frame, survey, receiver = capture
                ball = self.gate.detector.detect(frame, survey)
                latest = self.proxy.update(ball, receiver, now_s)
                max_dwell_s = max(max_dwell_s, latest.stable_duration_s)
                visible_count += int(latest.visible)
                measured_frames += 1
            if now_s >= TRIAL_DURATION_S - 1e-8:
                break
            if now_s + 1e-8 >= start_s:
                delta = CRADLE_TARGET_RATE_RAD_S * CONTROL_DT
                self.upper += np.clip(self.cradle - self.upper, -delta, delta)
            self.plant.advance(ZERO_DRIVE, ticks=1, upper_target=self.upper)
        self.pending_step = False

        retained = bool(
            latest is not None and 0 <= now_s - latest.timestamp_s <= self.proxy.max_age_s
            and latest.retained_proxy
        )
        final_dwell_s = (
            latest.stable_duration_s
            if latest is not None and 0 <= now_s - latest.timestamp_s <= self.proxy.max_age_s else 0.
        )
        # Dwell shaping uses only the final uninterrupted visible dwell.
        reward = float(retained) + .25 * min(final_dwell_s / self.proxy.required_duration_s, 1.)
        info = {
            "family": self.lob.family,
            "visual_gate_eligible": self.decision is not None,
            "gate_timestamp_s": self.decision.capture_timestamp_s if self.decision else None,
            "gate_lead_s": self.decision.time_to_crossing_s if self.decision else None,
            "requested_delay_ticks": requested_ticks,
            "applied_delay_s": delay,
            "max_observed_stable_dwell_s": max_dwell_s,
            "final_observed_stable_dwell_s": final_dwell_s,
            "retained_visual_proxy": retained,
            "post_decision_visible_frames": visible_count,
            "post_decision_frames": measured_frames,
            "final_proxy_status": latest.status if latest else "no_measurement",
        }
        if self.recorder is not None:
            self.recorder(info)
        terminal_observation = catch_observation(self.plant.proprioception(), None, 0.)
        return terminal_observation, reward, True, False, info

    def close(self) -> None:
        self.plant.close()


def _evaluate(env: CatchTimingEnv, predict, heldout: list[Lob]) -> list[dict]:
    rows = []
    for lob in heldout:
        observation, _ = env.reset(options={"lob": lob})
        _, reward, _, _, info = env.step(predict(observation))
        rows.append({**info, "lob": asdict(lob), "sensor_return": reward})
    return rows


def run_pilot(asset_root: Path, output: Path, *, seed: int, episodes_per_mode: int) -> dict:
    import stable_baselines3
    import torch

    start = monotonic()
    torch.set_num_threads(1)
    source_digests = _source_digests(asset_root)
    for name in ("catch_training", "catching", "catch_reward"):
        path = Path(__file__).with_name(f"{name}.py")
        source_digests[f"g1_receiver/{name}.py"] = sha256(path.read_bytes()).hexdigest()
    heldout_rng = np.random.default_rng(seed + 100_000)
    heldout = [sample_lob(heldout_rng, family) for family in FAMILIES]
    training = {}
    models = {}
    for mode in ("uniform", "sgs"):
        curriculum = SensorSGS(FAMILIES) if mode == "sgs" else None
        episodes = []

        def sampler(rng):
            probabilities = (
                curriculum.probabilities() if curriculum is not None else np.full(len(FAMILIES), 1 / len(FAMILIES))
            )
            return sample_lob(rng, str(rng.choice(tuple(FAMILIES), p=probabilities)))

        def record(info):
            episodes.append(dict(info))
            if curriculum is not None:
                curriculum.update(info["family"], info["retained_visual_proxy"])

        print(f"Training {mode} on {episodes_per_mode} simulated lobs", flush=True)
        env = CatchTimingEnv(asset_root, sampler, record)
        try:
            model = stable_baselines3.PPO(
                "MlpPolicy", env, seed=seed, device="cpu", n_steps=4, batch_size=4,
                n_epochs=4, gamma=1., policy_kwargs={"net_arch": [32, 32]}, verbose=1,
            )
            model.learn(total_timesteps=episodes_per_mode)
            checkpoint = output / f"{mode}.zip"
            model.save(str(checkpoint))
            models[mode] = model
            training[mode] = {
                "episodes": episodes,
                "environment_steps": model.num_timesteps,
                "requested_delay_tick_counts": dict(Counter(
                    row["requested_delay_ticks"] for row in episodes
                )),
                "applied_delay_tick_counts": dict(Counter(
                    round(row["applied_delay_s"] / CONTROL_DT) for row in episodes
                )),
                "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                "final_sampling_probabilities": (
                    dict(zip(FAMILIES, map(float, curriculum.probabilities())))
                    if curriculum is not None else dict.fromkeys(FAMILIES, 1 / len(FAMILIES))
                ),
            }
            (output / f"{mode}-training.json").write_text(json.dumps(training[mode], indent=2) + "\n")
        finally:
            env.close()
    print("Evaluating the immediate-cradle baseline and both learned policies on matched lobs", flush=True)
    env = CatchTimingEnv(asset_root, lambda rng: sample_lob(rng, next(iter(FAMILIES))))
    try:
        evaluation = {"immediate_cradle": _evaluate(env, lambda _: 0, heldout)}
        for mode, model in models.items():
            evaluation[mode] = _evaluate(env, lambda obs: model.predict(obs, deterministic=True)[0], heldout)
    finally:
        env.close()
    return {
        "evidence_class": "rendered_simulation_stationary_catch_timing_bandit",
        "evaluation_basis": "continuous_visible_ball_and_receiver_proxy",
        "ball_center_radial_offset_m": CATCH_BALL_CENTER_OFFSET_M,
        "ball_observation_basis": "visible_surface_with_radial_center_offset_heuristic",
        "actor_observation_schema": OBSERVATION_SCHEMA,
        "critic_observation_schema": OBSERVATION_SCHEMA,
        "camera_period_s": CAMERA_PERIOD_S,
        "control_period_s": CONTROL_DT,
        "trial_duration_s": TRIAL_DURATION_S,
        "max_additional_delay_s": MAX_DELAY_S,
        "action_schema": "categorical_cradle_delay_control_ticks",
        "requested_delay_ticks": list(range(MAX_DELAY_TICKS + 1)),
        "elapsed_s": monotonic() - start,
        "seed": seed,
        "sb3_version": stable_baselines3.__version__,
        "torch_version": str(torch.__version__),
        "source_sha256_at_launch": source_digests,
        "training": training,
        "paired_heldout_visual_proxy_evaluation": evaluation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-mode", type=int, default=8)
    parser.add_argument("--seed", type=int, default=173)
    args = parser.parse_args()
    if not 4 <= args.episodes_per_mode <= 32 or args.episodes_per_mode % 4 or args.seed < 0:
        parser.error("episode count must be a multiple of four from 4 to 32, and seed must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    report = run_pilot(args.asset_root, args.output, seed=args.seed, episodes_per_mode=args.episodes_per_mode)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for mode, rows in report["paired_heldout_visual_proxy_evaluation"].items():
        print(f"{mode}: {sum(r['retained_visual_proxy'] for r in rows)}/{len(rows)} retained visually")


if __name__ == "__main__":
    main()
