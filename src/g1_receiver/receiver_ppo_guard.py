"""Keep a PPO actor within a fixed reference budget on sensor replay observations."""

from __future__ import annotations

import math

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.torch_layers import FlattenExtractor
from torch.distributions import Normal, kl_divergence


class ActorReplayGuard:
    """Apply a post-update KL budget on a fixed array of training observations.

    The budget is the mean of KL(reference || current) over replay observations
    and action coordinates, before action clipping. If stock PPO exceeds the
    budget, backtrack along the actor update and leave the critic update intact.
    """

    def __init__(self, model: PPO, observations: np.ndarray, max_kl: float | None):
        policy = model.policy
        if (
            not isinstance(policy.action_dist, DiagGaussianDistribution)
            or type(policy.features_extractor) is not FlattenExtractor
            or type(policy.pi_features_extractor) is not FlattenExtractor
            or type(policy.vf_features_extractor) is not FlattenExtractor
        ):
            raise ValueError("the replay guard requires a Gaussian actor and parameter-free flattening")
        if max_kl is not None and (not math.isfinite(max_kl) or max_kl <= 0):
            raise ValueError("max_kl must be positive and finite")
        self.policy = policy
        self.observations = torch.as_tensor(observations, device=model.device)
        self.max_kl = max_kl
        self.parameters = (
            *policy.mlp_extractor.policy_net.parameters(),
            *policy.action_net.parameters(),
            policy.log_std,
        )
        with torch.no_grad():
            initial = self.distribution()
            self.reference = Normal(initial.mean.detach().clone(), initial.scale.detach().clone())

    def distribution(self) -> Normal:
        try:
            current = self.policy.get_distribution(self.observations).distribution
        except ValueError as error:
            raise RuntimeError("the PPO actor produced an invalid replay distribution") from error
        if not (torch.isfinite(current.mean).all() and torch.isfinite(current.scale).all()):
            raise RuntimeError("the PPO actor produced a nonfinite replay distribution")
        return current

    @torch.no_grad()
    def metrics(self) -> dict:
        current = self.distribution()
        delta = current.mean.clamp(-1, 1) - self.reference.mean.clamp(-1, 1)
        mean_kl = float(kl_divergence(self.reference, current).mean())
        if not math.isfinite(mean_kl):
            raise RuntimeError("the PPO actor produced nonfinite replay KL")
        return {
            "mean_kl_per_action": mean_kl,
            "clipped_action_rmse": float(delta.square().mean().sqrt()),
            "raw_action_saturation_fraction": float((current.mean.abs() > 1).float().mean()),
        }

    @torch.no_grad()
    def backtrack(self, previous: list[torch.Tensor]) -> tuple[float, dict]:
        full_update = [parameter.detach().clone() for parameter in self.parameters]
        for fraction in (.5, .25, .125, .0625, .03125, .015625, 0.):
            for parameter, before, after in zip(self.parameters, previous, full_update):
                parameter.copy_(torch.lerp(before, after, fraction) if fraction else before)
            metrics = self.metrics()
            if metrics["mean_kl_per_action"] <= self.max_kl:
                # The full PPO step did not become the new actor. Drop its Adam
                # moments so a rejected direction cannot accumulate in the actor.
                for parameter in self.parameters:
                    self.policy.optimizer.state.pop(parameter, None)
                return fraction, metrics
        raise RuntimeError("the previously accepted actor exceeds the replay KL budget")


class ReceiverGuardPPO(PPO):
    """Stock PPO with an optional post-rollout actor replay guard."""

    replay_guard: ActorReplayGuard | None = None

    def _excluded_save_params(self) -> list[str]:
        # Checkpoints remain loadable by ordinary PPO. They store the trained
        # weights, and guarded resume also needs the original reference inputs.
        return [*super()._excluded_save_params(), "replay_guard", "replay_log"]

    def train(self) -> None:
        guard = self.replay_guard
        if guard is None:
            super().train()
            return
        previous = (
            [parameter.detach().clone() for parameter in guard.parameters]
            if guard.max_kl is not None else []
        )
        super().train()
        proposed = guard.metrics()
        fraction, accepted = 1., proposed
        if guard.max_kl is not None and proposed["mean_kl_per_action"] > guard.max_kl:
            fraction, accepted = guard.backtrack(previous)
        self.replay_log.append({
            "step": self.num_timesteps,
            "actor_update_fraction": fraction,
            "proposed": proposed,
            "accepted": accepted,
        })
