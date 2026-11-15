"""Simulator-truth evaluator used only after held-out policy decisions."""

from __future__ import annotations

import numpy as np

from .receiver_env import ReceiverEnv


class PhysicalAuditReceiverEnv(ReceiverEnv):
    """The learner receives the same observations and rewards as ReceiverEnv."""

    def __init__(self, asset_root):
        super().__init__(asset_root)
        model = self.plant.model
        self._audit_ball = model.body("football").id
        self._audit_ball_geom = model.geom("football_surface").id
        self._audit_floor = model.geom("floor").id
        self._audit_pelvis = model.body("pelvis").id

    def reset(self, *, seed=None, options=None):
        self._audit_rows = []
        return super().reset(seed=seed, options=options)

    def _advance(self):
        super()._advance()
        data, model = self.plant.data, self.plant.model
        on_floor = False
        on_robot = False
        for contact in data.contact:
            first, second = int(contact.geom1), int(contact.geom2)
            if self._audit_ball_geom not in (first, second):
                continue
            other = second if first == self._audit_ball_geom else first
            on_floor |= other == self._audit_floor
            on_robot |= other >= 0 and model.geom_bodyid[other] not in (0, self._audit_ball)
        self._audit_rows.append((
            self.plant.timestamp_s, bool(on_robot), bool(on_floor),
            float(data.xpos[self._audit_ball, 2]), float(data.xmat[self._audit_pelvis, 8]),
            *data.xpos[self._audit_pelvis, :2],
        ))

    def sampled_truth(self) -> np.ndarray:
        return np.asarray(self._audit_rows, dtype=float)

    def physical_summary(self) -> dict:
        rows = self.sampled_truth()
        times = rows[:, 0]
        support = rows[:, 1].astype(bool)
        ground = rows[:, 2].astype(bool)
        end = times >= times[-1] - .30
        nonsupport = np.flatnonzero(~support)
        suffix = int(nonsupport[-1] + 1) if len(nonsupport) else 0
        duration = times[-1] - times[suffix] if suffix < len(times) else 0.
        upright = float(np.min(rows[:, 4]))
        supported_fraction = float(np.mean(support[end]))
        retained = (
            support.any() and not ground.any() and supported_fraction >= .85
            and np.min(rows[end, 3]) > .70 and upright >= .90
        )
        endpoint_error = None
        if self.route is not None:
            endpoint_error = float(np.linalg.norm(rows[-1, 5:7] - self.route.waypoints[-1]))
        return {
            "evidence_class": "heldout_simulator_truth_evaluation_only",
            "contact_sampling_hz": 50,
            "retained_sim_support": bool(retained),
            "retained_sim_support_2s": bool(retained and duration + 1e-8 >= 2.),
            "continuous_final_support_s": float(duration),
            "support_fraction_final_300ms": supported_fraction,
            "any_ball_floor_contact": bool(ground.any()),
            "final_ball_height_m": float(rows[-1, 3]),
            "min_upright_cosine": upright,
            "final_route_endpoint_error_m": endpoint_error,
        }
