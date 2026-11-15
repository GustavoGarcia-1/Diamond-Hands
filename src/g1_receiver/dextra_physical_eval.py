"""Physics-rate, evaluation-only contact measurements for paired Dextra hands."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

GROUPS = (
    "left_tip", "left_palm", "left_other_hand",
    "right_tip", "right_palm", "right_other_hand",
    "torso", "floor", "other",
)
TRACE_COLUMNS = (
    "timestamp_s", "ball_x_m", "ball_y_m", "ball_z_m", "ball_floor_contact", "applied_ball_load",
    *(f"{group}_normal_n" for group in GROUPS),
    *(f"{group}_vertical_n" for group in GROUPS),
)
TORSO_BODIES = ("pelvis", "waist_yaw_link", "waist_roll_link", "torso_link")


@dataclass(frozen=True)
class DextraRetentionCriteria:
    window_s: float = 2.0
    min_normal_per_hand_n: float = 0.10
    min_bilateral_fraction: float = 0.85
    max_bilateral_gap_s: float = 0.10
    max_nonhand_normal_n: float = 0.10
    min_mean_hand_weight_fraction: float = 0.80
    min_initial_free_flight_s: float = 0.02

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("retention criteria must be positive and finite")
        if not 0 < self.min_bilateral_fraction <= 1 or self.max_bilateral_gap_s >= self.window_s:
            raise ValueError("invalid bilateral retention window")


def _descends_from(model, body_id: int, ancestor: int) -> bool:
    while body_id:
        if body_id == ancestor:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def _max_gap(mask: np.ndarray, timestep: float) -> float:
    edges = np.flatnonzero(np.diff(np.r_[False, ~mask, False]))
    return float(np.max(edges[1::2] - edges[::2], initial=0) * timestep)


class DextraContactAudit:
    """Call after each `mj_step`. Keep this object outside the controller."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        palms: tuple[str, str] = ("dextra_left_palm", "dextra_right_palm"),
        fingertip_geoms: tuple[tuple[str, ...], tuple[str, ...]] | None = None,
        torso_bodies: tuple[str, ...] = TORSO_BODIES,
        criteria: DextraRetentionCriteria = DextraRetentionCriteria(),
    ) -> None:
        self.model = model
        self.criteria = criteria
        self.timestep = float(model.opt.timestep)
        self.ball_geom = model.geom("football_surface").id
        self.ball_body = int(model.geom_bodyid[self.ball_geom])
        self.floor_geom = model.geom("floor").id
        ball_joint = model.joint("football_free")
        if int(ball_joint.type[0]) != int(mujoco.mjtJoint.mjJNT_FREE) or ball_joint.bodyid[0] != self.ball_body:
            raise ValueError("football must have its own free joint")
        for index, kind in enumerate(model.eq_type):
            if int(kind) in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)):
                objects = (model.eq_obj1id[index], model.eq_obj2id[index])
                object_type = model.eq_objtype[index]
                bodies = (
                    tuple(model.site_bodyid[site] for site in objects if site >= 0)
                    if int(object_type) == int(mujoco.mjtObj.mjOBJ_SITE) else objects
                )
                if self.ball_body in bodies:
                    raise ValueError("football cannot be attached by an equality constraint")
        ball_dof = int(ball_joint.dofadr[0])
        self.ball_dofs = slice(ball_dof, ball_dof + 6)
        self.ball_weight_n = float(model.body_mass[self.ball_body] * np.linalg.norm(model.opt.gravity))
        if self.ball_weight_n <= 0:
            raise ValueError("retention audit requires a football with weight")
        palm_ids = [model.body(name).id for name in palms]
        if palm_ids[0] == palm_ids[1] or any(
            _descends_from(model, body_id, other) for body_id, other in (palm_ids, palm_ids[::-1])
        ):
            raise ValueError("left and right palms must be separate bodies")
        torso_ids = {model.body(i).id for i in range(model.nbody) if model.body(i).name in torso_bodies}
        self.geom_group = np.full(model.ngeom, GROUPS.index("other"), dtype=np.intp)
        self.geom_group[model.geom_bodyid == self.ball_body] = -1
        self.geom_group[self.floor_geom] = GROUPS.index("floor")
        self.geom_group[np.isin(model.geom_bodyid, list(torso_ids))] = GROUPS.index("torso")
        if fingertip_geoms is not None and len(fingertip_geoms) != len(palm_ids):
            raise ValueError("expected one fingertip geom list per palm")
        declared_tips = (
            [{model.geom(name).id for name in names} for names in fingertip_geoms]
            if fingertip_geoms is not None else None
        )
        for index, (side, palm_id) in enumerate(zip(("left", "right"), palm_ids, strict=True)):
            if declared_tips is not None and any(
                model.geom_bodyid[geom] == palm_id
                or not _descends_from(model, int(model.geom_bodyid[geom]), palm_id)
                for geom in declared_tips[index]
            ):
                raise ValueError(f"{side} fingertip geoms must belong to the corresponding hand fingers")
            for geom_id, body_id in enumerate(model.geom_bodyid):
                body_name = model.body(int(body_id)).name
                if body_id == palm_id:
                    suffix = "palm"
                elif _descends_from(model, int(body_id), palm_id):
                    is_tip = (geom_id in declared_tips[index] if declared_tips is not None
                              else body_name.endswith("_finger_tip"))
                    suffix = "tip" if is_tip else "other_hand"
                else:
                    continue
                self.geom_group[geom_id] = GROUPS.index(f"{side}_{suffix}")
            collidable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
            if any(
                not np.any(collidable & (self.geom_group == GROUPS.index(f"{side}_{kind}")))
                for kind in ("tip", "palm")
            ):
                raise ValueError(f"{side} hand requires identifiable palm and fingertip collision geoms")
        self._wrench = np.zeros(6)
        self._rows: list[np.ndarray] = []

    def record(self, data: mujoco.MjData) -> None:
        normal = np.zeros(len(GROUPS))
        vertical = np.zeros(len(GROUPS))
        floor_contact = False
        for index, contact in enumerate(data.contact):
            first, second = contact.geom
            if first == self.ball_geom:
                other, sign = second, -1.
            elif second == self.ball_geom:
                other, sign = first, 1.
            else:
                continue
            group = self.geom_group[other]
            if group < 0:
                continue
            mujoco.mj_contactForce(self.model, data, index, self._wrench)
            force = float(self._wrench[0])
            floor_contact |= other == self.floor_geom and (contact.dist <= 0 or force > 0)
            if force > 0:
                normal[group] += force
                vertical[group] += sign * float(contact.frame.reshape(3, 3)[:, 2] @ self._wrench[:3])
        applied_ball_load = bool(np.any(data.xfrc_applied[self.ball_body]) or np.any(data.qfrc_applied[self.ball_dofs]))
        self._rows.append(np.r_[data.time, data.geom_xpos[self.ball_geom], floor_contact,
                                applied_ball_load, normal, vertical])

    def trace(self) -> np.ndarray:
        return np.asarray(self._rows, dtype=float).reshape(-1, len(TRACE_COLUMNS))

    def summary(self) -> dict:
        rows = self.trace()
        if not len(rows) or not np.isfinite(rows).all():
            raise ValueError("contact audit requires finite physics samples")
        criteria = self.criteria
        samples = round(criteria.window_s / self.timestep)
        initial_samples = max(1, round(criteria.min_initial_free_flight_s / self.timestep))
        if samples < 1:
            raise ValueError("retention window is shorter than one physics step")
        full_window = len(rows) >= samples
        clock_ok = bool(np.allclose(np.diff(rows[:, 0]), self.timestep, rtol=0., atol=1e-7))
        floor_hit = bool(np.any(rows[:, 4]))
        applied_load = bool(np.any(rows[:, 5]))
        normals = rows[:, 6:6 + len(GROUPS)]
        window_normals = normals[-samples:]
        window_vertical = rows[-samples:, 6 + len(GROUPS):]
        palm_tip_normal = np.column_stack((window_normals[:, :2].sum(axis=1), window_normals[:, 3:5].sum(axis=1)))
        bilateral = (palm_tip_normal >= criteria.min_normal_per_hand_n).all(axis=1)
        bilateral_fraction = float(bilateral.mean())
        max_gap_s = _max_gap(bilateral, self.timestep)
        initial_free_flight = bool(
            len(rows) >= initial_samples and not np.any(rows[:initial_samples, 4:6])
            and np.max(normals[:initial_samples].sum(axis=1)) < criteria.min_normal_per_hand_n
        )
        nonhand = normals[:, GROUPS.index("torso"):].sum(axis=1)
        retention_nonhand = nonhand[-samples:]
        torso = window_normals[:, GROUPS.index("torso")]
        other_hand = window_normals[:, [GROUPS.index("left_other_hand"), GROUPS.index("right_other_hand")]].sum(axis=1)
        vertical_hand_fraction = float(window_vertical[:, :GROUPS.index("torso")].sum(axis=1).mean()
                                       / self.ball_weight_n)
        bilateral_retention = (
            full_window and clock_ok and not floor_hit and not applied_load
            and bilateral_fraction >= criteria.min_bilateral_fraction
            and max_gap_s <= criteria.max_bilateral_gap_s + 1e-8
            and bool(np.any(bilateral[-max(1, round(.02 / self.timestep)):]))
        )
        load_bearing = bilateral_retention and vertical_hand_fraction >= criteria.min_mean_hand_weight_fraction
        hands_only = bool(load_bearing and retention_nonhand.max() <= criteria.max_nonhand_normal_n)
        return {
            "evidence_class": "simulation_truth_evaluation_only",
            "physics_sample_hz": 1 / self.timestep,
            "samples": len(rows),
            "physics_clock_ok": clock_ok,
            "retention_window_s": criteria.window_s,
            "full_retention_window": full_window,
            "initial_free_flight": initial_free_flight,
            "ball_floor_contact": floor_hit,
            "applied_ball_load": applied_load,
            "bilateral_palm_tip_fraction": bilateral_fraction,
            "longest_bilateral_gap_s": max_gap_s,
            "mean_vertical_hand_force_as_ball_weight": vertical_hand_fraction,
            "max_ball_normal_force_n": dict(zip(GROUPS, map(float, normals.max(axis=0)), strict=True)),
            "peak_retention_torso_normal_n": float(torso.max()),
            "peak_episode_nonhand_normal_n": float(nonhand.max()),
            "peak_retention_nonhand_normal_n": float(retention_nonhand.max()),
            "peak_retention_other_hand_normal_n": float(other_hand.max()),
            "min_retention_ball_height_m": float(rows[-samples:, 3].min()),
            "final_ball_xyz_m": rows[-1, 1:4].tolist(),
            "bilateral_palm_tip_retention": bool(bilateral_retention),
            "bilateral_load_bearing_retention": bool(load_bearing),
            "torso_assisted_bilateral_retention": bool(
                bilateral_retention and torso.max() > criteria.max_nonhand_normal_n
            ),
            "hands_only_retention": hands_only,
            "hands_only_catch": bool(initial_free_flight and hands_only and nonhand.max() <= criteria.max_nonhand_normal_n),
        }
