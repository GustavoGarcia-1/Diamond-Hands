"""Acceptance invariants for contact evidence, independent of catch outcome."""
import pytest

from g1_receiver.locomotion import make_dynamic_scene
from g1_receiver.runtime import DEFAULT_RUNTIME_ROOT

mujoco = pytest.importorskip("mujoco")


def test_contact_model_has_free_robot_and_ball_without_hands_or_attachments():
    model = mujoco.MjModel.from_xml_string(make_dynamic_scene(DEFAULT_RUNTIME_ROOT))
    assert model.neq == 0, "contact evidence cannot depend on an equality attachment"
    assert model.nmocap == 0, "contact trials must not prescribe robot/ball motion"
    assert model.nu == 29
    free_joints = {
        model.joint(i).name for i, kind in enumerate(model.jnt_type)
        if kind == mujoco.mjtJoint.mjJNT_FREE
    }
    assert free_joints == {"floating_base_joint", "football_free"}
    assert not any("_hand_" in model.body(i).name for i in range(model.nbody))
    ball = model.geom("football_surface").id
    assert model.geom_contype[ball] != 0
    assert model.geom_conaffinity[ball] != 0
    assert model.body_mass[model.body("football").id] == pytest.approx(.42)
