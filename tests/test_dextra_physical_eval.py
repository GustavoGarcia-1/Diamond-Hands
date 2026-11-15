"""Physics-rate regression checks for evaluation-only paired-hand evidence."""

import pytest

mujoco = pytest.importorskip("mujoco")

from g1_receiver.dextra_physical_eval import DextraContactAudit


def model_fixture(*, tips_x=.065, torso_height=None, weld=False):
    torso = (
        f'<body name="torso_link"><geom type="box" pos="0 0 {torso_height}" size=".25 .25 .02"/></body>'
        if torso_height is not None else ""
    )
    equality = '<equality><weld body1="football"/></equality>' if weld else ""
    return mujoco.MjModel.from_xml_string(f'''
<mujoco>
  <option timestep=".002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="dextra_left_palm" pos="{-tips_x} 0 .80">
      <geom type="sphere" pos="-.15 0 0" size=".04"/>
      <body name="dextra_left_f1_finger_tip"><geom type="sphere" size=".06"/></body>
    </body>
    <body name="dextra_right_palm" pos="{tips_x} 0 .80">
      <geom type="sphere" pos=".15 0 0" size=".04"/>
      <body name="dextra_right_f1_finger_tip"><geom type="sphere" size=".06"/></body>
    </body>
    <body name="football" pos="0 0 1.05">
      <freejoint name="football_free"/>
      <geom name="football_surface" type="sphere" size=".085" mass=".42"/>
    </body>
    {torso}
  </worldbody>
  {equality}
</mujoco>
''')


def measure(model, *, force_pulse=False):
    data = mujoco.MjData(model)
    audit = DextraContactAudit(model)
    for step in range(round(2.5 / model.opt.timestep)):
        data.xfrc_applied[audit.ball_body, 2] = .05 if force_pulse and step == 1 else 0.
        mujoco.mj_step(model, data)
        audit.record(data)
    return audit.summary()


def test_tip_support_excludes_floor_and_torso():
    hand = measure(model_fixture())
    assert hand["hands_only_catch"]
    assert hand["bilateral_palm_tip_fraction"] == 1.
    assert hand["mean_vertical_hand_force_as_ball_weight"] == pytest.approx(1., abs=.01)
    assert hand["peak_episode_nonhand_normal_n"] == 0.
    torso = measure(model_fixture(tips_x=.30, torso_height=.86))
    assert not torso["hands_only_catch"]
    assert not torso["ball_floor_contact"]
    assert torso["peak_retention_torso_normal_n"] > 1.
    floor = measure(model_fixture(tips_x=.30))
    assert not floor["hands_only_catch"]
    assert floor["ball_floor_contact"]


def test_external_load_and_ball_weld_cannot_qualify():
    result = measure(model_fixture(), force_pulse=True)
    assert result["applied_ball_load"] and not result["hands_only_catch"]
    with pytest.raises(ValueError, match="attached"):
        DextraContactAudit(model_fixture(weld=True))
