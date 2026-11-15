import mujoco
import numpy as np
import pytest

from g1_receiver.scrimmage_throw import FixedQuarterback, ScrimmagePass


def test_ball_stays_at_scrimmage_then_travels_to_downfield_target():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep=".001"/><worldbody>
      <body name="football" pos="0 0 1.4">
        <freejoint name="football_free"/>
        <geom type="ellipsoid" size=".142 .085 .085" mass=".42"/>
      </body>
    </worldbody></mujoco>
    """)
    state = mujoco.MjData(model)
    origin = (0., 0., 1.4)
    aim = (9.7, 3.47, 1.008)
    fixture = FixedQuarterback(model, state, ScrimmagePass(origin, aim, .99))
    fixture.ready()
    mujoco.mj_step(model, state, nstep=400)
    np.testing.assert_allclose(state.qpos[:3], origin, atol=1e-9)
    throw = fixture.release(state.time)
    assert model.body_gravcomp[model.body("football").id] == 0.
    assert throw.release_velocity_mps[0] > 9. and throw.intercept_velocity_mps[2] < 0.
    with pytest.raises(RuntimeError, match="only once"):
        fixture.release(state.time)
    mujoco.mj_step(model, state, nstep=990)
    np.testing.assert_allclose(state.qpos[:3], aim, atol=.005)
    assert not np.any(state.qfrc_applied)
    assert not np.any(state.xfrc_applied)


def test_out_throw_horizontal_distance_is_explicit_and_rotation_invariant():
    import pytest
    from g1_receiver.scrimmage_throw import ScrimmagePass
    origin = (0., 0., 1.4)
    with pytest.raises(ValueError):
        ScrimmagePass(origin, (0., 6., 1.), .8).plan(1.)
    throw = ScrimmagePass(origin, (0., 6., 1.), .8, horizontal_distance_only=True).plan(1.)
    np.testing.assert_allclose(throw.intercept_xyz_m, (0., 6., 1.))
    with pytest.raises(ValueError):
        ScrimmagePass(origin, (0., 4.9, 1.), .8, horizontal_distance_only=True).plan(1.)
