from copy import deepcopy
from hashlib import sha256
import json
import mujoco
import pytest
from g1_receiver.anatomical_catch_eval import physics_fingerprint
from g1_receiver.collision_fingerprint import collision_fingerprint,ccd_options,CCD_FINGERPRINT_SCHEMA


def model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="1 1 .1"/>
      <body name="object" pos="0 0 1"><freejoint/><geom name="ball" type="sphere" size=".1" mass="1"/></body>
      </worldbody></mujoco>''')


def test_default_is_stable_matches_explicit_payload_and_preserves_legacy():
    m=model();before=physics_fingerprint(m);options=ccd_options(m)
    a=collision_fingerprint(m);b=collision_fingerprint(m)
    assert a==b
    expected=deepcopy(before['parameters']);expected['schema']=CCD_FINGERPRINT_SCHEMA;expected['ccd_options']=options
    assert a['sha256']==sha256(json.dumps(expected,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert a['legacy_v1_sha256']==before['sha256']
    assert a['sha256']!=before['sha256']
    assert physics_fingerprint(m)==before
    assert ccd_options(m)==options
    # A consumer editing returned metadata cannot mutate a model or future result.
    a['parameters']['ccd_options']['ccd_iterations']+=100
    assert collision_fingerprint(m)==b


@pytest.mark.parametrize('field,value',[
    ('disableflags',int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)),
    ('disableflags',int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)),
    ('enableflags',int(mujoco.mjtEnableBit.mjENBL_ENERGY)),
    ('ccd_iterations',75),
    ('ccd_tolerance',1e-5),
])
def test_ccd_change_cannot_falsely_match_default_but_legacy_is_unchanged(field,value):
    m=model();legacy=physics_fingerprint(m);default=collision_fingerprint(m)
    setattr(m.opt,field,value)
    changed=collision_fingerprint(m)
    assert changed['sha256']!=default['sha256']
    assert changed['parameters']['ccd_options'][field]==value
    assert physics_fingerprint(m)==legacy
    assert changed['legacy_v1_sha256']==legacy['sha256']


def test_extension_still_detects_material_changes():
    m=model();a=collision_fingerprint(m)
    m.geom_friction[1,0]*=1.1
    b=collision_fingerprint(m)
    assert a['sha256']!=b['sha256']
    assert a['legacy_v1_sha256']!=b['legacy_v1_sha256']
    assert a['parameters']['ccd_options']==b['parameters']['ccd_options']
