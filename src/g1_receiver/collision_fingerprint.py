"""Opt-in physical fingerprint that additionally records collision-pipeline options.

Legacy anatomical v1 is deliberately unchanged. This helper accepts only a static
model, never plant state or contacts. Host gains and the MuJoCo runtime version
still need separate provenance. A new hash does not imply material calibration.
"""
from copy import deepcopy
from hashlib import sha256
import json
from .anatomical_catch_eval import physics_fingerprint as _legacy_fingerprint

CCD_FINGERPRINT_SCHEMA = 'anatomical-physics-model-v1-plus-ccd-v1'


def ccd_options(model):
    """The complete flag bitfields plus native/legacy CCD stopping parameters."""
    return {
        name: float(getattr(model.opt, name)) if name == 'ccd_tolerance'
        else int(getattr(model.opt, name))
        for name in ('disableflags', 'enableflags', 'ccd_iterations', 'ccd_tolerance')
    }


def collision_fingerprint(model):
    """Extend a fresh legacy payload; never mutate the model or legacy result.

    This canonical payload matches the isolated CCD sensitivity experiment's
    explicit extension. Callers must use the extended default configuration as
    their reference; do not compare this digest to a legacy v1 digest.
    """
    legacy = _legacy_fingerprint(model)
    parameters = deepcopy(legacy['parameters'])
    parameters['schema'] = CCD_FINGERPRINT_SCHEMA
    parameters['ccd_options'] = ccd_options(model)
    digest = sha256(json.dumps(parameters, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {
        **legacy,
        'sha256': digest,
        'parameters': parameters,
        'legacy_v1_sha256': legacy['sha256'],
        'legacy_v1_omits_ccd': True,
    }
