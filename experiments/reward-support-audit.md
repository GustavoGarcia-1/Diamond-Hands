# Fixed-phase proxy/physical disagreement — independent read-only audit

All five physical scores and frozen sensor-reward ledgers recompute exactly. The prescribed normalized actions map to phase-rate multipliers1+0.35×action. They were fixed before physical outcomes, not selected by simulator truth.

- Action−1: reward4.150725, clean hold0, first forbidden support6.177s;56 qualified visual captures, maximum visual dwell0.88s.
- Action−0.5: reward1.040904, clean hold1.360s, no forbidden support.
- Action0: reward1.175612, clean hold1.785s, no forbidden support.
- Action+0.5 and+1: both reward0.811461, clean hold1ms, first forbidden support6.319s.

**Do not say action−1 earned its qualified dwell while supported by the torso.** Its initial torso touch lasts2ms(6.177–6.178), before first anatomical hand contact6.189, and invalidates the strict clean-hold score for all later times. Its56 qualified captures6.42–7.56 contain no concurrent torso, excluded-hand or floor support.53/56 are physically bilateral at the sampled instant. The maximum0.88s sensor-dwell window6.64–7.52 is99.773% physically bilateral, but bridges2ms of lost bilateral contact; raw longest bilateral without the initial-support cutoff is0.671s. Later torso support returns8.466s onward; final possession fails. Thus clean acquisition history and strict uninterrupted physical retention disagree with a proximity-based visual proxy.

There is also a direct hidden-support counterexample: actions+0.5/+1 each have excluded mount support at20 of21 reward-qualified captures. Qualified error/form/speed does not establish exclusive anatomical hand support.

Action−1 reward decomposes exactly into proximity0.195350 +relative-speed0.896714 +dwell1.060000 +milestone2.000000 −penalties0.001339. Reward uses a fresh confidence-gated ball estimate near the **mean** pad center plus declared offset; neither this center criterion nor low relative velocity identifies each load-bearing surface. Proximity/speed terms can pay even without qualified form/dwell. No simulator-force penalty was secretly supplied. The mismatch is an objective/observability limitation, not evidence of an undisclosed privileged reward input.

## Stopped R3b checkpoint

Independent nominal evaluation also reproduces exactly: reward1.1417878715 versus zero1.1756122162, clean hold1ms versus1.785s, first excluded mount6.277s, acquisition form fraction0.54545 and no final clean hold. This checkpoint is not a demonstrated improvement. `checkpoint-audit.json` pins the evaluated model manifest and input hashes.

## Sensor boundary and source caveat

Reviewed current relative/carry controllers receive measured29 body encoders, IMU, surveyed RGB-D marker pose, RGB-D forecast, phase and known targets. Their MuJoCo data is independently created FK state populated from those measurements, not the live plant. Model geometry, nominal compact/cup targets, catch plane/axis, offset and joint limits are explicit engineering commands/calibration; “sensor-only” does not mean they are learned or uncalibrated. The legacy native preload still uses the commanded release event. Relative freeze/carry timing derives from the causal visual forecast, not future simulator state.

Live plant pelvis/qpos/contact/ctrl reads in the runners feed recorded motion, passive contact witnesses, rendering and final evaluation. Native `sensordata` reads are actual modeled joint encoders. No simulator ball pose/root truth/contact forces enter the reviewed actor/critic/reward/relative/carry path. External AMP training provenance remains separately unverified.

Current source is not identical to every frozen artifact. `source-review.json` pins current versus frozen R3b files. A later diagnostic fix correctly labels `receiving_up_target` as a **soft orientation objective**, with hard form rejection disabled; it must not be called a guaranteed ≥0.65 upward/≤0.14m height guard. `clearance()` checks proximal-arm versus torso, not all interhand/native-finger geometry. Physical self-collision/form evaluation remains necessary.

A corrected related Go onboard study, `reports/catch-repair-20261114/go-side-cup/onboard-mount-observability-r2.json` (SHA44eafc2c79b0f83b05d587622f7090d00d4305810b42c8bb14806bef795c5bd3), correctly abstains. Its secure gaps1.04–4.28mm and same-step mount gaps0.676–0.939mm are separated as point estimates, but declared~17.7mm uncertainty makes every3σ interval include contact. Earlier R1 used unfiltered silhouette depths and is superseded; do not repeat its raw-range-overlap claim. R2 is a different play and not a validated Out classifier. Do not replace missing contact observability with truth labels or invent a successful geometric separator.

## Reproduce the audit

```sh
PYTHONPATH=src .venv/bin/python reports/catch-repair-20261114/validation/constant-phase-risk/audit.py
PYTHONPATH=src .venv/bin/python reports/catch-repair-20261114/validation/constant-phase-risk/audit_checkpoint.py
```

Inputs: `late-reach-training/constant-phase-linux/artifacts/study`, frozen reward `r3b-linux/artifacts/study/runtime/src/g1_receiver/late_reach_rl.py` SHA623ca6cc2676d26ff6233e563836929468f076a67750d4e9004c8be07384f097. Constant-study source archive SHA676ad0e3e1c721cab9dc2cb29dcd5522add5de97a081c83c83a07e26935875cf; retrieved archive SHAc0110fb9d55a921a74b48550f5f4e86930238b95f0a1edf55254f377eee61cd9. Per-trial input hashes and exposure-aligned support joins are in `audit.json`. Contact labels are evaluation-only;1ms preintegration reporting ambiguity does not change the long mount-support finding. No source, threshold, website or controller edits and no new simulations were made for this audit.
