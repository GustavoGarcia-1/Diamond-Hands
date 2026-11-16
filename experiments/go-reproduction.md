# Go QB-side cup: causal late reach, separate numerical contact assumption

This is a developmental approximately6.47 m fixed-machine receive while walking after the AMP startup. It is **not** the original10-yard running receive, a calibrated football material model, hardware validation, or a learned-policy improvement. The original20 ms model remains an archived failed anchor. No default physics, controller torque limits or scoring thresholds were changed.

## Selected nominal result and limits

The predeclared thumb-delay screen tested .30/.40 s after the observed RGBD grip gate under both original20 ms and explicitly scoped2.5 ms ball–hand contact time constants. Fingers retain .35 s delay, all ramps .25 s. This is delayed secure control after a bobble, not a clean first-try receive: firsttouch5.198 s and the legacy short receive window is41 ms. The chosen .30 s/2.5 ms candidate has2.684 s continuous legitimate bilateral support6.317–9.000 s, final possession, no excluded hand/body/floor support or applied loads,6.0003 mm maximum ball overlap, zero interhand penetration, and lower palms-up acquisition form. Its9-second exact replay matches **every** state/control/contact/trace/view/anatomical/interhand array. The11-second extension retains continuously4.085 s through10.401 s, then first excluded hand support10.402 s and loses endpoint possession. This longer failure is preserved.

The model-reference quality flag intentionally remains false for2.5 ms. The legacy `realistic_lower_catch` field therefore remains false even when all other selected9-second checks pass. Peak finger impact forces are approximately807/822 N; a6 mm overlap does not establish physically plausible impact force or a calibrated contact law. Numerical timestep convergence and physical material calibration are unresolved.

Original20 ms .30/.40 trials both give95 ms clean bilateral support and48.549 mm overlap; neither qualifies. Under2.5 ms, .40 s thumb delay gives2.733 s but first excluded support8.393 s and loses final possession. The prior .50 delay gives57 ms and loses possession. These are mechanism/timing comparisons on separately declared fixed models, not learning gains.

## Sensor and controller boundary

The arm cue uses only an actual RGBD ball history plus surveyed camera and fresh RGBD receiver pose. It fires4.580 s with measured predicted TTC.629075 s; the compact-to-cup reach is a.45 s quintic. There is **no fixture/release/absolute-time fallback for the arm reach**. Native preload still explicitly depends on the commanded machine-release event. Body locomotion and existing RGBD handoff are preserved; torso yaw goal−.65 and native schedules are fixed commanded targets. No contacts, exact ball state, forces, rewards, privileged teacher or critic drive control. qpos/contact access in the isolated wrapper is passive evaluation recording only.

A real wide camera at(3,−4,3) looking toward(3.5,−.4,2),80° vertical FOV, covers ascent and approach. Per-camera intrinsics are attached to its actual rendered frames; the inherited plant otherwise attaches global intrinsics. The identical frame is cached and used by causal arm cue and grip tracker. Root's pinned perception roundoff-tolerance fix is an explicit source overlay, not an old-source exactness claim.

The ball is free after release; initial orientation yaw100° is fixed at release only. Machine is(0,−1.8288,1.4), flight.85 s, commanded release4.35 s. Target center at nominal5.2 s is(6.30009081,−.36746073,1.17345236); planned aimZ includes the existing semiimplicit integration correction. No ball attachment or support is introduced.

## Mechanical development and failures

Initial compact-at-reset runs fell1.54 s. Preserving startup then preparing compact after the observed route handoff fixed this. An initial static audit mistakenly excluded all shoulder-named bodies; full collision review found shoulder-roll/torso obstruction and saturated torque. That omission was corrected before any pass. The final `upperarm-x35-y20.json` includes those colliders,≥6.885 mm static reach-path clearance, and reaches dynamically without settled selfcontact. Early preparation has transient robot contacts (documented in the r3 selfcontact audit), so this is not a hardware-safe trajectory claim. No gain/torque increase or fixed offset against obstruction was used.

The actual.45 s no-ball reach peaks2.393 rad/s; final200 ms peak.735 rad/s. Acquisition hands remain near shoulder height and palms upward. `actual-fixture-fit.json` fits the measured receiving facets with declared width≥10 mm and longitudinal≥5 mm margins before selecting the throw. All rejected FK, incoming geometry, calibration and startup attempts remain adjacent.

## Reproduction and evidence

`causal-pass-receipt.json`, `thumb-timing-receipt.json`, `causal-confirm-receipt.json` contain exact remote argv, immutable source roots, profile/source hashes and bounded1200-second job settings. The selected profile is `confirmed-thumb30-profile.json` (SHA256 df21abb5a5eba8741f1c04bc94d0a02b8bb59a20f569ea7179ac1ec06b46899a). `confirmation-comparison.json` records exact replay checks and both horizons. Full1 kHz actual state/control, ball and robot-self witnesses, actual native encoders, trace/view, anatomical/contact form and extended pair/CCD fingerprints are retained for every trial.

After unpacking the frozen source and placing matching Aero/AMP assets, the selected nominal command is:

```sh
python scripts/probe_go_side_causal_confirm.py --frozen-root SOURCE --profile profile.json --compact compact.json --cup cup.json --source AERO --amp-source AMP --output NEW_OUTPUT --timeconstant .0025 --cue-lead .65 --reach-duration .45 --duration 9
```

Use `--duration 11` for the archived extended failure. Runtime and dependency environment are the pinned Antioch Python in the receipt; controller source contains no environments, caches or credentials. The frozen source SHA256 is a51b42cb372b92092425173af576b894f9753ae72c9bcac8b6e43003a15522db. Separate fixed-catchplane throwY/Z±2 cm tests are predeclared in `fixedplane-receipt.json`; only the physical throw target changes, never the commanded catchplane, camera or policy inputs.

The fixed-plane ±2 cm results are in `fixedplane-summary.json`: Y+ retains3.281 s/end1/6.334 mm; Y− .444 s/end0; Z+2.384 s/end1/6.524 mm; Z−1.559 s/end0/31.144 mm. The field catch window and all camera/controller commands remain fixed; only the physical machine aim changes. This is not full spatial robustness.

## Sensitivity and sensor follow-up

With body PD held at1 kHz, physics timesteps1/.5/.25 ms produce2.684/2.119/.6295 s continuous clean support; only1 ms retains at9 s. At5.18 s the finer models put the actual receiving facets13–16/32–35 mm farther forward. Thus this is not numerical convergence. Exact1 ms anchor and full model fingerprints are retained in `timestep-*` evidence.

Constant20 ms actual image latency preserves every physical array of the nominal result;40 ms loses endpoint possession (.266 s),60 ms also fails (.419 s). All image timestamps remain unchanged, including failed/old frames. Grip gate stays4.98 s in0/20/40; arm cue moves4.58→4.60 under40 ms. An engineering adaptive-reach duration `clip(measured TTC−.18,.35,.45)` fails to improve40/60 ms and is **not selected**. These are not broad camera robustness or hardware latency claims.

Opening all four fingers completely before entry changes first force from an822 N index-tip impact at5.198 s to a523 N legitimate right palm8 impact at5.210 s,2.604 mm overlap. But the old finger capture starts5.33 s and retention is only2 ms, followed by floor contact. Intermediate .2 preload gives.628 s/5.875 mm but no endpoint. These failed mechanical controls support testing earlier capture on an initially open inlet; no new success is inferred from palm-first contact alone.

## Subsequent mechanical and measured-calibration controls

All-open finger preload plus capture delays.15/.20 s remains a failed receive:2 ms bilateral control under2.5 ms, despite genuine palm-first contact. Actual native tendon encoder travel is only0.53 mm by impact even with a6.57 mm earlier target displacement; response latency, not missing camera detection, matters. Partial preload.2 with earlier capture.15/.25 gives13/360 ms and no endpoint. All original20 ms counterparts remain deep-overlap failures. Full results/source receipts are `open-inlet-*`, `open-capture-*`, `partial-capture-*`; no selected source/profile is overwritten.

The default route pose becomes stale near entry (last frame4.80 s), so a millimetric throw fit cannot be justified from it. A passive close front-side RGB-D camera at(7.5,−.7,1.8)→(6.4,−.2,.85),50° actual intrinsics, gives18 usable measured poses over20 attempted entry frames while preserving **every nominal physical array exactly**. Joint encoders/velocities, IMU and past RGB-D pose regression predict the receiving facets at the machine's actual quantized arrival5.21 s. The prescribed aim correction is(+.378,+4.457,+14.752) mm, but pose uncertainty≈27–30 mm exceeds those shifts. The single declared fixture experiment fails to improve possession (2.018 s/endfail), and is not selected. The commanded catch plane remains fixed. `causal-sensor-facet-proposal.json` is sensor-based; adjacent `causal-actual-facet-proposal.json` is explicitly privileged **offline diagnostic only**, never an actor/reward input.

A camera-triggered yield prototype (`go_side_visual_yield.py`) uses fresh measured arrival/velocity and receiver pose, then encoder/IMU FK, joint/rate/clearance/form constraints. It latches a bounded8 cm forward/down command over.20 s when predicted TTC is20–100 ms; no fixture-clock fallback exists. Its physical no-ball check is upright with no selfcontact after yield and original torque limits. Zero action reproduces selected nominal exactly. Full yield nominal fails;40 ms delay gets2.335 s/endheld but actual palm-up form fraction.727 fails. Half yield with stricter commanded orientation margin passes form(.818) but loses possession before9 s. Neither is selected. No criterion is loosened to elevate these controls.

The subsequent common-prefix integration study keeps all state/control history exact through5.16 s, then changes physics integration to.5/.25 ms while keeping body PD1 kHz. True-timestamp support lasts2.7695/2.91425 s to9 s, with5.521/5.459 mm overlap and no excluded support. All acquisition forms pass. The original uniform-clock gate stays false for the variable-step records; these are supplemental numerical results, not qualification or full convergence. See `common-prefix-supplemental.json` and its source/receipt.

The final passive onboard camera investigation is documented in `ONBOARD.md`. It preserves every physical array while improving held-ball visibility, but explicitly **abstains** from mount-free retention because estimated clearance and uncertainty cannot separate valid hand support from excluded mount support. No controller/reward promotion follows from camera coverage.
