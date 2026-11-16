# Passive onboard RGB-D observability study

The hypothetical onboard view detects the ball throughout the long hand-control interval, but **does not establish legitimate hand retention**. The same view detects it during later excluded mount support. No controller, reward or quality gate uses this study; no result is promoted.

## Mount and sensor boundary

This is a proposed massless rigid camera on the existing `torso_link`, at local position(0.12,0,0.55) m, optical forward proportional to(1,0,−1),70° vertical FOV,320×180 pixels. It represents an upper torso/head-housing bracket. It is **not a verified built-in G1 camera, tested hardware installation, neck joint or active gaze**. The first attempt incorrectly assumed `head_link` was a body; it is a mesh attached to `torso_link`. That attempt failed before simulation; its source, receipt and logs remain in `onboard-observability-linux`. Revision2 corrects the mount; revision3 only adds body-encoder logs and10 Hz raw RGB-D snapshots.

MuJoCo renders the actual attached camera's RGB-D image. The observer never reads the moving camera's true world position or rotation. Its camera transform is reconstructed in a private static FK instance from the rigid mount calibration and the29 measured body joint encoders, with the floating base set to identity. Both observed ball points and known palm/mount geometry are expressed in `receiver_pelvis`. Native finger joint state is not an input. No world field position, marker pose, fixture release time, exact ball state, force or contact enters this sensor estimate. Existing external approach cameras and every controller command remain unchanged.

This is a noisy-shape/occlusion diagnostic, not a calibrated observation model for hardware. The original detector limits are unchanged. The default detector outputs a **visible surface point**, not the ball center. Its≈16 mm covariance accounts for the existing depth/segmentation floor, but initially assumes ideal static mount/FK. It does not by itself include actual hardware calibration, encoder, shape, lighting or depth-system errors.

## Coverage and exact physical preservation

Revision2 captures310 frames from4.82 s onward;290 detections pass the unchanged detector. Coverage is:

- Incoming/precontact4.8–5.198 s:0/19. This view **cannot replace the external anticipation camera**.
- Bobble5.198–6.317 s:55/56.
- Secure-control clock6.317–10.402 s:205/205, maximum detector positionσ16.14 mm.
- After first excluded mount support10.402 s:30/30.

Revision2 matches **every** recorded array in actual qpos/qvel/control/arm targets, ball/self contact witnesses, trace, view, anatomical record and interhand audit against the frozen11-second confirmation. Revision3 also matches all these arrays exactly. Proofs are `onboard-observability-r2-summary.json` and `onboard-r3-physical-exact.json`. This preserves the underlying result: delayed secure control,4.085 s continuous hand-only support through10.401 s, then left mount support at10.402 s and failed11-second possession. Passive camera coverage does not alter that failure.

## Known-shape mount-clearance diagnostic and abstention

`audit_go_onboard_mount_observability_r2.py` uses only saved RGB-D, measured29 body encoders and the static mount/model. It fits a known-shape ellipsoid with radii(.142,.085,.085) m to the visible orange depth surface, estimates center/major-axis orientation, and computes distance to rigid hand-adapter mount geometries in a private FK instance. This center is a **model-based estimate from a partial view**, not ball truth. No unobserved finger joints are used. The known-shape assumption and partial-view fitting bias limit generality.

The diagnostic explicitly adds illustrative uncertainty assumptions: mount positionσ5 mm, mount angleσ0.5°, encoderσ0.002 rad propagated through relative camera/mount Jacobians, and radiusσ5 mm, in addition to the unchanged detector covariance. These values are **declared assumptions, not measured hardware calibration**; the combinedσ proxy is≈17.7 mm. Partial-view bias and model errors may require larger bounds. No threshold is selected from physical labels.

Only after all sensor estimates are complete does the script join independent passive contact-force labels within±2 ms of each image. Exact contacts are used **only for evaluation**, never for fitting, controller inputs, reward, gate tuning or classifier training.

The original diagnostic incorrectly treated solver convergence as fit validity. Although robust soft-L1 fitting can ignore outliers,0.5–2.2% of orange silhouette pixels have distant background depth; these make raw all-point RMS65–160 mm in41 snapshots. The original report/source remain preserved (`onboard-mount-observability.json`, `onboard-mount-observability-r1-source.py`); their clearance summary is superseded.

Revision2 applies the **same existing detector depth-inlier band**, `abs(depth−median)<=max(.05,.035*median)`, before shape fitting. It separately reports solver convergence, raw residuals and exploratory sensor-fit validity (filtered RMS≤3×unchanged detectorσ, at least60% component depth inliers). This rule was declared before the corrected rerun and uses no physics labels. Seventeen snapshots have low raw RMS(.189–3.101 mm),41 have high raw RMS(65.5–160.4 mm); all58 filtered fits have RMS.177–1.018 mm and pass this exploratory bound. At least99.4% of fitted surface normals face the camera. This is not a calibrated confidence certificate for partial-view center fitting.

The corrected41 secure-clock fits estimate mount clearance1.04–4.28 mm (median3.06 mm). Four frames with force in the same physical sample estimate.676–.939 mm (median.888 mm). A separate±2 ms label-sensitivity group has six frames and must not be confused with contemporaneous support. Despite small point-estimate differences, the≈17.7 mm uncertainty proxy dominates both groups; **all illustrative3σ intervals include contact, and zero frames certify mount-free retention**. This evidence supports abstention for this sensor/geometry model, not a new retention reward or an impossibility claim for all RGB-D methods. The ball remains visible while support is physically invalid.

## Reproduction

`onboard-observability-r3-receipt.json` records exact remote argv, immutable base source, selected profile and source-overlay SHA256. `onboard-source-receipt.json` retrieves the source-only snapshot and manifest; source archive SHA256 is `f2e3cc004580adcb48250243552ea844980e06f10075e5e2bfdb9a16faa45599`. It contains source only, no environments, caches or credentials. The selected controller's separate earlier freeze remains unchanged (SHA256 `a51b42cb372b92092425173af576b894f9753ae72c9bcac8b6e43003a15522db`).

With the frozen source, matching environment and Aero/AMP assets from the receipt:

```sh
python scripts/probe_go_side_onboard_observability_r3.py --frozen-root SOURCE --profile profile.json --compact compact.json --cup cup.json --source AERO --amp-source AMP --output NEW_OUTPUT --timeconstant .0025 --cue-lead .65 --reach-duration .45 --duration 11
```

The study uses the explicitly separate, uncalibrated2.5 ms ball–hand numerical contact assumption. It does not validate that material law, hardware impact force or the original10-yard running catch.

From this repository root, rerun the passive sensor diagnostic:

```sh
.venv/bin/python scripts/audit_go_onboard_mount_observability_r2.py
```

The script reads `onboard-observability-r3-linux/artifacts/study/confirm-11s/{onboard-observations.json,onboard-keyframes.npz,scene.xml}` and writes `onboard-mount-observability-r2.json`. Raw RGB-D snapshots, all physical records and source receipts remain adjacent. `experiment-source-hashes.json` records local experiment-source/report identities. No shared controller, perception, form or RL module is changed by the camera study.
