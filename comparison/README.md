# Receiver comparison study: compact reproduction evidence

This package records 50 paired throws per route and method. The shared PCG64 scenarios perturb machine lateral aim and height independently by Uniform ±2 cm, and release time by Uniform ±20 ms. Three original methods compare current and earlier scripted controllers with a privileged exact-ball-observation diagnostic. A separate, posthoc appendix changes that diagnostic's assumed covariance from 1 µm to 1 mm standard deviation; it adds no actual noise. It is an estimator-compatibility investigation, not a calibrated camera or perfect-vision upper bound. See protocols and independent CSV/JSON summaries. Do not pool route success rates: route geometry, horizon and contact assumptions differ.

Controllers are scripted with a pretrained AMP gait; these comparisons do not measure learning gains. Exact-ball observations bypass detection visibility/confidence, but preserve camera timestamps, cadence and the original tracker. Actor inputs otherwise use RGBD, IMU and encoders. Native actuator bounds are optimistic symmetric stall envelopes (65.5555556 N tendon, 0.59 Nm abduction), not a hardware motor model. Go/Out use uncalibrated 2.5 ms ball-hand contact; Slant retains 20 ms. All use 1 ms physics, native CCD tolerance 1e-8/35 iterations. These are model-specific walking/delayed-control experiments, not safe-impact, running-catch or hardware validation.

## Sources and integrity

Extract with standard Python: `python -m zipfile -e receiver-comparison-study.zip study`. Large reports use ZIP LZMA compression (supported by Python's standard library; some older graphical unzip applications may not support it). Source bytes are deduplicated by SHA256 under `objects/`. Each `snapshots/*.json` lists every original tar member, mode and content hash, plus the original archive hash. No original archive is overwritten. From the extracted ZIP, run `python restore.py go-primary restored/go` (or go-floor, slant-primary, slant-floor, out-primary, out-floor). The standard-library restorer checks every byte, rejects traversal and refuses conflicting overwrites. Go archives restore directly at the destination; Slant archives have a `source/` prefix; Out archives contain `source/` and `runtimes/`. `package-manifest.json` checks evidence files. Dense state/contact/sensor NPZ and movies are intentionally excluded; raw archives remain local and their available checksums are in `raw-archives.json`. This compact package is sufficient to inspect source, profiles, commands, scores and protocol, but is not a substitute for independent raw-physics rescore.

## Environment and external assets

The frozen source archives contain src/scripts but omit pyproject.toml and uv.lock. `environment/pyproject.toml` and `environment/uv.lock` are supplementary copies from the packaging workspace, not byte-proven executed lockfiles. Prefer the recorded runtime versions in `environment/out-comparison/environment-receipt.json` (the routes use the same remote Python environment); these include Python 3.12.3, MuJoCo 3.15.0, NumPy 2.5.4, SciPy 1.17.1 and ONNX Runtime 1.31.0. Note recorded imageio 2.37.2 differs from the supplementary project pin. `environment/` provides setup and asset-fetch documentation. Python 3.12 on Linux with MuJoCo/EGL is the recorded environment (`MUJOCO_GL=egl`). Rendering is required for the controller's RGBD. macOS uses `mjpython` for rendering. Aero hand XML/meshes and the AMP repository/checkpoint are external inputs: obtain them using the frozen `scripts/fetch_aero_hand.py` and repository setup instructions, respecting their licenses/pins. Vendored body/runtime assets present in the snapshots are preserved. This is not claimed to be an offline, dependency-free installation; external Aero/AMP downloads are still required. No service API credentials are included.

## Portable single-trial commands

The recorded `results.json` retains the exact original argv and each profile SHA. Absolute `/workspace/...` paths are provenance, not portable paths. Below, replace `PYTHON`, `AERO`, `AMP`, `PKG`, `ROOT`, and `OUT` with your interpreter, external asset directories, extracted package, restored source root and fresh output directory. Set `PYTHONPATH=ROOT/src:ROOT/scripts` for the selected snapshot. Use the scenario-specific saved profile, not a regenerated approximation. These single-trial commands deliberately bypass batch-only exact-anchor checks, whose dense reference arrays are not in this ZIP.

Go primary (ROOT is restored/go):

```
PYTHON ROOT/scripts/probe_go_comparison.py --frozen-root ROOT --source AERO --amp-source AMP --profile PKG/evidence/go-primary/profiles/trial-00-current.json --compact PKG/inputs/go-side-cup/compact-upperarm.json --cup PKG/inputs/go-side-cup/upperarm-x35-y20.json --output OUT --timeconstant .0025 --cap-native-actuators --cue-lead .65 --reach-duration .45 --duration 11
```

For Go's privileged original method use its exact_ball_diagnostic profile and `--oracle-ball`. For the floor use `probe_go_uncertainty_comparison.py --oracle-ball --oracle-covariance-m2 1e-6` with the same remaining options. The floor snapshot's recorded command in `evidence/go-floor/results.json` adds the floor option; preserve its exact argv with only paths rebased. No Go single-trial probe requires the excluded nominal-anchor NPZ.

Slant primary (ROOT is restored/slant/source):

```
PYTHON ROOT/scripts/probe_slant_comparison.py --source AERO --amp-source AMP --profile PKG/evidence/slant-primary/profiles/trial-00-current.json --output OUT --method current
```

Use its recorded earlier/oracle method/profile pair for other methods. For the floor snapshot use the exact recorded probe/argv in its results. Keep the same method, profile and frozen source together.

Out (restore out-primary, then ROOT refers to the selected `runtimes/current` or `runtimes/earlier`): rebase the saved case config's `frozen_root`, `profile`, `source` and `amp_source`, and the route manifest's runtime paths. Keep all numeric values unchanged. Set PYTHONPATH to ROOT/src:ROOT/scripts plus the restored top-level source/scripts. Then:

```
PYTHON RESTORED/source/scripts/run_out_method_comparison.py single --manifest ROUTEMANIFEST --config CASECONFIG --method current --trial-id 6 --output OUT
```

`cfg.reference` is a required parser path but is not read in single/pair25 mode; it may name a declared empty local directory. Batch anchor evaluation does require excluded dense reference arrays. Floor uses `run_out_uncertainty_comparison.py` and method `exact_ball_uncertainty_floor`, without `--micro-check`. Preserve the config/profile for the chosen scenario and method.

## Scientific interpretation

All failures remain in results; no resampling or sensor-based success selection. Original Go counts are current 4/50, earlier 3/50, microcovariance oracle 0/50. The separately authorized Go floor appendix is 2/50. All original 450 plus all 150 posthoc appendix trials are included. Authoritative independent combined results and CSV are in `audits/comparison-50/independent-uncertainty-150/combined-600-summary.json` and `all-600-trials.csv`; original450 summaries remain separate. `missing-inputs.json` is empty in the final package. Rebuilding does not change frozen sources. High impact forces and contact sensitivity remain unresolved even where the declared model's possession/form gates pass.
