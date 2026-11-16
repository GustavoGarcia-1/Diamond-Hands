# Reproduction supplement: recorded environment, not a clean-machine guarantee

This supplement records the actual existing Linux worker and permits offline score reproduction. It does **not** demonstrate a fresh-machine installation or exact dynamics replay for the new Go/Out experiments. No active environment, frozen source or archive was changed.

## Offline score reproduction

Place `rescore.py` and `anatomical_score_portable.py` together. Python3.12 and NumPy are sufficient; no MuJoCo, Torch, GPU, Aero model or AMP weights are needed. Each trial directory must contain `report.json`, `anatomical-metadata.json`, `anatomical-contact.npz`, `anatomical-form.json` and `trace.npz`.

```sh
python -m venv offline-score-env
. offline-score-env/bin/activate
python -m pip install numpy==2.5.4
python rescore.py PATH/TO/trial --output rescored.json
```

This setup command is an explicit suggested offline-scoring environment, not a completed fresh-install test. The script exits nonzero if any full stored score differs; output includes all input hashes. The portable module exports the unchanged anatomical-v1 constants, criteria, run-length helper and score function; its source provenance is in `scorer-provenance.json`. All fourteen recorded zero/final-policy stress trials were rescored exactly (`rescore-verification.json`). The final policy model SHA is `436da7257d4483eaf9afee633c894babd902bf32c7893b8b570ca35a56ab2443`.

## Actual worker

Read-only capture: `collect_worker.py`, `worker-environment.json`, `requirements-observed-freeze.txt`. Python3.12.3, Linux x86_64/glibc2.39, RTX PRO6000; actual imports include NumPy2.5.4, MuJoCo3.15.0, Torch2.11.0+cu130, ONNXRuntime1.31.0, Gymnasium1.4.0, Stable-Baselines3 2.9.2, SciPy1.17.1. The JSON records driver/version, import paths and every installed distribution.

**Do not blindly install the full observed freeze.** The worker exposes system packages alongside its venv, including unrelated IsaacSim packages. Its `pip check` fails on pre-existing dependency conflicts, including Torch's Setuptools<82 requirement versus installed84.2.1 and incompatible unrelated IsaacSim pins. `pip freeze` is an environment inventory, not a resolver-valid lock. The project uv.lock selects a different Torch version and must not be presented as an exact reconstruction of this worker. Earlier clean-environment evidence for the separate native19456 component does not prove these new experiments replay exactly on a new installation.

## Assets and controller source

`asset-provenance.json` records all61 Aero and two AMP file hashes. Every remote Aero file matches the local `/tmp/g1-aero-source` content. Extracted worker asset folders lack `.git`; documented revisions are independently referenced from existing project provenance, not newly proved by remote git metadata:

- Aero Menagerie: `google-deepmind/mujoco_menagerie`, revision `fb36886b3023db08c198264e1e9956ab202c3e3b`. Use the recorded61-file manifest to verify the exact source tree actually consumed; upstream checkout layout may differ.
- AMP: `zihanwang0422/legged_rl_lab`, revision `386ee24bc62c8033641d65b940dc2cf1dec56545`. Required `deploy/g1_deploy/exported_policy/g1_run.onnx` SHA256 `17019141d56e57f53059c671c81e22d63c1dd2f7cc69a8b322c06790c00cca6f`; YAML SHA256 `3d45b9641f460eed486e2caa92feaedbfbf62d41b2434a2d0b38bb829a5319c8`.

AMP weights remain separately fetched; upstream licensing/redistribution and Aero hardware-design terms must be preserved. Source code alone does not confer hardware rights or establish actuator integration.

For actual Go rollout commands use the frozen source, profile, compact/cup JSON and exact argv in the accompanying Go reproduction notes/receipts. Rebase every absolute asset/output/source path. The Out downloaded runtime additionally requires a rebased JSON config, its frozen profile, source/model hashes and reference report directory used for passive evaluation. A model ZIP without that runtime/config is insufficient. Never substitute the active mutable source when claiming a frozen replay. Preserve original20ms versus explicitly experimental2.5ms ball–hand model identities, fixed1kHz servo, timestep, camera timing and device labels. CPU/CUDA contact trajectories can differ. No new ARM64 image, HIM runner qualification, hardware deployment, numerical convergence or material calibration is established here.

## Training-domain count

The final run completed149 logged episodes:71 noisy and78 nominal. The sampling configuration was50/50; realized counts need not be equal. `training-domain-count.json` records the two episode-log hashes. Reward and physical evaluation remain separate; increased visual reward did not improve this final model's physical catch.
