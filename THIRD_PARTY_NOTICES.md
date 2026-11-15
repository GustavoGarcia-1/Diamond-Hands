# Third-party notices

No project-wide license has been selected for Diamond Hands. This document does not grant a new license or replace upstream terms.

## NVIDIA G1 runtime

The asset manifest pins `NVlabs/GR00T-WholeBodyControl` revision `b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`, with individual source URLs, byte counts and SHA256 values. Robot XML/meshes and pretrained walk/balance ONNX files are **not committed**. `scripts/fetch_runtime_assets.py` restores and verifies them in ignored paths.

Preserved upstream terms are `src/g1_receiver/_vendor/g1_runtime/assets/upstream/CODE-LICENSE.txt` and `WEIGHTS-LICENSE.txt`; preserve the original `NOTICE.txt` and `ASSET-MANIFEST.json`. The model/checkpoint license must be reviewed for the intended use. Files under `_vendor/g1_runtime/pool` came from the author's existing g1-pool NumPy/ONNX adapter, as documented by its retained notice; do not assume all vendor code is newly authored here or covered by a new blanket license.

## Aero Hand Open simulation model

Aero simulation inputs are external: Google DeepMind MuJoCo Menagerie revision `fb36886b3023db08c198264e1e9956ab202c3e3b`, directory `tetheria_aero_hand_open`. The model has its own Apache-2.0 license, fetched with its XML and meshes. `scripts/fetch_aero_hand.py` verifies the pinned XML and referenced mesh digest through `g1_receiver.aero`. This simulation-model license does not license the separate physical hand design; hardware design terms and fabrication rights must be reviewed separately.

## AMP locomotion

The adapter pins `zihanwang0422/legged_rl_lab` revision `386ee24bc62c8033641d65b940dc2cf1dec56545`. Its ONNX checkpoint and YAML configuration are **not redistributed**. The inspected README advertised Apache-2.0, but the inspected revision had no root/model-specific license file. Obtain the upstream inputs directly and review permission for the intended use. Exact hashes are checked in `g1_receiver.running_policy`; no new sublicense is asserted.

## Python dependencies

MuJoCo, NumPy, ONNX Runtime, SciPy, PyTorch, Gymnasium, Stable-Baselines3 and other locked dependencies retain their own licenses. `uv.lock` pins distributions and hashes; dependency installation is separate from permission to redistribute model weights or hardware designs.

## Scientific context

The project uses pretrained locomotion, scripted/FK-based grasp geometry, RGB-D estimation, bounded residual PPO, paired physical tests and independent scoring. It is not a reproduction of Catch It All, Kimodo, Holosoma or a hardware system. Their ideas or offline pose proposals do not establish implementation equivalence, transfer or a new learned catch result.
