# Reproduction and verification

This repository is a curated working-source snapshot. The frozen 600-trial results used separate immutable controller bundles. Do not treat a command against today's source as an exact historical replay.

## November 15 additions

The curated source now includes the persistent encoder regulator, its opt in native grip integration, the sensor only precontact PPO interface and the transported camera target. The complete local suite passed **235 tests in 12.72 seconds** after these additions. This is a clean sibling checkout verification with pinned local assets, not a full replay of the frozen Linux studies. The focused CPU CI selection now contains 79 tests; it does not load training weights or robot assets.

`train_precontact_ppo.py` is a generic bridge. It requires a route driver and its frozen configuration from the separately archived experiment; this repository does not claim a standalone one command training replay. The 45 minute budgets, exact optimized versus collected step counts, sensor boundaries and camera provenance erratum are described in `precontact-ppo.md`. The source snapshot manifest records the current reporting metadata correction separately from the unchanged source hash used in the frozen runs.

The current submission package restores seven exact frozen source snapshots by content hash, includes complete raw evidence for its two primary Go plays, and documents pinned external AMP requirements. Secondary dense traces and all raw confirmation trajectories remain separate. See `OVERNIGHT_STUDY.md` for the package checksum and the scope of each result.

## Verified in the clean sibling checkout

These commands were executed here on macOS arm64 with Python 3.12.13. They are local verification, not an already completed GitHub Actions or Linux/ARM64 deployment run.

```sh
uv sync --locked --python 3.12 --extra training --extra test --extra video
uv run --no-sync python scripts/fetch_runtime_assets.py
uv run --no-sync python scripts/fetch_runtime_assets.py --check
uv run --no-sync python scripts/fetch_aero_hand.py assets/aero
AERO_SOURCE="$PWD/assets/aero" uv run --no-sync pytest -q
```

Result: **220 tests passed**. The runtime fetch downloaded and verified 56 files, 35,315,456 bytes, against the included source/size/SHA256 manifest. Aero was fetched into a new local directory and verified by its pinned loader. These generated data are ignored by Git.

The model-free CI subset was also run in a separate environment with only `--extra wbc --extra test`, without PyTorch, while robot/weight directories were temporarily absent and `AERO_SOURCE` pointed to an absent directory: **64 tests passed**. Hosted CI runs separately from this local verification. See `evidence/verification/model-free-tests.json`.

## Rendered perception smoke

```sh
mkdir -p outputs
uv run --no-sync python -m g1_receiver.sim --frames 60 \
  --sensor outputs/flight-sensor.npz --truth outputs/flight-truth.npz \
  --overview outputs/flight-overview.npz
uv run --no-sync python -m g1_receiver.perception outputs/flight-sensor.npz \
  --allow-synthetic-fixture --plane-point 1 0 0 --plane-normal 1 0 0
uv run --no-sync python scripts/summarize_evidence.py
```

The smoke rendered 60 RGB-D frames and produced sensor-only interception records. The local graphics backend warned that `ARB_clip_control` was unavailable and depth precision was limited. The truth NPZ is an evaluator output, not a perception input. This is a stationary synthetic flight fixture, not a route catch or hardware perception result. An environment with no usable graphics backend may fail; MuJoCo's supported headless backend must be configured appropriately there.

## Runtime data and optional route probes

NVIDIA inputs are restored in `src/g1_receiver/_vendor/g1_runtime` by the manifest fetcher; no model or ONNX file is committed. The runtime source, checkpoints and code/weight license terms are pinned independently. Aero input is the external MuJoCo Menagerie model, not a fabrication license for the physical hand.

AMP inputs remain an external checkout of `zihanwang0422/legged_rl_lab` at `386ee24bc62c8033641d65b940dc2cf1dec56545`. The adapter checks the ONNX/YAML hashes in `running_policy.py`. An upstream checkout can be obtained with:

```sh
git clone --no-checkout --filter=blob:none https://github.com/zihanwang0422/legged_rl_lab.git assets/amp
git -C assets/amp sparse-checkout set deploy/g1_deploy/config deploy/g1_deploy/exported_policy
git -C assets/amp checkout --detach 386ee24bc62c8033641d65b940dc2cf1dec56545
```

Those three checkout commands are documented from the pinned upstream workflow; they were **not rerun as part of this clean-checkout verification**. Review the external model's terms before use. Do not redistribute it under an assumed Diamond Hands license.

The curated Go, Out and Slant scripts accept explicit Aero/AMP paths. Their `--help` imports were checked here:

```sh
uv run --no-sync python scripts/probe_aero_out.py --help
uv run --no-sync python scripts/probe_go_side_causal_pass.py --help
uv run --no-sync python scripts/probe_slant_lower_palms.py --help
uv run --no-sync python scripts/train_out_late_reach.py --help
```

This verifies CLI/import closure, not their historical physical outcomes. Use explicit `--source assets/aero --amp-source assets/amp` where supported. Some historical defaults point to generic `/tmp` directories or omitted experiment profiles; those defaults are not fresh-checkout runnable. Training drivers/configurations and frozen comparison adapters may depend on an immutable runtime tree specified by their configuration. No one-command historical catch replay is claimed for this curated snapshot.

## Exact frozen study

Included protocols, seed/scenario files, source/checkpoint digests and independent summaries are in `evidence/`. The original 450 trials and separate posthoc 150 were independently rescored from dense traces, with free-ball, model, source, release, physical/form and support checks. The concise summaries retain the outcomes and fingerprints.

Dense NPZ trajectories, per-contact witnesses, selected MP4s, historical checkpoint files, full frozen controller/source archives and pretrained binary assets are intentionally excluded. The separate November15 evidence folder now includes the small final Slant arm checkpoint and its exact closing limit adapter, with hashes; this does not supply the historical runtime driver or assets. They are preserved separately in the originating research archive. **This Git repository alone cannot rerun or independently rescore all 600 historical trajectories.** The archives have not been published as GitHub release assets. Obtain the exact source/evidence archives identified by the included SHA256 values before making an exact-replay claim.

The later one-hour PPO records are fixed heterogeneous diagnostic comparisons, not new 50-trial success estimates. The reward-specificity audit shows why sensor return cannot select a physical catch. The small included Out identification record is a separate no-ball calibration, not evidence of a learned catch or hardware response.
