# Diamond Hands

**Can a humanoid robot learn to catch a ball in stride?**

Gustavo Garcia<br>
University of California San Diego

A simulation research project studying perception, hand geometry, physical contact and learning for a Unitree G1 football receiver. Pretrained locomotion and scripted catching provide the baselines; bounded PPO residuals are experiments. **A reliable learned catch in stride has not been demonstrated.**

[Research demos and results](https://g1-receiver-lab-1111.gag009.chatgpt.site) · [Reproduction](docs/REPRODUCTION.md) · [Sensor contract](docs/SENSOR_CONTRACT.md)

## What the experiments show

The frozen study contains 450 paired trials across three methods and three routes, plus a separate 150-trial posthoc exact-ball uncertainty diagnostic. Current-controller physical passes were **Go 4/50, Out 9/50, Slant 0/50**. These are seeded simulation engineering tests, not hardware reliability estimates. Selected retained receives are walking; they do not establish recurrent running catches.

The contact model is uncalibrated. Motor ceilings do not bound collision impulses, and low overlap does not establish safe impact. Exact-ball diagnostic inputs are privileged and exposed a tracker/covariance mismatch; they are not a deployable camera or a mechanical upper bound.

Three later one-hour PPO runs did not improve the matched physical catches. Sensor reward sometimes ranked a failed catch above a retained one, including after observer coverage improved. No learned catch controller from those runs is promoted. See the [included evidence](evidence/README.md).

There is no hardware validation, calibrated contact/material identification, proven camera/hand integration or qualified deployment. Upstream gait training may use privileged information; runtime sensor availability and transfer are separate questions.

## Start here

Python 3.12 and [uv](https://docs.astral.sh/uv/) are used for the locked development environment. The internal distribution/import name remains `g1-receiver` / `g1_receiver` to preserve source and lock compatibility.

```sh
uv sync --locked --python 3.12 --extra training --extra test --extra video
uv run --no-sync pytest -q tests/test_perception.py tests/test_catching.py \
  tests/test_observable.py tests/test_routes_evaluation.py \
  tests/test_running_catch_eval.py tests/test_receiver_comparison_plot.py \
  tests/test_rgbd_impairments.py tests/test_anatomical_catch_eval.py
uv run --no-sync python scripts/summarize_evidence.py
```

That 64-test subset uses synthetic fixtures and needs no downloaded robot models. The complete tests and simulation require pinned external runtime inputs:

```sh
uv run --no-sync python scripts/fetch_runtime_assets.py
uv run --no-sync python scripts/fetch_aero_hand.py assets/aero
AERO_SOURCE="$PWD/assets/aero" uv run --no-sync pytest -q
```

Fetched models and pretrained weights are ignored by Git. [Reproduction notes](docs/REPRODUCTION.md) distinguish commands checked in this clean working directory from historical frozen Linux replay proofs.

## Repository

- `src/g1_receiver/`: RGB-D estimation, localization, route/hand control, sensor-only training interfaces and separate physical evaluators.
- `scripts/`: a curated dependency closure of local probes, asset fetching and study validation. Remote launchers and credentials are excluded.
- `tests/`: perception, timing, sensor-boundary, controller and physical-score regression tests.
- `evidence/`: small frozen protocols, scenarios, independent summaries and provenance. Dense traces, frozen source archives, checkpoints and videos are not included.
- `docs/`: sensor contract, dependencies, methods and reproduction limits.

The working source snapshot is not a substitute for the immutable controller bundles used for the 600-trial study. Reproducing those exact trajectories requires their separate source/evidence archives.

## Licensing and provenance

A project-wide license has not yet been selected. Third-party terms are retained separately; no new license grant is made for their code, models or weights. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Aero and AMP inputs are fetched from pinned upstream revisions rather than committed. Existing NVIDIA runtime files are restored from a checksum manifest with their original license files.
