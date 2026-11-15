# Diamond Hands

**Can we teach a robot to play football?**

Gustavo Garcia<br>
University of California San Diego

A simulation research project studying perception, hand geometry, physical contact and learning for a Unitree G1 football receiver. Pretrained locomotion and scripted catching provide the baselines; bounded PPO residuals are experiments. **A reliable learned catch in stride has not been demonstrated.**

[Research demos and results](https://g1-receiver-lab-1111.gag009.chatgpt.site) · [Reproduction](docs/REPRODUCTION.md) · [Sensor contract](docs/SENSOR_CONTRACT.md)

## What the experiments show

The strongest population evidence is a scripted grip improvement: persistent finger encoder regulation retains **34 of 50 fresh Go confirmation throws versus 2 of 50** for the original controller. Development results were 31 versus 4 of 50. The confirmation set was frozen before evaluation and has not been reused for tuning. This is a walking receive near six metres, with pretrained locomotion.

The protected Out controller retains **9 of 50 development throws**. An entry and regulation bundle gives 8 of 50 and is not promoted. A trace audit explains why copying a successful route wholesale is insufficient: 32 of 50 original Out throws contact the torso before the hands, whereas original Go frequently achieves bilateral control and loses it later.

The newest Slant PPO improves nominal clean control from **171 to 873 milliseconds**, but does not retain the ball through the endpoint on its own. Adding one scripted limit on continued finger closure gives **1.995 seconds through the finish** in the nominal play. Across seven fixed development conditions, retained outcomes are zero for scripted preparation, one with the closing limit alone, zero for PPO alone and four for PPO with the same limit. A fresh matched confirmation then gives **19 of 50 retained catches for scripted arms versus 12 of 50 for learned arms**, both with the same closing limit. Six throws improve and thirteen worsen with learning; the paired difference is minus 14 percentage points with a 95% bootstrap interval from minus 30 to plus 2. The learned policy is not promoted. These data are excluded from further tuning.

A separate ten yard Go probe first receives at 9.970 yards and achieves a 1.645 second longest clean interval, with possession again at the finish. Both tested controllers pass two of seven fixed conditions. Its result is separate from the six metre confirmation distribution. All these newer receives fail the recurrent running check.

The historical study contains 450 paired trials across three methods and routes, plus 150 separate exact ball uncertainty diagnostics. Its original counts and negative PPO results remain in [the included evidence](evidence/README.md). The newer precontact PPO, negative Slant confirmation and scripted feedback rule are documented in [the November 15 study](docs/OVERNIGHT_STUDY.md).

Contact materials, camera uncertainty and actuator transmissions remain uncalibrated. Actual command and torque caps do not bound collision impact. There is no hardware validation, integrated camera or hand driver, ARM64 controller image or completed builder safety qualification. Upstream gait training may use privileged information; the new catching actor, critic and reward use sensor estimates and known commands only.

## Start here

Python 3.12 and [uv](https://docs.astral.sh/uv/) are used for the locked development environment. The internal distribution/import name remains `g1-receiver` / `g1_receiver` to preserve source and lock compatibility.

```sh
uv sync --locked --python 3.12 --extra training --extra test --extra video
uv run --no-sync pytest -q tests/test_perception.py tests/test_catching.py \
  tests/test_observable.py tests/test_routes_evaluation.py \
  tests/test_running_catch_eval.py tests/test_receiver_comparison_plot.py \
  tests/test_rgbd_impairments.py tests/test_anatomical_catch_eval.py \
  tests/test_precontact_native_rl.py tests/test_transport_precontact_rl.py \
  tests/test_persistent_finger_regulation.py
uv run --no-sync python scripts/summarize_evidence.py
```

That 79-test subset uses synthetic fixtures and needs no downloaded robot models. The complete tests and simulation require pinned external runtime inputs:

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
- `evidence/`: small frozen protocols, scenarios, independent summaries and provenance. The small final Slant arm checkpoint and its frozen closing limit adapter are included. Dense traces, complete frozen source archives, pretrained locomotion weights and videos remain separate.
- `docs/`: sensor contract, dependencies, methods and reproduction limits.

The working source snapshot is not a substitute for the immutable controller bundles used for the 600-trial study. Reproducing those exact trajectories requires their separate source/evidence archives.

## Licensing and provenance

A project-wide license has not yet been selected. Third-party terms are retained separately; no new license grant is made for their code, models or weights. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Aero and AMP inputs are fetched from pinned upstream revisions rather than committed. Existing NVIDIA runtime files are restored from a checksum manifest with their original license files.
