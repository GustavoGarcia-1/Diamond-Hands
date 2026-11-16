# Diamond Hands

**Can we teach a robot to play football?**

[Website and videos](https://gustavogarcia-1.github.io/Diamond-Hands/) · [Reproduction guide](docs/REPRODUCTION.md) · [Sensor contract](docs/SENSOR_CONTRACT.md)

I developed Diamond Hands for the [HIM Robotics Open Build Challenge](https://arena.himrobotics.com/challenges/open-build). In simulation, a Unitree G1 follows three receiver routes and tries to catch passes from a fixed machine. Pretrained locomotion moves the robot. A camera estimate times the reach, encoder feedback regulates the fingers, and a separate online PPO experiment learns small arm corrections.

## Results

- **Grip control:** On 50 fresh straight route throws, encoder regulated fingers retained 34 passes; fixed closure retained 2. Both controllers received while walking.
- **Arm learning:** On 50 fresh inward cutting throws, the PPO arms retained 12 passes; scripted arms retained 19. I did not promote the learned policy.
- **Outward cut:** The original controller retained 9 of 50 development throws; a timed entry revision retained 8. This was not an improvement.

These are separate comparisons, not one pooled success rate. A retained catch requires legitimate hand support at the finish, without help from the body, mounts, or field. The videos and the 50 throw studies use different experimental settings. [Methods, negative results, and provenance](docs/OVERNIGHT_STUDY.md) and the [frozen comparison evidence](evidence/README.md) give the full context.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/) are used for the locked development environment.

```sh
uv sync --locked --python 3.12 --extra training --extra test --extra video
```

The basic checks use synthetic fixtures and do not need downloaded robot models:

```sh
uv run --no-sync pytest -q tests/test_perception.py tests/test_catching.py
uv run --no-sync python scripts/summarize_evidence.py
```

For full simulation, fetch the pinned robot and hand assets and follow the [reproduction guide](docs/REPRODUCTION.md). The source tree is organized as `src/g1_receiver/` for controllers and evaluators, `scripts/` for experiments, `tests/` for regression checks, and `evidence/` for frozen summaries.

## Scope

The catching policy uses camera estimates, IMU, and joint encoders. Exact simulator ball state and contact are reserved for evaluation or clearly labeled diagnostics. No physical G1 has run this system; camera and Aero hand integration, contact calibration, and hardware safety checks remain open. See the [sensor contract](docs/SENSOR_CONTRACT.md) and [third party notices](THIRD_PARTY_NOTICES.md).
