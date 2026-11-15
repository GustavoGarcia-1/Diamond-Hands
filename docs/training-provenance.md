# Training provenance and transfer evidence

Privileged inputs during upstream locomotion training are distinct from the
observations available to its exported actor. The new catching experiments
keep actor, critic and reward within the documented sensor and command boundary.

The receiver runtime uses surveyed RGB-D, IMU and joint measurements, plus
commanded route/receiving geometry and calibration. The new late-reach actor,
critic and reward also share sensor-derived inputs. Independent inspection
found no exact simulator ball/root/contact input in these reviewed paths.
Simulator state remains available to the simulator, evaluator and renderer.

The pinned AMP source at `386ee24bc62c8033641d65b940dc2cf1dec56545` defines
privileged critic and motion-discriminator features. Its exported actor uses
the audited 384 IMU/joint/command/history inputs. This training/runtime
distinction is permitted; the exact exported checkpoint's training-run lineage
is not proven by the published configuration alone. Source copies, URLs and
hashes are archived under
`reports/catch-repair-20261114/validation/upstream-training-boundary/`.
The NVIDIA WBC's exported input audit is likewise separate from its training
provenance. No inherited checkpoint is renamed as a newly learned receiver.

Simulation can reduce hardware training, but this study does not establish
zero-shot physical transfer. Contact properties are uncalibrated; the selected
catch depends on a changed contact time constant, forces remain high, whole-play
timestep changes alter interception, and small aim/latency changes can break it.
Camera calibration, exposure timing, detection under real lighting and occlusion,
Aero mounting/actuation and the standard runner's missing camera/hand channels
remain physical integration work. Existing stress cases are engineering
perturbations, not measured hardware distributions or independent success rates.

Sensor-only reward is also insufficient proof of possession: fixed phase-rate
controls earn visual retention credit during hand-mount support. The stopped
PPO checkpoint regresses against its matched baseline. These findings support
improving perception and grasp geometry, not claiming successful transfer or
continuing to optimize a known inadequate proxy. No hardware catch, qualified
safety policy or ARM64 builder-kit result has been established.

Primary sources: [AMP observation configuration](https://github.com/zihanwang0422/legged_rl_lab/blob/386ee24bc62c8033641d65b940dc2cf1dec56545/source/legged_rl_lab/legged_rl_lab/tasks/locomotion/amp/amp_env_cfg.py)
and [G1 specialization](https://github.com/zihanwang0422/legged_rl_lab/blob/386ee24bc62c8033641d65b940dc2cf1dec56545/source/legged_rl_lab/legged_rl_lab/tasks/locomotion/amp/config/unitree_g1/amp_flat_env_cfg.py).
