# Runtime sensor contract

Diamond Hands separates what a controller can measure from what a simulator can score. A sensor-realizable interface is a design constraint, not evidence of hardware transfer.

Controllers use timestamped RGB and registered axial depth, calibrated intrinsics, fixed surveyed camera transforms or a declared capture-time pose estimate, measured body joint positions/velocities, IMU orientation/gyro, and known route/fixture commands. RGB-D ball and receiver estimates retain their original capture timestamps and confidence. Stale/missing estimates must abstain or follow the explicitly declared stored-forecast/frozen-command behavior; silently retimestamping an old frame is invalid.

Field coordinates are right-handed and z-up. Camera optical coordinates are right/down/forward. Receiver localization covariance is `(x, y, yaw)`, not `(x, y, z)`; depth-height uncertainty is separate. Capture-aligned private FK uses buffered encoder/IMU observations, not the simulator's exact pelvis state. Robot encoders alone do not establish field localization.

The Aero hand has seven native command channels per side: four finger cable lengths, thumb opposition angle, and two thumb cable lengths. Native tendon-length and abduction sensors do not reveal the complete underactuated finger configuration. CAD-based FK, rendered depth and offline full-state evaluation must not be conflated with measured hardware finger pose. Existing hand control updates at 50 Hz; the 1 kHz passive recorder does not make it a 1 kHz actor. Sample-and-hold slew budgets are not continuous hardware guarantees.

Some archived native profiles preload from a known passing-machine release command and therefore require a command-message link. Camera-TTC-only variants are separate experiments. Their timing dependencies must remain visible in reproduction claims. A fixed declared ball attitude is a fixture assumption, not observed postcontact orientation.

The recent residual learners use the same sensor-derived vector for actor and critic. Rewards use sensor/FK/context features; exact contacts, true ball/pelvis state and evaluation success labels stay outside actor, critic and reward. Privileged observations during upstream locomotion training are permitted; deployed observations and actual transfer remain separate audits. The exact-ball comparison deliberately violates the deployable observation boundary and is labeled a privileged diagnostic.

A separate evaluator records free-joint release, stationary prerelease ball poses, applied loads, per-surface forces, support exclusions, penetration, palm form, upright posture, route motion and foot clearance. Catch scores cannot be inferred from visible dwell or proximity. Native mounts, wrist/body support and floor contact are not legitimate hand retention. Collision impulse is not a motor torque or tendon-force limit.

**Reward limitation:** greater sensor reward is not demonstrated catching. The independent Go reward replay preserved actions/physics while augmenting observer coverage; it still assigned a higher return to a failed catch than to retained control. This is evidence against reward specificity, not permission to use simulator truth as reward.

No real camera synchronization/calibration, encoder error model, motor thermal/continuous-duty envelope, hand mounting, communications link, fail-stop or hardware transfer has been qualified. The original 2.5 ms ball-hand contact assumption and optimistic symmetric native stall ceilings are simulation assumptions, not measured material or continuous actuator capability.

The [preserved working record](reference/sensor-contract-working-record.md) contains historical per-variant camera configurations and disclosures. Its references to dense local experiment reports are provenance pointers; those reports are intentionally outside this repository. Runtime implementation is in `perception.py`, `observable.py`, `localization.py`, `out_late_cup_capture.py`, `rgbd_impairments.py`, `native_cup_residual_rl.py` and the selected scripts.
