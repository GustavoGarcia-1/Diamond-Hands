# Hardware and deployment status

No hardware is connected. The current results are simulated walking receives with uncalibrated contact materials, actuator transmission models and camera uncertainty. No physical catch, safe impact qualification or complete G1 deployment has been established.

The experiments require surveyed external RGB D cameras, a known marker rig, exposure timestamps, IMU measurements, body joint encoders and fourteen native Aero actuator encoder channels. The protected preload sequence additionally uses a commanded machine release event. Reproduction on hardware needs the corresponding camera installation, calibration, timestamped machine command link, hand mounting, hand driver and measured actuator feedback. These integrations have not been verified on HIM’s G1.

The standard HIM runner supplies joints and IMU, so the additional inputs must be declared and integrated. A supervised attempt also needs one Linux ARM64 controller image with its runtime dependencies, followed by the builder safety and counterfactual checks on that exact image. Source and video packaging does not satisfy those requirements. See the [HIM builder guide](https://arena.himrobotics.com/docs/g1-policy-checks) and [sensor contract](SENSOR_CONTRACT.md).

The source archive records actual command and actuator limit checks. These limits do not bound collision impulses, and the existing startup joint range excess is retained in the evidence. The visual loose ball intervention prevented one selected trip; it is not a general safety controller, graceful falling policy or standing recovery.

Before a physical receiving study, the installed hardware, frames, timestamps, observation latency, actuator ordering and transmission limits need direct measurement. Incoming trajectory estimation and retained possession require separate validation. Neither rendered camera availability nor simulated contact duration establishes those physical capabilities.
