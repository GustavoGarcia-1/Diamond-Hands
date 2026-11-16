# G1 receiver sensor contract


## Slant learned arms and scripted late closing floor

The Slant transport checkpoint controls fourteen bounded arm residuals. A
separate scripted native hand rule latches the eight requested finger cable
positions at the first control sample at or after the observed visual receive
cue plus 1.30 seconds. It applies the larger of each subsequent base request
and its latched position. Larger cable position means opening, so further
closing is limited while opening remains allowed. The six thumb requests are
unchanged. The same sampled 1.5 millimetre encoder closing error rule, target
slew limits and compiled actuator caps remain active. This rule uses the
causal visual cue, native encoder packets, known requests and the controller
clock. It does not use simulator contacts, physical forces, exact ball state
or exact root state. Learned arms and this scripted floor are distinct methods.

Across seven fixed unique development conditions, retained catches were 0
with zero arm actions and no floor, 1 with zero actions and the floor, 0 with
learned arms and no floor, and 4 with learned arms and the floor. These fixed
cases are development evidence, not a population rate or proof that learning
alone is necessary. The nominal learned plus floor interval is 1.995 seconds
through the endpoint under the declared engineering model.

The subsequent fresh paired confirmation used seed `2026111527`, now consumed,
for fifty independent requested aim and release perturbations. Zero actions
plus the same floor retained 19 of 50, compared with 12 of 50 for learned arms
plus the floor. There were six paired gains and thirteen losses. The learned
minus zero difference was minus fourteen percentage points, with a paired
bootstrap 95 percent interval from minus thirty to plus two percentage points.
This reverses the development ordering and does not support promoting the
learned controller for reliability. These probabilities concern only the
declared small jitter distribution, fixed cameras and engineering simulation
model. The historical realistic reference flag remains false. They establish
neither running at catch, complete route execution nor hardware transfer.

## November 15 precontact learning and visual hazard experiments

The new 169 feature PPO actor and critic use the same camera, IMU, body encoder,
native encoder and known command inputs. The policy requests fourteen bounded
arm residuals. Native grip, thumbs and locomotion remain fixed. Positive
alignment shaping ends at an absorbing arrival deadline estimated from fresh
camera trajectories. It does not use simulator contact or a physical success
score. The separately versioned transport experiment estimates horizontal
receiver motion from past field positions. Its equations, bounds and negative
initial results are in [the learning protocol](precontact-ppo.md).

The capture adapter currently reuses the optical frame label `camera_optical`
across physical views. The transport estimator therefore does not enforce its
originally stated single physical camera history guarantee. Independent
nominal Go and Out audits identify no mixed receiver camera endpoints, but that
does not cover every training episode. Three of ten positive nominal Slant
histories combine entry and pass camera estimates in the common field frame.
The source and frozen jobs are preserved
with an explicit erratum. A future provenance guard must carry the receiver
pose camera and cache origin from its own estimator; the ball camera identity
alone is insufficient. All positions still use their corresponding surveyed
transforms into the common field frame. No missing measurement is replaced
with simulator state.

The protected Out native preload also preserves an existing commanded machine
release event. The simulation supervisor provides that event; it is not a
camera measurement or an integrated physical machine signal. Reproducing this
controller requires a timestamped command link to the throwing machine, in
addition to the declared sensors. The later grasp transition still uses vision.
The longer flight diagnostic advances the same event by 160 milliseconds, so
preload and camera timing can both change. It is not a camera only controller
or an isolated change in ball descent angle.

The current failed Go catch avoidance experiment adds a dedicated RGB D stream
from the literal surveyed camera at `(3, -3, 2)` metres with a 48 degree field of
view. The existing controller's `pass` alias points to a different `late_cue`
camera and remains unchanged. The additional observer arms at the actual visual
receive cue, then requires two fresh low ball hazard observations within a
predicted foot corridor. Its geometry uses the visible ball surface, camera
uncertainty, IMU, encoders and commanded velocity. It never triggers from
simulator collision labels. A 250 millisecond velocity goal cancellation
prevented one recorded loose ball trip. Six nearby nonfall cases and a retained
catch control remained unchanged. This is scripted avoidance evidence, not a
trained fall recovery or a hardware safety qualification. Preexisting startup
joint range excess remains documented.

## November 15 native finger feedback

The optional `native_grip.finger_encoder_regulation` setting enables the frozen
persistent grip rule. It uses only the fourteen native actuator position
encoder channels, original nominal targets, the existing visual grasp event,
and monotonic control time. Eight finger tendon targets limit requested closing
position error to 1.5 millimeters after that event. All six thumb channels are
unchanged. Existing native target slew and compiled actuator limits remain in
force. Create a new grip controller for each play.

This position error rule does not measure contact, current, force or complete
underactuated finger posture. Its sampled error ceiling is not a continuous
force bound. The isolated Go studies used a 50 Hz control loop and verified
added encoder transport delays and quantization as engineering stresses. Those
stresses are not measured hardware specifications. Original delivery timestamps
and any inherited simulator sensor phase remain distinct from an added delay.

`src/g1_receiver/finger_encoder_regulation.py` preserves the exact helper used
in the frozen Go experiments. `NativeCupGrip` enables it only with an explicit
configuration; existing defaults are unchanged. The normal native trace retains
nominal commands, `native-finger-regulation.json` records the regulated requests,
and `native_applied_commands` records targets sent to the plant. Downstream rate
limits can make requested and applied values differ.

The grip improvement is scripted. Pretrained locomotion remains separate. All
contact, exact ball and root state, physical actuator force and full replay pose
data are evaluation or rendering inputs only in these new catching and visual
hazard paths. This statement does not describe upstream locomotion training. Physical camera, marker and Aero
hand integration remain unverified. The following sections retain earlier
experiments and their distinct sensor configurations.

**User clarification, November 14 evening:** privileged information during
locomotion training is permitted. The deployed controller must use physically
available sensor inputs; simulation performance alone does not establish real
transfer. Upstream training provenance is disclosed separately from runtime
input availability. See [training provenance and transfer evidence](training-provenance.md).

## November 14 evening audit: observability is not possession

The R7 relative-camera servo and new carry experiments still use rendered
RGB-D, surveyed calibration, timestamped IMU and encoders. Physical support
scores are evaluated afterward and are not fed back as reward or action inputs.
This does not make the visual reward equivalent to catching: fixed phase-rate
controls exposed reward credit during excluded hand-mount support, and visual
dwell can exceed actual uninterrupted bilateral hand support. PPO r3b was
intentionally stopped on this counterexample; its 12,288-step checkpoint is
preserved for diagnosis, not promoted. Reward v3 is not a qualified possession
detector. See the frozen constant-phase study and independent support audit.

A passive hypothetical camera on the torso/head housing at local
(0.12, 0, 0.55) m uses a rigid mount and encoder-only kinematics to express
visible depth surfaces in the pelvis frame. It does not consume the live
simulator camera world pose. It sees the held ball well but misses the approach,
and its estimated mount-clearance distributions overlap for legitimate hand
support and excluded mount support. With declared mounting, encoder and shape
uncertainty, no tested frame certifies positive three-sigma mount clearance.
The diagnostic abstains; it is not a new retention reward, onboard hardware
integration or active gaze controller. External approach views remain needed.

The full-play fall supervisor uses IMU/joint feedback and a causal visual
receive-phase event. Contact force labels and full poses belong to the matched
evaluation only. Early balance abort forfeits the catch; shallow bracing can
reduce impact but does not demonstrate getting up. These are scripted research
probes, not a hardware-qualified safety policy or a reproduced Disney learner.

## November 14 late-reach experiments (development only)

The new Out `late-reach-phase-65-capture-aligned-v2` observation contains 65
camera/IMU/encoder-derived features. Actor and critic receive the same vector.
The learned action is one bounded reach-phase rate residual (±35%); visual
activation, arm geometry, native grip and locomotion remain separately scripted
or pretrained. The reward uses these sensor packets, not contact, exact ball or
root state, future simulation trajectories, or physical evaluation scores.

Past IMU/joint measurements are buffered and interpolated to the camera's
exposure timestamp for palm forward kinematics. Receiver XY covariance and its
separate Z uncertainty are propagated explicitly; yaw variance is not a height
variance. Image age is observable, but absolute episode time is not. Fresh
visual evidence requires age ≤80 ms. Missing or repeated frames do not accrue
positive retention dwell. Visual retention additionally requires valid relative
velocity, uncertainty bounds, low/upward palms and an upright IMU. Occlusion
does not establish continuing possession.

The stress wrapper operates on rendered RGB/depth packets, preserves exposure
timestamps and corresponding calibration, and adds declared latency, frame
drops, noise, quantization or survey bias. It never fills missing observations
from geometry or exact state. These are engineering stress distributions, not a
calibrated hardware sensor model. Camera layout R6 and the newer coverage-only
R7 experiments are distinct frozen sources and profiles; the completed PPO r2
and its seven paired stress cases use R6. The PPO did not improve the measured
catch and is not promoted.

The separate Go side-cup experiment uses a surveyed camera at (3, -4, 3) aimed
at (3.5, -.4, 2), with an actual 80-degree vertical field of view and matching
per-frame intrinsics. A fresh RGB-D interception estimate triggers the late arm
reach without an absolute-time or machine-release fallback. Native preload
still depends on a commanded machine-release event. This event interface,
external cameras, marker calibration and Aero hand channels need physical
integration. There is no demonstrated onboard gaze controller or LiDAR input.

For these catching control and reward paths, simulator contacts, exact poses
and actuator forces remain passive evaluation and rendering data. Original body actuator torque caps are preserved, but those
caps do not bound passive impact forces. The pinned native Aero actuators have
no scalar force clamp; bounded tendon targets and target slew are not force
limits. An isolated full-episode stall-envelope diagnostic changes only those
native actuator bounds and records force as evaluation data, never as a new
controller observation. It is not a calibrated hardware motor model. See
`reports/catch-repair-20261114/go-side-cup/actuator-envelope/README.md`.
The upstream training history of the
externally pretrained locomotion policies has not been established as adhering
to this experiment's sensor-only actor/critic/reward boundary. No hardware
catch or calibrated contact model is claimed.

The receiver uses 29 joint position and velocity channels, the pelvis IMU, and surveyed fixed external RGB-D views. The standard HIM runner supplies the first two inputs. The external camera and marker survey need a separate connection and qualification.

The visible stadium replay camera is for presentation only. The selected plays
do not use a head-mounted camera or an active head-gaze controller. Their catch
timing comes from the external ball-camera measurements and causal trajectory
estimates. A head turn in a replay would not by itself establish onboard vision.

| Input | Format and timing |
| --- | --- |
| Joint state | All 29 G1 slots in the builder kit SDK order. Position in radians and velocity in radians per second. Target control rate 50 Hz. |
| Pelvis IMU | Unit quaternion in wxyz order and body-frame gyro in radians per second, sampled with the joint state. |
| Camera image | RGB uint8 and rectified, registered float axial depth in meters on the same pixel grid. The simulation used 320 × 180 at 25 Hz. A physical stream can use a different calibrated resolution, subject to latency and detection checks. Invalid depth is zero. |
| Intrinsics | Rectified color fx, fy, cx, cy in pixels. Apply any resampling to the calibration as well as the images. |
| Extrinsics | Rotation and translation from optical x right, y down, z forward to the surveyed field frame in meters, with z up. Supply a pose at depth exposure time and the survey uncertainty. |
| Pelvis markers | Two colored markers at measured offsets matching the localization model. Cyan is left and magenta is right. The nominal modeled centers are (0.15, 0.11, 0.08) and (0.15, −0.11, 0.08) m from the pelvis frame origin, with 35 mm radius. |

Map the camera exposure time into the same monotonic clock used by the joint and IMU adapter. Preserve the original timestamps and record the measured clock mapping. Reject mismatched RGB/depth pairs and future or out-of-order frames. The localizer allows up to 10 ms between the image and pose timestamps. The deployed observation assembly rejects camera ages above 120 ms. The independent route evaluation exercised 80 ms simulated delay; physical latency and jitter remain unmeasured.

The tracker estimates a ball surface position in field meters with a calibrated axial offset and causal velocity in meters per second. It does not consume simulator position or velocity. When the ball is occluded, the observation marks it invalid. The deployed observation/controller interface has stale-localization and proprioception guards, but the verified Go simulation does not run that interface: its configured IMU/encoder runup continues when camera tracking is lost. Its camera gates the grip and calibrated arm phases; camera loss does not stop locomotion in this archived Go. Do not assume fail-stop camera behavior from this replay. A future physical adapter must implement and validate stale-sensor handling and stop publishing on the checker's abort, leaving motor output to the host.

The current RealSense utility supports fixed calibrated capture files. The onboard RealSense model and stream capabilities remain unknown, and a moving head camera requires a separate measured pose source. The Go fallback uses its archived fixed surveyed camera views; a single camera is not established as sufficient for the selected Out. Use the [read-only commissioning procedure](hardware-qualification.md) to qualify a physical camera before connecting the controller.

## Aero hand and hardware status

The simulation preserves 29 G1 body actuator slots and adds 14 Aero hand actuators, seven per hand. These are additional hardware channels, not extra G1 body slots. Tendon-length channels use meters and thumb opposition uses radians; each needs a calibrated motor adapter, limits, power, timing and emergency-stop integration. The mounting transform is provisional simulation geometry. Physical Aero integration, fixed RGB-D registration and latency, and all hardware operation remain unverified. No Linux ARM64 deployment image has been built or tested; HIM builder safety and counterfactual checks have not been run.

## Historical camera-braked Out: four surveyed fixed RGB-D views

This older backpedaling reference used four calibrated views in the same world frame: break at(0.65,0.15,1.37)m aimed at(4.2,2.15,0.8)m; outside at(8.4,1.3,1.46)m aimed at(5.6,4.05,0.81)m; pass at(7.3,1.8,1.88)m aimed at(3.5,3.23,1.4)m, also recorded in `reports/out-scrimmage/frame-speed-20261113/finish-extra-camera-linux/artifacts/out/profile-00.json`; finish at(2,8.5,1.8)m aimed at(3.2,4.5,0.8)m. Camera rotations/intrinsics are constructed by the archived probe configuration and the finish calibration is recorded in its report. The finish view is a fallback only when normal route views have no marker estimate and at least0.7s has elapsed since visual grip closure. Fresh RGB-D marker localization triggers its outside-depth brake, not simulator position or contact.

The pass camera gates grip; break/outside/finish cameras localize pelvis markers. This selected Out rotates the body to look back while preserving outside travel, so rear-marker visibility and the finish view matter. All views require exposure timestamps mapped to the joint/IMU clock, surveyed extrinsics, registered depth and measured uncertainty/latency. Physical multi-camera transport, synchronization, occlusion handling and registration are unverified. The Go fallback retains its own archived survey and IMU runup behavior; do not apply the selected Out camera configuration retrospectively to its evidence.

## Archived Aero residual PPO: external retention camera

The archived training experiments added one surveyed external RGB-D view at
world(9.8,3.0,4.0)m, aimed at(10.0,3.4,0.9)m in the Go trial frame. This is an
elevated fixed mount in simulation, analogous to a calibrated tall tripod or
stand. The original pass view observes approach; after the visual grip trigger,
the added view observes the ball and pelvis markers near the hands. Fresh marker
estimates from the existing tracking views may fill a missing marker observation.
A prior estimate is never treated as current: the learner enforces80ms maximum
age and40ms ball/pose time skew, with only numerical floating-point tolerance.
Missing measurements clear visual dwell and do not become simulator state.

The actor and critic share111 sensor-derived features at25Hz. Sixteen bounded
residual actions adjust14 arm targets and2 hand closure commands; pretrained
locomotion remains the baseline. Reward uses camera ball position relative to
an encoder/IMU-derived hand cup, observed relative velocity and uncertainty,
plus posture and action penalties. A separate kinematic state computes the cup
from29 body encoders and IMU. Contacts, exact ball/root pose, and replay frames
are evaluation-only. The first256-step CUDA run verified optimizer execution
but exposed poor original-camera coverage; it establishes no catch improvement.
The added view's replay audit saw the ball in20/20 catch-window images and valid
same-frame ball+marker estimates in13/20, including the full brief physical
catch. See reports/aero-rl-20261113/visibility-* and the frozen job receipts.

The earlier camera-braked Out above is now a backpedaling grasp reference,
not the selected forward-left route. New forward-left and centerline-quarterback
trials preserve their own camera surveys. External-camera mounting, calibration,
synchronization, real RGB-D detection and Aero hand integration remain unverified.

## Current three held plays: exact camera configurations

The published held Go, forward-left Out and inward Slant use different surveyed
views. Positions and aim points below are world XYZ meters, before the renderer's
field-only translation. They must not be reused across route frames implicitly.
The passing machine is on the declared scrimmage line, two yards inward from
the receiver's initial position. It is a visual representation of the fixed
simulated release fixture, not an additional perception input.

- Go uses track (-1, 0, 1.5) aimed at (3, 0, .9), finish (5, 0, 1.5) aimed at
  (10, 0, .9), and pass (6, -4, 3.8) aimed at (9, 0, 1.5). Its field heading is
  zero. Camera depth triggers the 1.3 m handoff; the physical initial stance yaw
  is -0.7 rad. The final profile is
  `reports/aero-rl-20261113/go-cup-pass-linux/artifacts/go/profile-07.json`.
- Out uses break (.65, .15, 1.37) aimed at (4.2, 2.15, .8), outside
  (8.4, 1.3, 1.46) aimed at (5.6, 4.05, .81), and pass (6.8, 2.4, 3.5) aimed
  at (3.7, 2.4, 2.8). Its declared field heading is .415 rad. The selected held
  play does not enable the older fourth finish-camera fallback. The profile is
  `reports/out-scrimmage/left-turn-20261113/overhead-pass-linux/artifacts/out/profile-07.json`.
- Slant configures entry (1.1, .45, 1.28) aimed at (4.85, 2.45, .85), middle
  (4.2, 1.35, 1.36) aimed at (8, 3.2, .84), finish (8.8, 3.1, 1.4) aimed at
  (13.1, 3.8, .81), break (-1.4, -.60, 1.35) aimed at (.9, .82, .79), and
  pass (6, -4, 4.5) aimed at (8, -1, 2.5). Its declared stem heading is
  42.742662976 degrees. The profile is
  `reports/aero-amp-slant/retained-20261113/overhead-pass-profile-07.json`.

The profiles and frozen controller sources are authoritative for the exact
camera selection schedule. Localization freshness is at most 140 ms for route
events in these probes; the grip applies its own timestamp and tracking gates.
They do not implement a qualified hardware fail-stop: missing localization can
leave the established IMU/encoder motion command running, and missing ball
measurements prevent or delay a new visual grip trigger. After triggering, the
calibrated grip phases proceed without simulator contact feedback. Extra RGB-D
streams, calibration, hand channels, abort handling and approved host gains must
be integrated and qualified before a supervised physical attempt.

All three published held catches are scripted coordination around externally
pretrained locomotion. The evaluated residual PPO policies were not promoted.
The Kimodo reference is a separate motion experiment and is not used in these
plays. No trained residual, external camera, or generated reference should be
described as hardware-tested from this simulation evidence.

## November 14 catch-form residual training

The opt-in `go-cup-catch-form-context-v2-hold-hypothesis` observation has 119
features: the prior 111 sensor features plus eight explicit catch-phase and
causal interception-context features. Both actor and critic receive this same
vector. The new form reward uses a separate encoder/IMU kinematic model to
measure actual palm-face normals and hand height relative to the shoulders.
It never consumes live simulation contacts, root/ball truth, or replay poses.
Height and early-arm penalties require a fresh, uncertainty-gated RGB-D
prediction of interception height/time; a high ball's current height is not
substituted for that prediction. Catch type and receiving plane are commands.

The experimental Go training job adds callback-only ball retention RGB-D at
(10, -2.5, 2.4), aimed at (11.5, 0, 1.4), and receiver-marker RGB-D at
(8, 0, 1.5), aimed at (12, 0, .9). These are additional required surveyed camera
streams for that experiment, not inputs already available on HIM's runner.
The original route and grip cameras remain active. The fresh zero-action
rollout matches the archived physical trace exactly; 94/110 active captures
passed the unchanged uncertainty gates, with a longest 1.08 s visual dwell.
The physical hold lasts 3.82 s, so visual coverage remains imperfect.

Out's corresponding training views are ball RGB-D (1, 8, 2.4) aimed at
(3, 5, 1.5), and rear-marker RGB-D (4, 2, 1.4) aimed at (3, 5.5, .8).
Its zero-action preflight accepts 76/91 captures with 1.28 s longest visual
dwell. Slant adds ball RGB-D (8, 1, 2.4) aimed at (11, -1.5, 1.5), and
marker RGB-D (8, 1, 1.5) aimed at (11, -1.5, .8); its preflight accepts
51/90 captures with .44 s longest visual dwell. Both retain the selected
route/grip views and reproduce every archived physical trace/pose array.
These extra views and their intermittent coverage are experiment requirements,
not established capabilities of a physical G1.

A visually established hold hypothesis exempts post-catch reorientation from
the receive-form penalty through an occlusion. It does not preserve positive
catch reward: missing/stale observations still clear dwell, settle visual
potential and prevent the terminal retention bonus. Fresh visible ball escape
clears the form hypothesis. Its sensor state is delayed by one 20 ms control
tick. See `reports/catch-form-20261114/` for source hashes, calibrated axes,
engineering tolerances and training receipts. No learned result is promoted
until independent physical and form evaluation supports it.

## November 14 overnight native-hand experiments

The experimental `probe_amp_native_catch.py` branch adds fourteen native Aero
actuator encoder channels, seven per hand: index, middle, ring and pinky cable
lengths, thumb opposition, and the two thumb flexion cable lengths. These are
measured actuator channels, not the simulator's individual finger joint angles
or contact forces. `NativeCupGrip` validates and records them; it does not infer
tactile contact from them. The physical Aero driver and these channels remain
unverified on HIM's G1.

In the running experiment, a declared route phase at 1.6 seconds enables the
native ready posture. The explicit initial hand closure is matched between the
no-ball calibration and the passing trial. The grasp transition still requires
the surveyed RGB-D gate. Machine release timing is used by the simulated throw
fixture, not to activate this branch's native ready posture. Older lower-cup
screening trials used a commanded-release activation; those traces must retain
that distinct input declaration.

The frozen AMP policy receives measured IMU and body encoders. Its arm encoder
positions and velocities are centered on the current commanded upper-body
offset, while the physical controller receives the actual commanded arm target.
This is a deterministic observation transformation, not newly learned running.
Private encoder/IMU forward kinematics measure palm orientation and height.
Exact pelvis, ball, contact, clearance and replay pose arrays are confined to
independent evaluation and rendering. Offline no-ball traces set fixed throw
fixtures; they are never streamed back as runtime target tracking.

These experiments have not replaced the published held plays. The new
anatomical evaluator also checks impact penetration, inter-hand overlap, actual
palm-face orientation and excluded support. A long hold alone does not qualify
as a realistic lower catch under that evaluator.

The separate `native-cup-face-normal-169-v1` PPO experiment observes 169
features and controls 20 bounded residuals: fourteen body-arm targets and six
native hand groups (four fingers together, thumb opposition, and both thumb
flexors together, per hand). Native encoders are normalized using the declared
actuator strokes; unobserved finger joints and contact forces are excluded.
Actor, critic and reward use the same sensor boundary. Ball-to-cup position
and velocity are expressed in a yaw-aligned receiver frame with world Z up.
Receiver XY covariance and independently estimated Z uncertainty are added
to ball covariance, rotated into that frame, and include a 1 cm FK uncertainty
floor. Receiver-pose yaw variance is not treated as a Z-position variance.
Missing or stale images clear rewarded dwell and settle the visual potential;
an acquired form hypothesis does not prove continuing physical support.

That native PPO experiment uses ball RGB-D at (6, -2.5, 2.4) aimed at
(7, 0, 1), and marker RGB-D at (3.5, 0, 1.4) aimed at (6.5, 0, .9), in
addition to the existing measured-view selection. It currently preserves the
protected baseline's **commanded-release preload event**, distinct from the
route-stage activation in the running branch. A physical machine event link
has not been integrated. Recorded requested native targets enter the official
driver's rate limiter; they are not telemetry of its post-limiter controls.

The wrist-only running experiments optionally steer from a surveyed marker
pose no older than 140 ms. Heading correction is limited to .2 radians and
lateral velocity command to .2 m/s; missing pose produces zero correction.
They can relocate the finish RGB-D camera from X=5 to X=7 or X=8 m, still at
(X, 0, 1.5), aimed at (X+5, 0, .9), and reuse the pass view for marker fallback.
Each archived report declares its actual camera surveys and selection option.
These are additional experimental installation requirements, not verified
camera coverage on HIM's robot.

## November 15 observer ablation and replay diagnostics

The opt-in `AssociatedRearMarkerLocalizer` uses the same rendered RGB-D and
survey packets as `RearMarkerLocalizer`. It associates blue/yellow components
using the known 0.22 m marker-rig spacing (±0.045 m) and requires a unique
compatible pair. Ambiguity produces no pose. It does not read robot or ball
truth, contacts, pixel segmentation labels, or replay poses. The original
localizer and frozen policies are unchanged.

The Go observer ablation additionally requires fixed cameras at world
(10, −2.5, 2.4) m aimed at (11, 0, 1) m, and (8.5, 0, 1.4) m aimed at
(11, 0, 0.9) m. These are external surveyed cameras, not active head tracking
or demonstrated onboard sensing. They have not been installed on hardware.
The association experiment changes only the native residual observer; the
protected route/grip controller retains its original observation path.

`reports/perception-pair-benchmark-20261115/` compares original and associated
poses on identical image/depth/survey packets from two archived Go plays.
Replay state positions the renderer and supplies offline error labels; it
never enters localization. Hidden-marker and hidden-ball conditions alter
visual geometry only. Timestamp-age tests exercise freshness rejection on
fixed captures; they do not establish delayed-stream tracking robustness.
Neither improved observation coverage nor visual dwell establishes physical
hand support. The physical evaluator remains separate.
