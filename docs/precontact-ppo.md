# Precontact arm learning

The November 15 experiment learns a bounded correction to the receiving arms.
The route controller, visual reach cue, nominal arm motion, finger schedule and
encoder based grip regulation remain fixed. A new PPO policy is initialized
with an exactly zero deterministic output. Its actor and critic receive the
same 169 sensor and command features. They do not receive simulator contacts,
the exact football state, the exact root state or the physical catch score.

The observation includes 29 body joint positions and velocities, IMU gravity
and angular velocity, 14 native actuator encoders, known native targets,
calibrated rigid palm geometry and a camera based trajectory estimate. The
estimated trajectory is fitted from distinct RGB D captures in the surveyed
field frame. Native tendon encoders do not reveal every finger joint angle.
The rigid palm model therefore supplies an alignment target, rather than a
complete estimate of the grasp.

PPO requests 14 normalized arm actions at 25 Hz. The 50 Hz controller projects
their delivered residuals into the intersection of the joint ranges, a
0.18 radian magnitude bound and a 1.5 radian per second residual rate bound.
All native actuator requests are unchanged by the learned policy. The
original body and native actuator effort limits remain active.

## Reward and arrival boundary

Let the camera estimate a descending crossing of the commanded receiving
plane at time \(\hat T_k\), with estimated temporal standard deviation
\(\sigma_{T,k}\). A committed deadline can only move earlier:

\[
D_k=\min\{D_{k-1},\hat T_k-2\sigma_{T,k}\}.
\]

Once the sensor clock reaches this deadline, the positive alignment reward
closes for the rest of the episode. Later trajectory estimates cannot reopen
it. The factor of two is an engineering margin; the estimated uncertainty has
not been calibrated as a hardware confidence guarantee. Accepted arrival
estimates require a fresh incoming trajectory with temporal uncertainty no
greater than 60 milliseconds. This deadline does not use the measured contact
time from the simulator.

The alignment error \(e\) is the predicted football position at the receiving
plane, expressed relative to the current camera estimate of receiver position
and the current encoder based palm target. It does not forecast future receiver
translation. The palm target uses a fixed radius proxy of 0.1135 metres. It does
not measure football orientation or certify finite finger support.

For marginal position uncertainty \(\sigma_i\) and fixed scales
\(s=(0.12,0.08,0.10)\) metres, the shaping term is

\[
\phi(e,\sigma)=
\prod_{i=1}^{3}\frac{s_i}{\sqrt{s_i^2+\sigma_i^2}}
\exp\left[-\frac12\sum_{i=1}^{3}
\frac{e_i^2}{s_i^2+\sigma_i^2}\right].
\]

This is a Gaussian expectation under a diagonal uncertainty approximation.
Cross axis covariances are recorded in the diagnostic data but the reward
uses the three marginal variances. The frozen run metadata incorrectly calls
this full covariance; the original metadata is retained with this correction.
At exact alignment, larger uncertainty attenuates the reward. Away from exact
alignment, it changes the integrated expectation and need not lower it
monotonically. It is not silently replaced by a smaller fitted confidence. Only a distinct,
fresh capture can receive positive credit. Repeated images can supply a cached
actor feature for at most 80 milliseconds but cannot pay twice.

At each control step the reward ledger combines fresh alignment credit with
the existing receiving form cost, an upright cost and action regularization:

\[
r_k=25\,\Delta t_{\rm capture}\,\phi(e_k,\sigma_k)
\mathbf 1_{\rm fresh,\ enabled,\ t_k<D_k}
-\Delta t\left[c_{\rm form}
+2\max(0,0.9-u)
+0.05\,\overline{a^2}
+0.02\,\overline{\dot\delta^2}\right].
\]

Here \(u\) is the IMU upright cosine, \(a\) is the normalized arm request and
\(\delta\) is the delivered arm residual. The last delivered residual is held
after the deadline. Form and action costs continue, with no exemption based on
an inferred hold. There is no visual dwell milestone or terminal possession
bonus. The first reset reward is recorded separately from the return delivered
to PPO.

## Experiment controls

Each route first reproduces its archived zero action baseline, including the
physical pose and command arrays, model identity and complete anatomical
score. The initial actor head is exactly zero. Training uses a fresh seed,
two environments, a one hour wall budget, 64 step rollouts, four optimization
epochs, a learning rate of 0.0001 and an initial log standard deviation of
minus two. Declared small throw perturbations are sampled independently for
training. The commanded receiving plane remains fixed. Out perturbations use
its surveyed lateral axis and event relative release delay.

The Go R2 run has sparse receiver localization near contact. A separate Go R3
run adds one surveyed camera at `(4.0, 1.8, 1.6)` metres looking toward
`(6.3, -0.1, 0.95)`. Its zero replay reproduces the physical baseline exactly.
The camera can provide receiver and football observations through the same
documented view selector. The number of accepted enabled captures before the
deadline increases from two to eleven in the nominal diagnostic. This is
evidence of better observation coverage, not improved catch success. R2 and
R3 share a GPU during part of training, so their wall times do not establish a
controlled comparison of training efficiency.

The final checkpoints receive paired full play physical evaluations. Those
evaluations check retention, endpoint possession, support, penetration, form,
balance, actuator limits and ball freedom. They remain separate from the
training reward. Fixed development cases are not a population success rate.
The consumed Go confirmation seed `2026111501` is excluded from new training
and controller selection.

The earlier visual possession reward could give credit during excluded mount
support. Its checkpoints and negative physical results remain archived. This
new objective is deliberately described as precontact alignment. Whether it
improves catching is an empirical question answered by the physical evaluation.
Camera placement, hand transmission and material parameters still require
hardware validation.

## Receiver translation experiment

The first completed hour did not establish learned improvement. Go with the
original camera passed four of seven fixed throws after learning, compared
with six before learning. Out passed the same three of seven conditions before
and after learning. Slant passed none of its eight declared conditions under
either policy. Two Slant release commands mapped to the same physical control
tick, so there were seven unique physical conditions. These are fixed
development comparisons, not population estimates. The final partial PPO
rollout contributes collected steps but is not an additional optimizer update.

The first alignment objective compares a future ball position with a current
receiver position. Independent analysis of Out camera logs found a median
omitted horizontal receiver translation of 0.287 metres before arrival. At one
capture, the omission was 0.390 metres forward and 0.277 metres laterally in the
receiver frame. This motivates a separately versioned transport experiment;
it does not establish that transport will improve physical catching.

For two past receiver positions expressed in the surveyed field, the estimator uses

\[
\hat v_{xy,k}=\frac{\hat r_{xy,k}-\hat r_{xy,j}}{t_k-t_j},
\qquad h_k=\max(0,\hat T_k-t_k),
\qquad e'_k=e_k-Y_k[\hat v_{xy,k}h_k,0]^\top.
\]

The history contains at least four distinct camera poses and spans 0.16 to
0.40 seconds. It may contain gaps; it does not require uniformly spaced
captures. The latest pose must be fresh and synchronized with its image.
Different optical frame labels maintain separate histories. An audit found that
the simulation capture adapter reuses the label `camera_optical` for multiple
physical cameras, so this does not guarantee separation by physical camera.
The frozen runs are preserved, and their actual selected camera histories are
being audited. The earlier single camera description is withdrawn. Estimates
above 3 metres per second are rejected. There is no substitution of simulator
velocity or commanded velocity when camera history is unavailable.

Endpoint uncertainty is combined without assuming independent pose errors:

\[
\sigma_{v,i}=\frac{\sigma_{r_k,i}+\sigma_{r_j,i}}{t_k-t_j},
\quad d_i=h_k\sigma_{v,i}+\tfrac12 a_0h_k^2
+|\hat v_i|\sigma_T,
\quad \sigma'_k=\sigma_k+|Y_{k,:,xy}|d,
\qquad a_0=2\;\mathrm{m/s^2}.
\]

The acceleration term is a fixed engineering dispersion allowance, not a
verified hardware acceleration bound. The same Gaussian shaping uses
\(e'_k,\sigma'_k\). Velocity features now express ball motion relative to
the estimated receiver translation. Thus this experiment changes the target,
relative velocity features and uncertainty together; it is not an isolated
mean correction. It does not forecast future arm motion, yaw, finger contact
or possession. Its uncertainty remains uncalibrated.

The initial actor is again exactly zero. All archived physical arrays, the
complete score and every compiled numeric model array must match before
optimization. At least four fresh transported captures must provide positive
credit before the unchanged absorbing deadline. Separate 45 minute training
jobs and paired full play evaluations preserve the previous runs and their
negative results. The new checkpoint schema prevents accidental loading into
the original observation definition.

## Completed physical evaluations on November 15

The transported target did not improve retained catches for Go or Out. Go
passed five of seven fixed conditions after training, compared with six before
training. Its nominal uninterrupted interval decreased from 5.663 to 5.449
seconds. Out passed the same three of seven conditions, and all three successful
intervals became shorter. These models are archived without replacing the
stronger scripted controllers. Go failed the declared screening gate for a
broader development evaluation, so that additional job was not launched.

The final Slant checkpoint is the first of these runs to improve brief physical
receiving outcomes. Independently rescored full play outputs show a nominal clean interval
of 0.873 seconds versus 0.171 seconds, and four versus two short receives among
seven unique fixed conditions. The older short receive check restricts its
measurement to the first half second after fingertip contact; its nominal
interval is 0.293 seconds, not the full 0.873 seconds. All these Slant trials without the later scripted closing floor still
fail endpoint retention. All sixteen original full and short scores recompute exactly. Physical model
arrays match after explicit normalization of relocated asset paths; raw compiled
file byte identity is not claimed. The nominal interhand
penetration increases from 2.104 to 4.774 millimetres, and the corresponding
peak self contact force increases from 11.60 to 147.19 newtons. Longer contact
therefore does not establish gentler or hardware suitable behavior. The first
20 millisecond force peak decreases from 1274.43 to 650.14 newtons, while its
scalar impulse increases from 2.10870 to 2.23069 newton seconds. At the final
quarter second, allowed hand support is zero and mounts and wrist bodies
support the ball. There is no floor contact during the 9.5 second play. This
is loss of admissible bilateral control, not a demonstrated floor drop.

The Go, Out and Slant transported runs collected 1794, 3892 and 2714 steps,
respectively. Completed PPO rollouts contain 1792, 3840 and 2688 steps, with
56, 120 and 84 optimizer epochs. Final incomplete rollouts were saved but did
not receive optimization. The nominal 45 minute budgets are wall time budgets,
not equal sample budgets. The final checkpoint was specified in advance by
the budget; these development outcomes did not select among checkpoints.

Physical camera provenance was also audited. All recorded Go evaluation
translation endpoints use the retention pose camera. The nominal Out witness
uses its late pose camera. Three of ten positive nominal Slant histories mix
entry and pass camera estimates, contributing about 14.48 percent of its
transport alignment credit. These are causal estimates in a common surveyed
frame. They do not establish a calibrated cross camera error model, and the
generic frame label still cannot enforce a single camera history in unobserved
training episodes. The frozen implementation and its provenance limitation
are retained together.

## Scripted late closing floor and fresh confirmation

The final transported Slant arm checkpoint is unchanged in the subsequent
native hand experiment. At the first control sample at or after the causal
visual receive cue plus 1.30 seconds, a scripted rule latches eight requested
finger cable positions. Each final request is the larger of the base request
and its latch. Larger positions open the cables: additional closing is limited,
while opening remains allowed. The six thumb requests, learned arm residuals,
original gains, target slew and actuator caps are unchanged. The same sampled
1.5 millimetre encoder closing error rule remains active. The floor uses only
the visual event, controller clock, native encoders and known requests. It has
no contact, physical force, exact ball or exact root input. It is a scripted
component combined with learned arms, not an additional PPO policy.

All four methods were compared on the same seven unique fixed development
conditions. Retained catches were zero for zero arm actions without the floor,
one for zero actions with the floor, zero for learned actions without the
floor and four for learned actions with the floor. Exact anchor and physical
prefix comparisons preserve the earlier 0.171 and 0.873 second nominal
results. The combined nominal interval is 1.995 seconds through the endpoint.
These development outcomes suggested an interaction, not that learning alone
was necessary. All outcomes and failed conditions remain archived.

A separately frozen paired confirmation then consumed fresh seed
`2026111527`. Fifty IID requested throws independently perturb world Y aim
and height by uniform plus or minus one centimetre and release by uniform
plus or minus ten milliseconds. The original 20 millisecond machine release
cadence remains active. Both methods share the floor, perception, checkpoint
configuration and declared engineering model. No confirmation outcome was
used to tune a method.

Zero arm actions plus the floor retained 19 of 50 throws; learned arms plus
the floor retained 12 of 50. Six throws were paired gains and thirteen were
losses, with six successes shared by both and twenty five failures shared by
both. The learned minus zero difference is minus fourteen percentage points.
A paired percentile bootstrap with twenty thousand resamples and seed
`2026111528` gives a 95 percent interval from minus thirty to plus two
percentage points. Wilson 95 percent intervals are 25.86 to 51.85 percent
for zero actions and 14.30 to 37.41 percent for learned actions. These
intervals concern only this declared jitter distribution in the fixed model.
They do not describe hardware performance or variability across training seeds.

The confirmation reverses the fixed development ordering and does not support
promoting the learned controller for reliability. All one hundred outcomes
and both exact anchors were independently rescored. Full anatomical scores
match exactly. Short receive decisions match, with only numerical differences
up to 6.8 times ten to the minus fifteen in six floating point rows. Model,
free ball, full horizon, actuator cap and sensor deadline checks pass. The
historical realistic reference flag remains false. Neither this result nor
the nominal 1.995 second interval establishes running, complete route
execution, calibrated contact physics or hardware transfer.

The sensor only actor, critic and reward boundary described here applies to
the new catching experiments. The externally pretrained locomotion policies
have a separate upstream training history. Privileged information in upstream
gait training is permitted by the project requirements; it is not represented
as having followed this catching reward contract. Deployed gait and catching
controllers must still use physically available runtime inputs.
