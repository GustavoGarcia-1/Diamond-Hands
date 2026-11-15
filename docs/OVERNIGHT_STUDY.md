# November 15 receiver experiments

Diamond Hands separates route execution, brief receiving and retained possession. The newer plays are walking receives in a declared simulation model. Neither a long contact interval nor a visually convincing ending establishes running, legitimate hand support or hardware transfer.

## Persistent Go feedback

Eight finger tendon targets limit their requested closing error to 1.5 millimetres after the existing visual cue. Six thumb channels, force caps and target slew limits remain unchanged. This is encoder feedback, not contact sensing or a learned grip controller. The ceiling constrains a sampled position error and is not a continuous force guarantee.

On fifty paired development throws, retained catches improve from 4 to 31. A separately frozen fifty throw confirmation gives 2 versus 34, with 32 gains and no paired losses. The persistent proportion is 68%, with a Wilson 95% interval of 54.2% to 79.2%; the baseline is 4%, with interval 1.1% to 13.5%. The paired difference is 64 percentage points, with a declared bootstrap interval of 50 to 78 points. These intervals concern throw variation under fixed controllers, not training seeds or hardware uncertainty. Confirmation seed 2026111501 is consumed and excluded from further tuning.

The selected six metre Go example maintains 5.663 seconds of clean bilateral control through its eleven second finish. It is a development example, not a confirmation sample. A separate ten yard fixture receives at 9.970 yards of surveyed downfield depth, with a longest clean interval of 1.645 seconds and possession again at the finish. Both methods pass two of seven fixed conditions in that fixture. The six metre confirmation rate must not be transferred to the ten yard result.

## Why Out does not transfer wholesale

The protected original Out retains 9 of 50 development throws. A synchronized entry and regulation bundle retains 8, gaining two and losing three. It fails the promotion gate.

A read only audit of 150 archived plays finds 32 original Out cases with torso support before hand contact. Original Go instead has 28 cases that achieve at least 0.75 seconds of clean bilateral control and later lose admissible endpoint possession or support. Persistent Go regulation removes that latter failure category, while also helping some early acquisitions and worsening eight previously acquired cases below the duration gate. Differences across route fixtures are observational evidence, not a causal ranking of difficulty.

Copying Out preparation into Slant does not establish a general solution. Stronger preload and a shallower throw fail to improve its short receive. The latter also changes energy and cue timing, so it is not an isolated angle experiment. The complete negative outcomes remain in the originating archive.

## Slant learning and late grip closure

Removing a 160 millisecond delay after the same visual cue produces a scripted Slant baseline with 171 milliseconds of clean bilateral control. The final 45 minute PPO model extends that nominal interval to 873 milliseconds. Without another change, the ball later rests against mounts and wrist bodies and fails endpoint possession. No floor contact occurs within the learned nominal play.

Offline contact and encoder analysis shows continued finger shortening during wristward ball migration. A separate scripted rule latches the current requested cable positions 1.30 seconds after the same visual cue and limits additional shortening. The existing encoder regulator still applies; opening requests remain possible. The arms, thumbs, model, actuator limits and learned checkpoint remain unchanged. This combination produces 1.995 seconds of clean nominal control through the endpoint without excluded support.

All four combinations are evaluated on seven existing unique development conditions. Retained outcomes are 0 for scripted preparation, 1 for preparation with the late closing limit, 0 for PPO alone and 4 for PPO with that limit. The limit alone succeeds for the higher throw; learned arms add three retained conditions when the same limit is present. These results support a conditional interaction in the tested cases, not a claim that learning is always necessary or sufficient. All thirty archived full and short scores, two anchors, fourteen prefixes, physical model arrays, thumb commands and arm residuals pass independent checks.

A separately sealed confirmation reverses the development ordering: scripted arms with the closing limit retain **19 of 50**, compared with **12 of 50** for learned arms with the same limit. The Wilson 95% intervals are 25.86% to 51.85% and 14.30% to 37.41%. There are six paired gains, thirteen losses, six successes with both and twenty five failures with both. The learned minus scripted difference is minus 14 percentage points; its predeclared paired bootstrap 95% interval spans minus 30 to plus 2. No conditional learning gain is established, and the learned controller is not promoted. All hundred plays and two exact anchors complete and pass independent anatomical scoring, model, sensor and command limit audits. Every short receive classification matches; six derived floating point rows differ by at most 6.8 × 10⁻¹⁵, recorded in the numeric validation erratum. No scoring rule changed. Confirmation seed 2026111527 is consumed and excluded from retuning.

The nominal first 20 millisecond force peak decreases from 1274.43 N before learning to 650.14 N after learning, but scalar impulse increases from 2.10870 to 2.23069 N s. PPO alone increases self overlap from 2.104 to 4.774 mm; adding the closing limit restores the nominal maximum to 2.104 mm. One combined perturbation still exceeds the unchanged 5 mm self overlap limit. Actual caps do not imply safe collision impact.

The exact final arm checkpoint is included as `evidence/nov15/slant-final-ppo.zip`, SHA256 `58faa28a49460d1231b0672d0d5001451fcf9ce70b077a4f266c170a59559fd1`. Its frozen experiment adapter is included alongside it. Running that adapter still requires the matching route driver, configuration and assets from the restored source bundle.

The new learning objective uses camera estimates, IMU, body and native actuator encoders and known commands for both actor and critic. Positive alignment credit stops at a conservative camera predicted arrival deadline. Contacts, exact ball and root state, physical success and postcontact possession bonuses do not enter the reward. See [the full equations and protocol](precontact-ppo.md) and [sensor requirements](SENSOR_CONTRACT.md). A generic frame label does not guarantee physical camera identity; that provenance limitation is preserved with the frozen checkpoint.

## Missed catches

A dedicated camera observes a low ball in the predicted foot corridor after a failed receive. Two fresh observations trigger a 250 millisecond cancellation of the velocity goal and prevent one natural trip in the current scene. Six nearby cases are already nonfalls and remain unchanged, as does a successful catch control. This is one affected condition, not seven successful fall recoveries. Later IMU triggered responses have adverse impact tradeoffs and are not promoted.

## Reproducible artifacts and verification

The verified current submission package is `verified-current-study-20261115-r5/project.zip`, SHA256 `f6e20c2093c4ea33dd1e2618eaf822b4c9bc4ed902a9e2209254b24d508fe2be`. Its accompanying two play MP4 has SHA256 `861d0dd5fbafc1d2fc090c47ec02b6f97a838fa2dca03cfe719ab23408d5466e`. Their combined size is 86,076,903 bytes. The ZIP restores seven frozen source snapshots by content hash; all 2,270 frozen regular members were verified. Its two primary Go plays include dense raw evidence. Secondary dense traces and the hundred raw trajectories in each Go and Slant confirmation remain in separately archived artifacts. The Git repository and compact summaries alone cannot independently rescore every historical trajectory.

The primary reel contains all 552 decoded frames, joining only between its two complete plays. Source poses and timestamps are unchanged. External AMP assets require the pinned upstream revision and checksum; no license grant for those weights is inferred. WBC and Aero simulator notices remain with the source snapshots. No ARM64 image or builder safety qualification is supplied.

The curated repository passes 235 tests in its full local environment. The expanded CPU selection passes 79 tests without PyTorch or pretrained assets. Gymnasium is explicitly included in the test extra because the sensor audit imports the existing RL return ledger. Frozen experiment dependencies remain unchanged. See [source hashes and compact outcomes](../evidence/nov15/results.json).
