# Frozen-trajectory Go reward specificity audit

The coverage fix does **not** make the unchanged reward rank physical retention correctly on this paired development case. The physically retained scripted zero trajectory scores 8.9981 with the augmented observer, below the failed learned trajectory's 10.5114. No further PPO job was launched.

Four offline passes completed in 37.42 s on the existing idle Slant GPU, 2026-11-15 08:08:21–08:08:58 UTC, exit 0. Both original-observer passes reproduced every saved timestamp/raw observation/normalized observation/action/limiter array, complete sensor sample JSON, reward return, and terminal bonus exactly. These gates passed before the augmented passes. All four component ledgers close to returned reward within 1e-10. No physical trajectory or policy action changed.

`results.json` is the machine-readable result, including component totals, milestone timestamps and separately joined offline truth labels. `site-summary.json` is a compact figure input. `linux/artifacts/*/reward-events.json` contains passive original reward-line deltas; `offline-event-truth.json` is a later evaluation-only join.

## Paired observations and physical labels

- Scripted zero: archived clean bilateral hold 3.537 s; final bilateral fraction 1.0. Original observer return 1.152858; augmented 8.998118.
- Final learned: archived clean bilateral hold 0.131 s; final bilateral fraction 0.0; first excluded support 5.656 s. Original return 7.802738; augmented 10.511398.

The original reward preferred learned by 6.649880; augmentation narrows the difference to 1.513281 but preserves the wrong ordering. Both augmented passes qualify all 158 enabled fresh captures, compared with original 84/158 and 99/158. This coverage statement concerns the enabled reward interval, not all initial approach frames.

Augmented zero maximum visual dwell is 1.52 s, final 0.04 s; augmented learned maximum 1.88 s, final 0.08 s. Terminal bonuses are zero in all four passes. Both augmented trajectories receive one +2 milestone at 6.400 s. Offline truth labels that instant as clean bilateral for zero and bilateral **with excluded support** for learned. The original learned's +2 milestones at 6.040 and 7.280 s likewise coincide with excluded support; these labels were never supplied to the reward.

Augmented learned accumulates 4.68 s of positive dwell increments across reset segments, versus zero's 3.12 s. Only 0.16 s of learned increments coincide with the original scorer's clean bilateral label, versus zero's 3.04 s. Instantaneous bilateral contact accounts for 4.12 s and 3.04 s respectively. No positive dwell increment in either trajectory occurs without any anatomical hand force. Thus the observed failure is largely visual proximity/dwell credit during inadmissible support, rather than credit for a wholly detached ball. Original clean labels permanently terminate after first excluded support; the separate instantaneous labels preserve that distinction.

The augmented learned-minus-zero component difference is +1.56 dwell increments, −0.03049 dense proximity/motion integral, −0.02600 potential increments, +0.02713 potential settling, −0.00629 form cost, and −0.01108 action/rate cost; milestone difference is zero. The initial reset ledger adjustment is identical. Repeated positive dwell accumulation explains most of the remaining reward inversion.

The source scorer's `realistic_lower_catch` flag is false for both trajectories. Both fail reference-physics verification for the engineered model; learned also fails retention and excluded-support criteria. Physical retention and the original flag are explicitly separate. This paired audit diagnoses reward specificity on known trajectories. It is neither a success rate, heldout evaluation, nor proof that no causal observation/reward can distinguish admissible support.

## Reconstruction and boundary

The script hard-disables `mujoco.mj_step`; it never loads or evaluates a learned policy. It restores archived qpos/qvel into an offline scene, calls `mj_forward` to reconstruct simulated sensor channels, and renders RGB-D at the original camera timestamps. Full pose is consumed only by this offline rendering/sensor adapter. Reward receives RGB-D/calibrated survey, 29 joint position/velocity channels, IMU quaternion/gyro, 14 archived native encoders, 14 archived nominal native targets, and archived known enabled/action commands. It gets a separate static FK model and no exact ball/root/contact/physical-score channel.

Original plant source calls `mj_forward` after every 20 ms interval. The present-time archived q/dq channels match raw packets exactly; a 1 ms shift fails and was rejected before launch. Initial qpos comes from the archived t0 view; initial zero velocity follows the sealed reset source. All 20 ms native encoders/nominal targets are logged in `trace.npz`. The input body-target argument affects only discarded control outputs, never reward or observer; an encoder-position placeholder supplies this unused argument. No returned command is executed.

Original route marker localization remains unchanged. Augmentation uses precisely the prior associated-marker class and two extra fixed surveyed views, with the same confidence gates, acquisition rules, 0.75/2 s milestone thresholds, action costs and terminal rules. The altered raw/normalized observations in augmented passes are expected; archived actions and their limited residuals remain exact. Passive reward-assignment instrumentation records deltas and does not alter arithmetic or gates. The full exact-original replay is a stronger validation than merely comparing aggregate return.

Truth files are read only by local `analyze.py` after all replays complete. The raw runtime reward source has no truth-label join.

## Protocol and reproduction

`protocol.json` and source hashes were frozen before dispatch. Maximum four passes / 600 s. No new session, no shared API edit, no heldout50 use, no PPO, no reward adjustment, no rank-driven repair.

```sh
/Users/gustavogarcia/projects/simulation/.venv/bin/python reports/catch-repair-20261114/post-study/go-reward-specificity-r1/launch.py
/Users/gustavogarcia/projects/simulation/.venv/bin/python scripts/antioch_receiver.py status --receipt reports/catch-repair-20261114/post-study/go-reward-specificity-r1/receipt.json
/Users/gustavogarcia/projects/simulation/.venv/bin/python scripts/antioch_receiver.py retrieve --receipt reports/catch-repair-20261114/post-study/go-reward-specificity-r1/receipt.json
python reports/catch-repair-20261114/post-study/go-reward-specificity-r1/analyze.py
```

Launch creates a fresh root and refuses overwrite. Receipt includes the exact remote command, source archive identity, session and GPU. Source SHA256 `e90ef487dfdd2f45e176c011cf85e84613b11a40230273550732f1d9ee6f0b45`; runtime artifact SHA256 `f661fea568ac3b731020e1fdbaf8767ec4fc668ac041645ae2eb45528ef2d590`. Existing immutable mesh assets are referenced from the completed physics-study root, with asset/model source hashes preserved in that study. No integration into shared runtime was performed. Independent reviewer `/root/validation_r2` reviewed the protocol and driver before dispatch and is auditing the retrieved outputs.

A further learning attempt should wait for a predeclared sensor-only specificity test that can distinguish the observed inadmissible support mechanism, rather than optimizing more visual dwell. This audit does not authorize or select a replacement reward; no parameters were changed here.
