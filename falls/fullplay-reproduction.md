# Full receiver-play fall responses: reproduction and limits

This is a new matched reconstruction of the historical `offset-fixed-controlled` failed Go fixture. Do not call it an exact replay of the older archive: the first nonzero qpos difference is2.43e−8 at40ms and contact trajectories eventually diverge. Within this new study every response matches the fresh baseline exactly in recorded qpos, motion and ball contacts before its own IMU trigger. No external push, initial tilt, state overwrite or altered throw is introduced. The actual machine origin is(0,3.45,1.40)m; this is not the current2yard formation or a newly qualified straight Go.

The original20ms contact model, existing actuator limits and native hand gains are retained. Most variants retain body Kp/Kd; explicitly named half/quarter-stiffness variants reduce body Kp over150ms while retaining Kd. No torque limit increases. Passive ground impact forces can exceed what motor torque limits alone suggest. No hardware or contact-material calibration is established.

## Commands

Each launch source archive is a Python/reference overlay, not a standalone project. First extract the existing `../experiments/go-source.tar.gz` base to a new `SOURCE` folder, then extract the selected overlay into the SAME folder and run there using separately fetched matching Aero/AMP assets. The reused Go base contains58 non-Python vendored XML/mesh/WBC-weight/notice files, each verified against the actual visual-response worker; `runtime-assets-provenance.json` records every hash. All58 files in that downloadable Go base match the actual fall worker. Apply the selected overlay LAST. A separate minimal asset payload is retained only in the repository for packaging, not duplicated on the site. Source overlays include `src`, `scripts`, and the exact reference report at `reports/quarterback-pass/offset-fixed-controlled/report.json`. Each output path must be new. Set `PYTHONPATH=src:scripts` and `MUJOCO_GL=egl` on the original headless Linux worker. The exact worker argv, source archive SHA and source file manifest are recorded in the respective receipt.

```sh
mkdir SOURCE
tar -xzf go-source.tar.gz -C SOURCE
tar -xzf SELECTED_OVERLAY.tar.gz -C SOURCE
cd SOURCE
export PYTHONPATH=src:scripts
export MUJOCO_GL=egl
# The selected overlay must correspond to each command below; use separate SOURCE directories.
# Fresh baseline, startup-abort false-positive, and deep tuck (r3 source):
python scripts/probe_receiver_fall.py --source AERO --amp-source AMP --reference reports/quarterback-pass/offset-fixed-controlled/report.json --output NEW_BASE
# Half/quarter body-Kp variants (low-gains source):
python scripts/probe_receiver_fall.py --source AERO --amp-source AMP --reference reports/quarterback-pass/offset-fixed-controlled/report.json --output NEW_LOW --low-gains
# Late balance controllers (late-balance source):
python scripts/probe_receiver_fall.py --source AERO --amp-source AMP --reference reports/quarterback-pass/offset-fixed-controlled/report.json --output NEW_BALANCE --late-balance
# Featured visually armed early balance and shallow brace (visual source):
python scripts/probe_receiver_fall.py --source AERO --amp-source AMP --reference reports/quarterback-pass/offset-fixed-controlled/report.json --output NEW_VISUAL --visual-armed
# Same shallow brace, delayed IMU thresholds (late-brace source):
python scripts/probe_receiver_fall.py --source AERO --amp-source AMP --reference reports/quarterback-pass/offset-fixed-controlled/report.json --output NEW_LATE_BRACE --visual-late-brace
```

Archive/receipt prefixes: `fullplay-r3`, `fullplay-low-gains`, `fullplay-late-balance`, `fullplay-visual`, `fullplay-late-brace`. Files use `-receipt.source.tar.gz` and `-receipt.json` suffixes. Five snapshots are distinct; use the designated frozen snapshot for each command rather than mixing later scripts with earlier controllers. Two earlier bootstrap attempts failed on missing staged reference paths before simulation and are preserved in the repository, not used as outcomes.

The copied evaluator removes only its stop-on-fall evaluation termination, permitting all12s to be observed. It never hides or falsifies IMU tilt. Floor/contact data are passive post-step evaluation. The supervisor consumes IMU, encoders and known commands. Visual arming receives the existing RGB-D-derived ready command through a zero-residual callback; it does not read ball contacts, forces, root truth, a future trajectory or a release clock. The early visual responses forfeit the catch. No new learning was performed.

## Environment and assets

Observed worker Python3.12.3/Linux x86_64, NumPy2.5.4, MuJoCo3.15.0, Torch2.11.0+cu130, ONNXRuntime1.31.0 and Stable-Baselines3 2.9.2. The shared worker includes system packages and has dependency conflicts. An observed pip freeze is not a resolver-valid clean-install lock; no fresh-machine dynamics replay is promised. CPU/CUDA/library differences can change contact outcomes. No ARM64 deployment image or builder-kit validation is demonstrated.

Aero source is the documented Menagerie revision `fb36886b3023db08c198264e1e9956ab202c3e3b`. The observed61 remote files match the project's local source; extracted folders lack git metadata, so content hashes—not a newly observed git checkout—establish the consumed files. AMP revision is documented as `386ee24bc62c8033641d65b940dc2cf1dec56545` of `zihanwang0422/legged_rl_lab`; independently fetch rather than redistribute its weights. ONNX SHA256 `17019141d56e57f53059c671c81e22d63c1dd2f7cc69a8b322c06790c00cca6f`, YAML SHA256 `3d45b9641f460eed486e2caa92feaedbfbf62d41b2434a2d0b38bb829a5319c8`. Preserve notices and separate Aero hardware-design terms.

## Offline metrics

Download a selected `CASE-forces.npz` plus `rescore_fall.py`; Python and NumPy suffice:

```sh
python rescore_fall.py CASE-forces.npz
```

The force array columns are time, head-normal, hand-normal, torso-normal, other-normal(includes soles), and total-nonfoot-normal. Motion columns are time, pelvisXYZ, upright cosine, root linearXYZ velocity, root angularXYZ velocity. All evaluation traces are1kHz. Other-nonfoot is total-nonfoot minus head/hand/torso. Maximum150ms impulse is the greatest rolling sum of nonfoot normal force×1ms, not a vector impulse or injury estimate. First150ms after initial contact is a separate window and yields a different baseline/brace comparison.

Standing requires final2s height>.65/up>.9/planar speed<.15/angular speed<.35. Settled ground instead requires continuous nonfoot floor force>.1N, linear speed<.1/angular speed<.35 over final2s. The latter is not get-up recovery. These engineering descriptions are not clinical safety limits. One throw, posthoc mechanical diagnosis and deterministic matched tests do not establish generalized protection; all failed/degraded variants remain published.
