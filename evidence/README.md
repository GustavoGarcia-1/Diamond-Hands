# Included evidence

These small records make the claims inspectable without committing dense simulation runs. They do not replace the original traces or constitute a fresh rescore.

`study-600/protocol.json` and `scenarios.json` define the frozen 450-trial paired comparison. Independent summary `summary-450.json` reports current / earlier / exact-ball diagnostic outcomes for 50 scenarios per route. Current counts are Go 4, Out 9 and Slant 0 of 50. Methods share the declared route-specific fixture/model and scenario draws; results are simulation engineering evidence, not a general or hardware reliability rate.

`uncertainty-floor-protocol.json` and `summary-150-appendix.json` describe a **separate posthoc** 150-trial diagnostic using exact ball positions with a 1 mm covariance floor. It does not replace the original 450, simulate calibrated camera noise or establish a deployable perception upper bound. Its counts are Go 2, Out 4, Slant 0 of 50. The original near-zero-uncertainty exact-ball diagnostic interacted badly with tracker gravity assumptions; every diagnostic must retain its declared visibility/confidence contract.

`training/three-route-hour-summary.json` is a concise extraction from an independently verified full record. It retains source/artifact/checkpoint hashes, per-condition physical holds, return and outcomes. The full original record's SHA256 is retained; dense nested ledgers and local paths are omitted. The later final learners did not improve physical catches. These fixed heterogeneous conditions are not another statistical 50-trial comparison.

`training/go-reward-specificity-independent.json` preserves a particularly important counterexample. Replaying the same actions with an augmented observer improved selected sensor capture coverage to 100%, but the failed learned catch still received 10.511 return versus 8.998 for retained baseline control. Exact array/action/ledger checks passed. Coverage repair alone did not make the reward a possession detector.

`source-manifest.json` records original and repository hashes for curated source/evidence files. Concise extractions are explicitly distinguished from byte-identical copies. `verification/` records checks performed in the new repository. Git commit/tree identity identifies the final prepared snapshot.

Excluded evidence includes NPZ/full-state/contact traces, source TARs/ZIPs, MP4s, training checkpoints, remote launch receipts/session metadata and external model/asset trees. Their original archive/source digests remain in the selected records. They are preserved separately in the originating research archive; no public GitHub release or download URL has been invented. Exact historical reproduction needs those frozen inputs, as explained in [reproduction notes](../docs/REPRODUCTION.md).
