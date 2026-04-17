# Best Current Verifier

This directory records the verifier checkpoint that should be treated as the
current default artifact for downstream VCSR experiments.

The source-of-truth training run remains in its original experiment directory.
This folder exists to provide a stable reference point for future scripts and
evaluation runs.

## Selected Run

- Run: `results/verifier/ranking_aligned_round2/retrain_from_round1`
- Checkpoint: `results/verifier/ranking_aligned_round2/retrain_from_round1/best_model/model.pt`
- Metadata: `results/verifier/best_current/selection.yaml`
- Selection basis:
  - best downstream replay result on the original fixed pilot pool
  - strongest verifier checkpoint on the newer replay-tested round-2 pool
  - frozen as the project baseline for future robustness-focused verifier work

## Related Artifacts

- Training history:
  [train_history.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round2/retrain_from_round1/train_history.json)
- Validation metrics:
  [val_metrics.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round2/retrain_from_round1/val_metrics.json)
- Clean calibration report:
  [calibration_report.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round2/retrain_from_round1/calibration_report.json)
- Fixed-pool replay on original pilot pool:
  [replay_summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/replay_compare_ranking_round2/replay_summary.md)
- Replay on newer round-2 pool:
  [replay_summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_ranking_round2_pool/replay_compare/replay_summary.md)

## Note

The model weights are not duplicated here. This folder is a stable pointer and
metadata location, not a second copy of the artifact.
