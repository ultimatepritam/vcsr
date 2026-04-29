# Best Current Verifier

This directory records the verifier checkpoint that should be treated as the
current default artifact for downstream VCSR experiments.

The source-of-truth training run remains in its original experiment directory.
This folder exists to provide a stable reference point for future scripts and
evaluation runs.

## Selected Run

- Run: `results/verifier/ranking_aligned_round4/retrain_from_round3_focused`
- Checkpoint: `results/verifier/ranking_aligned_round4/retrain_from_round3_focused/best_model/model.pt`
- Metadata: `results/verifier/best_current/selection.yaml`
- Selection basis:
  - improved over round 3 on replay against the frozen held-out candidate pool
  - improved over round 3 on the fresh single-seed held-out run at both `K=4` and `K=8`
  - won the repeated fresh held-out comparison at `K=8` and roughly tied at `K=4`
  - promoted as the project default downstream verifier for the next VCSR phase

## Related Artifacts

- Training history:
  [train_history.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round4/retrain_from_round3_focused/train_history.json)
- Validation metrics:
  [val_metrics.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round4/retrain_from_round3_focused/val_metrics.json)
- Clean calibration report:
  [calibration_report.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round4/retrain_from_round3_focused/calibration_report.json)
- Replay against round 3 on the frozen held-out candidate pool:
  [replay_summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_round3_holdout_eval/replay_compare_round3_vs_round4_focused/replay_summary.md)
- Repeated fresh held-out comparison against round 3:
  [comparison_summary.md](/e:/Engineering/vcsr/results/vcsr/multiseed_holdout_compare/comparison_summary.md)

## Note

The model weights are not duplicated here. This folder is a stable pointer and
metadata location, not a second copy of the artifact.
