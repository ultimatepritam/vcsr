# Best Current Verifier

This directory records the verifier checkpoint that should be treated as the
current default artifact for downstream VCSR experiments.

The source-of-truth training run remains in its original experiment directory.
This folder exists to provide a stable reference point for future scripts and
evaluation runs.

## Selected Run

- Run: `results/verifier/ranking_aligned_round3/retrain_from_round2_multipool`
- Checkpoint: `results/verifier/ranking_aligned_round3/retrain_from_round2_multipool/best_model/model.pt`
- Metadata: `results/verifier/best_current/selection.yaml`
- Selection basis:
  - improved fixed-pool replay on the original 30-row pilot pool at `K=4`
  - matched the previous best `K=8` result on the original pilot pool
  - beat the round-2 baseline on the stronger 50-row replay pool at both `K=4` and `K=8`
  - frozen as the project baseline for the next end-to-end VCSR phase

## Related Artifacts

- Training history:
  [train_history.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round3/retrain_from_round2_multipool/train_history.json)
- Validation metrics:
  [val_metrics.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round3/retrain_from_round2_multipool/val_metrics.json)
- Clean calibration report:
  [calibration_report.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round3/retrain_from_round2_multipool/calibration_report.json)
- Fixed-pool replay on original pilot pool against round 2:
  [replay_summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/replay_compare_round2_vs_round3_multipool/replay_summary.md)
- Replay on stronger 50-row round-2 pool against round 2:
  [replay_summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_ranking_round2_pool/replay_compare_round2_vs_round3_multipool/replay_summary.md)

## Note

The model weights are not duplicated here. This folder is a stable pointer and
metadata location, not a second copy of the artifact.
