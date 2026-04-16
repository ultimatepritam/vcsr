# Best Current Verifier

This directory records the verifier checkpoint that should be treated as the
current default artifact for downstream VCSR experiments.

The source-of-truth training run remains in its original experiment directory.
This folder exists to provide a stable reference point for future scripts and
evaluation runs.

## Selected Run

- Run: `results/verifier/lr_sweep/lr_5em05`
- Checkpoint: `results/verifier/lr_sweep/lr_5em05/best_model/model.pt`
- Metadata: `results/verifier/best_current/selection.yaml`
- Selection basis:
  - highest validation AUC among the tested learning rates
  - best clean evaluation raw AUC under the calibration protocol
  - best best-threshold F1 on the untouched evaluation subset

## Related Artifacts

- Training history:
  [train_history.json](/e:/Engineering/vcsr/results/verifier/lr_sweep/lr_5em05/train_history.json)
- Validation metrics:
  [val_metrics.json](/e:/Engineering/vcsr/results/verifier/lr_sweep/lr_5em05/val_metrics.json)
- Clean calibration report:
  [calibration_report.json](/e:/Engineering/vcsr/results/verifier/lr_sweep/lr_5em05/calibration_report.json)
- Sweep summary:
  [summary.json](/e:/Engineering/vcsr/results/verifier/lr_sweep/summary.json)

## Note

The model weights are not duplicated here. This folder is a stable pointer and
metadata location, not a second copy of the artifact.
