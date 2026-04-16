# Experiments

This file tracks concrete runs, what they were for, what artifacts they produced,
and how to interpret the results in the context of the VCSR project.

## Conventions

- Record the exact config and output directory for each run.
- Distinguish clearly between `dry_run` / smoke tests and full experiments.
- Prefer interpretation in terms of project goals:
  semantic ranking quality, calibration readiness, and downstream usefulness for VCSR.

## 2026-04-16 / 2026-04-17

### Negative Generator Pilot

- Goal: build the first verifier-training dataset from Planetarium gold pairs,
  Bedrock-generated candidates, and perturbation-based hard negatives.
- Config: [configs/neggen.yaml](/e:/Engineering/vcsr/configs/neggen.yaml)
- Output: [results/neggen/pilot](/e:/Engineering/vcsr/results/neggen/pilot)
- Status: completed milestone

Key results from [run_stats.json](/e:/Engineering/vcsr/results/neggen/pilot/run_stats.json)
and [dataset_stats.json](/e:/Engineering/vcsr/results/neggen/pilot/dataset_stats.json):

- 500 Planetarium train rows processed
- 2,499 total assembled examples
- 2,395 parseable training examples
- 919 positives, 1,580 negatives
- 1,000 LLM candidates, 902 parseable, 417 equivalent
- 999 perturbations, 993 parseable, 2 equivalent before policy handling
- 72 equivalence timeouts
- About 8 hours wall-clock time

Interpretation:

- This run completed the Phase 2 data milestone.
- The dataset has meaningful positive/negative signal and enough size for a first verifier experiment.
- The pilot is still small relative to full Planetarium scale, so we should expect the verifier to be informative but imperfect.

### Verifier Dry Run / Smoke Test

- Goal: confirm that the verifier training path executes end to end on the local machine.
- Config lineage: verifier training config targeting `results/verifier/pilot`
- Output: [results/verifier/pilot](/e:/Engineering/vcsr/results/verifier/pilot)
- Status: debugging / smoke test only

Observed outcome:

- The training script ran successfully and produced metrics/artifacts.
- These artifacts should not be treated as a completed verifier experiment.

Interpretation:

- Useful as proof that the pipeline works locally.
- Not suitable as an experiment record for project conclusions.

### Verifier Full Run

- Goal: first real cross-encoder verifier training run on the pilot neggen dataset.
- Config: [configs/verifier_full.yaml](/e:/Engineering/vcsr/configs/verifier_full.yaml)
- Output: [results/verifier/full_run](/e:/Engineering/vcsr/results/verifier/full_run)
- Training entrypoint: [scripts/train_verifier.py](/e:/Engineering/vcsr/scripts/train_verifier.py)
- Status: completed full training run

Best validation results from [val_metrics.json](/e:/Engineering/vcsr/results/verifier/full_run/val_metrics.json):

- AUC: `0.7766`
- Accuracy: `0.7255`
- F1: `0.4787`
- Precision: `0.6618`
- Recall: `0.3750`
- Validation set size: `357`

Training trend from [train_history.json](/e:/Engineering/vcsr/results/verifier/full_run/train_history.json):

- Epoch 1: AUC `0.6834`, F1 `0.3977`
- Epoch 2: AUC `0.7025`, F1 `0.4444`
- Epoch 3: AUC `0.7436`, F1 `0.4973`
- Epoch 4: AUC `0.7545`, F1 `0.4762`
- Epoch 5: AUC `0.7766`, F1 `0.4787`

Validation split composition reproduced from the training split logic:

- Total validation rows: `357`
- Labels: `237` negatives, `120` positives
- By source:
  - `gold`: `75`, all positive
  - `llm_bedrock`: `133`, with `45` positive and `88` negative
  - `perturbation`: `149`, all negative

Interpretation:

- This is a meaningful step forward for the project: the verifier is now learning a usable ranking signal.
- The overall AUC near `0.78` is encouraging for the VCSR goal because candidate selection depends first on ranking quality, not just the hard `0.5` threshold.
- The model appears especially strong at separating Bedrock-generated candidates, which matters because those are the realistic candidates that verifier-ranked search will need to choose among.
- The current threshold behavior is not yet good enough for calibrated abstention or final deployment:
  recall remains modest (`0.375`), and some slices collapse under the default `0.5` decision rule.
- The per-source `gold` and `perturbation` AUC values of `0.0` should not be over-interpreted:
  each of those subsets is single-class in validation, so AUC is undefined there and currently reported as `0.0` by the metric helper.
- The per-domain `gripper` slice has non-trivial AUC (`0.7941`) but `F1=0.0`, which strongly suggests a threshold/calibration issue rather than absence of signal.

Project takeaway:

- We are now past "can the verifier train?" and into "how do we calibrate and use it safely?"
- The next experiments should focus on score calibration, threshold sweeps, and selective-risk analysis before implementing verifier-ranked search as the main inference policy.

### Verifier Calibration / Threshold Analysis

- Goal: understand whether the verifier's weak-looking fixed-threshold behavior is a ranking problem or a calibration / operating-point problem.
- Config/model source: [configs/verifier_full.yaml](/e:/Engineering/vcsr/configs/verifier_full.yaml),
  [results/verifier/full_run/best_model/model.pt](/e:/Engineering/vcsr/results/verifier/full_run/best_model/model.pt)
- Analysis script: [scripts/analyze_verifier.py](/e:/Engineering/vcsr/scripts/analyze_verifier.py)
- Outputs:
  - [results/verifier/full_run/score_analysis.json](/e:/Engineering/vcsr/results/verifier/full_run/score_analysis.json)
  - [results/verifier/full_run/val_scores.jsonl](/e:/Engineering/vcsr/results/verifier/full_run/val_scores.jsonl)
- Status: completed diagnostic analysis

Key findings from [score_analysis.json](/e:/Engineering/vcsr/results/verifier/full_run/score_analysis.json):

- Raw-score AUC: `0.7766`
- Raw-score log loss: `0.5038`
- Raw-score ECE (10 bins): `0.0777`
- Best raw-threshold F1 on this validation split: `0.6198` at threshold `0.20`
- Temperature scaling improved calibration modestly:
  - best temperature: `1.15`
  - log loss: `0.5018`
  - ECE: `0.0696`
- Isotonic regression fit the validation split much more aggressively:
  - AUC: `0.7934`
  - log loss: `0.4432`
  - ECE: effectively `0`
  - best F1: `0.6253` at threshold `0.30`

Interpretation:

- The main bottleneck after training is threshold selection and calibration, not total verifier failure.
- The default threshold of `0.5` is too conservative for this checkpoint and leaves substantial recall on the table.
- A lower threshold around `0.2` to `0.3` dramatically improves F1 on the held-out split.
- Temperature scaling gives a small but believable improvement and preserves ranking behavior.
- Isotonic looks stronger in this analysis, but because it is fit and evaluated on the same validation split, it is optimistic and should not yet be treated as a final reporting result.

Project takeaway:

- This verifier is likely ready for early verifier-ranked best-of-K experiments as a ranking component.
- It is not yet ready for final abstention claims without a cleaner calibration protocol.
- Before paper-facing conclusions, use a separate calibration split or nested validation and then recompute threshold / risk-coverage curves.

## Recommended Next Entries

- Threshold sweep on `results/verifier/full_run` validation scores
- Calibration experiment: temperature scaling vs isotonic
- Error analysis by domain, source, and natural-language style
- First verifier-ranked best-of-K comparison against greedy and random-valid baselines
