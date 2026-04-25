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

### Clean Calibration Protocol

- Goal: fit calibration on one split and evaluate threshold / risk-coverage behavior on a separate untouched split.
- Script: [scripts/calibrate_verifier.py](/e:/Engineering/vcsr/scripts/calibrate_verifier.py)
- Inputs:
  - [configs/verifier_full.yaml](/e:/Engineering/vcsr/configs/verifier_full.yaml)
  - [results/verifier/full_run/best_model/model.pt](/e:/Engineering/vcsr/results/verifier/full_run/best_model/model.pt)
- Outputs:
  - [results/verifier/full_run/calibration_report.json](/e:/Engineering/vcsr/results/verifier/full_run/calibration_report.json)
  - [results/verifier/full_run/calibration_eval_scores.jsonl](/e:/Engineering/vcsr/results/verifier/full_run/calibration_eval_scores.jsonl)
- Protocol:
  - start from the verifier's held-out validation pool (`357` rows)
  - split by template group into calibration (`180`) and evaluation (`177`) subsets
  - fit calibration only on the calibration subset
  - report threshold sweeps and risk-coverage curves only on the evaluation subset
- Status: completed

Key evaluation results from [calibration_report.json](/e:/Engineering/vcsr/results/verifier/full_run/calibration_report.json):

- Evaluation raw-score AUC: `0.7982`
- Evaluation raw-score log loss: `0.5176`
- Evaluation raw-score ECE: `0.0803`
- Temperature scaling:
  - fitted temperature: `1.05`
  - evaluation log loss: `0.5156`
  - evaluation ECE: `0.0776`
- Isotonic regression:
  - evaluation AUC: `0.7763`
  - evaluation log loss: `0.5378`
  - evaluation ECE: `0.0398`

Best threshold results on the untouched evaluation subset:

- Raw: best F1 `0.6374` at threshold `0.25`
- Temperature-scaled: best F1 `0.6374` at threshold `0.25`
- Isotonic: best F1 `0.6378` at threshold `0.30`

Risk-coverage examples on the evaluation subset:

- Raw scores at threshold `0.50`:
  - coverage `0.1864`
  - selective accuracy `0.6970`
  - selective risk `0.3030`
- Raw scores at threshold `0.25`:
  - coverage `0.6780`
  - selective accuracy `0.4833`
  - selective risk `0.5167`
- Temperature-scaled scores at threshold `0.90`:
  - coverage `0.0904`
  - selective accuracy `0.9375`
  - selective risk `0.0625`

Interpretation:

- This is the first calibration result we should treat as methodologically credible for internal decision-making.
- The verifier's ranking quality remains encouraging under the cleaner protocol (`AUC ≈ 0.80` on the untouched evaluation subset).
- Temperature scaling helps calibration a little, but does not materially change ranking or the best threshold.
- Isotonic improves calibration error more strongly, but loses some ranking quality on the untouched evaluation subset.
- The trustworthy operating region now looks clearer:
  - lower thresholds around `0.25` are better if we want higher recall / coverage
  - higher thresholds are useful for abstention-style high-confidence acceptance with lower coverage

Project takeaway:

- We now have a usable and cleaner basis for threshold selection and selective prediction analysis.
- The next natural experiment is verifier-ranked best-of-K with one or two chosen operating points from this report, rather than using the arbitrary default `0.5`.

### Verifier Learning-Rate Sweep

- Goal: make the planned `lr_sweep` in `configs/vcsr.yaml` real and compare verifier quality under a small controlled hyperparameter sweep.
- Runner: [scripts/run_verifier_lr_sweep.py](/e:/Engineering/vcsr/scripts/run_verifier_lr_sweep.py)
- Sweep source:
  - [configs/vcsr.yaml](/e:/Engineering/vcsr/configs/vcsr.yaml)
  - [configs/verifier_full.yaml](/e:/Engineering/vcsr/configs/verifier_full.yaml)
- Output root: [results/verifier/lr_sweep](/e:/Engineering/vcsr/results/verifier/lr_sweep)
- Summary:
  - [summary.json](/e:/Engineering/vcsr/results/verifier/lr_sweep/summary.json)
  - [summary.md](/e:/Engineering/vcsr/results/verifier/lr_sweep/summary.md)
- Status: completed

Swept learning rates:

- `1e-5`
- `2e-5`
- `5e-5`

Summary of results from [summary.json](/e:/Engineering/vcsr/results/verifier/lr_sweep/summary.json):

- `lr=1e-5`
  - val AUC: `0.7197`
  - eval raw AUC: `0.7247`
  - best raw F1 on clean eval split: `0.5990`
- `lr=2e-5`
  - val AUC: `0.7751`
  - eval raw AUC: `0.7925`
  - best raw F1 on clean eval split: `0.6417`
- `lr=5e-5`
  - val AUC: `0.7777`
  - eval raw AUC: `0.7929`
  - best raw F1 on clean eval split: `0.6630`

Interpretation:

- `1e-5` underperforms clearly and should not be our default.
- `2e-5` and `5e-5` are both strong, but `5e-5` is now the best candidate on the metrics most relevant to VCSR:
  highest validation AUC, highest clean evaluation AUC, and highest best-threshold F1.
- Calibration quality is mixed:
  `2e-5` has slightly better ECE than `5e-5`, but `5e-5` has the stronger ranking and operating-point performance.
- The project should now treat `5e-5` as the best current verifier checkpoint among the tested LRs.

Project takeaway:

- The sweep was worth doing; we got a measurable improvement over simply assuming the original `2e-5` was best.
- The verifier now has a more justified default training LR for future experiments.
- Next best step is to use the `5e-5` checkpoint for verifier-ranked best-of-K experiments, while keeping the clean calibration protocol in the loop for threshold selection.

### First Verifier-Ranked Best-of-K Pilot

- Goal: run the first end-to-end VCSR ranking experiment that uses the selected
  verifier checkpoint to rank multiple generated PDDL candidates and compare that
  choice against simple non-verifier baselines.
- Config: [configs/vcsr_bestofk_pilot.yaml](/e:/Engineering/vcsr/configs/vcsr_bestofk_pilot.yaml)
- Entrypoint: [scripts/run_verifier_bestofk.py](/e:/Engineering/vcsr/scripts/run_verifier_bestofk.py)
- Verifier source:
  [results/verifier/best_current/selection.yaml](/e:/Engineering/vcsr/results/verifier/best_current/selection.yaml)
- Output: [results/vcsr/bestofk_pilot](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot)
- Status: completed first ranking-only pilot

Pilot setup:

- Planetarium `test` split only
- In-domain rows only: `blocksworld` and `gripper`
- Fixed sample count: `30` rows with seed `42`
- Bedrock backend through the shared generation harness
- Candidate counts evaluated: `K in {1, 4, 8}`
- Policies compared:
  - `greedy_first`
  - `random_parseable`
  - `verifier_ranked`

Top-line results from
[summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/summary.md)
and
[aggregate_metrics.json](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/aggregate_metrics.json):

- `K=1`
  - all policies coincide by construction
  - parse rate: `0.9333`
  - equivalence rate: `0.4333`
- `K=4`
  - `greedy_first`: parse `0.9333`, equiv `0.4333`
  - `random_parseable`: parse `0.9667`, equiv `0.4333`
  - `verifier_ranked`: parse `0.9667`, equiv `0.4333`
  - oracle best-of-4 equivalence upper bound: `0.5667`
- `K=8`
  - `greedy_first`: parse `0.9333`, equiv `0.4333`
  - `random_parseable`: parse `0.9667`, equiv `0.5000`
  - `verifier_ranked`: parse `0.9667`, equiv `0.4667`
  - oracle best-of-8 equivalence upper bound: `0.6000`

Candidate-pool diagnostics:

- At `K=8`, the generator produced on average:
  - `7.6` parseable candidates per row
  - `3.37` equivalent candidates per row
- This means generation quality is not the main bottleneck on this pilot;
  there is recoverable headroom in the candidate pool if ranking improves.

Domain and slice observations:

- All successful equivalence selections in this pilot came from `blocksworld`.
- `gripper` had reasonable parseability, but `0` equivalent candidates for all
  reported policies on this 30-row sample.
- The verifier-ranked comparison should therefore be interpreted mainly as a
  `blocksworld` ranking signal check, not yet as a full in-domain conclusion.

Row-level ranking error analysis from
[candidate_dump.jsonl](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/candidate_dump.jsonl):

- At `K=8`, `18` of `30` rows had at least one parseable equivalent candidate
  in the pool.
- The verifier-selected candidate was equivalent on `14` of those `18` oracle-positive rows.
- The verifier therefore missed `4` rows where a good candidate was available,
  and all four misses were in `blocksworld`.
- These misses were not "no-signal" failures. In each case, the verifier scored
  a non-equivalent candidate slightly above an equivalent parseable candidate.

Representative failure modes:

- `blocksworld_swap_to_swap_blocks_list_10_3`
  - verifier selected candidate `1` with score `0.3433` and `equivalent=false`
  - candidate `7` was parseable and equivalent with score `0.3318`
- `blocksworld_invert_to_invert_blocks_list_1_3_3_4_5`
  - verifier scores were effectively tied around `0.9829`
  - verifier selected non-equivalent candidate `0`
  - equivalent candidates `1`, `2`, and `3` had the same rounded score
- `blocksworld_invert_to_invert_blocks_list_1_1_1_5_5_5`
  - verifier selected candidate `3` with score `0.5015` and `equivalent=false`
  - candidate `6` was equivalent with score `0.4998`
- `blocksworld_invert_to_invert_blocks_list_1_1_2_2_2_12`
  - verifier selected candidate `1` with score `0.4810` and `equivalent=false`
  - candidates `2` and `4` were equivalent with score `0.4563`

Interpretation:

- This pilot succeeded as an engineering milestone:
  the full verifier-ranked best-of-K loop now runs end to end and produces
  candidate-level and policy-level artifacts we can trust.
- The current verifier is useful enough to improve selected parse rate relative
  to greedy generation, but it is not yet a reliably superior ranker.
- The key project result is nuanced:
  verifier-ranked best-of-K did not clearly beat a simple `random_parseable`
  baseline on this first pilot, even though the candidate pool contained enough
  good options for a much stronger selector to do better.
- That makes the next problem a ranking-quality problem, not a search-plumbing problem.

Project takeaway:

- We now have the right experiment harness for VCSR.
- The selected verifier checkpoint is good enough to evaluate ranking policies,
  but not yet good enough to claim strong best-of-K selection gains.
- The next highest-value work is targeted rank-error analysis and verifier
  improvement:
  inspect near-miss rows, tighten supervision around semantically subtle
  negatives, and then rerun the same best-of-K pilot before expanding scope.

### Hard-Negative Mining + Verifier Retrain Round 1

- Goal: turn concrete verifier misranking failures from the first best-of-K
  pilot into targeted training signal and test whether the verifier improves as
  a ranker-oriented classifier.
- Mining script:
  [scripts/mine_verifier_hard_negatives.py](/e:/Engineering/vcsr/scripts/mine_verifier_hard_negatives.py)
- Retrain config:
  [configs/verifier_hardneg_round1.yaml](/e:/Engineering/vcsr/configs/verifier_hardneg_round1.yaml)
- Outputs:
  - mined dataset root:
    [results/verifier/hardneg_round1](/e:/Engineering/vcsr/results/verifier/hardneg_round1)
  - corrected fixed-validation retrain:
    [results/verifier/hardneg_round1/retrain_fixed_val](/e:/Engineering/vcsr/results/verifier/hardneg_round1/retrain_fixed_val)
- Status: completed first hard-negative development round

Method:

- Start from the first verifier-ranked best-of-K pilot output:
  [results/vcsr/bestofk_pilot/candidate_dump.jsonl](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/candidate_dump.jsonl)
- Identify rows where:
  - an equivalent parseable candidate existed in the `K=8` pool
  - `verifier_ranked` selected a parseable non-equivalent candidate
- Mine:
  - one positive anchor from the highest-scoring equivalent candidate
  - the verifier-selected wrong candidate
  - additional parseable non-equivalent candidates that outranked the positive
- Keep the original verifier validation split fixed by adding the mined rows as
  `extra_train_jsonl` instead of resplitting a merged dataset

Mining results from
[results/verifier/hardneg_round1/mining_report.json](/e:/Engineering/vcsr/results/verifier/hardneg_round1/mining_report.json):

- `30` pilot rows considered
- `18` rows had at least one equivalent candidate in the `K=8` pool
- `4` rows were verifier misses with a recoverable good candidate available
- `12` mined training examples created:
  - `4` positives
  - `8` negatives
- All mined examples came from `blocksworld`

Retrain results from
[val_metrics.json](/e:/Engineering/vcsr/results/verifier/hardneg_round1/retrain_fixed_val/val_metrics.json)
compared against the previous selected verifier at
[results/verifier/lr_sweep/lr_5em05/val_metrics.json](/e:/Engineering/vcsr/results/verifier/lr_sweep/lr_5em05/val_metrics.json):

- Overall validation metrics:
  - AUC: `0.7777 -> 0.7951`
  - F1: `0.4787 -> 0.5388`
  - Recall: `0.3750 -> 0.4917`
  - Precision: `0.6618 -> 0.5960`
  - Log loss: `0.4934 -> 0.4688`
- Blocksworld slice:
  - AUC: `0.7526 -> 0.7779`
  - F1: `0.5696 -> 0.6243`
  - Recall: `0.5000 -> 0.6556`
- Gripper slice:
  - AUC stayed similar: `0.7980 -> 0.7894`
  - thresholded positive predictions are still weak at the default `0.5`

Interpretation:

- This is the first verifier-improvement round that is clearly motivated by
  downstream VCSR failures rather than generic hyperparameter tuning.
- The result is encouraging:
  the verifier became meaningfully better on the fixed validation split,
  especially in recall and on the `blocksworld` slice that dominated the pilot
  ranking failures.
- The price of the improvement is lower precision at the default threshold,
  which is acceptable for ranking use but means calibration still matters.
- The new training path is methodologically better than a naive merged retrain:
  `extra_train_jsonl` preserves comparability of the base validation split.

Important caveat:

- Because the mined examples came from
  [results/vcsr/bestofk_pilot](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot),
  that 30-row pilot should now be treated as a development set.
- Rerunning the same pilot is still useful as a development check, but it is no
  longer a clean held-out evaluation for project claims.

Project takeaway:

- The hard-negative loop appears to be worth continuing.
- We now have a stronger verifier candidate for ranking-focused follow-up work.
- The next step should be:
  rerun the best-of-K pilot with this new checkpoint as a development check, and
  then create a fresh untouched evaluation sample before making stronger claims.

### Verifier Capacity Push Sweep

- Goal: test whether a larger effective batch and longer training budget can
  improve verifier quality on the current hard-negative training setup without
  changing the data distribution.
- Config:
  [configs/verifier_capacity_push.yaml](/e:/Engineering/vcsr/configs/verifier_capacity_push.yaml)
- Runner:
  [scripts/run_verifier_capacity_push.py](/e:/Engineering/vcsr/scripts/run_verifier_capacity_push.py)
- Output root:
  [results/verifier/capacity_push](/e:/Engineering/vcsr/results/verifier/capacity_push)
- Status: completed

Sweep design:

- training data:
  - base neggen JSONL
  - plus `extra_train_jsonl` mined from the first best-of-K pilot
- max epochs: `12`
- early stopping patience: `3`
- effective batch size: `128`
  - `batch_size=8`
  - `gradient_accumulation_steps=16`
- learning rates:
  - `2e-5`
  - `5e-5`
  - `7.5e-5`

Results from
[summary.md](/e:/Engineering/vcsr/results/verifier/capacity_push/summary.md):

- `lr=2e-5`
  - val AUC: `0.7972`
  - val F1: `0.4762`
  - eval raw AUC: `0.7975`
  - best raw F1: `0.6561` at threshold `0.35`
- `lr=5e-5`
  - val AUC: `0.7915`
  - val F1: `0.4742`
  - eval raw AUC: `0.7964`
  - best raw F1: `0.6667` at threshold `0.30`
- `lr=7.5e-5`
  - val AUC: `0.7815`
  - val F1: `0.4809`
  - eval raw AUC: `0.7926`
  - best raw F1: `0.6595` at threshold `0.35`

Interpretation:

- The capacity push helped ranking metrics a little, but not dramatically.
- `lr=2e-5` is now the best ranking-oriented result in this sweep by validation
  AUC and clean evaluation raw AUC.
- `lr=5e-5` gave the best thresholded clean-eval F1 in this sweep.
- Compared with the earlier hard-negative retrain, this sweep did not produce a
  clear win on default-threshold F1; its main benefit is modest ranking-quality
  improvement, not a broad across-the-board jump.
- This suggests the current bottleneck is still data quality and task alignment
  more than raw optimization budget.

Project takeaway:

- Bigger machine budget is useful, but it is not a substitute for better
  ranking-focused supervision.
- For ranking-first downstream experiments, `capacity_push/lr_2p0em05` is a
  reasonable candidate checkpoint.
- For thresholded verifier evaluation, `capacity_push/lr_5p0em05` is also worth
  keeping in mind because of its stronger best-threshold F1.
- The next most informative experiment is still downstream:
  rerun verifier-ranked best-of-K with the new capacity-push checkpoint and see
  whether the small ranking gains translate into better candidate selection.

### Best-of-K Development Rerun with Capacity-Push Verifier

- Goal: test whether the ranking-oriented winner from the capacity-push sweep
  improves downstream best-of-K selection.
- Config:
  [configs/vcsr_bestofk_capacity_push_lr2.yaml](/e:/Engineering/vcsr/configs/vcsr_bestofk_capacity_push_lr2.yaml)
- Verifier source:
  [results/verifier/capacity_push/lr_2p0em05/selection.yaml](/e:/Engineering/vcsr/results/verifier/capacity_push/lr_2p0em05/selection.yaml)
- Output:
  [results/vcsr/bestofk_capacity_push_lr2](/e:/Engineering/vcsr/results/vcsr/bestofk_capacity_push_lr2)
- Status: completed development rerun

Headline results from
[summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_capacity_push_lr2/summary.md):

- `K=1`
  - all policies coincide at equivalence `0.5333`
- `K=4`
  - `greedy_first`: `0.5333`
  - `random_parseable`: `0.4333`
  - `verifier_ranked`: `0.4000`
- `K=8`
  - `greedy_first`: `0.5333`
  - `random_parseable`: `0.5000`
  - `verifier_ranked`: `0.4333`
  - oracle best-of-8 upper bound: `0.6000`

Interpretation:

- This rerun did **not** produce a downstream win for verifier-ranked selection.
- However, it is not a clean controlled verifier comparison, because the
  candidate pool was regenerated through Bedrock and changed substantially from
  the first pilot.
- In this rerun, parse rate rose to `1.0` for all policies and greedy `K=1`
  performance improved markedly, which means the generator output distribution
  shifted enough to confound the verifier comparison.

Project takeaway:

- We should stop over-interpreting reruns that regenerate candidate pools.
- The project's key missing experimental tool is now clear:
  **fixed-pool replay evaluation**.
- That evaluator is needed before deciding whether more verifier training data,
  larger sweeps, or a changed ranking objective actually improve downstream VCSR.

### Fixed-Pool Replay Comparison Across Verifier Checkpoints

- Goal: compare verifier checkpoints on the exact same cached candidate pool so
  we can separate verifier ranking quality from generation randomness.
- Entrypoint:
  [scripts/replay_verifier_bestofk.py](/e:/Engineering/vcsr/scripts/replay_verifier_bestofk.py)
- Source candidate pool:
  [results/vcsr/bestofk_pilot/candidate_dump.jsonl](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/candidate_dump.jsonl)
- Compared verifier selections:
  - [results/verifier/best_current/selection.yaml](/e:/Engineering/vcsr/results/verifier/best_current/selection.yaml)
  - [results/verifier/capacity_push/lr_2p0em05/selection.yaml](/e:/Engineering/vcsr/results/verifier/capacity_push/lr_2p0em05/selection.yaml)
- Outputs:
  - [results/vcsr/bestofk_pilot/replay_compare/replay_summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/replay_compare/replay_summary.md)
  - [results/vcsr/bestofk_pilot/replay_compare/replay_summary.json](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/replay_compare/replay_summary.json)
  - [results/vcsr/bestofk_pilot/replay_compare/replay_dump.jsonl](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/replay_compare/replay_dump.jsonl)
- Status: completed

Headline replay results from
[replay_summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/replay_compare/replay_summary.md):

- `K=1`
  - all policies coincide at equivalence `0.4333`
  - this is the expected sanity check and confirms the replay path is behaving
    correctly
- `K=4`
  - for both compared verifiers:
    - `greedy_first`: `0.4333`
    - `random_parseable`: `0.4333`
    - `verifier_ranked`: `0.4000`
- `K=8`
  - with `lr_5em05`:
    - `greedy_first`: `0.4333`
    - `random_parseable`: `0.5000`
    - `verifier_ranked`: `0.4333`
  - with `lr_2p0em05`:
    - `greedy_first`: `0.4333`
    - `random_parseable`: `0.5000`
    - `verifier_ranked`: `0.4000`
  - oracle best-of-8 upper bound on the fixed pool: `0.6000`

Interpretation:

- This is one of the most important project results so far because it removes
  the main confound from earlier downstream comparisons.
- The weak downstream result was not just a consequence of regenerated Bedrock
  pools behaving differently across runs.
- On the same candidate pool, verifier-ranked selection still fails to beat the
  simple non-verifier baseline we care about.
- The capacity-push verifier also does not rescue the ranking problem on this
  pool, even though it looked slightly better on offline ranking metrics.
- That means the central problem is now much clearer:
  the verifier is not yet sufficiently aligned with the within-pool ranking task
  that VCSR actually needs.

Project takeaway:

- Fixed-pool replay is no longer a missing tool; it is now a completed and
  trusted part of the evaluation stack.
- Generic verifier improvement on validation AUC is not enough by itself.
- The next modeling step should focus on **ranking-aligned verifier training**:
  more real candidate-pool hard negatives and acceptance based on replay wins,
  not just offline metric gains.

### Ranking-Aligned Verifier Training Round 1

- Goal: perform the first verifier retraining round explicitly aligned with the
  downstream ranking task by mining whole candidate pools, warm-starting from
  the best ranking-oriented checkpoint, and validating the result with fixed-pool
  replay.
- Mining script:
  [scripts/mine_verifier_ranking_examples.py](/e:/Engineering/vcsr/scripts/mine_verifier_ranking_examples.py)
- Training config:
  [configs/verifier_ranking_aligned_round1.yaml](/e:/Engineering/vcsr/configs/verifier_ranking_aligned_round1.yaml)
- Outputs:
  - mined data root:
    [results/verifier/ranking_aligned_round1](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round1)
  - trained checkpoint:
    [results/verifier/ranking_aligned_round1/retrain_from_capacity_push](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round1/retrain_from_capacity_push)
  - replay comparison:
    [results/vcsr/bestofk_pilot/replay_compare_ranking_round1](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/replay_compare_ranking_round1)
- Status: completed first ranking-aligned development round

Method:

- Start from the same fixed candidate pool used in replay:
  [results/vcsr/bestofk_pilot/candidate_dump.jsonl](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/candidate_dump.jsonl)
- For rows with equivalent candidates in the `K=8` pool:
  - add up to `2` parseable equivalent positives
  - add up to `4` parseable non-equivalent negatives from the same pool,
    prioritizing verifier-selected wrong candidates and high-scoring near misses
- For rows with no equivalent candidate in-pool:
  - optionally add one top-scoring parseable negative so the model still sees
    realistic hard negatives from those rows
- Warm-start training from:
  [results/verifier/capacity_push/lr_2p0em05/selection.yaml](/e:/Engineering/vcsr/results/verifier/capacity_push/lr_2p0em05/selection.yaml)
- Keep the base verifier validation split fixed and inject the mined rows
  through `extra_train_jsonl`
- Repeat the mined set `4x` during training to make the ranking-focused signal
  matter relative to the base neggen data

Mining results from
[results/verifier/ranking_aligned_round1/mining_report.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round1/mining_report.json):

- `30` pilot rows considered
- `29` rows had at least one parseable candidate in the `K=8` pool
- `18` rows had at least one equivalent candidate in-pool
- `23` rows had parseable non-equivalent candidates in-pool
- `4` rows were verifier misses with a recoverable equivalent candidate available
- `11` rows contributed negative-only mining examples
- `63` mined examples created:
  - `25` positives
  - `38` negatives
- By domain:
  - `53` `blocksworld`
  - `10` `gripper`

Training results from
[val_metrics.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round1/retrain_from_capacity_push/val_metrics.json):

- Warm-start source:
  [results/verifier/capacity_push/lr_2p0em05/val_metrics.json](/e:/Engineering/vcsr/results/verifier/capacity_push/lr_2p0em05/val_metrics.json)
- Ranking-aligned round 1:
  - val AUC: `0.7995`
  - val F1: `0.4813`
  - val precision: `0.6716`
  - val recall: `0.3750`
- Compared with the warm-start checkpoint:
  - AUC moved slightly upward: `0.7972 -> 0.7995`
  - thresholded metrics stayed broadly similar

Clean calibration results from
[calibration_report.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round1/retrain_from_capacity_push/calibration_report.json):

- evaluation raw AUC: `0.7973`
- evaluation raw log loss: `0.5022`
- evaluation raw ECE: `0.0762`
- best raw-threshold F1 on the untouched eval subset: `0.6595` at threshold `0.35`

Fixed-pool replay results from
[replay_summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/replay_compare_ranking_round1/replay_summary.md):

- `K=1`
  - all policies still coincide at `0.4333`, as expected
- `K=4`
  - ranking-aligned round 1 did not change the replay result
  - `verifier_ranked` remained at `0.4000`
- `K=8`
  - previous best-current verifier (`lr_5em05`): `0.4333`
  - capacity-push verifier (`lr_2p0em05`): `0.4000`
  - ranking-aligned round 1: `0.4667`
  - `random_parseable` baseline remained stronger at `0.5000`

Interpretation:

- This is the first experiment that gives evidence the ranking-aligned direction
  is actually helping on the downstream task we care about.
- The improvement at `K=8` on the fixed pool is meaningful:
  the new checkpoint beat both previous verifier candidates on the same cached
  candidate set.
- At the same time, the gain is not yet sufficient for project success because
  the model still does not beat the simple `random_parseable` baseline.
- Offline validation and calibration metrics alone would have understated the
  value of this round; replay remains the more important acceptance criterion.

Project takeaway:

- Ranking-aligned training appears directionally correct.
- The first round was too small to fully solve the ranking problem.
- The next step should be a stronger round 2:
  mine larger real candidate pools, widen the ranking-focused supervision, and
  keep using fixed-pool replay as the main decision test.

### Ranking-Aligned Verifier Training Round 2

- Goal: scale up the ranking-aligned training signal using a larger real
  candidate pool, then test whether that stronger verifier finally improves
  downstream fixed-pool replay enough to matter for the VCSR objective.
- Pool-generation config:
  [configs/vcsr_bestofk_ranking_round2_pool.yaml](/e:/Engineering/vcsr/configs/vcsr_bestofk_ranking_round2_pool.yaml)
- Pool-generation output:
  [results/vcsr/bestofk_ranking_round2_pool](/e:/Engineering/vcsr/results/vcsr/bestofk_ranking_round2_pool)
- Mining script:
  [scripts/mine_verifier_ranking_examples.py](/e:/Engineering/vcsr/scripts/mine_verifier_ranking_examples.py)
- Training config:
  [configs/verifier_ranking_aligned_round2.yaml](/e:/Engineering/vcsr/configs/verifier_ranking_aligned_round2.yaml)
- Trained checkpoint:
  [results/verifier/ranking_aligned_round2/retrain_from_round1](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round2/retrain_from_round1)
- Replay comparison:
  [results/vcsr/bestofk_pilot/replay_compare_ranking_round2](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/replay_compare_ranking_round2)
- Status: completed

Method:

- Start from a larger in-domain best-of-K pool:
  `50` `test` rows from `blocksworld` and `gripper`
- Warm-start from the round-1 ranking-aligned verifier:
  [results/verifier/ranking_aligned_round1/retrain_from_capacity_push/selection.yaml](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round1/retrain_from_capacity_push/selection.yaml)
- Mine the same style of pool-conditioned positives and hard negatives, but from
  the larger round-2 candidate dump
- Keep the base verifier validation split fixed and inject the mined examples
  through `extra_train_jsonl`
- Repeat the mined set `4x` during training so the within-pool ranking signal is
  strong relative to the base neggen data

Important provenance note:

- There are two distinct `bestofk_ranking_round2_pool` generation passes in the
  development history.
- The replayed round-2 verifier in
  [results/verifier/ranking_aligned_round2/retrain_from_round1](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round2/retrain_from_round1)
  was trained from the **earlier** round-2 pool generation.
- The currently visible
  [results/vcsr/bestofk_ranking_round2_pool](/e:/Engineering/vcsr/results/vcsr/bestofk_ranking_round2_pool)
  artifacts come from a later clean rerun after proxy/logging fixes.
- Because candidate generation is stochastic, those two pool runs should not be
  treated as the same provenance source.
- The downstream replay result below remains valid for the trained checkpoint,
  but the current pool folder should not be described as the exact training
  source for that already-trained verifier.

Earlier round-2 mining results from
[results/verifier/ranking_aligned_round2/mining_report.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round2/mining_report.json):

- `50` rows considered
- `50` rows had at least one parseable candidate in-pool
- `30` rows had at least one equivalent candidate in-pool
- `41` rows had parseable non-equivalent candidates in-pool
- `11` rows were verifier misses with a recoverable equivalent candidate
  available
- `20` rows contributed negative-only mining examples
- `102` mined examples created:
  - `39` positives
  - `63` negatives
- By domain:
  - `82` `blocksworld`
  - `20` `gripper`

Training results from
[val_metrics.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round2/retrain_from_round1/val_metrics.json):

- round-2 ranking-aligned retrain:
  - val AUC: `0.8046`
  - val F1: `0.4813`
  - val precision: `0.6716`
  - val recall: `0.3750`
- compared with round 1:
  - AUC improved slightly: `0.7995 -> 0.8046`
  - thresholded metrics remained broadly similar

Clean calibration results from
[calibration_report.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round2/retrain_from_round1/calibration_report.json):

- evaluation raw AUC: `0.8072`
- best raw-threshold F1 on the untouched eval subset: `0.6667` at threshold
  `0.40`

Fixed-pool replay results from
[replay_summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/replay_compare_ranking_round2/replay_summary.md):

- `K=1`
  - all policies still coincide at `0.4333`, as expected
- `K=4`
  - previous best-current verifier (`lr_5em05`): `0.4000`
  - capacity-push verifier (`lr_2p0em05`): `0.4000`
  - ranking-aligned round 1 (`retrain_from_capacity_push`): `0.4000`
  - ranking-aligned round 2 (`retrain_from_round1`): `0.4667`
- `K=8`
  - previous best-current verifier (`lr_5em05`): `0.4333`
  - capacity-push verifier (`lr_2p0em05`): `0.4000`
  - ranking-aligned round 1 (`retrain_from_capacity_push`): `0.4667`
  - ranking-aligned round 2 (`retrain_from_round1`): `0.5333`
  - `random_parseable` baseline remained at `0.5000`
  - oracle best-of-8 upper bound remained `0.6000`

Later clean rerun of the round-2 pool from
[summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_ranking_round2_pool/summary.md):

- `K=4`
  - `greedy_first`: `0.4600`
  - `random_parseable`: `0.4800`
  - `verifier_ranked`: `0.4800`
- `K=8`
  - `greedy_first`: `0.4600`
  - `random_parseable`: `0.4400`
  - `verifier_ranked`: `0.4200`

Replay-style comparison on that newer pool from
[results/vcsr/bestofk_ranking_round2_pool/replay_compare/replay_summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_ranking_round2_pool/replay_compare/replay_summary.md):

- `K=4`
  - previous best-current verifier (`lr_5em05`): `0.4200`
  - capacity-push verifier (`lr_2p0em05`): `0.4200`
  - ranking-aligned round 1 (`retrain_from_capacity_push`): `0.4200`
  - ranking-aligned round 2 (`retrain_from_round1`): `0.4800`
  - `greedy_first`: `0.4600`
  - `random_parseable`: `0.4800`
  - oracle best-of-4 upper bound: `0.5600`
- `K=8`
  - previous best-current verifier (`lr_5em05`): `0.4200`
  - capacity-push verifier (`lr_2p0em05`): `0.3800`
  - ranking-aligned round 1 (`retrain_from_capacity_push`): `0.3800`
  - ranking-aligned round 2 (`retrain_from_round1`): `0.4600`
  - `greedy_first`: `0.4600`
  - `random_parseable`: `0.4400`
  - oracle best-of-8 upper bound: `0.6200`

Interpretation:

- This is the strongest downstream verifier result we have so far on the fixed
  replay benchmark.
- Round 2 is the first verifier checkpoint to beat `random_parseable` on the
  cached `K=8` pool, even if only modestly: `0.5333` versus `0.5000`.
- That means ranking-aligned training is no longer just directionally
  promising; it has now produced a real downstream win on the evaluation tool
  we trust most.
- On the newer round-2 pool, replay-style controlled comparison now shows that
  round 2 is still the best verifier checkpoint among the tested choices.
- However, the stronger fixed-pool win does **not** fully reproduce:
  on the newer pool, round 2 ties `greedy_first` at `K=8` and ties
  `random_parseable` at `K=4`, rather than clearly beating both.
- This makes the likely story more precise:
  the best round-2 run benefited from useful pool-mined supervision, but that
  gain is not yet stable enough across pools to count as solved robustness.
- The project question is therefore no longer "does ranking-aligned training
  help at all?" but "how do we make the replay gain reproducible across
  independently generated pools?"

Project takeaway:

- Ranking-aligned training is now the leading modeling direction by direct
  downstream evidence.
- The current best verifier for downstream replay is
  [results/verifier/ranking_aligned_round2/retrain_from_round1/selection.yaml](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round2/retrain_from_round1/selection.yaml).
- The next step should focus on robustness:
  more diverse ranking-aligned mining across multiple pools and acceptance based
  on replay wins that hold on more than one cached pool before spending on new
  end-to-end generation comparisons.

### Ranking-Aligned Verifier Training Round 3 (Multi-Pool)

- Goal: test whether a larger, more robust multi-pool ranking-aligned training
  set can turn the promising round-2 replay win into a more credible and more
  reproducible downstream result.
- Pool-generation config:
  [configs/vcsr_bestofk_round3_pool.yaml](/e:/Engineering/vcsr/configs/vcsr_bestofk_round3_pool.yaml)
- Pool-generation outputs:
  - [results/vcsr/round3_pool_seed43](/e:/Engineering/vcsr/results/vcsr/round3_pool_seed43)
  - [results/vcsr/round3_pool_seed44](/e:/Engineering/vcsr/results/vcsr/round3_pool_seed44)
  - [results/vcsr/round3_pool_seed45](/e:/Engineering/vcsr/results/vcsr/round3_pool_seed45)
  - [results/vcsr/round3_pool_seed46](/e:/Engineering/vcsr/results/vcsr/round3_pool_seed46)
  - [results/vcsr/round3_pool_seed47](/e:/Engineering/vcsr/results/vcsr/round3_pool_seed47)
- Dataset preparation:
  [scripts/prepare_ranking_round3_dataset.py](/e:/Engineering/vcsr/scripts/prepare_ranking_round3_dataset.py)
- Training config:
  [configs/verifier_ranking_aligned_round3.yaml](/e:/Engineering/vcsr/configs/verifier_ranking_aligned_round3.yaml)
- Trained checkpoint:
  [results/verifier/ranking_aligned_round3/retrain_from_round2_multipool](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round3/retrain_from_round2_multipool)
- Replay comparisons:
  - [results/vcsr/bestofk_pilot/replay_compare_round2_vs_round3_multipool](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/replay_compare_round2_vs_round3_multipool)
  - [results/vcsr/bestofk_ranking_round2_pool/replay_compare_round2_vs_round3_multipool](/e:/Engineering/vcsr/results/vcsr/bestofk_ranking_round2_pool/replay_compare_round2_vs_round3_multipool)
- Status: completed and promoted to `best_current`

Method:

- Freeze round 2 as the warm-start baseline:
  [results/verifier/ranking_aligned_round2/retrain_from_round1/selection.yaml](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round2/retrain_from_round1/selection.yaml)
- Generate five independent in-domain `K=8` pools over `blocksworld` and
  `gripper`, each with `50` test rows.
- Mine pool-conditioned positives and hard negatives across all five pools.
- Deduplicate the mined rows and append them as train-only extra supervision.
- Keep the base verifier validation split fixed.
- Calibrate the resulting checkpoint and judge it primarily by fixed-pool replay
  against the frozen round-2 verifier.

Multi-pool mining results from
[results/verifier/ranking_aligned_round3/mining_report.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round3/mining_report.json)
and
[augmented_train_stats.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round3/augmented_train_stats.json):

- `5` pool runs merged
- `411` raw mined examples before dedup
- `409` deduped mined examples kept
- `159` positives
- `250` negatives
- `2804` total rows in the augmented training set
- effective round-3 train size during retrain: `3674` rows after `extra_train_repeat: 4`

Training and calibration results from
[val_metrics.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round3/retrain_from_round2_multipool/val_metrics.json)
and
[calibration_report.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round3/retrain_from_round2_multipool/calibration_report.json):

- validation AUC: `0.7945`
- validation F1: `0.4656`
- clean evaluation raw AUC: `0.7759`
- best raw-threshold F1 on the untouched eval subset: `0.6630` at threshold `0.35`
- best temperature-scaled F1 on the untouched eval subset: `0.6630` at threshold `0.40`

Important offline interpretation:

- Round 3 did **not** improve the old offline verifier validation AUC versus
  round 2.
- That matters because it reinforces a core project lesson:
  offline verifier metrics alone are not enough to decide checkpoint quality for
  VCSR.

Fixed-pool replay on the original 30-row pilot pool from
[replay_summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/replay_compare_round2_vs_round3_multipool/replay_summary.md):

- `K=4`
  - round 2 `verifier_ranked`: `0.4667`
  - round 3 `verifier_ranked`: `0.5000`
- `K=8`
  - round 2 `verifier_ranked`: `0.5333`
  - round 3 `verifier_ranked`: `0.5333`

Fixed-pool replay on the stronger 50-row round-2 pool from
[replay_summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_ranking_round2_pool/replay_compare_round2_vs_round3_multipool/replay_summary.md):

- `K=4`
  - round 2 `verifier_ranked`: `0.4800`
  - round 3 `verifier_ranked`: `0.5200`
  - `random_parseable`: `0.4800`
  - oracle best-of-4 upper bound: `0.5600`
- `K=8`
  - round 2 `verifier_ranked`: `0.4600`
  - round 3 `verifier_ranked`: `0.5400`
  - `random_parseable`: `0.4400`
  - oracle best-of-8 upper bound: `0.6200`

Interpretation:

- This is the strongest and most encouraging verifier result in the project so
  far.
- The multi-pool round-3 checkpoint is the first verifier to look clearly
  better than round 2 on the stronger 50-row fixed replay pool.
- It also preserves the earlier progress on the original pilot pool, improving
  `K=4` and matching the prior best `K=8` result.
- The key lesson is now much sharper:
  downstream replay can improve even when offline validation AUC does not.
- That means the project finally has what it needed most:
  a verifier checkpoint whose value is supported by the task-level metric we
  actually care about.

Project takeaway:

- Round 3 should now be treated as the frozen `best_current` verifier.
- This is a real optimism point for the project because the ranking-aligned
  direction has now produced a stronger and more reproducible downstream gain.
- The next best step is no longer "prove the verifier can help at all."
- The next best step is to exploit this better verifier in a fresh end-to-end
  held-out best-of-K evaluation and then decide whether we need a still larger
  round 4 or a different ranking objective.

### Fresh Held-Out End-to-End Best-of-K Evaluation with Frozen Round 3

- Goal: test whether the replay-selected round-3 verifier improves the actual
  end-to-end VCSR decision loop on a fresh held-out run, rather than only on
  cached replay pools.
- Config:
  [configs/vcsr_bestofk_round3_holdout_eval.yaml](/e:/Engineering/vcsr/configs/vcsr_bestofk_round3_holdout_eval.yaml)
- Output:
  [results/vcsr/bestofk_round3_holdout_eval](/e:/Engineering/vcsr/results/vcsr/bestofk_round3_holdout_eval)
- Verifier source:
  [results/verifier/best_current/selection.yaml](/e:/Engineering/vcsr/results/verifier/best_current/selection.yaml)
- Status: completed

Setup:

- Fresh `test` split run on `blocksworld` and `gripper`
- Seed `48`, deliberately outside the round-3 pool generation seeds
- `50` evaluated rows
- Bedrock generation with `K=8`
- Policies compared:
  - `greedy_first`
  - `random_parseable`
  - `verifier_ranked`

Headline results from
[summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_round3_holdout_eval/summary.md)
and
[aggregate_metrics.json](/e:/Engineering/vcsr/results/vcsr/bestofk_round3_holdout_eval/aggregate_metrics.json):

- `K=1`
  - all policies coincide at equivalence `0.4400`
- `K=4`
  - `greedy_first`: parse `0.9400`, equiv `0.4400`
  - `random_parseable`: parse `1.0000`, equiv `0.4400`
  - `verifier_ranked`: parse `1.0000`, equiv `0.4200`
  - oracle best-of-4 upper bound: `0.5200`
- `K=8`
  - `greedy_first`: parse `0.9400`, equiv `0.4400`
  - `random_parseable`: parse `1.0000`, equiv `0.3800`
  - `verifier_ranked`: parse `1.0000`, equiv `0.4600`
  - oracle best-of-8 upper bound: `0.5400`

Important slice observations:

- `K=8` is the meaningful downstream win:
  round 3 beats both baselines on this fresh held-out run.
- `K=4` remains unstable:
  verifier-ranked underperforms both `greedy_first` and `random_parseable`.
- The held-out gain is concentrated in `blocksworld`.
- `gripper` again contributes parseable candidates but no equivalent selected
  outputs for the reported policies on this sample.
- Style breakdown remains asymmetric:
  `explicit/explicit` rows are much stronger than `abstract/abstract` rows.

Candidate-pool diagnostics:

- At `K=8`, the generator produced on average:
  - `7.58` parseable candidates per row
  - `3.34` equivalent candidates per row
- Oracle best-of-8 is `0.5400`, so there is still useful but now narrower
  headroom above the verifier's `0.4600`

Compact row-level interpretation:

- The fresh held-out run does **not** show a universal verifier win.
- Instead, it shows a more realistic and still encouraging shape:
  - at small `K`, ranking noise can still outweigh verifier benefit
  - at larger `K`, the verifier is now able to recover extra value from the
    candidate pool that greedy decoding misses
- That is exactly the regime where best-of-K selection is supposed to matter.

Interpretation:

- This run is more mixed than the replay comparisons, but still positive for the
  project direction.
- The key success is that the replay-selected round-3 verifier does translate
  into a fresh end-to-end `K=8` improvement over both simple baselines.
- That means the project is no longer relying only on replay evidence:
  we now have a real held-out end-to-end result pointing in the same direction.
- The remaining weakness at `K=4` suggests the verifier is not yet a uniformly
  better selector, and that some within-pool misrankings remain unresolved.

Project takeaway:

- Round 3 has now passed the most important test so far:
  replay wins were not just artifacts of cached pool evaluation.
- The verifier is beginning to create real downstream value at the larger `K`
  setting that matters most for best-of-K selection.
- The next best step is targeted error analysis of the remaining `K=4` and
  `K=8` misses, followed by deciding whether we need:
  - a modest round 4 focused on those residual failure modes, or
  - a shift to a more explicitly ranking-oriented loss.

### Held-Out Failure Analysis and Round-4 Decision Gate

- Goal: turn the frozen round-3 held-out artifacts into a decision-quality
  recommendation about whether the next step should be a focused round 4 or a
  more explicit ranking objective.
- Entrypoint:
  [scripts/analyze_bestofk_failures.py](/e:/Engineering/vcsr/scripts/analyze_bestofk_failures.py)
- Inputs:
  - [results/vcsr/bestofk_round3_holdout_eval/candidate_dump.jsonl](/e:/Engineering/vcsr/results/vcsr/bestofk_round3_holdout_eval/candidate_dump.jsonl)
  - [results/vcsr/bestofk_round3_holdout_eval/aggregate_metrics.json](/e:/Engineering/vcsr/results/vcsr/bestofk_round3_holdout_eval/aggregate_metrics.json)
- Outputs:
  - [results/vcsr/bestofk_round3_holdout_eval/failure_analysis/failure_summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_round3_holdout_eval/failure_analysis/failure_summary.md)
  - [results/vcsr/bestofk_round3_holdout_eval/failure_analysis/failure_summary.json](/e:/Engineering/vcsr/results/vcsr/bestofk_round3_holdout_eval/failure_analysis/failure_summary.json)
  - [results/vcsr/bestofk_round3_holdout_eval/failure_analysis/failure_cases.jsonl](/e:/Engineering/vcsr/results/vcsr/bestofk_round3_holdout_eval/failure_analysis/failure_cases.jsonl)
  - [results/vcsr/bestofk_round3_holdout_eval/failure_analysis/decision_recommendation.md](/e:/Engineering/vcsr/results/vcsr/bestofk_round3_holdout_eval/failure_analysis/decision_recommendation.md)
- Status: completed

Key findings:

- `50` total held-out rows
- oracle-positive rows:
  - `26` at `K=4`
  - `27` at `K=8`
- oracle-positive verifier misses:
  - `5` at `K=4`
  - `4` at `K=8`
- all oracle-positive verifier misses came from `blocksworld`
- most misses were concentrated in `abstract/abstract`
- most residual score gaps were small to moderate rather than huge

Decision:

- The resulting recommendation was `focused_round4`, not an immediate switch to
  pairwise/listwise training.
- The reason was that the remaining misses looked like concentrated within-pool
  ordering mistakes, not total semantic blindness.

Project takeaway:

- The failure-analysis gate did its job.
- It justified a small, targeted round-4 mining pass instead of another blind
  broad retrain.

### Focused Round-4 Verifier Training

- Goal: test whether a small held-out-failure-focused correction pass can
  improve the downstream selector without changing the backbone.
- Mining script:
  [scripts/prepare_ranking_round4_dataset.py](/e:/Engineering/vcsr/scripts/prepare_ranking_round4_dataset.py)
- Training config:
  [configs/verifier_ranking_aligned_round4.yaml](/e:/Engineering/vcsr/configs/verifier_ranking_aligned_round4.yaml)
- Run artifacts:
  - [results/verifier/ranking_aligned_round4](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round4)
  - [results/verifier/ranking_aligned_round4/retrain_from_round3_focused](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round4/retrain_from_round3_focused)
- Selection metadata:
  [results/verifier/ranking_aligned_round4/retrain_from_round3_focused/selection.yaml](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round4/retrain_from_round3_focused/selection.yaml)
- Status: completed, but not yet promoted

Focused mining results from
[mining_report.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round4/mining_report.json):

- `9` targeted failure cases selected from the held-out analysis
- `5` unique target rows
- `22` deduped mined examples kept
- `7` positives
- `15` negatives
- all mined rows from `blocksworld`
- style mix:
  - `19` `abstract/abstract`
  - `3` `explicit/explicit`

Training and calibration results from
[val_metrics.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round4/retrain_from_round3_focused/val_metrics.json)
and
[calibration_report.json](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round4/retrain_from_round3_focused/calibration_report.json):

- validation AUC: `0.8076`
- validation F1: `0.4757`
- clean evaluation raw AUC: `0.7934`
- best raw-threshold F1 on the untouched eval subset: `0.6739` at threshold `0.30`

Replay on the frozen held-out candidate pool from
[replay_summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_round3_holdout_eval/replay_compare_round3_vs_round4_focused/replay_summary.md):

- `K=4`
  - round 3 `verifier_ranked`: `0.4200`
  - round 4 `verifier_ranked`: `0.5000`
- `K=8`
  - round 3 `verifier_ranked`: `0.4400`
  - round 4 `verifier_ranked`: `0.4600`

Interpretation:

- Round 4 was a real verifier improvement over round 3.
- The focused failure-mining idea was worth doing.
- Replay improved enough to justify a fresh end-to-end held-out check.

Project takeaway:

- Round 4 became the leading provisional verifier candidate.
- But replay alone was not enough to justify immediate promotion.

### Fresh Held-Out End-to-End Best-of-K Evaluation with Focused Round 4

- Goal: test whether the focused round-4 verifier can beat the frozen round-3
  verifier and simple baselines on a fresh end-to-end held-out run.
- Config lineage:
  [configs/vcsr_bestofk_round3_holdout_eval.yaml](/e:/Engineering/vcsr/configs/vcsr_bestofk_round3_holdout_eval.yaml)
  with explicit selection override to
  [results/verifier/ranking_aligned_round4/retrain_from_round3_focused/selection.yaml](/e:/Engineering/vcsr/results/verifier/ranking_aligned_round4/retrain_from_round3_focused/selection.yaml)
- Output:
  [results/vcsr/bestofk_round4_holdout_eval_clean](/e:/Engineering/vcsr/results/vcsr/bestofk_round4_holdout_eval_clean)
- Status: completed

Headline results from
[summary.md](/e:/Engineering/vcsr/results/vcsr/bestofk_round4_holdout_eval_clean/summary.md)
and
[aggregate_metrics.json](/e:/Engineering/vcsr/results/vcsr/bestofk_round4_holdout_eval_clean/aggregate_metrics.json):

- `K=1`
  - all policies coincide at equivalence `0.4600`
- `K=4`
  - `greedy_first`: parse `0.9600`, equiv `0.4600`
  - `random_parseable`: parse `1.0000`, equiv `0.4200`
  - `verifier_ranked`: parse `1.0000`, equiv `0.4400`
  - oracle best-of-4 upper bound: `0.5200`
- `K=8`
  - `greedy_first`: parse `0.9600`, equiv `0.4600`
  - `random_parseable`: parse `1.0000`, equiv `0.5000`
  - `verifier_ranked`: parse `1.0000`, equiv `0.4800`
  - oracle best-of-8 upper bound: `0.5400`

Direct comparison against the frozen round-3 held-out run:

- `K=4` `verifier_ranked`: `0.4200 -> 0.4400`
- `K=8` `verifier_ranked`: `0.4600 -> 0.4800`

Interpretation:

- Round 4 improved over round 3 on the fresh held-out run at both `K=4` and
  `K=8`.
- But it still did **not** clearly win the overall selector comparison:
  - it lost to `greedy_first` at `K=4` (`0.4400 < 0.4600`)
  - it lost to `random_parseable` at `K=8` (`0.4800 < 0.5000`)
- The gaps are small, but that is exactly why this run should be interpreted as
  promising rather than decisive.

Project takeaway:

- Round 4 is the strongest provisional verifier candidate so far.
- It should **not** yet replace round 3 as the official `best_current`
  checkpoint based on this single fresh sample.
- The next best step is repeated fresh held-out evaluation across several seeds,
  not immediate promotion and not another blind retrain.

### Repeated Fresh Held-Out Comparison for Round 3 vs Round 4

- Goal: decide whether the focused round-4 verifier is strong enough to justify
  promotion once we stop relying on a single fresh 50-row held-out sample.
- Harness:
  [scripts/run_multiseed_holdout_compare.py](/e:/Engineering/vcsr/scripts/run_multiseed_holdout_compare.py)
- Config:
  [configs/vcsr_multiseed_holdout_compare.yaml](/e:/Engineering/vcsr/configs/vcsr_multiseed_holdout_compare.yaml)
- Output:
  [results/vcsr/multiseed_holdout_compare](/e:/Engineering/vcsr/results/vcsr/multiseed_holdout_compare)
- Status: completed

Headline results from
[comparison_summary.md](/e:/Engineering/vcsr/results/vcsr/multiseed_holdout_compare/comparison_summary.md)
and
[comparison_summary.json](/e:/Engineering/vcsr/results/vcsr/multiseed_holdout_compare/comparison_summary.json):

- Seeds evaluated: `48`, `49`, `50`
- Rows per run: `50`
- Proxy-cleared OpenRouter path used for stable generation

Mean `verifier_ranked` equivalence:

- round 3
  - `K=4`: `0.4000`
  - `K=8`: `0.4000`
- round 4
  - `K=4`: `0.4000`
  - `K=8`: `0.4267`

Seed-wise head-to-head for `verifier_ranked`:

- `K=4`
  - round 4 wins: `1`
  - round 3 wins: `1`
  - ties: `1`
- `K=8`
  - round 4 wins: `2`
  - round 3 wins: `0`
  - ties: `1`

Interpretation:

- The repeated held-out gate strengthens the round-4 case in a meaningful way.
- The strongest and cleanest new evidence is at `K=8`.
- `K=4` remains mixed enough that we should not oversell round 4 as a
  universally dominant selector.
- This is a more credible promotion signal than the earlier single-seed held-out
  run, because it repeats the same end-to-end protocol across multiple fresh
  seeds.

Project takeaway:

- Round 4 is now the strongest end-to-end verifier candidate so far.
- The promotion case is strongest if the project emphasis is best-of-`8`
  ranking.
- The repo still keeps round 3 as the official `best_current` pointer until we
  explicitly change that metadata.

### Pairwise-Ranking Round 5: Hybrid Pairwise Objective

- Goal: test whether explicit within-pool pairwise supervision improves the
  promoted round-4 verifier on cached best-of-K replay.
- Config:
  [configs/verifier_pairwise_round5.yaml](/e:/Engineering/vcsr/configs/verifier_pairwise_round5.yaml)
- Mining script:
  [scripts/prepare_pairwise_round5_dataset.py](/e:/Engineering/vcsr/scripts/prepare_pairwise_round5_dataset.py)
- Training output:
  [results/verifier/pairwise_round5/retrain_from_round4_hybrid_pairwise](/e:/Engineering/vcsr/results/verifier/pairwise_round5/retrain_from_round4_hybrid_pairwise)
- Status: completed, **not promoted**

Pairwise mining summary:

- Source pools: `round3_pool_seed43` through `round3_pool_seed47`
- Pairwise examples: `247`
- Pointwise retention examples: `464`
- Pairwise examples by `K`:
  - `K=4`: `84`
  - `K=8`: `163`
- Pair types:
  - `near_tie`: `101`
  - `selected_wrong`: `79`
  - `outranks_positive`: `50`
  - `moderate_gap`: `17`

Training summary:

- Warm start: promoted round-4 verifier from
  [results/verifier/best_current/selection.yaml](/e:/Engineering/vcsr/results/verifier/best_current/selection.yaml)
- Objective: `pairwise_logistic_loss + 0.5 * pointwise_bce_loss`
- Effective batch: `128` via microbatch `2` and gradient accumulation `64`
- Early stopping selected epoch `1`
- Offline metrics:
  - val AUC: `0.7927`
  - val F1: `0.4681`
  - pairwise val accuracy: `0.5789`
  - pairwise mean margin: `0.9000`

Replay gate against promoted round 4:

- On
  [bestofk_round4_holdout_eval_clean](/e:/Engineering/vcsr/results/vcsr/bestofk_round4_holdout_eval_clean/replay_compare_round4_vs_pairwise_round5/replay_summary.md):
  - round 4 `K=4`: `0.5000`
  - round 5 `K=4`: `0.5000`
  - round 4 `K=8`: `0.5200`
  - round 5 `K=8`: `0.5000`
- On
  [bestofk_round3_holdout_eval](/e:/Engineering/vcsr/results/vcsr/bestofk_round3_holdout_eval/replay_compare_round4_vs_pairwise_round5/replay_summary.md):
  - round 4 `K=4`: `0.5000`
  - round 5 `K=4`: `0.4600`
  - round 4 `K=8`: `0.4600`
  - round 5 `K=8`: `0.4400`

Interpretation:

- The pairwise training infrastructure works end to end.
- This first round-5 recipe does **not** pass the replay acceptance gate.
- The likely issue is not that pairwise ranking is the wrong research
  direction; it is that this specific pair set/objective balance is too small
  and too blocksworld-heavy, and the pairwise validation split is only `38`
  examples.
- Keep round 4 as `best_current`.
- Treat round 5 as an implemented negative result and a useful scaffold for a
  more careful ranking-loss experiment.

### Conservative Ranking Round 6 From Promoted Round 4

- Goal: improve round 4 without repeating the aggressive round-5 recipe.
- Regression diagnostic:
  [results/verifier/pairwise_round5/regression_analysis](/e:/Engineering/vcsr/results/verifier/pairwise_round5/regression_analysis)
- Dataset output:
  [results/verifier/ranking_round6](/e:/Engineering/vcsr/results/verifier/ranking_round6)
- Training config:
  [configs/verifier_ranking_round6.yaml](/e:/Engineering/vcsr/configs/verifier_ranking_round6.yaml)
- Training output:
  [results/verifier/ranking_round6/retrain_from_round4_conservative_pairwise](/e:/Engineering/vcsr/results/verifier/ranking_round6/retrain_from_round4_conservative_pairwise)
- Replay gate:
  [results/vcsr/replay_compare_round4_vs_round6/replay_gate_summary.md](/e:/Engineering/vcsr/results/vcsr/replay_compare_round4_vs_round6/replay_gate_summary.md)
- Status: completed, **rejected**

Round-5 regression diagnostic:

- Changed outcome rows across two replay pools: `6`
- Round 5 helped rows: `1`
- Round 5 hurt rows: `5`
- All changed rows were `blocksworld explicit/explicit`

Round-6 mining summary:

- Cached pools used: `10`
- Deduped pairwise rows: `558`
- Pairwise train/dev: `434 / 124`
- Pointwise retention examples: `817`
- Pairwise rows by `K`:
  - `K=4`: `186`
  - `K=8`: `372`

Round-6 training summary:

- Warm start: promoted round 4
- Objective: `1.0 * pointwise_bce + 0.25 * pairwise_logistic_loss`
- Early stopping metric: `val_auc`
- Best epoch: `1`
- Validation AUC: `0.7926`
- Pairwise dev accuracy: `0.5887`

Replay gate against round 4:

- Mean replay `K=4`: round 4 `0.5000`, round 6 `0.4733`
- Mean replay `K=8`: round 4 `0.5156`, round 6 `0.4978`

Interpretation:

- Round 6 does not pass the replay gate.
- The conservative objective avoided a catastrophic collapse, but still moved
  selection in the wrong direction on average.
- Do **not** run fresh generation for round 6.
- Keep round 4 as `best_current`.
- The next improvement should not be another immediate pairwise retrain; first
  analyze score behavior, candidate normalization, calibration-by-row, or
  selection policy alternatives.

### Round-4 Fixed-Model Selection Analysis

- Goal: improve how the promoted round-4 verifier is used in best-of-K without
  training another verifier.
- Entrypoint:
  [scripts/analyze_round4_selection.py](/e:/Engineering/vcsr/scripts/analyze_round4_selection.py)
- Outputs:
  - [results/vcsr/round4_selection_analysis/score_diagnostics.md](/e:/Engineering/vcsr/results/vcsr/round4_selection_analysis/score_diagnostics.md)
  - [results/vcsr/round4_selection_analysis/score_diagnostics.json](/e:/Engineering/vcsr/results/vcsr/round4_selection_analysis/score_diagnostics.json)
  - [results/vcsr/round4_selection_analysis/policy_replay_summary.md](/e:/Engineering/vcsr/results/vcsr/round4_selection_analysis/policy_replay_summary.md)
  - [results/vcsr/round4_selection_analysis/policy_replay_summary.json](/e:/Engineering/vcsr/results/vcsr/round4_selection_analysis/policy_replay_summary.json)
  - [results/vcsr/round4_selection_analysis/changed_rows.jsonl](/e:/Engineering/vcsr/results/vcsr/round4_selection_analysis/changed_rows.jsonl)
- Status: completed, **no selector policy accepted**

Inputs:

- [results/vcsr/bestofk_round4_holdout_eval_clean/candidate_dump.jsonl](/e:/Engineering/vcsr/results/vcsr/bestofk_round4_holdout_eval_clean/candidate_dump.jsonl)
- [results/vcsr/bestofk_round3_holdout_eval/candidate_dump.jsonl](/e:/Engineering/vcsr/results/vcsr/bestofk_round3_holdout_eval/candidate_dump.jsonl)
- [results/vcsr/bestofk_pilot/candidate_dump.jsonl](/e:/Engineering/vcsr/results/vcsr/bestofk_pilot/candidate_dump.jsonl)
- [results/vcsr/bestofk_ranking_round2_pool/candidate_dump.jsonl](/e:/Engineering/vcsr/results/vcsr/bestofk_ranking_round2_pool/candidate_dump.jsonl)

Policies evaluated:

- existing baselines: `greedy_first`, `random_parseable`, `verifier_ranked`
- margin fallback to greedy with margins `0.02`, `0.05`, `0.10`, `0.15`
- top-gap fallback to greedy with the same margin grid
- round-3/round-4 agreement fallback policies
- row-wise score normalization by z-score and min-max
- index-penalized scoring with alpha `0.00`, `0.01`, `0.03`, `0.05`

Mean cached-replay results:

- `K=4`
  - round-4 `verifier_ranked`: `0.5050`
  - best nontrivial changed policy: `round3_round4_agreement_lowest_parseable`
    at `0.4900`
  - all margin, top-gap, index-penalty, greedy, and random policies regressed
- `K=8`
  - round-4 `verifier_ranked`: `0.5167`
  - agreement policies tied round 4 at `0.5167` but changed rows with equal
    helped/hurt counts
  - the closest changed policy was `hybrid_rank_index_penalty_a0.01` at
    `0.5133`
  - all margin, top-gap, stronger index-penalty, greedy, and random policies
    regressed

Score diagnostics:

- rows where round 4 selected an equivalent candidate: `183`
- rows with no equivalent candidate in pool: `160`
- oracle-positive round-4 misses: `17`
- top-vs-second score gaps are often extremely small or tied:
  oracle-positive misses had median top-second gap `0.0` and p75 about
  `0.0010`

Interpretation:

- This was the correct follow-up after rounds 5 and 6 failed replay: it kept
  the round-4 verifier fixed and tested whether score-use heuristics could
  recover easy wins.
- No tested zero-training policy beat plain round-4 `verifier_ranked`.
- Row-wise score normalization was a no-op for ranking, as expected, and tied
  the baseline.
- Fallback policies usually hurt because they overrode correct round-4 choices
  more often than they rescued misses.
- The result strengthens the case that round 4 is a strong local baseline, not
  merely an artifact of a bad selector rule.

Project takeaway:

- Do not launch another small verifier retrain or simple selector tweak as the
  next move.
- If we need a stronger paper claim than the current round-4 story, the next
  improvement likely needs new information or a more structural change:
  generator diversity, semantic/planner-derived features, repair, or a more
  carefully designed verifier architecture/objective.
- Round 4 remains the promoted `best_current` verifier and the best fixed
  selector among the tested cached-replay policies.

### Focused Pointwise Round 7 From Promoted Round 4

- Goal: improve in the direction that made round 4 work: focused, pointwise,
  warm-started, and downstream-gated.
- Mining script:
  [scripts/prepare_focused_round7_dataset.py](/e:/Engineering/vcsr/scripts/prepare_focused_round7_dataset.py)
- Training config:
  [configs/verifier_focused_round7.yaml](/e:/Engineering/vcsr/configs/verifier_focused_round7.yaml)
- Mining output:
  [results/verifier/focused_round7](/e:/Engineering/vcsr/results/verifier/focused_round7)
- Training output:
  [results/verifier/focused_round7/retrain_from_round4_pointwise](/e:/Engineering/vcsr/results/verifier/focused_round7/retrain_from_round4_pointwise)
- Replay gate:
  [results/vcsr/replay_compare_round4_vs_round7_focused/replay_gate_summary.md](/e:/Engineering/vcsr/results/vcsr/replay_compare_round4_vs_round7_focused/replay_gate_summary.md)
- Status: completed training, **passed cached replay gate**, not yet promoted

Mining setup:

- cached development pools used:
  `bestofk_pilot`, `bestofk_ranking_round2_pool`, `round3_pool_seed43-47`,
  `bestofk_round3_holdout_eval`, and `bestofk_round4_holdout_eval_clean`
- round-4 verifier scores were used to prioritize mined candidates
- examples are standard pointwise verifier rows, not pairwise rows
- `K=4` and `K=8` were both mined

Mining results from
[mining_report.json](/e:/Engineering/vcsr/results/verifier/focused_round7/mining_report.json):

- `9` cached pools
- `788` deduped mined examples
- `316` positives
- `472` negatives
- source mix:
  - `316` focused positives
  - `225` focused negatives from oracle-positive pools
  - `247` negative-only examples
- domain mix:
  - `555` `blocksworld`
  - `233` `gripper`
- style mix:
  - `547` `abstract/abstract`
  - `241` `explicit/explicit`

Training setup:

- warm start: promoted round 4 from
  [results/verifier/best_current/selection.yaml](/e:/Engineering/vcsr/results/verifier/best_current/selection.yaml)
- objective: pure `pointwise`
- no pairwise/listwise loss
- extra train repeat: `4`
- best epoch: `3`
- early-stopped after epoch `5`

Offline validation:

- round 4 validation AUC: `0.8076`
- round 7 validation AUC: `0.8017`
- this is a small offline AUC drop, but within the pre-set non-collapse
  tolerance and not decisive by itself

Replay gate against round 4:

- mean `K=4`
  - round 4: `0.5050`
  - round 7: `0.5050`
  - delta: `+0.0000`
- mean `K=8`
  - round 4: `0.5167`
  - round 7: `0.5283`
  - delta: `+0.0117`
- row-level changes:
  - `K=4`: `1` helped, `1` hurt
  - `K=8`: `4` helped, `1` hurt
- `K=8` improved on `2` of `4` replay pools

Interpretation:

- This is the first post-round-4 training result that passes the cached replay
  gate.
- The key difference from rounds 5 and 6 is that round 7 returned to the
  successful round-4 recipe: pointwise training with focused mined examples.
- Round 7 should not be promoted yet, because the acceptance plan requires a
  fresh multiseed development comparison before changing `best_current`.

Project takeaway:

- The round-4 direction is still alive.
- Keep round 4 as `best_current` until round 7 passes a fresh gate.

### Fresh Multiseed Round 4 vs Round 7 Comparison

- Goal: test whether round 7's cached replay gain survives fresh generation.
- Config:
  [configs/vcsr_multiseed_round7_compare.yaml](/e:/Engineering/vcsr/configs/vcsr_multiseed_round7_compare.yaml)
- Output:
  [results/vcsr/multiseed_round7_compare](/e:/Engineering/vcsr/results/vcsr/multiseed_round7_compare)
- Status: completed, **round 7 not promoted**

Setup:

- Base best-of-K config:
  [configs/vcsr_bestofk_round3_holdout_eval.yaml](/e:/Engineering/vcsr/configs/vcsr_bestofk_round3_holdout_eval.yaml)
- Seeds: `56`, `57`, `58`
- Rows per verifier/seed: `50`
- Verifiers:
  - round 4 via [results/verifier/best_current/selection.yaml](/e:/Engineering/vcsr/results/verifier/best_current/selection.yaml)
  - round 7 via [results/verifier/focused_round7/retrain_from_round4_pointwise/selection.yaml](/e:/Engineering/vcsr/results/verifier/focused_round7/retrain_from_round4_pointwise/selection.yaml)

Mean fresh verifier-ranked equivalence from
[comparison_summary.md](/e:/Engineering/vcsr/results/vcsr/multiseed_round7_compare/comparison_summary.md):

- `K=4`
  - round 4: `0.4000`
  - round 7: `0.4133`
  - delta: `+0.0133`
- `K=8`
  - round 4: `0.4200`
  - round 7: `0.4200`
  - delta: `+0.0000`

Seed-wise verifier-ranked deltas:

- `K=4`: round 7 wins `2`, round 4 wins `1`, ties `0`
- `K=8`: round 7 wins `2`, round 4 wins `1`, ties `0`, but mean delta is
  exactly tied because seed `56` regressed by `-0.0800`

Interpretation:

- Round 7 did not fail catastrophically; it improved `K=4` mean and tied `K=8`
  mean on fresh generation.
- However, it did not satisfy the promotion rule because the main operating
  point is `K=8`, and the fresh mean did not improve there.
- The cached replay gain was therefore not strong enough to justify replacing
  round 4.

Project takeaway:

- Round 4 remains the official `best_current`.
- Round 7 is a useful provisional/diagnostic result showing that larger
  pointwise focused mining is safer than pairwise rounds 5 and 6.
- The next improvement should investigate why round 7 helps some seeds but
  regresses seed `56`, rather than promoting it or immediately scaling the same
  recipe again.

## Recommended Next Entries

- Analyze round-7 fresh gate changed rows, especially seed `56` at `K=8`.
- Keep round 4 as the promoted default.
- Selective prediction / abstention experiments only after the ranker baseline
  is actually stable
