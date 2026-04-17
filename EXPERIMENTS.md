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

## Recommended Next Entries

- Ranking-aligned verifier training round built from larger real candidate pools
- Replay-based comparison of the next hard-negative retrain against current best
  checkpoints
- Error analysis of replay failures by domain, style, and candidate type
- Fresh held-out downstream best-of-K evaluation after replay-based checkpoint
  selection
