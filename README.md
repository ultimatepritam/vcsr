# VCSR: Verifier-Calibrated Search and Repair for Text-to-PDDL

A research framework for faithful natural-language to PDDL translation using
calibrated semantic verification, best-of-K selection, and abstention-aware repair.

## Project Structure

```
configs/           YAML experiment configs (baseline, vcsr, neggen, verifier)
data/              Dataset loaders, splits, verifier JSONL assembly
  planetarium_loader.py   Template-hash splits for Planetarium
  verifier_dataset.py     Build (NL, PDDL, label) rows for verifier training
eval/              Planetarium equivalence wrappers, metrics
  equivalence.py        Lightweight + timed subprocess equivalence
generation/        LLM backends, prompts, perturbations
  prompts.py              NL→PDDL and repair prompts, PDDL extraction
  sampler.py              Bedrock, OpenRouter, OpenAI, HF, MultiSampler
  perturbations.py        Domain-aware gold PDDL mutations (hard negatives)
pddl_utils/        Oracle planner, Fast Downward + VAL wrappers (not `pddl/` — shadows PyPI `pddl`)
verifier/          Cross-encoder dataset/model/train/eval code
search/            (currently minimal) Best-of-K, abstention, repair loop
scripts/           Baselines, neggen, verifier training, calibration, sweeps
results/           Metrics, verifier runs, calibration reports, selected checkpoints
tools/             External tool installs (Fast Downward, VAL)
```

## Current Status

- Foundations and the negative-generation pilot are completed.
- Verifier training is implemented and has been run successfully on the pilot dataset.
- Clean calibration analysis, hard-negative retraining, ranking-aligned retraining, a capacity-push sweep, held-out failure analysis, and a focused round-4 verifier pass have also been completed.
- The current selected verifier checkpoint is recorded in `results/verifier/best_current/selection.yaml`.
- We have also completed verifier-ranked best-of-K pilots, replay-controlled evaluation on multiple cached pools, and fresh held-out end-to-end runs.
- We have now also completed a repeated fresh held-out comparison across seeds `48`, `49`, and `50`.
- Pairwise-ranking round 5 has been implemented and trained as a hybrid pairwise
  + pointwise verifier experiment, but the first recipe did not beat the
  promoted round-4 verifier on replay, so it is not promoted.
- Conservative ranking round 6 has also been implemented and trained from round
  4, but it failed replay against round 4 and is not promoted.
- A fixed-round-4 selector analysis has now tested margin fallback, top-gap
  fallback, agreement fallback, score normalization, and index-penalized
  policies without changing verifier weights. None beat plain round-4
  `verifier_ranked` on cached replay.
- Focused pointwise round 7 returned to the successful round-4 recipe with a
  larger cached-pool mined dataset. It passed cached replay against round 4 at
  `K=8` while tying at `K=4`.
- Fresh multiseed evaluation did **not** justify promoting round 7:
  it improved mean `K=4` but tied mean `K=8`.
- Row-level fresh-gate analysis shows why round 7 is not promoted:
  at `K=8`, it helped `11` rows and hurt `11` rows across `150` rows, and the
  seed `56` regression was mostly selector loss with equivalents still present.
- Fresh identical-pool comparison confirms the promotion decision:
  round 7 has a tiny clean `K=8` gain but regresses `K=4`, so round 4 remains
  `best_current`.
- Phase 3 cached planner/search ablation is complete:
  simple solvability-based policies did not beat round-4 `verifier_ranked`, so
  the next system step was a small repair-loop pilot.
- Phase 3 cached repair pilot is complete:
  one OpenRouter repair call converted `23 / 30` cached round-4 selected
  failures into equivalent PDDL, with `29 / 30` repaired candidates parseable.
  This is promising but still needs a fresh fixed-pool repair gate before it is
  treated as a paper-facing system result.

## Quick Start

```bash
# 1. Create and activate environment
python -m venv .venv
# Windows:
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Optional: install verifier training extras
#    (heavier stack; better suited to Linux/WSL or older Python versions)
pip install -r requirements-training.txt

# 4. Install Planetarium
pip install git+https://github.com/BatsResearch/planetarium.git

# 4. Optional: API keys for generation — create `.env` (not committed) with
#    AWS_* + BEDROCK_MODEL_ID for Bedrock; OPENROUTER_API_KEY for OpenRouter

# 5. Verify dataset loads
python -c "from data.planetarium_loader import PlanetariumDataset; ds = PlanetariumDataset(); print(ds.summary())"

# 6. Baseline reproduction (oracle / perturbed / solvability)
python scripts/reproduce_baselines.py --config configs/baseline.yaml

# 7. Verifier training data (negative generator pilot)
#    See configs/neggen.yaml — full run is long; use --dry_run for 5 rows
python scripts/generate_negatives.py --config configs/neggen.yaml --dry_run
# PowerShell (unbuffered log):
#   $env:PYTHONUNBUFFERED = "1"
#   .venv\Scripts\python -u scripts/generate_negatives.py --config configs/neggen.yaml 2>&1 | Tee-Object -FilePath results/neggen/pilot/run_log.txt

# 8. Stratified random sample from verifier JSONL (sanity check)
python scripts/sample_verifier_jsonl.py results/neggen/pilot/verifier_train.jsonl

# 9. Full verifier training run
python scripts/train_verifier.py --config configs/verifier_full.yaml

# 10. Clean calibration / threshold analysis
python scripts/calibrate_verifier.py --config configs/verifier_full.yaml

# 11. Optional: verifier LR sweep from configs/vcsr.yaml
python scripts/run_verifier_lr_sweep.py

# 12. Mine verifier misranking failures from the first Best-of-K pilot
python scripts/mine_verifier_hard_negatives.py

# 13. Retrain the verifier on the augmented dataset
python scripts/train_verifier.py --config configs/verifier_hardneg_round1.yaml

# 14. Capacity-push sweep on current hard-negative training setup
python scripts/run_verifier_capacity_push.py

# 15. End-to-end verifier-ranked best-of-K pilot
python scripts/run_verifier_bestofk.py --config configs/vcsr_bestofk_pilot.yaml

# 16. Replay verifier checkpoints on a fixed cached pool
python scripts/replay_verifier_bestofk.py --candidate_dump results/vcsr/bestofk_pilot/candidate_dump.jsonl --selection results/verifier/best_current/selection.yaml

# 17. Prepare merged multi-pool round-3 mining data
python scripts/prepare_ranking_round3_dataset.py --pool_dir results/vcsr/round3_pool_seed43 --pool_dir results/vcsr/round3_pool_seed44 --include_negative_only_rows

# 18. Round-3 verifier retrain with visible file logging
python scripts/train_verifier.py --config configs/verifier_ranking_aligned_round3.yaml

# 19. Focused round-4 dataset from held-out failure analysis
python scripts/prepare_ranking_round4_dataset.py

# 20. Round-4 focused verifier retrain
python scripts/train_verifier.py --config configs/verifier_ranking_aligned_round4.yaml

# 21. Fresh held-out evaluation with an explicit verifier selection
python scripts/run_verifier_bestofk.py --config configs/vcsr_bestofk_round3_holdout_eval.yaml --output_dir results/vcsr/bestofk_round4_holdout_eval_clean --selection_metadata results/verifier/ranking_aligned_round4/retrain_from_round3_focused/selection.yaml

# 22. Repeated fresh held-out comparison across multiple seeds
python scripts/run_multiseed_holdout_compare.py --config configs/vcsr_multiseed_holdout_compare.yaml

# 23. Mine pairwise ranking data from cached pools
python scripts/prepare_pairwise_round5_dataset.py

# 24. Hybrid pairwise verifier training from promoted round 4
python scripts/train_verifier.py --config configs/verifier_pairwise_round5.yaml

# 25. Replay pairwise round 5 against promoted round 4 before any promotion
python scripts/replay_verifier_bestofk.py --candidate_dump results/vcsr/bestofk_round4_holdout_eval_clean/candidate_dump.jsonl --selection results/verifier/best_current/selection.yaml --selection results/verifier/pairwise_round5/retrain_from_round4_hybrid_pairwise/selection.yaml --output_dir results/vcsr/bestofk_round4_holdout_eval_clean/replay_compare_round4_vs_pairwise_round5 --k_values 4 8

# 26. Analyze round-5 regression and prepare conservative ranking round 6
python scripts/analyze_round5_regression.py
python scripts/prepare_ranking_round6_dataset.py

# 27. Train and replay conservative ranking round 6
python scripts/train_verifier.py --config configs/verifier_ranking_round6.yaml
python scripts/replay_verifier_bestofk.py --candidate_dump results/vcsr/bestofk_round4_holdout_eval_clean/candidate_dump.jsonl --selection results/verifier/best_current/selection.yaml --selection results/verifier/ranking_round6/retrain_from_round4_conservative_pairwise/selection.yaml --output_dir results/vcsr/replay_compare_round4_vs_round6/round4_holdout_clean --k_values 4 8

# 28. Analyze fixed-round-4 selector policies without training
python scripts/analyze_round4_selection.py

# 29. Prepare and train focused pointwise round 7 from promoted round 4
python scripts/prepare_focused_round7_dataset.py
python scripts/train_verifier.py --config configs/verifier_focused_round7.yaml
python scripts/calibrate_verifier.py --config configs/verifier_focused_round7.yaml

# 30. Fresh multiseed gate for round 4 vs round 7, after cached replay passes
python scripts/run_multiseed_holdout_compare.py --config configs/vcsr_multiseed_round7_compare.yaml

# 31. Row-level analysis of the fresh round-7 gate
python scripts/analyze_round7_fresh_gate.py

# 32. Fresh fixed-pool verifier comparison for round 4 vs round 7
python scripts/run_fixed_pool_verifier_compare.py --config configs/vcsr_fixed_pool_round7_compare.yaml

# 33. Cached planner/search ablation with round 4 frozen
python scripts/analyze_search_ablation.py

# 34. Cached repair pilot with round 4 frozen
python scripts/run_repair_pilot.py --config configs/vcsr_repair_pilot.yaml
```

## Windows E: Drive Setup

If you want all writable runtime artifacts to stay inside this repository on
`E:`, use the bootstrap script before installing or running anything:

```powershell
PowerShell -ExecutionPolicy Bypass -File .\scripts\setup_windows_e_drive.ps1
```

That script creates `.venv` plus a repo-local `.local/` tree and sets common
cache/temp locations for `pip`, Hugging Face, `datasets`, Transformers, Torch,
`wandb`, Matplotlib, Jupyter, Python bytecode, and Windows profile-style temp
paths used by newer Python builds.

Recommended install flow in that same PowerShell session:

```powershell
$env:PIP_NO_INDEX = ""
python -m pip --python .\.venv\Scripts\python.exe install --upgrade pip
python -m pip --python .\.venv\Scripts\python.exe install -r requirements.txt
python -m pip --python .\.venv\Scripts\python.exe install git+https://github.com/BatsResearch/planetarium.git
```

Notes:

- `bitsandbytes` is optional for PEFT experiments and often problematic on Windows.
  If it fails, remove it from the install command for baseline dataset/evaluation work.
- The Python runtime also defaults `PlanetariumDataset` downloads and planner temp
  files into `.local/`, so the common baseline paths stay on `E:` by default.

## Verifier training data (pilot)

Pilot artifacts live under `results/neggen/pilot/` when you run `generate_negatives.py`:

| File | Description |
|------|-------------|
| `verifier_train.jsonl` | Parseable (NL, PDDL, label, source, …) rows for training |
| `verifier_all.jsonl` | All rows including unparseable candidates |
| `dataset_stats.json` | Aggregate counts by source/domain |
| `run_stats.json` | LLM/perturbation/equivalence-timeout counters |
| `run_config.yaml` | Frozen config for that run |

Configure generation in `configs/neggen.yaml` (backends, K, perturbations, **equivalence timeout**).

Rare **perturbation** rows can get Planetarium `label=1` (noise). Default **`labeling.perturbation_positive_policy: relabel`** forces them to negatives on new runs. To patch an existing JSONL without regenerating:

`python scripts/apply_perturbation_label_policy.py results/neggen/pilot/verifier_train.jsonl out.jsonl --policy relabel`

## Verifier Workflow

The verifier is trained on the neggen pilot JSONL, then analyzed with a
separate calibration/evaluation protocol.

Main verifier configs and scripts:

- `configs/verifier_full.yaml`
- `configs/verifier_hardneg_round1.yaml`
- `configs/verifier_capacity_push.yaml`
- `configs/verifier_ranking_aligned_round1.yaml`
- `configs/verifier_ranking_aligned_round2.yaml`
- `configs/verifier_ranking_aligned_round3.yaml`
- `configs/verifier_pairwise_round5.yaml`
- `scripts/train_verifier.py`
- `scripts/analyze_verifier.py`
- `scripts/calibrate_verifier.py`
- `scripts/run_verifier_lr_sweep.py`
- `scripts/run_verifier_capacity_push.py`
- `scripts/mine_verifier_hard_negatives.py`
- `scripts/mine_verifier_ranking_examples.py`
- `scripts/prepare_ranking_round3_dataset.py`
- `scripts/prepare_pairwise_round5_dataset.py`

Current key verifier artifacts:

| Path | Description |
|------|-------------|
| `results/verifier/pilot/` | Earlier dry-run / smoke-test outputs |
| `results/verifier/full_run/` | First completed verifier training run |
| `results/verifier/lr_sweep/` | LR sweep runs plus aggregate summaries |
| `results/verifier/ranking_aligned_round1/` | First ranking-aligned verifier retrain from cached candidate-pool supervision |
| `results/verifier/ranking_aligned_round2/` | Earlier replay-backed downstream verifier |
| `results/verifier/ranking_aligned_round3/` | Prior replay-backed verifier baseline selected from multi-pool replay wins |
| `results/verifier/ranking_aligned_round4/` | Current promoted verifier after replay gains plus the repeated fresh held-out gate |
| `results/verifier/pairwise_round5/` | Hybrid pairwise-ranking experiment; trained successfully but not promoted because replay regressed vs round 4 |
| `results/verifier/ranking_round6/` | Conservative ranking-aware successor from round 4; trained successfully but rejected by replay gate |
| `results/verifier/focused_round7/` | Focused pointwise round-7 dataset and training artifacts; passed cached replay but not promoted yet |
| `results/verifier/best_current/selection.yaml` | Stable metadata record for the current best verifier checkpoint |

As of the current repo state, the selected best verifier comes from:

- run: `results/verifier/ranking_aligned_round4/retrain_from_round3_focused`
- checkpoint: `results/verifier/ranking_aligned_round4/retrain_from_round3_focused/best_model/model.pt`

Round 4 is now the promoted default verifier in repo metadata.

We now also have a repeated fresh held-out comparison under:

- `results/vcsr/multiseed_holdout_compare/`

That multi-seed gate strengthens the case for round 4:

- mean round-3 `verifier_ranked`
  - `K=4`: `0.4000`
  - `K=8`: `0.4000`
- mean round-4 `verifier_ranked`
  - `K=4`: `0.4000`
  - `K=8`: `0.4267`
- seed-wise head-to-head
  - `K=4`: win / loss / tie = `1 / 1 / 1`
  - `K=8`: win / loss / tie = `2 / 0 / 1`

So the strongest current downstream case for round 4 is specifically at
best-of-`8`, not as a claim that it cleanly dominates every setting.

Pairwise round 5 has also been implemented and trained. It is useful as a
negative-result scaffold, but not as a new default: replay against the promoted
round-4 verifier tied at `K=4` and regressed at `K=8` on the clean round-4
held-out pool, and regressed at both `K=4` and `K=8` on the round-3 held-out
pool.

Conservative ranking round 6 was also implemented to address round 5's issues.
It used a larger cached-pool dataset, an explicit pairwise dev split, and a
pointwise-dominant hybrid loss. It still failed replay against round 4, so the
next step was score/selection diagnosis rather than another immediate ranking
retrain.

That fixed-round-4 selector analysis is under:

- `results/vcsr/round4_selection_analysis/`

It tested margin fallback, top-gap fallback, agreement fallback, row-wise score
normalization, and index-penalized ranking across cached pools. No tested
zero-training policy beat plain round-4 `verifier_ranked`; the baseline remains
`0.5050` mean equivalence at `K=4` and `0.5167` at `K=8` on the analyzed cached
replay pools.

Focused pointwise round 7 is under:

- `results/verifier/focused_round7/`

It mined `788` pointwise examples from cached pools, warm-started from round 4,
and used pure pointwise training. Cached replay against round 4 passed:

- mean `K=4`: round 4 `0.5050`, round 7 `0.5050`
- mean `K=8`: round 4 `0.5167`, round 7 `0.5283`

Fresh multiseed evaluation under `results/vcsr/multiseed_round7_compare/`
showed:

- mean `K=4`: round 4 `0.4000`, round 7 `0.4133`
- mean `K=8`: round 4 `0.4200`, round 7 `0.4200`

Round 7 is therefore not promoted. Round 4 remains `best_current`.

See `EXPERIMENTS.md` for the running experiment log and interpretation of these results.

Development note:

- `scripts/mine_verifier_hard_negatives.py` mines rows where the verifier-ranked
  policy picked a wrong parseable candidate even though an equivalent candidate
  existed in the same Best-of-K pool.
- The script writes both a merged dataset and a focused
  `results/verifier/hardneg_round1/mined_examples.jsonl` file.
- `configs/verifier_hardneg_round1.yaml` uses the mined JSONL as
  `extra_train_jsonl`, so those examples are appended to training while the base
  validation split stays comparable.
- Once we train on failures from `results/vcsr/bestofk_pilot/`, that pilot
  should be treated as a development set rather than a fresh benchmark.

## Best-of-K Status

Best-of-K experiment scripts and configs:

- `scripts/run_verifier_bestofk.py`
- `scripts/replay_verifier_bestofk.py`
- `configs/vcsr_bestofk_pilot.yaml`
- `configs/vcsr_bestofk_capacity_push_lr2.yaml`
- `configs/vcsr_bestofk_ranking_round2_pool.yaml`
- `configs/vcsr_bestofk_round3_pool.yaml`

Key downstream artifacts:

| Path | Description |
|------|-------------|
| `results/vcsr/bestofk_pilot/` | First verifier-ranked best-of-K pilot with the earlier selected verifier |
| `results/vcsr/bestofk_capacity_push_lr2/` | Development rerun using the ranking-oriented winner from the capacity-push sweep |
| `results/vcsr/bestofk_pilot/replay_compare_ranking_round2/` | Fixed-pool replay showing the strongest round-2 win on the original pilot pool |
| `results/vcsr/bestofk_ranking_round2_pool/` | Newer 50-row cached pool plus controlled replay across verifier checkpoints |
| `results/vcsr/bestofk_round3_holdout_eval/` | Fresh held-out end-to-end run with frozen round 3 |
| `results/vcsr/bestofk_round4_holdout_eval_clean/` | Fresh held-out end-to-end run with focused round 4 |
| `results/vcsr/multiseed_holdout_compare/` | Repeated fresh held-out round-3 vs round-4 comparison across seeds `48`, `49`, `50` |
| `results/vcsr/bestofk_round4_holdout_eval_clean/replay_compare_round4_vs_pairwise_round5/` | Fixed-pool replay showing pairwise round 5 did not beat round 4 on the clean round-4 held-out pool |
| `results/vcsr/bestofk_round3_holdout_eval/replay_compare_round4_vs_pairwise_round5/` | Fixed-pool replay showing pairwise round 5 also regressed on the earlier round-3 held-out pool |
| `results/vcsr/replay_compare_round4_vs_round6/` | Fixed-pool replay gate showing conservative ranking round 6 does not beat round 4 |
| `results/vcsr/round4_selection_analysis/` | Fixed-round-4 selector-policy analysis showing simple score-use policies do not beat round-4 `verifier_ranked` |
| `results/vcsr/replay_compare_round4_vs_round7_focused/` | Fixed-pool replay gate showing focused pointwise round 7 improves mean `K=8` while tying `K=4` |
| `results/vcsr/multiseed_round7_compare/` | Fresh multiseed gate showing round 7 improves `K=4` mean but ties round 4 at `K=8`, so it is not promoted |
| `results/vcsr/multiseed_round7_compare/fresh_gate_analysis/` | Row-level analysis showing round 7 helped and hurt equally at `K=8`, with seed-56 losses mostly due to selection despite available equivalents |
| `results/vcsr/fixed_pool_round7_compare/` | Fresh identical-pool comparison showing round 7 slightly improves `K=8` but regresses `K=4`, so it is still not promoted |
| `results/vcsr/search_ablation_round4/` | Phase 3 cached planner/search ablation showing solvability-based policies do not pass the cached gate |

Current project conclusion from these pilots:

- The verifier has real offline signal and ranking-aligned training has improved downstream ranking quality.
- Round 3 remains the important replay-backed baseline in the experiment history.
- Round 4 is now the promoted default verifier:
  it improved over round 3 on replay and also improved fresh held-out `verifier_ranked` from `0.42 -> 0.44` at `K=4` and `0.46 -> 0.48` at `K=8`.
- The new repeated fresh held-out comparison strengthens the round-4 case:
  at `K=8`, round 4 now shows a positive mean verifier-ranked gain over round 3
  (`0.4000 -> 0.4267`) with seed-wise results `2` wins, `1` tie, `0` losses.
- At `K=4`, the repeated fresh held-out comparison is effectively a tie:
  both rounds average `0.4000` verifier-ranked equivalence.
- Candidate generation quality is often good enough that a better selector should still be able to do better:
  oracle remains `0.5200` to `0.6200` across the development and held-out pools.
- Pairwise round 5 is implemented and trained, but the first hybrid recipe did
  not pass replay selection:
  it tied round 4 at `K=4` and lost at `K=8` on the clean round-4 held-out
  replay, and lost at both `K=4` and `K=8` on the round-3 held-out replay.
- Conservative ranking round 6 is also implemented and trained, but it failed
  replay against round 4:
  mean replay deltas were `-0.0267` at `K=4` and `-0.0178` at `K=8`.
- Fixed-round-4 selector analysis is also complete:
  no margin, top-gap, agreement, normalization, or index-penalty policy beat
  plain round-4 `verifier_ranked` on cached replay.
- Focused pointwise round 7 is complete and passed cached replay:
  mean `K=8` improved by `+0.0117`, while mean `K=4` tied round 4.
- Fresh multiseed evaluation is also complete:
  round 7 improved `K=4` mean by `+0.0133`, but tied `K=8` mean at `0.4200`.
- Round-7 fresh row analysis is complete:
  at `K=8`, round 7 helped `11` rows and hurt `11`; seed `56` had `6` selector
  losses with an equivalent candidate still available.
- Fresh identical-pool round-4-vs-round-7 comparison is complete:
  `K=4` regressed from `0.4067` to `0.3933`, while `K=8` only improved from
  `0.4400` to `0.4467`.
- Phase 3 cached planner/search ablation is complete:
  `solvable_then_verifier` tied `K=4` and improved `K=8` by only one row across
  `280` decisions, so no search policy was accepted.
- Round 7 is not promoted; round 4 remains the current best.

## Recommended Next Step

The highest-value next task is now:

- start the small repair-loop pilot; cached planner/search policies did not
  pass, and another blind verifier retrain is not justified

Why this matters:

- The promotion decision has now been made.
- The strongest positive evidence is at `K=8`, and the docs should say that plainly.
- Both round 5 and round 6 show that ranking-objective pressure alone is not
  yet producing a stronger selector.
- The fixed-model selector analysis shows that simple score-use heuristics are
  also not enough.
- Round 7 was the first post-round-4 training result to pass cached replay, but
  fresh generation did not preserve the `K=8` mean gain, and row-level analysis
  shows the remaining issue is not just pool luck.
- The identical-pool follow-up confirms that round 7's clean verifier gain is
  too small to justify promotion.
- The planner/search ablation shows solvability is not enough to identify
  semantic equivalence.

See `RECOMMENDATION.md` for the current project-level recommendation.

## External Tools

**Fast Downward** (classical planner) and **VAL** (plan validator) are C++ tools
best built under Linux/WSL. See `tools/README.md` for build instructions.

## Key References

- [Planetarium dataset](https://huggingface.co/datasets/BatsResearch/planetarium)
- [Planetarium code](https://github.com/BatsResearch/planetarium)
- [Fast Downward](https://github.com/aibasel/downward)
- [VAL](https://github.com/KCL-Planning/VAL)

Design background: `deep-research-report.md`. Contributor-oriented notes: `CLAUDE.md`. Experiment log: `EXPERIMENTS.md`.
