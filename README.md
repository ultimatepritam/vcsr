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
- The main open question is now modeling improvement:
  how do we improve beyond the promoted round-4 verifier while staying honest
  that the strongest repeated end-to-end evidence is at `K=8` and the `K=4`
  story remains more mixed?

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
- `scripts/train_verifier.py`
- `scripts/analyze_verifier.py`
- `scripts/calibrate_verifier.py`
- `scripts/run_verifier_lr_sweep.py`
- `scripts/run_verifier_capacity_push.py`
- `scripts/mine_verifier_hard_negatives.py`
- `scripts/mine_verifier_ranking_examples.py`
- `scripts/prepare_ranking_round3_dataset.py`

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

## Recommended Next Step

The highest-value next task is now:

- move to the next modeling improvement with round 4 as the promoted default,
  while keeping the write-up explicit that the strongest evidence is at `K=8`

Why this matters:

- The promotion decision has now been made.
- The strongest positive evidence is at `K=8`, and the docs should say that plainly.
- The next experiment should be a stronger ranking objective rather than another blind retrain.

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
