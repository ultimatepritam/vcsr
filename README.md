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
- Clean calibration analysis and a small learning-rate sweep have also been completed.
- The current selected verifier checkpoint is recorded in `results/verifier/best_current/selection.yaml`.
- The main remaining gap is downstream integration into verifier-ranked best-of-K, abstention, and later repair experiments.

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
- `scripts/train_verifier.py`
- `scripts/analyze_verifier.py`
- `scripts/calibrate_verifier.py`
- `scripts/run_verifier_lr_sweep.py`
- `scripts/run_verifier_capacity_push.py`
- `scripts/mine_verifier_hard_negatives.py`

Current key verifier artifacts:

| Path | Description |
|------|-------------|
| `results/verifier/pilot/` | Earlier dry-run / smoke-test outputs |
| `results/verifier/full_run/` | First completed verifier training run |
| `results/verifier/lr_sweep/` | LR sweep runs plus aggregate summaries |
| `results/verifier/best_current/selection.yaml` | Stable metadata record for the current best verifier checkpoint |

As of the current repo state, the selected best verifier comes from:

- run: `results/verifier/lr_sweep/lr_5em05`
- checkpoint: `results/verifier/lr_sweep/lr_5em05/best_model/model.pt`

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

## External Tools

**Fast Downward** (classical planner) and **VAL** (plan validator) are C++ tools
best built under Linux/WSL. See `tools/README.md` for build instructions.

## Key References

- [Planetarium dataset](https://huggingface.co/datasets/BatsResearch/planetarium)
- [Planetarium code](https://github.com/BatsResearch/planetarium)
- [Fast Downward](https://github.com/aibasel/downward)
- [VAL](https://github.com/KCL-Planning/VAL)

Design background: `deep-research-report.md`. Contributor-oriented notes: `CLAUDE.md`. Experiment log: `EXPERIMENTS.md`.
