# VCSR: Verifier-Calibrated Search and Repair for Text-to-PDDL

A research framework for faithful natural-language to PDDL translation using
calibrated semantic verification, best-of-K selection, and abstention-aware repair.

## Project Structure

```
configs/           YAML experiment configs (baseline, vcsr, neggen)
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
verifier/          (planned) Cross-encoder training, calibration
search/            (planned) Best-of-K, abstention, repair loop
scripts/           Entry points: baselines, negative generation, JSONL sampling
results/           Metrics, pilot verifier JSONL, run logs
tools/             External tool installs (Fast Downward, VAL)
```

## Quick Start

```bash
# 1. Create and activate environment
python -m venv .venv
# Windows:
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Planetarium
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
```

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

## External Tools

**Fast Downward** (classical planner) and **VAL** (plan validator) are C++ tools
best built under Linux/WSL. See `tools/README.md` for build instructions.

## Key References

- [Planetarium dataset](https://huggingface.co/datasets/BatsResearch/planetarium)
- [Planetarium code](https://github.com/BatsResearch/planetarium)
- [Fast Downward](https://github.com/aibasel/downward)
- [VAL](https://github.com/KCL-Planning/VAL)

Design background: `deep-research-report.md`. Contributor-oriented notes: `CLAUDE.md`.
