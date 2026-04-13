# VCSR: Verifier-Calibrated Search and Repair for Text-to-PDDL

A research framework for faithful natural-language to PDDL translation using
calibrated semantic verification, best-of-K selection, and abstention-aware repair.

## Project Structure

```
configs/        YAML experiment configurations
data/           Dataset loaders, caching, split definitions
pddl/           PDDL parsing, normalization, planner/validator wrappers
generation/     LLM prompts, sampling, constrained decoding
verifier/       Semantic verifier model, training, calibration
search/         Best-of-K selection, abstention logic, repair loop
eval/           Planetarium equivalence wrapper, metrics, reporting
scripts/        Reproducible experiment entry points
results/        Saved metrics, tables, plots, artifacts
tools/          External tool installs (Fast Downward, VAL)
notebooks/      Exploratory Jupyter notebooks
```

## Quick Start

```bash
# 1. Create and activate environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/WSL:
# source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Planetarium
pip install git+https://github.com/BatsResearch/planetarium.git

# 4. Verify dataset loads
python -c "from data.planetarium_loader import PlanetariumDataset; ds = PlanetariumDataset(); print(ds.summary())"

# 5. Run baseline reproduction
python scripts/reproduce_baselines.py --config configs/baseline.yaml
```

## External Tools

**Fast Downward** (classical planner) and **VAL** (plan validator) are C++ tools
best built under Linux/WSL. See `tools/README.md` for build instructions.

## Key References

- [Planetarium dataset](https://huggingface.co/datasets/BatsResearch/planetarium)
- [Planetarium code](https://github.com/BatsResearch/planetarium)
- [Fast Downward](https://github.com/aibasel/downward)
- [VAL](https://github.com/KCL-Planning/VAL)
