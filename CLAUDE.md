# VCSR: Verifier-Calibrated Search and Repair for Text-to-PDDL

## Project Overview

Research project for a cs.AI arXiv paper. The core problem: LLMs generate PDDL
that parses and may even be solvable by a classical planner, yet **semantically
misrepresents** the intended task. We call this the "valid-but-wrong" gap.

**VCSR** addresses this by training a semantic verifier on Planetarium's
equivalence labels, using it for best-of-K selection, and adding calibrated
abstention to avoid verifier-induced failures at scale.

Full design rationale is in `deep-research-report.md`.

## Hardware

- Local: RTX 3080 (10 GB VRAM), Ryzen 5800x, Windows 10
- Cloud GPU available for burst (training sweeps, large-scale generation)
- WSL2 (Ubuntu 22.04) available for Linux-native tools

## Repository Layout

```
configs/               baseline.yaml, vcsr.yaml, neggen.yaml
data/
  planetarium_loader.py   Template-hash splitter; HF `natural_language`, `problem_pddl`, etc.
  verifier_dataset.py     JSONL assembler for (NL, PDDL, label) verifier examples
eval/
  equivalence.py          Lightweight equivalence; timed subprocess kill for long graphs
generation/
  prompts.py              NL→PDDL + repair prompts; markdown PDDL extraction
  sampler.py              Bedrock (Converse), OpenRouter, OpenAI, HuggingFace, MultiSampler
  perturbations.py        Programmatic hard negatives (swap goals, drop init, …)
pddl_utils/
  oracle_planner.py       Planetarium oracle `plan()` / solvability
  planner.py              Fast Downward + VAL subprocess (native / WSL)
scripts/
  reproduce_baselines.py  Oracle, perturbed, solvability checks
  generate_negatives.py   Full neggen pipeline: LLM + perturb + label + checkpoint/resume
  sample_verifier_jsonl.py  Stratified random lines from verifier_train.jsonl
search/                   (empty) Best-of-K, abstention, repair
verifier/                 (empty) Cross-encoder training, calibration
results/
  baseline/               Baseline JSON
  neggen/pilot/           Pilot verifier JSONL + run_log + stats (when generated)
tools/                    FD/VAL setup scripts
```

**Important**: Use `pddl_utils/`, not `pddl/`, at repo root — the PyPI `pddl` package must not be shadowed.

## Dataset

**Planetarium** (`BatsResearch/planetarium` on HuggingFace). Data is **not** vendored in git; it is **downloaded and cached** (e.g. under `~/.cache/huggingface/datasets` on Windows unless `cache_dir` is set).

| Split | Rows    | Domains                              |
|-------|---------|--------------------------------------|
| Train | 103,983 | blocksworld (64,719), gripper (39,264) |
| Val   | 25,992  | blocksworld (15,893), gripper (10,099) |
| Test  | 15,943  | blocksworld (4,993), gripper (4,778), floor-tile (6,172 — OOD) |

Splitting uses **template-hash grouping** (by problem `name` field) to prevent leakage.

Each row includes: `natural_language`, `problem_pddl`, `domain`, `name`, `num_objects`,
`init_is_abstract`, `goal_is_abstract`, `is_placeholder`, etc.

## Key Dependencies

- Python 3.12+, PyTorch, transformers, datasets, peft, scikit-learn, wandb
- **planetarium** (pip from BatsResearch GitHub or PyPI as available)
- **boto3** (Bedrock), **openai** (OpenRouter + OpenAI APIs), **python-dotenv**
- See `requirements.txt`

## Equivalence Evaluation

Planetarium graph-based equivalence can be **very slow** on large problems (isomorphism / `fully_specify`). The pipeline uses:

- **`check_equivalence_lightweight()`** — in-process; fast for small instances.
- **`check_equivalence_lightweight_timed()`** — runs in a **spawned subprocess** and **terminates** after `equivalence_timeout_sec` (see `configs/neggen.yaml`). On timeout, label **not equivalent** (conservative for negatives).

For `num_objects >= equivalence_subprocess_min_objects` (default **8**), the neggen script uses the **timed** path so Windows runs cannot hang for hours. Set `equivalence_subprocess_min_objects: 0` to always use subprocess+timeout (slower on Windows due to spawn cost).

**Bedrock note:** Some Anthropic Bedrock models reject **both** `temperature` and `top_p` in one call. `generation/sampler.py` sends **only one** unless `BEDROCK_USE_TOP_P=1` in `.env`.

## Baseline Results (Pipeline Validation)

Run: `python scripts/reproduce_baselines.py --max_samples 30`

| Baseline   | Parse | Equiv | Solvable | Notes |
|------------|-------|-------|----------|-------|
| Oracle     | 1.000 | 1.000 | --       | Gold = candidate |
| Perturbed  | 1.000 | 0.667 | --       | Programmatic edits; parseable but often not equivalent |
| Solvability | --   | --    | 30/30    | Oracle planner on gold |

## Negative Generator Pilot (500 rows, completed)

Config: `configs/neggen.yaml`. Generator: **Bedrock** (`BEDROCK_MODEL_ID`, e.g. Claude Haiku), **K=2** samples per row, perturbations per gold, **seed 42** subsample of train.

**Final** stats from the completed pilot (`results/neggen/pilot/run_stats.json`):

- **500** rows processed; **1000** LLM calls (K=2); **902** parseable; **417** labeled equivalent to gold (**~46%** of parseable).
- **999** perturbation candidates (one row produced fewer than two); **993** parseable; **2** labeled equivalent (rare false positives — worth filtering if needed).
- **72** equivalence timeouts (120s subprocess cap); **2499** total assembled examples; **2395** parseable training lines.
- **~8 h** wall time (`elapsed_sec` ≈ 28747 s). Frontier API cost is non-trivial; weaker local generators may yield more hard negatives at lower $/row.

**Sanity tooling:** `python scripts/sample_verifier_jsonl.py results/neggen/pilot/verifier_train.jsonl`

**Resume:** `python scripts/generate_negatives.py --config configs/neggen.yaml --resume` (uses `checkpoint.jsonl` every `checkpoint_every` rows).

## Lessons Learned (engineering)

- **Logging:** `tqdm` + piped stdout can look “stuck”; use `PYTHONUNBUFFERED=1`, per-row `logger` lines, and optional `Tee-Object` to `run_log.txt`.
- **PowerShell:** use `Set-Location ...; $env:VAR = "1"; ...` — not bash `&& $env:VAR=...` syntax.
- **Label noise:** rare `perturbation` + `label=1` pairs possible (e.g. `negate_goal` quirks); optional filter before training.

---

## Timeline and Progress

### Phase 1: Foundations (Weeks 1-2) — DONE

- [x] Planetarium eval + baselines
- [x] Template-hash splits
- [x] Project skeleton, dependencies
- [x] Planner wrappers (oracle; FD/VAL script)
- [x] End-to-end baseline validation

### Phase 2: Verifier (Weeks 2-4) — IN PROGRESS

- [x] **Negative generator pipeline** (`generation/sampler.py`, `perturbations.py`, `generate_negatives.py`, `verifier_dataset.py`, `configs/neggen.yaml`)
- [x] **500-row pilot** verifier JSONL (`results/neggen/pilot/`)
- [ ] **Train text cross-encoder verifier** (`verifier/` — DeBERTa or similar, see `configs/vcsr.yaml`)
- [ ] **Calibration + abstention** on val set

### Phase 3: Search and Repair (Weeks 5-6)

- [ ] Best-of-K + planner-filter ablations
- [ ] Repair loop + analysis

### Phase 4: Paper and Release (Weeks 7-8)

- [ ] Paper, figures, artifact

---

## What To Work On Next

1. **Verifier training** on `results/neggen/pilot/verifier_train.jsonl` (and/or expand data with a weaker local generator for harder negatives).
2. **Val/test** evaluation using **held-out Planetarium** rows (template-hash splits); do **not** train on test.
3. **Baselines:** greedy LLM, best-of-K random, planner-valid-only vs **verifier-ranked**.

## Conventions

- Config via YAML; reproducible from config + seed (default **42**).
- Logging via `logging`; optional `wandb`.
- Stratify metrics by domain and NL style where applicable.
- Never add a root-level directory named `pddl/` (shadows the `pddl` package).

## Key External References

- Planetarium dataset: https://huggingface.co/datasets/BatsResearch/planetarium
- Planetarium code: https://github.com/BatsResearch/planetarium
- Fast Downward: https://github.com/aibasel/downward
- VAL: https://github.com/KCL-Planning/VAL
