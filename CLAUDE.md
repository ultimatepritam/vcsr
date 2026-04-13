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
configs/               Experiment YAML configs (baseline.yaml, vcsr.yaml)
data/                  Dataset loading, caching, split logic
  planetarium_loader.py   Template-hash splitter for BatsResearch/planetarium
eval/                  Evaluation modules
  equivalence.py          Planetarium graph-equivalence wrapper, batch metrics, stratified reporting
generation/            LLM generation for PDDL candidates
  prompts.py              System/domain prompts, repair prompts, response extraction
pddl_utils/            PDDL tooling (renamed from pddl/ to avoid shadowing the pddl package)
  oracle_planner.py       Wrapper around Planetarium's built-in oracle planners
  planner.py              Fast Downward + VAL subprocess wrapper (native and WSL modes)
search/                (empty) Best-of-K selection, abstention, repair loop
verifier/              (empty) Semantic verifier model, training, calibration
scripts/               Runnable experiment entry points
  reproduce_baselines.py  Pipeline validation: oracle, perturbed, solvability baselines
results/               Saved experiment outputs
  baseline/baseline_results.json
tools/                 External tool setup (Fast Downward, VAL)
  setup_planning_tools.sh  One-command WSL build script
notebooks/             (empty) Exploratory Jupyter notebooks
```

**Important**: The `pddl/` directory in the report's proposed layout was renamed
to `pddl_utils/` because Planetarium depends on an installed `pddl` Python
package and a local `pddl/` directory shadows it.

## Dataset

**Planetarium** (`BatsResearch/planetarium` on HuggingFace)

| Split | Rows    | Domains                              |
|-------|---------|--------------------------------------|
| Train | 103,983 | blocksworld (64,719), gripper (39,264) |
| Val   | 25,992  | blocksworld (15,893), gripper (10,099) |
| Test  | 15,943  | blocksworld (4,993), gripper (4,778), floor-tile (6,172 -- OOD) |

Splitting uses **template-hash grouping** (by problem `name` field) to prevent
data leakage. Train/val are derived from the HF `train` split; test is the HF
`test` split as-is.

Each row has: `natural_language`, `problem_pddl`, `domain`, `init_is_abstract`,
`goal_is_abstract`, `is_placeholder`, `num_objects`, and others.

## Key Dependencies

- Python 3.12, PyTorch 2.5.1+cu121, transformers 5.5.3, datasets 4.8.4
- peft 0.18.1, scikit-learn 1.8.0, wandb
- **planetarium 0.1.0** (installed from `BatsResearch/planetarium` GitHub)
- See `requirements.txt` for full list

## Equivalence Evaluation

Planetarium provides graph-isomorphism-based semantic equivalence:
- `planetarium.builder.build(pddl_str)` -> ProblemGraph
- `planetarium.oracle.fully_specify(graph)` -> reduced graph with inferred predicates
- `planetarium.metric.equals(g1, g2)` -> bool (isomorphism check)
- `planetarium.evaluate(gold, candidate)` -> (parseable, solveable, equivalent)

The `eval/equivalence.py` module wraps this with:
- `check_equivalence_lightweight()` -- no planner, graph-only
- `check_equivalence_full()` -- uses Planetarium's full evaluate (planner optional)
- `evaluate_batch()` -- batch evaluation with aggregate metrics
- `stratified_report()` -- metrics broken down by domain and description style

## Baseline Results (Pipeline Validation)

Run: `python scripts/reproduce_baselines.py --max_samples 30`

| Baseline   | Parse | Equiv | Solvable | Notes |
|------------|-------|-------|----------|-------|
| Oracle     | 1.000 | 1.000 | --       | Gold PDDL = candidate; confirms eval pipeline works |
| Perturbed  | 1.000 | 0.667 | --       | Programmatic edits to gold; parseable but semantically broken |
| Solvability| --    | --    | 30/30    | Oracle planner on gold PDDL; confirms planner integration |

**Interpretation**: The perturbed baseline demonstrates the core research
problem -- PDDL that looks valid (100% parseable) can fail semantic equivalence
(only 67% equivalent). This is the gap VCSR targets. No real LLM generation
has been run yet; the perturbed baseline is a controlled proxy.

---

## 8-Week Timeline and Progress

### Phase 1: Foundations (Weeks 1-2, Apr 6-19) -- DONE

- [x] Reproduce Planetarium eval + baselines
- [x] Data splits (template-hash) + caching
- [x] Project skeleton, environment, dependencies
- [x] Planner wrappers (oracle planner works; FD/VAL WSL script ready)
- [x] End-to-end pipeline validation (oracle + perturbed + solvability baselines)

### Phase 2: Verifier (Weeks 2-4, Apr 13 - May 1) -- NEXT

- [ ] **Build negative generator pipeline** (Apr 13-19)
  - Generate K candidates per NL description using a prompted LLM
  - Label each candidate via Planetarium equivalence against gold
  - Create hard negatives via domain-aware perturbations of gold PDDL
  - Assemble (NL, PDDL, label) triples for verifier training
- [ ] **Train text cross-encoder verifier** (Apr 20-26)
  - DeBERTa-v3-base backbone, `[CLS] NL [SEP] PDDL [SEP]` -> sigmoid
  - AdamW, LR sweep {1e-5, 2e-5, 5e-5}, batch 16, max_len 512
  - Train on positives + hard negatives, early stop on equiv-AUC
- [ ] **Calibration + abstention curves** (Apr 27 - May 1)
  - Temperature scaling or isotonic regression on val set
  - Sweep tau in {0.5, 0.6, 0.7, 0.8, 0.9}
  - Report selective risk-coverage curves

### Phase 3: Search and Repair (Weeks 5-6, May 2-15)

- [ ] Best-of-K + top-M validation ablations
  - K in {1, 4, 8, 16, 32}
  - Compare: random selection, planner-valid-only, verifier-ranked
- [ ] 1-step repair loop + analysis
  - Low-confidence candidates get repair prompt
  - Measure lift from repair vs. abstain

### Phase 4: Paper and Release (Weeks 7-8, May 16-28)

- [ ] Write paper + figures + tables
- [ ] Final reproducibility pass + artifact release

---

## What To Work On Next

**Immediate next task**: Build the negative generator pipeline.

This means:
1. Pick a generator model (API-based like GPT-4/Claude for speed, or local
   7B-class model for cost). Set the model ID in `configs/vcsr.yaml`.
2. Write `generation/generator.py` -- sample K PDDL candidates per NL input
   using stochastic decoding (temperature=0.8, top_p=0.95).
3. Write `data/build_verifier_dataset.py` -- for each training example:
   - Generate K candidates
   - Parse-filter them
   - Label each via `eval/equivalence.py` (equivalent to gold or not)
   - Also generate programmatic perturbations of gold PDDL
   - Output: `(nl, pddl_candidate, label)` triples saved to disk
4. Then move to verifier training in `verifier/`.

## Conventions

- Config via YAML in `configs/`. All experiments should be reproducible from config + seed.
- Logging via Python `logging` module. Use `wandb` for experiment tracking.
- Metrics always reported stratified by domain and description style.
- Seeds pinned: 42 is default across all configs.
- PDDL package shadowing: never create a directory named `pddl/` at project root.

## Key External References

- Planetarium dataset: https://huggingface.co/datasets/BatsResearch/planetarium
- Planetarium code: https://github.com/BatsResearch/planetarium
- Fast Downward: https://github.com/aibasel/downward
- VAL validator: https://github.com/KCL-Planning/VAL
