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
  train_verifier.py       Cross-encoder verifier training entry point
  analyze_verifier.py     Threshold / score analysis for trained verifier
  calibrate_verifier.py   Separate calibration/evaluation protocol + risk-coverage curves
  run_verifier_lr_sweep.py  Execute LR sweep and summarize verifier runs
  sample_verifier_jsonl.py  Stratified random lines from verifier_train.jsonl
search/                   (currently minimal) Best-of-K, abstention, repair
verifier/                 Dataset/model/train/eval code for cross-encoder verifier
results/
  baseline/               Baseline JSON
  neggen/pilot/           Pilot verifier JSONL + run_log + stats (when generated)
  verifier/pilot/         Dry-run / smoke-test verifier outputs (not a completed experiment)
  verifier/full_run/      First completed verifier training run + calibration analysis
  verifier/lr_sweep/      LR sweep runs, clean calibration reports, sweep summary
  verifier/best_current/  Stable pointer to the currently selected verifier checkpoint
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
- **Label noise:** rare `perturbation` + `label=1` pairs — use `labeling.perturbation_positive_policy` (`relabel` default, or `drop` / `keep`) in `configs/neggen.yaml`, or `scripts/apply_perturbation_label_policy.py` on existing JSONL.

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
- [x] **Verifier training code scaffold** (`scripts/train_verifier.py`, `verifier/`, `configs/verifier.yaml`)
- [x] **Run full text cross-encoder verifier experiment** (`results/verifier/full_run/`)
- [x] **Calibration analysis with separate calibration/evaluation split** (`scripts/calibrate_verifier.py`, `results/verifier/full_run/calibration_report.json`)
- [x] **Learning-rate sweep for verifier** (`results/verifier/lr_sweep/`)
- [x] **Select current best verifier checkpoint** (`results/verifier/best_current/selection.yaml`)
- [x] **Run first verifier-ranked best-of-K pilots**
- [x] **Isolate verifier impact with fixed-pool downstream evaluation**
- [x] **Move from pointwise verifier training toward ranking-aligned supervision**
- [x] **Freeze ranking-aligned round 2 as the current downstream baseline**
- [x] **Run robustness-focused multi-pool ranking-aligned round 3**
- [x] **Promote round 3 to `best_current` based on replay wins**
- [x] **Run fresh held-out end-to-end best-of-K evaluation with frozen round 3**
- [x] **Run held-out failure analysis and focused round-4 mining**
- [x] **Train focused ranking-aligned round 4 from the frozen round-3 baseline**
- [x] **Run fresh held-out end-to-end best-of-K evaluation with focused round 4**
- [x] **Run repeated fresh held-out comparison for round 3 vs round 4**
- [x] **Analyze fixed-round-4 selector policies without further training**
- [x] **Train focused pointwise round 7 from promoted round 4 and run cached replay gate**
- [x] **Run fresh multiseed round 4 vs round 7 gate**
- [x] **Analyze round-7 fresh gate row-level wins/losses**
- [x] **Run fresh identical-pool round 4 vs round 7 verifier comparison**
- [x] **Run Phase 3 cached planner/search ablation with round 4 frozen**
- [x] **Run Phase 3 cached repair pilot with round 4 frozen**
- [x] **Run Phase 3 fresh fixed-pool repair gate**
- [x] **Run Phase 3 domain-aware gripper repair prompt pilot**
- [x] **Run Phase 3 same-pool domain-aware repair gate**

### Phase 3: Search and Repair (Weeks 5-6)

- [x] Best-of-K + planner-filter ablations
- [x] Cached repair loop + analysis
- [x] Fresh fixed-pool repair gate
- [x] Domain-aware repair iteration
- [x] Integrate repair-augmented best-of-K selection
- [x] Run final fresh repair-augmented VCSR gate on untouched seeds `51-55`

### Phase 4: Paper and Release (Weeks 7-8)

- [ ] Paper, figures, artifact

---

## What To Work On Next

1. **Prepare paper tables and write-up**
   Final repair-augmented VCSR passed the untouched seed gate: mean `K=8`
   equivalence improved from `0.4200` with plain round-4 `verifier_ranked` to
   `0.7720` with `verifier_ranked_repair` on seeds `51-55`.
2. **Replay remains the checkpoint-selection rule**
   Continue to judge new verifier checkpoints primarily by replay on cached
   pools, not by offline AUC alone.
3. **Preserve provenance**
   Never reuse pool output directories; every long-running generation or
   training run must have its own output directory and visible `progress.log`.
4. **Keep the distinction clear**
   Rounds 5 and 6 failed because they pushed pairwise/ranking loss. Round 7 is
   different: it is a larger round-4-style focused pointwise retrain, and it
   passed cached replay but did not pass the fresh promotion gate.
5. **Do not tune on final seeds**
   Seeds `51-55` are now final evidence. Do not use them for prompt edits,
   checkpoint selection, selector-policy design, or repair gating.

## Current Status Notes

- The negative-generation pilot under `results/neggen/pilot/` is the completed data milestone for Phase 2.
- `results/verifier/pilot/` should still be treated as dry-run / debugging output from the earlier smoke-test stage.
- A completed verifier training run now exists under `results/verifier/full_run/`, along with threshold analysis and a cleaner calibration/evaluation report.
- We have now completed fixed-pool replay on multiple cached pools and verified that the ranking-aligned round-3 checkpoint is the current official best downstream verifier.
- `results/verifier/best_current/selection.yaml` now points to `results/verifier/ranking_aligned_round4/retrain_from_round3_focused`.
- The fresh held-out end-to-end runs under `results/vcsr/bestofk_round3_holdout_eval/` and `results/vcsr/bestofk_round4_holdout_eval_clean/` are complete.
- Round 4 improved over round 3 on replay and also improved fresh held-out `verifier_ranked` at both `K=4` and `K=8`.
- But round 4 still lost to `greedy_first` at `K=4` and to `random_parseable` at `K=8` on the fresh 50-row held-out sample.
- We have now also completed the repeated fresh held-out comparison under `results/vcsr/multiseed_holdout_compare/`.
- That multi-seed gate shows the strongest new round-4 evidence at `K=8`:
  mean `verifier_ranked` equivalence improves from `0.4000` to `0.4267`, with
  seed-wise results `2` wins, `1` tie, `0` losses.
- At `K=4`, the same gate is effectively tied:
  mean `verifier_ranked` equivalence is `0.4000` for both round 3 and round 4.
- The main uncertainty is no longer "can a verifier help downstream selection?" It can.
- Round 4 has now been promoted as the default verifier artifact in repo metadata.
- The main uncertainty is now how to improve beyond round 4 without overselling the result, given that the strongest gain is at `K=8` and the `K=4` story remains mixed.
- Pairwise round 5 is implemented under `results/verifier/pairwise_round5/`:
  it mines same-row equivalent-vs-non-equivalent candidate pairs, trains a
  hybrid pairwise + pointwise DeBERTa verifier from round 4, and writes normal
  progress/model/calibration artifacts.
- The first hybrid pairwise round-5 recipe is **not promoted**:
  replay tied round 4 at `K=4` and regressed at `K=8` on
  `bestofk_round4_holdout_eval_clean`, and regressed at both `K=4` and `K=8`
  on `bestofk_round3_holdout_eval`.
- Keep `results/verifier/best_current/selection.yaml` pointed at round 4 unless
  a later replay/fresh gate clearly beats it.
- Conservative ranking round 6 is also implemented under
  `results/verifier/ranking_round6/`. It warm-starts from round 4 with
  pointwise-dominant hybrid ranking loss and a larger cached-pool dataset, but
  it failed replay against round 4 and is not promoted.
- Fixed-round-4 selector analysis is implemented under
  `results/vcsr/round4_selection_analysis/`. It tested margin fallback,
  top-gap fallback, round-3/round-4 agreement fallback, score normalization, and
  index-penalized ranking without changing verifier weights. No policy beat
  plain round-4 `verifier_ranked` on cached replay.
- Focused pointwise round 7 is implemented under
  `results/verifier/focused_round7/`. It mined `788` pointwise examples from
  cached pools, warm-started from promoted round 4, and used pure pointwise
  training. Cached replay against round 4 passed:
  `K=4` tied at `0.5050`, and `K=8` improved from `0.5167` to `0.5283`.
- Fresh multiseed round-4-vs-round-7 evaluation is complete under
  `results/vcsr/multiseed_round7_compare/`. Round 7 improved mean `K=4`
  (`0.4000 -> 0.4133`) but tied mean `K=8` (`0.4200 -> 0.4200`), so it is not
  promoted.
- Round-7 fresh gate analysis is complete under
  `results/vcsr/multiseed_round7_compare/fresh_gate_analysis/`. At `K=8`,
  round 7 helped `11` rows and hurt `11` rows across `150` rows. The seed `56`
  regression was mostly selector loss with equivalents still present (`6` of
  `7` hurt rows), not just missing candidate-pool oracle availability.
- Fresh identical-pool round-4-vs-round-7 comparison is complete under
  `results/vcsr/fixed_pool_round7_compare/`. On the same generated candidate
  pools, round 7 regressed `K=4` (`0.4067 -> 0.3933`) and only slightly
  improved `K=8` (`0.4400 -> 0.4467`), so it is not promoted.
- Phase 3 cached search ablation is complete under
  `results/vcsr/search_ablation_round4/`. Simple solvability-based policies did
  not pass the cached gate. `solvable_then_verifier` improved mean `K=8` only
  from `0.4714` to `0.4750`, and the gain came from just one row / one pool.
- Phase 3 cached repair pilot is complete under
  `results/vcsr/repair_pilot_round4/`. It kept round 4 frozen, selected cached
  `K=8` failures where round-4 `verifier_ranked` chose parseable
  non-equivalent PDDL, and generated one repair per row. Repair parse rate was
  `0.9667`, repair equivalence was `0.7667`, and helped / hurt / tied was
  `23 / 0 / 7`.
- Phase 3 fresh fixed-pool repair gate is complete under
  `results/vcsr/fresh_repair_gate_round4/`. It generated fresh pools once for
  seeds `62`, `63`, and `64`, then repaired round-4 `K=8` selected failures on
  those same pools. Mean `K=8` improved from `0.5000` to `0.5467`, with
  helped / hurt / tied `7 / 0 / 66`, but it did not pass the stricter
  acceptance gate. All helped rows were blocksworld; gripper repair was
  `0 / 63`.
- Gripper-specific repair prompt pilot is complete under
  `results/vcsr/gripper_repair_prompt_pilot/`. The generic gripper failure was
  mainly a Planetarium schema mismatch: typed objects and missing unary
  `(room ...)`, `(ball ...)`, and `(gripper ...)` facts. The domain-specific
  prompt repaired `61 / 63` gripper failures with parse rate `1.0000`.
- Phase 3 same-pool domain-aware repair gate is complete under
  `results/vcsr/fresh_repair_gate_round4_domainaware/`. It reused the exact
  candidate pools from seeds `62`, `63`, and `64` and improved mean `K=8` from
  `0.5000` to `0.9600`. Helped / hurt / tied was `69 / 0 / 4`.
- Final repair-augmented VCSR gate is complete under
  `results/vcsr/final_repair_gate_round4/`. It ran untouched seeds `51-55` and
  improved mean `K=8` from plain round-4 `verifier_ranked` at `0.4200` to
  `verifier_ranked_repair` at `0.7720`. Repair parse rate was `0.9840`, and
  helped / hurt / tied was `104 / 16 / 130`.
- `results/verifier/best_current/selection.yaml` should remain pointed at round
  4.
- Current next-step bias: do not blindly retrain or add more simple selector
  heuristics. The next work should be paper-ready analysis: tables, figures,
  domain/style breakdowns, and a clear caveat that unconditional repair helps
  gripper dramatically but can hurt some already-correct blocksworld rows.

## Long-Run Visibility Rule

- Any long-running generation, mining, or training job must write visible file-based progress artifacts.
- For pool generation, use output directories that contain at least `progress.log` and `progress.json`.
- For verifier training, use output directories that contain `progress.log` and `progress.json` in addition to normal model and metrics artifacts.
- Do not rely on hidden shell output alone for expensive jobs.

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
