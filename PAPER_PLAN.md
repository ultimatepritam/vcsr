# Paper Plan: VCSR for Faithful Text-to-PDDL

Working title: **Verifier-Calibrated Search and Repair for Faithful Text-to-PDDL Generation**

## Core Claim

VCSR targets semantic faithfulness, not just parseability or planner solvability.
The paper-facing system combines frozen round-4 verifier-ranked best-of-K search
with one domain-aware repair call at the main `K=8` operating point.

Main claim:

> VCSR improves semantic equivalence for text-to-PDDL generation by combining
> verifier-ranked best-of-K search with one-step domain-aware repair. Verifier
> ranking alone is useful but not sufficient; repair is the decisive
> system-level improvement.

## Claim Hierarchy

- **Main system claim:** Repair-augmented VCSR improves semantic equivalence over
  greedy prompting, random parseable best-of-K, planner/solvability search, and
  verifier-only search at `K=8`.
- **Ablation claim:** `verifier_ranked` is the immediate pre-repair scaffold, so
  repair versus verifier-ranked isolates the repair contribution.
- **Negative-result claim:** Planner/solvability-only search and additional
  verifier training rounds did not reliably solve semantic selection.
- **Caveat:** Repair is a large net win but not uniformly harmless:
  it strongly fixes gripper while it can hurt already-correct blocksworld
  candidates.
- **Robustness claim:** A post-paper Claude-family benchmark suggests VCSR is a
  useful wrapper across stronger and weaker Claude generators, not only the
  Haiku generator used for the primary frozen result.

## Paper Structure

1. **Introduction**
   - Motivate the "valid-but-wrong" gap in text-to-PDDL.
   - Present VCSR as generate, verify, and repair.
   - Contributions: semantic verifier, replay methodology, search ablations,
     and final repair-augmented fresh-seed result.

2. **Method**
   - Task: Planetarium in-domain `blocksworld` and `gripper`.
   - Generator: OpenRouter/Claude Haiku, best-of-`K`, `K in {1,4,8}`.
   - Verifier: frozen round-4 DeBERTa cross-encoder from
     `results/verifier/best_current/selection.yaml`.
   - Search: parseable candidate ranking by verifier score.
   - Repair: one domain-aware repair call for the verifier-selected `K=8`
     candidate, without gold PDDL or equivalence labels in the prompt.
   - Evaluation: parse rate, semantic equivalence, equivalence-given-parse,
     domain/style slices, and helped/hurt/tied counts.

3. **Main Results**
   - Primary evidence: untouched seeds `51-55` under
     `results/vcsr/final_repair_gate_round4`.
   - Main `K=8` equivalence:
     - `greedy_first`: `0.3680`
     - `random_parseable`: `0.3760`
     - `verifier_ranked`: `0.4200`
     - `verifier_ranked_repair`: `0.7720`
   - Repair parse rate: `0.9840`.
   - Helped / hurt / tied: `104 / 16 / 130`.
   - Absolute lift over prompt-only / greedy generation: `+0.4040`.
   - Verifier-only search remains the immediate pre-repair ablation:
     `0.4200 -> 0.7720`.

4. **Ablations and Analysis**
   - Verifier-only progression: round 4 is the frozen verifier; rounds 5-7 are
     diagnostics and are not promoted.
   - Search ablation: solvability policies did not beat round-4
     `verifier_ranked` on cached replay.
   - Repair development: cached repair pilot, generic fresh repair gate,
     domain-aware repair gate, and final seed gate.
   - Guarded repair: replicated large gains on seeds `67-71`, but did not
     reduce blocksworld hurts, so it is not promoted.
   - Model robustness appendix: Claude Haiku 4.5, Sonnet 4.5, and Opus 4.6 all
     improve substantially under repair-augmented VCSR at `K=8`.

5. **Limitations**
   - Final claim is in-domain only: `blocksworld` and `gripper`.
   - No final claim is made for `floor-tile`.
   - Repair benefit is asymmetric:
     - `gripper`: verifier-ranked `2/124`, repair `98/124`.
     - `blocksworld`: verifier-ranked `103/126`, repair `95/126`.
   - Abstention/calibration is supporting evidence, not the final mechanism.
   - Future work: structural repair acceptance checks, abstention, richer
     semantic validators, and additional domains.

## Evidence Freeze

Do not tune prompts, checkpoints, guards, or selection rules using final seeds.
The paper-facing final evidence is frozen at:

- `results/vcsr/final_repair_gate_round4/final_repair_gate_summary.json`
- `results/vcsr/final_repair_gate_round4/final_repair_gate_summary.md`
- per-seed `aggregate_metrics.json`, `candidate_dump.jsonl`, and
  `repair_outputs.jsonl`

Supporting evidence:

- `results/vcsr/search_ablation_round4/search_ablation_summary.md`
- `results/vcsr/fixed_pool_round7_compare/comparison_summary.md`
- `results/vcsr/repair_pilot_round4/repair_summary.md`
- `results/vcsr/fresh_repair_gate_round4_domainaware/fresh_repair_gate_summary.md`
- `results/vcsr/final_guarded_repair_gate_round4/final_repair_gate_summary.md`
- `results/vcsr/model_benchmark/benchmark_summary.md`

Use `python scripts/export_paper_artifacts.py` to regenerate paper tables and
figure specs from frozen artifacts only.

## Post-Paper Robustness Benchmark

The multi-model benchmark in `configs/vcsr_model_benchmark.yaml` and
`scripts/run_model_benchmark.py` compares prompt-only generation, random
parseable best-of-K, verifier-ranked best-of-K, and repair-augmented VCSR across
OpenRouter models. It has been run for Claude Haiku 4.5, Claude Sonnet 4.5,
and Claude Opus 4.6 on seeds `72-74`, `10` rows per seed.

Main `K=8` result, using the paper's prompt-only vs VCSR framing:

| Model | Prompt K=1 | VCSR Repair K=8 | Delta vs Prompt | Verifier K=8 Ablation |
|---|---:|---:|---:|---:|
| Claude Haiku 4.5 | `0.4000` | `0.9000` | `+0.5000` | `0.4667` |
| Claude Sonnet 4.5 | `0.5000` | `0.9333` | `+0.4333` | `0.5333` |
| Claude Opus 4.6 | `0.3667` | `0.9000` | `+0.5333` | `0.4333` |

In the write-up, use a concrete plain-English reading, e.g. with prompt-only
Opus, about `36.7%` of rows were semantically correct; with full VCSR around
Opus, `90.0%` were semantically correct, a `+53.3` percentage-point gain on
this small benchmark.

Treat this as appendix/robustness evidence, not as a replacement for the frozen
final seed `51-55` result.
