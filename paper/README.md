# OSF Wiki Draft: VCSR Paper

## Title

**When Valid Is Not Faithful: Verifier-Calibrated Search and Repair for Structured Generation**

## Short Summary

This project studies a central failure mode in structured generation with large
language models: outputs can be syntactically valid while still being
semantically wrong. We evaluate this problem in text-to-PDDL generation using
the Planetarium benchmark and introduce **VCSR** (Verifier-Calibrated Search
and Repair), a system that:

1. samples multiple candidate PDDL problems,
2. selects a promising parseable candidate with a learned semantic verifier,
3. repairs that candidate using only non-oracle feedback.

The main finding is that **search alone helps only modestly, while search plus
repair produces a large gain in semantic faithfulness**.

## Main Result

Our primary paper-facing evidence is the final fresh evaluation on untouched
seeds `51-55` over the Planetarium `blocksworld` and `gripper` domains.

At the main operating point (`K=8`):

- prompt-only generation reaches **0.368** semantic equivalence,
- verifier-ranked search reaches **0.420**,
- full repair-augmented VCSR reaches **0.772**.

This is a gain of **+40.4 percentage points** over prompt-only generation.

## Final Evidence

The frozen final evidence for the paper is stored in:

- `../results/vcsr/final_repair_gate_round4/final_repair_gate_summary.md`
- `../results/vcsr/final_repair_gate_round4/final_repair_gate_summary.json`

That final gate uses:

- untouched seeds `51-55`,
- `50` rows per seed,
- the frozen round-4 verifier recorded in
  `../results/verifier/best_current/selection.yaml`.

Headline final-gate result:

- round-4 verifier-ranked at `K=8`: **0.4200**
- repair-augmented VCSR at `K=8`: **0.7720**

## What the Paper Claims

The paper supports the following scoped claim:

- in in-domain Planetarium text-to-PDDL generation for `blocksworld` and
  `gripper`, faithful generation benefits substantially from combining
  verifier-guided search with one-step repair.

The paper does **not** claim:

- broad out-of-domain generalization,
- a final result for `floor-tile`,
- that a stronger verifier alone solved the problem,
- that repair is uniformly harmless.

## Important Caveat

Repair is a large net win overall, but it is not uniformly beneficial.
In the final evaluation, repair is especially strong on `gripper`, while some
already-correct `blocksworld` candidates are harmed by unconditional repair.
A later guarded-repair follow-up reproduced the overall gain but did not remove
that caveat, so it is not the promoted final system.

Related follow-up artifact:

- `../results/vcsr/final_guarded_repair_gate_round4/final_repair_gate_summary.md`

## Repository Pointers

Key paper and artifact files:

- `main.tex`: LaTeX source for the manuscript
- `../references.bib`: bibliography
- `../PAPER_PLAN.md`: frozen claim hierarchy and paper plan
- `../results/paper/final_vcsr/`: exported paper tables and figure specs
- `../results/verifier/best_current/selection.yaml`: selected round-4 verifier
- `../results/vcsr/final_repair_gate_round4/`: primary final evidence

## Reproducibility Notes

During development, the project used cached-pool replay to compare verifier and
selection-policy changes without conflating them with generation randomness.
Those analyses were useful for rejecting several alternatives, including
additional verifier-training rounds and planner/solvability-only selection
policies.

The main paper-facing result is intentionally anchored to the untouched
final-seed gate above rather than to development-time replay evidence alone.

## Local Build

To compile the manuscript locally from this directory:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Contact

Author: **Pritam Mondal**  
Email: `ultimatepritam@hotmail.com`
