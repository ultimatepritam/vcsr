# When Valid Is Not Faithful: Verifier-Calibrated Search and Repair for Structured Generation

[![DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2F8TJMV-blue)](https://doi.org/10.17605/OSF.IO/8TJMV)

This directory contains the manuscript and paper-facing artifacts for *When
Valid Is Not Faithful: Verifier-Calibrated Search and Repair for Structured
Generation*.

The paper studies a failure mode in LLM-based structured generation: an output
can be syntactically valid while still encoding the wrong meaning. We evaluate
this problem in text-to-PDDL generation using the Planetarium benchmark and
introduce Verifier-Calibrated Search and Repair (VCSR), a system that samples
candidate PDDL problems, selects a promising parseable candidate with a learned
semantic verifier, and repairs that candidate using non-oracle feedback.

## Main Result

The primary paper evidence is the final fresh evaluation on untouched seeds
`51-55` over the Planetarium `blocksworld` and `gripper` domains.

At the main `K=8` operating point:

- prompt-only generation reaches **0.368** semantic equivalence,
- verifier-ranked search reaches **0.420**,
- repair-augmented VCSR reaches **0.772**.

This is a **+40.4 percentage point** improvement over prompt-only generation.
The result supports the paper's main conclusion: verifier-guided search is a
useful scaffold, but the large gain comes from adding repair.

## Final Evidence

The frozen final evidence is stored in:

- `../results/vcsr/final_repair_gate_round4/final_repair_gate_summary.md`
- `../results/vcsr/final_repair_gate_round4/final_repair_gate_summary.json`

The selected verifier is recorded in:

- `../results/verifier/best_current/selection.yaml`

That metadata points to the promoted round-4 verifier checkpoint:

- `../results/verifier/ranking_aligned_round4/retrain_from_round3_focused/best_model/model.pt`

## Scope and Caveats

The final claim is intentionally scoped to in-domain Planetarium
`blocksworld` and `gripper` text-to-PDDL generation. The paper does not claim a
final result for `floor-tile`, broad out-of-domain generalization, or that a
stronger verifier alone solves semantic faithfulness.

Repair is a large net win overall, especially on `gripper`, but it can hurt
some already-correct `blocksworld` candidates. A guarded-repair follow-up
reproduced the overall gain but did not remove that caveat, so the promoted
paper-facing system remains the final repair-augmented VCSR gate above.

Related follow-up artifact:

- `../results/vcsr/final_guarded_repair_gate_round4/final_repair_gate_summary.md`

## Files

- `main.tex`: LaTeX source for the manuscript
- `VCSR.pdf`: compiled manuscript PDF
- `../references.bib`: bibliography
- `../PAPER_PLAN.md`: paper plan and frozen claim hierarchy
- `../results/paper/final_vcsr/`: exported paper tables and figure specs
- `../results/vcsr/final_repair_gate_round4/`: primary final evidence

## Build

To compile the manuscript locally from this directory:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Contact

Pritam Mondal  
`ultimatepritam@hotmail.com`
