# Citation Plan

This file maps the paper narrative to the core references in
`references.bib`. Keep the citation spine focused: cite what the paper actually
uses, not every adjacent LLM-planning paper.

## Core Citation Spine

- `zuo2024planetarium`: primary benchmark, semantic equivalence metric, and the
  "parseable/solvable but semantically wrong" motivation.
- `planetarium_dataset` and `planetarium_code`: reproducibility references for
  the dataset and evaluation implementation.
- `ghallab1998pddl` and `fox2003pddl21`: PDDL / planning-language background.
- `helmert2006fastdownward` and `howey2004val`: planner and validator context;
  cite when discussing solvability/validation baselines.
- `gestrin2024nl2plan`: closest NL-to-PDDL planning-system related work.
- `oswald2024planningdomains`: LLM-generated planning domain models and
  automated evaluation against plan sets.
- `huang2024planningformalizers`: direct evidence that LLM formalization into
  PDDL is promising but degrades as descriptions become more natural.
- `silver2024generalizedplanning`: LLMs used with PDDL domains and feedback /
  debugging loops in generalized planning.
- `yu2025symbolicworldmodels`: test-time scaling / best-of-N style symbolic
  world-model generation with PDDL.
- `wang2023selfconsistency`: broad precedent for sampling multiple LLM outputs
  and selecting among them.
- `madaan2023selfrefine`: iterative feedback/refinement precedent for the
  repair stage.
- `guo2017calibration`: calibration background, especially if discussing the
  verifier calibration experiments or why raw verifier scores need caution.
- `geifman2017selective`: selective prediction / reject option and
  risk-coverage framing if abstention remains in the appendix.
- `he2021debertav3`: verifier backbone reference.

## Section-by-Section Citation Map

### Introduction

Use:

- `zuo2024planetarium`
- `ghallab1998pddl`
- `gestrin2024nl2plan`
- `huang2024planningformalizers`

Purpose:

- Establish text-to-PDDL as a natural-language-to-formal-planning task.
- Motivate semantic faithfulness as stricter than parseability or solvability.
- Position VCSR as a search-and-repair system for faithful formalization.

Suggested citation placement:

- First paragraph on PDDL: `\citep{ghallab1998pddl}`.
- Paragraph on valid-but-wrong failures: `\citep{zuo2024planetarium}`.
- Related systems using LLMs for PDDL/planning:
  `\citep{gestrin2024nl2plan,huang2024planningformalizers}`.

### Background and Related Work

Use:

- `zuo2024planetarium`
- `gestrin2024nl2plan`
- `oswald2024planningdomains`
- `huang2024planningformalizers`
- `silver2024generalizedplanning`
- `yu2025symbolicworldmodels`
- `wang2023selfconsistency`
- `madaan2023selfrefine`
- `helmert2006fastdownward`
- `howey2004val`

Purpose:

- Planetarium: benchmark and equivalence evaluation.
- NL2Plan: LLM-driven PDDL planning formalization.
- LLM planning-domain generation and formalization limits.
- Generalized planning / feedback loops in PDDL domains.
- Test-time scaling for symbolic world-model generation.
- Self-consistency: multi-sample generation / best-of-K motivation.
- Self-Refine: lightweight iterative repair/refinement precedent.
- Fast Downward / VAL: planner and validator context.

### Method

Use:

- `zuo2024planetarium`
- `he2021debertav3`
- `guo2017calibration`
- `geifman2017selective`
- `wang2023selfconsistency`
- `madaan2023selfrefine`

Purpose:

- Cite Planetarium for labels/equivalence.
- Cite DeBERTaV3 for the cross-encoder verifier backbone.
- Cite calibration if discussing calibrated scores or risk/coverage analysis.
- Cite selective classification only if including abstention/risk-coverage.
- Cite self-consistency or best-of-K sampling precedent when introducing
  candidate pools.
- Cite Self-Refine when introducing the repair step as test-time refinement.

### Evaluation

Use:

- `zuo2024planetarium`
- `planetarium_dataset`
- `planetarium_code`
- `helmert2006fastdownward`
- `howey2004val`

Purpose:

- Semantic equivalence metric and dataset splits.
- Planner/solvability ablations and why they are insufficient.
- Reproducibility details for code/data/tooling.

### Results and Analysis

Use:

- Usually cite fewer papers here. Results should mostly reference our tables.
- Cite `zuo2024planetarium` when interpreting semantic equivalence as the
  primary metric.
- Cite `guo2017calibration` only if discussing calibration or abstention
  experiments.

### Limitations and Future Work

Use:

- `zuo2024planetarium`
- `guo2017calibration`
- `geifman2017selective`
- `gestrin2024nl2plan`

Purpose:

- Planetarium domain limitations.
- Future calibration/abstention or stronger semantic validators.
- Broader planning formalization pipelines.

## References To Avoid Overusing

- Do not over-cite calibration in the final main claim. Calibration was useful
  during development, but repair-augmented VCSR is the paper-facing mechanism.
- Do not over-cite selective prediction unless abstention/risk-coverage appears
  in the final paper or appendix.
- Do not make VAL/Fast Downward sound like final correctness metrics. They are
  useful diagnostic baselines, while semantic equivalence remains primary.
- Do not cite the Claude-family model benchmark as external evidence; it is our
  appendix experiment.

## Missing Citation Checks Before Submission

- Verify final author/order metadata for Planetarium from the paper PDF.
- Verify whether NL2Plan has a venue update beyond arXiv.
- If we discuss OpenRouter/Claude model access in the methods, cite provider
  documentation in a footnote or artifact note rather than as a scholarly
  reference.
- If we include a final LaTeX version, ensure all `\citep{...}` keys resolve
  against `references.bib`.
