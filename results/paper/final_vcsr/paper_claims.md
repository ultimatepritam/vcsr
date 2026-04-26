# Paper Claim Sheet

## Recommended Main Claim

VCSR improves semantic equivalence for text-to-PDDL generation by combining
verifier-ranked best-of-K search with one-step domain-aware repair. The strongest
evidence is the final untouched seed gate (`51-55`) at `K=8`, where repair
raises semantic equivalence from `0.4200` to `0.7720`.

## What To Claim

- Semantic equivalence, not parseability or solvability, is the primary metric.
- Verifier-ranked search is useful as the scaffold for repair.
- Planner/solvability-only policies are weak semantic selectors.
- Repair-augmented VCSR is the paper-facing system.

## What Not To Claim

- Do not claim repair is uniformly harmless.
- Do not claim final results on `floor-tile` or unseen domains.
- Do not claim abstention is the final system mechanism.
- Do not promote guarded repair; it did not reduce blocksworld hurts.
- Do not tune anything on seeds `51-55`.
