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
- A Claude-family robustness benchmark supports the wrapper story: VCSR repair
  improves Haiku 4.5, Sonnet 4.5, and Opus 4.6 at `K=8`.

## What Not To Claim

- Do not claim repair is uniformly harmless.
- Do not claim final results on `floor-tile` or unseen domains.
- Do not claim abstention is the final system mechanism.
- Do not promote guarded repair; it did not reduce blocksworld hurts.
- Do not tune anything on seeds `51-55`.

## Secondary Robustness Result

The post-paper benchmark under `results/vcsr/model_benchmark/` uses seeds
`72-74`, `10` rows per seed, and the same frozen round-4 verifier. At `K=8`,
repair-augmented VCSR improves over prompt-only generation:

- Haiku 4.5: prompt-only `0.4000` -> VCSR `0.9000`
- Sonnet 4.5: prompt-only `0.5000` -> VCSR `0.9333`
- Opus 4.6: prompt-only `0.3667` -> VCSR `0.9000`

Verifier-ranked `K=8` remains a useful ablation for understanding the pipeline,
but it is not the headline comparison for this benchmark.

Use this as appendix/supporting evidence, not as replacement for the primary
untouched seed `51-55` result.
