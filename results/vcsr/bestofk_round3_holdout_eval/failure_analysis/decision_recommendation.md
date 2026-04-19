# Round-4 Decision Recommendation

- Recommended path: `focused_round4`
- Rationale: Most residual oracle-positive misses are still within-pool misrankings concentrated in blocksworld, especially abstract/abstract rows, with score gaps small enough to justify a targeted next mining round rather than an immediate objective change.

## Evidence

- Oracle-positive verifier misses considered: `9`
- Blocksworld miss ratio: `1.00`
- Abstract/abstract miss ratio: `0.78`
- Moderate-gap miss ratio: `1.00`
- Comparison recurrence support: `True`

## Next Experiment Definition

- Keep the DeBERTa cross-encoder backbone and current inference stack.
- Do not run a broad LR or batch sweep.
- Mine additional examples only from held-out-like blocksworld rows, especially abstract/abstract rows.
- Prioritize rows where a verifier-selected wrong candidate outranks an equivalent candidate.
- Prioritize rows with multiple equivalent candidates plus one or more high-scoring wrong candidates.
- Emphasize near-tie ranking negatives and keep replay as the checkpoint-selection rule.

Recommended next modeling path: focused_round4
