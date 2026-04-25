# Fresh Fixed-Pool Repair Gate

Base config: `configs/vcsr_bestofk_round3_holdout_eval.yaml`
Rows per seed: `50`
Seeds: `[62, 63, 64]`

Generation happened once per seed. Round-4 selected failures were repaired once on the same candidate pools.

## Mean K=8 Equivalence

| Policy | Mean Equiv | Per-Seed Values |
|---|---:|---|
| Round-4 verifier-ranked | 0.5000 | 0.5000, 0.5000, 0.5000 |
| Repair-augmented | 0.9600 | 1.0000, 0.9000, 0.9800 |

## Repair Outcomes

| Seed | Baseline K=8 | Repair-Aug K=8 | Delta | Failures Repaired | Helped | Hurt | Tied | Parse Rate | Repair Eq |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 62 | 0.5000 | 1.0000 | +0.5000 | 26 | 25 | 0 | 1 | 1.0000 | 0.9615 |
| 63 | 0.5000 | 0.9000 | +0.4000 | 23 | 20 | 0 | 3 | 1.0000 | 0.8696 |
| 64 | 0.5000 | 0.9800 | +0.4800 | 24 | 24 | 0 | 0 | 1.0000 | 1.0000 |

## Breakdown Over Repaired Failures

| Slice | Rows | Repair Eq | Helped | Hurt |
|---|---:|---:|---:|---:|
| domain=blocksworld | 10 | 0.8000 | 8 | 0 |
| domain=gripper | 63 | 0.9683 | 61 | 0 |
| style=abstract/abstract | 43 | 0.9302 | 40 | 0 |
| style=explicit/explicit | 30 | 0.9667 | 29 | 0 |

## Recommendation

Repair passes the fresh fixed-pool gate; implement repair-augmented selection in the main best-of-K entrypoint next.
