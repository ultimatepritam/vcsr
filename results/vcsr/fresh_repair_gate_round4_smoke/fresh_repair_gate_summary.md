# Fresh Fixed-Pool Repair Gate

Base config: `configs/vcsr_bestofk_round3_holdout_eval.yaml`
Rows per seed: `5`
Seeds: `[65]`

Generation happened once per seed. Round-4 selected failures were repaired once on the same candidate pools.

## Mean K=8 Equivalence

| Policy | Mean Equiv | Per-Seed Values |
|---|---:|---|
| Round-4 verifier-ranked | 0.2000 | 0.2000 |
| Repair-augmented | 0.4000 | 0.4000 |

## Repair Outcomes

| Seed | Baseline K=8 | Repair-Aug K=8 | Delta | Failures Repaired | Helped | Hurt | Tied | Parse Rate | Repair Eq |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 65 | 0.2000 | 0.4000 | +0.2000 | 2 | 1 | 0 | 1 | 1.0000 | 0.5000 |

## Breakdown Over Repaired Failures

| Slice | Rows | Repair Eq | Helped | Hurt |
|---|---:|---:|---:|---:|
| domain=blocksworld | 1 | 1.0000 | 1 | 0 |
| domain=gripper | 1 | 0.0000 | 0 | 0 |
| style=abstract/abstract | 2 | 0.5000 | 1 | 0 |

## Recommendation

Repair does not yet pass the fresh fixed-pool gate; inspect failures before scaling repair.
