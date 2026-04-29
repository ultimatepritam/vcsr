# Fresh Fixed-Pool Repair Gate

Base config: `configs/vcsr_bestofk_round3_holdout_eval.yaml`
Rows per seed: `50`
Seeds: `[62, 63, 64]`

Generation happened once per seed. Round-4 selected failures were repaired once on the same candidate pools.

## Mean K=8 Equivalence

| Policy | Mean Equiv | Per-Seed Values |
|---|---:|---|
| Round-4 verifier-ranked | 0.5000 | 0.5000, 0.5000, 0.5000 |
| Repair-augmented | 0.5467 | 0.5000, 0.5600, 0.5800 |

## Repair Outcomes

| Seed | Baseline K=8 | Repair-Aug K=8 | Delta | Failures Repaired | Helped | Hurt | Tied | Parse Rate | Repair Eq |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 62 | 0.5000 | 0.5000 | +0.0000 | 26 | 0 | 0 | 26 | 1.0000 | 0.0000 |
| 63 | 0.5000 | 0.5600 | +0.0600 | 23 | 3 | 0 | 20 | 1.0000 | 0.1304 |
| 64 | 0.5000 | 0.5800 | +0.0800 | 24 | 4 | 0 | 20 | 1.0000 | 0.1667 |

## Breakdown Over Repaired Failures

| Slice | Rows | Repair Eq | Helped | Hurt |
|---|---:|---:|---:|---:|
| domain=blocksworld | 10 | 0.7000 | 7 | 0 |
| domain=gripper | 63 | 0.0000 | 0 | 0 |
| style=abstract/abstract | 43 | 0.1628 | 7 | 0 |
| style=explicit/explicit | 30 | 0.0000 | 0 | 0 |

## Recommendation

Repair does not yet pass the fresh fixed-pool gate; inspect failures before scaling repair.
