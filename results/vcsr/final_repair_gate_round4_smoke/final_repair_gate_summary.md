# Final Repair-Augmented VCSR Gate

Base config: `configs/vcsr_bestofk_final_repair.yaml`
Rows per seed: `3`
Seeds: `[66]`

## Mean K=8 Equivalence

| Policy | Mean Equiv | Per-Seed Values |
|---|---:|---|
| Round-4 verifier-ranked | 0.3333 | 0.3333 |
| Repair-augmented | 1.0000 | 1.0000 |

## Per-Seed Results

| Seed | Baseline K=8 | Repair K=8 | Delta | Helped | Hurt | Repair Parse |
|---:|---:|---:|---:|---:|---:|---:|
| 66 | 0.3333 | 1.0000 | +0.6667 | 2 | 0 | 1.0000 |

## Acceptance

Accepted: `True`

Repair-augmented VCSR passes the final fresh gate.
