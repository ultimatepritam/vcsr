# Final Repair-Augmented VCSR Gate

Base config: `configs/vcsr_bestofk_guarded_repair.yaml`
Rows per seed: `50`
Seeds: `[67, 68, 69, 70, 71]`
Baseline policy: `verifier_ranked`
Repair policy: `verifier_ranked_repair_guarded`

## Mean K=8 Equivalence

| Policy | Mean Equiv | Per-Seed Values |
|---|---:|---|
| Round-4 verifier-ranked | 0.4360 | 0.4600, 0.3400, 0.4800, 0.4200, 0.4800 |
| Repair-augmented | 0.7960 | 0.8000, 0.7400, 0.8400, 0.8000, 0.8000 |
| Comparison repair policy | 0.7960 | see per-seed aggregate metrics |

## Per-Seed Results

| Seed | Baseline K=8 | Repair K=8 | Delta | Helped | Hurt | Repair Parse |
|---:|---:|---:|---:|---:|---:|---:|
| 67 | 0.4600 | 0.8000 | +0.3400 | 20 | 3 | 1.0000 |
| 68 | 0.3400 | 0.7400 | +0.4000 | 22 | 2 | 1.0000 |
| 69 | 0.4800 | 0.8400 | +0.3600 | 20 | 2 | 1.0000 |
| 70 | 0.4200 | 0.8000 | +0.3800 | 21 | 2 | 1.0000 |
| 71 | 0.4800 | 0.8000 | +0.3200 | 21 | 5 | 1.0000 |

## Guard Diagnostics

Total hurt rows: `14`
Comparison hurt rows: `14`
Blocksworld hurt rows: `14`
Comparison blocksworld hurt rows: `14`

## Acceptance

Accepted: `False`

Repair-augmented VCSR does not pass the final fresh gate; keep the claim cautious.
