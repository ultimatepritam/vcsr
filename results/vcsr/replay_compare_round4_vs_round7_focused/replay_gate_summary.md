# Round 4 vs Round 7 Focused Replay Gate

| K | Round 4 Mean | Round 7 Mean | Delta | Helped | Hurt | Changed |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.5050 | 0.5050 | +0.0000 | 1 | 1 | 47 |
| 8 | 0.5167 | 0.5283 | +0.0117 | 4 | 1 | 59 |

## Per-Pool

| Pool | K | Round 4 | Round 7 | Delta |
|---|---:|---:|---:|---:|
| round4_holdout_clean | 4 | 0.5000 | 0.5200 | +0.0200 |
| round4_holdout_clean | 8 | 0.5200 | 0.5400 | +0.0200 |
| round3_holdout | 4 | 0.5000 | 0.4800 | -0.0200 |
| round3_holdout | 8 | 0.4600 | 0.4600 | +0.0000 |
| round2_pool | 4 | 0.5200 | 0.5200 | +0.0000 |
| round2_pool | 8 | 0.5200 | 0.5800 | +0.0600 |
| pilot | 4 | 0.5000 | 0.5000 | +0.0000 |
| pilot | 8 | 0.5667 | 0.5333 | -0.0333 |

## Acceptance

- `accepted`: `True`
- `k8_improves`: `True`
- `k4_non_regress_with_tolerance_minus_0p01`: `True`
- `helped_more_than_hurt`: `True`
- `k8_improvement_pool_count`: `2`
