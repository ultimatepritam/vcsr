# Guarded Repair Policy Analysis

Selected margin: `0.05`

| Margin | Mean Baseline K=8 | Mean Guarded K=8 | Delta | Accepted | Helped | Hurt |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.5000 | 0.9200 | +0.4200 | 66 | 63 | 0 |
| 0.01 | 0.5000 | 0.9533 | +0.4533 | 72 | 68 | 0 |
| 0.02 | 0.5000 | 0.9533 | +0.4533 | 72 | 68 | 0 |
| 0.05 | 0.5000 | 0.9600 | +0.4600 | 73 | 69 | 0 |

## Decision

Use the selected margin for fresh guarded-repair validation. This analysis uses development repair artifacts only and does not tune on seeds `51-55`.
