# Multi-Seed Fresh Held-Out Comparison

Base config: `configs/vcsr_bestofk_round3_holdout_eval.yaml`
Rows per run: `50`
Seeds: `[56, 57, 58]`

## Mean Metrics

| Verifier | K | Policy | Mean Parse | Mean Equiv | Mean Equiv / Parse |
|---|---:|---|---:|---:|---:|
| round4_best_current | 1 | greedy_first | 0.9533 | 0.3600 | 0.3765 |
| round4_best_current | 1 | random_parseable | 0.9533 | 0.3600 | 0.3765 |
| round4_best_current | 1 | verifier_ranked | 0.9533 | 0.3600 | 0.3765 |
| round4_best_current | 4 | greedy_first | 0.9533 | 0.3600 | 0.3765 |
| round4_best_current | 4 | random_parseable | 0.9867 | 0.4000 | 0.4050 |
| round4_best_current | 4 | verifier_ranked | 0.9867 | 0.4000 | 0.4052 |
| round4_best_current | 8 | greedy_first | 0.9533 | 0.3600 | 0.3765 |
| round4_best_current | 8 | random_parseable | 0.9933 | 0.3467 | 0.3494 |
| round4_best_current | 8 | verifier_ranked | 0.9933 | 0.4200 | 0.4235 |
| round7_focused_pointwise | 1 | greedy_first | 0.9533 | 0.3667 | 0.3834 |
| round7_focused_pointwise | 1 | random_parseable | 0.9533 | 0.3667 | 0.3834 |
| round7_focused_pointwise | 1 | verifier_ranked | 0.9533 | 0.3667 | 0.3834 |
| round7_focused_pointwise | 4 | greedy_first | 0.9533 | 0.3667 | 0.3834 |
| round7_focused_pointwise | 4 | random_parseable | 0.9867 | 0.3733 | 0.3767 |
| round7_focused_pointwise | 4 | verifier_ranked | 0.9867 | 0.4133 | 0.4172 |
| round7_focused_pointwise | 8 | greedy_first | 0.9533 | 0.3667 | 0.3834 |
| round7_focused_pointwise | 8 | random_parseable | 1.0000 | 0.4067 | 0.4067 |
| round7_focused_pointwise | 8 | verifier_ranked | 1.0000 | 0.4200 | 0.4200 |

## Head-to-Head

| K | Policy | Round7 Wins | Round4 Wins | Ties | Mean Delta |
|---:|---|---:|---:|---:|---:|
| 4 | verifier_ranked | 2 | 1 | 0 | +0.0133 |
| 8 | verifier_ranked | 2 | 1 | 0 | +0.0000 |

## Per-Seed Verifier-Ranked

### K=4

| Seed | round4_best_current | round7_focused_pointwise | Delta |
|---:|---:|---:|---:|
| 56 | 0.4400 | 0.4000 | -0.0400 |
| 57 | 0.2600 | 0.2800 | +0.0200 |
| 58 | 0.5000 | 0.5600 | +0.0600 |

### K=8

| Seed | round4_best_current | round7_focused_pointwise | Delta |
|---:|---:|---:|---:|
| 56 | 0.4800 | 0.4000 | -0.0800 |
| 57 | 0.2600 | 0.2800 | +0.0200 |
| 58 | 0.5200 | 0.5800 | +0.0600 |

## Recommendation

Keep `round4_best_current` as the official frozen baseline for now. `round7_focused_pointwise` remains a provisional candidate, but the multi-seed fresh held-out gate did not show a clean enough verifier-ranked win profile at both K=4 and K=8 to justify promotion.
