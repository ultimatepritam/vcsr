# Multi-Seed Fresh Held-Out Comparison

Base config: `configs/vcsr_bestofk_round3_holdout_eval.yaml`
Rows per run: `50`
Seeds: `[48, 49, 50]`

## Mean Metrics

| Verifier | K | Policy | Mean Parse | Mean Equiv | Mean Equiv / Parse |
|---|---:|---|---:|---:|---:|
| round3_best_current | 1 | greedy_first | 0.9800 | 0.3800 | 0.3888 |
| round3_best_current | 1 | random_parseable | 0.9800 | 0.3800 | 0.3888 |
| round3_best_current | 1 | verifier_ranked | 0.9800 | 0.3800 | 0.3888 |
| round3_best_current | 4 | greedy_first | 0.9800 | 0.3800 | 0.3888 |
| round3_best_current | 4 | random_parseable | 0.9867 | 0.4000 | 0.4059 |
| round3_best_current | 4 | verifier_ranked | 0.9867 | 0.4000 | 0.4057 |
| round3_best_current | 8 | greedy_first | 0.9800 | 0.3800 | 0.3888 |
| round3_best_current | 8 | random_parseable | 0.9933 | 0.3600 | 0.3630 |
| round3_best_current | 8 | verifier_ranked | 0.9933 | 0.4000 | 0.4034 |
| round4_focused | 1 | greedy_first | 0.9533 | 0.3667 | 0.3869 |
| round4_focused | 1 | random_parseable | 0.9533 | 0.3667 | 0.3869 |
| round4_focused | 1 | verifier_ranked | 0.9533 | 0.3667 | 0.3869 |
| round4_focused | 4 | greedy_first | 0.9533 | 0.3667 | 0.3869 |
| round4_focused | 4 | random_parseable | 0.9933 | 0.3733 | 0.3763 |
| round4_focused | 4 | verifier_ranked | 0.9933 | 0.4000 | 0.4034 |
| round4_focused | 8 | greedy_first | 0.9533 | 0.3667 | 0.3869 |
| round4_focused | 8 | random_parseable | 1.0000 | 0.3867 | 0.3867 |
| round4_focused | 8 | verifier_ranked | 1.0000 | 0.4267 | 0.4267 |

## Head-to-Head

| K | Policy | Round4 Wins | Round3 Wins | Ties | Mean Delta |
|---:|---|---:|---:|---:|---:|
| 4 | verifier_ranked | 1 | 1 | 1 | +0.0000 |
| 8 | verifier_ranked | 2 | 0 | 1 | +0.0267 |

## Per-Seed Verifier-Ranked

### K=4

| Seed | Round3 | Round4 | Delta |
|---:|---:|---:|---:|
| 48 | 0.4800 | 0.5000 | +0.0200 |
| 49 | 0.3600 | 0.3400 | -0.0200 |
| 50 | 0.3600 | 0.3600 | +0.0000 |

### K=8

| Seed | Round3 | Round4 | Delta |
|---:|---:|---:|---:|
| 48 | 0.5000 | 0.5400 | +0.0400 |
| 49 | 0.3400 | 0.3400 | +0.0000 |
| 50 | 0.3600 | 0.4000 | +0.0400 |

## Recommendation

Promote `round4_focused` only if you are comfortable treating this multi-seed result as the new end-to-end gate: it beat or matched `round3_best_current` across the evaluated fresh held-out seeds and showed a positive mean verifier-ranked delta at both K=4 and K=8.
