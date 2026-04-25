# Round 4 Search Ablation

Verifier: `results/verifier/best_current/selection.yaml`
Pools: `6`
K values: `[4, 8]`

## Mean Metrics

| Policy | K | Equiv | Parse | Solvable | Eq / Parse | Helped | Hurt | Tied |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| verifier_ranked | 4 | 0.4500 | 0.9929 | 0.6964 | 0.4532 | 0 | 0 | 280 |
| solvable_then_verifier | 4 | 0.4500 | 0.7679 | 0.7679 | 0.5860 | 0 | 0 | 280 |
| verifier_then_solvable_tiebreak | 4 | 0.4500 | 0.9929 | 0.7679 | 0.4532 | 0 | 0 | 280 |
| parse_solvable_index | 4 | 0.4179 | 0.7679 | 0.7679 | 0.5442 | 1 | 10 | 269 |
| verifier_ranked | 8 | 0.4714 | 0.9964 | 0.6964 | 0.4731 | 0 | 0 | 280 |
| solvable_then_verifier | 8 | 0.4750 | 0.8107 | 0.8107 | 0.5859 | 1 | 0 | 279 |
| verifier_then_solvable_tiebreak | 8 | 0.4714 | 0.9964 | 0.8036 | 0.4731 | 0 | 0 | 280 |
| parse_solvable_index | 8 | 0.4179 | 0.8107 | 0.8107 | 0.5154 | 3 | 18 | 259 |

## Per-Pool Verifier-Ranked vs Search

| Pool | K | verifier_ranked | solvable_then_verifier | verifier_then_solvable_tiebreak | parse_solvable_index |
|---|---:|---:|---:|---:|---:|
| bestofk_round4_holdout_eval_clean | 4 | 0.5000 | 0.5000 | 0.5000 | 0.4800 |
| bestofk_round4_holdout_eval_clean | 8 | 0.5200 | 0.5200 | 0.5200 | 0.4800 |
| fixed_pool_round7_compare/seed_59 | 4 | 0.5000 | 0.5000 | 0.5000 | 0.4800 |
| fixed_pool_round7_compare/seed_59 | 8 | 0.5000 | 0.5000 | 0.5000 | 0.4800 |
| fixed_pool_round7_compare/seed_60 | 4 | 0.4000 | 0.4000 | 0.4000 | 0.3600 |
| fixed_pool_round7_compare/seed_60 | 8 | 0.4400 | 0.4400 | 0.4400 | 0.3600 |
| fixed_pool_round7_compare/seed_61 | 4 | 0.3200 | 0.3200 | 0.3200 | 0.2800 |
| fixed_pool_round7_compare/seed_61 | 8 | 0.3800 | 0.3800 | 0.3800 | 0.2800 |
| bestofk_round3_holdout_eval | 4 | 0.5000 | 0.5000 | 0.5000 | 0.4600 |
| bestofk_round3_holdout_eval | 8 | 0.4600 | 0.4800 | 0.4600 | 0.4600 |
| bestofk_pilot | 4 | 0.5000 | 0.5000 | 0.5000 | 0.4667 |
| bestofk_pilot | 8 | 0.5667 | 0.5667 | 0.5667 | 0.4667 |

## Candidate-Pool Diagnostics

| K | Oracle Best-of-K | Solvable Best-of-K | Avg Parseable | Avg Solvable |
|---:|---:|---:|---:|---:|
| 4 | 0.4893 | 0.7679 | 3.80 | 2.64 |
| 8 | 0.5214 | 0.8107 | 7.56 | 5.25 |

## Acceptance

No search policy passed cached replay. Do not spend on fresh generation for these policies; move to a small repair-loop pilot.
