# Fixed-Pool Replay Evaluation

Source pool: `results\vcsr\fixed_pool_round7_compare\pools\seed_60\candidate_dump.jsonl`

| Verifier | K | Policy | Parse | Equiv | Equiv / Parse | Avg Parseable | Avg Equivalent | Oracle Best-of-K |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| retrain_from_round3_focused | 4 | greedy_first | 1.0000 | 0.3600 | 0.3600 | 3.86 | 1.56 | 0.4400 |
| retrain_from_round3_focused | 4 | random_parseable | 1.0000 | 0.3800 | 0.3800 | 3.86 | 1.56 | 0.4400 |
| retrain_from_round3_focused | 4 | verifier_ranked | 1.0000 | 0.4000 | 0.4000 | 3.86 | 1.56 | 0.4400 |
| retrain_from_round3_focused | 8 | greedy_first | 1.0000 | 0.3600 | 0.3600 | 7.58 | 3.22 | 0.5000 |
| retrain_from_round3_focused | 8 | random_parseable | 1.0000 | 0.4000 | 0.4000 | 7.58 | 3.22 | 0.5000 |
| retrain_from_round3_focused | 8 | verifier_ranked | 1.0000 | 0.4400 | 0.4400 | 7.58 | 3.22 | 0.5000 |
| retrain_from_round4_pointwise | 4 | greedy_first | 1.0000 | 0.3600 | 0.3600 | 3.86 | 1.56 | 0.4400 |
| retrain_from_round4_pointwise | 4 | random_parseable | 1.0000 | 0.3800 | 0.3800 | 3.86 | 1.56 | 0.4400 |
| retrain_from_round4_pointwise | 4 | verifier_ranked | 1.0000 | 0.4000 | 0.4000 | 3.86 | 1.56 | 0.4400 |
| retrain_from_round4_pointwise | 8 | greedy_first | 1.0000 | 0.3600 | 0.3600 | 7.58 | 3.22 | 0.5000 |
| retrain_from_round4_pointwise | 8 | random_parseable | 1.0000 | 0.4000 | 0.4000 | 7.58 | 3.22 | 0.5000 |
| retrain_from_round4_pointwise | 8 | verifier_ranked | 1.0000 | 0.4600 | 0.4600 | 7.58 | 3.22 | 0.5000 |
