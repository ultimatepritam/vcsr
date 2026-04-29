# Fixed-Pool Replay Evaluation

Source pool: `results\vcsr\bestofk_round3_holdout_eval\candidate_dump.jsonl`

| Verifier | K | Policy | Parse | Equiv | Equiv / Parse | Avg Parseable | Avg Equivalent | Oracle Best-of-K |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| retrain_from_round3_focused | 4 | greedy_first | 0.9400 | 0.4400 | 0.4681 | 3.76 | 1.64 | 0.5200 |
| retrain_from_round3_focused | 4 | random_parseable | 1.0000 | 0.4400 | 0.4400 | 3.76 | 1.64 | 0.5200 |
| retrain_from_round3_focused | 4 | verifier_ranked | 1.0000 | 0.5000 | 0.5000 | 3.76 | 1.64 | 0.5200 |
| retrain_from_round3_focused | 8 | greedy_first | 0.9400 | 0.4400 | 0.4681 | 7.58 | 3.34 | 0.5400 |
| retrain_from_round3_focused | 8 | random_parseable | 1.0000 | 0.4200 | 0.4200 | 7.58 | 3.34 | 0.5400 |
| retrain_from_round3_focused | 8 | verifier_ranked | 1.0000 | 0.4600 | 0.4600 | 7.58 | 3.34 | 0.5400 |
| retrain_from_round4_conservative_pairwise | 4 | greedy_first | 0.9400 | 0.4400 | 0.4681 | 3.76 | 1.64 | 0.5200 |
| retrain_from_round4_conservative_pairwise | 4 | random_parseable | 1.0000 | 0.4400 | 0.4400 | 3.76 | 1.64 | 0.5200 |
| retrain_from_round4_conservative_pairwise | 4 | verifier_ranked | 1.0000 | 0.4400 | 0.4400 | 3.76 | 1.64 | 0.5200 |
| retrain_from_round4_conservative_pairwise | 8 | greedy_first | 0.9400 | 0.4400 | 0.4681 | 7.58 | 3.34 | 0.5400 |
| retrain_from_round4_conservative_pairwise | 8 | random_parseable | 1.0000 | 0.4200 | 0.4200 | 7.58 | 3.34 | 0.5400 |
| retrain_from_round4_conservative_pairwise | 8 | verifier_ranked | 1.0000 | 0.4600 | 0.4600 | 7.58 | 3.34 | 0.5400 |
