# Fixed-Pool Replay Evaluation

Source pool: `results\vcsr\bestofk_round3_holdout_eval\candidate_dump.jsonl`

| Verifier | K | Policy | Parse | Equiv | Equiv / Parse | Avg Parseable | Avg Equivalent | Oracle Best-of-K |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| retrain_from_round2_multipool | 1 | greedy_first | 0.9400 | 0.4400 | 0.4681 | 0.94 | 0.44 | 0.4400 |
| retrain_from_round2_multipool | 1 | random_parseable | 0.9400 | 0.4400 | 0.4681 | 0.94 | 0.44 | 0.4400 |
| retrain_from_round2_multipool | 1 | verifier_ranked | 0.9400 | 0.4400 | 0.4681 | 0.94 | 0.44 | 0.4400 |
| retrain_from_round2_multipool | 4 | greedy_first | 0.9400 | 0.4400 | 0.4681 | 3.76 | 1.64 | 0.5200 |
| retrain_from_round2_multipool | 4 | random_parseable | 1.0000 | 0.4400 | 0.4400 | 3.76 | 1.64 | 0.5200 |
| retrain_from_round2_multipool | 4 | verifier_ranked | 1.0000 | 0.4200 | 0.4200 | 3.76 | 1.64 | 0.5200 |
| retrain_from_round2_multipool | 8 | greedy_first | 0.9400 | 0.4400 | 0.4681 | 7.58 | 3.34 | 0.5400 |
| retrain_from_round2_multipool | 8 | random_parseable | 1.0000 | 0.4200 | 0.4200 | 7.58 | 3.34 | 0.5400 |
| retrain_from_round2_multipool | 8 | verifier_ranked | 1.0000 | 0.4400 | 0.4400 | 7.58 | 3.34 | 0.5400 |
| retrain_from_round3_focused | 1 | greedy_first | 0.9400 | 0.4400 | 0.4681 | 0.94 | 0.44 | 0.4400 |
| retrain_from_round3_focused | 1 | random_parseable | 0.9400 | 0.4400 | 0.4681 | 0.94 | 0.44 | 0.4400 |
| retrain_from_round3_focused | 1 | verifier_ranked | 0.9400 | 0.4400 | 0.4681 | 0.94 | 0.44 | 0.4400 |
| retrain_from_round3_focused | 4 | greedy_first | 0.9400 | 0.4400 | 0.4681 | 3.76 | 1.64 | 0.5200 |
| retrain_from_round3_focused | 4 | random_parseable | 1.0000 | 0.4400 | 0.4400 | 3.76 | 1.64 | 0.5200 |
| retrain_from_round3_focused | 4 | verifier_ranked | 1.0000 | 0.5000 | 0.5000 | 3.76 | 1.64 | 0.5200 |
| retrain_from_round3_focused | 8 | greedy_first | 0.9400 | 0.4400 | 0.4681 | 7.58 | 3.34 | 0.5400 |
| retrain_from_round3_focused | 8 | random_parseable | 1.0000 | 0.4200 | 0.4200 | 7.58 | 3.34 | 0.5400 |
| retrain_from_round3_focused | 8 | verifier_ranked | 1.0000 | 0.4600 | 0.4600 | 7.58 | 3.34 | 0.5400 |
