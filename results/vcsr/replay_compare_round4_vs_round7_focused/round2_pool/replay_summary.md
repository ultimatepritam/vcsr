# Fixed-Pool Replay Evaluation

Source pool: `results\vcsr\bestofk_ranking_round2_pool\candidate_dump.jsonl`

| Verifier | K | Policy | Parse | Equiv | Equiv / Parse | Avg Parseable | Avg Equivalent | Oracle Best-of-K |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| retrain_from_round3_focused | 4 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 3.94 | 1.72 | 0.5600 |
| retrain_from_round3_focused | 4 | random_parseable | 1.0000 | 0.4800 | 0.4800 | 3.94 | 1.72 | 0.5600 |
| retrain_from_round3_focused | 4 | verifier_ranked | 1.0000 | 0.5200 | 0.5200 | 3.94 | 1.72 | 0.5600 |
| retrain_from_round3_focused | 8 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 7.74 | 3.26 | 0.6200 |
| retrain_from_round3_focused | 8 | random_parseable | 1.0000 | 0.4400 | 0.4400 | 7.74 | 3.26 | 0.6200 |
| retrain_from_round3_focused | 8 | verifier_ranked | 1.0000 | 0.5200 | 0.5200 | 7.74 | 3.26 | 0.6200 |
| retrain_from_round4_pointwise | 4 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 3.94 | 1.72 | 0.5600 |
| retrain_from_round4_pointwise | 4 | random_parseable | 1.0000 | 0.4800 | 0.4800 | 3.94 | 1.72 | 0.5600 |
| retrain_from_round4_pointwise | 4 | verifier_ranked | 1.0000 | 0.5200 | 0.5200 | 3.94 | 1.72 | 0.5600 |
| retrain_from_round4_pointwise | 8 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 7.74 | 3.26 | 0.6200 |
| retrain_from_round4_pointwise | 8 | random_parseable | 1.0000 | 0.4400 | 0.4400 | 7.74 | 3.26 | 0.6200 |
| retrain_from_round4_pointwise | 8 | verifier_ranked | 1.0000 | 0.5800 | 0.5800 | 7.74 | 3.26 | 0.6200 |
