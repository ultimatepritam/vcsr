# Fixed-Pool Replay Evaluation

Source pool: `results\vcsr\bestofk_ranking_round2_pool\candidate_dump.jsonl`

| Verifier | K | Policy | Parse | Equiv | Equiv / Parse | Avg Parseable | Avg Equivalent | Oracle Best-of-K |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| retrain_from_round1 | 1 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| retrain_from_round1 | 1 | random_parseable | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| retrain_from_round1 | 1 | verifier_ranked | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| retrain_from_round1 | 4 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 3.94 | 1.72 | 0.5600 |
| retrain_from_round1 | 4 | random_parseable | 1.0000 | 0.4800 | 0.4800 | 3.94 | 1.72 | 0.5600 |
| retrain_from_round1 | 4 | verifier_ranked | 1.0000 | 0.4800 | 0.4800 | 3.94 | 1.72 | 0.5600 |
| retrain_from_round1 | 8 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 7.74 | 3.26 | 0.6200 |
| retrain_from_round1 | 8 | random_parseable | 1.0000 | 0.4400 | 0.4400 | 7.74 | 3.26 | 0.6200 |
| retrain_from_round1 | 8 | verifier_ranked | 1.0000 | 0.4600 | 0.4600 | 7.74 | 3.26 | 0.6200 |
| retrain_from_round2_multipool | 1 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| retrain_from_round2_multipool | 1 | random_parseable | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| retrain_from_round2_multipool | 1 | verifier_ranked | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| retrain_from_round2_multipool | 4 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 3.94 | 1.72 | 0.5600 |
| retrain_from_round2_multipool | 4 | random_parseable | 1.0000 | 0.4800 | 0.4800 | 3.94 | 1.72 | 0.5600 |
| retrain_from_round2_multipool | 4 | verifier_ranked | 1.0000 | 0.5200 | 0.5200 | 3.94 | 1.72 | 0.5600 |
| retrain_from_round2_multipool | 8 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 7.74 | 3.26 | 0.6200 |
| retrain_from_round2_multipool | 8 | random_parseable | 1.0000 | 0.4400 | 0.4400 | 7.74 | 3.26 | 0.6200 |
| retrain_from_round2_multipool | 8 | verifier_ranked | 1.0000 | 0.5400 | 0.5400 | 7.74 | 3.26 | 0.6200 |
