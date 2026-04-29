# Fixed-Pool Replay Evaluation

Source pool: `results\vcsr\bestofk_ranking_round2_pool\candidate_dump.jsonl`

| Verifier | K | Policy | Parse | Equiv | Equiv / Parse | Avg Parseable | Avg Equivalent | Oracle Best-of-K |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| lr_5em05 | 1 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| lr_5em05 | 1 | random_parseable | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| lr_5em05 | 1 | verifier_ranked | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| lr_5em05 | 4 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 3.94 | 1.72 | 0.5600 |
| lr_5em05 | 4 | random_parseable | 1.0000 | 0.4800 | 0.4800 | 3.94 | 1.72 | 0.5600 |
| lr_5em05 | 4 | verifier_ranked | 1.0000 | 0.4200 | 0.4200 | 3.94 | 1.72 | 0.5600 |
| lr_5em05 | 8 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 7.74 | 3.26 | 0.6200 |
| lr_5em05 | 8 | random_parseable | 1.0000 | 0.4400 | 0.4400 | 7.74 | 3.26 | 0.6200 |
| lr_5em05 | 8 | verifier_ranked | 1.0000 | 0.4200 | 0.4200 | 7.74 | 3.26 | 0.6200 |
| lr_2p0em05 | 1 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| lr_2p0em05 | 1 | random_parseable | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| lr_2p0em05 | 1 | verifier_ranked | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| lr_2p0em05 | 4 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 3.94 | 1.72 | 0.5600 |
| lr_2p0em05 | 4 | random_parseable | 1.0000 | 0.4800 | 0.4800 | 3.94 | 1.72 | 0.5600 |
| lr_2p0em05 | 4 | verifier_ranked | 1.0000 | 0.4200 | 0.4200 | 3.94 | 1.72 | 0.5600 |
| lr_2p0em05 | 8 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 7.74 | 3.26 | 0.6200 |
| lr_2p0em05 | 8 | random_parseable | 1.0000 | 0.4400 | 0.4400 | 7.74 | 3.26 | 0.6200 |
| lr_2p0em05 | 8 | verifier_ranked | 1.0000 | 0.3800 | 0.3800 | 7.74 | 3.26 | 0.6200 |
| retrain_from_capacity_push | 1 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| retrain_from_capacity_push | 1 | random_parseable | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| retrain_from_capacity_push | 1 | verifier_ranked | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| retrain_from_capacity_push | 4 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 3.94 | 1.72 | 0.5600 |
| retrain_from_capacity_push | 4 | random_parseable | 1.0000 | 0.4800 | 0.4800 | 3.94 | 1.72 | 0.5600 |
| retrain_from_capacity_push | 4 | verifier_ranked | 1.0000 | 0.4200 | 0.4200 | 3.94 | 1.72 | 0.5600 |
| retrain_from_capacity_push | 8 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 7.74 | 3.26 | 0.6200 |
| retrain_from_capacity_push | 8 | random_parseable | 1.0000 | 0.4400 | 0.4400 | 7.74 | 3.26 | 0.6200 |
| retrain_from_capacity_push | 8 | verifier_ranked | 1.0000 | 0.3800 | 0.3800 | 7.74 | 3.26 | 0.6200 |
| retrain_from_round1 | 1 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| retrain_from_round1 | 1 | random_parseable | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| retrain_from_round1 | 1 | verifier_ranked | 1.0000 | 0.4600 | 0.4600 | 1.00 | 0.46 | 0.4600 |
| retrain_from_round1 | 4 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 3.94 | 1.72 | 0.5600 |
| retrain_from_round1 | 4 | random_parseable | 1.0000 | 0.4800 | 0.4800 | 3.94 | 1.72 | 0.5600 |
| retrain_from_round1 | 4 | verifier_ranked | 1.0000 | 0.4800 | 0.4800 | 3.94 | 1.72 | 0.5600 |
| retrain_from_round1 | 8 | greedy_first | 1.0000 | 0.4600 | 0.4600 | 7.74 | 3.26 | 0.6200 |
| retrain_from_round1 | 8 | random_parseable | 1.0000 | 0.4400 | 0.4400 | 7.74 | 3.26 | 0.6200 |
| retrain_from_round1 | 8 | verifier_ranked | 1.0000 | 0.4600 | 0.4600 | 7.74 | 3.26 | 0.6200 |
