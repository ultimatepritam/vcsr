# Fixed-Pool Replay Evaluation

Source pool: `results\vcsr\fixed_pool_round7_compare\pools\seed_61\candidate_dump.jsonl`

| Verifier | K | Policy | Parse | Equiv | Equiv / Parse | Avg Parseable | Avg Equivalent | Oracle Best-of-K |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| retrain_from_round3_focused | 4 | greedy_first | 0.9200 | 0.2600 | 0.2826 | 3.62 | 1.06 | 0.4000 |
| retrain_from_round3_focused | 4 | random_parseable | 0.9800 | 0.3000 | 0.3061 | 3.62 | 1.06 | 0.4000 |
| retrain_from_round3_focused | 4 | verifier_ranked | 0.9800 | 0.3200 | 0.3265 | 3.62 | 1.06 | 0.4000 |
| retrain_from_round3_focused | 8 | greedy_first | 0.9200 | 0.2600 | 0.2826 | 7.36 | 2.10 | 0.4600 |
| retrain_from_round3_focused | 8 | random_parseable | 1.0000 | 0.3200 | 0.3200 | 7.36 | 2.10 | 0.4600 |
| retrain_from_round3_focused | 8 | verifier_ranked | 1.0000 | 0.3800 | 0.3800 | 7.36 | 2.10 | 0.4600 |
| retrain_from_round4_pointwise | 4 | greedy_first | 0.9200 | 0.2600 | 0.2826 | 3.62 | 1.06 | 0.4000 |
| retrain_from_round4_pointwise | 4 | random_parseable | 0.9800 | 0.3000 | 0.3061 | 3.62 | 1.06 | 0.4000 |
| retrain_from_round4_pointwise | 4 | verifier_ranked | 0.9800 | 0.2800 | 0.2857 | 3.62 | 1.06 | 0.4000 |
| retrain_from_round4_pointwise | 8 | greedy_first | 0.9200 | 0.2600 | 0.2826 | 7.36 | 2.10 | 0.4600 |
| retrain_from_round4_pointwise | 8 | random_parseable | 1.0000 | 0.3200 | 0.3200 | 7.36 | 2.10 | 0.4600 |
| retrain_from_round4_pointwise | 8 | verifier_ranked | 1.0000 | 0.3800 | 0.3800 | 7.36 | 2.10 | 0.4600 |
