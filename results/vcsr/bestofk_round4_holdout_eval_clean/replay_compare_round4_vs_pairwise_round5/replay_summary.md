# Fixed-Pool Replay Evaluation

Source pool: `results\vcsr\bestofk_round4_holdout_eval_clean\candidate_dump.jsonl`

| Verifier | K | Policy | Parse | Equiv | Equiv / Parse | Avg Parseable | Avg Equivalent | Oracle Best-of-K |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| retrain_from_round3_focused | 4 | greedy_first | 0.9600 | 0.4600 | 0.4792 | 3.86 | 1.72 | 0.5200 |
| retrain_from_round3_focused | 4 | random_parseable | 1.0000 | 0.4200 | 0.4200 | 3.86 | 1.72 | 0.5200 |
| retrain_from_round3_focused | 4 | verifier_ranked | 1.0000 | 0.5000 | 0.5000 | 3.86 | 1.72 | 0.5200 |
| retrain_from_round3_focused | 8 | greedy_first | 0.9600 | 0.4600 | 0.4792 | 7.58 | 3.42 | 0.5400 |
| retrain_from_round3_focused | 8 | random_parseable | 1.0000 | 0.4000 | 0.4000 | 7.58 | 3.42 | 0.5400 |
| retrain_from_round3_focused | 8 | verifier_ranked | 1.0000 | 0.5200 | 0.5200 | 7.58 | 3.42 | 0.5400 |
| retrain_from_round4_hybrid_pairwise | 4 | greedy_first | 0.9600 | 0.4600 | 0.4792 | 3.86 | 1.72 | 0.5200 |
| retrain_from_round4_hybrid_pairwise | 4 | random_parseable | 1.0000 | 0.4200 | 0.4200 | 3.86 | 1.72 | 0.5200 |
| retrain_from_round4_hybrid_pairwise | 4 | verifier_ranked | 1.0000 | 0.5000 | 0.5000 | 3.86 | 1.72 | 0.5200 |
| retrain_from_round4_hybrid_pairwise | 8 | greedy_first | 0.9600 | 0.4600 | 0.4792 | 7.58 | 3.42 | 0.5400 |
| retrain_from_round4_hybrid_pairwise | 8 | random_parseable | 1.0000 | 0.4000 | 0.4000 | 7.58 | 3.42 | 0.5400 |
| retrain_from_round4_hybrid_pairwise | 8 | verifier_ranked | 1.0000 | 0.5000 | 0.5000 | 7.58 | 3.42 | 0.5400 |
