# Fixed-Pool Replay Evaluation

Source pool: `results\vcsr\bestofk_pilot\candidate_dump.jsonl`

| Verifier | K | Policy | Parse | Equiv | Equiv / Parse | Avg Parseable | Avg Equivalent | Oracle Best-of-K |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| retrain_from_round1 | 1 | greedy_first | 0.9333 | 0.4333 | 0.4643 | 0.93 | 0.43 | 0.4333 |
| retrain_from_round1 | 1 | random_parseable | 0.9333 | 0.4333 | 0.4643 | 0.93 | 0.43 | 0.4333 |
| retrain_from_round1 | 1 | verifier_ranked | 0.9333 | 0.4333 | 0.4643 | 0.93 | 0.43 | 0.4333 |
| retrain_from_round1 | 4 | greedy_first | 0.9333 | 0.4333 | 0.4643 | 3.80 | 1.77 | 0.5667 |
| retrain_from_round1 | 4 | random_parseable | 0.9667 | 0.4333 | 0.4483 | 3.80 | 1.77 | 0.5667 |
| retrain_from_round1 | 4 | verifier_ranked | 0.9667 | 0.4667 | 0.4828 | 3.80 | 1.77 | 0.5667 |
| retrain_from_round1 | 8 | greedy_first | 0.9333 | 0.4333 | 0.4643 | 7.60 | 3.37 | 0.6000 |
| retrain_from_round1 | 8 | random_parseable | 0.9667 | 0.5000 | 0.5172 | 7.60 | 3.37 | 0.6000 |
| retrain_from_round1 | 8 | verifier_ranked | 0.9667 | 0.5333 | 0.5517 | 7.60 | 3.37 | 0.6000 |
| retrain_from_round2_multipool | 1 | greedy_first | 0.9333 | 0.4333 | 0.4643 | 0.93 | 0.43 | 0.4333 |
| retrain_from_round2_multipool | 1 | random_parseable | 0.9333 | 0.4333 | 0.4643 | 0.93 | 0.43 | 0.4333 |
| retrain_from_round2_multipool | 1 | verifier_ranked | 0.9333 | 0.4333 | 0.4643 | 0.93 | 0.43 | 0.4333 |
| retrain_from_round2_multipool | 4 | greedy_first | 0.9333 | 0.4333 | 0.4643 | 3.80 | 1.77 | 0.5667 |
| retrain_from_round2_multipool | 4 | random_parseable | 0.9667 | 0.4333 | 0.4483 | 3.80 | 1.77 | 0.5667 |
| retrain_from_round2_multipool | 4 | verifier_ranked | 0.9667 | 0.5000 | 0.5172 | 3.80 | 1.77 | 0.5667 |
| retrain_from_round2_multipool | 8 | greedy_first | 0.9333 | 0.4333 | 0.4643 | 7.60 | 3.37 | 0.6000 |
| retrain_from_round2_multipool | 8 | random_parseable | 0.9667 | 0.5000 | 0.5172 | 7.60 | 3.37 | 0.6000 |
| retrain_from_round2_multipool | 8 | verifier_ranked | 0.9667 | 0.5333 | 0.5517 | 7.60 | 3.37 | 0.6000 |
