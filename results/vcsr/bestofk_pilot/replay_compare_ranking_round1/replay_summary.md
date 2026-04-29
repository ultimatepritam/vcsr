# Fixed-Pool Replay Evaluation

Source pool: `results\vcsr\bestofk_pilot\candidate_dump.jsonl`

| Verifier | K | Policy | Parse | Equiv | Equiv|Parse | Avg Parseable | Avg Equivalent | Oracle Best-of-K |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| lr_5em05 | 1 | greedy_first | 0.9333 | 0.4333 | 0.4643 | 0.93 | 0.43 | 0.4333 |
| lr_5em05 | 1 | random_parseable | 0.9333 | 0.4333 | 0.4643 | 0.93 | 0.43 | 0.4333 |
| lr_5em05 | 1 | verifier_ranked | 0.9333 | 0.4333 | 0.4643 | 0.93 | 0.43 | 0.4333 |
| lr_5em05 | 4 | greedy_first | 0.9333 | 0.4333 | 0.4643 | 3.80 | 1.77 | 0.5667 |
| lr_5em05 | 4 | random_parseable | 0.9667 | 0.4333 | 0.4483 | 3.80 | 1.77 | 0.5667 |
| lr_5em05 | 4 | verifier_ranked | 0.9667 | 0.4000 | 0.4138 | 3.80 | 1.77 | 0.5667 |
| lr_5em05 | 8 | greedy_first | 0.9333 | 0.4333 | 0.4643 | 7.60 | 3.37 | 0.6000 |
| lr_5em05 | 8 | random_parseable | 0.9667 | 0.5000 | 0.5172 | 7.60 | 3.37 | 0.6000 |
| lr_5em05 | 8 | verifier_ranked | 0.9667 | 0.4333 | 0.4483 | 7.60 | 3.37 | 0.6000 |
| lr_2p0em05 | 1 | greedy_first | 0.9333 | 0.4333 | 0.4643 | 0.93 | 0.43 | 0.4333 |
| lr_2p0em05 | 1 | random_parseable | 0.9333 | 0.4333 | 0.4643 | 0.93 | 0.43 | 0.4333 |
| lr_2p0em05 | 1 | verifier_ranked | 0.9333 | 0.4333 | 0.4643 | 0.93 | 0.43 | 0.4333 |
| lr_2p0em05 | 4 | greedy_first | 0.9333 | 0.4333 | 0.4643 | 3.80 | 1.77 | 0.5667 |
| lr_2p0em05 | 4 | random_parseable | 0.9667 | 0.4333 | 0.4483 | 3.80 | 1.77 | 0.5667 |
| lr_2p0em05 | 4 | verifier_ranked | 0.9667 | 0.4000 | 0.4138 | 3.80 | 1.77 | 0.5667 |
| lr_2p0em05 | 8 | greedy_first | 0.9333 | 0.4333 | 0.4643 | 7.60 | 3.37 | 0.6000 |
| lr_2p0em05 | 8 | random_parseable | 0.9667 | 0.5000 | 0.5172 | 7.60 | 3.37 | 0.6000 |
| lr_2p0em05 | 8 | verifier_ranked | 0.9667 | 0.4000 | 0.4138 | 7.60 | 3.37 | 0.6000 |
| retrain_from_capacity_push | 1 | greedy_first | 0.9333 | 0.4333 | 0.4643 | 0.93 | 0.43 | 0.4333 |
| retrain_from_capacity_push | 1 | random_parseable | 0.9333 | 0.4333 | 0.4643 | 0.93 | 0.43 | 0.4333 |
| retrain_from_capacity_push | 1 | verifier_ranked | 0.9333 | 0.4333 | 0.4643 | 0.93 | 0.43 | 0.4333 |
| retrain_from_capacity_push | 4 | greedy_first | 0.9333 | 0.4333 | 0.4643 | 3.80 | 1.77 | 0.5667 |
| retrain_from_capacity_push | 4 | random_parseable | 0.9667 | 0.4333 | 0.4483 | 3.80 | 1.77 | 0.5667 |
| retrain_from_capacity_push | 4 | verifier_ranked | 0.9667 | 0.4000 | 0.4138 | 3.80 | 1.77 | 0.5667 |
| retrain_from_capacity_push | 8 | greedy_first | 0.9333 | 0.4333 | 0.4643 | 7.60 | 3.37 | 0.6000 |
| retrain_from_capacity_push | 8 | random_parseable | 0.9667 | 0.5000 | 0.5172 | 7.60 | 3.37 | 0.6000 |
| retrain_from_capacity_push | 8 | verifier_ranked | 0.9667 | 0.4667 | 0.4828 | 7.60 | 3.37 | 0.6000 |
