# Fresh Fixed-Pool Verifier Comparison

Base config: `configs/vcsr_bestofk_round3_holdout_eval.yaml`
Rows per seed: `50`
Seeds: `[59, 60, 61]`

Generation happened once per seed. Each verifier was replayed on the same candidate dump.

## Mean Verifier-Ranked Equivalence

| Verifier | K | Mean Equiv | Per-Seed Values |
|---|---:|---:|---|
| round4_best_current | 4 | 0.4067 | 0.5000, 0.4000, 0.3200 |
| round4_best_current | 8 | 0.4400 | 0.5000, 0.4400, 0.3800 |
| round7_focused_pointwise | 4 | 0.3933 | 0.5000, 0.4000, 0.2800 |
| round7_focused_pointwise | 8 | 0.4467 | 0.5000, 0.4600, 0.3800 |

## Head-to-Head

| K | Candidate - Baseline | Wins | Losses | Ties |
|---:|---:|---:|---:|---:|
| 4 | -0.0133 | 0 | 1 | 2 |
| 8 | +0.0067 | 1 | 0 | 2 |

## Replay Artifacts

| Seed | Pool | Replay |
|---:|---|---|
| 59 | `results\vcsr\fixed_pool_round7_compare\pools\seed_59\candidate_dump.jsonl` | `results\vcsr\fixed_pool_round7_compare\replays\seed_59` |
| 60 | `results\vcsr\fixed_pool_round7_compare\pools\seed_60\candidate_dump.jsonl` | `results\vcsr\fixed_pool_round7_compare\replays\seed_60` |
| 61 | `results\vcsr\fixed_pool_round7_compare\pools\seed_61\candidate_dump.jsonl` | `results\vcsr\fixed_pool_round7_compare\replays\seed_61` |

## Recommendation

Round 4 remains best_current; round 7 does not pass the identical-pool verifier gate.
