# Round 4 vs Round 6 Replay Gate

Round 6 is **not accepted**.

| Pool | K | Round 4 verifier-ranked | Round 6 verifier-ranked | Delta | Oracle |
|---|---:|---:|---:|---:|---:|
| `round4_holdout_clean` | 4 | 0.5000 | 0.4800 | -0.0200 | 0.5200 |
| `round4_holdout_clean` | 8 | 0.5200 | 0.5000 | -0.0200 | 0.5400 |
| `round3_holdout` | 4 | 0.5000 | 0.4400 | -0.0600 | 0.5200 |
| `round3_holdout` | 8 | 0.4600 | 0.4600 | +0.0000 | 0.5400 |
| `pilot` | 4 | 0.5000 | 0.5000 | +0.0000 | 0.5667 |
| `pilot` | 8 | 0.5667 | 0.5333 | -0.0333 | 0.6000 |

Mean replay result:

- `K=4`: round 4 `0.5000`, round 6 `0.4733`, delta `-0.0267`
- `K=8`: round 4 `0.5156`, round 6 `0.4978`, delta `-0.0178`

Interpretation:

- Conservative pairwise round 6 was safer than round 5 in spirit, but still did not improve downstream selection.
- The replay gate fails before any fresh generation is justified.
- Keep round 4 as `best_current`.
- Treat round 6 as a useful negative result showing that naive pairwise/listwise pressure is not enough; future improvement should first diagnose score behavior and perhaps focus on candidate normalization, row-level calibration, or selection policy rather than another immediate training run.
