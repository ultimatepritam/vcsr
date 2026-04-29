# Final Repair-Augmented VCSR Gate

Base config: `configs/vcsr_bestofk_final_repair.yaml`
Rows per seed: `50`
Seeds: `[51, 52, 53, 54, 55]`
Domains: `blocksworld`, `gripper`

This is the primary paper-facing final gate. These seeds are frozen final
evidence and should not be used for prompt tuning, checkpoint selection,
selector design, or repair gating.

## Mean K=8 Equivalence

| Policy | Mean Equiv | Per-Seed Values |
|---|---:|---|
| Prompt-only / first candidate | 0.3680 | n/a |
| Round-4 verifier-ranked | 0.4200 | 0.3600, 0.4200, 0.5200, 0.3400, 0.4600 |
| Repair-augmented | 0.7720 | 0.7800, 0.7800, 0.8000, 0.7600, 0.7400 |

## Per-Seed Results

| Seed | Baseline K=8 | Repair K=8 | Delta | Helped | Hurt | Repair Parse |
|---:|---:|---:|---:|---:|---:|---:|
| 51 | 0.3600 | 0.7800 | +0.4200 | 24 | 3 | 1.0000 |
| 52 | 0.4200 | 0.7800 | +0.3600 | 21 | 3 | 0.9800 |
| 53 | 0.5200 | 0.8000 | +0.2800 | 18 | 4 | 0.9600 |
| 54 | 0.3400 | 0.7600 | +0.4200 | 22 | 1 | 0.9800 |
| 55 | 0.4600 | 0.7400 | +0.2800 | 19 | 5 | 1.0000 |

## Acceptance

Accepted: `True`

Repair-augmented VCSR passes the final fresh gate.

## Paper Notes

- The `K=8` candidate-pool oracle is `0.4640`.
- Repair can exceed the candidate-pool oracle because repair creates a new
  candidate outside the original generated pool.
- Repair is a large net win overall, but it can hurt already-correct
  `blocksworld` candidates.
