# Paper Tables: Final VCSR Evidence

Source: `results/vcsr/final_repair_gate_round4`

## Main Result

- Final seeds: `[51, 52, 53, 54, 55]`
- Accepted: `True`
- Mean K=8 verifier-ranked: `0.4200`
- Mean K=8 repair-augmented VCSR: `0.7720`
- Mean K=8 delta: `+0.3520`
- Repair parse rate: `0.9840`
- Helped / hurt: `104 / 16`

## Table 1: Aggregate Metrics by Policy and K

| K | Policy | Parse | Equiv | Equiv / Parse |
| --- | --- | --- | --- | --- |
| 1 | Greedy first | 0.9240 | 0.3680 | 0.3994 |
| 1 | Random parseable | 0.9240 | 0.3680 | 0.3994 |
| 1 | Verifier-ranked | 0.9240 | 0.3680 | 0.3994 |
| 1 | VCSR repair-augmented | 0.9240 | 0.3680 | 0.3994 |
| 4 | Greedy first | 0.9240 | 0.3680 | 0.3994 |
| 4 | Random parseable | 0.9960 | 0.3760 | 0.3776 |
| 4 | Verifier-ranked | 0.9960 | 0.3920 | 0.3937 |
| 4 | VCSR repair-augmented | 0.9960 | 0.3920 | 0.3937 |
| 8 | Greedy first | 0.9240 | 0.3680 | 0.3994 |
| 8 | Random parseable | 1.0000 | 0.3760 | 0.3760 |
| 8 | Verifier-ranked | 1.0000 | 0.4200 | 0.4200 |
| 8 | VCSR repair-augmented | 1.0000 | 0.7720 | 0.7720 |

## Table 2: K=8 Domain and Style Breakdown

| Slice | Policy | Equiv Count | Equiv Rate | Parse Rate |
| --- | --- | --- | --- | --- |
| domain=blocksworld | Greedy first | 92/126 | 0.7302 | 0.9286 |
| domain=gripper | Greedy first | 0/124 | 0.0000 | 0.9194 |
| style=abstract/abstract | Greedy first | 28/126 | 0.2222 | 0.8492 |
| style=explicit/explicit | Greedy first | 64/124 | 0.5161 | 1.0000 |
| domain=blocksworld | Random parseable | 94/126 | 0.7460 | 1.0000 |
| domain=gripper | Random parseable | 0/124 | 0.0000 | 1.0000 |
| style=abstract/abstract | Random parseable | 30/126 | 0.2381 | 1.0000 |
| style=explicit/explicit | Random parseable | 64/124 | 0.5161 | 1.0000 |
| domain=blocksworld | Verifier-ranked | 103/126 | 0.8175 | 1.0000 |
| domain=gripper | Verifier-ranked | 2/124 | 0.0161 | 1.0000 |
| style=abstract/abstract | Verifier-ranked | 40/126 | 0.3175 | 1.0000 |
| style=explicit/explicit | Verifier-ranked | 65/124 | 0.5242 | 1.0000 |
| domain=blocksworld | VCSR repair-augmented | 95/126 | 0.7540 | 1.0000 |
| domain=gripper | VCSR repair-augmented | 98/124 | 0.7903 | 1.0000 |
| style=abstract/abstract | VCSR repair-augmented | 84/126 | 0.6667 | 1.0000 |
| style=explicit/explicit | VCSR repair-augmented | 109/124 | 0.8790 | 1.0000 |

## Table 3: Final Gate Per-Seed Outcomes

| Seed | Verifier K=8 | Repair K=8 | Delta | Helped | Hurt | Tied | Repair Parse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 51 | 0.3600 | 0.7800 | +0.4200 | 24 | 3 | 23 | 1.0000 |
| 52 | 0.4200 | 0.7800 | +0.3600 | 21 | 3 | 26 | 0.9800 |
| 53 | 0.5200 | 0.8000 | +0.2800 | 18 | 4 | 28 | 0.9600 |
| 54 | 0.3400 | 0.7600 | +0.4200 | 22 | 1 | 27 | 0.9800 |
| 55 | 0.4600 | 0.7400 | +0.2800 | 19 | 5 | 26 | 1.0000 |

## Repair Outcome Counts

```json
{
  "repair_helped": 104,
  "both_success": 89,
  "repair_hurt": 16,
  "both_fail": 41
}
```
