# Multi-Model Prompt-Only vs VCSR Benchmark

This is a post-paper robustness benchmark. It does not replace the frozen
paper evidence from seeds `51-55`.

Output root: `results\vcsr\model_benchmark_openrouter_claude_opus_4p6_smoke`
Seeds: `[72]`
Rows per seed: `3`
Main K: `8`

## Main Equivalence Summary

| Model | Prompt K=1 | Random K=8 | Verifier K=8 | VCSR Repair K=8 | Repair - Prompt | Repair - Verifier | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| openrouter_claude_opus_4p6 | 0.3333 | 0.6667 | 0.6667 | 1.0000 | +0.6667 | +0.3333 | 1 |

## Per-Seed Main Deltas

| Model | Seed | Prompt K=1 | Verifier K=8 | VCSR Repair K=8 | Repair - Prompt | Repair - Verifier |
|---|---:|---:|---:|---:|---:|---:|
| openrouter_claude_opus_4p6 | 72 | 0.3333 | 0.6667 | 1.0000 | +0.6667 | +0.3333 |

## Failed Runs

| Model | Seed | Output Dir | Error |
|---|---:|---|---|
| none |  |  |  |

## Interpretation Guide

- If VCSR repair improves most models over prompt-only, VCSR is a useful wrapper.
- If only the reference model improves, keep this as a limitation/generalization analysis.
- If a strong prompt-only model beats VCSR, report the cost/quality tradeoff honestly.
