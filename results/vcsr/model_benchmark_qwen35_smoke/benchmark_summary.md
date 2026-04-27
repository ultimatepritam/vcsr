# Multi-Model Prompt-Only vs VCSR Benchmark

This is a post-paper robustness benchmark. It does not replace the frozen
paper evidence from seeds `51-55`.

Output root: `results\vcsr\model_benchmark_qwen35_smoke`
Seeds: `[72]`
Rows per seed: `3`
Main K: `8`

## Main Equivalence Summary

| Model | Prompt K=1 | Random K=8 | Verifier K=8 | VCSR Repair K=8 | Repair - Prompt | Repair - Verifier | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|

## Per-Seed Main Deltas

| Model | Seed | Prompt K=1 | Verifier K=8 | VCSR Repair K=8 | Repair - Prompt | Repair - Verifier |
|---|---:|---:|---:|---:|---:|---:|

## Failed Runs

| Model | Seed | Output Dir | Error |
|---|---:|---|---|
| qwen_3p5_9b | 72 | `results\vcsr\model_benchmark_qwen35_smoke\qwen_3p5_9b\seed_72` | child run failed with exit code 4294967295 |

## Interpretation Guide

- If VCSR repair improves most models over prompt-only, VCSR is a useful wrapper.
- If only the reference model improves, keep this as a limitation/generalization analysis.
- If a strong prompt-only model beats VCSR, report the cost/quality tradeoff honestly.
