# Multi-Model Prompt-Only vs VCSR Benchmark

This is a post-paper robustness benchmark. It does not replace the frozen
paper evidence from seeds `51-55`.

Output root: `results\vcsr\model_benchmark`
Seeds: `[72, 73, 74]`
Rows per seed: `10`
Main K: `8`

## Main Prompt-Only vs VCSR Summary

| Model | Prompt K=1 | VCSR Repair K=8 | Delta vs Prompt | Random K=8 | Verifier K=8 Ablation | Runs |
|---|---:|---:|---:|---:|---:|---:|
| claude_haiku_4p5 | 0.4000 | 0.9000 | +0.5000 | 0.3000 | 0.4667 | 3 |
| claude_opus_4p6 | 0.3667 | 0.9000 | +0.5333 | 0.4333 | 0.4333 | 3 |
| claude_sonnet_4p5 | 0.5000 | 0.9333 | +0.4333 | 0.4667 | 0.5333 | 3 |

Plain-English reading: with prompt-only Opus, about `36.7%` of rows were
semantically correct. With full VCSR around Opus, `90.0%` were semantically
correct. So VCSR added `+53.3` percentage points on this small benchmark.

## Per-Seed Prompt-Only vs VCSR Deltas

| Model | Seed | Prompt K=1 | VCSR Repair K=8 | Delta vs Prompt | Verifier K=8 Ablation |
|---|---:|---:|---:|---:|---:|
| claude_haiku_4p5 | 72 | 0.3000 | 0.9000 | +0.6000 | 0.4000 |
| claude_haiku_4p5 | 73 | 0.5000 | 0.9000 | +0.4000 | 0.5000 |
| claude_haiku_4p5 | 74 | 0.4000 | 0.9000 | +0.5000 | 0.5000 |
| claude_opus_4p6 | 72 | 0.2000 | 0.9000 | +0.7000 | 0.3000 |
| claude_opus_4p6 | 73 | 0.5000 | 1.0000 | +0.5000 | 0.5000 |
| claude_opus_4p6 | 74 | 0.4000 | 0.8000 | +0.4000 | 0.5000 |
| claude_sonnet_4p5 | 72 | 0.4000 | 1.0000 | +0.6000 | 0.5000 |
| claude_sonnet_4p5 | 73 | 0.6000 | 0.8000 | +0.2000 | 0.6000 |
| claude_sonnet_4p5 | 74 | 0.5000 | 1.0000 | +0.5000 | 0.5000 |

## Failed Runs

| Model | Seed | Output Dir | Error |
|---|---:|---|---|
| none |  |  |  |

## Interpretation Guide

- If VCSR repair improves most models over prompt-only, VCSR is a useful wrapper.
- If only the reference model improves, keep this as a limitation/generalization analysis.
- If a strong prompt-only model beats VCSR, report the cost/quality tradeoff honestly.
