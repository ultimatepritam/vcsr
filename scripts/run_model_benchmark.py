"""Run a post-paper prompt-only vs VCSR benchmark across OpenRouter models.

The benchmark reuses scripts/run_verifier_bestofk.py for each model/seed run.
It writes a global benchmark summary while preserving per-run progress files.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vcsr_env  # noqa: F401
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip().lower())
    slug = slug.strip("._-")
    return slug or "model"


def _configure_file_logging(output_dir: Path) -> Path:
    log_path = output_dir / "progress.log"
    resolved = str(log_path.resolve())
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "") == resolved:
            return log_path
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger.addHandler(file_handler)
    return log_path


def _flush_logs() -> None:
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_progress(
    output_dir: Path,
    *,
    stage: str,
    total_runs: int,
    completed_runs: int,
    started_at: float,
    current_model: str | None = None,
    current_seed: int | None = None,
    failures: int = 0,
) -> None:
    elapsed_sec = max(0.0, time.time() - started_at)
    avg_sec_per_run = elapsed_sec / max(1, completed_runs) if completed_runs else None
    eta_sec = avg_sec_per_run * max(0, total_runs - completed_runs) if avg_sec_per_run is not None else None
    _write_json(
        output_dir / "progress.json",
        {
            "stage": stage,
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "remaining_runs": max(0, total_runs - completed_runs),
            "failed_runs": failures,
            "elapsed_sec": elapsed_sec,
            "avg_sec_per_run": avg_sec_per_run,
            "eta_sec": eta_sec,
            "current_model": current_model,
            "current_seed": current_seed,
        },
    )


def _clean_env(clear_proxy: bool) -> dict[str, str]:
    env = os.environ.copy()
    if clear_proxy:
        for key in PROXY_ENV_KEYS:
            env.pop(key, None)
    return env


def _run_command(command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]) -> int:
    logger.info("Running: %s", " ".join(command))
    _flush_logs()
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        return process.wait()


def _build_child_config(
    *,
    base_cfg: dict[str, Any],
    benchmark_cfg: dict[str, Any],
    model: dict[str, str],
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    child = deepcopy(base_cfg)
    dataset_cfg = benchmark_cfg.get("dataset", {})
    generation_cfg = benchmark_cfg.get("generation", {})
    repair_cfg = benchmark_cfg.get("repair", {})
    verifier_cfg = benchmark_cfg.get("verifier", {})
    backend_type = generation_cfg.get("backend_type", "openrouter")
    model_id = model["model_id"]
    generation_backend_kwargs = dict(generation_cfg.get("backend_kwargs", {}))
    repair_backend_kwargs = dict(repair_cfg.get("backend_kwargs", generation_backend_kwargs))

    child.setdefault("experiment", {})["name"] = f"vcsr_model_benchmark_{model['name']}"
    child["experiment"]["seed"] = int(seed)
    child.setdefault("dataset", {}).update(
        {
            "split": dataset_cfg.get("split", "test"),
            "domains": dataset_cfg.get("domains", ["blocksworld", "gripper"]),
            "max_samples": int(dataset_cfg.get("max_samples", 30)),
        }
    )
    child.setdefault("generation", {}).update(
        {
            "clear_proxy_env": bool(generation_cfg.get("clear_proxy_env", True)),
            "backends": [
                {
                    "type": backend_type,
                    "model": model_id,
                    "K": int(generation_cfg.get("K", 8)),
                    **generation_backend_kwargs,
                }
            ],
            "temperature": float(generation_cfg.get("temperature", 0.8)),
            "top_p": float(generation_cfg.get("top_p", 0.95)),
            "K_values": [int(k) for k in benchmark_cfg.get("k_values", [1, 4, 8])],
            "max_new_tokens": int(generation_cfg.get("max_new_tokens", 1024)),
            "retry_attempts": int(generation_cfg.get("retry_attempts", 3)),
            "retry_delay_sec": float(generation_cfg.get("retry_delay_sec", 2)),
        }
    )
    child.setdefault("verifier", {}).update(
        {
            "selection_metadata": verifier_cfg.get(
                "selection_metadata", "results/verifier/best_current/selection.yaml"
            ),
            "scoring_batch_size": int(verifier_cfg.get("scoring_batch_size", 8)),
        }
    )
    child.setdefault("repair", {}).update(
        {
            "enabled": bool(repair_cfg.get("enabled", True)),
            "K": int(repair_cfg.get("K", 8)),
            "generation": {
                "backends": [
                    {
                        "type": backend_type,
                        "model": model_id,
                        "K": 1,
                        **repair_backend_kwargs,
                    }
                ],
                "temperature": float(repair_cfg.get("temperature", 0.2)),
                "top_p": float(repair_cfg.get("top_p", 0.9)),
                "max_new_tokens": int(
                    repair_cfg.get("max_new_tokens", generation_cfg.get("max_new_tokens", 1024))
                ),
                "retry_attempts": int(repair_cfg.get("retry_attempts", generation_cfg.get("retry_attempts", 3))),
                "retry_delay_sec": float(
                    repair_cfg.get("retry_delay_sec", generation_cfg.get("retry_delay_sec", 2))
                ),
            },
        }
    )
    child.setdefault("evaluation", {}).update(
        {
            "check_parse": True,
            "check_equivalence": True,
            "planner_filter": False,
            "policies": benchmark_cfg.get(
                "policies",
                ["greedy_first", "random_parseable", "verifier_ranked", "verifier_ranked_repair"],
            ),
        }
    )
    child.setdefault("output", {})["dir"] = str(run_dir)
    child["output"]["save_predictions"] = True
    return child


def _policy_metric(summary: dict[str, Any], k: int, policy: str, metric: str) -> float:
    return float(summary["comparisons"][str(k)]["policies"][policy]["metrics"][metric])


def _policy_counts(summary: dict[str, Any], k: int, policy: str) -> dict[str, int]:
    metrics = summary["comparisons"][str(k)]["policies"][policy]["metrics"]
    return {
        "total": int(metrics["total"]),
        "parse_count": int(metrics["parse_count"]),
        "equiv_count": int(metrics["equiv_count"]),
    }


def _stratified_counts(summary: dict[str, Any], k: int, policy: str, slice_name: str) -> dict[str, int]:
    strata = summary["comparisons"][str(k)]["policies"][policy]["stratified"].get(slice_name, {})
    return {
        "total": int(strata.get("total", 0)),
        "parse_count": int(strata.get("parse_count", 0)),
        "equiv_count": int(strata.get("equiv_count", 0)),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _latency_summary(run_dir: Path) -> dict[str, Any]:
    candidate_rows = _read_jsonl(run_dir / "candidate_dump.jsonl")
    repair_rows = _read_jsonl(run_dir / "repair_outputs.jsonl")
    generation_latencies = [float(r.get("latency_sec", 0.0)) for r in candidate_rows if r.get("latency_sec") is not None]
    repair_latencies = [float(r.get("latency_sec", 0.0)) for r in repair_rows if r.get("latency_sec") is not None]
    generation_errors = sum(1 for r in candidate_rows if r.get("error"))
    repair_errors = sum(1 for r in repair_rows if r.get("repair_error") or r.get("error"))
    return {
        "generation_calls": len(candidate_rows),
        "repair_calls": len(repair_rows),
        "generation_errors": generation_errors,
        "repair_errors": repair_errors,
        "generation_latency_total_sec": sum(generation_latencies),
        "repair_latency_total_sec": sum(repair_latencies),
        "generation_latency_mean_sec": mean(generation_latencies) if generation_latencies else 0.0,
        "repair_latency_mean_sec": mean(repair_latencies) if repair_latencies else 0.0,
    }


def _merge_counts(target: Counter[str], counts: dict[str, int]) -> None:
    for key, value in counts.items():
        target[key] += int(value)


def _rate(counts: Counter[str], numerator: str, denominator: str = "total") -> float:
    denom = counts.get(denominator, 0)
    return counts.get(numerator, 0) / denom if denom else 0.0


def _aggregate_successful_runs(
    run_records: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    summary_cfg = cfg.get("summary", {})
    main_k = int(summary_cfg.get("main_k", 8))
    prompt_policy = summary_cfg.get("prompt_policy", "greedy_first")
    random_policy = summary_cfg.get("random_policy", "random_parseable")
    verifier_policy = summary_cfg.get("verifier_policy", "verifier_ranked")
    repair_policy = summary_cfg.get("repair_policy", "verifier_ranked_repair")
    k_values = [int(k) for k in cfg.get("k_values", [1, 4, 8])]
    policies = list(cfg.get("policies", [prompt_policy, random_policy, verifier_policy, repair_policy]))

    model_metrics: dict[str, dict[str, Any]] = {}
    for record in run_records:
        model_name = record["model_name"]
        model_block = model_metrics.setdefault(
            model_name,
            {
                "model_id": record["model_id"],
                "runs": [],
                "metrics": defaultdict(list),
                "counts": defaultdict(Counter),
                "slice_counts": defaultdict(Counter),
                "latency": Counter(),
            },
        )
        summary = record["summary"]
        run_dir = Path(record["output_dir"])
        latency = _latency_summary(run_dir)
        for key, value in latency.items():
            model_block["latency"][key] += value

        prompt_equiv = _policy_metric(summary, 1, prompt_policy, "equiv_rate")
        verifier_equiv = _policy_metric(summary, main_k, verifier_policy, "equiv_rate")
        random_equiv = _policy_metric(summary, main_k, random_policy, "equiv_rate")
        repair_equiv = _policy_metric(summary, main_k, repair_policy, "equiv_rate")
        model_block["runs"].append(
            {
                "seed": record["seed"],
                "output_dir": record["output_dir"],
                "prompt_only_equiv": prompt_equiv,
                "bestof8_random_parseable_equiv": random_equiv,
                "bestof8_verifier_ranked_equiv": verifier_equiv,
                "bestof8_vcsr_repair_equiv": repair_equiv,
                "verifier_minus_prompt": verifier_equiv - prompt_equiv,
                "repair_minus_prompt": repair_equiv - prompt_equiv,
                "repair_minus_verifier": repair_equiv - verifier_equiv,
                "latency": latency,
            }
        )
        for k in k_values:
            for policy in policies:
                key = f"k{k}:{policy}"
                model_block["metrics"][key].append(
                    {
                        "parse_rate": _policy_metric(summary, k, policy, "parse_rate"),
                        "equiv_rate": _policy_metric(summary, k, policy, "equiv_rate"),
                        "equiv_given_parse": _policy_metric(summary, k, policy, "equiv_given_parse"),
                    }
                )
                _merge_counts(model_block["counts"][key], _policy_counts(summary, k, policy))
                if k == main_k:
                    for slice_name in (
                        "domain=blocksworld",
                        "domain=gripper",
                        "style=abstract/abstract",
                        "style=explicit/explicit",
                    ):
                        _merge_counts(
                            model_block["slice_counts"][f"{key}:{slice_name}"],
                            _stratified_counts(summary, k, policy, slice_name),
                        )

    final_models = {}
    for model_name, block in sorted(model_metrics.items()):
        metric_means = {}
        for key, values in block["metrics"].items():
            metric_means[key] = {
                "parse_rate": mean([v["parse_rate"] for v in values]) if values else 0.0,
                "equiv_rate": mean([v["equiv_rate"] for v in values]) if values else 0.0,
                "equiv_given_parse": mean([v["equiv_given_parse"] for v in values]) if values else 0.0,
                "counts": dict(block["counts"][key]),
            }
        slice_breakdown = {}
        for key, counts in block["slice_counts"].items():
            slice_breakdown[key] = {
                "counts": dict(counts),
                "parse_rate": _rate(counts, "parse_count"),
                "equiv_rate": _rate(counts, "equiv_count"),
            }
        runs = block["runs"]
        final_models[model_name] = {
            "model_id": block["model_id"],
            "num_successful_runs": len(runs),
            "metrics": metric_means,
            "main_deltas": {
                "mean_verifier_minus_prompt": mean([r["verifier_minus_prompt"] for r in runs]) if runs else 0.0,
                "mean_repair_minus_prompt": mean([r["repair_minus_prompt"] for r in runs]) if runs else 0.0,
                "mean_repair_minus_verifier": mean([r["repair_minus_verifier"] for r in runs]) if runs else 0.0,
            },
            "per_seed": runs,
            "k8_slice_breakdown": slice_breakdown,
            "latency": dict(block["latency"]),
        }
    return final_models


def _markdown(report: dict[str, Any]) -> str:
    main_k = report["main_k"]
    lines = [
        "# Multi-Model Prompt-Only vs VCSR Benchmark",
        "",
        "This is a post-paper robustness benchmark. It does not replace the frozen",
        "paper evidence from seeds `51-55`.",
        "",
        f"Output root: `{report['output_dir']}`",
        f"Seeds: `{report['seeds']}`",
        f"Rows per seed: `{report['rows_per_seed']}`",
        f"Main K: `{main_k}`",
        "",
        "## Main Equivalence Summary",
        "",
        "| Model | Prompt K=1 | Random K=8 | Verifier K=8 | VCSR Repair K=8 | Repair - Prompt | Repair - Verifier | Runs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, block in report["models"].items():
        metrics = block["metrics"]
        prompt = metrics.get("k1:greedy_first", {}).get("equiv_rate", 0.0)
        random_val = metrics.get(f"k{main_k}:random_parseable", {}).get("equiv_rate", 0.0)
        verifier = metrics.get(f"k{main_k}:verifier_ranked", {}).get("equiv_rate", 0.0)
        repair = metrics.get(f"k{main_k}:verifier_ranked_repair", {}).get("equiv_rate", 0.0)
        deltas = block["main_deltas"]
        lines.append(
            f"| {model_name} | {prompt:.4f} | {random_val:.4f} | {verifier:.4f} | "
            f"{repair:.4f} | {deltas['mean_repair_minus_prompt']:+.4f} | "
            f"{deltas['mean_repair_minus_verifier']:+.4f} | {block['num_successful_runs']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Seed Main Deltas",
            "",
            "| Model | Seed | Prompt K=1 | Verifier K=8 | VCSR Repair K=8 | Repair - Prompt | Repair - Verifier |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name, block in report["models"].items():
        for row in block["per_seed"]:
            lines.append(
                f"| {model_name} | {row['seed']} | {row['prompt_only_equiv']:.4f} | "
                f"{row['bestof8_verifier_ranked_equiv']:.4f} | {row['bestof8_vcsr_repair_equiv']:.4f} | "
                f"{row['repair_minus_prompt']:+.4f} | {row['repair_minus_verifier']:+.4f} |"
            )

    lines.extend(
        [
            "",
            "## Failed Runs",
            "",
            "| Model | Seed | Output Dir | Error |",
            "|---|---:|---|---|",
        ]
    )
    if report["failed_runs"]:
        for row in report["failed_runs"]:
            error = str(row.get("error", "")).replace("|", "\\|")
            lines.append(f"| {row['model_name']} | {row['seed']} | `{row['output_dir']}` | {error} |")
    else:
        lines.append("| none |  |  |  |")

    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- If VCSR repair improves most models over prompt-only, VCSR is a useful wrapper.",
            "- If only the reference model improves, keep this as a limitation/generalization analysis.",
            "- If a strong prompt-only model beats VCSR, report the cost/quality tradeoff honestly.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vcsr_model_benchmark.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)
    base_cfg = _load_yaml(Path(cfg["base_config"]))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_file_logging(output_dir)
    started_at = time.time()
    _write_json(
        output_dir / "process_info.json",
        {
            "pid": os.getpid(),
            "started_at": started_at,
            "config": str(cfg_path),
            "output_dir": str(output_dir),
        },
    )
    with open(output_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    repo_root = Path(__file__).resolve().parent.parent
    models = cfg.get("models", [])
    seeds = [int(seed) for seed in cfg.get("seeds", [])]
    total_runs = len(models) * len(seeds)
    force = bool(args.force or cfg.get("execution", {}).get("force", False))
    continue_on_error = bool(cfg.get("execution", {}).get("continue_on_error", True))
    clear_proxy = bool(cfg.get("generation", {}).get("clear_proxy_env", True))
    env = _clean_env(clear_proxy)

    completed_runs = 0
    failed_runs: list[dict[str, Any]] = []
    successful_runs: list[dict[str, Any]] = []

    for model in models:
        model_name = str(model["name"])
        model_id = str(model["model_id"])
        model_dir = output_dir / _slug(model_name)
        for seed in seeds:
            run_dir = model_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            summary_path = run_dir / "aggregate_metrics.json"
            generated_cfg_path = run_dir / "benchmark_run_config.yaml"
            child_cfg = _build_child_config(
                base_cfg=base_cfg,
                benchmark_cfg=cfg,
                model={"name": model_name, "model_id": model_id},
                seed=seed,
                run_dir=run_dir,
            )
            with open(generated_cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(child_cfg, f, default_flow_style=False, sort_keys=False)

            _write_progress(
                output_dir,
                stage="run_model_seed",
                total_runs=total_runs,
                completed_runs=completed_runs,
                started_at=started_at,
                current_model=model_name,
                current_seed=seed,
                failures=len(failed_runs),
            )

            if not args.summarize_only and (force or not summary_path.exists()):
                return_code = _run_command(
                    [
                        sys.executable,
                        "-u",
                        "scripts/run_verifier_bestofk.py",
                        "--config",
                        str(generated_cfg_path),
                        "--output_dir",
                        str(run_dir),
                    ],
                    cwd=repo_root,
                    log_path=run_dir / "run_stdout.log",
                    env=env,
                )
                if return_code != 0:
                    failure = {
                        "model_name": model_name,
                        "model_id": model_id,
                        "seed": seed,
                        "output_dir": str(run_dir),
                        "return_code": return_code,
                        "error": f"child run failed with exit code {return_code}",
                    }
                    failed_runs.append(failure)
                    logger.error("%s", failure["error"])
                    completed_runs += 1
                    if not continue_on_error:
                        raise RuntimeError(failure["error"])
                    continue
            elif args.summarize_only and not summary_path.exists():
                logger.info("Skipping missing run in summarize-only mode: %s", run_dir)
                continue
            else:
                logger.info("Using existing benchmark run: %s", run_dir)

            try:
                summary = _read_json(summary_path)
                successful_runs.append(
                    {
                        "model_name": model_name,
                        "model_id": model_id,
                        "seed": seed,
                        "output_dir": str(run_dir),
                        "summary": summary,
                    }
                )
            except Exception as exc:
                failure = {
                    "model_name": model_name,
                    "model_id": model_id,
                    "seed": seed,
                    "output_dir": str(run_dir),
                    "error": str(exc),
                }
                failed_runs.append(failure)
                logger.error("Failed to read summary for %s: %s", run_dir, exc)
                if not continue_on_error:
                    raise

            completed_runs += 1
            _flush_logs()

    report = {
        "config": str(cfg_path),
        "base_config": cfg["base_config"],
        "output_dir": str(output_dir),
        "seeds": seeds,
        "rows_per_seed": int(cfg.get("dataset", {}).get("max_samples", 30)),
        "main_k": int(cfg.get("summary", {}).get("main_k", 8)),
        "successful_runs": len(successful_runs),
        "failed_runs": failed_runs,
        "models": _aggregate_successful_runs(successful_runs, cfg),
        "note": (
            "Post-paper robustness benchmark. Do not treat as replacement evidence "
            "for frozen final seeds 51-55."
        ),
    }
    _write_json(output_dir / "benchmark_summary.json", report)
    with open(output_dir / "benchmark_summary.md", "w", encoding="utf-8") as f:
        f.write(_markdown(report))
    _write_progress(
        output_dir,
        stage="complete",
        total_runs=total_runs,
        completed_runs=completed_runs,
        started_at=started_at,
        failures=len(failed_runs),
    )
    logger.info("Wrote benchmark summary to %s", output_dir / "benchmark_summary.md")


if __name__ == "__main__":
    main()
