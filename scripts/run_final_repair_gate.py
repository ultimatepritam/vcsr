"""
Run the final fresh repair-augmented VCSR gate across untouched seeds.

Each child run uses `scripts/run_verifier_bestofk.py` with repair enabled, so
the final artifacts are produced through the normal best-of-K entrypoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
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
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


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


def _write_progress(
    output_dir: Path,
    *,
    stage: str,
    total_runs: int,
    completed_runs: int,
    started_at: float,
    current_seed: int | None = None,
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
            "elapsed_sec": elapsed_sec,
            "avg_sec_per_run": avg_sec_per_run,
            "eta_sec": eta_sec,
            "current_seed": current_seed,
        },
    )


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in PROXY_ENV_KEYS:
        env.pop(key, None)
    return env


def _run_command(command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]) -> None:
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
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")


def _policy_metric(summary: dict[str, Any], k: int, policy: str, metric: str) -> float:
    return float(summary["comparisons"][str(k)]["policies"][policy]["metrics"][metric])


def _repair_outcome_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    parse_count = sum(1 for row in rows if row.get("final_parseable", row.get("repair_parseable")))
    equiv_count = sum(1 for row in rows if row.get("final_equivalent", row.get("repair_equivalent")))
    helped = sum(1 for row in rows if row.get("outcome") == "repair_helped")
    hurt = sum(1 for row in rows if row.get("outcome") == "repair_hurt")
    return {
        "total": total,
        "repair_parse_count": parse_count,
        "repair_equiv_count": equiv_count,
        "repair_parse_rate": parse_count / total if total else 0.0,
        "repair_equiv_rate": equiv_count / total if total else 0.0,
        "repair_equiv_given_parse": equiv_count / parse_count if parse_count else 0.0,
        "helped": helped,
        "hurt": hurt,
        "tied": total - helped - hurt,
    }


def _group_repair_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "unknown")), []).append(row)
    return {name: _repair_outcome_metrics(group_rows) for name, group_rows in sorted(grouped.items())}


def _repair_rows_for_policy(summary: dict[str, Any], policy: str) -> list[dict[str, Any]]:
    path = summary.get("repair", {}).get("repair_outputs")
    if not path:
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("policy") == policy:
                rows.append(row)
    return rows


def _aggregate(run_records: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    k = int(cfg.get("acceptance", {}).get("k", 8))
    baseline_policy = cfg.get("acceptance", {}).get("baseline_policy", "verifier_ranked")
    repair_policy = cfg.get("acceptance", {}).get("repair_policy", "verifier_ranked_repair")
    comparison_policy = cfg.get("acceptance", {}).get("comparison_policy")
    baseline_values = []
    repair_values = []
    comparison_values = []
    seed_reports = []
    total_helped = 0
    total_hurt = 0
    total_comparison_hurt = 0
    total_blocksworld_hurt = 0
    total_comparison_blocksworld_hurt = 0
    repair_parse_rates = []

    for record in run_records:
        summary = record["summary"]
        baseline = _policy_metric(summary, k, baseline_policy, "equiv_rate")
        repaired = _policy_metric(summary, k, repair_policy, "equiv_rate")
        compared = _policy_metric(summary, k, comparison_policy, "equiv_rate") if comparison_policy else None
        repair_summary = summary.get("repair", {})
        repair_rows = _repair_rows_for_policy(summary, repair_policy)
        comparison_rows = _repair_rows_for_policy(summary, comparison_policy) if comparison_policy else []
        repair_metrics = _repair_outcome_metrics(repair_rows) if repair_rows else repair_summary.get(
            "policy_breakdown", {}
        ).get(repair_policy, repair_summary.get("metrics", {}))
        repair_domain_breakdown = _group_repair_metrics(repair_rows, "domain") if repair_rows else {}
        comparison_metrics = _repair_outcome_metrics(comparison_rows) if comparison_rows else {}
        comparison_domain_breakdown = _group_repair_metrics(comparison_rows, "domain") if comparison_rows else {}
        baseline_values.append(baseline)
        repair_values.append(repaired)
        if compared is not None:
            comparison_values.append(compared)
        total_helped += int(repair_metrics.get("helped", 0))
        total_hurt += int(repair_metrics.get("hurt", 0))
        total_comparison_hurt += int(comparison_metrics.get("hurt", 0))
        total_blocksworld_hurt += int(repair_domain_breakdown.get("blocksworld", {}).get("hurt", 0))
        total_comparison_blocksworld_hurt += int(
            comparison_domain_breakdown.get("blocksworld", {}).get("hurt", 0)
        )
        repair_parse_rates.append(float(repair_metrics.get("repair_parse_rate", 0.0)))
        seed_reports.append(
            {
                "seed": record["seed"],
                "output_dir": record["output_dir"],
                "baseline_k8_equiv_rate": baseline,
                "repair_augmented_k8_equiv_rate": repaired,
                "delta_k8": repaired - baseline,
                "comparison_k8_equiv_rate": compared,
                "repair_metrics": repair_metrics,
                "repair_domain_breakdown": repair_domain_breakdown,
                "comparison_metrics": comparison_metrics,
                "comparison_domain_breakdown": comparison_domain_breakdown,
                "repair_summary": repair_summary,
            }
        )

    deltas = [row["delta_k8"] for row in seed_reports]
    acceptance_cfg = cfg.get("acceptance", {})
    min_mean_delta = float(acceptance_cfg.get("min_mean_delta", 0.10))
    min_nonnegative = int(acceptance_cfg.get("min_nonnegative_seeds", 4))
    max_large_regression = float(acceptance_cfg.get("max_large_regression", -0.05))
    min_parse_rate = float(acceptance_cfg.get("min_repair_parse_rate", 0.90))
    require_blocksworld_hurt_reduction = bool(acceptance_cfg.get("require_blocksworld_hurt_reduction", False))
    mean_delta = mean(deltas) if deltas else 0.0
    nonnegative_seeds = sum(1 for delta in deltas if delta >= 0)
    worst_delta = min(deltas) if deltas else 0.0
    mean_parse = mean(repair_parse_rates) if repair_parse_rates else 0.0
    comparison_mean = mean(comparison_values) if comparison_values else None
    comparison_ok = True
    if comparison_mean is not None:
        comparison_ok = mean(repair_values) >= comparison_mean - float(
            acceptance_cfg.get("max_mean_loss_vs_comparison", 0.0)
        )
    blocksworld_hurt_ok = True
    if require_blocksworld_hurt_reduction:
        blocksworld_hurt_ok = total_blocksworld_hurt < total_comparison_blocksworld_hurt
    accepted = (
        mean_delta >= min_mean_delta
        and nonnegative_seeds >= min_nonnegative
        and worst_delta >= max_large_regression
        and total_helped > total_hurt
        and mean_parse >= min_parse_rate
        and comparison_ok
        and blocksworld_hurt_ok
    )
    return {
        "base_config": cfg["base_config"],
        "seeds": cfg["seeds"],
        "rows_per_seed": cfg.get("max_rows"),
        "k": k,
        "baseline_policy": baseline_policy,
        "repair_policy": repair_policy,
        "seed_reports": seed_reports,
        "mean_metrics": {
            "baseline_k8_equiv_rate": mean(baseline_values) if baseline_values else 0.0,
            "repair_augmented_k8_equiv_rate": mean(repair_values) if repair_values else 0.0,
            "comparison_k8_equiv_rate": comparison_mean,
            "mean_delta_k8": mean_delta,
            "baseline_k8_values": baseline_values,
            "repair_augmented_k8_values": repair_values,
            "deltas": deltas,
            "nonnegative_seeds": nonnegative_seeds,
            "worst_delta": worst_delta,
            "repair_parse_rate_mean": mean_parse,
            "total_helped": total_helped,
            "total_hurt": total_hurt,
            "total_comparison_hurt": total_comparison_hurt,
            "total_blocksworld_hurt": total_blocksworld_hurt,
            "total_comparison_blocksworld_hurt": total_comparison_blocksworld_hurt,
            "comparison_ok": comparison_ok,
            "blocksworld_hurt_ok": blocksworld_hurt_ok,
        },
        "accepted": accepted,
        "recommendation": (
            "Repair-augmented VCSR passes the final fresh gate."
            if accepted
            else "Repair-augmented VCSR does not pass the final fresh gate; keep the claim cautious."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Final Repair-Augmented VCSR Gate",
        "",
        f"Base config: `{report['base_config']}`",
        f"Rows per seed: `{report['rows_per_seed']}`",
        f"Seeds: `{report['seeds']}`",
        f"Baseline policy: `{report.get('baseline_policy', 'verifier_ranked')}`",
        f"Repair policy: `{report.get('repair_policy', 'verifier_ranked_repair')}`",
        "",
        "## Mean K=8 Equivalence",
        "",
        "| Policy | Mean Equiv | Per-Seed Values |",
        "|---|---:|---|",
        (
            "| Round-4 verifier-ranked | "
            f"{report['mean_metrics']['baseline_k8_equiv_rate']:.4f} | "
            f"{', '.join(f'{v:.4f}' for v in report['mean_metrics']['baseline_k8_values'])} |"
        ),
        (
            "| Repair-augmented | "
            f"{report['mean_metrics']['repair_augmented_k8_equiv_rate']:.4f} | "
            f"{', '.join(f'{v:.4f}' for v in report['mean_metrics']['repair_augmented_k8_values'])} |"
        ),
    ]
    if report["mean_metrics"].get("comparison_k8_equiv_rate") is not None:
        lines.append(
            "| Comparison repair policy | "
            f"{report['mean_metrics']['comparison_k8_equiv_rate']:.4f} | "
            "see per-seed aggregate metrics |"
        )
    lines.extend(
        [
            "",
            "## Per-Seed Results",
            "",
            "| Seed | Baseline K=8 | Repair K=8 | Delta | Helped | Hurt | Repair Parse |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["seed_reports"]:
        metrics = row["repair_metrics"]
        lines.append(
            f"| {row['seed']} | {row['baseline_k8_equiv_rate']:.4f} | "
            f"{row['repair_augmented_k8_equiv_rate']:.4f} | {row['delta_k8']:+.4f} | "
            f"{metrics.get('helped', 0)} | {metrics.get('hurt', 0)} | "
            f"{metrics.get('repair_parse_rate', 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Guard Diagnostics",
            "",
            f"Total hurt rows: `{report['mean_metrics'].get('total_hurt', 0)}`",
            f"Comparison hurt rows: `{report['mean_metrics'].get('total_comparison_hurt', 0)}`",
            f"Blocksworld hurt rows: `{report['mean_metrics'].get('total_blocksworld_hurt', 0)}`",
            (
                "Comparison blocksworld hurt rows: "
                f"`{report['mean_metrics'].get('total_comparison_blocksworld_hurt', 0)}`"
            ),
            "",
            "## Acceptance",
            "",
            f"Accepted: `{report['accepted']}`",
            "",
            report["recommendation"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vcsr_final_repair_gate.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_file_logging(output_dir)
    started_at = time.time()
    with open(output_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    _write_json(
        output_dir / "process_info.json",
        {
            "pid": os.getpid(),
            "started_at": started_at,
            "config": args.config,
            "output_dir": str(output_dir),
        },
    )

    repo_root = Path(__file__).resolve().parent.parent
    env = _clean_env()
    seeds = [int(seed) for seed in cfg["seeds"]]
    total_runs = len(seeds)
    completed_runs = 0
    force = bool(args.force or cfg.get("force", False))
    run_records = []
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        summary_path = seed_dir / "aggregate_metrics.json"
        _write_progress(
            output_dir,
            stage="seed_run",
            total_runs=total_runs,
            completed_runs=completed_runs,
            started_at=started_at,
            current_seed=seed,
        )
        if force or not summary_path.exists():
            _run_command(
                [
                    sys.executable,
                    "-u",
                    "scripts/run_verifier_bestofk.py",
                    "--config",
                    cfg["base_config"],
                    "--seed",
                    str(seed),
                    "--max_rows",
                    str(cfg.get("max_rows", 50)),
                    "--output_dir",
                    str(seed_dir),
                ],
                cwd=repo_root,
                log_path=seed_dir / "run_stdout.log",
                env=env,
            )
        else:
            logger.info("Using existing final-gate run for seed %s", seed)
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        run_records.append({"seed": seed, "output_dir": str(seed_dir), "summary": summary})
        completed_runs += 1
        _flush_logs()

    report = _aggregate(run_records, cfg)
    _write_json(output_dir / "final_repair_gate_summary.json", report)
    with open(output_dir / "final_repair_gate_summary.md", "w", encoding="utf-8") as f:
        f.write(_markdown(report))
    _write_progress(
        output_dir,
        stage="complete",
        total_runs=total_runs,
        completed_runs=completed_runs,
        started_at=started_at,
    )
    logger.info("Wrote final repair gate summary to %s", output_dir / "final_repair_gate_summary.md")


if __name__ == "__main__":
    main()
