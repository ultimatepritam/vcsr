"""
Run repeated fresh held-out verifier comparisons across multiple seeds.

This script launches `scripts/run_verifier_bestofk.py` repeatedly with different
verifier selections and dataset seeds, then aggregates the resulting metrics into
one decision-friendly summary.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vcsr_env  # noqa: F401
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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


def _flush_all_logs() -> None:
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def _write_progress_snapshot(
    output_dir: Path,
    *,
    total_runs: int,
    completed_runs: int,
    started_at: float,
    current_label: str = "",
) -> None:
    elapsed_sec = max(0.0, time.time() - started_at)
    avg_sec_per_run = elapsed_sec / max(1, completed_runs) if completed_runs else None
    eta_sec = avg_sec_per_run * max(0, total_runs - completed_runs) if avg_sec_per_run is not None else None
    snapshot = {
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "remaining_runs": max(0, total_runs - completed_runs),
        "elapsed_sec": elapsed_sec,
        "avg_sec_per_run": avg_sec_per_run,
        "eta_sec": eta_sec,
        "current_label": current_label,
    }
    with open(output_dir / "progress.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)


def _resolve_verifier_name(selection_path: str, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    selection_file = Path(selection_path)
    if selection_file.stem == "selection":
        return selection_file.parent.name
    return selection_file.stem


def _load_metrics(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_metric(summary: dict, k: int, policy: str, metric_name: str) -> float:
    return float(summary["comparisons"][str(k)]["policies"][policy]["metrics"][metric_name])


def _markdown(report: dict) -> str:
    lines = [
        "# Multi-Seed Fresh Held-Out Comparison",
        "",
        f"Base config: `{report['base_config']}`",
        f"Rows per run: `{report['rows_per_run']}`",
        f"Seeds: `{report['seeds']}`",
        "",
        "## Mean Metrics",
        "",
        "| Verifier | K | Policy | Mean Parse | Mean Equiv | Mean Equiv / Parse |",
        "|---|---:|---|---:|---:|---:|",
    ]

    for verifier in report["verifiers"]:
        for k_key, policies in verifier["mean_metrics"].items():
            for policy_name, metrics in policies.items():
                lines.append(
                    f"| {verifier['name']} | {k_key} | {policy_name} | "
                    f"{metrics['parse_rate']:.4f} | {metrics['equiv_rate']:.4f} | {metrics['equiv_given_parse']:.4f} |"
                )

    lines.extend(
        [
            "",
            "## Head-to-Head",
            "",
            "| K | Policy | Round4 Wins | Round3 Wins | Ties | Mean Delta |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["head_to_head"]:
        lines.append(
            f"| {row['K']} | {row['policy']} | {row['candidate_wins']} | {row['baseline_wins']} | "
            f"{row['ties']} | {row['mean_equiv_delta']:+.4f} |"
        )

    baseline_label = report["head_to_head"][0]["baseline"] if report.get("head_to_head") else "baseline"
    candidate_label = report["head_to_head"][0]["candidate"] if report.get("head_to_head") else "candidate"

    lines.extend(["", "## Per-Seed Verifier-Ranked", ""])
    for k_key, seed_rows in report["verifier_ranked_by_seed"].items():
        lines.append(f"### K={k_key}")
        lines.append("")
        lines.append(f"| Seed | {baseline_label} | {candidate_label} | Delta |")
        lines.append("|---:|---:|---:|---:|")
        for seed_row in seed_rows:
            lines.append(
                f"| {seed_row['seed']} | {seed_row['baseline_equiv_rate']:.4f} | "
                f"{seed_row['candidate_equiv_rate']:.4f} | {seed_row['delta']:+.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Recommendation",
            "",
            report["recommendation"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated fresh held-out verifier comparisons")
    parser.add_argument("--config", type=str, default="configs/vcsr_multiseed_holdout_compare.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_config = str(cfg["base_config"])
    seeds = [int(seed) for seed in cfg.get("seeds", [48, 49, 50])]
    output_dir = Path(cfg.get("output_dir", "results/vcsr/multiseed_holdout_compare"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_file_logging(output_dir)

    with open(output_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    verifier_specs = cfg.get("verifiers", [])
    if len(verifier_specs) < 2:
        raise ValueError("Provide at least two verifier specs for comparison.")

    rows_per_run = int(cfg.get("max_rows", 50))
    total_runs = len(seeds) * len(verifier_specs)
    started_at = time.time()

    logger.info("Starting multi-seed held-out comparison")
    logger.info("Live progress log: %s", log_path)
    _write_progress_snapshot(output_dir, total_runs=total_runs, completed_runs=0, started_at=started_at)
    _flush_all_logs()

    run_records: list[dict] = []
    completed_runs = 0

    for seed in seeds:
        for verifier_spec in verifier_specs:
            verifier_name = _resolve_verifier_name(
                verifier_spec.get("selection"),
                verifier_spec.get("name"),
            )
            run_label = f"{verifier_name}/seed_{seed}"
            child_output_dir = output_dir / verifier_name / f"seed_{seed}"
            child_output_dir.mkdir(parents=True, exist_ok=True)
            stdout_log = child_output_dir / "run_stdout.log"

            logger.info("Starting child run: %s", run_label)
            _write_progress_snapshot(
                output_dir,
                total_runs=total_runs,
                completed_runs=completed_runs,
                started_at=started_at,
                current_label=run_label,
            )
            _flush_all_logs()

            command = [
                sys.executable,
                "-u",
                "scripts/run_verifier_bestofk.py",
                "--config",
                base_config,
                "--max_rows",
                str(rows_per_run),
                "--seed",
                str(seed),
                "--selection_metadata",
                str(verifier_spec["selection"]),
                "--output_dir",
                str(child_output_dir),
            ]

            clear_proxy = bool(cfg.get("clear_proxy_env", False))
            env = os.environ.copy()
            if clear_proxy:
                for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
                    env[key] = ""

            with open(stdout_log, "w", encoding="utf-8") as stdout_handle:
                stdout_handle.write("COMMAND: " + " ".join(command) + "\n")
                stdout_handle.flush()
                result = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parent.parent,
                    env=env,
                    stdout=stdout_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )

            if result.returncode != 0:
                raise RuntimeError(f"Child run failed for {run_label}; see {stdout_log}")

            summary_path = child_output_dir / "aggregate_metrics.json"
            summary = _load_metrics(summary_path)
            run_records.append(
                {
                    "seed": seed,
                    "verifier": verifier_name,
                    "selection": verifier_spec["selection"],
                    "output_dir": str(child_output_dir),
                    "summary_path": str(summary_path),
                    "summary": summary,
                }
            )

            completed_runs += 1
            logger.info("Completed child run: %s", run_label)
            _write_progress_snapshot(
                output_dir,
                total_runs=total_runs,
                completed_runs=completed_runs,
                started_at=started_at,
                current_label=run_label,
            )
            _flush_all_logs()

    by_verifier: dict[str, list[dict]] = {}
    for record in run_records:
        by_verifier.setdefault(record["verifier"], []).append(record)

    report_verifiers = []
    for verifier_spec in verifier_specs:
        verifier_name = _resolve_verifier_name(verifier_spec.get("selection"), verifier_spec.get("name"))
        records = sorted(by_verifier.get(verifier_name, []), key=lambda item: int(item["seed"]))
        if not records:
            continue

        sample_summary = records[0]["summary"]
        k_values = [int(k) for k in sample_summary["comparisons"].keys()]
        mean_metrics = {}
        for k in k_values:
            mean_metrics[str(k)] = {}
            for policy in ["greedy_first", "random_parseable", "verifier_ranked"]:
                parse_rates = [_safe_metric(r["summary"], k, policy, "parse_rate") for r in records]
                equiv_rates = [_safe_metric(r["summary"], k, policy, "equiv_rate") for r in records]
                eq_parse_rates = [_safe_metric(r["summary"], k, policy, "equiv_given_parse") for r in records]
                mean_metrics[str(k)][policy] = {
                    "parse_rate": sum(parse_rates) / len(parse_rates),
                    "equiv_rate": sum(equiv_rates) / len(equiv_rates),
                    "equiv_given_parse": sum(eq_parse_rates) / len(eq_parse_rates),
                }

        report_verifiers.append(
            {
                "name": verifier_name,
                "selection": verifier_spec["selection"],
                "runs": [
                    {
                        "seed": r["seed"],
                        "output_dir": r["output_dir"],
                        "summary_path": r["summary_path"],
                    }
                    for r in records
                ],
                "mean_metrics": mean_metrics,
            }
        )

    baseline_name = _resolve_verifier_name(verifier_specs[0].get("selection"), verifier_specs[0].get("name"))
    candidate_name = _resolve_verifier_name(verifier_specs[1].get("selection"), verifier_specs[1].get("name"))
    baseline_by_seed = {int(r["seed"]): r for r in by_verifier[baseline_name]}
    candidate_by_seed = {int(r["seed"]): r for r in by_verifier[candidate_name]}

    head_to_head = []
    verifier_ranked_by_seed = {}
    for k in [4, 8]:
        seed_rows = []
        candidate_wins = 0
        baseline_wins = 0
        ties = 0
        deltas = []
        for seed in seeds:
            base_rate = _safe_metric(baseline_by_seed[seed]["summary"], k, "verifier_ranked", "equiv_rate")
            cand_rate = _safe_metric(candidate_by_seed[seed]["summary"], k, "verifier_ranked", "equiv_rate")
            delta = cand_rate - base_rate
            deltas.append(delta)
            if delta > 1e-12:
                candidate_wins += 1
            elif delta < -1e-12:
                baseline_wins += 1
            else:
                ties += 1
            seed_rows.append(
                {
                    "seed": seed,
                    "baseline_equiv_rate": base_rate,
                    "candidate_equiv_rate": cand_rate,
                    "delta": delta,
                }
            )

        verifier_ranked_by_seed[str(k)] = seed_rows
        head_to_head.append(
            {
                "K": k,
                "policy": "verifier_ranked",
                "baseline": baseline_name,
                "candidate": candidate_name,
                "candidate_wins": candidate_wins,
                "baseline_wins": baseline_wins,
                "ties": ties,
                "mean_equiv_delta": sum(deltas) / len(deltas),
            }
        )

    recommend_promotion = all(
        row["candidate_wins"] >= row["baseline_wins"] and row["mean_equiv_delta"] > 0.0
        for row in head_to_head
    )
    if recommend_promotion:
        recommendation = (
            f"Promote `{candidate_name}` only if you are comfortable treating this multi-seed result as the new "
            f"end-to-end gate: it beat or matched `{baseline_name}` across the evaluated fresh held-out seeds and "
            "showed a positive mean verifier-ranked delta at both K=4 and K=8."
        )
    else:
        recommendation = (
            f"Keep `{baseline_name}` as the official frozen baseline for now. "
            f"`{candidate_name}` remains a provisional candidate, but the multi-seed fresh held-out gate did not show "
            "a clean enough verifier-ranked win profile at both K=4 and K=8 to justify promotion."
        )

    report = {
        "base_config": base_config,
        "rows_per_run": rows_per_run,
        "seeds": seeds,
        "clear_proxy_env": bool(cfg.get("clear_proxy_env", False)),
        "verifiers": report_verifiers,
        "head_to_head": head_to_head,
        "verifier_ranked_by_seed": verifier_ranked_by_seed,
        "recommendation": recommendation,
    }

    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(output_dir / "comparison_summary.md", "w", encoding="utf-8") as f:
        f.write(_markdown(report))

    logger.info("Saved comparison summary to %s", output_dir / "comparison_summary.json")
    logger.info("Saved comparison markdown to %s", output_dir / "comparison_summary.md")
    _flush_all_logs()


if __name__ == "__main__":
    main()
