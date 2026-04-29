"""
Replay guarded repair decisions on existing repair artifacts without generation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vcsr_env  # noqa: F401
import yaml


DEFAULT_CONFIG = {
    "input": {
        "dev_gate_summary": "results/vcsr/fresh_repair_gate_round4_domainaware/fresh_repair_gate_summary.json",
        "repair_glob": "results/vcsr/fresh_repair_gate_round4_domainaware/seed_*/repair/repair_outputs.jsonl",
    },
    "margins": [0.00, 0.01, 0.02, 0.05],
    "output_dir": "results/vcsr/guarded_repair_analysis",
}


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def _seed_from_path(path: str) -> int | None:
    for part in Path(path).parts:
        if part.startswith("seed_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                return None
    return None


def _guard_accept(row: dict[str, Any], margin: float) -> bool:
    if not row.get("repair_parseable"):
        return False
    original_score = row.get("original_selected_score")
    repair_score = row.get("repair_verifier_score")
    if original_score is None or repair_score is None:
        return False
    return float(repair_score) >= float(original_score) - float(margin)


def _outcome(original_equiv: bool, final_equiv: bool) -> str:
    if final_equiv and not original_equiv:
        return "repair_helped"
    if original_equiv and not final_equiv:
        return "repair_hurt"
    if final_equiv and original_equiv:
        return "both_success"
    return "both_fail"


def _metrics(rows: list[dict[str, Any]], margin: float) -> dict[str, Any]:
    total = len(rows)
    accepted = 0
    parseable = 0
    final_equiv = 0
    helped = 0
    hurt = 0
    outcomes = defaultdict(int)
    for row in rows:
        accept = _guard_accept(row, margin)
        accepted += int(accept)
        parseable += int(bool(row.get("repair_parseable")))
        original_equiv = bool(row.get("original_selected_equivalent"))
        selected_equiv = bool(row.get("repair_equivalent")) if accept else original_equiv
        final_equiv += int(selected_equiv)
        outcome = _outcome(original_equiv, selected_equiv)
        outcomes[outcome] += 1
        helped += int(outcome == "repair_helped")
        hurt += int(outcome == "repair_hurt")
    return {
        "repair_rows": total,
        "accepted": accepted,
        "rejected": total - accepted,
        "accept_rate": accepted / total if total else 0.0,
        "repair_parse_rate": parseable / total if total else 0.0,
        "final_equiv_count_on_repaired_rows": final_equiv,
        "final_equiv_rate_on_repaired_rows": final_equiv / total if total else 0.0,
        "helped": helped,
        "hurt": hurt,
        "tied": total - helped - hurt,
        "outcome_counts": dict(sorted(outcomes.items())),
    }


def _breakdown(rows: list[dict[str, Any]], margin: float, key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    return {name: _metrics(group_rows, margin) for name, group_rows in sorted(grouped.items())}


def _full_policy_metrics(
    *,
    rows: list[dict[str, Any]],
    margin: float,
    seed_reports: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        seed = _seed_from_path(row["_source_file"])
        if seed is not None:
            by_seed[seed].append(row)

    seed_rows = []
    for seed, repair_rows in sorted(by_seed.items()):
        report = seed_reports.get(seed, {})
        row_count = int(report.get("rows", 0) or 0)
        baseline_rate = float(report.get("baseline_k8_equiv_rate", 0.0))
        baseline_equiv = int(round(baseline_rate * row_count))
        row_metrics = _metrics(repair_rows, margin)
        guarded_equiv = baseline_equiv + row_metrics["helped"] - row_metrics["hurt"]
        seed_rows.append(
            {
                "seed": seed,
                "rows": row_count,
                "baseline_k8_equiv_rate": baseline_rate,
                "guarded_k8_equiv_rate": guarded_equiv / row_count if row_count else 0.0,
                "delta_k8": (guarded_equiv / row_count - baseline_rate) if row_count else 0.0,
                "repair_metrics": row_metrics,
            }
        )

    baseline_values = [row["baseline_k8_equiv_rate"] for row in seed_rows]
    guarded_values = [row["guarded_k8_equiv_rate"] for row in seed_rows]
    deltas = [row["delta_k8"] for row in seed_rows]
    return {
        "seed_reports": seed_rows,
        "mean_baseline_k8_equiv_rate": mean(baseline_values) if baseline_values else 0.0,
        "mean_guarded_k8_equiv_rate": mean(guarded_values) if guarded_values else 0.0,
        "mean_delta_k8": mean(deltas) if deltas else 0.0,
        "total_helped": sum(row["repair_metrics"]["helped"] for row in seed_rows),
        "total_hurt": sum(row["repair_metrics"]["hurt"] for row in seed_rows),
    }


def _choose_margin(margin_reports: list[dict[str, Any]]) -> float:
    best = sorted(
        margin_reports,
        key=lambda row: (
            -row["full_policy"]["mean_guarded_k8_equiv_rate"],
            row["repair_metrics"]["hurt"],
            row["margin"],
        ),
    )[0]
    return float(best["margin"])


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Guarded Repair Policy Analysis",
        "",
        f"Selected margin: `{report['selected_margin']:.2f}`",
        "",
        "| Margin | Mean Baseline K=8 | Mean Guarded K=8 | Delta | Accepted | Helped | Hurt |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["margin_reports"]:
        full = row["full_policy"]
        metrics = row["repair_metrics"]
        lines.append(
            f"| {row['margin']:.2f} | {full['mean_baseline_k8_equiv_rate']:.4f} | "
            f"{full['mean_guarded_k8_equiv_rate']:.4f} | {full['mean_delta_k8']:+.4f} | "
            f"{metrics['accepted']} | {metrics['helped']} | {metrics['hurt']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Use the selected margin for fresh guarded-repair validation. "
            "This analysis uses development repair artifacts only and does not tune on seeds `51-55`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    _write_json(
        output_dir / "process_info.json",
        {
            "pid": os.getpid(),
            "started_at": started_at,
            "config": args.config,
            "output_dir": str(output_dir),
        },
    )
    _write_json(output_dir / "progress.json", {"stage": "loading", "started_at": started_at})

    gate_summary = _read_json(Path(cfg["input"]["dev_gate_summary"]))
    seed_reports = {int(row["seed"]): row for row in gate_summary.get("seed_reports", [])}
    repair_files = sorted(Path().glob(cfg["input"]["repair_glob"]))
    rows: list[dict[str, Any]] = []
    for path in repair_files:
        rows.extend(_read_jsonl(path))

    margin_reports = []
    for margin in [float(value) for value in cfg.get("margins", [0.0])]:
        margin_reports.append(
            {
                "margin": margin,
                "repair_metrics": _metrics(rows, margin),
                "full_policy": _full_policy_metrics(rows=rows, margin=margin, seed_reports=seed_reports),
                "domain_breakdown": _breakdown(rows, margin, "domain"),
                "style_breakdown": _breakdown(rows, margin, "style"),
            }
        )
    selected_margin = _choose_margin(margin_reports)
    report = {
        "input": cfg["input"],
        "margins": cfg.get("margins", [0.0]),
        "repair_rows": len(rows),
        "selected_margin": selected_margin,
        "margin_reports": margin_reports,
    }
    _write_json(output_dir / "guarded_repair_analysis.json", report)
    with open(output_dir / "guarded_repair_analysis.md", "w", encoding="utf-8") as f:
        f.write(_markdown(report))
    _write_json(output_dir / "progress.json", {"stage": "complete", "repair_rows": len(rows)})


if __name__ == "__main__":
    main()
