"""
Analyze the fresh multi-seed round-4 vs round-7 gate at row level.

This script does not train, generate, or call any model APIs. It compares the
already-written candidate dumps from the fresh multi-seed gate and separates
verifier selection outcomes from candidate-pool availability effects.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vcsr_env  # noqa: F401

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


def _flush_logs() -> None:
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _style(row: dict[str, Any]) -> str:
    init = "abstract" if int(row.get("init_is_abstract", 0)) else "explicit"
    goal = "abstract" if int(row.get("goal_is_abstract", 0)) else "explicit"
    return f"{init}/{goal}"


def _row_key(record: dict[str, Any]) -> str:
    return f"{record.get('row_index')}::{record['planetarium_name']}"


def _load_candidate_dump(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = _row_key(record)
            name = record["planetarium_name"]
            row = rows.setdefault(
                key,
                {
                    "row_index": record.get("row_index"),
                    "planetarium_name": name,
                    "row_key": key,
                    "domain": record.get("domain"),
                    "init_is_abstract": record.get("init_is_abstract"),
                    "goal_is_abstract": record.get("goal_is_abstract"),
                    "style": _style(record),
                    "candidates": {},
                    "selections": defaultdict(dict),
                },
            )
            if "candidate_index" in record:
                row["candidates"][int(record["candidate_index"])] = record
            elif "policy" in record and "K" in record:
                row["selections"][int(record["K"])][record["policy"]] = record

    for row in rows.values():
        row["selections"] = {str(k): dict(v) for k, v in row["selections"].items()}
    return rows


def _safe_score(candidate: dict[str, Any] | None) -> float | None:
    if not candidate:
        return None
    score = candidate.get("verifier_score")
    return float(score) if score is not None else None


def _snapshot(row: dict[str, Any] | None, k: int, policy: str = "verifier_ranked") -> dict[str, Any]:
    if row is None:
        return {
            "present": False,
            "selected_index": None,
            "selected_parseable": False,
            "selected_equivalent": False,
            "selected_score": None,
            "parseable_count": 0,
            "equivalent_count": 0,
            "oracle_available": False,
            "best_equiv_index": None,
            "best_equiv_score": None,
            "best_wrong_index": None,
            "best_wrong_score": None,
            "selected_wrong_minus_best_equiv": None,
        }

    prefix = [c for i, c in sorted(row["candidates"].items()) if i < k]
    parseable = [c for c in prefix if bool(c.get("parseable"))]
    equiv = [c for c in parseable if bool(c.get("equivalent"))]
    wrong = [c for c in parseable if not bool(c.get("equivalent"))]
    best_equiv = max(equiv, key=lambda c: (_safe_score(c) if _safe_score(c) is not None else float("-inf"), -c["candidate_index"]), default=None)
    best_wrong = max(wrong, key=lambda c: (_safe_score(c) if _safe_score(c) is not None else float("-inf"), -c["candidate_index"]), default=None)

    selection = row["selections"].get(str(k), {}).get(policy, {})
    selected_index = selection.get("selected_index")
    selected = row["candidates"].get(int(selected_index)) if selected_index is not None else None
    selected_score = _safe_score(selected)
    best_equiv_score = _safe_score(best_equiv)
    margin = None
    if selected is not None and not bool(selected.get("equivalent")) and best_equiv_score is not None and selected_score is not None:
        margin = selected_score - best_equiv_score

    return {
        "present": True,
        "selected_index": selected_index,
        "selection_reason": selection.get("selection_reason"),
        "selected_parseable": bool(selected.get("parseable")) if selected else False,
        "selected_equivalent": bool(selected.get("equivalent")) if selected else False,
        "selected_score": selected_score,
        "parseable_count": len(parseable),
        "equivalent_count": len(equiv),
        "oracle_available": bool(equiv),
        "best_equiv_index": best_equiv.get("candidate_index") if best_equiv else None,
        "best_equiv_score": best_equiv_score,
        "best_wrong_index": best_wrong.get("candidate_index") if best_wrong else None,
        "best_wrong_score": _safe_score(best_wrong),
        "selected_wrong_minus_best_equiv": margin,
    }


def _outcome(r4: dict[str, Any], r7: dict[str, Any]) -> str:
    if r7["selected_equivalent"] and not r4["selected_equivalent"]:
        return "round7_helped"
    if r4["selected_equivalent"] and not r7["selected_equivalent"]:
        return "round7_hurt"
    if r4["selected_equivalent"] and r7["selected_equivalent"]:
        return "both_success"
    return "both_fail"


def _cause(r4: dict[str, Any], r7: dict[str, Any], outcome: str) -> str:
    if r7["oracle_available"] and not r4["oracle_available"]:
        return "oracle_availability_gain"
    if r4["oracle_available"] and not r7["oracle_available"]:
        return "oracle_availability_loss"
    if r4["oracle_available"] and r7["oracle_available"]:
        if outcome == "round7_helped":
            return "round7_selector_gain_with_oracle"
        if outcome == "round7_hurt":
            return "round7_selector_loss_with_oracle"
        return "both_oracle_available_same_outcome"
    return "both_no_oracle"


def _counter_table(counter: Counter, label: str) -> list[str]:
    lines = [
        f"### {label}",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for key, value in counter.most_common():
        lines.append(f"| {key} | {value} |")
    if not counter:
        lines.append("| none | 0 |")
    lines.append("")
    return lines


def _pct(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _markdown(report: dict[str, Any], changed_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Round 7 Fresh Gate Analysis",
        "",
        "This is a fixed-artifact analysis only. It does not generate, train, or rescore.",
        "",
        "Important caveat: the fresh multi-seed gate used separate generation runs for round 4 and round 7, so changed rows combine selector behavior with candidate-pool variance. Oracle-availability changes are therefore tracked separately from within-pool selection losses.",
        "",
        "## Top Line",
        "",
        "| K | Rows | Round 4 Eq | Round 7 Eq | Delta | Round 7 Helped | Round 7 Hurt | Ties |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k_key, summary in report["by_k"].items():
        lines.append(
            f"| {k_key} | {summary['rows']} | {summary['round4_equiv_rate']:.4f} | "
            f"{summary['round7_equiv_rate']:.4f} | {summary['delta']:+.4f} | "
            f"{summary['round7_helped']} | {summary['round7_hurt']} | {summary['ties']} |"
        )

    lines.extend(["", "## Per-Seed K=8", "", "| Seed | Round 4 Eq | Round 7 Eq | Delta | Helped | Hurt | Main Cause |", "|---:|---:|---:|---:|---:|---:|---|"])
    for seed_key, summary in report["seed_k8"].items():
        main_cause = summary["cause_counts"][0][0] if summary["cause_counts"] else "none"
        lines.append(
            f"| {seed_key} | {summary['round4_equiv_rate']:.4f} | {summary['round7_equiv_rate']:.4f} | "
            f"{summary['delta']:+.4f} | {summary['round7_helped']} | {summary['round7_hurt']} | {main_cause} |"
        )

    lines.extend(["", "## Cause Counts", ""])
    for k_key, summary in report["by_k"].items():
        lines.extend(_counter_table(Counter(dict(summary["cause_counts"])), f"K={k_key}"))

    lines.extend(["## Seed 56 K=8 Hurt Rows", "", "| Row | Domain | Style | Cause | R4 Eq Cnt | R7 Eq Cnt | R4 Sel | R7 Sel | R7 Wrong - Best Eq |", "|---|---|---|---|---:|---:|---:|---:|---:|"])
    seed56_hurts = [
        r
        for r in changed_rows
        if r["seed"] == 56 and r["K"] == 8 and r["outcome"] == "round7_hurt"
    ]
    for row in seed56_hurts[:20]:
        margin = row["round7"]["selected_wrong_minus_best_equiv"]
        margin_text = f"{margin:+.4f}" if margin is not None else ""
        lines.append(
            f"| {row['planetarium_name']} | {row['domain']} | {row['style']} | {row['cause']} | "
            f"{row['round4']['equivalent_count']} | {row['round7']['equivalent_count']} | "
            f"{row['round4']['selected_index']} | {row['round7']['selected_index']} | {margin_text} |"
        )
    if not seed56_hurts:
        lines.append("| none |  |  |  |  |  |  |  |  |")

    lines.extend(["", "## Interpretation", ""])
    interpretation = report["interpretation"]
    for item in interpretation:
        lines.append(f"- {item}")

    lines.extend(["", "## Recommendation", "", report["recommendation"], ""])
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_file_logging(output_dir)
    started_at = time.time()
    _write_json(
        output_dir / "process_info.json",
        {
            "pid": os.getpid(),
            "started_at": started_at,
            "input_dir": str(root),
            "baseline_name": args.baseline_name,
            "candidate_name": args.candidate_name,
            "seeds": args.seeds,
            "k_values": args.k_values,
        },
    )

    logger.info("Loading fresh gate candidate dumps")
    _write_json(output_dir / "progress.json", {"stage": "loading", "completed": False})
    data: dict[str, dict[int, dict[str, dict[str, Any]]]] = {args.baseline_name: {}, args.candidate_name: {}}
    for verifier in [args.baseline_name, args.candidate_name]:
        for seed in args.seeds:
            path = root / verifier / f"seed_{seed}" / "candidate_dump.jsonl"
            data[verifier][seed] = _load_candidate_dump(path)
            logger.info("Loaded %s seed %s rows=%s", verifier, seed, len(data[verifier][seed]))

    changed_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        r4_rows = data[args.baseline_name][seed]
        r7_rows = data[args.candidate_name][seed]
        for row_key in sorted(set(r4_rows) | set(r7_rows)):
            r4_row = r4_rows.get(row_key)
            r7_row = r7_rows.get(row_key)
            meta = r4_row or r7_row or {}
            for k in args.k_values:
                r4 = _snapshot(r4_row, k)
                r7 = _snapshot(r7_row, k)
                outcome = _outcome(r4, r7)
                cause = _cause(r4, r7, outcome)
                record = {
                    "seed": seed,
                    "K": k,
                    "row_key": row_key,
                    "planetarium_name": meta.get("planetarium_name"),
                    "domain": meta.get("domain"),
                    "style": meta.get("style"),
                    "row_index": meta.get("row_index"),
                    "outcome": outcome,
                    "cause": cause,
                    "round4": r4,
                    "round7": r7,
                }
                all_rows.append(record)
                if outcome in {"round7_helped", "round7_hurt"} or cause in {"oracle_availability_gain", "oracle_availability_loss"}:
                    changed_rows.append(record)

    report: dict[str, Any] = {
        "input_dir": str(root),
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "seeds": args.seeds,
        "k_values": args.k_values,
        "by_k": {},
        "seed_k8": {},
    }

    for k in args.k_values:
        rows = [r for r in all_rows if r["K"] == k]
        r4_success = sum(1 for r in rows if r["round4"]["selected_equivalent"])
        r7_success = sum(1 for r in rows if r["round7"]["selected_equivalent"])
        helped = sum(1 for r in rows if r["outcome"] == "round7_helped")
        hurt = sum(1 for r in rows if r["outcome"] == "round7_hurt")
        cause_counts = Counter(r["cause"] for r in rows)
        outcome_counts = Counter(r["outcome"] for r in rows)
        report["by_k"][str(k)] = {
            "rows": len(rows),
            "round4_equiv_count": r4_success,
            "round7_equiv_count": r7_success,
            "round4_equiv_rate": _pct(r4_success, len(rows)),
            "round7_equiv_rate": _pct(r7_success, len(rows)),
            "delta": _pct(r7_success, len(rows)) - _pct(r4_success, len(rows)),
            "round7_helped": helped,
            "round7_hurt": hurt,
            "ties": len(rows) - helped - hurt,
            "outcome_counts": outcome_counts.most_common(),
            "cause_counts": cause_counts.most_common(),
            "round4_oracle_available": sum(1 for r in rows if r["round4"]["oracle_available"]),
            "round7_oracle_available": sum(1 for r in rows if r["round7"]["oracle_available"]),
        }

    for seed in args.seeds:
        rows = [r for r in all_rows if r["K"] == 8 and r["seed"] == seed]
        r4_success = sum(1 for r in rows if r["round4"]["selected_equivalent"])
        r7_success = sum(1 for r in rows if r["round7"]["selected_equivalent"])
        report["seed_k8"][str(seed)] = {
            "rows": len(rows),
            "round4_equiv_rate": _pct(r4_success, len(rows)),
            "round7_equiv_rate": _pct(r7_success, len(rows)),
            "delta": _pct(r7_success, len(rows)) - _pct(r4_success, len(rows)),
            "round7_helped": sum(1 for r in rows if r["outcome"] == "round7_helped"),
            "round7_hurt": sum(1 for r in rows if r["outcome"] == "round7_hurt"),
            "cause_counts": Counter(r["cause"] for r in rows).most_common(),
        }

    for key in ["domain", "style"]:
        report[f"by_{key}"] = {}
        for k in args.k_values:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in all_rows:
                if row["K"] == k:
                    grouped[str(row.get(key))].append(row)
            report[f"by_{key}"][str(k)] = {
                group: {
                    "rows": len(rows),
                    "round4_equiv_rate": mean([1.0 if r["round4"]["selected_equivalent"] else 0.0 for r in rows]),
                    "round7_equiv_rate": mean([1.0 if r["round7"]["selected_equivalent"] else 0.0 for r in rows]),
                    "round7_helped": sum(1 for r in rows if r["outcome"] == "round7_helped"),
                    "round7_hurt": sum(1 for r in rows if r["outcome"] == "round7_hurt"),
                }
                for group, rows in grouped.items()
            }

    seed56_k8 = [r for r in all_rows if r["seed"] == 56 and r["K"] == 8]
    seed56_hurts = [r for r in seed56_k8 if r["outcome"] == "round7_hurt"]
    seed56_selection_losses = [r for r in seed56_hurts if r["cause"] == "round7_selector_loss_with_oracle"]
    seed56_oracle_losses = [r for r in seed56_hurts if r["cause"] == "oracle_availability_loss"]
    report["seed56_k8_focus"] = {
        "rows": len(seed56_k8),
        "round7_hurts": len(seed56_hurts),
        "selection_losses_with_oracle": len(seed56_selection_losses),
        "oracle_availability_losses": len(seed56_oracle_losses),
        "round4_oracle_available": sum(1 for r in seed56_k8 if r["round4"]["oracle_available"]),
        "round7_oracle_available": sum(1 for r in seed56_k8 if r["round7"]["oracle_available"]),
    }

    interpretation = []
    k8 = report["by_k"]["8"]
    if abs(k8["delta"]) < 0.005:
        interpretation.append("Round 7 tied round 4 at K=8 overall on the fresh gate, so it is not promotion-worthy despite replay gains.")
    if report["seed56_k8_focus"]["oracle_availability_losses"] or report["seed56_k8_focus"]["selection_losses_with_oracle"]:
        interpretation.append(
            "The seed-56 K=8 drop is mixed: it includes both candidate-pool/oracle-availability effects and within-pool round-7 selector losses."
        )
    if k8["round7_helped"] > 0 and k8["round7_hurt"] > 0:
        interpretation.append("Round 7 is not uniformly worse; it helped and hurt different rows, which supports keeping it as evidence but not as best_current.")
    if report["by_k"]["4"]["delta"] > 0:
        interpretation.append("Round 7 improved K=4 slightly on fresh seeds, but the project operating point is K=8 and the K=8 gate did not move.")

    report["interpretation"] = interpretation
    report["recommendation"] = (
        "Keep round 4 as the official `best_current`. Treat round 7 as a useful focused-pointwise diagnostic, not a promoted model. "
        "The next step should not be another blind retrain; either design a fixed-pool fresh comparison to isolate verifier effects, "
        "or shift attention to candidate-pool/generator diversity because fresh-pool variance is now large enough to hide small verifier gains."
    )

    _write_json(output_dir / "analysis_summary.json", report)
    _append_jsonl(output_dir / "changed_rows.jsonl", changed_rows)
    _append_jsonl(output_dir / "seed56_k8_cases.jsonl", seed56_k8)
    with open(output_dir / "analysis_summary.md", "w", encoding="utf-8") as f:
        f.write(_markdown(report, changed_rows))
    _write_json(
        output_dir / "progress.json",
        {
            "stage": "complete",
            "completed": True,
            "elapsed_sec": time.time() - started_at,
            "changed_rows": len(changed_rows),
            "output_dir": str(output_dir),
        },
    )
    _flush_logs()
    logger.info("Wrote analysis to %s", output_dir)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="results/vcsr/multiseed_round7_compare")
    parser.add_argument("--output_dir", default="results/vcsr/multiseed_round7_compare/fresh_gate_analysis")
    parser.add_argument("--baseline_name", default="round4_best_current")
    parser.add_argument("--candidate_name", default="round7_focused_pointwise")
    parser.add_argument("--seeds", nargs="+", type=int, default=[56, 57, 58])
    parser.add_argument("--k_values", nargs="+", type=int, default=[4, 8])
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
