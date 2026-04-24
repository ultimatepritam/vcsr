"""
Analyze where pairwise round 5 changed round-4 fixed-pool replay decisions.

The output is intentionally row-level: it shows whether round 5 helped, hurt,
or tied round 4 for verifier-ranked selection on cached candidate pools.
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


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_process_info(output_dir: Path, command: list[str]) -> None:
    _write_json(
        output_dir / "process_info.json",
        {
            "pid": os.getpid(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "command": " ".join(command),
            "output_dir": str(output_dir),
            "progress_log": str(output_dir / "progress.log"),
            "progress_json": str(output_dir / "progress.json"),
        },
    )


def _write_progress(output_dir: Path, *, status: str, started_at: float, completed: int, total: int) -> None:
    _write_json(
        output_dir / "progress.json",
        {
            "status": status,
            "completed_replay_dumps": completed,
            "total_replay_dumps": total,
            "elapsed_sec": max(0.0, time.time() - started_at),
        },
    )


def _load_replay_dump(path: Path) -> tuple[dict[str, dict[int, dict[int, dict]]], dict[str, dict[int, dict[int, dict]]]]:
    candidates: dict[str, dict[int, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    selections: dict[str, dict[int, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            verifier = record["verifier"]
            row_index = int(record["row_index"])
            if "candidate_index" in record:
                candidates[verifier][row_index][int(record["candidate_index"])] = record
            elif record.get("policy") == "verifier_ranked":
                selections[verifier][row_index][int(record["K"])] = record
    return candidates, selections


def _selected_candidate(
    candidates: dict[int, dict],
    selection: dict | None,
) -> dict | None:
    if not selection:
        return None
    selected_index = selection.get("selected_index")
    if selected_index is None:
        return None
    return candidates.get(int(selected_index))


def _style(record: dict) -> str:
    init = "abstract" if int(record.get("init_is_abstract", 0)) else "explicit"
    goal = "abstract" if int(record.get("goal_is_abstract", 0)) else "explicit"
    return f"{init}/{goal}"


def _analyze_one(path: Path, *, baseline_name: str, candidate_name: str, k_values: list[int]) -> tuple[list[dict], dict]:
    candidates, selections = _load_replay_dump(path)
    if baseline_name not in candidates or candidate_name not in candidates:
        available = sorted(candidates)
        raise ValueError(f"{path} does not contain both requested verifiers. Available: {available}")

    changed_rows: list[dict] = []
    counters = Counter()
    by_domain = Counter()
    by_style = Counter()
    by_k = Counter()

    row_indices = sorted(set(candidates[baseline_name]) & set(candidates[candidate_name]))
    for row_index in row_indices:
        base_candidates = candidates[baseline_name][row_index]
        cand_candidates = candidates[candidate_name][row_index]
        meta_source = next(iter(base_candidates.values()), {})
        for k_value in k_values:
            base_sel = _selected_candidate(base_candidates, selections[baseline_name][row_index].get(k_value))
            cand_sel = _selected_candidate(cand_candidates, selections[candidate_name][row_index].get(k_value))
            if base_sel is None or cand_sel is None:
                counters["missing_selection"] += 1
                continue

            base_equiv = bool(base_sel.get("equivalent"))
            cand_equiv = bool(cand_sel.get("equivalent"))
            if base_equiv == cand_equiv:
                counters["same_outcome"] += 1
                continue

            direction = "round5_helped" if cand_equiv and not base_equiv else "round5_hurt"
            counters[direction] += 1
            by_domain[meta_source.get("domain", "")] += 1
            by_style[_style(meta_source)] += 1
            by_k[str(k_value)] += 1
            changed_rows.append(
                {
                    "source_replay_dump": str(path),
                    "row_index": row_index,
                    "planetarium_name": meta_source.get("planetarium_name", ""),
                    "domain": meta_source.get("domain", ""),
                    "style": _style(meta_source),
                    "K": k_value,
                    "direction": direction,
                    "baseline_verifier": baseline_name,
                    "candidate_verifier": candidate_name,
                    "baseline_selected_index": int(base_sel["candidate_index"]),
                    "candidate_selected_index": int(cand_sel["candidate_index"]),
                    "baseline_selected_equivalent": base_equiv,
                    "candidate_selected_equivalent": cand_equiv,
                    "baseline_score": base_sel.get("replay_verifier_score"),
                    "candidate_score": cand_sel.get("replay_verifier_score"),
                }
            )

    summary = {
        "replay_dump": str(path),
        "baseline": baseline_name,
        "candidate": candidate_name,
        "rows_compared": len(row_indices),
        "k_values": k_values,
        "counts": dict(counters),
        "changed_rows": len(changed_rows),
        "by_domain": dict(by_domain.most_common()),
        "by_style": dict(by_style.most_common()),
        "by_k": dict(sorted(by_k.items())),
    }
    return changed_rows, summary


def _markdown(report: dict) -> str:
    lines = [
        "# Round 5 Regression Analysis",
        "",
        f"Baseline verifier: `{report['baseline']}`",
        f"Candidate verifier: `{report['candidate']}`",
        "",
        "## Top Line",
        "",
        f"- Replay dumps analyzed: `{len(report['replay_summaries'])}`",
        f"- Changed outcome rows: `{report['total_changed_rows']}`",
        f"- Round 5 helped rows: `{report['direction_counts'].get('round5_helped', 0)}`",
        f"- Round 5 hurt rows: `{report['direction_counts'].get('round5_hurt', 0)}`",
        "",
        "## Breakdown",
        "",
        f"- By K: `{report['by_k']}`",
        f"- By domain: `{report['by_domain']}`",
        f"- By style: `{report['by_style']}`",
        "",
        "## Highest-Value Changed Rows",
        "",
        "| Direction | K | Row | Domain | Style | Baseline idx | Candidate idx |",
        "|---|---:|---|---|---|---:|---:|",
    ]
    for row in report["changed_rows"][:20]:
        lines.append(
            f"| {row['direction']} | {row['K']} | `{row['planetarium_name']}` | {row['domain']} | "
            f"{row['style']} | {row['baseline_selected_index']} | {row['candidate_selected_index']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Round 5 should be treated as diagnostic signal, not a promoted checkpoint. "
            "Rows where round 5 hurt round 4 are priority negatives for round 6, while rows where it helped "
            "show what a ranking objective should preserve.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze round-5 replay regressions versus round 4")
    parser.add_argument(
        "--replay_dump",
        action="append",
        default=[
            "results/vcsr/bestofk_round4_holdout_eval_clean/replay_compare_round4_vs_pairwise_round5/replay_dump.jsonl",
            "results/vcsr/bestofk_round3_holdout_eval/replay_compare_round4_vs_pairwise_round5/replay_dump.jsonl",
        ],
    )
    parser.add_argument("--baseline_name", default="retrain_from_round3_focused")
    parser.add_argument("--candidate_name", default="retrain_from_round4_hybrid_pairwise")
    parser.add_argument("--k_values", type=int, nargs="*", default=[4, 8])
    parser.add_argument("--output_dir", default="results/verifier/pairwise_round5/regression_analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_file_logging(output_dir)
    _write_process_info(output_dir, sys.argv)
    started_at = time.time()
    replay_paths = [Path(path) for path in args.replay_dump]
    _write_progress(output_dir, status="starting", started_at=started_at, completed=0, total=len(replay_paths))

    all_changed: list[dict] = []
    replay_summaries: list[dict] = []
    for idx, replay_path in enumerate(replay_paths, start=1):
        logger.info("Analyzing replay dump %d/%d: %s", idx, len(replay_paths), replay_path)
        changed, summary = _analyze_one(
            replay_path,
            baseline_name=args.baseline_name,
            candidate_name=args.candidate_name,
            k_values=sorted(set(args.k_values)),
        )
        all_changed.extend(changed)
        replay_summaries.append(summary)
        _write_progress(output_dir, status="analyzing", started_at=started_at, completed=idx, total=len(replay_paths))

    direction_counts = Counter(row["direction"] for row in all_changed)
    by_domain = Counter(row["domain"] for row in all_changed)
    by_style = Counter(row["style"] for row in all_changed)
    by_k = Counter(str(row["K"]) for row in all_changed)
    all_changed.sort(key=lambda row: (row["direction"] != "round5_hurt", row["K"], row["planetarium_name"]))

    report = {
        "baseline": args.baseline_name,
        "candidate": args.candidate_name,
        "k_values": sorted(set(args.k_values)),
        "replay_summaries": replay_summaries,
        "total_changed_rows": len(all_changed),
        "direction_counts": dict(direction_counts.most_common()),
        "by_domain": dict(by_domain.most_common()),
        "by_style": dict(by_style.most_common()),
        "by_k": dict(sorted(by_k.items())),
        "changed_rows": all_changed,
    }

    _write_json(output_dir / "summary.json", report)
    with open(output_dir / "changed_rows.jsonl", "w", encoding="utf-8") as f:
        for row in all_changed:
            f.write(json.dumps(row) + "\n")
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(_markdown(report))
    _write_progress(output_dir, status="completed", started_at=started_at, completed=len(replay_paths), total=len(replay_paths))
    logger.info("Saved regression analysis to %s", output_dir)


if __name__ == "__main__":
    main()
