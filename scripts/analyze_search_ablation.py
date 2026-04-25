"""
Fixed-pool search-policy ablation around the promoted round-4 verifier.

This script does not generate or train. It replays cached candidate pools,
rescoring candidates with the current best verifier and adding Planetarium
oracle-planner solvability as a zero-training search signal.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vcsr_env  # noqa: F401
import yaml

from data.planetarium_loader import PlanetariumDataset
from pddl_utils.oracle_planner import check_solvability_oracle
from verifier.inference import VerifierScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_POOLS = [
    "results/vcsr/bestofk_round4_holdout_eval_clean/candidate_dump.jsonl",
    "results/vcsr/fixed_pool_round7_compare/pools/seed_59/candidate_dump.jsonl",
    "results/vcsr/fixed_pool_round7_compare/pools/seed_60/candidate_dump.jsonl",
    "results/vcsr/fixed_pool_round7_compare/pools/seed_61/candidate_dump.jsonl",
    "results/vcsr/bestofk_round3_holdout_eval/candidate_dump.jsonl",
    "results/vcsr/bestofk_pilot/candidate_dump.jsonl",
]

POLICIES = [
    "verifier_ranked",
    "solvable_then_verifier",
    "verifier_then_solvable_tiebreak",
    "parse_solvable_index",
]


@dataclass
class Candidate:
    index: int
    parseable: bool
    equivalent: bool
    pddl: str
    verifier_score: float | None = None
    solvable: bool = False
    planner_error: str | None = None


@dataclass
class Selection:
    policy: str
    selected_index: int | None
    reason: str


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
    completed_pools: int,
    total_pools: int,
    current_pool: str,
    started_at: float,
) -> None:
    _write_json(
        output_dir / "progress.json",
        {
            "stage": stage,
            "completed_pools": completed_pools,
            "total_pools": total_pools,
            "remaining_pools": max(0, total_pools - completed_pools),
            "current_pool": current_pool,
            "elapsed_sec": max(0.0, time.time() - started_at),
        },
    )


def _style(row: dict[str, Any]) -> str:
    init = "abstract" if int(row.get("init_is_abstract", 0)) else "explicit"
    goal = "abstract" if int(row.get("goal_is_abstract", 0)) else "explicit"
    return f"{init}/{goal}"


def _pool_name(candidate_dump: Path) -> str:
    parent = candidate_dump.parent
    if parent.name.startswith("seed_"):
        return f"{parent.parent.parent.name}/{parent.name}"
    return parent.name


def _load_run_config(candidate_dump: Path) -> dict[str, Any]:
    with open(candidate_dump.parent / "run_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_nl_lookup(candidate_dump: Path) -> dict[str, str]:
    cfg = _load_run_config(candidate_dump)
    ds_cfg = cfg.get("dataset", {})
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    dataset = PlanetariumDataset(
        split_strategy=ds_cfg.get("split_strategy", "template_hash"),
        seed=seed,
    )
    rows = dataset.get_split(ds_cfg.get("split", "test"))
    return {row["name"]: row["natural_language"] for row in rows}


def _load_candidate_dump(candidate_dump: Path) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    meta_by_row: dict[int, dict[str, Any]] = {}
    candidates_by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with open(candidate_dump, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if "candidate_index" not in record:
                continue
            row_index = int(record["row_index"])
            meta_by_row[row_index] = {
                "row_index": row_index,
                "planetarium_name": record["planetarium_name"],
                "domain": record["domain"],
                "init_is_abstract": int(record.get("init_is_abstract", 0)),
                "goal_is_abstract": int(record.get("goal_is_abstract", 0)),
                "style": _style(record),
            }
            candidates_by_row[row_index].append(record)
    for records in candidates_by_row.values():
        records.sort(key=lambda rec: int(rec["candidate_index"]))
    return meta_by_row, candidates_by_row


def _score_or_cached(
    *,
    scorer: VerifierScorer,
    nl_lookup: dict[str, str],
    meta_by_row: dict[int, dict[str, Any]],
    candidates_by_row: dict[int, list[dict[str, Any]]],
    batch_size: int,
) -> dict[int, list[Candidate]]:
    pairs: list[tuple[str, str]] = []
    refs: list[tuple[int, int]] = []
    scored: dict[int, list[Candidate]] = {}
    for row_index, records in candidates_by_row.items():
        name = meta_by_row[row_index]["planetarium_name"]
        nl = nl_lookup[name]
        scored[row_index] = []
        for rec in records:
            candidate = Candidate(
                index=int(rec["candidate_index"]),
                parseable=bool(rec.get("parseable")),
                equivalent=bool(rec.get("equivalent")),
                pddl=rec.get("pddl") or "",
            )
            scored[row_index].append(candidate)
            if candidate.parseable and candidate.pddl:
                pairs.append((nl, candidate.pddl))
                refs.append((row_index, candidate.index))

    scores = scorer.score_pairs(pairs, batch_size=batch_size) if pairs else []
    lookup = {ref: float(score) for ref, score in zip(refs, scores)}
    for row_index, candidates in scored.items():
        for candidate in candidates:
            candidate.verifier_score = lookup.get((row_index, candidate.index))
    return scored


def _add_solvability(scored: dict[int, list[Candidate]]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    cache: dict[str, tuple[bool, str | None]] = {}
    for row_index, candidates in scored.items():
        for candidate in candidates:
            if not candidate.parseable or not candidate.pddl:
                candidate.solvable = False
                candidate.planner_error = "not_parseable"
            else:
                if candidate.pddl not in cache:
                    result = check_solvability_oracle(candidate.pddl)
                    cache[candidate.pddl] = (bool(result.solvable), result.error)
                candidate.solvable, candidate.planner_error = cache[candidate.pddl]
            diagnostics.append(
                {
                    "row_index": row_index,
                    "candidate_index": candidate.index,
                    "parseable": candidate.parseable,
                    "solvable": candidate.solvable,
                    "equivalent": candidate.equivalent,
                    "verifier_score": candidate.verifier_score,
                    "planner_error": candidate.planner_error,
                }
            )
    return diagnostics


def _score(candidate: Candidate) -> float:
    return float("-inf") if candidate.verifier_score is None else float(candidate.verifier_score)


def _parseable(candidates: list[Candidate]) -> list[Candidate]:
    return [candidate for candidate in candidates if candidate.parseable]


def _select_verifier_ranked(candidates: list[Candidate]) -> Selection:
    eligible = _parseable(candidates)
    if not eligible:
        return Selection("verifier_ranked", None, "no_parseable_candidate")
    chosen = max(eligible, key=lambda cand: (_score(cand), -cand.index))
    return Selection("verifier_ranked", chosen.index, "highest_verifier_score")


def _select_solvable_then_verifier(candidates: list[Candidate]) -> Selection:
    eligible = [candidate for candidate in candidates if candidate.parseable and candidate.solvable]
    if not eligible:
        return Selection("solvable_then_verifier", None, "no_parseable_solvable_candidate")
    chosen = max(eligible, key=lambda cand: (_score(cand), -cand.index))
    return Selection("solvable_then_verifier", chosen.index, "highest_verifier_score_among_solvable")


def _select_verifier_then_solvable_tiebreak(candidates: list[Candidate], margin: float) -> Selection:
    eligible = _parseable(candidates)
    if not eligible:
        return Selection("verifier_then_solvable_tiebreak", None, "no_parseable_candidate")
    top_score = max(_score(candidate) for candidate in eligible)
    near_top = [candidate for candidate in eligible if top_score - _score(candidate) <= margin]
    solvable_near_top = [candidate for candidate in near_top if candidate.solvable]
    if solvable_near_top:
        chosen = max(solvable_near_top, key=lambda cand: (_score(cand), -cand.index))
        return Selection("verifier_then_solvable_tiebreak", chosen.index, f"solvable_within_margin_{margin}")
    chosen = max(eligible, key=lambda cand: (_score(cand), -cand.index))
    return Selection("verifier_then_solvable_tiebreak", chosen.index, "no_solvable_near_top")


def _select_parse_solvable_index(candidates: list[Candidate]) -> Selection:
    for candidate in candidates:
        if candidate.parseable and candidate.solvable:
            return Selection("parse_solvable_index", candidate.index, "first_parseable_solvable")
    return Selection("parse_solvable_index", None, "no_parseable_solvable_candidate")


def _select(policy: str, candidates: list[Candidate], margin: float) -> Selection:
    if policy == "verifier_ranked":
        return _select_verifier_ranked(candidates)
    if policy == "solvable_then_verifier":
        return _select_solvable_then_verifier(candidates)
    if policy == "verifier_then_solvable_tiebreak":
        return _select_verifier_then_solvable_tiebreak(candidates, margin)
    if policy == "parse_solvable_index":
        return _select_parse_solvable_index(candidates)
    raise ValueError(f"Unknown policy: {policy}")


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    parse_count = sum(1 for row in rows if row["selected_parseable"])
    solvable_count = sum(1 for row in rows if row["selected_solvable"])
    equiv_count = sum(1 for row in rows if row["selected_equivalent"])
    return {
        "total": total,
        "parse_count": parse_count,
        "solvable_count": solvable_count,
        "equiv_count": equiv_count,
        "parse_rate": parse_count / total if total else 0.0,
        "solvable_rate": solvable_count / total if total else 0.0,
        "equiv_rate": equiv_count / total if total else 0.0,
        "equiv_given_parse": equiv_count / parse_count if parse_count else 0.0,
    }


def _breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {name: _metrics(group_rows) for name, group_rows in sorted(grouped.items())}


def _policy_report(policy_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_by_key = {(row["pool"], row["row_index"], row["K"]): row for row in baseline_rows}
    helped = hurt = tied = 0
    for row in policy_rows:
        baseline = baseline_by_key[(row["pool"], row["row_index"], row["K"])]
        if row["selected_equivalent"] and not baseline["selected_equivalent"]:
            helped += 1
        elif baseline["selected_equivalent"] and not row["selected_equivalent"]:
            hurt += 1
        else:
            tied += 1
    return {
        "metrics": _metrics(policy_rows),
        "domain_breakdown": _breakdown(policy_rows, "domain"),
        "style_breakdown": _breakdown(policy_rows, "style"),
        "helped_vs_verifier_ranked": helped,
        "hurt_vs_verifier_ranked": hurt,
        "tied_vs_verifier_ranked": tied,
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Round 4 Search Ablation",
        "",
        f"Verifier: `{summary['verifier_selection']}`",
        f"Pools: `{len(summary['pools'])}`",
        f"K values: `{summary['k_values']}`",
        "",
        "## Mean Metrics",
        "",
        "| Policy | K | Equiv | Parse | Solvable | Eq / Parse | Helped | Hurt | Tied |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k_key, policies in summary["mean_metrics"].items():
        for policy, report in policies.items():
            metrics = report["metrics"]
            lines.append(
                f"| {policy} | {k_key} | {metrics['equiv_rate']:.4f} | {metrics['parse_rate']:.4f} | "
                f"{metrics['solvable_rate']:.4f} | {metrics['equiv_given_parse']:.4f} | "
                f"{report['helped_vs_verifier_ranked']} | {report['hurt_vs_verifier_ranked']} | "
                f"{report['tied_vs_verifier_ranked']} |"
            )

    lines.extend(["", "## Per-Pool Verifier-Ranked vs Search", ""])
    lines.extend(["| Pool | K | verifier_ranked | solvable_then_verifier | verifier_then_solvable_tiebreak | parse_solvable_index |", "|---|---:|---:|---:|---:|---:|"])
    for pool_name, pool_report in summary["per_pool"].items():
        for k_key, policies in pool_report.items():
            lines.append(
                f"| {pool_name} | {k_key} | "
                f"{policies['verifier_ranked']['metrics']['equiv_rate']:.4f} | "
                f"{policies['solvable_then_verifier']['metrics']['equiv_rate']:.4f} | "
                f"{policies['verifier_then_solvable_tiebreak']['metrics']['equiv_rate']:.4f} | "
                f"{policies['parse_solvable_index']['metrics']['equiv_rate']:.4f} |"
            )

    lines.extend(["", "## Candidate-Pool Diagnostics", "", "| K | Oracle Best-of-K | Solvable Best-of-K | Avg Parseable | Avg Solvable |", "|---:|---:|---:|---:|---:|"])
    for k_key, diag in summary["candidate_pool_diagnostics"].items():
        lines.append(
            f"| {k_key} | {diag['oracle_bestofk_equiv_rate']:.4f} | {diag['solvable_bestofk_rate']:.4f} | "
            f"{diag['avg_parseable_candidates']:.2f} | {diag['avg_solvable_candidates']:.2f} |"
        )

    lines.extend(["", "## Acceptance", "", summary["acceptance"]["recommendation"], ""])
    return "\n".join(lines)


def _acceptance(summary: dict[str, Any]) -> dict[str, Any]:
    accepted: list[str] = []
    reasons: dict[str, list[str]] = {}
    for policy in POLICIES:
        if policy == "verifier_ranked":
            continue
        policy_reasons = []
        k8 = summary["mean_metrics"]["8"][policy]["metrics"]["equiv_rate"]
        base8 = summary["mean_metrics"]["8"]["verifier_ranked"]["metrics"]["equiv_rate"]
        k4 = summary["mean_metrics"]["4"][policy]["metrics"]["equiv_rate"]
        base4 = summary["mean_metrics"]["4"]["verifier_ranked"]["metrics"]["equiv_rate"]
        helped8 = summary["mean_metrics"]["8"][policy]["helped_vs_verifier_ranked"]
        hurt8 = summary["mean_metrics"]["8"][policy]["hurt_vs_verifier_ranked"]
        pool_wins = 0
        for pool_report in summary["per_pool"].values():
            if pool_report["8"][policy]["metrics"]["equiv_rate"] > pool_report["8"]["verifier_ranked"]["metrics"]["equiv_rate"]:
                pool_wins += 1
        if k8 <= base8:
            policy_reasons.append("K=8 mean did not improve")
        if k4 < base4 - 0.01:
            policy_reasons.append("K=4 regressed beyond tolerance")
        if helped8 <= hurt8:
            policy_reasons.append("K=8 helped rows did not exceed hurt rows")
        if pool_wins <= 1:
            policy_reasons.append("improvement did not appear in more than one pool")
        if not policy_reasons:
            accepted.append(policy)
        reasons[policy] = policy_reasons or ["passes cached acceptance gate"]
    recommendation = (
        f"Cached search ablation accepts: {', '.join(accepted)}. Run a fresh fixed-pool gate next."
        if accepted
        else "No search policy passed cached replay. Do not spend on fresh generation for these policies; move to a small repair-loop pilot."
    )
    return {"accepted_policies": accepted, "reasons": reasons, "recommendation": recommendation}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_file_logging(output_dir)
    started_at = time.time()
    pools = [Path(path) for path in (args.pool or DEFAULT_POOLS)]
    k_values = [int(k) for k in args.k_values]
    scorer = VerifierScorer(selection_path=args.selection)

    _write_json(
        output_dir / "process_info.json",
        {
            "pid": os.getpid(),
            "started_at": started_at,
            "selection": args.selection,
            "pools": [str(pool) for pool in pools],
            "k_values": k_values,
            "output_dir": str(output_dir),
        },
    )

    all_rows: list[dict[str, Any]] = []
    candidate_diagnostics: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    pool_names: list[str] = []

    for pool_idx, candidate_dump in enumerate(pools):
        pool = _pool_name(candidate_dump)
        pool_names.append(pool)
        logger.info("Analyzing pool %s (%s/%s): %s", pool, pool_idx + 1, len(pools), candidate_dump)
        _write_progress(
            output_dir,
            stage="analyzing_pool",
            completed_pools=pool_idx,
            total_pools=len(pools),
            current_pool=str(candidate_dump),
            started_at=started_at,
        )
        nl_lookup = _load_nl_lookup(candidate_dump)
        meta_by_row, candidates_by_row = _load_candidate_dump(candidate_dump)
        scored = _score_or_cached(
            scorer=scorer,
            nl_lookup=nl_lookup,
            meta_by_row=meta_by_row,
            candidates_by_row=candidates_by_row,
            batch_size=args.scoring_batch_size,
        )
        diagnostics = _add_solvability(scored)
        for diagnostic in diagnostics:
            diagnostic["pool"] = pool
            candidate_diagnostics.append(diagnostic)

        for row_index, candidates in scored.items():
            meta = meta_by_row[row_index]
            candidate_by_index = {candidate.index: candidate for candidate in candidates}
            for k in k_values:
                subset = candidates[:k]
                baseline = _select("verifier_ranked", subset, args.tie_margin)
                baseline_candidate = candidate_by_index.get(baseline.selected_index)
                for policy in POLICIES:
                    selection = _select(policy, subset, args.tie_margin)
                    selected = candidate_by_index.get(selection.selected_index)
                    row = {
                        "pool": pool,
                        "row_index": row_index,
                        "planetarium_name": meta["planetarium_name"],
                        "domain": meta["domain"],
                        "style": meta["style"],
                        "K": k,
                        "policy": policy,
                        "selected_index": selection.selected_index,
                        "selection_reason": selection.reason,
                        "selected_parseable": bool(selected.parseable) if selected else False,
                        "selected_solvable": bool(selected.solvable) if selected else False,
                        "selected_equivalent": bool(selected.equivalent) if selected else False,
                        "selected_score": selected.verifier_score if selected else None,
                        "parseable_count": sum(1 for candidate in subset if candidate.parseable),
                        "solvable_count": sum(1 for candidate in subset if candidate.parseable and candidate.solvable),
                        "equivalent_count": sum(1 for candidate in subset if candidate.equivalent),
                        "oracle_bestofk_equiv": any(candidate.equivalent for candidate in subset),
                        "solvable_bestofk": any(candidate.parseable and candidate.solvable for candidate in subset),
                    }
                    all_rows.append(row)
                    if policy != "verifier_ranked":
                        helped = row["selected_equivalent"] and not bool(baseline_candidate and baseline_candidate.equivalent)
                        hurt = bool(baseline_candidate and baseline_candidate.equivalent) and not row["selected_equivalent"]
                        if helped or hurt:
                            changed_rows.append(
                                {
                                    **row,
                                    "change": "helped" if helped else "hurt",
                                    "baseline_selected_index": baseline.selected_index,
                                    "baseline_selected_equivalent": bool(baseline_candidate and baseline_candidate.equivalent),
                                    "baseline_selected_solvable": bool(baseline_candidate and baseline_candidate.solvable),
                                    "baseline_selected_score": baseline_candidate.verifier_score if baseline_candidate else None,
                                }
                            )
        _flush_logs()

    summary: dict[str, Any] = {
        "verifier_selection": args.selection,
        "pools": [str(pool) for pool in pools],
        "k_values": k_values,
        "policies": POLICIES,
        "tie_margin": args.tie_margin,
        "mean_metrics": {},
        "per_pool": {},
        "candidate_pool_diagnostics": {},
    }

    for pool in pool_names:
        summary["per_pool"][pool] = {}
        for k in k_values:
            summary["per_pool"][pool][str(k)] = {}
            baseline_rows = [row for row in all_rows if row["pool"] == pool and row["K"] == k and row["policy"] == "verifier_ranked"]
            for policy in POLICIES:
                rows = [row for row in all_rows if row["pool"] == pool and row["K"] == k and row["policy"] == policy]
                summary["per_pool"][pool][str(k)][policy] = _policy_report(rows, baseline_rows)

    for k in k_values:
        summary["mean_metrics"][str(k)] = {}
        baseline_rows = [row for row in all_rows if row["K"] == k and row["policy"] == "verifier_ranked"]
        for policy in POLICIES:
            rows = [row for row in all_rows if row["K"] == k and row["policy"] == policy]
            summary["mean_metrics"][str(k)][policy] = _policy_report(rows, baseline_rows)
        diagnostic_rows = [row for row in all_rows if row["K"] == k and row["policy"] == "verifier_ranked"]
        summary["candidate_pool_diagnostics"][str(k)] = {
            "oracle_bestofk_equiv_rate": mean([1.0 if row["oracle_bestofk_equiv"] else 0.0 for row in diagnostic_rows]),
            "solvable_bestofk_rate": mean([1.0 if row["solvable_bestofk"] else 0.0 for row in diagnostic_rows]),
            "avg_parseable_candidates": mean([float(row["parseable_count"]) for row in diagnostic_rows]),
            "avg_solvable_candidates": mean([float(row["solvable_count"]) for row in diagnostic_rows]),
            "avg_equivalent_candidates": mean([float(row["equivalent_count"]) for row in diagnostic_rows]),
        }

    summary["acceptance"] = _acceptance(summary)

    with open(output_dir / "candidate_diagnostics.jsonl", "w", encoding="utf-8") as f:
        for row in candidate_diagnostics:
            f.write(json.dumps(row) + "\n")
    with open(output_dir / "changed_rows.jsonl", "w", encoding="utf-8") as f:
        for row in changed_rows:
            f.write(json.dumps(row) + "\n")
    _write_json(output_dir / "search_ablation_summary.json", summary)
    with open(output_dir / "search_ablation_summary.md", "w", encoding="utf-8") as f:
        f.write(_markdown(summary))
    _write_progress(
        output_dir,
        stage="complete",
        completed_pools=len(pools),
        total_pools=len(pools),
        current_pool="",
        started_at=started_at,
    )
    logger.info("Saved search ablation summary to %s", output_dir / "search_ablation_summary.md")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default="results/verifier/best_current/selection.yaml")
    parser.add_argument("--output_dir", default="results/vcsr/search_ablation_round4")
    parser.add_argument("--pool", action="append", help="Candidate dump path; may be repeated")
    parser.add_argument("--k_values", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--tie_margin", type=float, default=0.02)
    parser.add_argument("--scoring_batch_size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
