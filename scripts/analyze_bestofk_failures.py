"""
Analyze verifier-ranked best-of-K failures on a cached candidate pool.

This script is intended to answer one practical question:
should the next modeling step be a focused round-4 data/mining pass, or does
the residual error pattern justify moving to a more explicit ranking objective?
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vcsr_env  # noqa: F401
import yaml

from eval.equivalence import BatchMetrics, EvalResult
from search.ranking import CandidateRecord, greedy_first, random_parseable, verifier_ranked

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_COMPARISON_DUMPS = [
    "results/vcsr/bestofk_pilot/candidate_dump.jsonl",
    "results/vcsr/bestofk_ranking_round2_pool/candidate_dump.jsonl",
]
DEFAULT_K_VALUES = [4, 8]
DEFAULT_POLICIES = ["greedy_first", "random_parseable", "verifier_ranked"]
NEAR_TIE_MARGIN = 0.03
MODERATE_GAP_MARGIN = 0.10


def _metrics_to_dict(metrics: BatchMetrics) -> dict[str, Any]:
    return {
        "total": metrics.total,
        "parse_count": metrics.parse_count,
        "solve_count": metrics.solve_count,
        "equiv_count": metrics.equiv_count,
        "error_count": metrics.error_count,
        "parse_rate": metrics.parse_rate,
        "solve_rate": metrics.solve_rate,
        "equiv_rate": metrics.equiv_rate,
        "equiv_given_parse": metrics.equiv_given_parse,
    }


def _compute_batch_metrics(results: list[EvalResult]) -> BatchMetrics:
    metrics = BatchMetrics(total=len(results))
    for res in results:
        if res.parseable:
            metrics.parse_count += 1
        if res.solveable:
            metrics.solve_count += 1
        if res.equivalent:
            metrics.equiv_count += 1
        if res.error:
            metrics.error_count += 1
    return metrics


def _style_label(record: dict[str, Any]) -> str:
    init_style = "abstract" if int(record.get("init_is_abstract", 0)) else "explicit"
    goal_style = "abstract" if int(record.get("goal_is_abstract", 0)) else "explicit"
    return f"{init_style}/{goal_style}"


def _load_run_config(candidate_dump_path: Path) -> dict[str, Any]:
    run_config_path = candidate_dump_path.parent / "run_config.yaml"
    if not run_config_path.exists():
        return {}
    with open(run_config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_aggregate_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_candidate_pool(path: Path) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], list[int]]:
    row_meta: dict[int, dict[str, Any]] = {}
    candidates_by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)

    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if "candidate_index" not in record:
                continue
            row_index = int(record["row_index"])
            row_meta[row_index] = {
                "planetarium_name": record["planetarium_name"],
                "domain": record["domain"],
                "init_is_abstract": int(record.get("init_is_abstract", 0)),
                "goal_is_abstract": int(record.get("goal_is_abstract", 0)),
                "style": _style_label(record),
            }
            candidates_by_row[row_index].append(record)

    ordered_row_indices = sorted(candidates_by_row)
    for row_index in ordered_row_indices:
        candidates_by_row[row_index].sort(key=lambda rec: int(rec["candidate_index"]))

    return row_meta, candidates_by_row, ordered_row_indices


def _pick_top_scored(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return max(
        records,
        key=lambda rec: (
            float("-inf") if rec.get("verifier_score") is None else float(rec["verifier_score"]),
            -int(rec["candidate_index"]),
        ),
    )


def _selection_to_result(
    selection_index: int | None,
    reason: str,
    by_index: dict[int, dict[str, Any]],
) -> tuple[EvalResult, dict[str, Any]]:
    if selection_index is None:
        return EvalResult(parseable=False, equivalent=False, error=reason), {
            "selected_index": None,
            "selection_reason": reason,
            "parseable": False,
            "equivalent": False,
            "verifier_score": None,
            "error": reason,
        }

    chosen = by_index[int(selection_index)]
    chosen_parseable = bool(chosen.get("parseable"))
    chosen_error = chosen.get("error")
    if not chosen_parseable and not chosen_error:
        # The cached candidate dump keeps `parseable: false` but not always the
        # derived `parse_failed` label that the online evaluator used in its
        # aggregate metrics. Reconstruct it here so validation matches exactly.
        chosen_error = "parse_failed"
    result = EvalResult(
        parseable=chosen_parseable,
        equivalent=bool(chosen.get("equivalent")),
        error=chosen_error,
    )
    details = {
        "selected_index": int(selection_index),
        "selection_reason": reason,
        "parseable": chosen_parseable,
        "equivalent": bool(chosen.get("equivalent")),
        "verifier_score": chosen.get("verifier_score"),
        "error": chosen_error,
    }
    return result, details


def _classify_verifier_miss(
    *,
    oracle_available: bool,
    verifier_equivalent: bool,
    score_margin: float | None,
) -> str:
    if not oracle_available:
        return "no_equivalent_in_pool"
    if verifier_equivalent:
        return "success"
    if score_margin is not None and score_margin <= NEAR_TIE_MARGIN:
        return "near_tie_misranking"
    return "equivalent_in_pool_but_misranked"


def _gain_reason(
    *,
    greedy_parseable: bool,
    greedy_equivalent: bool,
    verifier_equivalent: bool,
    gain_requires_extra_k: bool,
) -> str:
    if not verifier_equivalent:
        return "none"
    if greedy_equivalent:
        return "already_solved_by_greedy"
    if not greedy_parseable:
        return "dominated_by_unparseable_first_greedy_behavior"
    if gain_requires_extra_k:
        return "requires_extra_k_candidates"
    return "within_pool_reranking"


def _float_close(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def _summarize_breakdown(cases: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[str(case[key])].append(case)

    summary = []
    for group_key in sorted(groups):
        group = groups[group_key]
        summary.append(
            {
                key: group_key,
                "rows": len(group),
                "oracle_positive_rows": sum(1 for row in group if row["oracle_available"]),
                "verifier_hits": sum(1 for row in group if row["verifier_ranked"]["equivalent"]),
                "verifier_oracle_misses": sum(
                    1 for row in group if row["oracle_available"] and not row["verifier_ranked"]["equivalent"]
                ),
            }
        )
    return summary


def _analyze_pool(
    candidate_dump_path: Path,
    *,
    k_values: list[int],
    policies: list[str],
    comparison_role: str,
) -> dict[str, Any]:
    run_cfg = _load_run_config(candidate_dump_path)
    seed = int(run_cfg.get("experiment", {}).get("seed", 42))
    row_meta, candidates_by_row, ordered_row_indices = _load_candidate_pool(candidate_dump_path)

    selected_results: dict[tuple[int, str], list[EvalResult]] = {
        (k, policy): [] for k in k_values for policy in policies
    }
    pool_parseable_counts: dict[int, list[int]] = {k: [] for k in k_values}
    pool_equiv_counts: dict[int, list[int]] = {k: [] for k in k_values}
    pool_oracle_best: dict[int, list[int]] = {k: [] for k in k_values}

    cases_by_k: dict[int, list[dict[str, Any]]] = {k: [] for k in k_values}
    unique_rows: dict[int, dict[str, Any]] = {}

    for row_index in ordered_row_indices:
        meta = row_meta[row_index]
        candidates = candidates_by_row[row_index]
        unique_rows[row_index] = meta

        oracle4_available = any(bool(rec.get("equivalent")) for rec in candidates[:4])

        for k in k_values:
            subset = candidates[:k]
            by_index = {int(rec["candidate_index"]): rec for rec in subset}

            candidate_records = [
                CandidateRecord(
                    index=int(rec["candidate_index"]),
                    parseable=bool(rec.get("parseable")),
                    equivalent=bool(rec.get("equivalent")),
                    verifier_score=rec.get("verifier_score"),
                )
                for rec in subset
            ]

            rng = random.Random(seed + row_index * 1000 + k)
            selections = {
                "greedy_first": greedy_first(candidate_records),
                "random_parseable": random_parseable(candidate_records, rng),
                "verifier_ranked": verifier_ranked(candidate_records),
            }

            policy_details: dict[str, dict[str, Any]] = {}
            for policy_name in policies:
                selection = selections[policy_name]
                result, details = _selection_to_result(selection.selected_index, selection.reason, by_index)
                selected_results[(k, policy_name)].append(result)
                policy_details[policy_name] = details

            pool_parseable = sum(1 for rec in subset if bool(rec.get("parseable")))
            pool_equiv = sum(1 for rec in subset if bool(rec.get("equivalent")))
            oracle_available = pool_equiv > 0
            pool_parseable_counts[k].append(pool_parseable)
            pool_equiv_counts[k].append(pool_equiv)
            pool_oracle_best[k].append(1 if oracle_available else 0)

            parseable_equiv = [
                rec for rec in subset if bool(rec.get("parseable")) and bool(rec.get("equivalent"))
            ]
            parseable_wrong = [
                rec for rec in subset if bool(rec.get("parseable")) and not bool(rec.get("equivalent"))
            ]
            best_equiv = _pick_top_scored(parseable_equiv)
            best_wrong = _pick_top_scored(parseable_wrong)
            verifier_selected_index = policy_details["verifier_ranked"]["selected_index"]
            verifier_selected = by_index.get(verifier_selected_index) if verifier_selected_index is not None else None

            selected_wrong_score = None
            best_equiv_score = None
            score_margin = None
            if verifier_selected is not None and verifier_selected.get("verifier_score") is not None:
                selected_wrong_score = float(verifier_selected["verifier_score"])
            if best_equiv is not None and best_equiv.get("verifier_score") is not None:
                best_equiv_score = float(best_equiv["verifier_score"])
            if (
                oracle_available
                and verifier_selected is not None
                and not bool(verifier_selected.get("equivalent"))
                and selected_wrong_score is not None
                and best_equiv_score is not None
            ):
                score_margin = selected_wrong_score - best_equiv_score

            gain_requires_extra_k = bool(k == 8 and oracle_available and not oracle4_available)

            case_record = {
                "comparison_role": comparison_role,
                "candidate_dump": str(candidate_dump_path),
                "row_index": row_index,
                "planetarium_name": meta["planetarium_name"],
                "domain": meta["domain"],
                "init_is_abstract": meta["init_is_abstract"],
                "goal_is_abstract": meta["goal_is_abstract"],
                "style": meta["style"],
                "K": k,
                "oracle_available": oracle_available,
                "oracle_available_at_k4": oracle4_available,
                "gain_requires_extra_k_candidates": gain_requires_extra_k,
                "pool_parseable_count": pool_parseable,
                "pool_equivalent_count": pool_equiv,
                "pool_equivalent_candidate_count": len(parseable_equiv),
                "best_equivalent_candidate_index": None if best_equiv is None else int(best_equiv["candidate_index"]),
                "best_equivalent_verifier_score": None if best_equiv_score is None else best_equiv_score,
                "best_wrong_candidate_index": None if best_wrong is None else int(best_wrong["candidate_index"]),
                "best_wrong_verifier_score": None
                if best_wrong is None or best_wrong.get("verifier_score") is None
                else float(best_wrong["verifier_score"]),
                "selected_wrong_minus_best_equivalent_score_margin": score_margin,
                "greedy_first": policy_details["greedy_first"],
                "random_parseable": policy_details["random_parseable"],
                "verifier_ranked": policy_details["verifier_ranked"],
            }

            case_record["verifier_miss_type"] = _classify_verifier_miss(
                oracle_available=oracle_available,
                verifier_equivalent=bool(policy_details["verifier_ranked"]["equivalent"]),
                score_margin=score_margin,
            )
            case_record["verifier_gain_over_greedy"] = bool(
                policy_details["verifier_ranked"]["equivalent"] and not policy_details["greedy_first"]["equivalent"]
            )
            case_record["verifier_gain_reason"] = _gain_reason(
                greedy_parseable=bool(policy_details["greedy_first"]["parseable"]),
                greedy_equivalent=bool(policy_details["greedy_first"]["equivalent"]),
                verifier_equivalent=bool(policy_details["verifier_ranked"]["equivalent"]),
                gain_requires_extra_k=gain_requires_extra_k,
            )

            cases_by_k[k].append(case_record)

    metrics_by_k: dict[str, Any] = {}
    for k in k_values:
        metrics_by_k[str(k)] = {"policies": {}}
        for policy_name in policies:
            metrics = _compute_batch_metrics(selected_results[(k, policy_name)])
            metrics_by_k[str(k)]["policies"][policy_name] = {
                "metrics": _metrics_to_dict(metrics),
                "candidate_pool": {
                    "avg_parseable_candidates": sum(pool_parseable_counts[k]) / max(1, len(pool_parseable_counts[k])),
                    "avg_equivalent_candidates": sum(pool_equiv_counts[k]) / max(1, len(pool_equiv_counts[k])),
                    "oracle_bestofk_equiv_rate": sum(pool_oracle_best[k]) / max(1, len(pool_oracle_best[k])),
                },
            }

    summary_by_k = {}
    for k in k_values:
        cases = cases_by_k[k]
        summary_by_k[str(k)] = {
            "rows": len(cases),
            "oracle_positive_rows": sum(1 for case in cases if case["oracle_available"]),
            "verifier_hits": sum(1 for case in cases if case["verifier_ranked"]["equivalent"]),
            "verifier_oracle_misses": sum(
                1 for case in cases if case["oracle_available"] and not case["verifier_ranked"]["equivalent"]
            ),
            "verifier_miss_types": dict(Counter(case["verifier_miss_type"] for case in cases)),
            "verifier_gain_reasons": dict(
                Counter(case["verifier_gain_reason"] for case in cases if case["verifier_gain_reason"] != "none")
            ),
            "by_domain": _summarize_breakdown(cases, "domain"),
            "by_style": _summarize_breakdown(cases, "style"),
        }

    return {
        "candidate_dump": str(candidate_dump_path),
        "run_seed": seed,
        "k_values": k_values,
        "policies": policies,
        "metrics": metrics_by_k,
        "cases_by_k": cases_by_k,
        "summary_by_k": summary_by_k,
        "row_count": len(unique_rows),
    }


def _validate_against_saved_aggregate(
    analyzed: dict[str, Any],
    aggregate_metrics: dict[str, Any] | None,
    *,
    k_values: list[int],
    policies: list[str],
) -> dict[str, Any]:
    if aggregate_metrics is None:
        return {
            "aggregate_metrics_path": None,
            "validated": False,
            "reason": "aggregate_metrics_missing",
            "mismatches": [],
        }

    mismatches: list[str] = []
    for k in k_values:
        k_key = str(k)
        saved_block = aggregate_metrics["comparisons"][k_key]["policies"]
        analyzed_block = analyzed["metrics"][k_key]["policies"]
        for policy_name in policies:
            saved_metrics = saved_block[policy_name]["metrics"]
            analyzed_metrics = analyzed_block[policy_name]["metrics"]
            for metric_key in ["total", "parse_count", "equiv_count", "error_count"]:
                if int(saved_metrics[metric_key]) != int(analyzed_metrics[metric_key]):
                    mismatches.append(f"{k_key}/{policy_name}/{metric_key}")
            for metric_key in ["parse_rate", "equiv_rate", "equiv_given_parse"]:
                if not _float_close(saved_metrics[metric_key], analyzed_metrics[metric_key]):
                    mismatches.append(f"{k_key}/{policy_name}/{metric_key}")

            saved_pool = saved_block[policy_name]["candidate_pool"]
            analyzed_pool = analyzed_block[policy_name]["candidate_pool"]
            for pool_key in ["avg_parseable_candidates", "avg_equivalent_candidates", "oracle_bestofk_equiv_rate"]:
                if not _float_close(saved_pool[pool_key], analyzed_pool[pool_key]):
                    mismatches.append(f"{k_key}/{policy_name}/{pool_key}")

    return {
        "aggregate_metrics_path": aggregate_metrics.get("source_path") if isinstance(aggregate_metrics, dict) else None,
        "validated": not mismatches,
        "reason": "ok" if not mismatches else "metric_mismatch",
        "mismatches": mismatches,
    }


def _comparison_recurrence_summary(comparison_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for result in comparison_results:
        k8 = result["summary_by_k"].get("8", {})
        domain_rows = {row["domain"]: row for row in k8.get("by_domain", [])}
        style_rows = {row["style"]: row for row in k8.get("by_style", [])}
        summary.append(
            {
                "candidate_dump": result["candidate_dump"],
                "k8_verifier_oracle_misses": k8.get("verifier_oracle_misses", 0),
                "k8_blocksworld_misses": domain_rows.get("blocksworld", {}).get("verifier_oracle_misses", 0),
                "k8_gripper_misses": domain_rows.get("gripper", {}).get("verifier_oracle_misses", 0),
                "k8_abstract_abstract_misses": style_rows.get("abstract/abstract", {}).get("verifier_oracle_misses", 0),
                "k8_explicit_explicit_misses": style_rows.get("explicit/explicit", {}).get("verifier_oracle_misses", 0),
            }
        )
    return summary


def _decide_next_path(heldout: dict[str, Any], comparison_results: list[dict[str, Any]]) -> dict[str, Any]:
    miss_cases: list[dict[str, Any]] = []
    for k_key in ["4", "8"]:
        miss_cases.extend(
            case
            for case in heldout["cases_by_k"][int(k_key)]
            if case["oracle_available"] and not case["verifier_ranked"]["equivalent"]
        )

    if not miss_cases:
        recommendation = "focused_round4"
        rationale = "No oracle-positive verifier misses remain on the held-out run."
        evidence = {
            "miss_count": 0,
            "blocksworld_ratio": 1.0,
            "abstract_abstract_ratio": 1.0,
            "moderate_gap_ratio": 1.0,
            "comparison_recurrence_support": True,
        }
    else:
        blocksworld_ratio = sum(1 for case in miss_cases if case["domain"] == "blocksworld") / len(miss_cases)
        abstract_ratio = sum(1 for case in miss_cases if case["style"] == "abstract/abstract") / len(miss_cases)
        moderate_gap_ratio = sum(
            1
            for case in miss_cases
            if case["selected_wrong_minus_best_equivalent_score_margin"] is not None
            and case["selected_wrong_minus_best_equivalent_score_margin"] <= MODERATE_GAP_MARGIN
        ) / len(miss_cases)
        comparison_summary = _comparison_recurrence_summary(comparison_results)
        comparison_recurrence_support = any(
            item["k8_blocksworld_misses"] >= item["k8_gripper_misses"]
            and item["k8_abstract_abstract_misses"] >= item["k8_explicit_explicit_misses"]
            for item in comparison_summary
        )

        if (
            blocksworld_ratio >= 0.70
            and abstract_ratio >= 0.50
            and moderate_gap_ratio >= 0.60
            and comparison_recurrence_support
        ):
            recommendation = "focused_round4"
            rationale = (
                "Most residual oracle-positive misses are still within-pool misrankings concentrated in "
                "blocksworld, especially abstract/abstract rows, with score gaps small enough to justify a "
                "targeted next mining round rather than an immediate objective change."
            )
        else:
            recommendation = "ranking_objective_change"
            rationale = (
                "Residual oracle-positive misses are too broad or too large-gap to justify another pointwise "
                "data-only round as the default next step."
            )

        evidence = {
            "miss_count": len(miss_cases),
            "blocksworld_ratio": blocksworld_ratio,
            "abstract_abstract_ratio": abstract_ratio,
            "moderate_gap_ratio": moderate_gap_ratio,
            "comparison_recurrence_support": comparison_recurrence_support,
        }

    if recommendation == "focused_round4":
        next_experiment = [
            "Keep the DeBERTa cross-encoder backbone and current inference stack.",
            "Do not run a broad LR or batch sweep.",
            "Mine additional examples only from held-out-like blocksworld rows, especially abstract/abstract rows.",
            "Prioritize rows where a verifier-selected wrong candidate outranks an equivalent candidate.",
            "Prioritize rows with multiple equivalent candidates plus one or more high-scoring wrong candidates.",
            "Emphasize near-tie ranking negatives and keep replay as the checkpoint-selection rule.",
        ]
    else:
        next_experiment = [
            "Keep the DeBERTa cross-encoder backbone.",
            "Change the training objective before changing architecture.",
            "Move from pointwise classification toward pairwise or listwise ranking supervision using the existing pool-mined examples.",
            "Keep replay as the checkpoint-selection rule and do not accept the new checkpoint on offline AUC alone.",
        ]

    return {
        "recommended_path": recommendation,
        "rationale": rationale,
        "evidence": evidence,
        "next_experiment": next_experiment,
        "comparison_recurrence": _comparison_recurrence_summary(comparison_results),
    }


def _sort_high_value_misses(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        cases,
        key=lambda case: (
            0 if case["K"] == 8 else 1,
            0 if case["style"] == "abstract/abstract" else 1,
            -(case["pool_equivalent_candidate_count"]),
            999.0
            if case["selected_wrong_minus_best_equivalent_score_margin"] is None
            else case["selected_wrong_minus_best_equivalent_score_margin"],
            case["row_index"],
        ),
    )


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_None_\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _render_failure_summary_md(summary: dict[str, Any]) -> str:
    heldout = summary["heldout"]
    decision = summary["decision"]
    validation = summary["validation"]

    lines = [
        "# Held-Out Failure Analysis",
        "",
        f"Source pool: `{heldout['candidate_dump']}`",
        "",
        "## Validation",
        "",
        f"- Aggregate metric recomputation validated: `{validation['validated']}`",
    ]
    if validation["mismatches"]:
        lines.append(f"- Mismatches: `{', '.join(validation['mismatches'])}`")

    lines.extend(
        [
            "",
            "## Top-Line Counts",
            "",
            f"- Total rows: `{heldout['row_count']}`",
            f"- Oracle-positive rows at `K=4`: `{heldout['summary_by_k']['4']['oracle_positive_rows']}`",
            f"- Oracle-positive rows at `K=8`: `{heldout['summary_by_k']['8']['oracle_positive_rows']}`",
            f"- Oracle-positive verifier misses at `K=4`: `{heldout['summary_by_k']['4']['verifier_oracle_misses']}`",
            f"- Oracle-positive verifier misses at `K=8`: `{heldout['summary_by_k']['8']['verifier_oracle_misses']}`",
            "",
            "## Domain Breakdown (`K=8` verifier)",
            "",
            _markdown_table(
                ["Domain", "Rows", "Oracle+", "Verifier Hits", "Verifier Oracle Misses"],
                [
                    [
                        row["domain"],
                        str(row["rows"]),
                        str(row["oracle_positive_rows"]),
                        str(row["verifier_hits"]),
                        str(row["verifier_oracle_misses"]),
                    ]
                    for row in heldout["summary_by_k"]["8"]["by_domain"]
                ],
            ),
            "",
            "## Style Breakdown (`K=8` verifier)",
            "",
            _markdown_table(
                ["Style", "Rows", "Oracle+", "Verifier Hits", "Verifier Oracle Misses"],
                [
                    [
                        row["style"],
                        str(row["rows"]),
                        str(row["oracle_positive_rows"]),
                        str(row["verifier_hits"]),
                        str(row["verifier_oracle_misses"]),
                    ]
                    for row in heldout["summary_by_k"]["8"]["by_style"]
                ],
            ),
        ]
    )

    miss_rows = [
        [
            case["planetarium_name"],
            str(case["K"]),
            case["domain"],
            case["style"],
            case["verifier_miss_type"],
            str(case["pool_equivalent_candidate_count"]),
            "-" if case["verifier_ranked"]["selected_index"] is None else str(case["verifier_ranked"]["selected_index"]),
            "-"
            if case["best_equivalent_candidate_index"] is None
            else str(case["best_equivalent_candidate_index"]),
            "-"
            if case["selected_wrong_minus_best_equivalent_score_margin"] is None
            else f"{case['selected_wrong_minus_best_equivalent_score_margin']:.4f}",
        ]
        for case in summary["high_value_miss_cases"]
    ]
    lines.extend(
        [
            "",
            "## Highest-Value Miss Rows",
            "",
            _markdown_table(
                ["Row", "K", "Domain", "Style", "Miss Type", "Equiv Cands", "Selected", "Best Equiv", "Gap"],
                miss_rows,
            ),
        ]
    )

    gain_rows = [
        [
            case["planetarium_name"],
            case["domain"],
            case["style"],
            case["verifier_gain_reason"],
            str(case["verifier_ranked"]["selected_index"]),
            str(case["best_equivalent_candidate_index"]),
            "yes" if case["gain_requires_extra_k_candidates"] else "no",
        ]
        for case in summary["k8_gain_cases"]
    ]
    lines.extend(
        [
            "",
            "## `K=8` Verifier Wins Over Greedy",
            "",
            _markdown_table(
                ["Row", "Domain", "Style", "Gain Reason", "Selected", "Best Equiv", "Extra K Only"],
                gain_rows,
            ),
            "",
            "## Recommendation",
            "",
            f"- Recommended path: `{decision['recommended_path']}`",
            f"- Rationale: {decision['rationale']}",
            f"- Evidence: blocksworld ratio `{decision['evidence']['blocksworld_ratio']:.2f}`, "
            f"abstract/abstract ratio `{decision['evidence']['abstract_abstract_ratio']:.2f}`, "
            f"moderate-gap ratio `{decision['evidence']['moderate_gap_ratio']:.2f}`",
        ]
    )

    if decision["comparison_recurrence"]:
        lines.extend(
            [
                "",
                "## Comparison-Pool Recurrence (`K=8` miss slices)",
                "",
                _markdown_table(
                    ["Pool", "Oracle Misses", "Blocksworld", "Gripper", "Abstract/Abstract", "Explicit/Explicit"],
                    [
                        [
                            item["candidate_dump"],
                            str(item["k8_verifier_oracle_misses"]),
                            str(item["k8_blocksworld_misses"]),
                            str(item["k8_gripper_misses"]),
                            str(item["k8_abstract_abstract_misses"]),
                            str(item["k8_explicit_explicit_misses"]),
                        ]
                        for item in decision["comparison_recurrence"]
                    ],
                ),
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def _render_decision_md(summary: dict[str, Any]) -> str:
    decision = summary["decision"]
    lines = [
        "# Round-4 Decision Recommendation",
        "",
        f"- Recommended path: `{decision['recommended_path']}`",
        f"- Rationale: {decision['rationale']}",
        "",
        "## Evidence",
        "",
        f"- Oracle-positive verifier misses considered: `{decision['evidence']['miss_count']}`",
        f"- Blocksworld miss ratio: `{decision['evidence']['blocksworld_ratio']:.2f}`",
        f"- Abstract/abstract miss ratio: `{decision['evidence']['abstract_abstract_ratio']:.2f}`",
        f"- Moderate-gap miss ratio: `{decision['evidence']['moderate_gap_ratio']:.2f}`",
        f"- Comparison recurrence support: `{decision['evidence']['comparison_recurrence_support']}`",
        "",
        "## Next Experiment Definition",
        "",
    ]
    for item in decision["next_experiment"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            f"Recommended next modeling path: {decision['recommended_path']}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze best-of-K verifier failures on a cached candidate pool")
    parser.add_argument(
        "--candidate_dump",
        type=str,
        default="results/vcsr/bestofk_round3_holdout_eval/candidate_dump.jsonl",
    )
    parser.add_argument(
        "--aggregate_metrics",
        type=str,
        default="results/vcsr/bestofk_round3_holdout_eval/aggregate_metrics.json",
    )
    parser.add_argument("--comparison_dump", action="append", default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--k_values", type=int, nargs="*", default=DEFAULT_K_VALUES)
    args = parser.parse_args()

    candidate_dump_path = Path(args.candidate_dump)
    aggregate_metrics_path = Path(args.aggregate_metrics) if args.aggregate_metrics else None
    output_dir = Path(args.output_dir) if args.output_dir else candidate_dump_path.parent / "failure_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_dumps = args.comparison_dump[:] if args.comparison_dump else []
    if not comparison_dumps:
        for default_path in DEFAULT_COMPARISON_DUMPS:
            if default_path != str(candidate_dump_path) and Path(default_path).exists():
                comparison_dumps.append(default_path)

    k_values = sorted(set(int(k) for k in args.k_values))
    policies = DEFAULT_POLICIES[:]

    logger.info("Analyzing held-out pool: %s", candidate_dump_path)
    heldout = _analyze_pool(candidate_dump_path, k_values=k_values, policies=policies, comparison_role="heldout")

    aggregate_metrics = _load_aggregate_metrics(aggregate_metrics_path)
    if aggregate_metrics is not None:
        aggregate_metrics["source_path"] = str(aggregate_metrics_path)
    validation = _validate_against_saved_aggregate(heldout, aggregate_metrics, k_values=k_values, policies=policies)

    comparison_results = []
    for dump in comparison_dumps:
        dump_path = Path(dump)
        if not dump_path.exists():
            continue
        logger.info("Analyzing comparison pool: %s", dump_path)
        comparison_results.append(
            _analyze_pool(dump_path, k_values=k_values, policies=policies, comparison_role="comparison")
        )

    miss_cases = []
    for k in k_values:
        miss_cases.extend(
            case
            for case in heldout["cases_by_k"][k]
            if case["oracle_available"] and not case["verifier_ranked"]["equivalent"]
        )
    high_value_miss_cases = _sort_high_value_misses(miss_cases)[:8]
    k8_gain_cases = [
        case
        for case in heldout["cases_by_k"][8]
        if case["verifier_ranked"]["equivalent"] and not case["greedy_first"]["equivalent"]
    ][:8]

    decision = _decide_next_path(heldout, comparison_results)

    summary = {
        "inputs": {
            "candidate_dump": str(candidate_dump_path),
            "aggregate_metrics": str(aggregate_metrics_path) if aggregate_metrics_path else None,
            "comparison_dumps": comparison_dumps,
            "k_values": k_values,
            "policies": policies,
            "near_tie_margin": NEAR_TIE_MARGIN,
            "moderate_gap_margin": MODERATE_GAP_MARGIN,
        },
        "validation": validation,
        "heldout": {
            "candidate_dump": heldout["candidate_dump"],
            "row_count": heldout["row_count"],
            "summary_by_k": heldout["summary_by_k"],
            "metrics": heldout["metrics"],
        },
        "comparison_pools": [
            {
                "candidate_dump": result["candidate_dump"],
                "row_count": result["row_count"],
                "summary_by_k": result["summary_by_k"],
            }
            for result in comparison_results
        ],
        "high_value_miss_cases": high_value_miss_cases,
        "k8_gain_cases": k8_gain_cases,
        "decision": decision,
    }

    failure_cases_path = output_dir / "failure_cases.jsonl"
    with open(failure_cases_path, "w", encoding="utf-8") as f:
        for k in k_values:
            for case in heldout["cases_by_k"][k]:
                f.write(json.dumps(case) + "\n")

    with open(output_dir / "failure_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "failure_summary.md", "w", encoding="utf-8") as f:
        f.write(_render_failure_summary_md(summary))
    with open(output_dir / "decision_recommendation.md", "w", encoding="utf-8") as f:
        f.write(_render_decision_md(summary))

    logger.info("Saved failure summary to %s", output_dir / "failure_summary.json")
    logger.info("Saved failure markdown to %s", output_dir / "failure_summary.md")
    logger.info("Saved failure cases to %s", failure_cases_path)
    logger.info("Saved decision recommendation to %s", output_dir / "decision_recommendation.md")


if __name__ == "__main__":
    main()
