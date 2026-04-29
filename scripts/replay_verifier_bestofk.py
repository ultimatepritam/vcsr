"""
Replay verifier-ranked best-of-K evaluation on a fixed cached candidate pool.

This script reuses a previously saved candidate dump from `run_verifier_bestofk.py`,
rescoring the exact same parseable candidates with one or more verifier
checkpoints. This isolates verifier quality from generation randomness.

Usage:
    python scripts/replay_verifier_bestofk.py ^
      --candidate_dump results/vcsr/bestofk_pilot/candidate_dump.jsonl ^
      --selection results/verifier/best_current/selection.yaml ^
      --selection results/verifier/capacity_push/lr_2p0em05/selection.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vcsr_env  # noqa: F401
import yaml

from data.planetarium_loader import PlanetariumDataset
from eval.equivalence import BatchMetrics, EvalResult, stratified_report
from search.ranking import CandidateRecord, greedy_first, random_parseable, verifier_ranked
from verifier.inference import VerifierScorer, load_selected_verifier_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _metrics_to_dict(metrics: BatchMetrics) -> dict:
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


def _summarize_policy(
    rows: list[dict],
    results: list[EvalResult],
    pool_parseable_counts: list[int],
    pool_equiv_counts: list[int],
    pool_oracle_best: list[int],
) -> dict:
    metrics = _compute_batch_metrics(results)
    strata = stratified_report(rows, results)
    return {
        "metrics": _metrics_to_dict(metrics),
        "stratified": {k: _metrics_to_dict(v) for k, v in strata.items()},
        "candidate_pool": {
            "avg_parseable_candidates": sum(pool_parseable_counts) / max(1, len(pool_parseable_counts)),
            "avg_equivalent_candidates": sum(pool_equiv_counts) / max(1, len(pool_equiv_counts)),
            "oracle_bestofk_equiv_rate": sum(pool_oracle_best) / max(1, len(pool_oracle_best)),
        },
    }


def _markdown(summary: dict) -> str:
    lines = [
        "# Fixed-Pool Replay Evaluation",
        "",
        f"Source pool: `{summary['source_pool']['candidate_dump']}`",
        "",
        "| Verifier | K | Policy | Parse | Equiv | Equiv / Parse | Avg Parseable | Avg Equivalent | Oracle Best-of-K |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for verifier_block in summary["verifiers"]:
        verifier_name = verifier_block["name"]
        for k_key, comp in verifier_block["comparisons"].items():
            for policy, details in comp["policies"].items():
                metrics = details["metrics"]
                pool = details["candidate_pool"]
                lines.append(
                    f"| {verifier_name} | {k_key} | {policy} | {metrics['parse_rate']:.4f} | {metrics['equiv_rate']:.4f} | "
                    f"{metrics['equiv_given_parse']:.4f} | {pool['avg_parseable_candidates']:.2f} | "
                    f"{pool['avg_equivalent_candidates']:.2f} | {pool['oracle_bestofk_equiv_rate']:.4f} |"
                )
    return "\n".join(lines) + "\n"


def _load_run_context(candidate_dump_path: Path) -> tuple[dict, dict[str, dict]]:
    run_config_path = candidate_dump_path.parent / "run_config.yaml"
    with open(run_config_path, encoding="utf-8") as f:
        run_cfg = yaml.safe_load(f)

    ds_cfg = run_cfg.get("dataset", {})
    seed = int(run_cfg.get("experiment", {}).get("seed", 42))
    dataset = PlanetariumDataset(
        split_strategy=ds_cfg.get("split_strategy", "template_hash"),
        seed=seed,
    )
    split_name = ds_cfg.get("split", "test")
    split = dataset.get_split(split_name)
    rows_by_name = {row["name"]: row for row in split}
    return run_cfg, rows_by_name


def _load_candidate_pool(path: Path) -> tuple[dict[int, dict], dict[int, list[dict]], list[int]]:
    row_meta: dict[int, dict] = {}
    candidates_by_row: dict[int, list[dict]] = defaultdict(list)

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
            }
            candidates_by_row[row_index].append(record)

    ordered_row_indices = sorted(candidates_by_row)
    for row_index in ordered_row_indices:
        candidates_by_row[row_index].sort(key=lambda rec: int(rec["candidate_index"]))

    return row_meta, candidates_by_row, ordered_row_indices


def _resolve_verifier_name(selection_path: str) -> str:
    metadata = load_selected_verifier_metadata(selection_path)
    selected_run = str(metadata.get("selected_run", ""))
    return Path(selected_run).name or Path(selection_path).stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay verifier-ranked best-of-K on a fixed candidate pool")
    parser.add_argument("--candidate_dump", type=str, required=True)
    parser.add_argument("--selection", action="append", default=[], help="Selection YAML path; may be repeated")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--k_values", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.selection:
        raise ValueError("Provide at least one --selection path")

    candidate_dump_path = Path(args.candidate_dump)
    output_dir = Path(args.output_dir) if args.output_dir else candidate_dump_path.parent / "replay_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cfg, rows_by_name = _load_run_context(candidate_dump_path)
    row_meta, candidates_by_row, ordered_row_indices = _load_candidate_pool(candidate_dump_path)

    default_k_values = run_cfg.get("generation", {}).get("K_values", [1, 4, 8])
    k_values = sorted(set(int(k) for k in (args.k_values or default_k_values)))

    ordered_rows: list[dict] = []
    nl_by_row_index: dict[int, str] = {}
    for row_index in ordered_row_indices:
        meta = row_meta[row_index]
        row = rows_by_name[meta["planetarium_name"]]
        nl_by_row_index[row_index] = row["natural_language"]
        ordered_rows.append(
            {
                "domain": meta["domain"],
                "init_is_abstract": meta["init_is_abstract"],
                "goal_is_abstract": meta["goal_is_abstract"],
            }
        )

    verifier_summaries = []
    replay_dump = []

    for selection_path in args.selection:
        verifier_name = _resolve_verifier_name(selection_path)
        logger.info("Replaying fixed pool with verifier: %s", verifier_name)
        scorer = VerifierScorer(selection_path=selection_path)

        comparisons: dict[str, dict] = {str(k): {"policies": {}} for k in k_values}
        selected_results: dict[tuple[int, str], list[EvalResult]] = {
            (k, p): []
            for k in k_values
            for p in ["greedy_first", "random_parseable", "verifier_ranked"]
        }
        pool_parseable_counts: dict[int, list[int]] = {k: [] for k in k_values}
        pool_equiv_counts: dict[int, list[int]] = {k: [] for k in k_values}
        pool_oracle_best: dict[int, list[int]] = {k: [] for k in k_values}

        for row_index in ordered_row_indices:
            candidates = candidates_by_row[row_index]
            nl = nl_by_row_index[row_index]
            parseable_pairs = [
                (nl, rec["pddl"])
                for rec in candidates
                if rec.get("parseable") and rec.get("pddl")
            ]
            parseable_scores = scorer.score_pairs(parseable_pairs, batch_size=8) if parseable_pairs else []

            rescored_candidates = []
            score_iter = iter(parseable_scores)
            for rec in candidates:
                new_rec = dict(rec)
                new_rec["replay_verifier_score"] = next(score_iter) if rec.get("parseable") and rec.get("pddl") else None
                rescored_candidates.append(new_rec)
                replay_dump.append(
                    {
                        "verifier": verifier_name,
                        "row_index": row_index,
                        "planetarium_name": rec["planetarium_name"],
                        "domain": rec["domain"],
                        "candidate_index": int(rec["candidate_index"]),
                        "parseable": bool(rec.get("parseable")),
                        "equivalent": bool(rec.get("equivalent")),
                        "original_verifier_score": rec.get("verifier_score"),
                        "replay_verifier_score": new_rec["replay_verifier_score"],
                    }
                )

            eval_results = [
                EvalResult(
                    parseable=bool(rec.get("parseable")),
                    equivalent=bool(rec.get("equivalent")),
                    error=rec.get("error"),
                )
                for rec in rescored_candidates
            ]

            for k in k_values:
                subset_candidates = rescored_candidates[:k]
                subset_results = eval_results[:k]
                candidate_records = [
                    CandidateRecord(
                        index=int(rec["candidate_index"]),
                        parseable=bool(rec.get("parseable")),
                        equivalent=bool(rec.get("equivalent")),
                        verifier_score=rec.get("replay_verifier_score"),
                    )
                    for rec in subset_candidates
                ]

                pool_parseable_counts[k].append(sum(1 for rec in subset_candidates if rec.get("parseable")))
                pool_equiv_counts[k].append(sum(1 for rec in subset_candidates if rec.get("equivalent")))
                pool_oracle_best[k].append(1 if any(rec.get("equivalent") for rec in subset_candidates) else 0)

                rng = random.Random(args.seed + row_index * 1000 + k)
                selections = {
                    "greedy_first": greedy_first(candidate_records),
                    "random_parseable": random_parseable(candidate_records, rng),
                    "verifier_ranked": verifier_ranked(candidate_records),
                }

                for policy_name, selection in selections.items():
                    if selection.selected_index is None:
                        selected_results[(k, policy_name)].append(
                            EvalResult(parseable=False, equivalent=False, error=selection.reason)
                        )
                    else:
                        selected_results[(k, policy_name)].append(subset_results[selection.selected_index])
                    replay_dump.append(
                        {
                            "verifier": verifier_name,
                            "row_index": row_index,
                            "planetarium_name": row_meta[row_index]["planetarium_name"],
                            "K": k,
                            "policy": policy_name,
                            "selected_index": selection.selected_index,
                            "selection_reason": selection.reason,
                        }
                    )

        for k in k_values:
            k_key = str(k)
            for policy_name in ["greedy_first", "random_parseable", "verifier_ranked"]:
                comparisons[k_key]["policies"][policy_name] = _summarize_policy(
                    rows=ordered_rows,
                    results=selected_results[(k, policy_name)],
                    pool_parseable_counts=pool_parseable_counts[k],
                    pool_equiv_counts=pool_equiv_counts[k],
                    pool_oracle_best=pool_oracle_best[k],
                )

        verifier_summaries.append(
            {
                "name": verifier_name,
                "selection_metadata": selection_path,
                "comparisons": comparisons,
            }
        )

    summary = {
        "source_pool": {
            "candidate_dump": str(candidate_dump_path),
            "run_config": str(candidate_dump_path.parent / "run_config.yaml"),
            "rows_evaluated": len(ordered_row_indices),
            "evaluated_k_values": k_values,
        },
        "verifiers": verifier_summaries,
    }

    with open(output_dir / "replay_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "replay_summary.md", "w", encoding="utf-8") as f:
        f.write(_markdown(summary))
    with open(output_dir / "replay_dump.jsonl", "w", encoding="utf-8") as f:
        for record in replay_dump:
            f.write(json.dumps(record) + "\n")

    logger.info("Saved replay summary to %s", output_dir / "replay_summary.json")
    logger.info("Saved replay markdown to %s", output_dir / "replay_summary.md")
    logger.info("Saved replay dump to %s", output_dir / "replay_dump.jsonl")


if __name__ == "__main__":
    main()
