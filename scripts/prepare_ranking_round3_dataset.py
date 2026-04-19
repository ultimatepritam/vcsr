"""
Prepare a multi-pool ranking-aligned verifier dataset for round 3.

This script mines ranking-focused verifier examples from multiple immutable
best-of-K pool directories, merges and deduplicates them, and writes a manifest
that records exactly which pools contributed to the final dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vcsr_env  # noqa: F401

from data.verifier_dataset import VerifierDatasetBuilder, VerifierExample
from scripts.mine_verifier_ranking_examples import (
    _example_key,
    _load_candidate_dump,
    _load_planetarium_rows,
    _load_run_config,
    _negative_priority,
    _positive_sort_key,
    _score,
    _style_label,
)

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


def _pool_paths(pool_dir: Path) -> tuple[Path, Path]:
    candidate_dump_path = pool_dir / "candidate_dump.jsonl"
    run_config_path = pool_dir / "run_config.yaml"
    if not candidate_dump_path.exists():
        raise FileNotFoundError(f"Missing candidate dump: {candidate_dump_path}")
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run config: {run_config_path}")
    return candidate_dump_path, run_config_path


def _mine_single_pool(
    *,
    pool_dir: Path,
    k: int,
    max_positives_per_row: int,
    max_negatives_per_row: int,
    include_negative_only_rows: bool,
    max_negative_only_per_row: int,
) -> tuple[list[VerifierExample], dict]:
    candidate_dump_path, run_config_path = _pool_paths(pool_dir)
    run_config = _load_run_config(run_config_path)
    split_seed = int(run_config.get("experiment", {}).get("seed", 42))
    row_lookup = _load_planetarium_rows(run_config, split_seed=split_seed)
    row_meta, candidates_by_row, selections_by_row = _load_candidate_dump(candidate_dump_path)

    mined_examples: list[VerifierExample] = []
    detail_rows: list[dict] = []
    rows_considered = 0
    rows_with_parseable_pool = 0
    rows_with_positive_pool = 0
    rows_with_negative_pool = 0
    rows_with_verifier_miss = 0
    rows_with_negative_only_mining = 0

    for row_index in sorted(row_meta):
        rows_considered += 1
        meta = row_meta[row_index]
        planetarium_name = meta["planetarium_name"]
        source_row = row_lookup.get(planetarium_name)
        if source_row is None:
            logger.warning("Skipping %s because it was not found in the configured dataset split", planetarium_name)
            continue

        selection_block = selections_by_row.get(row_index, {}).get(k, {})
        verifier_selection = selection_block.get("verifier_ranked", {})
        selected_index = verifier_selection.get("selected_index")
        selected_negative_index = int(selected_index) if selected_index is not None else None

        candidate_subset = [
            candidates_by_row[row_index][cand_idx]
            for cand_idx in sorted(candidates_by_row[row_index])
            if cand_idx < k
        ]
        parseable_candidates = [rec for rec in candidate_subset if rec.get("parseable") and rec.get("pddl")]
        if not parseable_candidates:
            continue
        rows_with_parseable_pool += 1

        positive_candidates = [rec for rec in parseable_candidates if rec.get("equivalent")]
        negative_candidates = [rec for rec in parseable_candidates if not rec.get("equivalent")]
        if negative_candidates:
            rows_with_negative_pool += 1

        style = _style_label(meta["init_is_abstract"], meta["goal_is_abstract"])

        if positive_candidates:
            rows_with_positive_pool += 1
            sorted_positives = sorted(positive_candidates, key=_positive_sort_key, reverse=True)
            chosen_positives = sorted_positives[: max(1, max_positives_per_row)]
            best_positive_score = _score(sorted_positives[0])

            selected_wrong = any(
                int(rec["candidate_index"]) == selected_negative_index for rec in negative_candidates
            )
            if selected_wrong:
                rows_with_verifier_miss += 1

            ranked_negatives = sorted(
                negative_candidates,
                key=lambda rec: _negative_priority(
                    rec,
                    selected_negative_index=selected_negative_index,
                    best_positive_score=best_positive_score,
                ),
                reverse=True,
            )
            chosen_negatives = ranked_negatives[: max(1, max_negatives_per_row)]

            for rec in chosen_positives:
                mined_examples.append(
                    VerifierExample(
                        nl=source_row["natural_language"],
                        pddl=rec["pddl"],
                        label=1,
                        source="bestofk_ranking_positive",
                        source_model=rec.get("model", ""),
                        domain=meta["domain"],
                        init_is_abstract=meta["init_is_abstract"],
                        goal_is_abstract=meta["goal_is_abstract"],
                        parseable=True,
                        planetarium_name=planetarium_name,
                    )
                )

            for rec in chosen_negatives:
                mined_examples.append(
                    VerifierExample(
                        nl=source_row["natural_language"],
                        pddl=rec["pddl"],
                        label=0,
                        source="bestofk_ranking_negative",
                        source_model=rec.get("model", ""),
                        domain=meta["domain"],
                        init_is_abstract=meta["init_is_abstract"],
                        goal_is_abstract=meta["goal_is_abstract"],
                        parseable=True,
                        planetarium_name=planetarium_name,
                    )
                )

            detail_rows.append(
                {
                    "row_index": row_index,
                    "planetarium_name": planetarium_name,
                    "domain": meta["domain"],
                    "style": style,
                    "k": k,
                    "pool_type": "positive_pool",
                    "verifier_selected_index": selected_negative_index,
                    "verifier_selected_wrong": selected_wrong,
                    "available_positive_indices": [int(rec["candidate_index"]) for rec in sorted_positives],
                    "available_negative_indices": [int(rec["candidate_index"]) for rec in ranked_negatives],
                    "chosen_positive_indices": [int(rec["candidate_index"]) for rec in chosen_positives],
                    "chosen_negative_indices": [int(rec["candidate_index"]) for rec in chosen_negatives],
                    "best_positive_score": best_positive_score,
                }
            )
            continue

        if not include_negative_only_rows or not negative_candidates:
            continue

        rows_with_negative_only_mining += 1
        ranked_negatives = sorted(negative_candidates, key=_positive_sort_key, reverse=True)
        chosen_negatives = ranked_negatives[: max(1, max_negative_only_per_row)]
        for rec in chosen_negatives:
            mined_examples.append(
                VerifierExample(
                    nl=source_row["natural_language"],
                    pddl=rec["pddl"],
                    label=0,
                    source="bestofk_ranking_negative_only",
                    source_model=rec.get("model", ""),
                    domain=meta["domain"],
                    init_is_abstract=meta["init_is_abstract"],
                    goal_is_abstract=meta["goal_is_abstract"],
                    parseable=True,
                    planetarium_name=planetarium_name,
                )
            )

        detail_rows.append(
            {
                "row_index": row_index,
                "planetarium_name": planetarium_name,
                "domain": meta["domain"],
                "style": style,
                "k": k,
                "pool_type": "negative_only_pool",
                "chosen_negative_indices": [int(rec["candidate_index"]) for rec in chosen_negatives],
            }
        )

    mined_deduped: list[VerifierExample] = []
    seen_keys: set[tuple[str, str, int, str]] = set()
    for example in mined_examples:
        key = _example_key(example)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        mined_deduped.append(example)

    by_source = Counter(example.source for example in mined_deduped)
    by_domain = Counter(example.domain for example in mined_deduped)
    by_style = Counter(_style_label(example.init_is_abstract, example.goal_is_abstract) for example in mined_deduped)

    pool_report = {
        "pool_dir": str(pool_dir),
        "candidate_dump": str(candidate_dump_path),
        "run_config": str(run_config_path),
        "seed": split_seed,
        "summary": {
            "rows_considered": rows_considered,
            "rows_with_parseable_pool": rows_with_parseable_pool,
            "rows_with_positive_pool": rows_with_positive_pool,
            "rows_with_negative_pool": rows_with_negative_pool,
            "rows_with_verifier_miss": rows_with_verifier_miss,
            "rows_with_negative_only_mining": rows_with_negative_only_mining,
            "mined_examples_total": len(mined_deduped),
            "mined_positive_examples": sum(1 for ex in mined_deduped if ex.label == 1),
            "mined_negative_examples": sum(1 for ex in mined_deduped if ex.label == 0),
        },
        "by_source": dict(by_source.most_common()),
        "by_domain": dict(by_domain.most_common()),
        "by_style": dict(by_style.most_common()),
        "row_details": detail_rows,
    }
    return mined_deduped, pool_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multi-pool ranking-aligned verifier dataset")
    parser.add_argument("--pool_dir", action="append", required=True, help="Pool directory containing candidate_dump.jsonl and run_config.yaml")
    parser.add_argument(
        "--base_jsonl",
        type=str,
        default="results/neggen/pilot/verifier_train.relabeled.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/verifier/ranking_aligned_round3",
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max_positives_per_row", type=int, default=2)
    parser.add_argument("--max_negatives_per_row", type=int, default=4)
    parser.add_argument(
        "--include_negative_only_rows",
        action="store_true",
        help="Also add top-scoring parseable negatives from rows that have no equivalent candidate in-pool.",
    )
    parser.add_argument("--max_negative_only_per_row", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_file_logging(output_dir)
    logger.info("Live mining log: %s", log_path)

    pool_dirs = [Path(p) for p in args.pool_dir]
    all_examples: list[VerifierExample] = []
    pool_reports: list[dict] = []

    for idx, pool_dir in enumerate(pool_dirs, start=1):
        logger.info("Mining pool %d/%d: %s", idx, len(pool_dirs), pool_dir)
        pool_examples, pool_report = _mine_single_pool(
            pool_dir=pool_dir,
            k=args.k,
            max_positives_per_row=args.max_positives_per_row,
            max_negatives_per_row=args.max_negatives_per_row,
            include_negative_only_rows=args.include_negative_only_rows,
            max_negative_only_per_row=args.max_negative_only_per_row,
        )
        all_examples.extend(pool_examples)
        pool_reports.append(pool_report)
        logger.info(
            "Pool %s mined=%d positives=%d negatives=%d miss_rows=%d",
            pool_dir.name,
            pool_report["summary"]["mined_examples_total"],
            pool_report["summary"]["mined_positive_examples"],
            pool_report["summary"]["mined_negative_examples"],
            pool_report["summary"]["rows_with_verifier_miss"],
        )

    merged_deduped: list[VerifierExample] = []
    seen_keys: set[tuple[str, str, int, str]] = set()
    for example in all_examples:
        key = _example_key(example)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_deduped.append(example)

    merged_builder = VerifierDatasetBuilder()
    merged_builder.add_examples(merged_deduped)
    merged_builder.save_jsonl(output_dir / "mined_examples.jsonl")
    merged_builder.save_stats(output_dir / "mined_examples_stats.json")

    base_builder = VerifierDatasetBuilder.load_jsonl(Path(args.base_jsonl))
    base_keys = {_example_key(example) for example in base_builder.get_parseable_examples()}
    new_examples = [example for example in merged_deduped if _example_key(example) not in base_keys]
    base_builder.add_examples(new_examples)
    base_builder.save_jsonl(output_dir / "augmented_train.jsonl")
    base_builder.save_stats(output_dir / "augmented_train_stats.json")

    by_source = Counter(example.source for example in merged_deduped)
    by_domain = Counter(example.domain for example in merged_deduped)
    by_style = Counter(_style_label(example.init_is_abstract, example.goal_is_abstract) for example in merged_deduped)

    report = {
        "inputs": {
            "pool_dirs": [str(p) for p in pool_dirs],
            "base_jsonl": args.base_jsonl,
            "k": args.k,
            "max_positives_per_row": args.max_positives_per_row,
            "max_negatives_per_row": args.max_negatives_per_row,
            "include_negative_only_rows": args.include_negative_only_rows,
            "max_negative_only_per_row": args.max_negative_only_per_row,
        },
        "summary": {
            "pool_count": len(pool_dirs),
            "raw_examples_before_dedup": len(all_examples),
            "merged_examples_total": len(merged_deduped),
            "merged_positive_examples": sum(1 for ex in merged_deduped if ex.label == 1),
            "merged_negative_examples": sum(1 for ex in merged_deduped if ex.label == 0),
            "new_examples_added_to_augmented_train": len(new_examples),
            "augmented_train_total": len(base_builder),
        },
        "by_source": dict(by_source.most_common()),
        "by_domain": dict(by_domain.most_common()),
        "by_style": dict(by_style.most_common()),
        "pools": pool_reports,
    }

    with open(output_dir / "mining_manifest.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(output_dir / "mining_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Multi-pool mining complete: pools=%d raw=%d merged=%d added=%d",
        len(pool_dirs),
        len(all_examples),
        len(merged_deduped),
        len(new_examples),
    )


if __name__ == "__main__":
    main()
