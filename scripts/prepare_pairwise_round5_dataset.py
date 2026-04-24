"""
Prepare pairwise round-5 verifier data from cached best-of-K pools.

The pairwise file trains the scorer to prefer an equivalent candidate over a
non-equivalent candidate from the same NL row and same K-prefix. A companion
pointwise file keeps the broad BCE branch supplied with the same hard examples,
plus negative-only rows that are deliberately excluded from pairwise training.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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


def _flush_logs() -> None:
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def _write_progress(
    output_dir: Path,
    *,
    status: str,
    completed_pools: int,
    total_pools: int,
    current_pool: str,
    started_at: float,
) -> None:
    snapshot = {
        "status": status,
        "completed_pools": completed_pools,
        "total_pools": total_pools,
        "remaining_pools": max(0, total_pools - completed_pools),
        "current_pool": current_pool,
        "elapsed_sec": max(0.0, time.time() - started_at),
    }
    with open(output_dir / "progress.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)


def _write_process_info(output_dir: Path, command: list[str]) -> None:
    info = {
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": " ".join(command),
        "output_dir": str(output_dir),
        "progress_log": str(output_dir / "progress.log"),
        "progress_json": str(output_dir / "progress.json"),
    }
    with open(output_dir / "process_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def _pool_paths(pool_dir: Path) -> tuple[Path, Path]:
    candidate_dump_path = pool_dir / "candidate_dump.jsonl"
    run_config_path = pool_dir / "run_config.yaml"
    if not candidate_dump_path.exists():
        raise FileNotFoundError(f"Missing candidate dump: {candidate_dump_path}")
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run config: {run_config_path}")
    return candidate_dump_path, run_config_path


def _pair_key(pair: dict) -> tuple[str, str, str, str, int]:
    return (
        pair["nl"],
        pair["positive_pddl"],
        pair["negative_pddl"],
        pair["planetarium_name"],
        int(pair["K"]),
    )


def _pointwise_example(
    *,
    source_row: dict,
    rec: dict,
    label: int,
    source: str,
    meta: dict,
) -> VerifierExample:
    return VerifierExample(
        nl=source_row["natural_language"],
        pddl=rec["pddl"],
        label=label,
        source=source,
        source_model=rec.get("model", ""),
        domain=meta["domain"],
        init_is_abstract=meta["init_is_abstract"],
        goal_is_abstract=meta["goal_is_abstract"],
        parseable=True,
        planetarium_name=meta["planetarium_name"],
    )


def _select_negatives(
    *,
    negatives: list[dict],
    selected_wrong_index: int | None,
    best_positive_score: float,
    near_tie_margin: float,
    moderate_gap_margin: float,
    max_negatives: int,
) -> list[tuple[dict, str]]:
    chosen: list[tuple[dict, str]] = []
    seen: set[int] = set()

    def add(rec: dict, pair_type: str) -> None:
        idx = int(rec["candidate_index"])
        if idx in seen or len(chosen) >= max_negatives:
            return
        seen.add(idx)
        chosen.append((rec, pair_type))

    if selected_wrong_index is not None:
        for rec in negatives:
            if int(rec["candidate_index"]) == selected_wrong_index:
                add(rec, "selected_wrong")
                break

    outranking = [rec for rec in negatives if _score(rec) >= best_positive_score]
    near_ties = [
        rec
        for rec in negatives
        if 0 <= best_positive_score - _score(rec) <= near_tie_margin
    ]
    moderate = [
        rec
        for rec in negatives
        if near_tie_margin < best_positive_score - _score(rec) <= moderate_gap_margin
    ]

    buckets = [
        (outranking, "outranks_positive"),
        (near_ties, "near_tie"),
        (moderate, "moderate_gap"),
    ]
    for bucket, pair_type in buckets:
        for rec in sorted(bucket, key=lambda item: (_score(item), -int(item["candidate_index"])), reverse=True):
            add(rec, pair_type)

    return chosen


def _mine_pool(
    *,
    pool_dir: Path,
    k_values: list[int],
    max_positives_per_row: int,
    max_negatives_per_row: int,
    max_pairs_per_row_per_k: int,
    max_negative_only_per_row: int,
    near_tie_margin: float,
    moderate_gap_margin: float,
) -> tuple[list[dict], list[VerifierExample], dict]:
    candidate_dump_path, run_config_path = _pool_paths(pool_dir)
    run_config = _load_run_config(run_config_path)
    split_seed = int(run_config.get("experiment", {}).get("seed", 42))
    row_lookup = _load_planetarium_rows(run_config, split_seed=split_seed)
    row_meta, candidates_by_row, selections_by_row = _load_candidate_dump(candidate_dump_path)

    pair_rows: list[dict] = []
    pointwise_examples: list[VerifierExample] = []
    row_details: list[dict] = []
    counters = Counter()

    for row_index in sorted(row_meta):
        meta = row_meta[row_index]
        planetarium_name = meta["planetarium_name"]
        source_row = row_lookup.get(planetarium_name)
        if source_row is None:
            counters["missing_source_row"] += 1
            continue

        for k_value in k_values:
            candidate_subset = [
                candidates_by_row[row_index][cand_idx]
                for cand_idx in sorted(candidates_by_row[row_index])
                if cand_idx < k_value
            ]
            parseable = [rec for rec in candidate_subset if rec.get("parseable") and rec.get("pddl")]
            positives = [rec for rec in parseable if rec.get("equivalent")]
            negatives = [rec for rec in parseable if not rec.get("equivalent")]

            if not parseable:
                counters["rows_without_parseable_pool"] += 1
                continue

            style = _style_label(meta["init_is_abstract"], meta["goal_is_abstract"])

            if not positives:
                counters["negative_only_rows"] += 1
                for rec in sorted(negatives, key=lambda item: (_score(item), -int(item["candidate_index"])), reverse=True)[
                    :max(0, max_negative_only_per_row)
                ]:
                    pointwise_examples.append(
                        _pointwise_example(
                            source_row=source_row,
                            rec=rec,
                            label=0,
                            source="pairwise_round5_negative_only",
                            meta=meta,
                        )
                    )
                continue

            if not negatives:
                counters["positive_only_rows"] += 1
                for rec in sorted(positives, key=lambda item: (_score(item), -int(item["candidate_index"])), reverse=True)[
                    :max_positives_per_row
                ]:
                    pointwise_examples.append(
                        _pointwise_example(
                            source_row=source_row,
                            rec=rec,
                            label=1,
                            source="pairwise_round5_positive_only",
                            meta=meta,
                        )
                    )
                continue

            sorted_positives = sorted(
                positives,
                key=lambda item: (_score(item), -int(item["candidate_index"])),
                reverse=True,
            )[:max_positives_per_row]
            best_positive_score = _score(sorted_positives[0])
            selection = selections_by_row.get(row_index, {}).get(k_value, {}).get("verifier_ranked", {})
            selected_index = selection.get("selected_index")
            selected_wrong_index = int(selected_index) if selected_index is not None else None
            if selected_wrong_index is not None and not any(
                int(rec["candidate_index"]) == selected_wrong_index for rec in negatives
            ):
                selected_wrong_index = None

            chosen_negatives = _select_negatives(
                negatives=negatives,
                selected_wrong_index=selected_wrong_index,
                best_positive_score=best_positive_score,
                near_tie_margin=near_tie_margin,
                moderate_gap_margin=moderate_gap_margin,
                max_negatives=max_negatives_per_row,
            )
            if not chosen_negatives:
                counters["positive_negative_rows_without_selected_negative"] += 1
                continue

            pairs_for_row = 0
            for pos_rec in sorted_positives:
                pointwise_examples.append(
                    _pointwise_example(
                        source_row=source_row,
                        rec=pos_rec,
                        label=1,
                        source="pairwise_round5_pair_positive",
                        meta=meta,
                    )
                )
                for neg_rec, pair_type in chosen_negatives:
                    if pairs_for_row >= max_pairs_per_row_per_k:
                        break
                    pointwise_examples.append(
                        _pointwise_example(
                            source_row=source_row,
                            rec=neg_rec,
                            label=0,
                            source="pairwise_round5_pair_negative",
                            meta=meta,
                        )
                    )
                    pair_rows.append(
                        {
                            "nl": source_row["natural_language"],
                            "positive_pddl": pos_rec["pddl"],
                            "negative_pddl": neg_rec["pddl"],
                            "source": "bestofk_pairwise_round5",
                            "candidate_dump": str(candidate_dump_path),
                            "row_index": row_index,
                            "planetarium_name": planetarium_name,
                            "domain": meta["domain"],
                            "init_is_abstract": meta["init_is_abstract"],
                            "goal_is_abstract": meta["goal_is_abstract"],
                            "style": style,
                            "K": k_value,
                            "pair_type": pair_type,
                            "positive_candidate_index": int(pos_rec["candidate_index"]),
                            "negative_candidate_index": int(neg_rec["candidate_index"]),
                            "positive_score": _score(pos_rec),
                            "negative_score": _score(neg_rec),
                            "score_margin": _score(pos_rec) - _score(neg_rec),
                        }
                    )
                    counters[f"pair_type:{pair_type}"] += 1
                    pairs_for_row += 1

            counters["positive_negative_rows"] += 1
            counters["pair_rows"] += pairs_for_row
            row_details.append(
                {
                    "pool_dir": str(pool_dir),
                    "row_index": row_index,
                    "planetarium_name": planetarium_name,
                    "domain": meta["domain"],
                    "style": style,
                    "K": k_value,
                    "positive_indices": [int(rec["candidate_index"]) for rec in sorted_positives],
                    "negative_indices": [int(rec["candidate_index"]) for rec, _ in chosen_negatives],
                    "pairs": pairs_for_row,
                }
            )

    report = {
        "pool_dir": str(pool_dir),
        "candidate_dump": str(candidate_dump_path),
        "run_config": str(run_config_path),
        "seed": split_seed,
        "summary": dict(counters),
        "row_details": row_details,
    }
    return pair_rows, pointwise_examples, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare pairwise round-5 verifier dataset")
    parser.add_argument(
        "--pool_dir",
        action="append",
        default=[
            "results/vcsr/round3_pool_seed43",
            "results/vcsr/round3_pool_seed44",
            "results/vcsr/round3_pool_seed45",
            "results/vcsr/round3_pool_seed46",
            "results/vcsr/round3_pool_seed47",
        ],
    )
    parser.add_argument("--base_jsonl", type=str, default="results/neggen/pilot/verifier_train.relabeled.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/verifier/pairwise_round5")
    parser.add_argument("--k_values", type=int, nargs="*", default=[4, 8])
    parser.add_argument("--max_positives_per_row", type=int, default=2)
    parser.add_argument("--max_negatives_per_row", type=int, default=4)
    parser.add_argument("--max_pairs_per_row_per_k", type=int, default=6)
    parser.add_argument("--max_negative_only_per_row", type=int, default=1)
    parser.add_argument("--near_tie_margin", type=float, default=0.03)
    parser.add_argument("--moderate_gap_margin", type=float, default=0.10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_file_logging(output_dir)
    _write_process_info(output_dir, sys.argv)
    started_at = time.time()
    pool_dirs = [Path(path) for path in args.pool_dir]
    logger.info("Live pairwise mining log: %s", log_path)
    _write_progress(
        output_dir,
        status="starting",
        completed_pools=0,
        total_pools=len(pool_dirs),
        current_pool="",
        started_at=started_at,
    )

    all_pair_rows: list[dict] = []
    all_pointwise_examples: list[VerifierExample] = []
    pool_reports: list[dict] = []

    for idx, pool_dir in enumerate(pool_dirs, start=1):
        logger.info("Mining pool %d/%d: %s", idx, len(pool_dirs), pool_dir)
        _write_progress(
            output_dir,
            status="mining",
            completed_pools=idx - 1,
            total_pools=len(pool_dirs),
            current_pool=str(pool_dir),
            started_at=started_at,
        )
        pair_rows, pointwise_examples, pool_report = _mine_pool(
            pool_dir=pool_dir,
            k_values=sorted(set(args.k_values)),
            max_positives_per_row=max(1, args.max_positives_per_row),
            max_negatives_per_row=max(1, args.max_negatives_per_row),
            max_pairs_per_row_per_k=max(1, args.max_pairs_per_row_per_k),
            max_negative_only_per_row=max(0, args.max_negative_only_per_row),
            near_tie_margin=float(args.near_tie_margin),
            moderate_gap_margin=float(args.moderate_gap_margin),
        )
        all_pair_rows.extend(pair_rows)
        all_pointwise_examples.extend(pointwise_examples)
        pool_reports.append(pool_report)
        logger.info(
            "Pool %s produced pairs=%d pointwise=%d",
            pool_dir.name,
            len(pair_rows),
            len(pointwise_examples),
        )
        _flush_logs()

    deduped_pairs: list[dict] = []
    seen_pairs: set[tuple[str, str, str, str, int]] = set()
    for pair in all_pair_rows:
        key = _pair_key(pair)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        deduped_pairs.append(pair)

    deduped_pointwise: list[VerifierExample] = []
    seen_examples: set[tuple[str, str, int, str]] = set()
    for example in all_pointwise_examples:
        key = _example_key(example)
        if key in seen_examples:
            continue
        seen_examples.add(key)
        deduped_pointwise.append(example)

    with open(output_dir / "pairwise_train.jsonl", "w", encoding="utf-8") as f:
        for row in deduped_pairs:
            f.write(json.dumps(row) + "\n")

    pointwise_builder = VerifierDatasetBuilder()
    pointwise_builder.add_examples(deduped_pointwise)
    pointwise_builder.save_jsonl(output_dir / "pointwise_extra.jsonl")
    pointwise_builder.save_stats(output_dir / "pointwise_extra_stats.json")

    by_domain = Counter(row["domain"] for row in deduped_pairs)
    by_style = Counter(row["style"] for row in deduped_pairs)
    by_k = Counter(int(row["K"]) for row in deduped_pairs)
    by_pair_type = Counter(row["pair_type"] for row in deduped_pairs)
    report = {
        "inputs": {
            "pool_dirs": [str(path) for path in pool_dirs],
            "base_jsonl": args.base_jsonl,
            "k_values": sorted(set(args.k_values)),
            "max_positives_per_row": args.max_positives_per_row,
            "max_negatives_per_row": args.max_negatives_per_row,
            "max_pairs_per_row_per_k": args.max_pairs_per_row_per_k,
            "max_negative_only_per_row": args.max_negative_only_per_row,
            "near_tie_margin": args.near_tie_margin,
            "moderate_gap_margin": args.moderate_gap_margin,
        },
        "summary": {
            "pool_count": len(pool_dirs),
            "raw_pair_rows": len(all_pair_rows),
            "pair_rows": len(deduped_pairs),
            "raw_pointwise_examples": len(all_pointwise_examples),
            "pointwise_examples": len(deduped_pointwise),
        },
        "by_domain": dict(by_domain.most_common()),
        "by_style": dict(by_style.most_common()),
        "by_k": dict(sorted(by_k.items())),
        "by_pair_type": dict(by_pair_type.most_common()),
        "pools": pool_reports,
    }
    with open(output_dir / "pairwise_stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "total": len(deduped_pairs),
                "by_domain": dict(by_domain.most_common()),
                "by_style": dict(by_style.most_common()),
                "by_k": dict(sorted(by_k.items())),
                "by_pair_type": dict(by_pair_type.most_common()),
            },
            f,
            indent=2,
        )
    with open(output_dir / "mining_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _write_progress(
        output_dir,
        status="completed",
        completed_pools=len(pool_dirs),
        total_pools=len(pool_dirs),
        current_pool="",
        started_at=started_at,
    )
    logger.info("Pairwise round-5 mining complete: pairs=%d pointwise=%d", len(deduped_pairs), len(deduped_pointwise))


if __name__ == "__main__":
    main()
