"""
Prepare a round-4-style focused pointwise verifier dataset for round 7.

This is intentionally not pairwise training data. It mines standard
``(nl, candidate_pddl, label)`` verifier examples from cached best-of-K pools,
rescored with the promoted round-4 verifier, so the next model can be trained
with the same pointwise objective that made round 4 useful.
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

from data.verifier_dataset import VerifierDatasetBuilder, VerifierExample
from scripts.mine_verifier_ranking_examples import (
    _example_key,
    _load_candidate_dump,
    _load_planetarium_rows,
    _load_run_config,
    _style_label,
)
from verifier.inference import VerifierScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_POOL_DIRS = [
    "results/vcsr/bestofk_pilot",
    "results/vcsr/bestofk_ranking_round2_pool",
    "results/vcsr/round3_pool_seed43",
    "results/vcsr/round3_pool_seed44",
    "results/vcsr/round3_pool_seed45",
    "results/vcsr/round3_pool_seed46",
    "results/vcsr/round3_pool_seed47",
    "results/vcsr/bestofk_round3_holdout_eval",
    "results/vcsr/bestofk_round4_holdout_eval_clean",
]


def _configure_file_logging(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
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


def _write_progress(output_dir: Path, payload: dict) -> None:
    with open(output_dir / "progress.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _pool_paths(pool_dir: Path) -> tuple[Path, Path]:
    candidate_dump_path = pool_dir / "candidate_dump.jsonl"
    run_config_path = pool_dir / "run_config.yaml"
    if not candidate_dump_path.exists():
        raise FileNotFoundError(f"Missing candidate dump: {candidate_dump_path}")
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run config: {run_config_path}")
    return candidate_dump_path, run_config_path


def _score_candidate_pool(
    *,
    scorer: VerifierScorer,
    row_lookup: dict[str, dict],
    row_meta: dict[int, dict],
    candidates_by_row: dict[int, dict[int, dict]],
    batch_size: int,
) -> dict[tuple[int, int], float]:
    pairs: list[tuple[str, str]] = []
    keys: list[tuple[int, int]] = []
    for row_index in sorted(row_meta):
        planetarium_name = row_meta[row_index]["planetarium_name"]
        source_row = row_lookup.get(planetarium_name)
        if source_row is None:
            continue
        nl = source_row["natural_language"]
        for candidate_index in sorted(candidates_by_row[row_index]):
            rec = candidates_by_row[row_index][candidate_index]
            if rec.get("parseable") and rec.get("pddl"):
                keys.append((row_index, candidate_index))
                pairs.append((nl, rec["pddl"]))

    scores = scorer.score_pairs(pairs, batch_size=batch_size) if pairs else []
    return {key: float(score) for key, score in zip(keys, scores)}


def _round4_score(rec: dict, score_map: dict[tuple[int, int], float], row_index: int) -> float:
    return float(score_map.get((row_index, int(rec["candidate_index"])), -1.0))


def _select_positive_candidates(
    positives: list[dict],
    *,
    score_map: dict[tuple[int, int], float],
    row_index: int,
    limit: int,
) -> list[dict]:
    return sorted(
        positives,
        key=lambda rec: (_round4_score(rec, score_map, row_index), -int(rec["candidate_index"])),
        reverse=True,
    )[: max(1, limit)]


def _select_negative_candidates(
    negatives: list[dict],
    *,
    score_map: dict[tuple[int, int], float],
    row_index: int,
    selected_wrong_index: int | None,
    best_positive_score: float | None,
    limit: int,
) -> list[dict]:
    def priority(rec: dict) -> tuple[int, int, float, int]:
        idx = int(rec["candidate_index"])
        selected_wrong = 1 if selected_wrong_index is not None and idx == selected_wrong_index else 0
        outranks_positive = (
            1
            if best_positive_score is not None and _round4_score(rec, score_map, row_index) >= best_positive_score
            else 0
        )
        return (selected_wrong, outranks_positive, _round4_score(rec, score_map, row_index), -idx)

    return sorted(negatives, key=priority, reverse=True)[: max(1, limit)]


def _make_example(
    *,
    source_row: dict,
    candidate: dict,
    label: int,
    source: str,
    meta: dict,
) -> VerifierExample:
    return VerifierExample(
        nl=source_row["natural_language"],
        pddl=candidate["pddl"],
        label=int(label),
        source=source,
        source_model=candidate.get("model", ""),
        domain=meta["domain"],
        init_is_abstract=meta["init_is_abstract"],
        goal_is_abstract=meta["goal_is_abstract"],
        parseable=True,
        planetarium_name=meta["planetarium_name"],
    )


def _mine_pool(
    *,
    pool_dir: Path,
    scorer: VerifierScorer,
    k_values: list[int],
    max_positives_per_row: int,
    max_negatives_per_row: int,
    max_negative_only_per_row: int,
    batch_size: int,
) -> tuple[list[VerifierExample], dict]:
    candidate_dump_path, run_config_path = _pool_paths(pool_dir)
    run_config = _load_run_config(run_config_path)
    split_seed = int(run_config.get("experiment", {}).get("seed", 42))
    row_lookup = _load_planetarium_rows(run_config, split_seed=split_seed)
    row_meta, candidates_by_row, selections_by_row = _load_candidate_dump(candidate_dump_path)

    score_map = _score_candidate_pool(
        scorer=scorer,
        row_lookup=row_lookup,
        row_meta=row_meta,
        candidates_by_row=candidates_by_row,
        batch_size=batch_size,
    )

    examples: list[VerifierExample] = []
    row_details: list[dict] = []
    summary = Counter()
    by_domain = Counter()
    by_style = Counter()
    by_k = Counter()

    for row_index in sorted(row_meta):
        meta = row_meta[row_index]
        source_row = row_lookup.get(meta["planetarium_name"])
        if source_row is None:
            summary["rows_skipped_missing_source"] += 1
            continue

        style = _style_label(meta["init_is_abstract"], meta["goal_is_abstract"])
        for k in k_values:
            summary["row_k_considered"] += 1
            by_k[k] += 1
            selection_block = selections_by_row.get(row_index, {}).get(k, {})
            verifier_selection = selection_block.get("verifier_ranked", {})
            selected_index = verifier_selection.get("selected_index")
            selected_index = int(selected_index) if selected_index is not None else None

            subset = [
                candidates_by_row[row_index][candidate_index]
                for candidate_index in sorted(candidates_by_row[row_index])
                if candidate_index < k
            ]
            parseable = [rec for rec in subset if rec.get("parseable") and rec.get("pddl")]
            if not parseable:
                summary["row_k_no_parseable"] += 1
                continue

            positives = [rec for rec in parseable if rec.get("equivalent")]
            negatives = [rec for rec in parseable if not rec.get("equivalent")]
            if positives:
                summary["row_k_positive_pool"] += 1
                chosen_positives = _select_positive_candidates(
                    positives,
                    score_map=score_map,
                    row_index=row_index,
                    limit=max_positives_per_row,
                )
                best_positive_score = _round4_score(chosen_positives[0], score_map, row_index)
                selected_wrong = selected_index is not None and any(
                    int(rec["candidate_index"]) == selected_index for rec in negatives
                )
                if selected_wrong:
                    summary["row_k_round4_miss"] += 1

                chosen_negatives = _select_negative_candidates(
                    negatives,
                    score_map=score_map,
                    row_index=row_index,
                    selected_wrong_index=selected_index if selected_wrong else None,
                    best_positive_score=best_positive_score,
                    limit=max_negatives_per_row,
                )

                for rec in chosen_positives:
                    examples.append(
                        _make_example(
                            source_row=source_row,
                            candidate=rec,
                            label=1,
                            source="bestofk_round7_focus_positive",
                            meta=meta,
                        )
                    )
                for rec in chosen_negatives:
                    examples.append(
                        _make_example(
                            source_row=source_row,
                            candidate=rec,
                            label=0,
                            source="bestofk_round7_focus_negative",
                            meta=meta,
                        )
                    )

                row_details.append(
                    {
                        "pool_dir": str(pool_dir),
                        "candidate_dump": str(candidate_dump_path),
                        "row_index": row_index,
                        "planetarium_name": meta["planetarium_name"],
                        "domain": meta["domain"],
                        "style": style,
                        "K": k,
                        "pool_type": "positive_pool",
                        "round4_selected_index": selected_index,
                        "round4_selected_wrong": selected_wrong,
                        "available_positive_indices": [int(rec["candidate_index"]) for rec in positives],
                        "available_negative_indices": [int(rec["candidate_index"]) for rec in negatives],
                        "chosen_positive_indices": [int(rec["candidate_index"]) for rec in chosen_positives],
                        "chosen_negative_indices": [int(rec["candidate_index"]) for rec in chosen_negatives],
                        "best_positive_score": best_positive_score,
                    }
                )
            elif negatives:
                summary["row_k_negative_only_pool"] += 1
                chosen_negatives = sorted(
                    negatives,
                    key=lambda rec: (_round4_score(rec, score_map, row_index), -int(rec["candidate_index"])),
                    reverse=True,
                )[: max(1, max_negative_only_per_row)]
                for rec in chosen_negatives:
                    examples.append(
                        _make_example(
                            source_row=source_row,
                            candidate=rec,
                            label=0,
                            source="bestofk_round7_negative_only",
                            meta=meta,
                        )
                    )
                row_details.append(
                    {
                        "pool_dir": str(pool_dir),
                        "candidate_dump": str(candidate_dump_path),
                        "row_index": row_index,
                        "planetarium_name": meta["planetarium_name"],
                        "domain": meta["domain"],
                        "style": style,
                        "K": k,
                        "pool_type": "negative_only_pool",
                        "chosen_negative_indices": [int(rec["candidate_index"]) for rec in chosen_negatives],
                    }
                )

    deduped: list[VerifierExample] = []
    seen: set[tuple[str, str, int, str]] = set()
    for example in examples:
        key = _example_key(example)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
        by_domain[example.domain] += 1
        by_style[_style_label(example.init_is_abstract, example.goal_is_abstract)] += 1

    pool_report = {
        "pool_dir": str(pool_dir),
        "candidate_dump": str(candidate_dump_path),
        "run_config": str(run_config_path),
        "seed": split_seed,
        "summary": {
            **dict(summary),
            "raw_examples_before_dedup": len(examples),
            "mined_examples_total": len(deduped),
            "mined_positive_examples": sum(1 for ex in deduped if ex.label == 1),
            "mined_negative_examples": sum(1 for ex in deduped if ex.label == 0),
            "parseable_candidates_rescored": len(score_map),
        },
        "by_domain": dict(by_domain.most_common()),
        "by_style": dict(by_style.most_common()),
        "by_k": dict(sorted(by_k.items())),
        "row_details": row_details,
    }
    return deduped, pool_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare focused round-7 pointwise verifier dataset")
    parser.add_argument("--pool_dir", action="append", default=None)
    parser.add_argument("--base_jsonl", default="results/neggen/pilot/verifier_train.relabeled.jsonl")
    parser.add_argument("--output_dir", default="results/verifier/focused_round7")
    parser.add_argument("--selection", default="results/verifier/best_current/selection.yaml")
    parser.add_argument("--k_values", type=int, nargs="*", default=[4, 8])
    parser.add_argument("--max_positives_per_row", type=int, default=2)
    parser.add_argument("--max_negatives_per_row", type=int, default=4)
    parser.add_argument("--max_negative_only_per_row", type=int, default=1)
    parser.add_argument("--score_batch_size", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_file_logging(output_dir)
    _write_process_info(output_dir, sys.argv)
    started_at = time.time()
    logger.info("Live round-7 mining log: %s", log_path)

    pool_dirs = [Path(p) for p in (args.pool_dir or DEFAULT_POOL_DIRS)]
    missing = [str(p) for p in pool_dirs if not (p / "candidate_dump.jsonl").exists()]
    if missing:
        raise FileNotFoundError(f"Missing candidate dumps for pool dirs: {missing}")

    _write_progress(
        output_dir,
        {
            "status": "starting",
            "elapsed_sec": 0.0,
            "pool_count": len(pool_dirs),
            "current_pool": None,
            "completed_pools": 0,
        },
    )

    logger.info("Loading round-4 scorer from %s", args.selection)
    scorer = VerifierScorer(selection_path=args.selection)

    all_examples: list[VerifierExample] = []
    pool_reports: list[dict] = []
    for idx, pool_dir in enumerate(pool_dirs, start=1):
        logger.info("Mining pool %d/%d: %s", idx, len(pool_dirs), pool_dir)
        _write_progress(
            output_dir,
            {
                "status": "mining_pool",
                "elapsed_sec": time.time() - started_at,
                "pool_count": len(pool_dirs),
                "current_pool": str(pool_dir),
                "completed_pools": idx - 1,
            },
        )
        pool_examples, pool_report = _mine_pool(
            pool_dir=pool_dir,
            scorer=scorer,
            k_values=sorted(set(int(k) for k in args.k_values)),
            max_positives_per_row=args.max_positives_per_row,
            max_negatives_per_row=args.max_negatives_per_row,
            max_negative_only_per_row=args.max_negative_only_per_row,
            batch_size=args.score_batch_size,
        )
        all_examples.extend(pool_examples)
        pool_reports.append(pool_report)
        logger.info(
            "Pool %s mined=%d pos=%d neg=%d misses=%d",
            pool_dir.name,
            pool_report["summary"]["mined_examples_total"],
            pool_report["summary"]["mined_positive_examples"],
            pool_report["summary"]["mined_negative_examples"],
            pool_report["summary"].get("row_k_round4_miss", 0),
        )

    merged_deduped: list[VerifierExample] = []
    seen_keys: set[tuple[str, str, int, str]] = set()
    for example in all_examples:
        key = _example_key(example)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_deduped.append(example)

    mined_builder = VerifierDatasetBuilder()
    mined_builder.add_examples(merged_deduped)
    mined_builder.save_jsonl(output_dir / "mined_examples.jsonl")
    mined_builder.save_stats(output_dir / "mined_examples_stats.json")

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
            "selection": args.selection,
            "k_values": sorted(set(int(k) for k in args.k_values)),
            "max_positives_per_row": args.max_positives_per_row,
            "max_negatives_per_row": args.max_negatives_per_row,
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
    with open(output_dir / "mining_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(output_dir / "mining_manifest.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _write_progress(
        output_dir,
        {
            "status": "completed",
            "elapsed_sec": time.time() - started_at,
            "pool_count": len(pool_dirs),
            "completed_pools": len(pool_dirs),
            "merged_examples_total": len(merged_deduped),
            "merged_positive_examples": report["summary"]["merged_positive_examples"],
            "merged_negative_examples": report["summary"]["merged_negative_examples"],
        },
    )
    logger.info(
        "Round-7 focused mining complete: merged=%d pos=%d neg=%d added=%d",
        report["summary"]["merged_examples_total"],
        report["summary"]["merged_positive_examples"],
        report["summary"]["merged_negative_examples"],
        report["summary"]["new_examples_added_to_augmented_train"],
    )


if __name__ == "__main__":
    main()
