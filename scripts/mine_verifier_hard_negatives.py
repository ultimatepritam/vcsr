"""
Mine verifier hard negatives from a best-of-K pilot candidate dump.

This script identifies rows where the verifier-ranked policy selected a
parseable non-equivalent candidate even though an equivalent parseable
candidate was available in the same pool. It then writes:

1. A mined JSONL of verifier training examples in the standard schema
2. A merged training JSONL that appends those examples to a base verifier set
3. A mining report with row-level diagnostics

Usage:
    python scripts/mine_verifier_hard_negatives.py
    python scripts/mine_verifier_hard_negatives.py --k 8 --max_negatives_per_row 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vcsr_env  # noqa: F401
import yaml

from data.planetarium_loader import PlanetariumDataset
from data.verifier_dataset import VerifierDatasetBuilder, VerifierExample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _style_label(init_is_abstract: int, goal_is_abstract: int) -> str:
    init_label = "abstract" if init_is_abstract else "explicit"
    goal_label = "abstract" if goal_is_abstract else "explicit"
    return f"{init_label}/{goal_label}"


def _load_run_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_planetarium_rows(run_config: dict, split_seed: int) -> dict[str, dict]:
    ds_cfg = run_config.get("dataset", {})
    dataset = PlanetariumDataset(
        split_strategy=ds_cfg.get("split_strategy", "template_hash"),
        seed=split_seed,
    )
    split_name = ds_cfg.get("split", "test")
    rows = dataset.get_split(split_name)
    row_map: dict[str, dict] = {}
    for row in rows:
        row_map[row["name"]] = row
    logger.info("Loaded %d Planetarium rows from split=%s", len(row_map), split_name)
    return row_map


def _load_candidate_dump(path: Path) -> tuple[dict[int, dict], dict[int, dict[int, dict]], dict[int, dict]]:
    row_meta: dict[int, dict] = {}
    candidates_by_row: dict[int, dict[int, dict]] = defaultdict(dict)
    selections_by_row: dict[int, dict[int, dict]] = defaultdict(lambda: defaultdict(dict))

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            row_index = int(record["row_index"])
            row_meta[row_index] = {
                "planetarium_name": record["planetarium_name"],
                "domain": record["domain"],
                "init_is_abstract": int(record.get("init_is_abstract", 0)),
                "goal_is_abstract": int(record.get("goal_is_abstract", 0)),
            }
            if "candidate_index" in record:
                candidates_by_row[row_index][int(record["candidate_index"])] = record
            else:
                k_value = int(record["K"])
                policy = record["policy"]
                selections_by_row[row_index][k_value][policy] = record

    logger.info(
        "Loaded candidate dump: %d row groups, %d candidate records",
        len(row_meta),
        sum(len(v) for v in candidates_by_row.values()),
    )
    return row_meta, candidates_by_row, selections_by_row


def _pick_best_positive(positives: list[dict]) -> dict:
    return max(
        positives,
        key=lambda rec: (
            float(rec.get("verifier_score") if rec.get("verifier_score") is not None else -1.0),
            -int(rec["candidate_index"]),
        ),
    )


def _candidate_sort_key(rec: dict) -> tuple[float, int]:
    score = rec.get("verifier_score")
    return (float(score if score is not None else -1.0), -int(rec["candidate_index"]))


def _example_key(example: VerifierExample) -> tuple[str, str, int, str]:
    return (example.nl, example.pddl, int(example.label), example.planetarium_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine verifier hard negatives from best-of-K outputs")
    parser.add_argument(
        "--candidate_dump",
        type=str,
        default="results/vcsr/bestofk_pilot/candidate_dump.jsonl",
    )
    parser.add_argument(
        "--run_config",
        type=str,
        default="results/vcsr/bestofk_pilot/run_config.yaml",
    )
    parser.add_argument(
        "--base_jsonl",
        type=str,
        default="results/neggen/pilot/verifier_train.relabeled.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/verifier/hardneg_round1",
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max_negatives_per_row", type=int, default=3)
    parser.add_argument(
        "--include_selected_negative_only",
        action="store_true",
        help="Only add the verifier-selected wrong candidate per miss row.",
    )
    args = parser.parse_args()

    candidate_dump_path = Path(args.candidate_dump)
    run_config_path = Path(args.run_config)
    base_jsonl_path = Path(args.base_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = _load_run_config(run_config_path)
    split_seed = int(run_config.get("experiment", {}).get("seed", 42))
    row_lookup = _load_planetarium_rows(run_config, split_seed=split_seed)
    row_meta, candidates_by_row, selections_by_row = _load_candidate_dump(candidate_dump_path)

    mined_examples: list[VerifierExample] = []
    detail_rows: list[dict] = []
    rows_considered = 0
    rows_with_positive_pool = 0
    verifier_miss_rows = 0

    for row_index in sorted(row_meta):
        rows_considered += 1
        meta = row_meta[row_index]
        planetarium_name = meta["planetarium_name"]
        source_row = row_lookup.get(planetarium_name)
        if source_row is None:
            logger.warning("Skipping %s because it was not found in the configured dataset split", planetarium_name)
            continue

        selection_block = selections_by_row.get(row_index, {}).get(args.k, {})
        verifier_selection = selection_block.get("verifier_ranked")
        if verifier_selection is None:
            continue

        candidate_subset = [
            candidates_by_row[row_index][cand_idx]
            for cand_idx in sorted(candidates_by_row[row_index])
            if cand_idx < args.k
        ]
        parseable_candidates = [rec for rec in candidate_subset if rec.get("parseable")]
        positive_candidates = [rec for rec in parseable_candidates if rec.get("equivalent")]
        if not positive_candidates:
            continue

        rows_with_positive_pool += 1
        selected_index = verifier_selection.get("selected_index")
        selected_candidate = (
            candidates_by_row[row_index].get(int(selected_index))
            if selected_index is not None
            else None
        )
        if selected_candidate is None or not selected_candidate.get("parseable") or selected_candidate.get("equivalent"):
            continue

        verifier_miss_rows += 1
        best_positive = _pick_best_positive(positive_candidates)
        best_positive_score = float(
            best_positive.get("verifier_score") if best_positive.get("verifier_score") is not None else -1.0
        )

        negative_candidates = [selected_candidate]
        if not args.include_selected_negative_only:
            outranking_negatives = [
                rec
                for rec in parseable_candidates
                if not rec.get("equivalent")
                and rec.get("candidate_index") != selected_candidate.get("candidate_index")
                and float(rec.get("verifier_score") if rec.get("verifier_score") is not None else -1.0) >= best_positive_score
            ]
            outranking_negatives.sort(key=_candidate_sort_key, reverse=True)
            negative_candidates.extend(outranking_negatives[: max(0, args.max_negatives_per_row - 1)])

        deduped_negatives: list[dict] = []
        seen_negative_indices: set[int] = set()
        for rec in negative_candidates:
            cand_idx = int(rec["candidate_index"])
            if cand_idx in seen_negative_indices:
                continue
            seen_negative_indices.add(cand_idx)
            deduped_negatives.append(rec)

        style = _style_label(meta["init_is_abstract"], meta["goal_is_abstract"])

        positive_example = VerifierExample(
            nl=source_row["natural_language"],
            pddl=best_positive["pddl"],
            label=1,
            source="bestofk_mined_positive",
            source_model=best_positive.get("model", ""),
            domain=meta["domain"],
            init_is_abstract=meta["init_is_abstract"],
            goal_is_abstract=meta["goal_is_abstract"],
            parseable=True,
            planetarium_name=planetarium_name,
        )
        mined_examples.append(positive_example)

        negative_examples: list[VerifierExample] = []
        for rec in deduped_negatives:
            negative_examples.append(
                VerifierExample(
                    nl=source_row["natural_language"],
                    pddl=rec["pddl"],
                    label=0,
                    source="bestofk_mined_negative",
                    source_model=rec.get("model", ""),
                    domain=meta["domain"],
                    init_is_abstract=meta["init_is_abstract"],
                    goal_is_abstract=meta["goal_is_abstract"],
                    parseable=True,
                    planetarium_name=planetarium_name,
                )
            )
        mined_examples.extend(negative_examples)

        detail_rows.append(
            {
                "row_index": row_index,
                "planetarium_name": planetarium_name,
                "domain": meta["domain"],
                "style": style,
                "k": args.k,
                "selected_negative": {
                    "candidate_index": int(selected_candidate["candidate_index"]),
                    "verifier_score": selected_candidate.get("verifier_score"),
                    "equivalent": bool(selected_candidate.get("equivalent")),
                },
                "positive_anchor": {
                    "candidate_index": int(best_positive["candidate_index"]),
                    "verifier_score": best_positive.get("verifier_score"),
                    "equivalent": bool(best_positive.get("equivalent")),
                },
                "mined_negative_indices": [int(rec["candidate_index"]) for rec in deduped_negatives],
                "available_positive_indices": [int(rec["candidate_index"]) for rec in positive_candidates],
            }
        )

    mined_deduped: list[VerifierExample] = []
    seen_example_keys: set[tuple[str, str, int, str]] = set()
    for example in mined_examples:
        key = _example_key(example)
        if key in seen_example_keys:
            continue
        seen_example_keys.add(key)
        mined_deduped.append(example)

    mined_builder = VerifierDatasetBuilder()
    mined_builder.add_examples(mined_deduped)
    mined_builder.save_jsonl(output_dir / "mined_examples.jsonl")
    mined_builder.save_stats(output_dir / "mined_examples_stats.json")

    mined_counts_by_domain: Counter[str] = Counter()
    mined_counts_by_style: Counter[str] = Counter()
    for example in mined_deduped:
        if int(example.label) != 0:
            continue
        mined_counts_by_domain[example.domain] += 1
        mined_counts_by_style[_style_label(example.init_is_abstract, example.goal_is_abstract)] += 1

    base_builder = VerifierDatasetBuilder.load_jsonl(base_jsonl_path)
    base_keys = {_example_key(example) for example in base_builder.get_parseable_examples()}
    new_examples = [example for example in mined_deduped if _example_key(example) not in base_keys]
    base_builder.add_examples(new_examples)
    base_builder.save_jsonl(output_dir / "augmented_train.jsonl")
    base_builder.save_stats(output_dir / "augmented_train_stats.json")

    report = {
        "inputs": {
            "candidate_dump": str(candidate_dump_path),
            "run_config": str(run_config_path),
            "base_jsonl": str(base_jsonl_path),
            "k": args.k,
            "max_negatives_per_row": args.max_negatives_per_row,
            "include_selected_negative_only": args.include_selected_negative_only,
        },
        "summary": {
            "rows_considered": rows_considered,
            "rows_with_equivalent_candidate_pool": rows_with_positive_pool,
            "verifier_miss_rows": verifier_miss_rows,
            "mined_examples_total": len(mined_deduped),
            "mined_positive_examples": sum(1 for ex in mined_deduped if ex.label == 1),
            "mined_negative_examples": sum(1 for ex in mined_deduped if ex.label == 0),
            "new_examples_added_to_augmented_train": len(new_examples),
            "augmented_train_total": len(base_builder),
        },
        "by_domain_negative_examples": dict(mined_counts_by_domain.most_common()),
        "by_style_negative_examples": dict(mined_counts_by_style.most_common()),
        "row_details": detail_rows,
    }
    with open(output_dir / "mining_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved mined examples to %s", output_dir / "mined_examples.jsonl")
    logger.info("Saved augmented training set to %s", output_dir / "augmented_train.jsonl")
    logger.info(
        "Hard-negative mining complete: miss_rows=%d mined=%d added=%d",
        verifier_miss_rows,
        len(mined_deduped),
        len(new_examples),
    )


if __name__ == "__main__":
    main()
