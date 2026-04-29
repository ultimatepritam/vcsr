"""
Mine ranking-aligned verifier training examples from a cached best-of-K pool.

This script converts real candidate pools into extra verifier supervision that is
closer to the downstream decision we care about:

- positives are parseable equivalent candidates from the same NL pool
- negatives are parseable non-equivalent candidates that were scored highly,
  selected wrongly, or outranked the best positive

The output JSONL matches the standard verifier training schema so it can be fed
into `scripts/train_verifier.py` through `data.extra_train_jsonl`.

Usage:
    python scripts/mine_verifier_ranking_examples.py
    python scripts/mine_verifier_ranking_examples.py --k 8 --max_positives_per_row 2 --max_negatives_per_row 4
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
    selections_by_row: dict[int, dict[int, dict]] = defaultdict(dict)

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
                selections_by_row[row_index].setdefault(k_value, {})[policy] = record

    logger.info(
        "Loaded candidate dump: %d rows, %d candidate records",
        len(row_meta),
        sum(len(v) for v in candidates_by_row.values()),
    )
    return row_meta, candidates_by_row, selections_by_row


def _score(rec: dict) -> float:
    score = rec.get("verifier_score")
    return float(score if score is not None else -1.0)


def _positive_sort_key(rec: dict) -> tuple[float, int]:
    return (_score(rec), -int(rec["candidate_index"]))


def _negative_priority(
    rec: dict,
    *,
    selected_negative_index: int | None,
    best_positive_score: float | None,
) -> tuple[int, int, float, int]:
    idx = int(rec["candidate_index"])
    was_selected = 1 if selected_negative_index is not None and idx == selected_negative_index else 0
    outranks_positive = (
        1
        if best_positive_score is not None and _score(rec) >= best_positive_score
        else 0
    )
    return (was_selected, outranks_positive, _score(rec), -idx)


def _example_key(example: VerifierExample) -> tuple[str, str, int, str]:
    return (example.nl, example.pddl, int(example.label), example.planetarium_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine ranking-aligned verifier examples from best-of-K pools")
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
        default="results/verifier/ranking_aligned_round1",
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max_positives_per_row", type=int, default=2)
    parser.add_argument("--max_negatives_per_row", type=int, default=4)
    parser.add_argument(
        "--include_negative_only_rows",
        action="store_true",
        help="Also add top-scoring parseable negatives from rows that have no equivalent candidate in-pool.",
    )
    parser.add_argument(
        "--max_negative_only_per_row",
        type=int,
        default=1,
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

        selection_block = selections_by_row.get(row_index, {}).get(args.k, {})
        verifier_selection = selection_block.get("verifier_ranked", {})
        selected_index = verifier_selection.get("selected_index")
        selected_negative_index = int(selected_index) if selected_index is not None else None

        candidate_subset = [
            candidates_by_row[row_index][cand_idx]
            for cand_idx in sorted(candidates_by_row[row_index])
            if cand_idx < args.k
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
            chosen_positives = sorted_positives[: max(1, args.max_positives_per_row)]
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
            chosen_negatives = ranked_negatives[: max(1, args.max_negatives_per_row)]

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
                    "k": args.k,
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

        if not args.include_negative_only_rows or not negative_candidates:
            continue

        rows_with_negative_only_mining += 1
        ranked_negatives = sorted(negative_candidates, key=_positive_sort_key, reverse=True)
        chosen_negatives = ranked_negatives[: max(1, args.max_negative_only_per_row)]
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
                "k": args.k,
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

    mined_builder = VerifierDatasetBuilder()
    mined_builder.add_examples(mined_deduped)
    mined_builder.save_jsonl(output_dir / "mined_examples.jsonl")
    mined_builder.save_stats(output_dir / "mined_examples_stats.json")

    base_builder = VerifierDatasetBuilder.load_jsonl(base_jsonl_path)
    base_keys = {_example_key(example) for example in base_builder.get_parseable_examples()}
    new_examples = [example for example in mined_deduped if _example_key(example) not in base_keys]
    base_builder.add_examples(new_examples)
    base_builder.save_jsonl(output_dir / "augmented_train.jsonl")
    base_builder.save_stats(output_dir / "augmented_train_stats.json")

    by_source = Counter(example.source for example in mined_deduped)
    by_domain = Counter(example.domain for example in mined_deduped)
    by_style = Counter(_style_label(example.init_is_abstract, example.goal_is_abstract) for example in mined_deduped)

    report = {
        "inputs": {
            "candidate_dump": str(candidate_dump_path),
            "run_config": str(run_config_path),
            "base_jsonl": str(base_jsonl_path),
            "k": args.k,
            "max_positives_per_row": args.max_positives_per_row,
            "max_negatives_per_row": args.max_negatives_per_row,
            "include_negative_only_rows": args.include_negative_only_rows,
            "max_negative_only_per_row": args.max_negative_only_per_row,
        },
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
            "new_examples_added_to_augmented_train": len(new_examples),
            "augmented_train_total": len(base_builder),
        },
        "by_source": dict(by_source.most_common()),
        "by_domain": dict(by_domain.most_common()),
        "by_style": dict(by_style.most_common()),
        "row_details": detail_rows,
    }

    with open(output_dir / "mining_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved ranking-aligned mined examples to %s", output_dir / "mined_examples.jsonl")
    logger.info(
        "Ranking-aligned mining complete: rows_with_positive_pool=%d miss_rows=%d mined=%d added=%d",
        rows_with_positive_pool,
        rows_with_verifier_miss,
        len(mined_deduped),
        len(new_examples),
    )


if __name__ == "__main__":
    main()
