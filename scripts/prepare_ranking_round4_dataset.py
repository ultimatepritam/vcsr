"""
Prepare a focused round-4 ranking-aligned verifier dataset from held-out failures.

This script turns the frozen round-3 held-out failure analysis into a small,
decision-aligned verifier training set. It intentionally mines only rows that
look like the next bottleneck for the project:

- oracle-positive rows where an equivalent candidate exists in-pool
- verifier-ranked selection still picks the wrong parseable candidate
- emphasis on blocksworld, especially abstract/abstract
- emphasis on near-tie and moderate-gap within-pool misrankings

The output JSONL matches the standard verifier training schema so it can be fed
into `scripts/train_verifier.py` through `data.extra_train_jsonl`.
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

from data.verifier_dataset import VerifierDatasetBuilder, VerifierExample
from scripts.mine_verifier_ranking_examples import (
    _example_key,
    _load_candidate_dump,
    _load_planetarium_rows,
    _load_run_config,
    _score,
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


def _style_label(init_is_abstract: int, goal_is_abstract: int) -> str:
    init_label = "abstract" if int(init_is_abstract) else "explicit"
    goal_label = "abstract" if int(goal_is_abstract) else "explicit"
    return f"{init_label}/{goal_label}"


def _load_failure_cases(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _select_target_cases(
    *,
    cases: list[dict],
    comparison_role: str,
    k_values: set[int],
    focus_domain: str | None,
    focus_style: str | None,
    allowed_miss_types: set[str],
    max_score_gap: float,
) -> list[dict]:
    selected = []
    for case in cases:
        if case.get("comparison_role") != comparison_role:
            continue
        if int(case.get("K", 0)) not in k_values:
            continue
        if not bool(case.get("oracle_available")):
            continue
        if bool(case.get("verifier_ranked", {}).get("equivalent")):
            continue
        if focus_domain and case.get("domain") != focus_domain:
            continue
        if focus_style and case.get("style") != focus_style:
            continue
        if case.get("verifier_miss_type") not in allowed_miss_types:
            continue

        score_gap = case.get("selected_wrong_minus_best_equivalent_score_margin")
        if score_gap is not None and float(score_gap) > max_score_gap:
            continue
        selected.append(case)

    selected.sort(
        key=lambda case: (
            int(case["K"]),
            case["domain"],
            case["style"],
            float(case.get("selected_wrong_minus_best_equivalent_score_margin") or -1.0),
            case["planetarium_name"],
        )
    )
    return selected


def _load_pool_context(
    candidate_dump_path: Path,
    *,
    run_config_cache: dict[Path, dict],
    row_lookup_cache: dict[Path, dict[str, dict]],
    candidate_dump_cache: dict[Path, tuple[dict[int, dict], dict[int, dict[int, dict]], dict[int, dict]]],
) -> tuple[dict, dict[str, dict], dict[int, dict], dict[int, dict[int, dict]], dict[int, dict]]:
    if candidate_dump_path not in candidate_dump_cache:
        run_config_path = candidate_dump_path.parent / "run_config.yaml"
        run_config = _load_run_config(run_config_path)
        split_seed = int(run_config.get("experiment", {}).get("seed", 42))
        row_lookup = _load_planetarium_rows(run_config, split_seed=split_seed)
        row_meta, candidates_by_row, selections_by_row = _load_candidate_dump(candidate_dump_path)
        run_config_cache[candidate_dump_path] = run_config
        row_lookup_cache[candidate_dump_path] = row_lookup
        candidate_dump_cache[candidate_dump_path] = (row_meta, candidates_by_row, selections_by_row)

    run_config = run_config_cache[candidate_dump_path]
    row_lookup = row_lookup_cache[candidate_dump_path]
    row_meta, candidates_by_row, selections_by_row = candidate_dump_cache[candidate_dump_path]
    return run_config, row_lookup, row_meta, candidates_by_row, selections_by_row


def _top_equivalent_candidates(positives: list[dict], limit: int) -> list[dict]:
    return sorted(
        positives,
        key=lambda rec: (_score(rec), -int(rec["candidate_index"])),
        reverse=True,
    )[: max(1, limit)]


def _hard_negative_candidates(
    *,
    negatives: list[dict],
    selected_wrong_index: int | None,
    best_positive_score: float,
    near_tie_margin: float,
    moderate_gap_margin: float,
    limit: int,
) -> list[dict]:
    chosen: list[dict] = []
    seen_indices: set[int] = set()

    def _add(rec: dict) -> None:
        idx = int(rec["candidate_index"])
        if idx in seen_indices:
            return
        seen_indices.add(idx)
        chosen.append(rec)

    if selected_wrong_index is not None:
        for rec in negatives:
            if int(rec["candidate_index"]) == selected_wrong_index:
                _add(rec)
                break

    near_ties = []
    moderate = []
    rest = []
    for rec in negatives:
        gap = best_positive_score - _score(rec)
        if gap <= near_tie_margin:
            near_ties.append(rec)
        elif gap <= moderate_gap_margin:
            moderate.append(rec)
        else:
            rest.append(rec)

    for bucket in (near_ties, moderate, rest):
        bucket.sort(key=lambda rec: (_score(rec), -int(rec["candidate_index"])), reverse=True)
        for rec in bucket:
            if len(chosen) >= max(1, limit):
                return chosen
            _add(rec)

    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare focused round-4 verifier dataset from held-out failures")
    parser.add_argument(
        "--failure_cases",
        type=str,
        default="results/vcsr/bestofk_round3_holdout_eval/failure_analysis/failure_cases.jsonl",
    )
    parser.add_argument(
        "--base_jsonl",
        type=str,
        default="results/neggen/pilot/verifier_train.relabeled.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/verifier/ranking_aligned_round4",
    )
    parser.add_argument("--comparison_role", type=str, default="heldout")
    parser.add_argument("--k_values", type=int, nargs="*", default=[4, 8])
    parser.add_argument("--focus_domain", type=str, default="blocksworld")
    parser.add_argument(
        "--focus_style",
        type=str,
        default=None,
        help="Optional exact style filter such as abstract/abstract",
    )
    parser.add_argument(
        "--miss_types",
        type=str,
        nargs="*",
        default=["near_tie_misranking", "equivalent_in_pool_but_misranked"],
    )
    parser.add_argument("--max_score_gap", type=float, default=0.10)
    parser.add_argument("--near_tie_margin", type=float, default=0.03)
    parser.add_argument("--moderate_gap_margin", type=float, default=0.10)
    parser.add_argument("--max_positives_per_case", type=int, default=2)
    parser.add_argument("--max_negatives_per_case", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_file_logging(output_dir)
    logger.info("Live mining log: %s", log_path)

    failure_cases_path = Path(args.failure_cases)
    cases = _load_failure_cases(failure_cases_path)
    k_values = {int(k) for k in args.k_values}
    allowed_miss_types = set(args.miss_types)

    target_cases = _select_target_cases(
        cases=cases,
        comparison_role=args.comparison_role,
        k_values=k_values,
        focus_domain=args.focus_domain,
        focus_style=args.focus_style,
        allowed_miss_types=allowed_miss_types,
        max_score_gap=float(args.max_score_gap),
    )
    logger.info("Selected %d target failure cases from %s", len(target_cases), failure_cases_path)

    run_config_cache: dict[Path, dict] = {}
    row_lookup_cache: dict[Path, dict[str, dict]] = {}
    candidate_dump_cache: dict[Path, tuple[dict[int, dict], dict[int, dict[int, dict]], dict[int, dict]]] = {}

    mined_examples: list[VerifierExample] = []
    case_reports: list[dict] = []
    rows_skipped_missing_source = 0
    rows_skipped_no_parseable_positive = 0

    for idx, case in enumerate(target_cases, start=1):
        candidate_dump_path = Path(case["candidate_dump"])
        row_index = int(case["row_index"])
        k_value = int(case["K"])

        logger.info(
            "Target case %d/%d: K=%d %s (%s %s)",
            idx,
            len(target_cases),
            k_value,
            case["planetarium_name"],
            case["domain"],
            case["style"],
        )

        _, row_lookup, row_meta, candidates_by_row, _ = _load_pool_context(
            candidate_dump_path,
            run_config_cache=run_config_cache,
            row_lookup_cache=row_lookup_cache,
            candidate_dump_cache=candidate_dump_cache,
        )

        meta = row_meta.get(row_index)
        if meta is None:
            logger.warning("Skipping row_index=%d because it is missing in %s", row_index, candidate_dump_path)
            continue

        planetarium_name = meta["planetarium_name"]
        source_row = row_lookup.get(planetarium_name)
        if source_row is None:
            rows_skipped_missing_source += 1
            logger.warning("Skipping %s because it was not found in the configured dataset split", planetarium_name)
            continue

        candidate_subset = [
            candidates_by_row[row_index][cand_idx]
            for cand_idx in sorted(candidates_by_row[row_index])
            if cand_idx < k_value
        ]
        parseable_candidates = [rec for rec in candidate_subset if rec.get("parseable") and rec.get("pddl")]
        positive_candidates = [rec for rec in parseable_candidates if rec.get("equivalent")]
        negative_candidates = [rec for rec in parseable_candidates if not rec.get("equivalent")]

        if not positive_candidates:
            rows_skipped_no_parseable_positive += 1
            logger.warning("Skipping %s at K=%d because no parseable positive candidate remained", planetarium_name, k_value)
            continue

        chosen_positives = _top_equivalent_candidates(
            positive_candidates,
            limit=args.max_positives_per_case,
        )
        best_positive_score = _score(chosen_positives[0])
        selected_wrong_index = case.get("verifier_ranked", {}).get("selected_index")
        selected_wrong_index = int(selected_wrong_index) if selected_wrong_index is not None else None

        chosen_negatives = _hard_negative_candidates(
            negatives=negative_candidates,
            selected_wrong_index=selected_wrong_index,
            best_positive_score=best_positive_score,
            near_tie_margin=float(args.near_tie_margin),
            moderate_gap_margin=float(args.moderate_gap_margin),
            limit=args.max_negatives_per_case,
        )

        for rec in chosen_positives:
            mined_examples.append(
                VerifierExample(
                    nl=source_row["natural_language"],
                    pddl=rec["pddl"],
                    label=1,
                    source="bestofk_round4_focus_positive",
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
                    source="bestofk_round4_focus_negative",
                    source_model=rec.get("model", ""),
                    domain=meta["domain"],
                    init_is_abstract=meta["init_is_abstract"],
                    goal_is_abstract=meta["goal_is_abstract"],
                    parseable=True,
                    planetarium_name=planetarium_name,
                )
            )

        case_reports.append(
            {
                "candidate_dump": str(candidate_dump_path),
                "row_index": row_index,
                "planetarium_name": planetarium_name,
                "domain": meta["domain"],
                "style": _style_label(meta["init_is_abstract"], meta["goal_is_abstract"]),
                "K": k_value,
                "verifier_miss_type": case["verifier_miss_type"],
                "score_gap": case.get("selected_wrong_minus_best_equivalent_score_margin"),
                "selected_wrong_index": selected_wrong_index,
                "chosen_positive_indices": [int(rec["candidate_index"]) for rec in chosen_positives],
                "chosen_negative_indices": [int(rec["candidate_index"]) for rec in chosen_negatives],
                "available_positive_indices": [int(rec["candidate_index"]) for rec in positive_candidates],
                "available_negative_indices": [int(rec["candidate_index"]) for rec in negative_candidates],
                "best_positive_score": best_positive_score,
            }
        )

    merged_deduped: list[VerifierExample] = []
    seen_keys: set[tuple[str, str, int, str]] = set()
    for example in mined_examples:
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
    by_miss_type = Counter(case["verifier_miss_type"] for case in case_reports)
    by_k = Counter(int(case["K"]) for case in case_reports)

    report = {
        "inputs": {
            "failure_cases": args.failure_cases,
            "base_jsonl": args.base_jsonl,
            "comparison_role": args.comparison_role,
            "k_values": sorted(k_values),
            "focus_domain": args.focus_domain,
            "focus_style": args.focus_style,
            "miss_types": sorted(allowed_miss_types),
            "max_score_gap": args.max_score_gap,
            "near_tie_margin": args.near_tie_margin,
            "moderate_gap_margin": args.moderate_gap_margin,
            "max_positives_per_case": args.max_positives_per_case,
            "max_negatives_per_case": args.max_negatives_per_case,
        },
        "summary": {
            "target_cases_selected": len(target_cases),
            "target_cases_mined": len(case_reports),
            "unique_target_rows": len({(c["candidate_dump"], c["row_index"]) for c in case_reports}),
            "raw_examples_before_dedup": len(mined_examples),
            "merged_examples_total": len(merged_deduped),
            "merged_positive_examples": sum(1 for ex in merged_deduped if ex.label == 1),
            "merged_negative_examples": sum(1 for ex in merged_deduped if ex.label == 0),
            "new_examples_added_to_augmented_train": len(new_examples),
            "augmented_train_total": len(base_builder),
            "rows_skipped_missing_source": rows_skipped_missing_source,
            "rows_skipped_no_parseable_positive": rows_skipped_no_parseable_positive,
        },
        "by_source": dict(by_source.most_common()),
        "by_domain": dict(by_domain.most_common()),
        "by_style": dict(by_style.most_common()),
        "by_miss_type": dict(by_miss_type.most_common()),
        "by_k": dict(sorted(by_k.items())),
        "target_cases": case_reports,
    }

    with open(output_dir / "mining_manifest.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(output_dir / "mining_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Focused round-4 mining complete: target_cases=%d mined=%d added=%d",
        len(case_reports),
        len(merged_deduped),
        len(new_examples),
    )


if __name__ == "__main__":
    main()
