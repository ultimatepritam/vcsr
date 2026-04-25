"""
Run a small cached-failure repair pilot with the promoted round-4 verifier.

This script does not train a verifier and does not generate new best-of-K pools.
It selects cached round-4 verifier-ranked failures, asks the configured LLM to
repair the selected PDDL once, and evaluates the repaired candidate.
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
from planetarium.builder import build

from data.planetarium_loader import PlanetariumDataset
from eval.equivalence import EvalResult, check_equivalence_lightweight, check_equivalence_lightweight_timed
from generation.prompts import SYSTEM_PROMPT, extract_pddl_from_response, make_repair_prompt
from generation.sampler import MultiSampler, SampleResult, SamplerConfig
from pddl_utils.oracle_planner import check_solvability_oracle
from verifier.inference import VerifierScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@dataclass
class RepairCase:
    pool: str
    row_index: int
    planetarium_name: str
    domain: str
    style: str
    natural_language: str
    gold_pddl: str
    is_placeholder: bool
    selected_index: int
    selected_pddl: str
    selected_score: float
    selected_parseable: bool
    selected_solvable: bool
    selected_equivalent: bool
    selected_planner_error: str | None


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


def _clear_proxy_env() -> list[str]:
    """Keep hosted LLM calls off stale local proxy settings."""
    cleared: list[str] = []
    for key in PROXY_ENV_KEYS:
        if key in os.environ:
            cleared.append(key)
            os.environ.pop(key, None)
    if cleared:
        logger.info("Cleared proxy environment variables for repair generation: %s", ", ".join(cleared))
    return cleared


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_progress(
    output_dir: Path,
    *,
    stage: str,
    total_rows: int,
    completed_rows: int,
    started_at: float,
    current_case: str = "",
) -> None:
    elapsed_sec = max(0.0, time.time() - started_at)
    avg_sec_per_row = elapsed_sec / max(1, completed_rows) if completed_rows else None
    eta_sec = avg_sec_per_row * max(0, total_rows - completed_rows) if avg_sec_per_row is not None else None
    _write_json(
        output_dir / "progress.json",
        {
            "stage": stage,
            "total_rows": total_rows,
            "completed_rows": completed_rows,
            "remaining_rows": max(0, total_rows - completed_rows),
            "elapsed_sec": elapsed_sec,
            "avg_sec_per_row": avg_sec_per_row,
            "eta_sec": eta_sec,
            "current_case": current_case,
        },
    )


def _style(record: dict[str, Any]) -> str:
    init = "abstract" if int(record.get("init_is_abstract", 0)) else "explicit"
    goal = "abstract" if int(record.get("goal_is_abstract", 0)) else "explicit"
    return f"{init}/{goal}"


def _pool_name(candidate_dump: Path) -> str:
    parent = candidate_dump.parent
    if parent.name.startswith("seed_"):
        return f"{parent.parent.parent.name}/{parent.name}"
    return parent.name


def _try_parse_pddl(pddl: str) -> bool:
    if not pddl or not pddl.strip():
        return False
    try:
        build(pddl)
        return True
    except Exception:
        return False


def _load_run_config(candidate_dump: Path) -> dict[str, Any]:
    with open(candidate_dump.parent / "run_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dataset_rows(candidate_dump: Path) -> dict[str, dict[str, Any]]:
    cfg = _load_run_config(candidate_dump)
    ds_cfg = cfg.get("dataset", {})
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    dataset = PlanetariumDataset(
        split_strategy=ds_cfg.get("split_strategy", "template_hash"),
        seed=seed,
    )
    rows = dataset.get_split(ds_cfg.get("split", "test"))
    return {row["name"]: row for row in rows}


def _load_candidate_records(candidate_dump: Path) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
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


def _select_top_round4_failures(
    *,
    candidate_dump: Path,
    scorer: VerifierScorer,
    k: int,
    scoring_batch_size: int,
) -> list[RepairCase]:
    pool = _pool_name(candidate_dump)
    rows_by_name = _load_dataset_rows(candidate_dump)
    meta_by_row, candidates_by_row = _load_candidate_records(candidate_dump)
    cases: list[RepairCase] = []

    for row_index, candidates in candidates_by_row.items():
        subset = candidates[:k]
        meta = meta_by_row[row_index]
        dataset_row = rows_by_name[meta["planetarium_name"]]
        nl = dataset_row["natural_language"]
        pairs = [
            (nl, rec.get("pddl") or "")
            for rec in subset
            if bool(rec.get("parseable")) and rec.get("pddl")
        ]
        scores = scorer.score_pairs(pairs, batch_size=scoring_batch_size) if pairs else []
        score_iter = iter(scores)
        scored_candidates: list[dict[str, Any]] = []
        for rec in subset:
            new_rec = dict(rec)
            new_rec["round4_score"] = next(score_iter) if rec.get("parseable") and rec.get("pddl") else None
            scored_candidates.append(new_rec)

        selected = select_repair_candidate_from_scored(scored_candidates)
        if selected is None:
            continue

        plan_result = check_solvability_oracle(selected.get("pddl") or "")
        cases.append(
            RepairCase(
                pool=pool,
                row_index=row_index,
                planetarium_name=meta["planetarium_name"],
                domain=meta["domain"],
                style=meta["style"],
                natural_language=nl,
                gold_pddl=dataset_row["problem_pddl"],
                is_placeholder=bool(dataset_row.get("is_placeholder", 0)),
                selected_index=int(selected["candidate_index"]),
                selected_pddl=selected.get("pddl") or "",
                selected_score=float(selected["round4_score"]) if selected.get("round4_score") is not None else 0.0,
                selected_parseable=bool(selected.get("parseable")),
                selected_solvable=bool(plan_result.solvable),
                selected_equivalent=bool(selected.get("equivalent")),
                selected_planner_error=plan_result.error,
            )
        )
    return cases


def select_repair_candidate_from_scored(scored_candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        rec
        for rec in scored_candidates
        if rec.get("parseable") and rec.get("pddl")
    ]
    if not eligible:
        return None
    selected = max(
        eligible,
        key=lambda rec: (
            float("-inf") if rec.get("round4_score") is None else float(rec["round4_score"]),
            -int(rec["candidate_index"]),
        ),
    )
    if bool(selected.get("equivalent")):
        return None
    return selected


def _case_priority(case: RepairCase) -> tuple[int, int, float, str, int]:
    return (
        0 if case.domain == "blocksworld" else 1,
        0 if case.style == "abstract/abstract" else 1,
        -case.selected_score,
        case.pool,
        case.row_index,
    )


def select_repair_cases(
    *,
    candidate_pools: list[str],
    scorer: VerifierScorer,
    k: int,
    max_rows: int,
    seed: int,
    scoring_batch_size: int,
) -> list[RepairCase]:
    cases: list[RepairCase] = []
    for pool in candidate_pools:
        cases.extend(
            _select_top_round4_failures(
                candidate_dump=Path(pool),
                scorer=scorer,
                k=k,
                scoring_batch_size=scoring_batch_size,
            )
        )
    rng = random.Random(seed)
    rng.shuffle(cases)
    cases.sort(key=_case_priority)
    return cases[:max_rows]


def build_feedback(case: RepairCase) -> str:
    solvability = "solvable" if case.selected_solvable else "not confirmed solvable"
    return (
        "The candidate PDDL parses successfully. "
        f"The current verifier score for this candidate is {case.selected_score:.4f}. "
        f"A lightweight planner check says the candidate is {solvability}. "
        "This candidate was selected by the current verifier, but it may still fail to match the natural-language task. "
        "Repair the problem definition so the objects, initial state, and goal match the task description exactly."
    )


def _sample_repair(sampler: MultiSampler, prompt: str) -> SampleResult:
    backend, _ = sampler.backends[0]
    start = time.time()
    try:
        raw = backend._call_llm(prompt, system=SYSTEM_PROMPT)
        return SampleResult(
            raw_response=raw,
            extracted_pddl=extract_pddl_from_response(raw),
            backend=backend.backend_name,
            model=backend.model,
            latency_sec=time.time() - start,
        )
    except Exception as exc:
        return SampleResult(
            raw_response="",
            extracted_pddl="",
            backend=backend.backend_name,
            model=backend.model,
            latency_sec=time.time() - start,
            error=str(exc),
        )


def _evaluate_repair(
    *,
    case: RepairCase,
    repaired_pddl: str,
    timeout_sec: float,
) -> EvalResult:
    if timeout_sec and timeout_sec > 0:
        return check_equivalence_lightweight_timed(
            case.gold_pddl,
            repaired_pddl,
            is_placeholder=case.is_placeholder,
            timeout_sec=timeout_sec,
        )
    return check_equivalence_lightweight(
        case.gold_pddl,
        repaired_pddl,
        is_placeholder=case.is_placeholder,
    )


def _outcome(original_equiv: bool, repair_equiv: bool) -> str:
    if repair_equiv and not original_equiv:
        return "repair_helped"
    if original_equiv and not repair_equiv:
        return "repair_hurt"
    if repair_equiv and original_equiv:
        return "both_success"
    return "both_fail"


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    original_equiv = sum(1 for row in rows if row["original_selected_equivalent"])
    repair_parse = sum(1 for row in rows if row["repair_parseable"])
    repair_solvable = sum(1 for row in rows if row["repair_solvable"])
    repair_equiv = sum(1 for row in rows if row["repair_equivalent"])
    helped = sum(1 for row in rows if row["outcome"] == "repair_helped")
    hurt = sum(1 for row in rows if row["outcome"] == "repair_hurt")
    return {
        "total": total,
        "original_equiv_count": original_equiv,
        "repair_parse_count": repair_parse,
        "repair_solvable_count": repair_solvable,
        "repair_equiv_count": repair_equiv,
        "original_equiv_rate": original_equiv / total if total else 0.0,
        "repair_parse_rate": repair_parse / total if total else 0.0,
        "repair_solvable_rate": repair_solvable / total if total else 0.0,
        "repair_equiv_rate": repair_equiv / total if total else 0.0,
        "repair_equiv_given_parse": repair_equiv / repair_parse if repair_parse else 0.0,
        "helped": helped,
        "hurt": hurt,
        "tied": total - helped - hurt,
    }


def _breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {name: _metrics(group_rows) for name, group_rows in sorted(grouped.items())}


def _failure_type(row: dict[str, Any]) -> str:
    if not row["original_selected_solvable"]:
        return "unsolvable_selected_candidate"
    if row["original_selected_score"] >= 0.8:
        return "high_verifier_score_wrong_candidate"
    if row["original_selected_score"] < 0.4:
        return "low_verifier_score_wrong_candidate"
    return "solvable_non_equivalent_selected_candidate"


def _accepted(summary: dict[str, Any]) -> bool:
    metrics = summary["metrics"]
    if metrics["repair_equiv_rate"] - metrics["original_equiv_rate"] < 0.10:
        return False
    if metrics["helped"] <= metrics["hurt"]:
        return False
    if metrics["repair_parse_rate"] < 0.80:
        return False
    successful_families = {
        row["planetarium_name"].split("_")[1] if "_" in row["planetarium_name"] else row["planetarium_name"]
        for row in summary["rows"]
        if row["outcome"] == "repair_helped"
    }
    return metrics["helped"] >= 2 and len(successful_families) > 1


def _markdown(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    lines = [
        "# Round 4 Repair Pilot",
        "",
        f"Rows repaired: `{metrics['total']}`",
        "",
        "## Top Line",
        "",
        "| Original Eq | Repair Parse | Repair Eq | Repair Eq / Parse | Helped | Hurt | Tied | Accepted |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {metrics['original_equiv_rate']:.4f} | {metrics['repair_parse_rate']:.4f} | "
            f"{metrics['repair_equiv_rate']:.4f} | {metrics['repair_equiv_given_parse']:.4f} | "
            f"{metrics['helped']} | {metrics['hurt']} | {metrics['tied']} | {summary['accepted']} |"
        ),
        "",
        "## Domain Breakdown",
        "",
        "| Domain | Rows | Repair Eq | Helped | Hurt |",
        "|---|---:|---:|---:|---:|",
    ]
    for domain, data in summary["domain_breakdown"].items():
        lines.append(f"| {domain} | {data['total']} | {data['repair_equiv_rate']:.4f} | {data['helped']} | {data['hurt']} |")
    lines.extend(["", "## Style Breakdown", "", "| Style | Rows | Repair Eq | Helped | Hurt |", "|---|---:|---:|---:|---:|"])
    for style, data in summary["style_breakdown"].items():
        lines.append(f"| {style} | {data['total']} | {data['repair_equiv_rate']:.4f} | {data['helped']} | {data['hurt']} |")
    lines.extend(["", "## Failure-Type Breakdown", "", "| Failure Type | Rows | Repair Eq | Helped | Hurt |", "|---|---:|---:|---:|---:|"])
    for failure_type, data in summary["failure_type_breakdown"].items():
        lines.append(f"| {failure_type} | {data['total']} | {data['repair_equiv_rate']:.4f} | {data['helped']} | {data['hurt']} |")
    lines.extend(["", "## Recommendation", "", summary["recommendation"], ""])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.max_rows is not None:
        cfg.setdefault("selection", {})["max_rows"] = int(args.max_rows)
    if args.output_dir:
        cfg.setdefault("output", {})["dir"] = args.output_dir

    output_dir = Path(cfg.get("output", {}).get("dir", "results/vcsr/repair_pilot_round4"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_file_logging(output_dir)
    started_at = time.time()
    with open(output_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    _write_json(
        output_dir / "process_info.json",
        {
            "pid": os.getpid(),
            "started_at": started_at,
            "config": args.config,
            "output_dir": str(output_dir),
        },
    )

    selection_cfg = cfg.get("selection", {})
    generation_cfg = cfg.get("generation", {})
    eval_cfg = cfg.get("evaluation", {})
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    random.seed(seed)

    if bool(generation_cfg.get("clear_proxy_env", cfg.get("clear_proxy_env", True))):
        _clear_proxy_env()

    scorer = VerifierScorer(selection_path=selection_cfg["verifier_selection"])
    cases = select_repair_cases(
        candidate_pools=cfg["inputs"]["candidate_pools"],
        scorer=scorer,
        k=int(selection_cfg.get("K", 8)),
        max_rows=int(selection_cfg.get("max_rows", 30)),
        seed=seed,
        scoring_batch_size=int(selection_cfg.get("scoring_batch_size", 8)),
    )
    with open(output_dir / "repair_cases.jsonl", "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps({k: v for k, v in case.__dict__.items() if k != "gold_pddl"}) + "\n")

    sampler = MultiSampler(
        backend_specs=generation_cfg.get("backends", [{"type": "openrouter", "K": 1}]),
        config=SamplerConfig(
            temperature=float(generation_cfg.get("temperature", 0.4)),
            top_p=float(generation_cfg.get("top_p", 0.9)),
            max_new_tokens=int(generation_cfg.get("max_new_tokens", 1024)),
            retry_attempts=int(generation_cfg.get("retry_attempts", 3)),
            retry_delay_sec=float(generation_cfg.get("retry_delay_sec", 2)),
        ),
    )

    rows: list[dict[str, Any]] = []
    output_path = output_dir / "repair_outputs.jsonl"
    _write_progress(output_dir, stage="repairing", total_rows=len(cases), completed_rows=0, started_at=started_at)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, case in enumerate(cases):
            logger.info("Repair case %d/%d: %s", idx + 1, len(cases), case.planetarium_name)
            _write_progress(
                output_dir,
                stage="repairing",
                total_rows=len(cases),
                completed_rows=idx,
                started_at=started_at,
                current_case=case.planetarium_name,
            )
            feedback = build_feedback(case)
            prompt = make_repair_prompt(
                natural_language=case.natural_language,
                candidate_pddl=case.selected_pddl,
                domain=case.domain,
                feedback=feedback,
            )
            sample = _sample_repair(sampler, prompt)
            repair_parseable = _try_parse_pddl(sample.extracted_pddl)
            repair_eval = _evaluate_repair(
                case=case,
                repaired_pddl=sample.extracted_pddl,
                timeout_sec=float(eval_cfg.get("equivalence_timeout_sec", 0)),
            ) if repair_parseable else EvalResult(parseable=False, equivalent=False, error=sample.error or "parse_failed")
            repair_plan = check_solvability_oracle(sample.extracted_pddl) if repair_parseable else None
            repair_score = (
                scorer.score_pair(case.natural_language, sample.extracted_pddl)
                if repair_parseable and sample.extracted_pddl
                else None
            )
            row = {
                "pool": case.pool,
                "row_index": case.row_index,
                "planetarium_name": case.planetarium_name,
                "domain": case.domain,
                "style": case.style,
                "natural_language": case.natural_language,
                "original_selected_index": case.selected_index,
                "original_selected_pddl": case.selected_pddl,
                "original_selected_score": case.selected_score,
                "original_selected_parseable": case.selected_parseable,
                "original_selected_solvable": case.selected_solvable,
                "original_selected_equivalent": case.selected_equivalent,
                "original_selected_planner_error": case.selected_planner_error,
                "feedback": feedback,
                "repair_raw_response": sample.raw_response,
                "repair_pddl": sample.extracted_pddl,
                "repair_parseable": bool(repair_eval.parseable),
                "repair_solvable": bool(repair_plan.solvable) if repair_plan else False,
                "repair_planner_error": repair_plan.error if repair_plan else None,
                "repair_equivalent": bool(repair_eval.equivalent),
                "repair_error": repair_eval.error or sample.error,
                "repair_verifier_score": repair_score,
                "outcome": _outcome(case.selected_equivalent, bool(repair_eval.equivalent)),
                "latency_sec": sample.latency_sec,
                "backend": sample.backend,
                "model": sample.model,
            }
            row["failure_type"] = _failure_type(row)
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            _flush_logs()
            _write_progress(
                output_dir,
                stage="repairing",
                total_rows=len(cases),
                completed_rows=idx + 1,
                started_at=started_at,
                current_case=case.planetarium_name,
            )

    summary = {
        "config": cfg,
        "metrics": _metrics(rows),
        "domain_breakdown": _breakdown(rows, "domain"),
        "style_breakdown": _breakdown(rows, "style"),
        "failure_type_breakdown": _breakdown(rows, "failure_type"),
        "outcome_counts": Counter(row["outcome"] for row in rows).most_common(),
        "rows": [
            {
                "pool": row["pool"],
                "planetarium_name": row["planetarium_name"],
                "domain": row["domain"],
                "style": row["style"],
                "outcome": row["outcome"],
                "repair_parseable": row["repair_parseable"],
                "repair_equivalent": row["repair_equivalent"],
                "repair_verifier_score": row["repair_verifier_score"],
                "failure_type": row["failure_type"],
            }
            for row in rows
        ],
    }
    summary["accepted"] = _accepted(summary)
    summary["recommendation"] = (
        "Repair is promising; run a fresh fixed-pool repair gate next."
        if summary["accepted"]
        else "Repair did not pass the pilot gate; improve feedback or prompt before scaling."
    )
    _write_json(output_dir / "repair_summary.json", summary)
    with open(output_dir / "repair_summary.md", "w", encoding="utf-8") as f:
        f.write(_markdown(summary))
    _write_progress(output_dir, stage="complete", total_rows=len(cases), completed_rows=len(cases), started_at=started_at)
    logger.info("Saved repair summary to %s", output_dir / "repair_summary.md")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vcsr_repair_pilot.yaml")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
