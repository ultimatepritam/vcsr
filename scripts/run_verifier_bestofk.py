"""
Run the first verifier-ranked best-of-K pilot experiment.
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
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vcsr_env  # noqa: F401
import yaml
from planetarium.builder import build

from data.planetarium_loader import PlanetariumDataset
from eval.equivalence import BatchMetrics, EvalResult, check_equivalence_lightweight, stratified_report
from generation.prompts import SYSTEM_PROMPT, extract_pddl_from_response, make_repair_prompt
from generation.sampler import MultiSampler, SamplerConfig
from pddl_utils.oracle_planner import check_solvability_oracle
from search.ranking import CandidateRecord, greedy_first, random_parseable, verifier_ranked
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


def _try_parse_pddl(pddl: str) -> bool:
    if not pddl or not pddl.strip():
        return False
    try:
        build(pddl)
        return True
    except Exception:
        return False


def _clear_proxy_env() -> None:
    cleared = []
    for key in PROXY_ENV_KEYS:
        if os.environ.pop(key, None) is not None:
            cleared.append(key)
    if cleared:
        logger.info("Cleared proxy environment variables for generation: %s", ", ".join(cleared))


def _style(row: dict) -> str:
    init = "abstract" if int(row.get("init_is_abstract", 0)) else "explicit"
    goal = "abstract" if int(row.get("goal_is_abstract", 0)) else "explicit"
    return f"{init}/{goal}"


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


def _policy_markdown(summary: dict) -> str:
    lines = [
        "# Verifier-Ranked Best-of-K Pilot",
        "",
        "| K | Policy | Parse | Equiv | Equiv / Parse | Avg Parseable | Avg Equivalent | Oracle Best-of-K |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k_key, block in summary["comparisons"].items():
        for policy, details in block["policies"].items():
            metrics = details["metrics"]
            pool = details["candidate_pool"]
            lines.append(
                f"| {k_key} | {policy} | {metrics['parse_rate']:.4f} | {metrics['equiv_rate']:.4f} | "
                f"{metrics['equiv_given_parse']:.4f} | {pool['avg_parseable_candidates']:.2f} | "
                f"{pool['avg_equivalent_candidates']:.2f} | {pool['oracle_bestofk_equiv_rate']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def _metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    repair_parse = sum(1 for row in rows if row.get("repair_parseable"))
    repair_equiv = sum(1 for row in rows if row.get("repair_equivalent"))
    helped = sum(1 for row in rows if row.get("outcome") == "repair_helped")
    hurt = sum(1 for row in rows if row.get("outcome") == "repair_hurt")
    return {
        "total": total,
        "repair_parse_count": repair_parse,
        "repair_equiv_count": repair_equiv,
        "repair_parse_rate": repair_parse / total if total else 0.0,
        "repair_equiv_rate": repair_equiv / total if total else 0.0,
        "repair_equiv_given_parse": repair_equiv / repair_parse if repair_parse else 0.0,
        "helped": helped,
        "hurt": hurt,
        "tied": total - helped - hurt,
    }


def _repair_outcome(original_equiv: bool, repair_equiv: bool) -> str:
    if repair_equiv and not original_equiv:
        return "repair_helped"
    if original_equiv and not repair_equiv:
        return "repair_hurt"
    if repair_equiv and original_equiv:
        return "both_success"
    return "both_fail"


def _build_repair_feedback(*, parseable: bool, solvable: bool, verifier_score: float | None) -> str:
    parse_text = "parses successfully" if parseable else "does not parse"
    solvability = "solvable" if solvable else "not confirmed solvable"
    score_text = "unknown" if verifier_score is None else f"{verifier_score:.4f}"
    return (
        f"The candidate PDDL {parse_text}. "
        f"The current verifier score for this candidate is {score_text}. "
        f"A lightweight planner check says the candidate is {solvability}. "
        "This candidate was selected by the current verifier, but it may still fail to match the natural-language task. "
        "Repair the problem definition so the objects, initial state, and goal match the task description exactly."
    )


def should_attempt_repair(
    *,
    k: int,
    selected_index: int | None,
    selected_parseable: bool,
    repair_cfg: dict[str, Any],
) -> bool:
    if not repair_cfg.get("enabled", False):
        return False
    if int(repair_cfg.get("K", 8)) != int(k):
        return False
    if selected_index is None:
        return False
    return bool(selected_parseable)


def _sample_repair(repair_sampler: MultiSampler, prompt: str):
    backend, _ = repair_sampler.backends[0]
    started = time.time()
    raw_response = ""
    error = None
    try:
        raw_response = backend._call_llm(prompt, SYSTEM_PROMPT)  # noqa: SLF001 - repair uses a custom prompt.
        extracted = extract_pddl_from_response(raw_response)
    except Exception as exc:  # pragma: no cover - provider/runtime dependent
        extracted = ""
        error = str(exc)
    return {
        "backend": backend.backend_name,
        "model": backend.model,
        "raw_response": raw_response,
        "pddl": extracted,
        "latency_sec": time.time() - started,
        "error": error,
    }


def _breakdown_repair(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {name: _metrics_for_rows(group_rows) for name, group_rows in sorted(grouped.items())}


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


def _flush_all_logs() -> None:
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def _write_progress_snapshot(
    output_dir: Path,
    *,
    total_rows: int,
    completed_rows: int,
    started_at: float,
    current_row_name: str = "",
    current_domain: str = "",
) -> None:
    elapsed_sec = max(0.0, time.time() - started_at)
    avg_sec_per_row = elapsed_sec / max(1, completed_rows) if completed_rows else None
    eta_sec = avg_sec_per_row * max(0, total_rows - completed_rows) if avg_sec_per_row is not None else None
    snapshot = {
        "total_rows": total_rows,
        "completed_rows": completed_rows,
        "remaining_rows": max(0, total_rows - completed_rows),
        "elapsed_sec": elapsed_sec,
        "avg_sec_per_row": avg_sec_per_row,
        "eta_sec": eta_sec,
        "current_row_name": current_row_name,
        "current_domain": current_domain,
    }
    with open(output_dir / "progress.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run verifier-ranked best-of-K pilot")
    parser.add_argument("--config", type=str, default="configs/vcsr_bestofk_pilot.yaml")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--k_values", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--selection_metadata", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg.setdefault("experiment", {})["seed"] = int(args.seed)
    if args.output_dir:
        cfg.setdefault("output", {})["dir"] = args.output_dir
    if args.selection_metadata:
        cfg.setdefault("verifier", {})["selection_metadata"] = args.selection_metadata

    seed = cfg.get("experiment", {}).get("seed", 42)
    random.seed(seed)

    ds_cfg = cfg.get("dataset", {})
    gen_cfg = cfg.get("generation", {})
    verifier_cfg = cfg.get("verifier", {})
    eval_cfg = cfg.get("evaluation", {})
    repair_cfg = cfg.get("repair", {})
    out_cfg = cfg.get("output", {})

    output_dir = Path(out_cfg.get("dir", "results/vcsr/bestofk_pilot"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_file_logging(output_dir)

    with open(output_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    with open(output_dir / "process_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "pid": os.getpid(),
                "started_at": time.time(),
                "config": args.config,
                "output_dir": str(output_dir),
            },
            f,
            indent=2,
        )

    if os.environ.get("BEDROCK_MODEL_ID", "") == "" and os.environ.get("bedrock_model_id", ""):
        os.environ["BEDROCK_MODEL_ID"] = os.environ["bedrock_model_id"]
        logger.info("Normalized lowercase bedrock_model_id into BEDROCK_MODEL_ID for generation.")
    if bool(gen_cfg.get("clear_proxy_env", cfg.get("clear_proxy_env", False))):
        _clear_proxy_env()

    logger.info("Loading Planetarium dataset...")
    dataset = PlanetariumDataset(
        split_strategy=ds_cfg.get("split_strategy", "template_hash"),
        seed=seed,
    )
    split_name = ds_cfg.get("split", "test")
    split = dataset.get_split(split_name)

    domains = ds_cfg.get("domains", ["blocksworld", "gripper"])
    rows = [row for row in split if row["domain"] in domains]

    max_samples = args.max_rows or ds_cfg.get("max_samples", 30)
    if max_samples < len(rows):
        rng = random.Random(seed)
        indices = rng.sample(range(len(rows)), max_samples)
        indices.sort()
        rows = [rows[i] for i in indices]
    logger.info("Pilot rows selected: %d", len(rows))
    logger.info("Live progress log: %s", log_path)
    _flush_all_logs()

    sampler = MultiSampler(
        backend_specs=gen_cfg.get("backends", [{"type": "bedrock", "K": 8}]),
        config=SamplerConfig(
            temperature=gen_cfg.get("temperature", 0.8),
            top_p=gen_cfg.get("top_p", 0.95),
            max_new_tokens=gen_cfg.get("max_new_tokens", 1024),
            retry_attempts=gen_cfg.get("retry_attempts", 3),
            retry_delay_sec=gen_cfg.get("retry_delay_sec", 2.0),
        ),
    )
    repair_sampler = None
    if bool(repair_cfg.get("enabled", False)):
        repair_gen_cfg = repair_cfg.get("generation", {})
        repair_sampler = MultiSampler(
            backend_specs=repair_gen_cfg.get("backends", gen_cfg.get("backends", [{"type": "openrouter", "K": 1}])),
            config=SamplerConfig(
                temperature=repair_gen_cfg.get("temperature", 0.2),
                top_p=repair_gen_cfg.get("top_p", 0.9),
                max_new_tokens=repair_gen_cfg.get("max_new_tokens", gen_cfg.get("max_new_tokens", 1024)),
                retry_attempts=repair_gen_cfg.get("retry_attempts", gen_cfg.get("retry_attempts", 3)),
                retry_delay_sec=repair_gen_cfg.get("retry_delay_sec", gen_cfg.get("retry_delay_sec", 2.0)),
            ),
        )

    k_values = args.k_values or gen_cfg.get("K_values", [1, 4, 8])
    k_values = sorted(set(int(k) for k in k_values))
    max_k = max(k_values)
    if sampler.total_k < max_k:
        raise ValueError(
            f"Configured backends generate total K={sampler.total_k}, but K_values require {max_k}."
        )

    scorer = VerifierScorer(selection_path=verifier_cfg["selection_metadata"])
    scoring_batch_size = int(verifier_cfg.get("scoring_batch_size", 8))

    candidate_dump = []
    comparisons: dict[str, dict] = {}

    for k in k_values:
        comparisons[str(k)] = {
            "policies": {},
        }

    selected_results: dict[tuple[int, str], list[EvalResult]] = {
        (k, p): [] for k in k_values for p in eval_cfg.get("policies", ["greedy_first", "random_parseable", "verifier_ranked"])
    }
    pool_parseable_counts: dict[int, list[int]] = {k: [] for k in k_values}
    pool_equiv_counts: dict[int, list[int]] = {k: [] for k in k_values}
    pool_oracle_best: dict[int, list[int]] = {k: [] for k in k_values}
    repair_rows: list[dict[str, Any]] = []
    repair_outputs_path = output_dir / "repair_outputs.jsonl"
    started_at = time.time()
    _write_progress_snapshot(
        output_dir,
        total_rows=len(rows),
        completed_rows=0,
        started_at=started_at,
    )

    candidate_dump_path = output_dir / "candidate_dump.jsonl"
    with open(candidate_dump_path, "w", encoding="utf-8") as candidate_dump_file, open(
        repair_outputs_path, "w", encoding="utf-8"
    ) as repair_outputs_file:
        for row_idx, row in enumerate(rows):
            row_start = time.time()
            nl = row["natural_language"]
            gold_pddl = row["problem_pddl"]
            is_placeholder = bool(row.get("is_placeholder", 0))
            row_name = row["name"]
            logger.info(
                "Row %d/%d start: %s [%s]",
                row_idx + 1,
                len(rows),
                row_name,
                row["domain"],
            )
            _flush_all_logs()
            _write_progress_snapshot(
                output_dir,
                total_rows=len(rows),
                completed_rows=row_idx,
                started_at=started_at,
                current_row_name=row_name,
                current_domain=row["domain"],
            )

            samples = sampler.sample(natural_language=nl, domain=row["domain"])
            samples = samples[:max_k]

            all_scores = [None] * len(samples)
            parseable_pairs = []
            parseable_indices = []
            eval_results = []

            for i, sample in enumerate(samples):
                parseable = _try_parse_pddl(sample.extracted_pddl)
                if parseable:
                    res = check_equivalence_lightweight(
                        gold_pddl,
                        sample.extracted_pddl,
                        is_placeholder=is_placeholder,
                    )
                    parseable_pairs.append((nl, sample.extracted_pddl))
                    parseable_indices.append(i)
                else:
                    res = EvalResult(parseable=False, equivalent=False, error=sample.error or "parse_failed")
                eval_results.append(res)

            if parseable_pairs:
                scores = scorer.score_pairs(parseable_pairs, batch_size=scoring_batch_size)
                for idx, score in zip(parseable_indices, scores):
                    all_scores[idx] = score

            row_candidate_records = []
            for i, (sample, res) in enumerate(zip(samples, eval_results)):
                record = {
                    "row_index": row_idx,
                    "planetarium_name": row["name"],
                    "domain": row["domain"],
                    "init_is_abstract": row.get("init_is_abstract", 0),
                    "goal_is_abstract": row.get("goal_is_abstract", 0),
                    "candidate_index": i,
                    "backend": sample.backend,
                    "model": sample.model,
                    "latency_sec": sample.latency_sec,
                    "error": sample.error,
                    "raw_response": sample.raw_response,
                    "pddl": sample.extracted_pddl,
                    "parseable": res.parseable,
                    "equivalent": res.equivalent,
                    "verifier_score": all_scores[i],
                }
                row_candidate_records.append(record)
                candidate_dump.append(record)
                candidate_dump_file.write(json.dumps(record) + "\n")

            for k in k_values:
                subset_results = eval_results[:k]
                subset_scores = all_scores[:k]
                candidate_records = [
                    CandidateRecord(
                        index=i,
                        parseable=subset_results[i].parseable,
                        equivalent=subset_results[i].equivalent,
                        verifier_score=subset_scores[i],
                    )
                    for i in range(k)
                ]

                pool_parseable_counts[k].append(sum(1 for r in subset_results if r.parseable))
                pool_equiv_counts[k].append(sum(1 for r in subset_results if r.equivalent))
                pool_oracle_best[k].append(1 if any(r.equivalent for r in subset_results) else 0)

                rng = random.Random(seed + row_idx * 1000 + k)
                selections = {
                    "greedy_first": greedy_first(candidate_records),
                    "random_parseable": random_parseable(candidate_records, rng),
                    "verifier_ranked": verifier_ranked(candidate_records),
                }
                if "verifier_ranked_repair" in eval_cfg.get("policies", []):
                    selections["verifier_ranked_repair"] = selections["verifier_ranked"]

                for policy_name, selection in selections.items():
                    selected_eval = None
                    repair_row = None
                    if (
                        policy_name == "verifier_ranked_repair"
                        and should_attempt_repair(
                            k=k,
                            selected_index=selection.selected_index,
                            selected_parseable=(
                                False if selection.selected_index is None else subset_results[selection.selected_index].parseable
                            ),
                            repair_cfg=repair_cfg,
                        )
                        and repair_sampler is not None
                    ):
                        selected_idx = int(selection.selected_index)
                        selected_sample = samples[selected_idx]
                        selected_eval_original = subset_results[selected_idx]
                        selected_score = subset_scores[selected_idx]
                        selected_plan = check_solvability_oracle(selected_sample.extracted_pddl)
                        feedback = _build_repair_feedback(
                            parseable=selected_eval_original.parseable,
                            solvable=bool(selected_plan.solvable),
                            verifier_score=selected_score,
                        )
                        repair_prompt = make_repair_prompt(
                            natural_language=nl,
                            candidate_pddl=selected_sample.extracted_pddl,
                            domain=row["domain"],
                            feedback=feedback,
                        )
                        repair_sample = _sample_repair(repair_sampler, repair_prompt)
                        repair_parseable = _try_parse_pddl(repair_sample["pddl"])
                        if repair_parseable:
                            repair_eval = check_equivalence_lightweight(
                                gold_pddl,
                                repair_sample["pddl"],
                                is_placeholder=is_placeholder,
                            )
                            selected_eval = repair_eval
                            repair_plan = check_solvability_oracle(repair_sample["pddl"])
                            repair_score = scorer.score_pair(nl, repair_sample["pddl"])
                            final_source = "repair"
                        else:
                            repair_eval = EvalResult(
                                parseable=False,
                                equivalent=False,
                                error=repair_sample["error"] or "parse_failed",
                            )
                            selected_eval = selected_eval_original
                            repair_plan = None
                            repair_score = None
                            final_source = "original_fallback_repair_unparseable"

                        repair_row = {
                            "row_index": row_idx,
                            "planetarium_name": row["name"],
                            "domain": row["domain"],
                            "style": _style(row),
                            "K": k,
                            "policy": policy_name,
                            "original_selected_index": selected_idx,
                            "original_selected_pddl": selected_sample.extracted_pddl,
                            "original_selected_score": selected_score,
                            "original_selected_parseable": selected_eval_original.parseable,
                            "original_selected_solvable": bool(selected_plan.solvable),
                            "original_selected_equivalent": selected_eval_original.equivalent,
                            "original_selected_planner_error": selected_plan.error,
                            "feedback": feedback,
                            "repair_raw_response": repair_sample["raw_response"],
                            "repair_pddl": repair_sample["pddl"],
                            "repair_parseable": repair_eval.parseable,
                            "repair_solvable": bool(repair_plan.solvable) if repair_plan else False,
                            "repair_planner_error": repair_plan.error if repair_plan else None,
                            "repair_equivalent": repair_eval.equivalent,
                            "repair_error": repair_eval.error or repair_sample["error"],
                            "repair_verifier_score": repair_score,
                            "outcome": _repair_outcome(selected_eval_original.equivalent, selected_eval.equivalent),
                            "final_source": final_source,
                            "latency_sec": repair_sample["latency_sec"],
                            "backend": repair_sample["backend"],
                            "model": repair_sample["model"],
                        }
                        repair_rows.append(repair_row)
                        repair_outputs_file.write(json.dumps(repair_row) + "\n")
                        repair_outputs_file.flush()

                    if selection.selected_index is None:
                        selected_results[(k, policy_name)].append(
                            EvalResult(parseable=False, equivalent=False, error=selection.reason)
                        )
                    elif selected_eval is not None:
                        selected_results[(k, policy_name)].append(selected_eval)
                    else:
                        selected_results[(k, policy_name)].append(subset_results[selection.selected_index])

                    record = {
                        "row_index": row_idx,
                        "planetarium_name": row["name"],
                        "domain": row["domain"],
                        "init_is_abstract": row.get("init_is_abstract", 0),
                        "goal_is_abstract": row.get("goal_is_abstract", 0),
                        "K": k,
                        "policy": policy_name,
                        "selected_index": selection.selected_index,
                        "selection_reason": selection.reason,
                    }
                    if repair_row is not None:
                        record.update(
                            {
                                "repair_attempted": True,
                                "repair_equivalent": repair_row["repair_equivalent"],
                                "repair_parseable": repair_row["repair_parseable"],
                                "repair_outcome": repair_row["outcome"],
                                "repair_final_source": repair_row["final_source"],
                            }
                        )
                    candidate_dump.append(record)
                    candidate_dump_file.write(json.dumps(record) + "\n")

            candidate_dump_file.flush()
            repair_outputs_file.flush()
            row_parseable = sum(1 for r in eval_results if r.parseable)
            row_equiv = sum(1 for r in eval_results if r.equivalent)
            elapsed_row = time.time() - row_start
            elapsed_total = time.time() - started_at
            avg_sec_per_row = elapsed_total / max(1, row_idx + 1)
            eta_sec = avg_sec_per_row * max(0, len(rows) - (row_idx + 1))
            logger.info(
                "Row %d/%d done: parseable=%d/%d equivalent=%d/%d row_time=%.1fs elapsed=%.1fs eta=%.1fs",
                row_idx + 1,
                len(rows),
                row_parseable,
                len(samples),
                row_equiv,
                len(samples),
                elapsed_row,
                elapsed_total,
                eta_sec,
            )
            _flush_all_logs()
            _write_progress_snapshot(
                output_dir,
                total_rows=len(rows),
                completed_rows=row_idx + 1,
                started_at=started_at,
                current_row_name=row_name,
                current_domain=row["domain"],
            )

    for k in k_values:
        k_key = str(k)
        for policy_name in eval_cfg.get("policies", ["greedy_first", "random_parseable", "verifier_ranked"]):
            comparisons[k_key]["policies"][policy_name] = _summarize_policy(
                rows=rows,
                results=selected_results[(k, policy_name)],
                pool_parseable_counts=pool_parseable_counts[k],
                pool_equiv_counts=pool_equiv_counts[k],
                pool_oracle_best=pool_oracle_best[k],
            )

    summary = {
        "experiment": cfg.get("experiment", {}),
        "dataset": {
            "split": split_name,
            "domains": domains,
            "rows_evaluated": len(rows),
        },
        "generation": {
            "configured_total_k": sampler.total_k,
            "evaluated_k_values": k_values,
        },
        "verifier": {
            "selection_metadata": verifier_cfg["selection_metadata"],
        },
        "comparisons": comparisons,
    }
    if repair_rows:
        summary["repair"] = {
            "enabled": True,
            "metrics": _metrics_for_rows(repair_rows),
            "outcome_counts": Counter(row["outcome"] for row in repair_rows).most_common(),
            "domain_breakdown": _breakdown_repair(repair_rows, "domain"),
            "style_breakdown": _breakdown_repair(repair_rows, "style"),
            "repair_outputs": str(repair_outputs_path),
        }

    with open(output_dir / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(_policy_markdown(summary))
    logger.info("Saved aggregate metrics to %s", output_dir / "aggregate_metrics.json")
    logger.info("Saved candidate dump to %s", candidate_dump_path)
    logger.info("Saved markdown summary to %s", output_dir / "summary.md")
    _flush_all_logs()


if __name__ == "__main__":
    main()
