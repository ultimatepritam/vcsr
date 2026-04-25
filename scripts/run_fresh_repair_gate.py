"""
Run a fresh fixed-pool repair gate with the promoted round-4 verifier frozen.

For each seed, this script generates one best-of-K candidate pool, selects the
round-4 verifier-ranked K=8 failures from that exact pool, repairs each failure
once, and compares original round-4 selection against repair-augmented
selection on the same rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vcsr_env  # noqa: F401
import yaml

from eval.equivalence import EvalResult
from generation.prompts import SYSTEM_PROMPT, extract_pddl_from_response, make_repair_prompt
from generation.sampler import MultiSampler, SampleResult, SamplerConfig
from pddl_utils.oracle_planner import check_solvability_oracle
from scripts.run_repair_pilot import (
    RepairCase,
    _accepted,
    _evaluate_repair,
    _metrics,
    _outcome,
    _select_top_round4_failures,
    _try_parse_pddl,
    build_feedback,
)
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


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _clear_proxy_env(env: dict[str, str]) -> dict[str, str]:
    clean = dict(env)
    cleared = [key for key in PROXY_ENV_KEYS if key in clean]
    for key in PROXY_ENV_KEYS:
        clean.pop(key, None)
    if cleared:
        logger.info("Cleared proxy environment variables: %s", ", ".join(cleared))
    return clean


def _write_progress(
    output_dir: Path,
    *,
    stage: str,
    completed_steps: int,
    total_steps: int,
    started_at: float,
    seed: int | None = None,
    current_case: str = "",
) -> None:
    elapsed_sec = max(0.0, time.time() - started_at)
    avg_sec_per_step = elapsed_sec / max(1, completed_steps) if completed_steps else None
    eta_sec = avg_sec_per_step * max(0, total_steps - completed_steps) if avg_sec_per_step is not None else None
    _write_json(
        output_dir / "progress.json",
        {
            "stage": stage,
            "seed": seed,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "remaining_steps": max(0, total_steps - completed_steps),
            "elapsed_sec": elapsed_sec,
            "avg_sec_per_step": avg_sec_per_step,
            "eta_sec": eta_sec,
            "current_case": current_case,
        },
    )


def _run_command(command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]) -> None:
    logger.info("Running: %s", " ".join(command))
    _flush_logs()
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")


def _sample_repair(sampler: MultiSampler, prompt: str) -> SampleResult:
    backend, _ = sampler.backends[0]
    started = time.time()
    raw_response = ""
    error = None
    try:
        raw_response = backend._call_llm(prompt, SYSTEM_PROMPT)  # noqa: SLF001 - repair uses a custom prompt.
        extracted = extract_pddl_from_response(raw_response)
    except Exception as exc:  # pragma: no cover - provider/runtime dependent
        extracted = ""
        error = str(exc)
    return SampleResult(
        backend=backend.backend_name,
        model=backend.model,
        raw_response=raw_response,
        extracted_pddl=extracted,
        latency_sec=time.time() - started,
        error=error,
    )


def _repair_cases(
    *,
    cases: list[RepairCase],
    scorer: VerifierScorer,
    sampler: MultiSampler,
    timeout_sec: float,
    seed_dir: Path,
    gate_dir: Path,
    started_at: float,
    completed_steps: int,
    total_steps: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    output_path = seed_dir / "repair_outputs.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, case in enumerate(cases):
            logger.info("Repair seed %s case %d/%d: %s", seed, idx + 1, len(cases), case.planetarium_name)
            _write_progress(
                gate_dir,
                stage="repair",
                completed_steps=completed_steps,
                total_steps=total_steps,
                started_at=started_at,
                seed=seed,
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
            repair_eval = (
                _evaluate_repair(case=case, repaired_pddl=sample.extracted_pddl, timeout_sec=timeout_sec)
                if repair_parseable
                else EvalResult(parseable=False, equivalent=False, error=sample.error or "parse_failed")
            )
            repair_plan = check_solvability_oracle(sample.extracted_pddl) if repair_parseable else None
            repair_score = (
                scorer.score_pair(case.natural_language, sample.extracted_pddl)
                if repair_parseable and sample.extracted_pddl
                else None
            )
            row = {
                "seed": seed,
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
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            _flush_logs()
    return rows


def _read_aggregate_metrics(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _baseline_equiv_rate(metrics: dict[str, Any], k: int) -> float:
    return float(metrics["comparisons"][str(k)]["policies"]["verifier_ranked"]["metrics"]["equiv_rate"])


def _augmented_rate(*, baseline_rate: float, total_rows: int, helped: int) -> float:
    return (baseline_rate * total_rows + helped) / max(1, total_rows)


def _breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {name: _metrics(group_rows) for name, group_rows in sorted(grouped.items())}


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fresh Fixed-Pool Repair Gate",
        "",
        f"Base config: `{report['base_config']}`",
        f"Rows per seed: `{report['rows_per_seed']}`",
        f"Seeds: `{report['seeds']}`",
        "",
        "Generation happened once per seed. Round-4 selected failures were repaired once on the same candidate pools.",
        "",
        "## Mean K=8 Equivalence",
        "",
        "| Policy | Mean Equiv | Per-Seed Values |",
        "|---|---:|---|",
        (
            "| Round-4 verifier-ranked | "
            f"{report['mean_metrics']['baseline_k8_equiv_rate']:.4f} | "
            f"{', '.join(f'{v:.4f}' for v in report['mean_metrics']['baseline_k8_values'])} |"
        ),
        (
            "| Repair-augmented | "
            f"{report['mean_metrics']['repair_augmented_k8_equiv_rate']:.4f} | "
            f"{', '.join(f'{v:.4f}' for v in report['mean_metrics']['repair_augmented_k8_values'])} |"
        ),
        "",
        "## Repair Outcomes",
        "",
        "| Seed | Baseline K=8 | Repair-Aug K=8 | Delta | Failures Repaired | Helped | Hurt | Tied | Parse Rate | Repair Eq |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed_report in report["seed_reports"]:
        m = seed_report["repair_metrics"]
        lines.append(
            f"| {seed_report['seed']} | {seed_report['baseline_k8_equiv_rate']:.4f} | "
            f"{seed_report['repair_augmented_k8_equiv_rate']:.4f} | {seed_report['delta_k8']:+.4f} | "
            f"{m['total']} | {m['helped']} | {m['hurt']} | {m['tied']} | "
            f"{m['repair_parse_rate']:.4f} | {m['repair_equiv_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Breakdown Over Repaired Failures",
            "",
            "| Slice | Rows | Repair Eq | Helped | Hurt |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, data in report["domain_breakdown"].items():
        lines.append(f"| domain={name} | {data['total']} | {data['repair_equiv_rate']:.4f} | {data['helped']} | {data['hurt']} |")
    for name, data in report["style_breakdown"].items():
        lines.append(f"| style={name} | {data['total']} | {data['repair_equiv_rate']:.4f} | {data['helped']} | {data['hurt']} |")
    lines.extend(["", "## Recommendation", "", report["recommendation"], ""])
    return "\n".join(lines)


def _recommendation(report: dict[str, Any]) -> str:
    mean_delta = report["mean_metrics"]["mean_delta_k8"]
    helped = report["mean_metrics"]["total_helped"]
    hurt = report["mean_metrics"]["total_hurt"]
    parse_rate = report["mean_metrics"]["repair_parse_rate"]
    seed_deltas = [seed_report["delta_k8"] for seed_report in report["seed_reports"]]
    positive_seeds = sum(1 for delta in seed_deltas if delta > 0)
    if mean_delta >= 0.10 and helped > hurt and parse_rate >= 0.80 and positive_seeds >= 2:
        return "Repair passes the fresh fixed-pool gate; implement repair-augmented selection in the main best-of-K entrypoint next."
    return "Repair does not yet pass the fresh fixed-pool gate; inspect failures before scaling repair."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vcsr_fresh_repair_gate.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_file_logging(output_dir)
    repo_root = Path(__file__).resolve().parent.parent
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

    env = os.environ.copy()
    if cfg.get("clear_proxy_env", True):
        env = _clear_proxy_env(env)
        for key in PROXY_ENV_KEYS:
            os.environ.pop(key, None)

    repair_cfg = cfg["repair"]
    generation_cfg = repair_cfg.get("generation", {})
    scorer = VerifierScorer(selection_path=repair_cfg["verifier_selection"])
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

    seeds = [int(seed) for seed in cfg["pool_generation"]["seeds"]]
    total_steps = len(seeds) * 2
    completed_steps = 0
    force = bool(args.force or cfg.get("force", False))
    seed_reports: list[dict[str, Any]] = []
    all_repair_rows: list[dict[str, Any]] = []

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        pool_dir = seed_dir / "pool"
        repair_dir = seed_dir / "repair"
        pool_dir.mkdir(parents=True, exist_ok=True)
        repair_dir.mkdir(parents=True, exist_ok=True)
        candidate_dump = pool_dir / "candidate_dump.jsonl"

        _write_progress(
            output_dir,
            stage="generate_pool",
            completed_steps=completed_steps,
            total_steps=total_steps,
            started_at=started_at,
            seed=seed,
        )
        if force or not candidate_dump.exists():
            logger.info("Generating fresh candidate pool for seed %s", seed)
            _run_command(
                [
                    sys.executable,
                    "-u",
                    "scripts/run_verifier_bestofk.py",
                    "--config",
                    cfg["base_config"],
                    "--seed",
                    str(seed),
                    "--max_rows",
                    str(cfg["pool_generation"].get("max_rows", 50)),
                    "--output_dir",
                    str(pool_dir),
                    "--selection_metadata",
                    cfg["pool_generation"]["selection"],
                ],
                cwd=repo_root,
                log_path=pool_dir / "run_stdout.log",
                env=env,
            )
        else:
            logger.info("Using existing fresh pool for seed %s", seed)
        completed_steps += 1

        _write_progress(
            output_dir,
            stage="repair_pool",
            completed_steps=completed_steps,
            total_steps=total_steps,
            started_at=started_at,
            seed=seed,
        )
        repair_outputs = repair_dir / "repair_outputs.jsonl"
        if force or not repair_outputs.exists():
            cases = _select_top_round4_failures(
                candidate_dump=candidate_dump,
                scorer=scorer,
                k=int(repair_cfg.get("K", 8)),
                scoring_batch_size=int(repair_cfg.get("scoring_batch_size", 8)),
            )
            cases.sort(
                key=lambda case: (
                    1 if case.domain == "blocksworld" else 0,
                    1 if case.style == "abstract/abstract" else 0,
                    case.selected_score,
                    case.planetarium_name,
                ),
                reverse=True,
            )
            cases = cases[: int(repair_cfg.get("max_repairs_per_seed", 30))]
            with open(repair_dir / "repair_cases.jsonl", "w", encoding="utf-8") as f:
                for case in cases:
                    f.write(json.dumps({k: v for k, v in case.__dict__.items() if k != "gold_pddl"}) + "\n")
            repair_rows = _repair_cases(
                cases=cases,
                scorer=scorer,
                sampler=sampler,
                timeout_sec=float(repair_cfg.get("evaluation", {}).get("equivalence_timeout_sec", 0)),
                seed_dir=repair_dir,
                gate_dir=output_dir,
                started_at=started_at,
                completed_steps=completed_steps,
                total_steps=total_steps,
                seed=seed,
            )
        else:
            logger.info("Using existing repairs for seed %s", seed)
            with open(repair_outputs, encoding="utf-8") as f:
                repair_rows = [json.loads(line) for line in f if line.strip()]
        completed_steps += 1

        pool_metrics = _read_aggregate_metrics(pool_dir / "aggregate_metrics.json")
        baseline_k8 = _baseline_equiv_rate(pool_metrics, int(repair_cfg.get("K", 8)))
        total_rows = int(pool_metrics["comparisons"][str(repair_cfg.get("K", 8))]["policies"]["verifier_ranked"]["metrics"]["total"])
        repair_metrics = _metrics(repair_rows)
        augmented_k8 = _augmented_rate(
            baseline_rate=baseline_k8,
            total_rows=total_rows,
            helped=int(repair_metrics["helped"]),
        )
        seed_report = {
            "seed": seed,
            "candidate_dump": str(candidate_dump),
            "repair_dir": str(repair_dir),
            "rows": total_rows,
            "baseline_k8_equiv_rate": baseline_k8,
            "repair_augmented_k8_equiv_rate": augmented_k8,
            "delta_k8": augmented_k8 - baseline_k8,
            "repair_metrics": repair_metrics,
            "outcome_counts": Counter(row["outcome"] for row in repair_rows).most_common(),
        }
        _write_json(repair_dir / "repair_summary.json", seed_report)
        seed_reports.append(seed_report)
        all_repair_rows.extend(repair_rows)
        _flush_logs()

    baseline_values = [row["baseline_k8_equiv_rate"] for row in seed_reports]
    augmented_values = [row["repair_augmented_k8_equiv_rate"] for row in seed_reports]
    total_helped = sum(int(row["repair_metrics"]["helped"]) for row in seed_reports)
    total_hurt = sum(int(row["repair_metrics"]["hurt"]) for row in seed_reports)
    report = {
        "base_config": cfg["base_config"],
        "rows_per_seed": cfg["pool_generation"].get("max_rows", 50),
        "seeds": seeds,
        "seed_reports": seed_reports,
        "mean_metrics": {
            "baseline_k8_equiv_rate": mean(baseline_values) if baseline_values else 0.0,
            "repair_augmented_k8_equiv_rate": mean(augmented_values) if augmented_values else 0.0,
            "mean_delta_k8": mean([row["delta_k8"] for row in seed_reports]) if seed_reports else 0.0,
            "baseline_k8_values": baseline_values,
            "repair_augmented_k8_values": augmented_values,
            "repair_parse_rate": sum(row["repair_parseable"] for row in all_repair_rows) / max(1, len(all_repair_rows)),
            "repair_equiv_rate": sum(row["repair_equivalent"] for row in all_repair_rows) / max(1, len(all_repair_rows)),
            "total_helped": total_helped,
            "total_hurt": total_hurt,
            "total_repaired_failures": len(all_repair_rows),
        },
        "domain_breakdown": _breakdown(all_repair_rows, "domain"),
        "style_breakdown": _breakdown(all_repair_rows, "style"),
    }
    report["accepted"] = _recommendation(report).startswith("Repair passes")
    report["recommendation"] = _recommendation(report)
    _write_json(output_dir / "fresh_repair_gate_summary.json", report)
    with open(output_dir / "fresh_repair_gate_summary.md", "w", encoding="utf-8") as f:
        f.write(_summary_markdown(report))
    _write_progress(
        output_dir,
        stage="complete",
        completed_steps=completed_steps,
        total_steps=total_steps,
        started_at=started_at,
    )
    logger.info("Wrote fresh repair gate summary to %s", output_dir / "fresh_repair_gate_summary.md")


if __name__ == "__main__":
    main()
