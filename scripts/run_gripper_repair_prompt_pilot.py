"""
Rerun cached gripper repair failures with the stricter gripper repair prompt.

This script does not generate new best-of-K pools. It reuses gripper failures
from the fresh repair gate, applies the current domain-specific gripper repair
prompt once per row, and evaluates whether the prompt fixes the gripper gap.
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
from eval.equivalence import EvalResult
from generation.prompts import SYSTEM_PROMPT, extract_pddl_from_response, make_repair_prompt
from generation.sampler import MultiSampler, SampleResult, SamplerConfig
from pddl_utils.oracle_planner import check_solvability_oracle
from scripts.run_repair_pilot import RepairCase, _evaluate_repair, _metrics, _outcome, build_feedback
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


def _clear_proxy_env() -> None:
    cleared = []
    for key in PROXY_ENV_KEYS:
        if os.environ.pop(key, None) is not None:
            cleared.append(key)
    if cleared:
        logger.info("Cleared proxy environment variables: %s", ", ".join(cleared))


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


def _try_parse_pddl(pddl: str) -> bool:
    if not pddl or not pddl.strip():
        return False
    try:
        build(pddl)
        return True
    except Exception:
        return False


def _load_dataset_rows(pool_root: Path, seed: int) -> dict[str, dict[str, Any]]:
    run_config = pool_root / f"seed_{seed}" / "pool" / "run_config.yaml"
    with open(run_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ds_cfg = cfg.get("dataset", {})
    dataset = PlanetariumDataset(
        split_strategy=ds_cfg.get("split_strategy", "template_hash"),
        seed=int(cfg.get("experiment", {}).get("seed", seed)),
    )
    rows = dataset.get_split(ds_cfg.get("split", "test"))
    return {row["name"]: row for row in rows}


def _load_cases(cfg: dict[str, Any]) -> list[RepairCase]:
    selection_cfg = cfg.get("selection", {})
    domain = selection_cfg.get("domain", "gripper")
    pool_root = Path(cfg["inputs"].get("pool_root", "results/vcsr/fresh_repair_gate_round4"))
    rows_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    cases: list[RepairCase] = []
    for path_str in cfg["inputs"]["repair_outputs"]:
        with open(path_str, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("domain") != domain:
                    continue
                if row.get("original_selected_equivalent"):
                    continue
                seed = int(row["seed"])
                if seed not in rows_by_seed:
                    rows_by_seed[seed] = _load_dataset_rows(pool_root, seed)
                dataset_row = rows_by_seed[seed][row["planetarium_name"]]
                cases.append(
                    RepairCase(
                        pool=row.get("pool", f"seed_{seed}"),
                        row_index=int(row["row_index"]),
                        planetarium_name=row["planetarium_name"],
                        domain=row["domain"],
                        style=row["style"],
                        natural_language=row["natural_language"],
                        gold_pddl=dataset_row["problem_pddl"],
                        is_placeholder=bool(dataset_row.get("is_placeholder", 0)),
                        selected_index=int(row["original_selected_index"]),
                        selected_pddl=row["original_selected_pddl"],
                        selected_score=float(row["original_selected_score"]),
                        selected_parseable=bool(row["original_selected_parseable"]),
                        selected_solvable=bool(row["original_selected_solvable"]),
                        selected_equivalent=bool(row["original_selected_equivalent"]),
                        selected_planner_error=row.get("original_selected_planner_error"),
                    )
                )
    rng = random.Random(int(cfg.get("experiment", {}).get("seed", 42)))
    rng.shuffle(cases)
    cases.sort(key=lambda case: (-case.selected_score, case.planetarium_name, case.row_index))
    return cases[: int(selection_cfg.get("max_rows", len(cases)))]


def _sample_repair(sampler: MultiSampler, prompt: str) -> SampleResult:
    backend, _ = sampler.backends[0]
    started = time.time()
    raw_response = ""
    error = None
    try:
        raw_response = backend._call_llm(prompt, SYSTEM_PROMPT)  # noqa: SLF001 - custom repair prompt.
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


def _breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {name: _metrics(group_rows) for name, group_rows in sorted(grouped.items())}


def _markdown(summary: dict[str, Any]) -> str:
    m = summary["metrics"]
    lines = [
        "# Gripper Repair Prompt Pilot",
        "",
        f"Rows repaired: `{m['total']}`",
        "",
        "| Repair Parse | Repair Eq | Repair Eq / Parse | Helped | Hurt | Tied | Accepted |",
        "|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {m['repair_parse_rate']:.4f} | {m['repair_equiv_rate']:.4f} | "
            f"{m['repair_equiv_given_parse']:.4f} | {m['helped']} | {m['hurt']} | {m['tied']} | "
            f"{summary['accepted']} |"
        ),
        "",
        "## Style Breakdown",
        "",
        "| Style | Rows | Repair Eq | Helped | Hurt |",
        "|---|---:|---:|---:|---:|",
    ]
    for style, data in summary["style_breakdown"].items():
        lines.append(f"| {style} | {data['total']} | {data['repair_equiv_rate']:.4f} | {data['helped']} | {data['hurt']} |")
    lines.extend(["", "## Recommendation", "", summary["recommendation"], ""])
    return "\n".join(lines)


def _accepted(metrics: dict[str, Any]) -> bool:
    return metrics["repair_parse_rate"] >= 0.80 and metrics["repair_equiv_rate"] >= 0.15 and metrics["helped"] > metrics["hurt"]


def run(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.max_rows is not None:
        cfg.setdefault("selection", {})["max_rows"] = int(args.max_rows)
    if args.output_dir:
        cfg.setdefault("output", {})["dir"] = args.output_dir

    output_dir = Path(cfg.get("output", {}).get("dir", "results/vcsr/gripper_repair_prompt_pilot"))
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

    if bool(cfg.get("generation", {}).get("clear_proxy_env", True)):
        _clear_proxy_env()

    cases = _load_cases(cfg)
    with open(output_dir / "repair_cases.jsonl", "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps({k: v for k, v in case.__dict__.items() if k != "gold_pddl"}) + "\n")

    generation_cfg = cfg.get("generation", {})
    sampler = MultiSampler(
        backend_specs=generation_cfg.get("backends", [{"type": "openrouter", "K": 1}]),
        config=SamplerConfig(
            temperature=float(generation_cfg.get("temperature", 0.2)),
            top_p=float(generation_cfg.get("top_p", 0.9)),
            max_new_tokens=int(generation_cfg.get("max_new_tokens", 1024)),
            retry_attempts=int(generation_cfg.get("retry_attempts", 3)),
            retry_delay_sec=float(generation_cfg.get("retry_delay_sec", 2)),
        ),
    )
    scorer = VerifierScorer(selection_path=cfg["selection"]["verifier_selection"])
    timeout_sec = float(cfg.get("evaluation", {}).get("equivalence_timeout_sec", 0))

    rows: list[dict[str, Any]] = []
    _write_progress(output_dir, stage="repairing", total_rows=len(cases), completed_rows=0, started_at=started_at)
    with open(output_dir / "repair_outputs.jsonl", "w", encoding="utf-8") as f:
        for idx, case in enumerate(cases):
            logger.info("Gripper repair case %d/%d: %s", idx + 1, len(cases), case.planetarium_name)
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
            _write_progress(
                output_dir,
                stage="repairing",
                total_rows=len(cases),
                completed_rows=idx + 1,
                started_at=started_at,
                current_case=case.planetarium_name,
            )

    metrics = _metrics(rows)
    summary = {
        "config": cfg,
        "metrics": metrics,
        "outcome_counts": Counter(row["outcome"] for row in rows).most_common(),
        "style_breakdown": _breakdown(rows, "style"),
        "accepted": _accepted(metrics),
    }
    summary["recommendation"] = (
        "Gripper-specific repair is promising; rerun the fresh repair gate with domain-aware routing."
        if summary["accepted"]
        else "Gripper-specific repair is still weak; inspect outputs before another fresh gate."
    )
    _write_json(output_dir / "repair_summary.json", summary)
    with open(output_dir / "repair_summary.md", "w", encoding="utf-8") as f:
        f.write(_markdown(summary))
    _write_progress(output_dir, stage="complete", total_rows=len(cases), completed_rows=len(cases), started_at=started_at)
    logger.info("Saved gripper repair prompt summary to %s", output_dir / "repair_summary.md")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vcsr_gripper_repair_prompt_pilot.yaml")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
