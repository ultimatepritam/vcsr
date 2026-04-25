"""
Generate fresh candidate pools once, then replay multiple verifiers on them.

This is the clean comparison gate for verifier checkpoints: generation happens
once per seed, then every verifier scores and selects from the exact same
candidate dump. It separates checkpoint quality from fresh-pool variance.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vcsr_env  # noqa: F401
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROXY_ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]


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


def _metric(summary: dict[str, Any], verifier_index: int, k: int, policy: str, metric: str) -> float:
    return float(summary["verifiers"][verifier_index]["comparisons"][str(k)]["policies"][policy]["metrics"][metric])


def _verifier_name(summary: dict[str, Any], verifier_index: int, configured: list[dict[str, str]]) -> str:
    if verifier_index < len(configured) and configured[verifier_index].get("name"):
        return configured[verifier_index]["name"]
    return str(summary["verifiers"][verifier_index]["name"])


def _aggregate(replay_summaries: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    verifiers = cfg["verifiers"]
    k_values = [int(k) for k in cfg.get("k_values", [4, 8])]
    policies = ["greedy_first", "random_parseable", "verifier_ranked"]

    verifier_blocks = []
    for verifier_index, verifier_cfg in enumerate(verifiers):
        block = {
            "name": verifier_cfg["name"],
            "selection": verifier_cfg["selection"],
            "mean_metrics": {},
            "per_seed": [],
        }
        for k in k_values:
            block["mean_metrics"][str(k)] = {}
            for policy in policies:
                values = [
                    _metric(replay_summary["summary"], verifier_index, k, policy, "equiv_rate")
                    for replay_summary in replay_summaries
                ]
                block["mean_metrics"][str(k)][policy] = {
                    "equiv_rate": mean(values),
                    "values": values,
                }
        for replay_summary in replay_summaries:
            seed = replay_summary["seed"]
            seed_block = {"seed": seed, "metrics": {}}
            for k in k_values:
                seed_block["metrics"][str(k)] = {
                    policy: _metric(replay_summary["summary"], verifier_index, k, policy, "equiv_rate")
                    for policy in policies
                }
            block["per_seed"].append(seed_block)
        verifier_blocks.append(block)

    head_to_head = {}
    if len(verifiers) >= 2:
        for k in k_values:
            deltas = []
            wins = losses = ties = 0
            for replay_summary in replay_summaries:
                base = _metric(replay_summary["summary"], 0, k, "verifier_ranked", "equiv_rate")
                cand = _metric(replay_summary["summary"], 1, k, "verifier_ranked", "equiv_rate")
                delta = cand - base
                deltas.append(delta)
                if delta > 0:
                    wins += 1
                elif delta < 0:
                    losses += 1
                else:
                    ties += 1
            head_to_head[str(k)] = {
                "baseline": verifiers[0]["name"],
                "candidate": verifiers[1]["name"],
                "policy": "verifier_ranked",
                "mean_delta": mean(deltas),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "deltas": deltas,
            }

    return {
        "base_config": cfg["base_config"],
        "max_rows": cfg.get("max_rows"),
        "seeds": cfg["seeds"],
        "k_values": k_values,
        "output_dir": cfg["output_dir"],
        "verifiers": verifier_blocks,
        "head_to_head": head_to_head,
        "replay_outputs": replay_summaries,
        "recommendation": _recommendation(head_to_head),
    }


def _recommendation(head_to_head: dict[str, Any]) -> str:
    k8 = head_to_head.get("8")
    k4 = head_to_head.get("4")
    if not k8 or not k4:
        return "Inspect the fixed-pool summary before making a promotion decision."
    if k8["mean_delta"] > 0 and k4["mean_delta"] >= -0.01 and k8["wins"] > k8["losses"]:
        return "Round 7 passes the identical-pool verifier gate; consider a final fresh promotion check before changing best_current."
    return "Round 4 remains best_current; round 7 does not pass the identical-pool verifier gate."


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fresh Fixed-Pool Verifier Comparison",
        "",
        f"Base config: `{report['base_config']}`",
        f"Rows per seed: `{report['max_rows']}`",
        f"Seeds: `{report['seeds']}`",
        "",
        "Generation happened once per seed. Each verifier was replayed on the same candidate dump.",
        "",
        "## Mean Verifier-Ranked Equivalence",
        "",
        "| Verifier | K | Mean Equiv | Per-Seed Values |",
        "|---|---:|---:|---|",
    ]
    for verifier in report["verifiers"]:
        for k_key, metrics in verifier["mean_metrics"].items():
            values = metrics["verifier_ranked"]["values"]
            value_text = ", ".join(f"{v:.4f}" for v in values)
            lines.append(
                f"| {verifier['name']} | {k_key} | {metrics['verifier_ranked']['equiv_rate']:.4f} | {value_text} |"
            )

    lines.extend(["", "## Head-to-Head", "", "| K | Candidate - Baseline | Wins | Losses | Ties |", "|---:|---:|---:|---:|---:|"])
    for k_key, h2h in report["head_to_head"].items():
        lines.append(
            f"| {k_key} | {h2h['mean_delta']:+.4f} | {h2h['wins']} | {h2h['losses']} | {h2h['ties']} |"
        )

    lines.extend(["", "## Replay Artifacts", "", "| Seed | Pool | Replay |", "|---:|---|---|"])
    for item in report["replay_outputs"]:
        lines.append(f"| {item['seed']} | `{item['candidate_dump']}` | `{item['replay_dir']}` |")

    lines.extend(["", "## Recommendation", "", report["recommendation"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vcsr_fixed_pool_round7_compare.yaml")
    parser.add_argument("--force", action="store_true", help="Regenerate/replay even if artifacts already exist.")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_file_logging(output_dir)
    started_at = time.time()
    repo_root = Path(__file__).resolve().parent.parent

    env = os.environ.copy()
    if cfg.get("clear_proxy_env", False):
        for key in PROXY_ENV_KEYS:
            env.pop(key, None)

    _write_json(
        output_dir / "process_info.json",
        {
            "pid": os.getpid(),
            "started_at": started_at,
            "config": args.config,
            "output_dir": str(output_dir),
        },
    )

    force = bool(args.force or cfg.get("force", False))
    replay_summaries = []
    seeds = [int(seed) for seed in cfg["seeds"]]
    total_steps = len(seeds) * 2
    completed_steps = 0

    for seed in seeds:
        pool_dir = output_dir / "pools" / f"seed_{seed}"
        replay_dir = output_dir / "replays" / f"seed_{seed}"
        pool_dir.mkdir(parents=True, exist_ok=True)
        replay_dir.mkdir(parents=True, exist_ok=True)
        candidate_dump = pool_dir / "candidate_dump.jsonl"

        _write_json(
            output_dir / "progress.json",
            {
                "stage": "generate_pool",
                "seed": seed,
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "elapsed_sec": time.time() - started_at,
            },
        )
        if force or not candidate_dump.exists():
            logger.info("Generating fixed candidate pool for seed %s", seed)
            command = [
                sys.executable,
                "-u",
                "scripts/run_verifier_bestofk.py",
                "--config",
                cfg["base_config"],
                "--seed",
                str(seed),
                "--max_rows",
                str(cfg.get("max_rows", 50)),
                "--output_dir",
                str(pool_dir),
                "--selection_metadata",
                cfg["pool_generation"]["selection"],
            ]
            _run_command(command, cwd=repo_root, log_path=pool_dir / "run_stdout.log", env=env)
        else:
            logger.info("Using existing fixed candidate pool for seed %s", seed)
        completed_steps += 1

        _write_json(
            output_dir / "progress.json",
            {
                "stage": "replay_pool",
                "seed": seed,
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "elapsed_sec": time.time() - started_at,
            },
        )
        replay_summary_path = replay_dir / "replay_summary.json"
        if force or not replay_summary_path.exists():
            logger.info("Replaying verifiers on fixed pool for seed %s", seed)
            command = [
                sys.executable,
                "-u",
                "scripts/replay_verifier_bestofk.py",
                "--candidate_dump",
                str(candidate_dump),
                "--output_dir",
                str(replay_dir),
                "--k_values",
                *[str(k) for k in cfg.get("k_values", [4, 8])],
            ]
            for verifier in cfg["verifiers"]:
                command.extend(["--selection", verifier["selection"]])
            _run_command(command, cwd=repo_root, log_path=replay_dir / "replay_stdout.log", env=env)
        else:
            logger.info("Using existing replay for seed %s", seed)
        completed_steps += 1

        with open(replay_summary_path, encoding="utf-8") as f:
            replay_summary = json.load(f)
        replay_summaries.append(
            {
                "seed": seed,
                "candidate_dump": str(candidate_dump),
                "replay_dir": str(replay_dir),
                "summary": replay_summary,
            }
        )
        _flush_logs()

    report = _aggregate(replay_summaries, cfg)
    _write_json(output_dir / "comparison_summary.json", report)
    with open(output_dir / "comparison_summary.md", "w", encoding="utf-8") as f:
        f.write(_markdown(report))
    _write_json(
        output_dir / "progress.json",
        {
            "stage": "complete",
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "elapsed_sec": time.time() - started_at,
            "completed": True,
        },
    )
    logger.info("Wrote fixed-pool comparison summary to %s", output_dir / "comparison_summary.md")


if __name__ == "__main__":
    main()
