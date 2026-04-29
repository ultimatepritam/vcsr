"""
Baseline reproduction script for VCSR.

Validates the end-to-end evaluation pipeline by running multiple baselines:
1. Oracle baseline (gold PDDL) -- should achieve 100% equivalence
2. Perturbed baseline (corrupted gold PDDL) -- should show parse but low equivalence
3. LLM baseline (optional, requires API key) -- real generation evaluation

Usage:
    python scripts/reproduce_baselines.py --config configs/baseline.yaml
    python scripts/reproduce_baselines.py --max_samples 50
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vcsr_env  # noqa: F401
import yaml

from data.planetarium_loader import PlanetariumDataset
from eval.equivalence import (
    evaluate_batch,
    stratified_report,
    check_equivalence_lightweight,
    BatchMetrics,
)
from pddl_utils.oracle_planner import check_solvability_oracle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def perturb_pddl(pddl_str: str, seed: int = 0) -> str:
    """
    Apply a random perturbation to a gold PDDL string to create a hard negative.
    Perturbations are designed to keep the PDDL parseable but semantically wrong.
    """
    rng = random.Random(seed)
    perturbation = rng.choice(["swap_goal_pred", "drop_init_pred", "add_extra_object"])

    if perturbation == "swap_goal_pred":
        goal_match = re.search(r"\(:goal\s*\(and\s*(.*?)\)\s*\)", pddl_str, re.DOTALL)
        if goal_match:
            goal_body = goal_match.group(1)
            preds = re.findall(r"\([^()]+\)", goal_body)
            if len(preds) >= 2:
                i, j = rng.sample(range(len(preds)), 2)
                preds[i], preds[j] = preds[j], preds[i]
                new_goal = "(:goal (and " + " ".join(preds) + "))"
                pddl_str = pddl_str[:goal_match.start()] + new_goal + pddl_str[goal_match.end():]

    elif perturbation == "drop_init_pred":
        init_match = re.search(r"\(:init\s*(.*?)\)", pddl_str, re.DOTALL)
        if init_match:
            init_body = init_match.group(1)
            preds = re.findall(r"\([^()]+\)", init_body)
            if len(preds) > 2:
                drop_idx = rng.randint(0, len(preds) - 1)
                preds.pop(drop_idx)
                new_init = "(:init " + " ".join(preds) + ")"
                pddl_str = pddl_str[:init_match.start()] + new_init + pddl_str[init_match.end():]

    elif perturbation == "add_extra_object":
        obj_match = re.search(r"\(:objects\s*(.*?)\)", pddl_str)
        if obj_match:
            objects = obj_match.group(1).strip().split()
            objects.append(f"extra_{rng.randint(1, 99)}")
            new_objects = "(:objects " + " ".join(objects) + ")"
            pddl_str = pddl_str[:obj_match.start()] + new_objects + pddl_str[obj_match.end():]

    return pddl_str


def run_oracle_baseline(dataset, max_samples: int) -> dict:
    """
    Oracle baseline: use gold PDDL as the 'candidate'.
    Should achieve 100% parse and equivalence.
    """
    logger.info(f"=== Oracle Baseline (gold PDDL) on {max_samples} samples ===")

    test_data = dataset.test.select(range(min(max_samples, len(dataset.test))))

    gold_pddls = test_data["problem_pddl"]
    candidate_pddls = test_data["problem_pddl"]  # same as gold
    is_placeholders = [bool(row) for row in test_data["is_placeholder"]]

    metrics, results = evaluate_batch(
        gold_pddls, candidate_pddls, is_placeholders=is_placeholders
    )

    logger.info(f"Oracle baseline: {metrics}")

    rows = [dict(zip(test_data.column_names, vals)) for vals in zip(*[test_data[c] for c in test_data.column_names])]
    report = stratified_report(rows, results)
    for stratum, m in report.items():
        logger.info(f"  {stratum}: equiv={m.equiv_rate:.3f}, parse={m.parse_rate:.3f}")

    return {"metrics": str(metrics), "per_stratum": {k: str(v) for k, v in report.items()}}


def run_perturbed_baseline(dataset, max_samples: int) -> dict:
    """
    Perturbed baseline: apply random perturbations to gold PDDL.
    Should show high parse rate but reduced equivalence.
    """
    logger.info(f"=== Perturbed Baseline on {max_samples} samples ===")

    test_data = dataset.test.select(range(min(max_samples, len(dataset.test))))

    gold_pddls = test_data["problem_pddl"]
    candidate_pddls = [perturb_pddl(p, seed=i) for i, p in enumerate(gold_pddls)]
    is_placeholders = [bool(row) for row in test_data["is_placeholder"]]

    metrics, results = evaluate_batch(
        gold_pddls, candidate_pddls, is_placeholders=is_placeholders
    )

    logger.info(f"Perturbed baseline: {metrics}")

    rows = [dict(zip(test_data.column_names, vals)) for vals in zip(*[test_data[c] for c in test_data.column_names])]
    report = stratified_report(rows, results)
    for stratum, m in report.items():
        logger.info(f"  {stratum}: equiv={m.equiv_rate:.3f}, parse={m.parse_rate:.3f}")

    return {"metrics": str(metrics), "per_stratum": {k: str(v) for k, v in report.items()}}


def run_solvability_check(dataset, max_samples: int) -> dict:
    """
    Run oracle planner solvability check on gold PDDL test samples.
    Validates the planner integration works.
    """
    logger.info(f"=== Solvability Check (oracle planner) on {max_samples} samples ===")

    test_data = dataset.test.select(range(min(max_samples, len(dataset.test))))
    gold_pddls = test_data["problem_pddl"]

    solvable_count = 0
    error_count = 0

    for i, pddl in enumerate(gold_pddls):
        result = check_solvability_oracle(pddl)
        if result.solvable:
            solvable_count += 1
        if result.error:
            error_count += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  Checked {i+1}/{len(gold_pddls)}: solvable={solvable_count}, errors={error_count}")

    rate = solvable_count / max(len(gold_pddls), 1)
    logger.info(f"Solvability: {solvable_count}/{len(gold_pddls)} = {rate:.3f}")
    return {"solvable": solvable_count, "total": len(gold_pddls), "rate": rate, "errors": error_count}


def main():
    parser = argparse.ArgumentParser(description="VCSR Baseline Reproduction")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_solvability", action="store_true")
    args = parser.parse_args()

    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")

    max_samples = args.max_samples or config.get("dataset", {}).get("max_test_samples", 100)
    seed = config.get("experiment", {}).get("seed", 42)
    output_dir = args.output_dir or config.get("output", {}).get("dir", "results/baseline")

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Config: max_samples={max_samples}, seed={seed}, output_dir={output_dir}")

    logger.info("Loading dataset...")
    dataset = PlanetariumDataset(seed=seed)
    logger.info(dataset.summary())

    all_results = {}

    all_results["oracle"] = run_oracle_baseline(dataset, max_samples)

    all_results["perturbed"] = run_perturbed_baseline(dataset, max_samples)

    if not args.skip_solvability:
        all_results["solvability"] = run_solvability_check(dataset, max_samples)

    output_path = Path(output_dir) / "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    logger.info("\n=== SUMMARY ===")
    logger.info("Pipeline validation complete.")
    logger.info(f"  Oracle baseline: {all_results['oracle']['metrics']}")
    logger.info(f"  Perturbed baseline: {all_results['perturbed']['metrics']}")
    if "solvability" in all_results:
        s = all_results["solvability"]
        logger.info(f"  Solvability: {s['solvable']}/{s['total']} = {s['rate']:.3f}")


if __name__ == "__main__":
    main()
