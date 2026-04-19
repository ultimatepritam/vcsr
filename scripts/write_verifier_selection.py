"""
Write verifier selection metadata for a trained/calibrated run.

Usage:
    python scripts/write_verifier_selection.py ^
      --run_dir results/verifier/ranking_aligned_round4/retrain_from_round3_focused ^
      --summary "Focused round-4 verifier fine-tune..." ^
      --criterion "mined held-out oracle-positive misranking rows"
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import yaml


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
        return str(rel).replace("/", "\\")
    except Exception:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write verifier selection metadata YAML")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--criterion", action="append", default=[])
    parser.add_argument("--selected_at", type=str, default=str(date.today()))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    run_dir = Path(args.run_dir)
    output_path = Path(args.output) if args.output else run_dir / "selection.yaml"

    checkpoint_path = run_dir / "best_model" / "model.pt"
    train_history_path = run_dir / "train_history.json"
    val_metrics_path = run_dir / "val_metrics.json"
    calibration_report_path = run_dir / "calibration_report.json"

    with open(val_metrics_path, encoding="utf-8") as f:
        val_metrics = json.load(f)
    with open(calibration_report_path, encoding="utf-8") as f:
        calibration = json.load(f)

    payload = {
        "selected_at": args.selected_at,
        "selected_run": _relative_path(run_dir, repo_root),
        "checkpoint_path": _relative_path(checkpoint_path, repo_root),
        "training_history_path": _relative_path(train_history_path, repo_root),
        "val_metrics_path": _relative_path(val_metrics_path, repo_root),
        "calibration_report_path": _relative_path(calibration_report_path, repo_root),
        "selection_reason": {
            "summary": args.summary,
            "criteria": args.criterion,
        },
        "metrics_snapshot": {
            "val_auc": val_metrics.get("auc"),
            "val_f1": val_metrics.get("f1"),
            "eval_raw_auc": calibration["evaluation"]["raw"]["auc"],
            "best_raw_f1": calibration["evaluation"]["best_thresholds"]["raw"]["f1"],
            "best_raw_threshold": calibration["evaluation"]["best_thresholds"]["raw"]["threshold"],
            "best_temp_f1": calibration["evaluation"]["best_thresholds"]["temperature_scaled"]["f1"],
            "best_temp_threshold": calibration["evaluation"]["best_thresholds"]["temperature_scaled"]["threshold"],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
