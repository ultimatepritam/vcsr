"""
Run a single capacity-push verifier sweep on the current hard-negative setup.

This sweep keeps the training data fixed and varies only learning rate while
holding:
  - epochs = 12
  - early_stopping_patience = 3
  - effective batch size = 128

It trains each run, performs clean calibration analysis, and writes a summary.

Usage:
    python scripts/run_verifier_capacity_push.py
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent.parent
LR_VALUES = [2.0e-5, 5.0e-5, 7.5e-5]


def format_lr_tag(value: float) -> str:
    text = f"{value:.1e}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def run_step(args: list[str]) -> None:
    logger.info("Running: %s", " ".join(args))
    subprocess.run(args, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    base_config_path = REPO_ROOT / "configs" / "verifier_capacity_push.yaml"
    sweep_root = REPO_ROOT / "results" / "verifier" / "capacity_push"
    config_root = sweep_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)

    with open(base_config_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    results = []

    for lr in LR_VALUES:
        tag = format_lr_tag(lr)
        run_name = f"lr_{tag}"
        output_dir = sweep_root / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = json.loads(json.dumps(base_cfg))
        cfg["experiment"]["name"] = f"verifier_capacity_push_{run_name}"
        cfg["training"]["learning_rate"] = lr
        cfg["output"]["dir"] = str(output_dir.relative_to(REPO_ROOT)).replace("/", "\\")

        config_path = config_root / f"{run_name}.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        train_cmd = [
            sys.executable,
            "-u",
            "scripts/train_verifier.py",
            "--config",
            str(config_path.relative_to(REPO_ROOT)),
        ]
        run_step(train_cmd)

        calibrate_cmd = [
            sys.executable,
            "-u",
            "scripts/calibrate_verifier.py",
            "--config",
            str(config_path.relative_to(REPO_ROOT)),
        ]
        run_step(calibrate_cmd)

        val_metrics_path = output_dir / "val_metrics.json"
        calibration_report_path = output_dir / "calibration_report.json"
        with open(val_metrics_path, encoding="utf-8") as f:
            val_metrics = json.load(f)
        with open(calibration_report_path, encoding="utf-8") as f:
            calib = json.load(f)

        results.append(
            {
                "run_name": run_name,
                "learning_rate": lr,
                "output_dir": str(output_dir.relative_to(REPO_ROOT)).replace("/", "\\"),
                "val_auc": val_metrics.get("auc"),
                "val_accuracy": val_metrics.get("accuracy"),
                "val_f1": val_metrics.get("f1"),
                "val_precision": val_metrics.get("precision"),
                "val_recall": val_metrics.get("recall"),
                "eval_raw_auc": calib["evaluation"]["raw"]["auc"],
                "eval_raw_log_loss": calib["evaluation"]["raw"]["log_loss"],
                "eval_raw_ece": calib["evaluation"]["raw"]["ece_10bin"],
                "best_raw_threshold": calib["evaluation"]["best_thresholds"]["raw"]["threshold"],
                "best_raw_f1": calib["evaluation"]["best_thresholds"]["raw"]["f1"],
                "best_temp_threshold": calib["evaluation"]["best_thresholds"]["temperature_scaled"]["threshold"],
                "best_temp_f1": calib["evaluation"]["best_thresholds"]["temperature_scaled"]["f1"],
                "best_iso_threshold": calib["evaluation"]["best_thresholds"]["isotonic"]["threshold"],
                "best_iso_f1": calib["evaluation"]["best_thresholds"]["isotonic"]["f1"],
            }
        )

    results_sorted = sorted(
        results,
        key=lambda item: (item["eval_raw_auc"], item["best_raw_f1"]),
        reverse=True,
    )

    summary = {
        "source_config": str(base_config_path.relative_to(REPO_ROOT)).replace("/", "\\"),
        "sweep": {
            "learning_rates": LR_VALUES,
            "epochs": base_cfg["training"]["epochs"],
            "early_stopping_patience": base_cfg["training"]["early_stopping_patience"],
            "batch_size": base_cfg["training"]["batch_size"],
            "gradient_accumulation_steps": base_cfg["training"]["gradient_accumulation_steps"],
            "effective_batch_size": base_cfg["training"]["batch_size"] * base_cfg["training"]["gradient_accumulation_steps"],
        },
        "runs": results_sorted,
        "best_by_eval_raw_auc": results_sorted[0] if results_sorted else None,
    }

    with open(sweep_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Verifier Capacity Push Sweep",
        "",
        "| Run | LR | Val AUC | Val F1 | Val Recall | Eval Raw AUC | Best Raw F1 | Best Raw Tau |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results_sorted:
        lines.append(
            f"| {row['run_name']} | {row['learning_rate']:.2e} | {row['val_auc']:.4f} | {row['val_f1']:.4f} | "
            f"{row['val_recall']:.4f} | {row['eval_raw_auc']:.4f} | {row['best_raw_f1']:.4f} | "
            f"{row['best_raw_threshold']:.2f} |"
        )

    with open(sweep_root / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Capacity-push sweep complete. Summary written to %s", sweep_root / "summary.json")


if __name__ == "__main__":
    main()
