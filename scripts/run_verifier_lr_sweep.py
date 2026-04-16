"""
Run a verifier learning-rate sweep and summarize results.

This script:
  1. reads the LR list from configs/vcsr.yaml
  2. derives per-run configs from a base verifier config
  3. trains each verifier run into its own output directory
  4. runs the clean calibration protocol for each checkpoint
  5. writes an aggregate summary JSON/Markdown report

Usage:
    python scripts/run_verifier_lr_sweep.py
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


def format_lr_tag(value: float) -> str:
    text = f"{value:.0e}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def run_step(args: list[str]) -> None:
    logger.info("Running: %s", " ".join(args))
    subprocess.run(args, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    vcsr_config_path = REPO_ROOT / "configs" / "vcsr.yaml"
    base_config_path = REPO_ROOT / "configs" / "verifier_full.yaml"
    sweep_root = REPO_ROOT / "results" / "verifier" / "lr_sweep"
    config_root = sweep_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)

    with open(vcsr_config_path, encoding="utf-8") as f:
        vcsr_cfg = yaml.safe_load(f)
    with open(base_config_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    lr_values = [float(x) for x in vcsr_cfg.get("verifier", {}).get("lr_sweep", [])]
    if not lr_values:
        raise ValueError(f"No lr_sweep values found in {vcsr_config_path}")

    results = []

    for lr in lr_values:
        tag = format_lr_tag(lr)
        run_name = f"lr_{tag}"
        output_dir = sweep_root / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = json.loads(json.dumps(base_cfg))
        cfg["experiment"]["name"] = f"verifier_{run_name}"
        cfg["training"]["learning_rate"] = lr
        cfg["output"]["dir"] = str(output_dir.relative_to(REPO_ROOT)).replace("/", "\\")

        config_path = config_root / f"{run_name}.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        logger.info("Prepared config for %s at %s", run_name, config_path)

        train_cmd = [sys.executable, "-u", "scripts/train_verifier.py", "--config", str(config_path.relative_to(REPO_ROOT))]
        run_step(train_cmd)

        calibrate_cmd = [sys.executable, "-u", "scripts/calibrate_verifier.py", "--config", str(config_path.relative_to(REPO_ROOT))]
        run_step(calibrate_cmd)

        val_metrics_path = output_dir / "val_metrics.json"
        calibration_report_path = output_dir / "calibration_report.json"
        with open(val_metrics_path, encoding="utf-8") as f:
            val_metrics = json.load(f)
        with open(calibration_report_path, encoding="utf-8") as f:
            calib = json.load(f)

        record = {
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
            "eval_temp_log_loss": calib["evaluation"]["temperature_scaled"]["log_loss"],
            "eval_temp_ece": calib["evaluation"]["temperature_scaled"]["ece_10bin"],
            "eval_iso_log_loss": calib["evaluation"]["isotonic"]["log_loss"],
            "eval_iso_ece": calib["evaluation"]["isotonic"]["ece_10bin"],
            "best_raw_threshold": calib["evaluation"]["best_thresholds"]["raw"]["threshold"],
            "best_raw_f1": calib["evaluation"]["best_thresholds"]["raw"]["f1"],
            "best_temp_threshold": calib["evaluation"]["best_thresholds"]["temperature_scaled"]["threshold"],
            "best_temp_f1": calib["evaluation"]["best_thresholds"]["temperature_scaled"]["f1"],
            "best_iso_threshold": calib["evaluation"]["best_thresholds"]["isotonic"]["threshold"],
            "best_iso_f1": calib["evaluation"]["best_thresholds"]["isotonic"]["f1"],
        }
        results.append(record)

    def key_fn(item: dict) -> tuple[float, float]:
        return (item["eval_raw_auc"], item["best_raw_f1"])

    results_sorted = sorted(results, key=key_fn, reverse=True)

    summary = {
        "source_configs": {
            "vcsr": str(vcsr_config_path.relative_to(REPO_ROOT)).replace("/", "\\"),
            "base_verifier": str(base_config_path.relative_to(REPO_ROOT)).replace("/", "\\"),
        },
        "runs": results_sorted,
        "best_by_eval_raw_auc": results_sorted[0] if results_sorted else None,
    }

    with open(sweep_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Verifier LR Sweep",
        "",
        "| Run | LR | Val AUC | Val F1 | Eval Raw AUC | Best Raw F1 | Best Raw Tau | Temp ECE | Iso ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results_sorted:
        lines.append(
            f"| {row['run_name']} | {row['learning_rate']:.1e} | {row['val_auc']:.4f} | {row['val_f1']:.4f} | "
            f"{row['eval_raw_auc']:.4f} | {row['best_raw_f1']:.4f} | {row['best_raw_threshold']:.2f} | "
            f"{row['eval_temp_ece']:.4f} | {row['eval_iso_ece']:.4f} |"
        )

    with open(sweep_root / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Sweep complete. Summary written to %s", sweep_root / "summary.json")


if __name__ == "__main__":
    main()
