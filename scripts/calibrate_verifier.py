"""
Fit post-hoc calibration on a dedicated calibration split and evaluate on a
separate held-out evaluation split.

This script uses the verifier's existing validation pool, then splits that pool
by template group into:
  - calibration subset: fit temperature / isotonic mapping
  - evaluation subset: report threshold sweeps and risk-coverage curves

Usage:
    python scripts/calibrate_verifier.py --config configs/verifier_full.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vcsr_env  # noqa: F401
import numpy as np
import torch
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from verifier.dataset import VerifierDataset, VerifierRow, build_datasets, collate_fn
from verifier.model import VerifierModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def split_rows_by_template(
    rows: list[VerifierRow],
    fraction: float,
    seed: int,
) -> tuple[list[VerifierRow], list[VerifierRow]]:
    groups: dict[str, list[VerifierRow]] = defaultdict(list)
    for row in rows:
        groups[row.planetarium_name].append(row)

    keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    target = int(len(rows) * fraction)
    left: list[VerifierRow] = []
    right: list[VerifierRow] = []
    accumulated = 0

    for key in keys:
        group = groups[key]
        if accumulated < target:
            left.extend(group)
            accumulated += len(group)
        else:
            right.extend(group)

    return left, right


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(set(y_true.tolist())) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    result = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_score >= lo) & (y_score <= hi)
        else:
            mask = (y_score >= lo) & (y_score < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_score[mask]))
        result += (np.sum(mask) / total) * abs(acc - conf)
    return float(result)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    best_t = 1.0
    best_loss = math.inf
    for t in np.linspace(0.5, 5.0, 91):
        probs = sigmoid(logits / t)
        loss = log_loss(labels, probs, labels=[0, 1])
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_pred.mean()),
    }


def summarize_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "negatives": int(len(y_true) - y_true.sum()),
        "auc": safe_auc(y_true, y_score),
        "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])),
        "ece_10bin": ece(y_true, y_score, n_bins=10),
    }


def best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray, thresholds: list[float]) -> tuple[dict, list[dict]]:
    sweep = [binary_metrics(y_true, y_score, t) for t in thresholds]
    best = max(sweep, key=lambda x: x["f1"])
    return best, sweep


def risk_coverage_curve(y_true: np.ndarray, y_score: np.ndarray, thresholds: list[float]) -> list[dict]:
    curve = []
    n = len(y_true)
    for tau in thresholds:
        accepted = y_score >= tau
        coverage = float(np.mean(accepted))
        accepted_n = int(np.sum(accepted))
        if accepted_n == 0:
            curve.append(
                {
                    "threshold": float(tau),
                    "coverage": coverage,
                    "accepted_n": accepted_n,
                    "selective_accuracy": None,
                    "selective_risk": None,
                    "selective_f1": None,
                    "selective_precision": None,
                    "selective_recall": None,
                }
            )
            continue

        yt = y_true[accepted]
        ys = y_score[accepted]
        ypred = (ys >= tau).astype(int)
        acc = float(accuracy_score(yt, ypred))
        curve.append(
            {
                "threshold": float(tau),
                "coverage": coverage,
                "accepted_n": accepted_n,
                "rejected_n": int(n - accepted_n),
                "selective_accuracy": acc,
                "selective_risk": float(1.0 - acc),
                "selective_f1": float(f1_score(yt, ypred, zero_division=0)),
                "selective_precision": float(precision_score(yt, ypred, zero_division=0)),
                "selective_recall": float(recall_score(yt, ypred, zero_division=0)),
            }
        )
    return curve


@torch.no_grad()
def collect_logits(
    model: VerifierModel,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: list[float] = []
    all_labels: list[float] = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast("cuda", enabled=device.type == "cuda"):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                labels=batch["labels"],
            )
        all_logits.extend(out["logits"].detach().cpu().numpy().tolist())
        all_labels.extend(batch["labels"].detach().cpu().numpy().tolist())
    return np.array(all_logits, dtype=float), np.array(all_labels, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate verifier on separate split")
    parser.add_argument("--config", type=str, default="configs/verifier_full.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--calibration_seed", type=int, default=123)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("experiment", {}).get("seed", 42)
    torch.manual_seed(seed)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    out_cfg = cfg.get("output", {})

    revision = model_cfg.get("revision")
    backbone = model_cfg.get("backbone", "microsoft/deberta-v3-base")
    max_seq_len = model_cfg.get("max_seq_len", 512)
    backbone_source = vcsr_env.resolve_hf_snapshot(backbone, revision=revision) or backbone

    logger.info("Loading tokenizer/model from %s", backbone_source)
    tokenizer = AutoTokenizer.from_pretrained(
        str(backbone_source),
        revision=revision,
        local_files_only=True,
        use_fast=False,
    )

    _, val_ds, _, val_rows = build_datasets(
        jsonl_path=data_cfg.get("train_jsonl", "results/neggen/pilot/verifier_train.relabeled.jsonl"),
        tokenizer=tokenizer,
        max_length=max_seq_len,
        val_fraction=data_cfg.get("val_fraction", 0.15),
        seed=seed,
        filter_unparseable=data_cfg.get("filter_unparseable", True),
    )
    logger.info("Validation pool size before calibration split: %d", len(val_rows))

    calib_rows, eval_rows = split_rows_by_template(
        val_rows,
        fraction=args.calibration_fraction,
        seed=args.calibration_seed,
    )
    logger.info("Calibration rows: %d | Evaluation rows: %d", len(calib_rows), len(eval_rows))

    calib_ds = VerifierDataset(calib_rows, tokenizer, max_length=max_seq_len)
    eval_ds = VerifierDataset(eval_rows, tokenizer, max_length=max_seq_len)

    model = VerifierModel(
        backbone_name=str(backbone_source),
        dropout=model_cfg.get("dropout", 0.1),
        revision=revision,
    )
    output_dir = Path(out_cfg.get("dir", "results/verifier/full_run"))
    model_path = Path(args.model_path) if args.model_path else output_dir / "best_model" / "model.pt"
    logger.info("Loading checkpoint: %s", model_path)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = train_cfg.get("batch_size", 4) * 2
    calib_loader = DataLoader(calib_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    calib_logits, calib_labels = collect_logits(model, calib_loader, device)
    eval_logits, eval_labels = collect_logits(model, eval_loader, device)

    calib_raw = sigmoid(calib_logits)
    eval_raw = sigmoid(eval_logits)

    temperature = fit_temperature(calib_logits, calib_labels)
    eval_temp = sigmoid(eval_logits / temperature)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(calib_raw, calib_labels)
    eval_iso = np.clip(iso.predict(eval_raw), 1e-6, 1.0 - 1e-6)

    threshold_grid = [round(x, 2) for x in np.linspace(0.05, 0.95, 19).tolist()]

    raw_best, raw_sweep = best_threshold_by_f1(eval_labels, eval_raw, threshold_grid)
    temp_best, temp_sweep = best_threshold_by_f1(eval_labels, eval_temp, threshold_grid)
    iso_best, iso_sweep = best_threshold_by_f1(eval_labels, eval_iso, threshold_grid)

    summary = {
        "protocol": {
            "config": args.config,
            "model_path": str(model_path),
            "calibration_fraction": args.calibration_fraction,
            "calibration_seed": args.calibration_seed,
            "note": "Calibration is fit on the calibration split only and reported on the separate evaluation split.",
        },
        "split_summary": {
            "validation_pool_n": len(val_rows),
            "calibration_n": len(calib_rows),
            "evaluation_n": len(eval_rows),
            "calibration_positives": int(sum(r.label for r in calib_rows)),
            "evaluation_positives": int(sum(r.label for r in eval_rows)),
        },
        "calibration_fit": {
            "raw_calibration_summary": summarize_scores(calib_labels, calib_raw),
            "temperature": temperature,
        },
        "evaluation": {
            "raw": summarize_scores(eval_labels, eval_raw),
            "temperature_scaled": summarize_scores(eval_labels, eval_temp),
            "isotonic": summarize_scores(eval_labels, eval_iso),
            "best_thresholds": {
                "raw": raw_best,
                "temperature_scaled": temp_best,
                "isotonic": iso_best,
            },
            "threshold_sweeps": {
                "raw": raw_sweep,
                "temperature_scaled": temp_sweep,
                "isotonic": iso_sweep,
            },
            "risk_coverage_curves": {
                "raw": risk_coverage_curve(eval_labels, eval_raw, threshold_grid),
                "temperature_scaled": risk_coverage_curve(eval_labels, eval_temp, threshold_grid),
                "isotonic": risk_coverage_curve(eval_labels, eval_iso, threshold_grid),
            },
        },
        "notes": [
            "This is a cleaner protocol than fitting calibration on the same split used for threshold selection.",
            "For final reporting, consider nested validation or a dedicated dev/calibration split from the start.",
        ],
    }

    row_dump = []
    for row, raw, temp, iso_score in zip(eval_rows, eval_raw, eval_temp, eval_iso):
        row_dump.append(
            {
                "planetarium_name": row.planetarium_name,
                "domain": row.domain,
                "source": row.source,
                "label": int(row.label),
                "perturbation_type": row.perturbation_type,
                "raw_score": float(raw),
                "temperature_scaled_score": float(temp),
                "isotonic_score": float(iso_score),
            }
        )

    with open(output_dir / "calibration_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "calibration_eval_scores.jsonl", "w", encoding="utf-8") as f:
        for row in row_dump:
            f.write(json.dumps(row) + "\n")

    logger.info("Saved calibration report to %s", output_dir / "calibration_report.json")
    logger.info("Best raw threshold on eval split: %s", json.dumps(raw_best))
    logger.info("Best temperature-scaled threshold on eval split: %s", json.dumps(temp_best))
    logger.info("Best isotonic threshold on eval split: %s", json.dumps(iso_best))


if __name__ == "__main__":
    main()
