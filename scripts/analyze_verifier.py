"""
Analyze a trained verifier on the held-out validation split.

This script is meant for post-training diagnosis:
  - dump per-example scores/logits
  - sweep decision thresholds
  - fit simple post-hoc calibration models for inspection

Usage:
    python scripts/analyze_verifier.py --config configs/verifier_full.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
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

from verifier.dataset import build_datasets, collate_fn
from verifier.model import VerifierModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(set(y_true.tolist())) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_pred.mean()),
    }


def _ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0
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
        ece += (np.sum(mask) / total) * abs(acc - conf)
    return float(ece)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    best_t = 1.0
    best_loss = math.inf
    for t in np.linspace(0.5, 5.0, 91):
        probs = _sigmoid(logits / t)
        loss = log_loss(labels, probs, labels=[0, 1])
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t


@torch.no_grad()
def collect_predictions(model: VerifierModel, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
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
        logits = out["logits"].detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()
        all_logits.extend(logits.tolist())
        all_labels.extend(labels.tolist())
    return np.array(all_logits, dtype=float), np.array(all_labels, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze trained verifier outputs")
    parser.add_argument("--config", type=str, default="configs/verifier_full.yaml")
    parser.add_argument("--model_path", type=str, default=None)
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

    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 4) * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    logits, labels = collect_predictions(model, val_loader, device)
    raw_scores = _sigmoid(logits)

    base_summary = {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "negatives": int(len(labels) - labels.sum()),
        "auc": _safe_auc(labels, raw_scores),
        "log_loss": float(log_loss(labels, raw_scores, labels=[0, 1])),
        "ece_10bin": _ece(labels, raw_scores, n_bins=10),
    }

    threshold_grid = np.round(np.linspace(0.05, 0.95, 19), 2)
    raw_thresholds = [_binary_metrics(labels, raw_scores, float(t)) for t in threshold_grid]
    best_raw_f1 = max(raw_thresholds, key=lambda x: x["f1"])

    temperature = _fit_temperature(logits, labels)
    temp_scores = _sigmoid(logits / temperature)
    temp_summary = {
        "temperature": temperature,
        "auc": _safe_auc(labels, temp_scores),
        "log_loss": float(log_loss(labels, temp_scores, labels=[0, 1])),
        "ece_10bin": _ece(labels, temp_scores, n_bins=10),
    }
    temp_thresholds = [_binary_metrics(labels, temp_scores, float(t)) for t in threshold_grid]
    best_temp_f1 = max(temp_thresholds, key=lambda x: x["f1"])

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_scores, labels)
    iso_scores = np.clip(iso.predict(raw_scores), 1e-6, 1.0 - 1e-6)
    iso_summary = {
        "auc": _safe_auc(labels, iso_scores),
        "log_loss": float(log_loss(labels, iso_scores, labels=[0, 1])),
        "ece_10bin": _ece(labels, iso_scores, n_bins=10),
    }
    iso_thresholds = [_binary_metrics(labels, iso_scores, float(t)) for t in threshold_grid]
    best_iso_f1 = max(iso_thresholds, key=lambda x: x["f1"])

    score_dump = []
    for row, logit, raw, temp, iso_score in zip(val_rows, logits, raw_scores, temp_scores, iso_scores):
        score_dump.append(
            {
                "planetarium_name": row.planetarium_name,
                "domain": row.domain,
                "source": row.source,
                "label": int(row.label),
                "perturbation_type": row.perturbation_type,
                "logit": float(logit),
                "raw_score": float(raw),
                "temp_score": float(temp),
                "isotonic_score": float(iso_score),
            }
        )

    summary = {
        "base": base_summary,
        "best_raw_f1": best_raw_f1,
        "temperature_scaling": temp_summary,
        "best_temp_f1": best_temp_f1,
        "isotonic": iso_summary,
        "best_isotonic_f1": best_iso_f1,
        "threshold_sweeps": {
            "raw": raw_thresholds,
            "temperature_scaled": temp_thresholds,
            "isotonic": iso_thresholds,
        },
        "notes": [
            "Calibration here is exploratory because it is fit and evaluated on the same validation split.",
            "Use a separate calibration split or nested validation before reporting final selective-risk claims.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "score_analysis.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "val_scores.jsonl", "w", encoding="utf-8") as f:
        for row in score_dump:
            f.write(json.dumps(row) + "\n")

    logger.info("Base summary: %s", json.dumps(base_summary))
    logger.info("Best raw F1 threshold: %s", json.dumps(best_raw_f1))
    logger.info("Temperature scaling: %s", json.dumps(temp_summary))
    logger.info("Best temperature-scaled F1 threshold: %s", json.dumps(best_temp_f1))
    logger.info("Isotonic summary: %s", json.dumps(iso_summary))
    logger.info("Best isotonic F1 threshold: %s", json.dumps(best_iso_f1))
    logger.info("Saved analysis to %s", output_dir / "score_analysis.json")


if __name__ == "__main__":
    main()
