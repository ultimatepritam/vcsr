"""
Evaluation utilities for the cross-encoder verifier.

Computes AUC, accuracy, F1, precision, recall — overall and stratified by
source (gold / llm / perturbation) and domain (blocksworld / gripper).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.amp import autocast
from torch.utils.data import DataLoader

from verifier.model import VerifierModel

logger = logging.getLogger(__name__)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(set(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "auc": _safe_auc(y_true, y_score),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n": len(y_true),
    }


@torch.no_grad()
def evaluate(
    model: VerifierModel,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    val_rows: Optional[list] = None,
) -> dict:
    """
    Run evaluation. Returns dict with overall metrics, per_source, per_domain.

    If `val_rows` is provided (parallel list of VerifierRow objects matching
    the dataloader order), per-source and per-domain breakdowns are included.
    """
    model.eval()

    all_labels: list[float] = []
    all_scores: list[float] = []
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast("cuda", enabled=device.type == "cuda"):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                labels=batch["labels"],
            )
        total_loss += out["loss"].item()
        n_batches += 1
        probs = torch.sigmoid(out["logits"]).cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        all_scores.extend(probs.tolist())
        all_labels.extend(labels.tolist())

    y_true = np.array(all_labels)
    y_score = np.array(all_scores)

    metrics = _compute_metrics(y_true, y_score, threshold)
    metrics["loss"] = total_loss / max(1, n_batches)

    if val_rows is not None and len(val_rows) == len(y_true):
        by_source: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        by_domain: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))

        for i, row in enumerate(val_rows):
            src = getattr(row, "source", "unknown")
            dom = getattr(row, "domain", "unknown")
            by_source[src][0].append(y_true[i])
            by_source[src][1].append(y_score[i])
            by_domain[dom][0].append(y_true[i])
            by_domain[dom][1].append(y_score[i])

        metrics["per_source"] = {}
        for src, (yt, ys) in by_source.items():
            metrics["per_source"][src] = _compute_metrics(np.array(yt), np.array(ys), threshold)

        metrics["per_domain"] = {}
        for dom, (yt, ys) in by_domain.items():
            metrics["per_domain"][dom] = _compute_metrics(np.array(yt), np.array(ys), threshold)

    return metrics


@torch.no_grad()
def evaluate_pairwise(
    model: VerifierModel,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate whether the model scores positive candidates above negatives."""
    model.eval()

    margins: list[float] = []
    by_domain: dict[str, list[float]] = defaultdict(list)
    by_style: dict[str, list[float]] = defaultdict(list)
    by_k: dict[str, list[float]] = defaultdict(list)
    by_pair_type: dict[str, list[float]] = defaultdict(list)

    for batch in dataloader:
        pos = {k: v.to(device) for k, v in batch["positive"].items()}
        neg = {k: v.to(device) for k, v in batch["negative"].items()}
        with autocast("cuda", enabled=device.type == "cuda"):
            pos_logits = model(**pos)["logits"]
            neg_logits = model(**neg)["logits"]

        batch_margins = (pos_logits - neg_logits).detach().cpu().numpy().tolist()
        margins.extend(batch_margins)

        for margin, domain, style, k_value, pair_type in zip(
            batch_margins,
            batch.get("domain", []),
            batch.get("style", []),
            batch.get("k", []),
            batch.get("pair_type", []),
        ):
            by_domain[str(domain or "unknown")].append(float(margin))
            by_style[str(style or "unknown")].append(float(margin))
            by_k[str(k_value)].append(float(margin))
            by_pair_type[str(pair_type or "unknown")].append(float(margin))

    def summarize(values: list[float]) -> dict:
        arr = np.array(values, dtype=float)
        if len(arr) == 0:
            return {"n": 0, "pairwise_accuracy": 0.0, "mean_margin": 0.0}
        return {
            "n": int(len(arr)),
            "pairwise_accuracy": float((arr > 0).mean()),
            "mean_margin": float(arr.mean()),
        }

    return {
        **summarize(margins),
        "per_domain": {key: summarize(values) for key, values in by_domain.items()},
        "per_style": {key: summarize(values) for key, values in by_style.items()},
        "per_k": {key: summarize(values) for key, values in by_k.items()},
        "per_pair_type": {key: summarize(values) for key, values in by_pair_type.items()},
    }
