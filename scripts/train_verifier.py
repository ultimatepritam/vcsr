"""
Train the cross-encoder verifier on neggen JSONL data.

Usage:
    python scripts/train_verifier.py --config configs/verifier.yaml
    python scripts/train_verifier.py --config configs/verifier.yaml --dry_run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vcsr_env  # noqa: F401
import yaml
import torch
from transformers import AutoTokenizer

from verifier.dataset import build_datasets, collate_fn
from verifier.evaluate import evaluate
from verifier.model import VerifierModel
from verifier.train import TrainConfig, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train VCSR semantic verifier")
    parser.add_argument("--config", type=str, default="configs/verifier.yaml")
    parser.add_argument("--dry_run", action="store_true", help="1 epoch, small subset")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("experiment", {}).get("seed", 42)
    torch.manual_seed(seed)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    out_cfg = cfg.get("output", {})
    log_cfg = cfg.get("logging", {})

    backbone = model_cfg.get("backbone", "microsoft/deberta-v3-base")
    max_seq_len = model_cfg.get("max_seq_len", 512)

    # Tokenizer
    revision = model_cfg.get("revision")
    logger.info("Loading tokenizer: %s (revision=%s)", backbone, revision)
    tokenizer = AutoTokenizer.from_pretrained(backbone, revision=revision)

    # Datasets
    jsonl_path = data_cfg.get("train_jsonl", "results/neggen/pilot/verifier_train.relabeled.jsonl")
    val_fraction = data_cfg.get("val_fraction", 0.15)

    train_ds, val_ds, train_rows, val_rows = build_datasets(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        max_length=max_seq_len,
        val_fraction=val_fraction,
        seed=seed,
        filter_unparseable=data_cfg.get("filter_unparseable", True),
    )
    logger.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    # Dry-run: tiny subset
    if args.dry_run:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(64, len(train_ds)))))
        val_ds_eval = Subset(val_ds, list(range(min(32, len(val_ds)))))
        val_rows_eval = val_rows[:min(32, len(val_rows))]
        train_cfg["epochs"] = 1
        train_cfg["log_every_n_steps"] = 2
        logger.info("=== DRY RUN: train=%d val=%d, 1 epoch ===", len(train_ds), len(val_ds_eval))
    else:
        val_ds_eval = val_ds
        val_rows_eval = val_rows

    # Model
    logger.info("Initializing model: %s (dropout=%.2f, revision=%s)", backbone, model_cfg.get("dropout", 0.1), revision)
    model = VerifierModel(
        backbone_name=backbone,
        dropout=model_cfg.get("dropout", 0.1),
        revision=revision,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %s total, %s trainable", f"{n_params:,}", f"{n_trainable:,}")

    # Training config
    output_dir = out_cfg.get("dir", "results/verifier/pilot")
    tc = TrainConfig(
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        epochs=train_cfg.get("epochs", 5),
        batch_size=train_cfg.get("batch_size", 16),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        fp16=train_cfg.get("fp16", True),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 2),
        early_stopping_metric=train_cfg.get("early_stopping_metric", "val_auc"),
        eval_steps=eval_cfg.get("eval_steps", 0),
        log_every_n_steps=log_cfg.get("log_every_n_steps", 1),
        dataloader_workers=train_cfg.get("dataloader_workers", 0),
        save_best_model=out_cfg.get("save_best_model", True),
        save_last_model=out_cfg.get("save_last_model", False),
        output_dir=output_dir,
        wandb_project=log_cfg.get("wandb_project"),
    )

    # Train
    state = train(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds_eval,
        config=tc,
        val_rows=val_rows_eval,
    )

    # Save training history
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "train_history.json", "w") as f:
        json.dump(state.history, f, indent=2)

    # Save config snapshot
    with open(out_path / "train_config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    logger.info(
        "Training complete — best %s=%.4f at epoch %d",
        tc.early_stopping_metric, state.best_metric, state.best_epoch,
    )
    logger.info("Outputs: %s", out_path)

    # Final evaluation with best model
    if tc.save_best_model and (out_path / "best_model" / "model.pt").exists():
        logger.info("Loading best model for final evaluation...")
        model.load_state_dict(torch.load(out_path / "best_model" / "model.pt", weights_only=True))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        from torch.utils.data import DataLoader
        final_loader = DataLoader(
            val_ds_eval,
            batch_size=tc.batch_size * 2,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        final_metrics = evaluate(model, final_loader, device, val_rows=val_rows_eval)
        with open(out_path / "val_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2, default=str)
        logger.info("Final val metrics saved to %s", out_path / "val_metrics.json")


if __name__ == "__main__":
    main()
