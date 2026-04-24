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
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vcsr_env  # noqa: F401
import yaml
import torch
from transformers import AutoTokenizer

from verifier.dataset import (
    PairwiseVerifierDataset,
    VerifierDataset,
    build_datasets,
    collate_fn,
    collate_pairwise_fn,
    load_jsonl,
    load_pairwise_jsonl,
    split_by_template,
    split_pairwise_by_template,
)
from verifier.evaluate import evaluate
from verifier.model import VerifierModel
from verifier.train import TrainConfig, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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


def _write_process_info(output_dir: Path, command: list[str]) -> None:
    info = {
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": " ".join(command),
        "output_dir": str(output_dir),
        "progress_log": str(output_dir / "progress.log"),
        "progress_json": str(output_dir / "progress.json"),
    }
    with open(output_dir / "process_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train VCSR semantic verifier")
    parser.add_argument("--config", type=str, default="configs/verifier.yaml")
    parser.add_argument("--dry_run", action="store_true", help="1 epoch, small subset")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output.dir from config")
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
    if args.output_dir:
        out_cfg["dir"] = args.output_dir
    output_dir = out_cfg.get("dir", "results/verifier/pilot")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = _configure_file_logging(out_path)
    _write_process_info(out_path, sys.argv)
    logger.info("Live training log: %s", log_path)

    revision = model_cfg.get("revision")
    backbone = model_cfg.get("backbone", "microsoft/deberta-v3-base")
    max_seq_len = model_cfg.get("max_seq_len", 512)
    backbone_source = vcsr_env.resolve_hf_snapshot(backbone, revision=revision) or backbone

    # Tokenizer
    logger.info(
        "Loading tokenizer: %s (revision=%s, source=%s)",
        backbone,
        revision,
        backbone_source,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(backbone_source),
        revision=revision,
        local_files_only=True,
        use_fast=False,
    )

    # Datasets
    jsonl_path = data_cfg.get("train_jsonl", "results/neggen/pilot/verifier_train.relabeled.jsonl")
    extra_train_jsonl = data_cfg.get("extra_train_jsonl")
    extra_train_repeat = int(data_cfg.get("extra_train_repeat", 1))
    pairwise_train_jsonl = data_cfg.get("pairwise_train_jsonl")
    pairwise_val_jsonl = data_cfg.get("pairwise_val_jsonl")
    val_fraction = data_cfg.get("val_fraction", 0.15)
    pairwise_val_fraction = float(data_cfg.get("pairwise_val_fraction", val_fraction))
    filter_unparseable = data_cfg.get("filter_unparseable", True)

    if extra_train_jsonl:
        base_rows = load_jsonl(jsonl_path, filter_unparseable=filter_unparseable)
        train_rows, val_rows = split_by_template(base_rows, val_fraction=val_fraction, seed=seed)
        val_groups = {row.planetarium_name for row in val_rows}
        extra_rows = load_jsonl(extra_train_jsonl, filter_unparseable=filter_unparseable)
        extra_rows = [row for row in extra_rows if row.planetarium_name not in val_groups]
        if extra_train_repeat > 1:
            extra_rows = extra_rows * extra_train_repeat
        train_rows = train_rows + extra_rows
        train_ds = VerifierDataset(train_rows, tokenizer, max_length=max_seq_len)
        val_ds = VerifierDataset(val_rows, tokenizer, max_length=max_seq_len)
        logger.info(
            "Loaded base split plus extra training-only rows: base_train=%d extra_train=%d val=%d (extra_repeat=%d)",
            len(train_rows) - len(extra_rows),
            len(extra_rows),
            len(val_rows),
            max(1, extra_train_repeat),
        )
    else:
        train_ds, val_ds, train_rows, val_rows = build_datasets(
            jsonl_path=jsonl_path,
            tokenizer=tokenizer,
            max_length=max_seq_len,
            val_fraction=val_fraction,
            seed=seed,
            filter_unparseable=filter_unparseable,
        )
    logger.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    pairwise_train_ds = None
    pairwise_val_ds = None
    pairwise_val_rows = []
    if pairwise_train_jsonl:
        pairwise_rows = load_pairwise_jsonl(pairwise_train_jsonl)
        if pairwise_val_jsonl:
            pair_train_rows = pairwise_rows
            pair_val_rows = load_pairwise_jsonl(pairwise_val_jsonl)
            logger.info(
                "Using explicit pairwise validation file: train=%s val=%s",
                pairwise_train_jsonl,
                pairwise_val_jsonl,
            )
        else:
            pair_train_rows, pair_val_rows = split_pairwise_by_template(
                pairwise_rows,
                val_fraction=pairwise_val_fraction,
                seed=seed,
            )
        pairwise_train_ds = PairwiseVerifierDataset(pair_train_rows, tokenizer, max_length=max_seq_len)
        pairwise_val_ds = PairwiseVerifierDataset(pair_val_rows, tokenizer, max_length=max_seq_len)
        pairwise_val_rows = pair_val_rows
        logger.info(
            "Pairwise train: %d  Pairwise val: %d",
            len(pairwise_train_ds),
            len(pairwise_val_ds),
        )

    # Dry-run: tiny subset
    if args.dry_run:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(64, len(train_ds)))))
        val_ds_eval = Subset(val_ds, list(range(min(32, len(val_ds)))))
        val_rows_eval = val_rows[:min(32, len(val_rows))]
        if pairwise_train_ds is not None:
            pairwise_train_ds = Subset(pairwise_train_ds, list(range(min(64, len(pairwise_train_ds)))))
        if pairwise_val_ds is not None:
            pairwise_val_ds = Subset(pairwise_val_ds, list(range(min(32, len(pairwise_val_ds)))))
            pairwise_val_rows = pairwise_val_rows[:min(32, len(pairwise_val_rows))]
        train_cfg["epochs"] = 1
        train_cfg["log_every_n_steps"] = 2
        logger.info("=== DRY RUN: train=%d val=%d, 1 epoch ===", len(train_ds), len(val_ds_eval))
    else:
        val_ds_eval = val_ds
        val_rows_eval = val_rows

    # Model
    logger.info(
        "Initializing model: %s (dropout=%.2f, revision=%s, source=%s)",
        backbone,
        model_cfg.get("dropout", 0.1),
        revision,
        backbone_source,
    )
    model = VerifierModel(
        backbone_name=str(backbone_source),
        dropout=model_cfg.get("dropout", 0.1),
        revision=revision,
    )
    init_selection_path = model_cfg.get("init_selection_path")
    init_checkpoint_path = model_cfg.get("init_checkpoint_path")
    if init_checkpoint_path is None and init_selection_path:
        with open(init_selection_path, encoding="utf-8") as f:
            init_metadata = yaml.safe_load(f)
        init_checkpoint_path = init_metadata.get("checkpoint_path")
    if init_checkpoint_path:
        logger.info("Warm-starting verifier weights from %s", init_checkpoint_path)
        model.load_state_dict(torch.load(init_checkpoint_path, weights_only=True))
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %s total, %s trainable", f"{n_params:,}", f"{n_trainable:,}")

    # Training config
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
        objective=train_cfg.get("objective", "pointwise"),
        pairwise_loss_weight=train_cfg.get("pairwise_loss_weight", 1.0),
        pointwise_loss_weight=train_cfg.get("pointwise_loss_weight", 0.5),
    )

    # Train
    state = train(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds_eval,
        config=tc,
        val_rows=val_rows_eval,
        pairwise_train_ds=pairwise_train_ds,
        pairwise_val_ds=pairwise_val_ds,
    )

    # Save training history
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
        if pairwise_val_ds is not None and len(pairwise_val_ds) > 0:
            from verifier.evaluate import evaluate_pairwise
            pair_loader = DataLoader(
                pairwise_val_ds,
                batch_size=tc.batch_size * 2,
                shuffle=False,
                collate_fn=collate_pairwise_fn,
                num_workers=0,
            )
            final_metrics["pairwise"] = evaluate_pairwise(model, pair_loader, device)
        with open(out_path / "val_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2, default=str)
        logger.info("Final val metrics saved to %s", out_path / "val_metrics.json")


if __name__ == "__main__":
    main()
