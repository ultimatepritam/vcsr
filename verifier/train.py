"""
Training loop for the cross-encoder verifier.

Handles: AdamW + linear warmup, mixed-precision (fp16), gradient accumulation,
early stopping on val AUC, checkpoint saving, optional wandb logging.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from verifier.dataset import VerifierDataset, collate_fn
from verifier.evaluate import evaluate
from verifier.model import VerifierModel

logger = logging.getLogger(__name__)


def _flush_logs():
    for h in logging.root.handlers:
        h.flush()
    sys.stdout.flush()
    sys.stderr.flush()


@dataclass
class TrainConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    epochs: int = 5
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    fp16: bool = True
    early_stopping_patience: int = 2
    early_stopping_metric: str = "val_auc"
    eval_steps: int = 0
    log_every_n_steps: int = 1
    dataloader_workers: int = 0
    save_best_model: bool = True
    save_last_model: bool = False
    output_dir: str = "results/verifier/pilot"
    wandb_project: Optional[str] = None


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    best_epoch: int = 0
    patience_counter: int = 0
    history: list[dict] = field(default_factory=list)


def _get_linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return LambdaLR(optimizer, lr_lambda)


def train(
    model: VerifierModel,
    train_ds: VerifierDataset,
    val_ds: VerifierDataset,
    config: TrainConfig,
    val_rows: Optional[list] = None,
) -> TrainState:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.float().to(device)
    logger.info("Training on %s", device)

    wandb_run = None
    if config.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=config.wandb_project, config=vars(config))
        except Exception as e:
            logger.warning("wandb init failed: %s — continuing without", e)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.dataloader_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.dataloader_workers,
        pin_memory=device.type == "cuda",
    )

    no_decay = {"bias", "LayerNorm.weight", "layernorm.weight"}
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(param_groups, lr=config.learning_rate)

    batches_per_epoch = len(train_loader)
    opt_steps_per_epoch = math.ceil(batches_per_epoch / config.gradient_accumulation_steps)
    total_opt_steps = opt_steps_per_epoch * config.epochs
    warmup_steps = int(total_opt_steps * config.warmup_ratio)
    scheduler = _get_linear_warmup_scheduler(optimizer, warmup_steps, total_opt_steps)

    use_amp = config.fp16 and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    state = TrainState()

    logger.info(
        "Each epoch: %d batches → ~%d optimizer steps (accum=%d). First progress line was slow before; now logs every %d step(s).",
        batches_per_epoch,
        opt_steps_per_epoch,
        config.gradient_accumulation_steps,
        max(1, config.log_every_n_steps),
    )
    _flush_logs()

    for epoch in range(1, config.epochs + 1):
        state.epoch = epoch
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        optimizer.zero_grad()

        logger.info(
            "Epoch %d/%d start (%d batches, ~%d optimizer steps) …",
            epoch,
            config.epochs,
            batches_per_epoch,
            opt_steps_per_epoch,
        )
        _flush_logs()

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast("cuda", enabled=use_amp):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    labels=batch["labels"],
                )
                loss = out["loss"] / config.gradient_accumulation_steps

            scaler.scale(loss).backward()
            epoch_loss += out["loss"].item()
            n_batches += 1

            if batch_idx == 0:
                logger.info(
                    "  First micro-batch OK (batch 0/%d) — loss=%.4f",
                    batches_per_epoch - 1,
                    out["loss"].item(),
                )
                _flush_logs()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                state.global_step += 1

                log_this = config.log_every_n_steps <= 0 or (
                    state.global_step % max(1, config.log_every_n_steps) == 0
                )
                if log_this:
                    avg = epoch_loss / n_batches
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        "epoch %d optimizer_step %d/%d (epoch batch %d/%d) | loss=%.4f lr=%.2e",
                        epoch,
                        state.global_step,
                        total_opt_steps,
                        batch_idx + 1,
                        batches_per_epoch,
                        avg,
                        lr,
                    )
                    _flush_logs()
                    if wandb_run:
                        wandb_run.log({"train/loss": avg, "train/lr": lr}, step=state.global_step)

        epoch_time = time.time() - t0
        avg_loss = epoch_loss / max(1, n_batches)
        logger.info("Epoch %d done in %.1fs — avg loss %.4f", epoch, epoch_time, avg_loss)

        # Validation
        logger.info("Evaluating on val set...")
        metrics = evaluate(model, val_loader, device, val_rows=val_rows)
        logger.info(
            "Val: AUC=%.4f  acc=%.4f  F1=%.4f  loss=%.4f",
            metrics.get("auc", 0), metrics.get("accuracy", 0),
            metrics.get("f1", 0), metrics.get("loss", 0),
        )
        if metrics.get("per_source"):
            for src, src_m in metrics["per_source"].items():
                logger.info("  source=%s: AUC=%.4f acc=%.4f n=%d", src, src_m.get("auc", 0), src_m.get("accuracy", 0), src_m.get("n", 0))
        if metrics.get("per_domain"):
            for dom, dom_m in metrics["per_domain"].items():
                logger.info("  domain=%s: AUC=%.4f acc=%.4f n=%d", dom, dom_m.get("auc", 0), dom_m.get("accuracy", 0), dom_m.get("n", 0))

        record = {"epoch": epoch, "train_loss": avg_loss, **{f"val_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}}
        state.history.append(record)

        if wandb_run:
            wandb_run.log({f"val/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}, step=state.global_step)

        # Early stopping
        current = metrics.get(config.early_stopping_metric.replace("val_", ""), 0.0)
        if current > state.best_metric:
            state.best_metric = current
            state.best_epoch = epoch
            state.patience_counter = 0
            if config.save_best_model:
                best_dir = output_dir / "best_model"
                best_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_dir / "model.pt")
                logger.info("Saved best model (epoch %d, %s=%.4f)", epoch, config.early_stopping_metric, current)
        else:
            state.patience_counter += 1
            logger.info(
                "No improvement (%s=%.4f, best=%.4f, patience %d/%d)",
                config.early_stopping_metric, current, state.best_metric,
                state.patience_counter, config.early_stopping_patience,
            )
            if state.patience_counter >= config.early_stopping_patience:
                logger.info("Early stopping triggered.")
                break

    if config.save_last_model:
        last_dir = output_dir / "last_model"
        last_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), last_dir / "model.pt")

    if wandb_run:
        wandb_run.finish()

    return state
