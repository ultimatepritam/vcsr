"""
Verifier dataset: loads JSONL from the neggen pipeline, splits by template group,
and tokenizes (NL [SEP] PDDL) for a cross-encoder classifier.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class VerifierRow:
    nl: str
    pddl: str
    label: int
    source: str
    domain: str
    planetarium_name: str
    perturbation_type: str = ""


def load_jsonl(path: str | Path, filter_unparseable: bool = True) -> list[VerifierRow]:
    rows: list[VerifierRow] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if filter_unparseable and not d.get("parseable", True):
                continue
            rows.append(
                VerifierRow(
                    nl=d["nl"],
                    pddl=d["pddl"],
                    label=int(d["label"]),
                    source=d.get("source", ""),
                    domain=d.get("domain", ""),
                    planetarium_name=d.get("planetarium_name", ""),
                    perturbation_type=d.get("perturbation_type", ""),
                )
            )
    logger.info("Loaded %d rows from %s", len(rows), path)
    return rows


def split_by_template(
    rows: list[VerifierRow],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[VerifierRow], list[VerifierRow]]:
    """Group rows by planetarium_name, hold out val_fraction of groups for val."""
    groups: dict[str, list[VerifierRow]] = {}
    for r in rows:
        groups.setdefault(r.planetarium_name, []).append(r)

    keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_total = len(rows)
    train_target = int(n_total * (1.0 - val_fraction))

    train_rows: list[VerifierRow] = []
    val_rows: list[VerifierRow] = []
    accumulated = 0

    for key in keys:
        group = groups[key]
        if accumulated < train_target:
            train_rows.extend(group)
            accumulated += len(group)
        else:
            val_rows.extend(group)

    logger.info(
        "Split: train=%d val=%d (%.1f%% val) from %d groups",
        len(train_rows), len(val_rows),
        100 * len(val_rows) / max(1, len(rows)),
        len(groups),
    )
    return train_rows, val_rows


class VerifierDataset(Dataset):
    """PyTorch dataset that tokenizes (NL, PDDL) pairs for a cross-encoder."""

    def __init__(
        self,
        rows: list[VerifierRow],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        encoding = self.tokenizer(
            row.nl,
            row.pddl,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "token_type_ids": encoding.get("token_type_ids", [0] * len(encoding["input_ids"])),
            "label": row.label,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad a batch of variable-length encoded examples."""
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids = []
    attention_mask = []
    token_type_ids = []
    labels = []

    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [0] * pad_len)
        attention_mask.append(b["attention_mask"] + [0] * pad_len)
        token_type_ids.append(b["token_type_ids"] + [0] * pad_len)
        labels.append(b["label"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float),
    }


def build_datasets(
    jsonl_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    val_fraction: float = 0.15,
    seed: int = 42,
    filter_unparseable: bool = True,
) -> tuple[VerifierDataset, VerifierDataset, list[VerifierRow], list[VerifierRow]]:
    """Convenience: load → split → wrap in VerifierDataset. Returns (train_ds, val_ds, train_rows, val_rows)."""
    rows = load_jsonl(jsonl_path, filter_unparseable=filter_unparseable)
    train_rows, val_rows = split_by_template(rows, val_fraction=val_fraction, seed=seed)

    train_ds = VerifierDataset(train_rows, tokenizer, max_length=max_length)
    val_ds = VerifierDataset(val_rows, tokenizer, max_length=max_length)
    return train_ds, val_ds, train_rows, val_rows
