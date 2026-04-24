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


@dataclass
class PairwiseVerifierRow:
    nl: str
    positive_pddl: str
    negative_pddl: str
    source: str
    domain: str
    planetarium_name: str
    style: str = ""
    k: int = 8
    pair_type: str = ""
    positive_candidate_index: int = -1
    negative_candidate_index: int = -1
    score_margin: Optional[float] = None


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


def load_pairwise_jsonl(path: str | Path) -> list[PairwiseVerifierRow]:
    rows: list[PairwiseVerifierRow] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(
                PairwiseVerifierRow(
                    nl=d["nl"],
                    positive_pddl=d["positive_pddl"],
                    negative_pddl=d["negative_pddl"],
                    source=d.get("source", "bestofk_pairwise"),
                    domain=d.get("domain", ""),
                    planetarium_name=d.get("planetarium_name", ""),
                    style=d.get("style", ""),
                    k=int(d.get("K", d.get("k", 8))),
                    pair_type=d.get("pair_type", ""),
                    positive_candidate_index=int(d.get("positive_candidate_index", -1)),
                    negative_candidate_index=int(d.get("negative_candidate_index", -1)),
                    score_margin=d.get("score_margin"),
                )
            )
    logger.info("Loaded %d pairwise rows from %s", len(rows), path)
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


def split_pairwise_by_template(
    rows: list[PairwiseVerifierRow],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[PairwiseVerifierRow], list[PairwiseVerifierRow]]:
    groups: dict[str, list[PairwiseVerifierRow]] = {}
    for r in rows:
        groups.setdefault(r.planetarium_name, []).append(r)

    keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_total = len(rows)
    train_target = int(n_total * (1.0 - val_fraction))
    train_rows: list[PairwiseVerifierRow] = []
    val_rows: list[PairwiseVerifierRow] = []
    accumulated = 0

    for key in keys:
        group = groups[key]
        if accumulated < train_target:
            train_rows.extend(group)
            accumulated += len(group)
        else:
            val_rows.extend(group)

    logger.info(
        "Pairwise split: train=%d val=%d (%.1f%% val) from %d groups",
        len(train_rows),
        len(val_rows),
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


class PairwiseVerifierDataset(Dataset):
    """PyTorch dataset for positive-vs-negative candidates from the same pool."""

    def __init__(
        self,
        rows: list[PairwiseVerifierRow],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, nl: str, pddl: str) -> dict:
        encoding = self.tokenizer(
            nl,
            pddl,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "token_type_ids": encoding.get("token_type_ids", [0] * len(encoding["input_ids"])),
        }

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        return {
            "positive": self._encode(row.nl, row.positive_pddl),
            "negative": self._encode(row.nl, row.negative_pddl),
            "domain": row.domain,
            "style": row.style,
            "k": row.k,
            "pair_type": row.pair_type,
        }


def _pad_encoded_batch(batch: list[dict]) -> dict:
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids = []
    attention_mask = []
    token_type_ids = []

    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [0] * pad_len)
        attention_mask.append(b["attention_mask"] + [0] * pad_len)
        token_type_ids.append(b["token_type_ids"] + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
    }


def collate_fn(batch: list[dict]) -> dict:
    """Pad a batch of variable-length encoded examples."""
    encoded = _pad_encoded_batch(batch)
    encoded["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.float)
    return encoded


def collate_pairwise_fn(batch: list[dict]) -> dict:
    """Pad positive and negative candidates separately for pairwise loss."""
    return {
        "positive": _pad_encoded_batch([b["positive"] for b in batch]),
        "negative": _pad_encoded_batch([b["negative"] for b in batch]),
        "domain": [b["domain"] for b in batch],
        "style": [b["style"] for b in batch],
        "k": [b["k"] for b in batch],
        "pair_type": [b["pair_type"] for b in batch],
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
