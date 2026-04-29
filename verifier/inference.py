"""
Inference utilities for the trained verifier checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import yaml
from torch.amp import autocast
from transformers import AutoTokenizer

import vcsr_env
from verifier.model import VerifierModel


@dataclass
class LoadedVerifier:
    model: VerifierModel
    tokenizer: object
    device: torch.device
    checkpoint_path: Path
    backbone_name: str
    revision: Optional[str]
    max_seq_len: int


def load_selected_verifier_metadata(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_training_config_from_checkpoint(checkpoint_path: Path) -> dict:
    train_config_path = checkpoint_path.parent.parent / "train_config.yaml"
    with open(train_config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_verifier(
    selection_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> LoadedVerifier:
    if checkpoint_path is None:
        if selection_path is None:
            raise ValueError("Provide either selection_path or checkpoint_path")
        metadata = load_selected_verifier_metadata(selection_path)
        checkpoint_path = metadata["checkpoint_path"]

    checkpoint = Path(checkpoint_path)
    train_cfg = _load_training_config_from_checkpoint(checkpoint)

    model_cfg = train_cfg.get("model", {})
    backbone = model_cfg.get("backbone", "microsoft/deberta-v3-base")
    revision = model_cfg.get("revision")
    max_seq_len = model_cfg.get("max_seq_len", 512)
    dropout = model_cfg.get("dropout", 0.1)
    backbone_source = vcsr_env.resolve_hf_snapshot(backbone, revision=revision) or backbone

    tokenizer = AutoTokenizer.from_pretrained(
        str(backbone_source),
        revision=revision,
        local_files_only=True,
        use_fast=False,
    )

    model = VerifierModel(
        backbone_name=str(backbone_source),
        dropout=dropout,
        revision=revision,
    )
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return LoadedVerifier(
        model=model,
        tokenizer=tokenizer,
        device=device,
        checkpoint_path=checkpoint,
        backbone_name=str(backbone_source),
        revision=revision,
        max_seq_len=max_seq_len,
    )


class VerifierScorer:
    def __init__(
        self,
        selection_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
    ):
        self.loaded = load_verifier(selection_path=selection_path, checkpoint_path=checkpoint_path)

    def score_pair(self, nl: str, candidate_pddl: str) -> float:
        return self.score_pairs([(nl, candidate_pddl)])[0]

    @torch.no_grad()
    def score_pairs(self, pairs: Iterable[tuple[str, str]], batch_size: int = 8) -> list[float]:
        pairs = list(pairs)
        scores: list[float] = []
        tokenizer = self.loaded.tokenizer
        model = self.loaded.model
        device = self.loaded.device
        max_length = self.loaded.max_seq_len

        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start:start + batch_size]
            nls = [x[0] for x in batch_pairs]
            pddls = [x[1] for x in batch_pairs]
            enc = tokenizer(
                nls,
                pddls,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in enc.items()}
            with autocast("cuda", enabled=device.type == "cuda"):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    labels=None,
                )
            probs = torch.sigmoid(out["logits"]).detach().cpu().tolist()
            scores.extend(float(x) for x in probs)
        return scores
