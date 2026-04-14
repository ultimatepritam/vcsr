"""
Cross-encoder verifier: [CLS] NL [SEP] PDDL [SEP] → sigmoid score.

Uses a pre-trained transformer backbone (default: DeBERTa-v3-base) with a
single-neuron classification head on top of the [CLS] representation.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)


class VerifierModel(nn.Module):
    """Binary cross-encoder: p(equivalent | NL, PDDL)."""

    def __init__(
        self,
        backbone_name: str = "microsoft/deberta-v3-base",
        dropout: float = 0.1,
        hidden_size: Optional[int] = None,
        revision: Optional[str] = None,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(backbone_name, revision=revision)
        self.backbone = AutoModel.from_pretrained(backbone_name, config=config, revision=revision)
        h = hidden_size or config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h, 1),
        )
        self._hidden_size = h

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            try:
                kwargs["token_type_ids"] = token_type_ids
            except TypeError:
                pass

        outputs = self.backbone(**kwargs)
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.head(cls_repr).squeeze(-1)

        result: dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            result["loss"] = loss

        return result

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return calibrated probability (before any post-hoc calibration)."""
        with torch.no_grad():
            out = self.forward(input_ids, attention_mask, token_type_ids)
        return torch.sigmoid(out["logits"])

    @property
    def hidden_size(self) -> int:
        return self._hidden_size
