"""
Simple ranking policies for candidate selection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class CandidateRecord:
    index: int
    parseable: bool
    equivalent: bool
    verifier_score: Optional[float] = None


@dataclass
class SelectionResult:
    policy: str
    selected_index: Optional[int]
    reason: str


def greedy_first(candidates: list[CandidateRecord]) -> SelectionResult:
    if not candidates:
        return SelectionResult(policy="greedy_first", selected_index=None, reason="no_candidates")
    return SelectionResult(policy="greedy_first", selected_index=candidates[0].index, reason="first_candidate")


def random_parseable(candidates: list[CandidateRecord], rng: random.Random) -> SelectionResult:
    parseable = [c for c in candidates if c.parseable]
    if not parseable:
        return SelectionResult(policy="random_parseable", selected_index=None, reason="no_parseable_candidate")
    chosen = rng.choice(parseable)
    return SelectionResult(policy="random_parseable", selected_index=chosen.index, reason="random_parseable")


def verifier_ranked(candidates: list[CandidateRecord]) -> SelectionResult:
    parseable = [c for c in candidates if c.parseable]
    if not parseable:
        return SelectionResult(policy="verifier_ranked", selected_index=None, reason="no_parseable_candidate")
    chosen = max(
        parseable,
        key=lambda c: (
            float("-inf") if c.verifier_score is None else c.verifier_score,
            -c.index,
        ),
    )
    return SelectionResult(policy="verifier_ranked", selected_index=chosen.index, reason="highest_verifier_score")
