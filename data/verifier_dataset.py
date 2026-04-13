"""
Verifier training data assembler.

Combines three sources of (NL, PDDL, label) examples:
  1. Gold positives from Planetarium rows
  2. LLM-generated candidates labeled by Planetarium equivalence
  3. Programmatic perturbations of gold PDDL

Outputs a JSONL dataset and summary statistics for verifier training.
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VerifierExample:
    """A single (NL, PDDL, label) training example for the semantic verifier."""
    nl: str
    pddl: str
    label: int  # 1 = equivalent, 0 = not equivalent
    source: str  # "gold", "llm_bedrock", "llm_openrouter", "llm_openai", "llm_hf", "perturbation"
    source_model: str = ""
    perturbation_type: str = ""
    domain: str = ""
    init_is_abstract: int = 0
    goal_is_abstract: int = 0
    parseable: bool = True
    planetarium_name: str = ""

    def to_dict(self) -> dict:
        return {
            "nl": self.nl,
            "pddl": self.pddl,
            "label": self.label,
            "source": self.source,
            "source_model": self.source_model,
            "perturbation_type": self.perturbation_type,
            "domain": self.domain,
            "init_is_abstract": self.init_is_abstract,
            "goal_is_abstract": self.goal_is_abstract,
            "parseable": self.parseable,
            "planetarium_name": self.planetarium_name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VerifierExample":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DatasetStats:
    """Aggregate statistics for a verifier training dataset."""
    total: int = 0
    positives: int = 0
    negatives: int = 0
    unparseable: int = 0
    by_source: dict = field(default_factory=dict)
    by_domain: dict = field(default_factory=dict)
    by_perturbation_type: dict = field(default_factory=dict)
    by_source_model: dict = field(default_factory=dict)

    @property
    def pos_neg_ratio(self) -> float:
        return self.positives / max(self.negatives, 1)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "positives": self.positives,
            "negatives": self.negatives,
            "unparseable": self.unparseable,
            "pos_neg_ratio": round(self.pos_neg_ratio, 4),
            "by_source": self.by_source,
            "by_domain": self.by_domain,
            "by_perturbation_type": self.by_perturbation_type,
            "by_source_model": self.by_source_model,
        }

    def __repr__(self) -> str:
        return (
            f"DatasetStats(total={self.total}, pos={self.positives}, "
            f"neg={self.negatives}, unparseable={self.unparseable}, "
            f"ratio={self.pos_neg_ratio:.3f})"
        )


class VerifierDatasetBuilder:
    """
    Accumulates verifier training examples and writes them to disk.
    Supports incremental building with checkpoint/resume.
    """

    def __init__(self):
        self._examples: list[VerifierExample] = []

    def __len__(self) -> int:
        return len(self._examples)

    def add_gold_positive(
        self,
        nl: str,
        gold_pddl: str,
        domain: str = "",
        init_is_abstract: int = 0,
        goal_is_abstract: int = 0,
        planetarium_name: str = "",
    ):
        """Add a gold (NL, PDDL) pair as a positive example."""
        self._examples.append(
            VerifierExample(
                nl=nl,
                pddl=gold_pddl,
                label=1,
                source="gold",
                domain=domain,
                init_is_abstract=init_is_abstract,
                goal_is_abstract=goal_is_abstract,
                planetarium_name=planetarium_name,
            )
        )

    def add_llm_candidate(
        self,
        nl: str,
        candidate_pddl: str,
        label: int,
        backend: str,
        model: str = "",
        parseable: bool = True,
        domain: str = "",
        init_is_abstract: int = 0,
        goal_is_abstract: int = 0,
        planetarium_name: str = "",
    ):
        """Add an LLM-generated candidate with its equivalence label."""
        self._examples.append(
            VerifierExample(
                nl=nl,
                pddl=candidate_pddl,
                label=label,
                source=f"llm_{backend}",
                source_model=model,
                parseable=parseable,
                domain=domain,
                init_is_abstract=init_is_abstract,
                goal_is_abstract=goal_is_abstract,
                planetarium_name=planetarium_name,
            )
        )

    def add_perturbation(
        self,
        nl: str,
        perturbed_pddl: str,
        label: int,
        perturbation_type: str,
        parseable: bool = True,
        domain: str = "",
        init_is_abstract: int = 0,
        goal_is_abstract: int = 0,
        planetarium_name: str = "",
    ):
        """Add a programmatically perturbed negative."""
        self._examples.append(
            VerifierExample(
                nl=nl,
                pddl=perturbed_pddl,
                label=label,
                source="perturbation",
                perturbation_type=perturbation_type,
                parseable=parseable,
                domain=domain,
                init_is_abstract=init_is_abstract,
                goal_is_abstract=goal_is_abstract,
                planetarium_name=planetarium_name,
            )
        )

    def add_examples(self, examples: list[VerifierExample]):
        """Bulk-add pre-built examples."""
        self._examples.extend(examples)

    def compute_stats(self) -> DatasetStats:
        """Compute aggregate statistics over all accumulated examples."""
        stats = DatasetStats(total=len(self._examples))

        source_counter: Counter = Counter()
        domain_counter: Counter = Counter()
        ptype_counter: Counter = Counter()
        model_counter: Counter = Counter()

        for ex in self._examples:
            if ex.label == 1:
                stats.positives += 1
            else:
                stats.negatives += 1
            if not ex.parseable:
                stats.unparseable += 1

            source_counter[ex.source] += 1
            domain_counter[ex.domain or "unknown"] += 1
            if ex.perturbation_type:
                ptype_counter[ex.perturbation_type] += 1
            if ex.source_model:
                model_counter[ex.source_model] += 1

        stats.by_source = dict(source_counter.most_common())
        stats.by_domain = dict(domain_counter.most_common())
        stats.by_perturbation_type = dict(ptype_counter.most_common())
        stats.by_source_model = dict(model_counter.most_common())

        return stats

    def get_parseable_examples(self) -> list[VerifierExample]:
        """Return only examples that passed parse filter."""
        return [ex for ex in self._examples if ex.parseable]

    def save_jsonl(self, path: str | Path):
        """Write all examples to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for ex in self._examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(self._examples)} examples to {path}")

    def save_parseable_jsonl(self, path: str | Path):
        """Write only parseable examples to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        parseable = self.get_parseable_examples()
        with open(path, "w", encoding="utf-8") as f:
            for ex in parseable:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(parseable)} parseable examples to {path}")

    def save_stats(self, path: str | Path):
        """Write dataset statistics to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        stats = self.compute_stats()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f, indent=2)

        logger.info(f"Stats: {stats}")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "VerifierDatasetBuilder":
        """Load examples from a JSONL file into a new builder."""
        builder = cls()
        path = Path(path)
        if not path.exists():
            return builder

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    builder._examples.append(VerifierExample.from_dict(d))

        logger.info(f"Loaded {len(builder)} examples from {path}")
        return builder

    def save_checkpoint(self, output_dir: str | Path, checkpoint_name: str = "checkpoint"):
        """Save current state as a checkpoint for resume support."""
        output_dir = Path(output_dir)
        self.save_jsonl(output_dir / f"{checkpoint_name}.jsonl")
        self.save_stats(output_dir / f"{checkpoint_name}_stats.json")

    def merge(self, other: "VerifierDatasetBuilder"):
        """Merge another builder's examples into this one."""
        self._examples.extend(other._examples)
