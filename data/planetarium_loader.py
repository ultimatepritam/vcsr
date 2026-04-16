"""
Planetarium dataset loader with template-hash based splitting for VCSR.

Loads BatsResearch/planetarium from HuggingFace, provides train/val/test splits
grouped by underlying PDDL template identity to prevent data leakage, and exposes
utilities for filtering by domain and description style.
"""

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, Dataset
from vcsr_env import get_runtime_dir

logger = logging.getLogger(__name__)


@dataclass
class SplitStats:
    total: int = 0
    domains: dict = field(default_factory=dict)
    init_abstract: dict = field(default_factory=dict)
    goal_abstract: dict = field(default_factory=dict)
    num_objects_range: tuple = (0, 0)


def _normalize_pddl_for_hash(pddl_str: str) -> str:
    """Normalize PDDL whitespace for consistent hashing across formatting variants."""
    return " ".join(pddl_str.split())


def _template_hash(row: dict) -> str:
    """
    Compute a hash that identifies the underlying PDDL problem template.
    Uses the problem name field (e.g. 'blocksworld_on_table_to_stack_blocks_list_2')
    which encodes the domain, init/goal type, and object count.
    """
    return row["name"]


def _content_hash(row: dict) -> str:
    """
    Alternative hash using normalized gold PDDL content.
    More conservative -- groups rows that share identical ground-truth PDDL.
    """
    normalized = _normalize_pddl_for_hash(row["problem_pddl"])
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


class PlanetariumDataset:
    """
    Wraps the BatsResearch/planetarium HuggingFace dataset with
    template-hash based train/val/test splitting.
    """

    DATASET_ID = "BatsResearch/planetarium"

    def __init__(
        self,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_strategy: str = "template_hash",
        seed: int = 42,
        cache_dir: Optional[str] = None,
    ):
        assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
        assert split_strategy in ("template_hash", "content_hash", "random")

        self.split_ratios = split_ratios
        self.split_strategy = split_strategy
        self.seed = seed

        resolved_cache_dir = cache_dir or str(get_runtime_dir("hf_datasets"))

        logger.info(
            "Loading Planetarium dataset from HuggingFace (cache_dir=%s)...",
            resolved_cache_dir,
        )
        raw = load_dataset(self.DATASET_ID, cache_dir=resolved_cache_dir)

        self._raw_train: Dataset = raw["train"]
        self._raw_test: Dataset = raw["test"]

        self._train: Optional[Dataset] = None
        self._val: Optional[Dataset] = None
        self._test: Optional[Dataset] = None

        self._build_splits()

    def _build_splits(self):
        """
        Split the raw train set into train/val using template-hash grouping
        to prevent leakage. The raw test set is kept as-is (contains
        out-of-domain floor-tile examples for generalization evaluation).
        """
        if self.split_strategy == "template_hash":
            hash_fn = _template_hash
        elif self.split_strategy == "content_hash":
            hash_fn = _content_hash
        else:
            self._random_split()
            return

        groups: dict[str, list[int]] = {}
        for idx, row in enumerate(self._raw_train):
            key = hash_fn(row)
            groups.setdefault(key, []).append(idx)

        sorted_keys = sorted(groups.keys())

        import random
        rng = random.Random(self.seed)
        rng.shuffle(sorted_keys)

        n_total = len(self._raw_train)
        train_target = int(n_total * self.split_ratios[0])

        train_indices = []
        val_indices = []
        accumulated = 0

        for key in sorted_keys:
            indices = groups[key]
            if accumulated < train_target:
                train_indices.extend(indices)
                accumulated += len(indices)
            else:
                val_indices.extend(indices)

        self._train = self._raw_train.select(train_indices)
        self._val = self._raw_train.select(val_indices)
        self._test = self._raw_test

        logger.info(
            f"Splits: train={len(self._train)}, val={len(self._val)}, "
            f"test={len(self._test)} (strategy={self.split_strategy})"
        )

    def _random_split(self):
        split = self._raw_train.train_test_split(
            test_size=self.split_ratios[1] + self.split_ratios[2],
            seed=self.seed,
        )
        self._train = split["train"]

        remaining_val_ratio = self.split_ratios[1] / (
            self.split_ratios[1] + self.split_ratios[2]
        )
        val_test = split["test"].train_test_split(
            test_size=1.0 - remaining_val_ratio,
            seed=self.seed,
        )
        self._val = val_test["train"]
        self._test_from_train = val_test["test"]
        self._test = self._raw_test

    @property
    def train(self) -> Dataset:
        return self._train

    @property
    def val(self) -> Dataset:
        return self._val

    @property
    def test(self) -> Dataset:
        return self._test

    def get_split(self, split_name: str) -> Dataset:
        return {"train": self.train, "val": self.val, "test": self.test}[split_name]

    def filter_by_domain(self, dataset: Dataset, domain: str) -> Dataset:
        return dataset.filter(lambda row: row["domain"] == domain)

    def filter_by_style(
        self,
        dataset: Dataset,
        init_abstract: Optional[int] = None,
        goal_abstract: Optional[int] = None,
    ) -> Dataset:
        def predicate(row):
            if init_abstract is not None and row["init_is_abstract"] != init_abstract:
                return False
            if goal_abstract is not None and row["goal_is_abstract"] != goal_abstract:
                return False
            return True
        return dataset.filter(predicate)

    def compute_stats(self, dataset: Dataset) -> SplitStats:
        stats = SplitStats(total=len(dataset))
        stats.domains = dict(Counter(dataset["domain"]))
        stats.init_abstract = dict(Counter(dataset["init_is_abstract"]))
        stats.goal_abstract = dict(Counter(dataset["goal_is_abstract"]))
        obj_counts = dataset["num_objects"]
        if obj_counts:
            stats.num_objects_range = (min(obj_counts), max(obj_counts))
        return stats

    def summary(self) -> str:
        lines = ["=== Planetarium Dataset Summary ==="]
        for name, ds in [("train", self.train), ("val", self.val), ("test", self.test)]:
            s = self.compute_stats(ds)
            lines.append(f"\n{name}: {s.total} rows")
            lines.append(f"  Domains: {s.domains}")
            lines.append(f"  Init abstract: {s.init_abstract}")
            lines.append(f"  Goal abstract: {s.goal_abstract}")
            lines.append(f"  Object count range: {s.num_objects_range}")
        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ds = PlanetariumDataset()
    print(ds.summary())
