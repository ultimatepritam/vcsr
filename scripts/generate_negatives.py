"""
Negative generator pipeline: generate LLM candidates, label via Planetarium
equivalence, and assemble verifier training data.

Usage:
    python scripts/generate_negatives.py --config configs/neggen.yaml
    python scripts/generate_negatives.py --config configs/neggen.yaml --max_rows 5 --dry_run
    python scripts/generate_negatives.py --config configs/neggen.yaml --resume
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from tqdm import tqdm

from data.planetarium_loader import PlanetariumDataset
from data.verifier_dataset import VerifierDatasetBuilder
from eval.equivalence import (
    check_equivalence_lightweight,
    check_equivalence_lightweight_timed,
)
from generation.perturbations import generate_perturbations
from generation.sampler import MultiSampler, SamplerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _flush_logs():
    """Force log handlers to flush (helps when stdout is piped / line-buffering is off)."""
    for h in logging.root.handlers:
        h.flush()
    if hasattr(sys.stdout, "flush"):
        sys.stdout.flush()
    if hasattr(sys.stderr, "flush"):
        sys.stderr.flush()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def try_parse_pddl(pddl_str: str) -> bool:
    """Check if PDDL parses via Planetarium's builder."""
    if not pddl_str or not pddl_str.strip():
        return False
    try:
        from planetarium.builder import build
        build(pddl_str)
        return True
    except Exception:
        return False


def label_candidate(
    gold_pddl: str,
    candidate_pddl: str,
    is_placeholder: bool = False,
    equiv_timeout_sec: Optional[float] = None,
    num_objects: Optional[int] = None,
    subprocess_min_objects: int = 8,
) -> tuple[int, bool]:
    """
    Label a candidate via Planetarium equivalence. Returns (label, timed_out).

    For num_objects >= subprocess_min_objects (when set), runs the check in a
    subprocess with a hard timeout — avoids multi-hour hangs on large graphs.
    Smaller problems stay in-process (fast; ~milliseconds) to avoid Windows
    spawn overhead (~10–30s per subprocess call).
    Set subprocess_min_objects to 0 to always use the subprocess+timeout path.
    """
    try:
        use_subproc = (
            equiv_timeout_sec is not None
            and float(equiv_timeout_sec) > 0
            and (
                subprocess_min_objects <= 0
                or (num_objects is not None and num_objects >= subprocess_min_objects)
            )
        )
        if use_subproc:
            result = check_equivalence_lightweight_timed(
                gold_pddl,
                candidate_pddl,
                is_placeholder=is_placeholder,
                timeout_sec=float(equiv_timeout_sec),
            )
        else:
            result = check_equivalence_lightweight(
                gold_pddl, candidate_pddl, is_placeholder=is_placeholder
            )
        timed_out = result.error == "timeout"
        if timed_out:
            logger.warning("Equivalence timeout — labeling as not equivalent")
        return (1 if result.equivalent else 0), timed_out
    except Exception as e:
        logger.debug(f"Equivalence check failed: {e}")
        return 0, False


def load_checkpoint_progress(output_dir: Path) -> set[str]:
    """Load the set of already-processed planetarium_name values from checkpoint."""
    checkpoint_path = output_dir / "checkpoint.jsonl"
    if not checkpoint_path.exists():
        return set()

    processed = set()
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                name = d.get("planetarium_name", "")
                if name:
                    processed.add(name)

    logger.info(f"Checkpoint: {len(processed)} unique rows already processed")
    return processed


def process_row(
    row: dict,
    sampler: MultiSampler,
    builder: VerifierDatasetBuilder,
    perturbation_config: dict,
    seed: int,
    row_idx: int,
    equiv_timeout_sec: Optional[float] = 120.0,
    subprocess_min_objects: int = 8,
) -> dict:
    """
    Process a single Planetarium row:
      1. Add gold positive
      2. Generate LLM candidates, parse-filter, label
      3. Generate perturbations, parse-filter, label

    Returns per-row stats dict.
    """
    nl = row["natural_language"]
    gold_pddl = row["problem_pddl"]
    domain = row["domain"]
    name = row["name"]
    init_abstract = row.get("init_is_abstract", 0)
    goal_abstract = row.get("goal_is_abstract", 0)
    is_placeholder = bool(row.get("is_placeholder", 0))
    num_objects = row.get("num_objects")

    row_meta = dict(
        domain=domain,
        init_is_abstract=init_abstract,
        goal_is_abstract=goal_abstract,
        planetarium_name=name,
    )

    stats = {
        "llm_total": 0,
        "llm_parseable": 0,
        "llm_equivalent": 0,
        "llm_errors": 0,
        "pert_total": 0,
        "pert_parseable": 0,
        "pert_equivalent": 0,
        "equiv_timeouts": 0,
    }

    # 1. Gold positive
    builder.add_gold_positive(nl=nl, gold_pddl=gold_pddl, **row_meta)

    # 2. LLM candidates
    logger.info("Bedrock/LLM: sampling K=%s …", sampler.total_k)
    _flush_logs()
    try:
        results = sampler.sample(natural_language=nl, domain=domain)
    except Exception as e:
        logger.error(f"Row {row_idx} ({name}): sampler failed: {e}")
        results = []
    logger.info("Bedrock/LLM: got %d raw results", len(results))
    _flush_logs()

    for sr in results:
        stats["llm_total"] += 1

        if sr.error:
            stats["llm_errors"] += 1
            continue

        pddl = sr.extracted_pddl
        parseable = try_parse_pddl(pddl)
        if parseable:
            stats["llm_parseable"] += 1

        if parseable:
            label, t_out = label_candidate(
                gold_pddl,
                pddl,
                is_placeholder,
                equiv_timeout_sec=equiv_timeout_sec,
                num_objects=num_objects,
                subprocess_min_objects=subprocess_min_objects,
            )
            if t_out:
                stats["equiv_timeouts"] += 1
        else:
            label = 0

        if label == 1:
            stats["llm_equivalent"] += 1

        builder.add_llm_candidate(
            nl=nl,
            candidate_pddl=pddl,
            label=label,
            backend=sr.backend,
            model=sr.model,
            parseable=parseable,
            **row_meta,
        )

    # 3. Perturbations
    logger.info("Perturbations + equivalence labeling …")
    _flush_logs()
    pert_count = perturbation_config.get("count_per_gold", 2)
    pert_types = perturbation_config.get("types", None)
    perturbations = generate_perturbations(
        gold_pddl,
        domain=domain,
        n=pert_count,
        seed=seed + row_idx,
        allowed_types=pert_types,
    )

    for perturbed_pddl, ptype in perturbations:
        stats["pert_total"] += 1
        parseable = try_parse_pddl(perturbed_pddl)
        if parseable:
            stats["pert_parseable"] += 1

        if parseable:
            label, t_out = label_candidate(
                gold_pddl,
                perturbed_pddl,
                is_placeholder,
                equiv_timeout_sec=equiv_timeout_sec,
                num_objects=num_objects,
                subprocess_min_objects=subprocess_min_objects,
            )
            if t_out:
                stats["equiv_timeouts"] += 1
        else:
            label = 0

        if label == 1:
            stats["pert_equivalent"] += 1

        builder.add_perturbation(
            nl=nl,
            perturbed_pddl=perturbed_pddl,
            label=label,
            perturbation_type=ptype,
            parseable=parseable,
            **row_meta,
        )

    return stats


def main():
    parser = argparse.ArgumentParser(description="VCSR Negative Generator Pipeline")
    parser.add_argument("--config", type=str, default="configs/neggen.yaml")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Override max_rows from config")
    parser.add_argument("--dry_run", action="store_true",
                        help="Process only 5 rows for validation")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (skip already-processed rows)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # When stdout is piped (e.g. Tee-Object), block buffering hides logs; prefer line mode.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True)
            except Exception:
                pass

    config = load_config(args.config)
    seed = config.get("experiment", {}).get("seed", 42)
    ds_config = config.get("dataset", {})
    gen_config = config.get("generation", {})
    pert_config = config.get("perturbations", {})
    label_config = config.get("labeling", {})
    out_config = config.get("output", {})
    equiv_timeout_sec = label_config.get("equivalence_timeout_sec", 120.0)
    subprocess_min_objects = int(label_config.get("equivalence_subprocess_min_objects", 8))

    max_rows = args.max_rows or ds_config.get("max_rows", 500)
    if args.dry_run:
        max_rows = 5
        logger.info("=== DRY RUN MODE: processing 5 rows ===")

    output_dir = Path(args.output_dir or out_config.get("dir", "results/neggen/pilot"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_every = out_config.get("checkpoint_every", 50)

    # Save the config used for this run
    with open(output_dir / "run_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Load dataset
    logger.info("Loading Planetarium dataset...")
    dataset = PlanetariumDataset(seed=seed)
    train_data = dataset.train

    # Filter by domains if specified
    allowed_domains = ds_config.get("domains")
    if allowed_domains:
        train_data = train_data.filter(lambda row: row["domain"] in allowed_domains)
        logger.info(f"Filtered to domains {allowed_domains}: {len(train_data)} rows")

    # Subsample
    if max_rows < len(train_data):
        import random
        rng = random.Random(seed)
        indices = rng.sample(range(len(train_data)), max_rows)
        indices.sort()
        train_data = train_data.select(indices)

    logger.info(
        "Processing %s rows | equiv: timeout=%ss, subprocess if num_objects>=%s",
        len(train_data),
        equiv_timeout_sec,
        subprocess_min_objects,
    )

    # Initialize sampler
    sampler_config = SamplerConfig(
        temperature=gen_config.get("temperature", 0.8),
        top_p=gen_config.get("top_p", 0.95),
        max_new_tokens=gen_config.get("max_new_tokens", 1024),
        retry_attempts=gen_config.get("retry_attempts", 3),
        retry_delay_sec=gen_config.get("retry_delay_sec", 2.0),
    )

    sampler = MultiSampler(
        backend_specs=gen_config.get("backends", [{"type": "bedrock", "K": 4}]),
        config=sampler_config,
    )

    # Resume support
    processed_names = set()
    builder = VerifierDatasetBuilder()
    if args.resume:
        processed_names = load_checkpoint_progress(output_dir)
        if processed_names:
            builder = VerifierDatasetBuilder.load_jsonl(output_dir / "checkpoint.jsonl")

    # Aggregate stats
    agg_stats = {
        "rows_processed": 0,
        "rows_skipped": 0,
        "llm_total": 0,
        "llm_parseable": 0,
        "llm_equivalent": 0,
        "llm_errors": 0,
        "pert_total": 0,
        "pert_parseable": 0,
        "pert_equivalent": 0,
        "equiv_timeouts": 0,
    }

    t0 = time.time()
    n_total = len(train_data)

    # tqdm uses \\r updates; piped logs often look "stuck". Log every row + flush.
    for i, row in enumerate(tqdm(train_data, desc="Generating negatives", mininterval=2.0)):
        name = row["name"]

        if name in processed_names:
            agg_stats["rows_skipped"] += 1
            continue

        logger.info(
            "Row %d/%d start | domain=%s | name=%s",
            i + 1,
            n_total,
            row.get("domain", ""),
            (name[:100] + "…") if len(name) > 100 else name,
        )
        _flush_logs()

        row_t0 = time.time()
        row_stats = process_row(
            row=row,
            sampler=sampler,
            builder=builder,
            perturbation_config=pert_config,
            seed=seed,
            row_idx=i,
            equiv_timeout_sec=float(equiv_timeout_sec),
            subprocess_min_objects=subprocess_min_objects,
        )
        row_elapsed = time.time() - row_t0

        agg_stats["rows_processed"] += 1
        for k in [
            "llm_total",
            "llm_parseable",
            "llm_equivalent",
            "llm_errors",
            "pert_total",
            "pert_parseable",
            "pert_equivalent",
            "equiv_timeouts",
        ]:
            agg_stats[k] += row_stats[k]

        logger.info(
            "Row %d/%d done in %.1fs | llm_err=%d this_row",
            agg_stats["rows_processed"],
            n_total,
            row_elapsed,
            row_stats.get("llm_errors", 0),
        )
        _flush_logs()

        # Progress logging
        if (agg_stats["rows_processed"]) % 10 == 0:
            elapsed = time.time() - t0
            rate = agg_stats["rows_processed"] / elapsed if elapsed > 0 else 0
            logger.info(
                f"Progress: {agg_stats['rows_processed']} rows, "
                f"{agg_stats['llm_total']} LLM candidates "
                f"({agg_stats['llm_parseable']} parseable, "
                f"{agg_stats['llm_equivalent']} equiv), "
                f"{agg_stats['pert_total']} perturbations, "
                f"equiv_timeouts={agg_stats['equiv_timeouts']}, "
                f"rate={rate:.3f} rows/sec"
            )

        # Checkpoint
        if checkpoint_every > 0 and agg_stats["rows_processed"] % checkpoint_every == 0:
            logger.info(f"Saving checkpoint at row {agg_stats['rows_processed']}...")
            builder.save_checkpoint(output_dir)

    elapsed = time.time() - t0

    # Final save
    logger.info("Saving final dataset...")
    builder.save_jsonl(output_dir / "verifier_all.jsonl")
    builder.save_parseable_jsonl(output_dir / "verifier_train.jsonl")
    builder.save_stats(output_dir / "dataset_stats.json")

    # Clean up checkpoint if we finished successfully
    checkpoint_file = output_dir / "checkpoint.jsonl"
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        (output_dir / "checkpoint_stats.json").unlink(missing_ok=True)

    # Summary
    agg_stats["total_examples"] = len(builder)
    agg_stats["parseable_examples"] = len(builder.get_parseable_examples())
    agg_stats["elapsed_sec"] = round(elapsed, 1)

    with open(output_dir / "run_stats.json", "w") as f:
        json.dump(agg_stats, f, indent=2)

    logger.info("\n=== PIPELINE COMPLETE ===")
    logger.info(f"Rows processed: {agg_stats['rows_processed']}")
    logger.info(f"Rows skipped (resume): {agg_stats['rows_skipped']}")
    logger.info(f"Total examples: {agg_stats['total_examples']}")
    logger.info(f"Parseable examples: {agg_stats['parseable_examples']}")
    logger.info(
        f"LLM: {agg_stats['llm_total']} total, "
        f"{agg_stats['llm_parseable']} parseable, "
        f"{agg_stats['llm_equivalent']} equivalent, "
        f"{agg_stats['llm_errors']} errors"
    )
    logger.info(
        f"Perturbations: {agg_stats['pert_total']} total, "
        f"{agg_stats['pert_parseable']} parseable, "
        f"{agg_stats['pert_equivalent']} equivalent"
    )
    logger.info(f"Equivalence timeouts (labeled not equivalent): {agg_stats['equiv_timeouts']}")
    logger.info(f"Elapsed: {elapsed:.1f}s")
    logger.info(f"Output: {output_dir}")

    ds_stats = builder.compute_stats()
    logger.info(f"Dataset stats: {ds_stats}")


if __name__ == "__main__":
    main()
