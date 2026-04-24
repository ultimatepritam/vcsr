"""
Analyze zero-training selection policies around the fixed round-4 verifier.

This script does not train or generate. It replays cached candidate pools,
rescoring parseable candidates with the promoted round-4 verifier and the
historical round-3 verifier, then evaluates selection policies that use only
scores, parseability, and candidate indices.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vcsr_env  # noqa: F401
import yaml

from data.planetarium_loader import PlanetariumDataset
from verifier.inference import VerifierScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    index: int
    parseable: bool
    equivalent: bool
    pddl: str
    round4_score: float | None = None
    round3_score: float | None = None


@dataclass
class Selection:
    policy: str
    selected_index: int | None
    reason: str


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


def _flush_logs() -> None:
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_process_info(output_dir: Path, command: list[str]) -> None:
    _write_json(
        output_dir / "process_info.json",
        {
            "pid": os.getpid(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "command": " ".join(command),
            "output_dir": str(output_dir),
            "progress_log": str(output_dir / "progress.log"),
            "progress_json": str(output_dir / "progress.json"),
        },
    )


def _write_progress(
    output_dir: Path,
    *,
    status: str,
    completed_pools: int,
    total_pools: int,
    current_pool: str,
    started_at: float,
) -> None:
    _write_json(
        output_dir / "progress.json",
        {
            "status": status,
            "completed_pools": completed_pools,
            "total_pools": total_pools,
            "remaining_pools": max(0, total_pools - completed_pools),
            "current_pool": current_pool,
            "elapsed_sec": max(0.0, time.time() - started_at),
        },
    )


def _style(init_is_abstract: int, goal_is_abstract: int) -> str:
    init = "abstract" if int(init_is_abstract) else "explicit"
    goal = "abstract" if int(goal_is_abstract) else "explicit"
    return f"{init}/{goal}"


def _load_run_config(candidate_dump: Path) -> dict:
    run_config = candidate_dump.parent / "run_config.yaml"
    with open(run_config, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_nl_lookup(candidate_dump: Path) -> dict[str, str]:
    cfg = _load_run_config(candidate_dump)
    ds_cfg = cfg.get("dataset", {})
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    dataset = PlanetariumDataset(
        split_strategy=ds_cfg.get("split_strategy", "template_hash"),
        seed=seed,
    )
    rows = dataset.get_split(ds_cfg.get("split", "test"))
    return {row["name"]: row["natural_language"] for row in rows}


def _load_candidate_dump(candidate_dump: Path) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    meta_by_row: dict[int, dict] = {}
    candidates_by_row: dict[int, list[dict]] = defaultdict(list)
    with open(candidate_dump, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "candidate_index" not in row:
                continue
            row_index = int(row["row_index"])
            meta_by_row[row_index] = {
                "row_index": row_index,
                "planetarium_name": row["planetarium_name"],
                "domain": row["domain"],
                "init_is_abstract": int(row.get("init_is_abstract", 0)),
                "goal_is_abstract": int(row.get("goal_is_abstract", 0)),
            }
            candidates_by_row[row_index].append(row)
    for rows in candidates_by_row.values():
        rows.sort(key=lambda item: int(item["candidate_index"]))
    return meta_by_row, candidates_by_row


def _score_candidates(
    *,
    nl_by_name: dict[str, str],
    meta_by_row: dict[int, dict],
    candidates_by_row: dict[int, list[dict]],
    round4_scorer: VerifierScorer,
    round3_scorer: VerifierScorer,
    batch_size: int,
) -> dict[int, list[Candidate]]:
    scored: dict[int, list[Candidate]] = {}
    round4_pairs: list[tuple[str, str]] = []
    round3_pairs: list[tuple[str, str]] = []
    pair_refs: list[tuple[int, int]] = []
    for row_index, rows in candidates_by_row.items():
        name = meta_by_row[row_index]["planetarium_name"]
        nl = nl_by_name[name]
        scored[row_index] = []
        for rec in rows:
            cand = Candidate(
                index=int(rec["candidate_index"]),
                parseable=bool(rec.get("parseable")),
                equivalent=bool(rec.get("equivalent")),
                pddl=rec.get("pddl") or "",
            )
            scored[row_index].append(cand)
            if cand.parseable and cand.pddl:
                round4_pairs.append((nl, cand.pddl))
                round3_pairs.append((nl, cand.pddl))
                pair_refs.append((row_index, cand.index))

    round4_scores = round4_scorer.score_pairs(round4_pairs, batch_size=batch_size) if round4_pairs else []
    round3_scores = round3_scorer.score_pairs(round3_pairs, batch_size=batch_size) if round3_pairs else []
    score_lookup = {
        ref: (float(score4), float(score3))
        for ref, score4, score3 in zip(pair_refs, round4_scores, round3_scores)
    }
    for row_index, candidates in scored.items():
        for cand in candidates:
            if (row_index, cand.index) in score_lookup:
                cand.round4_score, cand.round3_score = score_lookup[(row_index, cand.index)]
    return scored


def _parseable(candidates: list[Candidate]) -> list[Candidate]:
    return [cand for cand in candidates if cand.parseable]


def _score(score: float | None) -> float:
    return float("-inf") if score is None else float(score)


def _select_by_round4(candidates: list[Candidate], policy: str = "verifier_ranked") -> Selection:
    parseable = _parseable(candidates)
    if not parseable:
        return Selection(policy, None, "no_parseable_candidate")
    chosen = max(parseable, key=lambda cand: (_score(cand.round4_score), -cand.index))
    return Selection(policy, chosen.index, "highest_round4_score")


def _select_by_round3(candidates: list[Candidate]) -> Selection:
    parseable = _parseable(candidates)
    if not parseable:
        return Selection("round3_ranked", None, "no_parseable_candidate")
    chosen = max(parseable, key=lambda cand: (_score(cand.round3_score), -cand.index))
    return Selection("round3_ranked", chosen.index, "highest_round3_score")


def greedy_first(candidates: list[Candidate]) -> Selection:
    if not candidates:
        return Selection("greedy_first", None, "no_candidates")
    return Selection("greedy_first", candidates[0].index, "first_candidate")


def random_parseable(candidates: list[Candidate], rng: random.Random) -> Selection:
    parseable = _parseable(candidates)
    if not parseable:
        return Selection("random_parseable", None, "no_parseable_candidate")
    chosen = rng.choice(parseable)
    return Selection("random_parseable", chosen.index, "random_parseable")


def verifier_margin_fallback(candidates: list[Candidate], margin: float) -> Selection:
    top = _select_by_round4(candidates, policy=f"verifier_margin_fallback_m{margin:g}")
    greedy = greedy_first(candidates)
    if top.selected_index is None:
        return top
    top_cand = _candidate_by_index(candidates, top.selected_index)
    greedy_cand = _candidate_by_index(candidates, greedy.selected_index)
    greedy_score = _score(greedy_cand.round4_score if greedy_cand else None)
    gap = _score(top_cand.round4_score) - greedy_score
    if gap >= margin:
        return Selection(top.policy, top.selected_index, f"top_beats_greedy_by_{gap:.4f}")
    return Selection(top.policy, greedy.selected_index, f"fallback_greedy_gap_{gap:.4f}")


def verifier_top_gap(candidates: list[Candidate], margin: float) -> Selection:
    policy = f"verifier_top_gap_m{margin:g}"
    parseable = sorted(_parseable(candidates), key=lambda cand: (_score(cand.round4_score), -cand.index), reverse=True)
    if not parseable:
        return Selection(policy, None, "no_parseable_candidate")
    if len(parseable) == 1:
        return Selection(policy, parseable[0].index, "single_parseable")
    gap = _score(parseable[0].round4_score) - _score(parseable[1].round4_score)
    if gap >= margin:
        return Selection(policy, parseable[0].index, f"top_gap_{gap:.4f}")
    return Selection(policy, greedy_first(candidates).selected_index, f"fallback_greedy_top_gap_{gap:.4f}")


def round3_round4_agreement(candidates: list[Candidate], fallback: str) -> Selection:
    policy = f"round3_round4_agreement_{fallback}"
    r4 = _select_by_round4(candidates, policy=policy)
    r3 = _select_by_round3(candidates)
    if r4.selected_index is None:
        return r4
    if r4.selected_index == r3.selected_index:
        return Selection(policy, r4.selected_index, "round3_round4_agree")
    if fallback == "lowest_parseable":
        parseable = sorted(_parseable(candidates), key=lambda cand: cand.index)
        if parseable:
            return Selection(policy, parseable[0].index, "fallback_lowest_parseable")
        return Selection(policy, None, "no_parseable_candidate")
    return Selection(policy, greedy_first(candidates).selected_index, "fallback_greedy_disagreement")


def score_normalized_rank(candidates: list[Candidate], method: str) -> Selection:
    policy = f"score_normalized_{method}"
    parseable = _parseable(candidates)
    if not parseable:
        return Selection(policy, None, "no_parseable_candidate")
    scores = [_score(cand.round4_score) for cand in parseable]
    finite_scores = [score for score in scores if math.isfinite(score)]
    if not finite_scores:
        return Selection(policy, parseable[0].index, "no_finite_scores")
    if method == "z":
        mu = mean(finite_scores)
        var = mean([(score - mu) ** 2 for score in finite_scores])
        denom = math.sqrt(var) if var > 0 else 1.0
        norm = {cand.index: (_score(cand.round4_score) - mu) / denom for cand in parseable}
    else:
        lo = min(finite_scores)
        hi = max(finite_scores)
        denom = hi - lo if hi > lo else 1.0
        norm = {cand.index: (_score(cand.round4_score) - lo) / denom for cand in parseable}
    chosen = max(parseable, key=lambda cand: (norm[cand.index], -cand.index))
    return Selection(policy, chosen.index, method)


def hybrid_rank_index_penalty(candidates: list[Candidate], alpha: float) -> Selection:
    policy = f"hybrid_rank_index_penalty_a{alpha:g}"
    parseable = _parseable(candidates)
    if not parseable:
        return Selection(policy, None, "no_parseable_candidate")
    chosen = max(parseable, key=lambda cand: (_score(cand.round4_score) - alpha * cand.index, -cand.index))
    return Selection(policy, chosen.index, f"score_minus_{alpha:g}_index")


def _candidate_by_index(candidates: list[Candidate], index: int | None) -> Candidate | None:
    if index is None:
        return None
    for cand in candidates:
        if cand.index == int(index):
            return cand
    return None


def _selection_equivalent(candidates: list[Candidate], selection: Selection) -> bool:
    cand = _candidate_by_index(candidates, selection.selected_index)
    return bool(cand and cand.equivalent)


def _selection_parseable(candidates: list[Candidate], selection: Selection) -> bool:
    cand = _candidate_by_index(candidates, selection.selected_index)
    return bool(cand and cand.parseable)


def _pool_stats(candidates: list[Candidate]) -> dict:
    parseable = _parseable(candidates)
    sorted_parseable = sorted(parseable, key=lambda cand: (_score(cand.round4_score), -cand.index), reverse=True)
    top = sorted_parseable[0] if sorted_parseable else None
    second = sorted_parseable[1] if len(sorted_parseable) > 1 else None
    greedy = candidates[0] if candidates else None
    top_score = top.round4_score if top else None
    second_score = second.round4_score if second else None
    greedy_score = greedy.round4_score if greedy else None
    return {
        "parseable_count": len(parseable),
        "equivalent_count": sum(1 for cand in candidates if cand.equivalent),
        "oracle": any(cand.equivalent for cand in candidates),
        "top_score": top_score,
        "second_score": second_score,
        "greedy_score": greedy_score,
        "top_second_gap": None if top_score is None or second_score is None else float(top_score) - float(second_score),
        "top_greedy_gap": None if top_score is None or greedy_score is None else float(top_score) - float(greedy_score),
    }


def _policy_factories(margins: list[float], alphas: list[float]) -> dict[str, Callable[[list[Candidate], random.Random], Selection]]:
    factories: dict[str, Callable[[list[Candidate], random.Random], Selection]] = {
        "greedy_first": lambda candidates, rng: greedy_first(candidates),
        "random_parseable": lambda candidates, rng: random_parseable(candidates, rng),
        "verifier_ranked": lambda candidates, rng: _select_by_round4(candidates),
        "round3_round4_agreement_greedy": lambda candidates, rng: round3_round4_agreement(candidates, "greedy"),
        "round3_round4_agreement_lowest_parseable": lambda candidates, rng: round3_round4_agreement(candidates, "lowest_parseable"),
        "score_normalized_z": lambda candidates, rng: score_normalized_rank(candidates, "z"),
        "score_normalized_minmax": lambda candidates, rng: score_normalized_rank(candidates, "minmax"),
    }
    for margin in margins:
        factories[f"verifier_margin_fallback_m{margin:g}"] = (
            lambda candidates, rng, m=margin: verifier_margin_fallback(candidates, m)
        )
        factories[f"verifier_top_gap_m{margin:g}"] = (
            lambda candidates, rng, m=margin: verifier_top_gap(candidates, m)
        )
    for alpha in alphas:
        factories[f"hybrid_rank_index_penalty_a{alpha:g}"] = (
            lambda candidates, rng, a=alpha: hybrid_rank_index_penalty(candidates, a)
        )
    return factories


def _run_self_tests() -> None:
    cands = [
        Candidate(0, True, False, "", 0.2, 0.2),
        Candidate(1, True, True, "", 0.9, 0.8),
        Candidate(2, False, False, "", None, None),
    ]
    assert greedy_first(cands).selected_index == 0
    assert _select_by_round4(cands).selected_index == 1
    assert verifier_margin_fallback(cands, 0.1).selected_index == 1
    assert verifier_margin_fallback(cands, 1.0).selected_index == 0
    assert verifier_top_gap(cands, 0.1).selected_index == 1
    assert round3_round4_agreement(cands, "greedy").selected_index == 1
    disagree = [
        Candidate(0, True, False, "", 0.2, 0.9),
        Candidate(1, True, True, "", 0.9, 0.1),
    ]
    assert round3_round4_agreement(disagree, "greedy").selected_index == 0
    assert round3_round4_agreement(disagree, "lowest_parseable").selected_index == 0
    assert score_normalized_rank(disagree, "z").selected_index == 1
    assert hybrid_rank_index_penalty(disagree, 1.0).selected_index == 0
    no_parse = [Candidate(0, False, False, "", None, None)]
    assert _select_by_round4(no_parse).selected_index is None
    assert greedy_first(no_parse).selected_index == 0
    print("self tests ok")


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    sorted_values = sorted(values)
    def q(frac: float) -> float:
        idx = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * frac)))
        return sorted_values[idx]
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "min": sorted_values[0],
        "p25": q(0.25),
        "median": q(0.5),
        "p75": q(0.75),
        "max": sorted_values[-1],
    }


def _analyze_pool(
    *,
    pool_name: str,
    candidate_dump: Path,
    scored_by_row: dict[int, list[Candidate]],
    meta_by_row: dict[int, dict],
    k_values: list[int],
    policy_fns: dict[str, Callable[[list[Candidate], random.Random], Selection]],
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    policy_rows: list[dict] = []
    changed_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    for row_index in sorted(scored_by_row):
        meta = meta_by_row[row_index]
        style = _style(meta["init_is_abstract"], meta["goal_is_abstract"])
        for k_value in k_values:
            candidates = scored_by_row[row_index][:k_value]
            stats = _pool_stats(candidates)
            rng = random.Random(seed + row_index * 1000 + k_value)
            baseline = policy_fns["verifier_ranked"](candidates, rng)
            baseline_equiv = _selection_equivalent(candidates, baseline)
            greedy = policy_fns["greedy_first"](candidates, rng)
            for policy_name, policy_fn in policy_fns.items():
                rng = random.Random(seed + row_index * 1000 + k_value)
                selection = policy_fn(candidates, rng)
                selected = _candidate_by_index(candidates, selection.selected_index)
                equiv = _selection_equivalent(candidates, selection)
                parseable = _selection_parseable(candidates, selection)
                changed = selection.selected_index != baseline.selected_index
                helped = changed and equiv and not baseline_equiv
                hurt = changed and (not equiv) and baseline_equiv
                tied_change = changed and equiv == baseline_equiv
                row = {
                    "pool": pool_name,
                    "candidate_dump": str(candidate_dump),
                    "row_index": row_index,
                    "planetarium_name": meta["planetarium_name"],
                    "domain": meta["domain"],
                    "style": style,
                    "K": k_value,
                    "policy": policy_name,
                    "selected_index": selection.selected_index,
                    "selected_parseable": parseable,
                    "selected_equivalent": equiv,
                    "selected_score": selected.round4_score if selected else None,
                    "reason": selection.reason,
                    "baseline_selected_index": baseline.selected_index,
                    "baseline_equivalent": baseline_equiv,
                    "greedy_selected_index": greedy.selected_index,
                    "greedy_equivalent": _selection_equivalent(candidates, greedy),
                    "changed_vs_round4": changed,
                    "helped_vs_round4": helped,
                    "hurt_vs_round4": hurt,
                    "tied_change_vs_round4": tied_change,
                    **stats,
                }
                policy_rows.append(row)
                if policy_name != "verifier_ranked" and changed:
                    changed_rows.append(row)
            verifier_selected = _candidate_by_index(candidates, baseline.selected_index)
            top_gap = stats.get("top_second_gap")
            top_greedy_gap = stats.get("top_greedy_gap")
            diagnostic_rows.append(
                {
                    "pool": pool_name,
                    "row_index": row_index,
                    "planetarium_name": meta["planetarium_name"],
                    "domain": meta["domain"],
                    "style": style,
                    "K": k_value,
                    "oracle": stats["oracle"],
                    "round4_correct": baseline_equiv,
                    "round4_selected_index": baseline.selected_index,
                    "round4_selected_score": verifier_selected.round4_score if verifier_selected else None,
                    "parseable_count": stats["parseable_count"],
                    "equivalent_count": stats["equivalent_count"],
                    "top_second_gap": top_gap,
                    "top_greedy_gap": top_greedy_gap,
                    "bucket": (
                        "round4_correct"
                        if baseline_equiv
                        else "oracle_positive_miss"
                        if stats["oracle"]
                        else "no_equivalent_in_pool"
                    ),
                }
            )
    return policy_rows, changed_rows, diagnostic_rows


def _aggregate_policy_rows(policy_rows: list[dict]) -> dict:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in policy_rows:
        grouped[(row["pool"], int(row["K"]), row["policy"])].append(row)
    per_pool = []
    for (pool, k_value, policy), rows in sorted(grouped.items()):
        total = len(rows)
        per_pool.append(
            {
                "pool": pool,
                "K": k_value,
                "policy": policy,
                "total": total,
                "parse_rate": sum(row["selected_parseable"] for row in rows) / max(1, total),
                "equiv_rate": sum(row["selected_equivalent"] for row in rows) / max(1, total),
                "oracle_rate": sum(row["oracle"] for row in rows) / max(1, total),
                "changed_vs_round4": sum(row["changed_vs_round4"] for row in rows),
                "helped_vs_round4": sum(row["helped_vs_round4"] for row in rows),
                "hurt_vs_round4": sum(row["hurt_vs_round4"] for row in rows),
                "tied_change_vs_round4": sum(row["tied_change_vs_round4"] for row in rows),
            }
        )
    mean_grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in per_pool:
        mean_grouped[(row["K"], row["policy"])].append(row)
    mean_rows = []
    for (k_value, policy), rows in sorted(mean_grouped.items()):
        mean_rows.append(
            {
                "K": k_value,
                "policy": policy,
                "mean_equiv_rate": sum(row["equiv_rate"] for row in rows) / len(rows),
                "mean_oracle_rate": sum(row["oracle_rate"] for row in rows) / len(rows),
                "total_changed_vs_round4": sum(row["changed_vs_round4"] for row in rows),
                "total_helped_vs_round4": sum(row["helped_vs_round4"] for row in rows),
                "total_hurt_vs_round4": sum(row["hurt_vs_round4"] for row in rows),
                "total_tied_change_vs_round4": sum(row["tied_change_vs_round4"] for row in rows),
                "pool_count": len(rows),
            }
        )
    return {"per_pool": per_pool, "mean": mean_rows}


def _aggregate_diagnostics(diagnostic_rows: list[dict]) -> dict:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in diagnostic_rows:
        if row["top_second_gap"] is not None:
            buckets[f"{row['bucket']}:top_second_gap"].append(float(row["top_second_gap"]))
        if row["top_greedy_gap"] is not None:
            buckets[f"{row['bucket']}:top_greedy_gap"].append(float(row["top_greedy_gap"]))
    by_domain = Counter()
    by_style = Counter()
    by_bucket = Counter()
    for row in diagnostic_rows:
        by_domain[(row["domain"], row["bucket"])] += 1
        by_style[(row["style"], row["bucket"])] += 1
        by_bucket[row["bucket"]] += 1
    return {
        "margin_distributions": {key: _quantiles(values) for key, values in sorted(buckets.items())},
        "by_bucket": dict(by_bucket.most_common()),
        "by_domain_bucket": {f"{domain}:{bucket}": count for (domain, bucket), count in by_domain.most_common()},
        "by_style_bucket": {f"{style}:{bucket}": count for (style, bucket), count in by_style.most_common()},
    }


def _policy_summary_markdown(summary: dict) -> str:
    baseline = {
        (row["K"], row["policy"]): row
        for row in summary["mean"]
        if row["policy"] == "verifier_ranked"
    }
    lines = [
        "# Round-4 Selection Policy Replay",
        "",
        "| K | Policy | Mean Equiv | Delta vs Round4 | Helped | Hurt | Changed |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary["mean"], key=lambda item: (item["K"], item["policy"])):
        base = baseline.get((row["K"], "verifier_ranked"), {"mean_equiv_rate": row["mean_equiv_rate"]})
        delta = row["mean_equiv_rate"] - base["mean_equiv_rate"]
        lines.append(
            f"| {row['K']} | `{row['policy']}` | {row['mean_equiv_rate']:.4f} | {delta:+.4f} | "
            f"{row['total_helped_vs_round4']} | {row['total_hurt_vs_round4']} | {row['total_changed_vs_round4']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _diagnostics_markdown(diagnostics: dict) -> str:
    lines = [
        "# Round-4 Score Diagnostics",
        "",
        "## Buckets",
        "",
    ]
    for bucket, count in diagnostics["by_bucket"].items():
        lines.append(f"- `{bucket}`: `{count}`")
    lines.extend(["", "## Margin Distributions", ""])
    for name, stats in diagnostics["margin_distributions"].items():
        lines.append(f"- `{name}`: `{stats}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze fixed round-4 selector policies on cached pools")
    parser.add_argument(
        "--candidate_dump",
        action="append",
        default=[
            "results/vcsr/bestofk_round4_holdout_eval_clean/candidate_dump.jsonl",
            "results/vcsr/bestofk_round3_holdout_eval/candidate_dump.jsonl",
            "results/vcsr/bestofk_pilot/candidate_dump.jsonl",
            "results/vcsr/bestofk_ranking_round2_pool/candidate_dump.jsonl",
        ],
    )
    parser.add_argument("--round4_selection", default="results/verifier/best_current/selection.yaml")
    parser.add_argument("--round3_selection", default="results/verifier/ranking_aligned_round3/retrain_from_round2_multipool/selection.yaml")
    parser.add_argument("--output_dir", default="results/vcsr/round4_selection_analysis")
    parser.add_argument("--k_values", type=int, nargs="*", default=[4, 8])
    parser.add_argument("--margins", type=float, nargs="*", default=[0.02, 0.05, 0.10, 0.15])
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.00, 0.01, 0.03, 0.05])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _run_self_tests()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_file_logging(output_dir)
    _write_process_info(output_dir, sys.argv)
    started_at = time.time()
    candidate_dumps = [Path(path) for path in args.candidate_dump]
    _write_progress(output_dir, status="starting", completed_pools=0, total_pools=len(candidate_dumps), current_pool="", started_at=started_at)

    logger.info("Loading round-4 scorer: %s", args.round4_selection)
    round4_scorer = VerifierScorer(selection_path=args.round4_selection)
    logger.info("Loading round-3 scorer: %s", args.round3_selection)
    round3_scorer = VerifierScorer(selection_path=args.round3_selection)
    policy_fns = _policy_factories(args.margins, args.alphas)

    all_policy_rows: list[dict] = []
    all_changed_rows: list[dict] = []
    all_diagnostic_rows: list[dict] = []

    for idx, candidate_dump in enumerate(candidate_dumps, start=1):
        pool_name = candidate_dump.parent.name
        logger.info("Analyzing pool %d/%d: %s", idx, len(candidate_dumps), candidate_dump)
        _write_progress(
            output_dir,
            status="analyzing",
            completed_pools=idx - 1,
            total_pools=len(candidate_dumps),
            current_pool=str(candidate_dump),
            started_at=started_at,
        )
        nl_by_name = _load_nl_lookup(candidate_dump)
        meta_by_row, candidates_by_row = _load_candidate_dump(candidate_dump)
        scored_by_row = _score_candidates(
            nl_by_name=nl_by_name,
            meta_by_row=meta_by_row,
            candidates_by_row=candidates_by_row,
            round4_scorer=round4_scorer,
            round3_scorer=round3_scorer,
            batch_size=args.batch_size,
        )
        policy_rows, changed_rows, diagnostic_rows = _analyze_pool(
            pool_name=pool_name,
            candidate_dump=candidate_dump,
            scored_by_row=scored_by_row,
            meta_by_row=meta_by_row,
            k_values=sorted(set(args.k_values)),
            policy_fns=policy_fns,
            seed=args.seed,
        )
        all_policy_rows.extend(policy_rows)
        all_changed_rows.extend(changed_rows)
        all_diagnostic_rows.extend(diagnostic_rows)
        logger.info(
            "Pool done: %s policy_rows=%d changed_rows=%d",
            pool_name,
            len(policy_rows),
            len(changed_rows),
        )
        _flush_logs()

    policy_summary = _aggregate_policy_rows(all_policy_rows)
    diagnostics = _aggregate_diagnostics(all_diagnostic_rows)
    hard_changes = [
        row
        for row in all_changed_rows
        if row["oracle"] and (row["helped_vs_round4"] or row["hurt_vs_round4"])
    ]
    hard_changes.sort(key=lambda row: (not row["hurt_vs_round4"], row["K"], row["planetarium_name"], row["policy"]))

    score_diagnostics = {
        "inputs": {
            "candidate_dumps": [str(path) for path in candidate_dumps],
            "round4_selection": args.round4_selection,
            "round3_selection": args.round3_selection,
            "k_values": sorted(set(args.k_values)),
        },
        "diagnostics": diagnostics,
        "diagnostic_rows": all_diagnostic_rows,
    }
    policy_report = {
        "inputs": score_diagnostics["inputs"],
        "policy_grid": {
            "margins": args.margins,
            "alphas": args.alphas,
        },
        "per_pool": policy_summary["per_pool"],
        "mean": policy_summary["mean"],
    }

    _write_json(output_dir / "score_diagnostics.json", score_diagnostics)
    _write_json(output_dir / "policy_replay_summary.json", policy_report)
    with open(output_dir / "score_diagnostics.md", "w", encoding="utf-8") as f:
        f.write(_diagnostics_markdown(diagnostics))
    with open(output_dir / "policy_replay_summary.md", "w", encoding="utf-8") as f:
        f.write(_policy_summary_markdown(policy_summary))
    with open(output_dir / "changed_rows.jsonl", "w", encoding="utf-8") as f:
        for row in hard_changes:
            f.write(json.dumps(row) + "\n")
    _write_progress(output_dir, status="completed", completed_pools=len(candidate_dumps), total_pools=len(candidate_dumps), current_pool="", started_at=started_at)
    logger.info("Saved round-4 selection analysis to %s", output_dir)


if __name__ == "__main__":
    main()
