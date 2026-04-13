"""
Equivalence evaluation wrapper around Planetarium's evaluation API.

Provides batch evaluation with parseability, solvability (optional),
and semantic equivalence metrics, stratified by domain and description style.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import planetarium
from planetarium.builder import build
from planetarium.oracle import fully_specify
from planetarium import metric

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    parseable: bool = False
    solveable: bool = False
    equivalent: bool = False
    error: Optional[str] = None


@dataclass
class BatchMetrics:
    total: int = 0
    parse_count: int = 0
    solve_count: int = 0
    equiv_count: int = 0
    error_count: int = 0

    @property
    def parse_rate(self) -> float:
        return self.parse_count / max(self.total, 1)

    @property
    def solve_rate(self) -> float:
        return self.solve_count / max(self.total, 1)

    @property
    def equiv_rate(self) -> float:
        return self.equiv_count / max(self.total, 1)

    @property
    def equiv_given_parse(self) -> float:
        return self.equiv_count / max(self.parse_count, 1)

    def __repr__(self) -> str:
        return (
            f"BatchMetrics(n={self.total}, "
            f"parse={self.parse_rate:.3f}, "
            f"solve={self.solve_rate:.3f}, "
            f"equiv={self.equiv_rate:.3f}, "
            f"equiv|parse={self.equiv_given_parse:.3f}, "
            f"errors={self.error_count})"
        )


def check_equivalence_lightweight(
    gold_pddl: str,
    candidate_pddl: str,
    is_placeholder: bool = False,
) -> EvalResult:
    """
    Check semantic equivalence WITHOUT running a planner (no solvability check).
    Uses Planetarium's graph-based equivalence only.
    """
    result = EvalResult()

    try:
        gold_graph = build(gold_pddl)
    except Exception as e:
        result.error = f"Failed to parse gold PDDL: {e}"
        return result

    try:
        cand_graph = build(candidate_pddl)
        result.parseable = True
    except Exception:
        return result

    try:
        if gold_graph == cand_graph:
            result.equivalent = True
        elif not metric.equals(gold_graph.init(), cand_graph.init()):
            result.equivalent = False
        else:
            result.equivalent = metric.equals(
                fully_specify(gold_graph, return_reduced=True),
                fully_specify(cand_graph, return_reduced=True),
                is_placeholder=is_placeholder,
            )
    except Exception as e:
        result.error = f"Equivalence check failed: {e}"

    return result


def check_equivalence_full(
    gold_pddl: str,
    candidate_pddl: str,
    domain_str: Optional[str] = None,
    is_placeholder: bool = False,
    check_solveable: bool = True,
) -> EvalResult:
    """
    Full evaluation using Planetarium's evaluate(), which includes
    optional planner-based solvability checking via Fast Downward + VAL.
    """
    result = EvalResult()

    try:
        p, s, e = planetarium.evaluate(
            gold_pddl,
            candidate_pddl,
            domain_str=domain_str,
            is_placeholder=is_placeholder,
            check_solveable=check_solveable,
        )
        result.parseable = p
        result.solveable = s
        result.equivalent = e
    except Exception as ex:
        result.error = str(ex)

    return result


def evaluate_batch(
    gold_pddls: list[str],
    candidate_pddls: list[str],
    is_placeholders: Optional[list[bool]] = None,
    use_planner: bool = False,
    domain_strs: Optional[list[str]] = None,
) -> tuple[BatchMetrics, list[EvalResult]]:
    """
    Evaluate a batch of (gold, candidate) PDDL pairs.

    Returns aggregate metrics and per-instance results.
    """
    assert len(gold_pddls) == len(candidate_pddls)
    n = len(gold_pddls)

    if is_placeholders is None:
        is_placeholders = [False] * n
    if domain_strs is None:
        domain_strs = [None] * n

    metrics = BatchMetrics(total=n)
    results = []

    for i in range(n):
        if use_planner:
            res = check_equivalence_full(
                gold_pddls[i],
                candidate_pddls[i],
                domain_str=domain_strs[i],
                is_placeholder=is_placeholders[i],
            )
        else:
            res = check_equivalence_lightweight(
                gold_pddls[i],
                candidate_pddls[i],
                is_placeholder=is_placeholders[i],
            )

        if res.parseable:
            metrics.parse_count += 1
        if res.solveable:
            metrics.solve_count += 1
        if res.equivalent:
            metrics.equiv_count += 1
        if res.error:
            metrics.error_count += 1

        results.append(res)

        if (i + 1) % 100 == 0:
            logger.info(f"Evaluated {i+1}/{n}: {metrics}")

    return metrics, results


def stratified_report(
    dataset_rows: list[dict],
    results: list[EvalResult],
) -> dict[str, BatchMetrics]:
    """
    Compute metrics stratified by domain and description style.
    Returns a dict mapping stratum name to BatchMetrics.
    """
    strata: dict[str, list[EvalResult]] = {}

    for row, res in zip(dataset_rows, results):
        domain = row["domain"]
        init_style = "abstract" if row["init_is_abstract"] else "explicit"
        goal_style = "abstract" if row["goal_is_abstract"] else "explicit"

        for key in [
            f"domain={domain}",
            f"init={init_style}",
            f"goal={goal_style}",
            f"style={init_style}/{goal_style}",
            "all",
        ]:
            strata.setdefault(key, []).append(res)

    report = {}
    for key, res_list in sorted(strata.items()):
        m = BatchMetrics(total=len(res_list))
        for r in res_list:
            if r.parseable:
                m.parse_count += 1
            if r.solveable:
                m.solve_count += 1
            if r.equivalent:
                m.equiv_count += 1
            if r.error:
                m.error_count += 1
        report[key] = m

    return report
