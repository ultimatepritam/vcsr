"""
Lightweight planner using Planetarium's built-in oracle planners.

For supported domains (blocksworld, gripper, floor-tile, rover-single),
this avoids needing Fast Downward entirely.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from planetarium.builder import build
from planetarium.oracle import plan, plan_to_string, ORACLES, DomainNotSupportedError

logger = logging.getLogger(__name__)

SUPPORTED_DOMAINS = list(ORACLES.keys())


@dataclass
class OraclePlanResult:
    solvable: bool
    plan_text: Optional[str] = None
    error: Optional[str] = None


def check_solvability_oracle(problem_pddl: str) -> OraclePlanResult:
    """
    Check if a PDDL problem is solvable using Planetarium's oracle planner.
    Only works for supported domains (blocksworld, gripper, floor-tile, rover-single).
    """
    try:
        problem_graph = build(problem_pddl)
    except Exception as e:
        return OraclePlanResult(solvable=False, error=f"Parse error: {e}")

    try:
        actions = plan(problem_graph)
        plan_str = plan_to_string(actions)
        return OraclePlanResult(solvable=True, plan_text=plan_str)
    except DomainNotSupportedError:
        return OraclePlanResult(
            solvable=False,
            error=f"Domain '{problem_graph.domain}' not supported by oracle planner",
        )
    except NotImplementedError as e:
        return OraclePlanResult(solvable=False, error=f"Not implemented: {e}")
    except Exception as e:
        return OraclePlanResult(solvable=False, error=f"Planning failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Supported oracle domains: {SUPPORTED_DOMAINS}")

    test_problem = """(define (problem test)
        (:domain blocksworld)
        (:requirements :strips)
        (:objects b1 b2)
        (:init (arm-empty) (clear b1) (clear b2) (on-table b1) (on-table b2))
        (:goal (and (arm-empty) (on b1 b2) (clear b1) (on-table b2)))
    )"""

    result = check_solvability_oracle(test_problem)
    print(f"\nSolvable: {result.solvable}")
    if result.plan_text:
        print(f"Plan:\n{result.plan_text}")
    if result.error:
        print(f"Error: {result.error}")
