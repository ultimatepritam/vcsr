"""
Wrappers for Fast Downward planner and VAL plan validator.

These tools are C++ binaries typically built under WSL/Linux.
This module provides Python interfaces that call them via subprocess.
Falls back gracefully when binaries are not available.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DOWNWARD_PATH = os.environ.get(
    "FAST_DOWNWARD_PATH", "fast-downward.py"
)
DEFAULT_VAL_PATH = os.environ.get("VAL_PATH", "validate")

WSL_DOWNWARD_PATH = os.environ.get(
    "WSL_FAST_DOWNWARD_PATH", "/opt/downward/fast-downward.py"
)
WSL_VAL_PATH = os.environ.get("WSL_VAL_PATH", "/opt/VAL/validate")


@dataclass
class PlanResult:
    success: bool
    plan: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1


@dataclass
class ValidationResult:
    valid: bool
    stdout: str = ""
    stderr: str = ""


def _run_command(
    cmd: list[str],
    timeout: int = 30,
    use_wsl: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command, optionally via WSL."""
    if use_wsl:
        cmd = ["wsl", "--distribution", "Ubuntu-22.04", "--"] + cmd

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _write_temp_files(
    domain_pddl: str,
    problem_pddl: str,
    use_wsl: bool = False,
) -> tuple[str, str, str]:
    """
    Write domain and problem PDDL to temp files.
    Returns (tmpdir, domain_path, problem_path).
    If use_wsl, writes to /tmp/ via WSL for cross-OS compatibility.
    """
    if use_wsl:
        tmpdir = "/tmp/vcsr_planner"
        _run_command(["mkdir", "-p", tmpdir], use_wsl=True)

        for name, content in [("domain.pddl", domain_pddl), ("problem.pddl", problem_pddl)]:
            path = f"{tmpdir}/{name}"
            escaped = content.replace("'", "'\\''")
            _run_command(["bash", "-c", f"cat > {path} << 'PDDLEOF'\n{content}\nPDDLEOF"], use_wsl=True)

        return tmpdir, f"{tmpdir}/domain.pddl", f"{tmpdir}/problem.pddl"
    else:
        tmpdir = tempfile.mkdtemp(prefix="vcsr_planner_")
        domain_path = os.path.join(tmpdir, "domain.pddl")
        problem_path = os.path.join(tmpdir, "problem.pddl")
        with open(domain_path, "w") as f:
            f.write(domain_pddl)
        with open(problem_path, "w") as f:
            f.write(problem_pddl)
        return tmpdir, domain_path, problem_path


def check_downward_available(use_wsl: bool = False) -> bool:
    """Check if Fast Downward is accessible."""
    try:
        path = WSL_DOWNWARD_PATH if use_wsl else DEFAULT_DOWNWARD_PATH
        result = _run_command(
            ["python3", path, "--help"],
            timeout=10,
            use_wsl=use_wsl,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def check_val_available(use_wsl: bool = False) -> bool:
    """Check if VAL (validate) is accessible."""
    try:
        path = WSL_VAL_PATH if use_wsl else DEFAULT_VAL_PATH
        result = _run_command(
            [path, "--help"],
            timeout=10,
            use_wsl=use_wsl,
        )
        return True  # VAL may return non-zero for --help but still be available
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def run_fast_downward(
    domain_pddl: str,
    problem_pddl: str,
    search_algorithm: str = "astar(lmcut())",
    timeout: int = 30,
    use_wsl: bool = False,
) -> PlanResult:
    """
    Run Fast Downward to find a plan.

    Returns PlanResult with success=True if a plan was found.
    """
    downward_path = WSL_DOWNWARD_PATH if use_wsl else DEFAULT_DOWNWARD_PATH
    tmpdir, domain_path, problem_path = _write_temp_files(
        domain_pddl, problem_pddl, use_wsl=use_wsl
    )

    plan_file = f"{tmpdir}/sas_plan" if use_wsl else os.path.join(tmpdir, "sas_plan")

    try:
        cmd = [
            "python3", downward_path,
            "--plan-file", plan_file,
            domain_path, problem_path,
            "--search", search_algorithm,
        ]

        result = _run_command(cmd, timeout=timeout, use_wsl=use_wsl)

        plan_text = None
        if use_wsl:
            cat_result = _run_command(["cat", plan_file], use_wsl=True)
            if cat_result.returncode == 0:
                plan_text = cat_result.stdout
        else:
            plan_path = Path(plan_file)
            if plan_path.exists():
                plan_text = plan_path.read_text()

        return PlanResult(
            success=plan_text is not None and len(plan_text.strip()) > 0,
            plan=plan_text,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )

    except subprocess.TimeoutExpired:
        return PlanResult(success=False, stderr="Timeout")
    except Exception as e:
        return PlanResult(success=False, stderr=str(e))
    finally:
        if not use_wsl and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


def run_val(
    domain_pddl: str,
    problem_pddl: str,
    plan_text: str,
    timeout: int = 10,
    use_wsl: bool = False,
) -> ValidationResult:
    """
    Run VAL to validate a plan against domain and problem.
    """
    val_path = WSL_VAL_PATH if use_wsl else DEFAULT_VAL_PATH
    tmpdir, domain_path, problem_path = _write_temp_files(
        domain_pddl, problem_pddl, use_wsl=use_wsl
    )

    plan_file = f"{tmpdir}/plan.txt" if use_wsl else os.path.join(tmpdir, "plan.txt")

    if use_wsl:
        _run_command(
            ["bash", "-c", f"cat > {plan_file} << 'PLANEOF'\n{plan_text}\nPLANEOF"],
            use_wsl=True,
        )
    else:
        with open(plan_file, "w") as f:
            f.write(plan_text)

    try:
        cmd = [val_path, domain_path, problem_path, plan_file]
        result = _run_command(cmd, timeout=timeout, use_wsl=use_wsl)

        valid = "Plan valid" in result.stdout or result.returncode == 0
        return ValidationResult(
            valid=valid,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    except subprocess.TimeoutExpired:
        return ValidationResult(valid=False, stderr="Timeout")
    except Exception as e:
        return ValidationResult(valid=False, stderr=str(e))
    finally:
        if not use_wsl and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


def check_solvability(
    domain_pddl: str,
    problem_pddl: str,
    timeout: int = 30,
    use_wsl: bool = False,
) -> bool:
    """Convenience: check if a problem is solvable via Fast Downward."""
    result = run_fast_downward(
        domain_pddl, problem_pddl, timeout=timeout, use_wsl=use_wsl
    )
    return result.success


def get_tool_status() -> dict[str, bool]:
    """Report availability of planning tools in both native and WSL modes."""
    status = {
        "fast_downward_native": check_downward_available(use_wsl=False),
        "val_native": check_val_available(use_wsl=False),
        "fast_downward_wsl": check_downward_available(use_wsl=True),
        "val_wsl": check_val_available(use_wsl=True),
    }
    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    status = get_tool_status()
    print("Planning tool status:")
    for tool, available in status.items():
        print(f"  {tool}: {'AVAILABLE' if available else 'NOT FOUND'}")
