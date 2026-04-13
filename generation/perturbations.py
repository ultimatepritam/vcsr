"""
Domain-aware PDDL perturbation engine for generating hard negatives.

Each perturbation keeps the PDDL parseable but introduces a targeted semantic
error. The perturbation type is returned alongside the mutated string so we
can track which kinds of negatives are most useful for verifier training.
"""

import random
import re
from typing import Optional


def _find_section(pddl: str, section: str) -> Optional[re.Match]:
    """Find a top-level PDDL section like :init, :goal, :objects."""
    pattern = rf"\(:{section}\s"
    match = re.search(pattern, pddl)
    if not match:
        return None

    start = match.start()
    depth = 0
    for i in range(start, len(pddl)):
        if pddl[i] == "(":
            depth += 1
        elif pddl[i] == ")":
            depth -= 1
            if depth == 0:
                return re.Match  # can't construct re.Match, use slice approach
    return None


def _extract_section(pddl: str, section: str) -> tuple[Optional[str], int, int]:
    """
    Extract a section's full text and its (start, end) positions.
    Returns (section_text, start, end) or (None, -1, -1).
    """
    pattern = rf"\(:{section}\s"
    match = re.search(pattern, pddl)
    if not match:
        return None, -1, -1

    start = match.start()
    depth = 0
    for i in range(start, len(pddl)):
        if pddl[i] == "(":
            depth += 1
        elif pddl[i] == ")":
            depth -= 1
            if depth == 0:
                return pddl[start : i + 1], start, i + 1
    return None, -1, -1


def _extract_predicates(section_body: str) -> list[str]:
    """Extract individual predicates from inside a section body."""
    return re.findall(r"\([^()]+\)", section_body)


def _replace_section(pddl: str, section: str, new_content: str) -> str:
    """Replace a PDDL section with new content."""
    old, start, end = _extract_section(pddl, section)
    if old is None:
        return pddl
    return pddl[:start] + new_content + pddl[end:]


# ---------------------------------------------------------------------------
# Core perturbation functions
# ---------------------------------------------------------------------------


def swap_goal_pred(pddl: str, rng: random.Random) -> Optional[str]:
    """Swap the arguments of a random goal predicate (e.g. (on b1 b2) -> (on b2 b1))."""
    goal_text, gs, ge = _extract_section(pddl, "goal")
    if goal_text is None:
        return None

    preds = _extract_predicates(goal_text)
    multi_arg = [p for p in preds if len(p.split()) >= 3]
    if not multi_arg:
        return None

    target = rng.choice(multi_arg)
    parts = target.strip("()").split()
    if len(parts) < 3:
        return None

    name = parts[0]
    args = parts[1:]
    rng.shuffle(args)
    new_pred = f"({name} {' '.join(args)})"

    return pddl.replace(target, new_pred, 1)


def drop_init_pred(pddl: str, rng: random.Random) -> Optional[str]:
    """Remove a random predicate from :init."""
    init_text, start, end = _extract_section(pddl, "init")
    if init_text is None:
        return None

    preds = _extract_predicates(init_text)
    if len(preds) <= 2:
        return None

    drop_idx = rng.randint(0, len(preds) - 1)
    preds.pop(drop_idx)
    new_init = "(:init " + " ".join(preds) + ")"
    return pddl[:start] + new_init + pddl[end:]


def add_extra_object(pddl: str, rng: random.Random) -> Optional[str]:
    """Add a spurious extra object to :objects."""
    obj_text, start, end = _extract_section(pddl, "objects")
    if obj_text is None:
        return None

    inner = obj_text[len("(:objects"): -1].strip()
    objects = inner.split()
    objects.append(f"extra_{rng.randint(1, 99)}")
    new_obj = "(:objects " + " ".join(objects) + ")"
    return pddl[:start] + new_obj + pddl[end:]


def drop_goal_conjunct(pddl: str, rng: random.Random) -> Optional[str]:
    """Remove a random conjunct from the goal."""
    goal_text, start, end = _extract_section(pddl, "goal")
    if goal_text is None:
        return None

    preds = _extract_predicates(goal_text)
    goal_preds = [p for p in preds if not p.startswith("(and")]
    if len(goal_preds) <= 1:
        return None

    drop = rng.choice(goal_preds)
    new_goal_text = goal_text.replace(drop, "", 1)
    new_goal_text = re.sub(r"\s+", " ", new_goal_text).strip()
    return pddl[:start] + new_goal_text + pddl[end:]


def duplicate_init_pred_wrong_args(pddl: str, rng: random.Random) -> Optional[str]:
    """Duplicate an :init predicate but with shuffled arguments."""
    init_text, start, end = _extract_section(pddl, "init")
    if init_text is None:
        return None

    preds = _extract_predicates(init_text)
    multi_arg = [p for p in preds if len(p.split()) >= 3]
    if not multi_arg:
        return None

    target = rng.choice(multi_arg)
    parts = target.strip("()").split()
    name = parts[0]
    args = parts[1:]
    rng.shuffle(args)
    new_pred = f"({name} {' '.join(args)})"

    if new_pred == target:
        return None

    all_preds = preds + [new_pred]
    new_init = "(:init " + " ".join(all_preds) + ")"
    return pddl[:start] + new_init + pddl[end:]


def negate_goal_predicate(pddl: str, rng: random.Random) -> Optional[str]:
    """
    Negate a goal predicate. For blocksworld: flip on<->on-table or clear<->holding.
    For gripper: flip at<->carry or free<->carry. Generic: wrap in (not ...).
    """
    goal_text, start, end = _extract_section(pddl, "goal")
    if goal_text is None:
        return None

    preds = _extract_predicates(goal_text)
    goal_preds = [p for p in preds if not p.startswith("(and")]
    if not goal_preds:
        return None

    target = rng.choice(goal_preds)
    parts = target.strip("()").split()
    name = parts[0]

    flip_map = {
        "on": "on-table",
        "on-table": "on",
        "clear": "holding",
        "holding": "clear",
        "at": "carry",
        "carry": "at",
        "free": "carry",
    }

    if name in flip_map:
        new_name = flip_map[name]
        if new_name in ("on-table", "holding", "clear", "free"):
            new_pred = f"({new_name} {parts[1]})" if len(parts) >= 2 else target
        elif new_name == "on":
            new_pred = (
                f"({new_name} {parts[1]} {parts[1]})"
                if len(parts) >= 2
                else target
            )
        else:
            args = " ".join(parts[1:]) if len(parts) > 1 else ""
            new_pred = f"({new_name} {args})".strip()
    else:
        new_pred = f"(not {target})"

    if new_pred == target:
        return None

    return pddl.replace(target, new_pred, 1)


def swap_pred_args(pddl: str, rng: random.Random) -> Optional[str]:
    """Swap arguments of a random multi-argument predicate in :init or :goal."""
    section = rng.choice(["init", "goal"])
    sec_text, start, end = _extract_section(pddl, section)
    if sec_text is None:
        return None

    preds = _extract_predicates(sec_text)
    multi_arg = [p for p in preds if len(p.split()) >= 3]
    if not multi_arg:
        return None

    target = rng.choice(multi_arg)
    parts = target.strip("()").split()
    name = parts[0]
    args = parts[1:]
    original_args = list(args)
    rng.shuffle(args)
    if args == original_args:
        args.reverse()
    if args == original_args:
        return None

    new_pred = f"({name} {' '.join(args)})"
    new_sec = sec_text.replace(target, new_pred, 1)
    return pddl[:start] + new_sec + pddl[end:]


def swap_blocksworld_objects(pddl: str, rng: random.Random) -> Optional[str]:
    """Swap two block names throughout the PDDL (changes semantics)."""
    obj_text, _, _ = _extract_section(pddl, "objects")
    if obj_text is None:
        return None

    inner = obj_text[len("(:objects"): -1].strip()
    objects = [o for o in inner.split() if o.startswith("b")]
    if len(objects) < 2:
        return None

    a, b = rng.sample(objects, 2)
    placeholder = f"__SWAP_PLACEHOLDER_{rng.randint(10000, 99999)}__"
    result = pddl.replace(f" {a} ", f" {placeholder} ")
    result = result.replace(f" {a})", f" {placeholder})")
    result = result.replace(f" {b} ", f" {a} ")
    result = result.replace(f" {b})", f" {a})")
    result = result.replace(f" {placeholder} ", f" {b} ")
    result = result.replace(f" {placeholder})", f" {b})")

    if result == pddl:
        return None
    return result


def swap_gripper_rooms(pddl: str, rng: random.Random) -> Optional[str]:
    """Swap rooma and roomb throughout the PDDL."""
    if "rooma" not in pddl or "roomb" not in pddl:
        return None

    placeholder = "__ROOM_SWAP__"
    result = pddl.replace("rooma", placeholder)
    result = result.replace("roomb", "rooma")
    result = result.replace(placeholder, "roomb")

    if result == pddl:
        return None
    return result


def add_contradictory_init(pddl: str, rng: random.Random) -> Optional[str]:
    """
    Add a predicate to :init that contradicts an existing one.
    E.g., if (clear b1) is present, add (holding b1).
    """
    init_text, start, end = _extract_section(pddl, "init")
    if init_text is None:
        return None

    preds = _extract_predicates(init_text)
    contradiction_map = {
        "clear": "holding",
        "on-table": "holding",
        "arm-empty": None,
        "free": "carry",
    }

    candidates = []
    for p in preds:
        parts = p.strip("()").split()
        name = parts[0]
        if name in contradiction_map and contradiction_map[name] is not None:
            candidates.append((p, parts))

    if not candidates:
        return None

    orig_pred, parts = rng.choice(candidates)
    name = parts[0]
    contra_name = contradiction_map[name]
    args = " ".join(parts[1:])

    if contra_name == "holding" and len(parts) >= 2:
        new_pred = f"({contra_name} {parts[1]})"
    elif contra_name == "carry" and len(parts) >= 2:
        new_pred = f"({contra_name} {parts[1]} left)"
    else:
        new_pred = f"({contra_name} {args})"

    all_preds = preds + [new_pred]
    new_init = "(:init " + " ".join(all_preds) + ")"
    return pddl[:start] + new_init + pddl[end:]


# ---------------------------------------------------------------------------
# Registry and public API
# ---------------------------------------------------------------------------

PERTURBATION_REGISTRY = {
    "swap_goal_pred": swap_goal_pred,
    "drop_init_pred": drop_init_pred,
    "add_extra_object": add_extra_object,
    "drop_goal_conjunct": drop_goal_conjunct,
    "duplicate_init_wrong_args": duplicate_init_pred_wrong_args,
    "negate_goal": negate_goal_predicate,
    "swap_pred_args": swap_pred_args,
    "swap_blocksworld_objects": swap_blocksworld_objects,
    "swap_gripper_rooms": swap_gripper_rooms,
    "add_contradictory_init": add_contradictory_init,
}

DOMAIN_PERTURBATIONS = {
    "blocksworld": [
        "swap_goal_pred",
        "drop_init_pred",
        "add_extra_object",
        "drop_goal_conjunct",
        "duplicate_init_wrong_args",
        "negate_goal",
        "swap_pred_args",
        "swap_blocksworld_objects",
        "add_contradictory_init",
    ],
    "gripper": [
        "swap_goal_pred",
        "drop_init_pred",
        "add_extra_object",
        "drop_goal_conjunct",
        "duplicate_init_wrong_args",
        "negate_goal",
        "swap_pred_args",
        "swap_gripper_rooms",
        "add_contradictory_init",
    ],
}

ALL_PERTURBATIONS = list(PERTURBATION_REGISTRY.keys())


def generate_perturbations(
    gold_pddl: str,
    domain: str = "",
    n: int = 2,
    seed: int = 0,
    allowed_types: Optional[list[str]] = None,
) -> list[tuple[str, str]]:
    """
    Generate n perturbations of a gold PDDL problem.

    Args:
        gold_pddl: The ground-truth PDDL string.
        domain: Domain name for domain-specific perturbation selection.
        n: Number of perturbations to generate.
        seed: Random seed for reproducibility.
        allowed_types: Restrict to these perturbation types. If None, uses
                       domain-aware defaults or all perturbations.

    Returns:
        List of (perturbed_pddl, perturbation_name) tuples.
        May return fewer than n if some perturbations are inapplicable.
    """
    rng = random.Random(seed)

    if allowed_types:
        pool = [t for t in allowed_types if t in PERTURBATION_REGISTRY]
    elif domain in DOMAIN_PERTURBATIONS:
        pool = DOMAIN_PERTURBATIONS[domain]
    else:
        pool = ALL_PERTURBATIONS

    results = []
    attempted = set()
    max_attempts = n * 3

    for attempt_i in range(max_attempts):
        if len(results) >= n:
            break

        ptype = rng.choice(pool)
        attempt_key = (ptype, attempt_i)
        if attempt_key in attempted:
            continue
        attempted.add(attempt_key)

        fn = PERTURBATION_REGISTRY[ptype]
        perturbed = fn(gold_pddl, rng)

        if perturbed is not None and perturbed != gold_pddl:
            results.append((perturbed, ptype))

    return results
