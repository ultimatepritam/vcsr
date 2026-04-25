"""
Prompt templates for text-to-PDDL generation.

Provides structured prompts that enforce strict PDDL formatting constraints
for use with instruction-tuned language models.
"""

import planetarium

SYSTEM_PROMPT = """You are an expert AI planning assistant that translates natural language task descriptions into PDDL (Planning Domain Definition Language) problem files.

Given a natural language description of a planning task in a known domain, generate ONLY the PDDL problem definition. Output nothing else -- no explanations, no commentary, just the PDDL problem wrapped in a code block.

Requirements:
- Output must be a valid PDDL problem definition starting with (define (problem ...)
- Include all required sections: :domain, :requirements, :objects, :init, :goal
- Use :strips requirement unless the domain requires :typing
- Ensure balanced parentheses
- List ALL objects mentioned or implied by the description
- Include ALL initial state predicates described
- Include ALL goal conditions described"""


DOMAIN_CONTEXT = {
    "blocksworld": """Domain: Blocksworld (STRIPS)
Predicates: (clear ?x), (on-table ?x), (arm-empty), (holding ?x), (on ?x ?y)
Actions: pickup, putdown, stack, unstack
Objects are blocks named b1, b2, b3, etc.""",

    "gripper": """Domain: Gripper (STRIPS)
Predicates: (room ?r), (ball ?b), (gripper ?g), (at-robby ?r), (at ?b ?r), (free ?g), (carry ?o ?g)
Actions: move, pick, drop
Rooms are typically rooma, roomb. Grippers are left, right. Balls are ball1, ball2, etc.""",

    "floor-tile": """Domain: Floor-tile (typed STRIPS)
Types: robot, tile, color
Predicates: (robot-at ?r ?x), (up ?x ?y), (right ?x ?y), (painted ?x ?c), (robot-has ?r ?c), (available-color ?c)
Actions: change-color, paint-up, paint-down, paint-right, paint-left, up, down, right, left""",
}


def make_generation_prompt(
    natural_language: str,
    domain: str = "",
    include_domain_context: bool = True,
) -> str:
    """
    Build a prompt for PDDL generation from a natural language description.

    Args:
        natural_language: The task description in natural language.
        domain: The PDDL domain name (for domain context injection).
        include_domain_context: Whether to include domain-specific predicate info.

    Returns:
        The formatted prompt string.
    """
    parts = [SYSTEM_PROMPT, ""]

    if include_domain_context and domain in DOMAIN_CONTEXT:
        parts.append(DOMAIN_CONTEXT[domain])
        parts.append("")

    parts.append(f"Task description:\n{natural_language}")
    parts.append("")
    parts.append("PDDL problem definition:")

    return "\n".join(parts)


def make_repair_prompt(
    natural_language: str,
    candidate_pddl: str,
    domain: str = "",
    feedback: str = "",
) -> str:
    """
    Build a prompt for repairing a candidate PDDL that may be incorrect.

    Args:
        natural_language: The original task description.
        candidate_pddl: The PDDL candidate to repair.
        domain: The PDDL domain name.
        feedback: Optional feedback about what's wrong.

    Returns:
        The formatted repair prompt string.
    """
    if domain == "gripper":
        return make_gripper_repair_prompt(
            natural_language=natural_language,
            candidate_pddl=candidate_pddl,
            feedback=feedback,
        )

    parts = [
        "You are an expert AI planning assistant. A candidate PDDL problem was generated "
        "from a natural language description but may contain errors. Fix the PDDL to "
        "accurately represent the described task. Output ONLY the corrected PDDL problem "
        "definition -- no explanations.",
        "",
    ]

    if domain in DOMAIN_CONTEXT:
        parts.append(DOMAIN_CONTEXT[domain])
        parts.append("")

    parts.append(f"Task description:\n{natural_language}")
    parts.append("")
    parts.append(f"Candidate PDDL (may contain errors):\n{candidate_pddl}")
    parts.append("")

    if feedback:
        parts.append(f"Issues identified: {feedback}")
        parts.append("")

    parts.append("Corrected PDDL problem definition:")

    return "\n".join(parts)


def make_gripper_repair_prompt(
    natural_language: str,
    candidate_pddl: str,
    feedback: str = "",
) -> str:
    """Build a stricter Planetarium-style gripper repair prompt."""
    parts = [
        "You are repairing a Planetarium gripper-domain PDDL problem.",
        "Output ONLY the corrected PDDL problem definition -- no explanations.",
        "",
        "Planetarium gripper PDDL conventions:",
        "- Use (:domain gripper).",
        "- Use (:requirements :strips). Do NOT use :typing.",
        "- List objects in one untyped :objects list.",
        "- Room names must be room1, room2, room3, ... when the task says first room, second room, etc.",
        "- Ball names must be ball1, ball2, ball3, ... exactly as in the task.",
        "- Gripper names must be gripper1, gripper2, gripper3, ... exactly as in the task.",
        "- The :init must include unary type facts for every object: (room roomN), (ball ballN), and (gripper gripperN).",
        "- The :init must include exactly one (at-robby roomN) fact matching the task.",
        "- The :init must include every described (at ball room), (carry ball gripper), and (free gripper) fact.",
        "- A gripper that is carrying a ball is NOT free; do not include (free gripperN) for any gripper in a (carry ball gripperN) fact.",
        "- The :goal must include every requested goal fact and only requested goal facts.",
        "- Do not add (free gripperN) to the goal unless the natural-language goal explicitly says that gripper should be free.",
        "- For 'bring all balls into the room which already has the least balls', choose the room with the fewest initial balls and put every ball there in the goal.",
        "- For juggle/swap carrying tasks, preserve the exact requested final ball-to-gripper assignments and free grippers.",
        "- Do not invent rooma/roomb, left/right grippers, typed declarations, or missing unary facts.",
        "",
        f"Task description:\n{natural_language}",
        "",
        f"Candidate PDDL (may contain errors):\n{candidate_pddl}",
        "",
    ]
    if feedback:
        parts.append(f"Non-oracle feedback: {feedback}")
        parts.append("")
    parts.extend(
        [
            "Before writing the final PDDL, internally check:",
            "1. number and names of rooms, balls, and grippers",
            "2. all unary object facts are present in :init",
            "3. all carry/free/at facts from the initial state are present",
            "4. every goal fact from the natural-language task is present",
            "5. no typed object syntax is used",
            "",
            "Corrected PDDL problem definition:",
        ]
    )
    return "\n".join(parts)


def extract_pddl_from_response(response: str) -> str:
    """
    Extract PDDL content from an LLM response that may contain
    markdown code blocks or surrounding text.
    """
    import re

    code_block = re.search(r"```(?:pddl)?\s*\n(.*?)```", response, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()

    define_match = re.search(r"(\(define\s.*)", response, re.DOTALL)
    if define_match:
        text = define_match.group(1)
        depth = 0
        end = 0
        for i, c in enumerate(text):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > 0:
            return text[:end].strip()

    return response.strip()
