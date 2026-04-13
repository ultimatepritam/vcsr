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
