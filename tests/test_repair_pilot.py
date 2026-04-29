import unittest

from generation.prompts import make_repair_prompt
from scripts.run_verifier_bestofk import should_accept_guarded_repair, should_attempt_repair
from scripts.run_repair_pilot import RepairCase, build_feedback, select_repair_candidate_from_scored


class RepairPilotTests(unittest.TestCase):
    def test_no_selected_candidate_when_no_candidates(self) -> None:
        self.assertIsNone(select_repair_candidate_from_scored([]))

    def test_no_selected_candidate_when_top_is_unparseable(self) -> None:
        rows = [
            {
                "candidate_index": 0,
                "parseable": False,
                "pddl": "",
                "equivalent": False,
                "round4_score": 0.99,
            }
        ]
        self.assertIsNone(select_repair_candidate_from_scored(rows))

    def test_no_selected_candidate_when_top_is_equivalent(self) -> None:
        rows = [
            {
                "candidate_index": 0,
                "parseable": True,
                "pddl": "(define (problem x))",
                "equivalent": True,
                "round4_score": 0.99,
            },
            {
                "candidate_index": 1,
                "parseable": True,
                "pddl": "(define (problem y))",
                "equivalent": False,
                "round4_score": 0.10,
            },
        ]
        self.assertIsNone(select_repair_candidate_from_scored(rows))

    def test_selected_candidate_when_top_is_parseable_non_equivalent(self) -> None:
        rows = [
            {
                "candidate_index": 0,
                "parseable": True,
                "pddl": "(define (problem x))",
                "equivalent": False,
                "round4_score": 0.99,
            },
            {
                "candidate_index": 1,
                "parseable": True,
                "pddl": "(define (problem y))",
                "equivalent": True,
                "round4_score": 0.10,
            },
        ]
        selected = select_repair_candidate_from_scored(rows)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["candidate_index"], 0)

    def test_feedback_does_not_leak_gold_or_equivalence_label(self) -> None:
        case = RepairCase(
            pool="pool",
            row_index=0,
            planetarium_name="name",
            domain="blocksworld",
            style="abstract/abstract",
            natural_language="Move blocks.",
            gold_pddl="SECRET_GOLD",
            is_placeholder=False,
            selected_index=0,
            selected_pddl="BAD_PDDL",
            selected_score=0.7,
            selected_parseable=True,
            selected_solvable=True,
            selected_equivalent=False,
            selected_planner_error=None,
        )
        feedback = build_feedback(case)
        self.assertNotIn("SECRET_GOLD", feedback)
        self.assertNotIn("BAD_PDDL", feedback)
        self.assertNotIn("equivalent", feedback.lower())

    def test_gripper_repair_prompt_uses_planetarium_conventions(self) -> None:
        prompt = make_repair_prompt(
            natural_language="You have 2 rooms. Gripper gripper1 is free.",
            candidate_pddl="(define (problem bad))",
            domain="gripper",
            feedback="parseable candidate",
        )
        self.assertIn("Do NOT use :typing", prompt)
        self.assertIn("Use (:requirements :strips)", prompt)
        self.assertIn("(room roomN)", prompt)
        self.assertIn("(ball ballN)", prompt)
        self.assertIn("(gripper gripperN)", prompt)
        self.assertIn("room1, room2", prompt)

    def test_repair_policy_does_not_repair_unparseable_selection(self) -> None:
        self.assertFalse(
            should_attempt_repair(
                k=8,
                selected_index=0,
                selected_parseable=False,
                repair_cfg={"enabled": True, "K": 8},
            )
        )

    def test_repair_policy_only_repairs_configured_k(self) -> None:
        self.assertFalse(
            should_attempt_repair(
                k=1,
                selected_index=0,
                selected_parseable=True,
                repair_cfg={"enabled": True, "K": 8},
            )
        )

    def test_guarded_repair_rejects_unparseable_repair(self) -> None:
        self.assertFalse(
            should_accept_guarded_repair(
                repair_parseable=False,
                original_score=0.8,
                repair_score=0.9,
                margin=0.05,
            )
        )

    def test_guarded_repair_rejects_score_below_margin(self) -> None:
        self.assertFalse(
            should_accept_guarded_repair(
                repair_parseable=True,
                original_score=0.8,
                repair_score=0.7,
                margin=0.05,
            )
        )

    def test_guarded_repair_accepts_score_within_margin(self) -> None:
        self.assertTrue(
            should_accept_guarded_repair(
                repair_parseable=True,
                original_score=0.8,
                repair_score=0.76,
                margin=0.05,
            )
        )
        self.assertTrue(
            should_attempt_repair(
                k=8,
                selected_index=0,
                selected_parseable=True,
                repair_cfg={"enabled": True, "K": 8},
            )
        )


if __name__ == "__main__":
    unittest.main()
