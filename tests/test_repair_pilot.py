import unittest

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


if __name__ == "__main__":
    unittest.main()
