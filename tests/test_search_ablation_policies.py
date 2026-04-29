import unittest

from scripts.analyze_search_ablation import (
    Candidate,
    _select_parse_solvable_index,
    _select_solvable_then_verifier,
    _select_verifier_ranked,
    _select_verifier_then_solvable_tiebreak,
)


class SearchAblationPolicyTests(unittest.TestCase):
    def test_no_candidates_returns_no_selection(self) -> None:
        self.assertIsNone(_select_verifier_ranked([]).selected_index)
        self.assertIsNone(_select_solvable_then_verifier([]).selected_index)
        self.assertIsNone(_select_parse_solvable_index([]).selected_index)

    def test_no_parseable_candidates_returns_no_selection(self) -> None:
        candidates = [Candidate(index=0, parseable=False, equivalent=False, pddl="", verifier_score=0.9)]
        self.assertIsNone(_select_verifier_ranked(candidates).selected_index)
        self.assertIsNone(_select_solvable_then_verifier(candidates).selected_index)
        self.assertIsNone(_select_parse_solvable_index(candidates).selected_index)

    def test_verifier_ranked_ignores_unparseable_high_score(self) -> None:
        candidates = [
            Candidate(index=0, parseable=False, equivalent=False, pddl="", verifier_score=0.99),
            Candidate(index=1, parseable=True, equivalent=False, pddl="x", verifier_score=0.2),
        ]
        self.assertEqual(_select_verifier_ranked(candidates).selected_index, 1)

    def test_solvable_then_verifier_requires_solvable_candidate(self) -> None:
        candidates = [
            Candidate(index=0, parseable=True, equivalent=False, pddl="x", verifier_score=0.9, solvable=False),
            Candidate(index=1, parseable=True, equivalent=False, pddl="y", verifier_score=0.4, solvable=True),
        ]
        self.assertEqual(_select_solvable_then_verifier(candidates).selected_index, 1)

    def test_verifier_then_solvable_tiebreak_prefers_solvable_near_top(self) -> None:
        candidates = [
            Candidate(index=0, parseable=True, equivalent=False, pddl="x", verifier_score=0.90, solvable=False),
            Candidate(index=1, parseable=True, equivalent=False, pddl="y", verifier_score=0.89, solvable=True),
        ]
        self.assertEqual(_select_verifier_then_solvable_tiebreak(candidates, margin=0.02).selected_index, 1)

    def test_verifier_then_solvable_tiebreak_keeps_large_score_gap_top(self) -> None:
        candidates = [
            Candidate(index=0, parseable=True, equivalent=False, pddl="x", verifier_score=0.90, solvable=False),
            Candidate(index=1, parseable=True, equivalent=False, pddl="y", verifier_score=0.70, solvable=True),
        ]
        self.assertEqual(_select_verifier_then_solvable_tiebreak(candidates, margin=0.02).selected_index, 0)

    def test_parse_solvable_index_uses_lowest_index(self) -> None:
        candidates = [
            Candidate(index=0, parseable=True, equivalent=False, pddl="x", verifier_score=0.1, solvable=False),
            Candidate(index=1, parseable=True, equivalent=False, pddl="y", verifier_score=0.9, solvable=True),
            Candidate(index=2, parseable=True, equivalent=False, pddl="z", verifier_score=0.95, solvable=True),
        ]
        self.assertEqual(_select_parse_solvable_index(candidates).selected_index, 1)


if __name__ == "__main__":
    unittest.main()
