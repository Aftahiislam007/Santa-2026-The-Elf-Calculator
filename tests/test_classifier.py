import unittest
from unittest.mock import patch

from elfcalc import Puzzle, load_practice
from elfcalc.classifier import classify_puzzle, partition_mixed_targets
from elfcalc.solvers import SOLVERS, exact, get_solver


class ClassifierAndSolverWrapperTest(unittest.TestCase):
    def test_practice_classification_accuracy(self):
        practice = load_practice()
        family_hits = sum(classify_puzzle(p)[0] == p.family for p in practice)
        tier_hits = sum(classify_puzzle(p)[1] == p.design_tier for p in practice)
        self.assertGreaterEqual(family_hits, 197)
        self.assertEqual(tier_hits, len(practice))

    def test_partition_mixed_targets_preserves_all_targets(self):
        for p in load_practice():
            if p.family == "mixed":
                structured, random_targets = partition_mixed_targets(p)
                self.assertEqual(sorted(structured + random_targets), sorted(p.targets))
                self.assertGreater(len(random_targets), 0)

    def test_all_solvers_registered_and_callable(self):
        expected = {
            "baseline",
            "exact",
            "greedy",
            "greedy_v2",
            "mitm_basis",
            "mixed_hybrid",
            "mult_factor",
            "shared_dag",
        }
        self.assertTrue(expected.issubset(set(SOLVERS)))
        for name in expected:
            self.assertTrue(callable(get_solver(name)))

    def test_exact_skips_out_of_envelope_without_subprocess(self):
        big_limit = Puzzle("p1", (1,), (2, 3), 2048)
        many_targets = Puzzle("p2", (1,), (2, 3, 4, 5, 6, 7), 100)
        with patch("elfcalc.solvers.engine.run_engine") as mock_run:
            self.assertIsNone(exact.solve(big_limit))
            self.assertIsNone(exact.solve(many_targets))
            mock_run.assert_not_called()

    def test_wrapper_forwards_options_and_timeout(self):
        tiny = Puzzle("p3", (1,), (2, 3), 100)
        expected_engines = [
            ("exact", 15.0),
            ("greedy", 120.0),
            ("mitm_basis", 120.0),
            ("shared_dag", 120.0),
            ("mult_factor", 120.0),
            ("mixed_hybrid", 120.0),
            ("greedy_v2", 120.0),
        ]
        for name, default_timeout in expected_engines:
            solver = get_solver(name)
            with patch("elfcalc.solvers.engine.run_engine", return_value=[("+", 1, 1)]) as mock_run:
                plan = solver(tiny, restarts="4")
                self.assertEqual(plan, [("+", 1, 1)])
                mock_run.assert_called_once_with(
                    name, tiny, args=["--restarts", "4"], timeout=default_timeout
                )
            with patch("elfcalc.solvers.engine.run_engine", return_value=[]) as mock_run:
                solver(tiny, timeout="7.5", seed=2)
                mock_run.assert_called_once_with(
                    name, tiny, args=["--seed", "2"], timeout=7.5
                )

    def test_classifier_and_partition_edge_cases(self):
        self.assertEqual(classify_puzzle(Puzzle("empty", (3,), (), 10)), ("tiny", "easy"))
        self.assertEqual(classify_puzzle(Puzzle("small", (3,), (1, 2, 5, 8), 9)), ("tiny", "easy"))
        indep = Puzzle("indep", (100,), (10**12 + 1, 10**12 + 2, 10**12 + 3, 10**12 + 4, 10**12 + 5), 10**18)
        self.assertEqual(classify_puzzle(indep), ("independent", "medium"))
        self.assertEqual(partition_mixed_targets(Puzzle("empty", (3,), (), 10)), ([], []))
        large_18 = Puzzle(
            "large_18",
            (2, 3),
            tuple(10**17 + i * 10**15 for i in range(1, 25)),
            10**18,
        )
        structured, random_t = partition_mixed_targets(large_18)
        self.assertEqual(sorted(structured + random_t), sorted(large_18.targets))


if __name__ == "__main__":
    unittest.main()
