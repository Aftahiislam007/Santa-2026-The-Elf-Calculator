import time
import unittest

from elfcalc import Puzzle, compile_plan, load_practice, simulate
from elfcalc.solvers import SOLVERS, exact, get_solver
from elfcalc.solvers.baseline import solve as baseline
from elfcalc.submission import merge_programs, validate_programs

SPECIALIZED_CASES = [
    ("mitm_basis", ("practice_0038", "practice_0005")),
    ("shared_dag", ("practice_0006", "practice_0004")),
    ("mult_factor", ("practice_0170", "practice_0007")),
    ("mixed_hybrid", ("practice_0114", "practice_0002")),
    ("greedy_v2", ("practice_0170", "practice_0006")),
]


class SolversTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.practice = load_practice()
        cls.by_id = {p.id: p for p in cls.practice}

    def test_exact_worked_example(self):
        puzzle = Puzzle("example_0001", (3,), (1, 2, 5, 8), 9)
        plan = exact.solve(puzzle)
        self.assertIsNotNone(plan)
        program = compile_plan(puzzle, plan)
        self.assertEqual(simulate(puzzle, program), 4)

    def test_exact_practice_tiny_optimal(self):
        tiny_puzzles = [p for p in self.practice if p.family == "tiny"]
        self.assertEqual(len(tiny_puzzles), 16)
        total_cost = 0
        for puzzle in tiny_puzzles:
            with self.subTest(puzzle_id=puzzle.id):
                plan = exact.solve(puzzle)
                self.assertIsNotNone(plan)
                program = compile_plan(puzzle, plan)
                cost = simulate(puzzle, program)
                self.assertGreaterEqual(cost, puzzle.lower_bound)
                total_cost += cost
        self.assertEqual(total_cost, 50)

    def test_exact_out_of_envelope_returns_none_immediately(self):
        puzzle = Puzzle("out_of_envelope", (1,), (2, 3), 1025)
        start = time.monotonic()
        result = exact.solve(puzzle)
        elapsed = time.monotonic() - start
        self.assertIsNone(result)
        self.assertLess(elapsed, 0.2)

    def test_specialized_solvers_compile_simulate_and_beat_baseline(self):
        for solver_name, puzzle_ids in SPECIALIZED_CASES:
            solver = get_solver(solver_name)
            for pid in puzzle_ids:
                puzzle = self.by_id[pid]
                with self.subTest(solver=solver_name, puzzle_id=pid, tier=puzzle.design_tier):
                    plan = solver(puzzle, time=1.0)
                    self.assertIsNotNone(plan)
                    program = compile_plan(puzzle, plan)
                    cost = simulate(puzzle, program)

                    base_program = compile_plan(puzzle, baseline(puzzle))
                    base_cost = simulate(puzzle, base_program)
                    self.assertLess(cost, base_cost)
                    self.assertLessEqual(cost * 2, base_cost)

    def test_portfolio_merge_selects_shortest_valid(self):
        puzzles = [self.by_id[pid] for pid in ("practice_0038", "practice_0006", "practice_0170")]
        baseline_progs = {p.id: compile_plan(p, baseline(p)) for p in puzzles}
        greedy_v2 = get_solver("greedy_v2")
        greedy_progs = {p.id: compile_plan(p, greedy_v2(p, time=1.0)) for p in puzzles}
        specialized_progs = {
            "practice_0038": compile_plan(
                self.by_id["practice_0038"],
                get_solver("mitm_basis")(self.by_id["practice_0038"], time=1.0),
            ),
            "practice_0006": compile_plan(
                self.by_id["practice_0006"],
                get_solver("shared_dag")(self.by_id["practice_0006"], time=1.0),
            ),
            "practice_0170": compile_plan(
                self.by_id["practice_0170"],
                get_solver("mult_factor")(self.by_id["practice_0170"], time=1.0),
            ),
        }
        invalid_progs = {p.id: [] for p in puzzles}

        sources = [invalid_progs, baseline_progs, greedy_progs, specialized_progs]
        merged = merge_programs(puzzles, sources)

        costs, errors = validate_programs(puzzles, merged, require_all=True)
        self.assertEqual(errors, {})
        for p in puzzles:
            expected_len = min(
                len(src[p.id]) for src in (baseline_progs, greedy_progs, specialized_progs)
            )
            self.assertEqual(costs[p.id], expected_len)
            self.assertEqual(len(merged[p.id]), expected_len)
            self.assertLess(costs[p.id], len(baseline_progs[p.id]))

    def _run_solver(self, name: str, puzzle: Puzzle, time_s: float = 1.0):
        solver = get_solver(name)
        if name in ("baseline", "exact", "greedy"):
            return solver(puzzle)
        return solver(puzzle, time=time_s)

    def test_all_solvers_worked_example(self):
        puzzle = Puzzle("example_0001", (3,), (1, 2, 5, 8), 9)
        for name in sorted(SOLVERS):
            with self.subTest(solver=name):
                plan = self._run_solver(name, puzzle)
                self.assertIsNotNone(plan)
                program = compile_plan(puzzle, plan)
                cost = simulate(puzzle, program)
                self.assertGreaterEqual(cost, 4)
                if name != "baseline":
                    self.assertEqual(len(plan), cost)

    def test_all_solvers_targets_already_in_starts(self):
        puzzle = Puzzle("already_covered", (3, 7, 15), (3, 15), 20)
        for name in sorted(SOLVERS):
            with self.subTest(solver=name):
                plan = self._run_solver(name, puzzle)
                self.assertEqual(plan, [])
                program = compile_plan(puzzle, plan)
                self.assertEqual(simulate(puzzle, program), 0)

    def test_all_solvers_large_value_limit_overflow_safe(self):
        puzzle = Puzzle("large_10e18", (3,), (10**18 - 7, 10**18), 10**18)
        self.assertIsNone(exact.solve(puzzle))
        for name in sorted(SOLVERS):
            if name == "exact":
                continue
            with self.subTest(solver=name):
                plan = self._run_solver(name, puzzle, time_s=1.0)
                self.assertIsNotNone(plan)
                program = compile_plan(puzzle, plan)
                cost = simulate(puzzle, program)
                self.assertGreater(cost, 0)
                if name != "baseline":
                    self.assertEqual(len(plan), cost)

    def test_all_solvers_tight_value_limit_and_starts_at_limit(self):
        cases = [
            Puzzle("tight_limit_13", (10,), (6, 9, 11, 13), 13),
            Puzzle("tight_limit_8", (5, 6, 7), (5, 8), 8),
            Puzzle("starts_at_limit", (3, 6, 12), (1, 5, 11), 12),
        ]
        for puzzle in cases:
            for name in sorted(SOLVERS):
                with self.subTest(puzzle_id=puzzle.id, solver=name):
                    plan = self._run_solver(name, puzzle, time_s=1.0)
                    self.assertIsNotNone(plan)
                    program = compile_plan(puzzle, plan)
                    cost = simulate(puzzle, program)
                    self.assertGreaterEqual(cost, puzzle.lower_bound)
                    if name != "baseline":
                        self.assertEqual(len(plan), cost)

    def test_shared_dag_previously_missed_practice_puzzles(self):
        shared_dag = get_solver("shared_dag")
        for pid in ("practice_0026", "practice_0034", "practice_0171", "practice_0184"):
            puzzle = self.by_id[pid]
            with self.subTest(puzzle_id=pid):
                plan = shared_dag(puzzle, time=1.0)
                self.assertIsNotNone(plan)
                program = compile_plan(puzzle, plan)
                cost = simulate(puzzle, program)
                self.assertEqual(len(plan), cost)
                base_cost = simulate(puzzle, compile_plan(puzzle, baseline(puzzle)))
                self.assertLess(cost, base_cost)


if __name__ == "__main__":
    unittest.main()
