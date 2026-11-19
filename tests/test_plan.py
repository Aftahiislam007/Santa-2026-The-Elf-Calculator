import unittest

from elfcalc import PlanError, Puzzle, compile_plan, execute_plan, load_practice, prune_plan, simulate
from elfcalc.solvers.baseline import solve as baseline

EXAMPLE = Puzzle("example_0001", (3,), (1, 2, 5, 8), 9)
EXAMPLE_PLAN = [("/", 3, 3), ("-", 3, 1), ("+", 3, 2), ("+", 3, 5)]


class PlanTest(unittest.TestCase):
    def test_worked_example(self):
        self.assertEqual(execute_plan(EXAMPLE, EXAMPLE_PLAN), [1, 2, 5, 8])
        program = compile_plan(EXAMPLE, EXAMPLE_PLAN)
        self.assertEqual(simulate(EXAMPLE, program), 4)

    def test_illegal_steps(self):
        for plan in ([("+", 3, 4)], [("-", 3, 3)], [("*", 3, 3), ("*", 9, 3)], [("/", 3, 2)]):
            with self.assertRaises(PlanError):
                execute_plan(EXAMPLE, plan)

    def test_prune_removes_dead_and_duplicate_steps(self):
        plan = [("/", 3, 3), ("+", 3, 3), ("/", 3, 3), ("-", 3, 1), ("+", 1, 1),
                ("+", 3, 2), ("+", 3, 5)]
        self.assertEqual(prune_plan(EXAMPLE, plan), EXAMPLE_PLAN)

    def test_registers_are_reused(self):
        puzzle = Puzzle("p", (1,), (200,), 1000)
        plan = [("+", 1, 1)] + [("+", v, 1) for v in range(2, 200)]
        program = compile_plan(puzzle, plan)
        self.assertEqual(simulate(puzzle, program), 199)
        self.assertLessEqual(max(max(ins[1:]) for ins in program), 47)

    def test_too_many_live_values(self):
        puzzle = Puzzle("p", (1,), (1275,), 10000)
        values = list(range(1, 51))
        plan = [("+", v, 1) for v in values[:-1]]
        acc = 50
        for v in reversed(values[:-1]):
            plan.append(("+", acc, v))
            acc += v
        with self.assertRaises(PlanError):
            compile_plan(puzzle, plan)

    def test_reorder_rescues_wide_plans(self):
        targets = tuple(range(1000, 1060))
        puzzle = Puzzle("p", (1, 1000), targets, 2000)
        plan = [("+", v, 1) for v in range(1, 59)] + [("+", 1000, v) for v in range(1, 60)]
        program = compile_plan(puzzle, plan)
        self.assertEqual(simulate(puzzle, program), 117)

    def test_baseline_solves_every_practice_puzzle(self):
        for puzzle in load_practice():
            program = compile_plan(puzzle, baseline(puzzle))
            self.assertGreaterEqual(simulate(puzzle, program), puzzle.lower_bound, puzzle.id)


if __name__ == "__main__":
    unittest.main()
