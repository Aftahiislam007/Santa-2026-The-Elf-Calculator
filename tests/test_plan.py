import unittest

from elfcalc import (InvalidProgram, PlanError, Puzzle, compile_plan, execute_plan, load_practice,
                     prune_plan, simulate)
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

    def test_recomputation_is_honoured_when_registers_run_out(self):
        puzzle = Puzzle("p", (1,), (1275,), 10000)
        values = list(range(1, 51))
        plan = [("+", v, 1) for v in values[:-1]]
        acc = 50
        for v in reversed(values[1:-1]):
            plan.append(("+", acc, v))
            acc += v
            if v > 2:
                plan.append(("-", v, 1))
        plan.append(("+", acc, 1))
        with self.assertRaises(PlanError):
            compile_plan(puzzle, [s for s in plan if s[0] == "+"])
        self.assertEqual(simulate(puzzle, compile_plan(puzzle, plan)), len(plan))

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

    def test_no_instruction_after_the_last_target(self):
        for puzzle in load_practice():
            program = compile_plan(puzzle, baseline(puzzle))
            if program:
                with self.assertRaises(InvalidProgram, msg=puzzle.id):
                    simulate(puzzle, program[:-1])  # the last instruction is needed

    def test_targets_already_in_starting_values_need_no_instructions(self):
        puzzle = Puzzle("p", (4, 7), (4, 7), 20)
        self.assertEqual(compile_plan(puzzle, []), [])
        self.assertEqual(compile_plan(puzzle, [("+", 4, 4), ("+", 4, 7)]), [])  # dead steps dropped
        puzzle = Puzzle("p", (4, 7), (4, 8), 10)
        self.assertEqual(simulate(puzzle, compile_plan(puzzle, [("+", 4, 4)])), 1)

    def test_steps_after_all_targets_are_covered_are_dropped(self):
        plan = EXAMPLE_PLAN + [("+", 8, 1), ("-", 8, 5)]  # legal but pointless (9, 3)
        self.assertEqual(execute_plan(EXAMPLE, plan), [1, 2, 5, 8, 9, 3])
        program = compile_plan(EXAMPLE, plan)
        self.assertEqual(len(program), 4)
        self.assertEqual(simulate(EXAMPLE, program), 4)

    def test_plan_that_misses_a_target_is_an_error(self):
        with self.assertRaises(PlanError):
            compile_plan(EXAMPLE, EXAMPLE_PLAN[:3])
        with self.assertRaises(PlanError):
            compile_plan(EXAMPLE, [])

    def test_execute_plan_respects_limit_inclusively(self):
        puzzle = Puzzle("p", (3,), (9,), 9)
        self.assertEqual(execute_plan(puzzle, [("*", 3, 3)]), [9])
        with self.assertRaises(PlanError):
            execute_plan(Puzzle("p", (3,), (9,), 8), [("*", 3, 3)])

    def test_execute_plan_rejects_bad_operators_and_missing_operands(self):
        for plan in ([("%", 3, 3)], [("+", 3, 7)], [("+", 5, 3)], [("-", 3, 1)]):
            with self.assertRaises(PlanError):
                execute_plan(EXAMPLE, plan)

    def test_limit_1e18_does_not_break_planning(self):
        limit = 10 ** 18
        puzzle = Puzzle("p", (10 ** 9, 10 ** 9 + 7), (10 ** 18, 10 ** 9 + 8), limit)
        plan = [("*", 10 ** 9, 10 ** 9), ("/", 10 ** 9 + 7, 10 ** 9 + 7), ("+", 10 ** 9 + 7, 1)]
        program = compile_plan(puzzle, plan)
        self.assertEqual(simulate(puzzle, program), 3)
        with self.assertRaises(PlanError):
            execute_plan(puzzle, [("*", 10 ** 9, 10 ** 9 + 7)])

    def test_duplicate_starting_values_are_harmless(self):
        puzzle = Puzzle("p", (3, 3), (6, 12), 20)
        program = compile_plan(puzzle, [("+", 3, 3), ("+", 6, 6)])
        self.assertEqual(simulate(puzzle, program), 2)

    def test_starting_registers_are_overwritten_when_all_48_are_full(self):
        starts = tuple(range(1, 49))
        puzzle = Puzzle("p", starts, (49, 98), 100)
        program = compile_plan(puzzle, [("+", 48, 1), ("+", 49, 49)])
        self.assertEqual(simulate(puzzle, program), 2)
        self.assertTrue(all(0 <= x <= 47 for ins in program for x in ins[1:]))

    def test_exactly_48_live_values_fit(self):
        # values 1..48 are all alive at once, then folded into a single sum
        plan = [("+", 1, 1)] + [("+", v, 1) for v in range(2, 48)]
        acc = 48
        for v in range(47, 0, -1):
            plan.append(("+", acc, v))
            acc += v
        puzzle = Puzzle("p", (1,), (acc,), 10 ** 6)
        program = compile_plan(puzzle, plan)
        self.assertEqual(simulate(puzzle, program), len(plan))
        self.assertEqual(max(ins[3] for ins in program), 47)

        # adding one more simultaneously-live value (49 total) must fail
        plan49 = [("+", 1, 1)] + [("+", v, 1) for v in range(2, 49)]
        acc49 = 49
        for v in range(48, 0, -1):
            plan49.append(("+", acc49, v))
            acc49 += v
        with self.assertRaises(PlanError):
            compile_plan(Puzzle("p", (1,), (acc49,), 10 ** 6), plan49)

    def test_compiled_programs_use_only_legal_registers_and_ops(self):
        for puzzle in load_practice()[:60]:
            for ins in compile_plan(puzzle, baseline(puzzle)):
                self.assertIn(ins[0], "+-*/")
                self.assertTrue(all(type(x) is int and 0 <= x < 48 for x in ins[1:]), puzzle.id)

    def test_compile_never_exceeds_plan_length(self):
        for puzzle in load_practice()[:60]:
            plan = baseline(puzzle)
            self.assertLessEqual(len(compile_plan(puzzle, plan)), len(plan), puzzle.id)


if __name__ == "__main__":
    unittest.main()
