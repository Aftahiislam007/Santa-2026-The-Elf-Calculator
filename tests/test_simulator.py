import unittest

from elfcalc import InvalidProgram, Puzzle, simulate

EXAMPLE = Puzzle("example_0001", (3,), (1, 2, 5, 8), 9)
EXAMPLE_PROGRAM = [["/", 0, 0, 1], ["-", 0, 1, 2], ["+", 0, 2, 3], ["+", 0, 3, 4]]


class SimulatorTest(unittest.TestCase):
    def assertInvalid(self, puzzle, program, fragment):
        with self.assertRaises(InvalidProgram) as ctx:
            simulate(puzzle, program)
        self.assertIn(fragment, str(ctx.exception))

    def test_worked_example(self):
        self.assertEqual(simulate(EXAMPLE, EXAMPLE_PROGRAM), 4)

    def test_missing_target(self):
        self.assertInvalid(EXAMPLE, EXAMPLE_PROGRAM[:3], "targets not produced")

    def test_target_stays_covered_after_overwrite(self):
        program = [["/", 0, 0, 1], ["-", 0, 1, 1], ["+", 0, 1, 1], ["+", 0, 1, 1]]
        self.assertEqual(simulate(EXAMPLE, program), 4)

    def test_starting_value_covers_target(self):
        self.assertEqual(simulate(Puzzle("p", (4, 7), (4, 7), 10), []), 0)

    def test_value_limit_applies_to_intermediates(self):
        puzzle = Puzzle("p", (3,), (6,), 8)
        self.assertInvalid(puzzle, [["*", 0, 0, 1], ["-", 1, 0, 2]], "exceeds value_limit")
        self.assertEqual(simulate(puzzle, [["+", 0, 0, 1]]), 1)

    def test_subtraction_must_stay_positive(self):
        puzzle = Puzzle("p", (3, 5), (2,), 10)
        self.assertInvalid(puzzle, [["-", 0, 1, 2]], "not positive")
        self.assertInvalid(puzzle, [["-", 0, 0, 2]], "not positive")
        self.assertEqual(simulate(puzzle, [["-", 1, 0, 2]]), 1)

    def test_division_must_be_exact(self):
        puzzle = Puzzle("p", (6, 4), (3,), 10)
        self.assertInvalid(puzzle, [["/", 0, 1, 2]], "not exact")
        self.assertInvalid(puzzle, [["/", 1, 0, 2]], "not exact")

    def test_undefined_source(self):
        self.assertInvalid(EXAMPLE, [["+", 0, 1, 2]], "undefined source")

    def test_sources_read_before_write(self):
        puzzle = Puzzle("p", (3,), (6, 12), 20)
        self.assertEqual(simulate(puzzle, [["+", 0, 0, 0], ["+", 0, 0, 0]]), 2)

    def test_malformed_instructions(self):
        self.assertInvalid(EXAMPLE, [["%", 0, 0, 1]], "bad operator")
        self.assertInvalid(EXAMPLE, [["+", 0, 0]], "four-element")
        self.assertInvalid(EXAMPLE, [["+", 0, 0, 48]], "out of range")
        self.assertInvalid(EXAMPLE, [["+", -1, 0, 1]], "out of range")
        self.assertInvalid(EXAMPLE, [["+", 0.0, 0, 1]], "out of range")
        self.assertInvalid(EXAMPLE, [["+", True, 0, 1]], "out of range")
        self.assertInvalid(EXAMPLE, {"+": 1}, "not a JSON array")

    def test_instruction_count_limit(self):
        puzzle = Puzzle("p", (1,), (1,), 10)
        self.assertEqual(simulate(puzzle, [["*", 0, 0, 0]] * 4096), 4096)
        self.assertInvalid(puzzle, [["*", 0, 0, 0]] * 4097, "exceed 4096")

    def test_useless_instructions_still_cost(self):
        self.assertEqual(simulate(EXAMPLE, EXAMPLE_PROGRAM + [["+", 0, 1, 5]]), 5)


if __name__ == "__main__":
    unittest.main()
