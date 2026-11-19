import unittest
from unittest import mock

from elfcalc import InvalidProgram, Puzzle, check, simulate
from elfcalc.simulator import MAX_CELL_BYTES, MAX_INSTRUCTIONS, OPS, encode_program

EXAMPLE = Puzzle("example_0001", (3,), (1, 2, 5, 8), 9)
EXAMPLE_PROGRAM = [["/", 0, 0, 1], ["-", 0, 1, 2], ["+", 0, 2, 3], ["+", 0, 3, 4]]


class SimulatorTest(unittest.TestCase):
    def assertInvalid(self, puzzle, program, fragment):
        with self.assertRaises(InvalidProgram) as ctx:
            simulate(puzzle, program)
        self.assertIn(fragment, str(ctx.exception))

    # --- official worked example -------------------------------------------------

    def test_worked_example(self):
        self.assertEqual(EXAMPLE.registers, 48)
        self.assertEqual(simulate(EXAMPLE, EXAMPLE_PROGRAM), 4)
        self.assertEqual(check(EXAMPLE, EXAMPLE_PROGRAM), (True, ""))

    def test_worked_example_encoding(self):
        self.assertEqual(encode_program(EXAMPLE_PROGRAM),
                         '[["/",0,0,1],["-",0,1,2],["+",0,2,3],["+",0,3,4]]')
        self.assertEqual(encode_program([]), "[]")

    def test_missing_target(self):
        self.assertInvalid(EXAMPLE, EXAMPLE_PROGRAM[:3], "targets not produced")
        self.assertInvalid(EXAMPLE, [], "targets not produced")
        ok, message = check(EXAMPLE, EXAMPLE_PROGRAM[:2])
        self.assertFalse(ok)
        self.assertIn("targets not produced", message)

    # --- targets -----------------------------------------------------------------

    def test_target_stays_covered_after_overwrite(self):
        program = [["/", 0, 0, 1], ["-", 0, 1, 1], ["+", 0, 1, 1], ["+", 0, 1, 1]]
        self.assertEqual(simulate(EXAMPLE, program), 4)

    def test_starting_value_covers_target(self):
        self.assertEqual(simulate(Puzzle("p", (4, 7), (4, 7), 10), []), 0)
        self.assertEqual(simulate(Puzzle("p", (1, 2), (1, 2), 5), []), 0)

    def test_starting_value_target_stays_covered_after_overwrite(self):
        puzzle = Puzzle("p", (4, 6), (4, 6, 10), 10)
        self.assertEqual(simulate(puzzle, [["+", 0, 1, 0]]), 1)  # 4 is gone, 6 stays
        self.assertEqual(simulate(puzzle, [["+", 0, 1, 1], ["/", 1, 1, 0]]), 2)

    # --- value rules -------------------------------------------------------------

    def test_value_limit_applies_to_intermediates(self):
        puzzle = Puzzle("p", (3,), (6,), 8)
        self.assertInvalid(puzzle, [["*", 0, 0, 1], ["-", 1, 0, 2]], "exceeds value_limit")
        self.assertEqual(simulate(puzzle, [["+", 0, 0, 1]]), 1)

    def test_value_limit_is_inclusive(self):
        self.assertEqual(simulate(Puzzle("p", (3,), (9,), 9), [["*", 0, 0, 1]]), 1)
        self.assertInvalid(Puzzle("p", (3,), (9,), 8), [["*", 0, 0, 1]], "exceeds value_limit")

    def test_limit_equal_to_max_target_and_to_a_start(self):
        puzzle = Puzzle("p", (5, 6, 7), (5, 8), 8)  # scored_0931-like
        self.assertEqual(simulate(puzzle, [["-", 2, 0, 3], ["+", 1, 3, 4]]), 2)
        self.assertInvalid(puzzle, [["+", 1, 2, 3]], "exceeds value_limit")

    def test_huge_limits_do_not_overflow_64_bit(self):
        limit = 10 ** 18
        puzzle = Puzzle("p", (10 ** 9, 10 ** 9 + 1), (10 ** 18,), limit)
        self.assertEqual(simulate(puzzle, [["*", 0, 0, 2]]), 1)
        self.assertInvalid(puzzle, [["*", 0, 1, 2]], "exceeds value_limit")
        big = Puzzle("p", (10 ** 10,), (10 ** 10,), limit)
        self.assertInvalid(big, [["*", 0, 0, 1]], "exceeds value_limit")
        self.assertInvalid(big, [["*", 0, 0, 0], ["*", 0, 0, 0]], "exceeds value_limit")
        # 2**33 squared is 2**66, which would wrap to 0 in 64-bit arithmetic
        wrap = Puzzle("p", (2 ** 33,), (2 ** 33,), limit)
        self.assertInvalid(wrap, [["*", 0, 0, 1]], "exceeds value_limit")
        self.assertEqual(simulate(Puzzle("p", (limit - 1, 1), (limit,), limit),
                                  [["+", 0, 1, 2]]), 1)
        self.assertInvalid(Puzzle("p", (limit, 1), (2,), limit), [["+", 0, 1, 2]],
                           "exceeds value_limit")

    def test_subtraction_must_stay_positive(self):
        puzzle = Puzzle("p", (3, 5), (2,), 10)
        self.assertInvalid(puzzle, [["-", 0, 1, 2]], "not positive")
        self.assertInvalid(puzzle, [["-", 0, 0, 2]], "not positive")
        self.assertEqual(simulate(puzzle, [["-", 1, 0, 2]]), 1)

    def test_division_must_be_exact(self):
        puzzle = Puzzle("p", (6, 4), (3,), 10)
        self.assertInvalid(puzzle, [["/", 0, 1, 2]], "not exact")
        self.assertInvalid(puzzle, [["/", 1, 0, 2]], "not exact")
        self.assertEqual(simulate(puzzle, [["/", 0, 0, 2], ["+", 2, 2, 3], ["/", 0, 3, 4],
                                           ["+", 4, 4, 5], ["+", 5, 2, 6]]), 5)
        self.assertInvalid(Puzzle("p", (1, 3), (2,), 10), [["/", 0, 1, 2]], "not exact")

    # --- registers ---------------------------------------------------------------

    def test_undefined_source(self):
        self.assertInvalid(EXAMPLE, [["+", 0, 1, 2]], "undefined source")

    def test_registers_beyond_starting_values_are_undefined(self):
        puzzle = Puzzle("p", (2, 3), (5,), 10)
        for src in (2, 47):
            self.assertInvalid(puzzle, [["+", 0, src, 2]], "undefined source")
            self.assertInvalid(puzzle, [["+", src, 0, 2]], "undefined source")
        # a destination becomes a legal source once written
        self.assertEqual(simulate(puzzle, [["+", 0, 1, 47], ["+", 47, 47, 2]]), 2)

    def test_starting_values_sit_in_listed_order(self):
        puzzle = Puzzle("p", (7, 2, 5), (3,), 10)
        self.assertEqual(simulate(puzzle, [["-", 2, 1, 3]]), 1)
        self.assertInvalid(puzzle, [["-", 1, 2, 3]], "not positive")
        self.assertEqual(simulate(puzzle, [["/", 0, 0, 4], ["+", 4, 1, 3]]), 2)

    def test_all_48_registers_are_addressable(self):
        puzzle = Puzzle("p", (1,), (2,), 10)
        self.assertEqual(simulate(puzzle, [["+", 0, 0, 47]]), 1)
        self.assertInvalid(puzzle, [["+", 0, 0, 48]], "out of range")

    def test_sources_read_before_write(self):
        puzzle = Puzzle("p", (3,), (6, 12), 20)
        self.assertEqual(simulate(puzzle, [["+", 0, 0, 0], ["+", 0, 0, 0]]), 2)

    def test_destination_may_equal_either_or_both_sources(self):
        puzzle = Puzzle("p", (3, 5), (8,), 100)
        self.assertEqual(simulate(puzzle, [["+", 0, 1, 0]]), 1)
        self.assertEqual(simulate(puzzle, [["+", 0, 1, 1]]), 1)
        puzzle = Puzzle("p", (3, 5), (6, 12, 24), 100)
        self.assertEqual(simulate(puzzle, [["+", 0, 0, 0]] * 3), 3)

    def test_same_register_as_both_sources(self):
        puzzle = Puzzle("p", (5,), (1, 10, 25), 100)
        self.assertEqual(simulate(puzzle, [["/", 0, 0, 1], ["+", 0, 0, 2], ["*", 0, 0, 3]]), 3)
        self.assertInvalid(Puzzle("p", (5,), (1,), 100), [["-", 0, 0, 1]], "not positive")

    # --- malformed input ---------------------------------------------------------

    def test_malformed_instructions(self):
        self.assertInvalid(EXAMPLE, [["%", 0, 0, 1]], "bad operator")
        self.assertInvalid(EXAMPLE, [["+", 0, 0]], "four-element")
        self.assertInvalid(EXAMPLE, [["+", 0, 0, 48]], "out of range")
        self.assertInvalid(EXAMPLE, [["+", -1, 0, 1]], "out of range")
        self.assertInvalid(EXAMPLE, [["+", 0.0, 0, 1]], "out of range")
        self.assertInvalid(EXAMPLE, [["+", True, 0, 1]], "out of range")
        self.assertInvalid(EXAMPLE, {"+": 1}, "not a JSON array")

    def test_indices_must_be_strict_integers_in_range(self):
        for ins in (["+", 0, 0, True], ["+", False, 0, 1], ["+", 0, True, 1],
                    ["+", 0, 0, 1.0], ["+", "0", 0, 1], ["+", None, 0, 1],
                    ["+", 0, 0, 48], ["+", 0, 0, -1], ["+", 0, 48, 1], ["+", 0, -1, 1],
                    ["+", [0], 0, 1], ["+", 2 ** 70, 0, 1]):
            self.assertInvalid(EXAMPLE, [ins], "out of range")

    def test_operator_must_be_one_of_four_strings(self):
        self.assertEqual(set(OPS), {"+", "-", "*", "/"})
        for op in ("%", "**", "x", "", "++", " +", 1, None, ["+"], True):
            self.assertInvalid(EXAMPLE, [[op, 0, 0, 1]], "bad operator")

    def test_instruction_shape(self):
        for ins in ([], ["+"], ["+", 0, 0], ["+", 0, 0, 1, 2], "+001", 5, None, {"op": "+"}):
            self.assertInvalid(EXAMPLE, [ins], "four-element")

    def test_program_must_be_a_list(self):
        for program in (None, "[]", {}, 4, (1, 2)):
            self.assertInvalid(EXAMPLE, program, "not a JSON array")

    def test_unserializable_program_is_invalid_not_a_crash(self):
        self.assertInvalid(EXAMPLE, [["+", 0, 0, object()]], "not JSON serializable")

    # --- limits and scoring ------------------------------------------------------

    def test_instruction_count_limit(self):
        puzzle = Puzzle("p", (1,), (1,), 10)
        self.assertEqual(MAX_INSTRUCTIONS, 4096)
        self.assertEqual(simulate(puzzle, [["*", 0, 0, 0]] * 4096), 4096)
        self.assertInvalid(puzzle, [["*", 0, 0, 0]] * 4097, "exceed 4096")

    def test_cell_byte_limit(self):
        self.assertEqual(MAX_CELL_BYTES, 131072)
        puzzle = Puzzle("p", (1,), (1,), 10)
        # the longest legal program (4096 instructions, widest indices) fits in 128 KiB
        widest = [["*", 47, 47, 47]] * 4096
        self.assertLessEqual(len(encode_program(widest).encode("utf-8")), MAX_CELL_BYTES)
        # and the byte check itself is enforced inclusively on the encoded cell
        program = [["*", 0, 0, 0]] * 5
        size = len(encode_program(program).encode("utf-8"))
        with mock.patch("elfcalc.simulator.MAX_CELL_BYTES", size):
            self.assertEqual(simulate(puzzle, program), 5)
        with mock.patch("elfcalc.simulator.MAX_CELL_BYTES", size - 1):
            self.assertInvalid(puzzle, program, "exceeds")

    def test_useless_instructions_still_cost(self):
        self.assertEqual(simulate(EXAMPLE, EXAMPLE_PROGRAM + [["+", 0, 1, 5]]), 5)
        self.assertEqual(simulate(EXAMPLE, EXAMPLE_PROGRAM + [["+", 0, 0, 5]] * 3), 7)
        self.assertEqual(simulate(EXAMPLE, EXAMPLE_PROGRAM + [["/", 0, 0, 0], ["/", 0, 0, 47]]), 6)

    def test_instructions_after_last_target_must_still_be_legal(self):
        for bad, fragment in ((["-", 0, 0, 5], "not positive"),
                              (["*", 4, 4, 5], "exceeds value_limit"),
                              (["/", 0, 4, 5], "not exact"),
                              (["+", 0, 30, 5], "undefined source"),
                              (["+", 0, 0, 48], "out of range")):
            self.assertInvalid(EXAMPLE, EXAMPLE_PROGRAM + [bad], fragment)


if __name__ == "__main__":
    unittest.main()
