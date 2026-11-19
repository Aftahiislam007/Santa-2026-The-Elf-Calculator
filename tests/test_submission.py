import csv
import tempfile
import unittest
from pathlib import Path

from elfcalc import Puzzle, load_practice, load_scored
from elfcalc.puzzle import SAMPLE_SUBMISSION_CSV
from elfcalc.simulator import MAX_CELL_BYTES, MAX_INSTRUCTIONS
from elfcalc.submission import merge_programs, read_programs, validate_programs, write_programs

EXAMPLE = Puzzle("example_0001", (3,), (1, 2, 5, 8), 9)
SHORT = [["/", 0, 0, 1], ["-", 0, 1, 2], ["+", 0, 2, 3], ["+", 0, 3, 4]]
LONG = SHORT + [["+", 0, 1, 5]]
BROKEN = SHORT[:2]
EXAMPLE_ROW = 'example_0001,"[[""/"",0,0,1],[""-"",0,1,2],[""+"",0,2,3],[""+"",0,3,4]]"'


def write_raw(path, text):
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(text)


class SubmissionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    # --- writing -----------------------------------------------------------------

    def test_round_trip_matches_competition_format(self):
        path = self.tmp / "out.csv"
        write_programs(path, {EXAMPLE.id: SHORT})
        self.assertEqual(path.read_text(encoding="utf-8").splitlines(),
                         ["id,program", EXAMPLE_ROW])
        self.assertEqual(read_programs(path), {EXAMPLE.id: SHORT})

    def test_official_row_is_read_back_exactly(self):
        path = self.tmp / "official.csv"
        write_raw(path, "id,program\n" + EXAMPLE_ROW + "\n")
        self.assertEqual(read_programs(path), {"example_0001": SHORT})

    def test_file_is_utf8_with_exact_header_and_csv_quoting(self):
        path = self.tmp / "out.csv"
        write_programs(path, {"b": SHORT, "a": []})
        raw = path.read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))  # no BOM
        raw.decode("utf-8")
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], ["id", "program"])
        self.assertEqual([r[0] for r in rows[1:]], ["a", "b"])
        self.assertEqual(rows[1][1], "[]")
        self.assertEqual(rows[2][1], '[["/",0,0,1],["-",0,1,2],["+",0,2,3],["+",0,3,4]]')
        self.assertTrue(all(len(r) == 2 for r in rows))

    def test_write_leaves_no_temp_file(self):
        write_programs(self.tmp / "sub" / "out.csv", {EXAMPLE.id: SHORT})
        self.assertEqual([p.name for p in (self.tmp / "sub").iterdir()], ["out.csv"])

    # --- reading -----------------------------------------------------------------

    def test_read_rejects_bad_headers(self):
        for header in ("program,id", "id", "id,program,extra", "ID,program", "id;program", ""):
            path = self.tmp / "bad.csv"
            write_raw(path, header + "\n")
            with self.assertRaises(ValueError, msg=header):
                read_programs(path)

    def test_read_rejects_duplicate_ids(self):
        path = self.tmp / "dup.csv"
        write_raw(path, "id,program\nx,[]\nx,[]\n")
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            read_programs(path)

    def test_read_rejects_malformed_rows(self):
        for body in ("x\n", "x,[],[]\n"):
            path = self.tmp / "bad.csv"
            write_raw(path, "id,program\n" + body)
            with self.assertRaises(ValueError, msg=body):
                read_programs(path)
        write_raw(self.tmp / "bad.csv", "id,program\nx,not json\n")
        with self.assertRaises(ValueError):
            read_programs(self.tmp / "bad.csv")

    def test_read_enforces_cell_byte_limit_on_raw_text(self):
        path = self.tmp / "fat.csv"
        # valid JSON, but the submitted cell is larger than 128 KiB because of whitespace
        write_raw(path, 'id,program\nx,"[' + " " * MAX_CELL_BYTES + ']"\n')
        with self.assertRaisesRegex(ValueError, "exceeds"):
            read_programs(path)
        ok = self.tmp / "ok.csv"
        write_raw(ok, 'id,program\nx,"[' + " " * (MAX_CELL_BYTES - 2) + ']"\n')
        self.assertEqual(read_programs(ok), {"x": []})

    # --- validation --------------------------------------------------------------

    def test_validate_reports_missing_invalid_and_unknown(self):
        costs, errors = validate_programs([EXAMPLE], {EXAMPLE.id: SHORT})
        self.assertEqual((costs, errors), ({EXAMPLE.id: 4}, {}))
        _, errors = validate_programs([EXAMPLE], {})
        self.assertEqual(errors, {EXAMPLE.id: "missing program"})
        _, errors = validate_programs([EXAMPLE], {EXAMPLE.id: BROKEN, "nope": []})
        self.assertEqual(set(errors), {EXAMPLE.id, "nope"})

    def test_validate_partial_allows_missing_ids(self):
        costs, errors = validate_programs([EXAMPLE, Puzzle("q", (1,), (2,), 5)],
                                          {EXAMPLE.id: SHORT}, require_all=False)
        self.assertEqual((costs, errors), ({EXAMPLE.id: 4}, {}))

    def test_validate_flags_example_id_in_a_scored_submission(self):
        scored = load_scored()
        programs = read_programs(SAMPLE_SUBMISSION_CSV)
        programs["example_0001"] = SHORT
        _, errors = validate_programs(scored, programs)
        self.assertEqual(errors, {"example_0001": "unknown puzzle id"})

    def test_validate_flags_each_missing_scored_id(self):
        scored = load_scored()
        programs = read_programs(SAMPLE_SUBMISSION_CSV)
        del programs["scored_0001"]
        del programs["scored_1000"]
        costs, errors = validate_programs(scored, programs)
        self.assertEqual(errors, {"scored_0001": "missing program", "scored_1000": "missing program"})
        self.assertEqual(len(costs), 998)

    def test_validate_rejects_non_list_and_oversized_programs(self):
        for program in (5, "[]", None, {"a": 1}):
            _, errors = validate_programs([EXAMPLE], {EXAMPLE.id: program})
            self.assertIn("not a JSON array", errors[EXAMPLE.id])
        too_long = [["+", 0, 0, 1]] * (MAX_INSTRUCTIONS + 1)
        _, errors = validate_programs([EXAMPLE], {EXAMPLE.id: too_long})
        self.assertIn("exceed", errors[EXAMPLE.id])

    def test_one_invalid_program_is_reported_among_valid_ones(self):
        other = Puzzle("other", (3,), (6,), 9)
        costs, errors = validate_programs(
            [EXAMPLE, other], {EXAMPLE.id: BROKEN, "other": [["+", 0, 0, 1]]})
        self.assertEqual(costs, {"other": 1})
        self.assertEqual(set(errors), {EXAMPLE.id})

    # --- merging -----------------------------------------------------------------

    def test_merge_keeps_shortest_valid(self):
        best = merge_programs([EXAMPLE], [{EXAMPLE.id: BROKEN}, {EXAMPLE.id: LONG},
                                          {EXAMPLE.id: SHORT}])
        self.assertEqual(best, {EXAMPLE.id: SHORT})

    def test_merge_never_picks_an_invalid_shorter_program(self):
        invalid_short = [["+", 0, 0, 1]]  # 6 is not a target set member -> targets missing
        best = merge_programs([EXAMPLE], [{EXAMPLE.id: invalid_short}, {EXAMPLE.id: LONG}])
        self.assertEqual(best, {EXAMPLE.id: LONG})
        self.assertEqual(merge_programs([EXAMPLE], [{EXAMPLE.id: BROKEN}]), {})

    def test_merge_ties_prefer_earlier_source_and_ignores_unknown_ids(self):
        alt = [["/", 0, 0, 1], ["-", 0, 1, 2], ["+", 0, 2, 3], ["+", 3, 0, 4]]
        best = merge_programs([EXAMPLE], [{EXAMPLE.id: alt, "zzz": SHORT}, {EXAMPLE.id: SHORT}])
        self.assertEqual(best, {EXAMPLE.id: alt})

    def test_merge_zero_instruction_program(self):
        puzzle = Puzzle("p", (1, 2), (1, 2), 5)
        self.assertEqual(merge_programs([puzzle], [{"p": [["+", 0, 0, 5]]}, {"p": []}]), {"p": []})

    # --- datasets ----------------------------------------------------------------

    def test_datasets_and_sample_submission(self):
        scored, practice = load_scored(), load_practice()
        self.assertEqual([p.id for p in scored], [f"scored_{i:04d}" for i in range(1, 1001)])
        self.assertEqual(len(practice), 200)
        costs, errors = validate_programs(scored, read_programs(SAMPLE_SUBMISSION_CSV))
        self.assertEqual(errors, {})
        self.assertEqual(sum(costs.values()), 1490379)

    def test_dataset_invariants(self):
        for puzzle in load_scored() + load_practice():
            self.assertEqual(puzzle.registers, 48, puzzle.id)
            self.assertTrue(1 <= puzzle.value_limit <= 10 ** 18, puzzle.id)
            self.assertTrue(all(v >= 1 for v in puzzle.starting_values + puzzle.targets), puzzle.id)
            self.assertLessEqual(len(puzzle.starting_values), puzzle.registers, puzzle.id)
            self.assertEqual(list(puzzle.targets), sorted(set(puzzle.targets)), puzzle.id)
        for puzzle in load_practice():
            self.assertIsNotNone(puzzle.family)

    def test_sample_submission_round_trips_through_write(self):
        programs = read_programs(SAMPLE_SUBMISSION_CSV)
        out = self.tmp / "copy.csv"
        write_programs(out, programs)
        self.assertEqual(read_programs(out), programs)
        costs, errors = validate_programs(load_scored(), read_programs(out))
        self.assertEqual((errors, sum(costs.values())), ({}, 1490379))

    def test_row_order_does_not_matter(self):
        p1 = Puzzle("scored_0001", (3,), (6,), 10)
        p2 = Puzzle("scored_0002", (5,), (10,), 10)
        path = self.tmp / "shuffled.csv"
        write_raw(path, 'id,program\nscored_0002,"[[""+"",0,0,1]]"\nscored_0001,"[[""+"",0,0,1]]"\n')
        costs, errors = validate_programs([p1, p2], read_programs(path))
        self.assertEqual((costs, errors), ({"scored_0001": 1, "scored_0002": 1}, {}))

    def test_cli_validate_handles_valid_and_malformed_csv(self):
        import contextlib
        import io
        from elfcalc.cli import main as cli_main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli_main(["validate", str(SAMPLE_SUBMISSION_CSV)])
        self.assertEqual(rc, 0)
        self.assertIn("VALID", buf.getvalue())

        bad = self.tmp / "bad_header.csv"
        write_raw(bad, "program,id\n[],scored_0001\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli_main(["validate", str(bad)])
        self.assertEqual(rc, 1)
        self.assertIn("REJECTED", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
