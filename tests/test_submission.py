import tempfile
import unittest
from pathlib import Path

from elfcalc import Puzzle, load_practice, load_scored
from elfcalc.puzzle import SAMPLE_SUBMISSION_CSV
from elfcalc.submission import merge_programs, read_programs, validate_programs, write_programs

EXAMPLE = Puzzle("example_0001", (3,), (1, 2, 5, 8), 9)
SHORT = [["/", 0, 0, 1], ["-", 0, 1, 2], ["+", 0, 2, 3], ["+", 0, 3, 4]]
LONG = SHORT + [["+", 0, 1, 5]]
BROKEN = SHORT[:2]


class SubmissionTest(unittest.TestCase):
    def test_round_trip_matches_competition_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            write_programs(path, {EXAMPLE.id: SHORT})
            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                ["id,program",
                 'example_0001,"[[""/"",0,0,1],[""-"",0,1,2],[""+"",0,2,3],[""+"",0,3,4]]"'],
            )
            self.assertEqual(read_programs(path), {EXAMPLE.id: SHORT})

    def test_validate_reports_missing_invalid_and_unknown(self):
        costs, errors = validate_programs([EXAMPLE], {EXAMPLE.id: SHORT})
        self.assertEqual((costs, errors), ({EXAMPLE.id: 4}, {}))
        _, errors = validate_programs([EXAMPLE], {})
        self.assertEqual(errors, {EXAMPLE.id: "missing program"})
        _, errors = validate_programs([EXAMPLE], {EXAMPLE.id: BROKEN, "nope": []})
        self.assertEqual(set(errors), {EXAMPLE.id, "nope"})

    def test_merge_keeps_shortest_valid(self):
        best = merge_programs([EXAMPLE], [{EXAMPLE.id: BROKEN}, {EXAMPLE.id: LONG},
                                          {EXAMPLE.id: SHORT}])
        self.assertEqual(best, {EXAMPLE.id: SHORT})

    def test_datasets_and_sample_submission(self):
        scored, practice = load_scored(), load_practice()
        self.assertEqual([p.id for p in scored], [f"scored_{i:04d}" for i in range(1, 1001)])
        self.assertEqual(len(practice), 200)
        costs, errors = validate_programs(scored, read_programs(SAMPLE_SUBMISSION_CSV))
        self.assertEqual(errors, {})
        self.assertEqual(sum(costs.values()), 1490379)


if __name__ == "__main__":
    unittest.main()
