"""Puzzle model and dataset loaders."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets"
SCORED_CSV = DATA_DIR / "puzzles.csv"
PRACTICE_CSV = DATA_DIR / "practice_puzzles.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"


@dataclass(frozen=True)
class Puzzle:
    id: str
    starting_values: tuple[int, ...]
    targets: tuple[int, ...]
    value_limit: int
    registers: int = 48
    family: str | None = None
    design_tier: str | None = None

    @property
    def missing_targets(self) -> frozenset[int]:
        """Targets that are not already covered by the starting values."""
        return frozenset(self.targets) - frozenset(self.starting_values)

    @property
    def lower_bound(self) -> int:
        """Each instruction covers at most one new target."""
        return len(self.missing_targets)


def load_puzzles(path: str | Path) -> list[Puzzle]:
    with open(path, newline="", encoding="utf-8") as f:
        return [
            Puzzle(
                id=row["id"],
                starting_values=tuple(json.loads(row["starting_values"])),
                targets=tuple(json.loads(row["targets"])),
                value_limit=int(row["value_limit"]),
                registers=int(row["registers"]),
                family=row.get("family") or None,
                design_tier=row.get("design_tier") or None,
            )
            for row in csv.DictReader(f)
        ]


def load_scored() -> list[Puzzle]:
    return load_puzzles(SCORED_CSV)


def load_practice() -> list[Puzzle]:
    return load_puzzles(PRACTICE_CSV)
