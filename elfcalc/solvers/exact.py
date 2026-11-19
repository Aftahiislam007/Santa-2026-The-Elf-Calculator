"""Exact optimal search for tiny puzzles (C++ engine `engine/exact.cpp`).

Skips puzzles outside the exact search envelope (`value_limit > 1024` or more
than 5 missing targets) without spawning a subprocess.
"""

from __future__ import annotations

from ..plan import Plan
from ..puzzle import Puzzle
from .engine import solve_engine


def solve(puzzle: Puzzle, **options: object) -> Plan | None:
    """Solve tiny puzzles with `engine/exact.cpp`, returning `None` out of scope."""
    if puzzle.value_limit > 1024 or len(puzzle.missing_targets) > 5:
        return None
    return solve_engine("exact", puzzle, default_timeout=15.0, **options)
