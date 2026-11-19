"""Greedy v2 solver wrapper (C++ engine `engine/greedy_v2.cpp`)."""

from __future__ import annotations

from ..plan import Plan
from ..puzzle import Puzzle
from .engine import solve_engine


def solve(puzzle: Puzzle, **options: object) -> Plan | None:
    """Run the `greedy_v2` C++ engine with forwarded CLI options."""
    return solve_engine("greedy_v2", puzzle, default_timeout=120.0, **options)
