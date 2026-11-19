"""Mixed-hybrid solver wrapper (C++ engine `engine/mixed_hybrid.cpp`)."""

from __future__ import annotations

from ..plan import Plan
from ..puzzle import Puzzle
from .engine import solve_engine


def solve(puzzle: Puzzle, **options: object) -> Plan | None:
    """Run the `mixed_hybrid` C++ engine with forwarded CLI options."""
    return solve_engine("mixed_hybrid", puzzle, default_timeout=120.0, **options)
