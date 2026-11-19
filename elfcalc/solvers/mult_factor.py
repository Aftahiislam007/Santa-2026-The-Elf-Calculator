"""Multiplicative-factorization solver wrapper (C++ engine `engine/mult_factor.cpp`)."""

from __future__ import annotations

from ..plan import Plan
from ..puzzle import Puzzle
from .engine import solve_engine


def solve(puzzle: Puzzle, **options: object) -> Plan | None:
    """Run the `mult_factor` C++ engine with forwarded CLI options."""
    return solve_engine("mult_factor", puzzle, default_timeout=120.0, **options)
