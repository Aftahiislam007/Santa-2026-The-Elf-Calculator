"""MITM basis solver wrapper (C++ engine `engine/mitm_basis.cpp`)."""

from __future__ import annotations

from ..plan import Plan
from ..puzzle import Puzzle
from .engine import solve_engine


def solve(puzzle: Puzzle, **options: object) -> Plan | None:
    """Run the `mitm_basis` C++ engine with forwarded CLI options."""
    return solve_engine("mitm_basis", puzzle, default_timeout=120.0, **options)
