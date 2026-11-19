"""Shared-DAG solver wrapper (C++ engine `engine/shared_dag.cpp`)."""

from __future__ import annotations

from ..plan import Plan
from ..puzzle import Puzzle
from .engine import solve_engine


def solve(puzzle: Puzzle, **options: object) -> Plan | None:
    """Run the `shared_dag` C++ engine with forwarded CLI options."""
    return solve_engine("shared_dag", puzzle, default_timeout=120.0, **options)
