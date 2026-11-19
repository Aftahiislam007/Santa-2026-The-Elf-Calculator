"""Pool-based greedy search (C++ engine `engine/greedy.cpp`).

Options are forwarded to the engine, e.g.
`python3 -m elfcalc solve greedy --opt restarts=8 --opt time=60 --opt seed=1`.
"""

from __future__ import annotations

from ..plan import Plan
from ..puzzle import Puzzle
from .engine import solve_engine


def solve(puzzle: Puzzle, **options: object) -> Plan | None:
    """Run the `greedy` C++ engine with forwarded CLI options."""
    return solve_engine("greedy", puzzle, default_timeout=120.0, **options)
