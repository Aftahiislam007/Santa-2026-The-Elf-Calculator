"""Pool-based greedy search (C++ engine `engine/greedy.cpp`).

Options are forwarded to the engine, e.g.
`python3 -m elfcalc solve greedy --opt restarts=8 --opt time=60 --opt seed=1`.
"""

from __future__ import annotations

from ..plan import Plan
from ..puzzle import Puzzle
from .engine import run_engine


def solve(puzzle: Puzzle, **options: str) -> Plan | None:
    args: list[str] = []
    for key, value in options.items():
        args += [f"--{key}", str(value)]
    return run_engine("greedy", puzzle, args)
