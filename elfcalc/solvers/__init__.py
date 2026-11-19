"""Solver registry.

A solver is a callable `solve(puzzle, **options) -> Plan | None` returning a
value-level plan (see `elfcalc.plan`). Register it here under a short name;
`python3 -m elfcalc solve <name>` runs it and stores the validated programs in
`solutions/<practice|scored>/<name>.csv`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Callable

SOLVERS: dict[str, str] = {
    "baseline": "elfcalc.solvers.baseline:solve",
}


def get_solver(name: str) -> Callable:
    if name not in SOLVERS:
        raise KeyError(f"unknown solver {name!r}; available: {', '.join(sorted(SOLVERS))}")
    module, func = SOLVERS[name].split(":")
    return getattr(import_module(module), func)
