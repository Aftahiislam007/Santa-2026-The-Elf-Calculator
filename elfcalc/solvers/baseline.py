"""Binary double-and-add, sharing every intermediate value between targets.

Always valid (prefixes of a binary expansion never exceed the target), so it
serves as the universal fallback and as a reference for the plan compiler.
"""

from __future__ import annotations

from ..plan import Plan
from ..puzzle import Puzzle


def solve(puzzle: Puzzle, **_: object) -> Plan:
    have = set(puzzle.starting_values)
    plan: Plan = []

    def emit(op: str, a: int, b: int, r: int) -> None:
        if r not in have:
            have.add(r)
            plan.append((op, a, b))

    s = puzzle.starting_values[0]
    emit("/", s, s, 1)
    for t in sorted(puzzle.missing_targets):
        acc = 1
        for bit in bin(t)[3:]:
            emit("+", acc, acc, 2 * acc)
            acc *= 2
            if bit == "1":
                emit("+", acc, 1, acc + 1)
                acc += 1
    return plan
