"""Value-level plans and their compilation to register programs.

A plan is a list of steps `(op, a, b)`: "compute `a op b`", where `a` and `b`
are values that already exist (starting values or results of earlier steps).
Solvers produce plans; `compile_plan` assigns registers.
"""

from __future__ import annotations

import heapq

from .puzzle import Puzzle

Step = tuple[str, int, int]
Plan = list[Step]


class PlanError(ValueError):
    pass


def apply_op(op: str, a: int, b: int, limit: int) -> int:
    if op == "+":
        r = a + b
    elif op == "*":
        r = a * b
    elif op == "-":
        if a <= b:
            raise PlanError(f"{a} - {b} is not positive")
        r = a - b
    elif op == "/":
        if a % b:
            raise PlanError(f"{a} / {b} is not exact")
        r = a // b
    else:
        raise PlanError(f"bad operator {op!r}")
    if r > limit:
        raise PlanError(f"{a} {op} {b} = {r} exceeds value_limit {limit}")
    return r


def execute_plan(puzzle: Puzzle, plan: Plan) -> list[int]:
    """Return the result of every step; raise PlanError on any illegal step."""
    have = set(puzzle.starting_values)
    results = []
    for n, (op, a, b) in enumerate(plan):
        if a not in have or b not in have:
            raise PlanError(f"step {n} ({a} {op} {b}): operand not available")
        r = apply_op(op, a, b, puzzle.value_limit)
        have.add(r)
        results.append(r)
    missing = set(puzzle.targets) - have
    if missing:
        raise PlanError(f"{len(missing)} targets not produced, e.g. {sorted(missing)[:3]}")
    return results


def prune_plan(puzzle: Puzzle, plan: Plan) -> Plan:
    """Drop steps that recompute an existing value or whose result is never needed."""
    results = execute_plan(puzzle, plan)
    have = set(puzzle.starting_values)
    fresh = []
    for step, r in zip(plan, results):
        if r not in have:
            have.add(r)
            fresh.append((step, r))
    needed = set(puzzle.missing_targets)
    kept = []
    for step, r in reversed(fresh):
        if r in needed:
            kept.append(step)
            needed.update(step[1:])
    kept.reverse()
    return kept


def _allocate(puzzle: Puzzle, steps: list[tuple[Step, int]]) -> list[list]:
    """Assign registers to pruned steps given in execution order."""
    last_use: dict[int, int] = {}
    for k, ((_, a, b), _) in enumerate(steps):
        last_use[a] = last_use[b] = k

    reg_of: dict[int, int] = {}
    free = list(range(len(puzzle.starting_values), puzzle.registers))
    for i, v in enumerate(puzzle.starting_values):
        if v in last_use and v not in reg_of:
            reg_of[v] = i
        else:
            free.append(i)
    heapq.heapify(free)

    program = []
    for k, ((op, a, b), r) in enumerate(steps):
        ra, rb = reg_of[a], reg_of[b]
        for v in {a, b}:
            if last_use[v] == k:
                heapq.heappush(free, reg_of.pop(v))
        if not free:
            raise PlanError(f"more than {puzzle.registers} live values at step {k}")
        if r in last_use:
            d = heapq.heappop(free)
            reg_of[r] = d
        else:
            d = free[0]
        program.append([op, ra, rb, d])
    return program


def _reorder(puzzle: Puzzle, steps: list[tuple[Step, int]]) -> list[tuple[Step, int]]:
    """Greedy topological order that keeps few values live at once."""
    users: dict[int, int] = {}
    for (_, a, b), _ in steps:
        for v in {a, b}:
            users[v] = users.get(v, 0) + 1
    have = set(puzzle.starting_values)
    pending = list(range(len(steps)))
    order = []
    while pending:
        best, best_gain = None, None
        for k in pending:
            (_, a, b), r = steps[k]
            if a not in have or b not in have:
                continue
            gain = sum(1 for v in {a, b} if users[v] == 1) - (1 if users.get(r) else 0)
            if best_gain is None or gain > best_gain:
                best, best_gain = k, gain
        (_, a, b), r = steps[best]
        for v in {a, b}:
            users[v] -= 1
        have.add(r)
        pending.remove(best)
        order.append(steps[best])
    return order


def compile_plan(puzzle: Puzzle, plan: Plan) -> list[list]:
    """Prune `plan` and turn it into a register program for `puzzle`."""
    pruned = prune_plan(puzzle, plan)
    steps = list(zip(pruned, execute_plan(puzzle, pruned)))
    try:
        return _allocate(puzzle, steps)
    except PlanError:
        return _allocate(puzzle, _reorder(puzzle, steps))
