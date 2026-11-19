"""Shared helpers for the `independent` family analysis (stdlib only).

Run every script from the repo root, e.g. `python3 analysis/independent/01_structure.py`.
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRACTICE = os.path.join(ROOT, "datasets", "practice_puzzles.csv")
SCORED = os.path.join(ROOT, "datasets", "puzzles.csv")


class Puzzle:
    __slots__ = ("id", "start", "targets", "limit", "registers", "family", "tier", "lower_bound")

    def __init__(self, row):
        self.id = row["id"]
        self.start = json.loads(row["starting_values"])
        self.targets = json.loads(row["targets"])
        self.limit = int(row["value_limit"])
        self.registers = int(row["registers"])
        self.family = row.get("family")
        self.tier = row.get("design_tier")
        lb = row.get("lower_bound")
        self.lower_bound = int(lb) if lb not in (None, "") else None


def load(path):
    with open(path, newline="") as f:
        return [Puzzle(r) for r in csv.DictReader(f)]


def practice():
    return load(PRACTICE)


def scored():
    return load(SCORED)


def independent():
    return [p for p in practice() if p.family == "independent"]


def check_plan(p, plan):
    """Value-level validity check of a plan [(op, a, b), ...].

    Every operand must already exist (starting value or earlier result), every result must be
    a positive integer <= value_limit, and every target must be produced. Register allocation
    is NOT checked here (see regs_needed). Returns the number of steps.
    """
    have = set(p.start)
    for k, (op, a, b) in enumerate(plan):
        if a not in have or b not in have:
            raise ValueError(f"{p.id}: step {k} {op} {a} {b}: operand missing")
        if op == "+":
            v = a + b
        elif op == "-":
            v = a - b
        elif op == "*":
            v = a * b
        elif op == "/":
            if a % b:
                raise ValueError(f"{p.id}: step {k}: inexact division")
            v = a // b
        else:
            raise ValueError(op)
        if v < 1 or v > p.limit:
            raise ValueError(f"{p.id}: step {k} {op} {a} {b} = {v} out of range (limit {p.limit})")
        have.add(v)
    missing = [t for t in p.targets if t not in have]
    if missing:
        raise ValueError(f"{p.id}: {len(missing)} targets missing")
    return len(plan)


def regs_needed(p, plan):
    """Peak number of simultaneously live values (values with a later use), i.e. the number of
    registers an ideal allocator needs. Targets need not stay live (rule 7)."""
    last = {}
    for k, (op, a, b) in enumerate(plan):
        last[a] = k
        last[b] = k
    live = set(v for v in p.start)  # starting values occupy registers from the start
    peak = len(live)
    for k, (op, a, b) in enumerate(plan):
        v = {"+": a + b, "-": a - b, "*": a * b, "/": a // b if b else 0}[op]
        # operands whose last use is this step are freed before the write
        for x in (a, b):
            if last.get(x, -1) <= k and x in live:
                live.discard(x)
        if last.get(v, -1) > k:
            live.add(v)
        # starting values never used again could be freed, be conservative: free if no later use
        live = set(x for x in live if last.get(x, -1) > k)
        peak = max(peak, len(live) + 1)
    return peak
