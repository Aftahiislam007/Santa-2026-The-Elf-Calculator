"""Authoritative implementation of the calculator rules.

This module is the single source of truth for program validity. It is written
for clarity, not speed, and deliberately shares no code with the solvers.
"""

from __future__ import annotations

import json

from .puzzle import Puzzle

OPS = ("+", "-", "*", "/")
MAX_INSTRUCTIONS = 4096
MAX_CELL_BYTES = 131072

Instruction = list  # ["op", i, j, d]
Program = list  # list[Instruction]


class InvalidProgram(ValueError):
    pass


def encode_program(program: Program) -> str:
    """Canonical compact JSON used for the `program` CSV cell."""
    return json.dumps(program, separators=(",", ":"))


def _is_index(x: object, registers: int) -> bool:
    return type(x) is int and 0 <= x < registers


def simulate(puzzle: Puzzle, program: Program) -> int:
    """Replay `program` on `puzzle`; return its cost or raise InvalidProgram."""
    if not isinstance(program, list):
        raise InvalidProgram("program is not a JSON array")
    if len(program) > MAX_INSTRUCTIONS:
        raise InvalidProgram(f"{len(program)} instructions exceed {MAX_INSTRUCTIONS}")
    if len(encode_program(program).encode("utf-8")) > MAX_CELL_BYTES:
        raise InvalidProgram(f"program cell exceeds {MAX_CELL_BYTES} bytes")

    limit = puzzle.value_limit
    regs: list[int | None] = [None] * puzzle.registers
    for i, v in enumerate(puzzle.starting_values):
        regs[i] = v
    pending = set(puzzle.targets) - set(puzzle.starting_values)

    for n, ins in enumerate(program):
        where = f"instruction {n} {ins!r}"
        if not isinstance(ins, (list, tuple)) or len(ins) != 4:
            raise InvalidProgram(f"{where}: not a four-element array")
        op, i, j, d = ins
        if op not in OPS:
            raise InvalidProgram(f"{where}: bad operator")
        if not (_is_index(i, puzzle.registers) and _is_index(j, puzzle.registers)
                and _is_index(d, puzzle.registers)):
            raise InvalidProgram(f"{where}: register index out of range")
        a, b = regs[i], regs[j]
        if a is None or b is None:
            raise InvalidProgram(f"{where}: undefined source register")
        if op == "+":
            r = a + b
        elif op == "*":
            r = a * b
        elif op == "-":
            if a <= b:
                raise InvalidProgram(f"{where}: {a} - {b} is not positive")
            r = a - b
        else:
            if a % b:
                raise InvalidProgram(f"{where}: {a} / {b} is not exact")
            r = a // b
        if r < 1:
            raise InvalidProgram(f"{where}: result {r} is not positive")
        if r > limit:
            raise InvalidProgram(f"{where}: result {r} exceeds value_limit {limit}")
        regs[d] = r
        pending.discard(r)

    if pending:
        some = sorted(pending)[:3]
        raise InvalidProgram(f"{len(pending)} targets not produced, e.g. {some}")
    return len(program)


def check(puzzle: Puzzle, program: Program) -> tuple[bool, str]:
    try:
        simulate(puzzle, program)
    except InvalidProgram as e:
        return False, str(e)
    return True, ""
