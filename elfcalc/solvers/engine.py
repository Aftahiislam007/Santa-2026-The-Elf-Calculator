"""Bridge to the C++ search engines in `engine/`.

Protocol (text, whitespace separated). The engine reads one puzzle on stdin:

    <id>
    <value_limit>
    <k> <s_1> ... <s_k>          starting values
    <n> <t_1> ... <t_n>          targets

and writes a value-level plan on stdout, one step per line: `<op> <a> <b>`
(`op` in `+ - * /`, `a` and `b` decimal values). Lines starting with `#` are
ignored. A non-zero exit status means "no solution".
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..plan import Plan
from ..puzzle import ROOT, Puzzle

ENGINE_BIN = ROOT / "engine" / "bin"


def engine_input(puzzle: Puzzle) -> str:
    return "\n".join([
        puzzle.id,
        str(puzzle.value_limit),
        " ".join(map(str, (len(puzzle.starting_values), *puzzle.starting_values))),
        " ".join(map(str, (len(puzzle.targets), *puzzle.targets))),
    ]) + "\n"


def parse_plan(text: str) -> Plan:
    plan: Plan = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        op, a, b = line.split()
        plan.append((op, int(a), int(b)))
    return plan


def run_engine(name: str, puzzle: Puzzle, args: list[str] | tuple[str, ...] = (),
               timeout: float | None = None) -> Plan | None:
    binary: Path = ENGINE_BIN / name
    if not binary.exists():
        raise FileNotFoundError(f"{binary} not built; run `make engine`")
    try:
        result = subprocess.run([str(binary), *map(str, args)], input=engine_input(puzzle),
                                capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    return parse_plan(result.stdout)


def solve_engine(
    name: str,
    puzzle: Puzzle,
    default_timeout: float | None = 120.0,
    **options: object,
) -> Plan | None:
    """Forward ``--key value`` options and ``timeout`` to engine ``name``."""
    raw_timeout = options.pop("timeout", default_timeout)
    timeout = None if raw_timeout in (None, "", "none", "None") else float(raw_timeout)
    args: list[str] = []
    for key, value in options.items():
        args += [f"--{key}", str(value)]
    return run_engine(name, puzzle, args=args, timeout=timeout)
