"""Reading, writing, validating and merging `id,program` CSV files."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from .puzzle import Puzzle
from .simulator import InvalidProgram, encode_program, simulate

Programs = dict[str, list]


def read_programs(path: str | Path) -> Programs:
    csv.field_size_limit(1 << 30)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "program"]:
            raise ValueError(f"{path}: expected columns id,program, got {reader.fieldnames}")
        programs: Programs = {}
        for row in reader:
            if row["id"] in programs:
                raise ValueError(f"{path}: duplicate id {row['id']}")
            programs[row["id"]] = json.loads(row["program"])
        return programs


def write_programs(path: str | Path, programs: Programs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "program"])
        for pid in sorted(programs):
            writer.writerow([pid, encode_program(programs[pid])])
    os.replace(tmp, path)


def validate_programs(
    puzzles: list[Puzzle], programs: Programs, require_all: bool = True
) -> tuple[dict[str, int], dict[str, str]]:
    """Return (cost per valid id, error per invalid id)."""
    costs: dict[str, int] = {}
    errors: dict[str, str] = {}
    known = {p.id for p in puzzles}
    for pid in programs:
        if pid not in known:
            errors[pid] = "unknown puzzle id"
    for p in puzzles:
        if p.id not in programs:
            if require_all:
                errors[p.id] = "missing program"
            continue
        try:
            costs[p.id] = simulate(p, programs[p.id])
        except InvalidProgram as e:
            errors[p.id] = str(e)
    return costs, errors


def merge_programs(puzzles: list[Puzzle], sources: list[Programs]) -> Programs:
    """Keep the shortest valid program per puzzle (earlier sources win ties)."""
    best: Programs = {}
    for p in puzzles:
        for src in sources:
            prog = src.get(p.id)
            if prog is None or (p.id in best and len(prog) >= len(best[p.id])):
                continue
            try:
                simulate(p, prog)
            except InvalidProgram:
                continue
            best[p.id] = prog
    return best
