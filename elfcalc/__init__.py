"""Core library for the Santa 2026 "Elf Calculator" competition."""

from .plan import PlanError, Step, compile_plan, execute_plan, prune_plan
from .puzzle import Puzzle, load_practice, load_puzzles, load_scored
from .simulator import InvalidProgram, check, simulate

__all__ = [
    "InvalidProgram",
    "PlanError",
    "Puzzle",
    "Step",
    "check",
    "compile_plan",
    "execute_plan",
    "load_practice",
    "load_puzzles",
    "load_scored",
    "prune_plan",
    "simulate",
]
