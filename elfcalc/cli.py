"""Command line interface: `python3 -m elfcalc <command>`."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from .plan import PlanError, compile_plan
from .puzzle import ROOT, SAMPLE_SUBMISSION_CSV, Puzzle, load_practice, load_scored
from .simulator import InvalidProgram, simulate
from .solvers import SOLVERS, get_solver
from .submission import merge_programs, read_programs, validate_programs, write_programs

SOLUTIONS_DIR = ROOT / "solutions"
SUBMISSION_CSV = ROOT / "submissions" / "submission.csv"


def _dataset(practice: bool) -> tuple[str, list[Puzzle]]:
    return ("practice", load_practice()) if practice else ("scored", load_scored())


def _solve_one(job: tuple[str, Puzzle, dict]) -> tuple[str, list | None, str]:
    name, puzzle, options = job
    try:
        plan = get_solver(name)(puzzle, **options)
        if plan is None:
            return puzzle.id, None, "no solution"
        program = compile_plan(puzzle, plan)
        simulate(puzzle, program)
        return puzzle.id, program, ""
    except (PlanError, InvalidProgram) as e:
        return puzzle.id, None, f"invalid: {e}"
    except Exception as e:  # a crashing solver must not take the batch down
        return puzzle.id, None, f"error: {type(e).__name__}: {e}"


def cmd_solve(args: argparse.Namespace) -> int:
    set_name, puzzles = _dataset(args.practice)
    if args.ids:
        wanted = set(args.ids.split(","))
        puzzles = [p for p in puzzles if p.id in wanted]
    if args.family:
        puzzles = [p for p in puzzles if p.family == args.family]
    if args.limit:
        puzzles = puzzles[: args.limit]
    options = {}
    for item in args.opt or []:
        key, _, value = item.partition("=")
        options[key] = value
    out = Path(args.out) if args.out else SOLUTIONS_DIR / set_name / f"{args.solver}.csv"
    best = read_programs(out) if out.exists() else {}

    jobs = [(args.solver, p, options) for p in puzzles]
    start, done, improved, failed = time.time(), 0, 0, 0
    last_flush = start
    with Pool(args.jobs) as pool:
        for pid, program, message in pool.imap_unordered(_solve_one, jobs):
            done += 1
            if program is None:
                failed += 1
                print(f"{pid}: {message}", file=sys.stderr)
            elif pid not in best or len(program) < len(best[pid]):
                best[pid] = program
                improved += 1
            if done == len(jobs) or time.time() - last_flush > 15:
                last_flush = time.time()
                write_programs(out, best)
                print(f"[{last_flush - start:7.1f}s] {done}/{len(jobs)} done, "
                      f"{improved} improved, {failed} failed", file=sys.stderr)
    total = sum(len(best[p.id]) for p in puzzles if p.id in best)
    print(f"{out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}: "
          f"{sum(p.id in best for p in puzzles)}/{len(puzzles)} puzzles, {total} instructions")
    return 1 if failed else 0


def cmd_validate(args: argparse.Namespace) -> int:
    _, puzzles = _dataset(args.practice)
    programs = read_programs(args.csv)
    costs, errors = validate_programs(puzzles, programs, require_all=not args.partial)
    for pid, message in sorted(errors.items())[:20]:
        print(f"INVALID {pid}: {message}")
    lower = sum(p.lower_bound for p in puzzles if p.id in costs)
    print(f"{len(costs)} valid programs, {len(errors)} errors, "
          f"score {sum(costs.values())} (lower bound {lower})")
    print("VALID" if not errors else "REJECTED")
    return 1 if errors else 0


def _sources(set_name: str) -> dict[str, dict]:
    sources = {}
    for path in sorted((SOLUTIONS_DIR / set_name).glob("*.csv")):
        sources[path.stem] = read_programs(path)
    if set_name == "scored":
        sources["sample_submission"] = read_programs(SAMPLE_SUBMISSION_CSV)
    return sources


def cmd_merge(args: argparse.Namespace) -> int:
    set_name, puzzles = _dataset(args.practice)
    best = merge_programs(puzzles, list(_sources(set_name).values()))
    missing = [p.id for p in puzzles if p.id not in best]
    if missing:
        print(f"no valid program for {len(missing)} puzzles, e.g. {missing[:5]}")
        return 1
    out = Path(args.out) if args.out else (
        SUBMISSION_CSV if set_name == "scored" else ROOT / "submissions" / "practice_best.csv")
    write_programs(out, best)
    costs, errors = validate_programs(puzzles, read_programs(out))
    if errors:
        print(f"merged file is INVALID: {list(errors.items())[:5]}")
        return 1
    print(f"{out}: {len(costs)} programs, score {sum(costs.values())} "
          f"(lower bound {sum(p.lower_bound for p in puzzles)})")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    set_name, puzzles = _dataset(args.practice)
    sources = _sources(set_name)
    best = merge_programs(puzzles, list(sources.values()))
    sources["BEST-OF"] = best
    groups: dict[str, list[Puzzle]] = defaultdict(list)
    for p in puzzles:
        groups[p.family or "all"].append(p)
    if len(groups) > 1:
        groups["all"] = puzzles
    for group, members in groups.items():
        lower = sum(p.lower_bound for p in members)
        print(f"\n== {set_name} / {group}: {len(members)} puzzles, lower bound {lower}")
        print(f"{'solver':<28}{'solved':>8}{'score':>10}{'ratio':>8}{'wins':>7}")
        for name, programs in sources.items():
            solved = [p for p in members if p.id in programs and p.id in best]
            score = sum(len(programs[p.id]) for p in solved)
            lb = sum(p.lower_bound for p in solved)
            wins = sum(len(programs[p.id]) == len(best[p.id]) for p in solved)
            ratio = f"{score / lb:.3f}" if lb else "-"
            print(f"{name:<28}{len(solved):>8}{score:>10}{ratio:>8}{wins:>7}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="elfcalc")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("solve", help="run a solver and store its programs")
    p.add_argument("solver", choices=sorted(SOLVERS))
    p.add_argument("--practice", action="store_true")
    p.add_argument("--ids", help="comma separated puzzle ids")
    p.add_argument("--family", help="practice only: restrict to one family")
    p.add_argument("--limit", type=int, help="only the first N puzzles")
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--opt", action="append", help="solver option key=value (repeatable)")
    p.add_argument("--out", help="output CSV (default solutions/<set>/<solver>.csv)")
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("validate", help="validate and score an id,program CSV")
    p.add_argument("csv")
    p.add_argument("--practice", action="store_true")
    p.add_argument("--partial", action="store_true", help="allow missing ids")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("merge", help="best-of merge of solutions/<set>/*.csv")
    p.add_argument("--practice", action="store_true")
    p.add_argument("--out")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("report", help="score table per solver")
    p.add_argument("--practice", action="store_true")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)
