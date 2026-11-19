# Santa 2026 - The Elf Calculator: project guide

## Problem (authoritative summary)

For each of 1,000 puzzles, write the shortest arithmetic program that produces every target.
Score = total number of instructions over all 1,000 programs. Lower is better. No partial credit:
one invalid program rejects the whole submission.

Puzzle fields (`datasets/puzzles.csv`, `datasets/practice_puzzles.csv`):

- `id` - `scored_0001..scored_1000` (practice: `practice_0001..practice_0200`)
- `starting_values` - JSON array of positive ints, placed in registers `0..len-1`
- `targets` - JSON array of positive ints (increasing) that must each appear at least once
- `value_limit` - max value allowed in any register at any time (1..10^18)
- `registers` - always 48
- practice only: `family` (`independent|shared|multiplicative|mixed|tiny`), `design_tier`
  (`hard|medium|easy`), `lower_bound` (= `len(set(targets) - set(starting_values))`)

Calculator rules:

1. 48 registers, indexed 0..47. Registers beyond the starting values are undefined.
2. Instruction `["op", i, j, d]`: read registers `i` and `j`, apply `op`, write to `d`.
   Both sources are read before the write; `d` may equal `i` and/or `j`; `i` may equal `j`.
3. `op` is one of `+ - * /`.
4. Every result must be a positive integer: `a - b` requires `a > b`; `a / b` requires `b | a`.
5. Every result (including intermediates) must be `<= value_limit`.
6. Source registers must be defined.
7. A target is covered once its value appears in any register (including at the start) and stays
   covered after the register is overwritten.
8. Limits: at most 4,096 instructions per program; program cell at most 131,072 UTF-8 bytes.
9. Every instruction costs 1, even useless ones.

Submission: UTF-8 CSV with columns `id,program`; `program` is a JSON array of 4-element
instructions; every scored id exactly once.

Reference numbers: `datasets/sample_submission.csv` scores 1,490,379 (binary double-and-add).
Sum of trivial lower bounds on the scored set is 26,898.

## Folder structure (keep it exactly like this)

```
.
├── AGENTS.md              project guide (this file)
├── README.md
├── Makefile               build / test / solve / merge / validate entry points
├── datasets/              competition data - READ ONLY, never modify
├── elfcalc/               core Python library (stdlib only)
│   ├── puzzle.py          Puzzle dataclass + CSV loaders
│   ├── simulator.py       authoritative rule checker (single source of truth for validity)
│   ├── plan.py            value-level plans -> register programs (register allocation)
│   ├── submission.py      read / write / validate / best-of merge of `id,program` CSVs
│   ├── cli.py             `python3 -m elfcalc <command>`
│   └── solvers/           one module per solver, registered in solvers/__init__.py
├── engine/                C++17 search engine sources; binaries go to engine/bin/ (gitignored)
├── analysis/<family>/     reverse-engineering scripts per puzzle family
├── tests/                 unittest suite (`python3 -m unittest discover -s tests`)
├── solutions/             per-solver outputs, one CSV per solver
│   ├── practice/<solver>.csv
│   └── scored/<solver>.csv
└── submissions/
    └── submission.csv     final best-of merge of solutions/scored/*.csv (always validated)
```

## Conventions

- Python 3 standard library only (no third-party packages). C++17 via `clang++ -O2 -std=c++17`.
- `elfcalc.simulator.simulate` is the only definition of validity. Every program written to
  `solutions/` or `submissions/` must pass it. Never weaken it.
- Solvers think in *values*, not registers: a plan is a list of steps `(op, a, b)` meaning
  "compute `a op b`", where `a` and `b` are values that already exist (starting values or
  results of earlier steps). `elfcalc.plan.compile_plan` turns a plan into register instructions
  and enforces the 48-register limit.
- Solver output files are CSVs with header `id,program` (same format as a submission), covering
  any subset of puzzle ids. `python3 -m elfcalc merge` keeps the shortest valid program per id.
- Never modify `datasets/`. Never hand-edit `submissions/submission.csv`; regenerate it via merge.
- Do not commit generated binaries (`engine/bin/`, `analysis/**/bin/`).

## Commands

```
python3 -m unittest discover -s tests            # run tests
python3 -m elfcalc validate <csv> [--practice]   # validate + score a solution/submission CSV
python3 -m elfcalc solve <solver> [--practice] [--ids ...] [--jobs N]
python3 -m elfcalc merge [--practice]            # best-of merge -> submissions/submission.csv
python3 -m elfcalc report [--practice]           # per-solver / per-family score table
```
