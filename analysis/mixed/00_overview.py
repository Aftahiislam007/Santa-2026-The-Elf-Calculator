#!/usr/bin/env python3
"""Overview of the `mixed` and `tiny` practice puzzles.

Run from the repo root:  python3 analysis/mixed/00_overview.py
"""
import collections
import csv
import json


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["starting_values"] = json.loads(r["starting_values"])
            r["targets"] = json.loads(r["targets"])
            r["value_limit"] = int(r["value_limit"])
            if "lower_bound" in r:
                r["lower_bound"] = int(r["lower_bound"])
            rows.append(r)
    return rows


def main():
    rows = load("datasets/practice_puzzles.csv")
    c = collections.Counter((r["family"], r["design_tier"]) for r in rows)
    for k in sorted(c):
        print(k, c[k])
    for fam in ("mixed", "tiny"):
        print("=" * 20, fam)
        for r in rows:
            if r["family"] != fam:
                continue
            t = r["targets"]
            print(r["id"], r["design_tier"], "sv=", r["starting_values"], "nT=", len(t),
                  "limit=", r["value_limit"], "lb=", r["lower_bound"])
            print("   ", t)


if __name__ == "__main__":
    main()
