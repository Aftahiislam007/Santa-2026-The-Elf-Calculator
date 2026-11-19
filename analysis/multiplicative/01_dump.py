"""Dump puzzles of given families with factorizations.

    python3 analysis/multiplicative/01_dump.py multiplicative,shared [max_puzzles] [tier]
"""
import sys
from collections import Counter

from common import practice, fstr

fams = sys.argv[1].split(",") if len(sys.argv) > 1 else ["multiplicative"]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
tier = sys.argv[3] if len(sys.argv) > 3 else None
ps = practice()
print(Counter((p.family, p.tier) for p in ps))
sel = [p for p in ps if p.family in fams and (tier is None or p.tier == tier)]
sel.sort(key=lambda p: max(p.targets))
for p in sel[:n]:
    print()
    print(p.id, p.family, p.tier, p.start, "limit", p.limit, "lb", p.lb)
    for t in p.targets:
        print("  %20d  %s" % (t, fstr(t)))
