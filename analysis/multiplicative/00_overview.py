"""Overview of the labelled practice set, with focus on `multiplicative`.

    python3 analysis/multiplicative/00_overview.py [n_dump]
"""
import sys
from collections import Counter

from common import practice, fstr

ps = practice()
print(Counter((p.family, p.tier) for p in ps))
mult = [p for p in ps if p.family == "multiplicative"]
print("id tier start ntargets limit limit/max lb min max")
for p in mult:
    print(p.id, p.tier, p.start, len(p.targets), p.limit, round(p.limit / max(p.targets), 3), p.lb,
          min(p.targets), max(p.targets))

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for p in mult[:n]:
    print()
    print(p.id, p.start, "limit", p.limit, fstr(p.limit))
    for t in p.targets:
        print("  %20d  %s" % (t, fstr(t)))
