"""Overview of the practice set and of the `independent` family.

python3 analysis/independent/00_overview.py
"""
import collections
import math

from common import practice, scored

ps = practice()
print("family x tier:", sorted(collections.Counter((p.family, p.tier) for p in ps).items()))
print()
print("per-family summary (practice):")
for fam in ("independent", "shared", "multiplicative", "mixed", "tiny"):
    q = [p for p in ps if p.family == fam]
    ns = [len(p.start) for p in q]
    nt = [len(p.targets) for p in q]
    smax = [max(p.start) for p in q]
    tb = [max(p.targets).bit_length() for p in q]
    lr = [p.limit / max(p.targets) for p in q]
    print(f"  {fam:14s} n={len(q):3d} |start| {min(ns)}..{max(ns)}  |targets| {min(nt)}..{max(nt)}"
          f"  max start {min(smax)}..{max(smax)}  max-target bits {min(tb)}..{max(tb)}"
          f"  limit/maxT {min(lr):.3f}..{max(lr):.3f}")
print()
print("independent puzzles:")
print("id tier nS nT start | minT maxT | limit limit/maxT log10(maxT) lb")
for p in ps:
    if p.family != "independent":
        continue
    t = p.targets
    print(p.id[-4:], p.tier[:1], len(p.start), len(t), p.start, "|", min(t), max(t), "|", p.limit,
          f"{p.limit / max(t):.3f}", f"{math.log10(max(t)):.2f}", p.lower_bound)
print()
sc = scored()
print("scored:", len(sc), "puzzles; total targets", sum(len(p.targets) for p in sc))
