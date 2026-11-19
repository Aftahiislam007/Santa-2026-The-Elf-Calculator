"""Heuristic puzzle family/tier classifier and mixed-target partitioner."""

from __future__ import annotations

import math
import statistics

from .puzzle import Puzzle

_MIXED_HARD_DECADES = (12, 15, 18)
_DOMINANT_DECADES = (6, 9, 12, 15, 18)


def _count_close_pairs(sorted_targets: list[int], rel_tol: float) -> int:
    return sum(
        1 for a, b in zip(sorted_targets, sorted_targets[1:])
        if (b - a) / a < rel_tol
    )


def _count_affine_links(sorted_targets: list[int], rel_tol: float = 1e-4) -> int:
    n = len(sorted_targets)
    count = 0
    for i in range(n):
        ti = sorted_targets[i]
        for j in range(i + 1, n):
            tj = sorted_targets[j]
            q = round(tj / ti)
            if 2 <= q <= 200 and abs(tj - q * ti) / ti < rel_tol:
                count += 1
    return count


def classify_puzzle(puzzle: Puzzle) -> tuple[str, str]:
    """Return ``(family, tier)`` for a puzzle.

    ``family`` is one of ``{"tiny", "independent", "mixed", "shared", "multiplicative"}``,
    and ``tier`` is one of ``{"easy", "medium", "hard"}``.
    """
    targets = sorted(puzzle.targets)
    n = len(targets)
    if n == 0 or max(targets) <= 20 or n <= 4:
        return ("tiny", "easy")

    tier = "medium" if n <= 20 else "hard"
    if max(puzzle.starting_values) > 34:
        return ("independent", tier)

    t1e4 = _count_close_pairs(targets, 1e-4)
    med_log10 = statistics.median(math.log10(t) for t in targets)

    if tier == "medium":
        t1e3 = _count_close_pairs(targets, 1e-3)
        frac_6digit = sum(1 for t in targets if len(str(t)) == 6) / n
        if med_log10 < 4.9 and t1e3 == 0 and frac_6digit < 0.30:
            return ("multiplicative", tier)
        if frac_6digit >= 0.45 and t1e4 == 0:
            return ("mixed", tier)
        return ("shared", tier)

    best_d = max(
        _MIXED_HARD_DECADES,
        key=lambda d: (sum(1 for t in targets if len(str(t)) == d), d),
    )
    in_d = [t for t in targets if len(str(t)) == best_d]
    best_frac = len(in_d) / n
    above = sum(1 for t in targets if t >= 10**best_d)
    t1e4_in_d = _count_close_pairs(in_d, 1e-4)
    if best_frac >= 0.48 and above <= 3 and t1e4_in_d <= 1:
        return ("mixed", tier)

    aff1e4 = _count_affine_links(targets, 1e-4)
    if t1e4 >= 2 or med_log10 > 8.1 or (t1e4 >= 1 and aff1e4 >= 2):
        return ("shared", tier)
    return ("multiplicative", tier)


def partition_mixed_targets(puzzle: Puzzle) -> tuple[list[int], list[int]]:
    """Split ``puzzle.targets`` into ``(structured_targets, random_targets)``.

    ``random_targets`` are isolated targets in the dominant top decade
    (``d in {6, 9, 12, 15, 18}``) with no close pair or small affine link.
    """
    targets = sorted(puzzle.targets)
    if not targets:
        return ([], [])

    best_d = max(
        _DOMINANT_DECADES,
        key=lambda d: (sum(1 for t in targets if len(str(t)) == d), d),
    )
    if sum(1 for t in targets if len(str(t)) == best_d) == 0:
        return (targets, [])

    linked: set[int] = set()
    n = len(targets)
    for i in range(n - 1):
        if (targets[i + 1] - targets[i]) / targets[i] < 1e-4:
            linked.add(i)
            linked.add(i + 1)

    for i in range(n):
        ti = targets[i]
        for j in range(i + 1, n):
            tj = targets[j]
            q = round(tj / ti)
            if 2 <= q <= 200 and abs(tj - q * ti) / ti < 1e-4:
                linked.add(i)
                linked.add(j)

    structured_targets: list[int] = []
    random_targets: list[int] = []
    for i, t in enumerate(targets):
        if len(str(t)) == best_d and i not in linked:
            random_targets.append(t)
        else:
            structured_targets.append(t)

    return (structured_targets, random_targets)
