"""Helpers shared by the analysis/shared/*.py scripts (Python 3 standard library only).

All scripts are meant to be run from the repo root, e.g.

    python3 analysis/shared/01_structure.py
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PRACTICE_CSV = os.path.join(ROOT, 'datasets', 'practice_puzzles.csv')
SCORED_CSV = os.path.join(ROOT, 'datasets', 'puzzles.csv')


def load(practice=True):
    """Return the puzzles as a list of dicts with int fields.

    Keys: id, starts, targets, limit, family, tier, lb.  `family` / `tier` are '' on the
    scored set; `lb` is always len(set(targets) - set(starts)).
    """
    path = PRACTICE_CSV if practice else SCORED_CSV
    out = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            starts = [int(x) for x in json.loads(r['starting_values'])]
            targets = [int(x) for x in json.loads(r['targets'])]
            out.append({
                'id': r['id'],
                'starts': starts,
                'targets': targets,
                'limit': int(r['value_limit']),
                'family': r.get('family') or '',
                'tier': r.get('design_tier') or '',
                'lb': len(set(targets) - set(starts)),
            })
    return out


def apply_op(op, a, b):
    """Value of `a op b` under the calculator rules, or None if the result is not a
    positive integer (the value_limit is checked by the caller)."""
    if op == '+':
        return a + b
    if op == '*':
        return a * b
    if op == '-':
        return a - b if a > b else None
    if op == '/':
        return a // b if b > 0 and a % b == 0 else None
    raise ValueError('unknown op %r' % (op,))


def combine(a, b, limit):
    """All valid one-instruction results of the unordered pair {a, b}.

    Returns a list of (op, x, y, value) with `x op y == value`, 1 <= value <= limit.
    """
    out = []
    s = a + b
    if s <= limit:
        out.append(('+', a, b, s))
    p = a * b
    if p <= limit:
        out.append(('*', a, b, p))
    if a > b:
        out.append(('-', a, b, a - b))
        if a % b == 0:
            out.append(('/', a, b, a // b))
    elif b > a:
        out.append(('-', b, a, b - a))
        if b % a == 0:
            out.append(('/', b, a, b // a))
    else:
        out.append(('/', a, a, 1))
    return out


def cheap_values(starts, limit, max_cost=2):
    """{value: tree cost} for every value whose formula over the starting values uses at
    most `max_cost` instructions (tree cost = cost(a) + cost(b) + 1, no sharing, so it is
    an upper bound on the real cost).  Starting values have cost 0."""
    cost = {}
    for s in starts:
        cost[s] = 0
    levels = [sorted(cost)]
    for k in range(1, max_cost + 1):
        new = set()
        for i in range(0, (k - 1) // 2 + 1):
            j = k - 1 - i
            for a in levels[i]:
                for b in levels[j]:
                    for _op, _x, _y, v in combine(a, b, limit):
                        if v not in cost:
                            new.add(v)
        for v in new:
            cost[v] = k
        levels.append(sorted(new))
    return cost


def verify_plan(starts, targets, limit, plan):
    """Check a value-level plan (sequence of steps whose first three items are op, a, b).

    Raises AssertionError on any rule violation (undefined operand, non-positive or
    fractional result, value_limit exceeded, missing target).  Returns the peak number of
    registers a liveness-based allocator needs (must be <= 48 for a legal program).
    """
    have = set(starts)
    created = {}
    for s in starts:
        created[s] = -1
    last_use = {}
    values = []
    for i, step in enumerate(plan):
        op, a, b = step[0], step[1], step[2]
        if a not in have or b not in have:
            raise AssertionError('step %d uses an undefined value: %r' % (i, step))
        v = apply_op(op, a, b)
        if v is None or v < 1:
            raise AssertionError('step %d is not a positive integer: %r' % (i, step))
        if v > limit:
            raise AssertionError('step %d exceeds value_limit: %r' % (i, step))
        last_use[a] = i
        last_use[b] = i
        if v not in have:
            have.add(v)
            created[v] = i
        values.append(v)
    missing = [t for t in targets if t not in have]
    if missing:
        raise AssertionError('targets not produced: %r' % (missing[:5],))
    peak = len(starts)
    for i, v in enumerate(values):
        live = 0
        for u, c in created.items():
            if c <= i and last_use.get(u, -1) > i:
                live += 1
        if last_use.get(v, -1) <= i:
            live += 1  # the fresh value still needs a register for one instant
        if live > peak:
            peak = live
    return peak


def small_factorization(n, bound=2000):
    """Trial division by primes <= bound.  Returns (list of (p, e), cofactor)."""
    fac = []
    p = 2
    while p <= bound and p * p <= n:
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            fac.append((p, e))
        p += 1 if p == 2 else 2
    if 1 < n <= bound:
        fac.append((n, 1))
        n = 1
    return fac, n


def fmt_factorization(n, bound=2000):
    fac, cof = small_factorization(n, bound)
    parts = ['%d^%d' % (p, e) if e > 1 else '%d' % p for p, e in fac]
    if cof != 1 or not parts:
        parts.append('[%d]' % cof)
    return ' * '.join(parts)


def quantiles(xs):
    """(min, q25, median, q75, max) of a non-empty list."""
    s = sorted(xs)
    n = len(s)

    def q(f):
        return s[min(n - 1, int(f * (n - 1) + 0.5))]

    return s[0], q(0.25), q(0.5), q(0.75), s[-1]
