"""Shared helpers for the `multiplicative` family analysis (stdlib only).

Run every script from the repo root, e.g.
    python3 analysis/multiplicative/01_structure.py
"""
import csv
import json
import math
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRACTICE = os.path.join(ROOT, "datasets", "practice_puzzles.csv")
SCORED = os.path.join(ROOT, "datasets", "puzzles.csv")


class Puzzle:
    __slots__ = ("id", "start", "targets", "limit", "family", "tier", "lb")

    def __init__(self, row):
        self.id = row["id"]
        self.start = json.loads(row["starting_values"])
        self.targets = json.loads(row["targets"])
        self.limit = int(row["value_limit"])
        self.family = row.get("family")
        self.tier = row.get("design_tier")
        lb = row.get("lower_bound")
        self.lb = int(lb) if lb not in (None, "") else len(set(self.targets) - set(self.start))


def load(path):
    with open(path, newline="") as f:
        return [Puzzle(r) for r in csv.DictReader(f)]


def practice():
    return load(PRACTICE)


def scored():
    return load(SCORED)


# ---------------------------------------------------------------- factoring
_SMALL_PRIMES = [p for p in range(2, 1000) if all(p % q for q in range(2, int(p ** 0.5) + 1))]


def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _rho(n):
    if n % 2 == 0:
        return 2
    rnd = random.Random(n)
    while True:
        c = rnd.randrange(1, n)
        y = rnd.randrange(0, n)
        g, r, q = 1, 1, 1
        x = ys = y
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(128, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += 128
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if g != n:
            return g


_FCACHE = {}


def factor(n):
    """Return {prime: exponent} for n >= 1."""
    if n in _FCACHE:
        return _FCACHE[n]
    n0 = n
    f = {}
    for p in _SMALL_PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            f[p] = f.get(p, 0) + 1
            n //= p
    stack = [n] if n > 1 else []
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            f[m] = f.get(m, 0) + 1
            continue
        d = _rho(m)
        stack.append(d)
        stack.append(m // d)
    _FCACHE[n0] = f
    return f


def divisors(n, cap=None):
    """All divisors of n (optionally only those <= cap)."""
    ds = [1]
    for p, e in factor(n).items():
        cur = []
        pe = 1
        for _ in range(e):
            pe *= p
            for d in ds:
                v = d * pe
                if cap is None or v <= cap:
                    cur.append(v)
        ds += cur
    return ds


def fstr(n):
    return " ".join(("%d^%d" % (p, e)) if e > 1 else str(p) for p, e in sorted(factor(n).items())) or "1"


def apply(op, a, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a // b if b and a % b == 0 else 0
    raise ValueError(op)


def check_plan(pz, plan):
    """Validate a value-level plan [(op, a, b), ...] under the calculator rules
    (ignores register allocation). Returns (ok, message)."""
    have = set(pz.start)
    for k, (op, a, b) in enumerate(plan):
        if a not in have or b not in have:
            return False, "step %d: operand missing %r" % (k, (op, a, b))
        if op == "-" and a <= b:
            return False, "step %d: non-positive %r" % (k, (op, a, b))
        if op == "/" and (b == 0 or a % b):
            return False, "step %d: inexact %r" % (k, (op, a, b))
        v = apply(op, a, b)
        if v < 1 or v > pz.limit:
            return False, "step %d: out of range %r -> %d" % (k, (op, a, b), v)
        have.add(v)
    missing = [t for t in pz.targets if t not in have]
    if missing:
        return False, "missing %d targets" % len(missing)
    return True, "ok"
