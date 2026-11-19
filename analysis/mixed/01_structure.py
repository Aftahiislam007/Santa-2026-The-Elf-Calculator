#!/usr/bin/env python3
"""Target partitioning of `mixed` puzzles and 5-family classification validation.

Run from the repo root:
    python3 analysis/mixed/01_structure.py
"""
import collections
import csv
import json
import math
import os


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRACTICE = os.path.join(ROOT, "datasets", "practice_puzzles.csv")
SCORED = os.path.join(ROOT, "datasets", "puzzles.csv")

_SMALL_PRIMES = [p for p in range(2, 600) if all(p % q for q in range(2, int(p ** 0.5) + 1))]


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "id": r["id"],
                "sv": json.loads(r["starting_values"]),
                "targets": json.loads(r["targets"]),
                "limit": int(r["value_limit"]),
                "family": r.get("family", ""),
                "tier": r.get("design_tier", ""),
            })
    return rows


def quantile(xs, q):
    s = sorted(xs)
    if not s:
        return 0.0
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(len(s) - 1, lo + 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def cheap_le1(starts, limit):
    out = set(starts)
    if 1 <= limit:
        out.add(1)
    for i in range(len(starts)):
        for j in range(i, len(starts)):
            a, b = starts[i], starts[j]
            if a + b <= limit:
                out.add(a + b)
            if a * b <= limit:
                out.add(a * b)
            if a != b:
                out.add(abs(a - b))
            if a >= b and a % b == 0:
                out.add(a // b)
            elif b > a and b % a == 0:
                out.add(b // a)
    return sorted(out)


def cofactor_after(n, bound=500):
    for p in _SMALL_PRIMES:
        if p > bound:
            break
        while n % p == 0:
            n //= p
    return n


def top_spike_decade(targets):
    """Return (top_D, count_in_D) for D in {6, 9, 12, 15, 18} with <= 2 targets > 10^D."""
    digs = [int(math.floor(math.log10(t))) + 1 for t in targets]
    cnt = collections.Counter(digs)
    cands = []
    for d in (6, 9, 12, 15, 18):
        if sum(1 for x in digs if x > d) <= 2:
            cands.append((cnt[d], d))
    if not cands:
        return 18, cnt[18]
    best_cnt, best_d = max(cands)
    return best_d, best_cnt


def partition_mixed_targets(puzzle):
    """Partition targets of a mixed-like puzzle into structured vs top-decade random."""
    ts = sorted(puzzle["targets"])
    n = len(ts)
    digs = [int(math.floor(math.log10(t))) + 1 for t in ts]
    d_top, _ = top_spike_decade(ts)

    cs = cheap_le1(puzzle["sv"], puzzle["limit"])
    aff_links = set()
    gcd_links = set()
    for i in range(n):
        for j in range(i + 1, n):
            lo, hi = ts[i], ts[j]
            q = round(hi / lo)
            if 1 <= q <= 512 and abs(hi - q * lo) <= max(5000, lo // 100000):
                aff_links.add(i)
                aff_links.add(j)
            if math.gcd(lo, hi) > 500:
                gcd_links.add(i)
                gcd_links.add(j)

    sm_top_bound = 20 if d_top <= 6 else (100 if d_top <= 9 else 300)
    sm_struct = set()
    for i, t in enumerate(ts):
        b = sm_top_bound if digs[i] == d_top else 300
        if cofactor_after(t, b) == 1 or any(
            (t + c <= puzzle["limit"] and cofactor_after(t + c, b) == 1)
            or (t > c and cofactor_after(t - c, b) == 1)
            for c in cs
        ):
            sm_struct.add(i)

    structured = []
    top_random = []
    linked_in_top = []
    for i, (t, d) in enumerate(zip(ts, digs)):
        if d != d_top:
            structured.append((i, t))
        elif (i in aff_links) or (i in gcd_links) or (i in sm_struct):
            structured.append((i, t))
            linked_in_top.append((i, t))
        else:
            top_random.append((i, t))

    return {
        "d_top": d_top,
        "n_lt": sum(1 for d in digs if d < d_top),
        "n_eq": sum(1 for d in digs if d == d_top),
        "n_gt": sum(1 for d in digs if d > d_top),
        "linked_in_top": len(linked_in_top),
        "structured": [t for _, t in structured],
        "top_random": [t for _, t in top_random],
        "struct_aff": sum(1 for i, _ in structured if i in aff_links),
        "struct_gcd": sum(1 for i, _ in structured if i in gcd_links),
        "struct_sm300": sum(1 for i, _ in structured if i in sm_struct),
    }


def classify_puzzle(puzzle):
    """Classify a puzzle into {tiny, independent, mixed, shared, multiplicative}."""
    ts = sorted(puzzle["targets"])
    n = len(ts)
    if ts[-1] <= 100:
        return "tiny"
    if max(puzzle["sv"]) > 40:
        return "independent"

    best_d, n_best_d = top_spike_decade(ts)
    sa_loose = set()
    ea, ea500 = 0, 0
    for i in range(n):
        tol = min(5000, ts[i] // 2000)
        tol500 = min(500, ts[i] // 50)
        tol_loose = max(5000, ts[i] // 100000)
        for j in range(i + 1, n):
            q = round(ts[j] / ts[i])
            if 1 <= q <= 512:
                d = abs(ts[j] - q * ts[i])
                if tol > 0 and d <= tol:
                    ea += 1
                if d <= tol500:
                    ea500 += 1
                if d <= tol_loose:
                    sa_loose.add(i)
                    sa_loose.add(j)

    if n > 14:
        if n_best_d >= 14:
            return "mixed"
    else:
        if n_best_d / n >= 0.6 and len(sa_loose) / n < 0.7:
            return "mixed"

    if n <= 14:
        return "shared" if ea500 >= 4 else "multiplicative"

    cs = cheap_le1(puzzle["sv"], puzzle["limit"])
    cofs = []
    for t in ts:
        mc = cofactor_after(t, 500)
        for c in cs:
            if t + c <= puzzle["limit"]:
                mc = min(mc, cofactor_after(t + c, 500))
            if t > c:
                mc = min(mc, cofactor_after(t - c, 500))
        cofs.append(mc)
    cofs.sort()
    p50 = cofs[n // 2]
    p75 = cofs[(3 * n) // 4]
    if ea <= 7 and p50 == 1 and p75 <= 1000:
        return "multiplicative"
    return "shared"


def analyze_mixed_partition(puzzles, label):
    print(f"=== Mixed Target Partitioning: {label} (n={len(puzzles)}) ===")
    by_d = collections.defaultdict(list)
    for p in puzzles:
        part = partition_mixed_targets(p)
        by_d[part["d_top"]].append((p, part))

    for d in sorted(by_d):
        items = by_d[d]
        n_lt = [pt["n_lt"] for _, pt in items]
        n_eq = [pt["n_eq"] for _, pt in items]
        n_gt = [pt["n_gt"] for _, pt in items]
        n_str = [len(pt["structured"]) for _, pt in items]
        n_rnd = [len(pt["top_random"]) for _, pt in items]
        n_lnk = [pt["linked_in_top"] for _, pt in items]
        # limit / max(top_decade_random_targets) when top_random non-empty
        lim_maxd = []
        for p, pt in items:
            if pt["top_random"]:
                lim_maxd.append(p["limit"] / max(pt["top_random"]))
        c15 = sum(1 for x in lim_maxd if x >= 1.5)
        c20 = sum(1 for x in lim_maxd if x >= 2.0)
        print(f"  D=10^{d:2d} (n={len(items):3d}): "
              f"<D:{min(n_lt)}..{max(n_lt)}  =D:{min(n_eq)}..{max(n_eq)} (linked:{min(n_lnk)}..{max(n_lnk)})  "
              f">D:{min(n_gt)}..{max(n_gt)} | "
              f"structured:{min(n_str)}..{max(n_str)} (med {int(quantile(n_str, 0.5))})  "
              f"top_random:{min(n_rnd)}..{max(n_rnd)} (med {int(quantile(n_rnd, 0.5))})")
        if lim_maxd:
            print(f"    limit/max(top_random): min={min(lim_maxd):.3f} p50={quantile(lim_maxd, 0.5):.3f} "
                  f"max={max(lim_maxd):.3f} | >=1.5:{c15}/{len(lim_maxd)} >=2.0:{c20}/{len(lim_maxd)}")

    tot_str = sum(len(pt["structured"]) for items in by_d.values() for _, pt in items)
    tot_rnd = sum(len(pt["top_random"]) for items in by_d.values() for _, pt in items)
    tot_aff = sum(pt["struct_aff"] for items in by_d.values() for _, pt in items)
    tot_gcd = sum(pt["struct_gcd"] for items in by_d.values() for _, pt in items)
    tot_sm3 = sum(pt["struct_sm300"] for items in by_d.values() for _, pt in items)
    tot_all = tot_str + tot_rnd
    print(f"  Overall partition: structured={tot_str}/{tot_all} ({100*tot_str/tot_all:.1f}%), "
          f"top_decade_random={tot_rnd}/{tot_all} ({100*tot_rnd/tot_all:.1f}%)")
    print(f"  Within structured: affine={100*tot_aff/tot_str:.1f}%, gcd>500={100*tot_gcd/tot_str:.1f}%, "
          f"1-step smooth<=300={100*tot_sm3/tot_str:.1f}%")
    print()


def main():
    pr = load(PRACTICE)
    sc = load(SCORED)

    mx_pr = [p for p in pr if p["family"] == "mixed"]
    analyze_mixed_partition(mx_pr, "practice mixed")

    # Validate top_spike_cnt separation gap on n > 14, max(sv) <= 40
    print("=== Top-Decade Spike Count Gap (nT > 14, max(sv) <= 40) ===")
    for name, ds in (("practice", pr), ("scored", sc)):
        non_mix_max = 0
        mix_min = 999
        gap_cnt = 0
        hist = collections.Counter()
        for p in ds:
            if max(p["targets"]) <= 100 or max(p["sv"]) > 40 or len(p["targets"]) <= 14:
                continue
            _, c = top_spike_decade(p["targets"])
            hist[c] += 1
            if 9 <= c <= 13:
                gap_cnt += 1
            if p["family"] == "mixed":
                mix_min = min(mix_min, c)
            elif p["family"] in ("shared", "multiplicative"):
                non_mix_max = max(non_mix_max, c)
        if name == "practice":
            print(f"  {name:8s}: non-mixed max={non_mix_max}, mixed min={mix_min}, in [9..13]={gap_cnt}")
        else:
            below = sum(v for k, v in hist.items() if k <= 8)
            above = sum(v for k, v in hist.items() if k >= 14)
            print(f"  {name:8s}: <=8: {below}, in [9..13]: {gap_cnt}, >=14: {above}")
    print()

    # Validate 5-family classification on practice and apply to scored
    print("=== 5-Family Classifier Validation ===")
    pr_preds = [(p, classify_puzzle(p)) for p in pr]
    cm = collections.Counter((p["family"], pred) for p, pred in pr_preds)
    acc = sum(v for (a, b), v in cm.items() if a == b)
    print(f"  Practice accuracy: {acc}/{len(pr)} ({100*acc/len(pr):.1f}%)")
    print("  Practice confusion matrix:")
    for (true_f, pred_f), cnt in sorted(cm.items()):
        print(f"    {true_f:14s} -> {pred_f:14s}: {cnt:3d}")
    errs = [(p["id"], p["family"], pred) for p, pred in pr_preds if p["family"] != pred]
    print(f"  Misclassified ({len(errs)}): {errs}")
    print()

    sc_preds = [(p, classify_puzzle(p)) for p in sc]
    sc_cnt = collections.Counter(pred for _, pred in sc_preds)
    print(f"  Scored family predictions (n={len(sc)}): {dict(sorted(sc_cnt.items()))}")
    for fam in ("tiny", "independent", "shared", "multiplicative", "mixed"):
        sub = [p for p, pred in sc_preds if pred == fam]
        med_n = sum(1 for p in sub if len(p["targets"]) <= 14)
        nt_sum = sum(len(p["targets"]) for p in sub)
        print(f"    {fam:14s}: n={len(sub):3d} (nT<=14: {med_n:3d}, nT>14: {len(sub)-med_n:3d}) | total_targets={nt_sum}")
    print()

    sc_mx = [p for p, pred in sc_preds if pred == "mixed"]
    analyze_mixed_partition(sc_mx, "scored predicted mixed")


if __name__ == "__main__":
    main()
