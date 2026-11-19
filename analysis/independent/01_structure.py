"""Structural analysis of the `independent` family (and `max(sv) > 40` scored set).

Run from the repo root:
    python3 analysis/independent/01_structure.py
"""
import bisect
import collections
import math
import random

from common import independent, scored


def quantile(xs, q):
    s = sorted(xs)
    if not s:
        return 0.0
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(len(s) - 1, lo + 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def decade_of(x):
    return int(math.floor(math.log10(x))) + 1


def eval_l1(basis, limit):
    """All distinct valid 1-step values in [1, limit] not in basis reachable from basis."""
    s_b = set(basis)
    l1 = set()
    n = len(basis)
    for i in range(n):
        a = basis[i]
        for j in range(i, n):
            b = basis[j]
            v = a + b
            if v <= limit and v not in s_b:
                l1.add(v)
            v = a * b
            if v <= limit and v not in s_b:
                l1.add(v)
            if a > b:
                if a - b not in s_b:
                    l1.add(a - b)
                if a % b == 0 and (a // b) not in s_b:
                    l1.add(a // b)
            elif b > a:
                if b - a not in s_b:
                    l1.add(b - a)
                if b % a == 0 and (b // a) not in s_b:
                    l1.add(b // a)
            else:
                if 1 not in s_b:
                    l1.add(1)
    return l1


def build_multiscale_basis(starts, size, limit, seed=0):
    """Construct a valid multi-scale basis B of `size` reachable from `starts`.

    Greedily extends B from L1(B) by maximizing distinct new L1 pairs while
    balancing half-decade log10 coverage from 1 up to limit.
    """
    rng = random.Random(seed)
    basis = list(dict.fromkeys(starts))
    if 1 not in basis:
        if len(starts) >= 1 and 1 <= limit:
            basis.append(1)  # 1 = s_0 / s_0
    if len(basis) > size:
        basis = basis[:size]

    while len(basis) < size:
        s_b = set(basis)
        l1 = eval_l1(basis, limit)
        if not l1:
            break
        # Half-decade bins across [0, log10(limit)]
        bins = collections.Counter(int(math.log10(x) * 2.0) for x in basis)
        cands = list(l1)
        if len(cands) > 90:
            cands = rng.sample(cands, 90)
        best_v = None
        best_score = -1e18
        for v in cands:
            bk = int(math.log10(v) * 2.0)
            new_l1 = 0
            for b in basis:
                s = v + b
                if s <= limit and s not in s_b and s not in l1:
                    new_l1 += 1
                p = v * b
                if p <= limit and p not in s_b and p not in l1:
                    new_l1 += 1
                d = abs(v - b)
                if d >= 1 and d not in s_b and d not in l1:
                    new_l1 += 1
            score = new_l1 - 8.0 * bins[bk] + 0.01 * rng.random()
            if score > best_score:
                best_score = score
                best_v = v
        basis.append(best_v)
    return basis


def analyze_dataset(name, puzzles):
    print(f"=== {name} (n={len(puzzles)}) ===")
    exact_buckets = collections.Counter()
    by_dec = collections.defaultdict(list)
    for p in puzzles:
        d_min = decade_of(min(p.targets))
        d_max = decade_of(max(p.targets))
        exact_buckets[(d_min, d_max)] += 1
        by_dec[d_max].append(p)
    print("  (minT_digits, maxT_digits) buckets:", sorted(exact_buckets.items()))

    lrs = [p.limit / max(p.targets) for p in puzzles]
    pcts = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    pvals = ", ".join(f"p{int(q*100):02d}={quantile(lrs, q):.3f}" for q in pcts)
    print(f"  limit/maxT percentiles: {pvals}")
    for thr in (1.5, 2.0, 3.0):
        cnt = sum(1 for x in lrs if x >= thr)
        print(f"  limit/maxT >= {thr:.1f}: {cnt:3d}/{len(lrs)} ({100.0 * cnt / len(lrs):5.1f}%)")

    all_t_ratios = [p.limit / t for p in puzzles for t in p.targets]
    for thr in (1.5, 2.0, 3.0):
        cnt = sum(1 for x in all_t_ratios if x >= thr)
        print(f"  per-target limit/t >= {thr:.1f}: {cnt:4d}/{len(all_t_ratios)} ({100.0 * cnt / len(all_t_ratios):5.1f}%)")

    print("  Per-decade breakdown:")
    for d in (6, 9, 12, 15, 18):
        sub = by_dec.get(d, [])
        if not sub:
            continue
        slrs = [p.limit / max(p.targets) for p in sub]
        ts = [p.limit / t for p in sub for t in p.targets]
        c15 = sum(1 for x in slrs if x >= 1.5)
        c20 = sum(1 for x in slrs if x >= 2.0)
        c30 = sum(1 for x in slrs if x >= 3.0)
        tc2 = sum(1 for x in ts if x >= 2.0)
        tc3 = sum(1 for x in ts if x >= 3.0)
        ns = f"{min(len(p.start) for p in sub)}..{max(len(p.start) for p in sub)}"
        nt = f"{min(len(p.targets) for p in sub)}..{max(len(p.targets) for p in sub)}"
        print(f"    10^{d:2d}: n={len(sub):3d} |sv|={ns:4s} |T|={nt:6s} "
              f"lim/maxT=[{min(slrs):.3f}..{max(slrs):.3f}] p50={quantile(slrs, 0.5):.3f} "
              f"(>=1.5:{c15:2d}, >=2.0:{c20:2d}, >=3.0:{c30:2d}) | "
              f"per-target lim/t>=2:{100*tc2/len(ts):4.1f}% >=3:{100*tc3/len(ts):4.1f}%")
    print()


def simulate_basis_coverage(puzzles):
    print("=== Synthetic Multi-Scale Basis Simulation (|B| in {34, 36, 38}) ===")
    by_dec = collections.defaultdict(list)
    for p in puzzles:
        by_dec[decade_of(max(p.targets))].append(p)

    # Compare powers-of-2 baseline vs constructive multi-scale basis across decades
    for d in (6, 9, 12, 15, 18):
        sub = by_dec.get(d, [])
        if not sub:
            continue
        # Sample up to 3 puzzles per decade (including one with high limit/maxT if available)
        sub_sorted = sorted(sub, key=lambda p: p.limit / max(p.targets))
        picks = [sub_sorted[0]]
        if len(sub_sorted) >= 3:
            picks = [sub_sorted[0], sub_sorted[len(sub_sorted) // 2], sub_sorted[-1]]
        elif len(sub_sorted) == 2:
            picks = sub_sorted

        # Powers-of-2 reference at size 36
        rep = picks[-1]
        pow2 = [1 << k for k in range(36) if (1 << k) <= rep.limit]
        pow2_l1 = eval_l1(pow2, rep.limit)
        print(f"  Decade 10^{d:2d} (pow2 ref: |B|={len(pow2)}, |L1|={len(pow2_l1)}):")

        for sz in (34, 36, 38):
            l1_sizes = []
            dec_b_cov = []
            dec_l1_cov = []
            top_l1_cnt = []
            sqrt_l1_cnt = []
            med_res_d1 = []
            med_res_td = []
            hit_2op_d1 = 0
            hit_2op_td = 0
            total_targets = 0

            half_d = (d + 1) // 2
            for idx, pz in enumerate(picks):
                basis = build_multiscale_basis(pz.start, sz, pz.limit, seed=100 + idx + sz)
                l1 = eval_l1(basis, pz.limit)
                M = sorted(set(basis) | l1)
                sM = set(M)
                l1_sizes.append(len(l1))

                db = set(decade_of(x) for x in basis)
                dl = set(decade_of(x) for x in l1)
                dec_b_cov.append(sum(1 for k in range(1, d + 1) if k in db))
                dec_l1_cov.append(sum(1 for k in range(1, d + 1) if k in dl))
                top_l1_cnt.append(sum(1 for x in l1 if decade_of(x) == d))
                sqrt_l1_cnt.append(sum(1 for x in l1 if half_d - 1 <= decade_of(x) <= half_d + 1))

                # Measure best residual |t*d_mult - a*q| with a, q, d_mult in M
                for t in pz.targets:
                    total_targets += 1
                    best_r1 = t
                    for a in M:
                        if a * a > t:
                            break
                        if a < 2:
                            continue
                        q_est = t // a
                        pos = bisect.bisect_left(M, q_est)
                        for k in (pos - 1, pos):
                            if 0 <= k < len(M):
                                prod = a * M[k]
                                if prod <= pz.limit:
                                    r = abs(t - prod)
                                    if r < best_r1:
                                        best_r1 = r
                    med_res_d1.append(best_r1)
                    if best_r1 == 0 or best_r1 in sM:
                        hit_2op_d1 += 1

                    best_r_td = best_r1
                    max_d = pz.limit // t
                    for dm in M:
                        if dm < 2:
                            continue
                        if dm > max_d:
                            break
                        td = t * dm
                        for a in M:
                            if a * a > td:
                                break
                            if a < 2:
                                continue
                            q_est = td // a
                            pos = bisect.bisect_left(M, q_est)
                            for k in (pos - 1, pos):
                                if 0 <= k < len(M):
                                    prod = a * M[k]
                                    if prod <= pz.limit:
                                        r = abs(td - prod)
                                        if r < best_r_td:
                                            best_r_td = r
                    med_res_td.append(best_r_td)
                    if best_r_td == 0 or best_r_td in sM:
                        hit_2op_td += 1

            avg_l1 = sum(l1_sizes) / len(l1_sizes)
            avg_db = sum(dec_b_cov) / len(dec_b_cov)
            avg_dl = sum(dec_l1_cov) / len(dec_l1_cov)
            avg_top = sum(top_l1_cnt) / len(top_l1_cnt)
            avg_sqrt = sum(sqrt_l1_cnt) / len(sqrt_l1_cnt)
            r1_log = math.log10(max(1.0, quantile(med_res_d1, 0.5)))
            rtd_log = math.log10(max(1.0, quantile(med_res_td, 0.5)))
            print(f"    |B|={sz}: mean|L1|={avg_l1:6.1f}  dec_B={avg_db:4.1f}/{d}  dec_L1={avg_dl:4.1f}/{d}  "
                  f"L1_top={avg_top:5.1f}  L1_sqrt={avg_sqrt:5.1f}  "
                  f"med_res(d=1)=10^{r1_log:4.2f}  med_res(t*d)=10^{rtd_log:4.2f}  "
                  f"3op_hit={hit_2op_d1}/{total_targets}->{hit_2op_td}/{total_targets}")
    print()


def main():
    ind = independent()
    sc = scored()
    sc_ind = [p for p in sc if max(p.start) > 40]
    sc_other = [p for p in sc if max(p.start) <= 40]
    print(f"Practice independent: {len(ind)} | max(sv) range: {min(max(p.start) for p in ind)}..{max(max(p.start) for p in ind)}")
    print(f"Scored max(sv) > 40:  {len(sc_ind)} | max(sv) range: {min(max(p.start) for p in sc_ind)}..{max(max(p.start) for p in sc_ind)}")
    print(f"Scored max(sv) <= 40: {len(sc_other)} | max(sv) range: {min(max(p.start) for p in sc_other)}..{max(max(p.start) for p in sc_other)}")
    print()

    analyze_dataset("practice independent", ind)
    analyze_dataset("scored independent-like (max(sv) > 40)", sc_ind)
    simulate_basis_coverage(sc_ind)


if __name__ == "__main__":
    main()
