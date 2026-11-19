"""Structural analysis of the `shared` family (practice set, n=51).

Run from the repo root:
    python3 analysis/shared/01_structure.py
"""
import collections
from fractions import Fraction
import math

from common import load, quantiles


def find_consensus_hubs(targets, max_q=512, max_a=16, r_cap=20000, rel_div=500):
    """Detect consensus hidden hubs H where >=2 targets satisfy t = q*H + r.

    Enforces `|r| <= min(r_cap, max(10, H // rel_div))` and `H > 100` so small divisors
    cannot artificially wrap large targets. Returns greedy set-cover list of
    `(H, [(target_idx, q, r), ...])`.
    """
    ts = sorted(targets)
    n = len(ts)
    cands = set()
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(ts[i], ts[j])
            if g > 100:
                cands.add(g)
            f = Fraction(ts[j], ts[i]).limit_denominator(max_a)
            b, a = f.numerator, f.denominator
            if 1 <= a <= max_a and 1 <= b <= max_q:
                h_est = round(ts[i] / a)
                if h_est > 100:
                    tol = min(r_cap, max(10, h_est // rel_div))
                    if abs(ts[i] - a * h_est) <= tol and abs(ts[j] - b * h_est) <= tol:
                        cands.add(h_est)
            q = round(ts[j] / ts[i])
            if 1 <= q <= max_q and ts[i] > 100:
                if abs(ts[j] - q * ts[i]) <= min(r_cap, max(10, ts[i] // rel_div)):
                    cands.add(ts[i])

    cands = sorted(cands)
    clusters = []
    for h in cands:
        if not clusters or h - clusters[-1][-1] > 2 * min(r_cap, max(10, h // rel_div)):
            clusters.append([h])
        else:
            clusters[-1].append(h)

    raw_hubs = []
    for cl in clusters:
        h0 = cl[len(cl) // 2]
        tol0 = min(r_cap, max(10, h0 // rel_div))
        test_h = set(cl)
        for t in ts:
            q = round(t / h0)
            if 1 <= q <= max_q and abs(t - q * h0) <= 2 * tol0:
                test_h.add(round(t / q))
                if t % q == 0:
                    test_h.add(t // q)
        best = None
        for h in test_h:
            if h <= 100:
                continue
            tol = min(r_cap, max(10, h // rel_div))
            mem = []
            for idx, t in enumerate(ts):
                q = round(t / h)
                if 1 <= q <= max_q:
                    r = t - q * h
                    if abs(r) <= tol:
                        mem.append((idx, q, r))
            if len(mem) >= 2:
                exact_cnt = sum(1 for _, _, r in mem if r == 0)
                score = (len(mem), exact_cnt, -sum(abs(m[2]) for m in mem), h)
                if best is None or score > best[0]:
                    best = (score, h, mem)
        if best is not None:
            raw_hubs.append((best[1], best[2]))

    # Greedy set-cover deduplication: keep hubs covering at least one new target
    raw_hubs.sort(key=lambda x: (-len(x[1]), -sum(1 for m in x[1] if m[2] == 0), sum(abs(m[2]) for m in x[1]), -x[0]))
    kept = []
    covered = set()
    for h, mem in raw_hubs:
        idxs = {m[0] for m in mem}
        if not (idxs - covered):
            continue
        kept.append((h, mem))
        covered |= idxs
    return kept


def analyze_links(puzzles, label):
    tot_t = 0
    diff_spec, diff_5k = 0, 0
    aff_spec, aff_5k = 0, 0
    gcd_100, gcd_1000 = 0, 0
    union_spec, union_strict = 0, 0
    comp_sizes = []
    spec_qs, spec_rs = [], []
    strict_qs, strict_rs = [], []

    for p in puzzles:
        ts = sorted(p["targets"])
        n = len(ts)
        tot_t += n
        sd_spec, sd_5k = set(), set()
        sa_spec, sa_5k = set(), set()
        sg_100, sg_1000 = set(), set()
        adj = collections.defaultdict(set)

        for i in range(n):
            for j in range(i + 1, n):
                tol_spec = max(5000, ts[i] // 1000)
                tol_strict = min(5000, ts[i] // 20)
                d_ij = ts[j] - ts[i]
                if d_ij <= tol_spec:
                    sd_spec.add(i)
                    sd_spec.add(j)
                if d_ij <= tol_strict:
                    sd_5k.add(i)
                    sd_5k.add(j)

                q = round(ts[j] / ts[i])
                if 1 <= q <= 512:
                    r = abs(ts[j] - q * ts[i])
                    if r <= tol_spec:
                        sa_spec.add(i)
                        sa_spec.add(j)
                        spec_qs.append(q)
                        spec_rs.append(r)
                        adj[i].add(j)
                        adj[j].add(i)
                    if r <= tol_strict:
                        sa_5k.add(i)
                        sa_5k.add(j)
                        strict_qs.append(q)
                        strict_rs.append(r)

                g = math.gcd(ts[i], ts[j])
                if g > 100:
                    sg_100.add(i)
                    sg_100.add(j)
                    adj[i].add(j)
                    adj[j].add(i)
                if g > 1000:
                    sg_1000.add(i)
                    sg_1000.add(j)

        diff_spec += len(sd_spec)
        diff_5k += len(sd_5k)
        aff_spec += len(sa_spec)
        aff_5k += len(sa_5k)
        gcd_100 += len(sg_100)
        gcd_1000 += len(sg_1000)
        union_spec += len(sa_spec | sg_100)
        union_strict += len(sa_5k | sg_1000)

        # Connected components under pairwise affine + gcd>100
        seen = set()
        for i in range(n):
            if i in seen or i not in adj:
                continue
            stack = [i]
            seen.add(i)
            sz = 0
            while stack:
                u = stack.pop()
                sz += 1
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            comp_sizes.append(sz)

    print(f"=== Pairwise Link Coverage: {label} (puzzles={len(puzzles)}, targets={tot_t}) ===")
    print(f"  diff |t_j-t_i| <= max(5000, t_i//1000): {diff_spec:4d}/{tot_t} ({100*diff_spec/tot_t:5.1f}%) | "
          f"strict <= min(5000, t_i//20): {diff_5k:4d}/{tot_t} ({100*diff_5k/tot_t:5.1f}%)")
    print(f"  affine 1<=q<=512, |r|<=spec:            {aff_spec:4d}/{tot_t} ({100*aff_spec/tot_t:5.1f}%) | "
          f"strict <= min(5000, t_i//20): {aff_5k:4d}/{tot_t} ({100*aff_5k/tot_t:5.1f}%)")
    print(f"  gcd(t_i, t_j) > 100:                    {gcd_100:4d}/{tot_t} ({100*gcd_100/tot_t:5.1f}%) | "
          f"gcd > 1000:                   {gcd_1000:4d}/{tot_t} ({100*gcd_1000/tot_t:5.1f}%)")
    print(f"  pairwise union (aff | gcd>100):         {union_spec:4d}/{tot_t} ({100*union_spec/tot_t:5.1f}%) | "
          f"strict union:                 {union_strict:4d}/{tot_t} ({100*union_strict/tot_t:5.1f}%)")
    if comp_sizes:
        print(f"  connected component sizes (n={len(comp_sizes)}): min/q25/med/q75/max = {quantiles(comp_sizes)}")
    if strict_qs:
        print(f"  strict affine q quantiles:   {quantiles(strict_qs)} | q==1: {sum(1 for x in strict_qs if x==1)}/{len(strict_qs)}")
        print(f"  strict affine |r| quantiles: {quantiles(strict_rs)} | |r|<=500: {sum(1 for x in strict_rs if x<=500)}/{len(strict_rs)}")
    print()


def analyze_hubs(shared, indep):
    print("=== Practice_0006 Consensus Hub Sanity Check ===")
    p6 = next(p for p in shared if p["id"] == "practice_0006")
    for h, mem in find_consensus_hubs(p6["targets"], r_cap=5000):
        print(f"  H={h} (deg={len(mem)}): {[(sorted(p6['targets'])[idx], q, r) for idx, q, r in mem]}")
    print()

    for r_cap in (5000, 20000, 100000):
        print(f"=== Consensus Hidden Hubs (r_cap={r_cap}, |r|<=min(r_cap, max(10, H//500)), 1<=q<=512) ===")
        for label, ds in (("shared (51)", shared), ("independent null (45)", indep)):
            tot_t = 0
            cov_hub = 0
            cov_union = 0
            degs, qs, rs = [], [], []
            hubs_per_pz = []
            for p in ds:
                ts = sorted(p["targets"])
                n = len(ts)
                tot_t += n
                hubs = find_consensus_hubs(ts, r_cap=r_cap)
                hubs_per_pz.append(len(hubs))
                s_hub = set()
                for _h, mem in hubs:
                    degs.append(len(mem))
                    for idx, q, r in mem:
                        s_hub.add(idx)
                        qs.append(q)
                        rs.append(abs(r))
                s_pair = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        if math.gcd(ts[i], ts[j]) > 100:
                            s_pair.add(i)
                            s_pair.add(j)
                        q = round(ts[j] / ts[i])
                        if 1 <= q <= 512 and abs(ts[j] - q * ts[i]) <= max(5000, ts[i] // 1000):
                            s_pair.add(i)
                            s_pair.add(j)
                cov_hub += len(s_hub)
                cov_union += len(s_hub | s_pair)
            print(f"  {label:22s}: hub_cov={cov_hub:4d}/{tot_t} ({100*cov_hub/tot_t:5.1f}%) | "
                  f"hub|aff|gcd={cov_union:4d}/{tot_t} ({100*cov_union/tot_t:5.1f}%) | "
                  f"n_hubs={len(degs)} (per puzzle: {quantiles(hubs_per_pz)})")
            if degs and label.startswith("shared"):
                deg_hist = dict(sorted(collections.Counter(degs).items()))
                print(f"    hub degree dist: {deg_hist}")
                print(f"    q quantiles (min/q25/med/q75/max): {quantiles(qs)} | "
                      f"q==1: {100*sum(1 for x in qs if x==1)/len(qs):.1f}% | "
                      f"q<=16: {100*sum(1 for x in qs if x<=16)/len(qs):.1f}% | "
                      f"q<=64: {100*sum(1 for x in qs if x<=64)/len(qs):.1f}%")
                print(f"    |r| quantiles (min/q25/med/q75/max): {quantiles(rs)} | "
                      f"r==0: {100*sum(1 for x in rs if x==0)/len(rs):.1f}% | "
                      f"|r|<=500: {100*sum(1 for x in rs if x<=500)/len(rs):.1f}% | "
                      f"|r|<=5000: {100*sum(1 for x in rs if x<=5000)/len(rs):.1f}%")
        print()


def main():
    all_p = load(practice=True)
    shared = [p for p in all_p if p["family"] == "shared"]
    indep = [p for p in all_p if p["family"] == "independent"]
    analyze_links(shared, "practice shared")
    analyze_links(indep, "practice independent null")
    analyze_hubs(shared, indep)


if __name__ == "__main__":
    main()
