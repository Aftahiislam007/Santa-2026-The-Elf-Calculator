"""Structural analysis of the `multiplicative` family (practice set, n=40).

Run from the repo root:
    python3 analysis/multiplicative/01_structure.py
"""
import collections
import math

from common import apply, factor, practice


_SMALL_PRIMES = [p for p in range(2, 500) if all(p % q for q in range(2, int(p ** 0.5) + 1))]


def quantile(xs, q):
    s = sorted(xs)
    if not s:
        return 0.0
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(len(s) - 1, lo + 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def cheap_values(starts, limit, depth=1):
    """Values reachable from `starts` in <= `depth` operations within [1, limit]."""
    out = set(starts)
    if 1 <= limit:
        out.add(1)  # s_0 / s_0 = 1
    for _ in range(depth):
        cur = sorted(out)
        nxt = set()
        for i in range(len(cur)):
            a = cur[i]
            for j in range(i, len(cur)):
                b = cur[j]
                for op in ("+", "*"):
                    v = apply(op, a, b)
                    if 1 <= v <= limit:
                        nxt.add(v)
                if a != b:
                    nxt.add(abs(a - b))
                if a >= b and a % b == 0:
                    nxt.add(a // b)
                elif b > a and b % a == 0:
                    nxt.add(b // a)
        out |= nxt
    return sorted(out)


def is_smooth(n, bound):
    for p in _SMALL_PRIMES:
        if p > bound:
            break
        while n % p == 0:
            n //= p
    return n == 1


def main():
    ps = practice()
    mult = [p for p in ps if p.family == "multiplicative"]
    print(f"Practice multiplicative puzzles: {len(mult)}")

    tot_t = 0
    raw_pmax = []
    raw_prime_targets = 0
    shift1_pmax = []
    shift1_c_used = []
    shift2_sm100 = 0
    shift2_sm300 = 0

    raw_share_p7 = 0
    raw_share_p13 = 0
    shift1_share_p7 = 0
    shift1_share_p13 = 0
    exposes_new_p7_on_best = 0
    exposes_any_shared_factor = 0
    drop_10x = 0
    drop_100x = 0

    comp_gcd_any = 0
    comp_gcd_gt100 = 0
    comp_gcd_gt1000 = 0
    max_sib_gcd = []
    shift1_new_gcd100 = 0

    raw_shared_p_per_pz = []
    shift1_shared_p_per_pz = []
    cheap1_counts = []
    top_reused_primes = collections.Counter()

    for pz in mult:
        cs1 = cheap_values(pz.start, pz.limit, depth=1)
        cs2 = [c for c in cheap_values(pz.start, pz.limit, depth=2) if c <= 50000]
        cheap1_counts.append(len(cs1))
        ts = pz.targets
        n = len(ts)
        tot_t += n

        raw_facs = [factor(t) for t in ts]
        raw_pcnt = collections.Counter()
        for f in raw_facs:
            for pr in f:
                if pr >= 7:
                    raw_pcnt[pr] += 1
        raw_shared_p_per_pz.append(sum(1 for pr, c in raw_pcnt.items() if c >= 2))
        for pr, c in raw_pcnt.items():
            if c >= 2:
                top_reused_primes[pr] += c

        # Best 1-step shift minimizing P_max(t +- c)
        best_facs = []
        best_vals = []
        for i, t in enumerate(ts):
            f0 = raw_facs[i]
            pm0 = max(f0) if f0 else 1
            raw_pmax.append(pm0)
            if pm0 == t and t > 1:
                raw_prime_targets += 1

            if any(raw_pcnt[pr] >= 2 for pr in f0 if pr >= 7):
                raw_share_p7 += 1
            if any(raw_pcnt[pr] >= 2 for pr in f0 if pr >= 13):
                raw_share_p13 += 1

            best_pm = pm0
            best_f = f0
            best_v = t
            best_c = 0
            for c in cs1:
                for v in (t + c, t - c):
                    if v < 1 or v > pz.limit:
                        continue
                    fv = factor(v)
                    pm = max(fv) if fv else 1
                    if pm < best_pm or (pm == best_pm and len(fv) < len(best_f)):
                        best_pm = pm
                        best_f = fv
                        best_v = v
                        best_c = c
            best_facs.append(best_f)
            best_vals.append(best_v)
            shift1_pmax.append(best_pm)
            if best_c:
                shift1_c_used.append(best_c)
            if pm0 > 10 * best_pm:
                drop_10x += 1
            if pm0 > 100 * best_pm:
                drop_100x += 1

            # Depth-2 smoothness check (fast trial division)
            if is_smooth(t, 100) or any(
                (t + c <= pz.limit and is_smooth(t + c, 100)) or (t > c and is_smooth(t - c, 100))
                for c in cs2
            ):
                shift2_sm100 += 1
            if is_smooth(t, 300) or any(
                (t + c <= pz.limit and is_smooth(t + c, 300)) or (t > c and is_smooth(t - c, 300))
                for c in cs2
            ):
                shift2_sm300 += 1

            # Composite GCD reuse across raw targets
            best_g = 1
            has_comp = False
            for j in range(n):
                if j == i:
                    continue
                g = math.gcd(t, ts[j])
                if g > best_g:
                    best_g = g
                if sum(factor(g).values()) >= 2:
                    has_comp = True
            max_sib_gcd.append(best_g)
            if has_comp:
                comp_gcd_any += 1
            if best_g > 100:
                comp_gcd_gt100 += 1
            if best_g > 1000:
                comp_gcd_gt1000 += 1

        # Shifted prime reuse & exposed shared factors on best smooth shift
        sh_pcnt = collections.Counter()
        for f in best_facs:
            for pr in f:
                if pr >= 7:
                    sh_pcnt[pr] += 1
        shift1_shared_p_per_pz.append(sum(1 for pr, c in sh_pcnt.items() if c >= 2))

        for i in range(n):
            f0 = raw_facs[i]
            bf = best_facs[i]
            bv = best_vals[i]
            if any(sh_pcnt[pr] >= 2 for pr in bf if pr >= 7):
                shift1_share_p7 += 1
            if any(sh_pcnt[pr] >= 2 for pr in bf if pr >= 13):
                shift1_share_p13 += 1
            # Exposed a shared prime >= 7 not present in raw target (on the best smooth shift!)
            if any(sh_pcnt[pr] >= 2 and pr not in f0 for pr in bf if pr >= 7):
                exposes_new_p7_on_best += 1
            # Exposed new gcd > 100 with a sibling (either raw sibling or shifted sibling) on best shift
            if bv != ts[i]:
                new_g = False
                for j in range(n):
                    if j == i:
                        continue
                    if math.gcd(bv, best_vals[j]) > 100 and math.gcd(ts[i], ts[j]) <= 100:
                        new_g = True
                        break
                if new_g:
                    shift1_new_gcd100 += 1
                if new_g or any(sh_pcnt[pr] >= 2 and pr not in f0 for pr in bf if pr >= 7):
                    exposes_any_shared_factor += 1

    print(f"Total targets: {tot_t} | prime targets: {raw_prime_targets}/{tot_t} ({100*raw_prime_targets/tot_t:.1f}%)")
    print(f"|cheap<=1| per puzzle: min={min(cheap1_counts)}, med={int(quantile(cheap1_counts, 0.5))}, max={max(cheap1_counts)}")
    print()

    print("=== Smoothness Distribution (Largest Prime Factor P_max) ===")
    pcts = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    print("  Raw P_max quantiles:        ", [int(quantile(raw_pmax, q)) for q in pcts])
    print("  1-step shift P_max quant.:  ", [int(quantile(shift1_pmax, q)) for q in pcts])
    for b in (30, 50, 100, 300, 1000):
        rc = sum(1 for x in raw_pmax if x <= b)
        sc = sum(1 for x in shift1_pmax if x <= b)
        print(f"  P_max <= {b:4d}: raw={rc:4d}/{tot_t} ({100*rc/tot_t:5.1f}%) | "
              f"1-step shift={sc:4d}/{tot_t} ({100*sc/tot_t:5.1f}%)")
    print(f"  2-step shift (c<=50k): <=100 smooth={shift2_sm100}/{tot_t} ({100*shift2_sm100/tot_t:.1f}%), "
          f"<=300 smooth={shift2_sm300}/{tot_t} ({100*shift2_sm300/tot_t:.1f}%)")
    print()

    print("=== 1-Step Shift Reduction & Shared Factor Exposure ===")
    print(f"  1-step shift drops P_max > 10x:  {drop_10x:4d}/{tot_t} ({100*drop_10x/tot_t:5.1f}%)")
    print(f"  1-step shift drops P_max > 100x: {drop_100x:4d}/{tot_t} ({100*drop_100x/tot_t:5.1f}%)")
    print(f"  Best smooth shift exposes new shared prime>=7:  {exposes_new_p7_on_best:4d}/{tot_t} ({100*exposes_new_p7_on_best/tot_t:5.1f}%)")
    print(f"  Best smooth shift exposes new sibling gcd>100:  {shift1_new_gcd100:4d}/{tot_t} ({100*shift1_new_gcd100/tot_t:5.1f}%)")
    print(f"  Best smooth shift exposes either (prime|gcd):   {exposes_any_shared_factor:4d}/{tot_t} ({100*exposes_any_shared_factor/tot_t:5.1f}%)")
    if shift1_c_used:
        print(f"  Winning |c| quantiles: {[int(quantile(shift1_c_used, q)) for q in pcts]}")
    print()

    print("=== Prime Reuse (>= 7) & Composite GCD Reuse ===")
    print(f"  Raw targets sharing prime >= 7:    {raw_share_p7:4d}/{tot_t} ({100*raw_share_p7/tot_t:5.1f}%) | "
          f"prime >= 13: {raw_share_p13:4d}/{tot_t} ({100*raw_share_p13/tot_t:5.1f}%)")
    print(f"  Shift-1 targets sharing prime >=7: {shift1_share_p7:4d}/{tot_t} ({100*shift1_share_p7/tot_t:5.1f}%) | "
          f"prime >= 13: {shift1_share_p13:4d}/{tot_t} ({100*shift1_share_p13/tot_t:5.1f}%)")
    print(f"  Distinct shared primes>=7/puzzle:  raw mean={sum(raw_shared_p_per_pz)/len(mult):.1f} "
          f"({min(raw_shared_p_per_pz)}..{max(raw_shared_p_per_pz)}) | "
          f"shift-1 mean={sum(shift1_shared_p_per_pz)/len(mult):.1f} "
          f"({min(shift1_shared_p_per_pz)}..{max(shift1_shared_p_per_pz)})")
    print(f"  Top reused raw primes >= 7 (target-occurrences): {top_reused_primes.most_common(12)}")
    print(f"  Targets sharing composite GCD with sibling: {comp_gcd_any:4d}/{tot_t} ({100*comp_gcd_any/tot_t:5.1f}%)")
    print(f"  Targets with sibling GCD > 100:             {comp_gcd_gt100:4d}/{tot_t} ({100*comp_gcd_gt100/tot_t:5.1f}%) | "
          f"> 1000: {comp_gcd_gt1000:4d}/{tot_t} ({100*comp_gcd_gt1000/tot_t:5.1f}%)")
    print(f"  Max sibling GCD quantiles: {[int(quantile(max_sib_gcd, q)) for q in pcts]}")


if __name__ == "__main__":
    main()
