// Factor-Graph & Shifted-Smooth Synthesis Engine for multiplicative puzzles.
//
// Exploits the structure of multiplicative puzzles via:
//   1. Fast shifted-target smooth factorization mining (raw + cheap<=1 + cheap<=2),
//      precomputing balanced divisor sets for up to 12 diverse smooth forms per target.
//   2. Global coordinate-descent form selection with pairwise GCD sharing.
//   3. Incremental L1 maintenance + LRU register management (with optional gentle
//      protection for active shared composite GCDs / multi-target divisors).
//   4. Multi-form live shifted-smooth exact-divisor synthesis at B=3, 4, 5, 6:
//        - B=3: S,c in L1; c in P & find2(S); S in P & find2(c)
//        - B=4: c in L1 & find2(S); c in P/0 & S = d*q (d,q in L1 or d in P, q in L2)
//        - B=5: c in L1 & S = d*q (d,q in L1); c in P/0 & S = d*q (d in L1, q in L2)
//        - B=6: c in P/0 & S = d*q (d in L2 AND q in L2, zero remainder for <=10^18!)
//   5. Unified L1 Re-Pair factor-multiset compression + Descending multi-scale
//      target ordering + GCD-first balanced smooth divisor splitting + full
//      enhanced decomp/construct fallback (bmax=8, 100% solve guarantee).
//
// I/O protocol: see elfcalc/solvers/engine.py.

#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using elf::FlatMap;
using elf::Puzzle;
using elf::Step;
using elf::Timer;
using elf::u128;
using elf::u64;

struct Options {
    u64 seed = 0;
    int restarts = 4;
    double time_limit = 4.0;
    int regs = 48;
    bool verbose = false;
};

// Ultra-fast open-addressed u64 -> u32 table (1-multiply Fibonacci hash) for hot inner loops.
struct FastTable {
    int bits = 12;
    size_t count = 0;
    std::vector<u64> keys;
    std::vector<uint32_t> vals;

    explicit FastTable(int init_bits = 12)
        : bits(init_bits),
          keys(size_t(1) << init_bits, 0),
          vals(size_t(1) << init_bits, 0) {}

    void clear() {
        std::fill(keys.begin(), keys.end(), 0);
        count = 0;
    }

    size_t slot(u64 k) const { return (k * 0x9E3779B97F4A7C15ULL) >> (64 - bits); }

    const uint32_t* find(u64 k) const {
        if (k == 0) return nullptr;
        size_t mask = keys.size() - 1;
        for (size_t i = slot(k);; i = (i + 1) & mask) {
            if (keys[i] == k) return &vals[i];
            if (!keys[i]) return nullptr;
        }
    }

    bool contains(u64 k) const { return find(k) != nullptr; }

    bool insert(u64 k, uint32_t v) {
        if (k == 0) return false;
        if ((count + 1) * 2 > keys.size()) grow();
        size_t mask = keys.size() - 1;
        for (size_t i = slot(k);; i = (i + 1) & mask) {
            if (keys[i] == k) return false;
            if (!keys[i]) {
                keys[i] = k;
                vals[i] = v;
                ++count;
                return true;
            }
        }
    }

    void grow() {
        std::vector<u64> ok;
        std::vector<uint32_t> ov;
        ok.swap(keys);
        ov.swap(vals);
        ++bits;
        keys.assign(size_t(1) << bits, 0);
        vals.assign(size_t(1) << bits, 0);
        count = 0;
        for (size_t i = 0; i < ok.size(); ++i) {
            if (ok[i]) insert(ok[i], ov[i]);
        }
    }
};

static inline int scale_bucket(u64 v) {
    if (v <= 1) return 0;
    int b = static_cast<int>(std::log2(static_cast<double>(v)) / 2.5);
    if (b < 0) b = 0;
    if (b > 31) b = 31;
    return b;
}

static inline bool is_smooth_300(u64 n) {
    static constexpr u64 kPrimes[] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
        127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
        199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
        283, 293};
    if (n <= 1) return true;
    for (u64 p : kPrimes) {
        if (p * p > n) break;
        while (n % p == 0) n /= p;
    }
    return n <= 300;
}

static int add_chain_len(int e) {
    static constexpr int kChain[65] = {
        0, 0, 1, 2, 2, 3, 3, 4, 3, 4, 4, 5, 4, 5, 5, 5,
        4, 5, 5, 6, 5, 6, 6, 6, 5, 6, 6, 6, 6, 7, 6, 7,
        5, 6, 6, 7, 6, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 8,
        6, 7, 7, 7, 7, 8, 7, 8, 7, 8, 8, 8, 7, 8, 8, 8, 6
    };
    if (e <= 64) return kChain[e];
    int bits = 0, pop = 0, x = e;
    while (x > 0) {
        pop += (x & 1);
        x >>= 1;
        ++bits;
    }
    return (bits - 1) + (pop - 1);
}

static std::vector<u64> sieve_primes(u64 limit) {
    std::vector<uint8_t> is_comp(limit + 1, 0);
    std::vector<u64> primes;
    for (u64 i = 2; i <= limit; ++i) {
        if (!is_comp[i]) {
            primes.push_back(i);
            if (i * i <= limit) {
                for (u64 j = i * i; j <= limit; j += i) is_comp[j] = 1;
            }
        }
    }
    return primes;
}

static const std::vector<u64>& primes_1000() {
    static const std::vector<u64> kPrimes = sieve_primes(1000);
    return kPrimes;
}

static const std::vector<u64>& primes_300() {
    static const std::vector<u64> kPrimes = sieve_primes(300);
    return kPrimes;
}

struct CheapEntry {
    u64 val = 0;
    uint8_t depth = 0;
    uint8_t step_cost = 0;
    Step step{};
};

struct CheapTable {
    FlatMap<CheapEntry> map;
    std::vector<u64> cheap1;
    std::vector<u64> cheap2_only;

    CheapTable() : map(2048) {}

    void build(const std::vector<u64>& starts, u64 limit) {
        map.clear();
        cheap1.clear();
        cheap2_only.clear();

        std::vector<u64> d0;
        for (u64 s : starts) {
            if (s == 0 || s > limit) continue;
            if (map.insert(s, CheapEntry{s, 0, 0, Step{}})) {
                d0.push_back(s);
                cheap1.push_back(s);
            }
        }

        std::vector<u64> d1;
        auto add_d1 = [&](char op, u64 a, u64 b, u64 r) {
            if (r == 0 || r > limit) return;
            if (map.insert(r, CheapEntry{r, 1, 1, Step{op, a, b, r}})) {
                d1.push_back(r);
                cheap1.push_back(r);
            }
        };
        for (size_t i = 0; i < d0.size(); ++i) {
            u64 a = d0[i];
            add_d1('/', a, a, 1);
            for (size_t j = i; j < d0.size(); ++j) {
                u64 b = d0[j];
                if (a <= limit - b) add_d1('+', a, b, a + b);
                u64 p = elf::safe_mul(a, b, limit);
                if (p) add_d1('*', a, b, p);
                if (a > b) {
                    add_d1('-', a, b, a - b);
                    if (a % b == 0) add_d1('/', a, b, a / b);
                } else if (b > a) {
                    add_d1('-', b, a, b - a);
                    if (b % a == 0) add_d1('/', b, a, b / a);
                }
            }
        }
        std::sort(cheap1.begin(), cheap1.end());

        constexpr u64 kMaxC2 = 50000;
        std::vector<std::pair<uint8_t, u64>> c2_list;
        auto add_d2 = [&](char op, u64 a, u64 b, u64 r, uint8_t depth, uint8_t scost) {
            if (r == 0 || r > limit || r > kMaxC2) return;
            if (map.insert(r, CheapEntry{r, depth, scost, Step{op, a, b, r}})) {
                c2_list.emplace_back(scost, r);
            }
        };
        for (u64 a : d1) {
            for (u64 b : d0) {
                if (a <= limit - b) add_d2('+', a, b, a + b, 2, 2);
                u64 p = elf::safe_mul(a, b, limit);
                if (p) add_d2('*', a, b, p, 2, 2);
                if (a > b) {
                    add_d2('-', a, b, a - b, 2, 2);
                    if (a % b == 0) add_d2('/', a, b, a / b, 2, 2);
                } else if (b > a) {
                    add_d2('-', b, a, b - a, 2, 2);
                    if (b % a == 0) add_d2('/', b, a, b / a, 2, 2);
                }
            }
        }
        for (size_t i = 0; i < d1.size(); ++i) {
            u64 a = d1[i];
            for (size_t j = i; j < d1.size(); ++j) {
                u64 b = d1[j];
                uint8_t scost = (a == b) ? 2 : 3;
                if (a <= limit - b) add_d2('+', a, b, a + b, 3, scost);
                u64 p = elf::safe_mul(a, b, limit);
                if (p) add_d2('*', a, b, p, 3, scost);
                if (a > b) {
                    add_d2('-', a, b, a - b, 3, scost);
                    if (a % b == 0) add_d2('/', a, b, a / b, 3, scost);
                } else if (b > a) {
                    add_d2('-', b, a, b - a, 3, scost);
                    if (b % a == 0) add_d2('/', b, a, b / a, 3, scost);
                }
            }
        }
        std::sort(c2_list.begin(), c2_list.end());
        cheap2_only.reserve(c2_list.size());
        for (const auto& kv : c2_list) cheap2_only.push_back(kv.second);
    }

    int step_cost_of(u64 v) const {
        if (const CheapEntry* e = map.find(v)) return e->step_cost;
        return -1;
    }

    double approx_prime_cost(u64 p) const {
        if (const CheapEntry* e = map.find(p)) return static_cast<double>(e->step_cost);
        if (p <= 2) return 1.0;
        double lg = std::log2(static_cast<double>(p));
        return std::max(1.5, 1.15 * lg - 1.0);
    }
};

struct ShiftedForm {
    u64 S = 0;
    u64 c = 0;
    int8_t sign = 0;  // t = S + sign * c
    uint8_t c_cost = 0;
    u64 pmax = 1;
    u64 cofactor = 1;
    int total_exp = 0;
    double base_score = 0.0;
    std::vector<std::pair<u64, int>> factors;
    std::vector<std::pair<u64, u64>> half_divs;  // (d, S/d) with 1 < d <= sqrt(S), balanced first
};

static void populate_half_divs(ShiftedForm& f) {
    f.half_divs.clear();
    if (f.factors.empty() || f.S <= 3) return;
    // Generate up to 1024 divisors of S.
    auto all_d = elf::get_divisors(f.factors, 1024);
    u64 root = elf::isqrt64(f.S);
    f.half_divs.reserve(all_d.size() / 2 + 1);
    for (u64 d : all_d) {
        if (d > 1 && d <= root) f.half_divs.emplace_back(d, f.S / d);
    }
    // Sort descending (closest to sqrt(S) first -> balanced splits checked first!).
    std::reverse(f.half_divs.begin(), f.half_divs.end());
    if (f.half_divs.size() > 192) f.half_divs.resize(192);
}

static void trial_factor_into(u64 n, const std::vector<u64>& plist,
                              std::vector<std::pair<u64, int>>& out, u64& rem) {
    out.clear();
    rem = n;
    for (u64 p : plist) {
        if (p * p > rem) break;
        if (rem % p == 0) {
            int e = 0;
            while (rem % p == 0) {
                rem /= p;
                ++e;
            }
            out.emplace_back(p, e);
        }
    }
    if (rem > 1 && rem <= plist.back() * plist.back()) {
        out.emplace_back(rem, 1);
        rem = 1;
    }
}

static void complete_factor(u64 rem, std::vector<std::pair<u64, int>>& out) {
    if (rem <= 1) return;
    auto extra = elf::factorize(rem);
    for (const auto& kv : extra) out.push_back(kv);
    std::sort(out.begin(), out.end());
    size_t w = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        if (w > 0 && out[w - 1].first == out[i].first) {
            out[w - 1].second += out[i].second;
        } else {
            out[w++] = out[i];
        }
    }
    out.resize(w);
}

static double compute_base_form_score(const ShiftedForm& f, const CheapTable& cheap) {
    double sc = (f.c == 0) ? 0.0 : (1.0 + 0.75 * static_cast<double>(f.c_cost));
    if (f.cofactor > 1) {
        sc += 1.8 * std::log2(static_cast<double>(f.cofactor)) + 6.0;
    }
    int mul_atoms = 0;
    for (const auto& kv : f.factors) {
        u64 p = kv.first;
        int e = kv.second;
        sc += 0.85 * cheap.approx_prime_cost(p);
        sc += static_cast<double>(add_chain_len(e));
        ++mul_atoms;
    }
    if (mul_atoms > 1) sc += static_cast<double>(mul_atoms - 1);
    if (f.pmax > 300) sc += 0.6 * std::log2(static_cast<double>(f.pmax) / 300.0);
    return sc;
}

static std::vector<std::vector<ShiftedForm>> mine_shifted_forms(
    const Puzzle& pz, const std::vector<u64>& miss, const CheapTable& cheap,
    const Timer& timer, double mine_deadline) {

    const auto& p1000 = primes_1000();
    const auto& p300 = primes_300();
    const size_t n = miss.size();
    std::vector<std::vector<ShiftedForm>> all_forms(n);
    std::vector<std::pair<u64, int>> tmp_fac;

    for (size_t ti = 0; ti < n; ++ti) {
        u64 t = miss[ti];
        std::vector<ShiftedForm> cands;
        cands.reserve(64);

        // 1. Raw target (c = 0).
        {
            ShiftedForm f;
            f.S = t;
            f.c = 0;
            f.sign = 0;
            f.c_cost = 0;
            u64 rem = 1;
            trial_factor_into(t, p1000, f.factors, rem);
            if (rem > 1) complete_factor(rem, f.factors);
            f.cofactor = 1;
            f.pmax = f.factors.empty() ? 1 : f.factors.back().first;
            for (const auto& kv : f.factors) f.total_exp += kv.second;
            f.base_score = compute_base_form_score(f, cheap);
            cands.push_back(std::move(f));
        }

        // 2. Scan all cheap1 shifts.
        struct RawCand {
            u64 S, c;
            int8_t sign;
            uint8_t c_cost;
            u64 rem;
            std::vector<std::pair<u64, int>> fac;
        };
        std::vector<RawCand> pending_rho;
        u64 best_c1_pmax = cands[0].pmax;

        for (u64 c : cheap.cheap1) {
            int cc = cheap.step_cost_of(c);
            uint8_t c_cost = cc >= 0 ? static_cast<uint8_t>(cc) : 1;
            for (int dir = 0; dir < 2; ++dir) {
                u64 S = 0;
                int8_t sgn = 0;
                if (dir == 0) {
                    if (t <= c) continue;
                    S = t - c;
                    sgn = +1;
                } else {
                    if (t > pz.limit - c) continue;
                    S = t + c;
                    sgn = -1;
                }
                u64 rem = 1;
                trial_factor_into(S, p1000, tmp_fac, rem);
                if (rem == 1) {
                    ShiftedForm f;
                    f.S = S;
                    f.c = c;
                    f.sign = sgn;
                    f.c_cost = c_cost;
                    f.factors = tmp_fac;
                    f.cofactor = 1;
                    f.pmax = f.factors.empty() ? 1 : f.factors.back().first;
                    for (const auto& kv : f.factors) f.total_exp += kv.second;
                    f.base_score = compute_base_form_score(f, cheap);
                    if (f.pmax < best_c1_pmax) best_c1_pmax = f.pmax;
                    cands.push_back(std::move(f));
                } else {
                    pending_rho.push_back(RawCand{S, c, sgn, c_cost, rem, tmp_fac});
                }
            }
        }

        std::sort(pending_rho.begin(), pending_rho.end(),
                  [](const RawCand& a, const RawCand& b) { return a.rem < b.rem; });
        size_t rho_cap = std::min<size_t>(pending_rho.size(), 10);
        for (size_t k = 0; k < rho_cap; ++k) {
            if (timer.elapsed_sec() > mine_deadline) break;
            auto& rc = pending_rho[k];
            if (rc.rem > 10000000000ULL && best_c1_pmax <= 300) continue;
            complete_factor(rc.rem, rc.fac);
            ShiftedForm f;
            f.S = rc.S;
            f.c = rc.c;
            f.sign = rc.sign;
            f.c_cost = rc.c_cost;
            f.factors = std::move(rc.fac);
            f.cofactor = 1;
            f.pmax = f.factors.empty() ? 1 : f.factors.back().first;
            for (const auto& kv : f.factors) f.total_exp += kv.second;
            f.base_score = compute_base_form_score(f, cheap);
            if (f.pmax < best_c1_pmax) best_c1_pmax = f.pmax;
            cands.push_back(std::move(f));
        }

        // 3. Scan cheap2 shifts (fast <= 300 trial division).
        size_t max_c2 = (best_c1_pmax <= 100) ? 250 : 900;
        max_c2 = std::min(max_c2, cheap.cheap2_only.size());
        int c2_hits = 0;
        for (size_t idx = 0; idx < max_c2; ++idx) {
            if ((idx & 63) == 0 && timer.elapsed_sec() > mine_deadline) break;
            u64 c = cheap.cheap2_only[idx];
            int cc = cheap.step_cost_of(c);
            uint8_t c_cost = cc >= 0 ? static_cast<uint8_t>(cc) : 2;
            for (int dir = 0; dir < 2; ++dir) {
                u64 S = 0;
                int8_t sgn = 0;
                if (dir == 0) {
                    if (t <= c) continue;
                    S = t - c;
                    sgn = +1;
                } else {
                    if (t > pz.limit - c) continue;
                    S = t + c;
                    sgn = -1;
                }
                u64 rem = 1;
                trial_factor_into(S, p300, tmp_fac, rem);
                if (rem == 1) {
                    u64 pm = tmp_fac.empty() ? 1 : tmp_fac.back().first;
                    if (pm > 1000) continue;
                    ShiftedForm f;
                    f.S = S;
                    f.c = c;
                    f.sign = sgn;
                    f.c_cost = c_cost;
                    f.factors = tmp_fac;
                    f.cofactor = 1;
                    f.pmax = pm;
                    for (const auto& kv : f.factors) f.total_exp += kv.second;
                    f.base_score = compute_base_form_score(f, cheap);
                    cands.push_back(std::move(f));
                    ++c2_hits;
                    if (pm <= 100 && c2_hits >= 12) break;
                }
            }
            if (c2_hits >= 18) break;
        }

        // Deduplicate by S and keep a diverse top K = 12 set.
        std::vector<ShiftedForm> kept;
        FlatMap<uint8_t> seen_S(64);
        auto try_keep = [&](const ShiftedForm& f) {
            if (kept.size() >= 12) return;
            if (seen_S.insert(f.S, 1)) kept.push_back(f);
        };
        try_keep(cands[0]);

        std::vector<size_t> order(cands.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return cands[a].base_score < cands[b].base_score;
        });
        for (size_t i = 0; i < std::min<size_t>(order.size(), 5); ++i) try_keep(cands[order[i]]);

        int c1_added = 0;
        for (size_t idx : order) {
            if (cands[idx].c_cost <= 1) {
                try_keep(cands[idx]);
                if (++c1_added >= 4) break;
            }
        }

        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (cands[a].pmax != cands[b].pmax) return cands[a].pmax < cands[b].pmax;
            return cands[a].base_score < cands[b].base_score;
        });
        for (size_t i = 0; i < std::min<size_t>(order.size(), 4); ++i) try_keep(cands[order[i]]);

        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return cands[a].base_score < cands[b].base_score;
        });
        for (size_t idx : order) {
            if (kept.size() >= 12) break;
            try_keep(cands[idx]);
        }

        for (auto& f : kept) populate_half_divs(f);
        all_forms[ti] = std::move(kept);
    }

    return all_forms;
}

enum class TargetOrder {
    Ascending,
    Descending,
    HubFirst,
    Shuffled
};

struct StrategyConfig {
    bool allow_cheap2 = true;
    double cheap2_penalty = 0.30;
    double gcd_bonus_weight = 0.75;
    bool use_repair = true;
    double repair_before_B3_thresh = 3.0;
    double repair_before_B4_thresh = 1.9;
    double repair_before_B5_thresh = 1.4;
    bool direct_B3_before_smooth = true;
    TargetOrder order_mode = TargetOrder::Descending;
    bool gentle_gcd_lru_boost = false;
    int evict_mode = 2;  // 0: pure LRU, 1: gentle GCD LRU, 2: multiplicative utility eviction
    bool use_affinity_pick = true;
    double aff_weight = 1.0;
    int bfull = 5;
    double noise = 0.0;
};

static int approx_mul_steps_of_gcd(u64 g, const std::vector<std::pair<u64, int>>& fac_a) {
    if (g <= 1) return 0;
    int muls = 0, distinct = 0;
    u64 rem = g;
    for (const auto& kv : fac_a) {
        u64 p = kv.first;
        if (rem % p == 0) {
            int e = 0;
            while (rem % p == 0) {
                rem /= p;
                ++e;
            }
            muls += add_chain_len(e);
            ++distinct;
        }
    }
    if (distinct > 1) muls += (distinct - 1);
    return muls;
}

static std::vector<size_t> select_forms(
    const Puzzle& pz,
    const std::vector<std::vector<ShiftedForm>>& forms,
    const CheapTable& cheap,
    const StrategyConfig& cfg,
    std::mt19937_64& rng) {

    const size_t n = forms.size();
    std::vector<size_t> chosen(n, 0);

    FlatMap<uint8_t> in_starts(pz.starts.size() * 2 + 8);
    for (u64 s : pz.starts) in_starts.insert(s, 1);

    auto allowed = [&](const ShiftedForm& f, size_t ti) -> bool {
        if (cfg.allow_cheap2 || f.c_cost <= 1) return true;
        for (const auto& alt : forms[ti]) {
            if (alt.c_cost <= 1 && alt.pmax <= 400) return false;
        }
        return true;
    };

    for (size_t i = 0; i < n; ++i) {
        double best = 1e18;
        size_t best_k = 0;
        for (size_t k = 0; k < forms[i].size(); ++k) {
            if (!allowed(forms[i][k], i)) continue;
            double sc = forms[i][k].base_score;
            if (forms[i][k].c_cost >= 2) sc += cfg.cheap2_penalty;
            if (sc < best) {
                best = sc;
                best_k = k;
            }
        }
        chosen[i] = best_k;
    }

    std::uniform_real_distribution<double> noise_dist(-cfg.noise, cfg.noise);
    for (int round = 0; round < 5; ++round) {
        FlatMap<int> prime_cnt(256);
        FlatMap<int> shift_cnt(128);
        for (size_t i = 0; i < n; ++i) {
            const auto& f = forms[i][chosen[i]];
            if (f.c > 0) {
                if (int* p = shift_cnt.find(f.c)) ++(*p);
                else shift_cnt.insert(f.c, 1);
            }
            for (const auto& kv : f.factors) {
                if (int* p = prime_cnt.find(kv.first)) ++(*p);
                else prime_cnt.insert(kv.first, 1);
            }
        }

        bool changed = false;
        for (size_t i = 0; i < n; ++i) {
            const auto& cur_f = forms[i][chosen[i]];
            if (cur_f.c > 0) {
                if (int* p = shift_cnt.find(cur_f.c)) --(*p);
            }
            for (const auto& kv : cur_f.factors) {
                if (int* p = prime_cnt.find(kv.first)) --(*p);
            }

            double best_sc = 1e18;
            size_t best_k = chosen[i];

            for (size_t k = 0; k < forms[i].size(); ++k) {
                const auto& f = forms[i][k];
                if (!allowed(f, i)) continue;

                double sc = 0.0;
                if (f.c > 0) {
                    const int* scnt = shift_cnt.find(f.c);
                    bool shared_c = (scnt && *scnt > 0) || in_starts.contains(f.c);
                    sc += shared_c ? 1.0 : (1.0 + 0.9 * static_cast<double>(f.c_cost));
                    if (f.c_cost >= 2) sc += cfg.cheap2_penalty;
                }

                int mul_atoms = 0;
                for (const auto& kv : f.factors) {
                    u64 p = kv.first;
                    int e = kv.second;
                    const int* pcnt = prime_cnt.find(p);
                    bool shared_p = (pcnt && *pcnt > 0);
                    if (!in_starts.contains(p)) {
                        double pc = cheap.approx_prime_cost(p);
                        sc += shared_p ? 0.15 : (1.25 * pc);
                    }
                    sc += static_cast<double>(add_chain_len(e));
                    ++mul_atoms;
                }
                if (mul_atoms > 1) sc += static_cast<double>(mul_atoms - 1);

                double best_gcd_bonus = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    if (j == i) continue;
                    u64 os = forms[j][chosen[j]].S;
                    if (os == f.S) {
                        best_gcd_bonus = std::max(best_gcd_bonus, static_cast<double>(mul_atoms + 2));
                        continue;
                    }
                    u64 g = std::gcd(f.S, os);
                    if (g <= 1) continue;
                    int gm = approx_mul_steps_of_gcd(g, f.factors);
                    double bonus = cfg.gcd_bonus_weight * static_cast<double>(gm);
                    if (f.S % os == 0 || os % f.S == 0) bonus += 1.5;
                    for (const auto& kv : f.factors) {
                        if (kv.first >= 30 && g % kv.first == 0) {
                            bonus += 0.45 * cheap.approx_prime_cost(kv.first);
                        }
                    }
                    if (bonus > best_gcd_bonus) best_gcd_bonus = bonus;
                }
                sc -= best_gcd_bonus;

                if (f.pmax > 25000) {
                    const int* pcnt = prime_cnt.find(f.pmax);
                    if (!pcnt || *pcnt == 0) sc += 10.0;
                }

                if (cfg.noise > 0.0) sc += noise_dist(rng);

                if (sc < best_sc) {
                    best_sc = sc;
                    best_k = k;
                }
            }

            if (best_k != chosen[i]) changed = true;
            chosen[i] = best_k;

            const auto& new_f = forms[i][chosen[i]];
            if (new_f.c > 0) {
                if (int* p = shift_cnt.find(new_f.c)) ++(*p);
                else shift_cnt.insert(new_f.c, 1);
            }
            for (const auto& kv : new_f.factors) {
                if (int* p = prime_cnt.find(kv.first)) ++(*p);
                else prime_cnt.insert(kv.first, 1);
            }
        }
        if (!changed) break;
    }

    return chosen;
}

struct SmoothTarget {
    u64 target = 0;
    u64 S = 0;
    u64 c = 0;
    int8_t sign = 0;
    bool smooth_active = false;
    bool covered = false;
    double hub_score = 0.0;
    std::vector<std::pair<u64, int>> factors;
    const std::vector<ShiftedForm>* all_forms = nullptr;
};

class FactorEngine {
public:
    FactorEngine(const Puzzle& pz, const CheapTable& cheap, int regs,
                 const StrategyConfig& cfg, u64 seed, const Timer& timer,
                 double deadline, size_t best_plan_len = static_cast<size_t>(-1))
        : pz_(pz),
          cheap_(cheap),
          regs_(regs),
          cfg_(cfg),
          rng_(seed),
          timer_(timer),
          deadline_(deadline),
          best_plan_len_(best_plan_len),
          pool_pos_(8),
          pins_(64),
          L1_idx_(13),
          in_starts_(pz.starts.size() * 2 + 8),
          all_targets_(pz.targets.size() * 2 + 8),
          failB_(1024),
          failB_div_(1024),
          ever_spec_built_(256),
          stale_B_(512),
          target_idx_(pz.targets.size() * 2 + 8) {
        for (u64 s : pz.starts) in_starts_.insert(s, 1);
        for (u64 t : pz.targets) all_targets_.insert(t, 1);
    }

    bool check_time() {
        if (ignore_soft_timeout_) return false;
        if (soft_timed_out_) return true;
        if ((++time_check_ctr_ & 63U) == 0) {
            if (timer_.elapsed_sec() >= deadline_) {
                soft_timed_out_ = true;
            }
        }
        return soft_timed_out_;
    }

    bool timed_out() { return check_time(); }

    double rem_time_sec() const {
        double rem = deadline_ - timer_.elapsed_sec();
        return rem > 0.0 ? rem : 0.0;
    }

    double reserve_time_sec() const {
        return std::min(0.26, 0.012 * static_cast<double>(remaining_.size()) + 0.025);
    }

    bool low_on_time() const {
        return soft_timed_out_ || rem_time_sec() <= reserve_time_sec();
    }

    bool should_abort() const {
        return aborted_ || (best_plan_len_ != static_cast<size_t>(-1) &&
                            plan_.size() + remaining_.size() >= best_plan_len_ + 4);
    }

    bool was_aborted() const { return aborted_; }

    std::vector<Step> run(const std::vector<u64>& miss,
                          const std::vector<std::vector<ShiftedForm>>& forms,
                          const std::vector<size_t>& chosen) {
        init_pool(pz_.starts);
        remaining_ = miss;
        targets_.clear();
        targets_.reserve(miss.size());

        FlatMap<int> global_p_cnt(256);
        for (size_t i = 0; i < miss.size(); ++i) {
            const auto& f = forms[i][chosen[i]];
            for (const auto& kv : f.factors) {
                if (int* p = global_p_cnt.find(kv.first)) ++(*p);
                else global_p_cnt.insert(kv.first, 1);
            }
        }

        for (size_t i = 0; i < miss.size(); ++i) {
            const auto& f = forms[i][chosen[i]];
            SmoothTarget st;
            st.target = miss[i];
            st.S = f.S;
            st.c = f.c;
            st.sign = f.sign;
            st.factors = f.factors;
            st.all_forms = &forms[i];
            st.covered = in_pool(st.target);
            bool pmax_shared = false;
            if (const int* pc = global_p_cnt.find(f.pmax)) pmax_shared = (*pc >= 2);
            st.smooth_active = (!st.covered) && (f.cofactor == 1) &&
                               (f.pmax <= 10000 || (pmax_shared && f.pmax <= 1000000));
            target_idx_.insert(st.target, i);
            targets_.push_back(std::move(st));
        }

        for (size_t i = 0; i < targets_.size(); ++i) {
            if (!targets_[i].smooth_active) continue;
            double hs = 0.0;
            for (size_t j = 0; j < targets_.size(); ++j) {
                if (i == j || !targets_[j].smooth_active) continue;
                u64 g = std::gcd(targets_[i].S, targets_[j].S);
                if (g > 100) {
                    hs += std::log2(static_cast<double>(g));
                    if (targets_[j].S % targets_[i].S == 0) hs += 25.0;
                }
            }
            targets_[i].hub_score = hs;
        }

        mark_covered();

        int steps_since_target = 0;
        while (!remaining_.empty()) {
            if (should_abort()) {
                aborted_ = true;
                return {};
            }
            if (!soft_timed_out_ && timer_.elapsed_sec() >= deadline_) {
                soft_timed_out_ = true;
            }
            size_t rem_before = remaining_.size();
            size_t plan_before = plan_.size();

            // 1. Level 1: Immediate L1 target hits!
            if (harvest_L1_targets()) {
                steps_since_target = 0;
                continue;
            }

            // 2. Level 2: L2 target voting across all uncovered targets!
            u64 best_l2_x = 0;
            int best_l2_votes = 0;
            find_best_L2_target_vote(best_l2_x, best_l2_votes);
            if (best_l2_votes >= 1) {
                if (emit_from_L1(best_l2_x)) {
                    steps_since_target += static_cast<int>(plan_.size() - plan_before);
                    if (remaining_.size() < rem_before) steps_since_target = 0;
                    continue;
                }
            }

            // If soft deadline reached, stay in the smart loop using fast B<=5 + smooth/M splits!
            if (soft_timed_out_) {
                if (harvest_shifted_smooth_B3()) {
                    steps_since_target = 0;
                    continue;
                }
                u64 t = pick_fallback_target();
                if (!construct_value(t, 5, 0, false)) {
                    if (should_abort()) {
                        aborted_ = true;
                        return {};
                    }
                    construct_value_fast_split(t);
                }
                steps_since_target = 0;
                continue;
            }

            bool can_speculate = cfg_.use_repair && (steps_since_target < 4) &&
                                 (regs_ == 0 || static_cast<int>(pool_.size()) < regs_ - 8);

            // 3. Very high-leverage Re-Pair step BEFORE B=3!
            if (can_speculate && has_active_smooth()) {
                u64 best_w = 0;
                double best_gain = 0.0;
                int hit_targets = 0;
                find_best_repair_L1(best_w, best_gain, hit_targets);
                if (best_w != 0 && hit_targets >= 2 && best_gain >= cfg_.repair_before_B3_thresh &&
                    !ever_spec_built_.contains(best_w)) {
                    ever_spec_built_.insert(best_w, 1);
                    if (emit_from_L1(best_w)) {
                        steps_since_target += 1;
                        if (remaining_.size() < rem_before) steps_since_target = 0;
                        continue;
                    }
                }
            }

            // 4. Level 3: Direct find3(t) + Shifted-Smooth B=3!
            if (cfg_.direct_B3_before_smooth) {
                if (solve_any_remaining_bounded(3, 3) || harvest_shifted_smooth_B3()) {
                    steps_since_target = 0;
                    continue;
                }
            } else {
                if (harvest_shifted_smooth_B3() || solve_any_remaining_bounded(3, 3)) {
                    steps_since_target = 0;
                    continue;
                }
            }

            // 5. High-leverage Re-Pair step BEFORE B=4!
            if (can_speculate && has_active_smooth()) {
                u64 best_w = 0;
                double best_gain = 0.0;
                int hit_targets = 0;
                find_best_repair_L1(best_w, best_gain, hit_targets);
                if (best_w != 0 && hit_targets >= 2 && best_gain >= cfg_.repair_before_B4_thresh &&
                    !ever_spec_built_.contains(best_w)) {
                    ever_spec_built_.insert(best_w, 1);
                    if (emit_from_L1(best_w)) {
                        steps_since_target += 1;
                        if (remaining_.size() < rem_before) steps_since_target = 0;
                        continue;
                    }
                }
            }

            // 6. Level 4: Shifted-smooth B=4 (via O(|divs|) exact divisor splits!) + enhanced decomp(4)!
            if (harvest_shifted_smooth_B4()) {
                steps_since_target = 0;
                continue;
            }
            if (cfg_.bfull >= 4 && solve_any_remaining_bounded(4, 4)) {
                steps_since_target = 0;
                continue;
            }

            // 7. Re-Pair step BEFORE B=5!
            if (can_speculate && has_active_smooth()) {
                u64 best_w = 0;
                double best_gain = 0.0;
                int hit_targets = 0;
                find_best_repair_L1(best_w, best_gain, hit_targets);
                if (best_w != 0 && hit_targets >= 2 && best_gain >= cfg_.repair_before_B5_thresh &&
                    !ever_spec_built_.contains(best_w)) {
                    ever_spec_built_.insert(best_w, 1);
                    if (emit_from_L1(best_w)) {
                        steps_since_target += 1;
                        if (remaining_.size() < rem_before) steps_since_target = 0;
                        continue;
                    }
                }
            }

            // 8. Level 5: Shifted-smooth B=5 (L1 x L2 exact splits + c in L1 L1 x L1!) + enhanced decomp(5)!
            if (harvest_shifted_smooth_B5()) {
                steps_since_target = 0;
                continue;
            }
            if (!low_on_time() && cfg_.bfull >= 5 && solve_any_remaining_bounded(5, 5)) {
                steps_since_target = 0;
                continue;
            }

            // 9. Level 6 Shifted-Smooth Exact Divisor Split (capped so it runs in a few ms)!
            if (!low_on_time() && harvest_shifted_smooth_B6()) {
                steps_since_target = 0;
                continue;
            }

            // 10. Deep Target Synthesis: handles B=6..8 + live-best smooth split + M-split
            //     in one unified pass (never double-runs B=6..8!).
            if (finish_hard_target()) {
                steps_since_target = 0;
                continue;
            }
            if (should_abort()) {
                aborted_ = true;
                return {};
            }
            u64 t = pick_fallback_target();
            construct_value_fast_split(t);
            steps_since_target = 0;
        }

        return plan_;
    }

private:
    // ---- Fast Incremental Pool + L1 Management ----

    bool in_pool(u64 x) const { return pool_pos_.contains(x); }

    int c01(u64 x) const {
        if (pool_pos_.contains(x)) return 0;
        if (L1_idx_.contains(x)) return 1;
        return 2;
    }

    const Step* find_L1(u64 x) const {
        if (const uint32_t* idx = L1_idx_.find(x)) return &L1_vec_[*idx];
        return nullptr;
    }

    void add_L1(u64 v, char op, u64 a, u64 b) {
        if (v == 0 || v > pz_.limit || in_pool(v)) return;
        if (L1_idx_.insert(v, static_cast<uint32_t>(L1_vec_.size()))) {
            L1_vec_.push_back(Step{op, a, b, v});
        }
    }

    void link_value(u64 v) {
        for (u64 p : pool_) {
            if (v <= pz_.limit - p) add_L1(v + p, '+', v, p);
            u64 prod = elf::safe_mul(v, p, pz_.limit);
            if (prod) add_L1(prod, '*', v, p);
            if (v > p) {
                add_L1(v - p, '-', v, p);
                if (v % p == 0) add_L1(v / p, '/', v, p);
            } else if (p > v) {
                add_L1(p - v, '-', p, v);
                if (p % v == 0) add_L1(p / v, '/', p, v);
            } else {
                add_L1(1, '/', v, v);
                break;
            }
        }
    }

    void init_pool(const std::vector<u64>& starts) {
        pool_.clear();
        stamp_.clear();
        pool_pos_.clear();
        pins_.clear();
        L1_vec_.clear();
        L1_idx_.clear();
        plan_.clear();
        tick_ = 0;
        created_ = 0;
        m_dirty_ = true;
        for (u64 s : starts) {
            if (s == 0 || s > pz_.limit || in_pool(s)) continue;
            if (regs_ > 0 && static_cast<int>(pool_.size()) >= regs_) break;
            pool_pos_.insert(s, static_cast<uint32_t>(pool_.size()));
            pool_.push_back(s);
            stamp_.push_back(++tick_);
            ++created_;
            link_value(s);
        }
    }

    void reindex_pool() {
        pool_pos_.clear();
        for (size_t j = 0; j < pool_.size(); ++j) {
            pool_pos_.insert(pool_[j], static_cast<uint32_t>(j));
        }
    }

    void rebuild_all_L1() {
        std::vector<u64> old;
        old.swap(pool_);
        pool_pos_.clear();
        L1_vec_.clear();
        L1_idx_.clear();
        for (u64 v : old) {
            pool_pos_.insert(v, static_cast<uint32_t>(pool_.size()));
            pool_.push_back(v);
            link_value(v);
        }
        m_dirty_ = true;
    }

    void pin(u64 v) {
        if (int* p = pins_.find(v)) ++(*p);
        else pins_.insert(v, 1);
    }

    void unpin(u64 v) {
        if (int* p = pins_.find(v)) {
            if (--(*p) <= 0) pins_.erase(v);
        }
    }

    u64 effective_lru_stamp(size_t idx) const {
        u64 st = stamp_[idx];
        if (!cfg_.gentle_gcd_lru_boost) return st;
        u64 v = pool_[idx];
        int div_cnt = 0;
        bool exact_S = false;
        for (const auto& t : targets_) {
            if (t.covered || !t.smooth_active) continue;
            if (v == t.S) {
                exact_S = true;
                div_cnt += 2;
            } else if (v > 1 && t.S % v == 0) {
                ++div_cnt;
            }
        }
        if (exact_S) return st + 200;
        if (div_cnt >= 2 && v > 10) return st + 50;
        if (div_cnt >= 3) return st + 35;
        if (in_starts_.contains(v)) return st + 15;
        return st;
    }

    // Utility-aware multiplicative eviction: protects 1, small constants, scale-bucket
    // diversity, composite divisors/GCDs of remaining targets & smooth forms, shift
    // constants, and recently built targets/composites; evicts old dead targets first.
    size_t pick_victim(const FlatMap<size_t>& last_need, size_t step_idx) const {
        if (cfg_.evict_mode <= 1) {
            size_t victim = pool_.size();
            u64 best_st = std::numeric_limits<u64>::max();
            for (size_t j = 0; j < pool_.size(); ++j) {
                u64 pv = pool_[j];
                const size_t* ln = last_need.find(pv);
                if ((ln && *ln > step_idx) || pins_.contains(pv)) continue;
                u64 es = effective_lru_stamp(j);
                if (victim == pool_.size() || es < best_st) {
                    victim = j;
                    best_st = es;
                }
            }
            return victim;
        }

        int bcount[32] = {};
        for (u64 v : pool_) ++bcount[scale_bucket(v)];

        FlatMap<uint8_t> rem_set(remaining_.size() * 2 + 8);
        for (u64 t : remaining_) rem_set.insert(t, 1);

        size_t victim = pool_.size();
        double victim_score = 0.0;
        const double inv_tick = 1.0 / static_cast<double>(tick_ + 1);
        const u64 root_lim = elf::isqrt64(pz_.limit);

        for (size_t j = 0; j < pool_.size(); ++j) {
            u64 v = pool_[j];
            const size_t* ln = last_need.find(v);
            if ((ln && *ln > step_idx) || pins_.contains(v)) continue;

            double sc = 0.0;
            if (v == 1) {
                sc += 1.0e6;
            } else if (v <= 64) {
                sc += 155.0 / (1.0 + std::log2(static_cast<double>(v)));
            }

            int b = scale_bucket(v);
            if (v <= root_lim) {
                if (bcount[b] == 1) sc += 55.0;
                else if (bcount[b] == 2) sc += 22.0;
                else if (bcount[b] >= 7) sc -= 12.0 * static_cast<double>(bcount[b] - 6);
            } else {
                sc -= 32.0;
            }

            int div_hits = 0;
            int smooth_div_hits = 0;
            int shift_hits = 0;
            int aff_hits = 0;
            int near_hits = 0;
            int gcd_hits = 0;
            bool exact_S = false;
            double comp_bonus = 0.0;

            for (const auto& st : targets_) {
                if (st.covered) continue;
                u64 t = st.target;
                if (v == st.S) exact_S = true;
                if (st.c > 0 && v == st.c) ++shift_hits;
                if (v > 1 && t % v == 0) {
                    ++div_hits;
                    if (v > 100) comp_bonus = std::max(comp_bonus, 12.0 * std::log2(static_cast<double>(v)));
                }
                if (st.smooth_active && v > 1 && st.S % v == 0) {
                    ++smooth_div_hits;
                    if (v > 100) comp_bonus = std::max(comp_bonus, 10.0 * std::log2(static_cast<double>(v)));
                }
                u64 diff = (t > v) ? (t - v) : (v - t);
                u64 near_tol = std::max<u64>(10000ULL, t / 100ULL);
                if (diff <= near_tol) ++near_hits;
                if (v < t && v >= 2) {
                    u64 q = (t + v / 2) / v;
                    if (q >= 1 && q <= 200) {
                        u128 prod = static_cast<u128>(q) * v;
                        u64 adiff = prod > t ? static_cast<u64>(prod - t)
                                             : static_cast<u64>(t - prod);
                        u64 aff_tol = std::max<u64>(5000ULL, t / 250ULL);
                        if (adiff <= aff_tol) ++aff_hits;
                    }
                }
                if (v > 500) {
                    if (std::gcd(v, t) > 500 || (st.smooth_active && std::gcd(v, st.S) > 500)) {
                        ++gcd_hits;
                    }
                }
            }

            if (exact_S) sc += 320.0;
            sc += std::min(160.0, shift_hits * 85.0);
            sc += std::min(300.0, div_hits * 85.0);
            sc += std::min(220.0, smooth_div_hits * 55.0);
            sc += std::min(180.0, comp_bonus);
            sc += std::min(360.0, aff_hits * 120.0 * cfg_.aff_weight);
            sc += std::min(300.0, near_hits * 160.0);
            sc += std::min(140.0, gcd_hits * 35.0);

            // Evict old finished targets that have zero utility for remaining targets,
            // while keeping freshly built targets hot for immediate a*q +- r reuse!
            bool is_old_dead_target = all_targets_.contains(v) && !rem_set.contains(v) &&
                                      !exact_S && div_hits == 0 && smooth_div_hits == 0 &&
                                      aff_hits == 0 && near_hits == 0 && gcd_hits == 0 &&
                                      shift_hits == 0 && (stamp_[j] + 10 < tick_);
            if (is_old_dead_target) sc -= 85.0;

            // Strong LRU recency component keeps active working set & recent composites hot.
            sc += 55.0 * (static_cast<double>(stamp_[j]) * inv_tick);

            if (victim == pool_.size() || sc < victim_score) {
                victim = j;
                victim_score = sc;
            }
        }
        return victim;
    }

    bool apply_batch(const std::vector<Step>& steps) {
        if (steps.empty()) return true;
        FlatMap<size_t> last_need(steps.size() * 2 + 8);
        for (size_t i = 0; i < steps.size(); ++i) {
            last_need.insert_or_assign(steps[i].a, i);
            last_need.insert_or_assign(steps[i].b, i);
        }
        bool evicted = false;
        for (size_t i = 0; i < steps.size(); ++i) {
            const Step& s = steps[i];
            const uint32_t* ia = pool_pos_.find(s.a);
            const uint32_t* ib = pool_pos_.find(s.b);
            if (!ia || !ib) return false;
            u64 r = elf::apply_op(s.op, s.a, s.b, pz_.limit);
            if (r == 0) return false;
            stamp_[*ia] = stamp_[*ib] = ++tick_;
            if (in_pool(r)) continue;
            if (regs_ > 0 && static_cast<int>(pool_.size()) >= regs_) {
                size_t victim = pick_victim(last_need, i);
                if (victim == pool_.size()) return false;
                pool_.erase(pool_.begin() + static_cast<ptrdiff_t>(victim));
                stamp_.erase(stamp_.begin() + static_cast<ptrdiff_t>(victim));
                reindex_pool();
                evicted = true;
            }
            plan_.push_back(Step{s.op, s.a, s.b, r});
            pool_pos_.insert(r, static_cast<uint32_t>(pool_.size()));
            pool_.push_back(r);
            stamp_.push_back(++tick_);
            ++created_;
            if (!evicted) link_value(r);
            m_dirty_ = true;
        }
        if (evicted) rebuild_all_L1();
        failB_.clear();
        failB_div_.clear();
        mark_covered();
        return true;
    }

    void build_M() {
        if (!m_dirty_) return;
        M_.clear();
        M_.reserve(pool_.size() + L1_vec_.size());
        for (u64 p : pool_) M_.emplace_back(p, 0);
        for (const Step& e : L1_vec_) {
            if (!in_pool(e.r)) M_.emplace_back(e.r, 1);
        }
        std::sort(M_.begin(), M_.end());
        m_dirty_ = false;
    }

    void mark_covered() {
        for (auto& st : targets_) {
            if (!st.covered && in_pool(st.target)) {
                st.covered = true;
                st.smooth_active = false;
            }
        }
        remaining_.erase(
            std::remove_if(remaining_.begin(), remaining_.end(),
                           [&](u64 t) { return in_pool(t); }),
            remaining_.end());
    }

    bool has_active_smooth() const {
        for (const auto& st : targets_) {
            if (st.smooth_active && !st.covered) return true;
        }
        return false;
    }

    bool emit_from_L1(u64 x) {
        const Step* s = find_L1(x);
        if (!s) return false;
        Step copy = *s;
        return apply_batch({copy});
    }

    bool harvest_L1_targets() {
        std::vector<Step> batch;
        for (u64 t : remaining_) {
            if (const Step* s = find_L1(t)) batch.push_back(*s);
        }
        if (batch.empty()) return false;
        return apply_batch(batch);
    }

    // ---- Level 2 ----

    bool find2(u64 v, std::vector<Step>* out, std::vector<u64>* collect) {
        work_ += static_cast<double>(pool_.size()) * 6.0;
        auto hit = [&](u64 x, char op, bool x_first, u64 c) -> bool {
            const Step* sx = find_L1(x);
            if (!sx) return false;
            if (collect) {
                collect->push_back(x);
                return false;
            }
            out->push_back(*sx);
            out->push_back(x_first ? Step{op, x, c, v} : Step{op, c, x, v});
            return true;
        };
        for (u64 c : pool_) {
            if (v > c && hit(v - c, '+', true, c)) return true;
            if (v <= pz_.limit - c && hit(v + c, '-', true, c)) return true;
            if (c > v) {
                if (hit(c - v, '-', false, c)) return true;
                if (c % v == 0 && hit(c / v, '/', false, c)) return true;
            }
            if (c > 1) {
                if (v % c == 0 && hit(v / c, '*', true, c)) return true;
                u64 p = elf::safe_mul(v, c, pz_.limit);
                if (p && hit(p, '/', true, c)) return true;
            }
        }
        if ((v & 1ULL) == 0 && hit(v / 2, '+', true, v / 2)) return true;
        u64 s = elf::isqrt64(v);
        if (s * s == v && hit(s, '*', true, s)) return true;
        return collect && !collect->empty();
    }

    bool find2_cached(u64 v, std::vector<Step>* out) {
        if (const int* fb = failB_div_.find(v); fb && *fb >= 2) return false;
        if (const int* fb = failB_.find(v); fb && *fb >= 2) return false;
        if (find2(v, out, nullptr)) return true;
        const int* fbd = failB_div_.find(v);
        if (!fbd || *fbd < 2) failB_div_.insert_or_assign(v, 2);
        return false;
    }

    void find_best_L2_target_vote(u64& best_x, int& best_votes) {
        best_x = 0;
        best_votes = 0;
        FlatMap<int> votes(256);
        std::vector<u64> vote_keys;
        std::vector<u64> xs;
        for (u64 t : remaining_) {
            xs.clear();
            find2(t, nullptr, &xs);
            std::sort(xs.begin(), xs.end());
            xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
            for (u64 x : xs) {
                if (int* c = votes.find(x)) {
                    ++(*c);
                } else {
                    votes.insert(x, 1);
                    vote_keys.push_back(x);
                }
            }
        }
        if (vote_keys.empty()) return;

        double best_tie = -1e18;
        for (u64 x : vote_keys) {
            int vc = *votes.find(x);
            if (vc < best_votes) continue;
            double tie = 0.0;
            for (const auto& st : targets_) {
                if (st.covered || !st.smooth_active) continue;
                if (x == st.S) tie += 25.0;
                else if (x > 1 && st.S % x == 0) tie += 6.0;
                if (st.c > 0 && x == st.c) tie += 4.0;
            }
            if (cfg_.order_mode == TargetOrder::Descending) {
                tie += 0.1 * std::log2(static_cast<double>(std::max<u64>(2, x)));
            }
            if (vc > best_votes || (vc == best_votes && tie > best_tie)) {
                best_votes = vc;
                best_tie = tie;
                best_x = x;
            }
        }
    }

    // ---- Multi-Form Shifted-Smooth Exact-Divisor Synthesis (B = 3, 4, 5, 6) ----

    std::vector<size_t> ordered_active_target_indices() {
        std::vector<size_t> idxs;
        idxs.reserve(remaining_.size());
        for (u64 t : remaining_) {
            if (const size_t* idx = target_idx_.find(t)) idxs.push_back(*idx);
        }
        switch (cfg_.order_mode) {
            case TargetOrder::Ascending:
                break;  // remaining_ is already sorted ascending
            case TargetOrder::Descending:
                std::reverse(idxs.begin(), idxs.end());
                break;
            case TargetOrder::HubFirst:
                std::sort(idxs.begin(), idxs.end(), [&](size_t a, size_t b) {
                    if (targets_[a].hub_score != targets_[b].hub_score)
                        return targets_[a].hub_score > targets_[b].hub_score;
                    return targets_[a].target > targets_[b].target;
                });
                break;
            case TargetOrder::Shuffled:
                std::shuffle(idxs.begin(), idxs.end(), rng_);
                break;
        }
        return idxs;
    }

    bool harvest_shifted_smooth_B3() {
        auto idxs = ordered_active_target_indices();
        for (size_t idx : idxs) {
            const auto& st = targets_[idx];
            if (st.covered || !st.all_forms) continue;
            for (const auto& f : *st.all_forms) {
                if (f.c == 0) continue;
                char op = (f.sign > 0) ? '+' : '-';
                int c_st = c01(f.c);
                int s_st = c01(f.S);
                if (s_st == 1 && c_st == 1) {
                    const Step* ss = find_L1(f.S);
                    const Step* cs = find_L1(f.c);
                    if (ss && cs) {
                        return apply_batch({*ss, *cs, Step{op, f.S, f.c, st.target}});
                    }
                }
                if (c_st == 0 && s_st >= 2) {
                    std::vector<Step> steps;
                    work_ = 0;
                    if (find2_cached(f.S, &steps)) {
                        steps.push_back(Step{op, f.S, f.c, st.target});
                        return apply_batch(steps);
                    }
                }
                if (s_st == 0 && c_st >= 2) {
                    std::vector<Step> steps;
                    work_ = 0;
                    if (find2_cached(f.c, &steps)) {
                        steps.push_back(Step{op, f.S, f.c, st.target});
                        return apply_batch(steps);
                    }
                }
            }
        }
        return false;
    }

    // O(|half_divs|) Shifted-Smooth Level 4:
    //   - c in L1 & find2(S) -> 4 steps
    //   - c in P/0 & S = d * q with d, q in L1 -> 3-4 steps (ANY magnitude!)
    //   - c in P/0 & S = d * q with d in P, q in L2 -> 3-4 steps
    bool harvest_shifted_smooth_B4() {
        auto idxs = ordered_active_target_indices();

        // Pass 1: c in L1 and find2(S).
        for (size_t idx : idxs) {
            const auto& st = targets_[idx];
            if (st.covered || !st.all_forms) continue;
            for (const auto& f : *st.all_forms) {
                if (f.c == 0) continue;
                if (c01(f.c) == 1 && c01(f.S) >= 2) {
                    std::vector<Step> steps;
                    work_ = 0;
                    if (find2_cached(f.S, &steps)) {
                        const Step* cs = find_L1(f.c);
                        if (cs) {
                            char op = (f.sign > 0) ? '+' : '-';
                            steps.push_back(*cs);
                            steps.push_back(Step{op, f.S, f.c, st.target});
                            return apply_batch(steps);
                        }
                    }
                }
            }
        }

        // Pass 2: c in P/0 and S = d * q using precomputed (d, q) half_divs!
        for (size_t idx : idxs) {
            const auto& st = targets_[idx];
            if (st.covered || !st.all_forms) continue;
            if (check_time()) return false;
            u64 skey = (st.target << 3) ^ 5ULL;
            if (const size_t* last_c = stale_B_.find(skey)) {
                if (created_ < *last_c + *last_c / 8 + 2) continue;
            }
            for (const auto& f : *st.all_forms) {
                if (f.c > 0 && !in_pool(f.c)) continue;
                u64 S = f.S;
                for (const auto& dq : f.half_divs) {
                    u64 d = dq.first;
                    u64 q = dq.second;
                    int cd = c01(d);
                    int cq = c01(q);
                    if (cd <= 1 && cq <= 1) {
                        std::vector<Step> steps;
                        if (!emit01(d, cd, steps)) continue;
                        if (q != d && !emit01(q, cq, steps)) continue;
                        steps.push_back(Step{'*', d, q, S});
                        if (f.c > 0) {
                            char op = (f.sign > 0) ? '+' : '-';
                            steps.push_back(Step{op, S, f.c, st.target});
                        }
                        return apply_batch(steps);
                    }
                    if (cd == 0 && cq == 2) {
                        std::vector<Step> steps;
                        work_ = 0;
                        if (find2_cached(q, &steps)) {
                            steps.push_back(Step{'*', d, q, S});
                            if (f.c > 0) {
                                char op = (f.sign > 0) ? '+' : '-';
                                steps.push_back(Step{op, S, f.c, st.target});
                            }
                            return apply_batch(steps);
                        }
                    } else if (cq == 0 && cd == 2) {
                        std::vector<Step> steps;
                        work_ = 0;
                        if (find2_cached(d, &steps)) {
                            steps.push_back(Step{'*', d, q, S});
                            if (f.c > 0) {
                                char op = (f.sign > 0) ? '+' : '-';
                                steps.push_back(Step{op, S, f.c, st.target});
                            }
                            return apply_batch(steps);
                        }
                    }
                }
            }
            stale_B_.insert_or_assign(skey, created_);
        }
        return false;
    }

    // O(|half_divs|) Shifted-Smooth Level 5:
    //   - c in L1 & S = d * q with d, q in P u L1 -> 4-5 steps
    //   - c in P/0 & S = d * q with one in L1 and the other in L2 -> 4-5 steps!
    bool harvest_shifted_smooth_B5() {
        auto idxs = ordered_active_target_indices();
        for (size_t idx : idxs) {
            const auto& st = targets_[idx];
            if (st.covered || !st.all_forms) continue;
            if (check_time()) return false;
            u64 skey = (st.target << 3) ^ 6ULL;
            if (const size_t* last_c = stale_B_.find(skey)) {
                if (created_ < *last_c + *last_c / 6 + 3) continue;
            }
            for (const auto& f : *st.all_forms) {
                int c_st = (f.c == 0) ? 0 : c01(f.c);
                if (c_st > 1) continue;
                u64 S = f.S;
                for (const auto& dq : f.half_divs) {
                    u64 d = dq.first;
                    u64 q = dq.second;
                    int cd = c01(d);
                    int cq = c01(q);
                    if (c_st == 1 && f.c > 0) {
                        if (cd <= 1 && cq <= 1) {
                            const Step* sc = find_L1(f.c);
                            if (!sc) break;
                            std::vector<Step> steps;
                            if (!emit01(d, cd, steps)) continue;
                            if (q != d && !emit01(q, cq, steps)) continue;
                            steps.push_back(Step{'*', d, q, S});
                            steps.push_back(*sc);
                            char op = (f.sign > 0) ? '+' : '-';
                            steps.push_back(Step{op, S, f.c, st.target});
                            return apply_batch(steps);
                        }
                    } else {
                        // c == 0 or c in P: check (cd == 1, cq == 2) or (cq == 1, cd == 2)
                        if (cd == 1 && cq == 2) {
                            std::vector<Step> steps;
                            work_ = 0;
                            if (find2_cached(q, &steps)) {
                                if (!emit01(d, 1, steps)) continue;
                                steps.push_back(Step{'*', d, q, S});
                                if (f.c > 0) {
                                    char op = (f.sign > 0) ? '+' : '-';
                                    steps.push_back(Step{op, S, f.c, st.target});
                                }
                                return apply_batch(steps);
                            }
                        } else if (cq == 1 && cd == 2) {
                            std::vector<Step> steps;
                            work_ = 0;
                            if (find2_cached(d, &steps)) {
                                if (!emit01(q, 1, steps)) continue;
                                steps.push_back(Step{'*', d, q, S});
                                if (f.c > 0) {
                                    char op = (f.sign > 0) ? '+' : '-';
                                    steps.push_back(Step{op, S, f.c, st.target});
                                }
                                return apply_batch(steps);
                            }
                        }
                    }
                }
            }
            stale_B_.insert_or_assign(skey, created_);
        }
        return false;
    }

    // Fast, tightly capped Shifted-Smooth Level 6 (L2 x L2 zero-remainder split!):
    //   - Uses failB_div_ caching so each divisor d is tested by find2 at most once per pool!
    //   - Strictly capped in time (<= 8ms) and live find2 probes (<= 384) per call so it
    //     NEVER burns the restart budget.
    bool harvest_shifted_smooth_B6() {
        double pass_deadline = std::min(deadline_, timer_.elapsed_sec() + 0.008);
        int find2_probes = 0;
        constexpr int kMaxFind2Probes = 384;

        auto idxs = ordered_active_target_indices();
        for (size_t idx : idxs) {
            const auto& st = targets_[idx];
            if (st.covered || !st.all_forms) continue;
            if (find2_probes >= kMaxFind2Probes || timer_.elapsed_sec() >= pass_deadline) {
                return false;
            }
            u64 skey = (st.target << 3) ^ 7ULL;
            if (const size_t* last_c = stale_B_.find(skey)) {
                if (created_ < *last_c + *last_c / 4 + 4) continue;
            }
            size_t max_forms = std::min<size_t>(st.all_forms->size(), 6);
            bool finished_target = true;
            for (size_t fi = 0; fi < max_forms; ++fi) {
                const auto& f = (*st.all_forms)[fi];
                int c_st = (f.c == 0) ? 0 : c01(f.c);
                if (c_st > 1) continue;
                u64 S = f.S;
                int checked = 0;
                for (const auto& dq : f.half_divs) {
                    if (++checked > 48) break;
                    if (find2_probes >= kMaxFind2Probes ||
                        ((checked & 15) == 0 && timer_.elapsed_sec() >= pass_deadline)) {
                        finished_target = false;
                        break;
                    }
                    u64 d = dq.first;
                    u64 q = dq.second;
                    int cd = c01(d);
                    int cq = c01(q);
                    if (c_st == 0) {
                        if (cd == 2 && cq == 2) {
                            if (const int* fb = failB_div_.find(d); fb && *fb >= 2) continue;
                            if (const int* fb = failB_div_.find(q); fb && *fb >= 2) continue;
                            ++find2_probes;
                            std::vector<Step> s1;
                            work_ = 0;
                            if (!find2_cached(d, &s1)) continue;
                            if (q == d) {
                                s1.push_back(Step{'*', d, d, S});
                                if (f.c > 0) {
                                    char op = (f.sign > 0) ? '+' : '-';
                                    s1.push_back(Step{op, S, f.c, st.target});
                                }
                                return apply_batch(s1);
                            }
                            ++find2_probes;
                            std::vector<Step> s2;
                            work_ = 0;
                            if (!find2_cached(q, &s2)) continue;
                            for (const Step& stp : s2) s1.push_back(stp);
                            s1.push_back(Step{'*', d, q, S});
                            if (f.c > 0) {
                                char op = (f.sign > 0) ? '+' : '-';
                                s1.push_back(Step{op, S, f.c, st.target});
                            }
                            return apply_batch(s1);
                        }
                    } else if (c_st == 1 && f.c > 0) {
                        if ((cd == 1 && cq == 2) || (cq == 1 && cd == 2)) {
                            u64 one = (cd == 1) ? d : q;
                            u64 two = (cd == 1) ? q : d;
                            if (const int* fb = failB_div_.find(two); fb && *fb >= 2) continue;
                            ++find2_probes;
                            std::vector<Step> steps;
                            work_ = 0;
                            if (find2_cached(two, &steps)) {
                                const Step* sc = find_L1(f.c);
                                if (!sc || !emit01(one, 1, steps)) continue;
                                steps.push_back(Step{'*', d, q, S});
                                steps.push_back(*sc);
                                char op = (f.sign > 0) ? '+' : '-';
                                steps.push_back(Step{op, S, f.c, st.target});
                                return apply_batch(steps);
                            }
                        }
                    }
                }
                if (!finished_target) break;
            }
            if (finished_target) stale_B_.insert_or_assign(skey, created_);
        }
        return false;
    }

    // ---- Unified L1 Re-Pair Factor-Graph Scorer ----

    double prime_build_cost_live(u64 p) const {
        if (in_pool(p)) return 0.0;
        if (L1_idx_.contains(p)) return 1.0;
        int cc = cheap_.step_cost_of(p);
        if (cc >= 0) return std::min<double>(static_cast<double>(cc), 2.5);
        double lg = std::log2(static_cast<double>(p));
        return std::max(1.8, 0.42 * lg);
    }

    int count_omega(u64 d, const std::vector<std::pair<u64, int>>& target_fac) const {
        int cnt = 0;
        u64 rem = d;
        for (const auto& kv : target_fac) {
            while (rem % kv.first == 0) {
                rem /= kv.first;
                ++cnt;
            }
            if (rem == 1) break;
        }
        return cnt;
    }

    double atom_total_impl(u64 cur, const std::vector<u64>& pdivs,
                           const std::vector<std::pair<u64, int>>& tfac,
                           FlatMap<double>& memo, int& budget) const {
        if (cur <= 1) return 0.0;
        if (in_pool(cur)) return 1.0;
        if (const double* m = memo.find(cur)) return *m;

        u64 rem = cur;
        double fixed_cost = 0.0;
        for (const auto& kv : tfac) {
            u64 p = kv.first;
            if (rem % p != 0) continue;
            bool covered = false;
            for (u64 d : pdivs) {
                if (d % p == 0 && cur % d == 0) {
                    covered = true;
                    break;
                }
            }
            if (!covered) {
                int e = 0;
                while (rem % p == 0) {
                    rem /= p;
                    ++e;
                }
                fixed_cost += 1.0 + prime_build_cost_live(p) + static_cast<double>(add_chain_len(e));
            }
        }
        if (rem != cur) {
            double res = fixed_cost + atom_total_impl(rem, pdivs, tfac, memo, budget);
            memo.insert_or_assign(cur, res);
            return res;
        }
        if (budget <= 0) {
            double gcost = 0.0;
            u64 x = cur;
            for (u64 d : pdivs) {
                if (x % d != 0) continue;
                int k = 0;
                while (x % d == 0) {
                    x /= d;
                    ++k;
                }
                gcost += 1.0 + static_cast<double>(add_chain_len(k));
                if (x == 1) break;
                if (in_pool(x)) {
                    gcost += 1.0;
                    x = 1;
                    break;
                }
            }
            if (x > 1) {
                for (const auto& kv : tfac) {
                    u64 p = kv.first;
                    if (x % p != 0) continue;
                    int e = 0;
                    while (x % p == 0) {
                        x /= p;
                        ++e;
                    }
                    gcost += 1.0 + prime_build_cost_live(p) + static_cast<double>(add_chain_len(e));
                }
            }
            return gcost;
        }
        --budget;

        double best = 1e9;
        int tried = 0;
        for (u64 d : pdivs) {
            if (cur % d != 0) continue;
            int max_k = 0;
            u64 q = cur;
            while (q % d == 0) {
                q /= d;
                ++max_k;
            }
            double c_max = 1.0 + static_cast<double>(add_chain_len(max_k)) +
                           atom_total_impl(q, pdivs, tfac, memo, budget);
            if (c_max < best) best = c_max;
            if (max_k >= 2 && tried == 0 && budget > 4) {
                double c1 = 1.0 + atom_total_impl(cur / d, pdivs, tfac, memo, budget);
                if (c1 < best) best = c1;
            }
            if (++tried >= 3) break;
        }
        if (tried == 0) {
            double pcost = 0.0;
            u64 x = cur;
            for (const auto& kv : tfac) {
                u64 p = kv.first;
                if (x % p != 0) continue;
                int e = 0;
                while (x % p == 0) {
                    x /= p;
                    ++e;
                }
                pcost += 1.0 + prime_build_cost_live(p) + static_cast<double>(add_chain_len(e));
            }
            best = std::min(best, pcost);
        }
        memo.insert_or_assign(cur, best);
        return best;
    }

    double build_cost_S(u64 n, const std::vector<u64>& pdivs,
                        const std::vector<std::pair<u64, int>>& tfac) const {
        if (n <= 1 || in_pool(n)) return 0.0;
        FlatMap<double> memo(32);
        int budget = 18;
        double at = atom_total_impl(n, pdivs, tfac, memo, budget);
        return std::max(0.0, at - 1.0);
    }

    void find_best_repair_L1(u64& best_w, double& best_gain, int& best_hits) {
        best_w = 0;
        best_gain = 0.0;
        best_hits = 0;

        struct ActiveInfo {
            size_t idx;
            u64 S, c;
            double base_cost;
            std::vector<u64> pdivs;
        };
        std::vector<ActiveInfo> active;
        active.reserve(targets_.size());
        FlatMap<int> missing_shift_cnt(64);

        for (size_t i = 0; i < targets_.size(); ++i) {
            const auto& st = targets_[i];
            if (st.covered || !st.smooth_active) continue;
            if (st.c > 0 && !in_pool(st.c)) {
                if (int* p = missing_shift_cnt.find(st.c)) ++(*p);
                else missing_shift_cnt.insert(st.c, 1);
            }
            if (in_pool(st.S)) continue;

            ActiveInfo info;
            info.idx = i;
            info.S = st.S;
            info.c = st.c;
            std::vector<std::pair<int, u64>> pd;
            for (u64 d : pool_) {
                if (d > 1 && st.S % d == 0) pd.emplace_back(count_omega(d, st.factors), d);
            }
            std::sort(pd.begin(), pd.end(), [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second > b.second;
            });
            info.pdivs.reserve(pd.size());
            for (const auto& kv : pd) info.pdivs.push_back(kv.second);
            info.base_cost = build_cost_S(st.S, info.pdivs, st.factors);
            active.push_back(std::move(info));
        }

        if (active.empty()) return;

        for (const Step& se : L1_vec_) {
            u64 w = se.r;
            if (w <= 1 || in_pool(w)) continue;
            double gain = 0.0;
            int hits = 0;

            for (const auto& info : active) {
                if (info.S % w != 0) continue;
                if (w == info.S) {
                    double sv = info.base_cost + ((info.c == 0 || in_pool(info.c)) ? 0.6 : 0.2);
                    gain += sv;
                    ++hits;
                    continue;
                }
                int max_k = 0;
                u64 q = info.S;
                while (q % w == 0) {
                    q /= w;
                    ++max_k;
                }
                double new_atom = 0.0;
                if (q == 1) {
                    new_atom = 1.0 + static_cast<double>(add_chain_len(max_k));
                } else {
                    double q_bc = build_cost_S(q, info.pdivs, targets_[info.idx].factors);
                    new_atom = 1.0 + static_cast<double>(add_chain_len(max_k)) + (q_bc + 1.0);
                    if (max_k >= 2) {
                        u64 q1 = info.S / w;
                        double q1_bc = build_cost_S(q1, info.pdivs, targets_[info.idx].factors);
                        double alt = 1.0 + (q1_bc + 1.0);
                        if (alt < new_atom) new_atom = alt;
                    }
                }
                double new_bc = std::max(0.0, new_atom - 1.0);
                if (new_bc + 0.01 < info.base_cost) {
                    gain += (info.base_cost - new_bc);
                    ++hits;
                }
            }

            if (const int* sc = missing_shift_cnt.find(w)) {
                gain += 0.75 * static_cast<double>(*sc);
                hits += *sc;
            }
            if (hits >= 2) gain += 0.12 * static_cast<double>(hits);

            if (gain > best_gain) {
                best_gain = gain;
                best_w = w;
                best_hits = hits;
            }
        }
    }

    // ---- Deep Target Synthesis (B=6..8 + Live-Best Smooth Split + Smart Construct) ----

    u64 pick_best_smooth_divisor(u64 V, const std::vector<std::pair<u64, int>>& vfac,
                                 bool check_sibling_gcd) const {
        u64 best_known = 0;
        double best_known_sc = -1e18;
        auto eval_known = [&](u64 d, int c_d) {
            if (d <= 1 || d >= V || V % d != 0) return;
            u64 q = V / d;
            int c_q = c01(q);
            double sc = -10.0 * static_cast<double>(c_d + c_q) + std::log2(static_cast<double>(d));
            if (sc > best_known_sc) {
                best_known_sc = sc;
                best_known = d;
            }
        };
        for (u64 d : pool_) eval_known(d, 0);
        for (const Step& e : L1_vec_) {
            if (!in_pool(e.r)) eval_known(e.r, 1);
        }
        if (best_known > 1) {
            u64 q = V / best_known;
            if (c01(q) <= 1 ||
                static_cast<u128>(best_known) * best_known * best_known >= V ||
                (in_pool(best_known) && best_known >= 16)) {
                return best_known;
            }
        }

        if (check_sibling_gcd) {
            u64 best_g = 0;
            for (const auto& st : targets_) {
                if (st.covered || !st.smooth_active || st.S == V) continue;
                u64 g = std::gcd(V, st.S);
                if (g > 100 && g < V && g > best_g) best_g = g;
            }
            if (best_g > 100) return best_g;
        }

        auto divs = elf::get_divisors(vfac, 1024);
        u64 root = elf::isqrt64(V);
        u64 best_bal = 0;
        double best_dist = 1e18;
        double log_root = std::log2(static_cast<double>(std::max<u64>(2, root)));
        for (u64 d : divs) {
            if (d <= 1 || d >= V) continue;
            u64 q = V / d;
            double dist = std::abs(std::log2(static_cast<double>(d)) - log_root);
            if (c01(d) <= 1) dist -= 2.0;
            if (c01(q) <= 1) dist -= 2.0;
            if (std::gcd(d, q) == 1) dist -= 1.0;  // keep prime powers intact
            if (dist < best_dist) {
                best_dist = dist;
                best_bal = d;
            }
        }
        return best_bal;
    }

    bool build_smooth_value(u64 V, int depth) {
        if (in_pool(V)) return true;
        if (L1_idx_.contains(V)) return emit_from_L1(V);
        if (should_abort()) return false;
        if (check_time()) return construct_value_fast_split(V);

        int b_try = (depth == 0) ? 5 : 4;
        cur_work_limit_ = (depth == 0) ? 1.0e7 : 5.0e6;
        for (int B = 2; B <= b_try; ++B) {
            if (check_time()) break;
            std::vector<Step> steps;
            work_ = 0;
            if (solve_bounded(V, B, steps, depth == 0)) return apply_batch(steps);
        }

        auto vfac = elf::factorize(V);
        if (depth >= 5 || (vfac.size() == 1 && vfac[0].second == 1)) {
            return construct_value(V, 6, depth + 1);
        }

        u64 d = pick_best_smooth_divisor(V, vfac, depth == 0);
        if (d <= 1 || d >= V || V % d != 0) return construct_value(V, 6, depth + 1);
        u64 q = V / d;
        if (d > q) std::swap(d, q);
        if (!build_smooth_value(d, depth + 1)) return false;
        if (in_pool(V)) return true;
        pin(d);
        bool ok = build_smooth_value(q, depth + 1);
        if (ok && !in_pool(V)) ok = apply_batch({Step{'*', d, q, V}});
        unpin(d);
        return ok && in_pool(V);
    }

    double live_form_est_cost(const ShiftedForm& f) const {
        double sc = (f.c == 0) ? 0.0 : (1.0 + (c01(f.c) == 0 ? 0.0 : (c01(f.c) == 1 ? 1.0 : 2.5)));
        if (in_pool(f.S)) return sc;
        if (L1_idx_.contains(f.S)) return sc + 1.0;
        std::vector<std::pair<int, u64>> pd;
        for (u64 d : pool_) {
            if (d > 1 && f.S % d == 0) pd.emplace_back(count_omega(d, f.factors), d);
        }
        std::sort(pd.begin(), pd.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second > b.second;
        });
        std::vector<u64> pdivs;
        pdivs.reserve(pd.size());
        for (const auto& kv : pd) pdivs.push_back(kv.second);
        sc += build_cost_S(f.S, pdivs, f.factors);
        return sc;
    }

    bool finish_hard_target() {
        u64 pick_t = pick_fallback_target();
        double rem = rem_time_sec();
        double res = reserve_time_sec();
        int max_B = (rem > res * 1.15) ? 8 : (rem > res * 0.65 ? 7 : 6);

        // 1. Try B = 6..max_B directly on pick_t FIRST (with tight work limits so one target
        //    never consumes the whole restart!).
        for (int B = 6; B <= max_B; ++B) {
            if (check_time()) break;
            cur_work_limit_ = (B == 6) ? 8.0e6 : (B == 7 ? 5.0e6 : 4.0e6);
            std::vector<Step> steps;
            work_ = 0;
            if (solve_bounded(pick_t, B, steps, true)) {
                return apply_batch(steps);
            }
            // Also check top 3 shifted smooth forms of pick_t with c in P u L1!
            if (B <= 7 && !low_on_time()) {
                if (const size_t* pi = target_idx_.find(pick_t)) {
                    const auto& st = targets_[*pi];
                    if (st.all_forms) {
                        int forms_tried = 0;
                        for (const auto& f : *st.all_forms) {
                            if (f.c == 0) continue;
                            int cc = c01(f.c);
                            if (cc > 1) continue;
                            int sb = B - 1 - cc;
                            if (sb < 2) continue;
                            if (++forms_tried > 3 || check_time()) break;
                            cur_work_limit_ = 2.5e6;
                            std::vector<Step> ssteps;
                            work_ = 0;
                            if (solve_bounded(f.S, sb, ssteps, false)) {
                                if (cc == 1 && !emit01(f.c, 1, ssteps)) continue;
                                char op = (f.sign > 0) ? '+' : '-';
                                ssteps.push_back(Step{op, f.S, f.c, st.target});
                                return apply_batch(ssteps);
                            }
                        }
                    }
                }
            }
        }

        // 2. Live-best composite smooth form split (handles monster 10^12..10^17 targets
        //    like 2^5*3*29^3*73^3*71147 even when pmax > 10000!).
        if (const size_t* pi = target_idx_.find(pick_t)) {
            const auto& st = targets_[*pi];
            const ShiftedForm* best_f = nullptr;
            double best_sc = 1e18;
            if (st.all_forms) {
                for (const auto& f : *st.all_forms) {
                    if (f.cofactor != 1) continue;
                    bool is_comp = (f.factors.size() >= 2) ||
                                   (f.factors.size() == 1 && f.factors[0].second >= 2);
                    if (!is_comp) continue;
                    bool good_smooth = (f.pmax <= 100000) ||
                                       (f.pmax <= 1000000 && f.S / f.pmax >= 100);
                    if (!good_smooth) continue;
                    double sc = live_form_est_cost(f);
                    if (sc < best_sc) {
                        best_sc = sc;
                        best_f = &f;
                    }
                }
            }
            if (best_f && (best_sc <= 16.0 || (best_f->pmax <= 100000 && best_sc <= 21.0))) {
                if (build_smooth_value(best_f->S, 0)) {
                    if (in_pool(st.target)) return true;
                    if (best_f->c > 0) {
                        pin(best_f->S);
                        bool ok_c = in_pool(best_f->c) || construct_value(best_f->c, 5, 1, false);
                        if (ok_c && !in_pool(st.target)) {
                            char op = (best_f->sign > 0) ? '+' : '-';
                            ok_c = apply_batch({Step{op, best_f->S, best_f->c, st.target}});
                        }
                        unpin(best_f->S);
                        if (in_pool(st.target)) return true;
                    }
                }
            }
        }

        // 3. Go straight to construct_value split (skipping redundant B=1..max_B on iter 0!).
        return construct_value(pick_t, 6, 0, true);
    }

    // ---- Enhanced L3 / decomp / construct Sub-Solver ----

    bool emit01(u64 x, int c, std::vector<Step>& out) const {
        if (c == 0) return true;
        const Step* s = find_L1(x);
        if (!s) return false;
        out.push_back(*s);
        return true;
    }

    bool find3(u64 v, std::vector<Step>& out) {
        if (check_time()) return false;
        work_ += static_cast<double>(L1_vec_.size()) * 6.0;
        auto both = [&](const Step& sx, u64 y, Step last) -> bool {
            const Step* sy = find_L1(y);
            if (!sy) return false;
            out.push_back(sx);
            if (y != sx.r) out.push_back(*sy);
            out.push_back(last);
            return true;
        };
        for (const Step& sx : L1_vec_) {
            u64 x = sx.r;
            if (in_pool(x)) continue;
            if (v > x && both(sx, v - x, Step{'+', x, v - x, v})) return true;
            if (x > v) {
                if (both(sx, x - v, Step{'-', x, x - v, v})) return true;
                if (x % v == 0 && both(sx, x / v, Step{'/', x, x / v, v})) return true;
            }
            if (v <= pz_.limit - x && both(sx, v + x, Step{'-', v + x, x, v})) return true;
            if (x > 1) {
                if (v % x == 0 && both(sx, v / x, Step{'*', x, v / x, v})) return true;
                u64 p = elf::safe_mul(v, x, pz_.limit);
                if (p && both(sx, p, Step{'/', p, x, v})) return true;
            }
        }
        auto chain = [&](u64 y, Step last) -> bool {
            size_t mark = out.size();
            if (!find2_cached(y, &out)) {
                out.resize(mark);
                return false;
            }
            out.push_back(last);
            return true;
        };
        for (u64 c : pool_) {
            if (v > c && chain(v - c, Step{'+', v - c, c, v})) return true;
            if (v <= pz_.limit - c && chain(v + c, Step{'-', v + c, c, v})) return true;
            if (c > v) {
                if (chain(c - v, Step{'-', c, c - v, v})) return true;
                if (c % v == 0 && chain(c / v, Step{'/', c, c / v, v})) return true;
            }
            if (c > 1) {
                if (v % c == 0 && chain(v / c, Step{'*', v / c, c, v})) return true;
                u64 p = elf::safe_mul(v, c, pz_.limit);
                if (p && chain(p, Step{'/', p, c, v})) return true;
            }
        }
        return false;
    }

    bool try_split(int B, u64 a, int ca, u64 q, u64 r, bool minus, std::vector<Step>& out) {
        if (q == 0 || r == 0) return false;
        int base = ca + 2;
        int lq = c01(q), lr = c01(r);
        if (base + lq + lr > B) return false;
        size_t mark = out.size();
        auto finish = [&]() -> bool {
            if (!emit01(a, ca, out)) return false;
            u64 prod = elf::safe_mul(a, q, pz_.limit);
            if (!prod) return false;
            out.push_back(Step{'*', a, q, prod});
            u64 res = minus ? (prod - r) : (prod + r);
            out.push_back(Step{minus ? '-' : '+', prod, r, res});
            return true;
        };
        int cap_r = std::min(2, B - base - lq);
        if (solve_bounded(r, cap_r, out, false)) {
            int cr = static_cast<int>(out.size() - mark);
            if (solve_bounded(q, B - base - cr, out, false)) return finish();
            out.resize(mark);
            return false;
        }
        out.resize(mark);
        int cap_q = std::min(2, B - base - std::max(lr, cap_r + 1));
        if (cap_q < lq) return false;
        if (solve_bounded(q, cap_q, out, false)) {
            int cq = static_cast<int>(out.size() - mark);
            if (solve_bounded(r, B - base - cq, out, false)) return finish();
        }
        out.resize(mark);
        return false;
    }

    bool decomp(u64 v, int B, std::vector<Step>& out, bool allow_div) {
        if (check_time() || work_ > cur_work_limit_) return false;
        build_M();
        work_ += static_cast<double>(M_.size()) * 16.0;

        struct Cand {
            u64 r, a, q;
            uint8_t ca, cq, kind;
            int rem;
        };
        std::vector<Cand> cands;
        cands.reserve(256);

        // 1. M * M +- r with immediate returns on exact product or cr <= 1!
        for (const auto& kv : M_) {
            u64 a = kv.first;
            uint8_t ca = kv.second;
            if (a < 2) continue;
            if (a >= v) break;
            auto it = std::lower_bound(M_.begin(), M_.end(), std::make_pair(v / a, uint8_t{0}));
            for (int d = -1; d <= 1; ++d) {
                auto jt = it + d;
                if (jt < M_.begin() || jt >= M_.end() || jt->first < 2) continue;
                u64 q = jt->first;
                int cost0 = ca + jt->second + 1;
                u64 prod = elf::safe_mul(a, q, pz_.limit);
                if (prod == 0) continue;
                if (prod == v) {
                    if (cost0 <= B) {
                        if (!emit01(a, ca, out)) continue;
                        if (q != a && !emit01(q, jt->second, out)) continue;
                        out.push_back(Step{'*', a, q, v});
                        return true;
                    }
                    continue;
                }
                if (cost0 >= B) continue;
                u64 r = (prod > v) ? (prod - v) : (v - prod);
                int rem = B - cost0 - 1;
                int cr = c01(r);
                if (rem < cr) continue;
                if (cr <= 1) {
                    size_t mark = out.size();
                    if (emit01(r, cr, out) && emit01(a, ca, out) &&
                        (q == a || emit01(q, jt->second, out))) {
                        out.push_back(Step{'*', a, q, prod});
                        out.push_back(Step{prod > v ? '-' : '+', prod, r, v});
                        return true;
                    }
                    out.resize(mark);
                    continue;
                }
                cands.push_back(Cand{r, a, q, ca, jt->second, uint8_t(prod > v ? 1 : 0), rem});
            }
        }

        // 2. Nearby M +- r with immediate returns on cr <= 1!
        auto it = std::lower_bound(M_.begin(), M_.end(), std::make_pair(v, uint8_t{0}));
        for (int d = -10; d < 10; ++d) {
            auto jt = it + d;
            if (jt < M_.begin() || jt >= M_.end() || jt->first == v) continue;
            u64 x = jt->first;
            u64 r = (x > v) ? (x - v) : (v - x);
            int rem = B - jt->second - 1;
            int cr = c01(r);
            if (rem < cr) continue;
            if (cr <= 1) {
                size_t mark = out.size();
                if (emit01(r, cr, out) && emit01(x, jt->second, out)) {
                    out.push_back(Step{x > v ? '-' : '+', x, r, v});
                    return true;
                }
                out.resize(mark);
                continue;
            }
            cands.push_back(Cand{r, x, 0, jt->second, 0, uint8_t(x > v ? 3 : 2), rem});
        }
        std::sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) {
            return x.r != y.r ? x.r < y.r : x.rem > y.rem;
        });

        int tried = 0;
        u64 last_r = 0;
        for (const Cand& c : cands) {
            if (c.r == last_r) continue;
            if (++tried > 24 || work_ > cur_work_limit_ || check_time()) break;
            last_r = c.r;
            size_t mark = out.size();
            if (!solve_bounded(c.r, c.rem, out, false)) {
                out.resize(mark);
                continue;
            }
            if (!emit01(c.a, c.ca, out)) {
                out.resize(mark);
                continue;
            }
            if (c.kind < 2) {
                if (c.q != c.a && !emit01(c.q, c.cq, out)) {
                    out.resize(mark);
                    continue;
                }
                u64 prod = c.a * c.q;
                out.push_back(Step{'*', c.a, c.q, prod});
                out.push_back(Step{c.kind ? '-' : '+', prod, c.r, v});
            } else {
                out.push_back(Step{c.kind == 3 ? '-' : '+', c.a, c.r, v});
            }
            return true;
        }

        // 3. Exact & remainder splits with a in pool (both a*q+r and a*(q+1)-(a-r))
        for (u64 a : pool_) {
            if (a < 2 || a >= v) continue;
            if (work_ > cur_work_limit_ || check_time()) return false;
            u64 q = v / a, r = v % a;
            if (r == 0) {
                size_t mark = out.size();
                if (solve_bounded(q, B - 1, out, false)) {
                    out.push_back(Step{'*', q, a, v});
                    return true;
                }
                out.resize(mark);
                continue;
            }
            if (try_split(B, a, 0, q, r, false, out)) return true;
            if (elf::safe_mul(a, q + 1, pz_.limit) != 0 &&
                try_split(B, a, 0, q + 1, a - r, true, out))
                return true;
        }

        // 4. Exact & remainder splits with a in L1 (exact r==0 from B>=3; capped deep splits at B>=6)
        int deep_l1_splits = 0;
        const int max_deep_l1 = (B >= 6) ? 64 : 1000000;
        for (const Step& se : L1_vec_) {
            u64 a = se.r;
            if (a < 2 || a >= v || in_pool(a)) continue;
            if (work_ > cur_work_limit_ || check_time()) return false;
            u64 q = v / a, r = v % a;
            if (r == 0) {
                if (B >= 3 && c01(q) <= B - 2) {
                    size_t mark = out.size();
                    if (solve_bounded(q, B - 2, out, false)) {
                        out.push_back(se);
                        out.push_back(Step{'*', a, q, v});
                        return true;
                    }
                    out.resize(mark);
                }
                continue;
            }
            int cq = c01(q), cr = c01(r);
            if (std::min(cq, cr) <= 1) {
                bool is_deep = (B >= 6 && std::max(cq, cr) >= 2);
                if (!is_deep || deep_l1_splits < max_deep_l1) {
                    if (is_deep) ++deep_l1_splits;
                    if (try_split(B, a, 1, q, r, false, out)) return true;
                }
            }
            if (elf::safe_mul(a, q + 1, pz_.limit) != 0) {
                int cq1 = c01(q + 1), cr1 = c01(a - r);
                if (std::min(cq1, cr1) <= 1) {
                    bool is_deep = (B >= 6 && std::max(cq1, cr1) >= 2);
                    if (!is_deep || deep_l1_splits < max_deep_l1) {
                        if (is_deep) ++deep_l1_splits;
                        if (try_split(B, a, 1, q + 1, a - r, true, out)) return true;
                    }
                }
            }
        }

        // 5. Exact-division lottery: v = (v * d) / d (catches (a * b) / d and (a*b +- r) / d!)
        if (allow_div && (B == 4 || B == 5 || B == 6) && !check_time() &&
            work_ < cur_work_limit_ * 0.35) {
            std::vector<u64> mults;
            mults.reserve(pool_.size());
            for (u64 d : pool_) {
                if (d >= 2 && (B == 4 || d <= 64) && elf::safe_mul(v, d, pz_.limit) != 0) {
                    mults.push_back(d);
                }
            }
            std::sort(mults.begin(), mults.end());
            int max_div = (B == 4) ? 10 : (B == 5 ? 6 : 4);
            int tried_div = 0;
            for (u64 d : mults) {
                if (++tried_div > max_div || work_ > cur_work_limit_ || check_time()) break;
                u64 vd = v * d;
                size_t mark = out.size();
                if (solve_bounded(vd, B - 1, out, false)) {
                    out.push_back(Step{'/', vd, d, v});
                    return true;
                }
                out.resize(mark);
            }
        }

        // 6. Smooth exact factorization split v = f1 * f2 (fast O(60) check!)
        if (allow_div && (B == 5 || B == 6) && v > 1000 && !check_time() &&
            work_ < cur_work_limit_ * 0.45 && is_smooth_300(v)) {
            u64 root_v = elf::isqrt64(v);
            auto fac = elf::factorize(v);
            auto divs = elf::get_divisors(fac, 256);
            std::vector<u64> balanced;
            for (u64 d : divs) {
                if (d >= 3 && d <= root_v) balanced.push_back(d);
            }
            std::sort(balanced.begin(), balanced.end(), std::greater<u64>());
            int tried_fac = 0;
            for (u64 f1 : balanced) {
                if (++tried_fac > 4 || work_ > cur_work_limit_ || check_time()) break;
                u64 f2 = v / f1;
                int b1 = std::min(3, B - 2);
                size_t mark = out.size();
                if (solve_bounded(f1, b1, out, false)) {
                    int c1 = static_cast<int>(out.size() - mark);
                    if (solve_bounded(f2, B - 1 - c1, out, false)) {
                        out.push_back(Step{'*', f1, f2, v});
                        return true;
                    }
                }
                out.resize(mark);
            }
        }

        return false;
    }

    bool solve_bounded(u64 v, int B, std::vector<Step>& out, bool allow_div = true) {
        int c = c01(v);
        if (c == 0) return true;
        if (B <= 0) return false;
        if (c == 1) return emit01(v, 1, out);
        if (B == 1) return false;
        if (work_ > cur_work_limit_ || check_time()) return false;

        int from = 2;
        if (const int* fbd = failB_div_.find(v)) from = std::max(from, *fbd + 1);
        if (!allow_div) {
            if (const int* fb = failB_.find(v)) from = std::max(from, *fb + 1);
        }
        size_t mark = out.size();
        for (int b = from; b <= B; ++b) {
            if (work_ > cur_work_limit_ || check_time()) return false;
            bool ok = (b == 2) ? find2(v, &out, nullptr)
                      : (b == 3) ? find3(v, out)
                                 : decomp(v, b, out, allow_div);
            if (ok) return true;
            out.resize(mark);
            if (work_ > cur_work_limit_ || check_time()) return false;
            if (allow_div || b == 2) {
                const int* fbd = failB_div_.find(v);
                if (!fbd || b > *fbd) failB_div_.insert_or_assign(v, b);
            } else {
                const int* fb = failB_.find(v);
                if (!fb || b > *fb) failB_.insert_or_assign(v, b);
            }
        }
        return false;
    }

    bool solve_any_remaining_bounded(int b_lo, int b_hi) {
        auto idxs = ordered_active_target_indices();
        for (int B = b_lo; B <= b_hi; ++B) {
            if (check_time()) return false;
            double pass_deadline = deadline_;
            if (B >= 5) {
                double avail = std::max(0.0, rem_time_sec() - reserve_time_sec());
                pass_deadline = std::min(deadline_, timer_.elapsed_sec() + std::max(0.012, avail * 0.30));
            }
            cur_work_limit_ = (B <= 3) ? 1.0e7 : (B == 4 ? 6.0e6 : 3.0e6);
            for (size_t idx : idxs) {
                u64 t = targets_[idx].target;
                if (check_time()) return false;
                if (B >= 5 && timer_.elapsed_sec() >= pass_deadline) return false;
                u64 key = (t << 3) ^ static_cast<u64>(B);
                if (B >= 4) {
                    if (const size_t* last_c = stale_B_.find(key)) {
                        if (created_ < *last_c + *last_c / 5 + 3) continue;
                    }
                }
                std::vector<Step> steps;
                work_ = 0;
                if (solve_bounded(t, B, steps, true)) {
                    return apply_batch(steps);
                }
                if (B >= 4) {
                    stale_B_.insert_or_assign(key, created_);
                }
            }
        }
        return false;
    }

    double est_cost(u64 x) const {
        int c = c01(x);
        if (c <= 1) return static_cast<double>(c);
        return 2.0 + 0.42 * std::log2(static_cast<double>(std::max<u64>(2, x)));
    }

    bool construct_value(u64 v, int bmax, int depth = 0, bool skip_bounded_first = false) {
        int guard = 0;
        while (!in_pool(v) && ++guard < 64) {
            if (should_abort()) {
                aborted_ = true;
                return false;
            }
            bool run_bounded = !(skip_bounded_first && guard == 1);
            if (run_bounded) {
                bool fast_mode = check_time();
                int cur_bmax = fast_mode ? 5 : (depth > 0 ? std::min(bmax, 6) : bmax);
                cur_work_limit_ = fast_mode ? 1.8e6 : (depth > 0 ? 4.0e6 : 6.0e6);

                std::vector<Step> steps;
                bool ok = false;
                bool prev_ignore = ignore_soft_timeout_;
                if (fast_mode) ignore_soft_timeout_ = true;
                for (int B = 1; B <= cur_bmax && !ok; ++B) {
                    steps.clear();
                    work_ = 0;
                    ok = solve_bounded(v, B, steps, depth == 0 && !fast_mode);
                    if (fast_mode && B >= 5) break;
                }
                ignore_soft_timeout_ = prev_ignore;
                if (ok) return apply_batch(steps);
            }

            // Try exact divisor in M, 300-smooth balanced factorization, or 1-step smooth shift!
            if (v > 256 && depth < 6) {
                u64 root = elf::isqrt64(v);
                u64 min_good_div = std::max<u64>(100ULL, static_cast<u64>(std::cbrt(static_cast<double>(v))));
                build_M();
                u64 best_m_div = 0;
                for (const auto& kv : M_) {
                    if (kv.first < min_good_div) continue;
                    if (kv.first > root) break;
                    if (v % kv.first == 0) best_m_div = kv.first;
                }
                if (best_m_div > 0) {
                    u64 f2 = v / best_m_div;
                    if (!construct_value(best_m_div, std::min(bmax, 6), depth + 1, false)) return false;
                    pin(best_m_div);
                    bool ok2 = construct_value(f2, std::min(bmax, 6), depth + 1, false);
                    if (ok2 && !in_pool(v)) ok2 = apply_batch({Step{'*', best_m_div, f2, v}});
                    unpin(best_m_div);
                    return ok2 && in_pool(v);
                }
                if (is_smooth_300(v)) {
                    auto fac = elf::factorize(v);
                    auto divs = elf::get_divisors(fac, 512);
                    u64 best_fac = 0;
                    for (u64 d : divs) {
                        if (d >= 3 && d <= root && d > best_fac) best_fac = d;
                    }
                    if (best_fac > 0) {
                        u64 f2 = v / best_fac;
                        if (!construct_value(best_fac, std::min(bmax, 6), depth + 1, false)) return false;
                        pin(best_fac);
                        bool ok2 = construct_value(f2, std::min(bmax, 6), depth + 1, false);
                        if (ok2 && !in_pool(v)) ok2 = apply_batch({Step{'*', best_fac, f2, v}});
                        unpin(best_fac);
                        return ok2 && in_pool(v);
                    }
                }
                if (depth == 0) {
                    u64 best_shift = 0;
                    bool shift_plus = false;
                    u64 best_c = 0;
                    for (u64 c : pool_) {
                        if (c == 0 || c >= v / 100ULL) continue;
                        if (v <= pz_.limit - c && is_smooth_300(v + c)) {
                            best_shift = v + c;
                            shift_plus = true;
                            best_c = c;
                            break;
                        }
                        if (v > c && is_smooth_300(v - c)) {
                            best_shift = v - c;
                            shift_plus = false;
                            best_c = c;
                            break;
                        }
                    }
                    if (best_shift > 0) {
                        pin(best_c);
                        bool ok_s = construct_value(best_shift, std::min(bmax, 6), depth + 1, false);
                        if (ok_s && !in_pool(v)) {
                            ok_s = apply_batch({Step{shift_plus ? '-' : '+', best_shift, best_c, v}});
                        }
                        unpin(best_c);
                        return ok_s && in_pool(v);
                    }
                }
            }

            // General split: evaluate BOTH a*q + r and a*(q+1) - (a-r) around sqrt(v) in M!
            build_M();
            u64 root = elf::isqrt64(v);
            struct SplitChoice {
                u64 a = 0, q = 0, r = 0;
                bool minus = false;
                double score = 1e18;
            };
            SplitChoice best_ch;
            auto eval_split = [&](u64 a, uint8_t ca) {
                if (a < 2 || a >= v) return;
                u64 q = v / a, r = v % a;
                if (q == 0) return;
                double sc_plus = ca + est_cost(q) + (r ? (est_cost(r) + 1.0) : 0.0);
                if (sc_plus < best_ch.score) {
                    best_ch = SplitChoice{a, q, r, false, sc_plus};
                }
                if (r > 0 && elf::safe_mul(a, q + 1, pz_.limit) != 0) {
                    u64 r2 = a - r;
                    double sc_minus = ca + est_cost(q + 1) + est_cost(r2) + 1.0;
                    if (sc_minus < best_ch.score) {
                        best_ch = SplitChoice{a, q + 1, r2, true, sc_minus};
                    }
                }
            };

            auto it = std::upper_bound(M_.begin(), M_.end(), std::make_pair(root, uint8_t{255}));
            auto it_dn = it;
            for (int seen = 0; it_dn != M_.begin() && seen < 24; ++seen) {
                --it_dn;
                if (it_dn->first < 2) break;
                eval_split(it_dn->first, it_dn->second);
            }
            auto it_up = it;
            for (int seen = 0; it_up != M_.end() && seen < 8; ++seen, ++it_up) {
                if (it_up->first >= v) break;
                eval_split(it_up->first, it_up->second);
            }
            if (best_ch.a == 0) {
                for (const auto& kv : M_) {
                    if (kv.first >= 2 && kv.first < v) {
                        eval_split(kv.first, kv.second);
                        break;
                    }
                }
            }
            if (best_ch.a == 0) {
                if (in_pool(1)) {
                    if (v == 2) return apply_batch({Step{'+', 1, 1, 2}});
                    return false;
                }
                u64 p0 = pool_[0];
                if (!apply_batch({Step{'/', p0, p0, 1}})) return false;
                continue;
            }
            if (!construct_value(best_ch.a, std::min(bmax, 6), depth + 1, false)) return false;
            pin(best_ch.a);
            if (!construct_value(best_ch.q, std::min(bmax, 6), depth + 1, false)) {
                unpin(best_ch.a);
                return false;
            }
            pin(best_ch.q);
            bool ok_final = false;
            if (best_ch.r > 0) {
                if (construct_value(best_ch.r, std::min(bmax, 6), depth + 1, false)) {
                    u64 prod = best_ch.a * best_ch.q;
                    ok_final = apply_batch({Step{'*', best_ch.a, best_ch.q, prod},
                                            Step{best_ch.minus ? '-' : '+', prod, best_ch.r, v}});
                }
            } else {
                ok_final = apply_batch({Step{'*', best_ch.a, best_ch.q, v}});
            }
            unpin(best_ch.q);
            unpin(best_ch.a);
            if (!ok_final) return false;
        }
        return in_pool(v);
    }

    bool construct_value_fast_split(u64 v) {
        if (in_pool(v)) return true;
        if (L1_idx_.contains(v)) return emit_from_L1(v);
        if (should_abort()) {
            aborted_ = true;
            return false;
        }
        // Smart fast completion: still tries B <= 4 with tiny work limit + M-split (+- r)!
        {
            double saved_wl = cur_work_limit_;
            bool saved_to = soft_timed_out_;
            soft_timed_out_ = false;
            cur_work_limit_ = 1.5e6;
            for (int B = 2; B <= 4; ++B) {
                std::vector<Step> steps;
                work_ = 0;
                if (solve_bounded(v, B, steps, false)) {
                    cur_work_limit_ = saved_wl;
                    soft_timed_out_ = saved_to;
                    return apply_batch(steps);
                }
            }
            cur_work_limit_ = saved_wl;
            soft_timed_out_ = saved_to;
        }

        build_M();
        u64 root = elf::isqrt64(v);
        struct SplitChoice {
            u64 a = 0, q = 0, r = 0;
            bool minus = false;
            double score = 1e18;
        };
        SplitChoice best_ch;
        auto eval_split = [&](u64 a, uint8_t ca) {
            if (a < 2 || a >= v) return;
            u64 q = v / a, r = v % a;
            if (q == 0) return;
            double sc_plus = ca + est_cost(q) + (r ? (est_cost(r) + 1.0) : 0.0);
            if (sc_plus < best_ch.score) best_ch = SplitChoice{a, q, r, false, sc_plus};
            if (r > 0 && elf::safe_mul(a, q + 1, pz_.limit) != 0) {
                u64 r2 = a - r;
                double sc_minus = ca + est_cost(q + 1) + est_cost(r2) + 1.0;
                if (sc_minus < best_ch.score) best_ch = SplitChoice{a, q + 1, r2, true, sc_minus};
            }
        };
        auto it = std::upper_bound(M_.begin(), M_.end(), std::make_pair(root, uint8_t{255}));
        auto it_dn = it;
        for (int seen = 0; it_dn != M_.begin() && seen < 16; ++seen) {
            --it_dn;
            if (it_dn->first < 2) break;
            eval_split(it_dn->first, it_dn->second);
        }
        auto it_up = it;
        for (int seen = 0; it_up != M_.end() && seen < 6; ++seen, ++it_up) {
            if (it_up->first >= v) break;
            eval_split(it_up->first, it_up->second);
        }
        if (best_ch.a == 0) {
            for (u64 p : pool_) {
                if (p >= 2 && p < v) {
                    eval_split(p, 0);
                    break;
                }
            }
        }
        if (best_ch.a == 0) {
            if (!in_pool(1)) {
                u64 p0 = pool_[0];
                if (!apply_batch({Step{'/', p0, p0, 1}})) return false;
            }
            if (v == 1) return true;
            if (!in_pool(2)) {
                if (!apply_batch({Step{'+', 1, 1, 2}})) return false;
            }
            eval_split(2, 0);
        }
        if (!in_pool(best_ch.a)) {
            if (!construct_value_fast_split(best_ch.a)) return false;
        }
        pin(best_ch.a);
        bool ok = construct_value_fast_split(best_ch.q);
        if (!ok) {
            unpin(best_ch.a);
            return false;
        }
        pin(best_ch.q);
        if (best_ch.r > 0) {
            ok = construct_value_fast_split(best_ch.r);
            if (ok && !in_pool(v)) {
                u64 prod = best_ch.a * best_ch.q;
                ok = apply_batch({Step{'*', best_ch.a, best_ch.q, prod},
                                  Step{best_ch.minus ? '-' : '+', prod, best_ch.r, v}});
            }
        } else if (!in_pool(v)) {
            ok = apply_batch({Step{'*', best_ch.a, best_ch.q, v}});
        }
        unpin(best_ch.q);
        unpin(best_ch.a);
        return ok;
    }

    double target_affinity(u64 t) const {
        double best = 0.0;
        for (u64 p : pool_) {
            if (p < 2 || p >= t) continue;
            if (t % p == 0) {
                best = std::max(best, 80.0 + std::log2(static_cast<double>(p)));
            }
            u64 q = (t + p / 2) / p;
            if (q >= 1 && q <= 256) {
                u128 prod = static_cast<u128>(q) * p;
                u64 diff = prod > t ? static_cast<u64>(prod - t) : static_cast<u64>(t - prod);
                if (diff == 0) {
                    best = std::max(best, 100.0);
                } else {
                    double ratio = static_cast<double>(diff) / static_cast<double>(t);
                    if (ratio < 0.02) {
                        double sc = 60.0 - 4.0 * std::log2(static_cast<double>(diff) + 1.0);
                        best = std::max(best, sc);
                    }
                }
            }
        }
        if (const size_t* idx = target_idx_.find(t)) {
            const auto& st = targets_[*idx];
            if (st.smooth_active) {
                for (u64 p : pool_) {
                    if (p > 10 && st.S % p == 0) {
                        best = std::max(best, 50.0 + 2.0 * std::log2(static_cast<double>(p)));
                    }
                }
            }
        }
        return best;
    }

    u64 pick_fallback_target() {
        if (cfg_.order_mode == TargetOrder::Shuffled && (rng_() % 3 == 0)) {
            return remaining_[rng_() % remaining_.size()];
        }
        if (cfg_.use_affinity_pick) {
            u64 best_t = remaining_[0];
            double best_sc = -1e18;
            for (u64 t : remaining_) {
                double aff = target_affinity(t);
                double sc = aff - 1.5 * std::log2(static_cast<double>(std::max<u64>(2, t)));
                if (sc > best_sc) {
                    best_sc = sc;
                    best_t = t;
                }
            }
            return best_t;
        }
        return *std::min_element(remaining_.begin(), remaining_.end());
    }

    const Puzzle& pz_;
    const CheapTable& cheap_;
    int regs_;
    const StrategyConfig& cfg_;
    std::mt19937_64 rng_;
    const Timer& timer_;
    double deadline_;
    size_t best_plan_len_;

    std::vector<u64> pool_;
    std::vector<u64> stamp_;
    FastTable pool_pos_;
    FlatMap<int> pins_;
    std::vector<Step> L1_vec_;
    FastTable L1_idx_;
    std::vector<std::pair<u64, uint8_t>> M_;
    bool m_dirty_ = true;
    std::vector<Step> plan_;
    u64 tick_ = 0;
    size_t created_ = 0;

    FlatMap<uint8_t> in_starts_;
    FlatMap<uint8_t> all_targets_;
    std::vector<u64> remaining_;
    std::vector<SmoothTarget> targets_;
    FlatMap<int> failB_;
    FlatMap<int> failB_div_;
    FlatMap<uint8_t> ever_spec_built_;
    FlatMap<size_t> stale_B_;
    FlatMap<size_t> target_idx_;
    double work_ = 0.0;
    double cur_work_limit_ = 1.0e7;
    uint32_t time_check_ctr_ = 0;
    bool soft_timed_out_ = false;
    bool ignore_soft_timeout_ = false;
    bool aborted_ = false;
};

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--seed") opt.seed = std::strtoull(next(), nullptr, 10);
        else if (a == "--restarts") opt.restarts = std::atoi(next());
        else if (a == "--time") opt.time_limit = std::atof(next());
        else if (a == "--regs") opt.regs = std::atoi(next());
        else if (a == "--verbose") opt.verbose = true;
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (opt.restarts < 1) opt.restarts = 1;

    Puzzle pz;
    if (!elf::read_puzzle(std::cin, pz)) {
        std::fprintf(stderr, "bad input\n");
        return 2;
    }

    std::vector<u64> miss = pz.missing_targets();
    if (miss.empty()) return 0;

    Timer total_timer;
    CheapTable cheap;
    cheap.build(pz.starts, pz.limit);

    double mine_deadline = std::min(opt.time_limit * 0.20, 0.8);
    auto all_forms = mine_shifted_forms(pz, miss, cheap, total_timer, mine_deadline);

    if (opt.verbose) {
        std::fprintf(stderr, "[%s] mined forms for %zu targets in %.3fs (cheap1=%zu, cheap2=%zu)\n",
                     pz.id.c_str(), miss.size(), total_timer.elapsed_sec(),
                     cheap.cheap1.size(), cheap.cheap2_only.size());
    }

    std::vector<Step> best_plan;
    bool have_best = false;

    for (int r = 0; r < opt.restarts; ++r) {
        double elapsed = total_timer.elapsed_sec();
        if (r > 0 && elapsed >= opt.time_limit * 0.90) break;

        StrategyConfig cfg;
        if (r == 0) {
            // Restart 0: Descending + Shifted-Smooth B=3..6 + Multiplicative Utility Eviction + Affinity Pick.
            cfg.allow_cheap2 = true;
            cfg.cheap2_penalty = 0.25;
            cfg.gcd_bonus_weight = 0.75;
            cfg.use_repair = false;
            cfg.direct_B3_before_smooth = true;
            cfg.order_mode = TargetOrder::Descending;
            cfg.evict_mode = 2;
            cfg.use_affinity_pick = true;
            cfg.aff_weight = 1.0;
            cfg.bfull = 5;
            cfg.noise = 0.0;
        } else if (r == 1) {
            // Restart 1: Ascending + Shifted-Smooth B=3..6 + Multiplicative Utility Eviction + Min-Target Pick.
            cfg.allow_cheap2 = true;
            cfg.cheap2_penalty = 0.25;
            cfg.gcd_bonus_weight = 0.75;
            cfg.use_repair = false;
            cfg.direct_B3_before_smooth = true;
            cfg.order_mode = TargetOrder::Ascending;
            cfg.evict_mode = 2;
            cfg.use_affinity_pick = false;
            cfg.aff_weight = 1.2;
            cfg.bfull = 5;
            cfg.noise = 0.0;
        } else if (r == 2) {
            // Restart 2: Descending + Unified L1 Re-Pair Compression + Multiplicative Utility Eviction.
            cfg.allow_cheap2 = true;
            cfg.cheap2_penalty = 0.30;
            cfg.gcd_bonus_weight = 0.85;
            cfg.use_repair = true;
            cfg.repair_before_B3_thresh = 2.8;
            cfg.repair_before_B4_thresh = 1.8;
            cfg.repair_before_B5_thresh = 1.3;
            cfg.direct_B3_before_smooth = false;
            cfg.order_mode = TargetOrder::Descending;
            cfg.evict_mode = 2;
            cfg.use_affinity_pick = true;
            cfg.aff_weight = 1.4;
            cfg.bfull = 5;
            cfg.noise = 0.0;
        } else if (r == 3) {
            // Restart 3: Descending + Shifted-Smooth B=3..6 + Gentle GCD LRU Boost (classic mult_factor diversity!).
            cfg.allow_cheap2 = true;
            cfg.cheap2_penalty = 0.25;
            cfg.gcd_bonus_weight = 0.80;
            cfg.use_repair = false;
            cfg.direct_B3_before_smooth = true;
            cfg.order_mode = TargetOrder::Descending;
            cfg.evict_mode = 1;
            cfg.gentle_gcd_lru_boost = true;
            cfg.use_affinity_pick = false;
            cfg.aff_weight = 1.0;
            cfg.bfull = 5;
            cfg.noise = 0.0;
        } else {
            cfg.allow_cheap2 = (r % 2 == 0);
            cfg.cheap2_penalty = 0.25 * (r % 3);
            cfg.gcd_bonus_weight = 0.65 + 0.15 * (r % 3);
            cfg.use_repair = (r % 2 == 1);
            cfg.repair_before_B3_thresh = 2.6 + 0.7 * (r % 3);
            cfg.repair_before_B4_thresh = 1.6 + 0.4 * (r % 3);
            cfg.repair_before_B5_thresh = 1.2 + 0.2 * (r % 3);
            cfg.direct_B3_before_smooth = (r % 2 == 0);
            cfg.order_mode = (r % 3 == 0) ? TargetOrder::HubFirst : TargetOrder::Shuffled;
            cfg.evict_mode = (r % 4 == 0) ? 1 : 2;
            cfg.gentle_gcd_lru_boost = (r % 2 == 1);
            cfg.use_affinity_pick = (r % 2 == 0);
            cfg.aff_weight = 1.0 + 0.25 * (r % 3);
            cfg.bfull = 5;
            cfg.noise = 0.25;
        }

        double rem_time = std::max(0.15, opt.time_limit - elapsed);
        int rem_restarts = opt.restarts - r;
        double slice = rem_time / static_cast<double>(rem_restarts);
        if (r == 0 && opt.restarts > 1) slice = std::max(slice, opt.time_limit * 0.42);
        double restart_deadline = std::min(opt.time_limit * 0.96, elapsed + std::max(0.18, slice));
        if (!have_best) restart_deadline = std::max(restart_deadline, opt.time_limit * 0.85);
        if (r == opt.restarts - 1) restart_deadline = opt.time_limit * 0.96;

        u64 rseed = elf::splitmix64(opt.seed + static_cast<u64>(r) * 0x9e3779b97f4a7c15ULL + 17ULL);
        std::mt19937_64 rng(rseed);

        auto chosen = select_forms(pz, all_forms, cheap, cfg, rng);

        size_t best_len = have_best ? best_plan.size() : static_cast<size_t>(-1);
        FactorEngine engine(pz, cheap, opt.regs, cfg, rseed, total_timer, restart_deadline, best_len);
        auto raw_plan = engine.run(miss, all_forms, chosen);

        if (engine.was_aborted()) {
            if (opt.verbose) {
                std::fprintf(stderr, "  restart %d: ABORTED (>= best %zu, %.3fs)\n",
                             r, best_plan.size(), total_timer.elapsed_sec());
            }
            continue;
        }

        auto verified = elf::verify_and_prune_plan(pz, raw_plan, opt.regs);
        if (verified) {
            size_t pruned_sz = verified->size();
            if (!have_best || pruned_sz < best_plan.size()) {
                best_plan = std::move(*verified);
                have_best = true;
            }
            if (opt.verbose) {
                std::fprintf(stderr, "  restart %d: raw=%zu pruned=%zu best=%zu (%.3fs)\n",
                             r, raw_plan.size(), pruned_sz, best_plan.size(),
                             total_timer.elapsed_sec());
            }
        } else if (opt.verbose) {
            std::fprintf(stderr, "  restart %d: FAILED verify (raw=%zu, %.3fs)\n",
                         r, raw_plan.size(), total_timer.elapsed_sec());
        }
    }

    if (!have_best) {
        std::fprintf(stderr, "mult_factor: failed to produce valid plan for %s\n", pz.id.c_str());
        return 1;
    }

    elf::emit_plan(best_plan);
    return 0;
}
