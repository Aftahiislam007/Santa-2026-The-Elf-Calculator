// Upgraded pool-based greedy search engine (greedy_v2).
//
// Improvements over greedy.cpp:
//   1. Utility-aware pool eviction (protects 1, small constants, affine/divisor
//      hubs for uncovered targets, scale-bucket diversity; evicts dead targets
//      and overcrowded scale duplicates first).
//   2. Strict inner-loop --time enforcement with fast fallback completion and
//      branch-and-bound early abort across restarts.
//   3. Richer decomposition operators:
//        - Exact-division lottery (v * d <= limit -> solve(v*d, B-1) / d)
//        - Both a*q + r and a*q - r in decomp AND fallback construct()
//        - Exact L1 factorization (v % a == 0 for a in L1) and balanced factor
//          splits via Pollard's Rho + 1-step smooth shifts v +- c
//        - Immediate returns on c01(r) <= 1 and deduplicated a <= sqrt(v) scan
//   4. Multi-restart portfolio (default 4 restarts) with affinity-guided target
//      ordering, randomized L2 voting/pick/split tie-breaking, and exact
//      elf::verify_and_prune_plan verification.
//
// I/O protocol: see elfcalc/solvers/engine.py.

#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
    double timeLimit = 4.0;
    int bfull = 5;
    int bmax = 8;
    int k = 24;
    double work = 1.5e8;
    int regs = 48;
    bool verbose = false;
};

// Ultra-fast open-addressed u64 -> u32 table (1-multiply Fibonacci hash) for hot inner loops.
struct FastTable {
    int bits = 12;
    size_t count = 0;
    std::vector<u64> keys;
    std::vector<uint32_t> vals;

    explicit FastTable(int initBits = 12)
        : bits(initBits),
          keys(size_t(1) << initBits, 0),
          vals(size_t(1) << initBits, 0) {}

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

struct L1Entry {
    u64 v = 0;
    u64 a = 0;
    u64 b = 0;
    char op = 0;
};

struct Cand {
    u64 r = 0;
    u64 a = 0;
    u64 q = 0;
    uint8_t ca = 0;
    uint8_t cq = 0;
    uint8_t kind = 0;  // 0: a*q+r, 1: a*q-r, 2: a+r, 3: a-r
    int rem = 0;
};

inline int scale_bucket(u64 v) {
    if (v <= 1) return 0;
    int b = static_cast<int>(std::log2(static_cast<double>(v)) / 2.5);
    if (b < 0) b = 0;
    if (b > 31) b = 31;
    return b;
}

inline bool is_smooth_300(u64 n) {
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

struct Engine {
    const Options& opt;
    const Timer& timer;
    double deadline = 1e9;
    size_t bestPlanLen = static_cast<size_t>(-1);
    int mode = 0;  // 0: det utility, 1: affinity-ordered, 2+: randomized
    double affWeight = 1.0;

    u64 limit = 0;
    std::vector<u64> pool;
    std::vector<u64> stamp;
    FlatMap<int> pins;
    FastTable poolIdx;
    FlatMap<uint8_t> allTargets;
    std::vector<L1Entry> l1;
    FastTable l1Idx;
    std::vector<Step> plan;
    std::vector<u64> remaining;
    FlatMap<int> failB;
    FlatMap<int> failBDiv;
    std::vector<std::pair<u64, uint8_t>> M;
    bool mDirty = true;

    u64 tick = 0;
    size_t created = 0;
    size_t evictions = 0;
    double work = 0.0;
    double curWorkLimit = 1.5e8;
    uint32_t timeCheckCtr = 0;
    bool softTimedOut = false;
    bool aborted = false;
    std::mt19937_64 rng;

    Engine(const Options& o, const Timer& tmr, double dl, size_t bestLen,
           int m, u64 lim, u64 seed)
        : opt(o),
          timer(tmr),
          deadline(dl),
          bestPlanLen(bestLen),
          mode(m),
          affWeight(m == 1 ? 1.5 : 1.0),
          limit(lim),
          pins(64),
          poolIdx(8),
          allTargets(128),
          l1Idx(13),
          failB(256),
          failBDiv(256),
          curWorkLimit(o.work),
          rng(seed) {}

    bool checkTime() {
        if (softTimedOut) return true;
        if ((++timeCheckCtr & 63U) == 0) {
            if (timer.elapsed_sec() >= deadline) {
                softTimedOut = true;
            }
        }
        return softTimedOut;
    }

    bool shouldAbort() const {
        return aborted || (bestPlanLen != static_cast<size_t>(-1) && plan.size() >= bestPlanLen);
    }

    u64 eval(char op, u64 a, u64 b) const {
        return elf::apply_op(op, a, b, limit);
    }

    bool inPool(u64 x) const { return poolIdx.contains(x); }

    int c01(u64 x) const { return inPool(x) ? 0 : (l1Idx.contains(x) ? 1 : 2); }

    void addL1(u64 v, char op, u64 a, u64 b) {
        if (v == 0 || inPool(v)) return;
        if (l1Idx.insert(v, static_cast<uint32_t>(l1.size()))) {
            l1.push_back(L1Entry{v, a, b, op});
        }
    }

    void linkValue(u64 v) {
        for (u64 p : pool) {
            addL1(eval('+', v, p), '+', v, p);
            addL1(eval('*', v, p), '*', v, p);
            if (v > p) {
                addL1(v - p, '-', v, p);
                if (v % p == 0) addL1(v / p, '/', v, p);
            } else if (p > v) {
                addL1(p - v, '-', p, v);
                if (p % v == 0) addL1(p / v, '/', p, v);
            } else {
                addL1(1, '/', v, v);
                break;
            }
        }
    }

    void addValue(u64 v, bool link) {
        if (!poolIdx.insert(v, static_cast<uint32_t>(pool.size()))) return;
        pool.push_back(v);
        stamp.push_back(++tick);
        ++created;
        if (link) linkValue(v);
        mDirty = true;
    }

    void reindex() {
        poolIdx.clear();
        for (size_t j = 0; j < pool.size(); ++j) {
            poolIdx.insert(pool[j], static_cast<uint32_t>(j));
        }
    }

    void rebuild() {
        std::vector<u64> old;
        old.swap(pool);
        poolIdx.clear();
        l1.clear();
        l1Idx.clear();
        for (u64 v : old) {
            poolIdx.insert(v, static_cast<uint32_t>(pool.size()));
            pool.push_back(v);
            linkValue(v);
        }
        mDirty = true;
    }

    // Utility-aware eviction: keeps 1, small constants, affine/divisor hubs,
    // and underrepresented scale buckets; evicts dead finished targets first.
    size_t pickVictim(const FlatMap<size_t>& lastNeed, size_t stepIdx) const {
        int bcount[32] = {};
        for (u64 v : pool) {
            ++bcount[scale_bucket(v)];
        }

        FlatMap<uint8_t> remSet(remaining.size() * 2 + 8);
        for (u64 t : remaining) remSet.insert(t, 1);

        size_t victim = pool.size();
        double victimScore = 0.0;
        const double invTick = 1.0 / static_cast<double>(tick + 1);

        const u64 rootLim = elf::isqrt64(limit);
        for (size_t j = 0; j < pool.size(); ++j) {
            u64 v = pool[j];
            if (const size_t* need = lastNeed.find(v); need && *need > stepIdx) continue;
            if (pins.contains(v)) continue;

            double sc = 0.0;
            if (v == 1) {
                sc += 1.0e6;
            } else if (v <= 64) {
                sc += 160.0 / (1.0 + std::log2(static_cast<double>(v)));
            }

            int b = scale_bucket(v);
            if (v <= rootLim) {
                if (bcount[b] == 1) {
                    sc += 55.0;
                } else if (bcount[b] == 2) {
                    sc += 20.0;
                } else if (bcount[b] >= 7) {
                    sc -= 12.0 * static_cast<double>(bcount[b] - 6);
                }
            } else {
                sc -= 40.0;
            }

            int divHits = 0;
            int affHits = 0;
            int nearHits = 0;
            int gcdHits = 0;
            for (u64 t : remaining) {
                if (v > 1 && t % v == 0) ++divHits;
                u64 diff = (t > v) ? (t - v) : (v - t);
                u64 nearTol = std::max<u64>(10000ULL, t / 100ULL);
                if (diff <= nearTol) ++nearHits;
                if (v < t && v >= 2) {
                    u64 q = (t + v / 2) / v;
                    if (q >= 1 && q <= 200) {
                        u128 prod = static_cast<u128>(q) * v;
                        u64 adiff = prod > t ? static_cast<u64>(prod - t)
                                             : static_cast<u64>(t - prod);
                        u64 affTol = std::max<u64>(5000ULL, t / 250ULL);
                        if (adiff <= affTol) ++affHits;
                    }
                }
                if (v > 500 && std::gcd(v, t) > 500) ++gcdHits;
            }

            sc += std::min(300.0, divHits * 85.0);
            sc += std::min(360.0, affHits * 120.0 * affWeight);
            sc += std::min(300.0, nearHits * 160.0);
            sc += std::min(120.0, gcdHits * 30.0);

            if (allTargets.contains(v) && !remSet.contains(v) &&
                divHits == 0 && affHits == 0 && nearHits == 0 && gcdHits == 0) {
                sc -= 90.0;
            }

            // Strong LRU recency component keeps active working set hot.
            sc += 45.0 * (static_cast<double>(stamp[j]) * invTick);

            if (victim == pool.size() || sc < victimScore) {
                victim = j;
                victimScore = sc;
            }
        }
        return victim;
    }

    bool apply(const std::vector<Step>& steps) {
        FlatMap<size_t> lastNeed(steps.size() * 2 + 8);
        for (size_t i = 0; i < steps.size(); ++i) {
            lastNeed.insert_or_assign(steps[i].a, i);
            lastNeed.insert_or_assign(steps[i].b, i);
        }
        bool evicted = false;
        for (size_t i = 0; i < steps.size(); ++i) {
            const Step& s = steps[i];
            const uint32_t* ia = poolIdx.find(s.a);
            const uint32_t* ib = poolIdx.find(s.b);
            if (!ia || !ib) return false;
            u64 r = eval(s.op, s.a, s.b);
            if (r == 0) return false;
            stamp[*ia] = stamp[*ib] = ++tick;
            if (inPool(r)) continue;
            plan.push_back(Step{s.op, s.a, s.b, r});
            if (opt.regs > 0 && pool.size() >= static_cast<size_t>(opt.regs)) {
                size_t victim = pickVictim(lastNeed, i);
                if (victim == pool.size()) return false;
                pool.erase(pool.begin() + static_cast<ptrdiff_t>(victim));
                stamp.erase(stamp.begin() + static_cast<ptrdiff_t>(victim));
                reindex();
                evicted = true;
                ++evictions;
            }
            addValue(r, !evicted);
        }
        if (evicted) rebuild();
        failB.clear();
        failBDiv.clear();
        remaining.erase(std::remove_if(remaining.begin(), remaining.end(),
                                       [&](u64 t) { return inPool(t); }),
                        remaining.end());
        return true;
    }

    void emitL1(u64 x, std::vector<Step>& out) const {
        const uint32_t* idx = l1Idx.find(x);
        const L1Entry& e = l1[*idx];
        out.push_back(Step{e.op, e.a, e.b, e.v});
    }

    void emit01(u64 x, int c, std::vector<Step>& out) const {
        if (c == 1) emitL1(x, out);
    }

    void pinVal(u64 v) {
        if (int* p = pins.find(v)) {
            ++(*p);
        } else {
            pins.insert(v, 1);
        }
    }

    void unpinVal(u64 v) {
        if (int* p = pins.find(v)) {
            if (--(*p) <= 0) pins.erase(v);
        }
    }

    bool find2(u64 v, std::vector<Step>* out, std::vector<u64>* collect) {
        work += static_cast<double>(pool.size()) * 6.0;
        auto hit = [&](u64 x, char op, bool xFirst, u64 c) -> bool {
            if (!l1Idx.contains(x)) return false;
            if (collect) {
                collect->push_back(x);
                return false;
            }
            emitL1(x, *out);
            out->push_back(xFirst ? Step{op, x, c, v} : Step{op, c, x, v});
            return true;
        };
        for (u64 c : pool) {
            if (v > c && hit(v - c, '+', true, c)) return true;
            if (v <= limit - c && hit(v + c, '-', true, c)) return true;
            if (c > v) {
                if (hit(c - v, '-', false, c)) return true;
                if (c % v == 0 && hit(c / v, '/', false, c)) return true;
            }
            if (c > 1) {
                if (v % c == 0 && hit(v / c, '*', true, c)) return true;
                u128 p = static_cast<u128>(v) * c;
                if (p <= limit && hit(static_cast<u64>(p), '/', true, c)) return true;
            }
        }
        if ((v & 1ULL) == 0 && hit(v / 2, '+', true, v / 2)) return true;
        u64 s = elf::isqrt64(v);
        if (s * s == v && hit(s, '*', true, s)) return true;
        return collect && !collect->empty();
    }

    bool find3(u64 v, std::vector<Step>& out) {
        if (checkTime()) return false;
        work += static_cast<double>(l1.size()) * 6.0;
        auto both = [&](u64 x, u64 y, Step last) -> bool {
            if (!l1Idx.contains(y)) return false;
            emitL1(x, out);
            if (y != x) emitL1(y, out);
            out.push_back(last);
            return true;
        };
        for (const L1Entry& e : l1) {
            u64 x = e.v;
            if (v > x && both(x, v - x, Step{'+', x, v - x, v})) return true;
            if (x > v) {
                if (both(x, x - v, Step{'-', x, x - v, v})) return true;
                if (x % v == 0 && both(x, x / v, Step{'/', x, x / v, v})) return true;
            }
            if (v <= limit - x && both(x, v + x, Step{'-', v + x, x, v})) return true;
            if (x > 1) {
                if (v % x == 0 && both(x, v / x, Step{'*', x, v / x, v})) return true;
                u128 p = static_cast<u128>(v) * x;
                if (p <= limit && both(x, static_cast<u64>(p), Step{'/', static_cast<u64>(p), x, v}))
                    return true;
            }
        }
        auto chain = [&](u64 y, Step last) -> bool {
            size_t mark = out.size();
            if (!find2(y, &out, nullptr)) {
                out.resize(mark);
                return false;
            }
            out.push_back(last);
            return true;
        };
        for (u64 c : pool) {
            if (v > c && chain(v - c, Step{'+', v - c, c, v})) return true;
            if (v <= limit - c && chain(v + c, Step{'-', v + c, c, v})) return true;
            if (c > v) {
                if (chain(c - v, Step{'-', c, c - v, v})) return true;
                if (c % v == 0 && chain(c / v, Step{'/', c, c / v, v})) return true;
            }
            if (c > 1) {
                if (v % c == 0 && chain(v / c, Step{'*', v / c, c, v})) return true;
                u128 p = static_cast<u128>(v) * c;
                if (p <= limit && chain(static_cast<u64>(p), Step{'/', static_cast<u64>(p), c, v}))
                    return true;
            }
        }
        return false;
    }

    void buildM() {
        if (!mDirty) return;
        M.clear();
        M.reserve(pool.size() + l1.size());
        for (u64 p : pool) M.emplace_back(p, static_cast<uint8_t>(0));
        for (const L1Entry& e : l1) {
            if (!inPool(e.v)) M.emplace_back(e.v, static_cast<uint8_t>(1));
        }
        std::sort(M.begin(), M.end());
        mDirty = false;
    }

    bool trySplit(int B, u64 a, int ca, u64 q, u64 r, bool minus, std::vector<Step>& out) {
        if (q == 0 || r == 0) return false;
        int base = ca + 2;
        int lq = c01(q);
        int lr = c01(r);
        if (base + lq + lr > B) return false;
        size_t mark = out.size();
        auto finish = [&]() -> bool {
            emit01(a, ca, out);
            out.push_back(Step{'*', a, q, a * q});
            u64 res = minus ? (a * q - r) : (a * q + r);
            out.push_back(Step{minus ? '-' : '+', a * q, r, res});
            return true;
        };
        int capR = std::min(2, B - base - lq);
        if (solve(r, capR, out, false)) {
            int cr = static_cast<int>(out.size() - mark);
            if (solve(q, B - base - cr, out, false)) return finish();
            out.resize(mark);
            return false;
        }
        out.resize(mark);
        int capQ = std::min(2, B - base - std::max(lr, capR + 1));
        if (capQ < lq) return false;
        if (solve(q, capQ, out, false)) {
            int cq = static_cast<int>(out.size() - mark);
            if (solve(r, B - base - cq, out, false)) return finish();
        }
        out.resize(mark);
        return false;
    }

    bool decomp(u64 v, int B, std::vector<Step>& out, bool allowDiv) {
        if (checkTime() || work > curWorkLimit) return false;
        buildM();
        work += static_cast<double>(M.size()) * 14.0;
        std::vector<Cand> cands;
        cands.reserve(256);

        // 1. M * M +- r (full scan a < v so no (a, q) neighborhood is missed)
        for (const auto& [a, ca] : M) {
            if (a < 2) continue;
            if (a >= v) break;
            auto it = std::lower_bound(M.begin(), M.end(),
                                       std::make_pair(v / a, static_cast<uint8_t>(0)));
            for (int d = -1; d <= 1; ++d) {
                auto jt = it + d;
                if (jt < M.begin() || jt >= M.end() || jt->first < 2) continue;
                u64 q = jt->first;
                int cost0 = ca + jt->second + 1;
                u128 p = static_cast<u128>(a) * q;
                if (p > limit) continue;
                u64 prod = static_cast<u64>(p);
                if (prod == v) {
                    if (cost0 <= B) {
                        emit01(a, ca, out);
                        if (q != a) emit01(q, jt->second, out);
                        out.push_back(Step{'*', a, q, v});
                        return true;
                    }
                    continue;
                }
                if (cost0 >= B) continue;
                u64 r = prod > v ? (prod - v) : (v - prod);
                int rem = B - cost0 - 1;
                int cr = c01(r);
                if (rem < cr) continue;
                if (cr <= 1) {
                    emit01(r, cr, out);
                    emit01(a, ca, out);
                    if (q != a) emit01(q, jt->second, out);
                    out.push_back(Step{'*', a, q, prod});
                    out.push_back(Step{prod > v ? '-' : '+', prod, r, v});
                    return true;
                }
                cands.push_back(Cand{r, a, q, ca, jt->second,
                                     static_cast<uint8_t>(prod > v ? 1 : 0), rem});
            }
        }

        // 2. Nearby M +- r
        auto it = std::lower_bound(M.begin(), M.end(), std::make_pair(v, static_cast<uint8_t>(0)));
        for (int d = -10; d < 10; ++d) {
            auto jt = it + d;
            if (jt < M.begin() || jt >= M.end() || jt->first == v) continue;
            u64 x = jt->first;
            u64 r = x > v ? (x - v) : (v - x);
            int rem = B - jt->second - 1;
            int cr = c01(r);
            if (rem < cr) continue;
            if (cr <= 1) {
                emit01(r, cr, out);
                emit01(x, jt->second, out);
                out.push_back(Step{x > v ? '-' : '+', x, r, v});
                return true;
            }
            cands.push_back(Cand{r, x, 0, jt->second, 0,
                                 static_cast<uint8_t>(x > v ? 3 : 2), rem});
        }

        std::sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) {
            if (x.r != y.r) return x.r < y.r;
            return x.rem > y.rem;
        });

        int tried = 0;
        u64 lastR = 0;
        for (const Cand& c : cands) {
            if (c.r == lastR) continue;
            if (++tried > opt.k || work > curWorkLimit || checkTime()) break;
            lastR = c.r;
            size_t mark = out.size();
            if (!solve(c.r, c.rem, out, false)) {
                out.resize(mark);
                continue;
            }
            emit01(c.a, c.ca, out);
            if (c.kind < 2) {
                if (c.q != c.a) emit01(c.q, c.cq, out);
                out.push_back(Step{'*', c.a, c.q, c.a * c.q});
                out.push_back(Step{c.kind ? '-' : '+', c.a * c.q, c.r, v});
            } else {
                out.push_back(Step{c.kind == 3 ? '-' : '+', c.a, c.r, v});
            }
            return true;
        }

        // 3. Exact & remainder splits with a in pool (both +r and -r)
        for (u64 a : pool) {
            if (a < 2 || a >= v) continue;
            if (work > curWorkLimit || checkTime()) return false;
            u64 q = v / a;
            u64 r = v % a;
            if (r == 0) {
                size_t mark = out.size();
                if (solve(q, B - 1, out, false)) {
                    out.push_back(Step{'*', q, a, v});
                    return true;
                }
                out.resize(mark);
                continue;
            }
            if (trySplit(B, a, 0, q, r, false, out)) return true;
            if (static_cast<u128>(a) * (q + 1) <= limit &&
                trySplit(B, a, 0, q + 1, a - r, true, out)) {
                return true;
            }
        }

        // 4. Exact & remainder splits with a in L1 (INCLUDING r == 0!)
        for (const L1Entry& e : l1) {
            u64 a = e.v;
            if (a < 2 || a >= v || inPool(a)) continue;
            if (work > curWorkLimit || checkTime()) return false;
            u64 q = v / a;
            u64 r = v % a;
            if (r == 0) {
                if (B >= 3 && c01(q) <= B - 2) {
                    size_t mark = out.size();
                    if (solve(q, B - 2, out, false)) {
                        emitL1(a, out);
                        out.push_back(Step{'*', q, a, v});
                        return true;
                    }
                    out.resize(mark);
                }
                continue;
            }
            if (std::min(c01(q), c01(r)) <= 1 && trySplit(B, a, 1, q, r, false, out)) return true;
            if (static_cast<u128>(a) * (q + 1) <= limit &&
                std::min(c01(q + 1), c01(a - r)) <= 1 &&
                trySplit(B, a, 1, q + 1, a - r, true, out)) {
                return true;
            }
        }

        // 5. Fast exact-division lottery: v = (v * d) / d for d in pool (zero bloat!)
        if (allowDiv && (B == 4 || B == 5) && !checkTime() && work < curWorkLimit * 0.35) {
            std::vector<u64> mults;
            mults.reserve(pool.size());
            for (u64 d : pool) {
                if (d >= 2 && (B == 4 || d <= 32) && static_cast<u128>(v) * d <= limit) {
                    mults.push_back(d);
                }
            }
            std::sort(mults.begin(), mults.end());
            int maxDiv = (B == 4) ? 8 : 4;
            int triedDiv = 0;
            for (u64 d : mults) {
                if (++triedDiv > maxDiv || work > curWorkLimit || checkTime()) break;
                u64 vd = v * d;
                size_t mark = out.size();
                if (solve(vd, B - 1, out, false)) {
                    out.push_back(Step{'/', vd, d, v});
                    return true;
                }
                out.resize(mark);
            }
        }

        // 6. Smooth exact factorization split v = f1 * f2 (fast O(60) check!)
        if (allowDiv && B == 5 && v > 1000 && !checkTime() &&
            work < curWorkLimit * 0.5 && is_smooth_300(v)) {
            u64 rootV = elf::isqrt64(v);
            auto fac = elf::factorize(v);
            auto divs = elf::get_divisors(fac, 256);
            std::vector<u64> balanced;
            for (u64 d : divs) {
                if (d >= 3 && d <= rootV) balanced.push_back(d);
            }
            std::sort(balanced.begin(), balanced.end(), std::greater<u64>());
            int triedFac = 0;
            for (u64 f1 : balanced) {
                if (++triedFac > 4 || work > curWorkLimit || checkTime()) break;
                u64 f2 = v / f1;
                int b1 = std::min(3, B - 2);
                size_t mark = out.size();
                if (solve(f1, b1, out, false)) {
                    int c1 = static_cast<int>(out.size() - mark);
                    if (solve(f2, B - 1 - c1, out, false)) {
                        out.push_back(Step{'*', f1, f2, v});
                        return true;
                    }
                }
                out.resize(mark);
            }
        }

        return false;
    }

    bool solve(u64 v, int B, std::vector<Step>& out, bool allowDiv = true) {
        int c = c01(v);
        if (c == 0) return true;
        if (B <= 0) return false;
        if (c == 1) {
            emitL1(v, out);
            return true;
        }
        if (B == 1) return false;
        if (checkTime() || work > curWorkLimit) return false;

        int from = 2;
        if (const int* fbd = failBDiv.find(v)) from = std::max(from, *fbd + 1);
        if (!allowDiv) {
            if (const int* fb = failB.find(v)) from = std::max(from, *fb + 1);
        }
        size_t mark = out.size();
        for (int b = from; b <= B; ++b) {
            if (work > curWorkLimit || checkTime()) return false;
            bool ok = false;
            if (b == 2) {
                ok = find2(v, &out, nullptr);
            } else if (b == 3) {
                ok = find3(v, out);
            } else {
                ok = decomp(v, b, out, allowDiv);
            }
            if (ok) return true;
            out.resize(mark);
            if (work > curWorkLimit || checkTime()) return false;
            if (allowDiv) {
                const int* fbd = failBDiv.find(v);
                if (!fbd || b > *fbd) failBDiv.insert_or_assign(v, b);
            } else {
                const int* fb = failB.find(v);
                if (!fb || b > *fb) failB.insert_or_assign(v, b);
            }
        }
        return false;
    }

    double estCost(u64 x) const {
        int c = c01(x);
        if (c <= 1) return static_cast<double>(c);
        return 2.0 + 0.42 * std::log2(static_cast<double>(x));
    }

    bool construct(u64 v, int depth = 0) {
        while (!inPool(v)) {
            if (shouldAbort()) {
                aborted = true;
                return false;
            }
            int curBmax = softTimedOut ? 5 : (depth > 0 ? std::min(opt.bmax, 6) : opt.bmax);
            curWorkLimit = softTimedOut ? 4.0e6 : opt.work;
            std::vector<Step> steps;
            bool ok = false;
            for (int B = 1; B <= curBmax && !ok; ++B) {
                steps.clear();
                work = 0.0;
                ok = solve(v, B, steps, depth == 0 && !softTimedOut);
                if (softTimedOut && B >= 5) break;
            }
            if (ok) {
                if (!apply(steps)) return false;
                return true;
            }

            // Try exact factorization ONLY when v is smooth, has a substantial divisor in M,
            // or v +- c (c in pool) is 300-smooth!
            if (v > 256 && depth < 6) {
                u64 root = elf::isqrt64(v);
                u64 minGoodDiv = std::max<u64>(100ULL, static_cast<u64>(std::cbrt(static_cast<double>(v))));
                // (a) Exact divisor already in M (zero Pollard's Rho!)
                buildM();
                u64 bestMDiv = 0;
                for (const auto& [a, ca] : M) {
                    if (a < minGoodDiv) continue;
                    if (a > root) break;
                    if (v % a == 0) bestMDiv = a;
                }
                if (bestMDiv > 0) {
                    u64 f2 = v / bestMDiv;
                    if (!construct(bestMDiv, depth + 1)) return false;
                    pinVal(bestMDiv);
                    bool ok2 = construct(f2, depth + 1);
                    if (ok2) ok2 = apply({Step{'*', bestMDiv, f2, v}});
                    unpinVal(bestMDiv);
                    if (!ok2) return false;
                    return true;
                }
                // (b) 300-smooth exact balanced factorization
                if (is_smooth_300(v)) {
                    auto fac = elf::factorize(v);
                    auto divs = elf::get_divisors(fac, 512);
                    u64 bestFac = 0;
                    for (u64 d : divs) {
                        if (d >= 3 && d <= root) {
                            if (bestFac == 0 || d > bestFac) bestFac = d;
                        }
                    }
                    if (bestFac > 0) {
                        u64 f2 = v / bestFac;
                        if (!construct(bestFac, depth + 1)) return false;
                        pinVal(bestFac);
                        bool ok2 = construct(f2, depth + 1);
                        if (ok2) ok2 = apply({Step{'*', bestFac, f2, v}});
                        unpinVal(bestFac);
                        if (!ok2) return false;
                        return true;
                    }
                }
                // (c) 1-step 300-smooth shift v +- c for small c in pool
                if (depth == 0) {
                    u64 bestShift = 0;
                    bool shiftPlus = false;
                    u64 bestC = 0;
                    for (u64 c : pool) {
                        if (c == 0 || c >= v / 100ULL) continue;
                        if (v <= limit - c && is_smooth_300(v + c)) {
                            bestShift = v + c;
                            shiftPlus = true;
                            bestC = c;
                            break;
                        }
                        if (v > c && is_smooth_300(v - c)) {
                            bestShift = v - c;
                            shiftPlus = false;
                            bestC = c;
                            break;
                        }
                    }
                    if (bestShift > 0) {
                        pinVal(bestC);
                        bool okS = construct(bestShift, depth + 1);
                        if (okS) {
                            okS = apply({Step{shiftPlus ? '-' : '+', bestShift, bestC, v}});
                        }
                        unpinVal(bestC);
                        if (!okS) return false;
                        return true;
                    }
                }
            }

            // General split: evaluate both a*q + r and a*(q+1) - (a-r) around sqrt(v).
            buildM();
            u64 root = elf::isqrt64(v);
            struct SplitChoice {
                u64 a = 0;
                u64 q = 0;
                u64 r = 0;
                bool minus = false;
                double score = 1e18;
            };
            std::vector<SplitChoice> choices;
            choices.reserve(48);

            auto evalSplit = [&](u64 a, uint8_t ca) {
                if (a < 2 || a >= v) return;
                u64 q = v / a;
                u64 r = v % a;
                if (q == 0) return;
                double scPlus = ca + estCost(q) + (r ? (estCost(r) + 1.0) : 0.0);
                choices.push_back(SplitChoice{a, q, r, false, scPlus});
                if (r > 0 && static_cast<u128>(a) * (q + 1) <= limit) {
                    u64 r2 = a - r;
                    double scMinus = ca + estCost(q + 1) + estCost(r2) + 1.0;
                    choices.push_back(SplitChoice{a, q + 1, r2, true, scMinus});
                }
            };

            auto it = std::upper_bound(M.begin(), M.end(),
                                       std::make_pair(root, static_cast<uint8_t>(255)));
            auto itDown = it;
            for (int seen = 0; itDown != M.begin() && seen < 24; ++seen) {
                --itDown;
                if (itDown->first < 2) break;
                evalSplit(itDown->first, itDown->second);
            }
            auto itUp = it;
            for (int seen = 0; itUp != M.end() && seen < 8; ++seen, ++itUp) {
                if (itUp->first >= v) break;
                evalSplit(itUp->first, itUp->second);
            }

            if (choices.empty()) {
                for (const auto& [a, ca] : M) {
                    if (a >= 2 && a < v) {
                        evalSplit(a, ca);
                        break;
                    }
                }
            }
            if (choices.empty()) {
                if (inPool(1)) {
                    // v must be 1 or unreachable; handle v == 2 via 1 + 1
                    if (v == 2) {
                        if (!apply({Step{'+', 1, 1, 2}})) return false;
                        return true;
                    }
                    return false;
                }
                if (!apply({Step{'/', pool[0], pool[0], 1}})) return false;
                continue;
            }

            std::sort(choices.begin(), choices.end(),
                      [](const SplitChoice& x, const SplitChoice& y) { return x.score < y.score; });

            size_t pickIdx = 0;
            if (mode >= 2 && choices.size() > 1 && (rng() % 3 == 0)) {
                size_t topK = std::min<size_t>(3, choices.size());
                pickIdx = rng() % topK;
            }
            const SplitChoice chosen = choices[pickIdx];

            if (!construct(chosen.a, depth + 1)) return false;
            pinVal(chosen.a);
            bool okQ = construct(chosen.q, depth + 1);
            if (!okQ) {
                unpinVal(chosen.a);
                return false;
            }
            pinVal(chosen.q);
            bool okAll = true;
            if (chosen.r > 0) {
                okAll = construct(chosen.r, depth + 1);
                if (okAll) {
                    u64 prod = chosen.a * chosen.q;
                    okAll = apply({Step{'*', chosen.a, chosen.q, prod},
                                   Step{chosen.minus ? '-' : '+', prod, chosen.r, v}});
                }
            } else {
                okAll = apply({Step{'*', chosen.a, chosen.q, v}});
            }
            unpinVal(chosen.q);
            unpinVal(chosen.a);
            if (!okAll) return false;
        }
        return true;
    }

    // Score how affine/GCD-similar a remaining target is to the current pool.
    double targetAffinity(u64 t) const {
        double best = 0.0;
        for (u64 p : pool) {
            if (p < 2 || p >= t) continue;
            if (t % p == 0) best = std::max(best, 100.0 + std::log2(static_cast<double>(p)));
            u64 q = (t + p / 2) / p;
            if (q >= 1 && q <= 200) {
                u128 prod = static_cast<u128>(q) * p;
                u64 diff = prod > t ? static_cast<u64>(prod - t) : static_cast<u64>(t - prod);
                if (diff <= std::max<u64>(5000ULL, t / 500ULL)) {
                    best = std::max(best, 120.0 - std::log2(static_cast<double>(diff + 1)));
                }
            }
            if (p > 500) {
                u64 g = std::gcd(p, t);
                if (g > 500) best = std::max(best, 50.0 + std::log2(static_cast<double>(g)));
            }
        }
        // Also bonus if t is an affine base for another remaining target!
        for (u64 u : remaining) {
            if (u <= t) continue;
            u64 q = (u + t / 2) / t;
            if (q >= 1 && q <= 200) {
                u128 prod = static_cast<u128>(q) * t;
                u64 diff = prod > u ? static_cast<u64>(prod - u) : static_cast<u64>(u - prod);
                if (diff <= std::max<u64>(5000ULL, u / 500ULL)) {
                    best += 80.0;
                }
            }
        }
        return best;
    }

    u64 pickTarget() {
        if (mode == 1) {
            u64 bestT = remaining[0];
            double bestSc = -1e18;
            for (u64 t : remaining) {
                double sc = targetAffinity(t) - 1.5 * std::log10(static_cast<double>(t) + 1.0);
                if (sc > bestSc) {
                    bestSc = sc;
                    bestT = t;
                }
            }
            return bestT;
        }
        if (mode >= 2 && (rng() % 3 == 0)) {
            return remaining[rng() % remaining.size()];
        }
        return *std::min_element(remaining.begin(), remaining.end());
    }

    bool run(const Puzzle& pz) {
        for (u64 t : pz.targets) allTargets.insert(t, 1);
        for (u64 s : pz.starts) addValue(s, true);
        remaining = pz.missing_targets();
        if (remaining.empty()) return true;

        FlatMap<uint64_t> stale(remaining.size() * 2 + 8);
        std::vector<Step> steps;

        while (!remaining.empty()) {
            if (shouldAbort()) {
                aborted = true;
                return false;
            }
            checkTime();

            // 1. Level-1 harvest
            steps.clear();
            for (u64 t : remaining) {
                if (c01(t) == 1) emitL1(t, steps);
            }
            if (!steps.empty()) {
                if (!apply(steps)) return false;
                continue;
            }

            // 2. Level-2 voting
            FlatMap<int> votes(remaining.size() * 32 + 64);
            std::vector<u64> votedKeys;
            std::vector<u64> xs;
            for (u64 t : remaining) {
                xs.clear();
                find2(t, nullptr, &xs);
                std::sort(xs.begin(), xs.end());
                xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
                for (u64 x : xs) {
                    if (int* cnt = votes.find(x)) {
                        ++(*cnt);
                    } else {
                        votes.insert(x, 1);
                        votedKeys.push_back(x);
                    }
                }
            }
            if (!votedKeys.empty()) {
                std::vector<u64> top;
                int bestVotes = 0;
                for (u64 x : votedKeys) {
                    int n = *votes.find(x);
                    if (n > bestVotes) {
                        bestVotes = n;
                        top.clear();
                    }
                    if (n == bestVotes) top.push_back(x);
                }
                std::sort(top.begin(), top.end());
                u64 pickX = (mode >= 2) ? top[rng() % top.size()] : top[0];
                emitL1(pickX, steps);
                if (!apply(steps)) return false;
                continue;
            }

            // 3. Budgeted shallow search across all remaining targets
            bool found = false;
            if (!softTimedOut) {
                std::vector<u64> order = remaining;
                if (mode == 1) {
                    std::sort(order.begin(), order.end(), [&](u64 a, u64 b) {
                        return targetAffinity(a) > targetAffinity(b);
                    });
                } else if (mode >= 2) {
                    std::shuffle(order.begin(), order.end(), rng);
                }

                curWorkLimit = std::min(opt.work, 1.5e7);
                for (int B = 3; B <= opt.bfull && !found && !softTimedOut; ++B) {
                    for (u64 t : order) {
                        if (checkTime() || shouldAbort()) break;
                        u64 key = (t << 4) ^ static_cast<u64>(B);
                        if (key == 0) key = 1;
                        if (B > 3) {
                            if (const u64* lastCr = stale.find(key)) {
                                if (created < *lastCr + (*lastCr) / 5 + 3) continue;
                            }
                        }
                        steps.clear();
                        work = 0.0;
                        if (solve(t, B, steps, true)) {
                            if (!apply(steps)) return false;
                            found = true;
                            break;
                        }
                        stale.insert_or_assign(key, static_cast<u64>(created));
                    }
                }
                curWorkLimit = opt.work;
            }
            if (found) continue;

            // 4. Deep construction of chosen target
            u64 t = pickTarget();
            if (!construct(t, 0)) return false;
        }
        return true;
    }
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
        else if (a == "--time") opt.timeLimit = std::atof(next());
        else if (a == "--bfull") opt.bfull = std::atoi(next());
        else if (a == "--bmax") opt.bmax = std::atoi(next());
        else if (a == "--k") opt.k = std::atoi(next());
        else if (a == "--work") opt.work = std::atof(next());
        else if (a == "--regs") opt.regs = std::atoi(next());
        else if (a == "--verbose") opt.verbose = true;
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }

    Puzzle pz;
    if (!elf::read_puzzle(std::cin, pz)) {
        std::fprintf(stderr, "bad input\n");
        return 2;
    }

    const size_t lowerBound = pz.missing_targets().size();
    if (lowerBound == 0) {
        return 0;
    }

    Timer timer;
    std::vector<Step> best;
    bool haveBest = false;

    int totalRestarts = std::max(1, opt.restarts);
    for (int r = 0; r < totalRestarts; ++r) {
        if (haveBest && best.size() <= lowerBound) break;
        double elapsed = timer.elapsed_sec();
        if (haveBest && elapsed >= opt.timeLimit * 0.90) break;

        double remainingTime = std::max(0.15, opt.timeLimit - elapsed);
        int restartsLeft = totalRestarts - r;
        double slice = remainingTime / static_cast<double>(restartsLeft);
        if (r == 0 && totalRestarts > 1) slice = std::max(slice, opt.timeLimit * 0.55);
        double restartDeadline = std::min(opt.timeLimit * 0.96, elapsed + slice);
        if (!haveBest) restartDeadline = std::max(restartDeadline, opt.timeLimit * 0.85);

        Options rOpt = opt;
        if (r > 0) {
            rOpt.bmax = std::min(opt.bmax, 7);
            rOpt.work = std::min(opt.work, 7.0e7);
        }

        u64 rSeed = (opt.seed + static_cast<u64>(r) + 1ULL) * 0x9E3779B97F4A7C15ULL + 1ULL;
        Engine engine(rOpt, timer, restartDeadline,
                      haveBest ? best.size() : static_cast<size_t>(-1),
                      r, pz.limit, rSeed);

        bool ok = engine.run(pz);
        if (ok) {
            auto verified = elf::verify_and_prune_plan(pz, engine.plan, opt.regs);
            if (verified.has_value()) {
                if (!haveBest || verified->size() < best.size()) {
                    best = std::move(*verified);
                    haveBest = true;
                }
            }
        }
        if (opt.verbose) {
            std::fprintf(stderr, "restart %d: ok=%d raw=%zu evict=%zu best=%zu t=%.2fs\n",
                         r, ok ? 1 : 0, engine.plan.size(), engine.evictions,
                         haveBest ? best.size() : 0, timer.elapsed_sec());
        }
    }

    if (!haveBest) {
        // Emergency ultra-fast deterministic fallback (no time limit cutoff in split).
        Options fastOpt = opt;
        fastOpt.bfull = 3;
        fastOpt.bmax = 4;
        Engine engine(fastOpt, timer, timer.elapsed_sec() + 30.0,
                      static_cast<size_t>(-1), 0, pz.limit, 1ULL);
        if (engine.run(pz)) {
            auto verified = elf::verify_and_prune_plan(pz, engine.plan, opt.regs);
            if (verified.has_value()) {
                best = std::move(*verified);
                haveBest = true;
            }
        }
    }

    if (!haveBest) return 1;
    elf::emit_plan(best);
    return 0;
}
