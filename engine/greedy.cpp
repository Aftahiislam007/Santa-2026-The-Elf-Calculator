// Pool-based greedy search engine.
//
// The pool P holds every value computed so far. L1 is the set of values one
// instruction away from P (maintained incrementally). Targets are produced
// cheapest-first:
//   level 1   t in L1
//   level 2   t = x o c, x in L1, c in P; the x that unlocks most targets wins
//   level 3   t = x o y (x, y in L1) or t = (x o c) o c'
//   level 4+  decompositions t = a*q +- r and t = x +- r with a, q, x in P u L1
//             and the remainder solved recursively (iterative deepening)
// If nothing fits the budget the target is split as t = a*q + r with a close
// to sqrt(t) and both parts are constructed recursively, which always
// terminates.
//
// I/O protocol: see elfcalc/solvers/engine.py.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using u64 = uint64_t;
using u128 = __uint128_t;

struct Step {
    char op;
    u64 a, b;
};

struct Table {  // open addressing u64 -> u32, keys are non-zero
    int bits = 12;
    size_t count = 0;
    std::vector<u64> keys = std::vector<u64>(size_t(1) << bits, 0);
    std::vector<uint32_t> vals = std::vector<uint32_t>(size_t(1) << bits, 0);

    size_t slot(u64 k) const { return (k * 0x9E3779B97F4A7C15ULL) >> (64 - bits); }

    const uint32_t* find(u64 k) const {
        size_t mask = keys.size() - 1;
        for (size_t i = slot(k);; i = (i + 1) & mask) {
            if (keys[i] == k) return &vals[i];
            if (!keys[i]) return nullptr;
        }
    }

    bool insert(u64 k, uint32_t v) {
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
        std::vector<u64> oldKeys;
        std::vector<uint32_t> oldVals;
        oldKeys.swap(keys);
        oldVals.swap(vals);
        ++bits;
        keys.assign(size_t(1) << bits, 0);
        vals.assign(size_t(1) << bits, 0);
        count = 0;
        for (size_t i = 0; i < oldKeys.size(); ++i)
            if (oldKeys[i]) insert(oldKeys[i], oldVals[i]);
    }
};

struct Options {
    uint64_t seed = 0;
    int restarts = 1;
    double timeLimit = 1e9;
    int bfull = 5;      // budgets up to bfull are tried for every target before going deep
    int bmax = 8;       // deepest budget tried on a single target before splitting it
    int k = 24;         // remainder candidates explored per decomposition
    double work = 2e8;  // table lookups allowed per top-level search
    int regs = 48;      // values that can be held at once (0 = unlimited)
    bool verbose = false;
};

static u64 isqrt(u64 v) {
    u64 s = (u64)std::sqrt((long double)v);
    while ((u128)s * s > v) --s;
    while ((u128)(s + 1) * (s + 1) <= v) ++s;
    return s;
}

struct Engine {
    struct L1E {
        u64 v, a, b;
        char op;
    };
    struct Cand {
        u64 r, a, q;
        uint8_t ca, cq, kind;  // kind 0: a*q+r, 1: a*q-r, 2: a+r, 3: a-r
        int rem;
    };

    const Options& opt;
    u64 limit;
    std::vector<u64> pool;
    std::vector<u64> stamp;  // parallel to pool: tick of creation or last read
    std::unordered_map<u64, int> pins;
    u64 tick = 0;
    size_t created = 0, evictions = 0;
    Table poolIdx;
    std::vector<L1E> l1;
    Table l1Idx;
    std::vector<Step> plan;
    std::vector<u64> remaining;
    std::unordered_map<u64, int> failB;
    std::vector<std::pair<u64, uint8_t>> M;
    bool mDirty = true;
    double work = 0;
    std::mt19937_64 rng;

    Engine(const Options& o, u64 lim, uint64_t seed) : opt(o), limit(lim), rng(seed) {}

    [[noreturn]] void fatal(const char* msg) {
        std::fprintf(stderr, "engine bug: %s\n", msg);
        std::exit(2);
    }

    u64 eval(char op, u64 a, u64 b) const {
        switch (op) {
            case '+': { u64 r = a + b; return r <= limit ? r : 0; }
            case '*': { u128 r = (u128)a * b; return r <= limit ? (u64)r : 0; }
            case '-': return a > b ? a - b : 0;
            case '/': return b && a % b == 0 ? a / b : 0;
        }
        return 0;
    }

    bool inPool(u64 x) const { return poolIdx.find(x) != nullptr; }

    int c01(u64 x) const { return inPool(x) ? 0 : l1Idx.find(x) ? 1 : 2; }

    void addL1(u64 v, char op, u64 a, u64 b) {
        if (!v || inPool(v)) return;
        if (l1Idx.insert(v, (uint32_t)l1.size())) l1.push_back({v, a, b, op});
    }

    // L1 entries pairing v with every pool value up to and including itself.
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
        if (!poolIdx.insert(v, (uint32_t)pool.size())) return;
        pool.push_back(v);
        stamp.push_back(++tick);
        ++created;
        if (link) linkValue(v);
        mDirty = true;
    }

    void reindex() {
        poolIdx = Table();
        for (size_t j = 0; j < pool.size(); ++j) poolIdx.insert(pool[j], (uint32_t)j);
    }

    void rebuild() {
        std::vector<u64> old;
        old.swap(pool);
        poolIdx = Table();
        l1.clear();
        l1Idx = Table();
        for (u64 v : old) {
            poolIdx.insert(v, (uint32_t)pool.size());
            pool.push_back(v);
            linkValue(v);
        }
        mDirty = true;
    }

    // The pool mirrors the registers: when it is full the least recently used
    // value that no pending step needs is overwritten.
    void apply(const std::vector<Step>& steps) {
        std::unordered_map<u64, size_t> lastNeed;
        for (size_t i = 0; i < steps.size(); ++i) lastNeed[steps[i].a] = lastNeed[steps[i].b] = i;
        bool evicted = false;
        for (size_t i = 0; i < steps.size(); ++i) {
            const Step& s = steps[i];
            const uint32_t *ia = poolIdx.find(s.a), *ib = poolIdx.find(s.b);
            if (!ia || !ib) fatal("operand not in pool");
            u64 r = eval(s.op, s.a, s.b);
            if (!r) fatal("illegal step");
            stamp[*ia] = stamp[*ib] = ++tick;
            if (inPool(r)) continue;
            plan.push_back(s);
            if (opt.regs && pool.size() >= (size_t)opt.regs) {
                size_t victim = pool.size();
                for (size_t j = 0; j < pool.size(); ++j) {
                    auto need = lastNeed.find(pool[j]);
                    if ((need != lastNeed.end() && need->second > i) || pins.count(pool[j])) continue;
                    if (victim == pool.size() || stamp[j] < stamp[victim]) victim = j;
                }
                if (victim == pool.size()) fatal("no register to overwrite");
                pool.erase(pool.begin() + victim);
                stamp.erase(stamp.begin() + victim);
                reindex();
                evicted = true;
                ++evictions;
            }
            addValue(r, !evicted);
        }
        if (evicted) rebuild();
        failB.clear();
        remaining.erase(std::remove_if(remaining.begin(), remaining.end(),
                                       [&](u64 t) { return inPool(t); }),
                        remaining.end());
    }

    void emitL1(u64 x, std::vector<Step>& out) const {
        const L1E& e = l1[*l1Idx.find(x)];
        out.push_back({e.op, e.a, e.b});
    }

    void emit01(u64 x, int c, std::vector<Step>& out) const {
        if (c == 1) emitL1(x, out);
    }

    // v = x o c with x in L1, c in P. With `collect`, gather every such x instead.
    bool find2(u64 v, std::vector<Step>* out, std::vector<u64>* collect) {
        work += pool.size() * 6;
        auto hit = [&](u64 x, char op, bool xFirst, u64 c) {
            if (!l1Idx.find(x)) return false;
            if (collect) {
                collect->push_back(x);
                return false;
            }
            emitL1(x, *out);
            out->push_back(xFirst ? Step{op, x, c} : Step{op, c, x});
            return true;
        };
        for (u64 c : pool) {
            if (v > c && hit(v - c, '+', true, c)) return true;
            if (v + c <= limit && hit(v + c, '-', true, c)) return true;
            if (c > v) {
                if (hit(c - v, '-', false, c)) return true;
                if (c % v == 0 && hit(c / v, '/', false, c)) return true;
            }
            if (c > 1) {
                if (v % c == 0 && hit(v / c, '*', true, c)) return true;
                u128 p = (u128)v * c;
                if (p <= limit && hit((u64)p, '/', true, c)) return true;
            }
        }
        if (!(v & 1) && hit(v / 2, '+', true, v / 2)) return true;
        u64 s = isqrt(v);
        if (s * s == v && hit(s, '*', true, s)) return true;
        return collect && !collect->empty();
    }

    bool find3(u64 v, std::vector<Step>& out) {
        work += l1.size() * 6;
        auto both = [&](u64 x, u64 y, Step last) {
            if (!l1Idx.find(y)) return false;
            emitL1(x, out);
            emitL1(y, out);
            out.push_back(last);
            return true;
        };
        for (const L1E& e : l1) {
            u64 x = e.v;
            if (v > x && both(x, v - x, {'+', x, v - x})) return true;
            if (x > v) {
                if (both(x, x - v, {'-', x, x - v})) return true;
                if (x % v == 0 && both(x, x / v, {'/', x, x / v})) return true;
            }
            if (v + x <= limit && both(x, v + x, {'-', v + x, x})) return true;
            if (x > 1) {
                if (v % x == 0 && both(x, v / x, {'*', x, v / x})) return true;
                u128 p = (u128)v * x;
                if (p <= limit && both(x, (u64)p, {'/', (u64)p, x})) return true;
            }
        }
        auto chain = [&](u64 y, Step last) {
            if (!find2(y, &out, nullptr)) return false;
            out.push_back(last);
            return true;
        };
        for (u64 c : pool) {
            if (v > c && chain(v - c, {'+', v - c, c})) return true;
            if (v + c <= limit && chain(v + c, {'-', v + c, c})) return true;
            if (c > v) {
                if (chain(c - v, {'-', c, c - v})) return true;
                if (c % v == 0 && chain(c / v, {'/', c, c / v})) return true;
            }
            if (c > 1) {
                if (v % c == 0 && chain(v / c, {'*', v / c, c})) return true;
                u128 p = (u128)v * c;
                if (p <= limit && chain((u64)p, {'/', (u64)p, c})) return true;
            }
        }
        return false;
    }

    void buildM() {
        if (!mDirty) return;
        M.clear();
        for (u64 p : pool) M.push_back({p, 0});
        for (const L1E& e : l1)
            if (!inPool(e.v)) M.push_back({e.v, 1});
        std::sort(M.begin(), M.end());
        mDirty = false;
    }

    // v = a*q +- r, where q and r are solved with whatever budget is left.
    bool trySplit(int B, u64 a, int ca, u64 q, u64 r, bool minus, std::vector<Step>& out) {
        int base = ca + 2, lq = c01(q), lr = c01(r);
        if (base + lq + lr > B) return false;
        size_t mark = out.size();
        auto finish = [&] {
            emit01(a, ca, out);
            out.push_back({'*', a, q});
            out.push_back({minus ? '-' : '+', a * q, r});
            return true;
        };
        int capR = std::min(2, B - base - lq);
        if (solve(r, capR, out)) {
            int cr = (int)(out.size() - mark);
            if (solve(q, B - base - cr, out)) return finish();
            out.resize(mark);
            return false;
        }
        out.resize(mark);
        int capQ = std::min(2, B - base - std::max(lr, capR + 1));
        if (capQ < lq) return false;
        if (solve(q, capQ, out)) {
            int cq = (int)(out.size() - mark);
            if (solve(r, B - base - cq, out)) return finish();
        }
        out.resize(mark);
        return false;
    }

    bool decomp(u64 v, int B, std::vector<Step>& out) {
        buildM();
        work += M.size() * 20;
        std::vector<Cand> cands;
        for (const auto& [a, ca] : M) {
            if (a < 2) continue;
            if (a >= v) break;
            auto it = std::lower_bound(M.begin(), M.end(), std::make_pair(v / a, (uint8_t)0));
            for (int d = -1; d <= 1; ++d) {
                auto jt = it + d;
                if (jt < M.begin() || jt >= M.end() || jt->first < 2) continue;
                u64 q = jt->first;
                int cost0 = ca + jt->second + 1;
                u128 p = (u128)a * q;
                if (cost0 >= B || p > limit || (u64)p == v) continue;
                u64 prod = (u64)p, r = prod > v ? prod - v : v - prod;
                int rem = B - cost0 - 1;
                if (rem < c01(r)) continue;
                cands.push_back({r, a, q, ca, jt->second, (uint8_t)(prod > v ? 1 : 0), rem});
            }
        }
        auto it = std::lower_bound(M.begin(), M.end(), std::make_pair(v, (uint8_t)0));
        for (int d = -8; d < 8; ++d) {
            auto jt = it + d;
            if (jt < M.begin() || jt >= M.end() || jt->first == v) continue;
            u64 x = jt->first, r = x > v ? x - v : v - x;
            int rem = B - jt->second - 1;
            if (rem < c01(r)) continue;
            cands.push_back({r, x, 0, jt->second, 0, (uint8_t)(x > v ? 3 : 2), rem});
        }
        std::sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) {
            return x.r != y.r ? x.r < y.r : x.rem > y.rem;
        });
        int tried = 0;
        u64 lastR = 0;
        for (const Cand& c : cands) {
            if (c.r == lastR) continue;
            if (++tried > opt.k || work > opt.work) break;
            lastR = c.r;
            size_t mark = out.size();
            if (!solve(c.r, c.rem, out)) {
                out.resize(mark);
                continue;
            }
            emit01(c.a, c.ca, out);
            if (c.kind < 2) {
                emit01(c.q, c.cq, out);
                out.push_back({'*', c.a, c.q});
                out.push_back({c.kind ? '-' : '+', c.a * c.q, c.r});
            } else {
                out.push_back({c.kind == 3 ? '-' : '+', c.a, c.r});
            }
            return true;
        }

        for (u64 a : pool) {
            if (a < 2 || a >= v) continue;
            if (work > opt.work) return false;
            u64 q = v / a, r = v % a;
            if (r == 0) {
                size_t mark = out.size();
                if (solve(q, B - 1, out)) {
                    out.push_back({'*', q, a});
                    return true;
                }
                out.resize(mark);
                continue;
            }
            if (trySplit(B, a, 0, q, r, false, out)) return true;
            if ((u128)a * (q + 1) <= limit && trySplit(B, a, 0, q + 1, a - r, true, out))
                return true;
        }
        for (const L1E& e : l1) {
            u64 a = e.v;
            if (a < 2 || a >= v || inPool(a)) continue;
            if (work > opt.work) return false;
            u64 q = v / a, r = v % a;
            if (r == 0) continue;
            if (std::min(c01(q), c01(r)) <= 1 && trySplit(B, a, 1, q, r, false, out)) return true;
            if ((u128)a * (q + 1) <= limit && std::min(c01(q + 1), c01(a - r)) <= 1 &&
                trySplit(B, a, 1, q + 1, a - r, true, out))
                return true;
        }
        return false;
    }

    // Append to `out` a derivation of v using at most B new instructions.
    bool solve(u64 v, int B, std::vector<Step>& out) {
        int c = c01(v);
        if (c == 0) return true;
        if (B <= 0) return false;
        if (c == 1) {
            emitL1(v, out);
            return true;
        }
        if (B == 1) return false;
        auto it = failB.find(v);
        int from = it == failB.end() ? 2 : it->second + 1;
        size_t mark = out.size();
        for (int b = from; b <= B; ++b) {
            if (work > opt.work) return false;
            bool ok = b == 2 ? find2(v, &out, nullptr) : b == 3 ? find3(v, out) : decomp(v, b, out);
            if (ok) return true;
            out.resize(mark);
            if (work > opt.work) return false;
            failB[v] = b;
        }
        return false;
    }

    void construct(u64 v) {
        while (!inPool(v)) {
            std::vector<Step> steps;
            bool ok = false;
            for (int B = 1; B <= opt.bmax && !ok; ++B) {
                steps.clear();
                work = 0;
                ok = solve(v, B, steps);
            }
            if (ok) {
                apply(steps);
                return;
            }
            buildM();
            u64 root = isqrt(v), best = 0;
            int bestScore = 99, seen = 0;
            auto it = std::upper_bound(M.begin(), M.end(), std::make_pair(root, (uint8_t)255));
            while (it != M.begin() && seen < 16) {
                --it;
                u64 a = it->first;
                if (a < 2) break;
                ++seen;
                int score = it->second + c01(v / a) + (v % a ? c01(v % a) : 0);
                if (score < bestScore) bestScore = score, best = a;
            }
            if (!best) {
                for (const auto& [a, ca] : M)
                    if (a >= 2 && a < v) {
                        best = a;
                        break;
                    }
            }
            if (!best) {
                if (inPool(1)) fatal("cannot split");
                apply({{'/', pool[0], pool[0]}});
                continue;
            }
            u64 q = v / best, r = v % best;
            construct(best);
            ++pins[best];
            construct(q);
            ++pins[q];
            if (r) {
                construct(r);
                apply({{'*', best, q}, {'+', best * q, r}});
            } else {
                apply({{'*', best, q}});
            }
            for (u64 x : {best, q})
                if (--pins[x] == 0) pins.erase(x);
        }
    }

    u64 pick() {
        if (opt.seed == 0 || rng() % 4) return *std::min_element(remaining.begin(), remaining.end());
        return remaining[rng() % remaining.size()];
    }

    void run(const std::vector<u64>& starts, const std::vector<u64>& targets) {
        for (u64 s : starts) addValue(s, true);
        for (u64 t : targets)
            if (!inPool(t) && std::find(remaining.begin(), remaining.end(), t) == remaining.end())
                remaining.push_back(t);
        std::sort(remaining.begin(), remaining.end());
        std::unordered_map<u64, std::vector<size_t>> stale;
        std::vector<Step> steps;

        while (!remaining.empty()) {
            steps.clear();
            for (u64 t : remaining)
                if (c01(t) == 1) emitL1(t, steps);
            if (!steps.empty()) {
                apply(steps);
                continue;
            }

            std::unordered_map<u64, int> votes;
            std::vector<u64> xs;
            for (u64 t : remaining) {
                xs.clear();
                find2(t, nullptr, &xs);
                std::sort(xs.begin(), xs.end());
                xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
                for (u64 x : xs) ++votes[x];
            }
            if (!votes.empty()) {
                std::vector<u64> top;
                int bestVotes = 0;
                for (const auto& [x, n] : votes) {
                    if (n > bestVotes) bestVotes = n, top.clear();
                    if (n == bestVotes) top.push_back(x);
                }
                std::sort(top.begin(), top.end());
                emitL1(opt.seed ? top[rng() % top.size()] : top[0], steps);
                apply(steps);
                continue;
            }

            bool found = false;
            std::vector<u64> order = remaining;
            if (opt.seed) std::shuffle(order.begin(), order.end(), rng);
            for (int B = 3; B <= opt.bfull && !found; ++B) {
                for (u64 t : order) {
                    auto& seen = stale[t];
                    if (seen.size() <= (size_t)B) seen.resize(B + 1, 0);
                    if (B > 3 && seen[B] && created < seen[B] + seen[B] / 5 + 3) continue;
                    steps.clear();
                    work = 0;
                    if (solve(t, B, steps)) {
                        apply(steps);
                        found = true;
                        break;
                    }
                    seen[B] = created;
                }
            }
            if (found) continue;

            u64 t = pick();
            size_t before = plan.size();
            construct(t);
            if (opt.verbose)
                std::fprintf(stderr, "deep %llu: %zu steps (pool %zu, L1 %zu)\n",
                             (unsigned long long)t, plan.size() - before, pool.size(), l1.size());
        }
    }
};

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

    std::string id;
    u64 limit;
    size_t k, n;
    std::cin >> id >> limit >> k;
    std::vector<u64> starts(k);
    for (u64& s : starts) std::cin >> s;
    std::cin >> n;
    std::vector<u64> targets(n);
    for (u64& t : targets) std::cin >> t;
    if (!std::cin) {
        std::fprintf(stderr, "bad input\n");
        return 2;
    }

    auto begin = std::chrono::steady_clock::now();
    std::vector<Step> best;
    bool have = false;
    for (int r = 0; r < opt.restarts; ++r) {
        Options o = opt;
        o.seed = opt.seed + r;
        Engine engine(o, limit, o.seed * 0x9E3779B97F4A7C15ULL + 1);
        engine.run(starts, targets);
        if (!have || engine.plan.size() < best.size()) best = engine.plan, have = true;
        std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - begin;
        if (opt.verbose)
            std::fprintf(stderr, "restart %d: %zu steps, %zu evictions, best %zu, %.1fs\n", r,
                         engine.plan.size(), engine.evictions, best.size(), elapsed.count());
        if (elapsed.count() * (r + 2) / (r + 1) > opt.timeLimit) break;
    }
    for (const Step& s : best)
        std::printf("%c %llu %llu\n", s.op, (unsigned long long)s.a, (unsigned long long)s.b);
    return 0;
}
