#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace elf {

using u64 = uint64_t;
using u128 = __uint128_t;

struct Step {
    char op = 0;
    u64 a = 0;
    u64 b = 0;
    u64 r = 0;
};

struct StepNode {
    char op = 0;
    u64 a = 0;
    u64 b = 0;
    uint8_t cost = 0;
};

struct Puzzle {
    std::string id;
    u64 limit = 0;
    std::vector<u64> starts;
    std::vector<u64> targets;

    std::vector<u64> missing_targets() const {
        std::vector<u64> sv = starts;
        std::sort(sv.begin(), sv.end());
        sv.erase(std::unique(sv.begin(), sv.end()), sv.end());
        std::vector<u64> miss;
        for (u64 t : targets) {
            if (!std::binary_search(sv.begin(), sv.end(), t)) {
                miss.push_back(t);
            }
        }
        std::sort(miss.begin(), miss.end());
        miss.erase(std::unique(miss.begin(), miss.end()), miss.end());
        return miss;
    }
};

// Exact stdin parser matching elfcalc/solvers/engine.py:
//   Line 1: puzzle id
//   Line 2: value_limit
//   Line 3: k s_1 ... s_k
//   Line 4: n t_1 ... t_n
inline bool read_puzzle(std::istream& in, Puzzle& pz) {
    pz = Puzzle{};
    size_t k = 0;
    size_t n = 0;
    if (!(in >> pz.id >> pz.limit >> k)) return false;
    pz.starts.resize(k);
    for (size_t i = 0; i < k; ++i) {
        if (!(in >> pz.starts[i])) return false;
    }
    if (!(in >> n)) return false;
    pz.targets.resize(n);
    for (size_t i = 0; i < n; ++i) {
        if (!(in >> pz.targets[i])) return false;
    }
    return true;
}

inline Puzzle read_puzzle(std::istream& in) {
    Puzzle pz;
    read_puzzle(in, pz);
    return pz;
}

inline void emit_plan(const std::vector<Step>& plan) {
    for (const Step& s : plan) {
        std::printf("%c %llu %llu\n", s.op,
                    static_cast<unsigned long long>(s.a),
                    static_cast<unsigned long long>(s.b));
    }
}

inline void emit_plan(const std::vector<Step>& plan, std::ostream& out) {
    for (const Step& s : plan) {
        out << s.op << ' ' << s.a << ' ' << s.b << '\n';
    }
}

// Safe arithmetic (0 indicates illegal operation or overflow past limit).
inline u64 safe_mul(u64 a, u64 b, u64 limit) {
    if (a == 0 || b == 0) return 0;
    u128 r = static_cast<u128>(a) * static_cast<u128>(b);
    return r <= limit ? static_cast<u64>(r) : 0;
}

inline u64 apply_op(char op, u64 a, u64 b, u64 limit) {
    if (a == 0 || b == 0) return 0;
    switch (op) {
        case '+':
            return (a <= limit && b <= limit - a) ? (a + b) : 0;
        case '-':
            return (a > b && (a - b) <= limit) ? (a - b) : 0;
        case '*':
            return safe_mul(a, b, limit);
        case '/': {
            if (a % b != 0) return 0;
            u64 q = a / b;
            return q <= limit ? q : 0;
        }
        default:
            return 0;
    }
}

inline u64 isqrt64(u64 n) {
    if (n == 0) return 0;
    u64 s = static_cast<u64>(std::sqrt(static_cast<long double>(n)));
    while (static_cast<u128>(s) * s > n) --s;
    while (static_cast<u128>(s + 1) * (s + 1) <= n) ++s;
    return s;
}

inline u64 splitmix64(u64 x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Fast open-addressed hash map with non-zero u64 keys and SplitMix64 hashing.
template <typename Val>
class FlatMap {
public:
    explicit FlatMap(size_t initial_cap = 16) {
        size_t cap = 8;
        while (cap < initial_cap) cap <<= 1;
        keys_.assign(cap, 0);
        vals_.resize(cap);
    }

    size_t size() const { return count_; }
    bool empty() const { return count_ == 0; }
    size_t capacity() const { return keys_.size(); }

    void clear() {
        std::fill(keys_.begin(), keys_.end(), 0);
        count_ = 0;
    }

    void reserve(size_t n) {
        size_t needed = 8;
        while (needed < n * 2) needed <<= 1;
        if (needed > keys_.size()) rehash(needed);
    }

    const Val* find(u64 k) const {
        if (k == 0) return nullptr;
        size_t mask = keys_.size() - 1;
        for (size_t i = slot(k, mask);; i = (i + 1) & mask) {
            if (keys_[i] == k) return &vals_[i];
            if (keys_[i] == 0) return nullptr;
        }
    }

    Val* find(u64 k) {
        if (k == 0) return nullptr;
        size_t mask = keys_.size() - 1;
        for (size_t i = slot(k, mask);; i = (i + 1) & mask) {
            if (keys_[i] == k) return &vals_[i];
            if (keys_[i] == 0) return nullptr;
        }
    }

    bool contains(u64 k) const { return find(k) != nullptr; }

    // Inserts (k, v) if k is not present. Returns true if inserted.
    bool insert(u64 k, const Val& v) {
        if (k == 0) return false;
        if ((count_ + 1) * 2 > keys_.size()) rehash(keys_.size() << 1);
        size_t mask = keys_.size() - 1;
        for (size_t i = slot(k, mask);; i = (i + 1) & mask) {
            if (keys_[i] == k) return false;
            if (keys_[i] == 0) {
                keys_[i] = k;
                vals_[i] = v;
                ++count_;
                return true;
            }
        }
    }

    // Inserts or overwrites existing value for k. Returns true if newly inserted.
    bool insert_or_assign(u64 k, const Val& v) {
        if (k == 0) return false;
        if ((count_ + 1) * 2 > keys_.size()) rehash(keys_.size() << 1);
        size_t mask = keys_.size() - 1;
        for (size_t i = slot(k, mask);; i = (i + 1) & mask) {
            if (keys_[i] == k) {
                vals_[i] = v;
                return false;
            }
            if (keys_[i] == 0) {
                keys_[i] = k;
                vals_[i] = v;
                ++count_;
                return true;
            }
        }
    }

    // Backward-shift deletion (no tombstones). Returns true if key existed.
    bool erase(u64 k) {
        if (k == 0 || count_ == 0) return false;
        size_t mask = keys_.size() - 1;
        size_t i = slot(k, mask);
        while (true) {
            if (keys_[i] == 0) return false;
            if (keys_[i] == k) break;
            i = (i + 1) & mask;
        }
        --count_;
        size_t hole = i;
        size_t cur = (hole + 1) & mask;
        while (keys_[cur] != 0) {
            size_t home = slot(keys_[cur], mask);
            if (((cur - home) & mask) >= ((cur - hole) & mask)) {
                keys_[hole] = keys_[cur];
                vals_[hole] = std::move(vals_[cur]);
                hole = cur;
            }
            cur = (cur + 1) & mask;
        }
        keys_[hole] = 0;
        vals_[hole] = Val{};
        return true;
    }

    template <typename Fn>
    void for_each(Fn&& fn) const {
        for (size_t i = 0; i < keys_.size(); ++i) {
            if (keys_[i] != 0) fn(keys_[i], vals_[i]);
        }
    }

private:
    static size_t slot(u64 k, size_t mask) {
        return static_cast<size_t>(splitmix64(k)) & mask;
    }

    void rehash(size_t new_cap) {
        std::vector<u64> old_keys;
        std::vector<Val> old_vals;
        old_keys.swap(keys_);
        old_vals.swap(vals_);
        keys_.assign(new_cap, 0);
        vals_.resize(new_cap);
        count_ = 0;
        size_t mask = new_cap - 1;
        for (size_t i = 0; i < old_keys.size(); ++i) {
            u64 k = old_keys[i];
            if (k == 0) continue;
            for (size_t j = slot(k, mask);; j = (j + 1) & mask) {
                if (keys_[j] == 0) {
                    keys_[j] = k;
                    vals_[j] = std::move(old_vals[i]);
                    ++count_;
                    break;
                }
            }
        }
    }

    std::vector<u64> keys_;
    std::vector<Val> vals_;
    size_t count_ = 0;
};

// Number theory: deterministic 64-bit Miller-Rabin, Brent's Pollard's Rho, divisors.
inline u64 mul_mod(u64 a, u64 b, u64 m) {
    return static_cast<u64>((static_cast<u128>(a) * static_cast<u128>(b)) % m);
}

inline u64 pow_mod(u64 base, u64 exp, u64 mod) {
    u64 res = 1 % mod;
    base %= mod;
    while (exp > 0) {
        if (exp & 1ULL) res = mul_mod(res, base, mod);
        base = mul_mod(base, base, mod);
        exp >>= 1;
    }
    return res;
}

inline bool is_prime(u64 n) {
    if (n < 2) return false;
    static constexpr u64 kBases[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (u64 p : kBases) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }
    u64 d = n - 1;
    int s = 0;
    while ((d & 1ULL) == 0) {
        d >>= 1;
        ++s;
    }
    for (u64 a : kBases) {
        if (a % n == 0) continue;
        u64 x = pow_mod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool witness = true;
        for (int r = 1; r < s; ++r) {
            x = mul_mod(x, x, n);
            if (x == n - 1) {
                witness = false;
                break;
            }
        }
        if (witness) return false;
    }
    return true;
}

namespace detail {

inline u64 pollard_brent(u64 n) {
    if ((n & 1ULL) == 0) return 2;
    if (n % 3ULL == 0) return 3;
    auto f = [n](u64 x, u64 c) -> u64 {
        return static_cast<u64>((static_cast<u128>(x) * x + c) % n);
    };
    for (u64 attempt = 1;; ++attempt) {
        u64 c = (attempt % (n - 1)) + 1;
        u64 y = (splitmix64(attempt ^ n) % (n - 2)) + 2;
        u64 r = 1;
        u64 q = 1;
        u64 g = 1;
        u64 x = y;
        u64 ys = y;
        constexpr u64 m = 128;
        while (g == 1) {
            x = y;
            for (u64 i = 0; i < r; ++i) y = f(y, c);
            u64 k = 0;
            while (k < r && g == 1) {
                ys = y;
                u64 lim = std::min(m, r - k);
                for (u64 i = 0; i < lim; ++i) {
                    y = f(y, c);
                    u64 diff = x > y ? x - y : y - x;
                    u64 prod = mul_mod(q, diff, n);
                    if (prod == 0) {
                        q = 0;
                        break;
                    }
                    q = prod;
                }
                g = (q == 0) ? n : std::gcd(q, n);
                k += lim;
            }
            r <<= 1;
        }
        if (g == n) {
            g = 1;
            while (g == 1) {
                ys = f(ys, c);
                u64 diff = x > ys ? x - ys : ys - x;
                g = std::gcd(diff, n);
            }
        }
        if (g > 1 && g < n) return g;
    }
}

inline void factor_rec(u64 n, std::vector<u64>& out) {
    if (n <= 1) return;
    if (is_prime(n)) {
        out.push_back(n);
        return;
    }
    u64 d = pollard_brent(n);
    factor_rec(d, out);
    factor_rec(n / d, out);
}

}  // namespace detail

inline std::vector<std::pair<u64, int>> factorize(u64 n) {
    std::vector<std::pair<u64, int>> res;
    if (n <= 1) return res;
    static constexpr u64 kSmallPrimes[] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
        41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
    for (u64 p : kSmallPrimes) {
        if (p * p > n) break;
        if (n % p == 0) {
            int e = 0;
            while (n % p == 0) {
                n /= p;
                ++e;
            }
            res.emplace_back(p, e);
        }
    }
    if (n > 1) {
        std::vector<u64> rem;
        detail::factor_rec(n, rem);
        std::sort(rem.begin(), rem.end());
        for (u64 p : rem) {
            if (!res.empty() && res.back().first == p) {
                ++res.back().second;
            } else {
                res.emplace_back(p, 1);
            }
        }
    }
    std::sort(res.begin(), res.end());
    return res;
}

inline std::vector<u64> get_divisors(const std::vector<std::pair<u64, int>>& fac,
                                     size_t max_divs = 4096) {
    std::vector<u64> divs;
    if (max_divs == 0) return divs;
    divs.push_back(1);
    for (const auto& kv : fac) {
        u64 p = kv.first;
        int e = kv.second;
        size_t cur = divs.size();
        u64 pk = 1;
        for (int i = 0; i < e; ++i) {
            pk *= p;
            for (size_t j = 0; j < cur; ++j) {
                if (divs.size() >= max_divs) break;
                divs.push_back(divs[j] * pk);
            }
            if (divs.size() >= max_divs) break;
        }
        if (divs.size() >= max_divs) break;
    }
    std::sort(divs.begin(), divs.end());
    return divs;
}

struct Timer {
    std::chrono::steady_clock::time_point start;

    Timer() : start(std::chrono::steady_clock::now()) {}

    void reset() { start = std::chrono::steady_clock::now(); }

    double elapsed_sec() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
};

// Build L1: all values reachable in 1 legal instruction from `pool` that are not already in `pool`.
inline void build_L1(const std::vector<u64>& pool, u64 limit,
                     FlatMap<Step>& L1, std::vector<u64>& L1_sorted) {
    L1.clear();
    L1_sorted.clear();
    FlatMap<uint8_t> in_pool(pool.size() * 2 + 8);
    for (u64 v : pool) in_pool.insert(v, 1);

    auto try_add = [&](char op, u64 a, u64 b, u64 r) {
        if (r == 0 || r > limit || in_pool.contains(r)) return;
        if (L1.insert(r, Step{op, a, b, r})) {
            L1_sorted.push_back(r);
        }
    };

    const size_t n = pool.size();
    for (size_t i = 0; i < n; ++i) {
        u64 u = pool[i];
        if (u == 0) continue;
        for (size_t j = i; j < n; ++j) {
            u64 v = pool[j];
            if (v == 0) continue;
            if (u <= limit && v <= limit - u) try_add('+', u, v, u + v);
            u64 prod = safe_mul(u, v, limit);
            if (prod != 0) try_add('*', u, v, prod);
            if (u > v) {
                try_add('-', u, v, u - v);
                if (u % v == 0) try_add('/', u, v, u / v);
            } else if (v > u) {
                try_add('-', v, u, v - u);
                if (v % u == 0) try_add('/', v, u, v / u);
            } else {
                try_add('/', u, u, 1);
            }
        }
    }
    std::sort(L1_sorted.begin(), L1_sorted.end());
}

namespace detail {

// Check register pressure for a deduplicated plan (matches elfcalc.plan._allocate).
inline bool check_regs_unique(const Puzzle& pz, const std::vector<Step>& steps, int max_regs) {
    if (max_regs <= 0) return true;
    FlatMap<size_t> last_use(steps.size() * 2 + pz.starts.size() * 2 + 8);
    for (size_t k = 0; k < steps.size(); ++k) {
        last_use.insert_or_assign(steps[k].a, k);
        last_use.insert_or_assign(steps[k].b, k);
    }
    int live = 0;
    FlatMap<uint8_t> seen_start(pz.starts.size() * 2 + 8);
    for (u64 v : pz.starts) {
        if (last_use.contains(v) && seen_start.insert(v, 1)) {
            ++live;
        }
    }
    if (live > max_regs) return false;
    for (size_t k = 0; k < steps.size(); ++k) {
        const Step& s = steps[k];
        if (const size_t* lu = last_use.find(s.a); lu && *lu == k) --live;
        if (s.b != s.a) {
            if (const size_t* lu = last_use.find(s.b); lu && *lu == k) --live;
        }
        if (live >= max_regs) return false;  // need at least 1 free register for destination
        if (last_use.contains(s.r)) ++live;
    }
    return true;
}

// Instance-based pruning and register check (matches elfcalc.plan._allocate_instances).
inline std::optional<std::vector<Step>> prune_and_check_instances(
    const Puzzle& pz, const std::vector<Step>& validated, int max_regs) {
    const size_t m = validated.size();
    // Instance IDs: starting values are 1..starts.size(), step k is starts.size() + 1 + k.
    const size_t offset = pz.starts.size() + 1;
    FlatMap<size_t> current(m * 2 + pz.starts.size() * 2 + 8);
    FlatMap<uint8_t> covered(m * 2 + pz.starts.size() * 2 + 8);
    for (size_t i = 0; i < pz.starts.size(); ++i) {
        current.insert(pz.starts[i], i + 1);
        covered.insert(pz.starts[i], 1);
    }

    FlatMap<uint8_t> miss_set(pz.targets.size() * 2 + 8);
    for (u64 t : pz.targets) {
        if (!covered.contains(t)) miss_set.insert(t, 1);
    }

    std::vector<std::pair<size_t, size_t>> operands(m);
    std::vector<uint8_t> needed_inst(offset + m, 0);
    for (size_t k = 0; k < m; ++k) {
        const Step& s = validated[k];
        const size_t* ia = current.find(s.a);
        const size_t* ib = current.find(s.b);
        if (!ia || !ib) return std::nullopt;
        operands[k] = {*ia, *ib};
        current.insert_or_assign(s.r, offset + k);
        if (miss_set.contains(s.r) && !covered.contains(s.r)) {
            needed_inst[offset + k] = 1;
        }
        covered.insert(s.r, 1);
    }

    for (size_t k = m; k-- > 0;) {
        if (needed_inst[offset + k]) {
            needed_inst[operands[k].first] = 1;
            needed_inst[operands[k].second] = 1;
        }
    }

    std::vector<size_t> last_use(offset + m, 0);
    std::vector<uint8_t> has_use(offset + m, 0);
    for (size_t k = 0; k < m; ++k) {
        if (!needed_inst[offset + k]) continue;
        size_t ia = operands[k].first;
        size_t ib = operands[k].second;
        last_use[ia] = k;
        has_use[ia] = 1;
        last_use[ib] = k;
        has_use[ib] = 1;
    }

    int live = 0;
    for (size_t i = 0; i < pz.starts.size(); ++i) {
        if (has_use[i + 1]) ++live;
    }
    if (max_regs > 0 && live > max_regs) return std::nullopt;

    std::vector<Step> out;
    out.reserve(m);
    for (size_t k = 0; k < m; ++k) {
        if (!needed_inst[offset + k]) continue;
        size_t ia = operands[k].first;
        size_t ib = operands[k].second;
        if (has_use[ia] && last_use[ia] == k) --live;
        if (ib != ia && has_use[ib] && last_use[ib] == k) --live;
        if (max_regs > 0 && live >= max_regs) return std::nullopt;
        if (has_use[offset + k]) ++live;
        out.push_back(validated[k]);
    }
    return out;
}

}  // namespace detail

// Validate all steps value-level, verify all targets are covered, prune dead/redundant steps,
// and verify register feasibility (<= max_regs). Returns std::nullopt if invalid.
inline std::optional<std::vector<Step>> verify_and_prune_plan(
    const Puzzle& pz, const std::vector<Step>& plan, int max_regs = 48) {
    FlatMap<uint8_t> have(plan.size() * 2 + pz.starts.size() * 2 + 8);
    for (u64 v : pz.starts) {
        if (v == 0 || v > pz.limit) return std::nullopt;
        have.insert(v, 1);
    }

    std::vector<Step> validated;
    validated.reserve(plan.size());
    for (const Step& s : plan) {
        if (!have.contains(s.a) || !have.contains(s.b)) return std::nullopt;
        u64 r = apply_op(s.op, s.a, s.b, pz.limit);
        if (r == 0) return std::nullopt;
        if (s.r != 0 && s.r != r) return std::nullopt;
        have.insert(r, 1);
        validated.push_back(Step{s.op, s.a, s.b, r});
    }

    for (u64 t : pz.targets) {
        if (!have.contains(t)) return std::nullopt;
    }

    // Deduplicated value-level prune (matches elfcalc.plan.prune_plan).
    FlatMap<uint8_t> seen(validated.size() * 2 + pz.starts.size() * 2 + 8);
    for (u64 v : pz.starts) seen.insert(v, 1);
    std::vector<Step> fresh;
    fresh.reserve(validated.size());
    for (const Step& s : validated) {
        if (seen.insert(s.r, 1)) {
            fresh.push_back(s);
        }
    }

    FlatMap<uint8_t> needed(fresh.size() * 2 + pz.targets.size() * 2 + 8);
    for (u64 t : pz.missing_targets()) needed.insert(t, 1);
    std::vector<Step> pruned;
    pruned.reserve(fresh.size());
    for (size_t i = fresh.size(); i-- > 0;) {
        if (needed.contains(fresh[i].r)) {
            pruned.push_back(fresh[i]);
            needed.insert(fresh[i].a, 1);
            needed.insert(fresh[i].b, 1);
        }
    }
    std::reverse(pruned.begin(), pruned.end());

    if (detail::check_regs_unique(pz, pruned, max_regs)) {
        return pruned;
    }
    // Fallback to instance-based pruning when recomputations are needed for <= max_regs.
    return detail::prune_and_check_instances(pz, validated, max_regs);
}

// Tracks a live value pool of size <= max_regs with pinning, custom usefulness scores,
// and safe step emission so engines never exceed the register limit.
class PoolTracker {
public:
    explicit PoolTracker(u64 limit, int max_regs = 48)
        : limit_(limit), max_regs_(max_regs), pos_(128), pins_(64), scores_(128) {}

    void init(const std::vector<u64>& starts) {
        pool_.clear();
        last_touch_.clear();
        pos_.clear();
        pins_.clear();
        scores_.clear();
        plan_.clear();
        tick_ = 0;
        evictions_ = 0;
        evicted_since_reset_ = false;
        for (u64 v : starts) {
            if (v == 0 || v > limit_) continue;
            if (pos_.contains(v)) continue;
            if (max_regs_ > 0 && pool_.size() >= static_cast<size_t>(max_regs_)) break;
            pos_.insert(v, static_cast<uint32_t>(pool_.size()));
            pool_.push_back(v);
            last_touch_.push_back(++tick_);
        }
    }

    u64 limit() const { return limit_; }
    int max_regs() const { return max_regs_; }
    size_t size() const { return pool_.size(); }
    bool contains(u64 v) const { return pos_.contains(v); }
    const std::vector<u64>& values() const { return pool_; }
    const std::vector<Step>& steps() const { return plan_; }
    size_t evictions() const { return evictions_; }
    bool evicted_recently() const { return evicted_since_reset_; }
    void clear_evicted_flag() { evicted_since_reset_ = false; }

    void pin(u64 v) {
        if (int* p = pins_.find(v)) {
            ++(*p);
        } else {
            pins_.insert(v, 1);
        }
    }

    void unpin(u64 v) {
        if (int* p = pins_.find(v)) {
            if (--(*p) <= 0) pins_.erase(v);
        }
    }

    bool is_pinned(u64 v) const { return pins_.contains(v); }

    void set_score(u64 v, double score) { scores_.insert_or_assign(v, score); }

    double get_score(u64 v) const {
        const double* s = scores_.find(v);
        return s ? *s : 0.0;
    }

    // Emit one step (op, a, b), evicting the least useful unpinned value if full.
    // Returns the computed result r (> 0), or 0 on error (missing operand, illegal op, or all pinned).
    template <typename ScoreFn>
    u64 emit_step(char op, u64 a, u64 b, ScoreFn&& score_fn) {
        uint32_t* ia = pos_.find(a);
        uint32_t* ib = pos_.find(b);
        if (!ia || !ib) return 0;
        u64 r = apply_op(op, a, b, limit_);
        if (r == 0) return 0;
        last_touch_[*ia] = ++tick_;
        last_touch_[*ib] = tick_;
        if (uint32_t* ir = pos_.find(r)) {
            last_touch_[*ir] = tick_;
            return r;
        }
        if (max_regs_ > 0 && pool_.size() >= static_cast<size_t>(max_regs_)) {
            size_t victim = pool_.size();
            double victim_score = 0.0;
            for (size_t i = 0; i < pool_.size(); ++i) {
                u64 v = pool_[i];
                if (pins_.contains(v)) continue;
                double sc = score_fn(v);
                if (victim == pool_.size() || sc < victim_score ||
                    (sc == victim_score && last_touch_[i] < last_touch_[victim])) {
                    victim = i;
                    victim_score = sc;
                }
            }
            if (victim == pool_.size()) return 0;
            evict_at(victim);
        }
        plan_.push_back(Step{op, a, b, r});
        pos_.insert(r, static_cast<uint32_t>(pool_.size()));
        pool_.push_back(r);
        last_touch_.push_back(++tick_);
        return r;
    }

    u64 emit_step(char op, u64 a, u64 b) {
        return emit_step(op, a, b, [this](u64 v) { return get_score(v); });
    }

    // Apply a sequence of steps, temporarily protecting operands needed later in the batch.
    template <typename ScoreFn>
    bool apply_steps(const std::vector<Step>& batch, ScoreFn&& score_fn) {
        FlatMap<size_t> last_need(batch.size() * 2 + 8);
        for (size_t i = 0; i < batch.size(); ++i) {
            last_need.insert_or_assign(batch[i].a, i);
            last_need.insert_or_assign(batch[i].b, i);
        }
        for (size_t i = 0; i < batch.size(); ++i) {
            const Step& s = batch[i];
            // Temporarily pin anything still needed after step i.
            std::vector<u64> temp_pins;
            for (u64 v : pool_) {
                if (const size_t* ln = last_need.find(v); ln && *ln > i) {
                    pin(v);
                    temp_pins.push_back(v);
                }
            }
            u64 r = emit_step(s.op, s.a, s.b, score_fn);
            for (u64 v : temp_pins) unpin(v);
            if (r == 0) return false;
        }
        return true;
    }

    bool apply_steps(const std::vector<Step>& batch) {
        return apply_steps(batch, [this](u64 v) { return get_score(v); });
    }

private:
    void evict_at(size_t idx) {
        u64 evicted_val = pool_[idx];
        u64 back_val = pool_.back();
        pool_[idx] = back_val;
        last_touch_[idx] = last_touch_.back();
        pool_.pop_back();
        last_touch_.pop_back();
        pos_.erase(evicted_val);
        if (idx < pool_.size()) {
            pos_.insert_or_assign(back_val, static_cast<uint32_t>(idx));
        }
        ++evictions_;
        evicted_since_reset_ = true;
    }

    u64 limit_;
    int max_regs_;
    std::vector<u64> pool_;
    std::vector<u64> last_touch_;
    FlatMap<uint32_t> pos_;
    FlatMap<int> pins_;
    FlatMap<double> scores_;
    std::vector<Step> plan_;
    u64 tick_ = 0;
    size_t evictions_ = 0;
    bool evicted_since_reset_ = false;
};

}  // namespace elf
