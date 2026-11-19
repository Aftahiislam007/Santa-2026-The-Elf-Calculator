// Provably globally optimal solver for tiny/small Elf Calculator puzzles.
//
// Mode A (value_limit <= 24):
//   Exact BFS over uint32_t bitmask states of 1..value_limit.
//   Since value_limit <= 24 < 48 registers, values never need to be evicted,
//   so the shortest path in the value-set state graph is provably optimal.
//
// Mode B (24 < value_limit <= 1024 && missing_targets <= 4):
//   Exact iterative deepening (IDA*) up to depth <= 10 with admissible bounds,
//   O(|pool|) 1-step & 2-step goal completion, and collision-free transposition
//   pruning. Verifies the 48-register constraint before accepting a solution.
//
// Larger puzzles exit with status 1 so `exact` only emits proven optima.

#include "common.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using elf::FlatMap;
using elf::Puzzle;
using elf::Step;
using elf::Timer;
using elf::u64;

struct Options {
    double time_limit = 2.0;
    int max_depth = 10;
    bool verbose = false;
};

// ---------------------------------------------------------------------------
// Mode A: Exact bitmask BFS for value_limit <= 24
// ---------------------------------------------------------------------------

struct VisitedNode {
    uint32_t parent = 0;
    char op = 0;
    uint8_t a = 0;
    uint8_t b = 0;
    uint8_t r = 0;
};

Step find_step_mask(uint32_t mask, int r, int limit) {
    int vals[24];
    int nv = 0;
    for (uint32_t m = mask; m; m &= m - 1) {
        vals[nv++] = __builtin_ctz(m) + 1;
    }
    for (int i = 0; i < nv; ++i) {
        int a = vals[i];
        for (int j = i; j < nv; ++j) {
            int b = vals[j];
            if (a + b == r && r <= limit) {
                return Step{'+', static_cast<u64>(a), static_cast<u64>(b), static_cast<u64>(r)};
            }
            if (a * b == r && r <= limit) {
                return Step{'*', static_cast<u64>(a), static_cast<u64>(b), static_cast<u64>(r)};
            }
            if (b > a) {
                if (b - a == r) {
                    return Step{'-', static_cast<u64>(b), static_cast<u64>(a), static_cast<u64>(r)};
                }
                if (b % a == 0 && b / a == r) {
                    return Step{'/', static_cast<u64>(b), static_cast<u64>(a), static_cast<u64>(r)};
                }
            } else if (r == 1) {
                return Step{'/', static_cast<u64>(a), static_cast<u64>(a), 1};
            }
        }
    }
    return Step{};
}

bool solve_mode_a(const Puzzle& pz, const Options& opt, const Timer& timer,
                  std::vector<Step>& out_plan) {
    const int L = static_cast<int>(pz.limit);
    uint32_t start_mask = 0;
    for (u64 s : pz.starts) {
        if (s == 0 || s > pz.limit) return false;
        start_mask |= (1u << (static_cast<int>(s) - 1));
    }
    if (start_mask == 0) return false;

    uint32_t goal_mask = 0;
    for (u64 t : pz.targets) {
        if (t == 0 || t > pz.limit) return false;
        goal_mask |= (1u << (static_cast<int>(t) - 1));
    }
    if ((start_mask & goal_mask) == goal_mask) {
        out_plan.clear();
        return true;
    }

    uint32_t pair_mask[25][25] = {};
    for (int a = 1; a <= L; ++a) {
        for (int b = a; b <= L; ++b) {
            uint32_t m = 0;
            if (a + b <= L) m |= (1u << (a + b - 1));
            if (a * b <= L) m |= (1u << (a * b - 1));
            if (b > a) {
                m |= (1u << (b - a - 1));
                if (b % a == 0) m |= (1u << (b / a - 1));
            } else {
                m |= 1u;  // a / a == 1
            }
            pair_mask[a][b] = m;
        }
    }

    FlatMap<VisitedNode> visited(4096);
    std::vector<uint32_t> queue;
    queue.reserve(4096);
    visited.insert(start_mask, VisitedNode{0, 0, 0, 0, 0});
    queue.push_back(start_mask);

    auto reconstruct = [&](uint32_t end_mask, const Step* final_step) {
        out_plan.clear();
        if (final_step) out_plan.push_back(*final_step);
        uint32_t cur = end_mask;
        while (cur != start_mask) {
            const VisitedNode* node = visited.find(cur);
            out_plan.push_back(Step{node->op, node->a, node->b, node->r});
            cur = node->parent;
        }
        std::reverse(out_plan.begin(), out_plan.end());
    };

    size_t head = 0;
    while (head < queue.size()) {
        if ((head & 65535u) == 0 && timer.elapsed_sec() > opt.time_limit) {
            return false;
        }
        uint32_t cur = queue[head++];

        int vals[24];
        int nv = 0;
        for (uint32_t m = cur; m; m &= m - 1) {
            vals[nv++] = __builtin_ctz(m) + 1;
        }

        uint32_t reach = 0;
        for (int i = 0; i < nv; ++i) {
            const uint32_t* row = pair_mask[vals[i]];
            for (int j = i; j < nv; ++j) {
                reach |= row[vals[j]];
            }
        }
        reach &= ~cur;

        uint32_t missing = goal_mask & ~cur;
        if ((missing & (missing - 1)) == 0) {
            // Only one target bit left: check immediate completion.
            if (reach & missing) {
                int r = __builtin_ctz(missing) + 1;
                Step last = find_step_mask(cur, r, L);
                reconstruct(cur, &last);
                if (opt.verbose) {
                    std::fprintf(stderr, "exact[A] %s: len=%zu visited=%zu time=%.4fs\n",
                                 pz.id.c_str(), out_plan.size(), visited.size(),
                                 timer.elapsed_sec());
                }
                return true;
            }
        }

        for (uint32_t m = reach; m; m &= m - 1) {
            uint32_t bit = m & -m;
            uint32_t nxt = cur | bit;
            if (visited.contains(nxt)) continue;
            int r = __builtin_ctz(bit) + 1;
            Step s = find_step_mask(cur, r, L);
            visited.insert(nxt, VisitedNode{cur, s.op, static_cast<uint8_t>(s.a),
                                            static_cast<uint8_t>(s.b), static_cast<uint8_t>(r)});
            if ((nxt & goal_mask) == goal_mask) {
                reconstruct(nxt, nullptr);
                if (opt.verbose) {
                    std::fprintf(stderr, "exact[A] %s: len=%zu visited=%zu time=%.4fs\n",
                                 pz.id.c_str(), out_plan.size(), visited.size(),
                                 timer.elapsed_sec());
                }
                return true;
            }
            queue.push_back(nxt);
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Mode B: Exact IDA* for value_limit <= 1024 && missing_targets <= 4
// ---------------------------------------------------------------------------

struct Bits1024 {
    uint64_t w[16] = {};

    bool test(int v) const {
        int idx = v - 1;
        return (w[idx >> 6] >> (idx & 63)) & 1ULL;
    }
    void set(int v) {
        int idx = v - 1;
        w[idx >> 6] |= (1ULL << (idx & 63));
    }
    void reset(int v) {
        int idx = v - 1;
        w[idx >> 6] &= ~(1ULL << (idx & 63));
    }
};

struct TTEntry {
    std::array<uint16_t, 10> key = {};
    uint8_t count = 0;
    uint8_t failed_rem = 0;
};

struct ExactTT {
    static constexpr size_t kBits = 19;  // 524,288 slots (~12 MB)
    static constexpr size_t kCap = size_t(1) << kBits;
    static constexpr size_t kMask = kCap - 1;

    std::vector<TTEntry> table;
    size_t used = 0;

    ExactTT() : table(kCap) {}

    static u64 hash_key(const std::array<uint16_t, 10>& key, uint8_t count) {
        u64 h = elf::splitmix64(count);
        for (uint8_t i = 0; i < count; ++i) {
            h = elf::splitmix64(h ^ (static_cast<u64>(key[i]) * 0x9e3779b97f4a7c15ULL));
        }
        return h;
    }

    uint8_t lookup(const std::array<uint16_t, 10>& key, uint8_t count) const {
        if (count == 0) return 0;
        size_t i = static_cast<size_t>(hash_key(key, count)) & kMask;
        for (size_t probe = 0; probe < 32; ++probe) {
            const TTEntry& e = table[i];
            if (e.count == 0) return 0;
            if (e.count == count && e.key == key) return e.failed_rem;
            i = (i + 1) & kMask;
        }
        return 0;
    }

    void store(const std::array<uint16_t, 10>& key, uint8_t count, uint8_t rem) {
        if (count == 0 || used * 2 >= kCap) return;
        size_t i = static_cast<size_t>(hash_key(key, count)) & kMask;
        for (size_t probe = 0; probe < 32; ++probe) {
            TTEntry& e = table[i];
            if (e.count == 0) {
                e.key = key;
                e.count = count;
                e.failed_rem = rem;
                ++used;
                return;
            }
            if (e.count == count && e.key == key) {
                if (rem > e.failed_rem) e.failed_rem = rem;
                return;
            }
            i = (i + 1) & kMask;
        }
    }
};

struct ModeBSolver {
    const Puzzle& pz;
    const Options& opt;
    const Timer& timer;
    int limit;
    std::vector<int> targets;
    Bits1024 in_pool;
    Bits1024 in_l1;
    std::vector<int> pool;
    std::vector<Step> cur_plan;
    std::vector<Step> best_plan;
    std::array<uint16_t, 10> added = {};
    uint8_t added_cnt = 0;
    bool use_tt = false;
    ExactTT tt;
    bool timed_out = false;
    uint64_t nodes = 0;

    ModeBSolver(const Puzzle& p, const Options& o, const Timer& t, const std::vector<u64>& miss)
        : pz(p), opt(o), timer(t), limit(static_cast<int>(p.limit)) {
        for (u64 v : miss) targets.push_back(static_cast<int>(v));
        for (u64 s : pz.starts) {
            int v = static_cast<int>(s);
            if (!in_pool.test(v)) {
                in_pool.set(v);
                pool.push_back(v);
            }
        }
        for (size_t i = 0; i < pool.size(); ++i) {
            add_l1_pairs(pool[i], i);
        }
    }

    void add_l1_pairs(int x, size_t max_idx) {
        for (size_t j = 0; j <= max_idx; ++j) {
            int p = pool[j];
            if (x + p <= limit && !in_pool.test(x + p)) in_l1.set(x + p);
            if (x * p <= limit && !in_pool.test(x * p)) in_l1.set(x * p);
            if (x > p) {
                if (!in_pool.test(x - p)) in_l1.set(x - p);
                if (x % p == 0 && !in_pool.test(x / p)) in_l1.set(x / p);
            } else if (p > x) {
                if (!in_pool.test(p - x)) in_l1.set(p - x);
                if (p % x == 0 && !in_pool.test(p / x)) in_l1.set(p / x);
            } else {
                if (!in_pool.test(1)) in_l1.set(1);
            }
        }
    }

    Step find_l1_step(int v) const {
        for (int c : pool) {
            if (v > c && in_pool.test(v - c)) {
                return Step{'+', static_cast<u64>(v - c), static_cast<u64>(c), static_cast<u64>(v)};
            }
            if (v + c <= limit && in_pool.test(v + c)) {
                return Step{'-', static_cast<u64>(v + c), static_cast<u64>(c), static_cast<u64>(v)};
            }
            if (c > v && in_pool.test(c - v)) {
                return Step{'-', static_cast<u64>(c), static_cast<u64>(c - v), static_cast<u64>(v)};
            }
            if (v % c == 0 && in_pool.test(v / c)) {
                return Step{'*', static_cast<u64>(v / c), static_cast<u64>(c), static_cast<u64>(v)};
            }
            if (v * c <= limit && in_pool.test(v * c)) {
                return Step{'/', static_cast<u64>(v * c), static_cast<u64>(c), static_cast<u64>(v)};
            }
            if (c % v == 0 && in_pool.test(c / v)) {
                return Step{'/', static_cast<u64>(c), static_cast<u64>(c / v), static_cast<u64>(v)};
            }
        }
        if (v == 1 && !pool.empty()) {
            return Step{'/', static_cast<u64>(pool[0]), static_cast<u64>(pool[0]), 1};
        }
        return Step{};
    }

    bool accept_current_plan() {
        auto checked = elf::verify_and_prune_plan(pz, cur_plan, 48);
        if (!checked.has_value()) return false;
        best_plan = std::move(*checked);
        return true;
    }

    // Check if single remaining target `t` can be completed in 2 steps: x in L1, then t = x op c (or x op x).
    bool try_complete_2(int t) {
        auto try_x = [&](int x, char op2, int a2, int b2) -> bool {
            if (x <= 0 || x > limit || !in_l1.test(x)) return false;
            Step s1 = find_l1_step(x);
            if (s1.op == 0) return false;
            cur_plan.push_back(s1);
            cur_plan.push_back(Step{op2, static_cast<u64>(a2), static_cast<u64>(b2), static_cast<u64>(t)});
            bool ok = accept_current_plan();
            cur_plan.pop_back();
            cur_plan.pop_back();
            return ok;
        };

        if ((t & 1) == 0 && try_x(t / 2, '+', t / 2, t / 2)) return true;
        int sq = static_cast<int>(elf::isqrt64(static_cast<u64>(t)));
        if (sq * sq == t && try_x(sq, '*', sq, sq)) return true;

        for (int c : pool) {
            if (t > c && try_x(t - c, '+', t - c, c)) return true;
            if (t + c <= limit && try_x(t + c, '-', t + c, c)) return true;
            if (c > t && try_x(c - t, '-', c, c - t)) return true;
            if (t % c == 0 && try_x(t / c, '*', t / c, c)) return true;
            if (t * c <= limit && try_x(t * c, '/', t * c, c)) return true;
            if (c % t == 0 && try_x(c / t, '/', c, c / t)) return true;
        }
        return false;
    }

    // Admissible check: can max_val in pool grow to >= max_target within rem_left steps?
    bool can_reach_max(int cur_max, int max_target, int rem_left) const {
        u64 m = static_cast<u64>(cur_max);
        for (int i = 0; i < rem_left && m < static_cast<u64>(max_target); ++i) {
            m = (m <= 1) ? 2 : std::min<u64>(static_cast<u64>(limit), m * m);
        }
        return m >= static_cast<u64>(max_target);
    }

    bool dfs(int rem_left, int cur_max) {
        if ((++nodes & 4095u) == 0 && timer.elapsed_sec() > opt.time_limit) {
            timed_out = true;
            return false;
        }

        int rem_targets[4];
        int n_rem = 0;
        int max_t = 0;
        for (int t : targets) {
            if (!in_pool.test(t)) {
                rem_targets[n_rem++] = t;
                if (t > max_t) max_t = t;
            }
        }
        if (n_rem == 0) return accept_current_plan();
        if (n_rem > rem_left) return false;
        if (!can_reach_max(cur_max, max_t, rem_left)) return false;

        // Exact 1-step base case
        if (rem_left == 1) {
            int t = rem_targets[0];
            if (!in_l1.test(t)) return false;
            Step s = find_l1_step(t);
            if (s.op == 0) return false;
            cur_plan.push_back(s);
            bool ok = accept_current_plan();
            cur_plan.pop_back();
            return ok;
        }

        // Exact 2-step shortcut for a single remaining target
        if (rem_left == 2 && n_rem == 1) {
            int t = rem_targets[0];
            if (in_l1.test(t)) {
                Step s = find_l1_step(t);
                if (s.op != 0) {
                    cur_plan.push_back(s);
                    bool ok = accept_current_plan();
                    cur_plan.pop_back();
                    if (ok) return true;
                }
            }
            return try_complete_2(t);
        }

        std::array<uint16_t, 10> sorted_key = {};
        if (use_tt && added_cnt > 0 && added_cnt <= 10) {
            sorted_key = added;
            std::sort(sorted_key.begin(), sorted_key.begin() + added_cnt);
            if (tt.lookup(sorted_key, added_cnt) >= static_cast<uint8_t>(rem_left)) {
                return false;
            }
        }

        auto try_branch = [&](int x) -> bool {
            Step s = find_l1_step(x);
            if (s.op == 0) return false;
            Bits1024 saved_l1 = in_l1;
            in_pool.set(x);
            in_l1.reset(x);
            pool.push_back(x);
            add_l1_pairs(x, pool.size() - 1);
            cur_plan.push_back(s);
            if (added_cnt < 10) added[added_cnt] = static_cast<uint16_t>(x);
            ++added_cnt;

            bool ok = dfs(rem_left - 1, std::max(cur_max, x));

            --added_cnt;
            if (added_cnt < 10) added[added_cnt] = 0;
            cur_plan.pop_back();
            pool.pop_back();
            in_pool.reset(x);
            in_l1 = saved_l1;
            return ok;
        };

        // Zero-spare-steps pruning: if n_rem == rem_left, the next step MUST cover a target.
        if (n_rem == rem_left) {
            for (int i = 0; i < n_rem; ++i) {
                int t = rem_targets[i];
                if (in_l1.test(t)) {
                    if (try_branch(t)) return true;
                    if (timed_out) return false;
                }
            }
            if (use_tt && !timed_out && added_cnt > 0 && added_cnt <= 10) {
                tt.store(sorted_key, added_cnt, static_cast<uint8_t>(rem_left));
            }
            return false;
        }

        // Move ordering: targets in L1 first, then all other values in L1.
        std::vector<int> cands;
        cands.reserve(256);
        for (int i = 0; i < n_rem; ++i) {
            if (in_l1.test(rem_targets[i])) cands.push_back(rem_targets[i]);
        }
        int num_target_cands = static_cast<int>(cands.size());
        for (int w = 0; w < 16; ++w) {
            uint64_t bits = in_l1.w[w];
            while (bits) {
                int b = __builtin_ctzll(bits);
                bits &= bits - 1;
                int v = (w << 6) + b + 1;
                if (v > limit) break;
                bool is_t = false;
                for (int i = 0; i < num_target_cands; ++i) {
                    if (cands[i] == v) {
                        is_t = true;
                        break;
                    }
                }
                if (!is_t) cands.push_back(v);
            }
        }

        for (int x : cands) {
            if (try_branch(x)) return true;
            if (timed_out) return false;
        }

        if (use_tt && !timed_out && added_cnt > 0 && added_cnt <= 10) {
            tt.store(sorted_key, added_cnt, static_cast<uint8_t>(rem_left));
        }
        return false;
    }

    bool solve(std::vector<Step>& out_plan) {
        int cur_max = 0;
        for (int v : pool) cur_max = std::max(cur_max, v);
        int lb = static_cast<int>(targets.size());
        int cap = std::min(opt.max_depth, 10);
        for (int depth = lb; depth <= cap; ++depth) {
            use_tt = (pz.starts.size() + static_cast<size_t>(depth) <= 47);
            if (dfs(depth, cur_max)) {
                out_plan = std::move(best_plan);
                if (opt.verbose) {
                    std::fprintf(stderr, "exact[B] %s: len=%zu depth=%d nodes=%llu time=%.4fs\n",
                                 pz.id.c_str(), out_plan.size(), depth,
                                 static_cast<unsigned long long>(nodes), timer.elapsed_sec());
                }
                return true;
            }
            if (timed_out) return false;
        }
        return false;
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
        if (a == "--time") {
            opt.time_limit = std::atof(next());
        } else if (a == "--max-depth") {
            opt.max_depth = std::atoi(next());
        } else if (a == "--verbose") {
            opt.verbose = true;
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }

    Puzzle pz;
    if (!elf::read_puzzle(std::cin, pz)) {
        std::fprintf(stderr, "bad input\n");
        return 2;
    }

    Timer timer;
    std::vector<Step> plan;

    if (pz.limit == 0 || pz.starts.empty()) return 1;

    if (pz.limit <= 24) {
        if (!solve_mode_a(pz, opt, timer, plan)) return 1;
    } else if (pz.limit <= 1024) {
        std::vector<u64> miss = pz.missing_targets();
        if (miss.empty()) {
            return 0;
        }
        if (miss.size() > 4) return 1;
        for (u64 s : pz.starts) {
            if (s == 0 || s > pz.limit) return 1;
        }
        for (u64 t : pz.targets) {
            if (t == 0 || t > pz.limit) return 1;
        }
        ModeBSolver solver(pz, opt, timer, miss);
        if (!solver.solve(plan)) return 1;
    } else {
        return 1;
    }

    auto verified = elf::verify_and_prune_plan(pz, plan, 48);
    if (!verified.has_value()) {
        std::fprintf(stderr, "internal verification failed for %s\n", pz.id.c_str());
        return 2;
    }
    elf::emit_plan(*verified);
    return 0;
}
