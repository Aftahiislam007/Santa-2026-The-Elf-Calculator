// Mixed-Hybrid search engine (`engine/mixed_hybrid.cpp`).
//
// Exploits the bimodal structure of `mixed` puzzles (and scales cleanly to
// `shared` and `independent` puzzles):
//   Step 1: Automatic Target Partitioning into `structured` (off-decade,
//           close pairs, affine links, sibling GCD > 500, 1-step smooth,
//           near-squares, and consensus hidden-hub members) vs `randomTargets`
//           (isolated top-decade targets in D in {6,9,12,15,18}).
//   Step 2: Phase 1 — Full Shared-DAG Hub-Mining Engine (ported from
//           `shared_dag.cpp`): continued-fraction rational consensus hubs,
//           pairwise GCDs, high-q shifted-affine links q*(H + d) + r,
//           near-square hubs, hub-to-hub DAG, demand-aware hub pinning,
//           hub-cluster DFS sweeping with on-the-fly sibling chaining,
//           multi-target + hub L2 voting, self-consistent 6-strategy construct,
//           and strict deadline/cycle/progress guards.
//   Step 3: Warm-Start Multi-Scale Basis Curation + Greedy L1-Entropy Top-Up +
//           Online Basis Absorption from Phase 1's live pool (preserving top
//           Phase-1 hubs while guaranteeing balanced half-decade logarithmic
//           coverage including sqrt(10^D) golden-zone entries up to |B|=40..42).
//   Step 4: Phase 2 — Full MITM Basis Engine (ported from `mitm_basis.cpp`):
//           unified Node table H = B u L1(B) u L2(B) (~150k..240k entries),
//           256 KB 2-hash Bloom filter, fast exact solvers (H, H op B, M op H,
//           (H op B) op B, M*H +- H, H +- H, (M op H) op B), t*d lottery,
//           O(|H|) two-pointer H x H MITM with layered residual solving,
//           linear-combo two-pointer scan V = (b*x +- y) +- r, divisor-remainder
//           beam search, per-target time budgeting, cross-target scratch
//           promotion, and exact verify_and_prune_plan verification.
//
// I/O protocol: see elfcalc/solvers/engine.py.

#include "common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using elf::FlatMap;
using elf::Puzzle;
using elf::Step;
using elf::Timer;
using elf::u128;
using elf::u64;
using i64 = int64_t;
using i128 = __int128_t;

struct Options {
    u64 seed = 0;
    int restarts = 3;
    double timeLimit = 4.0;
    int regs = 48;
    int basis = 41;
    bool basisExplicit = false;
    int beam = 48;
    bool verbose = false;
};

// ============================================================================
// Common Math, Hub Mining (`shared_dag`), and Target Partitioning
// ============================================================================

inline int digits10(u64 x) {
    int d = 1;
    while (x >= 10ULL) {
        x /= 10ULL;
        ++d;
    }
    return d;
}

inline u64 abs_i64(i64 x) {
    return x < 0 ? static_cast<u64>(-(x + 1)) + 1ULL : static_cast<u64>(x);
}

inline i64 round_div_i128(i128 num, u64 den) {
    if (den == 0) return 0;
    i128 d = static_cast<i128>(den);
    if (num >= 0) {
        return static_cast<i64>((num + d / 2) / d);
    }
    return -static_cast<i64>(((-num) + d / 2) / d);
}

inline double rough_build_cost(u64 x) {
    if (x == 0) return 0.0;
    if (x == 1) return 1.0;
    if (x <= 16) return 1.2 + 0.25 * std::log2(static_cast<double>(x));
    if (x <= 1024) return 0.48 * std::log2(static_cast<double>(x)) - 0.5;
    return 0.42 * std::log2(static_cast<double>(x)) - 0.9;
}

inline double rough_edge_cost(u64 q, i64 d, i64 r) {
    double c = 0.0;
    if (d != 0) c += rough_build_cost(abs_i64(d)) + 1.0;
    if (q > 1) c += rough_build_cost(q) + 1.0;
    if (r != 0) c += rough_build_cost(abs_i64(r)) + 1.0;
    return c;
}

inline bool is_smooth_bound(u64 n, u64 bound) {
    static constexpr u64 kPrimes[] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
        127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
        199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
        283, 293};
    if (n <= 1) return true;
    for (u64 p : kPrimes) {
        if (p > bound) break;
        while (n % p == 0) n /= p;
    }
    return n == 1;
}

static constexpr u64 kSmoothPrimes[] = {
    2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,  43,  47,
    53,  59,  61,  67,  71,  73,  79,  83,  89,  97,  101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
    199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
    383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
    467, 479, 487, 491, 499};

struct HubEdge {
    u64 target = 0;
    u64 q = 1;
    i64 d = 0;  // t = q * (H + d) + r
    i64 r = 0;
    double est_cost = 1e9;
};

struct HubLink {
    int parent_hub = -1;
    u64 q = 1;
    i64 d = 0;
    i64 r = 0;
};

struct HubInfo {
    u64 H = 0;
    bool is_target = false;
    std::vector<HubEdge> edges;
    HubLink parent_link;
    std::vector<int> child_hubs;
    double savings = 0.0;
    int exact_count = 0;
};

struct MinedDAG {
    std::vector<HubInfo> hubs;
    std::unordered_map<u64, int> hub_idx_of;
    std::unordered_map<u64, std::vector<int>> target_hubs;
    std::unordered_map<u64, int> freq_constants;
};

inline bool match_shifted_affine(u64 t, u64 H, u64 limit, u64 max_q,
                                 double min_save, bool strict_caps, u64 r_cap, u64 d_cap,
                                 u64& out_q, i64& out_d, i64& out_r, double& out_cost) {
    if (H <= 100 || t < H / 2) return false;
    u64 q = (t + H / 2) / H;
    if (q < 1 || q > max_q) return false;

    double tc = rough_build_cost(t);
    double best_c = 1e18;
    bool found = false;
    u64 bq = 1;
    i64 bd = 0, br = 0;

    auto try_q = [&](u64 qq) {
        if (qq < 1 || qq > max_q) return;
        i128 diff = static_cast<i128>(t) - static_cast<i128>(qq) * static_cast<i128>(H);
        // 1. Direct affine (d = 0)
        if (static_cast<u128>(qq) * static_cast<u128>(H) <= limit) {
            i128 adiff = diff < 0 ? -diff : diff;
            if (adiff < static_cast<i128>(t / 2)) {
                u64 ar = static_cast<u64>(adiff);
                bool cap_ok = !strict_caps || ar <= std::min<u64>(r_cap, std::max<u64>(500, H / 500));
                if (cap_ok && (qq == 1 || ar < H)) {
                    i64 r64 = static_cast<i64>(diff);
                    double ec = rough_edge_cost(qq, 0, r64);
                    if (ec + min_save <= tc || (t == H && qq == 1 && r64 == 0)) {
                        if (ec < best_c) {
                            best_c = ec;
                            bq = qq;
                            bd = 0;
                            br = r64;
                            found = true;
                        }
                    }
                }
            }
        }
        // 2. Shifted affine (qq >= 2): diff = qq * d + r
        if (qq >= 2) {
            i64 d = round_div_i128(diff, qq);
            i128 rem = diff - static_cast<i128>(qq) * static_cast<i128>(d);
            u64 ad = abs_i64(d);
            i128 arem = rem < 0 ? -rem : rem;
            if (ad > 0 && ad < H / 4 && arem < static_cast<i128>(t / 4)) {
                u64 ar = static_cast<u64>(arem);
                bool cap_ok =
                    !strict_caps ||
                    (ad <= std::min<u64>(d_cap, std::max<u64>(500, H / 500)) &&
                     ar <= std::min<u64>(r_cap, std::max<u64>(500, H / 500)));
                if (cap_ok && (d > 0 || ad < H)) {
                    u64 H_sh = d >= 0 ? (H + ad) : (H - ad);
                    if (H_sh > 0 && H_sh <= limit &&
                        static_cast<u128>(qq) * static_cast<u128>(H_sh) <= limit) {
                        i64 r64 = static_cast<i64>(rem);
                        double ec = rough_edge_cost(qq, d, r64);
                        if (ec + min_save <= tc) {
                            if (ec < best_c) {
                                best_c = ec;
                                bq = qq;
                                bd = d;
                                br = r64;
                                found = true;
                            }
                        }
                    }
                }
            }
        }
    };

    try_q(q);
    if (q > 1) try_q(q - 1);
    try_q(q + 1);

    if (found) {
        out_q = bq;
        out_d = bd;
        out_r = br;
        out_cost = best_c;
        return true;
    }
    return false;
}

void rational_approx(u64 num, u64 den, u64 max_den, u64 max_num,
                     std::vector<std::pair<u64, u64>>& out) {
    out.clear();
    if (den == 0 || num == 0) return;
    u64 p_prev = 0, p_cur = 1;
    u64 q_prev = 1, q_cur = 0;
    u64 a = num, b = den;
    while (b > 0) {
        u64 quot = a / b;
        u64 rem = a % b;
        if (quot > 0) {
            u128 q_next = static_cast<u128>(quot) * q_cur + q_prev;
            u128 p_next = static_cast<u128>(quot) * p_cur + p_prev;
            if (q_next > max_den || p_next > max_num) {
                u64 k_max = quot;
                if (q_cur > 0) k_max = std::min(k_max, (max_den - q_prev) / q_cur);
                if (p_cur > 0) k_max = std::min(k_max, (max_num - p_prev) / p_cur);
                if (k_max > 0) {
                    u64 ps = k_max * p_cur + p_prev;
                    u64 qs = k_max * q_cur + q_prev;
                    if (qs >= 1 && qs <= max_den && ps >= 1 && ps <= max_num) {
                        out.emplace_back(ps, qs);
                    }
                }
                break;
            }
            p_prev = p_cur;
            q_prev = q_cur;
            p_cur = static_cast<u64>(p_next);
            q_cur = static_cast<u64>(q_next);
            if (q_cur >= 1 && q_cur <= max_den && p_cur >= 1 && p_cur <= max_num) {
                out.emplace_back(p_cur, q_cur);
            }
        }
        a = b;
        b = rem;
    }
}

MinedDAG mine_shared_dag(const std::vector<u64>& targets, u64 limit,
                         u64 r_cap, u64 d_cap, u64 max_q,
                         int min_hub_deg, double min_savings, bool add_target_links,
                         const Timer& timer, double hard_dl) {
    MinedDAG dag;
    std::vector<u64> ts = targets;
    std::sort(ts.begin(), ts.end());
    ts.erase(std::unique(ts.begin(), ts.end()), ts.end());
    const size_t n = ts.size();
    if (n < 2) return dag;

    std::unordered_set<u64> target_set(ts.begin(), ts.end());
    std::vector<u64> cands;
    cands.reserve(n * n * 2);

    std::vector<std::pair<u64, u64>> rats;
    for (size_t i = 0; i < n; ++i) {
        if (timer.elapsed_sec() >= hard_dl) break;
        u64 ti = ts[i];
        if (ti > 10000) {
            u64 s = elf::isqrt64(ti);
            if (ti - s * s <= std::min<u64>(r_cap, s / 10)) {
                cands.push_back(s * s);
            } else if (static_cast<u128>(s + 1) * (s + 1) <= limit &&
                       (s + 1) * (s + 1) - ti <= std::min<u64>(r_cap, s / 10)) {
                cands.push_back((s + 1) * (s + 1));
            }
        }

        for (size_t j = i + 1; j < n; ++j) {
            u64 tj = ts[j];

            // 1. Direct additive siblings
            u64 diff_cap = std::min<u64>(100000, std::max<u64>(5000, ti / 100));
            if (tj - ti <= diff_cap && ti > 100) {
                cands.push_back(ti);
                cands.push_back(tj);
            }

            // 2. Pairwise GCD hubs
            u64 g = std::gcd(ti, tj);
            if (g > 500 && ti / g <= max_q && tj / g <= max_q) {
                cands.push_back(g);
            } else if (g > 100 && ti / g <= 512 && tj / g <= 512) {
                cands.push_back(g);
            }

            // 3. Direct & shifted affine links (up to max_q)
            if (ti > 100) {
                u64 q = (tj + ti / 2) / ti;
                if (q >= 1 && q <= max_q) {
                    i128 diff = static_cast<i128>(tj) - static_cast<i128>(q) * static_cast<i128>(ti);
                    u64 tol_i = std::min<u64>(r_cap, std::max<u64>(500, ti / 500));
                    if (q == 1) {
                        if (diff >= -static_cast<i128>(tol_i) && diff <= static_cast<i128>(tol_i)) {
                            cands.push_back(ti);
                        }
                    } else {
                        i64 d = round_div_i128(diff, q);
                        i128 rem = diff - static_cast<i128>(q) * static_cast<i128>(d);
                        u64 ad = abs_i64(d);
                        if (ad <= tol_i &&
                            rem >= -static_cast<i128>(tol_i) && rem <= static_cast<i128>(tol_i)) {
                            if (d >= 0 || ad < ti) {
                                u64 h_est = d >= 0 ? (ti + ad) : (ti - ad);
                                if (h_est > 100 && h_est <= limit) {
                                    cands.push_back(h_est);
                                    cands.push_back(ti);
                                }
                            }
                        }
                    }
                }
            }

            // 4. Rational consensus hidden hubs via continued fractions
            if (ti > 100 && tj / ti <= 64) {
                rational_approx(tj, ti, 32, 512, rats);
                for (const auto& [p, q] : rats) {
                    if (q < 2 || p <= q) continue;
                    u64 h1 = (ti + q / 2) / q;
                    u64 h2 = (tj + p / 2) / p;
                    for (u64 h_est : {h1, h2}) {
                        if (h_est <= 100 || h_est > limit) continue;
                        u64 tol = std::min<u64>(r_cap, std::max<u64>(500, h_est / 500));
                        i128 ri = static_cast<i128>(ti) - static_cast<i128>(q) * static_cast<i128>(h_est);
                        i128 rj = static_cast<i128>(tj) - static_cast<i128>(p) * static_cast<i128>(h_est);
                        if (ri >= -static_cast<i128>(tol) && ri <= static_cast<i128>(tol) &&
                            rj >= -static_cast<i128>(tol) && rj <= static_cast<i128>(tol)) {
                            cands.push_back(h_est);
                        }
                    }
                }
            }
        }
    }

    std::sort(cands.begin(), cands.end());
    cands.erase(std::unique(cands.begin(), cands.end()), cands.end());

    // Cluster nearby candidate hubs
    std::vector<std::vector<u64>> clusters;
    for (u64 h : cands) {
        if (h <= 100 || h > limit) continue;
        u64 gap = 2 * std::min<u64>(r_cap, std::max<u64>(500, h / 500));
        if (clusters.empty() || h - clusters.back().back() > gap) {
            clusters.push_back({h});
        } else {
            clusters.back().push_back(h);
        }
    }

    std::vector<HubInfo> raw_hubs;
    for (const auto& cl : clusters) {
        if (timer.elapsed_sec() >= hard_dl) break;
        u64 h0 = cl[cl.size() / 2];
        u64 tol0 = std::min<u64>(r_cap, std::max<u64>(500, h0 / 500));
        std::vector<u64> test_h = cl;

        for (u64 t : ts) {
            if (t < h0 / 2) continue;
            u64 q = (t + h0 / 2) / h0;
            if (q >= 1 && q <= max_q) {
                i128 diff = static_cast<i128>(t) - static_cast<i128>(q) * static_cast<i128>(h0);
                i64 d = round_div_i128(diff, q);
                u64 ad = abs_i64(d);
                if (ad <= 2 * tol0) {
                    u64 h_ref = (t + q / 2) / q;
                    if (h_ref > 100 && h_ref <= limit) test_h.push_back(h_ref);
                    if (t % q == 0 && t / q > 100) test_h.push_back(t / q);
                }
            }
        }
        {
            u64 s = elf::isqrt64(h0);
            if (h0 - s * s <= 2 * tol0 && s * s > 100) test_h.push_back(s * s);
            if (static_cast<u128>(s + 1) * (s + 1) <= limit &&
                (s + 1) * (s + 1) - h0 <= 2 * tol0) {
                test_h.push_back((s + 1) * (s + 1));
            }
        }
        std::sort(test_h.begin(), test_h.end());
        test_h.erase(std::unique(test_h.begin(), test_h.end()), test_h.end());

        if (test_h.size() > 64) {
            std::vector<u64> sub;
            sub.reserve(64);
            for (u64 h : test_h) {
                if (target_set.count(h)) sub.push_back(h);
            }
            size_t step = std::max<size_t>(1, test_h.size() / 48);
            for (size_t i = 0; i < test_h.size() && sub.size() < 64; i += step) {
                sub.push_back(test_h[i]);
            }
            std::sort(sub.begin(), sub.end());
            sub.erase(std::unique(sub.begin(), sub.end()), sub.end());
            test_h.swap(sub);
        }

        bool found_best = false;
        HubInfo best_hub;
        double best_rank = -1e18;

        for (u64 h : test_h) {
            if (h <= 100 || h > limit) continue;
            std::vector<HubEdge> edges;
            int exact_cnt = 0;
            double edge_savings = 0.0;
            for (u64 t : ts) {
                u64 q = 0;
                i64 d = 0, r = 0;
                double ec = 0.0;
                if (match_shifted_affine(t, h, limit, max_q, 1.5, true, r_cap, d_cap,
                                         q, d, r, ec)) {
                    double tc = rough_build_cost(t);
                    edges.push_back(HubEdge{t, q, d, r, ec});
                    if (r == 0 && d == 0) ++exact_cnt;
                    edge_savings += std::max(0.0, tc - ec);
                }
            }
            if (static_cast<int>(edges.size()) < min_hub_deg) continue;
            double h_cost = rough_build_cost(h);
            u64 s = elf::isqrt64(h);
            if (s * s == h) h_cost = rough_build_cost(s) + 1.0;
            double net_savings = edge_savings - h_cost;
            if (net_savings < min_savings) continue;

            double rank = net_savings + 0.6 * exact_cnt + (target_set.count(h) ? 1.0 : 0.0);
            if (!found_best || rank > best_rank) {
                found_best = true;
                best_rank = rank;
                best_hub = HubInfo{};
                best_hub.H = h;
                best_hub.is_target = target_set.count(h) > 0;
                best_hub.edges = std::move(edges);
                best_hub.savings = net_savings;
                best_hub.exact_count = exact_cnt;
            }
        }
        if (found_best) {
            raw_hubs.push_back(std::move(best_hub));
        }
    }

    // Greedy set-cover selection of consensus hubs
    std::sort(raw_hubs.begin(), raw_hubs.end(), [](const HubInfo& a, const HubInfo& b) {
        if (a.savings != b.savings) return a.savings > b.savings;
        if (a.edges.size() != b.edges.size()) return a.edges.size() > b.edges.size();
        return a.H < b.H;
    });

    std::unordered_map<u64, double> best_t_cost;
    for (u64 t : ts) best_t_cost[t] = rough_build_cost(t);

    for (auto& hub : raw_hubs) {
        double marginal_gain = 0.0;
        int improved_cnt = 0;
        for (const auto& e : hub.edges) {
            double cur = best_t_cost[e.target];
            if (e.est_cost < cur) {
                marginal_gain += (cur - e.est_cost);
                ++improved_cnt;
            }
        }
        double h_cost = rough_build_cost(hub.H);
        u64 s = elf::isqrt64(hub.H);
        if (s * s == hub.H) h_cost = rough_build_cost(s) + 1.0;

        if (improved_cnt >= min_hub_deg && marginal_gain - h_cost >= min_savings) {
            for (const auto& e : hub.edges) {
                if (e.est_cost < best_t_cost[e.target]) {
                    best_t_cost[e.target] = e.est_cost;
                }
            }
            dag.hubs.push_back(std::move(hub));
        }
    }

    // Target-to-target bit-saving shifted-affine edges
    if (add_target_links) {
        std::unordered_map<u64, size_t> existing_hub;
        for (size_t i = 0; i < dag.hubs.size(); ++i) existing_hub[dag.hubs[i].H] = i;

        for (size_t j = 0; j < n; ++j) {
            u64 tj = ts[j];
            double cur_best = best_t_cost[tj];
            int best_parent_i = -1;
            u64 bq = 1;
            i64 bd = 0, br = 0;
            double bec = cur_best;

            for (size_t i = 0; i < j; ++i) {
                u64 ti = ts[i];
                u64 q = 0;
                i64 d = 0, r = 0;
                double ec = 0.0;
                if (match_shifted_affine(tj, ti, limit, 10000000ULL, 2.0, false,
                                         r_cap, d_cap, q, d, r, ec)) {
                    if (ec + 1.0 < bec) {
                        bec = ec;
                        best_parent_i = static_cast<int>(i);
                        bq = q;
                        bd = d;
                        br = r;
                    }
                }
            }
            if (best_parent_i >= 0) {
                u64 ti = ts[best_parent_i];
                best_t_cost[tj] = bec;
                auto it = existing_hub.find(ti);
                if (it != existing_hub.end()) {
                    auto& edges = dag.hubs[it->second].edges;
                    bool dup = false;
                    for (auto& e : edges) {
                        if (e.target == tj) {
                            if (bec < e.est_cost) e = HubEdge{tj, bq, bd, br, bec};
                            dup = true;
                            break;
                        }
                    }
                    if (!dup) edges.push_back(HubEdge{tj, bq, bd, br, bec});
                    dag.hubs[it->second].savings += (cur_best - bec);
                } else {
                    HubInfo h;
                    h.H = ti;
                    h.is_target = true;
                    h.edges.push_back(HubEdge{ti, 1, 0, 0, 0.0});
                    h.edges.push_back(HubEdge{tj, bq, bd, br, bec});
                    h.savings = (cur_best - bec);
                    h.exact_count = (br == 0 && bd == 0) ? 1 : 0;
                    existing_hub[ti] = dag.hubs.size();
                    dag.hubs.push_back(std::move(h));
                }
            }
        }
    }

    std::sort(dag.hubs.begin(), dag.hubs.end(),
              [](const HubInfo& a, const HubInfo& b) { return a.H < b.H; });

    // Super-hubs via pairwise GCD of selected hubs
    {
        std::unordered_set<u64> existing_h;
        existing_h.reserve(dag.hubs.size() * 2 + 8);
        for (const auto& h : dag.hubs) existing_h.insert(h.H);

        std::vector<u64> extra_gcds;
        for (size_t i = 0; i < dag.hubs.size(); ++i) {
            for (size_t j = i + 1; j < dag.hubs.size(); ++j) {
                u64 g = std::gcd(dag.hubs[i].H, dag.hubs[j].H);
                if (g > 500 && dag.hubs[i].H / g <= 4096 && dag.hubs[j].H / g <= 4096) {
                    if (!existing_h.count(g)) extra_gcds.push_back(g);
                }
            }
        }
        std::sort(extra_gcds.begin(), extra_gcds.end());
        extra_gcds.erase(std::unique(extra_gcds.begin(), extra_gcds.end()), extra_gcds.end());
        for (u64 g : extra_gcds) {
            if (existing_h.count(g)) continue;
            int cnt = 0;
            for (const auto& h : dag.hubs) {
                if (h.H % g == 0 && h.H / g <= 4096) ++cnt;
            }
            if (cnt >= 2) {
                HubInfo sh;
                sh.H = g;
                sh.is_target = target_set.count(g) > 0;
                sh.savings = rough_build_cost(g);
                existing_h.insert(g);
                dag.hubs.push_back(std::move(sh));
            }
        }
        std::sort(dag.hubs.begin(), dag.hubs.end(),
                  [](const HubInfo& a, const HubInfo& b) { return a.H < b.H; });
    }

    dag.hub_idx_of.clear();
    for (size_t i = 0; i < dag.hubs.size(); ++i) {
        dag.hub_idx_of[dag.hubs[i].H] = static_cast<int>(i);
    }

    for (size_t i = 0; i < dag.hubs.size(); ++i) {
        auto& edges = dag.hubs[i].edges;
        std::sort(edges.begin(), edges.end(), [](const HubEdge& a, const HubEdge& b) {
            if (a.est_cost != b.est_cost) return a.est_cost < b.est_cost;
            return a.target < b.target;
        });
        for (const auto& e : edges) {
            dag.target_hubs[e.target].push_back(static_cast<int>(i));
        }
    }

    // Hub-to-hub DAG links (H_a -> H_b for a < b)
    for (size_t b = 0; b < dag.hubs.size(); ++b) {
        double best_cost = rough_build_cost(dag.hubs[b].H) - 1.5;
        int best_a = -1;
        u64 best_q = 1;
        i64 best_d = 0, best_r = 0;
        for (size_t a = 0; a < b; ++a) {
            u64 q = 0;
            i64 d = 0, r = 0;
            double ec = 0.0;
            if (match_shifted_affine(dag.hubs[b].H, dag.hubs[a].H, limit, 10000000ULL,
                                     1.5, false, r_cap, d_cap, q, d, r, ec)) {
                if (ec < best_cost) {
                    best_cost = ec;
                    best_a = static_cast<int>(a);
                    best_q = q;
                    best_d = d;
                    best_r = r;
                }
            }
        }
        if (best_a >= 0) {
            dag.hubs[b].parent_link = HubLink{best_a, best_q, best_d, best_r};
            dag.hubs[best_a].child_hubs.push_back(static_cast<int>(b));
        }
    }

    // Frequent small constants
    auto tally = [&](u64 v) {
        if (v >= 2 && v <= 50000) ++dag.freq_constants[v];
    };
    for (const auto& h : dag.hubs) {
        for (const auto& e : h.edges) {
            if (e.q >= 2) tally(e.q);
            if (e.d != 0) tally(abs_i64(e.d));
            if (e.r != 0) tally(abs_i64(e.r));
        }
        if (h.parent_link.parent_hub >= 0) {
            if (h.parent_link.q >= 2) tally(h.parent_link.q);
            if (h.parent_link.d != 0) tally(abs_i64(h.parent_link.d));
            if (h.parent_link.r != 0) tally(abs_i64(h.parent_link.r));
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            u64 diff = ts[j] - ts[i];
            if (diff <= 50000) tally(diff);
            else break;
        }
    }
    for (auto it = dag.freq_constants.begin(); it != dag.freq_constants.end();) {
        if (it->second < 2) it = dag.freq_constants.erase(it);
        else ++it;
    }

    return dag;
}

struct PartitionResult {
    int topD = 0;
    std::vector<u64> structured;
    std::vector<u64> randomTargets;
};

PartitionResult partition_targets(const Puzzle& pz) {
    PartitionResult res;
    std::vector<u64> ts = pz.missing_targets();
    const size_t n = ts.size();
    if (n == 0) return res;

    std::vector<int> digs(n);
    int cnt[21] = {};
    for (size_t i = 0; i < n; ++i) {
        digs[i] = digits10(ts[i]);
        if (digs[i] <= 20) ++cnt[digs[i]];
    }

    static constexpr int kDecades[5] = {6, 9, 12, 15, 18};
    int bestD = 0;
    int bestCnt = -1;
    for (int d : kDecades) {
        int above = 0;
        for (size_t i = 0; i < n; ++i) {
            if (digs[i] > d) ++above;
        }
        if (above <= 2) {
            if (cnt[d] > bestCnt || (cnt[d] == bestCnt && d > bestD)) {
                bestCnt = cnt[d];
                bestD = d;
            }
        }
    }
    if (bestCnt <= 0) {
        for (int d : kDecades) {
            if (cnt[d] > bestCnt || (cnt[d] == bestCnt && d > bestD)) {
                bestCnt = cnt[d];
                bestD = d;
            }
        }
    }
    res.topD = bestD;
    if (bestCnt <= 0) {
        res.structured = ts;
        return res;
    }

    // Cheap <=1-step constants from starting values
    std::vector<u64> cs;
    {
        FlatMap<uint8_t> seen(128);
        auto addC = [&](u64 v) {
            if (v > 0 && v <= pz.limit && seen.insert(v, 1)) cs.push_back(v);
        };
        addC(1);
        for (u64 s : pz.starts) addC(s);
        for (size_t i = 0; i < pz.starts.size(); ++i) {
            for (size_t j = i; j < pz.starts.size(); ++j) {
                u64 a = pz.starts[i], b = pz.starts[j];
                if (a <= pz.limit && b <= pz.limit - a) addC(a + b);
                u64 prod = elf::safe_mul(a, b, pz.limit);
                if (prod) addC(prod);
                if (a > b) {
                    addC(a - b);
                    if (a % b == 0) addC(a / b);
                } else if (b > a) {
                    addC(b - a);
                    if (b % a == 0) addC(b / a);
                }
            }
        }
        std::sort(cs.begin(), cs.end());
    }

    std::vector<uint8_t> isStruct(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (digs[i] != bestD) isStruct[i] = 1;
        // Tight near-square top-decade targets are structured!
        if (!isStruct[i] && ts[i] >= 1000000ULL) {
            u64 s = elf::isqrt64(ts[i]);
            u64 cap = std::min<u64>(5000ULL, s / 100ULL);
            if (ts[i] - s * s <= cap) {
                isStruct[i] = 1;
            } else if (static_cast<u128>(s + 1) * (s + 1) <= pz.limit &&
                       (s + 1) * (s + 1) - ts[i] <= cap) {
                isStruct[i] = 1;
            }
        }
    }

    // Close neighbors, affine links, sibling GCD > 500
    for (size_t i = 0; i < n; ++i) {
        u64 lo = ts[i];
        for (size_t j = i + 1; j < n; ++j) {
            u64 hi = ts[j];
            if (static_cast<u128>(hi - lo) * 1000ULL < lo) {
                isStruct[i] = 1;
                isStruct[j] = 1;
            }
            u64 q = (hi + lo / 2) / lo;
            if (q >= 1 && q <= 512) {
                u128 prod = static_cast<u128>(q) * lo;
                u64 diff = prod > hi ? static_cast<u64>(prod - hi)
                                     : static_cast<u64>(hi - prod);
                if (diff <= std::max<u64>(5000ULL, lo / 100000ULL)) {
                    isStruct[i] = 1;
                    isStruct[j] = 1;
                }
            }
            if (std::gcd(lo, hi) > 500ULL) {
                isStruct[i] = 1;
                isStruct[j] = 1;
            }
        }
    }

    // 1-step <=B_sm smooth check for top-decade targets
    const u64 smBound = (bestD <= 6) ? 20ULL : (bestD <= 9 ? 100ULL : 300ULL);
    for (size_t i = 0; i < n; ++i) {
        if (isStruct[i]) continue;
        u64 t = ts[i];
        bool sm = is_smooth_bound(t, smBound);
        if (!sm) {
            for (u64 c : cs) {
                if (t <= pz.limit - c && is_smooth_bound(t + c, smBound)) {
                    sm = true;
                    break;
                }
                if (t > c && is_smooth_bound(t - c, smBound)) {
                    sm = true;
                    break;
                }
            }
        }
        if (sm) isStruct[i] = 1;
    }

    for (size_t i = 0; i < n; ++i) {
        if (isStruct[i]) {
            res.structured.push_back(ts[i]);
        } else {
            res.randomTargets.push_back(ts[i]);
        }
    }
    return res;
}

// ============================================================================
// Step 2: Phase 1 — Shared-DAG Hub-Mining Engine (ported from `shared_dag.cpp`)
// ============================================================================

struct Phase1Config {
    u64 seed = 0;
    u64 r_cap = 20000;
    u64 d_cap = 10000;
    u64 max_q = 500000;
    int min_hub_deg = 2;
    double min_savings = 1.5;
    bool add_target_links = true;
    bool strict_hub_first = true;
    int b_full = 3;
    int b_construct = 5;
    double work_limit = 3.5e6;
    bool randomize_order = false;
    bool allow_smooth_shift = true;
};

struct Phase1Engine {
    struct L1E {
        u64 v, a, b;
        char op;
    };
    struct Cand {
        u64 r, a, q;
        uint8_t ca, cq, kind;  // 0: a*q+r, 1: a*q-r, 2: a+r, 3: a-r
        int rem;
    };

    int maxRegs = 48;
    u64 limit = 0;
    const Phase1Config& cfg;
    const Timer& timer;
    double deadline = 1e18;
    double hard_deadline = 1e18;
    double fatal_deadline = 1e18;
    bool can_abort_on_deadline = false;
    bool fast_fallback_mode = false;
    bool timed_out = false;
    bool aborted = false;

    std::vector<u64> pool;
    std::vector<u64> stamp;
    std::unordered_map<u64, int> pins;
    u64 tick = 0;
    size_t created = 0;
    size_t evictions = 0;
    FlatMap<uint32_t> pool_idx;
    std::vector<L1E> l1;
    FlatMap<uint32_t> l1_idx;
    std::vector<Step> plan;

    std::vector<u64> remaining;
    std::unordered_set<u64> remaining_set;
    std::unordered_set<u64> covered_targets;
    std::unordered_set<u64> start_set;
    FlatMap<uint8_t> coveredEver;

    MinedDAG dag;
    std::vector<int> hub_rem_children;
    std::vector<int> hub_rem_child_hubs;
    std::vector<uint8_t> hub_built;
    std::unordered_map<u64, int> freq_rem;

    FlatMap<int> fail_b;
    std::vector<std::pair<u64, uint8_t>> M;
    bool m_dirty = true;
    double work = 0.0;
    static constexpr size_t kMaxPlanSteps = 20000;
    static constexpr int kMaxConstructDepth = 16;
    std::vector<u64> construct_stack;
    std::mt19937_64 rng;

    Phase1Engine(int regs, u64 lim, const Phase1Config& c, const Timer& t, double dl)
        : maxRegs(regs),
          limit(lim),
          cfg(c),
          timer(t),
          deadline(dl),
          pool_idx(128),
          l1_idx(4096),
          coveredEver(256),
          fail_b(1024),
          rng(c.seed * 0x9E3779B97F4A7C15ULL + 1ULL) {}

    bool check_time() {
        if (aborted) return true;
        double el = timer.elapsed_sec();
        if (el >= fatal_deadline || plan.size() > kMaxPlanSteps) {
            timed_out = true;
            aborted = true;
            return true;
        }
        if (el >= hard_deadline) {
            timed_out = true;
            if (can_abort_on_deadline) {
                aborted = true;
            } else {
                fast_fallback_mode = true;
            }
            return true;
        }
        if (el >= deadline) {
            timed_out = true;
            return true;
        }
        return timed_out;
    }

    bool on_stack(u64 x) const {
        for (u64 s : construct_stack) {
            if (s == x) return true;
        }
        return false;
    }

    struct StackGuard {
        std::vector<u64>& st;
        StackGuard(std::vector<u64>& s, u64 v) : st(s) { st.push_back(v); }
        ~StackGuard() { st.pop_back(); }
    };

    struct PinGuard {
        Phase1Engine& eng;
        u64 val;
        bool active;
        PinGuard(Phase1Engine& e, u64 v, bool act = true) : eng(e), val(v), active(act) {
            if (active) eng.pin(val);
        }
        void release() {
            if (active) {
                eng.unpin(val);
                active = false;
            }
        }
        ~PinGuard() { release(); }
    };

    inline u64 eval(char op, u64 a, u64 b) const { return elf::apply_op(op, a, b, limit); }
    inline bool in_pool(u64 x) const { return pool_idx.contains(x); }
    inline int c01(u64 x) const { return in_pool(x) ? 0 : (l1_idx.contains(x) ? 1 : 2); }

    void add_l1(u64 v, char op, u64 a, u64 b) {
        if (v == 0 || in_pool(v)) return;
        if (l1_idx.insert(v, static_cast<uint32_t>(l1.size()))) {
            l1.push_back({v, a, b, op});
        }
    }

    void link_value(u64 v) {
        for (u64 p : pool) {
            add_l1(eval('+', v, p), '+', v, p);
            add_l1(eval('*', v, p), '*', v, p);
            if (v > p) {
                add_l1(v - p, '-', v, p);
                if (v % p == 0) add_l1(v / p, '/', v, p);
            } else if (p > v) {
                add_l1(p - v, '-', p, v);
                if (p % v == 0) add_l1(p / v, '/', p, v);
            } else {
                add_l1(1, '/', v, v);
                break;
            }
        }
    }

    void add_value(u64 v, bool link) {
        coveredEver.insert(v, 1);
        if (!pool_idx.insert(v, static_cast<uint32_t>(pool.size()))) return;
        pool.push_back(v);
        stamp.push_back(++tick);
        ++created;
        if (link) link_value(v);
        m_dirty = true;
    }

    void reindex() {
        pool_idx.clear();
        for (size_t j = 0; j < pool.size(); ++j) {
            pool_idx.insert(pool[j], static_cast<uint32_t>(j));
        }
    }

    void rebuild() {
        std::vector<u64> old;
        old.swap(pool);
        pool_idx.clear();
        l1.clear();
        l1_idx.clear();
        for (u64 v : old) {
            pool_idx.insert(v, static_cast<uint32_t>(pool.size()));
            pool.push_back(v);
            link_value(v);
        }
        m_dirty = true;
    }

    void pin(u64 v) { ++pins[v]; }
    void unpin(u64 v) {
        auto it = pins.find(v);
        if (it != pins.end()) {
            if (--it->second <= 0) pins.erase(it);
        }
    }

    double evict_score(u64 v) const {
        auto hit = dag.hub_idx_of.find(v);
        if (hit != dag.hub_idx_of.end()) {
            int hi = hit->second;
            int need = hub_rem_children[hi] + 2 * hub_rem_child_hubs[hi];
            if (need > 0) {
                return 500.0 + 100.0 * static_cast<double>(need);
            }
        }
        auto fit = freq_rem.find(v);
        if (fit != freq_rem.end() && fit->second > 0) {
            return 50.0 + 10.0 * static_cast<double>(fit->second);
        }
        if (v == 1 || v == 2) return 90.0;
        if (start_set.count(v) && v <= 64) return 75.0;
        if (v <= 64) return 30.0;
        if (v <= 1024) return 5.0;
        if (covered_targets.count(v)) return -1000.0;
        return -100.0;
    }

    void on_value_produced(u64 v) {
        coveredEver.insert(v, 1);
        if (remaining_set.erase(v)) {
            covered_targets.insert(v);
            auto it = dag.target_hubs.find(v);
            if (it != dag.target_hubs.end()) {
                for (int hi : it->second) {
                    if (hub_rem_children[hi] > 0) --hub_rem_children[hi];
                    for (const auto& e : dag.hubs[hi].edges) {
                        if (e.target == v) {
                            auto dec = [&](u64 c) {
                                auto fi = freq_rem.find(c);
                                if (fi != freq_rem.end() && fi->second > 0) --fi->second;
                            };
                            if (e.q >= 2) dec(e.q);
                            if (e.d != 0) dec(abs_i64(e.d));
                            if (e.r != 0) dec(abs_i64(e.r));
                            break;
                        }
                    }
                }
            }
        }
        auto hit = dag.hub_idx_of.find(v);
        if (hit != dag.hub_idx_of.end()) {
            int hi = hit->second;
            if (!hub_built[hi]) {
                hub_built[hi] = 1;
                int p = dag.hubs[hi].parent_link.parent_hub;
                if (p >= 0 && hub_rem_child_hubs[p] > 0) {
                    --hub_rem_child_hubs[p];
                }
            }
        }
    }

    bool apply(const std::vector<Step>& steps) {
        std::unordered_map<u64, size_t> last_need;
        for (size_t i = 0; i < steps.size(); ++i) {
            last_need[steps[i].a] = i;
            last_need[steps[i].b] = i;
        }
        bool evicted = false;
        for (size_t i = 0; i < steps.size(); ++i) {
            const Step& s = steps[i];
            const uint32_t* ia = pool_idx.find(s.a);
            const uint32_t* ib = pool_idx.find(s.b);
            if (!ia || !ib) return false;
            u64 r = eval(s.op, s.a, s.b);
            if (r == 0) return false;
            stamp[*ia] = ++tick;
            stamp[*ib] = tick;
            if (in_pool(r)) {
                on_value_produced(r);
                continue;
            }
            if (maxRegs > 0 && pool.size() >= static_cast<size_t>(maxRegs)) {
                size_t victim = pool.size();
                double victim_score = 0.0;
                for (size_t j = 0; j < pool.size(); ++j) {
                    u64 pv = pool[j];
                    auto need = last_need.find(pv);
                    if ((need != last_need.end() && need->second > i) || pins.count(pv)) {
                        continue;
                    }
                    double sc = evict_score(pv);
                    if (victim == pool.size() || sc < victim_score ||
                        (sc == victim_score && stamp[j] < stamp[victim])) {
                        victim = j;
                        victim_score = sc;
                    }
                }
                if (victim == pool.size()) {
                    for (size_t j = 0; j < pool.size(); ++j) {
                        u64 pv = pool[j];
                        auto need = last_need.find(pv);
                        if (need != last_need.end() && need->second > i) continue;
                        if (victim == pool.size() || stamp[j] < stamp[victim]) victim = j;
                    }
                    if (victim == pool.size()) return false;
                }
                pool.erase(pool.begin() + static_cast<std::ptrdiff_t>(victim));
                stamp.erase(stamp.begin() + static_cast<std::ptrdiff_t>(victim));
                reindex();
                evicted = true;
                ++evictions;
            }
            plan.push_back(Step{s.op, s.a, s.b, r});
            add_value(r, !evicted);
            on_value_produced(r);
        }
        if (evicted) rebuild();
        fail_b.clear();
        remaining.erase(
            std::remove_if(remaining.begin(), remaining.end(),
                           [&](u64 t) { return !remaining_set.count(t); }),
            remaining.end());
        return true;
    }

    void emit_l1(u64 x, std::vector<Step>& out) const {
        const uint32_t* idx = l1_idx.find(x);
        if (!idx) return;
        const L1E& e = l1[*idx];
        out.push_back(Step{e.op, e.a, e.b, e.v});
    }

    void emit_01(u64 x, int c, std::vector<Step>& out) const {
        if (c == 1) emit_l1(x, out);
    }

    void grab_l1_targets() {
        while (!aborted) {
            std::vector<Step> steps;
            for (u64 t : remaining) {
                if (c01(t) == 1) emit_l1(t, steps);
            }
            if (steps.empty()) break;
            size_t before_plan = plan.size();
            size_t before_rem = remaining.size();
            if (!apply(steps)) break;
            if (plan.size() == before_plan && remaining.size() == before_rem) break;
        }
    }

    // Grab any random targets that happen to be 1-step reachable from Phase-1's live pool.
    void harvest_extra_l1(const std::vector<u64>& extra_targets) {
        if (aborted || extra_targets.empty()) return;
        for (int pass = 0; pass < 3; ++pass) {
            std::vector<Step> steps;
            for (u64 t : extra_targets) {
                if (!coveredEver.contains(t) && c01(t) == 1) {
                    emit_l1(t, steps);
                }
            }
            if (steps.empty()) break;
            if (!apply(steps)) break;
        }
    }

    bool find2(u64 v, std::vector<Step>* out, std::vector<u64>* collect) {
        work += static_cast<double>(pool.size()) * 6.0;
        auto hit = [&](u64 x, char op, bool x_first, u64 c) -> bool {
            if (!l1_idx.contains(x)) return false;
            if (collect) {
                collect->push_back(x);
                return false;
            }
            emit_l1(x, *out);
            out->push_back(x_first ? Step{op, x, c, v} : Step{op, c, x, v});
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
                u128 p = static_cast<u128>(v) * static_cast<u128>(c);
                if (p <= limit && hit(static_cast<u64>(p), '/', true, c)) return true;
            }
        }
        if ((v & 1ULL) == 0 && hit(v / 2, '+', true, v / 2)) return true;
        u64 s = elf::isqrt64(v);
        if (s * s == v && hit(s, '*', true, s)) return true;
        return collect && !collect->empty();
    }

    bool find3(u64 v, std::vector<Step>& out) {
        if (check_time()) return false;
        work += static_cast<double>(l1.size()) * 6.0;
        auto both = [&](u64 x, u64 y, Step last) -> bool {
            if (!l1_idx.contains(y)) return false;
            emit_l1(x, out);
            if (y != x) emit_l1(y, out);
            out.push_back(last);
            return true;
        };
        for (const L1E& e : l1) {
            u64 x = e.v;
            if (v > x && both(x, v - x, Step{'+', x, v - x, v})) return true;
            if (x > v) {
                if (both(x, x - v, Step{'-', x, x - v, v})) return true;
                if (x % v == 0 && both(x, x / v, Step{'/', x, x / v, v})) return true;
            }
            if (v <= limit - x && both(x, v + x, Step{'-', v + x, x, v})) return true;
            if (x > 1) {
                if (v % x == 0 && both(x, v / x, Step{'*', x, v / x, v})) return true;
                u128 p = static_cast<u128>(v) * static_cast<u128>(x);
                if (p <= limit && both(x, static_cast<u64>(p), Step{'/', static_cast<u64>(p), x, v})) {
                    return true;
                }
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
            if (work > cfg.work_limit || check_time()) return false;
            if (v > c && chain(v - c, Step{'+', v - c, c, v})) return true;
            if (v <= limit - c && chain(v + c, Step{'-', v + c, c, v})) return true;
            if (c > v) {
                if (chain(c - v, Step{'-', c, c - v, v})) return true;
                if (c % v == 0 && chain(c / v, Step{'/', c, c / v, v})) return true;
            }
            if (c > 1) {
                if (v % c == 0 && chain(v / c, Step{'*', v / c, c, v})) return true;
                u128 p = static_cast<u128>(v) * static_cast<u128>(c);
                if (p <= limit && chain(static_cast<u64>(p), Step{'/', static_cast<u64>(p), c, v})) {
                    return true;
                }
            }
        }
        if ((v & 1ULL) == 0 && chain(v / 2, Step{'+', v / 2, v / 2, v})) return true;
        u64 s = elf::isqrt64(v);
        if (s * s == v && chain(s, Step{'*', s, s, v})) return true;
        return false;
    }

    void build_M() {
        if (!m_dirty) return;
        M.clear();
        M.reserve(pool.size() + l1.size());
        for (u64 p : pool) M.emplace_back(p, 0);
        for (const L1E& e : l1) {
            if (!in_pool(e.v)) M.emplace_back(e.v, 1);
        }
        std::sort(M.begin(), M.end());
        m_dirty = false;
    }

    bool try_split(int B, u64 a, int ca, u64 q, u64 r, bool minus, std::vector<Step>& out) {
        int base = ca + 2;
        int lq = c01(q);
        int lr = c01(r);
        if (base + lq + lr > B) return false;
        size_t mark = out.size();
        auto finish = [&]() -> bool {
            emit_01(a, ca, out);
            u64 prod = a * q;
            out.push_back(Step{'*', a, q, prod});
            out.push_back(Step{minus ? '-' : '+', prod, r, minus ? (prod - r) : (prod + r)});
            return true;
        };
        int cap_r = std::min(2, B - base - lq);
        if (solve(r, cap_r, out)) {
            int cr = static_cast<int>(out.size() - mark);
            if (solve(q, B - base - cr, out)) return finish();
            out.resize(mark);
            return false;
        }
        out.resize(mark);
        int cap_q = std::min(2, B - base - std::max(lr, cap_r + 1));
        if (cap_q < lq) return false;
        if (solve(q, cap_q, out)) {
            int cq = static_cast<int>(out.size() - mark);
            if (solve(r, B - base - cq, out)) return finish();
        }
        out.resize(mark);
        return false;
    }

    bool decomp(u64 v, int B, std::vector<Step>& out) {
        if (check_time() || work > cfg.work_limit) return false;
        build_M();
        work += static_cast<double>(M.size()) * 16.0;
        std::vector<Cand> cands;
        cands.reserve(256);

        for (const auto& [a, ca] : M) {
            if (a < 2) continue;
            if (a >= v) break;
            auto it = std::lower_bound(M.begin(), M.end(), std::make_pair(v / a, uint8_t{0}));
            for (int d = -1; d <= 1; ++d) {
                auto jt = it + d;
                if (jt < M.begin() || jt >= M.end() || jt->first < 2) continue;
                u64 q = jt->first;
                int cost0 = ca + jt->second + 1;
                u128 p = static_cast<u128>(a) * static_cast<u128>(q);
                if (cost0 >= B || p > limit || static_cast<u64>(p) == v) continue;
                u64 prod = static_cast<u64>(p);
                u64 r = prod > v ? prod - v : v - prod;
                int rem = B - cost0 - 1;
                if (rem < c01(r)) continue;
                cands.push_back({r, a, q, ca, jt->second, static_cast<uint8_t>(prod > v ? 1 : 0), rem});
            }
        }
        auto it = std::lower_bound(M.begin(), M.end(), std::make_pair(v, uint8_t{0}));
        for (int d = -8; d < 8; ++d) {
            auto jt = it + d;
            if (jt < M.begin() || jt >= M.end() || jt->first == v) continue;
            u64 x = jt->first;
            u64 r = x > v ? x - v : v - x;
            int rem = B - jt->second - 1;
            if (rem < c01(r)) continue;
            cands.push_back({r, x, 0, jt->second, 0, static_cast<uint8_t>(x > v ? 3 : 2), rem});
        }
        std::sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) {
            return x.r != y.r ? x.r < y.r : x.rem > y.rem;
        });

        int tried = 0;
        u64 last_r = 0;
        for (const Cand& c : cands) {
            if (c.r == last_r) continue;
            if (++tried > 20 || work > cfg.work_limit || check_time()) break;
            last_r = c.r;
            size_t mark = out.size();
            if (!solve(c.r, c.rem, out)) {
                out.resize(mark);
                continue;
            }
            emit_01(c.a, c.ca, out);
            if (c.kind < 2) {
                emit_01(c.q, c.cq, out);
                u64 prod = c.a * c.q;
                out.push_back(Step{'*', c.a, c.q, prod});
                out.push_back(Step{c.kind ? '-' : '+', prod, c.r, v});
            } else {
                out.push_back(Step{c.kind == 3 ? '-' : '+', c.a, c.r, v});
            }
            return true;
        }

        // Exact division check: v = (v * d) / d
        for (u64 d : pool) {
            if (d < 2) continue;
            u128 vd = static_cast<u128>(v) * static_cast<u128>(d);
            if (vd > limit) continue;
            if (work > cfg.work_limit || check_time()) return false;
            size_t mark = out.size();
            if (solve(static_cast<u64>(vd), B - 1, out)) {
                out.push_back(Step{'/', static_cast<u64>(vd), d, v});
                return true;
            }
            out.resize(mark);
        }

        for (u64 a : pool) {
            if (a < 2 || a >= v) continue;
            if (work > cfg.work_limit || check_time()) return false;
            u64 q = v / a, r = v % a;
            if (r == 0) {
                size_t mark = out.size();
                if (solve(q, B - 1, out)) {
                    out.push_back(Step{'*', q, a, v});
                    return true;
                }
                out.resize(mark);
                continue;
            }
            if (try_split(B, a, 0, q, r, false, out)) return true;
            if (static_cast<u128>(a) * (q + 1) <= limit &&
                try_split(B, a, 0, q + 1, a - r, true, out)) {
                return true;
            }
        }
        for (const L1E& e : l1) {
            u64 a = e.v;
            if (a < 2 || a >= v || in_pool(a)) continue;
            if (work > cfg.work_limit || check_time()) return false;
            u64 q = v / a, r = v % a;
            if (r == 0) {
                if (c01(q) <= 1 && B >= 2 + c01(q)) {
                    emit_l1(a, out);
                    emit_01(q, c01(q), out);
                    out.push_back(Step{'*', a, q, v});
                    return true;
                }
                continue;
            }
            if (std::min(c01(q), c01(r)) <= 1 && try_split(B, a, 1, q, r, false, out)) return true;
            if (static_cast<u128>(a) * (q + 1) <= limit &&
                std::min(c01(q + 1), c01(a - r)) <= 1 &&
                try_split(B, a, 1, q + 1, a - r, true, out)) {
                return true;
            }
        }
        return false;
    }

    bool solve(u64 v, int B, std::vector<Step>& out) {
        int c = c01(v);
        if (c == 0) return true;
        if (B <= 0) return false;
        if (c == 1) {
            emit_l1(v, out);
            return true;
        }
        if (B == 1) return false;
        const int* fb = fail_b.find(v);
        int from = fb ? (*fb + 1) : 2;
        size_t mark = out.size();
        for (int b = from; b <= B; ++b) {
            if (work > cfg.work_limit || check_time()) return false;
            bool ok = (b == 2) ? find2(v, &out, nullptr)
                               : (b == 3 ? find3(v, out) : decomp(v, b, out));
            if (ok) return true;
            out.resize(mark);
            if (work > cfg.work_limit || check_time()) return false;
            fail_b.insert_or_assign(v, b);
        }
        return false;
    }

    double est_cost(u64 x) const {
        if (x == 0) return 0.0;
        int c = c01(x);
        if (c <= 1) return static_cast<double>(c);
        return rough_build_cost(x);
    }

    double est_factored_cost(u64 x, const std::vector<std::pair<u64, int>>& fac) const {
        if (x <= 1) return est_cost(x);
        int c = c01(x);
        if (c <= 1) return static_cast<double>(c);
        bool all_even = !fac.empty();
        for (const auto& [p, e] : fac) {
            if (e & 1) {
                all_even = false;
                break;
            }
        }
        if (all_even) {
            u64 s = elf::isqrt64(x);
            std::vector<std::pair<u64, int>> half_fac;
            half_fac.reserve(fac.size());
            for (const auto& [p, e] : fac) half_fac.emplace_back(p, e / 2);
            return std::min(est_cost(s), est_factored_cost(s, half_fac)) + 1.0;
        }
        double total = 0.0;
        int pieces = 0;
        for (const auto& [p, e] : fac) {
            double pc = est_cost(p);
            int extra_muls = 0;
            if (e > 1) {
                int tmp = e, bits = 0, ones = 0;
                while (tmp > 0) {
                    ++bits;
                    ones += (tmp & 1);
                    tmp >>= 1;
                }
                extra_muls = (bits - 1) + (ones - 1);
            }
            total += pc + static_cast<double>(extra_muls);
            ++pieces;
        }
        if (pieces > 1) total += static_cast<double>(pieces - 1);
        return std::min(est_cost(x), total);
    }

    double est_divisor_cost(u64 d, const std::vector<std::pair<u64, int>>& parent_fac) const {
        int c = c01(d);
        if (c <= 1) return static_cast<double>(c);
        if (d <= 256) return est_cost(d);
        std::vector<std::pair<u64, int>> sub;
        u64 tmp = d;
        for (const auto& [p, e] : parent_fac) {
            if (tmp % p == 0) {
                int cnt = 0;
                while (tmp % p == 0) {
                    tmp /= p;
                    ++cnt;
                }
                sub.emplace_back(p, cnt);
            }
            if (tmp == 1) break;
        }
        return est_factored_cost(d, sub);
    }

    enum class PlanKind {
        NONE,
        DIRECT_ADDSUB,
        EXACT_DIV,
        BASE_SPLIT,
        SHIFTED_AFFINE,
        FACTOR_SPLIT,
        NEAR_SQUARE,
        SMOOTH_SHIFT
    };

    struct ConstructChoice {
        PlanKind kind = PlanKind::NONE;
        double score = 1e18;
        u64 a = 0;
        u64 b = 0;
        u64 q = 0;
        i64 d = 0;
        i64 r = 0;
        bool minus = false;
    };

    bool monotonic_construct(u64 v) {
        if (v == 0 || v > limit) return false;
        if (in_pool(v)) return true;
        if (c01(v) == 1) {
            std::vector<Step> st;
            emit_l1(v, st);
            if (apply(st) && in_pool(v)) return true;
        }
        if (!in_pool(1)) {
            if (pool.empty() || !apply({Step{'/', pool[0], pool[0], 1}})) return false;
        }
        if (v == 1) return in_pool(1);

        PinGuard pin1(*this, 1, true);
        int top = 63 - __builtin_clzll(v);
        int start_bit = top - 1;
        u64 acc = 1;
        for (int sh = 1; sh < top; ++sh) {
            u64 pref = v >> sh;
            if (pref > 1 && in_pool(pref)) {
                acc = pref;
                start_bit = sh - 1;
                break;
            }
        }
        bool acc_pinned = (acc != 1);
        if (acc_pinned) pin(acc);
        for (int b = start_bit; b >= 0; --b) {
            if (plan.size() > kMaxPlanSteps) {
                if (acc_pinned) unpin(acc);
                aborted = true;
                return false;
            }
            u64 dbl = acc + acc;
            if (!apply({Step{'+', acc, acc, dbl}})) {
                if (acc_pinned) unpin(acc);
                return false;
            }
            if (acc_pinned) unpin(acc);
            acc = dbl;
            acc_pinned = (acc != 1);
            if (acc_pinned) pin(acc);

            if ((v >> b) & 1ULL) {
                if (!in_pool(1)) {
                    if (!apply({Step{'/', acc, acc, 1}})) {
                        if (acc_pinned) unpin(acc);
                        return false;
                    }
                }
                u64 nxt = acc + 1;
                if (!apply({Step{'+', acc, 1, nxt}})) {
                    if (acc_pinned) unpin(acc);
                    return false;
                }
                if (acc_pinned) unpin(acc);
                acc = nxt;
                acc_pinned = (acc != 1);
                if (acc_pinned) pin(acc);
            }
        }
        if (acc_pinned) unpin(acc);
        return in_pool(v);
    }

    void fast_finish_remaining() {
        while (!remaining.empty() && !aborted) {
            if (timer.elapsed_sec() >= fatal_deadline || plan.size() > kMaxPlanSteps) {
                aborted = true;
                return;
            }
            grab_l1_targets();
            if (remaining.empty()) break;
            u64 t = *std::min_element(remaining.begin(), remaining.end());
            if (!monotonic_construct(t)) {
                aborted = true;
                return;
            }
        }
    }

    void construct(u64 v, int depth = 0) {
        if (in_pool(v)) return;
        if (v == 0 || v > limit) {
            aborted = true;
            return;
        }
        if (check_time()) {
            if (aborted) return;
            if (fast_fallback_mode) {
                if (!monotonic_construct(v)) aborted = true;
                return;
            }
        }
        if (depth >= kMaxConstructDepth || on_stack(v)) {
            if (!monotonic_construct(v)) aborted = true;
            return;
        }
        StackGuard sg(construct_stack, v);

        int guard = 0;
        while (!in_pool(v)) {
            if (aborted || plan.size() > kMaxPlanSteps) {
                aborted = true;
                return;
            }
            if (check_time()) {
                if (aborted) return;
                if (fast_fallback_mode) {
                    if (!monotonic_construct(v)) aborted = true;
                    return;
                }
            }
            if (++guard > 4) {
                if (!monotonic_construct(v)) aborted = true;
                return;
            }
            if (c01(v) == 1) {
                std::vector<Step> st;
                emit_l1(v, st);
                if (apply(st) && in_pool(v)) return;
                if (!monotonic_construct(v)) aborted = true;
                return;
            }
            if (v <= 3) {
                if (!monotonic_construct(v)) aborted = true;
                return;
            }

            int b_lim = (depth == 0) ? cfg.b_construct : (depth == 1 ? 4 : 3);
            if (timed_out) b_lim = 2;
            std::vector<Step> steps;
            bool ok = false;
            for (int B = 2; B <= b_lim && !ok; ++B) {
                if (B >= 3 && check_time()) break;
                steps.clear();
                work = 0.0;
                ok = solve(v, B, steps);
            }
            if (aborted) return;
            if (ok) {
                if (apply(steps) && in_pool(v)) return;
                continue;
            }

            build_M();
            bool has_small = false;
            for (const auto& [a, ca] : M) {
                if (a >= 2 && a < v) {
                    has_small = true;
                    break;
                }
            }
            if (!has_small) {
                if (!in_pool(1)) {
                    apply({Step{'/', pool[0], pool[0], 1}});
                    continue;
                }
                if (!in_pool(2) && 2 <= limit) {
                    apply({Step{'+', 1, 1, 2}});
                    continue;
                }
            }

            bool low_pin_mode = (pins.size() + 8 >= static_cast<size_t>(maxRegs)) || depth >= 10;
            ConstructChoice best;

            // Strategy 0: Direct add/sub from nearby M and pool values + exact division
            {
                auto try_addsub = [&](u64 x, int cx) {
                    if (x == v || x == 0 || on_stack(x)) return;
                    u64 r = x > v ? (x - v) : (v - x);
                    if (r == 0 || r > v / 2 || on_stack(r)) return;
                    double sc = cx + est_cost(r) + 1.0;
                    if (sc < best.score) {
                        best = {PlanKind::DIRECT_ADDSUB, sc, x, 0, 0, 0,
                                static_cast<i64>(r), x > v};
                    }
                };
                auto it = std::lower_bound(M.begin(), M.end(), std::make_pair(v, uint8_t{0}));
                for (int off = -4; off <= 4; ++off) {
                    auto jt = it + off;
                    if (jt < M.begin() || jt >= M.end()) continue;
                    try_addsub(jt->first, jt->second);
                }
                for (u64 p : pool) {
                    u64 diff = p > v ? (p - v) : (v - p);
                    if (diff > 0 && diff <= v / 16) try_addsub(p, 0);
                }
                for (u64 p : pool) {
                    if (p > v && p % v == 0) {
                        u64 dv = p / v;
                        if (dv < v && !on_stack(dv)) {
                            double sc = est_cost(dv) + 1.0;
                            if (sc < best.score) {
                                best = {PlanKind::EXACT_DIV, sc, p, dv, 0, 0, 0, false};
                            }
                        }
                    }
                }
            }

            // Strategy 5 (baseline): Base split v = a*q +- r with a in M
            {
                u64 root = elf::isqrt64(v);
                auto eval_a = [&](u64 a, int ca) {
                    if (a < 2 || a >= v || on_stack(a)) return;
                    u64 q = v / a, r = v % a;
                    if (q < 2 || q >= v) return;
                    if (!on_stack(q) && (r == 0 || !on_stack(r))) {
                        double sc_plus = ca + est_cost(q) + 1.0 + (r ? (est_cost(r) + 1.0) : 0.0);
                        if (sc_plus < best.score) {
                            best = {PlanKind::BASE_SPLIT, sc_plus, a, 0, q, 0,
                                    static_cast<i64>(r), false};
                        }
                    }
                    if (r > 0 && q + 1 < v && static_cast<u128>(a) * (q + 1) <= limit) {
                        u64 rm = a - r;
                        if (!on_stack(q + 1) && !on_stack(rm)) {
                            double sc_minus = ca + est_cost(q + 1) + 1.0 + est_cost(rm) + 1.0;
                            if (sc_minus < best.score) {
                                best = {PlanKind::BASE_SPLIT, sc_minus, a, 0, q + 1, 0,
                                        static_cast<i64>(rm), true};
                            }
                        }
                    }
                };
                auto it = std::upper_bound(M.begin(), M.end(), std::make_pair(root, uint8_t{255}));
                int seen = 0;
                auto jt = it;
                while (jt != M.begin() && seen < 24) {
                    --jt;
                    if (jt->first < 2) break;
                    eval_a(jt->first, jt->second);
                    ++seen;
                }
                jt = it;
                seen = 0;
                while (jt != M.end() && seen < 12 && jt->first < v) {
                    eval_a(jt->first, jt->second);
                    ++jt;
                    ++seen;
                }
                for (u64 p : pool) {
                    if (p >= 2 && p < v) eval_a(p, 0);
                }
            }

            if (!low_pin_mode) {
                // Strategy 1: Shifted-affine from pool values p < v
                for (u64 p : pool) {
                    if (p < 100 || p >= v) continue;
                    u64 q0 = (v + p / 2) / p;
                    auto try_sa_q = [&](u64 q) {
                        if (q < 2 || q > 10000000ULL || q >= v || on_stack(q)) return;
                        i128 diff = static_cast<i128>(v) - static_cast<i128>(q) * static_cast<i128>(p);
                        i64 d = round_div_i128(diff, q);
                        i128 rem = diff - static_cast<i128>(q) * static_cast<i128>(d);
                        u64 ad = abs_i64(d);
                        if (ad > 0 && ad < p / 4 && (d > 0 || ad < p)) {
                            u64 p_sh = d >= 0 ? (p + ad) : (p - ad);
                            if (p_sh > 0 && p_sh < v && p_sh <= limit &&
                                static_cast<u128>(q) * static_cast<u128>(p_sh) <= limit &&
                                rem >= -static_cast<i128>(v / 2) && rem <= static_cast<i128>(v / 2)) {
                                i64 r = static_cast<i64>(rem);
                                u64 ar = abs_i64(r);
                                if (ar < v && ad < v && !on_stack(ad) && (ar == 0 || !on_stack(ar))) {
                                    double sc = (in_pool(p_sh) ? 0.0 : (est_cost(ad) + 1.0)) +
                                                est_cost(q) + 1.0 +
                                                (ar > 0 ? (est_cost(ar) + 1.0) : 0.0);
                                    if (sc < best.score) {
                                        best = {PlanKind::SHIFTED_AFFINE, sc, p, p_sh, q, d, r, false};
                                    }
                                }
                            }
                        }
                    };
                    try_sa_q(q0);
                }

                // Strategy 2: Exact Factorization & Square Split
                if (v >= 16 && !timed_out) {
                    auto fac = elf::factorize(v);
                    u64 s = elf::isqrt64(v);
                    if (s * s == v && s > 1 && s < v && !on_stack(s)) {
                        std::vector<std::pair<u64, int>> half_fac;
                        for (const auto& [p, e] : fac) half_fac.emplace_back(p, e / 2);
                        double sc = est_factored_cost(s, half_fac) + 1.0;
                        if (sc < best.score) {
                            best = {PlanKind::FACTOR_SPLIT, sc, s, s, 0, 0, 0, false};
                        }
                    }
                    if (!(fac.size() == 1 && fac[0].second == 1)) {
                        auto divs = elf::get_divisors(fac, 256);
                        for (u64 f1 : divs) {
                            if (f1 <= 1 || static_cast<u128>(f1) * f1 > v) continue;
                            u64 f2 = v / f1;
                            if (f2 <= 1 || f2 >= v || on_stack(f1) || on_stack(f2)) continue;
                            double c1 = est_divisor_cost(f1, fac);
                            double c2 = est_divisor_cost(f2, fac);
                            double sc = c1 + c2 + 1.0;
                            if (sc < best.score) {
                                best = {PlanKind::FACTOR_SPLIT, sc, f1, f2, 0, 0, 0, false};
                            }
                        }
                        for (u64 p : pool) {
                            if (p > 1 && p < v && v % p == 0) {
                                u64 f2 = v / p;
                                if (f2 > 1 && f2 < v && !on_stack(f2)) {
                                    double sc = est_divisor_cost(f2, fac) + 1.0;
                                    if (sc < best.score) {
                                        best = {PlanKind::FACTOR_SPLIT, sc, p, f2, 0, 0, 0, false};
                                    }
                                }
                            }
                        }
                    }
                }

                // Strategy 3: Near-Square Shortcut v = s^2 +- r
                if (v >= 10000) {
                    u64 s0 = elf::isqrt64(v);
                    auto check_sq = [&](u64 s, bool minus) {
                        if (s <= 1 || s >= v || on_stack(s)) return;
                        u128 sq = static_cast<u128>(s) * static_cast<u128>(s);
                        if (sq > limit) return;
                        u64 r = minus ? (static_cast<u64>(sq) - v) : (v - static_cast<u64>(sq));
                        if (r == 0 || r >= v / 100 || on_stack(r)) return;
                        double cs = est_cost(s);
                        if (s > 1000 && !timed_out) {
                            auto sfac = elf::factorize(s);
                            cs = est_factored_cost(s, sfac);
                        }
                        double sc = cs + 1.0 + est_cost(r) + 1.0;
                        if (sc < best.score) {
                            best = {PlanKind::NEAR_SQUARE, sc, s, 0, 0, 0,
                                    static_cast<i64>(r), minus};
                        }
                    };
                    check_sq(s0, false);
                    check_sq(s0 + 1, true);
                }

                // Strategy 4: Nearby Smooth Shift v = (v +- d) -+ d
                if (cfg.allow_smooth_shift && depth == 0 && v >= 100000 && !timed_out) {
                    auto check_smooth = [&](u64 d_val, bool add_then_sub) {
                        if (d_val == 0 || d_val >= v / 100 || on_stack(d_val)) return;
                        if (add_then_sub && v > limit - d_val) return;
                        u64 u = add_then_sub ? (v + d_val) : (v - d_val);
                        if (u < 16 || on_stack(u)) return;
                        u64 tmp = u;
                        int distinct = 0;
                        int extra_muls = 0;
                        double pcost = 0.0;
                        for (u64 p : kSmoothPrimes) {
                            if (p * p > tmp) break;
                            if (tmp % p == 0) {
                                int e = 0;
                                while (tmp % p == 0) {
                                    tmp /= p;
                                    ++e;
                                }
                                ++distinct;
                                pcost += est_cost(p);
                                if (e > 1) {
                                    int te = e, bits = 0, ones = 0;
                                    while (te > 0) {
                                        ++bits;
                                        ones += (te & 1);
                                        te >>= 1;
                                    }
                                    extra_muls += (bits - 1) + (ones - 1);
                                }
                            }
                        }
                        if (tmp > 1) {
                            if (tmp > 5000) return;
                            ++distinct;
                            pcost += est_cost(tmp);
                        }
                        if (distinct < 2) return;
                        double sc = pcost + extra_muls + (distinct - 1) + est_cost(d_val) + 1.0;
                        if (sc + 1.2 < best.score) {
                            best = {PlanKind::SMOOTH_SHIFT, sc, u, d_val, 0, 0, 0, add_then_sub};
                        }
                    };
                    for (u64 d = 1; d <= 48; ++d) {
                        check_smooth(d, false);
                        check_smooth(d, true);
                    }
                    for (u64 p : pool) {
                        if (p > 48 && p <= 5000) {
                            check_smooth(p, false);
                            check_smooth(p, true);
                        }
                    }
                }
            }

            if (best.kind == PlanKind::NONE) {
                if (!monotonic_construct(v)) aborted = true;
                return;
            }

            switch (best.kind) {
                case PlanKind::DIRECT_ADDSUB: {
                    u64 x = best.a;
                    u64 r = static_cast<u64>(best.r);
                    if (!in_pool(x)) {
                        if (c01(x) == 1) {
                            std::vector<Step> st;
                            emit_l1(x, st);
                            apply(st);
                        }
                    }
                    if (aborted || !in_pool(x)) break;
                    PinGuard px(*this, x);
                    construct(r, depth + 1);
                    if (aborted || !in_pool(x) || !in_pool(r)) break;
                    apply({Step{best.minus ? '-' : '+', x, r, v}});
                    break;
                }
                case PlanKind::EXACT_DIV: {
                    u64 p = best.a;
                    u64 dv = best.b;
                    if (!in_pool(p)) break;
                    PinGuard pp(*this, p);
                    construct(dv, depth + 1);
                    if (aborted || !in_pool(p) || !in_pool(dv)) break;
                    apply({Step{'/', p, dv, v}});
                    break;
                }
                case PlanKind::BASE_SPLIT: {
                    u64 a = best.a;
                    u64 q = best.q;
                    u64 r = static_cast<u64>(best.r);
                    if (c01(a) == 1 && q > a) {
                        std::vector<Step> st;
                        emit_l1(a, st);
                        apply(st);
                    }
                    if (aborted) break;
                    u64 prod = a * q;
                    if (!in_pool(prod)) {
                        if (in_pool(a)) {
                            PinGuard pa(*this, a);
                            construct(q, depth + 1);
                            if (aborted || !in_pool(a) || !in_pool(q)) break;
                            apply({Step{'*', a, q, prod}});
                        } else if (est_cost(q) >= est_cost(a)) {
                            construct(q, depth + 1);
                            if (aborted || !in_pool(q)) break;
                            PinGuard pq(*this, q);
                            if (!in_pool(a) && c01(a) == 1) {
                                std::vector<Step> st;
                                emit_l1(a, st);
                                apply(st);
                            } else {
                                construct(a, depth + 1);
                            }
                            if (aborted || !in_pool(a) || !in_pool(q)) break;
                            apply({Step{'*', a, q, prod}});
                        } else {
                            if (c01(a) == 1) {
                                std::vector<Step> st;
                                emit_l1(a, st);
                                apply(st);
                            } else {
                                construct(a, depth + 1);
                            }
                            if (aborted || !in_pool(a)) break;
                            PinGuard pa(*this, a);
                            construct(q, depth + 1);
                            if (aborted || !in_pool(a) || !in_pool(q)) break;
                            apply({Step{'*', a, q, prod}});
                        }
                    }
                    if (aborted || !in_pool(prod)) break;
                    if (r > 0) {
                        PinGuard pprod(*this, prod);
                        construct(r, depth + 1);
                        if (aborted || !in_pool(prod) || !in_pool(r)) break;
                        apply({Step{best.minus ? '-' : '+', prod, r, v}});
                    }
                    break;
                }
                case PlanKind::SHIFTED_AFFINE: {
                    u64 p = best.a;
                    u64 p_sh = best.b;
                    u64 q = best.q;
                    u64 ad = abs_i64(best.d);
                    u64 ar = abs_i64(best.r);
                    if (!in_pool(p_sh)) {
                        if (!in_pool(p)) break;
                        PinGuard pp(*this, p);
                        construct(ad, depth + 1);
                        if (aborted || !in_pool(p) || !in_pool(ad)) break;
                        apply({Step{best.d >= 0 ? '+' : '-', p, ad, p_sh}});
                    }
                    if (aborted || !in_pool(p_sh)) break;
                    u64 prod = p_sh * q;
                    if (!in_pool(prod)) {
                        PinGuard psh(*this, p_sh);
                        construct(q, depth + 1);
                        if (aborted || !in_pool(p_sh) || !in_pool(q)) break;
                        apply({Step{'*', p_sh, q, prod}});
                    }
                    if (aborted || !in_pool(prod)) break;
                    if (ar > 0) {
                        PinGuard pprod(*this, prod);
                        construct(ar, depth + 1);
                        if (aborted || !in_pool(prod) || !in_pool(ar)) break;
                        apply({Step{best.r >= 0 ? '+' : '-', prod, ar, v}});
                    }
                    break;
                }
                case PlanKind::FACTOR_SPLIT: {
                    u64 f1 = best.a;
                    u64 f2 = best.b;
                    if (f1 == f2) {
                        construct(f1, depth + 1);
                        if (aborted || !in_pool(f1)) break;
                        apply({Step{'*', f1, f1, v}});
                    } else if (in_pool(f1)) {
                        PinGuard pf1(*this, f1);
                        construct(f2, depth + 1);
                        if (aborted || !in_pool(f1) || !in_pool(f2)) break;
                        apply({Step{'*', f1, f2, v}});
                    } else if (in_pool(f2)) {
                        PinGuard pf2(*this, f2);
                        construct(f1, depth + 1);
                        if (aborted || !in_pool(f1) || !in_pool(f2)) break;
                        apply({Step{'*', f1, f2, v}});
                    } else if (est_cost(f2) >= est_cost(f1)) {
                        construct(f2, depth + 1);
                        if (aborted || !in_pool(f2)) break;
                        PinGuard pf2(*this, f2);
                        construct(f1, depth + 1);
                        if (aborted || !in_pool(f1) || !in_pool(f2)) break;
                        apply({Step{'*', f1, f2, v}});
                    } else {
                        construct(f1, depth + 1);
                        if (aborted || !in_pool(f1)) break;
                        PinGuard pf1(*this, f1);
                        construct(f2, depth + 1);
                        if (aborted || !in_pool(f1) || !in_pool(f2)) break;
                        apply({Step{'*', f1, f2, v}});
                    }
                    break;
                }
                case PlanKind::NEAR_SQUARE: {
                    u64 s = best.a;
                    u64 r = static_cast<u64>(best.r);
                    u64 sq = s * s;
                    if (!in_pool(sq)) {
                        construct(s, depth + 1);
                        if (aborted || !in_pool(s)) break;
                        apply({Step{'*', s, s, sq}});
                    }
                    if (aborted || !in_pool(sq)) break;
                    if (r > 0) {
                        PinGuard psq(*this, sq);
                        construct(r, depth + 1);
                        if (aborted || !in_pool(sq) || !in_pool(r)) break;
                        apply({Step{best.minus ? '-' : '+', sq, r, v}});
                    }
                    break;
                }
                case PlanKind::SMOOTH_SHIFT: {
                    u64 u = best.a;
                    u64 d_val = best.b;
                    construct(u, depth + 1);
                    if (aborted || !in_pool(u)) break;
                    PinGuard pu(*this, u);
                    construct(d_val, depth + 1);
                    if (aborted || !in_pool(u) || !in_pool(d_val)) break;
                    apply({Step{best.minus ? '-' : '+', u, d_val, v}});
                    break;
                }
                default:
                    break;
            }
        }
    }

    void synthesize_from_hub(u64 base_hub, u64 target, u64 q, i64 d, i64 r) {
        if (in_pool(target) || aborted) return;
        if (check_time()) {
            if (aborted) return;
            if (fast_fallback_mode) {
                if (!monotonic_construct(target)) aborted = true;
                return;
            }
        }
        if (c01(target) == 1) {
            std::vector<Step> st;
            emit_l1(target, st);
            apply(st);
            return;
        }

        auto edge_pool_cost = [&](u64 b, u64 qq, i64 dd, i64 rr) -> double {
            double c = 0.0;
            if (dd != 0) {
                u64 ad = abs_i64(dd);
                u64 b_sh = (dd >= 0) ? (b + ad) : (b - ad);
                if (!in_pool(b_sh)) c += est_cost(ad) + 1.0;
            }
            if (qq > 1) c += est_cost(qq) + 1.0;
            if (rr != 0) c += est_cost(abs_i64(rr)) + 1.0;
            return c;
        };

        u64 best_base = base_hub;
        u64 best_q = q;
        i64 best_d = d;
        i64 best_r = r;
        double best_c = in_pool(base_hub) ? edge_pool_cost(base_hub, q, d, r) : 1e18;

        for (u64 p : pool) {
            if (p < 100 || p == target) continue;
            u64 diff_abs = (target > p) ? (target - p) : (p - target);
            if (diff_abs < target / 4) {
                double c = est_cost(diff_abs) + 1.0;
                if (c + 0.3 < best_c) {
                    best_c = c;
                    best_base = p;
                    best_q = 1;
                    best_d = 0;
                    best_r = (target > p) ? static_cast<i64>(diff_abs)
                                          : -static_cast<i64>(diff_abs);
                }
            }
            if (p < target) {
                u64 qq = 0;
                i64 dd = 0, rr = 0;
                double dummy_ec = 0.0;
                if (match_shifted_affine(target, p, limit, 500000ULL, 1.5, false, 0, 0,
                                         qq, dd, rr, dummy_ec)) {
                    double c = edge_pool_cost(p, qq, dd, rr);
                    if (c + 0.3 < best_c) {
                        best_c = c;
                        best_base = p;
                        best_q = qq;
                        best_d = dd;
                        best_r = rr;
                    }
                }
            }
        }

        {
            std::vector<Step> st;
            work = 0.0;
            int b_try = (!timed_out && best_c > 3.5) ? 3 : 2;
            if (solve(target, b_try, st)) {
                if (static_cast<double>(st.size()) <= best_c + 0.5) {
                    apply(st);
                    return;
                }
            }
            if (aborted) return;
        }

        if (!in_pool(best_base)) {
            construct(target);
            return;
        }

        u64 ad = abs_i64(best_d);
        if (best_d < 0 && ad >= best_base) {
            construct(target);
            return;
        }
        u64 h_sh = (best_d >= 0) ? (best_base + ad) : (best_base - ad);
        if (h_sh == 0 || h_sh > limit) {
            construct(target);
            return;
        }
        if (static_cast<u128>(best_q) * static_cast<u128>(h_sh) > limit) {
            construct(target);
            return;
        }
        u64 prod = best_q * h_sh;
        u64 ar = abs_i64(best_r);
        if ((best_r > 0 && prod + ar != target) ||
            (best_r < 0 && (prod <= ar || prod - ar != target)) ||
            (best_r == 0 && prod != target)) {
            construct(target);
            return;
        }

        PinGuard pbase(*this, best_base);
        if (best_d != 0 && !in_pool(h_sh)) {
            construct(ad, 1);
            if (aborted) return;
            if (!in_pool(best_base) || !in_pool(ad)) {
                pbase.release();
                construct(target);
                return;
            }
            apply({Step{best_d >= 0 ? '+' : '-', best_base, ad, h_sh}});
            grab_l1_targets();
        }
        if (aborted || in_pool(target)) return;

        if (best_q > 1 && !in_pool(prod)) {
            PinGuard psh(*this, h_sh, h_sh != best_base);
            construct(best_q, 1);
            if (aborted) return;
            if (!in_pool(h_sh) || !in_pool(best_q)) {
                psh.release();
                pbase.release();
                construct(target);
                return;
            }
            apply({Step{'*', h_sh, best_q, prod}});
            psh.release();
            grab_l1_targets();
        }
        if (aborted || in_pool(target)) return;

        if (best_r != 0) {
            PinGuard pprod(*this, prod, prod != best_base);
            construct(ar, 1);
            if (aborted) return;
            if (!in_pool(prod) || !in_pool(ar)) {
                pprod.release();
                pbase.release();
                construct(target);
                return;
            }
            apply({Step{best_r >= 0 ? '+' : '-', prod, ar, target}});
            pprod.release();
        }
        pbase.release();
        grab_l1_targets();
    }

    void sweep_hub(int hi) {
        if (aborted || fast_fallback_mode) return;
        u64 H = dag.hubs[hi].H;
        if (!in_pool(H)) return;
        PinGuard pH(*this, H);

        std::vector<const HubEdge*> todo;
        for (const auto& e : dag.hubs[hi].edges) {
            if (remaining_set.count(e.target)) {
                todo.push_back(&e);
            }
        }
        std::sort(todo.begin(), todo.end(), [](const HubEdge* a, const HubEdge* b) {
            bool a0 = (a->d == 0);
            bool b0 = (b->d == 0);
            if (a0 != b0) return a0 > b0;
            if (a->d != b->d) return a->d < b->d;
            return a->est_cost < b->est_cost;
        });

        for (const HubEdge* ep : todo) {
            if (check_time() && (aborted || fast_fallback_mode)) return;
            if (!remaining_set.count(ep->target)) continue;
            if (!in_pool(H)) break;
            synthesize_from_hub(H, ep->target, ep->q, ep->d, ep->r);
            if (aborted || fast_fallback_mode) return;
            auto it = dag.hub_idx_of.find(ep->target);
            if (it != dag.hub_idx_of.end() && it->second != hi) {
                int sub = it->second;
                if ((hub_rem_children[sub] > 0 || hub_rem_child_hubs[sub] > 0) &&
                    in_pool(ep->target)) {
                    sweep_hub(sub);
                }
            }
        }

        for (int ch : dag.hubs[hi].child_hubs) {
            if (check_time() && (aborted || fast_fallback_mode)) return;
            if (hub_rem_children[ch] == 0 && hub_rem_child_hubs[ch] == 0) continue;
            if (!in_pool(H)) break;
            u64 H2 = dag.hubs[ch].H;
            if (!in_pool(H2)) {
                const HubLink& lk = dag.hubs[ch].parent_link;
                synthesize_from_hub(H, H2, lk.q, lk.d, lk.r);
            }
            if (aborted || fast_fallback_mode) return;
            if (in_pool(H2)) {
                sweep_hub(ch);
            }
        }
    }

    void refresh_hub_counts() {
        const size_t nh = dag.hubs.size();
        for (size_t k = nh; k-- > 0;) {
            int cnt = 0;
            for (const auto& e : dag.hubs[k].edges) {
                if (remaining_set.count(e.target)) ++cnt;
            }
            hub_rem_children[k] = cnt;
            int ch_cnt = 0;
            for (int ch : dag.hubs[k].child_hubs) {
                if (ch <= static_cast<int>(k) || ch >= static_cast<int>(nh)) continue;
                if (in_pool(dag.hubs[ch].H)) continue;
                if (hub_rem_children[ch] > 0 || hub_rem_child_hubs[ch] > 0) ++ch_cnt;
            }
            hub_rem_child_hubs[k] = ch_cnt;
            if (in_pool(dag.hubs[k].H)) hub_built[k] = 1;
        }
    }

    int pick_next_hub() {
        std::vector<int> cands;
        for (size_t i = 0; i < dag.hubs.size(); ++i) {
            int hi = static_cast<int>(i);
            if (hub_built[hi]) continue;
            bool useful = (hub_rem_children[hi] >= 2) ||
                          (dag.hubs[hi].is_target && hub_rem_children[hi] >= 1) ||
                          (hub_rem_child_hubs[hi] >= 1);
            if (!useful) continue;
            int p = dag.hubs[hi].parent_link.parent_hub;
            if (p >= 0 && !hub_built[p] &&
                (hub_rem_children[p] >= 1 || hub_rem_child_hubs[p] >= 1)) {
                continue;
            }
            cands.push_back(hi);
        }
        if (cands.empty()) return -1;

        if (cfg.randomize_order && cands.size() > 1 && (rng() % 3 == 0)) {
            return cands[rng() % cands.size()];
        }
        int best = cands[0];
        double best_key = -1e18;
        for (int hi : cands) {
            double h_cost = rough_build_cost(dag.hubs[hi].H);
            double key = cfg.strict_hub_first
                             ? (-h_cost + 0.35 * dag.hubs[hi].savings)
                             : (dag.hubs[hi].savings / std::max(1.0, h_cost));
            if (key > best_key) {
                best_key = key;
                best = hi;
            }
        }
        return best;
    }

    u64 pick_standalone_target() {
        if (cfg.randomize_order && remaining.size() > 1 && (rng() % 4 == 0)) {
            return remaining[rng() % remaining.size()];
        }
        return *std::min_element(remaining.begin(), remaining.end());
    }

    bool run(const std::vector<u64>& starts, const std::vector<u64>& targets) {
        for (u64 s : starts) {
            start_set.insert(s);
            add_value(s, true);
        }
        for (u64 t : targets) {
            if (!coveredEver.contains(t) && remaining_set.insert(t).second) {
                remaining.push_back(t);
            }
        }
        std::sort(remaining.begin(), remaining.end());
        if (remaining.empty()) return true;

        dag = mine_shared_dag(remaining, limit, cfg.r_cap, cfg.d_cap, cfg.max_q,
                              cfg.min_hub_deg, cfg.min_savings, cfg.add_target_links,
                              timer, hard_deadline);

        hub_rem_children.assign(dag.hubs.size(), 0);
        hub_rem_child_hubs.assign(dag.hubs.size(), 0);
        hub_built.assign(dag.hubs.size(), 0);
        for (size_t i = 0; i < dag.hubs.size(); ++i) {
            for (const auto& e : dag.hubs[i].edges) {
                if (remaining_set.count(e.target)) ++hub_rem_children[i];
            }
            hub_rem_child_hubs[i] = static_cast<int>(dag.hubs[i].child_hubs.size());
            if (in_pool(dag.hubs[i].H)) hub_built[i] = 1;
        }
        freq_rem = dag.freq_constants;

        std::unordered_map<u64, std::vector<size_t>> stale;
        std::vector<Step> steps;
        size_t last_plan_size = static_cast<size_t>(-1);
        size_t last_remaining = static_cast<size_t>(-1);
        int stuck_iters = 0;
        int iters_since_cover = 0;
        int l2_vote_streak = 0;

        while (!remaining.empty()) {
            check_time();
            if (aborted || plan.size() > kMaxPlanSteps) {
                aborted = true;
                return false;
            }
            if (fast_fallback_mode) {
                fast_finish_remaining();
                return !aborted && remaining.empty();
            }
            if (remaining.size() == last_remaining) {
                ++iters_since_cover;
                if (plan.size() == last_plan_size) {
                    ++stuck_iters;
                } else {
                    stuck_iters = 0;
                }
            } else {
                stuck_iters = 0;
                iters_since_cover = 0;
                l2_vote_streak = 0;
            }
            last_plan_size = plan.size();
            last_remaining = remaining.size();

            if (stuck_iters >= 4 || iters_since_cover >= 16) {
                if (!monotonic_construct(pick_standalone_target())) {
                    aborted = true;
                    return false;
                }
                continue;
            }
            if (stuck_iters >= 2 || iters_since_cover >= 10) {
                construct(pick_standalone_target(), 0);
                continue;
            }
            refresh_hub_counts();

            // 1. Immediate L1 coverage
            grab_l1_targets();
            if (remaining.empty()) break;

            // 2. Hot Hub-Cluster Sweep
            bool swept = false;
            for (size_t i = 0; i < dag.hubs.size(); ++i) {
                int hi = static_cast<int>(i);
                if ((hub_rem_children[hi] > 0 || hub_rem_child_hubs[hi] > 0) &&
                    in_pool(dag.hubs[hi].H)) {
                    sweep_hub(hi);
                    swept = true;
                    break;
                }
            }
            if (swept) {
                l2_vote_streak = 0;
                continue;
            }

            // Active hub in L1 -> emit & sweep immediately!
            int l1_hub = -1;
            for (size_t i = 0; i < dag.hubs.size(); ++i) {
                int hi = static_cast<int>(i);
                if ((hub_rem_children[hi] >= 2 || hub_rem_child_hubs[hi] >= 1) &&
                    c01(dag.hubs[hi].H) == 1) {
                    l1_hub = hi;
                    break;
                }
            }
            if (l1_hub >= 0) {
                steps.clear();
                emit_l1(dag.hubs[l1_hub].H, steps);
                apply(steps);
                sweep_hub(l1_hub);
                l2_vote_streak = 0;
                continue;
            }

            // 3. Multi-Target + Active-Hub L2 Voting
            if (l2_vote_streak < 6) {
                std::unordered_map<u64, int> votes;
                std::vector<u64> xs;
                for (u64 t : remaining) {
                    xs.clear();
                    find2(t, nullptr, &xs);
                    std::sort(xs.begin(), xs.end());
                    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
                    for (u64 x : xs) votes[x] += 10;
                }
                for (size_t i = 0; i < dag.hubs.size(); ++i) {
                    int hi = static_cast<int>(i);
                    if (hub_built[hi] || hub_rem_children[hi] < 2) continue;
                    xs.clear();
                    find2(dag.hubs[hi].H, nullptr, &xs);
                    std::sort(xs.begin(), xs.end());
                    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
                    int w = 8 * std::min(4, hub_rem_children[hi]);
                    for (u64 x : xs) votes[x] += w;
                }
                if (!votes.empty()) {
                    for (auto& [x, v] : votes) {
                        auto fi = freq_rem.find(x);
                        if (fi != freq_rem.end() && fi->second > 0) {
                            v += std::min(6, fi->second * 2);
                        }
                    }
                    std::vector<u64> top;
                    int best_v = -1;
                    for (const auto& [x, v] : votes) {
                        if (v > best_v) {
                            best_v = v;
                            top.clear();
                        }
                        if (v == best_v) top.push_back(x);
                    }
                    std::sort(top.begin(), top.end());
                    u64 chosen = cfg.randomize_order ? top[rng() % top.size()] : top[0];
                    steps.clear();
                    emit_l1(chosen, steps);
                    if (apply(steps)) {
                        ++l2_vote_streak;
                        continue;
                    }
                }
            }

            // 4. Short-path iterative deepening (B = 3..b_full)
            bool found = false;
            if (!check_time()) {
                std::vector<u64> cand_vals = remaining;
                for (size_t i = 0; i < dag.hubs.size(); ++i) {
                    int hi = static_cast<int>(i);
                    if (!hub_built[hi] && hub_rem_children[hi] >= 2) {
                        cand_vals.push_back(dag.hubs[hi].H);
                    }
                }
                if (cfg.randomize_order) std::shuffle(cand_vals.begin(), cand_vals.end(), rng);

                for (int B = 3; B <= cfg.b_full && !found; ++B) {
                    if (B >= 4 && check_time()) break;
                    for (u64 v : cand_vals) {
                        if (check_time()) break;
                        auto& seen = stale[v];
                        if (seen.size() <= static_cast<size_t>(B)) seen.resize(B + 1, 0);
                        if (B > 3 && seen[B] && created < seen[B] + seen[B] / 4 + 3) continue;
                        steps.clear();
                        work = 0.0;
                        if (solve(v, B, steps)) {
                            apply(steps);
                            found = true;
                            break;
                        }
                        seen[B] = created;
                    }
                }
            }
            if (found) {
                l2_vote_streak = 0;
                continue;
            }

            // 5. Hub-First Construct OR Standalone Target Construct
            l2_vote_streak = 0;
            int next_hub = pick_next_hub();
            if (next_hub >= 0) {
                u64 H = dag.hubs[next_hub].H;
                construct(H, 0);
                if (!aborted && !fast_fallback_mode && in_pool(H)) {
                    sweep_hub(next_hub);
                }
                continue;
            }

            u64 t = pick_standalone_target();
            construct(t, 0);
        }
        return !aborted && remaining.empty();
    }
};

// ============================================================================
// Step 3 & 4: Phase 2 — Warm-Started MITM Basis Engine (`mitm_basis` core)
// ============================================================================

struct Node {
    u64 val = 0;
    uint32_t p1 = 0;
    uint32_t p2 = 0;
    char op = 0;
    uint8_t cost = 0;  // 0: in B, 1: in L1(B), 2: in L2(B)
};

struct ValIdx {
    u64 val = 0;
    uint32_t idx = 0;
    uint8_t cost = 0;
};

// 256 KB (2^21 bits) 2-hash Bloom filter: rejects ~96% of non-hits in L2 cache.
class BloomFilter {
public:
    static constexpr size_t kBits = size_t(1) << 21;
    static constexpr size_t kMask = kBits - 1;

    BloomFilter() : words_(kBits >> 6, 0) {}

    void clear() { std::fill(words_.begin(), words_.end(), 0); }

    void insert(u64 k) {
        u64 h = elf::splitmix64(k);
        size_t b1 = static_cast<size_t>(h) & kMask;
        size_t b2 = static_cast<size_t>(h >> 21) & kMask;
        words_[b1 >> 6] |= (u64(1) << (b1 & 63));
        words_[b2 >> 6] |= (u64(1) << (b2 & 63));
    }

    inline bool maybe_contains(u64 k) const {
        u64 h = elf::splitmix64(k);
        size_t b1 = static_cast<size_t>(h) & kMask;
        if ((words_[b1 >> 6] & (u64(1) << (b1 & 63))) == 0) return false;
        size_t b2 = static_cast<size_t>(h >> 21) & kMask;
        return (words_[b2 >> 6] & (u64(1) << (b2 & 63))) != 0;
    }

private:
    std::vector<u64> words_;
};

// Deduplicating topological emitter for ephemeral target mini-plans.
struct LocalEmitter {
    const std::vector<Node>& H_vec;
    FlatMap<uint8_t> have;
    std::vector<Step> steps;

    LocalEmitter(const std::vector<Node>& h, const std::vector<u64>& basis)
        : H_vec(h), have(basis.size() * 2 + 32) {
        for (u64 v : basis) have.insert(v, 1);
    }

    void reset(const std::vector<u64>& basis) {
        have.clear();
        for (u64 v : basis) have.insert(v, 1);
        steps.clear();
    }

    void emit_node(uint32_t idx) {
        const Node& nd = H_vec[idx];
        if (nd.cost == 0 || have.contains(nd.val)) return;
        emit_node(nd.p1);
        emit_node(nd.p2);
        if (!have.contains(nd.val)) {
            steps.push_back(Step{nd.op, H_vec[nd.p1].val, H_vec[nd.p2].val, nd.val});
            have.insert(nd.val, 1);
        }
    }

    void emit_op(char op, u64 a, u64 b, u64 r) {
        if (have.contains(r)) return;
        steps.push_back(Step{op, a, b, r});
        have.insert(r, 1);
    }

    void append_steps(const std::vector<Step>& sub) {
        for (const Step& s : sub) {
            if (!have.contains(s.r)) {
                steps.push_back(s);
                have.insert(s.r, 1);
            }
        }
    }
};

struct ResidualCand {
    u64 r = 0;
    uint32_t u_idx = 0;
    uint32_t v_idx = 0;
    char sign = '+';  // '+' means V = u*v + r, '-' means V = u*v - r
    uint8_t oc = 0;   // outer cost: c(u) + c(v) + 2
};

struct LinearResCand {
    u64 r = 0;
    uint32_t b_idx = 0;
    uint32_t x_idx = 0;
    uint32_t y_idx = 0;
    uint8_t kind = 0;   // 0: b*x + y, 1: b*x - y, 2: y - b*x
    char r_sign = '+';  // '+' means V = s + r, '-' means V = s - r
    uint8_t oc = 0;     // c(x) + c(y) + 3
};

struct BeamCand {
    u64 q = 0;
    uint32_t x_idx = 0;
    uint32_t r_idx = UINT32_MAX;
    char sign = '*';
    uint8_t oc = 0;
    double score = 0.0;
};

struct BeamState {
    u64 q = 0;
    uint8_t cost = 0;
    double score = 0.0;
    struct Link {
        uint32_t x_idx = 0;
        uint32_t r_idx = UINT32_MAX;
        char sign = '*';
        u64 parent_val = 0;
    };
    std::vector<Link> chain;
};

inline int log10_half_bucket(u64 v) {
    if (v <= 1) return 0;
    return static_cast<int>(std::log10(static_cast<double>(v)) * 2.0);
}

inline double log2_approx(u64 v) {
    if (v <= 1) return 0.0;
    return std::log2(static_cast<double>(v));
}

struct Phase2Config {
    int targetBasis = 41;
    int warmCap = 36;
    int maxPerBucket = 2;
    int maxUpfrontTopup = 40;
    bool onlineAbsorb = true;
    bool jitter = false;
    int mode = 0;
    bool allowAbort = false;
};

class Phase2Engine {
public:
    Phase2Engine(const Puzzle& pz, const Options& opt, const Timer& timer,
                 double deadline, u64 seed)
        : pz_(pz),
          opt_(opt),
          timer_(timer),
          deadline_(deadline),
          rng_(seed),
          M_map_(8192),
          H_map_(524288) {}

    bool time_up() const { return timer_.elapsed_sec() >= deadline_; }

    size_t basisSize() const { return basis_.size(); }
    size_t mSize() const { return M_sorted_.size(); }
    size_t hSize() const { return H_sorted_.size(); }

    std::optional<std::vector<Step>> run(
        const std::vector<u64>& p1_pool,
        const std::vector<Step>& p1_plan,
        const std::vector<u64>& p1_hubs,
        const FlatMap<uint8_t>& p1_covered,
        const std::vector<u64>& random_targets,
        const Phase2Config& cfg) {
        // Filter out targets already covered by Phase 1.
        rem_targets_.clear();
        for (u64 t : random_targets) {
            if (!p1_covered.contains(t)) rem_targets_.push_back(t);
        }
        std::sort(rem_targets_.begin(), rem_targets_.end());
        rem_targets_.erase(std::unique(rem_targets_.begin(), rem_targets_.end()),
                           rem_targets_.end());

        if (rem_targets_.empty()) {
            return elf::verify_and_prune_plan(pz_, p1_plan, opt_.regs);
        }

        max_t_ = *std::max_element(rem_targets_.begin(), rem_targets_.end());
        min_t_ = *std::min_element(rem_targets_.begin(), rem_targets_.end());
        dec_d_ = std::max(1.0, std::log10(static_cast<double>(max_t_)));

        const int max_reg_cap = opt_.regs > 0 ? std::max(6, opt_.regs - 6) : 42;
        const int final_basis = std::min(cfg.targetBasis, max_reg_cap);

        basis_.clear();
        basis_steps_.clear();
        FlatMap<uint8_t> in_b(128);

        // 1. Curate warm-start basis from Phase 1's live pool (ZERO extra steps!)
        curate_warm_basis(p1_pool, p1_hubs, std::min(cfg.warmCap, final_basis),
                          cfg.maxPerBucket, cfg.jitter, in_b);

        // Ensure 1 and 2 are available in B if missing (at most 1-2 steps).
        if (!in_b.contains(1) && !basis_.empty() && static_cast<int>(basis_.size()) < final_basis) {
            u64 s0 = basis_[0];
            basis_steps_.push_back(Step{'/', s0, s0, 1});
            basis_.push_back(1);
            in_b.insert(1, 1);
        }
        if (!in_b.contains(2) && in_b.contains(1) && pz_.limit >= 2 &&
            static_cast<int>(basis_.size()) < final_basis) {
            basis_steps_.push_back(Step{'+', 1, 1, 2});
            basis_.push_back(2);
            in_b.insert(2, 1);
        }

        // 2. Determine upfront top-up target vs online absorption target.
        const bool use_absorb = cfg.onlineAbsorb && rem_targets_.size() >= 5 &&
                                static_cast<int>(basis_.size()) < final_basis;
        int init_target_basis = final_basis;
        if (use_absorb) {
            if (p1_pool.size() < 20 && rem_targets_.size() >= 12) {
                // Start lean (14..26) like mitm_basis so the first 3 targets pay for the basis!
                int lean = (cfg.mode == 0) ? 16 : std::max(18, final_basis - 14);
                init_target_basis = std::max<int>(static_cast<int>(basis_.size()) + 4, lean);
            } else {
                int leave_for_absorb = (rem_targets_.size() <= 12) ? 6 : 10;
                int min_init = std::min(final_basis, std::max<int>(static_cast<int>(basis_.size()), 20));
                init_target_basis = std::max(min_init, final_basis - leave_for_absorb);
            }
            init_target_basis = std::min(init_target_basis, final_basis);
        }

        // Respect maxUpfrontTopup ONLY as a cap above current basis (never starve low-pool!).
        int allowed_upfront = std::min(init_target_basis,
                                       static_cast<int>(basis_.size()) + cfg.maxUpfrontTopup);
        if (static_cast<int>(basis_.size()) < allowed_upfront) {
            extend_basis_greedy(cfg.mode, allowed_upfront, basis_steps_, in_b, cfg.jitter);
        }

        build_tables();
        size_t last_built_bsz = basis_.size();

        // Track covered targets (Phase-1 covered + any lucky hit during basis top-up)
        FlatMap<uint8_t> covered(rem_targets_.size() * 2 + 128);
        for (u64 s : pz_.starts) covered.insert(s, 1);
        for (const Step& s : p1_plan) covered.insert(s.r, 1);
        for (const Step& s : basis_steps_) covered.insert(s.r, 1);

        // Prefix starts with Phase-1 plan + upfront basis top-up steps.
        std::vector<Step> absorbed_prefix;
        absorbed_prefix.reserve(p1_plan.size() + basis_steps_.size() + 64);
        for (const Step& s : p1_plan) absorbed_prefix.push_back(s);
        for (const Step& s : basis_steps_) absorbed_prefix.push_back(s);

        // Save Phase-1 + initial basis prefix for emergency fallback.
        const std::vector<Step> safe_prefix = absorbed_prefix;

        std::vector<u64> order = rem_targets_;
        if (cfg.jitter && !use_absorb && order.size() > 2) {
            std::shuffle(order.begin(), order.end(), rng_);
        }

        std::vector<std::vector<Step>> post_target_steps;
        post_target_steps.reserve(order.size());

        int targets_solved = 0;
        bool basis_frozen = (!use_absorb) || (static_cast<int>(basis_.size()) >= final_basis);

        for (size_t ti = 0; ti < order.size(); ++ti) {
            if (cfg.allowAbort && time_up()) return std::nullopt;
            u64 t = order[ti];
            if (covered.contains(t)) continue;

            // Per-target soft deadline so EVERY remaining target gets full MITM + b*x+-y!
            size_t rem_cnt = order.size() - ti;
            double now = timer_.elapsed_sec();
            double rem_sec = std::max(0.0, deadline_ - now);
            double target_dl = std::min(deadline_, now + (rem_sec / static_cast<double>(rem_cnt)) * 1.25);

            std::vector<Step> t_plan = solve_target(t, target_dl, !cfg.allowAbort);
            for (const Step& s : t_plan) covered.insert(s.r, 1);
            ++targets_solved;

            if (!basis_frozen) {
                for (const Step& s : t_plan) absorbed_prefix.push_back(s);

                std::vector<u64> newly_created;
                for (const Step& s : t_plan) {
                    if (s.r >= 2 && s.r <= max_t_ && !in_b.contains(s.r)) {
                        newly_created.push_back(s.r);
                    }
                }

                std::vector<int> bins(
                    static_cast<size_t>(std::max(4, static_cast<int>(dec_d_ * 2.0) + 8)), 0);
                for (u64 x : basis_) {
                    int bk = std::min(static_cast<int>(bins.size()) - 1, log10_half_bucket(x));
                    ++bins[bk];
                }
                int half_bk = std::max(1, static_cast<int>(dec_d_));

                std::sort(newly_created.begin(), newly_created.end(),
                          [&](u64 a, u64 b) {
                              int bka = std::min(static_cast<int>(bins.size()) - 1,
                                                 log10_half_bucket(a));
                              int bkb = std::min(static_cast<int>(bins.size()) - 1,
                                                 log10_half_bucket(b));
                              double sa = -8.0 * bins[bka] - 0.5 * std::abs(bka - half_bk);
                              double sb = -8.0 * bins[bkb] - 0.5 * std::abs(bkb - half_bk);
                              if (a > max_t_) sa -= 50.0;
                              if (b > max_t_) sb -= 50.0;
                              return sa > sb;
                          });

                bool added_any = false;
                for (u64 v : newly_created) {
                    if (static_cast<int>(basis_.size()) >= final_basis) break;
                    if (!in_b.insert(v, 1)) continue;
                    basis_.push_back(v);
                    int bk = std::min(static_cast<int>(bins.size()) - 1, log10_half_bucket(v));
                    ++bins[bk];
                    added_any = true;
                }

                if (targets_solved >= 3 && static_cast<int>(basis_.size()) < final_basis &&
                    order.size() >= 12) {
                    int rem_fill = std::min(final_basis - static_cast<int>(basis_.size()), 8);
                    extend_basis_greedy(cfg.mode, static_cast<int>(basis_.size()) + rem_fill,
                                        absorbed_prefix, in_b, cfg.jitter);
                    basis_frozen = true;
                    added_any = true;
                } else if (static_cast<int>(basis_.size()) >= final_basis ||
                           targets_solved >= 3) {
                    basis_frozen = true;
                }

                // Batch table rebuilds to avoid burning ~35ms after every single absorbed target!
                if (added_any && ti + 1 < order.size() &&
                    (basis_frozen || basis_.size() >= last_built_bsz + 8)) {
                    build_tables();
                    last_built_bsz = basis_.size();
                }
            } else {
                post_target_steps.push_back(std::move(t_plan));
            }
        }

        return assemble_and_optimize(absorbed_prefix, safe_prefix, post_target_steps);
    }

private:
    const Puzzle& pz_;
    const Options& opt_;
    const Timer& timer_;
    double deadline_;
    std::mt19937_64 rng_;

    std::vector<u64> rem_targets_;
    u64 max_t_ = 1;
    u64 min_t_ = 1;
    double dec_d_ = 1.0;

    std::vector<u64> basis_;
    std::vector<Step> basis_steps_;
    std::vector<Node> H_vec_;
    FlatMap<uint32_t> M_map_;
    FlatMap<uint32_t> H_map_;
    BloomFilter H_bloom_;
    std::vector<ValIdx> M_sorted_;
    std::vector<ValIdx> H_sorted_;
    u64 H_max_val_ = 0;

    // -------------------------------------------------------------------------
    // Curate Warm-Start Basis from Phase 1's Live Pool (preserving top hubs &
    // half-decade logarithmic diversity without clustering or giant-value bloat)
    // -------------------------------------------------------------------------
    void curate_warm_basis(const std::vector<u64>& p1_pool,
                           const std::vector<u64>& p1_hubs,
                           int warm_cap, int max_per_bucket, bool jitter,
                           FlatMap<uint8_t>& in_b) {
        if (p1_pool.empty() || warm_cap <= 0) return;

        int max_bk = std::max(2, static_cast<int>(dec_d_ * 2.0));
        int half_bk = std::max(1, static_cast<int>(dec_d_));
        std::vector<int> bins(static_cast<size_t>(max_bk + 16), 0);
        const u64 max_lottery_d = std::min<u64>(200ULL, pz_.limit / std::max<u64>(1ULL, min_t_));

        FlatMap<uint8_t> pool_set(p1_pool.size() * 2 + 8);
        std::vector<u64> uniq_pool;
        uniq_pool.reserve(p1_pool.size());
        for (u64 v : p1_pool) {
            if (v >= 1 && v <= pz_.limit && pool_set.insert(v, 1)) {
                uniq_pool.push_back(v);
            }
        }

        // If Phase 1 barely ran (small pool <= warm_cap and < 16), keep everything useful!
        if (static_cast<int>(uniq_pool.size()) <= std::min(warm_cap, 16)) {
            for (u64 v : uniq_pool) {
                if (static_cast<int>(basis_.size()) >= warm_cap) break;
                if (in_b.insert(v, 1)) {
                    basis_.push_back(v);
                    int bk = std::min(static_cast<int>(bins.size()) - 1, log10_half_bucket(v));
                    ++bins[bk];
                }
            }
            return;
        }

        auto add_val = [&](u64 v) {
            if (static_cast<int>(basis_.size()) >= warm_cap) return false;
            if (!in_b.insert(v, 1)) return false;
            basis_.push_back(v);
            int bk = std::min(static_cast<int>(bins.size()) - 1, log10_half_bucket(v));
            ++bins[bk];
            return true;
        };

        // 1. Always keep 1, 2, and up to 4 small constants in [3..25]
        if (pool_set.contains(1)) add_val(1);
        if (pool_set.contains(2)) add_val(2);
        std::vector<u64> smalls;
        for (u64 v : uniq_pool) {
            if (v >= 3 && v <= 25) smalls.push_back(v);
        }
        std::sort(smalls.begin(), smalls.end());
        for (u64 v : smalls) {
            if (static_cast<int>(basis_.size()) >= std::min(warm_cap, 6)) break;
            add_val(v);
        }

        // 2. Preserve up to 8-10 top well-spaced Phase-1 hubs that are still in p1_pool!
        int hubs_kept = 0;
        const int max_hubs_keep = std::min(10, std::max(4, warm_cap / 4));
        for (u64 h : p1_hubs) {
            if (hubs_kept >= max_hubs_keep || static_cast<int>(basis_.size()) >= warm_cap) break;
            if (!pool_set.contains(h) || in_b.contains(h) || h > max_t_) continue;
            int bk = std::min(static_cast<int>(bins.size()) - 1, log10_half_bucket(h));
            if (bins[bk] >= 2) continue;
            bool too_close = false;
            for (u64 u : basis_) {
                u64 hi = std::max(u, h), lo = std::min(u, h);
                if (hi > 10 && (hi - lo) * 40ULL < hi) {
                    too_close = true;
                    break;
                }
            }
            if (too_close) continue;
            if (add_val(h)) ++hubs_kept;
        }

        // 3. Curate diverse non-clustered pool values across half-decade buckets
        while (static_cast<int>(basis_.size()) < warm_cap) {
            u64 best_v = 0;
            double best_sc = -1e18;

            for (u64 v : uniq_pool) {
                if (in_b.contains(v)) continue;
                int bk = std::min(static_cast<int>(bins.size()) - 1, log10_half_bucket(v));
                if (bins[bk] >= max_per_bucket) continue;

                double sc = 0.0;
                if (v > max_t_) {
                    sc -= 50.0;
                } else if (bk <= half_bk) {
                    sc += (std::abs(bk - half_bk) <= 3) ? 16.0 : 11.0;
                } else if (bk <= static_cast<int>(dec_d_ * 1.75)) {
                    sc += 7.5;
                } else {
                    sc -= 6.0;
                }

                if (v >= 2 && v <= max_lottery_d) sc += 7.0;
                sc -= 12.0 * static_cast<double>(bins[bk]);

                bool too_close = false;
                int good_prods = 0;
                for (u64 u : basis_) {
                    u64 hi = std::max(u, v), lo = std::min(u, v);
                    if (hi > 10 && (hi - lo) * 40ULL < hi) too_close = true;
                    u128 p = static_cast<u128>(u) * v;
                    if (p <= pz_.limit && p <= max_t_) {
                        int pbk = log10_half_bucket(static_cast<u64>(p));
                        if (std::abs(pbk - half_bk) <= 3 || pbk >= max_bk - 2) ++good_prods;
                    }
                }
                if (too_close) sc -= 28.0;
                sc += std::min(12.0, static_cast<double>(good_prods) * 1.2);
                if (jitter) {
                    sc += static_cast<double>(rng_() & 0xFFFFULL) / 65536.0 * 3.0;
                }

                if (sc > best_sc) {
                    best_sc = sc;
                    best_v = v;
                }
            }
            if (best_v == 0 || best_sc < -15.0) break;
            add_val(best_v);
        }
    }

    // -------------------------------------------------------------------------
    // Greedy Multi-Scale Dense Basis Extension (from `mitm_basis.cpp`)
    // -------------------------------------------------------------------------
    void extend_basis_greedy(int mode, int cap, std::vector<Step>& out_steps,
                             FlatMap<uint8_t>& in_b, bool jitter) {
        FlatMap<uint8_t> target_set(rem_targets_.size() * 2 + 8);
        for (u64 t : rem_targets_) target_set.insert(t, 1);

        int m = mode % 3;
        int max_bk = std::max(2, static_cast<int>(dec_d_ * 2.0));
        int half_bk = std::max(1, static_cast<int>(dec_d_));

        FlatMap<Step> l1_map(4096);
        std::vector<u64> l1_sorted;
        std::vector<int> bins(static_cast<size_t>(max_bk + 16), 0);

        while (static_cast<int>(basis_.size()) < cap) {
            elf::build_L1(basis_, pz_.limit, l1_map, l1_sorted);
            if (l1_sorted.empty()) break;

            std::fill(bins.begin(), bins.end(), 0);
            for (u64 x : basis_) {
                int bk = std::min(static_cast<int>(bins.size()) - 1, log10_half_bucket(x));
                ++bins[bk];
            }

            std::vector<u64> cands;
            if (l1_sorted.size() <= 200) {
                cands = l1_sorted;
            } else {
                std::vector<std::vector<u64>> by_bk(bins.size());
                for (u64 v : l1_sorted) {
                    if (target_set.contains(v)) {
                        cands.push_back(v);
                        continue;
                    }
                    int bk = std::min(static_cast<int>(bins.size()) - 1, log10_half_bucket(v));
                    by_bk[bk].push_back(v);
                }
                for (size_t bk = 0; bk < by_bk.size(); ++bk) {
                    auto& vec = by_bk[bk];
                    if (vec.empty()) continue;
                    size_t take = std::min<size_t>(vec.size(), 12);
                    if (vec.size() <= take) {
                        for (u64 v : vec) cands.push_back(v);
                    } else {
                        for (size_t i = 0; i < take; ++i) {
                            cands.push_back(vec[rng_() % vec.size()]);
                        }
                    }
                }
                std::sort(cands.begin(), cands.end());
                cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
            }

            u64 best_v = 0;
            double best_score = -1e18;

            for (u64 v : cands) {
                if (target_set.contains(v)) {
                    best_v = v;
                    best_score = 1e18;
                    break;
                }

                int bk = std::min(static_cast<int>(bins.size()) - 1, log10_half_bucket(v));

                double l1_gain = 0.0;
                for (u64 b : basis_) {
                    auto score_new = [&](u64 r) {
                        if (r == 0 || r > pz_.limit) return;
                        if (in_b.contains(r) || l1_map.contains(r)) return;
                        double w = 1.0;
                        if (r <= max_t_) {
                            int rbk = log10_half_bucket(r);
                            int dist_half = std::abs(rbk - half_bk);
                            if (rbk >= max_bk - 2) w = 1.45;
                            else if (dist_half <= 3) w = (m == 2) ? 1.65 : 1.35;
                            else if (rbk <= half_bk) w = (m == 1) ? 1.35 : 1.15;
                            else w = 1.1;
                        } else {
                            w = 0.2;
                        }
                        if (target_set.contains(r)) w += 40.0;
                        l1_gain += w;
                    };

                    if (v <= pz_.limit && b <= pz_.limit - v) score_new(v + b);
                    u64 prod = elf::safe_mul(v, b, pz_.limit);
                    if (prod != 0) score_new(prod);
                    if (v > b) {
                        score_new(v - b);
                        if (v % b == 0) score_new(v / b);
                    } else if (b > v) {
                        score_new(b - v);
                        if (b % v == 0) score_new(b / v);
                    }
                }

                double bucket_pref = 0.0;
                if (v > max_t_) {
                    bucket_pref = -45.0;
                } else if (bk <= half_bk) {
                    bucket_pref = (m == 1) ? 10.0 : 7.0;
                } else if (bk <= static_cast<int>(dec_d_ * 1.75)) {
                    bucket_pref = (m == 2) ? 11.0 : 5.5;
                } else {
                    bucket_pref = -4.0;
                }

                if (v >= 3 && v <= 25 && bins[0] + bins[1] < 5) {
                    bucket_pref += 16.0;
                }

                double t_bonus = 0.0;
                if (v >= 3 && v <= max_t_) {
                    int exact_divs = 0;
                    for (u64 t : rem_targets_) {
                        if (t % v == 0) ++exact_divs;
                    }
                    t_bonus += exact_divs * 6.0;
                }

                double bucket_pen = (m == 2 ? 12.0 : 10.5) * static_cast<double>(bins[bk]);
                double noise = jitter
                                   ? (static_cast<double>(rng_() & 0xFFFFULL) / 65536.0 * 4.5)
                                   : (static_cast<double>(rng_() & 0xFFFFULL) / 65536.0 * 1.5);
                double sc = l1_gain + bucket_pref + t_bonus - bucket_pen + noise;

                if (sc > best_score) {
                    best_score = sc;
                    best_v = v;
                }
            }

            if (best_v == 0) break;
            const Step* st = l1_map.find(best_v);
            if (!st) break;
            out_steps.push_back(*st);
            basis_.push_back(best_v);
            in_b.insert(best_v, 1);
        }
    }

    // -------------------------------------------------------------------------
    // Precompute M = B u L1(B) and H = B u L1(B) u L2(B)
    // -------------------------------------------------------------------------
    void build_tables() {
        H_vec_.clear();
        M_map_.clear();
        H_map_.clear();
        H_bloom_.clear();
        M_sorted_.clear();
        H_sorted_.clear();

        H_vec_.reserve(260000);
        M_map_.reserve(4096);
        H_map_.reserve(320000);

        for (u64 v : basis_) {
            uint32_t idx = static_cast<uint32_t>(H_vec_.size());
            H_vec_.push_back(Node{v, 0, 0, 0, 0});
            M_map_.insert(v, idx);
            H_map_.insert(v, idx);
            H_bloom_.insert(v);
        }
        const uint32_t B_sz = static_cast<uint32_t>(H_vec_.size());

        auto add_l1 = [&](char op, uint32_t i1, uint32_t i2, u64 r) {
            if (r == 0 || r > pz_.limit) return;
            if (H_map_.contains(r)) return;
            uint32_t idx = static_cast<uint32_t>(H_vec_.size());
            H_vec_.push_back(Node{r, i1, i2, op, 1});
            M_map_.insert(r, idx);
            H_map_.insert(r, idx);
            H_bloom_.insert(r);
        };

        for (uint32_t i = 0; i < B_sz; ++i) {
            u64 u = H_vec_[i].val;
            for (uint32_t j = i; j < B_sz; ++j) {
                u64 v = H_vec_[j].val;
                if (u <= pz_.limit && v <= pz_.limit - u) add_l1('+', i, j, u + v);
                u64 prod = elf::safe_mul(u, v, pz_.limit);
                if (prod != 0) add_l1('*', i, j, prod);
                if (u > v) {
                    add_l1('-', i, j, u - v);
                    if (u % v == 0) add_l1('/', i, j, u / v);
                } else if (v > u) {
                    add_l1('-', j, i, v - u);
                    if (v % u == 0) add_l1('/', j, i, v / u);
                } else {
                    add_l1('/', i, i, 1);
                }
            }
        }
        const uint32_t M_sz = static_cast<uint32_t>(H_vec_.size());

        M_sorted_.reserve(M_sz);
        for (uint32_t i = 0; i < M_sz; ++i) {
            M_sorted_.push_back(ValIdx{H_vec_[i].val, i, H_vec_[i].cost});
        }
        std::sort(M_sorted_.begin(), M_sorted_.end(),
                  [](const ValIdx& a, const ValIdx& b) { return a.val < b.val; });

        constexpr size_t kMaxH = 300000;
        auto add_l2 = [&](char op, uint32_t i1, uint32_t i2, u64 r) {
            if (r == 0 || r > pz_.limit || H_vec_.size() >= kMaxH) return;
            if (H_map_.contains(r)) return;
            uint32_t idx = static_cast<uint32_t>(H_vec_.size());
            H_vec_.push_back(Node{r, i1, i2, op, 2});
            H_map_.insert(r, idx);
            H_bloom_.insert(r);
        };

        for (uint32_t i = B_sz; i < M_sz; ++i) {
            u64 x = H_vec_[i].val;
            if (x <= pz_.limit - x) add_l2('+', i, i, x + x);
            u64 sq = elf::safe_mul(x, x, pz_.limit);
            if (sq != 0) add_l2('*', i, i, sq);
        }

        for (uint32_t i = B_sz; i < M_sz && H_vec_.size() < kMaxH; ++i) {
            u64 x = H_vec_[i].val;
            for (uint32_t j = 0; j < B_sz; ++j) {
                u64 b = H_vec_[j].val;
                if (x <= pz_.limit - b) add_l2('+', i, j, x + b);
                u64 prod = elf::safe_mul(x, b, pz_.limit);
                if (prod != 0) add_l2('*', i, j, prod);
                if (x > b) {
                    add_l2('-', i, j, x - b);
                    if (x % b == 0) add_l2('/', i, j, x / b);
                } else if (b > x) {
                    add_l2('-', j, i, b - x);
                    if (b % x == 0) add_l2('/', j, i, b / x);
                }
            }
        }

        H_sorted_.reserve(H_vec_.size());
        for (uint32_t i = 0; i < H_vec_.size(); ++i) {
            H_sorted_.push_back(ValIdx{H_vec_[i].val, i, H_vec_[i].cost});
        }
        std::sort(H_sorted_.begin(), H_sorted_.end(),
                  [](const ValIdx& a, const ValIdx& b) { return a.val < b.val; });
        H_max_val_ = H_sorted_.empty() ? 0 : H_sorted_.back().val;
    }

    inline const uint32_t* find_H(u64 v) const {
        if (v == 0 || v > H_max_val_ || !H_bloom_.maybe_contains(v)) return nullptr;
        return H_map_.find(v);
    }

    // Ultra-fast O(|B|) cost-<=3 checker: V in H, self-op(H), or H op B (~0.5 us)
    int solve_c3_hb(u64 V, int max_cost, std::vector<Step>& out) const {
        out.clear();
        if (V == 0 || V > pz_.limit || max_cost < 0) return -1;

        if (const uint32_t* hi = find_H(V)) {
            if (static_cast<int>(H_vec_[*hi].cost) <= max_cost) {
                LocalEmitter em(H_vec_, basis_);
                em.emit_node(*hi);
                out = std::move(em.steps);
                return static_cast<int>(out.size());
            }
            return -1;
        }
        if (max_cost <= 2) return -1;

        auto emit_pair = [&](uint32_t i1, uint32_t i2, char op) -> int {
            LocalEmitter em(H_vec_, basis_);
            em.emit_node(i1);
            em.emit_node(i2);
            em.emit_op(op, H_vec_[i1].val, H_vec_[i2].val, V);
            if (static_cast<int>(em.steps.size()) <= max_cost) {
                out = std::move(em.steps);
                return static_cast<int>(out.size());
            }
            return -1;
        };

        if ((V & 1ULL) == 0) {
            if (const uint32_t* hi = find_H(V >> 1)) {
                int c = emit_pair(*hi, *hi, '+');
                if (c >= 0) return c;
            }
        }
        {
            u64 s = elf::isqrt64(V);
            if (s * s == V) {
                if (const uint32_t* hi = find_H(s)) {
                    int c = emit_pair(*hi, *hi, '*');
                    if (c >= 0) return c;
                }
            }
        }

        const uint32_t B_sz = static_cast<uint32_t>(basis_.size());
        for (uint32_t bi = 0; bi < B_sz; ++bi) {
            u64 b = basis_[bi];
            if (V > b) {
                if (const uint32_t* hi = find_H(V - b)) {
                    int c = emit_pair(*hi, bi, '+');
                    if (c >= 0) return c;
                }
            }
            if (V <= pz_.limit - b) {
                if (const uint32_t* hi = find_H(V + b)) {
                    int c = emit_pair(*hi, bi, '-');
                    if (c >= 0) return c;
                }
            }
            if (b > V) {
                if (const uint32_t* hi = find_H(b - V)) {
                    int c = emit_pair(bi, *hi, '-');
                    if (c >= 0) return c;
                }
                if (b % V == 0) {
                    if (const uint32_t* hi = find_H(b / V)) {
                        int c = emit_pair(bi, *hi, '/');
                        if (c >= 0) return c;
                    }
                }
            }
            if (b > 1) {
                if (V % b == 0) {
                    if (const uint32_t* hi = find_H(V / b)) {
                        int c = emit_pair(*hi, bi, '*');
                        if (c >= 0) return c;
                    }
                }
                u64 prod = elf::safe_mul(V, b, pz_.limit);
                if (prod != 0) {
                    if (const uint32_t* hi = find_H(prod)) {
                        int c = emit_pair(*hi, bi, '/');
                        if (c >= 0) return c;
                    }
                }
            }
        }
        return -1;
    }

    // Full sub-solver up to max_cost (3..7): H op B, M op H, (H op B) op B, M*H +- H, H +- H
    int solve_sub(u64 V, int max_cost, bool allow_two_sum, bool allow_m_x_h,
                  std::vector<Step>& out, bool allow_hb_b = true) const {
        out.clear();
        if (V == 0 || V > pz_.limit || max_cost < 0) return -1;

        int c3 = solve_c3_hb(V, std::min(max_cost, 3), out);
        if (c3 >= 0) return c3;
        if (max_cost <= 2) return -1;

        int best_c = max_cost + 1;
        std::vector<Step> best_steps;

        auto accept_op = [&](uint32_t i1, uint32_t i2, char op, u64 r) {
            int est = static_cast<int>(H_vec_[i1].cost) + static_cast<int>(H_vec_[i2].cost) + 1;
            if (est >= best_c && i1 != i2) {
                if (!(est == best_c && H_vec_[i1].cost == 2 && H_vec_[i2].cost == 2 &&
                      H_vec_[i1].p1 == H_vec_[i2].p1)) {
                    return;
                }
            }
            LocalEmitter em(H_vec_, basis_);
            em.emit_node(i1);
            em.emit_node(i2);
            em.emit_op(op, H_vec_[i1].val, H_vec_[i2].val, r);
            int sz = static_cast<int>(em.steps.size());
            if (sz < best_c) {
                best_c = sz;
                best_steps = std::move(em.steps);
            }
        };

        // Cost 3 (M op M) and Cost 4 (M op H)
        for (const ValIdx& mx : M_sorted_) {
            if (mx.cost == 0) continue;
            u64 x = mx.val;
            int rem_c = best_c - 1 - static_cast<int>(mx.cost) - 1;
            if (rem_c < 0) continue;
            auto check_other = [&](u64 y, char op, bool x_first) {
                const uint32_t* hi = find_H(y);
                if (!hi) return;
                if (static_cast<int>(H_vec_[*hi].cost) > rem_c) return;
                if (x_first) accept_op(mx.idx, *hi, op, V);
                else accept_op(*hi, mx.idx, op, V);
            };
            if (V > x) check_other(V - x, '+', true);
            if (V <= pz_.limit - x) check_other(V + x, '-', false);
            if (x > V) {
                check_other(x - V, '-', true);
                if (x % V == 0) check_other(x / V, '/', true);
            }
            if (x > 1) {
                if (V % x == 0) check_other(V / x, '*', true);
                u64 prod = elf::safe_mul(V, x, pz_.limit);
                if (prod != 0) check_other(prod, '/', false);
            }
            if (!best_steps.empty() && best_c <= 3) {
                out = std::move(best_steps);
                return best_c;
            }
        }

        if (!best_steps.empty() && best_c <= 4) {
            out = std::move(best_steps);
            return best_c;
        }
        if (max_cost <= 3) return -1;

        // Cost 4: (H op B) op B
        const uint32_t B_sz = static_cast<uint32_t>(basis_.size());
        if (allow_hb_b && best_c > 4 && max_cost >= 4) {
            auto check_hb = [&](u64 V2, uint32_t bi2, char op2, bool v2_first) -> bool {
                if (V2 == 0 || V2 > pz_.limit) return false;
                for (uint32_t bi1 = 0; bi1 < B_sz; ++bi1) {
                    u64 b1 = basis_[bi1];
                    auto finish = [&](uint32_t hi, char op1, bool h_first) -> bool {
                        LocalEmitter em(H_vec_, basis_);
                        em.emit_node(hi);
                        if (h_first) em.emit_op(op1, H_vec_[hi].val, b1, V2);
                        else em.emit_op(op1, b1, H_vec_[hi].val, V2);
                        if (v2_first) em.emit_op(op2, V2, basis_[bi2], V);
                        else em.emit_op(op2, basis_[bi2], V2, V);
                        int sz = static_cast<int>(em.steps.size());
                        if (sz < best_c) {
                            best_c = sz;
                            best_steps = std::move(em.steps);
                            return true;
                        }
                        return false;
                    };
                    if (V2 > b1) {
                        if (const uint32_t* hi = find_H(V2 - b1)) {
                            if (finish(*hi, '+', true) && best_c <= 4) return true;
                        }
                    }
                    if (V2 <= pz_.limit - b1) {
                        if (const uint32_t* hi = find_H(V2 + b1)) {
                            if (finish(*hi, '-', true) && best_c <= 4) return true;
                        }
                    }
                    if (b1 > V2) {
                        if (const uint32_t* hi = find_H(b1 - V2)) {
                            if (finish(*hi, '-', false) && best_c <= 4) return true;
                        }
                    }
                    if (b1 > 1) {
                        if (V2 % b1 == 0) {
                            if (const uint32_t* hi = find_H(V2 / b1)) {
                                if (finish(*hi, '*', true) && best_c <= 4) return true;
                            }
                        }
                        u64 prod = elf::safe_mul(V2, b1, pz_.limit);
                        if (prod != 0) {
                            if (const uint32_t* hi = find_H(prod)) {
                                if (finish(*hi, '/', true) && best_c <= 4) return true;
                            }
                        }
                    }
                }
                return false;
            };

            for (uint32_t bi2 = 0; bi2 < B_sz && best_c > 4; ++bi2) {
                u64 b2 = basis_[bi2];
                if (V > b2 && check_hb(V - b2, bi2, '+', true)) break;
                if (V <= pz_.limit - b2 && check_hb(V + b2, bi2, '-', true)) break;
                if (b2 > V && check_hb(b2 - V, bi2, '-', false)) break;
                if (b2 > 1) {
                    if (V % b2 == 0 && check_hb(V / b2, bi2, '*', true)) break;
                    u64 prod = elf::safe_mul(V, b2, pz_.limit);
                    if (prod != 0 && check_hb(prod, bi2, '/', true)) break;
                }
            }
        }

        if (!best_steps.empty() && best_c <= 4) {
            out = std::move(best_steps);
            return best_c;
        }

        // Cost 4..7: M * H +- H
        if (allow_m_x_h && max_cost >= 4) {
            for (const ValIdx& mx : M_sorted_) {
                u64 x = mx.val;
                if (x < 2) continue;
                if (x > V) break;
                if (max_cost == 4 && mx.cost > 0) continue;
                u64 q = V / x;
                auto it = std::lower_bound(
                    H_sorted_.begin(), H_sorted_.end(), q,
                    [](const ValIdx& a, u64 val) { return a.val < val; });

                for (int delta = -1; delta <= 1; ++delta) {
                    if (delta < 0 && it == H_sorted_.begin()) continue;
                    auto jt = it + delta;
                    if (jt < H_sorted_.begin() || jt >= H_sorted_.end()) continue;
                    u64 y = jt->val;
                    if (y < 2) continue;
                    u128 p128 = static_cast<u128>(x) * y;
                    if (p128 > pz_.limit) continue;
                    u64 prod = static_cast<u64>(p128);
                    int base_c = static_cast<int>(mx.cost) + static_cast<int>(jt->cost) + 1;
                    if (prod == V) {
                        if (base_c <= best_c) accept_op(mx.idx, jt->idx, '*', V);
                        continue;
                    }
                    if (base_c + 1 > best_c) continue;
                    u64 r = prod > V ? prod - V : V - prod;
                    const uint32_t* ri = find_H(r);
                    if (!ri) continue;
                    int tot = base_c + static_cast<int>(H_vec_[*ri].cost) + 1;
                    if (tot > best_c) continue;
                    LocalEmitter em(H_vec_, basis_);
                    em.emit_node(mx.idx);
                    em.emit_node(jt->idx);
                    em.emit_op('*', x, y, prod);
                    em.emit_node(*ri);
                    em.emit_op(prod > V ? '-' : '+', prod, r, V);
                    int sz = static_cast<int>(em.steps.size());
                    if (sz < best_c) {
                        best_c = sz;
                        best_steps = std::move(em.steps);
                    }
                }
            }
        }

        if (max_cost <= 4) {
            if (!best_steps.empty() && best_c <= max_cost) {
                out = std::move(best_steps);
                return best_c;
            }
            return -1;
        }

        // Cost 5: H +- H two-pointer
        if (allow_two_sum && best_c > 5 && max_cost >= 5 && !H_sorted_.empty()) {
            const size_t Hn = H_sorted_.size();
            if (V <= H_max_val_ * 2ULL) {
                size_t i = 0;
                size_t j = Hn - 1;
                while (i <= j) {
                    u64 sum = H_sorted_[i].val + H_sorted_[j].val;
                    if (sum == V) {
                        accept_op(H_sorted_[i].idx, H_sorted_[j].idx, '+', V);
                        if (best_c <= 5) break;
                        ++i;
                        if (j > 0) --j;
                    } else if (sum < V) {
                        ++i;
                    } else {
                        if (j == 0) break;
                        --j;
                    }
                }
            }
            if (best_c > 5 && V < H_max_val_) {
                size_t i = 0;
                size_t j = 0;
                while (j < Hn && i < Hn) {
                    u64 hi_v = H_sorted_[j].val;
                    u64 lo_v = H_sorted_[i].val;
                    if (hi_v <= lo_v) {
                        ++j;
                        continue;
                    }
                    u64 diff = hi_v - lo_v;
                    if (diff == V) {
                        accept_op(H_sorted_[j].idx, H_sorted_[i].idx, '-', V);
                        if (best_c <= 5) break;
                        ++i;
                        ++j;
                    } else if (diff < V) {
                        ++j;
                    } else {
                        ++i;
                    }
                }
            }
        }

        if (!best_steps.empty() && best_c <= max_cost) {
            out = std::move(best_steps);
            return best_c;
        }
        return -1;
    }

    int solve_moh_op_b(u64 V, int max_cost, std::vector<Step>& out) const {
        out.clear();
        if (max_cost < 5 || V == 0 || V > pz_.limit) return -1;
        const uint32_t B_sz = static_cast<uint32_t>(basis_.size());
        int best_c = max_cost + 1;
        std::vector<Step> best_steps;

        auto try_inner = [&](u64 V2, uint32_t bi, char op2, bool v2_first) {
            if (V2 == 0 || V2 > pz_.limit) return;
            for (const ValIdx& mx : M_sorted_) {
                if (mx.cost == 0) continue;
                u64 x = mx.val;
                auto finish = [&](u64 y, char op1, bool x_first) {
                    const uint32_t* hi = find_H(y);
                    if (!hi) return;
                    int est = static_cast<int>(mx.cost) + static_cast<int>(H_vec_[*hi].cost) + 2;
                    if (est >= best_c + 1) return;
                    LocalEmitter em(H_vec_, basis_);
                    em.emit_node(mx.idx);
                    em.emit_node(*hi);
                    if (x_first) em.emit_op(op1, x, y, V2);
                    else em.emit_op(op1, y, x, V2);
                    if (v2_first) em.emit_op(op2, V2, basis_[bi], V);
                    else em.emit_op(op2, basis_[bi], V2, V);
                    int sz = static_cast<int>(em.steps.size());
                    if (sz < best_c) {
                        best_c = sz;
                        best_steps = std::move(em.steps);
                    }
                };
                if (V2 > x) finish(V2 - x, '+', true);
                if (V2 <= pz_.limit - x) finish(V2 + x, '-', false);
                if (x > V2) {
                    finish(x - V2, '-', true);
                    if (x % V2 == 0) finish(x / V2, '/', true);
                }
                if (x > 1) {
                    if (V2 % x == 0) finish(V2 / x, '*', true);
                    u64 prod = elf::safe_mul(V2, x, pz_.limit);
                    if (prod != 0) finish(prod, '/', false);
                }
                if (best_c <= 4) return;
            }
        };

        for (uint32_t bi = 0; bi < B_sz && best_c > 5; ++bi) {
            u64 b = basis_[bi];
            if (V > b) try_inner(V - b, bi, '+', true);
            if (V <= pz_.limit - b) try_inner(V + b, bi, '-', true);
            if (b > V) try_inner(b - V, bi, '-', false);
            if (b > 1) {
                if (V % b == 0) try_inner(V / b, bi, '*', true);
                u64 prod = elf::safe_mul(V, b, pz_.limit);
                if (prod != 0) try_inner(prod, bi, '/', true);
            }
        }

        if (!best_steps.empty() && best_c <= max_cost) {
            out = std::move(best_steps);
            return best_c;
        }
        return -1;
    }

    // Linear-Combo Two-Pointer Scan: V = (b * x +- y) +- r
    template <typename WrapFn>
    void scan_linear_bxy(u64 V, int& budget, int max_b_count, bool collect_res,
                         double soft_dl, WrapFn&& wrap_and_update,
                         int& best_cost_ref, int extra_cost) const {
        if (budget < 4 || H_sorted_.empty()) return;

        std::vector<std::pair<u64, uint32_t>> b_list;
        b_list.reserve(basis_.size());
        for (uint32_t bi = 0; bi < basis_.size(); ++bi) {
            u64 b = basis_[bi];
            if (b >= 2 && b <= V) b_list.emplace_back(b, bi);
        }
        std::sort(b_list.begin(), b_list.end());

        std::vector<std::pair<u64, uint32_t>> picked_b;
        if (static_cast<int>(b_list.size()) <= max_b_count) {
            picked_b = std::move(b_list);
        } else {
            for (int k = 0; k < max_b_count; ++k) {
                size_t idx = (b_list.size() - 1) * static_cast<size_t>(k) /
                             static_cast<size_t>(std::max(1, max_b_count - 1));
                picked_b.push_back(b_list[idx]);
            }
        }

        const size_t Hn = H_sorted_.size();
        const size_t max_keep_lin = collect_res ? 240 : 0;
        std::array<std::vector<LinearResCand>, 8> lin_buckets;
        std::array<u64, 8> lin_thresh;
        lin_thresh.fill(std::numeric_limits<u64>::max());

        auto push_lin = [&](u64 r, uint32_t bi, uint32_t xi, uint32_t yi,
                            uint8_t kind, char r_sign, uint8_t oc) {
            if (!collect_res || oc > 7 || oc + 3 > budget) return;
            if (r >= lin_thresh[oc] && lin_buckets[oc].size() >= max_keep_lin * 2) return;
            lin_buckets[oc].push_back(LinearResCand{r, bi, xi, yi, kind, r_sign, oc});
            if (lin_buckets[oc].size() >= max_keep_lin * 4) {
                auto& bk = lin_buckets[oc];
                std::nth_element(bk.begin(), bk.begin() + max_keep_lin, bk.end(),
                                 [](const LinearResCand& a, const LinearResCand& c) {
                                     return a.r < c.r;
                                 });
                bk.resize(max_keep_lin);
                u64 mx = 0;
                for (const auto& e : bk) mx = std::max(mx, e.r);
                lin_thresh[oc] = mx;
            }
        };

        auto emit_bxy_r = [&](uint32_t bi, uint32_t xi, uint32_t yi,
                              uint8_t kind, char r_sign, uint32_t ri) {
            LocalEmitter em(H_vec_, basis_);
            em.emit_node(xi);
            em.emit_node(yi);
            u64 b = basis_[bi];
            u64 x = H_vec_[xi].val;
            u64 y = H_vec_[yi].val;
            u64 bx = b * x;
            em.emit_op('*', b, x, bx);
            u64 s = 0;
            if (kind == 0) {
                s = bx + y;
                em.emit_op('+', bx, y, s);
            } else if (kind == 1) {
                s = bx - y;
                em.emit_op('-', bx, y, s);
            } else {
                s = y - bx;
                em.emit_op('-', y, bx, s);
            }
            if (ri != UINT32_MAX) {
                em.emit_node(ri);
                em.emit_op(r_sign, s, H_vec_[ri].val, V);
            }
            wrap_and_update(em.steps);
            budget = best_cost_ref - 1 - extra_cost;
        };

        for (size_t bp = 0; bp < picked_b.size(); ++bp) {
            if (budget < 4 || time_up()) break;
            if (bp >= 4 && timer_.elapsed_sec() >= soft_dl) break;
            u64 b = picked_b[bp].first;
            uint32_t bi = picked_b[bp].second;
            u64 max_x_lim = pz_.limit / b;
            u64 max_x_v = (V - 1) / b;

            // Case 0: b * x + y ~= V
            {
                size_t j = Hn - 1;
                for (size_t i = 0; i < Hn; ++i) {
                    u64 x = H_sorted_[i].val;
                    if (x > max_x_v) break;
                    u64 bx = b * x;
                    u64 need = V - bx;
                    uint8_t cx = H_sorted_[i].cost;
                    while (j > 0 && H_sorted_[j].val > need) --j;

                    {
                        u64 y = H_sorted_[j].val;
                        if (y <= need) {
                            u64 r = need - y;
                            uint8_t cy = H_sorted_[j].cost;
                            if (r == 0) {
                                if (cx + cy + 2 <= budget) {
                                    emit_bxy_r(bi, H_sorted_[i].idx, H_sorted_[j].idx, 0, '+',
                                               UINT32_MAX);
                                }
                            } else {
                                uint8_t oc = static_cast<uint8_t>(cx + cy + 3);
                                if (oc <= budget) {
                                    if (const uint32_t* ri = find_H(r)) {
                                        if (oc + H_vec_[*ri].cost <= budget + 1) {
                                            emit_bxy_r(bi, H_sorted_[i].idx, H_sorted_[j].idx, 0,
                                                       '+', *ri);
                                        }
                                    } else if (oc + 3 <= budget) {
                                        push_lin(r, bi, H_sorted_[i].idx, H_sorted_[j].idx, 0, '+',
                                                 oc);
                                    }
                                }
                            }
                        }
                    }
                    if (j + 1 < Hn) {
                        u64 y = H_sorted_[j + 1].val;
                        if (bx <= pz_.limit - y && bx + y > V) {
                            u64 r = (bx + y) - V;
                            uint8_t cy = H_sorted_[j + 1].cost;
                            uint8_t oc = static_cast<uint8_t>(cx + cy + 3);
                            if (oc <= budget) {
                                if (const uint32_t* ri = find_H(r)) {
                                    if (oc + H_vec_[*ri].cost <= budget + 1) {
                                        emit_bxy_r(bi, H_sorted_[i].idx, H_sorted_[j + 1].idx, 0,
                                                   '-', *ri);
                                    }
                                } else if (oc + 3 <= budget) {
                                    push_lin(r, bi, H_sorted_[i].idx, H_sorted_[j + 1].idx, 0, '-',
                                             oc);
                                }
                            }
                        }
                    }
                }
            }

            // Case 1: b * x - y ~= V
            if (max_x_lim > max_x_v && budget >= 4) {
                auto it_start = std::upper_bound(
                    H_sorted_.begin(), H_sorted_.end(), max_x_v,
                    [](u64 val, const ValIdx& a) { return val < a.val; });
                size_t i_start = static_cast<size_t>(it_start - H_sorted_.begin());
                size_t j = 0;
                for (size_t i = i_start; i < Hn; ++i) {
                    u64 x = H_sorted_[i].val;
                    if (x > max_x_lim) break;
                    u64 bx = b * x;
                    if (bx <= V) continue;
                    u64 need = bx - V;
                    uint8_t cx = H_sorted_[i].cost;
                    while (j + 1 < Hn && H_sorted_[j + 1].val <= need) ++j;

                    if (H_sorted_[j].val <= need) {
                        u64 y = H_sorted_[j].val;
                        u64 r = need - y;
                        uint8_t cy = H_sorted_[j].cost;
                        if (r == 0) {
                            if (cx + cy + 2 <= budget) {
                                emit_bxy_r(bi, H_sorted_[i].idx, H_sorted_[j].idx, 1, '-',
                                           UINT32_MAX);
                            }
                        } else {
                            uint8_t oc = static_cast<uint8_t>(cx + cy + 3);
                            if (oc <= budget) {
                                if (const uint32_t* ri = find_H(r)) {
                                    if (oc + H_vec_[*ri].cost <= budget + 1) {
                                        emit_bxy_r(bi, H_sorted_[i].idx, H_sorted_[j].idx, 1, '-',
                                                   *ri);
                                    }
                                } else if (oc + 3 <= budget) {
                                    push_lin(r, bi, H_sorted_[i].idx, H_sorted_[j].idx, 1, '-', oc);
                                }
                            }
                        }
                    }
                    if (j + 1 < Hn && H_sorted_[j + 1].val < bx && H_sorted_[j + 1].val > need) {
                        u64 y = H_sorted_[j + 1].val;
                        u64 r = y - need;
                        uint8_t cy = H_sorted_[j + 1].cost;
                        uint8_t oc = static_cast<uint8_t>(cx + cy + 3);
                        if (oc <= budget) {
                            if (const uint32_t* ri = find_H(r)) {
                                if (oc + H_vec_[*ri].cost <= budget + 1) {
                                    emit_bxy_r(bi, H_sorted_[i].idx, H_sorted_[j + 1].idx, 1, '+',
                                               *ri);
                                }
                            } else if (oc + 3 <= budget) {
                                push_lin(r, bi, H_sorted_[i].idx, H_sorted_[j + 1].idx, 1, '+', oc);
                            }
                        }
                    }
                }
            }
        }

        // Solve top collected residuals from b*x +- y via ultra-fast solve_c3_hb
        if (collect_res && budget >= 7 && !time_up()) {
            std::vector<LinearResCand> top_lin;
            for (int oc = 3; oc <= 7; ++oc) {
                auto& bk = lin_buckets[oc];
                if (bk.empty() || oc + 3 > budget) continue;
                std::sort(bk.begin(), bk.end(),
                          [](const LinearResCand& a, const LinearResCand& c) { return a.r < c.r; });
                bk.erase(std::unique(bk.begin(), bk.end(),
                                     [](const LinearResCand& a, const LinearResCand& c) {
                                         return a.r == c.r;
                                     }),
                         bk.end());
                if (bk.size() > max_keep_lin) bk.resize(max_keep_lin);
                for (const auto& e : bk) top_lin.push_back(e);
            }
            std::sort(top_lin.begin(), top_lin.end(),
                      [](const LinearResCand& a, const LinearResCand& c) {
                          return (a.oc + 0.25 * log2_approx(a.r)) <
                                 (c.oc + 0.25 * log2_approx(c.r));
                      });

            for (size_t k = 0; k < top_lin.size() && k < 600; ++k) {
                if ((k & 31u) == 0 && (time_up() || timer_.elapsed_sec() >= soft_dl)) break;
                const auto& lc = top_lin[k];
                int rem_b = budget - static_cast<int>(lc.oc);
                if (rem_b < 3) continue;
                std::vector<Step> r_steps;
                int rc = solve_c3_hb(lc.r, std::min(rem_b, 3), r_steps);
                if (rc < 0 && rem_b >= 4 && k < 32 && timer_.elapsed_sec() < soft_dl) {
                    rc = solve_sub(lc.r, std::min(rem_b, 4), false, false, r_steps);
                }
                if (rc >= 0) {
                    LocalEmitter em(H_vec_, basis_);
                    em.emit_node(lc.x_idx);
                    em.emit_node(lc.y_idx);
                    u64 b = basis_[lc.b_idx];
                    u64 x = H_vec_[lc.x_idx].val;
                    u64 y = H_vec_[lc.y_idx].val;
                    u64 bx = b * x;
                    em.emit_op('*', b, x, bx);
                    u64 s = (lc.kind == 0) ? (bx + y) : (lc.kind == 1 ? (bx - y) : (y - bx));
                    if (lc.kind == 0) em.emit_op('+', bx, y, s);
                    else if (lc.kind == 1) em.emit_op('-', bx, y, s);
                    else em.emit_op('-', y, bx, s);
                    em.append_steps(r_steps);
                    em.emit_op(lc.r_sign, s, lc.r, V);
                    wrap_and_update(em.steps);
                    budget = best_cost_ref - 1 - extra_cost;
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Full Target Solver (with strict per-target soft deadline)
    // -------------------------------------------------------------------------
    std::vector<Step> solve_target(u64 target, double target_dl, bool is_first_restart) const {
        std::vector<Step> best_steps;
        int best_cost = std::numeric_limits<int>::max();

        auto update_best = [&](std::vector<Step>&& cand) {
            if (!cand.empty() && static_cast<int>(cand.size()) < best_cost) {
                best_cost = static_cast<int>(cand.size());
                best_steps = std::move(cand);
            }
        };

        // 1. Direct fast check on target itself (cost <= 5)
        {
            std::vector<Step> sub;
            int c = solve_sub(target, 5, true, true, sub);
            if (c >= 0) update_best(std::move(sub));
            if (best_cost <= 3) return best_steps;
        }

        // 2. Multiplier lottery list: d = 1, then d in B, then small d in L1
        struct MultCand {
            u64 d;
            uint32_t d_idx;
            int extra;
        };
        std::vector<MultCand> mults;
        mults.push_back(MultCand{1, UINT32_MAX, 0});

        u64 max_d = pz_.limit / target;
        if (max_d >= 2) {
            std::vector<MultCand> b_mults;
            for (uint32_t bi = 0; bi < basis_.size(); ++bi) {
                u64 d = basis_[bi];
                if (d >= 2 && d <= max_d) {
                    b_mults.push_back(MultCand{d, bi, 1});
                }
            }
            std::sort(b_mults.begin(), b_mults.end(),
                      [](const MultCand& a, const MultCand& b) { return a.d < b.d; });
            for (const auto& mc : b_mults) {
                if (mults.size() >= 14) break;
                mults.push_back(mc);
            }
            if (mults.size() < 8) {
                for (const ValIdx& mx : M_sorted_) {
                    if (mx.val < 2 || mx.cost == 0) continue;
                    if (mx.val > max_d) break;
                    mults.push_back(MultCand{mx.val, mx.idx, 2});
                    if (mults.size() >= 10) break;
                }
            }
        }

        // Quick pass: cost <= 4 on all multipliers (skip slow (H op B) op B when V > 10^12)
        for (size_t mi = 1; mi < mults.size(); ++mi) {
            const auto& mc = mults[mi];
            int budget = best_cost - 1 - mc.extra;
            if (budget <= 0) continue;
            u64 V = target * mc.d;
            std::vector<Step> sub;
            int c = solve_sub(V, std::min(budget, 4), false, true, sub, V <= 1000000000000ULL);
            if (c >= 0) {
                LocalEmitter em(H_vec_, basis_);
                em.append_steps(sub);
                em.emit_node(mc.d_idx);
                em.emit_op('/', V, mc.d, target);
                update_best(std::move(em.steps));
            }
        }
        if (best_cost <= 4) return best_steps;

        // Only run O(|B|*|M|) solve_moh_op_b when V <= 10^11 (tiny hit rate above 10^11!)
        if (best_cost > 5 && target <= 100000000000ULL && !time_up()) {
            std::vector<Step> sub;
            if (solve_moh_op_b(target, 5, sub) >= 0) {
                update_best(std::move(sub));
            }
            if (best_cost <= 5) return best_steps;
        }

        // 3. Unified Two-Pointer H x H + Linear b*x +- y + Layered Residuals + Beam
        const size_t Hn = H_sorted_.size();

        for (size_t mi = 0; mi < mults.size(); ++mi) {
            if (time_up()) break;
            if (mi > 0 && timer_.elapsed_sec() >= target_dl) break;

            const auto& mc = mults[mi];
            int budget = best_cost - 1 - mc.extra;
            if (budget < 4) continue;

            u64 V = target * mc.d;

            auto wrap_and_update = [&](const std::vector<Step>& v_plan) {
                if (mc.d == 1) {
                    if (static_cast<int>(v_plan.size()) < best_cost) {
                        best_cost = static_cast<int>(v_plan.size());
                        best_steps = v_plan;
                    }
                } else {
                    LocalEmitter em(H_vec_, basis_);
                    em.append_steps(v_plan);
                    em.emit_node(mc.d_idx);
                    em.emit_op('/', V, mc.d, target);
                    update_best(std::move(em.steps));
                }
            };

            if (mi > 0 && budget >= 5) {
                std::vector<Step> sub;
                int c = solve_sub(V, std::min(budget, 5), false, true, sub);
                if (c >= 0) {
                    wrap_and_update(sub);
                    budget = best_cost - 1 - mc.extra;
                    if (budget < 4) continue;
                }
            }

            // 3a. Unified two-pointer H x H scan + quotient beam hook
            std::array<std::vector<ResidualCand>, 7> res_buckets;
            const size_t max_keep = (mi == 0) ? 400 : 100;
            std::array<u64, 7> res_thresh;
            res_thresh.fill(std::numeric_limits<u64>::max());

            std::vector<BeamCand> beam_cands;
            beam_cands.reserve(256);

            auto push_res = [&](u64 r, uint32_t ui, uint32_t vi, char sign, uint8_t oc) {
                if (oc > 6 || oc + 3 > budget) return;
                if (r >= res_thresh[oc] && res_buckets[oc].size() >= max_keep * 2) return;
                res_buckets[oc].push_back(ResidualCand{r, ui, vi, sign, oc});
                if (res_buckets[oc].size() >= max_keep * 4) {
                    auto& b = res_buckets[oc];
                    std::nth_element(b.begin(), b.begin() + max_keep, b.end(),
                                     [](const ResidualCand& a, const ResidualCand& c) {
                                         return a.r < c.r;
                                     });
                    b.resize(max_keep);
                    u64 mx = 0;
                    for (const auto& e : b) mx = std::max(mx, e.r);
                    res_thresh[oc] = mx;
                }
            };

            auto emit_uvr = [&](uint32_t ui, uint32_t vi, u64 prod, char sign, uint32_t ri) {
                LocalEmitter em(H_vec_, basis_);
                em.emit_node(ui);
                em.emit_node(vi);
                em.emit_op('*', H_vec_[ui].val, H_vec_[vi].val, prod);
                if (ri != UINT32_MAX) {
                    em.emit_node(ri);
                    em.emit_op(sign, prod, H_vec_[ri].val, V);
                }
                wrap_and_update(em.steps);
                budget = best_cost - 1 - mc.extra;
            };

            u64 root_v = elf::isqrt64(V);
            size_t j = Hn > 0 ? Hn - 1 : 0;

            for (size_t i = 0; i < Hn; ++i) {
                if ((i & 8191u) == 0 && time_up()) break;
                u64 u = H_sorted_[i].val;
                if (u < 2) continue;
                if (u > root_v) break;
                uint8_t cu = H_sorted_[i].cost;
                uint32_t ui = H_sorted_[i].idx;

                u64 q = V / u;
                u64 rem = V - q * u;

                if (rem == 0) {
                    uint8_t oc = static_cast<uint8_t>(cu + 1);
                    if (oc < budget) {
                        beam_cands.push_back(BeamCand{q, ui, UINT32_MAX, '*', oc,
                                                      oc + 0.30 * log2_approx(q)});
                    }
                } else if (cu <= 1) {
                    const uint32_t* ri = find_H(rem);
                    if (ri) {
                        uint8_t oc = static_cast<uint8_t>(cu + H_vec_[*ri].cost + 2);
                        if (oc < budget && q >= 2) {
                            beam_cands.push_back(BeamCand{q, ui, *ri, '+', oc,
                                                          oc + 0.30 * log2_approx(q)});
                        }
                    }
                    u64 rem2 = u - rem;
                    if (static_cast<u128>(q + 1) * u <= pz_.limit) {
                        const uint32_t* ri2 = find_H(rem2);
                        if (ri2) {
                            uint8_t oc = static_cast<uint8_t>(cu + H_vec_[*ri2].cost + 2);
                            if (oc < budget) {
                                beam_cands.push_back(BeamCand{q + 1, ui, *ri2, '-', oc,
                                                              oc + 0.30 * log2_approx(q + 1)});
                            }
                        }
                    }
                }

                while (j > 0 && H_sorted_[j].val > q) --j;

                {
                    u64 v = H_sorted_[j].val;
                    if (v >= 2 && v <= q) {
                        u64 prod = u * v;
                        u64 r = V - prod;
                        uint8_t cv = H_sorted_[j].cost;
                        if (r == 0) {
                            if (cu + cv + 1 <= budget) {
                                emit_uvr(ui, H_sorted_[j].idx, prod, '+', UINT32_MAX);
                            }
                        } else {
                            uint8_t oc = static_cast<uint8_t>(cu + cv + 2);
                            if (oc <= budget) {
                                if (const uint32_t* ri = find_H(r)) {
                                    if (oc + H_vec_[*ri].cost <= budget + 1) {
                                        emit_uvr(ui, H_sorted_[j].idx, prod, '+', *ri);
                                    }
                                } else if (oc + 3 <= budget) {
                                    push_res(r, ui, H_sorted_[j].idx, '+', oc);
                                }
                            }
                        }
                    }
                }

                if (j + 1 < Hn) {
                    u64 v = H_sorted_[j + 1].val;
                    u128 p128 = static_cast<u128>(u) * v;
                    if (v >= 2 && p128 <= pz_.limit && p128 > V) {
                        u64 prod = static_cast<u64>(p128);
                        u64 r = prod - V;
                        uint8_t cv = H_sorted_[j + 1].cost;
                        uint8_t oc = static_cast<uint8_t>(cu + cv + 2);
                        if (oc <= budget) {
                            if (const uint32_t* ri = find_H(r)) {
                                if (oc + H_vec_[*ri].cost <= budget + 1) {
                                    emit_uvr(ui, H_sorted_[j + 1].idx, prod, '-', *ri);
                                }
                            } else if (oc + 3 <= budget) {
                                push_res(r, ui, H_sorted_[j + 1].idx, '-', oc);
                            }
                        }
                    }
                }
            }

            // Scan M_sorted with u > root_v for quotient-beam
            for (const ValIdx& mx : M_sorted_) {
                u64 u = mx.val;
                if (u <= root_v) continue;
                if (u >= V) break;
                u64 q = V / u;
                u64 rem = V - q * u;
                if (rem == 0) {
                    uint8_t oc = static_cast<uint8_t>(mx.cost + 1);
                    if (oc < budget) {
                        beam_cands.push_back(BeamCand{q, mx.idx, UINT32_MAX, '*', oc,
                                                      oc + 0.30 * log2_approx(q)});
                    }
                } else {
                    if (const uint32_t* ri = find_H(rem)) {
                        uint8_t oc = static_cast<uint8_t>(mx.cost + H_vec_[*ri].cost + 2);
                        if (oc < budget && q >= 2) {
                            beam_cands.push_back(BeamCand{q, mx.idx, *ri, '+', oc,
                                                          oc + 0.30 * log2_approx(q)});
                        }
                    }
                    u64 rem2 = u - rem;
                    if (static_cast<u128>(q + 1) * u <= pz_.limit) {
                        if (const uint32_t* ri2 = find_H(rem2)) {
                            uint8_t oc = static_cast<uint8_t>(mx.cost + H_vec_[*ri2].cost + 2);
                            if (oc < budget) {
                                beam_cands.push_back(BeamCand{q + 1, mx.idx, *ri2, '-', oc,
                                                              oc + 0.30 * log2_approx(q + 1)});
                            }
                        }
                    }
                }
            }

            // 3b. Layered residual solver on H x H residuals
            std::vector<ResidualCand> all_res;
            for (int oc = 2; oc <= 6; ++oc) {
                auto& b = res_buckets[oc];
                if (b.empty() || oc + 3 > budget) continue;
                std::sort(b.begin(), b.end(),
                          [](const ResidualCand& a, const ResidualCand& c) { return a.r < c.r; });
                b.erase(std::unique(b.begin(), b.end(),
                                    [](const ResidualCand& a, const ResidualCand& c) {
                                        return a.r == c.r;
                                    }),
                        b.end());
                if (b.size() > max_keep) b.resize(max_keep);
                for (const auto& rc : b) all_res.push_back(rc);
            }

            std::sort(all_res.begin(), all_res.end(),
                      [](const ResidualCand& a, const ResidualCand& b) {
                          return (a.oc + 0.22 * log2_approx(a.r)) <
                                 (b.oc + 0.22 * log2_approx(b.r));
                      });

            auto emit_res_sol = [&](const ResidualCand& rc, const std::vector<Step>& r_steps) {
                u64 u = H_vec_[rc.u_idx].val;
                u64 v = H_vec_[rc.v_idx].val;
                u64 prod = u * v;
                LocalEmitter em(H_vec_, basis_);
                em.emit_node(rc.u_idx);
                em.emit_node(rc.v_idx);
                em.emit_op('*', u, v, prod);
                em.append_steps(r_steps);
                em.emit_op(rc.sign, prod, rc.r, V);
                wrap_and_update(em.steps);
                budget = best_cost - 1 - mc.extra;
            };

            // Layer 1 (ultra-fast ~0.5 us each): solve_c3_hb on top residuals
            const size_t c3_lim = (mi == 0) ? 1200 : 280;
            for (size_t k = 0; k < all_res.size() && k < c3_lim; ++k) {
                const auto& rc = all_res[k];
                if (static_cast<int>(rc.oc) + 3 > budget) continue;
                std::vector<Step> r_steps;
                if (solve_c3_hb(rc.r, 3, r_steps) >= 0) {
                    emit_res_sol(rc, r_steps);
                }
            }

            // Layer 2: M op H & B*H+-B on top residuals (plus (H op B) op B on top 12)
            const size_t c4_lim = (mi == 0) ? 64 : 16;
            for (size_t k = 0; k < all_res.size() && k < c4_lim; ++k) {
                if ((k & 7u) == 0 && (time_up() || (k >= 16 && timer_.elapsed_sec() >= target_dl))) {
                    break;
                }
                const auto& rc = all_res[k];
                int rem_b = budget - static_cast<int>(rc.oc);
                if (rem_b < 4) continue;
                std::vector<Step> r_steps;
                if (solve_sub(rc.r, 4, false, true, r_steps, mi == 0 && k < 12) >= 0) {
                    emit_res_sol(rc, r_steps);
                }
            }

            // 3c. Linear-Combo Two-Pointer Scan: V = (b * x +- y) +- r
            if (!time_up() && budget >= 5 &&
                (mi == 0 || (mi == 1 && is_first_restart && best_cost > 10 &&
                             timer_.elapsed_sec() < target_dl))) {
                int max_b = (mi == 0) ? 14 : 6;
                bool collect_lin = (mi == 0 && budget >= 8);
                scan_linear_bxy(V, budget, max_b, collect_lin, target_dl,
                                wrap_and_update, best_cost, mc.extra);
            }

            // Layer 3 (medium): M * H +- H and H +- H on top residuals (mi == 0, time-budgeted)
            if (mi == 0 && !time_up() && timer_.elapsed_sec() < target_dl && budget >= 8) {
                const size_t c5_lim = 14;
                size_t two_sum_tries = 0;
                for (size_t k = 0; k < all_res.size() && k < c5_lim; ++k) {
                    if (time_up() || timer_.elapsed_sec() >= target_dl) break;
                    const auto& rc = all_res[k];
                    int rem_b = budget - static_cast<int>(rc.oc);
                    if (rem_b < 5) continue;
                    bool try_2sum = (two_sum_tries < 6 && rc.r <= H_max_val_ * 2ULL);
                    if (try_2sum) ++two_sum_tries;
                    std::vector<Step> r_steps;
                    if (solve_sub(rc.r, std::min(rem_b, 7), try_2sum, true, r_steps, k < 4) >= 0) {
                        emit_res_sol(rc, r_steps);
                    }
                }
            }

            // 3d. Divisor-Remainder Beam Search (depth 1..3)
            if (!time_up() && !beam_cands.empty() && budget >= 4 &&
                (mi == 0 || timer_.elapsed_sec() < target_dl)) {
                int bw = (mi == 0) ? opt_.beam : std::min(opt_.beam, 16);
                double beam_dl = (mi == 0 && best_steps.empty())
                                     ? std::min(deadline_, std::max(target_dl, timer_.elapsed_sec() + 0.010))
                                     : target_dl;
                run_beam_search(V, budget, beam_cands, bw, beam_dl, wrap_and_update);
                budget = best_cost - 1 - mc.extra;
            }

            // Layer 4: 2-level H x H on top 3 residuals if still >= 10
            if (mi == 0 && !time_up() && timer_.elapsed_sec() < target_dl &&
                budget >= 10 && !all_res.empty()) {
                size_t deep_cnt = std::min<size_t>(all_res.size(), 3);
                for (size_t k = 0; k < deep_cnt; ++k) {
                    if (time_up() || timer_.elapsed_sec() >= target_dl) break;
                    const auto& rc = all_res[k];
                    int rem_b = budget - static_cast<int>(rc.oc);
                    if (rem_b < 6) continue;
                    std::vector<Step> r_steps;
                    if (solve_one_level_hxh(rc.r, rem_b, r_steps) >= 0) {
                        emit_res_sol(rc, r_steps);
                    }
                }
            }
        }

        // 4. Guaranteed Fallback
        if (best_steps.empty()) {
            LocalEmitter em(H_vec_, basis_);
            fallback_build(target, em);
            update_best(std::move(em.steps));
        }

        return best_steps;
    }

    int solve_one_level_hxh(u64 V, int max_cost, std::vector<Step>& out) const {
        out.clear();
        if (V == 0 || V > pz_.limit || max_cost < 3) return -1;
        int best_c = max_cost + 1;
        int c0 = solve_sub(V, std::min(max_cost, 6), true, true, out);
        if (c0 >= 0) best_c = c0;
        if (best_c <= 5) return best_c;

        const size_t Hn = H_sorted_.size();
        u64 root_v = elf::isqrt64(V);
        size_t j = Hn > 0 ? Hn - 1 : 0;

        for (size_t i = 0; i < Hn; ++i) {
            if ((i & 8191u) == 0 && time_up()) break;
            u64 u = H_sorted_[i].val;
            if (u < 2) continue;
            if (u > root_v) break;
            uint8_t cu = H_sorted_[i].cost;
            uint32_t ui = H_sorted_[i].idx;
            u64 q = V / u;
            while (j > 0 && H_sorted_[j].val > q) --j;

            auto check_v = [&](size_t idx_v) {
                u64 v = H_sorted_[idx_v].val;
                if (v < 2) return;
                u128 p128 = static_cast<u128>(u) * v;
                if (p128 > pz_.limit) return;
                u64 prod = static_cast<u64>(p128);
                uint8_t cv = H_sorted_[idx_v].cost;
                if (prod == V) {
                    if (cu + cv + 1 < best_c) {
                        LocalEmitter em(H_vec_, basis_);
                        em.emit_node(ui);
                        em.emit_node(H_sorted_[idx_v].idx);
                        em.emit_op('*', u, v, prod);
                        if (static_cast<int>(em.steps.size()) < best_c) {
                            best_c = static_cast<int>(em.steps.size());
                            out = std::move(em.steps);
                        }
                    }
                    return;
                }
                u64 r = prod > V ? prod - V : V - prod;
                int oc = static_cast<int>(cu) + static_cast<int>(cv) + 2;
                if (oc >= best_c) return;
                if (const uint32_t* ri = find_H(r)) {
                    if (oc + static_cast<int>(H_vec_[*ri].cost) < best_c + 1) {
                        LocalEmitter em(H_vec_, basis_);
                        em.emit_node(ui);
                        em.emit_node(H_sorted_[idx_v].idx);
                        em.emit_op('*', u, v, prod);
                        em.emit_node(*ri);
                        em.emit_op(prod > V ? '-' : '+', prod, r, V);
                        if (static_cast<int>(em.steps.size()) < best_c) {
                            best_c = static_cast<int>(em.steps.size());
                            out = std::move(em.steps);
                        }
                    }
                }
            };

            check_v(j);
            if (j + 1 < Hn) check_v(j + 1);
            if (best_c <= 5) break;
        }
        return (best_c <= max_cost) ? best_c : -1;
    }

    template <typename WrapFn>
    void run_beam_search(u64 V, int& budget, std::vector<BeamCand>& init_cands,
                         int beam_width, double soft_dl, WrapFn&& wrap_and_update) const {
        if (init_cands.empty() || beam_width <= 0) return;

        std::sort(init_cands.begin(), init_cands.end(),
                  [](const BeamCand& a, const BeamCand& b) { return a.score < b.score; });

        std::vector<BeamState> beam;
        beam.reserve(static_cast<size_t>(beam_width));
        FlatMap<uint8_t> seen_q(static_cast<size_t>(beam_width) * 4 + 16);

        for (const auto& bc : init_cands) {
            if (bc.oc >= budget) continue;
            if (!seen_q.insert(bc.q, 1)) continue;
            BeamState st;
            st.q = bc.q;
            st.cost = bc.oc;
            st.score = bc.score;
            st.chain.push_back(BeamState::Link{bc.x_idx, bc.r_idx, bc.sign, V});
            beam.push_back(std::move(st));
            if (static_cast<int>(beam.size()) >= beam_width) break;
        }

        auto emit_beam_solution = [&](const BeamState& st, const std::vector<Step>& q_steps) {
            LocalEmitter em(H_vec_, basis_);
            em.append_steps(q_steps);
            u64 cur = st.q;
            for (size_t k = st.chain.size(); k-- > 0;) {
                const auto& lk = st.chain[k];
                u64 x = H_vec_[lk.x_idx].val;
                em.emit_node(lk.x_idx);
                u64 prod = cur * x;
                em.emit_op('*', cur, x, prod);
                if (lk.sign != '*' && lk.r_idx != UINT32_MAX) {
                    em.emit_node(lk.r_idx);
                    em.emit_op(lk.sign, prod, H_vec_[lk.r_idx].val, lk.parent_val);
                }
                cur = lk.parent_val;
            }
            wrap_and_update(em.steps);
        };

        const int max_depth = (budget >= 10 && dec_d_ >= 14.5) ? 3 : 2;
        for (int depth = 0; depth < max_depth && !beam.empty(); ++depth) {
            std::vector<BeamState> next_beam;
            next_beam.reserve(static_cast<size_t>(beam_width) * 4);
            FlatMap<uint8_t> next_seen(static_cast<size_t>(beam_width) * 8 + 16);

            for (size_t bi = 0; bi < beam.size(); ++bi) {
                if (time_up() || (bi >= 8 && timer_.elapsed_sec() >= soft_dl)) break;
                const BeamState& st = beam[bi];
                int rem_b = budget - static_cast<int>(st.cost);
                if (rem_b < 0) continue;

                std::vector<Step> q_steps;
                int qc = solve_sub(st.q, std::min(rem_b, 5), rem_b >= 5 && bi < 4, rem_b >= 4,
                                   q_steps, bi < 6);
                if (qc >= 0) {
                    emit_beam_solution(st, q_steps);
                    int new_len = static_cast<int>(st.cost) + qc;
                    if (new_len - 1 < budget) budget = new_len - 1;
                    continue;
                }

                if (depth + 1 < max_depth && rem_b >= 4 &&
                    bi < static_cast<size_t>(std::min(beam_width, depth == 0 ? 16 : 10))) {
                    u64 cur_v = st.q;
                    for (const ValIdx& mx : M_sorted_) {
                        u64 x = mx.val;
                        if (x < 2) continue;
                        if (x >= cur_v) break;
                        u64 q2 = cur_v / x;
                        u64 rem = cur_v - q2 * x;
                        if (rem == 0) {
                            uint8_t nc = static_cast<uint8_t>(st.cost + mx.cost + 1);
                            if (nc < budget && next_seen.insert(q2, 1)) {
                                BeamState ns = st;
                                ns.q = q2;
                                ns.cost = nc;
                                ns.score = nc + 0.30 * log2_approx(q2);
                                ns.chain.push_back(BeamState::Link{mx.idx, UINT32_MAX, '*', cur_v});
                                next_beam.push_back(std::move(ns));
                            }
                        } else {
                            if (const uint32_t* ri = find_H(rem)) {
                                uint8_t nc = static_cast<uint8_t>(st.cost + mx.cost + H_vec_[*ri].cost + 2);
                                if (nc < budget && q2 >= 2 && next_seen.insert(q2, 1)) {
                                    BeamState ns = st;
                                    ns.q = q2;
                                    ns.cost = nc;
                                    ns.score = nc + 0.30 * log2_approx(q2);
                                    ns.chain.push_back(BeamState::Link{mx.idx, *ri, '+', cur_v});
                                    next_beam.push_back(std::move(ns));
                                }
                            }
                            u64 rem2 = x - rem;
                            if (static_cast<u128>(q2 + 1) * x <= pz_.limit) {
                                if (const uint32_t* ri2 = find_H(rem2)) {
                                    uint8_t nc = static_cast<uint8_t>(st.cost + mx.cost + H_vec_[*ri2].cost + 2);
                                    if (nc < budget && next_seen.insert(q2 + 1, 1)) {
                                        BeamState ns = st;
                                        ns.q = q2 + 1;
                                        ns.cost = nc;
                                        ns.score = nc + 0.30 * log2_approx(q2 + 1);
                                        ns.chain.push_back(BeamState::Link{mx.idx, *ri2, '-', cur_v});
                                        next_beam.push_back(std::move(ns));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (next_beam.empty()) break;
            std::sort(next_beam.begin(), next_beam.end(),
                      [](const BeamState& a, const BeamState& b) { return a.score < b.score; });
            if (static_cast<int>(next_beam.size()) > beam_width) {
                next_beam.resize(static_cast<size_t>(beam_width));
            }
            beam.swap(next_beam);
        }
    }

    void fallback_build(u64 v, LocalEmitter& em) const {
        if (em.have.contains(v)) return;
        if (const uint32_t* hi = find_H(v)) {
            em.emit_node(*hi);
            return;
        }
        std::vector<Step> sub;
        if (solve_sub(v, 4, false, false, sub, false) >= 0) {
            em.append_steps(sub);
            return;
        }
        // Choose best split a in M_sorted_ near sqrt(v) testing both +r and -(a-r)
        u64 root = elf::isqrt64(v);
        auto it = std::upper_bound(M_sorted_.begin(), M_sorted_.end(), root,
                                   [](u64 val, const ValIdx& a) { return val < a.val; });
        uint32_t best_idx = UINT32_MAX;
        u64 best_a = 0, best_q = 0, best_r = 0;
        bool best_minus = false;
        double best_sc = 1e18;

        auto eval_m = [&](const ValIdx& mv) {
            u64 a = mv.val;
            if (a < 2 || a >= v) return;
            u64 q = v / a, r = v % a;
            if (q == 0) return;
            double sc1 = mv.cost + log2_approx(q) + (r ? log2_approx(r) : -3.0);
            if (sc1 < best_sc) {
                best_sc = sc1;
                best_idx = mv.idx;
                best_a = a;
                best_q = q;
                best_r = r;
                best_minus = false;
            }
            if (r > 0 && static_cast<u128>(a) * (q + 1) <= pz_.limit) {
                u64 r2 = a - r;
                double sc2 = mv.cost + log2_approx(q + 1) + log2_approx(r2);
                if (sc2 < best_sc) {
                    best_sc = sc2;
                    best_idx = mv.idx;
                    best_a = a;
                    best_q = q + 1;
                    best_r = r2;
                    best_minus = true;
                }
            }
        };

        auto it_d = it;
        for (int s = 0; it_d != M_sorted_.begin() && s < 24; ++s) {
            --it_d;
            eval_m(*it_d);
        }
        if (best_a < 2) {
            for (const ValIdx& mv : M_sorted_) {
                if (mv.val >= 2 && mv.val < v) {
                    eval_m(mv);
                    break;
                }
            }
        }
        if (best_a < 2) {
            if (!em.have.contains(1)) em.emit_op('/', basis_[0], basis_[0], 1);
            if (!em.have.contains(2)) em.emit_op('+', 1, 1, 2);
            best_a = 2;
            best_q = v / 2;
            best_r = v % 2;
            best_minus = false;
            best_idx = UINT32_MAX;
        }
        fallback_build(best_q, em);
        if (best_idx != UINT32_MAX) em.emit_node(best_idx);
        u64 prod = best_q * best_a;
        em.emit_op('*', best_q, best_a, prod);
        if (best_r > 0) {
            fallback_build(best_r, em);
            em.emit_op(best_minus ? '-' : '+', prod, best_r, v);
        }
    }

    std::optional<std::vector<Step>> assemble_and_optimize(
        const std::vector<Step>& prefix_steps,
        const std::vector<Step>& safe_prefix,
        const std::vector<std::vector<Step>>& post_target) const {
        auto build_plan_with_promoted = [&](const std::vector<uint32_t>& promoted_nodes) {
            std::vector<Step> full = prefix_steps;
            FlatMap<uint8_t> pinned(basis_.size() * 2 + promoted_nodes.size() * 2 + 16);
            for (u64 v : basis_) pinned.insert(v, 1);

            for (uint32_t idx : promoted_nodes) {
                LocalEmitter em(H_vec_, basis_);
                em.emit_node(idx);
                for (const Step& s : em.steps) {
                    if (pinned.insert(s.r, 1)) full.push_back(s);
                }
            }

            for (const auto& tp : post_target) {
                for (const Step& s : tp) {
                    if (pinned.contains(s.r)) continue;
                    full.push_back(s);
                }
            }
            return full;
        };

        std::vector<Step> raw = build_plan_with_promoted({});
        auto best_verified = elf::verify_and_prune_plan(pz_, raw, opt_.regs);
        if (!best_verified) {
            LocalEmitter em(H_vec_, basis_);
            std::vector<Step> fb = safe_prefix;
            FlatMap<uint8_t> cov(fb.size() * 2 + pz_.starts.size() * 2 + 32);
            for (u64 s : pz_.starts) cov.insert(s, 1);
            for (const Step& s : fb) cov.insert(s.r, 1);
            for (u64 t : pz_.missing_targets()) {
                if (cov.contains(t)) continue;
                em.reset(basis_);
                fallback_build(t, em);
                for (const Step& s : em.steps) {
                    fb.push_back(s);
                    cov.insert(s.r, 1);
                }
            }
            return elf::verify_and_prune_plan(pz_, fb, opt_.regs);
        }

        // Cross-target scratch promotion
        FlatMap<uint32_t> freq(1024);
        FlatMap<uint8_t> in_b(basis_.size() * 2 + 8);
        for (u64 v : basis_) in_b.insert(v, 1);

        for (const auto& tp : post_target) {
            FlatMap<uint8_t> seen_in_t(tp.size() * 2 + 8);
            for (const Step& s : tp) {
                if (in_b.contains(s.r) || !seen_in_t.insert(s.r, 1)) continue;
                if (const uint32_t* mi = M_map_.find(s.r)) {
                    if (uint32_t* f = freq.find(*mi)) ++(*f);
                    else freq.insert(*mi, 1);
                }
            }
        }

        std::vector<std::pair<uint32_t, uint32_t>> hot;
        freq.for_each([&](u64 idx_key, uint32_t cnt) {
            if (cnt >= 2) hot.emplace_back(cnt, static_cast<uint32_t>(idx_key));
        });
        std::sort(hot.begin(), hot.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        if (!hot.empty()) {
            std::vector<uint32_t> promoted;
            int max_prom = std::max(0, (opt_.regs > 0 ? opt_.regs - 5 : 43) -
                                           static_cast<int>(basis_.size()));
            for (size_t k = 0;
                 k < hot.size() && static_cast<int>(promoted.size()) < max_prom && !time_up();
                 ++k) {
                promoted.push_back(hot[k].second);
                std::vector<Step> cand = build_plan_with_promoted(promoted);
                auto verified = elf::verify_and_prune_plan(pz_, cand, opt_.regs);
                if (verified && verified->size() < best_verified->size()) {
                    best_verified = std::move(verified);
                } else {
                    promoted.pop_back();
                }
            }
        }

        return best_verified;
    }
};

// Peak-2-register binary double-and-add fallback (guaranteed valid for any regs >= 2).
std::vector<Step> binary_fallback(const Puzzle& pz) {
    std::vector<Step> plan;
    auto missing = pz.missing_targets();
    if (missing.empty() || pz.starts.empty()) return plan;
    bool have_one = false;
    for (u64 s : pz.starts) {
        if (s == 1) {
            have_one = true;
            break;
        }
    }
    if (!have_one) {
        plan.push_back(Step{'/', pz.starts[0], pz.starts[0], 1});
    }
    for (u64 t : missing) {
        if (t == 1) continue;
        int top = 63 - __builtin_clzll(t);
        u64 acc = 1;
        for (int b = top - 1; b >= 0; --b) {
            plan.push_back(Step{'+', acc, acc, acc + acc});
            acc += acc;
            if ((t >> b) & 1ULL) {
                plan.push_back(Step{'+', acc, 1, acc + 1});
                ++acc;
            }
        }
    }
    return plan;
}

Phase1Config make_p1_config(int r, u64 seed) {
    Phase1Config cfg;
    cfg.seed = seed;
    if (r == 0) {
        cfg.r_cap = 20000;
        cfg.d_cap = 10000;
        cfg.max_q = 500000;
        cfg.min_hub_deg = 2;
        cfg.min_savings = 1.5;
        cfg.add_target_links = true;
        cfg.strict_hub_first = true;
        cfg.b_full = 3;
        cfg.b_construct = 5;
        cfg.work_limit = 3.5e6;
        cfg.randomize_order = false;
        cfg.allow_smooth_shift = true;
    } else if (r == 1) {
        cfg.r_cap = 50000;
        cfg.d_cap = 20000;
        cfg.max_q = 1000000;
        cfg.min_hub_deg = 2;
        cfg.min_savings = 2.5;
        cfg.add_target_links = true;
        cfg.strict_hub_first = false;
        cfg.b_full = 4;
        cfg.b_construct = 6;
        cfg.work_limit = 4.5e6;
        cfg.randomize_order = false;
        cfg.allow_smooth_shift = true;
    } else if (r == 2) {
        cfg.r_cap = 20000;
        cfg.d_cap = 10000;
        cfg.max_q = 500000;
        cfg.min_hub_deg = 3;
        cfg.min_savings = 3.0;
        cfg.add_target_links = true;
        cfg.strict_hub_first = false;
        cfg.b_full = 4;
        cfg.b_construct = 6;
        cfg.work_limit = 4.0e6;
        cfg.randomize_order = false;
        cfg.allow_smooth_shift = false;
    } else {
        std::mt19937_64 rgen(seed * 0xBF58476D1CE4E5B9ULL + 7ULL);
        static constexpr u64 r_caps[3] = {5000, 20000, 100000};
        static constexpr u64 d_caps[3] = {2000, 10000, 30000};
        cfg.r_cap = r_caps[rgen() % 3];
        cfg.d_cap = d_caps[rgen() % 3];
        cfg.max_q = 1000000;
        cfg.min_hub_deg = (r % 2 == 0) ? 2 : 3;
        cfg.min_savings = (r % 3 == 0) ? 1.0 : ((r % 3 == 1) ? 2.5 : 4.0);
        cfg.add_target_links = (r != 5);
        cfg.strict_hub_first = (r % 2 == 0);
        cfg.b_full = (r % 2 == 1) ? 4 : 3;
        cfg.b_construct = (r % 3 == 1) ? 6 : 5;
        cfg.work_limit = 3.5e6;
        cfg.randomize_order = true;
        cfg.allow_smooth_shift = (r % 2 == 0);
    }
    return cfg;
}

int choose_target_basis(const Options& opt, size_t n_rnd, int topD, int poolSz, int r) {
    const int max_reg_cap = opt.regs > 0 ? std::max(6, opt.regs - 6) : 42;
    if (opt.basisExplicit) {
        return std::min(opt.basis, max_reg_cap);
    }
    int base = 41;
    // When Phase 1 already populated >= 28 live registers for FREE, ALWAYS keep 41-42!
    if (poolSz >= 28) {
        base = 41;
    } else if (n_rnd <= 6) {
        base = std::min(max_reg_cap, static_cast<int>(18 + n_rnd * 2));
    } else if (n_rnd <= 12) {
        base = (topD >= 15) ? 38 : 34;
    } else if (topD >= 15) {
        base = 41;
    } else if (topD >= 12) {
        base = 39;
    } else {
        base = 35;
    }
    static constexpr int kOffsets[4] = {0, 1, -1, 0};
    return std::max(12, std::min(max_reg_cap, base + kOffsets[r % 4]));
}

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
        else if (a == "--regs") opt.regs = std::atoi(next());
        else if (a == "--basis") {
            opt.basis = std::atoi(next());
            opt.basisExplicit = true;
        } else if (a == "--beam") opt.beam = std::atoi(next());
        else if (a == "--verbose") opt.verbose = true;
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (opt.restarts < 1) opt.restarts = 1;
    if (opt.timeLimit < 0.05) opt.timeLimit = 0.05;
    if (opt.beam < 1) opt.beam = 1;

    Puzzle pz;
    if (!elf::read_puzzle(std::cin, pz)) {
        std::fprintf(stderr, "bad input\n");
        return 2;
    }

    const size_t lowerBound = pz.missing_targets().size();
    if (lowerBound == 0) return 0;

    Timer timer;
    PartitionResult part = partition_targets(pz);
    if (opt.verbose) {
        std::fprintf(stderr, "%s: D=%d structured=%zu random=%zu\n",
                     pz.id.c_str(), part.topD, part.structured.size(), part.randomTargets.size());
    }

    std::vector<Step> best;
    bool haveBest = false;
    const int totalRestarts = std::max(1, opt.restarts);

    for (int r = 0; r < totalRestarts; ++r) {
        if (haveBest && best.size() <= lowerBound) break;
        double elapsed = timer.elapsed_sec();
        if (haveBest && elapsed >= opt.timeLimit * 0.90) break;

        double remainingTime = std::max(0.08, opt.timeLimit - elapsed);
        double slice = remainingTime / static_cast<double>(totalRestarts - r);
        double rDeadline = std::min(opt.timeLimit * 0.96, elapsed + slice);

        u64 rSeed = elf::splitmix64(opt.seed + 0x9E3779B97F4A7C15ULL * static_cast<u64>(r + 1));

        // When randomTargets is tiny (<=3), last restart tries solving everything via Phase 1!
        bool allPhase1 = part.randomTargets.empty() ||
                         (part.randomTargets.size() <= 3 && r == totalRestarts - 1);

        std::optional<std::vector<Step>> verified;
        size_t p1Len = 0, basisSz = 0, mSz = 0, hSz = 0, p1HubsCnt = 0;
        double p1Sec = 0.0;

        if (allPhase1) {
            Phase1Config p1cfg = make_p1_config(r, rSeed);
            Phase1Engine p1(opt.regs, pz.limit, p1cfg, timer,
                            std::min(rDeadline, elapsed + slice * 0.85));
            p1.can_abort_on_deadline = (r > 0 && haveBest);
            p1.hard_deadline = rDeadline;
            p1.fatal_deadline = std::min(opt.timeLimit + 0.25, rDeadline + 0.15);
            if (p1.run(pz.starts, pz.missing_targets()) && !p1.aborted) {
                p1Len = p1.plan.size();
                p1HubsCnt = p1.dag.hubs.size();
                p1Sec = timer.elapsed_sec() - elapsed;
                verified = elf::verify_and_prune_plan(pz, p1.plan, opt.regs);
            }
        } else {
            Phase1Config p1cfg = make_p1_config(r, rSeed);
            double p1Soft = part.structured.empty()
                                ? elapsed
                                : (elapsed + slice * 0.30);
            double p1Hard = part.structured.empty()
                                ? elapsed + 0.05
                                : (elapsed + slice * 0.38);
            Phase1Engine p1(opt.regs, pz.limit, p1cfg, timer, p1Soft);
            p1.can_abort_on_deadline = (r > 0 && haveBest);
            p1.hard_deadline = p1Hard;
            p1.fatal_deadline = std::min(opt.timeLimit + 0.15, elapsed + slice * 0.48);

            if (p1.run(pz.starts, part.structured) && !p1.aborted) {
                // Grab any random targets that are 1-step reachable from Phase-1's live pool
                if (!part.structured.empty()) {
                    p1.harvest_extra_l1(part.randomTargets);
                }
                p1Len = p1.plan.size();
                p1HubsCnt = p1.dag.hubs.size();
                p1Sec = timer.elapsed_sec() - elapsed;

                std::vector<u64> remRnd;
                for (u64 t : part.randomTargets) {
                    if (!p1.coveredEver.contains(t)) remRnd.push_back(t);
                }

                // Collect Phase-1 mined hubs sorted by savings descending for warm-start priority
                std::vector<std::pair<double, u64>> hubRank;
                hubRank.reserve(p1.dag.hubs.size());
                for (const auto& hb : p1.dag.hubs) {
                    if (p1.in_pool(hb.H)) hubRank.emplace_back(hb.savings, hb.H);
                }
                std::sort(hubRank.begin(), hubRank.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
                std::vector<u64> p1HubVals;
                p1HubVals.reserve(hubRank.size());
                for (const auto& [sv, hv] : hubRank) {
                    (void)sv;
                    p1HubVals.push_back(hv);
                }

                const int poolSz = static_cast<int>(p1.pool.size());
                Phase2Config p2cfg;
                const int targetBasis = choose_target_basis(opt, remRnd.size(),
                                                            part.topD > 0 ? part.topD : 12,
                                                            poolSz, r);
                p2cfg.targetBasis = targetBasis;
                p2cfg.jitter = (r > 0);
                p2cfg.mode = r;
                p2cfg.allowAbort = (r > 0 && haveBest);

                if (poolSz < 20) {
                    // Low Phase-1 pool (e.g., independent or struct <= 1):
                    // NEVER starve |B|! Build full 40..42 balanced multi-scale basis with
                    // online absorption (r=0,2) or pure upfront greedy (r=1), matching mitm_basis.
                    p2cfg.warmCap = std::min(poolSz, targetBasis);
                    p2cfg.maxPerBucket = 4;
                    p2cfg.onlineAbsorb = (r % 2 == 0) && (remRnd.size() >= 6);
                    p2cfg.maxUpfrontTopup = std::max(0, targetBasis - poolSz);
                } else if (remRnd.size() <= 4) {
                    p2cfg.warmCap = targetBasis;
                    p2cfg.maxPerBucket = 4;
                    p2cfg.onlineAbsorb = false;
                    p2cfg.maxUpfrontTopup = (r == 0) ? 0 : 2;
                } else if (r == 0) {
                    // Balanced warm reuse (up to targetBasis-4 diverse free entries incl. top Phase-1 hubs)
                    // + targeted golden-zone top-up + free online absorption to full targetBasis!
                    p2cfg.warmCap = std::max(18, targetBasis - 4);
                    p2cfg.maxPerBucket = 4;
                    p2cfg.onlineAbsorb = true;
                    p2cfg.maxUpfrontTopup = std::min(6, std::max(3, static_cast<int>(remRnd.size()) / 3));
                } else if (r == 1) {
                    // High free warm reuse (up to full targetBasis) + light patch for missing scales
                    p2cfg.warmCap = targetBasis;
                    p2cfg.maxPerBucket = 4;
                    p2cfg.onlineAbsorb = true;
                    p2cfg.maxUpfrontTopup = (poolSz < 30) ? 10 : 2;
                } else {
                    // Fresh golden-zone diversity: keep ~8-10 top Phase-1 hubs + best 2/bucket,
                    // then build fresh golden-zone entries + online absorption up to targetBasis!
                    p2cfg.warmCap = std::min(targetBasis - 10, 26);
                    p2cfg.maxPerBucket = 2;
                    p2cfg.onlineAbsorb = (r % 2 == 0);
                    p2cfg.maxUpfrontTopup = std::min(14, std::max(6, static_cast<int>(remRnd.size())));
                }

                Phase2Engine p2(pz, opt, timer, rDeadline,
                                rSeed ^ 0x517cc1b727220a95ULL);
                verified = p2.run(p1.pool, p1.plan, p1HubVals, p1.coveredEver, remRnd, p2cfg);
                basisSz = p2.basisSize();
                mSz = p2.mSize();
                hSz = p2.hSize();
            }
        }

        const bool okRes = verified.has_value() && verified->size() <= 4096;
        const size_t totSz = okRes ? verified->size() : 0;
        if (okRes) {
            if (!haveBest || totSz < best.size()) {
                best = std::move(*verified);
                haveBest = true;
            }
        }

        if (opt.verbose) {
            std::fprintf(stderr,
                         "  restart %d: ok=%d p1=%zu(hubs=%zu,%.2fs) |B|=%zu |M|=%zu |H|=%zu tot=%zu best=%zu t=%.2fs\n",
                         r, okRes ? 1 : 0, p1Len, p1HubsCnt, p1Sec,
                         basisSz, mSz, hSz, totSz,
                         haveBest ? best.size() : 0, timer.elapsed_sec());
        }
    }

    if (!haveBest) {
        auto fb = binary_fallback(pz);
        auto verified = elf::verify_and_prune_plan(pz, fb, opt.regs);
        if (verified.has_value()) {
            best = std::move(*verified);
            haveBest = true;
        }
    }

    if (!haveBest) return 1;
    elf::emit_plan(best);
    return 0;
}
