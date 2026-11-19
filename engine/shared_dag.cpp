// Hub-DAG + Shifted-Affine + Factorization Search Engine for `shared` puzzles.
//
// Exploits the structure of `shared` puzzles:
//   1. Phase 1 mines:
//      - Consensus hidden hubs H via continued-fraction rational approximations,
//        pairwise GCDs, high-q shifted-affine links t = q*(H + d) + r, and near-squares.
//      - Target-to-target bit-saving shifted-affine edges (including degree-1 links
//        with large q up to 10^7 and arbitrary cost-saving shifts d), so parents are
//        kept hot in registers until their children are synthesized.
//      - Hub-to-hub DAG links and super-hub GCDs.
//      - Frequent small constants/offsets/multipliers.
//   2. Phase 2 runs a 48-register demand-aware pool engine with:
//      - Hub-cluster DFS sweeping with on-the-fly sibling chaining (always checking
//        if any already-hot pool value gives a cheaper shifted-affine path).
//      - Demand-aware pinning & usefulness eviction (protect active hubs/parents and
//        frequent constants; evict dead covered targets first).
//      - Multi-target + hub L2 voting and short-path search (B = 1..4).
//      - Self-consistent cost-guided construct(v) comparing:
//        (1) Direct add/sub & exact div from pool/L1
//        (2) Shifted-affine q*(p + d) + r from pool values (q up to 10^7)
//        (3) Exact factorization / square / balanced smooth splits (elf::factorize)
//        (4) Near-square shortcuts s^2 +- r
//        (5) Nearby smooth shifts (v +- d) -+ d
//        (6) Base splits a*q +- r (building L1 radix first to unlock a^2!)
//   3. Multi-restart diversity with strict wall-clock time enforcement and
//      exact verification + pruning via elf::verify_and_prune_plan.
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
#include <limits>
#include <numeric>
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
    int restarts = 4;
    double time_limit = 4.0;
    int regs = 48;
    bool verbose = false;
};

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

// Self-consistent build cost estimate:
// Calibrated so that for a base split v = a*q + r with a in pool (cost 0),
// c(q) + c(r) + 2 ≈ c(v), which naturally rejects peeling tiny factors (like 2, 3, 5)
// while rewarding balanced/smooth factorizations, squares, and shifted-affine edges.
inline double rough_build_cost(u64 x) {
    if (x == 0) return 0.0;
    if (x == 1) return 1.0;
    if (x <= 16) return 1.2 + 0.25 * std::log2(static_cast<double>(x));
    if (x <= 1024) return 0.48 * std::log2(static_cast<double>(x)) - 0.5;
    return 0.42 * std::log2(static_cast<double>(x)) - 0.9;
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

inline double rough_edge_cost(u64 q, i64 d, i64 r) {
    double c = 0.0;
    if (d != 0) c += rough_build_cost(abs_i64(d)) + 1.0;
    if (q > 1) c += rough_build_cost(q) + 1.0;
    if (r != 0) c += rough_build_cost(abs_i64(r)) + 1.0;
    return c;
}

// Compute best bit-saving shifted-affine representation t = q * (H + d) + r for given H.
// Uses cost-savings criterion rather than tiny hard caps on d/r/q!
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

MinedDAG mine_shared_dag(const Puzzle& pz, u64 r_cap, u64 d_cap, u64 max_q,
                         int min_hub_deg, double min_savings, bool add_target_links,
                         const Timer& timer, double hard_dl) {
    MinedDAG dag;
    std::vector<u64> ts = pz.missing_targets();
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
            } else if (static_cast<u128>(s + 1) * (s + 1) <= pz.limit &&
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
                                if (h_est > 100 && h_est <= pz.limit) {
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
                        if (h_est <= 100 || h_est > pz.limit) continue;
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
        if (h <= 100 || h > pz.limit) continue;
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
                    if (h_ref > 100 && h_ref <= pz.limit) test_h.push_back(h_ref);
                    if (t % q == 0 && t / q > 100) test_h.push_back(t / q);
                }
            }
        }
        {
            u64 s = elf::isqrt64(h0);
            if (h0 - s * s <= 2 * tol0 && s * s > 100) test_h.push_back(s * s);
            if (static_cast<u128>(s + 1) * (s + 1) <= pz.limit &&
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
            if (h <= 100 || h > pz.limit) continue;
            std::vector<HubEdge> edges;
            int exact_cnt = 0;
            double edge_savings = 0.0;
            for (u64 t : ts) {
                u64 q = 0;
                i64 d = 0, r = 0;
                double ec = 0.0;
                if (match_shifted_affine(t, h, pz.limit, max_q, 1.5, true, r_cap, d_cap,
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

    // Target-to-target bit-saving shifted-affine edges (captures high-q, large-d, and
    // degree-1 parent->child target chains like t[28] = 2041*(t[14] - 254749)!).
    // Since a target parent t_i must be built anyway, its marginal hub cost is 0!
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
                if (match_shifted_affine(tj, ti, pz.limit, 10000000ULL, 2.0, false,
                                         r_cap, d_cap, q, d, r, ec)) {
                    if (ec < bec) {
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
                    // Add edge to existing hub if not already present
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

    // Sort selected hubs by H ascending so parent hubs always have smaller index
    std::sort(dag.hubs.begin(), dag.hubs.end(),
              [](const HubInfo& a, const HubInfo& b) { return a.H < b.H; });

    // Check for super-hubs via pairwise GCD of selected hubs
    {
        std::unordered_set<u64> existing_h;
        existing_h.reserve(dag.hubs.size() * 2 + 8);
        for (const auto& h : dag.hubs) existing_h.insert(h.H);

        std::vector<u64> extra_gcds;
        for (size_t i = 0; i < dag.hubs.size(); ++i) {
            for (size_t j = i + 1; j < dag.hubs.size(); ++j) {
                u64 g = std::gcd(dag.hubs[i].H, dag.hubs[j].H);
                if (g > 500 && dag.hubs[i].H / g <= 4096 && dag.hubs[j].H / g <= 4096) {
                    if (!existing_h.count(g)) {
                        extra_gcds.push_back(g);
                    }
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

    // Detect hub-to-hub links (H_a -> H_b for a < b)
    for (size_t b = 0; b < dag.hubs.size(); ++b) {
        double best_cost = rough_build_cost(dag.hubs[b].H) - 1.5;
        int best_a = -1;
        u64 best_q = 1;
        i64 best_d = 0, best_r = 0;
        for (size_t a = 0; a < b; ++a) {
            u64 q = 0;
            i64 d = 0, r = 0;
            double ec = 0.0;
            if (match_shifted_affine(dag.hubs[b].H, dag.hubs[a].H, pz.limit, 10000000ULL,
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

    // Mine frequent small constants
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

static constexpr u64 kSmoothPrimes[] = {
    2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,  43,  47,
    53,  59,  61,  67,  71,  73,  79,  83,  89,  97,  101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
    199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
    383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
    467, 479, 487, 491, 499};

struct RestartConfig {
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

struct SharedEngine {
    struct L1E {
        u64 v, a, b;
        char op;
    };
    struct H2E {
        u64 v, a, b;
        char op;
        uint8_t cost;  // 2 (pool op L1 or L1^2/2*L1) or 3 (L1 op L1)
    };
    struct Cand {
        u64 r, a, q;
        uint8_t ca, cq, kind;  // 0: a*q+r, 1: a*q-r, 2: a+r, 3: a-r
        int rem;
    };

    const Puzzle& pz;
    const Options& opt;
    const RestartConfig& cfg;
    const Timer& timer;
    double deadline;

    u64 limit;
    std::vector<u64> pool;
    std::vector<u64> stamp;
    std::unordered_map<u64, int> pins;
    u64 tick = 0;
    size_t created = 0;
    size_t evictions = 0;
    size_t mitm_hits = 0;
    FlatMap<uint32_t> pool_idx;
    std::vector<L1E> l1;
    FlatMap<uint32_t> l1_idx;
    std::vector<H2E> h2;
    FlatMap<uint32_t> h2_idx;
    std::vector<std::pair<u64, uint32_t>> h2_sorted;
    size_t h2_ver_created = static_cast<size_t>(-1);
    size_t h2_ver_evictions = static_cast<size_t>(-1);
    std::vector<Step> plan;

    std::vector<u64> remaining;
    std::unordered_set<u64> remaining_set;
    std::unordered_set<u64> covered_targets;
    u64 last_covered_target = 0;
    std::unordered_set<u64> start_set;

    MinedDAG dag;
    std::vector<int> hub_rem_children;
    std::vector<int> hub_rem_child_hubs;
    std::vector<uint8_t> hub_built;
    std::unordered_map<u64, int> freq_rem;

    FlatMap<int> fail_b;
    std::vector<std::pair<u64, uint8_t>> M;
    bool m_dirty = true;
    double work = 0.0;
    bool timed_out = false;
    bool aborted = false;
    bool can_abort_on_deadline = false;
    bool fast_fallback_mode = false;
    double hard_deadline = 1e18;
    double fatal_deadline = 1e18;
    static constexpr size_t kMaxPlanSteps = 20000;
    static constexpr int kMaxConstructDepth = 16;
    std::vector<u64> construct_stack;
    std::mt19937_64 rng;

    SharedEngine(const Puzzle& p, const Options& o, const RestartConfig& c,
                 const Timer& t, double dl)
        : pz(p),
          opt(o),
          cfg(c),
          timer(t),
          deadline(dl),
          limit(p.limit),
          pool_idx(128),
          l1_idx(4096),
          h2_idx(16384),
          fail_b(1024),
          rng(c.seed * 0x9E3779B97F4A7C15ULL + 1) {}

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
        SharedEngine& eng;
        u64 val;
        bool active;
        PinGuard(SharedEngine& e, u64 v, bool act = true) : eng(e), val(v), active(act) {
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

    inline u64 eval(char op, u64 a, u64 b) const {
        return elf::apply_op(op, a, b, limit);
    }

    inline bool in_pool(u64 x) const { return pool_idx.contains(x); }

    inline int c01(u64 x) const {
        return in_pool(x) ? 0 : (l1_idx.contains(x) ? 1 : 2);
    }

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
        // Protect non-trivial shared multiplicative divisors of remaining targets.
        {
            int div_cnt = 0;
            for (u64 t : remaining) {
                if (t > v && t % v == 0) ++div_cnt;
            }
            if (div_cnt >= 2) {
                return 60.0 + 25.0 * static_cast<double>(div_cnt);
            }
        }
        if (v <= 1024) return 5.0;
        // Keep the most recently covered target warm (for sibling/chain reuse) while
        // evicting stale older covered targets before active intermediates.
        if (v == last_covered_target) return -50.0;
        if (covered_targets.count(v)) return -1000.0;
        return -100.0;
    }

    void on_value_produced(u64 v) {
        if (remaining_set.erase(v)) {
            covered_targets.insert(v);
            last_covered_target = v;
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
            if (opt.regs > 0 && pool.size() >= static_cast<size_t>(opt.regs)) {
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
            // Stop if apply() fails or makes no progress (would otherwise spin forever).
            if (!apply(steps)) break;
            if (plan.size() == before_plan && remaining.size() == before_rem) break;
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
                               : (b == 3) ? find3(v, out)
                                          : decomp(v, b, out);
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

    // Estimate factored cost of x given prime factorization of x (or a subset of factors).
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

    // Factor subset of `fac` that divides `d` to get accurate cost of divisor `d`.
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
        DIRECT_ADDSUB,  // p +- r (p in M)
        EXACT_DIV,      // p / (p / v) (p in pool)
        BASE_SPLIT,     // a * q +- r
        SHIFTED_AFFINE, // q * (p + d) + r
        FACTOR_SPLIT,   // f1 * f2 (f1 == f2 for square)
        NEAR_SQUARE,    // s * s +- r
        SMOOTH_SHIFT,   // (v +- d) -+ d
        PROD_RESID      // (x * y) +- r or h1 +- r with x in M U H2, y in M
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

    // Iterative, zero-recursion, strictly monotonic O(log v) constructor.
    // Uses at most 2 pins ({1, acc}) and can jump-start from any bit-prefix already in pool.
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
        // Jump-start from the largest bit-prefix (v >> sh) already in pool.
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

    // Fast deterministic fallback: finish all remaining targets in < 0.05s.
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

    // Build a bounded, pool-version-cached H2 table (values reachable in 2-3 steps from pool)
    // for fast on-demand MITM synthesis of large unlinked targets and hubs.
    void build_H2() {
        if (h2_ver_created == created && h2_ver_evictions == evictions) return;
        build_M();
        h2.clear();
        h2_idx.clear();
        h2_sorted.clear();
        h2_ver_created = created;
        h2_ver_evictions = evictions;

        static constexpr size_t kMaxH2 = 8192;
        h2.reserve(4096);

        auto add_h2 = [&](u64 val, char op, u64 a, u64 b, uint8_t cost) {
            if (val == 0 || val > limit || in_pool(val) || l1_idx.contains(val)) return;
            if (h2.size() >= kMaxH2) return;
            if (h2_idx.insert(val, static_cast<uint32_t>(h2.size()))) {
                h2.push_back(H2E{val, a, b, op, cost});
            }
        };

        std::vector<u64> l1_vals;
        l1_vals.reserve(l1.size());
        for (const auto& [val, c] : M) {
            if (c == 1) l1_vals.push_back(val);
        }
        if (l1_vals.empty()) return;

        // Pass 1a (cost 2): squares & doubles of ALL L1 values (unlocks huge 10^12..10^17 values).
        for (u64 x : l1_vals) {
            u64 sq = eval('*', x, x);
            if (sq) add_h2(sq, '*', x, x, 2);
            u64 dbl = eval('+', x, x);
            if (dbl) add_h2(dbl, '+', x, x, 2);
        }
        if (check_time()) {
            h2_sorted.reserve(h2.size());
            for (uint32_t i = 0; i < h2.size(); ++i) h2_sorted.emplace_back(h2[i].v, i);
            std::sort(h2_sorted.begin(), h2_sorted.end());
            return;
        }

        // Select a diverse bounded subset of L1: smallest 48, largest 64, and 16 spaced.
        std::vector<u64> l1_sel;
        l1_sel.reserve(128);
        const size_t n1 = l1_vals.size();
        const size_t n_small = std::min<size_t>(48, n1);
        for (size_t i = 0; i < n_small; ++i) l1_sel.push_back(l1_vals[i]);
        const size_t n_large = std::min<size_t>(64, n1);
        for (size_t i = n1 - n_large; i < n1; ++i) l1_sel.push_back(l1_vals[i]);
        if (n1 > 112) {
            for (size_t k = 1; k < 16; ++k) {
                l1_sel.push_back(l1_vals[(k * n1) / 16]);
            }
        }
        std::sort(l1_sel.begin(), l1_sel.end());
        l1_sel.erase(std::unique(l1_sel.begin(), l1_sel.end()), l1_sel.end());

        // Pass 1b (cost 2): pool op l1_sel.
        for (size_t xi = 0; xi < l1_sel.size() && h2.size() < kMaxH2; ++xi) {
            if ((xi & 31U) == 0 && check_time()) break;
            u64 x = l1_sel[xi];
            for (u64 p : pool) {
                add_h2(eval('+', x, p), '+', x, p, 2);
                add_h2(eval('*', x, p), '*', x, p, 2);
                if (x > p) {
                    add_h2(x - p, '-', x, p, 2);
                    if (x % p == 0) add_h2(x / p, '/', x, p, 2);
                } else if (p > x) {
                    add_h2(p - x, '-', p, x, 2);
                    if (p % x == 0) add_h2(p / x, '/', p, x, 2);
                }
            }
        }

        // Pass 2 (cost 3): small-L1 x small-L1 (+, -, *) and large-L1 x large-L1 (*).
        if (!check_time() && h2.size() < kMaxH2) {
            size_t ns = std::min<size_t>(32, l1_sel.size());
            for (size_t i = 0; i < ns && h2.size() < kMaxH2; ++i) {
                u64 a = l1_sel[i];
                for (size_t j = i + 1; j < ns && h2.size() < kMaxH2; ++j) {
                    u64 b = l1_sel[j];
                    add_h2(eval('+', a, b), '+', a, b, 3);
                    add_h2(b - a, '-', b, a, 3);
                    add_h2(eval('*', a, b), '*', a, b, 3);
                }
            }
            size_t nl = std::min<size_t>(24, l1_sel.size());
            size_t base_l = l1_sel.size() - nl;
            for (size_t i = base_l; i < l1_sel.size() && h2.size() < kMaxH2; ++i) {
                u64 a = l1_sel[i];
                for (size_t j = i + 1; j < l1_sel.size() && h2.size() < kMaxH2; ++j) {
                    u64 prod = eval('*', a, l1_sel[j]);
                    if (prod) add_h2(prod, '*', a, l1_sel[j], 3);
                }
            }
        }

        h2_sorted.reserve(h2.size());
        for (uint32_t i = 0; i < h2.size(); ++i) h2_sorted.emplace_back(h2[i].v, i);
        std::sort(h2_sorted.begin(), h2_sorted.end());
    }

    int lookup_m_h2_cost(u64 x) const {
        if (x == 0) return 0;
        if (in_pool(x)) return 0;
        if (l1_idx.contains(x)) return 1;
        if (const uint32_t* hi = h2_idx.find(x)) return static_cast<int>(h2[*hi].cost);
        return 99;
    }

    void emit_m_h2_val(u64 x, std::vector<Step>& out) const {
        if (x == 0 || in_pool(x)) return;
        if (l1_idx.contains(x)) {
            emit_l1(x, out);
            return;
        }
        if (const uint32_t* hi = h2_idx.find(x)) {
            const H2E& e = h2[*hi];
            if (l1_idx.contains(e.a)) emit_l1(e.a, out);
            if (e.b != e.a && l1_idx.contains(e.b)) emit_l1(e.b, out);
            out.push_back(Step{e.op, e.a, e.b, e.v});
        }
    }

    double est_m_h2_cost(u64 x) const {
        if (x == 0) return 0.0;
        int c = lookup_m_h2_cost(x);
        if (c <= 3) return static_cast<double>(c);
        return rough_build_cost(x);
    }

    // Search M*M +- r, H2*M +- r, and H2 +- r (updating `best` with PROD_RESID for both
    // exact H2 hits AND approximate hits with small recursive residual r).
    // Also returns true + exact steps in `exact_out` if a short exact (possibly /d) hit beats `best`.
    bool mitm_search(u64 v, ConstructChoice& best, std::vector<Step>& exact_out) {
        if (check_time()) return false;
        build_H2();
        if (check_time()) return false;

        int best_exact_len = std::min(8, static_cast<int>(std::floor(best.score)) + 1);
        bool found_exact = false;
        std::vector<Step> cand;
        cand.reserve(10);

        auto try_record_exact = [&](std::vector<Step>& raw) {
            std::vector<Step> clean;
            clean.reserve(raw.size());
            std::vector<u64> made;
            made.reserve(raw.size());
            auto avail = [&](u64 x) -> bool {
                if (in_pool(x)) return true;
                for (u64 m : made) {
                    if (m == x) return true;
                }
                return false;
            };
            for (const Step& s : raw) {
                if (avail(s.r)) continue;
                if (!avail(s.a) || !avail(s.b)) return;
                u64 res = eval(s.op, s.a, s.b);
                if (res == 0 || res != s.r) return;
                clean.push_back(s);
                made.push_back(res);
            }
            if (clean.empty() || clean.back().r != v) return;
            if (static_cast<int>(clean.size()) < best_exact_len) {
                best_exact_len = static_cast<int>(clean.size());
                exact_out = std::move(clean);
                found_exact = true;
            }
        };

        auto consider_prod_resid = [&](u64 x, int cx, u64 y, int cy, u64 prod) {
            if (prod == v) {
                double sc = static_cast<double>(cx + cy + (y ? 1 : 0));
                if (sc < best.score) {
                    best = {PlanKind::PROD_RESID, sc, x, y, 0, 0, 0, false};
                }
                return;
            }
            u64 r = (prod > v) ? (prod - v) : (v - prod);
            if (r == 0 || r > v / 2 || on_stack(r)) return;
            int cr = lookup_m_h2_cost(r);
            // Only accept exact M/H2 residual hits OR genuinely small residuals (r <= 10000),
            // so PROD_RESID never replaces a structured split with a huge one-off residual!
            if (cr > 2 && r > 10000) return;
            double rc = (cr <= 2) ? static_cast<double>(cr) : rough_build_cost(r);
            double sc = static_cast<double>(cx + cy + (y ? 1 : 0)) + rc + 1.0;
            if (sc + 0.5 < best.score) {
                best = {PlanKind::PROD_RESID, sc, x, y, 0, 0, static_cast<i64>(r), prod > v};
            }
        };

        // 1. Direct H2 +- r (near v in h2_sorted)
        if (!h2_sorted.empty()) {
            auto it = std::lower_bound(h2_sorted.begin(), h2_sorted.end(),
                                       std::make_pair(v, uint32_t{0}));
            for (int off = -8; off <= 8; ++off) {
                auto jt = it + off;
                if (jt < h2_sorted.begin() || jt >= h2_sorted.end()) continue;
                u64 h1 = jt->first;
                const H2E& e1 = h2[jt->second];
                consider_prod_resid(h1, e1.cost, 0, 0, h1);
            }
        }

        // 2. Full M * M +- r (x <= sqrt(v), y = lower_bound(M, v / x))
        {
            u64 x_lim = elf::isqrt64(v) + 2;
            for (size_t i = 0; i < M.size(); ++i) {
                if ((i & 255U) == 0 && check_time()) break;
                u64 x = M[i].first;
                if (x < 2) continue;
                if (x > x_lim) break;
                int cx = M[i].second;
                auto it = std::lower_bound(M.begin(), M.end(), std::make_pair(v / x, uint8_t{0}));
                for (int d = -1; d <= 1; ++d) {
                    auto jt = it + d;
                    if (jt < M.begin() || jt >= M.end()) continue;
                    u64 y = jt->first;
                    if (y < x) continue;
                    u128 p128 = static_cast<u128>(x) * static_cast<u128>(y);
                    if (p128 > limit) continue;
                    consider_prod_resid(x, cx, y, jt->second, static_cast<u64>(p128));
                }
            }
        }

        // 3. H2 * M +- r (m in pool + bounded small/large M, h = lower_bound(h2_sorted, v / m))
        if (!h2_sorted.empty() && !check_time()) {
            auto check_hm = [&](u64 m, int cm) {
                auto it = std::lower_bound(h2_sorted.begin(), h2_sorted.end(),
                                           std::make_pair(v / m, uint32_t{0}));
                for (int d = -1; d <= 1; ++d) {
                    auto jt = it + d;
                    if (jt < h2_sorted.begin() || jt >= h2_sorted.end()) continue;
                    u64 h = jt->first;
                    if (h < 2) continue;
                    const H2E& he = h2[jt->second];
                    u128 p128 = static_cast<u128>(h) * static_cast<u128>(m);
                    if (p128 > limit) continue;
                    consider_prod_resid(h, he.cost, m, cm, static_cast<u64>(p128));
                }
            };
            for (u64 p : pool) {
                if (p >= 2 && p < v) check_hm(p, 0);
            }
            size_t m_sz = M.size();
            size_t small_lim = std::min<size_t>(256, m_sz);
            for (size_t i = 0; i < small_lim; ++i) {
                if (M[i].second == 0) continue;
                u64 m = M[i].first;
                if (m < 2) continue;
                if (m >= v) break;
                check_hm(m, 1);
            }
            if (m_sz > 256 && !check_time()) {
                size_t start_l = (m_sz > 384) ? (m_sz - 128) : 256;
                for (size_t i = start_l; i < m_sz; ++i) {
                    if (M[i].second == 0) continue;
                    u64 m = M[i].first;
                    if (m < 2 || m >= v) continue;
                    check_hm(m, 1);
                }
            }
        }

        // 4. Exact division wrapper: tv = v * d <= limit for small d in pool (exact hits only).
        best_exact_len = std::min(best_exact_len, static_cast<int>(std::floor(best.score)) + 1);
        if (!check_time() && best_exact_len > 4) {
            std::vector<u64> small_d;
            small_d.reserve(12);
            for (u64 p : pool) {
                if (p >= 2 && p <= 256 && static_cast<u128>(v) * static_cast<u128>(p) <= limit) {
                    small_d.push_back(p);
                }
            }
            std::sort(small_d.begin(), small_d.end());
            if (small_d.size() > 12) small_d.resize(12);
            for (u64 d : small_d) {
                if (check_time()) break;
                u64 tv = v * d;
                for (u64 m : pool) {
                    if (m < 2 || m >= tv) continue;
                    auto it = std::lower_bound(h2_sorted.begin(), h2_sorted.end(),
                                               std::make_pair(tv / m, uint32_t{0}));
                    for (int off = -1; off <= 1; ++off) {
                        auto jt = it + off;
                        if (jt < h2_sorted.begin() || jt >= h2_sorted.end()) continue;
                        u64 h = jt->first;
                        if (h < 2) continue;
                        const H2E& he = h2[jt->second];
                        int base_c = static_cast<int>(he.cost) + 2;  // *m and /d
                        if (base_c >= best_exact_len) continue;
                        u128 p128 = static_cast<u128>(h) * static_cast<u128>(m);
                        if (p128 > limit) continue;
                        u64 prod = static_cast<u64>(p128);
                        if (prod == tv) {
                            cand.clear();
                            emit_m_h2_val(h, cand);
                            cand.push_back(Step{'*', h, m, prod});
                            cand.push_back(Step{'/', tv, d, v});
                            try_record_exact(cand);
                        } else {
                            u64 r = (prod > tv) ? (prod - tv) : (tv - prod);
                            int cr = lookup_m_h2_cost(r);
                            if (base_c + cr + 1 < best_exact_len) {
                                cand.clear();
                                emit_m_h2_val(h, cand);
                                cand.push_back(Step{'*', h, m, prod});
                                emit_m_h2_val(r, cand);
                                cand.push_back(Step{prod > tv ? '-' : '+', prod, r, tv});
                                cand.push_back(Step{'/', tv, d, v});
                                try_record_exact(cand);
                            }
                        }
                    }
                }
            }
        }
        return found_exact;
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

            int b_lim = (depth == 0) ? cfg.b_construct : (depth == 1 ? 5 : 3);
            if (timed_out) b_lim = 2;
            std::vector<Step> steps;
            bool ok = false;
            for (int B = 2; B <= b_lim && !ok; ++B) {
                if (B >= 3 && check_time()) break;
                steps.clear();
                work = (depth == 1 && B >= 5) ? (cfg.work_limit * 0.85) : 0.0;
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

            bool low_pin_mode = (pins.size() + 8 >= static_cast<size_t>(opt.regs)) || depth >= 10;
            ConstructChoice best;

            // --- Strategy 0: Direct add/sub from nearby M and pool values ---
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
                // Also explicitly check pool values (cost 0!) even if L1 crowds M around v.
                for (u64 p : pool) {
                    try_addsub(p, 0);
                }
                // Exact division from pool values p > v
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

            // --- Strategy 5 (baseline): Base split v = a*q +- r with a in M ---
            {
                u64 root = elf::isqrt64(v);
                auto eval_a = [&](u64 a, int ca) {
                    if (a < 2 || a >= v || on_stack(a)) return;
                    u64 q = v / a, r = v % a;
                    if (q < 2 || q >= v) return;  // q==1 handled cleanly by DIRECT_ADDSUB!
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
                // --- Strategy 1: Shifted-affine from pool values p < v (q up to 10^7!) ---
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
                    if (q0 > 2) try_sa_q(q0 - 1);
                    try_sa_q(q0 + 1);
                }

                // --- Strategy 2: Exact Factorization & Square Split (self-consistent!) ---
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
                        // Explicitly check exact pool divisors even if divs truncated at 256.
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

                // --- Strategy 3: Near-Square Shortcut v = s^2 +- r ---
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

                // --- Strategy 4: Nearby Smooth Shift v = (v +- d) -+ d ---
                if (cfg.allow_smooth_shift && depth == 0 && v >= 10000 && !timed_out) {
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
                    for (u64 d = 1; d <= 64; ++d) {
                        check_smooth(d, false);
                        check_smooth(d, true);
                    }
                    for (u64 p : pool) {
                        if (p > 64 && p <= 5000) {
                            check_smooth(p, false);
                            check_smooth(p, true);
                        }
                    }
                }
            }

            // --- Lightweight on-demand H2 MITM + M*M/H2*M +- r for hard large targets / hubs ---
            if (depth == 0 && v >= 100000 && !check_time() && !low_pin_mode && best.score >= 7.5) {
                std::vector<Step> mitm_steps;
                if (mitm_search(v, best, mitm_steps)) {
                    if (apply(mitm_steps) && in_pool(v)) {
                        ++mitm_hits;
                        return;
                    }
                }
                if (aborted) return;
            }

            if (best.kind == PlanKind::NONE) {
                if (!monotonic_construct(v)) aborted = true;
                return;
            }

            switch (best.kind) {
                case PlanKind::DIRECT_ADDSUB: {
                    u64 x = best.a;
                    u64 r = static_cast<u64>(best.r);
                    // x comes from M (pool or L1): materialize directly without upward recursion!
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
                    // If `a` is already in L1 (1 step!) and q is huge, emit `a` FIRST!
                    // Putting `a` into pool unlocks `a^2` and `a*p` in L1 for building `q`!
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
                case PlanKind::PROD_RESID: {
                    u64 x = best.a;
                    u64 y = best.b;
                    u64 r = static_cast<u64>(best.r);
                    u64 prod = (y > 0) ? (x * y) : x;
                    std::vector<Step> batch;
                    batch.reserve(8);
                    if (!in_pool(prod)) {
                        emit_m_h2_val(x, batch);
                        if (y > 0) {
                            if (y != x) emit_m_h2_val(y, batch);
                            batch.push_back(Step{'*', x, y, prod});
                        }
                    }
                    bool r_direct = (r > 0 && lookup_m_h2_cost(r) <= 3);
                    if (r_direct) {
                        emit_m_h2_val(r, batch);
                        batch.push_back(Step{best.minus ? '-' : '+', prod, r, v});
                    }
                    // Deduplicate batch steps before apply()
                    if (!batch.empty()) {
                        std::vector<Step> clean;
                        std::vector<u64> made;
                        auto avail = [&](u64 val) -> bool {
                            if (in_pool(val)) return true;
                            for (u64 m : made) {
                                if (m == val) return true;
                            }
                            return false;
                        };
                        for (const Step& s : batch) {
                            if (avail(s.r)) continue;
                            if (!avail(s.a) || !avail(s.b)) break;
                            clean.push_back(s);
                            made.push_back(s.r);
                        }
                        if (!clean.empty()) apply(clean);
                    }
                    if (aborted) break;
                    if (in_pool(v)) {
                        ++mitm_hits;
                        break;
                    }
                    if (r > 0 && in_pool(prod)) {
                        ++mitm_hits;
                        PinGuard pprod(*this, prod);
                        construct(r, depth + 1);
                        if (aborted || !in_pool(prod) || !in_pool(r)) break;
                        apply({Step{best.minus ? '-' : '+', prod, r, v}});
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }

    // Synthesize `target` from hot `base_hub` (or an even better already-hot pool sibling!).
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

        // On-the-fly sibling chaining: check if ANY value currently in `pool` gives a
        // strictly cheaper shifted-affine or add/sub formula for `target`!
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
            // Direct add/sub from sibling p (whether p < target or p > target!)
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

        // Check if target is already reachable in 2..3 steps faster than formula!
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

        // Verify arithmetic safety
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
            // If this newly-produced target is ITSELF a hub with children, sweep it right now!
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

    // Recompute hub bookkeeping from the real engine state. The incremental counters
    // maintained by on_value_produced() can go stale (e.g. a child hub that is already a
    // starting value, a child hub with no remaining work that is never built, or duplicate
    // hub values), which used to make run() re-select the same hub forever without
    // emitting any step. Child hubs always have a larger index than their parent (hubs are
    // sorted by H and parent links only go from smaller to larger index), so a reverse
    // sweep sees every child before its parent.
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

    void run() {
        for (u64 s : pz.starts) {
            start_set.insert(s);
            add_value(s, true);
        }
        remaining = pz.missing_targets();
        for (u64 t : remaining) remaining_set.insert(t);
        if (remaining.empty()) return;

        dag = mine_shared_dag(pz, cfg.r_cap, cfg.d_cap, cfg.max_q,
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

        if (opt.verbose) {
            std::fprintf(stderr, "[shared_dag] mined %zu hubs, %zu freq_constants\n",
                         dag.hubs.size(), freq_rem.size());
        }

        std::unordered_map<u64, std::vector<size_t>> stale;
        std::vector<Step> steps;
        size_t last_plan_size = static_cast<size_t>(-1);
        size_t last_remaining = static_cast<size_t>(-1);
        size_t last_evictions = 0;
        int stuck_iters = 0;
        int evict_stall_iters = 0;
        int l2_vote_evict_streak = 0;

        while (!remaining.empty()) {
            // Strict deadline enforcement: abort on restart r > 0, or finish fast on restart 0.
            check_time();
            if (aborted || plan.size() > kMaxPlanSteps) {
                aborted = true;
                return;
            }
            if (fast_fallback_mode) {
                fast_finish_remaining();
                return;
            }
            // Progress guard: detect both zero-step stalls AND eviction ping-pong loops
            // where evictions keep occurring without covering any new target.
            if (remaining.size() == last_remaining) {
                if (plan.size() == last_plan_size) {
                    ++stuck_iters;
                } else {
                    stuck_iters = 0;
                }
                if (evictions > last_evictions) {
                    ++evict_stall_iters;
                }
            } else {
                stuck_iters = 0;
                evict_stall_iters = 0;
                l2_vote_evict_streak = 0;
            }
            last_plan_size = plan.size();
            last_remaining = remaining.size();
            last_evictions = evictions;

            if (stuck_iters >= 4 || evict_stall_iters >= 32) {
                if (!monotonic_construct(pick_standalone_target())) {
                    aborted = true;
                    return;
                }
                continue;
            }
            if (stuck_iters >= 2 || evict_stall_iters >= 20) {
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
                l2_vote_evict_streak = 0;
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
                l2_vote_evict_streak = 0;
                continue;
            }

            // 3. Multi-Target + Active-Hub L2 Voting (capped eviction streak prevents ping-pong)
            if (l2_vote_evict_streak < 12) {
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
                    size_t ev_before = evictions;
                    if (apply(steps)) {
                        if (evictions > ev_before) ++l2_vote_evict_streak;
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
                l2_vote_evict_streak = 0;
                continue;
            }

            // 5. Hub-First Construct OR Standalone Target Construct
            l2_vote_evict_streak = 0;
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

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    bool restarts_explicit = false;
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
        else if (a == "--restarts") {
            opt.restarts = std::atoi(next());
            restarts_explicit = true;
        } else if (a == "--time") opt.time_limit = std::atof(next());
        else if (a == "--regs") opt.regs = std::atoi(next());
        else if (a == "--verbose") opt.verbose = true;
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (opt.restarts < 1) opt.restarts = 1;
    if (opt.time_limit < 0.05) opt.time_limit = 0.05;

    Puzzle pz;
    if (!elf::read_puzzle(std::cin, pz)) {
        std::fprintf(stderr, "bad input\n");
        return 2;
    }

    const size_t lb = pz.missing_targets().size();
    const int base_restarts = opt.restarts;
    const int max_restarts = restarts_explicit ? opt.restarts : std::max(opt.restarts, 10);

    Timer total_timer;
    std::vector<Step> best_plan;
    bool have_best = false;

    for (int r = 0; r < max_restarts; ++r) {
        if (have_best && best_plan.size() <= lb) break;
        double elapsed = total_timer.elapsed_sec();
        if (r > 0 && elapsed >= opt.time_limit * 0.88) break;

        RestartConfig cfg;
        cfg.seed = opt.seed + static_cast<u64>(r);
        if (r == 0) {
            // Restart 0: Full Hub-DAG + Target-to-Target Shifted-Affine + Self-Consistent Factor
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
            // Restart 1: Deeper Search (B=4, construct B=6) + Wider Caps + Savings/Cost Order
            cfg.r_cap = 50000;
            cfg.d_cap = 20000;
            cfg.max_q = 1000000;
            cfg.min_hub_deg = 2;
            cfg.min_savings = 2.5;
            cfg.add_target_links = true;
            cfg.strict_hub_first = false;
            cfg.b_full = 4;
            cfg.b_construct = 6;
            cfg.work_limit = 5e6;
            cfg.randomize_order = false;
            cfg.allow_smooth_shift = true;
        } else if (r == 2) {
            // Restart 2: Strong Hubs Only (deg >= 3) + Target-to-Target Links + Deeper Search
            cfg.r_cap = 20000;
            cfg.d_cap = 10000;
            cfg.max_q = 500000;
            cfg.min_hub_deg = 3;
            cfg.min_savings = 3.0;
            cfg.add_target_links = true;
            cfg.strict_hub_first = false;
            cfg.b_full = 4;
            cfg.b_construct = 6;
            cfg.work_limit = 4e6;
            cfg.randomize_order = false;
            cfg.allow_smooth_shift = false;
        } else {
            // Restart 3+: Diverse randomized hub thresholds, search depths & orderings
            std::mt19937_64 rgen(cfg.seed * 0xBF58476D1CE4E5B9ULL + 7);
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

        double remaining_time = std::max(0.05, opt.time_limit - elapsed);
        int denom = (r < base_restarts) ? (base_restarts - r) : (max_restarts - r);
        double slice = (r == 0 && base_restarts > 1)
                           ? std::min(remaining_time * 0.45, std::max(0.25, opt.time_limit / denom))
                           : (remaining_time / denom);
        double dl = std::min(opt.time_limit, elapsed + std::max(0.08, slice));

        SharedEngine engine(pz, opt, cfg, total_timer, dl);
        if (r > 0 && have_best) {
            engine.can_abort_on_deadline = true;
            engine.deadline = dl;
            engine.hard_deadline = std::min(dl + 0.08, opt.time_limit + 0.05);
            engine.fatal_deadline = opt.time_limit + 0.15;
        } else {
            engine.can_abort_on_deadline = false;
            engine.deadline = std::min(dl, opt.time_limit * 0.85);
            engine.hard_deadline = opt.time_limit;
            engine.fatal_deadline = opt.time_limit + 0.25;
        }
        engine.run();
        if (engine.aborted && opt.verbose) {
            std::fprintf(stderr, "restart %d: aborted (past hard deadline), %.2fs\n",
                         r, total_timer.elapsed_sec());
        }
        if (engine.aborted) continue;

        auto verified = elf::verify_and_prune_plan(pz, engine.plan, opt.regs);
        if (verified.has_value()) {
            size_t pruned_sz = verified->size();
            if (!have_best || pruned_sz < best_plan.size()) {
                best_plan = std::move(*verified);
                have_best = true;
            }
            if (opt.verbose) {
                std::fprintf(stderr, "restart %d: %zu -> %zu steps, %zu evictions, %zu mitm, best %zu, %.2fs\n",
                             r, engine.plan.size(), pruned_sz, engine.evictions, engine.mitm_hits,
                             best_plan.size(), total_timer.elapsed_sec());
            }
        } else if (opt.verbose) {
            std::fprintf(stderr, "restart %d: verification failed (%zu raw steps), %.2fs\n",
                         r, engine.plan.size(), total_timer.elapsed_sec());
        }
    }

    if (!have_best) {
        auto fb = binary_fallback(pz);
        auto verified = elf::verify_and_prune_plan(pz, fb, opt.regs);
        if (verified.has_value()) {
            best_plan = std::move(*verified);
            have_best = true;
        } else {
            std::fprintf(stderr, "shared_dag: fallback verification failed for %s\n", pz.id.c_str());
            return 1;
        }
    }

    elf::emit_plan(best_plan);
    return 0;
}
