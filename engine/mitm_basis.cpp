// MITM Basis Engine for Santa 2026 - The Elf Calculator.
//
// Architecture:
//   Phase 1: Deliberate multi-scale pinned basis B (|B| <= regs - 6), with
//            optional online absorption of useful intermediate values from
//            the first few targets to save upfront basis steps while reaching
//            |B| = 40..42.
//   Tables:  Unified Node table H = B U L1(B) U L2(B) (~150k..240k entries),
//            with L1-cache M_map, 256 KB 2-hash Bloom filter for H_map, and
//            sorted arrays M_sorted, H_sorted.
//   Phase 2: Per-target solver using ONLY B + ephemeral scratch registers:
//     1) Fast exact solvers (H, H op B, M op H, (H op B) op B, M*H +- H, H +- H)
//     2) t * d multiplier lottery (d in B U M with t * d <= limit)
//     3) Unified O(|H|) two-pointer H x H scan (V = u * v +- r) + quotient hook
//     4) Linear-combo two-pointer scan V = (b * x +- y) +- r (b in B, x,y in H)
//        checking ~3.5e12 cost-<=6 triples & ~1e18 cost-<=9 quads in ~8 ms
//     5) Layered residual solver: ultra-fast H op B on top 1,200 residuals,
//        M op H on top 120, M*H +- H / H +- H on top 24, deep H x H fallback
//     6) Divisor-remainder beam search (V = q * x +- r) + guaranteed fallback
//     7) Cross-target scratch promotion + verify_and_prune_plan verification.

#include "common.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>

using namespace elf;

namespace {

struct Options {
    u64 seed = 0;
    int restarts = 3;
    double time_limit = 4.0;
    int basis = 36;
    bool basis_explicit = false;
    int beam = 64;
    int regs = 48;
    bool verbose = false;
};

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

// 256 KB (32768 x 64-bit words) single-cache-line 3-bit blocked Bloom filter:
// 1 L2 cache read + 1 multiply-shift hash, ~0.9% FP rate at |H| = 150k.
class BloomFilter {
public:
    static constexpr size_t kWords = size_t(1) << 15;  // 32768 words = 256 KB

    BloomFilter() : words_(kWords, 0) {}

    void clear() { std::fill(words_.begin(), words_.end(), 0); }

    static inline u64 hash64(u64 k) {
        u64 h = k * 0x9E3779B97F4A7C15ULL;
        return h ^ (h >> 32);
    }

    inline void insert(u64 k) {
        u64 h = hash64(k);
        size_t w = static_cast<size_t>(h >> 49);
        u64 mask = (u64(1) << (h & 63u)) |
                   (u64(1) << ((h >> 6) & 63u)) |
                   (u64(1) << ((h >> 12) & 63u));
        words_[w] |= mask;
    }

    inline bool maybe_contains(u64 k) const {
        u64 h = hash64(k);
        size_t w = static_cast<size_t>(h >> 49);
        u64 mask = (u64(1) << (h & 63u)) |
                   (u64(1) << ((h >> 6) & 63u)) |
                   (u64(1) << ((h >> 12) & 63u));
        return (words_[w] & mask) == mask;
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

// Residual from V = s +- r where s is (b * x + y), (b * x - y), or (y - b * x).
struct LinearResCand {
    u64 r = 0;
    uint32_t b_idx = 0;
    uint32_t x_idx = 0;
    uint32_t y_idx = 0;
    uint8_t kind = 0;  // 0: b*x + y, 1: b*x - y, 2: y - b*x
    char r_sign = '+'; // '+' means V = s + r, '-' means V = s - r
    uint8_t oc = 0;    // c(x) + c(y) + 3
};

struct BeamCand {
    u64 q = 0;
    uint32_t x_idx = 0;
    uint32_t r_idx = UINT32_MAX;
    char sign = '*';
    uint8_t oc = 0;
    double score = 0.0;
};

struct RemBranch {
    u64 r = 0;
    uint32_t u_idx = 0;
    uint32_t v_idx = 0;
    char sign = '+';  // '+' means V = u*v + r, '-' means V = u*v - r
    uint8_t oc = 0;   // outer cost: c(u) + c(v) + 2
};

// Unified 3-layer beam node encoding a reduction chain back to target:
//   kind == 0: root V = target * d (x_idx = d_idx or UINT32_MAX for d = 1)
//   kind == 1: quotient move:  parent = val * x (+|-) r (x_idx = x, r_idx = r or UINT32_MAX)
//   kind == 2: remainder move: parent = u * v (+|-) val (x_idx = u, r_idx = v, sign = '+'|'-')
struct BeamNode {
    u64 val = 0;
    int cost = 0;
    int parent = -1;
    uint8_t kind = 0;
    uint32_t x_idx = UINT32_MAX;
    uint32_t r_idx = UINT32_MAX;
    char sign = '*';
};

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

inline int log10_half_bucket(u64 v) {
    if (v <= 1) return 0;
    return static_cast<int>(std::log10(static_cast<double>(v)) * 2.0);
}

inline double log2_approx(u64 v) {
    if (v <= 1) return 0.0;
    return std::log2(static_cast<double>(v));
}

class MitmBasisSolver {
public:
    MitmBasisSolver(const Puzzle& pz, const Options& opt, const Timer& timer, double deadline)
        : pz_(pz), opt_(opt), timer_(timer), deadline_(deadline),
          M_map_(8192), H_map_(524288) {}

    bool time_up() const { return timer_.elapsed_sec() >= deadline_; }
    bool local_time_up(double local_dl) const { return timer_.elapsed_sec() >= local_dl; }

    std::optional<std::vector<Step>> solve_restart(int restart_idx, u64 rng_seed,
                                                   int target_basis, bool online_absorb) {
        rng_.seed(rng_seed);
        missing_ = pz_.missing_targets();
        if (missing_.empty()) return std::vector<Step>{};

        max_t_ = *std::max_element(missing_.begin(), missing_.end());
        dec_d_ = std::max(1.0, std::log10(static_cast<double>(max_t_)));

        int max_reg_cap = opt_.regs > 0 ? std::max(6, opt_.regs - 5) : 43;
        int final_basis = std::min(target_basis, max_reg_cap);

        // Analyze target structure (pairwise GCD > 500, close/affine pairs, smooth shifts)
        // for Target-First Bootstrap ordering and close-pair anchoring.
        const size_t n_miss = missing_.size();
        std::vector<uint8_t> is_easy(n_miss, 0);
        std::vector<int> easy_prio(n_miss, 0);
        FlatMap<u64> close_anchor(n_miss * 2 + 16);
        FlatMap<uint8_t> is_anchor_target(n_miss * 2 + 16);
        std::vector<std::pair<int, u64>> gcd_helpers;  // (multiplicity, gcd_val)

        {
            FlatMap<int> gcd_counts(n_miss * 2 + 16);
            for (size_t i = 0; i < n_miss; ++i) {
                u64 lo = missing_[i];
                for (size_t j = i + 1; j < n_miss; ++j) {
                    u64 hi = missing_[j];
                    if (static_cast<u128>(hi - lo) * 1000ULL < lo) {
                        is_easy[i] = 1;
                        is_easy[j] = 1;
                        easy_prio[i] += 30;
                        easy_prio[j] += 30;
                        if (!close_anchor.contains(hi)) {
                            close_anchor.insert(hi, lo);
                            is_anchor_target.insert(lo, 1);
                        }
                    } else {
                        u64 q = (hi + lo / 2) / lo;
                        if (q >= 2 && q <= 256) {
                            u128 prod = static_cast<u128>(q) * lo;
                            u64 diff = prod > hi ? static_cast<u64>(prod - hi)
                                                 : static_cast<u64>(hi - prod);
                            if (diff <= std::max<u64>(5000ULL, lo / 100000ULL)) {
                                is_easy[i] = 1;
                                is_easy[j] = 1;
                                easy_prio[i] += 25;
                                easy_prio[j] += 25;
                                if (!close_anchor.contains(hi)) {
                                    close_anchor.insert(hi, lo);
                                    is_anchor_target.insert(lo, 1);
                                }
                            }
                        }
                    }
                    u64 g = std::gcd(lo, hi);
                    if (g > 500ULL && g < lo) {
                        is_easy[i] = 1;
                        is_easy[j] = 1;
                        easy_prio[i] += 20;
                        easy_prio[j] += 20;
                        if (int* c = gcd_counts.find(g)) ++(*c);
                        else gcd_counts.insert(g, 1);
                    }
                }
            }

            // 1-step smooth shifts from {1} U starts
            const u64 sm_bound = (dec_d_ <= 6.5) ? 20ULL : (dec_d_ <= 9.5 ? 100ULL : 300ULL);
            std::vector<u64> sm_shifts = {1ULL};
            for (u64 s : pz_.starts) {
                if (s > 1) sm_shifts.push_back(s);
            }
            for (size_t i = 0; i < n_miss; ++i) {
                u64 t = missing_[i];
                bool sm = is_smooth_bound(t, sm_bound);
                if (!sm) {
                    for (u64 c : sm_shifts) {
                        if ((t <= pz_.limit - c && is_smooth_bound(t + c, sm_bound)) ||
                            (t > c && is_smooth_bound(t - c, sm_bound))) {
                            sm = true;
                            break;
                        }
                    }
                }
                if (sm) {
                    is_easy[i] = 1;
                    easy_prio[i] += 35;
                }
            }

            // Collect high-value shared GCDs (dividing >= 2 targets and >= 1500, or >= 3 targets)
            gcd_counts.for_each([&](u64 g, int pairs) {
                int mult = 0;
                for (u64 t : missing_) {
                    if (t % g == 0) ++mult;
                }
                if (mult >= 2 && (g >= 1500ULL || mult >= 3 || pairs >= 2)) {
                    gcd_helpers.emplace_back(mult * 100 + static_cast<int>(log2_approx(g)), g);
                }
            });
            std::sort(gcd_helpers.begin(), gcd_helpers.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            if (gcd_helpers.size() > 3) gcd_helpers.resize(3);
        }

        // Determine initial upfront basis size:
        // Target-First Bootstrap (online_absorb=true, Restarts 0 & 2) seeds ONLY 10-14 fast
        // multi-scale steps, then solves 2-5 easiest targets first and absorbs their
        // intermediates directly into B up to final_basis (41..42)!
        // Restart 1 keeps the pure multi-scale upfront basis for struct<=1 puzzles.
        int init_basis = final_basis;
        if (online_absorb && n_miss >= 4) {
            int s_cnt = static_cast<int>(pz_.starts.size());
            if (n_miss <= 10) {
                init_basis = std::max(s_cnt + 3, (restart_idx == 0) ? 7 : 6);
            } else if (n_miss <= 18) {
                init_basis = std::max(s_cnt + 5, (restart_idx == 0) ? 10 : 9);
            } else {
                // Restart 0: tight ~12-element seed; Restart 2: slightly different ~14-element seed
                init_basis = (restart_idx == 0)
                                 ? std::max(s_cnt + 7, 12)
                                 : std::max(s_cnt + 9, 14);
            }
            init_basis = std::min(init_basis, final_basis);
        }

        // Phase 1: Build initial multi-scale basis B.
        build_initial_basis(restart_idx, init_basis);
        build_tables();

        if (opt_.verbose) {
            std::fprintf(stderr,
                         "[r%d] init |B|=%zu/%d (steps=%zu) |M|=%zu |H|=%zu absorb=%d gcds=%zu t=%.3fs\n",
                         restart_idx, basis_.size(), final_basis, basis_steps_.size(),
                         M_sorted_.size(), H_sorted_.size(), online_absorb ? 1 : 0,
                         gcd_helpers.size(), timer_.elapsed_sec());
        }

        FlatMap<uint8_t> covered(n_miss * 2 + 128);
        FlatMap<uint8_t> in_b(128);
        for (u64 s : pz_.starts) covered.insert(s, 1);
        for (u64 v : basis_) in_b.insert(v, 1);
        for (const Step& s : basis_steps_) covered.insert(s.r, 1);

        std::vector<Step> absorbed_prefix = basis_steps_;

        // Absorb helper lambda: adds useful scale-diverse intermediates from a solved step list into B.
        auto absorb_from_steps = [&](const std::vector<Step>& steps, bool allow_top_anchor) -> bool {
            std::vector<u64> newly_created;
            for (const Step& s : steps) {
                if (s.r >= 2 && s.r <= max_t_ && !in_b.contains(s.r)) {
                    newly_created.push_back(s.r);
                }
            }
            if (newly_created.empty()) return false;

            const int num_bins = std::max(4, static_cast<int>(dec_d_ * 2.0) + 8);
            std::vector<int> bins(static_cast<size_t>(num_bins), 0);
            for (u64 x : basis_) {
                int bk = std::min(num_bins - 1, log10_half_bucket(x));
                ++bins[bk];
            }
            int half_bk = std::max(1, static_cast<int>(dec_d_));
            int top_cap_bk = std::max(half_bk + 1, static_cast<int>(dec_d_ * 1.80));

            std::sort(newly_created.begin(), newly_created.end(),
                      [&](u64 a, u64 b) {
                          int bka = std::min(num_bins - 1, log10_half_bucket(a));
                          int bkb = std::min(num_bins - 1, log10_half_bucket(b));
                          double sa = -10.0 * bins[bka] - 0.45 * std::abs(bka - half_bk);
                          double sb = -10.0 * bins[bkb] - 0.45 * std::abs(bkb - half_bk);
                          if (bka > top_cap_bk) sa -= 35.0;
                          if (bkb > top_cap_bk) sb -= 35.0;
                          if (a >= 3 && a <= 64) sa += 4.0;
                          if (b >= 3 && b <= 64) sb += 4.0;
                          return sa > sb;
                      });

            bool added = false;
            for (u64 v : newly_created) {
                if (static_cast<int>(basis_.size()) >= final_basis) break;
                int bk = std::min(num_bins - 1, log10_half_bucket(v));
                // Avoid wasting basis slots on top-decade targets unless it's an anchor or bucket is empty
                if (bk > top_cap_bk && !allow_top_anchor && !is_anchor_target.contains(v) &&
                    bins[bk] >= 1) {
                    continue;
                }
                // Keep scale balance across half-decades while bootstrapping
                int per_bk_cap = (std::abs(bk - half_bk) <= 3) ? 4 : 3;
                if (bins[bk] >= per_bk_cap && static_cast<int>(basis_.size()) + 4 < final_basis) {
                    continue;
                }
                if (!in_b.insert(v, 1)) continue;
                basis_.push_back(v);
                ++bins[bk];
                added = true;
            }
            return added;
        };

        // Pre-seed high-value shared GCD helpers into B (only during online_absorb or when
        // they save steps across multiple targets).
        if (online_absorb && !gcd_helpers.empty() && static_cast<int>(basis_.size()) < final_basis) {
            bool added_gcd = false;
            for (const auto& [score_g, g] : gcd_helpers) {
                (void)score_g;
                if (in_b.contains(g) || static_cast<int>(basis_.size()) >= final_basis) continue;
                if (timer_.elapsed_sec() >= deadline_ - 0.08) break;
                double g_dl = std::min(deadline_ - 0.05, timer_.elapsed_sec() + 0.03);
                std::vector<Step> g_plan = solve_target(g, g_dl, 0, true);
                if (!g_plan.empty() && g_plan.size() <= 8) {
                    for (const Step& s : g_plan) {
                        absorbed_prefix.push_back(s);
                        covered.insert(s.r, 1);
                    }
                    // Ensure g itself goes into B first
                    if (in_b.insert(g, 1)) {
                        basis_.push_back(g);
                        added_gcd = true;
                    }
                    if (absorb_from_steps(g_plan, false)) added_gcd = true;
                }
            }
            if (added_gcd) build_tables();
        }

        // Build target solving order:
        //   1. Quick cheapness probe (smallest solve_c3_hb / B-divisibility) + structured targets first!
        //   2. Keep close/affine partner hi IMMEDIATELY after anchor lo so hi gets the anchor shortcut!
        std::vector<u64> order;
        order.reserve(n_miss);
        {
            std::vector<size_t> idxs(n_miss);
            std::iota(idxs.begin(), idxs.end(), 0);

            std::vector<double> t_rank(n_miss, 0.0);
            for (size_t i = 0; i < n_miss; ++i) {
                u64 t = missing_[i];
                double rk = 0.0;
                if (online_absorb) {
                    rk -= static_cast<double>(easy_prio[i]) * 2.0;
                    // Bonus if directly cheap or divisible by a basis element >= 3
                    std::vector<Step> tmp;
                    int c3 = solve_c3_hb(t, 3, tmp);
                    if (c3 >= 0) rk -= 120.0 - c3 * 15.0;
                    u64 best_div = 1;
                    for (u64 b : basis_) {
                        if (b >= 3 && b < t && t % b == 0 && b > best_div) best_div = b;
                    }
                    if (best_div > 1) rk -= 4.0 * log2_approx(best_div);
                    // Prefer slightly smaller targets first while basis is small
                    rk += 0.35 * log2_approx(t);
                    if (restart_idx >= 2) {
                        rk += static_cast<double>(rng_() & 0xFFULL) / 256.0 * 3.0;
                    }
                } else {
                    rk = static_cast<double>(i);
                }
                t_rank[i] = rk;
            }

            std::sort(idxs.begin(), idxs.end(),
                      [&](size_t a, size_t b) { return t_rank[a] < t_rank[b]; });

            // Emit order ensuring every close/affine partner follows its anchor immediately.
            FlatMap<uint8_t> emitted(n_miss * 2 + 16);
            for (size_t idx : idxs) {
                u64 t = missing_[idx];
                if (emitted.contains(t)) continue;
                // If t has an unemitted anchor, emit the anchor first!
                if (const u64* anc = close_anchor.find(t)) {
                    if (!emitted.contains(*anc)) {
                        emitted.insert(*anc, 1);
                        order.push_back(*anc);
                    }
                }
                emitted.insert(t, 1);
                order.push_back(t);
                // Immediately pull in any targets anchored on t!
                for (size_t j = 0; j < n_miss; ++j) {
                    u64 tj = missing_[j];
                    if (emitted.contains(tj)) continue;
                    if (const u64* anc = close_anchor.find(tj)) {
                        if (*anc == t) {
                            emitted.insert(tj, 1);
                            order.push_back(tj);
                        }
                    }
                }
            }
        }

        std::vector<std::vector<Step>> post_target_steps;
        post_target_steps.reserve(order.size());

        int targets_solved = 0;
        bool basis_frozen = (static_cast<int>(basis_.size()) >= final_basis);
        u64 last_emitted_target = 0;

        for (size_t ti = 0; ti < order.size(); ++ti) {
            u64 t = order[ti];
            if (covered.contains(t)) {
                last_emitted_target = t;
                continue;
            }

            // Count remaining unsolved targets to compute fair per-target time slice!
            size_t rem_unsolved = 0;
            for (size_t k = ti; k < order.size(); ++k) {
                if (!covered.contains(order[k])) ++rem_unsolved;
            }
            double now = timer_.elapsed_sec();
            double reserve = 0.04;
            double work_rem = std::max(0.0, (deadline_ - reserve) - now);
            double t_slice = work_rem / static_cast<double>(std::max<size_t>(1, rem_unsolved));
            // Give bootstrap targets slightly less time when |B| is tiny, and never exceed slice
            double target_deadline = now + std::max(0.004, t_slice);

            // Check if t has a live close/affine anchor (in B or immediately preceding target)
            u64 anchor = 0;
            if (const u64* anc = close_anchor.find(t)) {
                if (in_b.contains(*anc) || last_emitted_target == *anc) {
                    anchor = *anc;
                }
            }

            std::vector<Step> t_plan = solve_target(t, target_deadline, anchor, !basis_frozen);
            for (const Step& s : t_plan) covered.insert(s.r, 1);
            last_emitted_target = t;
            ++targets_solved;

            if (!basis_frozen) {
                // Append this target's steps into the prefix, and absorb its best
                // intermediates into B (zero extra step cost!).
                for (const Step& s : t_plan) absorbed_prefix.push_back(s);

                bool added_any = absorb_from_steps(t_plan, false);

                // Absorb across the first 2-5 targets until B reaches final_basis,
                // and ONLY THEN fill any missing half-decade buckets up to final_basis!
                int max_boot_targets = (n_miss >= 18) ? 5 : 4;
                bool ready_to_freeze =
                    static_cast<int>(basis_.size()) >= final_basis ||
                    targets_solved >= max_boot_targets ||
                    (targets_solved >= 3 && static_cast<int>(basis_.size()) >= final_basis - 4);

                if (ready_to_freeze) {
                    if (static_cast<int>(basis_.size()) < final_basis && n_miss >= 18) {
                        int rem_fill = std::min(final_basis - static_cast<int>(basis_.size()), 8);
                        extend_basis_greedy(restart_idx, static_cast<int>(basis_.size()) + rem_fill,
                                            absorbed_prefix, in_b);
                        added_any = true;
                    } else if (static_cast<int>(basis_.size()) < final_basis && n_miss >= 11) {
                        int rem_fill = std::min(final_basis - static_cast<int>(basis_.size()), 3);
                        extend_basis_greedy(restart_idx, static_cast<int>(basis_.size()) + rem_fill,
                                            absorbed_prefix, in_b);
                        added_any = true;
                    }
                    basis_frozen = true;
                }

                if (added_any && ti + 1 < order.size()) {
                    build_tables();
                    if (opt_.verbose && basis_frozen) {
                        std::fprintf(stderr,
                                     "[r%d] froze |B|=%zu (prefix=%zu after %d targets) |M|=%zu |H|=%zu t=%.3fs\n",
                                     restart_idx, basis_.size(), absorbed_prefix.size(),
                                     targets_solved, M_sorted_.size(), H_sorted_.size(),
                                     timer_.elapsed_sec());
                    }
                }
            } else {
                post_target_steps.push_back(std::move(t_plan));
            }
        }

        // Assemble full plan and run scratch promotion + verify_and_prune_plan.
        return assemble_and_optimize(absorbed_prefix, post_target_steps);
    }

private:
    const Puzzle& pz_;
    const Options& opt_;
    const Timer& timer_;
    double deadline_;
    std::mt19937_64 rng_;

    std::vector<u64> missing_;
    u64 max_t_ = 1;
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
    // Phase 1: Deliberate Multi-Scale Dense Basis Builder
    // -------------------------------------------------------------------------
    void build_initial_basis(int restart_idx, int target_basis) {
        basis_.clear();
        basis_steps_.clear();

        int max_allowed = opt_.regs > 0 ? std::max(4, opt_.regs - 5) : 43;
        int cap = std::min(target_basis, max_allowed);

        FlatMap<uint8_t> in_b(128);
        std::vector<u64> uniq_starts;
        for (u64 s : pz_.starts) {
            if (s >= 1 && s <= pz_.limit && in_b.insert(s, 1)) {
                uniq_starts.push_back(s);
            }
        }
        in_b.clear();
        if (static_cast<int>(uniq_starts.size()) > cap) {
            std::sort(uniq_starts.begin(), uniq_starts.end());
            for (int i = 0; i < cap; ++i) {
                size_t pos = (uniq_starts.size() - 1) * static_cast<size_t>(i) /
                             static_cast<size_t>(std::max(1, cap - 1));
                u64 v = uniq_starts[pos];
                if (in_b.insert(v, 1)) basis_.push_back(v);
            }
        } else {
            for (u64 v : uniq_starts) {
                if (in_b.insert(v, 1)) basis_.push_back(v);
            }
        }

        // Ensure 1 is in B (s0 / s0 = 1).
        if (!in_b.contains(1) && !basis_.empty() && static_cast<int>(basis_.size()) < cap) {
            u64 s0 = basis_[0];
            basis_steps_.push_back(Step{'/', s0, s0, 1});
            basis_.push_back(1);
            in_b.insert(1, 1);
        }

        // Ensure 2 is in B if limit >= 2.
        if (!in_b.contains(2) && in_b.contains(1) && pz_.limit >= 2 &&
            static_cast<int>(basis_.size()) < cap) {
            basis_steps_.push_back(Step{'+', 1, 1, 2});
            basis_.push_back(2);
            in_b.insert(2, 1);
        }

        extend_basis_greedy(restart_idx, cap, basis_steps_, in_b);
    }

    void extend_basis_greedy(int restart_idx, int cap, std::vector<Step>& out_steps,
                             FlatMap<uint8_t>& in_b) {
        FlatMap<uint8_t> target_set(missing_.size() * 2 + 8);
        for (u64 t : missing_) target_set.insert(t, 1);

        int mode = restart_idx % 3;
        int max_bk = std::max(2, static_cast<int>(dec_d_ * 2.0));
        int half_bk = std::max(1, static_cast<int>(dec_d_));

        FlatMap<Step> l1_map(4096);
        std::vector<u64> l1_sorted;
        std::vector<int> bins(static_cast<size_t>(max_bk + 16), 0);

        while (static_cast<int>(basis_.size()) < cap) {
            build_L1(basis_, pz_.limit, l1_map, l1_sorted);
            if (l1_sorted.empty()) break;

            std::fill(bins.begin(), bins.end(), 0);
            for (u64 x : basis_) {
                int bk = std::min(static_cast<int>(bins.size()) - 1, log10_half_bucket(x));
                ++bins[bk];
            }

            // Stratified candidate sampling across half-decade buckets.
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
                            if (rbk >= max_bk - 2) w = 1.45;  // top decade helps b*x +- y!
                            else if (dist_half <= 3) w = (mode == 2) ? 1.65 : 1.35;
                            else if (rbk <= half_bk) w = (mode == 1) ? 1.35 : 1.15;
                            else w = 1.1;
                        } else {
                            w = 0.2;
                        }
                        if (target_set.contains(r)) w += 40.0;
                        l1_gain += w;
                    };

                    if (v <= pz_.limit && b <= pz_.limit - v) score_new(v + b);
                    u64 prod = safe_mul(v, b, pz_.limit);
                    if (prod != 0) score_new(prod);
                    if (v > b) {
                        score_new(v - b);
                        if (v % b == 0) score_new(v / b);
                    } else if (b > v) {
                        score_new(b - v);
                        if (b % v == 0) score_new(b / v);
                    }
                }

                // Cover [0 .. half_bk] densely AND put a few elements in (half_bk .. 1.8*half_bk]
                // so H2 populates BOTH [1..sqrt(maxT)] AND [0.1*maxT..maxT]!
                double bucket_pref = 0.0;
                if (v > max_t_) {
                    bucket_pref = -45.0;
                } else if (bk <= half_bk) {
                    bucket_pref = (mode == 1) ? 10.0 : 7.0;
                } else if (bk <= static_cast<int>(dec_d_ * 1.75)) {
                    bucket_pref = (mode == 2) ? 11.0 : 5.5;
                } else {
                    bucket_pref = -4.0;
                }

                // Ensure a handful of small multipliers & remainders in [3..25]
                if (v >= 3 && v <= 25 && bins[0] + bins[1] < 5) {
                    bucket_pref += 16.0;
                }

                double t_bonus = 0.0;
                if (v >= 3 && v <= max_t_) {
                    int exact_divs = 0;
                    for (u64 t : missing_) {
                        if (t % v == 0) ++exact_divs;
                    }
                    t_bonus += exact_divs * 6.0;
                }

                double bucket_pen = (mode == 2 ? 12.0 : 10.5) * static_cast<double>(bins[bk]);
                double noise = static_cast<double>(rng_() & 0xFFFFULL) / 65536.0 * 3.5;
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
    // Precompute M = B U L1(B) and H = B U L1(B) U L2(B)
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
                u64 prod = safe_mul(u, v, pz_.limit);
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
            u64 sq = safe_mul(x, x, pz_.limit);
            if (sq != 0) add_l2('*', i, i, sq);
        }

        for (uint32_t i = B_sz; i < M_sz && H_vec_.size() < kMaxH; ++i) {
            u64 x = H_vec_[i].val;
            for (uint32_t j = 0; j < B_sz; ++j) {
                u64 b = H_vec_[j].val;
                if (x <= pz_.limit - b) add_l2('+', i, j, x + b);
                u64 prod = safe_mul(x, b, pz_.limit);
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

    // -------------------------------------------------------------------------
    // Ultra-fast O(|B|) cost-<=3 checker: V in H, self-op(H), or H op B.
    // Takes ~0.5 microseconds (Bloom-filtered); safe to run on 1,500+ residuals!
    // Returns step count (>= 0) or -1.
    // -------------------------------------------------------------------------
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
            u64 s = isqrt64(V);
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
                u64 prod = safe_mul(V, b, pz_.limit);
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

    // -------------------------------------------------------------------------
    // Full sub-solver up to max_cost (3..7). Returns step count (>= 0) or -1.
    // -------------------------------------------------------------------------
    int solve_sub(u64 V, int max_cost, bool allow_two_sum, bool allow_m_x_h,
                  std::vector<Step>& out) const {
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
                u64 prod = safe_mul(V, x, pz_.limit);
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
        if (best_c > 4 && max_cost >= 4) {
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
                        u64 prod = safe_mul(V2, b1, pz_.limit);
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
                    u64 prod = safe_mul(V, b2, pz_.limit);
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
                if (max_cost == 4 && mx.cost > 0) continue;  // fast path for cost 4: B * H +- B
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

    // Check V = (M op H) op B in <= 5 steps.
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
                    u64 prod = safe_mul(V2, x, pz_.limit);
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
                u64 prod = safe_mul(V, b, pz_.limit);
                if (prod != 0) try_inner(prod, bi, '/', true);
            }
        }

        if (!best_steps.empty() && best_c <= max_cost) {
            out = std::move(best_steps);
            return best_c;
        }
        return -1;
    }

    // -------------------------------------------------------------------------
    // Phase 2: Linear-Combo Two-Pointer Scan: V = (b * x +- y) +- r
    //   with b in B (cost 0), x, y in H (cost <= 2), r in {0} U H U (H op B).
    // Strictly enforces `local_dl` inside outer and inner loops.
    // -------------------------------------------------------------------------
    template <typename WrapFn>
    void scan_linear_bxy(u64 V, int& budget, int max_b_count, bool collect_res,
                         double local_dl, WrapFn&& wrap_and_update, int& best_cost_ref,
                         int extra_cost) const {
        if (budget < 4 || H_sorted_.empty() || local_time_up(local_dl)) return;

        // Pick diverse basis multipliers b in B with b >= 3 and b <= V.
        // Order largest b first so each scan touches fewer H_sorted_ entries and uses cheaper x!
        std::vector<std::pair<u64, uint32_t>> b_list;
        b_list.reserve(basis_.size());
        for (uint32_t bi = 0; bi < basis_.size(); ++bi) {
            u64 b = basis_[bi];
            if (b >= 3 && b <= V) b_list.emplace_back(b, bi);
        }
        std::sort(b_list.begin(), b_list.end());

        // Select up to max_b_count well-spread values across b_list, then reverse (largest first)
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
        std::reverse(picked_b.begin(), picked_b.end());

        const size_t Hn = H_sorted_.size();
        const size_t max_keep_lin = collect_res ? 140 : 0;
        const u64 max_lin_r = V / 64ULL;
        std::array<std::vector<LinearResCand>, 8> lin_buckets;
        std::array<u64, 8> lin_thresh;
        lin_thresh.fill(std::numeric_limits<u64>::max());

        auto push_lin = [&](u64 r, uint32_t bi, uint32_t xi, uint32_t yi,
                            uint8_t kind, char r_sign, uint8_t oc) {
            if (!collect_res || oc > 7 || oc + 3 > budget || r >= max_lin_r) return;
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

        for (const auto& [b, bi] : picked_b) {
            if (budget < 4 || local_time_up(local_dl)) break;
            u64 max_x_lim = pz_.limit / b;
            u64 max_x_v = (V - 1) / b;  // b * x < V

            // Case 0: b * x + y ≈ V (with b * x < V)
            {
                size_t j = Hn - 1;
                for (size_t i = 0; i < Hn; ++i) {
                    if ((i & 4095u) == 0 && local_time_up(local_dl)) break;
                    u64 x = H_sorted_[i].val;
                    if (x > max_x_v) break;
                    u64 bx = b * x;
                    u64 need = V - bx;
                    uint8_t cx = H_sorted_[i].cost;
                    while (j > 0 && H_sorted_[j].val > need) --j;

                    // y_lo = H_sorted_[j] (bx + y_lo <= V)
                    {
                        u64 y = H_sorted_[j].val;
                        if (y <= need) {
                            u64 r = need - y;
                            uint8_t cy = H_sorted_[j].cost;
                            if (r == 0) {
                                if (cx + cy + 2 <= budget) {
                                    emit_bxy_r(bi, H_sorted_[i].idx, H_sorted_[j].idx, 0, '+', UINT32_MAX);
                                }
                            } else {
                                uint8_t oc = static_cast<uint8_t>(cx + cy + 3);
                                if (oc <= budget) {
                                    if (const uint32_t* ri = find_H(r)) {
                                        if (oc + H_vec_[*ri].cost <= budget + 1) {
                                            emit_bxy_r(bi, H_sorted_[i].idx, H_sorted_[j].idx, 0, '+', *ri);
                                        }
                                    } else if (oc + 3 <= budget) {
                                        push_lin(r, bi, H_sorted_[i].idx, H_sorted_[j].idx, 0, '+', oc);
                                    }
                                }
                            }
                        }
                    }
                    // y_hi = H_sorted_[j + 1] (bx + y_hi > V, <= limit)
                    if (j + 1 < Hn) {
                        u64 y = H_sorted_[j + 1].val;
                        if (bx <= pz_.limit - y && bx + y > V) {
                            u64 r = (bx + y) - V;
                            uint8_t cy = H_sorted_[j + 1].cost;
                            uint8_t oc = static_cast<uint8_t>(cx + cy + 3);
                            if (oc <= budget) {
                                if (const uint32_t* ri = find_H(r)) {
                                    if (oc + H_vec_[*ri].cost <= budget + 1) {
                                        emit_bxy_r(bi, H_sorted_[i].idx, H_sorted_[j + 1].idx, 0, '-', *ri);
                                    }
                                } else if (oc + 3 <= budget) {
                                    push_lin(r, bi, H_sorted_[i].idx, H_sorted_[j + 1].idx, 0, '-', oc);
                                }
                            }
                        }
                    }
                }
            }

            // Case 1: b * x - y ≈ V (with V < b * x <= limit)
            if (max_x_lim > max_x_v && budget >= 4 && !local_time_up(local_dl)) {
                auto it_start = std::upper_bound(
                    H_sorted_.begin(), H_sorted_.end(), max_x_v,
                    [](u64 val, const ValIdx& a) { return val < a.val; });
                size_t i_start = static_cast<size_t>(it_start - H_sorted_.begin());
                size_t j = 0;
                for (size_t i = i_start; i < Hn; ++i) {
                    if (((i - i_start) & 4095u) == 0 && local_time_up(local_dl)) break;
                    u64 x = H_sorted_[i].val;
                    if (x > max_x_lim) break;
                    u64 bx = b * x;
                    if (bx <= V) continue;
                    u64 need = bx - V;
                    uint8_t cx = H_sorted_[i].cost;
                    while (j + 1 < Hn && H_sorted_[j + 1].val <= need) ++j;

                    // y_lo = H_sorted_[j] (<= need => bx - y_lo >= V)
                    if (H_sorted_[j].val <= need) {
                        u64 y = H_sorted_[j].val;
                        u64 r = need - y;  // (bx - y) - V
                        uint8_t cy = H_sorted_[j].cost;
                        if (r == 0) {
                            if (cx + cy + 2 <= budget) {
                                emit_bxy_r(bi, H_sorted_[i].idx, H_sorted_[j].idx, 1, '-', UINT32_MAX);
                            }
                        } else {
                            uint8_t oc = static_cast<uint8_t>(cx + cy + 3);
                            if (oc <= budget) {
                                if (const uint32_t* ri = find_H(r)) {
                                    if (oc + H_vec_[*ri].cost <= budget + 1) {
                                        emit_bxy_r(bi, H_sorted_[i].idx, H_sorted_[j].idx, 1, '-', *ri);
                                    }
                                } else if (oc + 3 <= budget) {
                                    push_lin(r, bi, H_sorted_[i].idx, H_sorted_[j].idx, 1, '-', oc);
                                }
                            }
                        }
                    }
                    // y_hi = H_sorted_[j + 1] (> need, < bx => bx - y_hi < V)
                    if (j + 1 < Hn && H_sorted_[j + 1].val < bx && H_sorted_[j + 1].val > need) {
                        u64 y = H_sorted_[j + 1].val;
                        u64 r = y - need;  // V - (bx - y)
                        uint8_t cy = H_sorted_[j + 1].cost;
                        uint8_t oc = static_cast<uint8_t>(cx + cy + 3);
                        if (oc <= budget) {
                            if (const uint32_t* ri = find_H(r)) {
                                if (oc + H_vec_[*ri].cost <= budget + 1) {
                                    emit_bxy_r(bi, H_sorted_[i].idx, H_sorted_[j + 1].idx, 1, '+', *ri);
                                }
                            } else if (oc + 3 <= budget) {
                                push_lin(r, bi, H_sorted_[i].idx, H_sorted_[j + 1].idx, 1, '+', oc);
                            }
                        }
                    }
                }
            }
        }

        // Always harvest top collected residuals from b*x +- y via ultra-fast solve_c3_hb (~0.15 ms!)
        if (collect_res && budget >= 7) {
            double res_dl = std::min(deadline_ - 0.03, timer_.elapsed_sec() + 0.0015);
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
                          return (a.oc + 0.16 * log2_approx(a.r)) < (c.oc + 0.16 * log2_approx(c.r));
                      });

            for (size_t k = 0; k < top_lin.size() && k < 320; ++k) {
                if ((k & 31u) == 0 && local_time_up(res_dl)) break;
                const auto& lc = top_lin[k];
                int rem_b = budget - static_cast<int>(lc.oc);
                if (rem_b < 3) continue;
                std::vector<Step> r_steps;
                int rc = solve_c3_hb(lc.r, std::min(rem_b, 3), r_steps);
                if (rc < 0 && rem_b >= 4 && k < 12 && !local_time_up(local_dl)) {
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
    // Fused Single-Pass Beam Base-Case + Expansion Solver (costs 0..7, ~90 us!):
    //   1) H, self-op(H), H op B (cost <= 3)
    //   2) SINGLE fused pass over M_sorted_:
    //      - M op H (+, -, and guarded * / without extra divisions)
    //      - O(1) H * M +- (0 / H) + collects small H*M remainders (out_rems)
    //      - Quotient moves V = q * x +- r (out_qs) reusing the SAME q0 = V / x!
    //   3) Fast 2-way Two-Pointer M * M +- (0 / H / solve_c3_hb(r)) + M*M remainders
    //   4) Fast Nearby H +- H (cost 5)
    // -------------------------------------------------------------------------
    int solve_beam_base(u64 V, int max_cost, std::vector<Step>& out,
                        std::vector<RemBranch>* out_rems,
                        std::vector<BeamCand>* out_qs = nullptr,
                        int expand_budget = 0) const {
        out.clear();
        if (out_rems) out_rems->clear();
        if (out_qs) out_qs->clear();
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

        const u64 rem_cap_16 = V / 16ULL;
        const u64 rem_cap_8 = V / 8ULL;
        const u64 max_mul_x = (V > 0) ? (std::min(pz_.limit, H_max_val_) / V) : 0ULL;
        const bool want_expand = (out_qs != nullptr || out_rems != nullptr);

        // SINGLE FUSED PASS over M_sorted_: M op H + H * M +- (0/H) + Quotient & H*M expansion!
        for (const ValIdx& mx : M_sorted_) {
            u64 x = mx.val;
            if (x < 2) continue;

            // (a) M op H for cost <= 4 (add/sub + guarded mul/div, no extra division!)
            if (mx.cost > 0) {
                int rem_c = best_c - 1 - static_cast<int>(mx.cost) - 1;
                if (rem_c >= 0) {
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
                        if (x >= V * 2ULL && x % V == 0) check_other(x / V, '/', true);
                    }
                    if (x <= max_mul_x) {
                        check_other(V * x, '/', false);
                    }
                }
            }

            if (x >= V) continue;

            // (b) Compute q0 = V / x, r0 = V % x ONCE and reuse for H*M AND Quotient expansion!
            int base_x = static_cast<int>(mx.cost) + 1;
            if (base_x >= best_c && !want_expand) continue;
            u64 q0 = V / x;
            u64 r0 = V - q0 * x;

            if (r0 == 0) {
                if (const uint32_t* qi = find_H(q0)) {
                    int bc = static_cast<int>(H_vec_[*qi].cost) + base_x;
                    if (bc <= best_c) accept_op(*qi, mx.idx, '*', V);
                }
                if (out_qs && base_x < expand_budget) {
                    out_qs->push_back(BeamCand{q0, mx.idx, UINT32_MAX, '*',
                                               static_cast<uint8_t>(base_x), 0.0});
                }
            } else {
                const uint32_t* ri0 = find_H(r0);
                if (ri0) {
                    if (out_qs && q0 >= 2) {
                        uint8_t cr0 = H_vec_[*ri0].cost;
                        if (cr0 <= 1 || x >= 512ULL) {
                            int oc = base_x + static_cast<int>(cr0) + 1;
                            if (oc < expand_budget) {
                                out_qs->push_back(
                                    BeamCand{q0, mx.idx, *ri0, '+', static_cast<uint8_t>(oc), 0.0});
                            }
                        }
                    }
                    if (max_cost >= 4) {
                        if (const uint32_t* qi = find_H(q0)) {
                            int bc = static_cast<int>(H_vec_[*qi].cost) + base_x;
                            if (bc + static_cast<int>(H_vec_[*ri0].cost) + 1 <= best_c) {
                                LocalEmitter em(H_vec_, basis_);
                                em.emit_node(*qi);
                                em.emit_node(mx.idx);
                                u64 prod = q0 * x;
                                em.emit_op('*', q0, x, prod);
                                em.emit_node(*ri0);
                                em.emit_op('+', prod, r0, V);
                                int sz = static_cast<int>(em.steps.size());
                                if (sz < best_c) {
                                    best_c = sz;
                                    best_steps = std::move(em.steps);
                                }
                            }
                        }
                    }
                } else if (out_rems && r0 < rem_cap_16 && base_x + 2 < expand_budget) {
                    if (const uint32_t* qi = find_H(q0)) {
                        int oc = static_cast<int>(H_vec_[*qi].cost) + base_x + 1;
                        if (oc < expand_budget) {
                            out_rems->push_back(
                                RemBranch{r0, *qi, mx.idx, '+', static_cast<uint8_t>(oc)});
                        }
                    }
                }

                if (static_cast<u128>(x) * (q0 + 1) <= pz_.limit) {
                    u64 q1 = q0 + 1;
                    u64 r1 = x - r0;
                    const uint32_t* ri1 = find_H(r1);
                    if (ri1) {
                        if (out_qs) {
                            uint8_t cr1 = H_vec_[*ri1].cost;
                            if (cr1 <= 1 || x >= 512ULL) {
                                int oc = base_x + static_cast<int>(cr1) + 1;
                                if (oc < expand_budget) {
                                    out_qs->push_back(BeamCand{
                                        q1, mx.idx, *ri1, '-', static_cast<uint8_t>(oc), 0.0});
                                }
                            }
                        }
                        if (max_cost >= 4) {
                            if (const uint32_t* qi = find_H(q1)) {
                                int bc = static_cast<int>(H_vec_[*qi].cost) + base_x;
                                if (bc + static_cast<int>(H_vec_[*ri1].cost) + 1 <= best_c) {
                                    LocalEmitter em(H_vec_, basis_);
                                    em.emit_node(*qi);
                                    em.emit_node(mx.idx);
                                    u64 prod = q1 * x;
                                    em.emit_op('*', q1, x, prod);
                                    em.emit_node(*ri1);
                                    em.emit_op('-', prod, r1, V);
                                    int sz = static_cast<int>(em.steps.size());
                                    if (sz < best_c) {
                                        best_c = sz;
                                        best_steps = std::move(em.steps);
                                    }
                                }
                            }
                        }
                    } else if (out_rems && r1 < rem_cap_16 && base_x + 2 < expand_budget) {
                        if (const uint32_t* qi = find_H(q1)) {
                            int oc = static_cast<int>(H_vec_[*qi].cost) + base_x + 1;
                            if (oc < expand_budget) {
                                out_rems->push_back(
                                    RemBranch{r1, *qi, mx.idx, '-', static_cast<uint8_t>(oc)});
                            }
                        }
                    }
                }
            }
        }

        if (!best_steps.empty() && best_c <= 3) {
            out = std::move(best_steps);
            return best_c;
        }
        if (max_cost <= 3) return -1;

        // Cost 2..7: Fast 2-way Two-Pointer M * M +- (0 / H / solve_c3_hb(r)) + small remainders!
        std::vector<RemBranch> mm_rems;
        if (max_cost >= 4 && !M_sorted_.empty()) {
            mm_rems.reserve(96);
            const size_t Mn = M_sorted_.size();
            size_t i = 0;
            size_t j = Mn - 1;
            while (i <= j) {
                u64 xi = M_sorted_[i].val;
                if (xi < 2) {
                    ++i;
                    continue;
                }
                while (j > i && static_cast<u128>(xi) * M_sorted_[j].val > V) --j;
                uint8_t ci = M_sorted_[i].cost;
                uint32_t ui = M_sorted_[i].idx;
                for (int dk = 0; dk <= 1; ++dk) {
                    size_t k = j + static_cast<size_t>(dk);
                    if (k < i || k >= Mn) continue;
                    u64 yk = M_sorted_[k].val;
                    if (yk < 2) continue;
                    u128 p128 = static_cast<u128>(xi) * yk;
                    if (p128 > pz_.limit) continue;
                    u64 prod = static_cast<u64>(p128);
                    int base = static_cast<int>(ci) + static_cast<int>(M_sorted_[k].cost) + 1;
                    if (prod == V) {
                        if (base <= best_c) accept_op(ui, M_sorted_[k].idx, '*', V);
                        continue;
                    }
                    u64 r = prod > V ? (prod - V) : (V - prod);
                    char sgn = (prod > V) ? '-' : '+';
                    if (const uint32_t* ri = find_H(r)) {
                        if (base + static_cast<int>(H_vec_[*ri].cost) + 1 <= best_c) {
                            LocalEmitter em(H_vec_, basis_);
                            em.emit_node(ui);
                            em.emit_node(M_sorted_[k].idx);
                            em.emit_op('*', xi, yk, prod);
                            em.emit_node(*ri);
                            em.emit_op(sgn, prod, r, V);
                            int sz = static_cast<int>(em.steps.size());
                            if (sz < best_c) {
                                best_c = sz;
                                best_steps = std::move(em.steps);
                            }
                        }
                    } else if (r < rem_cap_8) {
                        mm_rems.push_back(
                            RemBranch{r, ui, M_sorted_[k].idx, sgn, static_cast<uint8_t>(base + 1)});
                    }
                }
                ++i;
            }
        }

        if (!mm_rems.empty()) {
            std::sort(mm_rems.begin(), mm_rems.end(),
                      [](const RemBranch& a, const RemBranch& b) {
                          return a.r != b.r ? a.r < b.r : a.oc < b.oc;
                      });
            std::vector<RemBranch> uniq_mm;
            uniq_mm.reserve(20);
            u64 last_r = 0;
            for (const auto& mr : mm_rems) {
                if (mr.r == last_r) continue;
                last_r = mr.r;
                uniq_mm.push_back(mr);
                if (uniq_mm.size() >= 18) break;
            }
            // 2-level MITM: M * M +- solve_c3_hb(r) (up to cost 6..7 in ~8 us!)
            if (best_c > 5) {
                for (const auto& mr : uniq_mm) {
                    int rem_cap = std::min(3, best_c - static_cast<int>(mr.oc) - 1);
                    if (rem_cap < 3) continue;
                    std::vector<Step> r_steps;
                    if (solve_c3_hb(mr.r, rem_cap, r_steps) >= 0) {
                        u64 xi = H_vec_[mr.u_idx].val;
                        u64 yk = H_vec_[mr.v_idx].val;
                        u64 prod = xi * yk;
                        LocalEmitter em(H_vec_, basis_);
                        em.emit_node(mr.u_idx);
                        em.emit_node(mr.v_idx);
                        em.emit_op('*', xi, yk, prod);
                        em.append_steps(r_steps);
                        em.emit_op(mr.sign, prod, mr.r, V);
                        int sz = static_cast<int>(em.steps.size());
                        if (sz < best_c) {
                            best_c = sz;
                            best_steps = std::move(em.steps);
                        }
                    }
                }
            }
            if (out_rems) {
                for (const auto& mr : uniq_mm) {
                    if (mr.r < rem_cap_16) out_rems->push_back(mr);
                }
            }
        }

        // Cost 5: Fast Nearby H +- H
        if (best_c > 5 && max_cost >= 5 && !H_sorted_.empty()) {
            auto it = std::lower_bound(
                H_sorted_.begin(), H_sorted_.end(), V,
                [](const ValIdx& a, u64 val) { return a.val < val; });
            for (int d = -16; d <= 16; ++d) {
                ptrdiff_t pos = (it - H_sorted_.begin()) + d;
                if (pos < 0 || pos >= static_cast<ptrdiff_t>(H_sorted_.size())) continue;
                u64 h = H_sorted_[pos].val;
                if (h == V) continue;
                u64 r = (h > V) ? (h - V) : (V - h);
                if (const uint32_t* ri = find_H(r)) {
                    if (static_cast<int>(H_sorted_[pos].cost) +
                            static_cast<int>(H_vec_[*ri].cost) + 1 <=
                        best_c) {
                        accept_op(H_sorted_[pos].idx, *ri, (h > V) ? '-' : '+', V);
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

    // -------------------------------------------------------------------------
    // Phase 2: Full Target Solver with Per-Target Time Slicing & 3-Layer Beam
    // -------------------------------------------------------------------------
    std::vector<Step> solve_target(u64 target, double target_deadline, u64 prev_anchor,
                                   bool is_bootstrap) const {
        std::vector<Step> best_steps;
        int best_cost = std::numeric_limits<int>::max();

        auto update_best = [&](std::vector<Step>&& cand) {
            if (!cand.empty() && static_cast<int>(cand.size()) < best_cost) {
                best_cost = static_cast<int>(cand.size());
                best_steps = std::move(cand);
            }
        };

        // 0. Close / Affine Anchor Shortcut (e.g., t2 = t1 + solve(t2 - t1)):
        if (prev_anchor >= 2 && prev_anchor < target) {
            u64 diff = target - prev_anchor;
            if (static_cast<u128>(diff) * 500ULL < prev_anchor) {
                double anc_dl = std::min(target_deadline, timer_.elapsed_sec() + 0.012);
                std::vector<Step> d_steps = solve_target(diff, anc_dl, 0, true);
                if (!d_steps.empty()) {
                    d_steps.push_back(Step{'+', prev_anchor, diff, target});
                    update_best(std::move(d_steps));
                }
            } else {
                u64 q = (target + prev_anchor / 2) / prev_anchor;
                if (q >= 2 && q <= 256) {
                    u128 p128 = static_cast<u128>(q) * prev_anchor;
                    if (p128 <= pz_.limit) {
                        u64 prod = static_cast<u64>(p128);
                        u64 d = (prod > target) ? (prod - target) : (target - prod);
                        if (d <= std::max<u64>(5000ULL, prev_anchor / 100000ULL)) {
                            const uint32_t* qi = find_H(q);
                            if (qi && H_vec_[*qi].cost <= 2) {
                                LocalEmitter em(H_vec_, basis_);
                                em.emit_node(*qi);
                                em.emit_op('*', prev_anchor, q, prod);
                                if (d > 0) {
                                    double anc_dl = std::min(target_deadline, timer_.elapsed_sec() + 0.010);
                                    std::vector<Step> d_steps = solve_target(d, anc_dl, 0, true);
                                    em.append_steps(d_steps);
                                    em.emit_op((prod > target) ? '-' : '+', prod, d, target);
                                }
                                update_best(std::move(em.steps));
                            }
                        }
                    }
                }
            }
        }

        // 1. Direct fast check on target itself (cost <= 6 via solve_beam_base).
        std::vector<RemBranch> d1_base_rems;
        {
            std::vector<Step> sub;
            int c = solve_beam_base(target, std::min(best_cost - 1, 6), sub, &d1_base_rems);
            if (c >= 0) update_best(std::move(sub));
            if (best_cost <= 3) return best_steps;
        }

        // 2. Multiplier lottery list: d = 1, then d in B, then small d in L1.
        struct MultCand {
            u64 d;
            uint32_t d_idx;
            int extra;  // 0 for d=1, 1 for d in B, 2 for d in L1
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
                if (mults.size() >= 18) break;
                mults.push_back(mc);
            }
            for (const ValIdx& mx : M_sorted_) {
                if (mx.val < 2 || mx.cost == 0) continue;
                if (mx.val > max_d) break;
                mults.push_back(MultCand{mx.val, mx.idx, 2});
                if (mults.size() >= 28) break;
            }
        }

        // 3. Fast H x H two-pointer scan on d = 1 (produces exact hits, residuals, and quotient cands).
        const size_t Hn = H_sorted_.size();
        const double alpha_log2 = (dec_d_ >= 14.5) ? 0.15 : 0.17;
        std::vector<ResidualCand> all_res;
        std::vector<BeamCand> d1_beam_cands;
        d1_beam_cands.reserve(256);

        auto wrap_d1 = [&](const std::vector<Step>& v_plan) {
            if (!v_plan.empty() && static_cast<int>(v_plan.size()) < best_cost) {
                best_cost = static_cast<int>(v_plan.size());
                best_steps = v_plan;
            }
        };

        {
            u64 V = target;
            int budget = best_cost - 1;
            std::array<std::vector<ResidualCand>, 7> res_buckets;
            const size_t max_keep = 320;
            std::array<u64, 7> res_thresh;
            res_thresh.fill(std::numeric_limits<u64>::max());

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
                wrap_d1(em.steps);
                budget = best_cost - 1;
            };

            u64 root_v = isqrt64(V);
            size_t j = Hn > 0 ? Hn - 1 : 0;

            for (size_t i = 0; i < Hn; ++i) {
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
                        d1_beam_cands.push_back(
                            BeamCand{q, ui, UINT32_MAX, '*', oc, oc + alpha_log2 * log2_approx(q)});
                    }
                } else {
                    const uint32_t* ri = (cu <= 1) ? find_H(rem) : M_map_.find(rem);
                    if (ri) {
                        uint8_t oc = static_cast<uint8_t>(cu + H_vec_[*ri].cost + 2);
                        if (oc < budget && q >= 2) {
                            d1_beam_cands.push_back(
                                BeamCand{q, ui, *ri, '+', oc, oc + alpha_log2 * log2_approx(q)});
                        }
                    }
                    u64 rem2 = u - rem;
                    if (static_cast<u128>(q + 1) * u <= pz_.limit) {
                        const uint32_t* ri2 = (cu <= 1) ? find_H(rem2) : M_map_.find(rem2);
                        if (ri2) {
                            uint8_t oc = static_cast<uint8_t>(cu + H_vec_[*ri2].cost + 2);
                            if (oc < budget) {
                                d1_beam_cands.push_back(BeamCand{
                                    q + 1, ui, *ri2, '-', oc, oc + alpha_log2 * log2_approx(q + 1)});
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

            // Collect and rank top H x H residuals
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
                      [&](const ResidualCand& a, const ResidualCand& b) {
                          return (a.oc + alpha_log2 * log2_approx(a.r)) <
                                 (b.oc + alpha_log2 * log2_approx(b.r));
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
                wrap_d1(em.steps);
                budget = best_cost - 1;
            };

            // Layer 1 (0.5 us each): solve_c3_hb on top 1,000 H x H residuals
            for (size_t k = 0; k < all_res.size() && k < 1000; ++k) {
                const auto& rc = all_res[k];
                if (static_cast<int>(rc.oc) + 3 > budget) continue;
                std::vector<Step> r_steps;
                if (solve_c3_hb(rc.r, 3, r_steps) >= 0) {
                    emit_res_sol(rc, r_steps);
                }
            }

            // Layer 2 (~25 us each): M op H & (H op B) op B on top 60 residuals
            for (size_t k = 0; k < all_res.size() && k < 60; ++k) {
                if ((k & 15u) == 0 && local_time_up(target_deadline)) break;
                const auto& rc = all_res[k];
                int rem_b = budget - static_cast<int>(rc.oc);
                if (rem_b < 4) continue;
                std::vector<Step> r_steps;
                if (solve_sub(rc.r, 4, false, false, r_steps) >= 0) {
                    emit_res_sol(rc, r_steps);
                }
            }

            if (best_cost <= 4) return best_steps;

            // 4. Linear-Combo Two-Pointer Scan V = (b * x +- y) +- r on d = 1
            // Cap its time to at most 40% of remaining target slice so the 3-layer beam ALWAYS runs!
            if (budget >= 5 && !local_time_up(target_deadline)) {
                double now = timer_.elapsed_sec();
                double rem = std::max(0.0, target_deadline - now);
                double lin_dl = now + rem * 0.40;
                int max_b = is_bootstrap ? 12 : 20;
                bool collect_lin = (!is_bootstrap && budget >= 8 && rem >= 0.012);
                scan_linear_bxy(V, budget, max_b, collect_lin, lin_dl, wrap_d1, best_cost, 0);
            }
        }

        if (best_cost <= 4) return best_steps;

        // 5. Unified 3-Layer Quotient + Remainder Beam Search across ALL t * d roots + H x H rems!
        {
            // Ensure beam gets time even if earlier passes were close to target_deadline
            double beam_dl = std::max(target_deadline, std::min(deadline_ - 0.03, timer_.elapsed_sec() + 0.008));
            std::vector<RemBranch> init_hx_rems;
            init_hx_rems.reserve(24);
            const u64 rem_lim = target / 16ULL;
            for (const auto& rc : all_res) {
                if (rc.r > 0 && rc.r < rem_lim && static_cast<int>(rc.oc) + 2 < best_cost) {
                    init_hx_rems.push_back(RemBranch{rc.r, rc.u_idx, rc.v_idx, rc.sign, rc.oc});
                    if (init_hx_rems.size() >= 18) break;
                }
            }
            for (const auto& mr : d1_base_rems) {
                if (mr.r > 0 && mr.r < rem_lim && static_cast<int>(mr.oc) + 2 < best_cost) {
                    init_hx_rems.push_back(mr);
                    if (init_hx_rems.size() >= 28) break;
                }
            }

            run_beam_search(target, mults, d1_beam_cands, init_hx_rems, beam_dl,
                            best_cost, update_best, is_bootstrap);
        }

        // 6. Optional bonus passes ONLY when spare target time remains!
        if (!local_time_up(target_deadline) && best_cost > 5) {
            // 6a. Layer 3: solve_beam_base up to cost 7 on top 16 H x H residuals
            int budget = best_cost - 1;
            for (size_t k = 0; k < all_res.size() && k < 16; ++k) {
                if (local_time_up(target_deadline)) break;
                const auto& rc = all_res[k];
                int rem_b = budget - static_cast<int>(rc.oc);
                if (rem_b < 5) continue;
                std::vector<Step> r_steps;
                if (solve_beam_base(rc.r, std::min(rem_b, 7), r_steps, nullptr) >= 0) {
                    u64 u = H_vec_[rc.u_idx].val;
                    u64 v = H_vec_[rc.v_idx].val;
                    u64 prod = u * v;
                    LocalEmitter em(H_vec_, basis_);
                    em.emit_node(rc.u_idx);
                    em.emit_node(rc.v_idx);
                    em.emit_op('*', u, v, prod);
                    em.append_steps(r_steps);
                    em.emit_op(rc.sign, prod, rc.r, target);
                    wrap_d1(em.steps);
                    budget = best_cost - 1;
                }
            }

            // 6b. Check (M op H) op B only on <= 10^13 targets (where cost-5 hits are realistic)
            if (dec_d_ <= 13.0 && best_cost > 5 && !local_time_up(target_deadline)) {
                std::vector<Step> sub;
                if (solve_moh_op_b(target, 5, sub) >= 0) {
                    update_best(std::move(sub));
                }
            }

            // 6c. 2-level H x H on top 3 residuals if still >= 10 and time permits
            if (!local_time_up(target_deadline) && best_cost - 1 >= 10 && !all_res.empty()) {
                for (size_t k = 0; k < all_res.size() && k < 3; ++k) {
                    if (local_time_up(target_deadline)) break;
                    const auto& rc = all_res[k];
                    int rem_b = (best_cost - 1) - static_cast<int>(rc.oc);
                    if (rem_b < 6) continue;
                    std::vector<Step> r_steps;
                    if (solve_one_level_hxh(rc.r, rem_b, r_steps) >= 0) {
                        u64 u = H_vec_[rc.u_idx].val;
                        u64 v = H_vec_[rc.v_idx].val;
                        u64 prod = u * v;
                        LocalEmitter em(H_vec_, basis_);
                        em.emit_node(rc.u_idx);
                        em.emit_node(rc.v_idx);
                        em.emit_op('*', u, v, prod);
                        em.append_steps(r_steps);
                        em.emit_op(rc.sign, prod, rc.r, target);
                        wrap_d1(em.steps);
                    }
                }
            }

            // 6d. Quick H x H scan on top 2 d in B multipliers if time permits
            for (size_t mi = 1; mi < mults.size() && mi <= 2 && !local_time_up(target_deadline); ++mi) {
                const auto& mc = mults[mi];
                if (mc.extra > 1) break;
                int mbudget = best_cost - 1 - mc.extra;
                if (mbudget < 5) continue;
                u64 V = target * mc.d;
                std::vector<Step> sub;
                if (solve_one_level_hxh(V, std::min(mbudget, 7), sub) >= 0) {
                    LocalEmitter em(H_vec_, basis_);
                    em.append_steps(sub);
                    em.emit_node(mc.d_idx);
                    em.emit_op('/', V, mc.d, target);
                    update_best(std::move(em.steps));
                }
            }
        }

        // 7. Guaranteed Fallback
        if (best_steps.empty()) {
            LocalEmitter em(H_vec_, basis_);
            fallback_build(target, em, 0);
            update_best(std::move(em.steps));
        }

        return best_steps;
    }

    int solve_one_level_hxh(u64 V, int max_cost, std::vector<Step>& out) const {
        out.clear();
        if (V == 0 || V > pz_.limit || max_cost < 3) return -1;
        int best_c = max_cost + 1;
        int c0 = solve_beam_base(V, std::min(max_cost, 6), out, nullptr);
        if (c0 >= 0) best_c = c0;
        if (best_c <= 5) return best_c;

        const size_t Hn = H_sorted_.size();
        u64 root_v = isqrt64(V);
        size_t j = Hn > 0 ? Hn - 1 : 0;

        for (size_t i = 0; i < Hn; ++i) {
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

    // -------------------------------------------------------------------------
    // Upgraded 3-Layer Unified Quotient + Remainder Beam Search (depth < 3):
    //   - Layer 0: t * d lottery roots (d = 1, d in B, top d in L1)
    //   - Layer 1 & Layer 2: Quotient moves (V = q * x +- r -> q) AND
    //     Remainder moves (V = u * v +- r -> r, r < V / 16 from HxH, H*M, M*M)
    //   - Every layer evaluated with fast cost-<=7 solve_beam_base!
    // -------------------------------------------------------------------------
    template <typename MultVec, typename UpdateFn>
    void run_beam_search(u64 target, const MultVec& mults,
                         const std::vector<BeamCand>& d1_q_cands,
                         const std::vector<RemBranch>& init_hx_rems,
                         double local_dl, int& best_cost, UpdateFn&& update_best,
                         bool is_bootstrap) const {
        if (mults.empty() || best_cost <= 4) return;

        std::vector<BeamNode> all_nodes;
        all_nodes.reserve(1024);
        std::vector<int> cur_layer;
        cur_layer.reserve(mults.size());

        for (const auto& mc : mults) {
            if (mc.extra >= best_cost - 1) continue;
            int idx = static_cast<int>(all_nodes.size());
            all_nodes.push_back(BeamNode{target * mc.d, mc.extra, -1, 0, mc.d_idx, UINT32_MAX, '/'});
            cur_layer.push_back(idx);
        }
        if (cur_layer.empty()) return;

        auto emit_beam_solution = [&](int node_idx, const std::vector<Step>& base_steps) {
            LocalEmitter em(H_vec_, basis_);
            em.append_steps(base_steps);
            u64 cur = all_nodes[node_idx].val;
            int ni = node_idx;
            while (ni >= 0) {
                const BeamNode& nd = all_nodes[ni];
                if (nd.kind == 0) {
                    if (nd.x_idx != UINT32_MAX) {
                        em.emit_node(nd.x_idx);
                        em.emit_op('/', cur, H_vec_[nd.x_idx].val, target);
                    }
                    break;
                } else if (nd.kind == 1) {
                    // parent = cur * x (+|-) r
                    u64 parent_val = all_nodes[nd.parent].val;
                    u64 x = H_vec_[nd.x_idx].val;
                    em.emit_node(nd.x_idx);
                    u64 prod = cur * x;
                    em.emit_op('*', cur, x, prod);
                    if (nd.sign != '*' && nd.r_idx != UINT32_MAX) {
                        em.emit_node(nd.r_idx);
                        em.emit_op(nd.sign, prod, H_vec_[nd.r_idx].val, parent_val);
                    }
                    cur = parent_val;
                    ni = nd.parent;
                } else {
                    // parent = u * v (+|-) cur
                    u64 parent_val = all_nodes[nd.parent].val;
                    u64 u = H_vec_[nd.x_idx].val;
                    u64 v = H_vec_[nd.r_idx].val;
                    em.emit_node(nd.x_idx);
                    em.emit_node(nd.r_idx);
                    u64 prod = u * v;
                    em.emit_op('*', u, v, prod);
                    em.emit_op(nd.sign, prod, cur, parent_val);
                    cur = parent_val;
                    ni = nd.parent;
                }
            }
            update_best(std::move(em.steps));
        };

        const double alpha_log2 = (dec_d_ >= 14.5) ? 0.145 : 0.165;
        const int max_depth = (dec_d_ >= 13.5 || best_cost >= 8) ? 3 : 2;
        const int bw_base = std::max(16, opt_.beam);
        const size_t beam_widths[3] = {
            static_cast<size_t>(is_bootstrap ? std::min(bw_base, 36) : std::min(bw_base, 48)),
            static_cast<size_t>(is_bootstrap ? std::min(bw_base, 24) : std::min(bw_base, 32)),
            static_cast<size_t>(is_bootstrap ? 16 : 20)};

        struct CandNext {
            BeamNode node;
            double est = 1e18;
        };

        for (int depth = 0; depth < max_depth && !cur_layer.empty(); ++depth) {
            if (depth > 0 && local_time_up(local_dl) && best_cost < 1000) break;

            std::vector<CandNext> next_cands;
            next_cands.reserve(2048);

            // At depth == 0, inject pre-computed d=1 HxH small remainders & H quotient moves!
            if (depth == 0) {
                for (const auto& bc : d1_q_cands) {
                    if (bc.oc >= best_cost - 1) continue;
                    double est = static_cast<double>(bc.oc) + alpha_log2 * log2_approx(bc.q);
                    next_cands.push_back(CandNext{
                        BeamNode{bc.q, static_cast<int>(bc.oc), 0, 1, bc.x_idx, bc.r_idx, bc.sign},
                        est});
                }
                for (const auto& rm : init_hx_rems) {
                    if (rm.oc >= best_cost - 1) continue;
                    double est = static_cast<double>(rm.oc) + alpha_log2 * log2_approx(rm.r);
                    next_cands.push_back(CandNext{
                        BeamNode{rm.r, static_cast<int>(rm.oc), 0, 2, rm.u_idx, rm.v_idx, rm.sign},
                        est});
                }
            }

            std::vector<RemBranch> node_rems;
            const size_t expand_cap = (depth == 0) ? cur_layer.size() : std::min<size_t>(cur_layer.size(), 24);

            for (size_t ci = 0; ci < cur_layer.size(); ++ci) {
                if ((ci & 3u) == 0 && local_time_up(local_dl) && best_cost < 1000) break;
                int node_idx = cur_layer[ci];
                const BeamNode nd = all_nodes[node_idx];
                if (nd.cost >= best_cost - 1) continue;

                // Evaluate base case up to cost 7 (skip d=1 root at depth 0, already evaluated!)
                bool need_rems = (depth + 1 < max_depth && ci < expand_cap && nd.cost < best_cost - 2);
                node_rems.clear();
                if (!(depth == 0 && node_idx == 0)) {
                    int cap = std::min(7, best_cost - nd.cost - 1);
                    if (cap >= 0) {
                        std::vector<Step> base_steps;
                        int fc = solve_beam_base(nd.val, cap, base_steps,
                                                 need_rems ? &node_rems : nullptr);
                        if (fc >= 0) {
                            emit_beam_solution(node_idx, base_steps);
                        }
                    }
                }

                if (!need_rems || nd.cost >= best_cost - 2) continue;
                if (local_time_up(local_dl) && best_cost < 1000) continue;

                // Expansion A: Quotient moves nd.val = q * x (+|-) r (x in M, r in {0} U H)
                // (Skip d=1 at depth 0 for x <= sqrt(target) since HxH scan already added them,
                //  still check x > sqrt(target) in M!)
                u64 root_v = (depth == 0 && node_idx == 0) ? isqrt64(nd.val) : 0ULL;
                const u64 rem_lim_100 = nd.val / 100ULL;
                for (const ValIdx& mx : M_sorted_) {
                    u64 x = mx.val;
                    if (x < 2 || x <= root_v) continue;
                    if (x >= nd.val) break;
                    u64 q0 = nd.val / x;
                    u64 r0 = nd.val - q0 * x;
                    if (r0 == 0) {
                        int nc = nd.cost + static_cast<int>(mx.cost) + 1;
                        if (nc < best_cost - 1) {
                            double est = nc + alpha_log2 * log2_approx(q0);
                            next_cands.push_back(CandNext{
                                BeamNode{q0, nc, node_idx, 1, mx.idx, UINT32_MAX, '*'}, est});
                        }
                    } else {
                        if (const uint32_t* ri = find_H(r0)) {
                            uint8_t cr = H_vec_[*ri].cost;
                            if (cr <= 1 || x >= 512ULL) {
                                int nc = nd.cost + static_cast<int>(mx.cost) + static_cast<int>(cr) + 2;
                                if (nc < best_cost - 1 && q0 >= 2) {
                                    double est = nc + alpha_log2 * log2_approx(q0);
                                    next_cands.push_back(CandNext{
                                        BeamNode{q0, nc, node_idx, 1, mx.idx, *ri, '+'}, est});
                                }
                            }
                        } else if (depth == 0 && node_idx == 0 && r0 < rem_lim_100) {
                            // Also catch H * M small remainders for x > root_v on d = 1
                            if (const uint32_t* qi = find_H(q0)) {
                                int nc = nd.cost + static_cast<int>(H_vec_[*qi].cost) +
                                         static_cast<int>(mx.cost) + 2;
                                if (nc < best_cost - 1) {
                                    double est = nc + alpha_log2 * log2_approx(r0);
                                    next_cands.push_back(CandNext{
                                        BeamNode{r0, nc, node_idx, 2, *qi, mx.idx, '+'}, est});
                                }
                            }
                        }
                        if (static_cast<u128>(x) * (q0 + 1) <= pz_.limit) {
                            u64 r1 = x - r0;
                            if (const uint32_t* ri1 = find_H(r1)) {
                                uint8_t cr1 = H_vec_[*ri1].cost;
                                if (cr1 <= 1 || x >= 512ULL) {
                                    int nc = nd.cost + static_cast<int>(mx.cost) +
                                             static_cast<int>(cr1) + 2;
                                    if (nc < best_cost - 1) {
                                        double est = nc + alpha_log2 * log2_approx(q0 + 1);
                                        next_cands.push_back(CandNext{
                                            BeamNode{q0 + 1, nc, node_idx, 1, mx.idx, *ri1, '-'},
                                            est});
                                    }
                                }
                            } else if (depth == 0 && node_idx == 0 && r1 < rem_lim_100) {
                                if (const uint32_t* qi1 = find_H(q0 + 1)) {
                                    int nc = nd.cost + static_cast<int>(H_vec_[*qi1].cost) +
                                             static_cast<int>(mx.cost) + 2;
                                    if (nc < best_cost - 1) {
                                        double est = nc + alpha_log2 * log2_approx(r1);
                                        next_cands.push_back(CandNext{
                                            BeamNode{r1, nc, node_idx, 2, *qi1, mx.idx, '-'}, est});
                                    }
                                }
                            }
                        }
                    }
                }

                // Expansion B: Remainder moves nd.val = u * v (+|-) r -> r (from H*M and M*M!)
                if (!node_rems.empty()) {
                    std::sort(node_rems.begin(), node_rems.end(),
                              [&](const RemBranch& a, const RemBranch& b) {
                                  return (a.oc + alpha_log2 * log2_approx(a.r)) <
                                         (b.oc + alpha_log2 * log2_approx(b.r));
                              });
                    int added_rm = 0;
                    u64 last_r = 0;
                    for (const auto& rm : node_rems) {
                        if (rm.r == 0 || rm.r == last_r || rm.r >= nd.val / 16ULL) continue;
                        last_r = rm.r;
                        int nc = nd.cost + static_cast<int>(rm.oc);
                        if (nc >= best_cost - 1) continue;
                        double est = nc + alpha_log2 * log2_approx(rm.r);
                        next_cands.push_back(CandNext{
                            BeamNode{rm.r, nc, node_idx, 2, rm.u_idx, rm.v_idx, rm.sign}, est});
                        if (++added_rm >= 10) break;
                    }
                }
            }

            if (next_cands.empty()) break;
            std::sort(next_cands.begin(), next_cands.end(),
                      [](const CandNext& a, const CandNext& b) { return a.est < b.est; });

            cur_layer.clear();
            FlatMap<uint8_t> seen_next(next_cands.size() * 2 + 16);
            size_t bw = beam_widths[std::min(depth, 2)];
            for (const auto& cn : next_cands) {
                if (cn.node.cost >= best_cost - 1) continue;
                if (!seen_next.insert(cn.node.val, 1)) continue;
                int idx = static_cast<int>(all_nodes.size());
                all_nodes.push_back(cn.node);
                cur_layer.push_back(idx);
                if (cur_layer.size() >= bw) break;
            }
        }
    }

    void fallback_build(u64 v, LocalEmitter& em, int depth) const {
        if (em.have.contains(v)) return;
        if (const uint32_t* hi = find_H(v)) {
            em.emit_node(*hi);
            return;
        }
        std::vector<Step> sub;
        if (solve_beam_base(v, 6, sub, nullptr) >= 0) {
            em.append_steps(sub);
            return;
        }
        u64 best_x = 0;
        uint32_t best_idx = UINT32_MAX;
        u64 best_q = 0, best_r = 0;
        bool best_minus = false;
        double best_sc = 1e18;

        if (depth <= 12 && !M_sorted_.empty()) {
            u64 root = isqrt64(v);
            auto it = std::upper_bound(
                M_sorted_.begin(), M_sorted_.end(), root,
                [](u64 val, const ValIdx& a) { return val < a.val; });
            auto eval_m = [&](const ValIdx& mx) {
                u64 a = mx.val;
                if (a < 2 || a >= v) return;
                u64 q = v / a, r = v % a;
                if (q == 0) return;
                double sc1 = mx.cost + log2_approx(q) + (r ? log2_approx(r) : -4.0);
                if (sc1 < best_sc) {
                    best_sc = sc1;
                    best_x = a;
                    best_idx = mx.idx;
                    best_q = q;
                    best_r = r;
                    best_minus = false;
                }
                if (r > 0 && static_cast<u128>(a) * (q + 1) <= pz_.limit) {
                    u64 r2 = a - r;
                    double sc2 = mx.cost + log2_approx(q + 1) + log2_approx(r2);
                    if (sc2 < best_sc) {
                        best_sc = sc2;
                        best_x = a;
                        best_idx = mx.idx;
                        best_q = q + 1;
                        best_r = r2;
                        best_minus = true;
                    }
                }
            };
            auto it_d = it;
            for (int s = 0; it_d != M_sorted_.begin() && s < 20; ++s) {
                --it_d;
                eval_m(*it_d);
            }
        }

        if (best_x < 2) {
            u64 root = isqrt64(v);
            for (u64 b : basis_) {
                if (b >= 2 && b <= root && b > best_x) best_x = b;
            }
            if (best_x < 2) {
                for (u64 b : basis_) {
                    if (b >= 2 && b < v && (best_x == 0 || b < best_x)) best_x = b;
                }
            }
            if (best_x < 2) {
                if (!em.have.contains(1)) em.emit_op('/', basis_[0], basis_[0], 1);
                if (!em.have.contains(2)) em.emit_op('+', 1, 1, 2);
                best_x = 2;
            }
            best_idx = UINT32_MAX;
            best_q = v / best_x;
            best_r = v % best_x;
            best_minus = false;
        }

        fallback_build(best_q, em, depth + 1);
        if (best_idx != UINT32_MAX) em.emit_node(best_idx);
        u64 prod = best_q * best_x;
        em.emit_op('*', best_q, best_x, prod);
        if (best_r > 0) {
            fallback_build(best_r, em, depth + 1);
            em.emit_op(best_minus ? '-' : '+', prod, best_r, v);
        }
    }

    // -------------------------------------------------------------------------
    // Assembly, Cross-Target Scratch Promotion, & Verification
    // -------------------------------------------------------------------------
    std::optional<std::vector<Step>> assemble_and_optimize(
        const std::vector<Step>& prefix_steps,
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
        auto best_verified = verify_and_prune_plan(pz_, raw, opt_.regs);
        if (!best_verified) {
            LocalEmitter em(H_vec_, basis_);
            std::vector<Step> fb = basis_steps_;
            for (u64 t : missing_) {
                em.reset(basis_);
                fallback_build(t, em, 0);
                for (const Step& s : em.steps) fb.push_back(s);
            }
            return verify_and_prune_plan(pz_, fb, opt_.regs);
        }

        // Cross-target scratch promotion: promote frequently reused L1 values into B
        // when global deduplication had to fall back to instances.
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
            int max_prom = std::max(0, (opt_.regs > 0 ? opt_.regs - 5 : 44) -
                                           static_cast<int>(basis_.size()));
            for (size_t k = 0; k < hot.size() && static_cast<int>(promoted.size()) < max_prom; ++k) {
                promoted.push_back(hot[k].second);
                std::vector<Step> cand = build_plan_with_promoted(promoted);
                auto verified = verify_and_prune_plan(pz_, cand, opt_.regs);
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

int choose_basis_size(const Puzzle& pz, const Options& opt, int restart_idx) {
    int max_reg_cap = opt.regs > 0 ? std::max(6, opt.regs - 5) : 43;
    if (opt.basis_explicit) {
        return std::min(opt.basis, max_reg_cap);
    }

    size_t n_miss = pz.missing_targets().size();
    u64 max_t = 1;
    for (u64 t : pz.targets) max_t = std::max(max_t, t);
    double dec = std::log10(static_cast<double>(std::max<u64>(1, max_t)));

    int base = opt.basis;
    if (n_miss <= 10) {
        base = std::min(opt.basis, static_cast<int>(14 + n_miss));
    } else if (n_miss <= 18) {
        base = std::min(opt.basis, 32);
    } else if (dec >= 14.5) {
        base = 41;
    } else if (dec >= 11.5) {
        base = 39;
    } else {
        base = 34;
    }

    static constexpr int kOffsets[5] = {0, 1, -2, 2, -4};
    int off = kOffsets[restart_idx % 5];
    return std::max(10, std::min(max_reg_cap, base + off));
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
        else if (a == "--time") opt.time_limit = std::atof(next());
        else if (a == "--basis") {
            opt.basis = std::atoi(next());
            opt.basis_explicit = true;
        } else if (a == "--beam") opt.beam = std::atoi(next());
        else if (a == "--regs") opt.regs = std::atoi(next());
        else if (a == "--verbose") opt.verbose = true;
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }

    if (opt.restarts < 1) opt.restarts = 1;
    if (opt.beam < 1) opt.beam = 1;
    if (opt.basis < 4) opt.basis = 4;

    Puzzle pz;
    if (!read_puzzle(std::cin, pz)) {
        std::fprintf(stderr, "bad input\n");
        return 2;
    }

    Timer timer;
    std::vector<Step> best_plan;
    bool have_best = false;

    const double total_budget = std::max(0.15, opt.time_limit * 0.94);

    for (int r = 0; r < opt.restarts; ++r) {
        double elapsed = timer.elapsed_sec();
        double remaining = total_budget - elapsed;
        if (r > 0 && remaining <= 0.08) break;

        double slice = std::max(0.10, remaining / static_cast<double>(opt.restarts - r));
        double deadline = elapsed + slice;
        int bsz = choose_basis_size(pz, opt, r);
        // Restart 0 & 2 use Target-First Bootstrap; Restart 1 uses pure upfront multi-scale basis.
        bool online_absorb = (r % 2 == 0);
        u64 rseed = splitmix64(opt.seed + 0x9e3779b97f4a7c15ULL * static_cast<u64>(r + 1));

        MitmBasisSolver solver(pz, opt, timer, deadline);
        auto res = solver.solve_restart(r, rseed, bsz, online_absorb);
        if (res && res->size() <= 4096) {
            size_t r_sz = res->size();
            if (!have_best || r_sz < best_plan.size()) {
                best_plan = std::move(*res);
                have_best = true;
            }
            if (opt.verbose) {
                std::fprintf(stderr, "restart %d (basis=%d absorb=%d): %zu steps (best %zu), %.3fs\n",
                             r, bsz, online_absorb ? 1 : 0, r_sz, best_plan.size(),
                             timer.elapsed_sec());
            }
        }
    }

    if (!have_best) {
        std::fprintf(stderr, "mitm_basis failed to produce valid plan for %s\n", pz.id.c_str());
        return 1;
    }

    emit_plan(best_plan);
    return 0;
}
