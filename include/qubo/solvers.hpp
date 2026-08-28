#pragma once

#include "qubo.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace qubo {

struct Solution {
    std::vector<int> x;
    double energy;
};

// Exhaustive search over all 2^n assignments. Exact, but only feasible for
// small n (fine up to ~20-24 bits depending on patience).
inline Solution solve_brute_force(const Problem& p) {
    const std::size_t n = p.size();
    std::vector<int> x(n, 0);
    Solution best{x, std::numeric_limits<double>::infinity()};

    const std::size_t total = std::size_t(1) << n;
    for (std::size_t mask = 0; mask < total; ++mask) {
        for (std::size_t i = 0; i < n; ++i) x[i] = (mask >> i) & 1;
        double e = p.energy(x);
        if (e < best.energy) best = {x, e};
    }
    return best;
}

// Simulated annealing: start random, repeatedly propose a single bit flip,
// accept it if it improves the energy or (with a temperature-dependent
// probability) even if it doesn't. Temperature decays geometrically from
// t_start to t_end. No optimality guarantee, but scales far beyond brute
// force and is the classic entry point into QUBO/Ising heuristics.
inline Solution solve_simulated_annealing(const Problem& p,
                                           int iterations = 20000,
                                           double t_start = 4.0,
                                           double t_end = 0.01,
                                           unsigned seed = 42) {
    const std::size_t n = p.size();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::size_t> pick_bit(0, n - 1);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    std::vector<int> x(n);
    for (auto& b : x) b = unit(rng) < 0.5 ? 1 : 0;
    double e = p.energy(x);
    Solution best{x, e};

    for (int step = 0; step < iterations; ++step) {
        double t = t_start * std::pow(t_end / t_start, double(step) / iterations);
        std::size_t i = pick_bit(rng);

        // Delta from flipping bit i, computed locally instead of
        // recomputing the full x^T Q x from scratch each step.
        double delta = 0.0;
        double flip = (x[i] == 0) ? 1.0 : -1.0; // x_i: 0->1 is +1, 1->0 is -1
        delta += flip * p.at(i, i);
        for (std::size_t j = 0; j < n; ++j) {
            if (j != i && x[j]) delta += flip * (p.at(i, j) + p.at(j, i));
        }

        if (delta <= 0.0 || unit(rng) < std::exp(-delta / t)) {
            x[i] ^= 1;
            e += delta;
            if (e < best.energy) best = {x, e};
        }
    }
    return best;
}

} // namespace qubo
