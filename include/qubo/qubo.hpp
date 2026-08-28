#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace qubo {

// A QUBO (Quadratic Unconstrained Binary Optimization) problem:
//
//     minimize   x^T Q x     over x in {0,1}^n
//
// Q is kept symmetric (the standard convention): the weight of the pair
// (i, j) is split evenly between Q[i][j] and Q[j][i], so add_term() below
// is the normal way to build one up term by term.
class Problem {
public:
    explicit Problem(std::size_t n) : n_(n), Q_(n, std::vector<double>(n, 0.0)) {}

    std::size_t size() const { return n_; }

    double at(std::size_t i, std::size_t j) const { return Q_[i][j]; }

    // Adds a term w*x_i*x_j to the objective. For i == j this is a linear
    // term (since x_i^2 == x_i for binary x_i); for i != j the weight w is
    // mirrored onto both Q[i][j] and Q[j][i] to keep Q symmetric.
    void add_term(std::size_t i, std::size_t j, double w) {
        if (i == j) {
            Q_[i][i] += w;
        } else {
            Q_[i][j] += w;
            Q_[j][i] += w;
        }
    }

    double energy(const std::vector<int>& x) const {
        if (x.size() != n_) throw std::invalid_argument("x has wrong size for this Problem");
        double e = 0.0;
        for (std::size_t i = 0; i < n_; ++i) {
            if (!x[i]) continue;
            for (std::size_t j = 0; j < n_; ++j) {
                if (x[j]) e += Q_[i][j];
            }
        }
        return e;
    }

    bool is_symmetric() const {
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t j = i + 1; j < n_; ++j)
                if (Q_[i][j] != Q_[j][i]) return false;
        return true;
    }

    const std::vector<std::vector<double>>& matrix() const { return Q_; }

private:
    std::size_t n_;
    std::vector<std::vector<double>> Q_;
};

} // namespace qubo
