// Max-Cut as a QUBO, worked end to end.
//
// Max-Cut: split the graph's nodes into two sets so that the total weight
// of edges crossing between the sets is as large as possible.
//
// For an edge (i, j) with weight w, the term
//
//     w * (x_i + x_j - 2*x_i*x_j)
//
// equals w when x_i != x_j (the edge is cut) and 0 when x_i == x_j (it
// isn't). Summing this over all edges gives the total cut weight to
// *maximize*. QUBO solvers minimize, so we negate it:
//
//     minimize  sum_edges  w * (2*x_i*x_j - x_i - x_j)
//
// which is exactly what build_maxcut_qubo() below encodes via add_term().
#include "qubo/qubo.hpp"
#include "qubo/solvers.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

struct Edge {
    std::size_t i, j;
    double w;
};

qubo::Problem build_maxcut_qubo(std::size_t n, const std::vector<Edge>& edges) {
    qubo::Problem p(n);
    for (const auto& e : edges) {
        p.add_term(e.i, e.j, e.w);   // +2w*x_i*x_j once symmetrized
        p.add_term(e.i, e.i, -e.w);  // -w*x_i
        p.add_term(e.j, e.j, -e.w);  // -w*x_j
    }
    return p;
}

double cut_weight(const std::vector<Edge>& edges, const std::vector<int>& x) {
    double cut = 0.0;
    for (const auto& e : edges)
        if (x[e.i] != x[e.j]) cut += e.w;
    return cut;
}

void print_matrix(const qubo::Problem& p) {
    const auto& Q = p.matrix();
    for (std::size_t i = 0; i < p.size(); ++i) {
        for (std::size_t j = 0; j < p.size(); ++j)
            std::cout << std::setw(6) << Q[i][j];
        std::cout << '\n';
    }
}

void print_solution(const std::string& label, const qubo::Solution& sol,
                     const std::vector<Edge>& edges) {
    std::cout << label << ":  energy = " << sol.energy
              << "   cut weight = " << cut_weight(edges, sol.x) << "\n  x = ";
    for (int b : sol.x) std::cout << b;
    std::cout << "  (set A = {";
    bool first = true;
    for (std::size_t i = 0; i < sol.x.size(); ++i)
        if (sol.x[i] == 0) { std::cout << (first ? "" : ",") << i; first = false; }
    std::cout << "}, set B = {";
    first = true;
    for (std::size_t i = 0; i < sol.x.size(); ++i)
        if (sol.x[i] == 1) { std::cout << (first ? "" : ",") << i; first = false; }
    std::cout << "})\n";
}

int main() {
    // A 6-node bipartite-ish graph: a 6-cycle plus two "diagonals". Every
    // edge only ever connects opposite-parity nodes, so it's exactly
    // 2-colorable and the max cut visibly equals every edge weight summed.
    const std::size_t n = 6;
    const std::vector<Edge> edges = {
        {0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 4, 1}, {4, 5, 1}, {5, 0, 1},
        {0, 3, 1}, {1, 4, 1},
    };

    qubo::Problem p = build_maxcut_qubo(n, edges);

    std::cout << "=== QUBO properties ===\n";
    std::cout << "n (bits)   : " << p.size() << "\n";
    std::cout << "symmetric  : " << (p.is_symmetric() ? "yes" : "no") << "\n";
    std::cout << "Q matrix   :\n";
    print_matrix(p);
    std::cout << "\n";

    std::cout << "=== Solving ===\n";
    auto exact = qubo::solve_brute_force(p);
    auto annealed = qubo::solve_simulated_annealing(p);

    print_solution("brute force        ", exact, edges);
    print_solution("simulated annealing", annealed, edges);

    return 0;
}
