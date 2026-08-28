# QUBO Playground (C++)

A small, self-contained setup for learning **QUBO** — Quadratic Unconstrained
Binary Optimization — by reading and running real code.

## What is a QUBO?

A QUBO problem asks: given a symmetric matrix `Q` of size `n x n`, find the
binary vector `x` in `{0,1}^n` that minimizes

```
x^T Q x  =  sum_i sum_j  Q[i][j] * x[i] * x[j]
```

That's it — no constraints, just binary variables and a quadratic objective.
A few properties worth internalizing early:

- **Diagonal terms are linear.** Since `x_i` is 0 or 1, `x_i * x_i == x_i`,
  so `Q[i][i]` acts as a per-variable linear cost/reward, not a true
  quadratic term.
- **Off-diagonal terms are pairwise interactions.** `Q[i][j] + Q[j][i]` is
  the total cost of turning on both `x_i` and `x_j` together — this is
  where QUBO expresses "these two choices interact."
- **`Q` is conventionally symmetric.** Any quadratic form can be
  symmetrized without changing its value, so by convention the weight of a
  pair is split evenly across `Q[i][j]` and `Q[j][i]`. See
  [`Problem::add_term`](include/qubo/qubo.hpp) for how that's enforced here.
- **"Unconstrained" is a bit of a lie.** Real constraints (e.g. "exactly one
  of these must be true") are folded into `Q` as penalty terms that raise
  the energy of infeasible assignments. This is the main modeling skill in
  QUBO — the [Max-Cut example](src/main.cpp) below shows the technique on
  a simple case.
- **It's equivalent to an Ising model.** Substituting `x_i = (1 + s_i) / 2`
  with `s_i in {-1, +1}` turns a QUBO into an Ising Hamiltonian. This is why
  QUBO is the standard input format for quantum annealers (e.g. D-Wave) and
  QAOA on gate-model quantum computers — both are built to minimize Ising
  energy.
- **It's NP-hard in general**, since Max-Cut, graph coloring, knapsack, and
  many other NP-hard problems reduce to it directly. That's also what makes
  it a useful sandbox for heuristics like simulated annealing.

## Layout

```
include/qubo/qubo.hpp     Problem: holds Q, evaluates x^T Q x, checks symmetry
include/qubo/solvers.hpp  solve_brute_force, solve_simulated_annealing
src/main.cpp              Worked example: Max-Cut encoded as a QUBO
CMakeLists.txt
```

## The example: Max-Cut

Max-Cut asks: split a graph's nodes into two sets so the total weight of
edges *crossing* between them is maximized. For an edge `(i, j)` with weight
`w`, the expression

```
w * (x_i + x_j - 2*x_i*x_j)
```

equals `w` when `x_i != x_j` (the edge is cut) and `0` when `x_i == x_j`
(it isn't). Summing over all edges and negating (QUBO minimizes) gives the
objective built in `build_maxcut_qubo()` in [src/main.cpp](src/main.cpp).

The example graph is a 6-node ring plus two diagonals, built so every edge
connects opposite-parity nodes — it's exactly 2-colorable, so you can
predict the answer before running anything: alternating the nodes
`0,1,0,1,0,1` around the ring cuts every single edge.

## Build & run

```sh
cmake -S . -B build
cmake --build build
./build/qubo_maxcut
```

Expected output: both the brute-force solver (exact, checks all `2^6 = 64`
assignments) and simulated annealing (heuristic) land on energy `-8`,
i.e. a full cut of weight 8, splitting the nodes into `{1,3,5}` and
`{0,2,4}` — exactly the alternating coloring predicted above.

## Ideas for extending this

Once the Max-Cut example makes sense, good next steps to build your own
intuition:

- **Number partitioning** — split a set of numbers into two groups with
  equal sums. QUBO: `(sum_i s_i * (2*x_i - 1))^2`.
- **Graph coloring with a fixed number of colors** — one-hot binary
  variables per (node, color) pair, plus penalty terms for adjacent nodes
  sharing a color and for a node not getting exactly one color.
- **0/1 Knapsack** — needs a penalty term to enforce the weight
  constraint, a good exercise in the "unconstrained is a bit of a lie"
  point above.
- Swap `solve_simulated_annealing`'s bit-flip proposal for a bulkier move,
  or plot energy vs. temperature, to build intuition for annealing
  schedules.
