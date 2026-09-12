# fmap — a functional maps pipeline in C++

Shape correspondence via functional maps (Ovsjanikov et al., 2012), built as a
set of reusable blocks wired together by a single `main`.

```
mesh → cotangent Laplacian → spectral basis → descriptors → map C → point map
```

## Building

No manual dependency setup — CMake fetches everything.

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Requires CMake ≥ 3.24 and a C++17 compiler.

## Running

```sh
./build/fmap --list                        # what meshes are available
./build/fmap -s scape:0 -t scape:1         # run the pipeline
./build/fmap -s tosca:cat0 -t tosca:cat3 -k 40
```

Meshes are named by dataset spec (`scape:0`, `tosca:cat3`) or by path to a
`.off` / `.obj` file. `--help` lists the options.

Output goes to `out/`: the target coloured by position, and the source coloured
by pulling those colours back through the recovered map. If the map is right the
two look anatomically matched; a left/right symmetry flip shows up as mirrored
colour.

## Layout

| Path | Responsibility |
|---|---|
| `cmake/Dependencies.cmake` | Fetches Eigen, Spectra, nanoflann, tinyobjloader |
| `include/fmap/dataset.hpp` | Mesh discovery and spec → path resolution |
| `include/fmap/mesh.hpp` | Mesh container, OFF/OBJ loading, colour output |
| `include/fmap/laplacian.hpp` | Cotangent stiffness + lumped mass matrix |
| `include/fmap/spectral_basis.hpp` | Smallest-k generalized eigenpairs |
| `include/fmap/descriptors.hpp` | HKS and WKS |
| `include/fmap/functional_map.hpp` | Least-squares solve for C |
| `include/fmap/correspondence.hpp` | Point-to-point recovery (swappable) |
| `include/fmap/evaluation.hpp` | Geodesic error vs. ground truth |
| `include/fmap/utils.hpp` | `parallel_for`, `Timer` |
| `src/main.cpp` | The only executable; wires the blocks in order |
| `tests/` | One suite per block |

Dependencies are all header-only and are fetched *without* configuring their own
CMake projects (see `cmake/Dependencies.cmake` for why). Adding a dataset means
one row in `kDatasetDirs` in `src/dataset.cpp` and one `.gitignore` line; no
filename or directory appears anywhere else.

## Adding a recovery method

`correspondence.hpp` defines `CorrespondenceStrategy` with a single `recover()`.
Implement it, add an enum value, and add a branch to `make_strategy()`. It
receives both full spectral bases — not just the embeddings — so a combinatorial
method has the eigenvalues and mass vector available for smoothness or
area-preservation terms.

## Results and known limits

Two datasets are supported and both provide free ground truth: SCAPE meshes all
share one vertex numbering, as do the meshes within a TOSCA shape class, so the
true map is the identity permutation.

Geodesic error is normalized by √area. Representative numbers at defaults:

| Pair | mean error | within 5% | within 10% | target coverage |
|---|---|---|---|---|
| `scape:0 → scape:1` | 0.030 | 89% | 99.7% | 39% |
| `tosca:cat0 → cat3` (k=100) | 0.32 | 24% | 36% | 7.5% |
| `tosca:cat0 → cat3` (k=40) | 0.15 | 35% | 69% | 6.1% |

A mesh mapped to itself recovers the identity permutation on all 12500 vertices
exactly, and gives C = I to within 4e-4 — that is the test that pins the whole
chain down.

Three real limitations, all inherent to this baseline rather than bugs:

- **Nearest-neighbour recovery is not injective.** Each source vertex picks its
  closest target independently, so targets get reused heavily: on a SCAPE pair
  only 39% of target vertices are hit at all, and up to 60 source vertices land
  on a single target. On TOSCA it is far worse (5–8% coverage, up to 441
  collisions). This is the stage a combinatorial formulation would replace.

- **Bigger bases are not better.** On TOSCA, k=40 beats k=100 by a factor of two,
  and the map's deviation from orthogonality grows from 1.0 to 7.5 as k goes
  40 → 150. High-frequency eigenfunctions are unstable across shapes, so past
  some point they contribute noise instead of detail. Treat k as per-dataset.

- **Symmetry flips.** About a third of a SCAPE mesh's first 100 eigenvalues come
  in near-degenerate pairs, and within such a pair the eigenfunctions are only
  defined up to a rotation of their shared eigenspace. On bilaterally symmetric
  shapes this lets the map send points to their mirror counterpart. The TOSCA
  error curves show it directly: they plateau around 50% and stop improving,
  meaning the remaining points are not slightly wrong but on the wrong side.

Natural next steps: ZoomOut-style iterative refinement, a bijectivity-aware
recovery step, and descriptors with better discriminative power than HKS/WKS.

## Data

`scape-bim/` (71 meshes, 12500 vertices each) and `tosca-offs/` (82 meshes in 10
classes, 4.3k–52.6k vertices) are gitignored — they are inputs, not source.
Point `$FMAP_DATA_ROOT` elsewhere if they live outside the repo.

Two quirks worth knowing, both handled: SCAPE has no `mesh051`, and TOSCA's
index ranges have gaps (`horse` jumps 7 → 10 → 15 → 17), so meshes are
discovered by scanning rather than by generating filenames. `horse0_partial` has
a different vertex count from the rest of the horse class and is deliberately
kept in a class of its own so it is never treated as vertex-aligned with it.
