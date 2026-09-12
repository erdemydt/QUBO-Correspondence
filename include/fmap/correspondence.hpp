// Turning a functional map back into a point-to-point correspondence.
//
// The functional map C lives between function spaces; downstream you usually
// want an actual vertex-to-vertex map. The standard route is through the
// spectral embedding: a vertex's delta function has coefficients equal to its
// row of Phi, so applying C to source vertex i's embedding gives where that
// point lands in the target's spectral coordinates. The corresponding target
// vertex is the one whose own embedding sits closest.
//
//     T(i) = argmin_j || C * Phi1(i,:)^T - Phi2(j,:)^T ||
//
// This is where recovery strategies plug in. Nearest neighbour is the classic
// choice and the baseline implemented here, but it is also the weakest link in
// the pipeline: it treats every vertex independently, so nothing stops it from
// mapping two source vertices to the same target vertex, or from producing a
// map that tears neighbouring points apart. Formulating recovery as a
// combinatorial assignment problem instead -- QUBO among them -- is a way to
// impose the bijectivity and smoothness that this step throws away.

#pragma once

#include <Eigen/Dense>

#include <memory>
#include <string>

#include "fmap/functional_map.hpp"
#include "fmap/spectral_basis.hpp"

namespace fmap {

// For each source vertex, the index of the target vertex it maps to.
// Size n1, values in [0, n2).
using PointMap = Eigen::VectorXi;

// Available recovery methods. A QUBO-based strategy is intended to join this
// enum without disturbing anything around it.
enum class RecoveryMethod {
  NearestNeighbor,
};

std::string to_string(RecoveryMethod method);

// Parse a method name; throws std::runtime_error listing the valid names.
RecoveryMethod recovery_method_from_string(const std::string& name);

// Interface every recovery method implements.
//
// The inputs are deliberately the full spectral bases rather than just the
// embeddings: a combinatorial method will want the eigenvalues and the mass
// vector for smoothness or area-preservation terms, which a nearest-neighbour
// search has no use for.
class CorrespondenceStrategy {
 public:
  virtual ~CorrespondenceStrategy() = default;

  virtual std::string name() const = 0;

  virtual PointMap recover(const FunctionalMap& map, const SpectralBasis& source,
                           const SpectralBasis& target) const = 0;
};

// Nearest neighbour in the target's spectral embedding, via a KD-tree.
class NearestNeighborStrategy : public CorrespondenceStrategy {
 public:
  std::string name() const override { return "nearest-neighbor"; }

  PointMap recover(const FunctionalMap& map, const SpectralBasis& source,
                   const SpectralBasis& target) const override;
};

std::unique_ptr<CorrespondenceStrategy> make_strategy(RecoveryMethod method);

// Diagnostics that do not require ground truth.
struct MapStatistics {
  // Distinct target vertices actually hit. A bijection would hit all of them;
  // nearest-neighbour recovery typically collapses many source vertices onto
  // the same target, and this is the number that shows it.
  Eigen::Index distinct_targets = 0;

  // distinct_targets as a fraction of the target vertex count.
  double coverage = 0.0;

  // Most source vertices mapped onto any single target vertex.
  Eigen::Index max_collisions = 0;
};

MapStatistics analyze_map(const PointMap& map, Eigen::Index target_vertices);

}  // namespace fmap
