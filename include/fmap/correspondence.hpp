// Turning a functional map back into a vertex-to-vertex correspondence.
//
// A vertex's delta function has coefficients equal to its row of Phi, so
// applying C to source vertex i's embedding says where it lands in the target's
// spectral coordinates; the answer is the nearest target embedding:
//
//     T(i) = argmin_j || C * Phi1(i,:)^T - Phi2(j,:)^T ||
//
// This is the extension point. Nearest neighbour is the classic baseline and
// also the pipeline's weakest link: it treats every vertex independently, so
// nothing stops it collapsing many sources onto one target or tearing
// neighbours apart. Posing recovery as combinatorial assignment -- QUBO among
// them -- is how the bijectivity and smoothness lost here get imposed.

#pragma once

#include <Eigen/Dense>

#include <memory>
#include <string>

#include "fmap/functional_map.hpp"
#include "fmap/spectral_basis.hpp"

namespace fmap {

// Target index per source vertex: size n1, values in [0, n2).
using PointMap = Eigen::VectorXi;

enum class RecoveryMethod {
  NearestNeighbor,
};

std::string to_string(RecoveryMethod method);

// Throws std::runtime_error listing the valid names.
RecoveryMethod recovery_method_from_string(const std::string& name);

// Takes the full bases rather than just the embeddings: a combinatorial method
// will want the eigenvalues and mass for smoothness or area-preservation terms.
class CorrespondenceStrategy {
 public:
  virtual ~CorrespondenceStrategy() = default;

  virtual std::string name() const = 0;

  virtual PointMap recover(const FunctionalMap& map, const SpectralBasis& source,
                           const SpectralBasis& target) const = 0;
};

class NearestNeighborStrategy : public CorrespondenceStrategy {
 public:
  std::string name() const override { return "nearest-neighbor"; }

  PointMap recover(const FunctionalMap& map, const SpectralBasis& source,
                   const SpectralBasis& target) const override;
};

std::unique_ptr<CorrespondenceStrategy> make_strategy(RecoveryMethod method);

// Diagnostics that need no ground truth. A bijection would hit every target;
// these are the numbers that show how far from one the recovery is.
struct MapStatistics {
  Eigen::Index distinct_targets = 0;
  double coverage = 0.0;  // distinct_targets / target vertex count
  Eigen::Index max_collisions = 0;
};

MapStatistics analyze_map(const PointMap& map, Eigen::Index target_vertices);

}  // namespace fmap
