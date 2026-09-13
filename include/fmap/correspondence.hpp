// Functional map -> vertex map. A vertex's delta function has coefficients
// equal to its row of Phi, so T(i) = argmin_j ||C Phi1(i,:)^T - Phi2(j,:)^T||.
//
// The extension point. Nearest neighbour is the baseline and the weakest link:
// it treats each vertex independently, so nothing stops it collapsing many
// sources onto one target. Combinatorial assignment -- QUBO among them -- is
// how the lost bijectivity and smoothness get imposed.

#pragma once

#include <Eigen/Dense>

#include <memory>
#include <string>

#include "fmap/functional_map.hpp"
#include "fmap/spectral_basis.hpp"

namespace fmap {

// Size n1, values in [0, n2).
using PointMap = Eigen::VectorXi;

enum class RecoveryMethod {
  NearestNeighbor,
};

std::string to_string(RecoveryMethod method);
RecoveryMethod recovery_method_from_string(const std::string& name);

// Takes full bases, not just embeddings: a combinatorial method wants the
// eigenvalues and mass for smoothness or area-preservation terms.
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

// How far from a bijection the recovery is; needs no ground truth.
struct MapStatistics {
  Eigen::Index distinct_targets = 0;
  double coverage = 0.0;
  Eigen::Index max_collisions = 0;
};

MapStatistics analyze_map(const PointMap& map, Eigen::Index target_vertices);

}  // namespace fmap
