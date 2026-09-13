// C takes a function's source-basis coefficients to target ones: a k2 x k1
// matrix in place of an n x n permutation.
//
// Least squares over descriptor preservation ||C A - B||^2 and Laplacian
// commutativity ||C D1 - D2 C||^2. Both Laplacians are diagonal in their own
// eigenbasis, so the latter is sum_ij C_ij^2 (l1_j - l2_i)^2, which separates
// by row: k2 independent ridge regressions, not one k1*k2 system.

#pragma once

#include <Eigen/Dense>

#include "fmap/spectral_basis.hpp"

namespace fmap {

struct FunctionalMapOptions {
  // Eigenvalues are normalized first, so this is mesh-independent. Swept on
  // SCAPE at k=60, max|C^T C - I|: 0 -> 566, 0.01 -> 12.7, 1.0 -> 4.7,
  // 100 -> 3.7 at 3.5x the descriptor residual. 1.0 is the knee.
  double commutativity_weight = 1.0;
};

struct FunctionalMap {
  Eigen::MatrixXd C;                    // k2 x k1, C * a ~= b
  double descriptor_residual = 0.0;     // ||C A - B||_F / ||B||_F
  double commutativity_residual = 0.0;

  Eigen::Index source_size() const { return C.cols(); }
  Eigen::Index target_size() const { return C.rows(); }

  // max |C^T C - I|. Large means non-isometric shapes, weak descriptors, or a
  // basis that is too small.
  double orthogonality_error() const;
};

// Descriptors are per-vertex, n1 x d and n2 x d with matching d; projected here.
FunctionalMap solve_functional_map(const SpectralBasis& source,
                                   const SpectralBasis& target,
                                   const Eigen::MatrixXd& source_descriptors,
                                   const Eigen::MatrixXd& target_descriptors,
                                   const FunctionalMapOptions& options = {});

}  // namespace fmap
