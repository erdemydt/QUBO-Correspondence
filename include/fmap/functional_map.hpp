// The functional map matrix C: a correspondence as a linear operator between
// function spaces rather than a pairing of points. C is k2 x k1 and takes a
// function's coefficients on the source to its coefficients on the target,
// replacing an n x n permutation with a small dense matrix.
//
// Least squares over two energies: descriptor preservation ||C A - B||^2, and
// Laplacian commutativity ||C D1 - D2 C||^2, which a true isometry satisfies.
// Both Laplacians are diagonal in their own eigenbasis, so the second term is
// just sum_ij C_ij^2 (lambda1_j - lambda2_i)^2 -- it separates by row, and the
// solve is k2 independent k1 x k1 ridge regressions rather than one big system.

#pragma once

#include <Eigen/Dense>

#include "fmap/spectral_basis.hpp"

namespace fmap {

struct FunctionalMapOptions {
  // Relative to descriptor preservation. Eigenvalues are normalized by the
  // largest first, so this means the same thing on any mesh.
  //
  // Swept on a SCAPE pair at k=60 (max|C^T C - I|, energy within 3 of the
  // diagonal): 0 -> 566, 25%; 0.01 -> 12.7, 94%; 1.0 -> 4.7, 99.4%; 100 -> 3.7,
  // 99.9% but 3.5x the descriptor residual. 1.0 is the knee.
  double commutativity_weight = 1.0;
};

struct FunctionalMap {
  Eigen::MatrixXd C;  // k2 x k1, so that C * a ~= b

  double descriptor_residual = 0.0;     // ||C A - B||_F / ||B||_F
  double commutativity_residual = 0.0;  // ||C D1 - D2 C||_F, normalized lambdas

  Eigen::Index source_size() const { return C.cols(); }
  Eigen::Index target_size() const { return C.rows(); }

  // max |C^T C - I|. Near zero for isometric shapes; large means the shapes
  // differ non-isometrically, the descriptors were weak, or k is too small.
  double orthogonality_error() const;
};

// Descriptors are per-vertex (n1 x d and n2 x d, same d), and are projected
// into their respective bases here.
FunctionalMap solve_functional_map(const SpectralBasis& source,
                                   const SpectralBasis& target,
                                   const Eigen::MatrixXd& source_descriptors,
                                   const Eigen::MatrixXd& target_descriptors,
                                   const FunctionalMapOptions& options = {});

}  // namespace fmap
