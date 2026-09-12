// Solving for the functional map matrix C.
//
// A functional map represents a correspondence not as a pairing of points, but
// as a linear operator between function spaces. Given spectral bases Phi1 and
// Phi2, C is the small k2 x k1 matrix taking a function's coefficients on the
// source to its coefficients on the target. Replacing an n x n permutation with
// a k x k dense matrix is the whole point of the method.
//
// C is found by least squares from two energies:
//
//   1. Descriptor preservation, ||C A - B||^2. Corresponding points have
//      matching descriptors, so their projections must map to one another.
//   2. Laplacian commutativity, ||C Delta1 - Delta2 C||^2. A map that is a real
//      isometry commutes with the Laplace-Beltrami operator. This term is what
//      makes the result usable -- descriptors alone leave C badly
//      under-determined.
//
// Both Laplacians are diagonal in their own eigenbasis, so the commutativity
// term collapses to a per-entry weight:
//
//     ||C Delta1 - Delta2 C||_F^2 = sum_ij C_ij^2 (lambda1_j - lambda2_i)^2
//
// which means the energy separates by row of C. Rather than one k1*k2 x k1*k2
// system, we solve k2 independent k1 x k1 ridge regressions -- each penalising
// entries that pair mismatched frequencies.

#pragma once

#include <Eigen/Dense>

#include "fmap/spectral_basis.hpp"

namespace fmap {

struct FunctionalMapOptions {
  // Weight on the commutativity term, relative to descriptor preservation.
  // Eigenvalues are normalized by the largest before use, so this is a
  // dimensionless knob that means the same thing on any mesh.
  //
  // Chosen by sweeping a SCAPE pose pair at k=60. The term matters enormously:
  //
  //   weight   descriptor residual   max|C^T C - I|   energy within 3 of diag
  //        0               0.0031            566.3                     25.4%
  //     0.01               0.0038             12.7                     94.1%
  //      1.0               0.0076              4.7                     99.4%
  //      100               0.0267              3.7                     99.9%
  //
  // 1.0 sits at the knee: near-best orthogonality while the descriptor fit is
  // still tight. Past it, C is forced toward diagonal at the cost of actually
  // matching the descriptors.
  double commutativity_weight = 1.0;
};

struct FunctionalMap {
  // k2 x k1. Maps source coefficients to target coefficients: C * a ~= b.
  Eigen::MatrixXd C;

  // ||C A - B||_F, normalized by ||B||_F. How well the descriptors are matched.
  double descriptor_residual = 0.0;

  // ||C Delta1 - Delta2 C||_F, with normalized eigenvalues. How close C is to
  // commuting with the Laplacian -- a proxy for how isometric the map is.
  double commutativity_residual = 0.0;

  Eigen::Index source_size() const { return C.cols(); }
  Eigen::Index target_size() const { return C.rows(); }

  // Largest deviation from orthogonality, max |C^T C - I|.
  //
  // For a map between genuinely isometric shapes C should be close to
  // orthogonal. A large value means the shapes differ non-isometrically, the
  // descriptors were uninformative, or the basis size is too small.
  double orthogonality_error() const;
};

// Solve for C.
//
// `source_descriptors` and `target_descriptors` are per-vertex matrices
// (n1 x d and n2 x d) with the same d columns, i.e. the same descriptor
// computed on both shapes. They are projected into their respective bases here.
FunctionalMap solve_functional_map(const SpectralBasis& source,
                                   const SpectralBasis& target,
                                   const Eigen::MatrixXd& source_descriptors,
                                   const Eigen::MatrixXd& target_descriptors,
                                   const FunctionalMapOptions& options = {});

}  // namespace fmap
