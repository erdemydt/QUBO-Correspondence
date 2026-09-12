// Cotangent stiffness L and lumped mass M, defining the generalized
// eigenproblem L phi = lambda M phi whose eigenfunctions are the spectral
// basis. See spectral_basis.hpp for the solve.

#pragma once

#include <Eigen/Sparse>

#include "fmap/mesh.hpp"

namespace fmap {

struct LaplacianOperator {
  // n x n, symmetric, rows sum to zero. Positive semi-definite convention
  // (L = D - W, off-diagonals -w_ij), so eigenvalues are >= 0 with lambda_0 = 0
  // on the constant function. Much of the literature uses the opposite sign;
  // this is the one the eigensolver wants.
  Eigen::SparseMatrix<double> L;

  // Lumped: M_ii is a third of the incident triangle area, so entries sum to
  // the surface area. Diagonal keeps the eigenproblem cheap and M^{-1/2}
  // trivial; the accuracy cost over a full Galerkin mass matrix is not material
  // at these mesh densities.
  Eigen::SparseMatrix<double> M;

  Eigen::VectorXd mass;  // diagonal of M, which is what most callers want

  Eigen::Index size() const { return mass.size(); }

  // Obtuse triangles give negative cotangent weights. A few are harmless; a
  // large fraction means L may not be PSD and the eigensolve can return small
  // negative eigenvalues.
  Eigen::Index obtuse_triangles = 0;
};

LaplacianOperator build_laplacian(const Mesh& mesh);

}  // namespace fmap
