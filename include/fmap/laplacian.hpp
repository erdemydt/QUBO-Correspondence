// Discrete Laplace-Beltrami operator on a triangle mesh.
//
// Builds the cotangent stiffness matrix L and the lumped mass matrix M, which
// together define the generalized eigenproblem
//
//     L phi = lambda M phi
//
// whose eigenfunctions form the spectral basis the functional map is expressed
// in. See spectral_basis.hpp for the solve.

#pragma once

#include <Eigen/Sparse>

#include "fmap/mesh.hpp"

namespace fmap {

struct LaplacianOperator {
  // Cotangent stiffness matrix, n x n. Symmetric, rows sum to zero, and
  // positive semi-definite for well-shaped meshes.
  //
  // Sign convention: L is positive semi-definite, i.e. L = D - W with
  // off-diagonals -w_ij. Eigenvalues are therefore >= 0, with lambda_0 = 0 on
  // the constant function. Much of the literature writes the operator with the
  // opposite sign; this convention is the one the eigensolver wants.
  Eigen::SparseMatrix<double> L;

  // Lumped (diagonal) mass matrix: M_ii is a third of the total area of the
  // triangles incident to vertex i, so the entries sum to the surface area.
  //
  // Lumped rather than the full Galerkin mass matrix because it keeps M
  // diagonal, which makes the generalized eigenproblem cheap and M^{-1/2}
  // trivial. The accuracy difference is not material at these mesh densities.
  Eigen::SparseMatrix<double> M;

  // Diagonal of M, kept separately since most callers want it as a vector.
  Eigen::VectorXd mass;

  Eigen::Index size() const { return mass.size(); }

  // Number of triangles with an obtuse angle, which produce negative cotangent
  // weights. A few are harmless; a large fraction means L may not be positive
  // semi-definite and the eigensolve can return small negative eigenvalues.
  Eigen::Index obtuse_triangles = 0;
};

// Assemble L and M for a mesh.
LaplacianOperator build_laplacian(const Mesh& mesh);

}  // namespace fmap
