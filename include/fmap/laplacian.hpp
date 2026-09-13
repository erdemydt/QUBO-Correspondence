// Cotangent stiffness and lumped mass for L phi = lambda M phi.

#pragma once

#include <Eigen/Sparse>

#include "fmap/mesh.hpp"

namespace fmap {

struct LaplacianOperator {
  // PSD convention (L = D - W): lambda >= 0, lambda_0 = 0. The literature often
  // uses the opposite sign; this is the one the eigensolver wants.
  Eigen::SparseMatrix<double> L;

  // Lumped, a third of each incident triangle: entries sum to surface area and
  // M^{-1/2} is trivial, at negligible accuracy cost here.
  Eigen::SparseMatrix<double> M;

  Eigen::VectorXd mass;  // diagonal of M
  Eigen::Index size() const { return mass.size(); }

  Eigen::Index obtuse_triangles = 0;  // negative weights; many => L not PSD
};

LaplacianOperator build_laplacian(const Mesh& mesh);

}  // namespace fmap
