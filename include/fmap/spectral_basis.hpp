// The k smallest eigenpairs of L phi = lambda M phi, ordered smooth to
// oscillatory. Shift-invert, because plain Lanczos converges badly down there.

#pragma once

#include <Eigen/Dense>

#include "fmap/laplacian.hpp"

namespace fmap {

struct SpectralOptions {
  int k = 100;

  // Fraction of the mean eigenvalue magnitude. Must be negative: lambda_0 is 0,
  // so shifting to 0 makes the factorization singular. Relative because
  // eigenvalues scale as 1/area and mesh areas vary hugely across datasets.
  double shift_fraction = -1e-6;

  double ncv_factor = 2.0;  // Lanczos subspace, as a multiple of k
  int max_iterations = 1000;
  double tolerance = 1e-10;
};

struct SpectralBasis {
  Eigen::VectorXd eigenvalues;   // ascending, eigenvalues(0) ~ 0
  Eigen::MatrixXd eigenvectors;  // n x k, M-orthonormal
  Eigen::VectorXd mass;

  Eigen::Index size() const { return eigenvectors.rows(); }   // vertices
  Eigen::Index count() const { return eigenvectors.cols(); }  // basis functions

  // Phi^T M f -- mass-weighted, not a plain dot product. Getting this wrong is
  // a classic and quiet source of error.
  Eigen::VectorXd project(const Eigen::VectorXd& f) const;
  Eigen::MatrixXd project(const Eigen::MatrixXd& functions) const;

  Eigen::VectorXd reconstruct(const Eigen::VectorXd& coefficients) const;
  Eigen::MatrixXd reconstruct(const Eigen::MatrixXd& coefficients) const;

  double orthonormality_error() const;  // max |Phi^T M Phi - I|
};

// Throws on non-convergence rather than returning a partial basis that would
// silently corrupt every later stage.
SpectralBasis compute_spectral_basis(const LaplacianOperator& op,
                                     const SpectralOptions& options = {});

}  // namespace fmap
