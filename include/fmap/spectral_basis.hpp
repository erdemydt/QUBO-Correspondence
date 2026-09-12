// The k smallest eigenpairs of L phi = lambda M phi. The eigenfunctions form a
// basis for functions on the surface, ordered smooth to oscillatory, and a
// functional map is a change of basis between two of them.
//
// Smallest is the awkward end of the spectrum -- plain Lanczos converges there
// very slowly -- so this uses shift-invert, factorizing (L - sigma*M) once.

#pragma once

#include <Eigen/Dense>

#include "fmap/laplacian.hpp"

namespace fmap {

struct SpectralOptions {
  int k = 100;  // the literature's usual working value

  // Shift, as a fraction of the mean eigenvalue magnitude. Must be negative:
  // lambda_0 is exactly 0, so shifting to 0 makes the factorization singular.
  // Relative because eigenvalues carry units of 1/area (cotangent weights are
  // dimensionless, mass is an area), and mesh areas vary hugely across datasets.
  double shift_fraction = -1e-6;

  double ncv_factor = 2.0;  // Lanczos subspace size, as a multiple of k
  int max_iterations = 1000;
  double tolerance = 1e-10;
};

struct SpectralBasis {
  Eigen::VectorXd eigenvalues;   // ascending; eigenvalues(0) ~ 0
  Eigen::MatrixXd eigenvectors;  // n x k, M-orthonormal: phi_i^T M phi_j = d_ij
  Eigen::VectorXd mass;          // needed to project into this basis

  Eigen::Index size() const { return eigenvectors.rows(); }   // vertices
  Eigen::Index count() const { return eigenvectors.cols(); }  // basis functions

  // Phi^T M f, the M-weighted inner product -- not a plain dot product, which
  // is a classic and quiet source of error.
  Eigen::VectorXd project(const Eigen::VectorXd& f) const;
  Eigen::MatrixXd project(const Eigen::MatrixXd& functions) const;

  Eigen::VectorXd reconstruct(const Eigen::VectorXd& coefficients) const;
  Eigen::MatrixXd reconstruct(const Eigen::MatrixXd& coefficients) const;

  // max |Phi^T M Phi - I|; a sanity metric, should be near machine precision.
  double orthonormality_error() const;
};

// Throws std::runtime_error on non-convergence rather than returning a
// partially converged basis that would silently corrupt every later stage.
SpectralBasis compute_spectral_basis(const LaplacianOperator& op,
                                     const SpectralOptions& options = {});

}  // namespace fmap
