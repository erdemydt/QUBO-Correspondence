// Laplacian eigenbasis -- the "spectral" half of functional maps.
//
// Solves the generalized eigenproblem
//
//     L phi = lambda M phi
//
// for the k smallest eigenpairs. The eigenfunctions are a basis for functions
// on the surface, ordered from smooth to oscillatory, and a functional map is
// nothing more than a change of basis between two of these.
//
// The smallest eigenvalues are the awkward end of the spectrum: plain Lanczos
// converges there very slowly. This uses shift-invert instead, which factorizes
// (L - sigma*M) once and converges in a handful of iterations.

#pragma once

#include <Eigen/Dense>

#include "fmap/laplacian.hpp"

namespace fmap {

struct SpectralOptions {
  // Number of eigenpairs. 100 is the usual working value in the functional
  // maps literature: enough to capture shape structure, few enough that the
  // map matrix C stays small.
  int k = 100;

  // Shift, as a fraction of the mean eigenvalue magnitude. Must be negative:
  // lambda_0 is exactly 0, so shifting to 0 would make the factorization
  // singular. Expressed relatively because the absolute scale of the spectrum
  // varies with mesh area -- cotangent weights are dimensionless while mass is
  // an area, so eigenvalues carry units of 1/area.
  double shift_fraction = -1e-6;

  // Lanczos subspace size, as a multiple of k. Larger converges in fewer
  // iterations but costs memory.
  double ncv_factor = 2.0;

  int max_iterations = 1000;
  double tolerance = 1e-10;
};

struct SpectralBasis {
  // k eigenvalues in ascending order. lambda(0) is ~0 (the constant function).
  Eigen::VectorXd eigenvalues;

  // n x k eigenfunctions, M-orthonormal: phi_i^T M phi_j = delta_ij.
  Eigen::MatrixXd eigenvectors;

  // Lumped mass, needed to project functions into this basis.
  Eigen::VectorXd mass;

  Eigen::Index size() const { return eigenvectors.rows(); }   // vertices
  Eigen::Index count() const { return eigenvectors.cols(); }  // basis functions

  // Project a per-vertex function into the basis: returns k coefficients.
  // This is the M-weighted inner product Phi^T M f, not a plain dot product --
  // using the wrong one is a classic and quiet source of error.
  Eigen::VectorXd project(const Eigen::VectorXd& f) const;

  // Project d functions at once: n x d in, k x d out.
  Eigen::MatrixXd project(const Eigen::MatrixXd& functions) const;

  // Reconstruct a per-vertex function from its coefficients.
  Eigen::VectorXd reconstruct(const Eigen::VectorXd& coefficients) const;
  Eigen::MatrixXd reconstruct(const Eigen::MatrixXd& coefficients) const;

  // Largest deviation from M-orthonormality, i.e. max |Phi^T M Phi - I|.
  // A sanity metric; should be near machine precision.
  double orthonormality_error() const;
};

// Solve for the k smallest eigenpairs of (L, M).
//
// Throws std::runtime_error if the solver fails to converge, rather than
// returning a partially converged basis that would silently corrupt every
// later stage.
SpectralBasis compute_spectral_basis(const LaplacianOperator& op,
                                     const SpectralOptions& options = {});

}  // namespace fmap
