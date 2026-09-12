#include "fmap/spectral_basis.hpp"

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/SymGEigsShiftSolver.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace fmap {

Eigen::VectorXd SpectralBasis::project(const Eigen::VectorXd& f) const {
  if (f.size() != size()) {
    throw std::runtime_error("project(): function has " +
                             std::to_string(f.size()) + " values but the mesh "
                             "has " + std::to_string(size()) + " vertices");
  }
  return eigenvectors.transpose() * (mass.asDiagonal() * f);
}

Eigen::MatrixXd SpectralBasis::project(const Eigen::MatrixXd& functions) const {
  if (functions.rows() != size()) {
    throw std::runtime_error("project(): functions have " +
                             std::to_string(functions.rows()) + " rows but the "
                             "mesh has " + std::to_string(size()) +
                             " vertices");
  }
  return eigenvectors.transpose() * (mass.asDiagonal() * functions);
}

Eigen::VectorXd SpectralBasis::reconstruct(
    const Eigen::VectorXd& coefficients) const {
  return eigenvectors * coefficients;
}

Eigen::MatrixXd SpectralBasis::reconstruct(
    const Eigen::MatrixXd& coefficients) const {
  return eigenvectors * coefficients;
}

double SpectralBasis::orthonormality_error() const {
  const Eigen::MatrixXd gram =
      eigenvectors.transpose() * (mass.asDiagonal() * eigenvectors);
  const Eigen::MatrixXd identity =
      Eigen::MatrixXd::Identity(gram.rows(), gram.cols());
  return (gram - identity).cwiseAbs().maxCoeff();
}

SpectralBasis compute_spectral_basis(const LaplacianOperator& op,
                                     const SpectralOptions& options) {
  const Eigen::Index n = op.size();
  if (options.k <= 0) {
    throw std::runtime_error("spectral basis needs k >= 1");
  }
  if (options.k >= n) {
    throw std::runtime_error(
        "k = " + std::to_string(options.k) +
        " must be smaller than the vertex count " + std::to_string(n) +
        "; shift-invert needs room for a Lanczos subspace");
  }
  if (options.shift_fraction >= 0.0) {
    throw std::runtime_error(
        "shift_fraction must be negative: lambda_0 is exactly 0, so a "
        "non-negative shift makes (L - sigma*M) singular");
  }

  // Rough magnitude of a typical eigenvalue, used to place the shift on the
  // same scale as the spectrum. sum(L_ii)/sum(M_ii) is a Rayleigh quotient
  // over the diagonal and carries the right units (1/area).
  const double scale = op.L.diagonal().sum() / op.mass.sum();
  if (!(scale > 0.0) || !std::isfinite(scale)) {
    throw std::runtime_error("degenerate Laplacian: cannot estimate spectrum "
                             "scale");
  }
  const double sigma = options.shift_fraction * scale;

  // Spectra requires k < ncv <= n. A larger subspace converges in fewer
  // restarts at the cost of memory.
  const Eigen::Index wanted_ncv =
      static_cast<Eigen::Index>(options.ncv_factor * options.k) + 1;
  const int ncv = static_cast<int>(std::min<Eigen::Index>(n, wanted_ncv));
  if (ncv <= options.k) {
    throw std::runtime_error("ncv must exceed k; mesh is too small for k = " +
                             std::to_string(options.k));
  }

  using OpType = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
  using BOpType = Spectra::SparseSymMatProd<double>;

  OpType shift_invert(op.L, op.M);
  BOpType mass_product(op.M);

  Spectra::SymGEigsShiftSolver<OpType, BOpType, Spectra::GEigsMode::ShiftInvert>
      solver(shift_invert, mass_product, options.k, ncv, sigma);

  solver.init();
  const int converged =
      solver.compute(Spectra::SortRule::LargestMagn, options.max_iterations,
                     options.tolerance);

  if (solver.info() != Spectra::CompInfo::Successful || converged < options.k) {
    std::ostringstream msg;
    msg << "eigensolver failed: converged " << converged << " of "
        << options.k << " requested eigenpairs";
    if (solver.info() == Spectra::CompInfo::NotConverging) {
      msg << " (not converging; try a larger ncv_factor or more iterations)";
    } else if (solver.info() == Spectra::CompInfo::NumericalIssue) {
      msg << " (numerical issue in the shift-invert factorization; the mesh "
             "may have degenerate geometry)";
    }
    throw std::runtime_error(msg.str());
  }

  SpectralBasis basis;
  basis.mass = op.mass;

  // Shift-invert finds the eigenvalues of the transformed problem in
  // descending magnitude, which is ascending distance from sigma. Since sigma
  // sits just below 0 and the spectrum is non-negative, that means the
  // eigenvalues arrive largest-first and need reversing.
  const Eigen::VectorXd raw_values = solver.eigenvalues();
  const Eigen::MatrixXd raw_vectors = solver.eigenvectors();

  basis.eigenvalues = raw_values.reverse();
  basis.eigenvectors = raw_vectors.rowwise().reverse();

  // Guard the assumption above rather than trusting it: if a future Spectra
  // release changes the ordering, this fails loudly instead of producing a
  // basis whose modes are in the wrong order.
  for (Eigen::Index i = 1; i < basis.eigenvalues.size(); ++i) {
    if (basis.eigenvalues(i) < basis.eigenvalues(i - 1) - 1e-9 * scale) {
      throw std::runtime_error(
          "eigenvalues are not ascending after reversal; the solver's output "
          "ordering is not what this code assumes");
    }
  }

  // The first eigenvalue is analytically 0 on a closed surface. Tiny negative
  // values are normal -- roughly 30% of triangles in these datasets are obtuse,
  // which produces negative cotangent weights -- so clamp rather than reject.
  for (Eigen::Index i = 0; i < basis.eigenvalues.size(); ++i) {
    if (basis.eigenvalues(i) < 0.0) basis.eigenvalues(i) = 0.0;
  }

  // Spectra returns M-orthonormal vectors, but normalize explicitly so the
  // guarantee holds regardless of solver internals. Callers rely on it for
  // projection to be a simple transpose.
  for (Eigen::Index c = 0; c < basis.eigenvectors.cols(); ++c) {
    const double norm = std::sqrt(
        basis.eigenvectors.col(c).cwiseProduct(op.mass).dot(
            basis.eigenvectors.col(c)));
    if (norm > 0.0) basis.eigenvectors.col(c) /= norm;
  }

  // Sign is arbitrary for an eigenvector, and Spectra's choice is not stable
  // across meshes. Fix it by convention so repeated runs on the same mesh are
  // reproducible. This does NOT make signs consistent *between* two different
  // meshes -- resolving that is precisely the functional map's job.
  for (Eigen::Index c = 0; c < basis.eigenvectors.cols(); ++c) {
    if (basis.eigenvectors.col(c).sum() < 0.0) {
      basis.eigenvectors.col(c) *= -1.0;
    }
  }

  return basis;
}

}  // namespace fmap
