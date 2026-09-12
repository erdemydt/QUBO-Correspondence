#include "fmap/functional_map.hpp"

#include <cmath>
#include <stdexcept>

namespace fmap {

double FunctionalMap::orthogonality_error() const {
  const Eigen::MatrixXd gram = C.transpose() * C;
  const Eigen::MatrixXd identity =
      Eigen::MatrixXd::Identity(gram.rows(), gram.cols());
  return (gram - identity).cwiseAbs().maxCoeff();
}

FunctionalMap solve_functional_map(const SpectralBasis& source,
                                   const SpectralBasis& target,
                                   const Eigen::MatrixXd& source_descriptors,
                                   const Eigen::MatrixXd& target_descriptors,
                                   const FunctionalMapOptions& options) {
  if (source_descriptors.cols() != target_descriptors.cols()) {
    throw std::runtime_error(
        "descriptor count differs between shapes: " +
        std::to_string(source_descriptors.cols()) + " vs " +
        std::to_string(target_descriptors.cols()) +
        "; the same descriptors must be computed on both");
  }
  if (source_descriptors.rows() != source.size()) {
    throw std::runtime_error("source descriptors do not match the source mesh");
  }
  if (target_descriptors.rows() != target.size()) {
    throw std::runtime_error("target descriptors do not match the target mesh");
  }
  if (options.commutativity_weight < 0.0) {
    throw std::runtime_error("commutativity_weight must be non-negative");
  }

  const Eigen::Index k1 = source.count();
  const Eigen::Index k2 = target.count();

  // Project the descriptors into each basis. A is k1 x d, B is k2 x d.
  const Eigen::MatrixXd A = source.project(source_descriptors);
  const Eigen::MatrixXd B = target.project(target_descriptors);

  if (A.cols() == 0) {
    throw std::runtime_error("no descriptors supplied; C is unconstrained");
  }

  // Normalize eigenvalues so the commutativity weights are O(1) and the
  // weight parameter means the same thing regardless of mesh scale.
  const double scale1 = source.eigenvalues(k1 - 1);
  const double scale2 = target.eigenvalues(k2 - 1);
  if (!(scale1 > 0.0) || !(scale2 > 0.0)) {
    throw std::runtime_error("degenerate spectrum; cannot normalize");
  }
  const Eigen::VectorXd lambda1 = source.eigenvalues / scale1;
  const Eigen::VectorXd lambda2 = target.eigenvalues / scale2;

  // Normal equations, shared across all rows: A A^T is k1 x k1 and does not
  // depend on which row of C is being solved.
  const Eigen::MatrixXd AAt = A * A.transpose();
  const Eigen::MatrixXd ABt = A * B.transpose();  // k1 x k2

  FunctionalMap result;
  result.C.resize(k2, k1);

  // One ridge regression per row of C. The penalty is diagonal but differs
  // per row, which is exactly why this decomposes so cleanly: row i of C pairs
  // target frequency lambda2_i against every source frequency, and entries
  // pairing very different frequencies are pushed toward zero.
  // A A^T has rank at most d, so with fewer descriptors than basis functions it
  // is singular. The commutativity penalty regularizes most of that away, but
  // not entry (0, 0): lambda1_0 and lambda2_0 are both zero on a closed
  // surface, so the penalty there vanishes identically. A small Tikhonov ridge
  // covers that hole. It is scaled to the data so it stays negligible relative
  // to the descriptor term rather than biasing the fit.
  const double ridge = 1e-9 * AAt.trace() / static_cast<double>(k1);

  Eigen::MatrixXd system(k1, k1);
  for (Eigen::Index i = 0; i < k2; ++i) {
    system = AAt;
    for (Eigen::Index j = 0; j < k1; ++j) {
      const double gap = lambda1(j) - lambda2(i);
      system(j, j) += options.commutativity_weight * gap * gap + ridge;
    }

    // LDLT: the system is symmetric and, with the ridge above, positive
    // definite.
    const Eigen::LDLT<Eigen::MatrixXd> solver(system);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error(
          "functional map solve failed at row " + std::to_string(i) + ". The "
          "normal equations are rank deficient: there are " +
          std::to_string(A.cols()) + " descriptor constraints for a basis of "
          "size " + std::to_string(k1) +
          ". Supply more descriptors (lower DescriptorOptions::step), reduce "
          "the basis size, or raise commutativity_weight.");
    }
    result.C.row(i) = solver.solve(ABt.col(i)).transpose();
  }

  if (!result.C.allFinite()) {
    throw std::runtime_error(
        "functional map solve produced non-finite entries; the descriptor "
        "system is likely rank deficient");
  }

  // Residuals, for reporting.
  const double b_norm = B.norm();
  result.descriptor_residual =
      (result.C * A - B).norm() / (b_norm > 0.0 ? b_norm : 1.0);

  double commutativity = 0.0;
  for (Eigen::Index i = 0; i < k2; ++i) {
    for (Eigen::Index j = 0; j < k1; ++j) {
      const double gap = lambda1(j) - lambda2(i);
      const double term = result.C(i, j) * gap;
      commutativity += term * term;
    }
  }
  result.commutativity_residual = std::sqrt(commutativity);

  return result;
}

}  // namespace fmap
