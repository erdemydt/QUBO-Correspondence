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

  // A is k1 x d, B is k2 x d.
  const Eigen::MatrixXd A = source.project(source_descriptors);
  const Eigen::MatrixXd B = target.project(target_descriptors);

  if (A.cols() == 0) {
    throw std::runtime_error("no descriptors supplied; C is unconstrained");
  }

  // Normalized so the weights are O(1) and the knob is mesh-independent.
  const double scale1 = source.eigenvalues(k1 - 1);
  const double scale2 = target.eigenvalues(k2 - 1);
  if (!(scale1 > 0.0) || !(scale2 > 0.0)) {
    throw std::runtime_error("degenerate spectrum; cannot normalize");
  }
  const Eigen::VectorXd lambda1 = source.eigenvalues / scale1;
  const Eigen::VectorXd lambda2 = target.eigenvalues / scale2;

  // Shared across all rows: A A^T is k1 x k1 and row-independent.
  const Eigen::MatrixXd AAt = A * A.transpose();
  const Eigen::MatrixXd ABt = A * B.transpose();  // k1 x k2

  FunctionalMap result;
  result.C.resize(k2, k1);

  // One ridge regression per row: the penalty is diagonal but row-dependent,
  // which is what decouples them. A A^T has rank at most d, so d < k leaves it
  // singular; the penalty fixes that everywhere but (0,0), where both lambda_0
  // are 0 and it vanishes. The Tikhonov ridge covers that hole, scaled to the
  // data so it does not bias the fit.
  const double ridge = 1e-9 * AAt.trace() / static_cast<double>(k1);

  Eigen::MatrixXd system(k1, k1);
  for (Eigen::Index i = 0; i < k2; ++i) {
    system = AAt;
    for (Eigen::Index j = 0; j < k1; ++j) {
      const double gap = lambda1(j) - lambda2(i);
      system(j, j) += options.commutativity_weight * gap * gap + ridge;
    }

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
