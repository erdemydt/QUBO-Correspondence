#include "fmap/descriptors.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace fmap {
namespace {

void validate(const SpectralBasis& basis, const DescriptorOptions& options) {
  if (basis.count() < 3) {
    throw std::runtime_error(
        "descriptors need at least 3 eigenpairs; got " +
        std::to_string(basis.count()));
  }
  if (options.num_samples < 1) {
    throw std::runtime_error("num_samples must be >= 1");
  }
  if (options.step < 1) {
    throw std::runtime_error("step must be >= 1");
  }
}

// Indices of the samples to keep, given the subsampling step.
std::vector<int> retained_samples(int num_samples, int step) {
  std::vector<int> keep;
  for (int i = 0; i < num_samples; i += step) keep.push_back(i);
  return keep;
}

// First meaningfully non-zero eigenvalue. lambda_0 is 0 by construction and WKS
// takes a logarithm, so the constant mode must be skipped; near-duplicates of 0
// also appear when the mesh has several connected components.
Eigen::Index first_nonzero_mode(const SpectralBasis& basis) {
  const double scale = basis.eigenvalues(basis.count() - 1);
  const double floor_value = 1e-8 * std::max(scale, 1e-300);
  for (Eigen::Index i = 0; i < basis.count(); ++i) {
    if (basis.eigenvalues(i) > floor_value) return i;
  }
  throw std::runtime_error(
      "spectrum is entirely zero; the mesh may be degenerate");
}

}  // namespace

void normalize_descriptors(Eigen::MatrixXd& descriptors,
                           const Eigen::VectorXd& mass) {
  if (descriptors.rows() != mass.size()) {
    throw std::runtime_error("descriptor rows do not match the mass vector");
  }
  for (Eigen::Index c = 0; c < descriptors.cols(); ++c) {
    const double norm = std::sqrt(
        descriptors.col(c).cwiseProduct(mass).dot(descriptors.col(c)));
    // A flat column is legitimate (an unreachable time scale); leave it rather
    // than amplify numerical noise.
    if (norm > 1e-12) descriptors.col(c) /= norm;
  }
}

Eigen::MatrixXd compute_hks(const SpectralBasis& basis,
                            const DescriptorOptions& options) {
  validate(basis, options);

  const Eigen::Index n = basis.size();
  const Eigen::Index k = basis.count();
  const Eigen::Index first = first_nonzero_mode(basis);

  const double lambda_min = basis.eigenvalues(first);
  const double lambda_max = basis.eigenvalues(k - 1);

  // Time range from Sun et al.: where the heat kernel carries information,
  // bounded by the spectrum's extremes and so adapting to mesh scale for free.
  const double kSpread = 4.0 * std::log(10.0);
  const double t_min = kSpread / lambda_max;
  const double t_max = kSpread / lambda_min;

  const std::vector<int> keep = retained_samples(options.num_samples,
                                                 options.step);
  Eigen::MatrixXd out(n, static_cast<Eigen::Index>(keep.size()));

  // Log spacing: diffusion is scale-free, so equal ratios of t carry equal
  // information.
  const double log_t_min = std::log(t_min);
  const double log_t_max = std::log(t_max);
  const double dt = (options.num_samples > 1)
                        ? (log_t_max - log_t_min) / (options.num_samples - 1)
                        : 0.0;

  const Eigen::MatrixXd phi_squared = basis.eigenvectors.array().square();

  for (std::size_t c = 0; c < keep.size(); ++c) {
    const double t = std::exp(log_t_min + dt * keep[c]);

    Eigen::VectorXd decay(k);
    for (Eigen::Index i = 0; i < k; ++i) {
      decay(i) = std::exp(-basis.eigenvalues(i) * t);
    }

    Eigen::VectorXd column = phi_squared * decay;

    // Divide by the heat trace, or the descriptor is dominated by its overall
    // magnitude -- which carries no correspondence information -- not its shape.
    const double trace = decay.sum();
    if (trace > 1e-300) column /= trace;

    out.col(static_cast<Eigen::Index>(c)) = column;
  }

  if (options.normalize) normalize_descriptors(out, basis.mass);
  return out;
}

Eigen::MatrixXd compute_wks(const SpectralBasis& basis,
                            const DescriptorOptions& options) {
  validate(basis, options);

  const Eigen::Index n = basis.size();
  const Eigen::Index k = basis.count();
  const Eigen::Index first = first_nonzero_mode(basis);

  // WKS works in log-energy, which is why lambda_0 = 0 must be excluded.
  const Eigen::Index num_modes = k - first;
  Eigen::VectorXd log_lambda(num_modes);
  for (Eigen::Index i = 0; i < num_modes; ++i) {
    log_lambda(i) = std::log(basis.eigenvalues(first + i));
  }

  const double e_min = log_lambda(0);
  // The 1.02 divisor is the reference implementation's: it pulls the top of the
  // range just inside the last eigenvalue, where truncation hurts most.
  const double e_max = log_lambda(num_modes - 1) / 1.02;

  const double de = (options.num_samples > 1)
                        ? (e_max - e_min) / (options.num_samples - 1)
                        : 0.0;
  const double sigma = de * options.wks_variance;
  if (!(sigma > 0.0)) {
    throw std::runtime_error(
        "WKS energy range collapsed; the spectrum may be degenerate");
  }
  const double inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);

  const std::vector<int> keep = retained_samples(options.num_samples,
                                                 options.step);
  Eigen::MatrixXd out(n, static_cast<Eigen::Index>(keep.size()));

  const Eigen::MatrixXd phi_squared =
      basis.eigenvectors.rightCols(num_modes).array().square();

  for (std::size_t c = 0; c < keep.size(); ++c) {
    const double e = e_min + de * keep[c];

    Eigen::VectorXd weights(num_modes);
    for (Eigen::Index i = 0; i < num_modes; ++i) {
      const double d = e - log_lambda(i);
      weights(i) = std::exp(-d * d * inv_two_sigma_sq);
    }

    // C_e in the paper: each band becomes a weighted average, not a sum, so
    // bands with few modes are not systematically smaller.
    const double total = weights.sum();
    Eigen::VectorXd column = phi_squared * weights;
    if (total > 1e-300) column /= total;

    out.col(static_cast<Eigen::Index>(c)) = column;
  }

  if (options.normalize) normalize_descriptors(out, basis.mass);
  return out;
}

Eigen::MatrixXd compute_descriptors(const SpectralBasis& basis,
                                    DescriptorType type,
                                    const DescriptorOptions& options) {
  switch (type) {
    case DescriptorType::HKS:
      return compute_hks(basis, options);
    case DescriptorType::WKS:
      return compute_wks(basis, options);
  }
  throw std::runtime_error("unknown descriptor type");
}

Eigen::MatrixXd compute_all_descriptors(const SpectralBasis& basis,
                                        const DescriptorOptions& options) {
  const Eigen::MatrixXd hks = compute_hks(basis, options);
  const Eigen::MatrixXd wks = compute_wks(basis, options);

  Eigen::MatrixXd out(hks.rows(), hks.cols() + wks.cols());
  out << hks, wks;
  return out;
}

}  // namespace fmap
