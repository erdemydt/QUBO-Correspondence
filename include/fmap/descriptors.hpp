// Pointwise spectral descriptors -- the linear constraints that pin down an
// otherwise under-determined functional map. Sampling ranges come from the
// eigenvalues, not constants, which is what makes them scale-invariant: TOSCA
// meshes have ~10^4 times the area of SCAPE ones.

#pragma once

#include <Eigen/Dense>

#include "fmap/spectral_basis.hpp"

namespace fmap {

enum class DescriptorType {
  HKS,  // Sun et al. 2009
  WKS,  // Aubry et al. 2011
};

struct DescriptorOptions {
  int num_samples = 100;

  // Keep every step-th sample. Must leave more descriptors than basis
  // functions: the map's normal equations have rank at most d, so d < k is
  // singular. HKS and WKS concatenate, so the default gives 2 * num_samples.
  int step = 1;

  double wks_variance = 6.0;  // log-energy window, in sample spacings
  bool normalize = true;      // unit mass-weighted norm, so none dominates
};

// h(x,t) = sum_i exp(-lambda_i t) phi_i(x)^2. Returns n x d.
Eigen::MatrixXd compute_hks(const SpectralBasis& basis,
                            const DescriptorOptions& options = {});

// w(x,e) = C_e sum_i exp(-(e - log lambda_i)^2 / 2 sigma^2) phi_i(x)^2. Bands
// the spectrum in log-energy rather than damping it, separating scales more
// sharply than HKS. Returns n x d.
Eigen::MatrixXd compute_wks(const SpectralBasis& basis,
                            const DescriptorOptions& options = {});

Eigen::MatrixXd compute_descriptors(const SpectralBasis& basis,
                                    DescriptorType type,
                                    const DescriptorOptions& options = {});

// Both, columns concatenated -- they capture different structure.
Eigen::MatrixXd compute_all_descriptors(const SpectralBasis& basis,
                                        const DescriptorOptions& options = {});

void normalize_descriptors(Eigen::MatrixXd& descriptors,
                           const Eigen::VectorXd& mass);

}  // namespace fmap
