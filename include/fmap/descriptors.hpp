// Pointwise spectral descriptors. These are what pin down an otherwise
// under-determined functional map: corresponding points on two poses get
// near-identical values, and projected into the basis they become the linear
// constraints the map solve fits.
//
// Both take their sampling range from the eigenvalues rather than from
// constants, which makes them scale-invariant -- TOSCA meshes have ~10^4 times
// the area of SCAPE meshes, so any fixed time or energy range would be
// meaningless on one of the two.

#pragma once

#include <Eigen/Dense>

#include "fmap/spectral_basis.hpp"

namespace fmap {

enum class DescriptorType {
  HKS,  // Heat Kernel Signature (Sun et al. 2009)
  WKS,  // Wave Kernel Signature (Aubry et al. 2011)
};

struct DescriptorOptions {
  int num_samples = 100;  // time (HKS) or energy (WKS) samples

  // Keep every step-th sample. Must leave more descriptors than basis
  // functions: the map's normal equations have rank at most d, so d < k is
  // singular. HKS and WKS are concatenated, so the default gives 2 *
  // num_samples -- comfortably above the usual k = 100. Raising it trades
  // constraints for speed, and neighbouring samples are highly correlated.
  int step = 1;

  // WKS log-energy window width, in multiples of the sample spacing. 6 is the
  // original paper's reference value.
  double wks_variance = 6.0;

  // Scale each function to unit mass-weighted L2 norm so none dominates the fit.
  bool normalize = true;
};

// h(x, t) = sum_i exp(-lambda_i t) phi_i(x)^2 -- heat remaining at x at time t
// after a unit impulse there. Small t probes curvature, large t global
// structure. Returns n x d, one column per retained sample.
Eigen::MatrixXd compute_hks(const SpectralBasis& basis,
                            const DescriptorOptions& options = {});

// w(x, e) = C_e sum_i exp(-(e - log lambda_i)^2 / 2 sigma^2) phi_i(x)^2 --
// bands the spectrum in log-energy instead of damping it, separating scales
// more sharply than HKS. Returns n x d.
Eigen::MatrixXd compute_wks(const SpectralBasis& basis,
                            const DescriptorOptions& options = {});

Eigen::MatrixXd compute_descriptors(const SpectralBasis& basis,
                                    DescriptorType type,
                                    const DescriptorOptions& options = {});

// HKS and WKS columns concatenated; they capture different structure, and
// using both is the usual default.
Eigen::MatrixXd compute_all_descriptors(const SpectralBasis& basis,
                                        const DescriptorOptions& options = {});

// Rescale each column to unit mass-weighted L2 norm, in place.
void normalize_descriptors(Eigen::MatrixXd& descriptors,
                           const Eigen::VectorXd& mass);

}  // namespace fmap
