// Pointwise spectral descriptors: HKS and WKS.
//
// A functional map is under-determined by structure alone -- descriptors are
// what pin it down. Each descriptor assigns every vertex a vector that depends
// only on intrinsic geometry, so corresponding points on two poses of the same
// shape get near-identical values. Projected into the spectral basis, they
// become the linear constraints that the map solve fits.
//
// Both descriptors take their sampling range from the eigenvalues rather than
// from fixed constants. That makes them scale-invariant, which matters here:
// TOSCA meshes have ~10^4 times the surface area of SCAPE meshes, so their
// eigenvalues differ by the same factor, and any hardcoded time or energy
// range would be meaningless on one of the two.

#pragma once

#include <Eigen/Dense>

#include "fmap/spectral_basis.hpp"

namespace fmap {

enum class DescriptorType {
  HKS,  // Heat Kernel Signature (Sun et al. 2009)
  WKS,  // Wave Kernel Signature (Aubry et al. 2011)
};

struct DescriptorOptions {
  // Number of time (HKS) or energy (WKS) samples to evaluate.
  int num_samples = 100;

  // Keep every step-th sample as a constraint.
  //
  // Defaults to 1 (keep everything). The functional map solve needs the
  // descriptor constraints to over-determine the basis: with d descriptors and
  // a basis of size k, its normal equations have rank at most d, so d < k
  // leaves the system singular. Since HKS and WKS are concatenated, the default
  // yields 2 * num_samples constraints, comfortably above the usual k = 100.
  //
  // Raising this trades constraints for speed; neighbouring samples are highly
  // correlated, so the loss is smaller than the count suggests.
  int step = 1;

  // Width of the WKS log-energy window, as a multiple of the sample spacing.
  // 6 is the value from the original paper's reference implementation.
  double wks_variance = 6.0;

  // Scale each descriptor function to unit mass-weighted L2 norm, so that no
  // single one dominates the least-squares fit in functional_map.hpp.
  bool normalize = true;
};

// Heat Kernel Signature.
//
//     h(x, t) = sum_i exp(-lambda_i t) phi_i(x)^2
//
// Physically: how much heat remains at x at time t after a unit impulse there.
// Small t probes local curvature, large t probes global structure.
//
// Returns an n x d matrix, one column per retained time sample.
Eigen::MatrixXd compute_hks(const SpectralBasis& basis,
                            const DescriptorOptions& options = {});

// Wave Kernel Signature.
//
//     w(x, e) = C_e sum_i exp(-(e - log lambda_i)^2 / 2 sigma^2) phi_i(x)^2
//
// Bands the spectrum in log-energy instead of damping it, which separates
// scales more sharply than HKS and is usually the better discriminator.
//
// Returns an n x d matrix, one column per retained energy sample.
Eigen::MatrixXd compute_wks(const SpectralBasis& basis,
                            const DescriptorOptions& options = {});

// Dispatch on type.
Eigen::MatrixXd compute_descriptors(const SpectralBasis& basis,
                                    DescriptorType type,
                                    const DescriptorOptions& options = {});

// HKS and WKS side by side, columns concatenated. The two capture different
// structure, and using both is the usual default.
Eigen::MatrixXd compute_all_descriptors(const SpectralBasis& basis,
                                        const DescriptorOptions& options = {});

// Rescale each column to unit mass-weighted L2 norm, in place.
void normalize_descriptors(Eigen::MatrixXd& descriptors,
                           const Eigen::VectorXd& mass);

}  // namespace fmap
