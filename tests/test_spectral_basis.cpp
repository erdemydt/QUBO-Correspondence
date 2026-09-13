// The eigenbasis properties every later stage assumes: ascending non-negative
// eigenvalues, a constant first mode, M-orthonormality, and low-pass round trip.

#include "fmap/spectral_basis.hpp"

#include <chrono>

#include "test_support.hpp"

using namespace fmap::test;
using fmap::LaplacianOperator;
using fmap::Mesh;
using fmap::SpectralBasis;
using fmap::SpectralOptions;

namespace {

void test_spectrum(const char* spec, int k) {
  section(std::string("spectrum of ") + spec + " (k=" + std::to_string(k) + ")");

  const Mesh mesh = fmap::load_mesh(spec);
  const LaplacianOperator op = fmap::build_laplacian(mesh);

  SpectralOptions options;
  options.k = k;

  const auto start = std::chrono::steady_clock::now();
  const SpectralBasis basis = fmap::compute_spectral_basis(op, options);
  const auto elapsed = std::chrono::steady_clock::now() - start;
  const double seconds =
      std::chrono::duration<double>(elapsed).count();

  check(basis.count() == k, "returned k eigenpairs");
  check(basis.size() == mesh.num_vertices(), "eigenvectors span the mesh");

  bool ascending = true;
  for (Eigen::Index i = 1; i < basis.eigenvalues.size(); ++i) {
    if (basis.eigenvalues(i) < basis.eigenvalues(i - 1)) ascending = false;
  }
  check(ascending, "eigenvalues are ascending");
  check(basis.eigenvalues.minCoeff() >= 0.0, "eigenvalues are non-negative");

  // Closed surface: first mode constant, eigenvalue 0.
  const double scale = basis.eigenvalues(basis.count() - 1);
  check(basis.eigenvalues(0) < 1e-8 * scale, "lambda_0 is ~0");
  check(basis.eigenvalues(1) > 1e-6 * scale, "lambda_1 is clearly positive");

  const Eigen::VectorXd first = basis.eigenvectors.col(0);
  const double spread =
      (first.maxCoeff() - first.minCoeff()) / std::abs(first.mean());
  check(spread < 1e-6, "first eigenfunction is constant");

  // What makes projection a plain transpose.
  check(basis.orthonormality_error() < 1e-8, "basis is M-orthonormal");

  // Eigenproblem residual for a mid-spectrum mode.
  const Eigen::Index probe = basis.count() / 2;
  const Eigen::VectorXd phi = basis.eigenvectors.col(probe);
  const Eigen::VectorXd residual =
      op.L * phi - basis.eigenvalues(probe) * (op.mass.asDiagonal() * phi);
  const double relative =
      residual.norm() / std::max(1e-300, (op.L * phi).norm());
  check(relative < 1e-6, "L phi = lambda M phi holds for a mid mode");

  std::cout << "       lambda_1 = " << basis.eigenvalues(1)
            << ", lambda_" << (k - 1) << " = " << basis.eigenvalues(k - 1)
            << ", solved in " << seconds << "s\n";
}

void test_reconstruction() {
  section("projection and reconstruction");

  const Mesh mesh = fmap::load_mesh("scape:0");
  const LaplacianOperator op = fmap::build_laplacian(mesh);
  SpectralOptions options;
  options.k = 100;
  const SpectralBasis basis = fmap::compute_spectral_basis(op, options);

  // A constant lies in the first mode's span, so it must be exact.
  const Eigen::VectorXd constant =
      Eigen::VectorXd::Constant(mesh.num_vertices(), 3.0);
  const Eigen::VectorXd round_trip =
      basis.reconstruct(basis.project(constant));
  check((round_trip - constant).cwiseAbs().maxCoeff() < 1e-6,
        "constant function reconstructs exactly");

  // A coordinate is smooth: the low-pass behaviour the method relies on.
  const Eigen::VectorXd x = mesh.V.col(0);
  const Eigen::VectorXd x_hat = basis.reconstruct(basis.project(x));
  const double relative_error =
      (x_hat - x).norm() / x.norm();
  check(relative_error < 0.05, "a coordinate function reconstructs to <5%");
  std::cout << "       coordinate reconstruction error: "
            << relative_error * 100.0 << "%\n";

  // Must be mass-weighted: <phi_0,1>_M = area * phi_0.
  const Eigen::VectorXd coeffs = basis.project(constant);
  check(std::abs(coeffs(0)) > 1e-6, "constant has a non-trivial DC coefficient");
  check(coeffs.tail(basis.count() - 1).cwiseAbs().maxCoeff() < 1e-6,
        "constant has no energy in higher modes");
}

void test_validation() {
  section("input validation");

  const Mesh mesh = fmap::load_mesh("scape:0");
  const LaplacianOperator op = fmap::build_laplacian(mesh);

  auto rejects = [&](SpectralOptions o) {
    try {
      fmap::compute_spectral_basis(op, o);
      return false;
    } catch (const std::runtime_error&) {
      return true;
    }
  };

  SpectralOptions bad_k;
  bad_k.k = 0;
  check(rejects(bad_k), "k = 0 is rejected");

  SpectralOptions huge_k;
  huge_k.k = static_cast<int>(mesh.num_vertices());
  check(rejects(huge_k), "k >= n is rejected");

  // Non-negative shift makes (L - sigma*M) singular, since lambda_0 == 0.
  SpectralOptions bad_shift;
  bad_shift.shift_fraction = 0.0;
  check(rejects(bad_shift), "non-negative shift is rejected");
}

}  // namespace

int main() {
  // Areas differ by ~4 orders of magnitude and eigenvalues scale as 1/area, so
  // running both proves the shift is relative, not hardcoded.
  test_spectrum("scape:0", 100);
  test_spectrum("tosca:cat0", 100);
  test_reconstruction();
  test_validation();
  return summary();
}
