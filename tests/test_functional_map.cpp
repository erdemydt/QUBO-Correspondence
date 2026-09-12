// The functional map solve. The self-map pins everything down: mapping a mesh
// to itself must give C = I, so any error in projection, in the basis, or in
// the normal equations shows up immediately.

#include "fmap/functional_map.hpp"

#include "fmap/descriptors.hpp"

#include "test_support.hpp"

using namespace fmap::test;
using fmap::FunctionalMap;
using fmap::FunctionalMapOptions;
using fmap::Mesh;
using fmap::SpectralBasis;

namespace {

constexpr int kBasisSize = 60;

SpectralBasis basis_for(const Mesh& mesh) {
  fmap::SpectralOptions options;
  options.k = kBasisSize;
  return fmap::compute_spectral_basis(fmap::build_laplacian(mesh), options);
}

void test_self_map() {
  section("self map is the identity");

  const Mesh mesh = fmap::load_mesh("scape:0");
  const SpectralBasis basis = basis_for(mesh);
  const Eigen::MatrixXd desc = fmap::compute_all_descriptors(basis);

  const FunctionalMap map =
      fmap::solve_functional_map(basis, basis, desc, desc);

  check(map.C.rows() == kBasisSize && map.C.cols() == kBasisSize,
        "C is k x k");

  const Eigen::MatrixXd identity =
      Eigen::MatrixXd::Identity(kBasisSize, kBasisSize);
  const double deviation = (map.C - identity).cwiseAbs().maxCoeff();
  check(deviation < 0.05, "C is the identity");
  std::cout << "       max |C - I| = " << deviation
            << ", descriptor residual = " << map.descriptor_residual << '\n';

  check(map.descriptor_residual < 0.05, "descriptors are well fitted");
  check(map.orthogonality_error() < 0.1, "C is near-orthogonal");

  // The constant function must map to the constant function: both shapes have
  // the same total area here, so C(0,0) is 1.
  check_near(map.C(0, 0), 1.0, 0.01, "constant mode maps to itself");
}

void test_pose_pair() {
  section("two poses of the same subject");

  const Mesh a = fmap::load_mesh("scape:0");
  const Mesh b = fmap::load_mesh("scape:1");
  const SpectralBasis ba = basis_for(a);
  const SpectralBasis bb = basis_for(b);

  const FunctionalMap map = fmap::solve_functional_map(
      ba, bb, fmap::compute_all_descriptors(ba),
      fmap::compute_all_descriptors(bb));

  check(map.C.allFinite(), "C is finite");
  check(map.descriptor_residual < 0.5, "descriptors are reasonably fitted");

  // For near-isometric shapes C is banded -- "funnel shaped" -- not diagonal:
  // low frequencies map to low frequencies, concentrating energy near the
  // diagonal. Asserting strict diagonality would be wrong, because about a
  // third of this mesh's eigenvalues are near-degenerate pairs (see
  // test_descriptors) whose eigenfunctions are defined only up to a rotation of
  // their shared eigenspace, which C absorbs as an off-diagonal block. Band
  // energy is what actually characterizes a good map.
  const Eigen::Index probe = 20;
  const Eigen::MatrixXd block = map.C.topLeftCorner(probe, probe);
  double diagonal_energy = 0.0;
  double band_energy = 0.0;
  for (Eigen::Index i = 0; i < probe; ++i) {
    for (Eigen::Index j = 0; j < probe; ++j) {
      const double e = block(i, j) * block(i, j);
      if (i == j) diagonal_energy += e;
      if (std::abs(i - j) <= 3) band_energy += e;
    }
  }
  const double total = block.squaredNorm();
  check(band_energy / total > 0.95,
        "low-frequency block is banded around the diagonal");
  check(diagonal_energy / total > 0.4, "with substantial energy on it");

  std::cout << "       descriptor residual = " << map.descriptor_residual
            << ", diagonal " << diagonal_energy / total * 100.0
            << "%, within 3 of diagonal " << band_energy / total * 100.0
            << "%, orthogonality error = " << map.orthogonality_error() << '\n';
}

void test_regularization_effect() {
  section("commutativity regularizer");

  const Mesh a = fmap::load_mesh("scape:0");
  const Mesh b = fmap::load_mesh("scape:1");
  const SpectralBasis ba = basis_for(a);
  const SpectralBasis bb = basis_for(b);
  const Eigen::MatrixXd da = fmap::compute_all_descriptors(ba);
  const Eigen::MatrixXd db = fmap::compute_all_descriptors(bb);

  FunctionalMapOptions weak;
  weak.commutativity_weight = 0.0;
  FunctionalMapOptions strong;
  strong.commutativity_weight = 1.0;

  const FunctionalMap unregularized =
      fmap::solve_functional_map(ba, bb, da, db, weak);
  const FunctionalMap regularized =
      fmap::solve_functional_map(ba, bb, da, db, strong);

  // More regularization trades descriptor fit for commutativity, which is the
  // entire point of the term.
  check(regularized.commutativity_residual <
            unregularized.commutativity_residual,
        "regularization reduces the commutativity residual");
  check(regularized.descriptor_residual >= unregularized.descriptor_residual,
        "at the cost of descriptor fit");

  std::cout << "       lambda=0:   commutativity "
            << unregularized.commutativity_residual << ", descriptor "
            << unregularized.descriptor_residual << '\n'
            << "       lambda=1:   commutativity "
            << regularized.commutativity_residual << ", descriptor "
            << regularized.descriptor_residual << '\n';
}

void test_validation() {
  section("input validation");

  const Mesh mesh = fmap::load_mesh("scape:0");
  const SpectralBasis basis = basis_for(mesh);
  const Eigen::MatrixXd desc = fmap::compute_all_descriptors(basis);

  bool threw = false;
  try {
    // Mismatched descriptor counts between the two shapes.
    fmap::solve_functional_map(basis, basis, desc, desc.leftCols(2));
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "mismatched descriptor counts are rejected");

  threw = false;
  try {
    FunctionalMapOptions bad;
    bad.commutativity_weight = -1.0;
    fmap::solve_functional_map(basis, basis, desc, desc, bad);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "negative regularization weight is rejected");
}

}  // namespace

int main() {
  test_self_map();
  test_pose_pair();
  test_regularization_effect();
  test_validation();
  return summary();
}
