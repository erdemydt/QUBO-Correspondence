// Point-to-point recovery. The self-map case is exact: mapping a mesh to itself
// must recover the identity on every vertex. That pins the embedding
// convention -- notably which side C is applied on, easy to transpose and hard
// to notice otherwise.

#include "fmap/correspondence.hpp"

#include "fmap/descriptors.hpp"
#include "fmap/utils.hpp"

#include "test_support.hpp"

using namespace fmap::test;
using fmap::Mesh;
using fmap::PointMap;
using fmap::SpectralBasis;

namespace {

constexpr int kBasisSize = 60;

SpectralBasis basis_for(const Mesh& mesh) {
  fmap::SpectralOptions options;
  options.k = kBasisSize;
  return fmap::compute_spectral_basis(fmap::build_laplacian(mesh), options);
}

void test_self_map_is_identity() {
  section("self map recovers the identity permutation");

  const Mesh mesh = fmap::load_mesh("scape:0");
  const SpectralBasis basis = basis_for(mesh);
  const Eigen::MatrixXd desc = fmap::compute_all_descriptors(basis);
  const auto map = fmap::solve_functional_map(basis, basis, desc, desc);

  const auto strategy =
      fmap::make_strategy(fmap::RecoveryMethod::NearestNeighbor);

  fmap::Timer timer;
  const PointMap recovered = strategy->recover(map, basis, basis);
  const double seconds = timer.seconds();

  check(recovered.size() == mesh.num_vertices(), "one entry per source vertex");

  Eigen::Index exact = 0;
  for (Eigen::Index i = 0; i < recovered.size(); ++i) {
    if (recovered(i) == i) ++exact;
  }
  const double rate = static_cast<double>(exact) /
                      static_cast<double>(recovered.size());
  check(rate > 0.999, "every vertex maps to itself");
  std::cout << "       " << exact << " / " << recovered.size()
            << " exact (" << rate * 100.0 << "%), " << seconds << "s\n";

  // A perfect self map is a bijection, so coverage must be complete.
  const auto stats = fmap::analyze_map(recovered, mesh.num_vertices());
  check(stats.coverage > 0.999, "self map is a bijection");
}

void test_pose_pair() {
  section("recovery between two poses");

  const Mesh a = fmap::load_mesh("scape:0");
  const Mesh b = fmap::load_mesh("scape:1");
  const SpectralBasis ba = basis_for(a);
  const SpectralBasis bb = basis_for(b);

  const auto map = fmap::solve_functional_map(
      ba, bb, fmap::compute_all_descriptors(ba),
      fmap::compute_all_descriptors(bb));

  const auto strategy =
      fmap::make_strategy(fmap::RecoveryMethod::NearestNeighbor);

  fmap::Timer timer;
  const PointMap recovered = strategy->recover(map, ba, bb);
  const double seconds = timer.seconds();

  check(recovered.minCoeff() >= 0 && recovered.maxCoeff() < b.num_vertices(),
        "all indices are in range");

  const auto stats = fmap::analyze_map(recovered, b.num_vertices());

  // NN recovery is not injective -- each source picks its target independently,
  // so targets get reused. This is the baseline a combinatorial method beats.
  check(stats.coverage < 1.0, "recovery is not a bijection");
  check(stats.coverage > 0.3, "but it does spread across the target");

  std::cout << "       coverage " << stats.coverage * 100.0 << "%, worst "
            << stats.max_collisions << " sources on one target, " << seconds
            << "s\n";
}

void test_strategy_registry() {
  section("strategy selection");

  const auto strategy =
      fmap::make_strategy(fmap::RecoveryMethod::NearestNeighbor);
  check(strategy != nullptr, "factory returns a strategy");
  check(strategy->name() == "nearest-neighbor", "strategy reports its name");

  check(fmap::recovery_method_from_string("nn") ==
            fmap::RecoveryMethod::NearestNeighbor,
        "short alias parses");
  check(fmap::to_string(fmap::RecoveryMethod::NearestNeighbor) ==
            "nearest-neighbor",
        "name round-trips");

  bool threw = false;
  try {
    fmap::recovery_method_from_string("qubo");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "unimplemented method is rejected by name");
}

void test_parallel_for() {
  section("parallel_for utility");

  constexpr std::size_t n = 10000;
  std::vector<int> values(n, 0);
  fmap::parallel_for(n, [&](std::size_t i) {
    values[i] = static_cast<int>(i) * 2;
  });

  bool all_correct = true;
  for (std::size_t i = 0; i < n; ++i) {
    if (values[i] != static_cast<int>(i) * 2) all_correct = false;
  }
  check(all_correct, "every index is visited exactly once");

  std::vector<int> empty;
  fmap::parallel_for(0, [&](std::size_t) { empty.push_back(1); });
  check(empty.empty(), "zero count does nothing");
}

}  // namespace

int main() {
  test_self_map_is_identity();
  test_pose_pair();
  test_strategy_registry();
  test_parallel_for();
  return summary();
}
