// Calibrated against two known answers: a perfect map scores zero, a random one
// near chance. A metric that cannot separate those measures nothing.

#include "fmap/evaluation.hpp"

#include <random>

#include "test_support.hpp"

using namespace fmap::test;
using fmap::EvaluationOptions;
using fmap::Mesh;
using fmap::PointMap;

namespace {

EvaluationOptions fast_options() {
  EvaluationOptions options;
  options.num_samples = 200;  // each sample is a Dijkstra run
  return options;
}

void test_geodesic_distances() {
  section("geodesic distances");

  const Mesh mesh = fmap::load_mesh("scape:0");
  const Eigen::VectorXd d = fmap::geodesic_distances_from(mesh, 0);

  check(d.size() == mesh.num_vertices(), "one distance per vertex");
  check(d(0) == 0.0, "distance to self is zero");
  check(d.minCoeff() >= 0.0, "distances are non-negative");
  check(d.allFinite(), "mesh is connected, so all are finite");

  // An edge path can never beat the straight line.
  bool dominates = true;
  for (Eigen::Index i = 0; i < mesh.num_vertices(); i += 97) {
    const double euclidean = (mesh.V.row(i) - mesh.V.row(0)).norm();
    if (d(i) < euclidean - 1e-9) dominates = false;
  }
  check(dominates, "geodesic distance is at least Euclidean distance");
}

void test_ground_truth_detection() {
  section("ground truth availability");

  const Mesh a = fmap::load_mesh("scape:0");
  const Mesh b = fmap::load_mesh("scape:1");
  check(fmap::has_identity_ground_truth(a, b), "SCAPE poses are aligned");

  const Mesh cat0 = fmap::load_mesh("tosca:cat0");
  const Mesh cat1 = fmap::load_mesh("tosca:cat1");
  check(fmap::has_identity_ground_truth(cat0, cat1),
        "TOSCA meshes in a class are aligned");

  // Different classes and the partial mesh must be rejected.
  const Mesh horse = fmap::load_mesh("tosca:horse0");
  check(!fmap::has_identity_ground_truth(cat0, horse),
        "different shape classes are not aligned");

  const Mesh partial = fmap::load_mesh("tosca:horse0_partial");
  check(!fmap::has_identity_ground_truth(horse, partial),
        "the partial horse is not aligned with its class");

  // Shared registration, different triangulation: ground truth must not use it.
  Eigen::Index differing = 0;
  for (Eigen::Index f = 0; f < a.F.rows(); ++f) {
    if (a.F.row(f) != b.F.row(f)) ++differing;
  }
  check(differing > 0, "SCAPE poses really do differ in triangulation");
  std::cout << "       " << differing << " / " << a.F.rows()
            << " faces differ between two aligned SCAPE poses\n";

  // The dataset-level signal, which isolates the partial mesh.
  check(fmap::same_shape_class(fmap::resolve_mesh("scape:0"),
                               fmap::resolve_mesh("scape:1")),
        "SCAPE poses are in the same class");
  check(!fmap::same_shape_class(fmap::resolve_mesh("tosca:horse0"),
                                fmap::resolve_mesh("tosca:horse0_partial")),
        "the partial horse is in a different class");
  check(!fmap::same_shape_class(fmap::resolve_mesh("tosca:cat0"),
                                fmap::resolve_mesh("tosca:dog0")),
        "cat and dog are different classes");
}

void test_perfect_map() {
  section("a perfect map scores zero");

  const Mesh a = fmap::load_mesh("scape:0");
  const Mesh b = fmap::load_mesh("scape:1");

  PointMap identity(a.num_vertices());
  for (Eigen::Index i = 0; i < identity.size(); ++i) identity(i) = static_cast<int>(i);

  const auto result =
      fmap::evaluate_against_identity(identity, a, b, fast_options());

  check_near(result.exact_match_rate, 1.0, 1e-12, "every match is exact");
  check_near(result.mean_geodesic_error, 0.0, 1e-12, "zero geodesic error");
  check_near(result.within(0.01), 1.0, 1e-12, "all points within 1%");
}

void test_random_map() {
  section("a random map scores near chance");

  const Mesh a = fmap::load_mesh("scape:0");
  const Mesh b = fmap::load_mesh("scape:1");

  std::mt19937 rng(7);
  std::uniform_int_distribution<int> pick(0, static_cast<int>(b.num_vertices()) - 1);
  PointMap random(a.num_vertices());
  for (Eigen::Index i = 0; i < random.size(); ++i) random(i) = pick(rng);

  const auto result =
      fmap::evaluate_against_identity(random, a, b, fast_options());

  check(result.exact_match_rate < 0.01, "essentially no exact matches");
  check(result.mean_geodesic_error > 0.1, "large geodesic error");
  check(result.within(0.01) < 0.05, "almost nothing lands close");

  std::cout << "       random map: mean error " << result.mean_geodesic_error
            << ", within 1%: " << result.within(0.01) * 100.0 << "%\n";
}

void test_rejects_unaligned() {
  section("unaligned meshes are rejected");

  const Mesh cat = fmap::load_mesh("tosca:cat0");
  const Mesh horse = fmap::load_mesh("tosca:horse0");
  PointMap map = PointMap::Zero(cat.num_vertices());

  bool threw = false;
  try {
    fmap::evaluate_against_identity(map, cat, horse, fast_options());
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "scoring without ground truth is an error, not a silent zero");
}

}  // namespace

int main() {
  test_geodesic_distances();
  test_ground_truth_detection();
  test_perfect_map();
  test_random_map();
  test_rejects_unaligned();
  return summary();
}
