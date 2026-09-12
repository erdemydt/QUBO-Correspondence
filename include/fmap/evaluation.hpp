// Scoring a recovered correspondence. Both datasets make ground truth free:
// vertex i is the same material point across SCAPE meshes, and across meshes
// within a TOSCA class, so the true map is the identity permutation.
//
// Exact-match rate alone is too harsh -- landing one vertex away is nearly
// right -- so the main metric is geodesic error, normalized by sqrt(area) to
// stay comparable across meshes and datasets.

#pragma once

#include <Eigen/Dense>

#include <vector>

#include "fmap/correspondence.hpp"
#include "fmap/mesh.hpp"

namespace fmap {

// Dijkstra with Euclidean edge weights. Forcing paths onto edges overestimates
// true geodesic distance, but on meshes this well tessellated the bias is small
// and uniform, which is all a comparative metric needs. Fast marching or MMP
// would be the upgrade if absolute numbers ever mattered.
Eigen::VectorXd geodesic_distances_from(const Mesh& mesh, Eigen::Index source);

struct EvaluationOptions {
  // Each sample costs one Dijkstra run over the target, hence sampling rather
  // than exhaustive. <= 0 evaluates every vertex.
  int num_samples = 1000;

  unsigned seed = 42;

  double max_curve_threshold = 0.25;  // in normalized geodesic units
  int curve_points = 26;
};

struct EvaluationResult {
  Eigen::Index samples = 0;
  double exact_match_rate = 0.0;

  // Normalized by sqrt(target surface area).
  double mean_geodesic_error = 0.0;
  double median_geodesic_error = 0.0;
  double max_geodesic_error = 0.0;

  // Princeton-style cumulative curve: fractions[i] is the share of samples
  // with error at most thresholds[i].
  std::vector<double> thresholds;
  std::vector<double> fractions;

  double within(double threshold) const;
};

// Checked structurally rather than assumed -- TOSCA's horse0_partial has a
// different vertex count from the rest of its class.
bool has_identity_ground_truth(const Mesh& source, const Mesh& target);

// Throws std::runtime_error if the meshes are not vertex-aligned, rather than
// silently reporting meaningless numbers.
EvaluationResult evaluate_against_identity(const PointMap& map,
                                           const Mesh& source,
                                           const Mesh& target,
                                           const EvaluationOptions& options = {});

// The cumulative curve as a compact text plot for the console.
std::string format_error_curve(const EvaluationResult& result);

}  // namespace fmap
