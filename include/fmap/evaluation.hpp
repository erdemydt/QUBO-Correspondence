// Scoring a recovered map. Vertex i is the same material point across SCAPE
// meshes and within a TOSCA class, so the truth is the identity permutation.
// Exact-match rate alone is too harsh -- one vertex off is nearly right -- so
// the metric is geodesic error normalized by sqrt(area).

#pragma once

#include <Eigen/Dense>

#include <vector>

#include "fmap/correspondence.hpp"
#include "fmap/mesh.hpp"

namespace fmap {

// Dijkstra on mesh edges. Forcing paths onto edges overestimates, but the bias
// is small and uniform at these tessellations, which is all a comparative
// metric needs. Fast marching would be the upgrade if absolutes mattered.
Eigen::VectorXd geodesic_distances_from(const Mesh& mesh, Eigen::Index source);

struct EvaluationOptions {
  int num_samples = 1000;  // one Dijkstra each, hence sampled; <= 0 means all
  unsigned seed = 42;
  double max_curve_threshold = 0.25;
  int curve_points = 26;
};

struct EvaluationResult {
  Eigen::Index samples = 0;
  double exact_match_rate = 0.0;

  // Normalized by sqrt(target surface area).
  double mean_geodesic_error = 0.0;
  double median_geodesic_error = 0.0;
  double max_geodesic_error = 0.0;

  // Princeton-style cumulative curve.
  std::vector<double> thresholds;
  std::vector<double> fractions;

  double within(double threshold) const;
};

// Checked, not assumed: TOSCA's horse0_partial differs from its class.
bool has_identity_ground_truth(const Mesh& source, const Mesh& target);

// Throws if the meshes are not vertex-aligned, rather than reporting
// meaningless numbers.
EvaluationResult evaluate_against_identity(const PointMap& map,
                                           const Mesh& source,
                                           const Mesh& target,
                                           const EvaluationOptions& options = {});

std::string format_error_curve(const EvaluationResult& result);

}  // namespace fmap
