// Scoring a recovered correspondence against ground truth.
//
// Both datasets here make ground truth free: SCAPE meshes all share one vertex
// numbering, and so do the meshes within a TOSCA class. Vertex i on one mesh is
// the same material point as vertex i on the other, so the true map is the
// identity permutation and a recovered map can be scored exactly.
//
// Raw exact-match rate is too harsh on its own -- landing one vertex away is
// nearly right, not wrong -- so the main metric is geodesic error: how far
// across the target surface the recovered point is from the true one,
// normalized by sqrt(area) so it is comparable across meshes and datasets.

#pragma once

#include <Eigen/Dense>

#include <vector>

#include "fmap/correspondence.hpp"
#include "fmap/mesh.hpp"

namespace fmap {

// Geodesic distance from `source` to every vertex, approximated by shortest
// paths along mesh edges (Dijkstra with Euclidean edge weights).
//
// This overestimates true geodesic distance, since paths are forced onto edges
// rather than allowed to cross faces. On meshes this well tessellated the bias
// is small and uniform, which is all that is needed for a comparative metric.
// An exact method (fast marching, MMP) would be the upgrade if absolute numbers
// ever matter.
Eigen::VectorXd geodesic_distances_from(const Mesh& mesh, Eigen::Index source);

struct EvaluationOptions {
  // Vertices to evaluate. Each costs one Dijkstra run over the target mesh, so
  // this is sampled rather than exhaustive. <= 0 evaluates every vertex.
  int num_samples = 1000;

  unsigned seed = 42;

  // Upper end of the cumulative error curve, in normalized geodesic units.
  double max_curve_threshold = 0.25;
  int curve_points = 26;
};

struct EvaluationResult {
  Eigen::Index samples = 0;

  // Fraction of sampled vertices mapped exactly to their ground-truth partner.
  double exact_match_rate = 0.0;

  // Geodesic error normalized by sqrt(target surface area).
  double mean_geodesic_error = 0.0;
  double median_geodesic_error = 0.0;
  double max_geodesic_error = 0.0;

  // Princeton-benchmark style cumulative curve: fractions[i] is the share of
  // sampled vertices whose error is at most thresholds[i].
  std::vector<double> thresholds;
  std::vector<double> fractions;

  // Share of samples within 1% and 5% of the true point.
  double within(double threshold) const;
};

// True when the two meshes share a vertex numbering, so the identity is a valid
// ground-truth map. Checked structurally rather than assumed: TOSCA's
// horse0_partial has a different vertex count from the rest of its class.
bool has_identity_ground_truth(const Mesh& source, const Mesh& target);

// Score `map` against the identity ground truth on `target`.
//
// Throws std::runtime_error if the meshes are not vertex-aligned, rather than
// silently reporting meaningless numbers.
EvaluationResult evaluate_against_identity(const PointMap& map,
                                           const Mesh& source,
                                           const Mesh& target,
                                           const EvaluationOptions& options = {});

// Render the cumulative error curve as a compact text plot for the console.
std::string format_error_curve(const EvaluationResult& result);

}  // namespace fmap
