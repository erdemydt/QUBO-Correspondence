#include "fmap/evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "fmap/utils.hpp"

namespace fmap {
namespace {

// CSR adjacency with Euclidean edge weights.
struct EdgeGraph {
  std::vector<Eigen::Index> offsets;    // size n+1, CSR-style
  std::vector<Eigen::Index> neighbors;
  std::vector<double> weights;
};

EdgeGraph build_edge_graph(const Mesh& mesh) {
  const Eigen::Index n = mesh.num_vertices();

  // Degrees first, so CSR fills without per-vertex vectors.
  std::vector<Eigen::Index> degree(static_cast<std::size_t>(n), 0);
  auto count_edge = [&](int a, int b) {
    ++degree[static_cast<std::size_t>(a)];
    ++degree[static_cast<std::size_t>(b)];
  };
  for (Eigen::Index f = 0; f < mesh.F.rows(); ++f) {
    count_edge(mesh.F(f, 0), mesh.F(f, 1));
    count_edge(mesh.F(f, 1), mesh.F(f, 2));
    count_edge(mesh.F(f, 2), mesh.F(f, 0));
  }

  EdgeGraph graph;
  graph.offsets.assign(static_cast<std::size_t>(n) + 1, 0);
  for (Eigen::Index i = 0; i < n; ++i) {
    graph.offsets[static_cast<std::size_t>(i) + 1] =
        graph.offsets[static_cast<std::size_t>(i)] +
        degree[static_cast<std::size_t>(i)];
  }
  graph.neighbors.resize(
      static_cast<std::size_t>(graph.offsets[static_cast<std::size_t>(n)]));
  graph.weights.resize(graph.neighbors.size());

  std::vector<Eigen::Index> cursor(graph.offsets.begin(), graph.offsets.end() - 1);
  auto add_edge = [&](int a, int b) {
    const double length = (mesh.V.row(a) - mesh.V.row(b)).norm();
    graph.neighbors[static_cast<std::size_t>(cursor[static_cast<std::size_t>(a)])] = b;
    graph.weights[static_cast<std::size_t>(cursor[static_cast<std::size_t>(a)])] = length;
    ++cursor[static_cast<std::size_t>(a)];
    graph.neighbors[static_cast<std::size_t>(cursor[static_cast<std::size_t>(b)])] = a;
    graph.weights[static_cast<std::size_t>(cursor[static_cast<std::size_t>(b)])] = length;
    ++cursor[static_cast<std::size_t>(b)];
  };
  for (Eigen::Index f = 0; f < mesh.F.rows(); ++f) {
    add_edge(mesh.F(f, 0), mesh.F(f, 1));
    add_edge(mesh.F(f, 1), mesh.F(f, 2));
    add_edge(mesh.F(f, 2), mesh.F(f, 0));
  }
  // Duplicate edges left in: Dijkstra is indifferent, dedup costs more.
  return graph;
}

Eigen::VectorXd dijkstra(const EdgeGraph& graph, Eigen::Index n,
                         Eigen::Index source) {
  Eigen::VectorXd distance =
      Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
  distance(source) = 0.0;

  using Entry = std::pair<double, Eigen::Index>;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
  queue.emplace(0.0, source);

  while (!queue.empty()) {
    const auto [d, u] = queue.top();
    queue.pop();
    if (d > distance(u)) continue;  // stale entry

    const std::size_t begin = static_cast<std::size_t>(graph.offsets[static_cast<std::size_t>(u)]);
    const std::size_t end = static_cast<std::size_t>(graph.offsets[static_cast<std::size_t>(u) + 1]);
    for (std::size_t e = begin; e < end; ++e) {
      const Eigen::Index v = graph.neighbors[e];
      const double next = d + graph.weights[e];
      if (next < distance(v)) {
        distance(v) = next;
        queue.emplace(next, v);
      }
    }
  }
  return distance;
}

}  // namespace

double EvaluationResult::within(double threshold) const {
  for (std::size_t i = 0; i < thresholds.size(); ++i) {
    if (thresholds[i] >= threshold) return fractions[i];
  }
  return fractions.empty() ? 0.0 : fractions.back();
}

Eigen::VectorXd geodesic_distances_from(const Mesh& mesh, Eigen::Index source) {
  if (source < 0 || source >= mesh.num_vertices()) {
    throw std::runtime_error("geodesic source vertex is out of range");
  }
  const EdgeGraph graph = build_edge_graph(mesh);
  return dijkstra(graph, mesh.num_vertices(), source);
}

bool has_identity_ground_truth(const Mesh& source, const Mesh& target) {
  // Vertex counts only. Requiring identical faces is tempting but wrong: SCAPE
  // poses share a registration yet ~24% of their faces disagree (quads split
  // along opposite diagonals), the same four vertices either way. Whether a
  // shared numbering IS a registration is dataset knowledge -- see
  // same_shape_class(). This is just the structural precondition, which already
  // rejects what matters: other classes and the partial horse differ in count.
  return source.num_vertices() == target.num_vertices() &&
         source.num_vertices() > 0;
}

EvaluationResult evaluate_against_identity(const PointMap& map,
                                           const Mesh& source,
                                           const Mesh& target,
                                           const EvaluationOptions& options) {
  if (!has_identity_ground_truth(source, target)) {
    throw std::runtime_error(
        "meshes are not vertex-aligned (" +
        std::to_string(source.num_vertices()) + "/" +
        std::to_string(source.num_faces()) + " vs " +
        std::to_string(target.num_vertices()) + "/" +
        std::to_string(target.num_faces()) +
        "), so the identity is not a valid ground-truth map");
  }
  if (map.size() != source.num_vertices()) {
    throw std::runtime_error("map size does not match the source mesh");
  }

  const Eigen::Index n = source.num_vertices();

  // Sampled: each scored vertex costs a full Dijkstra.
  std::vector<Eigen::Index> samples;
  if (options.num_samples <= 0 || options.num_samples >= n) {
    samples.resize(static_cast<std::size_t>(n));
    std::iota(samples.begin(), samples.end(), Eigen::Index{0});
  } else {
    std::vector<Eigen::Index> all(static_cast<std::size_t>(n));
    std::iota(all.begin(), all.end(), Eigen::Index{0});
    std::mt19937 rng(options.seed);
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(static_cast<std::size_t>(options.num_samples));
    samples = std::move(all);
  }

  // sqrt(area) makes SCAPE and TOSCA numbers directly comparable.
  const double normalizer = std::sqrt(target.surface_area());
  if (!(normalizer > 0.0)) {
    throw std::runtime_error("target mesh has zero area");
  }

  const EdgeGraph graph = build_edge_graph(target);
  std::vector<double> errors(samples.size(), 0.0);

  parallel_for(samples.size(), [&](std::size_t s) {
    const Eigen::Index truth = samples[s];
    const Eigen::Index got = map(truth);
    const Eigen::VectorXd distance = dijkstra(graph, target.num_vertices(), truth);
    const double d = distance(got);
    errors[s] = std::isfinite(d) ? d / normalizer : 1.0;
  }, /*min_per_thread=*/1);

  EvaluationResult result;
  result.samples = static_cast<Eigen::Index>(samples.size());

  Eigen::Index exact = 0;
  for (std::size_t s = 0; s < samples.size(); ++s) {
    if (map(samples[s]) == samples[s]) ++exact;
  }
  result.exact_match_rate =
      static_cast<double>(exact) / static_cast<double>(samples.size());

  result.mean_geodesic_error =
      std::accumulate(errors.begin(), errors.end(), 0.0) /
      static_cast<double>(errors.size());
  result.max_geodesic_error =
      *std::max_element(errors.begin(), errors.end());

  std::vector<double> sorted = errors;
  std::sort(sorted.begin(), sorted.end());
  result.median_geodesic_error = sorted[sorted.size() / 2];

  const int points = std::max(2, options.curve_points);
  result.thresholds.reserve(static_cast<std::size_t>(points));
  result.fractions.reserve(static_cast<std::size_t>(points));
  for (int i = 0; i < points; ++i) {
    const double t = options.max_curve_threshold * i / (points - 1);
    const std::size_t count = static_cast<std::size_t>(
        std::upper_bound(sorted.begin(), sorted.end(), t) - sorted.begin());
    result.thresholds.push_back(t);
    result.fractions.push_back(static_cast<double>(count) /
                               static_cast<double>(sorted.size()));
  }

  return result;
}

std::string format_error_curve(const EvaluationResult& result) {
  std::ostringstream out;
  out << "  geodesic error curve (fraction of points within a distance)\n";
  constexpr int kWidth = 40;
  for (std::size_t i = 0; i < result.thresholds.size(); i += 5) {
    const double fraction = result.fractions[i];
    const int filled = static_cast<int>(fraction * kWidth + 0.5);
    out << "    " << std::fixed << std::setprecision(3) << std::setw(5)
        << result.thresholds[i] << "  |";
    for (int c = 0; c < kWidth; ++c) out << (c < filled ? '#' : ' ');
    out << "|  " << std::setprecision(1) << std::setw(5) << fraction * 100.0
        << "%\n";
  }
  return out.str();
}

}  // namespace fmap
