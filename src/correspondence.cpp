#include "fmap/correspondence.hpp"

#include <nanoflann.hpp>

#include <atomic>
#include <stdexcept>
#include <vector>

#include "fmap/utils.hpp"

namespace fmap {
namespace {

// Row-major: the tree build and per-query scans walk points row by row.
using EmbeddingMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

}  // namespace

std::string to_string(RecoveryMethod method) {
  switch (method) {
    case RecoveryMethod::NearestNeighbor:
      return "nearest-neighbor";
  }
  return "unknown";
}

RecoveryMethod recovery_method_from_string(const std::string& name) {
  if (name == "nearest-neighbor" || name == "nn") {
    return RecoveryMethod::NearestNeighbor;
  }
  throw std::runtime_error("unknown recovery method '" + name +
                           "'; valid methods: nearest-neighbor");
}

PointMap NearestNeighborStrategy::recover(const FunctionalMap& map,
                                          const SpectralBasis& source,
                                          const SpectralBasis& target) const {
  if (map.source_size() != source.count()) {
    throw std::runtime_error(
        "functional map expects a source basis of size " +
        std::to_string(map.source_size()) + " but was given " +
        std::to_string(source.count()));
  }
  if (map.target_size() != target.count()) {
    throw std::runtime_error(
        "functional map expects a target basis of size " +
        std::to_string(map.target_size()) + " but was given " +
        std::to_string(target.count()));
  }

  // Row i of Phi1 * C^T is C * Phi1(i,:)^T: where source i lands in target.
  const EmbeddingMatrix queries = source.eigenvectors * map.C.transpose();
  const EmbeddingMatrix points = target.eigenvectors;

  const int dimension = static_cast<int>(points.cols());
  const Eigen::Index num_targets = points.rows();
  const Eigen::Index num_sources = queries.rows();

  if (num_targets == 0 || num_sources == 0) {
    throw std::runtime_error("cannot recover a map between empty meshes");
  }

  using KDTree = nanoflann::KDTreeEigenMatrixAdaptor<EmbeddingMatrix>;
  // Leaf 16: at ~100 dims leaf scans dominate and vectorize better than the
  // library default.
  KDTree tree(dimension, std::cref(points), 16);

  PointMap result(num_sources);

  parallel_for(static_cast<std::size_t>(num_sources), [&](std::size_t i) {
    Eigen::Index index = 0;
    double distance_squared = 0.0;
    tree.query(queries.row(static_cast<Eigen::Index>(i)).data(), 1, &index,
               &distance_squared);
    result(static_cast<Eigen::Index>(i)) = static_cast<int>(index);
  });

  return result;
}

std::unique_ptr<CorrespondenceStrategy> make_strategy(RecoveryMethod method) {
  switch (method) {
    case RecoveryMethod::NearestNeighbor:
      return std::make_unique<NearestNeighborStrategy>();
  }
  throw std::runtime_error("unhandled recovery method");
}

MapStatistics analyze_map(const PointMap& map, Eigen::Index target_vertices) {
  MapStatistics stats;
  if (map.size() == 0 || target_vertices == 0) return stats;

  std::vector<int> hits(static_cast<std::size_t>(target_vertices), 0);
  for (Eigen::Index i = 0; i < map.size(); ++i) {
    const int t = map(i);
    if (t < 0 || t >= target_vertices) {
      throw std::runtime_error("recovered map contains an out-of-range index");
    }
    ++hits[static_cast<std::size_t>(t)];
  }

  for (const int count : hits) {
    if (count > 0) ++stats.distinct_targets;
    if (count > stats.max_collisions) stats.max_collisions = count;
  }
  stats.coverage = static_cast<double>(stats.distinct_targets) /
                   static_cast<double>(target_vertices);
  return stats;
}

}  // namespace fmap
