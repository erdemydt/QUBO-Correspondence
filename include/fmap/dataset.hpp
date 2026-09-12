// Dataset discovery and mesh reference resolution.
//
// The point of this file is that no other source file ever spells out a
// directory or filename. Callers name a mesh with a short spec string --
// "scape:0", "tosca:cat3" -- and get back a resolved, existing path.
//
// Datasets are discovered by scanning their directory, not by constructing
// filenames from a formula. That is a requirement, not a style choice: the
// indices in these datasets have gaps (SCAPE has no mesh051; TOSCA's horse
// class jumps 7 -> 10 -> 15 -> 17), so any formula would invent paths that do
// not exist.

#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace fmap {

// A resolved reference to one mesh file.
struct MeshRef {
  std::string dataset;            // "scape", "tosca"
  std::string group;              // shape class: "mesh", "cat", "horse", ...
  std::string stem;               // filename without extension: "cat3"
  std::filesystem::path path;     // absolute, verified to exist

  // The spec string that resolves back to this mesh, e.g. "tosca:cat3".
  std::string spec() const { return dataset + ":" + stem; }
};

// Scans the known dataset directories once on construction and answers
// lookups from the result.
class DatasetRegistry {
 public:
  // Uses the baked-in project root, overridden by $FMAP_DATA_ROOT if set.
  DatasetRegistry();

  // Scans under an explicit root. Mainly for tests.
  explicit DatasetRegistry(std::filesystem::path root);

  // Resolve a spec of the form "<dataset>:<id>".
  //
  // <id> is matched against the filename stem first ("tosca:cat3"). If that
  // fails and <id> is a bare integer, it is also tried against the numeric
  // suffix of each stem, so "scape:7" finds mesh007.off.
  //
  // Throws std::runtime_error naming the available alternatives on failure.
  MeshRef resolve(std::string_view spec) const;

  // True if `spec` resolves, without throwing.
  bool contains(std::string_view spec) const;

  // Names of the datasets that were found on disk.
  std::vector<std::string> datasets() const;

  // Shape classes within a dataset, e.g. {"cat", "centaur", ...} for TOSCA.
  std::vector<std::string> groups(std::string_view dataset) const;

  // All meshes in a dataset, optionally restricted to one group. Sorted by
  // stem, numerically where the stems share a prefix.
  std::vector<MeshRef> meshes(std::string_view dataset,
                              std::string_view group = {}) const;

  // Root that was scanned.
  const std::filesystem::path& root() const { return root_; }

  // Directory for generated output; created on first call.
  std::filesystem::path output_dir() const;

 private:
  std::filesystem::path root_;
  std::vector<MeshRef> meshes_;

  void scan();
};

// Resolve a spec against a default-constructed registry. Convenience for
// one-off lookups; prefer holding a registry when resolving several meshes.
MeshRef resolve_mesh(std::string_view spec);

// True when two meshes come from the same dataset and shape class, and so are
// expected to share a vertex registration -- vertex i being the same material
// point on both.
//
// This is dataset knowledge, not geometry: it cannot be recovered by comparing
// the meshes. SCAPE poses share a registration while disagreeing on roughly a
// quarter of their triangulation (quads split along opposite diagonals), so
// connectivity is not a usable proxy for it.
//
// Combined with matching vertex counts, this is what licenses scoring a
// recovered map against the identity. See evaluation.hpp.
bool same_shape_class(const MeshRef& a, const MeshRef& b);

}  // namespace fmap
