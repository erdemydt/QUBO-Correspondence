// Dataset discovery, so that no other source file spells out a directory or
// filename. Callers name a mesh with a spec string -- "scape:0", "tosca:cat3"
// -- and get back a resolved, existing path.
//
// Directories are scanned rather than having filenames built from a formula.
// That is a requirement, not a style choice: the indices have gaps (SCAPE has
// no mesh051, TOSCA's horse class jumps 7 -> 10 -> 15 -> 17), so any formula
// would invent paths that do not exist.

#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace fmap {

struct MeshRef {
  std::string dataset;         // "scape", "tosca"
  std::string group;           // shape class: "mesh", "cat", "horse", ...
  std::string stem;            // filename without extension: "cat3"
  std::filesystem::path path;  // absolute, verified to exist

  std::string spec() const { return dataset + ":" + stem; }
};

// Scans once on construction and answers lookups from the result.
class DatasetRegistry {
 public:
  // Baked-in project root, overridden by $FMAP_DATA_ROOT if set.
  DatasetRegistry();

  explicit DatasetRegistry(std::filesystem::path root);

  // "<dataset>:<id>", where <id> matches a filename stem ("tosca:cat3") or,
  // failing that and if it is a bare integer, a stem's numeric suffix -- so
  // "scape:7" finds mesh007.off. Throws naming the alternatives on failure.
  MeshRef resolve(std::string_view spec) const;

  bool contains(std::string_view spec) const;
  std::vector<std::string> datasets() const;
  std::vector<std::string> groups(std::string_view dataset) const;

  // Sorted by stem, numerically where the stems share a prefix.
  std::vector<MeshRef> meshes(std::string_view dataset,
                              std::string_view group = {}) const;

  const std::filesystem::path& root() const { return root_; }

  // Created on first call.
  std::filesystem::path output_dir() const;

 private:
  std::filesystem::path root_;
  std::vector<MeshRef> meshes_;

  void scan();
};

// Convenience for one-off lookups; prefer holding a registry for several.
MeshRef resolve_mesh(std::string_view spec);

// Whether the two are expected to share a vertex registration, vertex i being
// the same material point on both. This is dataset knowledge and cannot be
// recovered by comparing meshes: SCAPE poses share a registration while
// disagreeing on roughly a quarter of their triangulation (quads split along
// opposite diagonals), so connectivity is not a usable proxy. Together with
// matching vertex counts it licenses scoring against the identity.
bool same_shape_class(const MeshRef& a, const MeshRef& b);

}  // namespace fmap
