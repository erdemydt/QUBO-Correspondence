// Mesh discovery by spec string ("scape:0", "tosca:cat3"), so no other file
// spells out a directory or filename. Directories are scanned, not generated
// from a formula: the indices have gaps (no SCAPE mesh051; TOSCA's horse jumps
// 7 -> 10 -> 15 -> 17), so a formula would invent paths that do not exist.

#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace fmap {

struct MeshRef {
  std::string dataset;         // "scape", "tosca"
  std::string group;           // shape class: "cat", "horse", ...
  std::string stem;            // "cat3"
  std::filesystem::path path;  // absolute, exists

  std::string spec() const { return dataset + ":" + stem; }
};

class DatasetRegistry {
 public:
  // Baked-in project root, overridden by $FMAP_DATA_ROOT.
  DatasetRegistry();

  explicit DatasetRegistry(std::filesystem::path root);

  // <id> matches a stem, or failing that a stem's numeric suffix, so "scape:7"
  // finds mesh007.off. Throws naming the alternatives.
  MeshRef resolve(std::string_view spec) const;

  bool contains(std::string_view spec) const;
  std::vector<std::string> datasets() const;
  std::vector<std::string> groups(std::string_view dataset) const;

  // Sorted naturally, so mesh10 follows mesh9.
  std::vector<MeshRef> meshes(std::string_view dataset,
                              std::string_view group = {}) const;

  const std::filesystem::path& root() const { return root_; }
  std::filesystem::path output_dir() const;  // created on first call

 private:
  std::filesystem::path root_;
  std::vector<MeshRef> meshes_;

  void scan();
};

// Prefer holding a registry when resolving several.
MeshRef resolve_mesh(std::string_view spec);

// Whether vertex i is the same material point on both. Dataset knowledge, not
// geometry: SCAPE poses share a registration while ~a quarter of their faces
// disagree (quads split along opposite diagonals), so connectivity is no proxy.
// With matching vertex counts, this licenses scoring against the identity.
bool same_shape_class(const MeshRef& a, const MeshRef& b);

}  // namespace fmap
