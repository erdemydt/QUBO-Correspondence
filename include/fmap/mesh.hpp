// Triangle mesh container, loading, and debug output.
//
// Loading is by spec string ("scape:0") or path; callers never build a filename
// themselves. See dataset.hpp for how specs resolve.

#pragma once

#include <Eigen/Dense>

#include <filesystem>
#include <string>
#include <string_view>

#include "fmap/dataset.hpp"

namespace fmap {

// A triangle mesh.
//
// Row-major storage on purpose: every consumer here iterates over vertices or
// faces, so keeping each vertex's coordinates contiguous is the friendlier
// layout. Eigen defaults to column-major, hence the explicit option.
using VertexMatrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using FaceMatrix = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>;
using ColorMatrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

struct Mesh {
  VertexMatrix V;   // n x 3 positions
  FaceMatrix F;     // m x 3 triangle vertex indices
  std::string name; // spec or stem it was loaded from, for logging and output

  Eigen::Index num_vertices() const { return V.rows(); }
  Eigen::Index num_faces() const { return F.rows(); }

  // Sum of triangle areas.
  double surface_area() const;

  // Axis-aligned bounding box diagonal length. Useful for scale-normalizing
  // thresholds so they mean the same thing on TOSCA (centimetre-ish units) and
  // SCAPE (metre-ish units).
  double bounding_box_diagonal() const;
};

// Load by dataset spec, e.g. "scape:0" or "tosca:cat3".
Mesh load_mesh(std::string_view spec);

// Load a resolved reference.
Mesh load_mesh(const MeshRef& ref);

// Load a path directly, dispatching on extension (.off, .obj).
Mesh load_mesh_file(const std::filesystem::path& path);

// Write plain OFF.
void write_off(const std::filesystem::path& path, const Mesh& mesh);

// Write COFF: OFF with per-vertex colour. Colours are RGB in [0, 1] and must
// have one row per vertex.
void write_colored_off(const std::filesystem::path& path, const Mesh& mesh,
                       const ColorMatrix& colors);

// Colour each vertex by its normalized position within the mesh's bounding box.
//
// This is the standard way to eyeball a correspondence: colour the source by
// position, push those colours through the recovered map onto the target, and
// compare. A correct map looks smooth and anatomically matched; a bad one looks
// like noise, and a left/right symmetry flip -- the classic functional-maps
// failure -- is immediately visible as mirrored colour.
ColorMatrix position_colors(const Mesh& mesh);

}  // namespace fmap
