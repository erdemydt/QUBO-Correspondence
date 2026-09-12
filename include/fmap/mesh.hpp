// Triangle mesh container, loading, and debug output. Loading is by spec string
// ("scape:0") or path; callers never build a filename. See dataset.hpp.

#pragma once

#include <Eigen/Dense>

#include <filesystem>
#include <string>
#include <string_view>

#include "fmap/dataset.hpp"

namespace fmap {

// Row-major on purpose: every consumer here iterates over vertices or faces, so
// keeping each row contiguous is the friendlier layout. Eigen defaults to
// column-major, hence the explicit option.
using VertexMatrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using FaceMatrix = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>;
using ColorMatrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

struct Mesh {
  VertexMatrix V;    // n x 3 positions
  FaceMatrix F;      // m x 3 triangle vertex indices
  std::string name;  // spec or stem it came from, for logging and output

  Eigen::Index num_vertices() const { return V.rows(); }
  Eigen::Index num_faces() const { return F.rows(); }

  double surface_area() const;

  // Useful for scale-normalizing thresholds, so they mean the same thing on
  // TOSCA (centimetre-ish units) and SCAPE (metre-ish).
  double bounding_box_diagonal() const;
};

Mesh load_mesh(std::string_view spec);
Mesh load_mesh(const MeshRef& ref);

// Dispatches on extension (.off, .obj).
Mesh load_mesh_file(const std::filesystem::path& path);

void write_off(const std::filesystem::path& path, const Mesh& mesh);

// COFF: OFF with per-vertex colour, RGB in [0, 1], one row per vertex.
void write_colored_off(const std::filesystem::path& path, const Mesh& mesh,
                       const ColorMatrix& colors);

// Colour by normalized position within the bounding box. The standard way to
// eyeball a correspondence: colour the source, push the colours through the map
// onto the target, compare. A correct map looks anatomically matched; a
// left/right symmetry flip shows up immediately as mirrored colour.
ColorMatrix position_colors(const Mesh& mesh);

}  // namespace fmap
