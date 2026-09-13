// Mesh container, loading, and debug output. Load by spec ("scape:0") or path;
// callers never build a filename. See dataset.hpp.

#pragma once

#include <Eigen/Dense>

#include <filesystem>
#include <string>
#include <string_view>

#include "fmap/dataset.hpp"

namespace fmap {

// Row-major: every consumer iterates over vertices or faces, so contiguous rows
// are the friendlier layout. Eigen defaults to column-major.
using VertexMatrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using FaceMatrix = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>;
using ColorMatrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

struct Mesh {
  VertexMatrix V;    // n x 3
  FaceMatrix F;      // m x 3
  std::string name;  // for logging and output filenames

  Eigen::Index num_vertices() const { return V.rows(); }
  Eigen::Index num_faces() const { return F.rows(); }

  double surface_area() const;

  // For scale-normalizing thresholds across TOSCA and SCAPE units.
  double bounding_box_diagonal() const;
};

Mesh load_mesh(std::string_view spec);
Mesh load_mesh(const MeshRef& ref);
Mesh load_mesh_file(const std::filesystem::path& path);  // .off or .obj

void write_off(const std::filesystem::path& path, const Mesh& mesh);

// COFF. Colours are RGB in [0, 1], one row per vertex.
void write_colored_off(const std::filesystem::path& path, const Mesh& mesh,
                       const ColorMatrix& colors);

// Colour by position in the bounding box. Push these through a map onto the
// target and compare: a correct map looks anatomically matched, and the classic
// left/right symmetry flip shows up immediately as mirrored colour.
ColorMatrix position_colors(const Mesh& mesh);

}  // namespace fmap
