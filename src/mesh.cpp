#include "fmap/mesh.hpp"

#include <tiny_obj_loader.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fmap {
namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("could not open mesh file: " + path.string());
  }
  std::ostringstream buf;
  buf << in.rdbuf();
  return buf.str();
}

// Whitespace-separated token scanner that treats '#' as a comment to
// end-of-line. OFF files in the wild vary in how they wrap lines -- counts
// sometimes share a line with the "OFF" header -- so tokenizing the whole file
// is more robust than reading it line by line.
class TokenScanner {
 public:
  explicit TokenScanner(const std::string& text) : text_(text) {}

  bool next(std::string_view& out) {
    skip_filler();
    if (pos_ >= text_.size()) return false;
    const std::size_t start = pos_;
    while (pos_ < text_.size() &&
           !std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
    out = std::string_view(text_).substr(start, pos_ - start);
    return true;
  }

  double next_double(const char* what) {
    std::string_view tok;
    if (!next(tok)) throw std::runtime_error(missing(what));
    // strtod needs NUL termination; the scratch string avoids allocating a
    // fresh std::string per token in the hot loop.
    scratch_.assign(tok);
    const char* begin = scratch_.c_str();
    char* end = nullptr;
    const double value = std::strtod(begin, &end);
    if (end == begin) throw std::runtime_error(bad(what, tok));
    return value;
  }

  long next_long(const char* what) {
    std::string_view tok;
    if (!next(tok)) throw std::runtime_error(missing(what));
    scratch_.assign(tok);
    const char* begin = scratch_.c_str();
    char* end = nullptr;
    const long value = std::strtol(begin, &end, 10);
    if (end == begin) throw std::runtime_error(bad(what, tok));
    return value;
  }

  // Advance past the remainder of the current line.
  void finish_line() {
    while (pos_ < text_.size() && text_[pos_] != '\n') ++pos_;
    if (pos_ < text_.size()) ++pos_;
  }

  // Next line that carries actual content, with any trailing comment removed.
  bool next_content_line(std::string_view& out) {
    while (pos_ < text_.size()) {
      const std::size_t start = pos_;
      finish_line();
      std::string_view line = std::string_view(text_).substr(
          start, (pos_ > start ? pos_ - start : 0));
      if (const std::size_t hash = line.find('#');
          hash != std::string_view::npos) {
        line = line.substr(0, hash);
      }
      // Content means at least one non-space character.
      if (line.find_first_not_of(" \t\r\n\f\v") != std::string_view::npos) {
        out = line;
        return true;
      }
    }
    return false;
  }

 private:
  void skip_filler() {
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if (std::isspace(static_cast<unsigned char>(c))) {
        ++pos_;
      } else if (c == '#') {
        while (pos_ < text_.size() && text_[pos_] != '\n') ++pos_;
      } else {
        return;
      }
    }
  }

  static std::string missing(const char* what) {
    return std::string("unexpected end of file while reading ") + what;
  }
  static std::string bad(const char* what, std::string_view tok) {
    return std::string("malformed ") + what + ": '" + std::string(tok) + "'";
  }

  const std::string& text_;
  std::size_t pos_ = 0;
  std::string scratch_;
};

Mesh load_off(const std::filesystem::path& path) {
  const std::string text = read_file(path);
  TokenScanner scan(text);

  std::string_view magic;
  if (!scan.next(magic)) {
    throw std::runtime_error("empty OFF file: " + path.string());
  }
  // Accept the OFF variants: COFF (per-vertex colour), NOFF (normals), STOFF
  // (texture coords), 4OFF (homogeneous coordinates). The vertex block is
  // parsed a line at a time and only the leading coordinates are taken, so any
  // combination of trailing per-vertex attributes is handled without the reader
  // having to know how many numbers each adds.
  //
  // That matters because the colour block is genuinely ambiguous: COFF permits
  // 1, 3, or 4 channels and the header does not say which.
  const bool homogeneous = magic.find('4') != std::string_view::npos;
  if (magic.find("OFF") == std::string_view::npos) {
    throw std::runtime_error("not an OFF file (bad magic '" +
                             std::string(magic) + "'): " + path.string());
  }

  const long num_vertices = scan.next_long("vertex count");
  const long num_faces = scan.next_long("face count");
  // The third header field is the edge count. It is unreliable -- TOSCA's
  // horse0_partial.off declares 0 -- so it is read and discarded rather than
  // used to size anything.
  (void)scan.next_long("edge count");

  if (num_vertices <= 0 || num_faces < 0) {
    throw std::runtime_error("OFF file declares no geometry: " + path.string());
  }

  Mesh mesh;
  mesh.V.resize(num_vertices, 3);
  const int coords = homogeneous ? 4 : 3;

  for (long i = 0; i < num_vertices; ++i) {
    std::string_view line;
    if (!scan.next_content_line(line)) {
      throw std::runtime_error("OFF file ended after " + std::to_string(i) +
                               " of " + std::to_string(num_vertices) +
                               " vertices: " + path.string());
    }

    // Take the leading coordinates and ignore whatever follows on the line.
    const std::string buf(line);
    const char* cursor = buf.c_str();
    double xyzw[4] = {0, 0, 0, 1};
    for (int k = 0; k < coords; ++k) {
      char* end = nullptr;
      xyzw[k] = std::strtod(cursor, &end);
      if (end == cursor) {
        throw std::runtime_error("malformed vertex line in " + path.string() +
                                 ": '" + buf + "'");
      }
      cursor = end;
    }

    // 4OFF stores homogeneous coordinates; divide through to get Cartesian.
    const double w = homogeneous ? xyzw[3] : 1.0;
    const double scale = (w != 0.0) ? 1.0 / w : 1.0;
    mesh.V(i, 0) = xyzw[0] * scale;
    mesh.V(i, 1) = xyzw[1] * scale;
    mesh.V(i, 2) = xyzw[2] * scale;
  }

  // Faces may be polygons; fan-triangulate anything above 3 sides so the
  // loader doesn't reject an otherwise usable mesh.
  std::vector<std::array<int, 3>> tris;
  tris.reserve(static_cast<std::size_t>(num_faces));
  for (long f = 0; f < num_faces; ++f) {
    const long sides = scan.next_long("face valence");
    if (sides < 3) {
      throw std::runtime_error("degenerate face with " + std::to_string(sides) +
                               " sides in " + path.string());
    }
    std::vector<int> poly(static_cast<std::size_t>(sides));
    for (long k = 0; k < sides; ++k) {
      const long idx = scan.next_long("face index");
      if (idx < 0 || idx >= num_vertices) {
        throw std::runtime_error("face index " + std::to_string(idx) +
                                 " out of range in " + path.string());
      }
      poly[static_cast<std::size_t>(k)] = static_cast<int>(idx);
    }
    for (std::size_t k = 1; k + 1 < poly.size(); ++k) {
      tris.push_back({poly[0], poly[k], poly[k + 1]});
    }
  }

  mesh.F.resize(static_cast<Eigen::Index>(tris.size()), 3);
  for (std::size_t i = 0; i < tris.size(); ++i) {
    mesh.F(static_cast<Eigen::Index>(i), 0) = tris[i][0];
    mesh.F(static_cast<Eigen::Index>(i), 1) = tris[i][1];
    mesh.F(static_cast<Eigen::Index>(i), 2) = tris[i][2];
  }
  return mesh;
}

Mesh load_obj(const std::filesystem::path& path) {
  tinyobj::ObjReader reader;
  tinyobj::ObjReaderConfig cfg;
  cfg.triangulate = true;
  cfg.vertex_color = false;

  if (!reader.ParseFromFile(path.string(), cfg)) {
    throw std::runtime_error("failed to parse OBJ " + path.string() + ": " +
                             reader.Error());
  }

  const tinyobj::attrib_t& attrib = reader.GetAttrib();
  const std::size_t num_vertices = attrib.vertices.size() / 3;
  if (num_vertices == 0) {
    throw std::runtime_error("OBJ has no vertices: " + path.string());
  }

  Mesh mesh;
  mesh.V.resize(static_cast<Eigen::Index>(num_vertices), 3);
  for (std::size_t i = 0; i < num_vertices; ++i) {
    mesh.V(static_cast<Eigen::Index>(i), 0) = attrib.vertices[3 * i + 0];
    mesh.V(static_cast<Eigen::Index>(i), 1) = attrib.vertices[3 * i + 1];
    mesh.V(static_cast<Eigen::Index>(i), 2) = attrib.vertices[3 * i + 2];
  }

  // tinyobjloader hands back per-shape index lists with separate position,
  // normal and texcoord indices; we keep only the position index, which is the
  // one that defines the mesh connectivity.
  std::vector<std::array<int, 3>> tris;
  for (const tinyobj::shape_t& shape : reader.GetShapes()) {
    const auto& indices = shape.mesh.indices;
    for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
      tris.push_back({indices[i + 0].vertex_index, indices[i + 1].vertex_index,
                      indices[i + 2].vertex_index});
    }
  }
  if (tris.empty()) {
    throw std::runtime_error("OBJ has no faces: " + path.string());
  }

  mesh.F.resize(static_cast<Eigen::Index>(tris.size()), 3);
  for (std::size_t i = 0; i < tris.size(); ++i) {
    for (int k = 0; k < 3; ++k) {
      const int idx = tris[i][static_cast<std::size_t>(k)];
      if (idx < 0 || idx >= static_cast<int>(num_vertices)) {
        throw std::runtime_error("OBJ face index out of range in " +
                                 path.string());
      }
      mesh.F(static_cast<Eigen::Index>(i), k) = idx;
    }
  }
  return mesh;
}

}  // namespace

double Mesh::surface_area() const {
  double total = 0.0;
  for (Eigen::Index f = 0; f < F.rows(); ++f) {
    const Eigen::Vector3d a = V.row(F(f, 0));
    const Eigen::Vector3d b = V.row(F(f, 1));
    const Eigen::Vector3d c = V.row(F(f, 2));
    total += 0.5 * (b - a).cross(c - a).norm();
  }
  return total;
}

double Mesh::bounding_box_diagonal() const {
  if (V.rows() == 0) return 0.0;
  return (V.colwise().maxCoeff() - V.colwise().minCoeff()).norm();
}

Mesh load_mesh_file(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("mesh file does not exist: " + path.string());
  }

  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  Mesh mesh;
  if (ext == ".off") {
    mesh = load_off(path);
  } else if (ext == ".obj") {
    mesh = load_obj(path);
  } else {
    throw std::runtime_error("unsupported mesh format '" + ext +
                             "' (expected .off or .obj): " + path.string());
  }

  mesh.name = path.stem().string();
  return mesh;
}

Mesh load_mesh(const MeshRef& ref) {
  Mesh mesh = load_mesh_file(ref.path);
  mesh.name = ref.spec();
  return mesh;
}

Mesh load_mesh(std::string_view spec) { return load_mesh(resolve_mesh(spec)); }

void write_off(const std::filesystem::path& path, const Mesh& mesh) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("could not write " + path.string());

  out << "OFF\n"
      << mesh.num_vertices() << ' ' << mesh.num_faces() << " 0\n";
  for (Eigen::Index i = 0; i < mesh.V.rows(); ++i) {
    out << mesh.V(i, 0) << ' ' << mesh.V(i, 1) << ' ' << mesh.V(i, 2) << '\n';
  }
  for (Eigen::Index f = 0; f < mesh.F.rows(); ++f) {
    out << "3 " << mesh.F(f, 0) << ' ' << mesh.F(f, 1) << ' ' << mesh.F(f, 2)
        << '\n';
  }
}

void write_colored_off(const std::filesystem::path& path, const Mesh& mesh,
                       const ColorMatrix& colors) {
  if (colors.rows() != mesh.num_vertices()) {
    throw std::runtime_error(
        "colour matrix has " + std::to_string(colors.rows()) +
        " rows but the mesh has " + std::to_string(mesh.num_vertices()) +
        " vertices");
  }

  std::ofstream out(path);
  if (!out) throw std::runtime_error("could not write " + path.string());

  // Integer 0-255 colour channels: the most widely accepted COFF flavour.
  out << "COFF\n"
      << mesh.num_vertices() << ' ' << mesh.num_faces() << " 0\n";
  for (Eigen::Index i = 0; i < mesh.V.rows(); ++i) {
    out << mesh.V(i, 0) << ' ' << mesh.V(i, 1) << ' ' << mesh.V(i, 2);
    for (int k = 0; k < 3; ++k) {
      const double c = std::clamp(colors(i, k), 0.0, 1.0);
      out << ' ' << static_cast<int>(c * 255.0 + 0.5);
    }
    out << " 255\n";
  }
  for (Eigen::Index f = 0; f < mesh.F.rows(); ++f) {
    out << "3 " << mesh.F(f, 0) << ' ' << mesh.F(f, 1) << ' ' << mesh.F(f, 2)
        << '\n';
  }
}

ColorMatrix position_colors(const Mesh& mesh) {
  ColorMatrix colors(mesh.num_vertices(), 3);
  if (mesh.num_vertices() == 0) return colors;

  const Eigen::RowVector3d lo = mesh.V.colwise().minCoeff();
  const Eigen::RowVector3d hi = mesh.V.colwise().maxCoeff();
  const Eigen::RowVector3d extent =
      (hi - lo).cwiseMax(Eigen::RowVector3d::Constant(1e-12));

  for (Eigen::Index i = 0; i < mesh.V.rows(); ++i) {
    colors.row(i) = (mesh.V.row(i) - lo).cwiseQuotient(extent);
  }
  return colors;
}

}  // namespace fmap
