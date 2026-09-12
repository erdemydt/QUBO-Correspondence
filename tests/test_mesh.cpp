// Mesh loading, writing, and the OFF variant handling.

#include "fmap/mesh.hpp"

#include <cstdio>
#include <filesystem>

#include "test_support.hpp"

using namespace fmap::test;
using fmap::Mesh;

namespace {

void test_load_scape() {
  section("SCAPE .off");

  const Mesh m = fmap::load_mesh("scape:0");
  check(m.num_vertices() == 12500, "12500 vertices");
  check(m.num_faces() == 24998, "24998 faces");
  check(m.name == "scape:mesh000", "name carries the spec");

  check(m.F.minCoeff() >= 0 && m.F.maxCoeff() < m.num_vertices(),
        "all face indices are in range");
  check(m.surface_area() > 0.0, "positive surface area");
  check(m.bounding_box_diagonal() > 0.0, "non-degenerate bounding box");

  // An unused vertex gives the Laplacian an all-zero row and the eigensolve
  // spurious null modes.
  Eigen::VectorXi used = Eigen::VectorXi::Zero(m.num_vertices());
  for (Eigen::Index f = 0; f < m.F.rows(); ++f) {
    for (int k = 0; k < 3; ++k) used(m.F(f, k)) = 1;
  }
  check(used.minCoeff() == 1, "no isolated vertices");
}

void test_load_tosca() {
  section("TOSCA .off");

  const Mesh cat = fmap::load_mesh("tosca:cat0");
  check(cat.num_vertices() == 27894, "cat0 has 27894 vertices");
  check(cat.num_faces() == 55712, "cat0 has 55712 faces");

  // Declares an edge count of 0 in its header. Loading it proves that field is
  // being ignored rather than trusted.
  const Mesh partial = fmap::load_mesh("tosca:horse0_partial");
  check(partial.num_vertices() == 15974,
        "horse0_partial loads despite a bogus edge count");

  const Mesh horse = fmap::load_mesh("tosca:horse0");
  check(horse.num_vertices() != partial.num_vertices(),
        "partial mesh is not vertex-aligned with its class");
}

void test_round_trip() {
  section("write / read round trip");

  const Mesh original = fmap::load_mesh("scape:0");
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "fmap_test";
  std::filesystem::create_directories(dir);

  const std::filesystem::path plain = dir / "plain.off";
  fmap::write_off(plain, original);
  const Mesh reloaded = fmap::load_mesh_file(plain);
  check(reloaded.num_vertices() == original.num_vertices(),
        "vertex count survives a round trip");
  check(reloaded.num_faces() == original.num_faces(),
        "face count survives a round trip");
  check((reloaded.V - original.V).cwiseAbs().maxCoeff() < 1e-4,
        "positions survive a round trip");
  check((reloaded.F - original.F).cwiseAbs().maxCoeff() == 0,
        "connectivity survives a round trip");

  // The variant-header path: skip the extra colour channels, still land on the
  // face block.
  const std::filesystem::path colored = dir / "colored.off";
  fmap::write_colored_off(colored, original, fmap::position_colors(original));
  const Mesh recolored = fmap::load_mesh_file(colored);
  check(recolored.num_vertices() == original.num_vertices(),
        "COFF vertex count parses");
  check(recolored.num_faces() == original.num_faces(),
        "COFF face count parses");
  check((recolored.F - original.F).cwiseAbs().maxCoeff() == 0,
        "COFF connectivity parses");

  std::filesystem::remove_all(dir);
}

void test_colors() {
  section("position colours");

  const Mesh m = fmap::load_mesh("scape:0");
  const fmap::ColorMatrix c = fmap::position_colors(m);

  check(c.rows() == m.num_vertices(), "one colour per vertex");
  check(c.minCoeff() >= 0.0 && c.maxCoeff() <= 1.0, "colours are in [0, 1]");
  check(c.maxCoeff() > 0.99, "colours span the full range");
}

void test_errors() {
  section("error handling");

  bool threw = false;
  try {
    fmap::load_mesh_file("/nonexistent/mesh.off");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "missing file is reported");

  threw = false;
  try {
    fmap::load_mesh_file("/etc/hostname");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "unsupported extension is reported");
}

}  // namespace

int main() {
  test_load_scape();
  test_load_tosca();
  test_round_trip();
  test_colors();
  test_errors();
  return summary();
}
