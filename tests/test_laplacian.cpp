// The invariants the spectral basis depends on. Break one and the
// eigenfunctions go wrong in ways that are hard to diagnose later.

#include "fmap/laplacian.hpp"

#include <Eigen/Dense>

#include "test_support.hpp"

using namespace fmap::test;
using fmap::LaplacianOperator;
using fmap::Mesh;

namespace {

// Small enough to check by hand, so the place to pin the sign convention.
Mesh unit_square() {
  Mesh m;
  m.V.resize(4, 3);
  m.V << 0, 0, 0,
         1, 0, 0,
         1, 1, 0,
         0, 1, 0;
  m.F.resize(2, 3);
  m.F << 0, 1, 2,
         0, 2, 3;
  m.name = "unit_square";
  return m;
}

void test_analytic() {
  section("unit square (hand-checkable)");

  const Mesh m = unit_square();
  const LaplacianOperator op = fmap::build_laplacian(m);

  check_near(op.mass.sum(), 1.0, 1e-12, "total mass equals the area");
  check_near(op.mass(0), 1.0 / 3.0, 1e-12,
             "corner shared by both triangles gets 2/6");
  check_near(op.mass(1), 1.0 / 6.0, 1e-12,
             "corner in one triangle gets 1/6");

  // Right isoceles: the right angle gives cot = 0, the 45s cot = 1. Edge 0-2
  // is opposite the right angle in both triangles, so its weight is 0.
  check_near(op.L.coeff(0, 2), 0.0, 1e-12,
             "diagonal edge has zero cotangent weight");
  check_near(op.L.coeff(0, 1), -0.5, 1e-12, "boundary edge weight is -1/2");

  // PSD convention, so the diagonal is positive.
  check(op.L.coeff(0, 0) > 0.0, "diagonal is positive (PSD convention)");
}

void test_invariants(const char* spec) {
  section(std::string("invariants on ") + spec);

  const Mesh m = fmap::load_mesh(spec);
  const LaplacianOperator op = fmap::build_laplacian(m);

  check(op.size() == m.num_vertices(), "operator matches the vertex count");

  const Eigen::SparseMatrix<double> asym =
      Eigen::SparseMatrix<double>(op.L.transpose()) - op.L;
  check(asym.norm() < 1e-10, "L is symmetric");

  // L annihilates constants: almost every assembly bug breaks this.
  const Eigen::VectorXd ones = Eigen::VectorXd::Ones(op.size());
  const Eigen::VectorXd row_sums = op.L * ones;
  check(row_sums.cwiseAbs().maxCoeff() < 1e-9,
        "L annihilates the constant function");

  // Positive, and accounting for exactly the surface area.
  check(op.mass.minCoeff() > 0.0, "all lumped masses are positive");
  check_near(op.mass.sum(), m.surface_area(), 1e-8,
             "lumped mass sums to the surface area");

  // Sampled PSD: the Rayleigh quotient must be non-negative anywhere.
  double min_quotient = 0.0;
  for (int trial = 0; trial < 20; ++trial) {
    Eigen::VectorXd x = Eigen::VectorXd::Random(op.size());
    x.array() -= x.mean();  // orthogonal to the null space
    const double q = x.dot(op.L * x) / x.squaredNorm();
    min_quotient = std::min(min_quotient, q);
  }
  check(min_quotient >= -1e-12, "Rayleigh quotients are non-negative");

  std::cout << "       " << op.obtuse_triangles << " / " << m.num_faces()
            << " triangles obtuse\n";
}

void test_alignment_consistency() {
  section("vertex-aligned meshes give same-sized operators");

  // Shared connectivity means structurally identical operators.
  const LaplacianOperator a = fmap::build_laplacian(fmap::load_mesh("scape:0"));
  const LaplacianOperator b = fmap::build_laplacian(fmap::load_mesh("scape:1"));

  check(a.size() == b.size(), "same dimension");
  check(a.L.nonZeros() == b.L.nonZeros(), "same sparsity pattern size");
  check((a.mass - b.mass).cwiseAbs().maxCoeff() > 0.0,
        "but different geometry, so different masses");
}

}  // namespace

int main() {
  test_analytic();
  test_invariants("scape:0");
  test_invariants("tosca:cat0");
  test_alignment_consistency();
  return summary();
}
