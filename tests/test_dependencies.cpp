// Proves the third-party stack compiles and runs together before any pipeline
// code is built on it. The Spectra section is the point: SymGEigsShiftSolver is
// the most failure-prone part of the pipeline, so it is exercised here on a
// problem shaped like the real one -- a Neumann Laplacian, smallest eigenvalue
// exactly 0.

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/SymGEigsShiftSolver.h>
#include <nanoflann.hpp>

#include <vector>

#include "test_support.hpp"

using namespace fmap::test;

namespace {

// 1D Neumann Laplacian on a path graph: symmetric, PSD, rows sum to zero, so
// lambda_0 == 0 on a constant eigenvector -- the cotangent Laplacian's structure.
Eigen::SparseMatrix<double> path_laplacian(int n) {
  std::vector<Eigen::Triplet<double>> t;
  for (int i = 0; i + 1 < n; ++i) {
    t.emplace_back(i, i, 1.0);
    t.emplace_back(i + 1, i + 1, 1.0);
    t.emplace_back(i, i + 1, -1.0);
    t.emplace_back(i + 1, i, -1.0);
  }
  Eigen::SparseMatrix<double> l(n, n);
  l.setFromTriplets(t.begin(), t.end());
  return l;
}

void test_spectra_shift_invert() {
  section("Spectra: generalized shift-invert for smallest eigenpairs");

  const int n = 40;
  const int k = 3;
  Eigen::SparseMatrix<double> l = path_laplacian(n);

  // Lumped mass matrix stand-in: diagonal and positive, like the real one.
  Eigen::SparseMatrix<double> m(n, n);
  m.setIdentity();
  m *= 0.5;

  using OpType = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
  using BOpType = Spectra::SparseSymMatProd<double>;

  OpType op(l, m);
  BOpType bop(m);

  // Just below zero: the Laplacian is singular at exactly 0, so shifting there
  // makes the factorization rank-deficient.
  const double sigma = -1e-8;
  Spectra::SymGEigsShiftSolver<OpType, BOpType, Spectra::GEigsMode::ShiftInvert>
      eigs(op, bop, k, 2 * k + 1, sigma);

  eigs.init();
  const int converged = eigs.compute(Spectra::SortRule::LargestMagn);

  check(eigs.info() == Spectra::CompInfo::Successful, "solver converged");
  check(converged == k, "all requested eigenpairs converged");

  const Eigen::VectorXd vals = eigs.eigenvalues();
  // Spectra returns eigenvalues in descending order here; we want ascending.
  check_near(vals(k - 1), 0.0, 1e-8, "lambda_0 is zero");
  check(vals(k - 2) > vals(k - 1), "eigenvalues are strictly increasing");
  check(vals.minCoeff() > -1e-8, "no negative eigenvalues");
}

void test_nanoflann_knn() {
  section("nanoflann: KD-tree nearest neighbour on an Eigen matrix");

  // 4 points on a line in 2D; the nearest neighbour of (2.9, 0) is row 3.
  Eigen::MatrixXd points(4, 2);
  points << 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0;

  using KDTree = nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd>;
  KDTree tree(2, std::cref(points), 10);

  const double query[2] = {2.9, 0.0};
  Eigen::Index index = -1;
  double dist_sq = 0.0;
  tree.query(query, 1, &index, &dist_sq);

  check(index == 3, "found the expected nearest neighbour");
  check_near(dist_sq, 0.01, 1e-9, "squared distance is correct");
}

void test_tinyobjloader_parse() {
  section("tinyobjloader: parse a single triangle");

  const std::string obj =
      "v 0.0 0.0 0.0\n"
      "v 1.0 0.0 0.0\n"
      "v 0.0 1.0 0.0\n"
      "f 1 2 3\n";

  tinyobj::ObjReader reader;
  const bool ok = reader.ParseFromString(obj, /*mtl_text=*/"");

  check(ok, "parser accepted the input");
  check(reader.GetAttrib().vertices.size() == 9, "read 3 vertices (9 floats)");
  check(reader.GetShapes().size() == 1, "read one shape");
  if (!reader.GetShapes().empty()) {
    check(reader.GetShapes()[0].mesh.indices.size() == 3,
          "read one triangle (3 indices)");
  }
}

}  // namespace

int main() {
  test_spectra_shift_invert();
  test_nanoflann_knn();
  test_tinyobjloader_parse();
  return summary();
}
