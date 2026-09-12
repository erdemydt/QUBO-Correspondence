#include "fmap/laplacian.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace fmap {

LaplacianOperator build_laplacian(const Mesh& mesh) {
  const Eigen::Index n = mesh.num_vertices();
  if (n == 0 || mesh.num_faces() == 0) {
    throw std::runtime_error("cannot build a Laplacian for an empty mesh");
  }

  LaplacianOperator op;
  op.mass = Eigen::VectorXd::Zero(n);

  std::vector<Eigen::Triplet<double>> stiffness;
  // Each triangle contributes to 3 edges, each edge to 4 matrix entries.
  stiffness.reserve(static_cast<std::size_t>(mesh.num_faces()) * 12);

  for (Eigen::Index f = 0; f < mesh.F.rows(); ++f) {
    const int idx[3] = {mesh.F(f, 0), mesh.F(f, 1), mesh.F(f, 2)};
    const Eigen::Vector3d p[3] = {mesh.V.row(idx[0]), mesh.V.row(idx[1]),
                                  mesh.V.row(idx[2])};

    const Eigen::Vector3d cross = (p[1] - p[0]).cross(p[2] - p[0]);
    const double double_area = cross.norm();

    // Degenerate triangle: no well-defined angles. Skip it rather than emit
    // infinities; an isolated sliver does not meaningfully change the operator.
    if (double_area < 1e-20) continue;

    const double area = 0.5 * double_area;

    // Barycentric lumping: each vertex takes a third of each incident triangle.
    // The entries therefore sum to the total surface area.
    for (int k = 0; k < 3; ++k) op.mass(idx[k]) += area / 3.0;

    bool obtuse = false;
    for (int k = 0; k < 3; ++k) {
      const int a = k;              // angle at this vertex
      const int b = (k + 1) % 3;    // the edge opposite it runs b -- c
      const int c = (k + 2) % 3;

      const Eigen::Vector3d u = p[b] - p[a];
      const Eigen::Vector3d v = p[c] - p[a];

      // cot(theta) = cos/sin = dot(u,v) / |u x v|, and |u x v| is the same
      // 2*area for all three angles of the triangle.
      const double dot = u.dot(v);
      if (dot < 0.0) obtuse = true;
      const double cotangent = dot / double_area;

      // Standard cotangent weight: each interior edge picks up cot of the
      // angle opposite it in each of its two adjacent triangles, halved.
      const double w = 0.5 * cotangent;
      stiffness.emplace_back(idx[b], idx[b], w);
      stiffness.emplace_back(idx[c], idx[c], w);
      stiffness.emplace_back(idx[b], idx[c], -w);
      stiffness.emplace_back(idx[c], idx[b], -w);
    }
    if (obtuse) ++op.obtuse_triangles;
  }

  op.L.resize(n, n);
  op.L.setFromTriplets(stiffness.begin(), stiffness.end());
  op.L.makeCompressed();

  // A vertex with zero area would make M singular and the generalized
  // eigenproblem ill-posed. This should not happen on a mesh where every
  // vertex is used by a face, so treat it as a real error rather than clamping.
  if (op.mass.minCoeff() <= 0.0) {
    throw std::runtime_error(
        "mesh has a vertex with zero lumped area; the mass matrix is singular");
  }

  op.M.resize(n, n);
  std::vector<Eigen::Triplet<double>> mass_triplets;
  mass_triplets.reserve(static_cast<std::size_t>(n));
  for (Eigen::Index i = 0; i < n; ++i) {
    mass_triplets.emplace_back(i, i, op.mass(i));
  }
  op.M.setFromTriplets(mass_triplets.begin(), mass_triplets.end());
  op.M.makeCompressed();

  return op;
}

}  // namespace fmap
