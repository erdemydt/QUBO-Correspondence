// HKS and WKS. The load-bearing test is pose invariance: two SCAPE meshes are
// one person in two poses sharing a vertex numbering, so vertex i is the same
// material point on both. Descriptors are only useful if they agree there and
// disagree elsewhere.

#include "fmap/descriptors.hpp"

#include <random>

#include "test_support.hpp"

using namespace fmap::test;
using fmap::DescriptorOptions;
using fmap::Mesh;
using fmap::SpectralBasis;

namespace {

SpectralBasis basis_for(const Mesh& mesh, int k = 100) {
  fmap::SpectralOptions options;
  options.k = k;
  return fmap::compute_spectral_basis(fmap::build_laplacian(mesh), options);
}

void test_shape(const char* name, const Eigen::MatrixXd& d,
                Eigen::Index vertices) {
  section(std::string("well-formedness: ") + name);

  check(d.rows() == vertices, "one row per vertex");
  check(d.cols() > 0, "produced at least one sample");
  check(d.allFinite(), "all values are finite");

  // A descriptor that is constant across the surface carries no information.
  Eigen::Index degenerate = 0;
  for (Eigen::Index c = 0; c < d.cols(); ++c) {
    const double spread = d.col(c).maxCoeff() - d.col(c).minCoeff();
    if (spread < 1e-12) ++degenerate;
  }
  check(degenerate == 0, "no constant (uninformative) columns");
}

void test_pose_invariance() {
  section("pose invariance across two SCAPE poses");

  const Mesh a = fmap::load_mesh("scape:0");
  const Mesh b = fmap::load_mesh("scape:1");
  check(a.num_vertices() == b.num_vertices(), "meshes are vertex-aligned");

  const Eigen::MatrixXd da = fmap::compute_all_descriptors(basis_for(a));
  const Eigen::MatrixXd db = fmap::compute_all_descriptors(basis_for(b));

  // Same vertex across meshes vs. an unrelated vertex. The former must be much
  // closer or the descriptors are not doing their job.
  std::mt19937 rng(12345);
  std::uniform_int_distribution<Eigen::Index> pick(0, a.num_vertices() - 1);

  double matched = 0.0;
  double random = 0.0;
  const int trials = 500;
  for (int i = 0; i < trials; ++i) {
    const Eigen::Index v = pick(rng);
    Eigen::Index other = pick(rng);
    while (other == v) other = pick(rng);

    matched += (da.row(v) - db.row(v)).norm();
    random += (da.row(v) - db.row(other)).norm();
  }
  matched /= trials;
  random /= trials;

  check(matched < random, "corresponding vertices are closer than random ones");
  check(matched * 3.0 < random, "and by a wide margin");
  std::cout << "       matched " << matched << " vs random " << random
            << "  (ratio " << random / matched << "x)\n";
}

void test_scale_invariance() {
  section("scale invariance");

  const Mesh original = fmap::load_mesh("scape:0");
  Mesh scaled = original;
  scaled.V *= 7.5;  // arbitrary non-unit factor

  check_near(scaled.surface_area(), original.surface_area() * 7.5 * 7.5, 1e-6,
             "scaling really did change the geometry");

  const Eigen::MatrixXd d0 = fmap::compute_all_descriptors(basis_for(original));
  const Eigen::MatrixXd d1 = fmap::compute_all_descriptors(basis_for(scaled));

  // Sampling ranges come from the eigenvalues, which scale inversely with area,
  // so the two should agree. Normalization is mass-weighted and mass scales
  // with area, hence the rescale.
  const double factor = 7.5;
  const Eigen::MatrixXd diff = (d0 - d1 * factor).cwiseAbs();
  const double denom = d0.cwiseAbs().maxCoeff();
  const double relative = diff.maxCoeff() / denom;
  const double mean_relative = diff.mean() / denom;

  // Diagnostic: how well do the two spectra agree after undoing the scaling?
  const SpectralBasis b0 = basis_for(original);
  const SpectralBasis b1 = basis_for(scaled);
  const Eigen::VectorXd rescaled = b1.eigenvalues * (factor * factor);
  double worst_eig = 0.0;
  Eigen::Index worst_at = 0;
  for (Eigen::Index i = 1; i < b0.eigenvalues.size(); ++i) {
    const double r = std::abs(rescaled(i) - b0.eigenvalues(i)) /
                     b0.eigenvalues(i);
    if (r > worst_eig) {
      worst_eig = r;
      worst_at = i;
    }
  }
  // Within a near-degenerate pair the eigenvectors are not uniquely determined
  // -- they rotate freely inside the shared eigenspace -- so phi_i(x)^2 is
  // unstable even where the eigenvalues are exact. That is the mechanism behind
  // the outlier deviations below, and behind the left/right symmetry flips
  // functional maps are prone to on human shapes.
  Eigen::Index near_degenerate = 0;
  for (Eigen::Index i = 1; i < b0.eigenvalues.size(); ++i) {
    const double gap = b0.eigenvalues(i) - b0.eigenvalues(i - 1);
    if (gap < 0.01 * b0.eigenvalues(i)) ++near_degenerate;
  }

  std::cout << "       eigenvalue agreement after rescaling: worst "
            << worst_eig * 100.0 << "% at mode " << worst_at << "\n";
  std::cout << "       descriptor deviation: max " << relative * 100.0
            << "%, mean " << mean_relative * 100.0 << "%\n";
  std::cout << "       near-degenerate eigenvalue pairs: " << near_degenerate
            << " of " << b0.count() << "\n";

  // Eigenvalues are uniquely determined and must match almost exactly: this is
  // the real test that the operator and shift placement are scale-invariant.
  check(worst_eig < 1e-9, "spectrum is scale-invariant to machine precision");

  // Asserted on the mean, not the max: the max is dominated by the outliers
  // from degenerate subspaces above, which is not what this test is about.
  check(mean_relative < 0.01, "descriptors are invariant to uniform scaling");
  check(relative < 0.15, "outliers stay bounded");
  check(near_degenerate > 0,
        "this mesh does have near-degenerate modes (explains the outliers)");
}

void test_options() {
  section("options");

  const SpectralBasis basis = basis_for(fmap::load_mesh("scape:0"));

  DescriptorOptions opts;
  opts.num_samples = 100;
  opts.step = 1;
  const Eigen::MatrixXd dense = fmap::compute_hks(basis, opts);
  check(dense.cols() == 100, "step=1 keeps every sample");

  opts.step = 5;
  const Eigen::MatrixXd sparse = fmap::compute_hks(basis, opts);
  check(sparse.cols() == 20, "step=5 keeps every fifth sample");
  check((dense.col(0) - sparse.col(0)).cwiseAbs().maxCoeff() < 1e-12,
        "subsampling picks the same samples, not different ones");

  const Eigen::MatrixXd both = fmap::compute_all_descriptors(basis, opts);
  check(both.cols() == sparse.cols() * 2, "combined descriptor concatenates");

  // Normalization should make every column unit norm under the mass metric.
  opts.normalize = true;
  const Eigen::MatrixXd normalized = fmap::compute_wks(basis, opts);
  double worst = 0.0;
  for (Eigen::Index c = 0; c < normalized.cols(); ++c) {
    const double norm = std::sqrt(
        normalized.col(c).cwiseProduct(basis.mass).dot(normalized.col(c)));
    worst = std::max(worst, std::abs(norm - 1.0));
  }
  check(worst < 1e-9, "normalized columns have unit mass-weighted norm");
}

}  // namespace

int main() {
  const Mesh mesh = fmap::load_mesh("scape:0");
  const SpectralBasis basis = basis_for(mesh);

  test_shape("HKS", fmap::compute_hks(basis), mesh.num_vertices());
  test_shape("WKS", fmap::compute_wks(basis), mesh.num_vertices());
  test_pose_invariance();
  test_scale_invariance();
  test_options();
  return summary();
}
