// The single entry point. Thin on purpose: parse arguments, then run the blocks
// in include/fmap/ in order. Algorithmic work belongs in the blocks.
//
//     mesh -> Laplacian -> spectral basis -> descriptors -> C -> point map

#include "fmap/config.hpp"
#include "fmap/correspondence.hpp"
#include "fmap/dataset.hpp"
#include "fmap/descriptors.hpp"
#include "fmap/evaluation.hpp"
#include "fmap/functional_map.hpp"
#include "fmap/laplacian.hpp"
#include "fmap/mesh.hpp"
#include "fmap/spectral_basis.hpp"
#include "fmap/utils.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

struct Settings {
  std::string source;
  std::string target;
  int basis_size = 100;
  double commutativity_weight = 1.0;
  int descriptor_step = 1;
  int evaluation_samples = 1000;
  fmap::RecoveryMethod method = fmap::RecoveryMethod::NearestNeighbor;
  bool write_output = true;
  bool list = false;
  std::string list_filter;
};

void print_usage() {
  std::cout
      << "fmap " << fmap::config::kVersion << " -- functional maps pipeline\n\n"
      << "usage:\n"
      << "  fmap --source <mesh> --target <mesh> [options]\n"
      << "  fmap --list [dataset]\n\n"
      << "<mesh> is a dataset spec such as scape:0 or tosca:cat3, or a path to\n"
      << "a .off / .obj file.\n\n"
      << "options:\n"
      << "  -k, --basis <int>       spectral basis size (default 100)\n"
      << "      --weight <float>    Laplacian commutativity weight (default 1.0)\n"
      << "      --desc-step <int>   descriptor subsampling step (default 1)\n"
      << "      --eval <int>        vertices to score, 0 for all (default 1000)\n"
      << "      --method <name>     recovery method (default nearest-neighbor)\n"
      << "      --no-output         skip writing meshes to out/\n";
}

void stage(const std::string& label, double seconds) {
  std::cout << "  " << std::left << std::setw(34) << label << std::right
            << std::fixed << std::setprecision(3) << std::setw(8) << seconds
            << "s\n"
            << std::defaultfloat;
}

// Spec or path, so external meshes need no special casing.
fmap::Mesh load_any(const std::string& what) {
  if (what.find(':') != std::string::npos) return fmap::load_mesh(what);
  return fmap::load_mesh_file(what);
}

// A registry entry when the input was a spec: how we tell whether the identity
// is valid ground truth.
std::optional<fmap::MeshRef> try_resolve(const fmap::DatasetRegistry& registry,
                                         const std::string& what) {
  if (what.find(':') == std::string::npos) return std::nullopt;
  try {
    return registry.resolve(what);
  } catch (const std::runtime_error&) {
    return std::nullopt;
  }
}

int list_meshes(const std::string& which) {
  const fmap::DatasetRegistry registry;

  if (registry.datasets().empty()) {
    std::cerr << "no datasets found under " << registry.root() << '\n'
              << "set $" << fmap::config::kDataRootEnvVar
              << " if the meshes live elsewhere.\n";
    return 1;
  }

  for (const std::string& dataset : registry.datasets()) {
    if (!which.empty() && which != dataset) continue;
    std::cout << dataset << "  (" << registry.meshes(dataset).size()
              << " meshes)\n";
    for (const std::string& group : registry.groups(dataset)) {
      const auto meshes = registry.meshes(dataset, group);
      std::cout << "  " << std::left << std::setw(16) << group << std::setw(5)
                << meshes.size() << " ";
      // Ids, not a range: these datasets have gaps.
      for (std::size_t i = 0; i < meshes.size() && i < 8; ++i) {
        std::cout << meshes[i].stem << ' ';
      }
      if (meshes.size() > 8) std::cout << "...";
      std::cout << '\n';
    }
  }
  return 0;
}

int run_pipeline(const Settings& settings) {
  const fmap::DatasetRegistry registry;
  fmap::Timer timer;
  fmap::Timer total;

  // ---- load -------------------------------------------------------------
  const fmap::Mesh source = load_any(settings.source);
  const fmap::Mesh target = load_any(settings.target);

  std::cout << "meshes\n"
            << "  source  " << std::left << std::setw(22) << source.name
            << source.num_vertices() << " verts, " << source.num_faces()
            << " faces\n"
            << "  target  " << std::setw(22) << target.name
            << target.num_vertices() << " verts, " << target.num_faces()
            << " faces\n"
            << std::right;

  if (settings.basis_size >= source.num_vertices() ||
      settings.basis_size >= target.num_vertices()) {
    std::cerr << "error: basis size " << settings.basis_size
              << " must be smaller than both meshes' vertex counts\n";
    return 1;
  }

  std::cout << "\npipeline\n";

  // ---- operators and spectral bases --------------------------------------
  timer.reset();
  const fmap::LaplacianOperator source_operator = fmap::build_laplacian(source);
  const fmap::LaplacianOperator target_operator = fmap::build_laplacian(target);
  stage("cotangent Laplacian", timer.seconds());

  fmap::SpectralOptions spectral;
  spectral.k = settings.basis_size;

  timer.reset();
  const fmap::SpectralBasis source_basis =
      fmap::compute_spectral_basis(source_operator, spectral);
  const fmap::SpectralBasis target_basis =
      fmap::compute_spectral_basis(target_operator, spectral);
  stage("spectral basis (k=" + std::to_string(settings.basis_size) + ")",
        timer.seconds());

  // ---- descriptors --------------------------------------------------------
  fmap::DescriptorOptions descriptor_options;
  descriptor_options.step = settings.descriptor_step;

  timer.reset();
  const Eigen::MatrixXd source_descriptors =
      fmap::compute_all_descriptors(source_basis, descriptor_options);
  const Eigen::MatrixXd target_descriptors =
      fmap::compute_all_descriptors(target_basis, descriptor_options);
  stage("descriptors (HKS+WKS, " +
            std::to_string(source_descriptors.cols()) + ")",
        timer.seconds());

  // ---- functional map -----------------------------------------------------
  fmap::FunctionalMapOptions map_options;
  map_options.commutativity_weight = settings.commutativity_weight;

  timer.reset();
  const fmap::FunctionalMap map =
      fmap::solve_functional_map(source_basis, target_basis, source_descriptors,
                                 target_descriptors, map_options);
  stage("functional map solve", timer.seconds());

  // ---- point-to-point recovery --------------------------------------------
  const auto strategy = fmap::make_strategy(settings.method);
  timer.reset();
  const fmap::PointMap point_map =
      strategy->recover(map, source_basis, target_basis);
  stage("recovery (" + strategy->name() + ")", timer.seconds());

  std::cout << "  " << std::left << std::setw(34) << "total" << std::right
            << std::fixed << std::setprecision(3) << std::setw(8)
            << total.seconds() << "s\n"
            << std::defaultfloat;

  // ---- report -------------------------------------------------------------
  const auto stats = fmap::analyze_map(point_map, target.num_vertices());

  std::cout << "\nfunctional map\n"
            << "  descriptor residual     " << map.descriptor_residual << '\n'
            << "  commutativity residual  " << map.commutativity_residual
            << '\n'
            << "  orthogonality error     " << map.orthogonality_error()
            << '\n';

  std::cout << "\nrecovered map\n"
            << "  target coverage         " << std::fixed
            << std::setprecision(1) << stats.coverage * 100.0 << "%  ("
            << stats.distinct_targets << " of " << target.num_vertices()
            << " vertices hit)\n"
            << "  worst collision         " << stats.max_collisions
            << " source vertices on one target\n"
            << std::defaultfloat;

  // ---- scoring ------------------------------------------------------------
  // Only meaningful when the meshes share a vertex registration.
  const auto source_ref = try_resolve(registry, settings.source);
  const auto target_ref = try_resolve(registry, settings.target);
  const bool same_class =
      source_ref && target_ref && fmap::same_shape_class(*source_ref, *target_ref);
  const bool structurally_ok = fmap::has_identity_ground_truth(source, target);

  if (same_class && structurally_ok) {
    fmap::EvaluationOptions evaluation;
    evaluation.num_samples = settings.evaluation_samples;

    timer.reset();
    const fmap::EvaluationResult score =
        fmap::evaluate_against_identity(point_map, source, target, evaluation);

    std::cout << "\naccuracy vs identity ground truth  (" << score.samples
              << " sampled vertices, " << std::fixed << std::setprecision(2)
              << timer.seconds() << "s)\n"
              << std::defaultfloat
              << "  exact matches           " << std::fixed
              << std::setprecision(2) << score.exact_match_rate * 100.0 << "%\n"
              << "  mean geodesic error     " << std::setprecision(4)
              << score.mean_geodesic_error << '\n'
              << "  median geodesic error   " << score.median_geodesic_error
              << '\n'
              << "  within 1% / 5% / 10%    " << std::setprecision(1)
              << score.within(0.01) * 100.0 << "% / "
              << score.within(0.05) * 100.0 << "% / "
              << score.within(0.10) * 100.0 << "%\n\n"
              << std::defaultfloat
              << fmap::format_error_curve(score);
  } else if (structurally_ok && !same_class) {
    std::cout << "\nno ground truth: meshes have matching vertex counts but are "
                 "not from\nthe same shape class, so the identity is not a "
                 "valid reference map.\n";
  } else {
    std::cout << "\nno ground truth available for this pair.\n";
  }

  // ---- visual output ------------------------------------------------------
  if (settings.write_output) {
    const std::filesystem::path out = registry.output_dir();

    // Colour the target by position and pull it back onto the source. A good
    // map looks anatomically matched; errors speckle, and a symmetry flip
    // mirrors the colour.
    const fmap::ColorMatrix target_colors = fmap::position_colors(target);
    fmap::ColorMatrix source_colors(source.num_vertices(), 3);
    for (Eigen::Index i = 0; i < source.num_vertices(); ++i) {
      source_colors.row(i) = target_colors.row(point_map(i));
    }

    const std::filesystem::path target_path = out / "target_reference.off";
    const std::filesystem::path source_path = out / "source_mapped.off";
    fmap::write_colored_off(target_path, target, target_colors);
    fmap::write_colored_off(source_path, source, source_colors);

    std::cout << "\nwrote\n  " << target_path << "  (reference colouring)\n  "
              << source_path << "  (pulled back through the map)\n";
  }

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);
  Settings settings;

  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    auto value = [&](const char* name) -> std::string {
      if (i + 1 >= args.size()) {
        throw std::runtime_error(std::string(name) + " needs a value");
      }
      return args[++i];
    };

    try {
      if (a == "--source" || a == "-s") {
        settings.source = value("--source");
      } else if (a == "--target" || a == "-t") {
        settings.target = value("--target");
      } else if (a == "--basis" || a == "-k") {
        settings.basis_size = std::stoi(value("--basis"));
      } else if (a == "--weight") {
        settings.commutativity_weight = std::stod(value("--weight"));
      } else if (a == "--desc-step") {
        settings.descriptor_step = std::stoi(value("--desc-step"));
      } else if (a == "--eval") {
        settings.evaluation_samples = std::stoi(value("--eval"));
      } else if (a == "--method") {
        settings.method = fmap::recovery_method_from_string(value("--method"));
      } else if (a == "--no-output") {
        settings.write_output = false;
      } else if (a == "--list" || a == "-l") {
        settings.list = true;
        if (i + 1 < args.size() && args[i + 1].rfind('-', 0) != 0) {
          settings.list_filter = args[++i];
        }
      } else if (a == "--help" || a == "-h") {
        print_usage();
        return 0;
      } else {
        std::cerr << "unknown argument: " << a << "\n\n";
        print_usage();
        return 2;
      }
    } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << '\n';
      return 2;
    }
  }

  if (settings.list) return list_meshes(settings.list_filter);

  if (settings.source.empty() || settings.target.empty()) {
    print_usage();
    return (settings.source.empty() && settings.target.empty()) ? 0 : 2;
  }

  try {
    return run_pipeline(settings);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
