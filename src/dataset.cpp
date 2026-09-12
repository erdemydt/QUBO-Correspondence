#include "fmap/dataset.hpp"

#include "fmap/config.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace fmap {
namespace {

// The one place directory names appear. A new dataset is a row here, plus a
// .gitignore line if the data lives in the repo.
struct DatasetDir {
  const char* name;       // spec prefix, e.g. "scape"
  const char* directory;  // directory under the data root
};

constexpr DatasetDir kDatasetDirs[] = {
    {"scape", "scape-bim"},
    {"tosca", "tosca-offs"},
};

bool is_mesh_extension(const std::filesystem::path& ext) {
  return ext == ".off" || ext == ".obj" || ext == ".ply";
}

// Shape class: strip the trailing index, "cat3" -> "cat". Meshes in a group are
// the ones that may share a vertex numbering. This deliberately leaves
// "horse0_partial" in its own group -- it has a different vertex count and is
// not vertex-aligned with the horse class.
std::string group_of(const std::string& stem) {
  std::size_t end = stem.size();
  while (end > 0 && std::isdigit(static_cast<unsigned char>(stem[end - 1]))) {
    --end;
  }
  // A stem that is all digits has no class name; treat the whole thing as one.
  return end == 0 ? stem : stem.substr(0, end);
}

// Trailing integer, or -1. For natural ordering: mesh10 sorts after mesh9.
long index_of(const std::string& stem) {
  std::size_t end = stem.size();
  while (end > 0 && std::isdigit(static_cast<unsigned char>(stem[end - 1]))) {
    --end;
  }
  if (end == stem.size()) return -1;
  return std::stol(stem.substr(end));
}

std::filesystem::path default_root() {
  if (const char* env = std::getenv(config::kDataRootEnvVar);
      env != nullptr && *env != '\0') {
    return std::filesystem::path(env);
  }
  return std::filesystem::path(config::kProjectRoot);
}

}  // namespace

DatasetRegistry::DatasetRegistry() : DatasetRegistry(default_root()) {}

DatasetRegistry::DatasetRegistry(std::filesystem::path root)
    : root_(std::move(root)) {
  scan();
}

void DatasetRegistry::scan() {
  std::error_code ec;
  for (const DatasetDir& dir : kDatasetDirs) {
    const std::filesystem::path dir_path = root_ / dir.directory;
    if (!std::filesystem::is_directory(dir_path, ec)) continue;

    for (const auto& entry : std::filesystem::directory_iterator(dir_path, ec)) {
      if (!entry.is_regular_file(ec)) continue;
      if (!is_mesh_extension(entry.path().extension())) continue;

      const std::string stem = entry.path().stem().string();
      meshes_.push_back(MeshRef{dir.name, group_of(stem), stem, entry.path()});
    }
  }

  std::sort(meshes_.begin(), meshes_.end(),
            [](const MeshRef& a, const MeshRef& b) {
              if (a.dataset != b.dataset) return a.dataset < b.dataset;
              if (a.group != b.group) return a.group < b.group;
              const long ia = index_of(a.stem);
              const long ib = index_of(b.stem);
              if (ia != ib) return ia < ib;
              return a.stem < b.stem;
            });
}

MeshRef DatasetRegistry::resolve(std::string_view spec) const {
  const std::size_t colon = spec.find(':');
  if (colon == std::string_view::npos) {
    std::ostringstream msg;
    msg << "mesh spec '" << spec << "' is missing a ':'; expected "
        << "<dataset>:<id>, e.g. scape:0 or tosca:cat3";
    throw std::runtime_error(msg.str());
  }

  const std::string dataset(spec.substr(0, colon));
  const std::string id(spec.substr(colon + 1));

  // Exact stem match.
  for (const MeshRef& m : meshes_) {
    if (m.dataset == dataset && m.stem == id) return m;
  }

  // Bare integer: match against the numeric suffix, so "scape:7" finds
  // mesh007 without the caller knowing the zero-padding width.
  const bool numeric = !id.empty() &&
                       std::all_of(id.begin(), id.end(), [](unsigned char c) {
                         return std::isdigit(c);
                       });
  if (numeric) {
    const long wanted = std::stol(id);
    const MeshRef* match = nullptr;
    for (const MeshRef& m : meshes_) {
      if (m.dataset == dataset && index_of(m.stem) == wanted) {
        // Ambiguous if several groups share the index (TOSCA cat0, dog0, ...).
        if (match != nullptr) {
          match = nullptr;
          break;
        }
        match = &m;
      }
    }
    if (match != nullptr) return *match;
  }

  // Failed: report what is actually available rather than just the miss.
  std::ostringstream msg;
  msg << "could not resolve mesh spec '" << spec << "'.\n";

  const std::vector<std::string> known = datasets();
  if (std::find(known.begin(), known.end(), dataset) == known.end()) {
    msg << "  unknown dataset '" << dataset << "'; found:";
    if (known.empty()) {
      msg << " (none under " << root_ << ")";
    } else {
      for (const std::string& d : known) msg << ' ' << d;
    }
  } else {
    msg << "  dataset '" << dataset << "' has no mesh '" << id << "'.\n";
    msg << "  groups:";
    for (const std::string& g : groups(dataset)) msg << ' ' << g;
    msg << "\n  example ids:";
    int shown = 0;
    for (const MeshRef& m : meshes_) {
      if (m.dataset == dataset && shown < 6) {
        msg << ' ' << m.stem;
        ++shown;
      }
    }
    if (shown == 6) msg << " ...";
  }
  throw std::runtime_error(msg.str());
}

bool DatasetRegistry::contains(std::string_view spec) const {
  try {
    resolve(spec);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

std::vector<std::string> DatasetRegistry::datasets() const {
  std::vector<std::string> out;
  for (const MeshRef& m : meshes_) {
    if (std::find(out.begin(), out.end(), m.dataset) == out.end()) {
      out.push_back(m.dataset);
    }
  }
  return out;
}

std::vector<std::string> DatasetRegistry::groups(
    std::string_view dataset) const {
  std::vector<std::string> out;
  for (const MeshRef& m : meshes_) {
    if (m.dataset != dataset) continue;
    if (std::find(out.begin(), out.end(), m.group) == out.end()) {
      out.push_back(m.group);
    }
  }
  return out;
}

std::vector<MeshRef> DatasetRegistry::meshes(std::string_view dataset,
                                             std::string_view group) const {
  std::vector<MeshRef> out;
  for (const MeshRef& m : meshes_) {
    if (m.dataset != dataset) continue;
    if (!group.empty() && m.group != group) continue;
    out.push_back(m);
  }
  return out;
}

std::filesystem::path DatasetRegistry::output_dir() const {
  const std::filesystem::path dir =
      std::filesystem::path(config::kProjectRoot) / config::kOutputDirName;
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  return dir;
}

MeshRef resolve_mesh(std::string_view spec) {
  return DatasetRegistry().resolve(spec);
}

bool same_shape_class(const MeshRef& a, const MeshRef& b) {
  return a.dataset == b.dataset && a.group == b.group;
}

}  // namespace fmap
