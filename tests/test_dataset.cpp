// Dataset discovery and spec resolution. These assertions encode facts about
// the data on disk that later stages rely on -- notably the index gaps, which
// are why the registry scans directories instead of building filenames.

#include "fmap/dataset.hpp"

#include <algorithm>
#include <stdexcept>

#include "test_support.hpp"

using namespace fmap::test;
using fmap::DatasetRegistry;
using fmap::MeshRef;

namespace {

void test_discovery(const DatasetRegistry& reg) {
  section("discovery");

  const std::vector<std::string> ds = reg.datasets();
  check(std::find(ds.begin(), ds.end(), "scape") != ds.end(), "found scape");
  check(std::find(ds.begin(), ds.end(), "tosca") != ds.end(), "found tosca");

  check(reg.meshes("scape").size() == 71, "scape has 71 meshes");
  check(reg.meshes("tosca").size() == 82, "tosca has 82 meshes");
}

void test_groups(const DatasetRegistry& reg) {
  section("shape classes");

  const std::vector<std::string> tosca = reg.groups("tosca");
  check(std::find(tosca.begin(), tosca.end(), "cat") != tosca.end(),
        "tosca has a cat class");
  check(reg.meshes("tosca", "cat").size() == 11, "cat class has 11 meshes");

  // horse0_partial is not vertex-aligned with the horse class (15974 vs 19248),
  // so grouping it there would make evaluation assume false ground truth.
  check(std::find(tosca.begin(), tosca.end(), "horse0_partial") != tosca.end(),
        "horse0_partial is its own class");
  const auto horses = reg.meshes("tosca", "horse");
  check(std::none_of(horses.begin(), horses.end(),
                     [](const MeshRef& m) { return m.stem == "horse0_partial"; }),
        "horse class excludes the partial mesh");

  check(reg.groups("scape").size() == 1, "scape is a single class");
}

void test_resolution(const DatasetRegistry& reg) {
  section("spec resolution");

  // Bare integer resolves through the zero-padding.
  const MeshRef a = reg.resolve("scape:7");
  check(a.stem == "mesh007", "scape:7 -> mesh007");
  check(std::filesystem::exists(a.path), "resolved path exists");
  check(a.spec() == "scape:mesh007", "spec round-trips");

  // Exact stem also works.
  check(reg.resolve("scape:mesh007").path == a.path, "stem and index agree");

  // TOSCA goes by stem: an index alone is ambiguous (cat0, dog0, wolf0, ...).
  const MeshRef c = reg.resolve("tosca:cat3");
  check(c.group == "cat", "tosca:cat3 is in the cat class");
  check(std::filesystem::exists(c.path), "resolved tosca path exists");
  check(!reg.contains("tosca:0"), "bare ambiguous index is rejected");
}

void test_gaps(const DatasetRegistry& reg) {
  section("index gaps are respected");

  // Real gaps. A formula would return paths for them and fail later, confusingly.
  check(!reg.contains("scape:51"), "scape has no mesh051");
  check(!reg.contains("tosca:horse8"), "tosca horse class skips 8");
  check(reg.contains("tosca:horse17"), "but horse17 does exist");

  // Natural ordering: horse10 must sort after horse7, not after horse1.
  const auto horses = reg.meshes("tosca", "horse");
  check(!horses.empty() && horses.front().stem == "horse0",
        "horse class starts at horse0");
  check(!horses.empty() && horses.back().stem == "horse17",
        "horse class ends at horse17");
}

void test_errors(const DatasetRegistry& reg) {
  section("error reporting");

  bool threw = false;
  try {
    reg.resolve("scape7");  // missing colon
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "spec without ':' is rejected");

  threw = false;
  try {
    reg.resolve("nosuch:0");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "unknown dataset is rejected");
}

}  // namespace

int main() {
  const DatasetRegistry reg;

  if (reg.datasets().empty()) {
    std::cout << "no datasets found under " << reg.root()
              << "; nothing to verify\n";
    return 1;
  }

  test_discovery(reg);
  test_groups(reg);
  test_resolution(reg);
  test_gaps(reg);
  test_errors(reg);
  return summary();
}
