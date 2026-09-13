// Deliberately not a test framework: GTest just to compare doubles would be the
// heaviest thing in the build. A plain executable returning non-zero is all
// CTest needs.

#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace fmap::test {

inline int failures = 0;

inline void check(bool ok, const std::string& what) {
  if (ok) {
    std::cout << "  ok   " << what << '\n';
  } else {
    std::cout << "  FAIL " << what << '\n';
    ++failures;
  }
}

// Absolute tolerance: these are physical quantities of known scale.
inline void check_near(double got, double want, double tol,
                       const std::string& what) {
  const bool ok = std::abs(got - want) <= tol;
  if (ok) {
    std::cout << "  ok   " << what << " (" << got << ")\n";
  } else {
    std::cout << "  FAIL " << what << ": got " << got << ", want " << want
              << " +/- " << tol << '\n';
    ++failures;
  }
}

inline void section(const std::string& name) {
  std::cout << name << '\n';
}

// Return from main().
inline int summary() {
  if (failures == 0) {
    std::cout << "PASS\n";
    return 0;
  }
  std::cout << "FAIL (" << failures << " check(s))\n";
  return 1;
}

}  // namespace fmap::test
