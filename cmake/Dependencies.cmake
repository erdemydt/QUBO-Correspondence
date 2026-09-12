# Third-party dependencies.
#
# Every dependency this project uses is header-only. Rather than letting each
# one configure its own CMake project, we fetch the sources and wrap them in
# INTERFACE targets ourselves. That buys three things:
#
#   1. Immunity to upstream CMake rot. CMake 4.x removed compatibility with
#      cmake_minimum_required(VERSION <3.5), which several of these projects
#      still declare -- configuring them directly is a hard error.
#   2. No stray build targets. Upstream test/example/benchmark targets default
#      to ON in some of these projects and would drag in GTest and friends.
#   3. Fast configure. Nothing downstream is configured or compiled; we only
#      need the headers on the include path.
#
# Sources are marked SYSTEM so warnings from third-party headers don't drown
# out warnings from our own code.

include(FetchContent)

# Fetch a header-only dependency and expose it as an INTERFACE target.
#
#   fmap_header_only_dependency(
#     NAME     <fetchcontent name>        # lowercase; drives <name>_SOURCE_DIR
#     REPO     <git url>
#     TAG      <tag or commit>
#     TARGET   <target name to create>    # e.g. Eigen3::Eigen
#     INCLUDE  <subdir of source tree>    # optional; "" means the root
#   )
function(fmap_header_only_dependency)
  cmake_parse_arguments(ARG "" "NAME;REPO;TAG;TARGET;INCLUDE" "" ${ARGN})

  FetchContent_Declare(${ARG_NAME}
    GIT_REPOSITORY ${ARG_REPO}
    GIT_TAG        ${ARG_TAG}
    GIT_SHALLOW    TRUE
    # Point SOURCE_SUBDIR at a path that holds no CMakeLists.txt. FetchContent
    # then downloads the sources but skips add_subdirectory(), so upstream's
    # build system never runs. This is the mechanism that makes points 1-3
    # above work.
    SOURCE_SUBDIR  cmake-entry-point-intentionally-absent
  )
  FetchContent_MakeAvailable(${ARG_NAME})

  set(_src "${${ARG_NAME}_SOURCE_DIR}")
  if(ARG_INCLUDE)
    set(_inc "${_src}/${ARG_INCLUDE}")
  else()
    set(_inc "${_src}")
  endif()

  if(NOT EXISTS "${_inc}")
    message(FATAL_ERROR "${ARG_NAME}: expected include dir not found: ${_inc}")
  endif()

  # Two-target dance: INTERFACE library with a namespaced ALIAS, so consumers
  # always link the ::-qualified name and typos fail at configure time.
  string(REPLACE "::" "_" _impl "${ARG_TARGET}")
  add_library(${_impl} INTERFACE)
  target_include_directories(${_impl} SYSTEM INTERFACE "${_inc}")
  add_library(${ARG_TARGET} ALIAS ${_impl})

  message(STATUS "  ${ARG_TARGET} <- ${_inc}")
endfunction()

message(STATUS "Resolving dependencies:")

# Eigen -- dense/sparse linear algebra, the base type for everything here.
# Pinned to the 3.4 maintenance branch: Spectra 1.x targets the Eigen 3 API.
fmap_header_only_dependency(
  NAME    eigen
  REPO    https://gitlab.com/libeigen/eigen.git
  TAG     3.4.1
  TARGET  Eigen3::Eigen
)

# Spectra -- sparse eigensolvers built on Eigen. Supplies SymGEigsShiftSolver,
# which is how we get the *smallest* eigenpairs of the Laplacian; plain Lanczos
# converges badly at that end of the spectrum.
fmap_header_only_dependency(
  NAME    spectra
  REPO    https://github.com/yixuan/spectra.git
  TAG     v1.2.0
  TARGET  Spectra::Spectra
  INCLUDE include
)

# nanoflann -- KD-trees, used to pull a point-to-point map out of the spectral
# embedding by nearest-neighbour search.
fmap_header_only_dependency(
  NAME    nanoflann
  REPO    https://github.com/jlblancoc/nanoflann.git
  TAG     v1.10.0
  TARGET  nanoflann::nanoflann
  INCLUDE include
)

# tinyobjloader -- .obj reading. The benchmark datasets here are .off, but this
# keeps the loader honest for real-world .obj files, which carry v/vt/vn index
# triples and negative (relative) indices that a quick hand-rolled parser gets
# wrong.
fmap_header_only_dependency(
  NAME    tinyobjloader
  REPO    https://github.com/tinyobjloader/tinyobjloader.git
  TAG     v2.0.0rc13
  TARGET  tinyobjloader::tinyobjloader
)
