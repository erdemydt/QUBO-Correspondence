# Every dependency here is header-only, so we fetch the sources and wrap them in
# INTERFACE targets rather than letting each configure its own CMake project.
# That avoids upstream CMake rot (4.x rejects the cmake_minimum_required(<3.5)
# several of these still declare), keeps their test/example targets -- some ON
# by default, pulling in GTest -- out of the build, and makes configure fast.
# Marked SYSTEM so third-party warnings don't drown out our own.

include(FetchContent)

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
    # A path with no CMakeLists.txt: FetchContent downloads but skips
    # add_subdirectory(), so upstream's build system never runs. This is the
    # mechanism the header above describes.
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

  # INTERFACE library plus namespaced ALIAS, so consumers link the ::-qualified
  # name and typos fail at configure time.
  string(REPLACE "::" "_" _impl "${ARG_TARGET}")
  add_library(${_impl} INTERFACE)
  target_include_directories(${_impl} SYSTEM INTERFACE "${_inc}")
  add_library(${ARG_TARGET} ALIAS ${_impl})

  message(STATUS "  ${ARG_TARGET} <- ${_inc}")
endfunction()

message(STATUS "Resolving dependencies:")

# Linear algebra, the base type for everything here. Pinned to the 3.4
# maintenance branch: Spectra 1.x targets the Eigen 3 API.
fmap_header_only_dependency(
  NAME    eigen
  REPO    https://gitlab.com/libeigen/eigen.git
  TAG     3.4.1
  TARGET  Eigen3::Eigen
)

# SymGEigsShiftSolver, which is how we get the *smallest* Laplacian eigenpairs;
# plain Lanczos converges badly at that end of the spectrum.
fmap_header_only_dependency(
  NAME    spectra
  REPO    https://github.com/yixuan/spectra.git
  TAG     v1.2.0
  TARGET  Spectra::Spectra
  INCLUDE include
)

# KD-trees, for pulling a point-to-point map out of the spectral embedding.
fmap_header_only_dependency(
  NAME    nanoflann
  REPO    https://github.com/jlblancoc/nanoflann.git
  TAG     v1.10.0
  TARGET  nanoflann::nanoflann
  INCLUDE include
)

# .obj reading. The datasets here are .off, but real .obj files carry v/vt/vn
# index triples and negative (relative) indices a hand-rolled parser gets wrong.
fmap_header_only_dependency(
  NAME    tinyobjloader
  REPO    https://github.com/tinyobjloader/tinyobjloader.git
  TAG     v2.0.0rc13
  TARGET  tinyobjloader::tinyobjloader
)
