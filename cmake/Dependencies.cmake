# All deps are header-only, so we fetch sources and wrap them in INTERFACE
# targets rather than configuring each upstream project. That dodges CMake rot
# (4.x rejects the cmake_minimum_required(<3.5) several still declare) and keeps
# their test/example targets, some ON by default, from pulling in GTest.
# SYSTEM so third-party warnings don't drown out ours.

include(FetchContent)

#   NAME <fetchcontent name>  REPO <url>  TAG <tag>
#   TARGET <e.g. Eigen3::Eigen>  INCLUDE <subdir, or omit for root>
function(fmap_header_only_dependency)
  cmake_parse_arguments(ARG "" "NAME;REPO;TAG;TARGET;INCLUDE" "" ${ARGN})

  FetchContent_Declare(${ARG_NAME}
    GIT_REPOSITORY ${ARG_REPO}
    GIT_TAG        ${ARG_TAG}
    GIT_SHALLOW    TRUE
    # No CMakeLists.txt there, so FetchContent downloads but skips
    # add_subdirectory() and upstream's build never runs. That is the mechanism.
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

  # Namespaced ALIAS so typos fail at configure time.
  string(REPLACE "::" "_" _impl "${ARG_TARGET}")
  add_library(${_impl} INTERFACE)
  target_include_directories(${_impl} SYSTEM INTERFACE "${_inc}")
  add_library(${ARG_TARGET} ALIAS ${_impl})

  message(STATUS "  ${ARG_TARGET} <- ${_inc}")
endfunction()

message(STATUS "Resolving dependencies:")

# Pinned to 3.4: Spectra 1.x targets the Eigen 3 API.
fmap_header_only_dependency(
  NAME    eigen
  REPO    https://gitlab.com/libeigen/eigen.git
  TAG     3.4.1
  TARGET  Eigen3::Eigen
)

# SymGEigsShiftSolver: the *smallest* Laplacian eigenpairs.
fmap_header_only_dependency(
  NAME    spectra
  REPO    https://github.com/yixuan/spectra.git
  TAG     v1.2.0
  TARGET  Spectra::Spectra
  INCLUDE include
)

# KD-trees for recovery in the spectral embedding.
fmap_header_only_dependency(
  NAME    nanoflann
  REPO    https://github.com/jlblancoc/nanoflann.git
  TAG     v1.10.0
  TARGET  nanoflann::nanoflann
  INCLUDE include
)

# .obj: v/vt/vn index triples and negative indices a hand-rolled parser botches.
fmap_header_only_dependency(
  NAME    tinyobjloader
  REPO    https://github.com/tinyobjloader/tinyobjloader.git
  TAG     v2.0.0rc13
  TARGET  tinyobjloader::tinyobjloader
)
