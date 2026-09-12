cmake_minimum_required(VERSION 3.24)
project(fmap
  VERSION     0.1.0
  DESCRIPTION "Functional maps pipeline for shape correspondence"
  LANGUAGES   CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Always emit compile_commands.json so clangd/editors work without extra setup.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# This pipeline is numerically heavy; an unoptimized build is unusably slow on
# real meshes. Default to Release when the user hasn't chosen.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Dependencies)

# Warning set applied to our code only (dependencies are included as SYSTEM).
add_library(fmap_warnings INTERFACE)
target_compile_options(fmap_warnings INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

# config.hpp bakes in the project root so no source file carries a literal path.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/fmap/config.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/fmap/config.hpp"
  @ONLY
)

# The pipeline blocks. main.cpp and every test link against this, so each block
# is compiled once rather than re-instantiating Eigen templates per test binary.
add_library(fmap_core STATIC
  src/dataset.cpp
  src/mesh.cpp
  src/laplacian.cpp
  src/spectral_basis.cpp
  src/descriptors.cpp
  src/functional_map.cpp
  src/correspondence.cpp
  src/evaluation.cpp
  src/third_party_impl.cpp
)
target_include_directories(fmap_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
target_link_libraries(fmap_core
  PUBLIC  Eigen3::Eigen
  PRIVATE fmap_warnings tinyobjloader::tinyobjloader nanoflann::nanoflann Spectra::Spectra
)

add_executable(fmap src/main.cpp)
target_link_libraries(fmap PRIVATE fmap_core fmap_warnings)

enable_testing()
add_subdirectory(tests)
