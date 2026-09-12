# One smoke test per pipeline block, so a regression in an early stage is caught
# before it propagates into the later ones.

# Compiles with our warning set; sources can include test_support.hpp.
function(fmap_add_test name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LINK" ${ARGN})

  add_executable(${name} ${ARG_SOURCES})
  target_link_libraries(${name} PRIVATE fmap_warnings ${ARG_LINK})

  add_test(NAME ${name} COMMAND ${name})
endfunction()

# Guards the third-party stack itself: all four dependencies compiling together,
# and Spectra's shift-invert converging on a Laplacian-shaped problem.
fmap_add_test(test_dependencies
  SOURCES test_dependencies.cpp
  LINK    Eigen3::Eigen Spectra::Spectra nanoflann::nanoflann
          tinyobjloader::tinyobjloader
)

fmap_add_test(test_dataset
  SOURCES test_dataset.cpp
  LINK    fmap_core
)

fmap_add_test(test_mesh
  SOURCES test_mesh.cpp
  LINK    fmap_core
)

fmap_add_test(test_laplacian
  SOURCES test_laplacian.cpp
  LINK    fmap_core
)

fmap_add_test(test_spectral_basis
  SOURCES test_spectral_basis.cpp
  LINK    fmap_core Spectra::Spectra
)

fmap_add_test(test_descriptors
  SOURCES test_descriptors.cpp
  LINK    fmap_core Spectra::Spectra
)

fmap_add_test(test_functional_map
  SOURCES test_functional_map.cpp
  LINK    fmap_core Spectra::Spectra
)

fmap_add_test(test_correspondence
  SOURCES test_correspondence.cpp
  LINK    fmap_core Spectra::Spectra
)

fmap_add_test(test_evaluation
  SOURCES test_evaluation.cpp
  LINK    fmap_core Spectra::Spectra
)
