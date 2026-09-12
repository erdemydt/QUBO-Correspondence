// Single translation unit that instantiates the header-only dependencies which
// need an explicit implementation macro.
//
// Keeping this in one place means no other file has to care about which header
// is "the implementation" one, and makes duplicate-symbol errors impossible if
// a second consumer includes the same header.

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
