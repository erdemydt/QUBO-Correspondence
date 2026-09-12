// The one TU that instantiates header-only dependencies needing an explicit
// implementation macro, so no other file has to know which header is "the"
// implementation and duplicate symbols are impossible.

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
