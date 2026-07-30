#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <filesystem>

namespace blackframe::engine {

// Both entry points require an absolute path with the matching lower-case
// extension. OBJ faces must be triangles with complete v/vt/vn corners. PLY
// input is the ASCII 1.0 variant with aligned x/y/z, nx/ny/nz, and u/v vertex
// properties. Unsupported encodings and records fail explicitly.
[[nodiscard]] core::Result<TriangleMesh>
load_obj_triangle_mesh(const std::filesystem::path& absolute_path);

[[nodiscard]] core::Result<TriangleMesh>
load_ply_triangle_mesh(const std::filesystem::path& absolute_path);

} // namespace blackframe::engine
