#include <Blackframe/Engine/TriangleMesh.hpp>
#include <cmath>
#include <limits>
#include <utility>

namespace blackframe::engine {
namespace {

[[nodiscard]] core::Error mesh_error(const core::StatusCode code, const char* const message) {
    return core::Error{
        .code = code,
        .message = message,
    };
}

[[nodiscard]] bool finite(const renderer::Point3 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

[[nodiscard]] bool finite(const renderer::Point2 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y);
}

[[nodiscard]] bool unit_normal(const renderer::Normal3 value) noexcept {
    const auto squared_length =
        std::fma(value.x, value.x, std::fma(value.y, value.y, value.z * value.z));
    constexpr auto tolerance =
        std::numeric_limits<renderer::TransportScalar>::epsilon() * renderer::TransportScalar{256};
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) &&
           std::isfinite(squared_length) &&
           std::abs(squared_length - renderer::TransportScalar{1}) <= tolerance;
}

} // namespace

core::Result<TriangleMesh> TriangleMesh::create(std::vector<renderer::Point3> positions,
                                                std::vector<renderer::Normal3> normals,
                                                std::vector<renderer::Point2> texture_coordinates,
                                                std::vector<TriangleVertexIndices> triangles) {
    if (positions.empty()) {
        return std::unexpected(mesh_error(core::StatusCode::invalid_argument,
                                          "A triangle mesh requires at least one aligned vertex."));
    }
    if (triangles.empty()) {
        return std::unexpected(
            mesh_error(core::StatusCode::invalid_argument,
                       "A triangle mesh requires at least one indexed triangle."));
    }
    if (positions.size() != normals.size() || positions.size() != texture_coordinates.size()) {
        return std::unexpected(mesh_error(
            core::StatusCode::invalid_argument,
            "Triangle mesh positions, normals, and texture coordinates must be aligned."));
    }
    if (positions.size() > std::numeric_limits<std::uint32_t>::max() ||
        triangles.size() > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(
            mesh_error(core::StatusCode::resource_exhausted,
                       "Triangle mesh counts exceed the supported 32-bit index domain."));
    }

    for (const auto position : positions) {
        if (!finite(position)) {
            return std::unexpected(
                mesh_error(core::StatusCode::invalid_argument,
                           "Triangle mesh positions must contain only finite values."));
        }
    }
    for (const auto normal : normals) {
        if (!unit_normal(normal)) {
            return std::unexpected(
                mesh_error(core::StatusCode::invalid_argument,
                           "Triangle mesh normals must be finite unit-length values."));
        }
    }
    for (const auto texture_coordinate : texture_coordinates) {
        if (!finite(texture_coordinate)) {
            return std::unexpected(
                mesh_error(core::StatusCode::invalid_argument,
                           "Triangle mesh texture coordinates must contain only finite values."));
        }
    }

    for (const auto& triangle : triangles) {
        for (const auto vertex : triangle.vertices) {
            if (static_cast<std::size_t>(vertex) >= positions.size()) {
                return std::unexpected(
                    mesh_error(core::StatusCode::invalid_argument,
                               "A triangle mesh index references an unknown aligned vertex."));
            }
        }
        const auto geometric_triangle = renderer::Triangle::create(positions[triangle.vertices[0]],
                                                                   positions[triangle.vertices[1]],
                                                                   positions[triangle.vertices[2]]);
        if (!geometric_triangle) {
            return std::unexpected(
                mesh_error(core::StatusCode::invalid_argument,
                           "A triangle mesh contains a degenerate or unrepresentable triangle."));
        }
    }

    return TriangleMesh{std::move(positions), std::move(normals), std::move(texture_coordinates),
                        std::move(triangles)};
}

std::span<const renderer::Point3> TriangleMesh::positions() const noexcept {
    return positions_;
}

std::span<const renderer::Normal3> TriangleMesh::normals() const noexcept {
    return normals_;
}

std::span<const renderer::Point2> TriangleMesh::texture_coordinates() const noexcept {
    return texture_coordinates_;
}

std::span<const TriangleVertexIndices> TriangleMesh::triangles() const noexcept {
    return triangles_;
}

core::Result<renderer::Triangle>
TriangleMesh::geometric_triangle(const std::size_t triangle_index) const {
    if (triangle_index >= triangles_.size()) {
        return std::unexpected(
            mesh_error(core::StatusCode::not_found,
                       "The triangle mesh does not contain the requested triangle."));
    }
    const auto& indices = triangles_[triangle_index].vertices;
    auto triangle = renderer::Triangle::create(positions_[indices[0]], positions_[indices[1]],
                                               positions_[indices[2]]);
    if (!triangle) {
        return std::unexpected(
            mesh_error(core::StatusCode::internal_error,
                       "A validated triangle mesh could not reconstruct its geometric triangle."));
    }
    return triangle;
}

TriangleMesh::TriangleMesh(std::vector<renderer::Point3>&& positions,
                           std::vector<renderer::Normal3>&& normals,
                           std::vector<renderer::Point2>&& texture_coordinates,
                           std::vector<TriangleVertexIndices>&& triangles) noexcept
    : positions_{std::move(positions)}, normals_{std::move(normals)},
      texture_coordinates_{std::move(texture_coordinates)}, triangles_{std::move(triangles)} {}

} // namespace blackframe::engine
