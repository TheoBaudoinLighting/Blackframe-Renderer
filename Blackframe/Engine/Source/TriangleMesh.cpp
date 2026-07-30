#include <Blackframe/Engine/TriangleMesh.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>
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

using AlignedVertexBits = std::array<std::uint32_t, 8>;

[[nodiscard]] AlignedVertexBits
aligned_vertex_bits(const renderer::Point3 position, const renderer::Normal3 normal,
                    const renderer::Point2 texture_coordinate) noexcept {
    return {
        std::bit_cast<std::uint32_t>(position.x),
        std::bit_cast<std::uint32_t>(position.y),
        std::bit_cast<std::uint32_t>(position.z),
        std::bit_cast<std::uint32_t>(normal.x),
        std::bit_cast<std::uint32_t>(normal.y),
        std::bit_cast<std::uint32_t>(normal.z),
        std::bit_cast<std::uint32_t>(texture_coordinate.x),
        std::bit_cast<std::uint32_t>(texture_coordinate.y),
    };
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

core::Result<TriangleMesh> TriangleMesh::compacted() const {
    try {
        auto compact_index_by_vertex = std::map<AlignedVertexBits, std::uint32_t>{};
        auto source_vertex_by_compact_index = std::vector<std::uint32_t>{};
        auto maximum_referenced_vertices = positions_.size();
        constexpr auto corners_per_triangle = std::size_t{3};
        if (triangles_.size() <= std::numeric_limits<std::size_t>::max() / corners_per_triangle) {
            maximum_referenced_vertices =
                std::min(maximum_referenced_vertices, triangles_.size() * corners_per_triangle);
        }
        source_vertex_by_compact_index.reserve(maximum_referenced_vertices);
        auto compact_triangles = std::vector<TriangleVertexIndices>(triangles_.size());

        for (auto triangle_index = std::size_t{}; triangle_index < triangles_.size();
             ++triangle_index) {
            const auto& source_triangle = triangles_[triangle_index];
            auto& compact_triangle = compact_triangles[triangle_index];
            for (auto corner = std::size_t{}; corner < source_triangle.vertices.size(); ++corner) {
                const auto source_index = source_triangle.vertices[corner];
                const auto key =
                    aligned_vertex_bits(positions_[source_index], normals_[source_index],
                                        texture_coordinates_[source_index]);
                const auto existing = compact_index_by_vertex.find(key);
                if (existing != compact_index_by_vertex.end()) {
                    compact_triangle.vertices[corner] = existing->second;
                    continue;
                }

                if (source_vertex_by_compact_index.size() >=
                    std::numeric_limits<std::uint32_t>::max()) {
                    return std::unexpected(
                        mesh_error(core::StatusCode::resource_exhausted,
                                   "Compacted triangle mesh exceeds the 32-bit index domain."));
                }
                const auto compact_index =
                    static_cast<std::uint32_t>(source_vertex_by_compact_index.size());
                compact_index_by_vertex.emplace(key, compact_index);
                source_vertex_by_compact_index.push_back(source_index);
                compact_triangle.vertices[corner] = compact_index;
            }
        }

        auto compact_positions =
            std::vector<renderer::Point3>(source_vertex_by_compact_index.size());
        auto compact_normals =
            std::vector<renderer::Normal3>(source_vertex_by_compact_index.size());
        auto compact_texture_coordinates =
            std::vector<renderer::Point2>(source_vertex_by_compact_index.size());
        for (auto compact_index = std::size_t{};
             compact_index < source_vertex_by_compact_index.size(); ++compact_index) {
            const auto source_index = source_vertex_by_compact_index[compact_index];
            compact_positions[compact_index] = positions_[source_index];
            compact_normals[compact_index] = normals_[source_index];
            compact_texture_coordinates[compact_index] = texture_coordinates_[source_index];
        }

        return TriangleMesh::create(std::move(compact_positions), std::move(compact_normals),
                                    std::move(compact_texture_coordinates),
                                    std::move(compact_triangles));
    } catch (const std::bad_alloc&) {
        return std::unexpected(mesh_error(core::StatusCode::resource_exhausted,
                                          "Triangle mesh compaction exhausted host memory."));
    } catch (const std::length_error&) {
        return std::unexpected(
            mesh_error(core::StatusCode::resource_exhausted,
                       "Triangle mesh compaction exceeded host container limits."));
    }
}

TriangleMeshMemoryReport TriangleMesh::memory_report() const noexcept {
    constexpr auto position_bytes = std::uint64_t{sizeof(renderer::Point3)};
    constexpr auto normal_bytes = std::uint64_t{sizeof(renderer::Normal3)};
    constexpr auto texture_coordinate_bytes = std::uint64_t{sizeof(renderer::Point2)};
    constexpr auto index_bytes = std::uint64_t{sizeof(TriangleVertexIndices)};
    constexpr auto attributes_per_corner = position_bytes + normal_bytes + texture_coordinate_bytes;

    const auto vertex_count = static_cast<std::uint64_t>(positions_.size());
    const auto triangle_count = static_cast<std::uint64_t>(triangles_.size());
    const auto report_position_bytes = vertex_count * position_bytes;
    const auto report_normal_bytes = vertex_count * normal_bytes;
    const auto report_texture_coordinate_bytes = vertex_count * texture_coordinate_bytes;
    const auto report_index_bytes = triangle_count * index_bytes;
    return TriangleMeshMemoryReport{
        .position_bytes = report_position_bytes,
        .normal_bytes = report_normal_bytes,
        .texture_coordinate_bytes = report_texture_coordinate_bytes,
        .index_bytes = report_index_bytes,
        .payload_bytes = report_position_bytes + report_normal_bytes +
                         report_texture_coordinate_bytes + report_index_bytes,
        .expanded_triangle_bytes = triangle_count * 3U * attributes_per_corner,
    };
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
