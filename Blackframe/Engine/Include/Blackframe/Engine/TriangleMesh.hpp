#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/Triangle.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace blackframe::engine {

struct TriangleVertexIndices final {
    std::array<std::uint32_t, 3> vertices{};

    [[nodiscard]] constexpr bool operator==(const TriangleVertexIndices&) const noexcept = default;
};

// A transport mesh owns one aligned position/normal/UV tuple per vertex.
// Importers must split source vertices at attribute seams instead of repairing
// missing attributes or assuming that source index domains are identical.
class TriangleMesh final {
  public:
    [[nodiscard]] static core::Result<TriangleMesh>
    create(std::vector<renderer::Point3> positions, std::vector<renderer::Normal3> normals,
           std::vector<renderer::Point2> texture_coordinates,
           std::vector<TriangleVertexIndices> triangles);

    [[nodiscard]] std::span<const renderer::Point3> positions() const noexcept;
    [[nodiscard]] std::span<const renderer::Normal3> normals() const noexcept;
    [[nodiscard]] std::span<const renderer::Point2> texture_coordinates() const noexcept;
    [[nodiscard]] std::span<const TriangleVertexIndices> triangles() const noexcept;

    [[nodiscard]] core::Result<renderer::Triangle>
    geometric_triangle(std::size_t triangle_index) const;

  private:
    TriangleMesh(std::vector<renderer::Point3>&& positions,
                 std::vector<renderer::Normal3>&& normals,
                 std::vector<renderer::Point2>&& texture_coordinates,
                 std::vector<TriangleVertexIndices>&& triangles) noexcept;

    std::vector<renderer::Point3> positions_;
    std::vector<renderer::Normal3> normals_;
    std::vector<renderer::Point2> texture_coordinates_;
    std::vector<TriangleVertexIndices> triangles_;
};

} // namespace blackframe::engine
