#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <Blackframe/Renderer/Triangle.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace blackframe::engine {

struct TriangleVertexIndices final {
    std::array<std::uint32_t, 3> vertices{};

    [[nodiscard]] constexpr bool operator==(const TriangleVertexIndices&) const noexcept = default;
};

static_assert(std::is_standard_layout_v<TriangleVertexIndices>);
static_assert(std::is_trivially_copyable_v<TriangleVertexIndices>);
static_assert(sizeof(TriangleVertexIndices) == 3 * sizeof(std::uint32_t));

// Logical bytes copied by a renderer from the four POD mesh buffers, plus the
// equivalent fully expanded corner payload for comparison. The report
// deliberately excludes vector capacity, allocator overhead, and the owning
// C++ object because none of those belong in a CPU or CUDA payload.
struct TriangleMeshMemoryReport final {
    std::uint64_t position_bytes{};
    std::uint64_t normal_bytes{};
    std::uint64_t texture_coordinate_bytes{};
    std::uint64_t index_bytes{};
    std::uint64_t payload_bytes{};
    std::uint64_t expanded_triangle_bytes{};

    [[nodiscard]] constexpr bool
    operator==(const TriangleMeshMemoryReport&) const noexcept = default;
};

static_assert(std::is_standard_layout_v<TriangleMeshMemoryReport>);
static_assert(std::is_trivially_copyable_v<TriangleMeshMemoryReport>);

// A transport mesh owns one aligned position/normal/UV tuple per vertex.
// Importers must split source vertices at attribute seams instead of repairing
// missing attributes or assuming that source index domains are identical.
class TriangleMesh final {
  public:
    // Closed frame scenes may share a mesh with its creator. Construction stays
    // available for value returns, but assignment cannot replace shared buffers.
    TriangleMesh(const TriangleMesh&) = default;
    TriangleMesh(TriangleMesh&&) noexcept = default;
    TriangleMesh& operator=(const TriangleMesh&) = delete;
    TriangleMesh& operator=(TriangleMesh&&) = delete;

    [[nodiscard]] static core::Result<TriangleMesh>
    create(std::vector<renderer::Point3> positions, std::vector<renderer::Normal3> normals,
           std::vector<renderer::Point2> texture_coordinates,
           std::vector<TriangleVertexIndices> triangles);

    [[nodiscard]] std::span<const renderer::Point3> positions() const noexcept;
    [[nodiscard]] std::span<const renderer::Normal3> normals() const noexcept;
    [[nodiscard]] std::span<const renderer::Point2> texture_coordinates() const noexcept;
    [[nodiscard]] std::span<const TriangleVertexIndices> triangles() const noexcept;

    // Visits triangle corners in stable order, removes unreferenced vertices,
    // and merges only bit-identical aligned tuples. No tolerant welding,
    // quantization, index-width substitution, or source fallback is applied.
    [[nodiscard]] core::Result<TriangleMesh> compacted() const;

    [[nodiscard]] TriangleMeshMemoryReport memory_report() const noexcept;

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
