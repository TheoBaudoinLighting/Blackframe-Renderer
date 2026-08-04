#include <Blackframe/Backends/GPU/CUDA/SceneBvh.hpp>
#include <Blackframe/Renderer/Triangle.hpp>
#include <Blackframe/XPU/Shared/SceneSoaAbi.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

namespace bvh_array = xpu::shared::scene_bvh_array;
namespace scene_column = xpu::shared::scene_soa_column;
using xpu::shared::SceneBvhBlas;
using xpu::shared::SceneBvhHeader;
using xpu::shared::SceneBvhInstanceReference;
using xpu::shared::SceneBvhNode;
using xpu::shared::SceneSoaHeader;

static_assert(std::endian::native == std::endian::little,
              "CUDA BVH serialization requires a little-endian host.");
static_assert(std::numeric_limits<float>::is_iec559);
static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));

struct Bounds final {
    std::array<float, 3> minimum{};
    std::array<float, 3> maximum{};
};

struct BuildItem final {
    Bounds bounds{};
    std::array<double, 3> centroid{};
    std::uint32_t stable_key{};
    SceneBvhInstanceReference instance_reference{};
};

struct SourceGeometry final {
    std::uint32_t id{};
    std::uint64_t vertex_offset{};
    std::uint64_t vertex_count{};
    std::uint64_t triangle_offset{};
    std::uint64_t triangle_count{};
};

struct DownloadedScene final {
    SceneSoaHeader header{};
    std::vector<std::uint8_t> bytes;
};

struct BuiltBvh final {
    SceneBvhHeader header{};
    std::vector<SceneBvhBlas> blases;
    std::vector<SceneBvhNode> blas_nodes;
    std::vector<std::uint32_t> primitive_references;
    std::vector<SceneBvhNode> tlas_nodes;
    std::vector<SceneBvhInstanceReference> instance_references;
    std::vector<std::uint8_t> bytes;
};

[[nodiscard]] core::Error bvh_error(const core::StatusCode code, std::string message) {
    return core::Error{.code = code, .message = std::move(message)};
}

[[nodiscard]] core::Error cuda_error(const cudaError_t status, const char* operation,
                                     const std::size_t byte_count) {
    return bvh_error(xpu::cuda::cuda_memory_status_code(static_cast<std::int32_t>(status)),
                     std::string{"CUDA BVH "} + operation + " failed for " +
                         std::to_string(byte_count) + " bytes: " + cudaGetErrorName(status) + " (" +
                         cudaGetErrorString(status) + ").");
}

[[nodiscard]] constexpr float canonical_zero(const float value) noexcept {
    return value == 0.0F ? 0.0F : value;
}

[[nodiscard]] bool valid(const Bounds& bounds) noexcept {
    for (auto axis = std::size_t{0}; axis < 3U; ++axis) {
        if (!std::isfinite(bounds.minimum[axis]) || !std::isfinite(bounds.maximum[axis]) ||
            bounds.minimum[axis] > bounds.maximum[axis]) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] Bounds unite(const Bounds& left, const Bounds& right) noexcept {
    auto result = Bounds{};
    for (auto axis = std::size_t{0}; axis < 3U; ++axis) {
        result.minimum[axis] = canonical_zero(std::min(left.minimum[axis], right.minimum[axis]));
        result.maximum[axis] = canonical_zero(std::max(left.maximum[axis], right.maximum[axis]));
    }
    return result;
}

[[nodiscard]] bool same_bounds(const SceneBvhNode& node, const Bounds& bounds) noexcept {
    return node.minimum_x == bounds.minimum[0] && node.minimum_y == bounds.minimum[1] &&
           node.minimum_z == bounds.minimum[2] && node.maximum_x == bounds.maximum[0] &&
           node.maximum_y == bounds.maximum[1] && node.maximum_z == bounds.maximum[2];
}

[[nodiscard]] Bounds node_bounds(const SceneBvhNode& node) noexcept {
    return Bounds{
        .minimum = {node.minimum_x, node.minimum_y, node.minimum_z},
        .maximum = {node.maximum_x, node.maximum_y, node.maximum_z},
    };
}

[[nodiscard]] SceneBvhNode make_node(const Bounds& bounds) noexcept {
    return SceneBvhNode{
        .minimum_x = bounds.minimum[0],
        .minimum_y = bounds.minimum[1],
        .minimum_z = bounds.minimum[2],
        .kind = static_cast<std::uint32_t>(xpu::shared::SceneBvhNodeKind::internal),
        .maximum_x = bounds.maximum[0],
        .maximum_y = bounds.maximum[1],
        .maximum_z = bounds.maximum[2],
    };
}

[[nodiscard]] std::array<double, 3> centroid(const Bounds& bounds) noexcept {
    return {
        (static_cast<double>(bounds.minimum[0]) + static_cast<double>(bounds.maximum[0])) * 0.5,
        (static_cast<double>(bounds.minimum[1]) + static_cast<double>(bounds.maximum[1])) * 0.5,
        (static_cast<double>(bounds.minimum[2]) + static_cast<double>(bounds.maximum[2])) * 0.5,
    };
}

[[nodiscard]] std::uint64_t normalized_scene_hash(const std::span<const std::uint8_t> bytes) {
    auto hash = xpu::shared::SceneSoaFnv1aOffsetBasis;
    constexpr auto hash_begin = xpu::shared::SceneSoaContentHashOffset;
    constexpr auto hash_end = hash_begin + sizeof(std::uint64_t);
    for (auto index = std::size_t{0}; index < bytes.size(); ++index) {
        const auto value = index >= hash_begin && index < hash_end ? std::uint8_t{0} : bytes[index];
        hash ^= value;
        hash *= xpu::shared::SceneSoaFnv1aPrime;
    }
    return hash;
}

[[nodiscard]] std::uint64_t normalized_bvh_hash(const std::span<const std::uint8_t> bytes) {
    auto hash = xpu::shared::SceneBvhFnv1aOffsetBasis;
    constexpr auto hash_begin = xpu::shared::SceneBvhContentHashOffset;
    constexpr auto hash_end = hash_begin + sizeof(std::uint64_t);
    for (auto index = std::size_t{0}; index < bytes.size(); ++index) {
        const auto value = index >= hash_begin && index < hash_end ? std::uint8_t{0} : bytes[index];
        hash ^= value;
        hash *= xpu::shared::SceneBvhFnv1aPrime;
    }
    return hash;
}

[[nodiscard]] core::Result<DownloadedScene> download_scene(const CudaSceneSoA& scene) {
    if (!scene || scene.device_data() == nullptr || scene.device_ordinal() < 0 ||
        scene.size_bytes() < sizeof(SceneSoaHeader)) {
        return std::unexpected(
            bvh_error(core::StatusCode::invalid_argument,
                      "CUDA BVH construction requires an open serialized CUDA scene."));
    }

    int active_device = -1;
    auto status = cudaGetDevice(&active_device);
    if (status != cudaSuccess) {
        return std::unexpected(cuda_error(status, "device query", scene.size_bytes()));
    }
    if (active_device != scene.device_ordinal()) {
        return std::unexpected(bvh_error(
            core::StatusCode::invalid_argument,
            "CUDA BVH construction requires the serialized scene's device to be active."));
    }

    auto downloaded = DownloadedScene{};
    downloaded.bytes.resize(scene.size_bytes());
    status = cudaMemcpy(downloaded.bytes.data(), scene.device_data(), downloaded.bytes.size(),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        return std::unexpected(cuda_error(status, "scene download", downloaded.bytes.size()));
    }
    std::memcpy(&downloaded.header, downloaded.bytes.data(), sizeof(downloaded.header));
    if (std::memcmp(&downloaded.header, &scene.header(), sizeof(downloaded.header)) != 0) {
        return std::unexpected(bvh_error(
            core::StatusCode::invalid_argument,
            "CUDA BVH construction found a device scene header that differs from its owner."));
    }
    if (xpu::shared::validate_scene_soa_header(downloaded.header) !=
            xpu::shared::SceneSoaHeaderValidationStatus::valid ||
        downloaded.header.total_size_bytes != downloaded.bytes.size()) {
        return std::unexpected(
            bvh_error(core::StatusCode::invalid_argument,
                      "CUDA BVH construction rejected an invalid serialized scene layout."));
    }
    if (normalized_scene_hash(downloaded.bytes) != downloaded.header.content_hash) {
        return std::unexpected(
            bvh_error(core::StatusCode::invalid_argument,
                      "CUDA BVH construction rejected a serialized scene with a mismatched hash."));
    }
    return downloaded;
}

template <typename Value>
[[nodiscard]] Value scene_value(const DownloadedScene& scene, const std::uint32_t column,
                                const std::uint64_t index) noexcept {
    static_assert(std::is_trivially_copyable_v<Value>);
    const auto& descriptor = scene.header.columns[column];
    auto value = Value{};
    const auto offset = static_cast<std::size_t>(descriptor.offset_bytes) +
                        static_cast<std::size_t>(index) * sizeof(Value);
    std::memcpy(&value, scene.bytes.data() + offset, sizeof(Value));
    return value;
}

[[nodiscard]] core::Status require_u32_count(const std::uint64_t count, const char* const message) {
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(bvh_error(core::StatusCode::resource_exhausted, message));
    }
    return {};
}

[[nodiscard]] core::Result<std::vector<SourceGeometry>>
read_geometries(const DownloadedScene& source) {
    if (auto status = require_u32_count(source.header.geometry_count,
                                        "CUDA BVH geometry count exceeds its 32-bit index domain.");
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    if (auto status = require_u32_count(
            source.header.triangle_count,
            "CUDA BVH primitive-reference count exceeds its 32-bit index domain.");
        !status) {
        return std::unexpected(std::move(status.error()));
    }

    auto geometries = std::vector<SourceGeometry>{};
    geometries.reserve(static_cast<std::size_t>(source.header.geometry_count));
    auto previous_id = std::uint32_t{};
    auto have_previous = false;
    auto expected_vertex_offset = std::uint64_t{0};
    auto expected_triangle_offset = std::uint64_t{0};
    for (auto geometry_index = std::uint64_t{0}; geometry_index < source.header.geometry_count;
         ++geometry_index) {
        const auto geometry = SourceGeometry{
            .id = scene_value<std::uint32_t>(source, scene_column::geometry_id, geometry_index),
            .vertex_offset = scene_value<std::uint64_t>(
                source, scene_column::geometry_vertex_offset, geometry_index),
            .vertex_count = scene_value<std::uint64_t>(source, scene_column::geometry_vertex_count,
                                                       geometry_index),
            .triangle_offset = scene_value<std::uint64_t>(
                source, scene_column::geometry_triangle_offset, geometry_index),
            .triangle_count = scene_value<std::uint64_t>(
                source, scene_column::geometry_triangle_count, geometry_index),
        };
        if (have_previous && geometry.id <= previous_id) {
            return std::unexpected(
                bvh_error(core::StatusCode::invalid_argument,
                          "CUDA BVH construction requires strictly ordered geometry identifiers."));
        }
        if (geometry.vertex_count == 0U || geometry.triangle_count == 0U ||
            geometry.vertex_offset != expected_vertex_offset ||
            geometry.triangle_offset != expected_triangle_offset ||
            geometry.vertex_offset > source.header.vertex_count ||
            geometry.vertex_count > source.header.vertex_count - geometry.vertex_offset ||
            geometry.triangle_offset > source.header.triangle_count ||
            geometry.triangle_count > source.header.triangle_count - geometry.triangle_offset) {
            return std::unexpected(
                bvh_error(core::StatusCode::invalid_argument,
                          "CUDA BVH construction rejected an invalid geometry range."));
        }
        previous_id = geometry.id;
        have_previous = true;
        expected_vertex_offset += geometry.vertex_count;
        expected_triangle_offset += geometry.triangle_count;
        geometries.push_back(geometry);
    }
    if (expected_vertex_offset != source.header.vertex_count ||
        expected_triangle_offset != source.header.triangle_count) {
        return std::unexpected(bvh_error(
            core::StatusCode::invalid_argument,
            "CUDA BVH construction requires geometry ranges to cover the serialized meshes."));
    }
    return geometries;
}

[[nodiscard]] core::Result<std::array<std::uint32_t, 3>>
triangle_indices(const DownloadedScene& source, const SourceGeometry& geometry,
                 const std::uint32_t primitive) {
    const auto global_triangle = geometry.triangle_offset + primitive;
    auto indices = std::array<std::uint32_t, 3>{};
    for (auto corner = std::uint32_t{0}; corner < 3U; ++corner) {
        indices[corner] = scene_value<std::uint32_t>(
            source, scene_column::triangle_vertex_0 + corner, global_triangle);
        if (indices[corner] >= geometry.vertex_count) {
            return std::unexpected(
                bvh_error(core::StatusCode::invalid_argument,
                          "CUDA BVH construction rejected a triangle index outside its geometry."));
        }
    }
    return indices;
}

[[nodiscard]] renderer::Point3 source_position(const DownloadedScene& source,
                                               const SourceGeometry& geometry,
                                               const std::uint32_t vertex) noexcept {
    const auto global_vertex = geometry.vertex_offset + vertex;
    return renderer::Point3{
        .x = scene_value<float>(source, scene_column::position_x, global_vertex),
        .y = scene_value<float>(source, scene_column::position_y, global_vertex),
        .z = scene_value<float>(source, scene_column::position_z, global_vertex),
    };
}

[[nodiscard]] core::Result<Bounds> primitive_bounds(const DownloadedScene& source,
                                                    const SourceGeometry& geometry,
                                                    const std::uint32_t primitive) {
    auto indices = triangle_indices(source, geometry, primitive);
    if (!indices) {
        return std::unexpected(std::move(indices.error()));
    }
    const auto first = source_position(source, geometry, (*indices)[0]);
    auto bounds = Bounds{
        .minimum = {canonical_zero(first.x), canonical_zero(first.y), canonical_zero(first.z)},
        .maximum = {canonical_zero(first.x), canonical_zero(first.y), canonical_zero(first.z)},
    };
    for (auto corner = std::size_t{1}; corner < indices->size(); ++corner) {
        const auto point = source_position(source, geometry, (*indices)[corner]);
        const auto components = std::array{point.x, point.y, point.z};
        for (auto axis = std::size_t{0}; axis < 3U; ++axis) {
            if (!std::isfinite(components[axis])) {
                return std::unexpected(
                    bvh_error(core::StatusCode::invalid_argument,
                              "CUDA BVH construction requires finite triangle positions."));
            }
            bounds.minimum[axis] = canonical_zero(std::min(bounds.minimum[axis], components[axis]));
            bounds.maximum[axis] = canonical_zero(std::max(bounds.maximum[axis], components[axis]));
        }
    }
    if (!valid(bounds)) {
        return std::unexpected(bvh_error(core::StatusCode::invalid_argument,
                                         "CUDA BVH produced invalid primitive bounds."));
    }
    return bounds;
}

template <typename EmitReference>
[[nodiscard]] core::Result<std::uint32_t>
build_node(std::vector<BuildItem>& items, const std::size_t begin, const std::size_t end,
           std::vector<SceneBvhNode>& nodes, EmitReference& emit_reference) {
    if (begin >= end || nodes.size() >= xpu::shared::SceneBvhInvalidIndex) {
        return std::unexpected(
            bvh_error(core::StatusCode::resource_exhausted,
                      "CUDA BVH node construction exceeded its 32-bit index domain."));
    }

    const auto node_index = static_cast<std::uint32_t>(nodes.size());
    nodes.emplace_back();
    if (end - begin == xpu::shared::SceneBvhLeafCapacity) {
        auto reference_offset = emit_reference(items[begin]);
        if (!reference_offset) {
            return std::unexpected(std::move(reference_offset.error()));
        }
        auto node = make_node(items[begin].bounds);
        node.kind = static_cast<std::uint32_t>(xpu::shared::SceneBvhNodeKind::leaf);
        node.reference_offset = *reference_offset;
        node.reference_count = 1U;
        nodes[node_index] = node;
        return node_index;
    }

    auto centroid_minimum = items[begin].centroid;
    auto centroid_maximum = items[begin].centroid;
    for (auto index = begin + 1U; index < end; ++index) {
        for (auto axis = std::size_t{0}; axis < 3U; ++axis) {
            centroid_minimum[axis] = std::min(centroid_minimum[axis], items[index].centroid[axis]);
            centroid_maximum[axis] = std::max(centroid_maximum[axis], items[index].centroid[axis]);
        }
    }
    auto split_axis = std::uint32_t{0};
    auto largest_extent = centroid_maximum[0] - centroid_minimum[0];
    for (auto axis = std::uint32_t{1}; axis < 3U; ++axis) {
        const auto extent = centroid_maximum[axis] - centroid_minimum[axis];
        if (extent > largest_extent) {
            largest_extent = extent;
            split_axis = axis;
        }
    }

    std::sort(items.begin() + static_cast<std::ptrdiff_t>(begin),
              items.begin() + static_cast<std::ptrdiff_t>(end),
              [split_axis](const BuildItem& left, const BuildItem& right) {
                  if (left.centroid[split_axis] < right.centroid[split_axis]) {
                      return true;
                  }
                  if (right.centroid[split_axis] < left.centroid[split_axis]) {
                      return false;
                  }
                  return left.stable_key < right.stable_key;
              });

    const auto middle = begin + (end - begin) / 2U;
    auto first_child = build_node(items, begin, middle, nodes, emit_reference);
    if (!first_child) {
        return std::unexpected(std::move(first_child.error()));
    }
    auto second_child = build_node(items, middle, end, nodes, emit_reference);
    if (!second_child) {
        return std::unexpected(std::move(second_child.error()));
    }

    const auto bounds = unite(node_bounds(nodes[*first_child]), node_bounds(nodes[*second_child]));
    auto node = make_node(bounds);
    node.split_axis = split_axis;
    node.first_child = *first_child;
    node.second_child = *second_child;
    nodes[node_index] = node;
    return node_index;
}

[[nodiscard]] core::Result<std::uint32_t>
checked_expected_node_count(const std::uint64_t reference_count) {
    if (reference_count == 0U) {
        return std::uint32_t{0};
    }
    if (reference_count >
        (static_cast<std::uint64_t>(xpu::shared::SceneBvhInvalidIndex) + 1U) / 2U) {
        return std::unexpected(
            bvh_error(core::StatusCode::resource_exhausted,
                      "CUDA BVH binary topology exceeds its 32-bit node-index domain."));
    }
    const auto node_count = reference_count * 2U - 1U;
    if (node_count > xpu::shared::SceneBvhInvalidIndex) {
        return std::unexpected(
            bvh_error(core::StatusCode::resource_exhausted,
                      "CUDA BVH binary topology exceeds its 32-bit node-count domain."));
    }
    return static_cast<std::uint32_t>(node_count);
}

[[nodiscard]] core::Status reserve_topology_storage(BuiltBvh& built, const DownloadedScene& source,
                                                    const std::vector<SourceGeometry>& geometries) {
    auto blas_node_count = std::uint64_t{0};
    for (const auto& geometry : geometries) {
        auto node_count = checked_expected_node_count(geometry.triangle_count);
        if (!node_count) {
            return std::unexpected(std::move(node_count.error()));
        }
        if (blas_node_count > xpu::shared::SceneBvhInvalidIndex - *node_count) {
            return std::unexpected(
                bvh_error(core::StatusCode::resource_exhausted,
                          "CUDA BLAS nodes exceed their aggregate 32-bit index domain."));
        }
        blas_node_count += *node_count;
    }
    auto tlas_node_count = checked_expected_node_count(source.header.instance_count);
    if (!tlas_node_count) {
        return std::unexpected(std::move(tlas_node_count.error()));
    }

    built.blases.reserve(geometries.size());
    built.blas_nodes.reserve(static_cast<std::size_t>(blas_node_count));
    built.primitive_references.reserve(static_cast<std::size_t>(source.header.triangle_count));
    built.tlas_nodes.reserve(*tlas_node_count);
    built.instance_references.reserve(static_cast<std::size_t>(source.header.instance_count));
    return {};
}

[[nodiscard]] core::Status build_blases(const DownloadedScene& source,
                                        const std::vector<SourceGeometry>& geometries,
                                        BuiltBvh& built) {
    for (const auto& geometry : geometries) {
        auto items = std::vector<BuildItem>{};
        items.reserve(static_cast<std::size_t>(geometry.triangle_count));
        for (auto primitive = std::uint32_t{0}; primitive < geometry.triangle_count; ++primitive) {
            auto bounds = primitive_bounds(source, geometry, primitive);
            if (!bounds) {
                return std::unexpected(std::move(bounds.error()));
            }
            items.push_back(BuildItem{
                .bounds = *bounds,
                .centroid = centroid(*bounds),
                .stable_key = primitive,
            });
        }

        const auto node_offset = static_cast<std::uint32_t>(built.blas_nodes.size());
        const auto reference_offset = static_cast<std::uint32_t>(built.primitive_references.size());
        auto emit = [&](const BuildItem& item) -> core::Result<std::uint32_t> {
            if (built.primitive_references.size() >= xpu::shared::SceneBvhInvalidIndex) {
                return std::unexpected(
                    bvh_error(core::StatusCode::resource_exhausted,
                              "CUDA BLAS primitive references exceed their 32-bit index domain."));
            }
            const auto offset = static_cast<std::uint32_t>(built.primitive_references.size());
            built.primitive_references.push_back(item.stable_key);
            return offset;
        };
        auto root = build_node(items, 0U, items.size(), built.blas_nodes, emit);
        if (!root) {
            return std::unexpected(std::move(root.error()));
        }
        built.blases.push_back(SceneBvhBlas{
            .geometry_id = geometry.id,
            .root_node = *root,
            .node_offset = node_offset,
            .node_count = static_cast<std::uint32_t>(built.blas_nodes.size() - node_offset),
            .primitive_reference_offset = reference_offset,
            .primitive_reference_count = static_cast<std::uint32_t>(geometry.triangle_count),
        });
    }
    return {};
}

[[nodiscard]] std::optional<std::uint32_t>
find_geometry(const std::vector<SourceGeometry>& geometries, const std::uint32_t id) noexcept {
    const auto found =
        std::lower_bound(geometries.begin(), geometries.end(), id,
                         [](const SourceGeometry& geometry, const std::uint32_t value) {
                             return geometry.id < value;
                         });
    if (found == geometries.end() || found->id != id) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(found - geometries.begin());
}

[[nodiscard]] std::array<float, 16> instance_world_matrix(const DownloadedScene& source,
                                                          const std::uint32_t instance_index) {
    auto matrix = std::array<float, 16>{};
    for (auto element = std::uint32_t{0}; element < matrix.size(); ++element) {
        matrix[element] = scene_value<float>(
            source, scene_column::instance_local_to_world + element, instance_index);
    }
    return matrix;
}

[[nodiscard]] renderer::Point3 transform_point(const std::array<float, 16>& matrix,
                                               const renderer::Point3 point) noexcept {
    return renderer::Point3{
        .x = matrix[0] * point.x + matrix[1] * point.y + matrix[2] * point.z + matrix[3],
        .y = matrix[4] * point.x + matrix[5] * point.y + matrix[6] * point.z + matrix[7],
        .z = matrix[8] * point.x + matrix[9] * point.y + matrix[10] * point.z + matrix[11],
    };
}

[[nodiscard]] core::Result<Bounds> transform_point_bounds(const std::array<float, 16>& matrix,
                                                          const renderer::Point3 point) {
    constexpr auto scalar_epsilon = static_cast<double>(std::numeric_limits<float>::epsilon());
    constexpr auto gamma7 = (7.0 * scalar_epsilon) / (1.0 - 7.0 * scalar_epsilon);
    constexpr auto underflow_allowance =
        7.0 * static_cast<double>(std::numeric_limits<float>::denorm_min());
    constexpr auto maximum = static_cast<double>(std::numeric_limits<float>::max());
    const auto source = std::array{point.x, point.y, point.z};
    auto result = Bounds{};

    for (auto row = std::size_t{0}; row < 3U; ++row) {
        auto exact = static_cast<double>(matrix[row * 4U + 3U]);
        auto magnitude = std::abs(exact);
        for (auto column = std::size_t{0}; column < 3U; ++column) {
            const auto product = static_cast<double>(matrix[row * 4U + column]) *
                                 static_cast<double>(source[column]);
            exact += product;
            magnitude += std::abs(product);
        }
        const auto error = std::fma(gamma7, magnitude, underflow_allowance);
        const auto lower = exact - error;
        const auto upper = exact + error;
        if (!std::isfinite(exact) || !std::isfinite(magnitude) || !std::isfinite(error) ||
            !std::isfinite(lower) || !std::isfinite(upper) || lower < -maximum || upper > maximum) {
            return std::unexpected(
                bvh_error(core::StatusCode::invalid_argument,
                          "CUDA TLAS conservative point bounds are not finitely representable."));
        }

        auto rounded_lower = static_cast<float>(lower);
        auto rounded_upper = static_cast<float>(upper);
        if (static_cast<double>(rounded_lower) > lower) {
            rounded_lower = std::nextafter(rounded_lower, -std::numeric_limits<float>::infinity());
        }
        if (static_cast<double>(rounded_upper) < upper) {
            rounded_upper = std::nextafter(rounded_upper, std::numeric_limits<float>::infinity());
        }
        result.minimum[row] = canonical_zero(rounded_lower);
        result.maximum[row] = canonical_zero(rounded_upper);
    }
    if (!valid(result)) {
        return std::unexpected(
            bvh_error(core::StatusCode::invalid_argument,
                      "CUDA TLAS conservative point bounds are not finitely representable."));
    }
    return result;
}

[[nodiscard]] core::Result<Bounds> instance_bounds(const DownloadedScene& source,
                                                   const SourceGeometry& geometry,
                                                   const std::array<float, 16>& matrix) {
    for (const auto value : matrix) {
        if (!std::isfinite(value)) {
            return std::unexpected(
                bvh_error(core::StatusCode::invalid_argument,
                          "CUDA TLAS construction requires a finite world transform."));
        }
    }
    if (matrix[12] != 0.0F || matrix[13] != 0.0F || matrix[14] != 0.0F || matrix[15] != 1.0F) {
        return std::unexpected(
            bvh_error(core::StatusCode::invalid_argument,
                      "CUDA TLAS construction requires an affine world transform."));
    }

    auto have_bounds = false;
    auto result = Bounds{};
    for (auto primitive = std::uint32_t{0}; primitive < geometry.triangle_count; ++primitive) {
        auto indices = triangle_indices(source, geometry, primitive);
        if (!indices) {
            return std::unexpected(std::move(indices.error()));
        }
        auto world = std::array<renderer::Point3, 3>{};
        for (auto corner = std::size_t{0}; corner < world.size(); ++corner) {
            const auto object_point = source_position(source, geometry, (*indices)[corner]);
            world[corner] = transform_point(matrix, object_point);
            if (!std::isfinite(world[corner].x) || !std::isfinite(world[corner].y) ||
                !std::isfinite(world[corner].z)) {
                return std::unexpected(bvh_error(
                    core::StatusCode::invalid_argument,
                    "CUDA TLAS construction produced a non-finite world-space triangle."));
            }
            auto point_bounds = transform_point_bounds(matrix, object_point);
            if (!point_bounds) {
                return std::unexpected(std::move(point_bounds.error()));
            }
            if (!have_bounds) {
                result = *point_bounds;
                have_bounds = true;
            } else {
                result = unite(result, *point_bounds);
            }
        }
        if (!renderer::Triangle::create(world[0], world[1], world[2])) {
            return std::unexpected(bvh_error(
                core::StatusCode::invalid_argument,
                "CUDA TLAS construction produced an unrepresentable world-space triangle."));
        }
    }
    if (!have_bounds || !valid(result)) {
        return std::unexpected(bvh_error(core::StatusCode::internal_error,
                                         "CUDA TLAS failed to derive instance bounds."));
    }

    return result;
}

[[nodiscard]] core::Status build_tlas(const DownloadedScene& source,
                                      const std::vector<SourceGeometry>& geometries,
                                      BuiltBvh& built) {
    if (auto status =
            require_u32_count(source.header.instance_count,
                              "CUDA TLAS instance count exceeds its 32-bit index domain.");
        !status) {
        return status;
    }
    if (source.header.instance_count == 0U) {
        built.header.tlas_root_node = xpu::shared::SceneBvhInvalidIndex;
        return {};
    }

    auto items = std::vector<BuildItem>{};
    items.reserve(static_cast<std::size_t>(source.header.instance_count));
    auto previous_id = std::uint32_t{};
    auto have_previous = false;
    for (auto instance_index = std::uint32_t{0}; instance_index < source.header.instance_count;
         ++instance_index) {
        const auto instance_id =
            scene_value<std::uint32_t>(source, scene_column::instance_id, instance_index);
        if (have_previous && instance_id <= previous_id) {
            return std::unexpected(bvh_error(
                core::StatusCode::invalid_argument,
                "CUDA TLAS construction requires strictly ordered instance identifiers."));
        }
        const auto geometry_id =
            scene_value<std::uint32_t>(source, scene_column::instance_geometry_id, instance_index);
        const auto geometry_index = find_geometry(geometries, geometry_id);
        if (!geometry_index) {
            return std::unexpected(
                bvh_error(core::StatusCode::invalid_argument,
                          "CUDA TLAS construction found an instance with no matching BLAS."));
        }
        auto bounds = instance_bounds(source, geometries[*geometry_index],
                                      instance_world_matrix(source, instance_index));
        if (!bounds) {
            return std::unexpected(std::move(bounds.error()));
        }
        const auto reference = SceneBvhInstanceReference{
            .scene_instance_index = instance_index,
            .instance_id = instance_id,
            .scene_geometry_index = *geometry_index,
            .geometry_id = geometry_id,
            .blas_index = *geometry_index,
            .visibility_mask = scene_value<std::uint32_t>(
                source, scene_column::instance_visibility_mask, instance_index),
        };
        items.push_back(BuildItem{
            .bounds = *bounds,
            .centroid = centroid(*bounds),
            .stable_key = instance_id,
            .instance_reference = reference,
        });
        previous_id = instance_id;
        have_previous = true;
    }

    auto emit = [&](const BuildItem& item) -> core::Result<std::uint32_t> {
        if (built.instance_references.size() >= xpu::shared::SceneBvhInvalidIndex) {
            return std::unexpected(
                bvh_error(core::StatusCode::resource_exhausted,
                          "CUDA TLAS instance references exceed their 32-bit index domain."));
        }
        const auto offset = static_cast<std::uint32_t>(built.instance_references.size());
        built.instance_references.push_back(item.instance_reference);
        return offset;
    };
    auto root = build_node(items, 0U, items.size(), built.tlas_nodes, emit);
    if (!root) {
        return std::unexpected(std::move(root.error()));
    }
    built.header.tlas_root_node = *root;
    return {};
}

[[nodiscard]] core::Status validate_tree(const std::span<const SceneBvhNode> nodes,
                                         const std::uint32_t node_offset,
                                         const std::uint32_t node_count, const std::uint32_t root,
                                         const std::uint32_t reference_offset,
                                         const std::uint32_t reference_count) {
    if (reference_count == 0U || node_count != reference_count * 2U - 1U || root != node_offset ||
        node_offset > nodes.size() || node_count > nodes.size() - node_offset) {
        return std::unexpected(bvh_error(core::StatusCode::internal_error,
                                         "CUDA BVH produced inconsistent tree ranges."));
    }
    auto seen_nodes = std::vector<std::uint8_t>(node_count, std::uint8_t{0});
    auto seen_references = std::vector<std::uint8_t>(reference_count, std::uint8_t{0});
    auto stack = std::vector<std::uint32_t>{root};
    while (!stack.empty()) {
        const auto index = stack.back();
        stack.pop_back();
        if (index < node_offset || index >= node_offset + node_count ||
            seen_nodes[index - node_offset] != 0U) {
            return std::unexpected(bvh_error(core::StatusCode::internal_error,
                                             "CUDA BVH topology is cyclic or out of range."));
        }
        seen_nodes[index - node_offset] = 1U;
        const auto& node = nodes[index];
        const auto bounds = node_bounds(node);
        if (!valid(bounds)) {
            return std::unexpected(bvh_error(core::StatusCode::internal_error,
                                             "CUDA BVH produced invalid node bounds."));
        }
        if (node.kind == static_cast<std::uint32_t>(xpu::shared::SceneBvhNodeKind::leaf)) {
            if (node.split_axis != xpu::shared::SceneBvhInvalidIndex ||
                node.first_child != xpu::shared::SceneBvhInvalidIndex ||
                node.second_child != xpu::shared::SceneBvhInvalidIndex ||
                node.reference_count != 1U || node.reference_offset < reference_offset ||
                node.reference_offset >= reference_offset + reference_count) {
                return std::unexpected(bvh_error(core::StatusCode::internal_error,
                                                 "CUDA BVH produced an invalid leaf node."));
            }
            auto& seen = seen_references[node.reference_offset - reference_offset];
            if (seen != 0U) {
                return std::unexpected(
                    bvh_error(core::StatusCode::internal_error,
                              "CUDA BVH topology references one payload more than once."));
            }
            seen = 1U;
            continue;
        }
        if (node.kind != static_cast<std::uint32_t>(xpu::shared::SceneBvhNodeKind::internal) ||
            node.split_axis > 2U || node.reference_offset != 0U || node.reference_count != 0U ||
            node.first_child == node.second_child || node.first_child < node_offset ||
            node.first_child >= node_offset + node_count || node.second_child < node_offset ||
            node.second_child >= node_offset + node_count) {
            return std::unexpected(bvh_error(core::StatusCode::internal_error,
                                             "CUDA BVH produced an invalid internal node."));
        }
        const auto union_bounds =
            unite(node_bounds(nodes[node.first_child]), node_bounds(nodes[node.second_child]));
        if (!same_bounds(node, union_bounds)) {
            return std::unexpected(
                bvh_error(core::StatusCode::internal_error,
                          "CUDA BVH internal bounds do not equal the union of their children."));
        }
        stack.push_back(node.second_child);
        stack.push_back(node.first_child);
    }
    if (std::ranges::find(seen_nodes, std::uint8_t{0}) != seen_nodes.end() ||
        std::ranges::find(seen_references, std::uint8_t{0}) != seen_references.end()) {
        return std::unexpected(bvh_error(core::StatusCode::internal_error,
                                         "CUDA BVH topology contains unreachable data."));
    }
    return {};
}

[[nodiscard]] core::Status validate_built_topology(const BuiltBvh& built) {
    auto expected_node_offset = std::uint32_t{0};
    auto expected_reference_offset = std::uint32_t{0};
    for (const auto& blas : built.blases) {
        if (blas.node_offset != expected_node_offset ||
            blas.primitive_reference_offset != expected_reference_offset ||
            blas.reserved != std::array<std::uint32_t, 2>{}) {
            return std::unexpected(bvh_error(core::StatusCode::internal_error,
                                             "CUDA BLAS descriptors are not canonical."));
        }
        if (auto status =
                validate_tree(built.blas_nodes, blas.node_offset, blas.node_count, blas.root_node,
                              blas.primitive_reference_offset, blas.primitive_reference_count);
            !status) {
            return status;
        }
        auto primitive_seen = std::vector<std::uint8_t>(blas.primitive_reference_count, 0U);
        for (auto index = std::uint32_t{0}; index < blas.primitive_reference_count; ++index) {
            const auto primitive =
                built.primitive_references[blas.primitive_reference_offset + index];
            if (primitive >= primitive_seen.size() || primitive_seen[primitive] != 0U) {
                return std::unexpected(
                    bvh_error(core::StatusCode::internal_error,
                              "CUDA BLAS primitive references are not a complete permutation."));
            }
            primitive_seen[primitive] = 1U;
        }
        expected_node_offset += blas.node_count;
        expected_reference_offset += blas.primitive_reference_count;
    }
    if (expected_node_offset != built.blas_nodes.size() ||
        expected_reference_offset != built.primitive_references.size()) {
        return std::unexpected(bvh_error(core::StatusCode::internal_error,
                                         "CUDA BLAS descriptors do not cover their arrays."));
    }

    if (built.instance_references.empty()) {
        if (!built.tlas_nodes.empty() ||
            built.header.tlas_root_node != xpu::shared::SceneBvhInvalidIndex) {
            return std::unexpected(bvh_error(core::StatusCode::internal_error,
                                             "CUDA TLAS empty-scene state is inconsistent."));
        }
        return {};
    }
    if (auto status =
            validate_tree(built.tlas_nodes, 0U, static_cast<std::uint32_t>(built.tlas_nodes.size()),
                          built.header.tlas_root_node, 0U,
                          static_cast<std::uint32_t>(built.instance_references.size()));
        !status) {
        return status;
    }
    auto instance_seen = std::vector<std::uint8_t>(built.instance_references.size(), 0U);
    for (const auto& reference : built.instance_references) {
        if (reference.scene_instance_index >= instance_seen.size() ||
            instance_seen[reference.scene_instance_index] != 0U ||
            reference.blas_index >= built.blases.size() ||
            reference.scene_geometry_index != reference.blas_index ||
            reference.geometry_id != built.blases[reference.blas_index].geometry_id ||
            reference.reserved != std::array<std::uint32_t, 2>{}) {
            return std::unexpected(
                bvh_error(core::StatusCode::internal_error,
                          "CUDA TLAS instance references are not a complete canonical mapping."));
        }
        instance_seen[reference.scene_instance_index] = 1U;
    }
    return {};
}

[[nodiscard]] bool add_overflows(const std::uint64_t left, const std::uint64_t right) noexcept {
    return left > std::numeric_limits<std::uint64_t>::max() - right;
}

[[nodiscard]] core::Status prepare_header_layout(SceneBvhHeader& header,
                                                 const std::size_t host_max_size,
                                                 const xpu::cuda::DeviceMemoryBudget budget) {
    header.arrays = {};
    header.total_size_bytes = 0U;
    header.content_hash = 0U;
    auto cursor = xpu::shared::scene_bvh_align_up(sizeof(SceneBvhHeader));
    for (auto array = std::uint32_t{0}; array < bvh_array::count; ++array) {
        auto& descriptor = header.arrays[array];
        descriptor.element_count = xpu::shared::scene_bvh_array_count(header, array);
        descriptor.element_size = xpu::shared::scene_bvh_array_element_size(array);
        if (descriptor.element_count == 0U) {
            continue;
        }
        if (cursor > std::numeric_limits<std::uint64_t>::max() -
                         (xpu::shared::SceneBvhArrayAlignment - 1U)) {
            return std::unexpected(bvh_error(core::StatusCode::resource_exhausted,
                                             "CUDA BVH array alignment overflowed."));
        }
        cursor = xpu::shared::scene_bvh_align_up(cursor);
        if (descriptor.element_count >
            std::numeric_limits<std::uint64_t>::max() / descriptor.element_size) {
            return std::unexpected(bvh_error(core::StatusCode::resource_exhausted,
                                             "CUDA BVH array byte count overflowed."));
        }
        const auto byte_count = descriptor.element_count * descriptor.element_size;
        if (add_overflows(cursor, byte_count)) {
            return std::unexpected(bvh_error(core::StatusCode::resource_exhausted,
                                             "CUDA BVH aggregate byte count overflowed."));
        }
        descriptor.offset_bytes = cursor;
        cursor += byte_count;
    }
    header.total_size_bytes = cursor;
    if (cursor > budget.maximum_bytes) {
        return std::unexpected(
            bvh_error(core::StatusCode::resource_exhausted,
                      "CUDA BVH exceeds its explicit device-memory budget before allocation."));
    }
    if (cursor > std::numeric_limits<std::size_t>::max() ||
        cursor > static_cast<std::uint64_t>(host_max_size) ||
        cursor > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        return std::unexpected(bvh_error(core::StatusCode::resource_exhausted,
                                         "CUDA BVH exceeds the addressable host range."));
    }
    if (xpu::shared::validate_scene_bvh_header(header) !=
        xpu::shared::SceneBvhHeaderValidationStatus::valid) {
        return std::unexpected(bvh_error(core::StatusCode::internal_error,
                                         "CUDA BVH layout violated its frozen schema."));
    }
    return {};
}

[[nodiscard]] core::Status preflight_layout(const SceneSoaHeader& source,
                                            const xpu::cuda::DeviceMemoryBudget budget) {
    if (xpu::shared::validate_scene_soa_header(source) !=
        xpu::shared::SceneSoaHeaderValidationStatus::valid) {
        return std::unexpected(
            bvh_error(core::StatusCode::invalid_argument,
                      "CUDA BVH preflight requires a valid serialized scene header."));
    }
    if (auto status = require_u32_count(source.geometry_count,
                                        "CUDA BVH geometry count exceeds its 32-bit index domain.");
        !status) {
        return status;
    }
    if (auto status = require_u32_count(
            source.triangle_count,
            "CUDA BVH primitive-reference count exceeds its 32-bit index domain.");
        !status) {
        return status;
    }
    if (auto status = require_u32_count(
            source.instance_count, "CUDA TLAS instance count exceeds its 32-bit index domain.");
        !status) {
        return status;
    }
    if (source.triangle_count < source.geometry_count) {
        return std::unexpected(
            bvh_error(core::StatusCode::invalid_argument,
                      "CUDA BVH preflight requires at least one triangle per geometry."));
    }

    const auto blas_node_count = source.triangle_count == 0U
                                     ? std::uint64_t{0}
                                     : source.triangle_count * 2U - source.geometry_count;
    if (blas_node_count > xpu::shared::SceneBvhInvalidIndex) {
        return std::unexpected(
            bvh_error(core::StatusCode::resource_exhausted,
                      "CUDA BLAS nodes exceed their aggregate 32-bit index domain."));
    }
    auto tlas_node_count = checked_expected_node_count(source.instance_count);
    if (!tlas_node_count) {
        return std::unexpected(std::move(tlas_node_count.error()));
    }

    auto planned = SceneBvhHeader{};
    planned.magic = xpu::shared::SceneBvhMagic;
    planned.abi_major = xpu::shared::SceneBvhAbiMajor;
    planned.abi_minor = xpu::shared::SceneBvhAbiMinor;
    planned.header_size = sizeof(SceneBvhHeader);
    planned.array_count = bvh_array::count;
    planned.hash_algorithm = xpu::shared::SceneBvhHashAlgorithmFnv1a64;
    planned.source_scene_hash = source.content_hash;
    planned.blas_count = source.geometry_count;
    planned.blas_node_count = blas_node_count;
    planned.primitive_reference_count = source.triangle_count;
    planned.tlas_node_count = *tlas_node_count;
    planned.instance_reference_count = source.instance_count;
    planned.tlas_root_node = source.instance_count == 0U ? xpu::shared::SceneBvhInvalidIndex : 0U;
    planned.leaf_capacity = xpu::shared::SceneBvhLeafCapacity;
    return prepare_header_layout(planned, std::vector<std::uint8_t>{}.max_size(), budget);
}

[[nodiscard]] core::Status prepare_layout(BuiltBvh& built,
                                          const xpu::cuda::DeviceMemoryBudget budget) {
    auto& header = built.header;
    header.magic = xpu::shared::SceneBvhMagic;
    header.abi_major = xpu::shared::SceneBvhAbiMajor;
    header.abi_minor = xpu::shared::SceneBvhAbiMinor;
    header.header_size = sizeof(SceneBvhHeader);
    header.array_count = bvh_array::count;
    header.hash_algorithm = xpu::shared::SceneBvhHashAlgorithmFnv1a64;
    header.leaf_capacity = xpu::shared::SceneBvhLeafCapacity;
    header.blas_count = built.blases.size();
    header.blas_node_count = built.blas_nodes.size();
    header.primitive_reference_count = built.primitive_references.size();
    header.tlas_node_count = built.tlas_nodes.size();
    header.instance_reference_count = built.instance_references.size();
    return prepare_header_layout(header, built.bytes.max_size(), budget);
}

template <typename Value>
[[nodiscard]] core::Status append_array(BuiltBvh& built, const std::uint32_t array,
                                        const std::span<const Value> values) {
    static_assert(std::is_trivially_copyable_v<Value>);
    const auto& descriptor = built.header.arrays[array];
    if (descriptor.element_count != values.size() || descriptor.element_size != sizeof(Value)) {
        return std::unexpected(bvh_error(core::StatusCode::internal_error,
                                         "CUDA BVH array disagrees with its layout."));
    }
    if (values.empty()) {
        return {};
    }
    const auto offset = static_cast<std::size_t>(descriptor.offset_bytes);
    const auto byte_count = values.size_bytes();
    if (offset > built.bytes.size() || byte_count > built.bytes.size() - offset) {
        return std::unexpected(bvh_error(core::StatusCode::internal_error,
                                         "CUDA BVH array exceeds its serialized blob."));
    }
    std::memcpy(built.bytes.data() + offset, values.data(), byte_count);
    return {};
}

[[nodiscard]] core::Status serialize(BuiltBvh& built, const xpu::cuda::DeviceMemoryBudget budget) {
    if (auto status = prepare_layout(built, budget); !status) {
        return status;
    }
    built.bytes.assign(static_cast<std::size_t>(built.header.total_size_bytes), std::uint8_t{0});
    std::memcpy(built.bytes.data(), &built.header, sizeof(built.header));
    if (auto status =
            append_array(built, bvh_array::blas, std::span<const SceneBvhBlas>{built.blases});
        !status) {
        return status;
    }
    if (auto status = append_array(built, bvh_array::blas_node,
                                   std::span<const SceneBvhNode>{built.blas_nodes});
        !status) {
        return status;
    }
    if (auto status = append_array(built, bvh_array::primitive_reference,
                                   std::span<const std::uint32_t>{built.primitive_references});
        !status) {
        return status;
    }
    if (auto status = append_array(built, bvh_array::tlas_node,
                                   std::span<const SceneBvhNode>{built.tlas_nodes});
        !status) {
        return status;
    }
    if (auto status =
            append_array(built, bvh_array::instance_reference,
                         std::span<const SceneBvhInstanceReference>{built.instance_references});
        !status) {
        return status;
    }
    built.header.content_hash = normalized_bvh_hash(built.bytes);
    std::memcpy(built.bytes.data(), &built.header, sizeof(built.header));
    return {};
}

[[nodiscard]] core::Result<BuiltBvh> build_host_bvh(const DownloadedScene& source,
                                                    const xpu::cuda::DeviceMemoryBudget budget) {
    auto geometries = read_geometries(source);
    if (!geometries) {
        return std::unexpected(std::move(geometries.error()));
    }
    auto built = BuiltBvh{};
    built.header.source_scene_hash = source.header.content_hash;
    if (auto status = reserve_topology_storage(built, source, *geometries); !status) {
        return std::unexpected(std::move(status.error()));
    }
    if (auto status = build_blases(source, *geometries, built); !status) {
        return std::unexpected(std::move(status.error()));
    }
    if (auto status = build_tlas(source, *geometries, built); !status) {
        return std::unexpected(std::move(status.error()));
    }
    if (auto status = validate_built_topology(built); !status) {
        return std::unexpected(std::move(status.error()));
    }
    if (auto status = serialize(built, budget); !status) {
        return std::unexpected(std::move(status.error()));
    }
    return built;
}

} // namespace

CudaSceneBvh::CudaSceneBvh(SceneBvhHeader header,
                           xpu::cuda::DeviceBuffer<std::uint8_t> device_bytes) noexcept
    : header_(header), device_bytes_(std::move(device_bytes)) {}

CudaSceneBvh::CudaSceneBvh(CudaSceneBvh&& other) noexcept
    : header_(std::exchange(other.header_, SceneBvhHeader{})),
      device_bytes_(std::move(other.device_bytes_)) {}

core::Result<CudaSceneBvh> CudaSceneBvh::build(const CudaSceneSoA& scene,
                                               const CudaSceneBvhBuildOptions options) try {
    if (options.abi_major != xpu::shared::SceneBvhAbiMajor ||
        options.abi_minor != xpu::shared::SceneBvhAbiMinor) {
        return std::unexpected(bvh_error(core::StatusCode::incompatible,
                                         "Requested CUDA BVH ABI version is not supported."));
    }
    if (auto status = preflight_layout(scene.header(), options.device_memory_budget); !status) {
        return std::unexpected(std::move(status.error()));
    }
    auto source = download_scene(scene);
    if (!source) {
        return std::unexpected(std::move(source.error()));
    }
    auto built = build_host_bvh(*source, options.device_memory_budget);
    if (!built) {
        return std::unexpected(std::move(built.error()));
    }

    auto allocation = xpu::cuda::DeviceBuffer<std::uint8_t>::allocate(built->bytes.size(),
                                                                      options.device_memory_budget);
    if (!allocation) {
        return std::unexpected(std::move(allocation.error()));
    }
    auto device_bytes = std::move(*allocation);
    const auto copy_status = cudaMemcpy(device_bytes.data(), built->bytes.data(),
                                        built->bytes.size(), cudaMemcpyHostToDevice);
    if (copy_status != cudaSuccess) {
        return std::unexpected(cuda_error(copy_status, "upload", built->bytes.size()));
    }
    return CudaSceneBvh{built->header, std::move(device_bytes)};
} catch (const std::bad_alloc&) {
    return std::unexpected(bvh_error(core::StatusCode::resource_exhausted,
                                     "CUDA BVH construction exhausted host memory."));
} catch (const std::length_error&) {
    return std::unexpected(bvh_error(core::StatusCode::resource_exhausted,
                                     "CUDA BVH construction exceeded host container limits."));
}

core::Status CudaSceneBvh::close() {
    auto status = device_bytes_.close();
    if (device_bytes_.empty()) {
        header_ = SceneBvhHeader{};
    }
    return status;
}

} // namespace blackframe::engine
