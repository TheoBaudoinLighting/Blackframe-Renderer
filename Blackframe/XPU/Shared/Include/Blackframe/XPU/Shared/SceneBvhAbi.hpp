#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace blackframe::xpu::shared {

inline constexpr std::uint64_t SceneBvhMagic = 0x31485642464B4C42ULL; // "BLKFBVH1"
inline constexpr std::uint16_t SceneBvhAbiMajor = 1U;
inline constexpr std::uint16_t SceneBvhAbiMinor = 0U;
inline constexpr std::uint32_t SceneBvhHashAlgorithmFnv1a64 = 1U;
inline constexpr std::uint64_t SceneBvhFnv1aOffsetBasis = 14695981039346656037ULL;
inline constexpr std::uint64_t SceneBvhFnv1aPrime = 1099511628211ULL;
inline constexpr std::uint64_t SceneBvhArrayAlignment = 16U;
inline constexpr std::uint32_t SceneBvhLeafCapacity = 1U;
inline constexpr std::uint32_t SceneBvhInvalidIndex = std::numeric_limits<std::uint32_t>::max();

namespace scene_bvh_array {

inline constexpr std::uint32_t blas = 0U;
inline constexpr std::uint32_t blas_node = 1U;
inline constexpr std::uint32_t primitive_reference = 2U;
inline constexpr std::uint32_t tlas_node = 3U;
inline constexpr std::uint32_t instance_reference = 4U;
inline constexpr std::uint32_t count = 5U;

} // namespace scene_bvh_array

enum class SceneBvhNodeKind : std::uint32_t {
    internal = 0U,
    leaf = 1U,
};

struct alignas(16) SceneBvhNode final {
    float minimum_x{};
    float minimum_y{};
    float minimum_z{};
    std::uint32_t kind{};
    float maximum_x{};
    float maximum_y{};
    float maximum_z{};
    std::uint32_t split_axis{SceneBvhInvalidIndex};
    std::uint32_t first_child{SceneBvhInvalidIndex};
    std::uint32_t second_child{SceneBvhInvalidIndex};
    std::uint32_t reference_offset{};
    std::uint32_t reference_count{};
};

struct alignas(16) SceneBvhBlas final {
    std::uint32_t geometry_id{};
    std::uint32_t root_node{SceneBvhInvalidIndex};
    std::uint32_t node_offset{};
    std::uint32_t node_count{};
    std::uint32_t primitive_reference_offset{};
    std::uint32_t primitive_reference_count{};
    std::array<std::uint32_t, 2> reserved{};
};

struct alignas(16) SceneBvhInstanceReference final {
    std::uint32_t scene_instance_index{};
    std::uint32_t instance_id{};
    std::uint32_t scene_geometry_index{};
    std::uint32_t geometry_id{};
    std::uint32_t blas_index{};
    std::uint32_t visibility_mask{};
    std::array<std::uint32_t, 2> reserved{};
};

struct SceneBvhArrayDescriptor final {
    std::uint64_t offset_bytes{};
    std::uint64_t element_count{};
    std::uint32_t element_size{};
    std::uint32_t reserved{};
};

struct alignas(16) SceneBvhHeader final {
    std::uint64_t magic{};
    std::uint16_t abi_major{};
    std::uint16_t abi_minor{};
    std::uint32_t header_size{};
    std::uint32_t array_count{};
    std::uint32_t hash_algorithm{};
    std::uint64_t total_size_bytes{};
    std::uint64_t content_hash{};
    std::uint64_t source_scene_hash{};
    std::uint64_t blas_count{};
    std::uint64_t blas_node_count{};
    std::uint64_t primitive_reference_count{};
    std::uint64_t tlas_node_count{};
    std::uint64_t instance_reference_count{};
    std::uint32_t tlas_root_node{SceneBvhInvalidIndex};
    std::uint32_t leaf_capacity{};
    std::array<SceneBvhArrayDescriptor, scene_bvh_array::count> arrays{};
    std::array<std::uint64_t, 5> reserved{};
};

enum class SceneBvhHeaderValidationStatus : std::uint32_t {
    valid = 0U,
    bad_magic,
    incompatible_version,
    bad_header_size,
    bad_array_count,
    unsupported_hash_algorithm,
    nonzero_reserved,
    bad_leaf_capacity,
    invalid_topology_counts,
    invalid_tlas_root,
    invalid_array_descriptor,
    size_overflow,
    bad_total_size,
};

[[nodiscard]] constexpr std::uint64_t scene_bvh_array_count(const SceneBvhHeader& header,
                                                            const std::uint32_t array) noexcept {
    switch (array) {
    case scene_bvh_array::blas:
        return header.blas_count;
    case scene_bvh_array::blas_node:
        return header.blas_node_count;
    case scene_bvh_array::primitive_reference:
        return header.primitive_reference_count;
    case scene_bvh_array::tlas_node:
        return header.tlas_node_count;
    case scene_bvh_array::instance_reference:
        return header.instance_reference_count;
    default:
        return 0U;
    }
}

[[nodiscard]] constexpr std::uint32_t
scene_bvh_array_element_size(const std::uint32_t array) noexcept {
    switch (array) {
    case scene_bvh_array::blas:
        return sizeof(SceneBvhBlas);
    case scene_bvh_array::blas_node:
    case scene_bvh_array::tlas_node:
        return sizeof(SceneBvhNode);
    case scene_bvh_array::primitive_reference:
        return sizeof(std::uint32_t);
    case scene_bvh_array::instance_reference:
        return sizeof(SceneBvhInstanceReference);
    default:
        return 0U;
    }
}

[[nodiscard]] constexpr std::uint64_t scene_bvh_align_up(const std::uint64_t value) noexcept {
    const auto remainder = value % SceneBvhArrayAlignment;
    return remainder == 0U ? value : value + (SceneBvhArrayAlignment - remainder);
}

[[nodiscard]] constexpr SceneBvhHeaderValidationStatus
validate_scene_bvh_header(const SceneBvhHeader& header) noexcept {
    if (header.magic != SceneBvhMagic) {
        return SceneBvhHeaderValidationStatus::bad_magic;
    }
    if (header.abi_major != SceneBvhAbiMajor || header.abi_minor != SceneBvhAbiMinor) {
        return SceneBvhHeaderValidationStatus::incompatible_version;
    }
    if (header.header_size != sizeof(SceneBvhHeader)) {
        return SceneBvhHeaderValidationStatus::bad_header_size;
    }
    if (header.array_count != scene_bvh_array::count) {
        return SceneBvhHeaderValidationStatus::bad_array_count;
    }
    if (header.hash_algorithm != SceneBvhHashAlgorithmFnv1a64) {
        return SceneBvhHeaderValidationStatus::unsupported_hash_algorithm;
    }
    for (const auto value : header.reserved) {
        if (value != 0U) {
            return SceneBvhHeaderValidationStatus::nonzero_reserved;
        }
    }
    if (header.leaf_capacity != SceneBvhLeafCapacity) {
        return SceneBvhHeaderValidationStatus::bad_leaf_capacity;
    }
    if (header.blas_count > SceneBvhInvalidIndex || header.blas_node_count > SceneBvhInvalidIndex ||
        header.primitive_reference_count > SceneBvhInvalidIndex ||
        header.tlas_node_count > SceneBvhInvalidIndex ||
        header.instance_reference_count > SceneBvhInvalidIndex ||
        (header.blas_count == 0U &&
         (header.blas_node_count != 0U || header.primitive_reference_count != 0U)) ||
        (header.blas_count != 0U &&
         (header.primitive_reference_count < header.blas_count ||
          header.blas_node_count != header.primitive_reference_count * 2U - header.blas_count)) ||
        (header.instance_reference_count == 0U && header.tlas_node_count != 0U) ||
        (header.instance_reference_count != 0U &&
         header.tlas_node_count != header.instance_reference_count * 2U - 1U)) {
        return SceneBvhHeaderValidationStatus::invalid_topology_counts;
    }
    if ((header.tlas_node_count == 0U && header.tlas_root_node != SceneBvhInvalidIndex) ||
        (header.tlas_node_count != 0U && header.tlas_root_node != 0U)) {
        return SceneBvhHeaderValidationStatus::invalid_tlas_root;
    }
    if (header.total_size_bytes < sizeof(SceneBvhHeader)) {
        return SceneBvhHeaderValidationStatus::bad_total_size;
    }

    auto cursor = scene_bvh_align_up(sizeof(SceneBvhHeader));
    for (auto array = std::uint32_t{0}; array < scene_bvh_array::count; ++array) {
        const auto& descriptor = header.arrays[array];
        const auto expected_count = scene_bvh_array_count(header, array);
        const auto expected_size = scene_bvh_array_element_size(array);
        if (descriptor.element_count != expected_count ||
            descriptor.element_size != expected_size || descriptor.reserved != 0U) {
            return SceneBvhHeaderValidationStatus::invalid_array_descriptor;
        }
        if (expected_count == 0U) {
            if (descriptor.offset_bytes != 0U) {
                return SceneBvhHeaderValidationStatus::invalid_array_descriptor;
            }
            continue;
        }
        if (cursor > std::numeric_limits<std::uint64_t>::max() - (SceneBvhArrayAlignment - 1U)) {
            return SceneBvhHeaderValidationStatus::size_overflow;
        }
        cursor = scene_bvh_align_up(cursor);
        if (descriptor.offset_bytes != cursor) {
            return SceneBvhHeaderValidationStatus::invalid_array_descriptor;
        }
        if (expected_count > std::numeric_limits<std::uint64_t>::max() / expected_size) {
            return SceneBvhHeaderValidationStatus::size_overflow;
        }
        const auto byte_count = expected_count * expected_size;
        if (cursor > std::numeric_limits<std::uint64_t>::max() - byte_count) {
            return SceneBvhHeaderValidationStatus::size_overflow;
        }
        cursor += byte_count;
    }
    if (header.total_size_bytes != cursor) {
        return SceneBvhHeaderValidationStatus::bad_total_size;
    }
    return SceneBvhHeaderValidationStatus::valid;
}

inline constexpr std::size_t SceneBvhContentHashOffset = offsetof(SceneBvhHeader, content_hash);

static_assert(sizeof(float) == 4U);
static_assert(sizeof(SceneBvhNodeKind) == sizeof(std::uint32_t));
static_assert(sizeof(SceneBvhHeaderValidationStatus) == sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<SceneBvhNode>);
static_assert(std::is_trivially_copyable_v<SceneBvhNode>);
static_assert(sizeof(SceneBvhNode) == 48U);
static_assert(alignof(SceneBvhNode) == 16U);
static_assert(offsetof(SceneBvhNode, minimum_x) == 0U);
static_assert(offsetof(SceneBvhNode, kind) == 12U);
static_assert(offsetof(SceneBvhNode, maximum_x) == 16U);
static_assert(offsetof(SceneBvhNode, split_axis) == 28U);
static_assert(offsetof(SceneBvhNode, first_child) == 32U);
static_assert(offsetof(SceneBvhNode, reference_offset) == 40U);
static_assert(std::is_standard_layout_v<SceneBvhBlas>);
static_assert(std::is_trivially_copyable_v<SceneBvhBlas>);
static_assert(sizeof(SceneBvhBlas) == 32U);
static_assert(alignof(SceneBvhBlas) == 16U);
static_assert(offsetof(SceneBvhBlas, geometry_id) == 0U);
static_assert(offsetof(SceneBvhBlas, primitive_reference_offset) == 16U);
static_assert(std::is_standard_layout_v<SceneBvhInstanceReference>);
static_assert(std::is_trivially_copyable_v<SceneBvhInstanceReference>);
static_assert(sizeof(SceneBvhInstanceReference) == 32U);
static_assert(alignof(SceneBvhInstanceReference) == 16U);
static_assert(offsetof(SceneBvhInstanceReference, scene_instance_index) == 0U);
static_assert(offsetof(SceneBvhInstanceReference, blas_index) == 16U);
static_assert(std::is_standard_layout_v<SceneBvhArrayDescriptor>);
static_assert(std::is_trivially_copyable_v<SceneBvhArrayDescriptor>);
static_assert(sizeof(SceneBvhArrayDescriptor) == 24U);
static_assert(alignof(SceneBvhArrayDescriptor) == 8U);
static_assert(std::is_standard_layout_v<SceneBvhHeader>);
static_assert(std::is_trivially_copyable_v<SceneBvhHeader>);
static_assert(sizeof(SceneBvhHeader) == 256U);
static_assert(alignof(SceneBvhHeader) == 16U);
static_assert(offsetof(SceneBvhHeader, magic) == 0U);
static_assert(offsetof(SceneBvhHeader, total_size_bytes) == 24U);
static_assert(offsetof(SceneBvhHeader, content_hash) == 32U);
static_assert(offsetof(SceneBvhHeader, source_scene_hash) == 40U);
static_assert(offsetof(SceneBvhHeader, tlas_root_node) == 88U);
static_assert(offsetof(SceneBvhHeader, arrays) == 96U);
static_assert(offsetof(SceneBvhHeader, reserved) == 216U);

} // namespace blackframe::xpu::shared
