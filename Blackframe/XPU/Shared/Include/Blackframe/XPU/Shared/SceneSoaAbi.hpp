#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace blackframe::xpu::shared {

inline constexpr std::uint64_t SceneSoaMagic = 0x33414F53464B4C42ULL; // "BLKFSOA3"
inline constexpr std::uint16_t SceneSoaAbiMajor = 3U;
inline constexpr std::uint16_t SceneSoaAbiMinor = 0U;
inline constexpr std::uint32_t SceneSoaHashAlgorithmFnv1a64 = 1U;
inline constexpr std::uint64_t SceneSoaFnv1aOffsetBasis = 14695981039346656037ULL;
inline constexpr std::uint64_t SceneSoaFnv1aPrime = 1099511628211ULL;
inline constexpr std::uint64_t SceneSoaColumnAlignment = 16U;
inline constexpr std::uint32_t SceneSoaSpectrumLaneCount = 4U;
inline constexpr std::uint32_t SceneSoaClosureParameterScalarCount = 10U;
inline constexpr std::uint32_t SceneSoaMatrixElementCount = 16U;

namespace scene_soa_column {

inline constexpr std::uint32_t object_id = 0U;

inline constexpr std::uint32_t geometry_id = 1U;
inline constexpr std::uint32_t geometry_vertex_offset = 2U;
inline constexpr std::uint32_t geometry_vertex_count = 3U;
inline constexpr std::uint32_t geometry_triangle_offset = 4U;
inline constexpr std::uint32_t geometry_triangle_count = 5U;

inline constexpr std::uint32_t position_x = 6U;
inline constexpr std::uint32_t position_y = 7U;
inline constexpr std::uint32_t position_z = 8U;
inline constexpr std::uint32_t normal_x = 9U;
inline constexpr std::uint32_t normal_y = 10U;
inline constexpr std::uint32_t normal_z = 11U;
inline constexpr std::uint32_t texture_coordinate_x = 12U;
inline constexpr std::uint32_t texture_coordinate_y = 13U;

inline constexpr std::uint32_t triangle_vertex_0 = 14U;
inline constexpr std::uint32_t triangle_vertex_1 = 15U;
inline constexpr std::uint32_t triangle_vertex_2 = 16U;

inline constexpr std::uint32_t material_id = 17U;
inline constexpr std::uint32_t material_spectral_present = 18U;
inline constexpr std::uint32_t material_wavelength_nanometers = 19U;
inline constexpr std::uint32_t material_wavelength_pdf = 23U;
inline constexpr std::uint32_t material_wavelength_measure = 27U;
inline constexpr std::uint32_t material_closure_offset = 31U;
inline constexpr std::uint32_t material_closure_count = 32U;
inline constexpr std::uint32_t material_closure_frame_mode = 33U;
inline constexpr std::uint32_t material_closure_tangent_rotation_radians = 34U;
inline constexpr std::uint32_t material_emitted_radiance = 35U;

inline constexpr std::uint32_t closure_kind = 39U;
inline constexpr std::uint32_t closure_lobes = 40U;
inline constexpr std::uint32_t closure_weight = 41U;
inline constexpr std::uint32_t closure_parameters = 45U;
inline constexpr std::uint32_t closure_probability =
    closure_parameters + SceneSoaClosureParameterScalarCount;

inline constexpr std::uint32_t instance_id = 56U;
inline constexpr std::uint32_t instance_parent_present = 57U;
inline constexpr std::uint32_t instance_parent_id = 58U;
inline constexpr std::uint32_t instance_object_id = 59U;
inline constexpr std::uint32_t instance_geometry_id = 60U;
inline constexpr std::uint32_t instance_material_id = 61U;
inline constexpr std::uint32_t instance_visibility_mask = 62U;
inline constexpr std::uint32_t instance_local_to_parent = 63U;
inline constexpr std::uint32_t instance_parent_to_local = 79U;
inline constexpr std::uint32_t instance_local_to_world = 95U;
inline constexpr std::uint32_t instance_world_to_local = 111U;

inline constexpr std::uint32_t punctual_kind = 127U;
inline constexpr std::uint32_t punctual_position_x = 128U;
inline constexpr std::uint32_t punctual_position_y = 129U;
inline constexpr std::uint32_t punctual_position_z = 130U;
inline constexpr std::uint32_t punctual_position_error_x = 131U;
inline constexpr std::uint32_t punctual_position_error_y = 132U;
inline constexpr std::uint32_t punctual_position_error_z = 133U;
inline constexpr std::uint32_t punctual_direction_x = 134U;
inline constexpr std::uint32_t punctual_direction_y = 135U;
inline constexpr std::uint32_t punctual_direction_z = 136U;
inline constexpr std::uint32_t punctual_inner_half_angle = 137U;
inline constexpr std::uint32_t punctual_outer_half_angle = 138U;
inline constexpr std::uint32_t punctual_spectrum = 139U;

inline constexpr std::uint32_t mesh_area_light_instance_id = 143U;

inline constexpr std::uint32_t environment_wavelength_nanometers = 144U;
inline constexpr std::uint32_t environment_wavelength_pdf = 148U;
inline constexpr std::uint32_t environment_wavelength_measure = 152U;
inline constexpr std::uint32_t environment_radiance = 156U;

inline constexpr std::uint32_t texture_id = 160U;
inline constexpr std::uint32_t texture_kind = 161U;
inline constexpr std::uint32_t texture_value = 162U;

inline constexpr std::uint32_t count = 166U;

} // namespace scene_soa_column

enum class SceneSoaPunctualKind : std::uint32_t {
    point = 0U,
    directional = 1U,
    spot = 2U,
};

struct SceneSoaColumnDescriptor final {
    std::uint64_t offset_bytes{};
    std::uint64_t element_count{};
    std::uint32_t element_size{};
    std::uint32_t reserved{};
};

struct alignas(16) SceneSoaHeader final {
    std::uint64_t magic{};
    std::uint16_t abi_major{};
    std::uint16_t abi_minor{};
    std::uint32_t header_size{};
    std::uint32_t column_count{};
    std::uint32_t hash_algorithm{};
    std::uint64_t total_size_bytes{};
    std::uint64_t content_hash{};
    std::uint64_t object_count{};
    std::uint64_t geometry_count{};
    std::uint64_t vertex_count{};
    std::uint64_t triangle_count{};
    std::uint64_t material_count{};
    std::uint64_t closure_count{};
    std::uint64_t instance_count{};
    std::uint64_t punctual_light_count{};
    std::uint64_t mesh_area_light_count{};
    std::uint64_t environment_count{};
    std::uint64_t texture_count{};
    std::array<SceneSoaColumnDescriptor, scene_soa_column::count> columns{};
    std::array<std::uint64_t, 4> reserved{};
};

enum class SceneSoaHeaderValidationStatus : std::uint32_t {
    valid = 0U,
    bad_magic,
    incompatible_version,
    bad_header_size,
    bad_column_count,
    unsupported_hash_algorithm,
    nonzero_reserved,
    invalid_environment_count,
    invalid_column_descriptor,
    size_overflow,
    bad_total_size,
};

[[nodiscard]] constexpr std::uint64_t scene_soa_column_count(const SceneSoaHeader& header,
                                                             const std::uint32_t column) noexcept {
    using namespace scene_soa_column;
    if (column == object_id) {
        return header.object_count;
    }
    if (column >= geometry_id && column <= geometry_triangle_count) {
        return header.geometry_count;
    }
    if (column >= position_x && column <= texture_coordinate_y) {
        return header.vertex_count;
    }
    if (column >= triangle_vertex_0 && column <= triangle_vertex_2) {
        return header.triangle_count;
    }
    if (column >= material_id && column < closure_kind) {
        return header.material_count;
    }
    if (column >= closure_kind && column < instance_id) {
        return header.closure_count;
    }
    if (column >= instance_id && column < punctual_kind) {
        return header.instance_count;
    }
    if (column >= punctual_kind && column < mesh_area_light_instance_id) {
        return header.punctual_light_count;
    }
    if (column == mesh_area_light_instance_id) {
        return header.mesh_area_light_count;
    }
    if (column >= environment_wavelength_nanometers && column < texture_id) {
        return header.environment_count;
    }
    if (column >= texture_id && column < count) {
        return header.texture_count;
    }
    return 0U;
}

[[nodiscard]] constexpr std::uint32_t
scene_soa_column_element_size(const std::uint32_t column) noexcept {
    using namespace scene_soa_column;
    if (column >= count) {
        return 0U;
    }
    if ((column >= geometry_vertex_offset && column <= geometry_triangle_count) ||
        column == material_closure_offset || column == material_closure_count) {
        return sizeof(std::uint64_t);
    }
    if (column == material_spectral_present ||
        (column >= material_wavelength_measure && column < material_closure_offset) ||
        column == material_closure_frame_mode || column == instance_parent_present ||
        (column >= environment_wavelength_measure && column < environment_radiance)) {
        return sizeof(std::uint8_t);
    }
    if ((column >= position_x && column <= texture_coordinate_y) ||
        (column >= material_wavelength_nanometers && column < material_wavelength_measure) ||
        column == material_closure_tangent_rotation_radians ||
        (column >= material_emitted_radiance && column < closure_kind) ||
        (column >= closure_weight && column < instance_id) ||
        (column >= instance_local_to_parent && column < punctual_kind) ||
        (column >= punctual_position_x && column < mesh_area_light_instance_id) ||
        (column >= environment_wavelength_nanometers && column < environment_wavelength_measure) ||
        (column >= environment_radiance && column < texture_id) ||
        (column >= texture_value && column < count)) {
        return sizeof(float);
    }
    return sizeof(std::uint32_t);
}

[[nodiscard]] constexpr std::uint64_t scene_soa_align_up(const std::uint64_t value) noexcept {
    const auto remainder = value % SceneSoaColumnAlignment;
    return remainder == 0U ? value : value + (SceneSoaColumnAlignment - remainder);
}

[[nodiscard]] constexpr SceneSoaHeaderValidationStatus
validate_scene_soa_header(const SceneSoaHeader& header) noexcept {
    if (header.magic != SceneSoaMagic) {
        return SceneSoaHeaderValidationStatus::bad_magic;
    }
    if (header.abi_major != SceneSoaAbiMajor || header.abi_minor != SceneSoaAbiMinor) {
        return SceneSoaHeaderValidationStatus::incompatible_version;
    }
    if (header.header_size != sizeof(SceneSoaHeader)) {
        return SceneSoaHeaderValidationStatus::bad_header_size;
    }
    if (header.column_count != scene_soa_column::count) {
        return SceneSoaHeaderValidationStatus::bad_column_count;
    }
    if (header.hash_algorithm != SceneSoaHashAlgorithmFnv1a64) {
        return SceneSoaHeaderValidationStatus::unsupported_hash_algorithm;
    }
    for (const auto value : header.reserved) {
        if (value != 0U) {
            return SceneSoaHeaderValidationStatus::nonzero_reserved;
        }
    }
    if (header.environment_count > 1U) {
        return SceneSoaHeaderValidationStatus::invalid_environment_count;
    }
    if (header.total_size_bytes < sizeof(SceneSoaHeader)) {
        return SceneSoaHeaderValidationStatus::bad_total_size;
    }

    auto cursor = scene_soa_align_up(sizeof(SceneSoaHeader));
    for (auto column = std::uint32_t{0}; column < scene_soa_column::count; ++column) {
        const auto& descriptor = header.columns[column];
        const auto expected_count = scene_soa_column_count(header, column);
        const auto expected_size = scene_soa_column_element_size(column);
        if (descriptor.element_count != expected_count ||
            descriptor.element_size != expected_size || descriptor.reserved != 0U) {
            return SceneSoaHeaderValidationStatus::invalid_column_descriptor;
        }
        if (expected_count == 0U) {
            if (descriptor.offset_bytes != 0U) {
                return SceneSoaHeaderValidationStatus::invalid_column_descriptor;
            }
            continue;
        }
        if (cursor > std::numeric_limits<std::uint64_t>::max() - (SceneSoaColumnAlignment - 1U)) {
            return SceneSoaHeaderValidationStatus::size_overflow;
        }
        cursor = scene_soa_align_up(cursor);
        if (descriptor.offset_bytes != cursor) {
            return SceneSoaHeaderValidationStatus::invalid_column_descriptor;
        }
        if (expected_count > std::numeric_limits<std::uint64_t>::max() / expected_size) {
            return SceneSoaHeaderValidationStatus::size_overflow;
        }
        const auto byte_count = expected_count * expected_size;
        if (cursor > std::numeric_limits<std::uint64_t>::max() - byte_count) {
            return SceneSoaHeaderValidationStatus::size_overflow;
        }
        cursor += byte_count;
    }
    if (header.total_size_bytes != cursor) {
        return SceneSoaHeaderValidationStatus::bad_total_size;
    }
    return SceneSoaHeaderValidationStatus::valid;
}

inline constexpr std::size_t SceneSoaContentHashOffset = offsetof(SceneSoaHeader, content_hash);

static_assert(sizeof(float) == 4U);
static_assert(sizeof(std::uint8_t) == 1U);
static_assert(sizeof(std::uint16_t) == 2U);
static_assert(sizeof(std::uint32_t) == 4U);
static_assert(sizeof(std::uint64_t) == 8U);
static_assert(sizeof(SceneSoaPunctualKind) == sizeof(std::uint32_t));
static_assert(sizeof(SceneSoaHeaderValidationStatus) == sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<SceneSoaColumnDescriptor>);
static_assert(std::is_trivially_copyable_v<SceneSoaColumnDescriptor>);
static_assert(std::is_trivially_destructible_v<SceneSoaColumnDescriptor>);
static_assert(sizeof(SceneSoaColumnDescriptor) == 24U);
static_assert(alignof(SceneSoaColumnDescriptor) == 8U);
static_assert(offsetof(SceneSoaColumnDescriptor, offset_bytes) == 0U);
static_assert(offsetof(SceneSoaColumnDescriptor, element_count) == 8U);
static_assert(offsetof(SceneSoaColumnDescriptor, element_size) == 16U);
static_assert(offsetof(SceneSoaColumnDescriptor, reserved) == 20U);
static_assert(std::is_standard_layout_v<SceneSoaHeader>);
static_assert(std::is_trivially_copyable_v<SceneSoaHeader>);
static_assert(std::is_trivially_destructible_v<SceneSoaHeader>);
static_assert(sizeof(SceneSoaHeader) == 4144U);
static_assert(alignof(SceneSoaHeader) == 16U);
static_assert(offsetof(SceneSoaHeader, magic) == 0U);
static_assert(offsetof(SceneSoaHeader, abi_major) == 8U);
static_assert(offsetof(SceneSoaHeader, abi_minor) == 10U);
static_assert(offsetof(SceneSoaHeader, header_size) == 12U);
static_assert(offsetof(SceneSoaHeader, column_count) == 16U);
static_assert(offsetof(SceneSoaHeader, hash_algorithm) == 20U);
static_assert(offsetof(SceneSoaHeader, total_size_bytes) == 24U);
static_assert(offsetof(SceneSoaHeader, content_hash) == 32U);
static_assert(offsetof(SceneSoaHeader, object_count) == 40U);
static_assert(offsetof(SceneSoaHeader, geometry_count) == 48U);
static_assert(offsetof(SceneSoaHeader, vertex_count) == 56U);
static_assert(offsetof(SceneSoaHeader, triangle_count) == 64U);
static_assert(offsetof(SceneSoaHeader, material_count) == 72U);
static_assert(offsetof(SceneSoaHeader, closure_count) == 80U);
static_assert(offsetof(SceneSoaHeader, instance_count) == 88U);
static_assert(offsetof(SceneSoaHeader, punctual_light_count) == 96U);
static_assert(offsetof(SceneSoaHeader, mesh_area_light_count) == 104U);
static_assert(offsetof(SceneSoaHeader, environment_count) == 112U);
static_assert(offsetof(SceneSoaHeader, texture_count) == 120U);
static_assert(offsetof(SceneSoaHeader, columns) == 128U);
static_assert(offsetof(SceneSoaHeader, reserved) == 4112U);

} // namespace blackframe::xpu::shared
