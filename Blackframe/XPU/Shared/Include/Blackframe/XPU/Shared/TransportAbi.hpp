#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace blackframe::xpu::shared {

// This is an in-process host/device memory ABI, not the native extension ABI and not a serialized
// wire format. Records use fixed-width scalars, explicit padding, and no native pointers. Renderer
// objects have different layouts and must be converted field by field rather than copied as bytes.
inline constexpr std::uint16_t HostDeviceTransportAbiMajor = 1U;
inline constexpr std::uint16_t HostDeviceTransportAbiMinor = 0U;
inline constexpr std::uint32_t HostDeviceSpectrumLaneCount = 4U;

struct alignas(16) TransportSpectrum final {
    float values[HostDeviceSpectrumLaneCount];
};

// The two xyz groups are packed with their parameter bounds. The final word is reserved and must
// be zero. This deliberately differs from renderer::Ray's host-only layout.
struct alignas(16) TransportRay final {
    float origin_x;
    float origin_y;
    float origin_z;
    float t_min;
    float direction_x;
    float direction_y;
    float direction_z;
    float t_max;
    float time;
    std::uint32_t visibility_mask;
    std::uint32_t current_medium;
    std::uint32_t reserved;
};

struct alignas(8) SampleStreamIndex final {
    std::uint32_t pixel_x;
    std::uint32_t pixel_y;
    std::uint64_t sample_index;
    std::uint64_t seed;
};

struct PathSlot final {
    std::uint32_t value;
};

// Every queue buffer begins with this header. Unknown ABI versions, sizes, queue kinds, or non-zero
// reserved fields are errors; a consumer must never reinterpret them as an older layout.
struct alignas(16) QueueHeader final {
    std::uint16_t abi_major;
    std::uint16_t abi_minor;
    std::uint32_t struct_size;
    std::uint32_t queue_kind;
    std::uint32_t capacity;
    std::uint32_t size;
    std::uint32_t overflow_count;
    std::uint32_t rejected_count;
    std::uint32_t reserved;
};

enum class QueueHeaderValidationStatus : std::uint8_t {
    valid = 0U,
    unsupported_abi_version = 1U,
    unexpected_struct_size = 2U,
    unknown_queue_kind = 3U,
    size_exceeds_capacity = 4U,
    nonzero_reserved = 5U,
};

[[nodiscard]] constexpr QueueHeaderValidationStatus
validate_queue_header(const QueueHeader& header) noexcept {
    if (header.abi_major != HostDeviceTransportAbiMajor ||
        header.abi_minor != HostDeviceTransportAbiMinor) {
        return QueueHeaderValidationStatus::unsupported_abi_version;
    }
    if (header.struct_size != static_cast<std::uint32_t>(sizeof(QueueHeader))) {
        return QueueHeaderValidationStatus::unexpected_struct_size;
    }
    // Queue-kind codes are the fixed-width representation of the canonical renderer convention.
    if (header.queue_kind > 6U) {
        return QueueHeaderValidationStatus::unknown_queue_kind;
    }
    if (header.size > header.capacity) {
        return QueueHeaderValidationStatus::size_exceeds_capacity;
    }
    if (header.reserved != 0U) {
        return QueueHeaderValidationStatus::nonzero_reserved;
    }
    return QueueHeaderValidationStatus::valid;
}

struct alignas(16) SurfaceIdentifiers final {
    std::uint32_t object;
    std::uint32_t instance;
    std::uint32_t geometry;
    std::uint32_t primitive;
    std::uint32_t material;
    std::uint32_t reserved[3U];
};

struct alignas(16) ClosestHit final {
    float parameter;
    float barycentric_vertex0;
    float barycentric_vertex1;
    float barycentric_vertex2;
    float geometric_normal_x;
    float geometric_normal_y;
    float geometric_normal_z;
    std::uint32_t reserved;
    SurfaceIdentifiers identifiers;
};

// This is one float path lane transferred across the boundary, not a declaration of queue storage.
// Wavefront buffers may decompose these fields into SoA columns. Wavelength probability measures
// remain explicit, all reserved words must be zero, and the double scalar reference stays
// host-only.
struct alignas(16) TransportPathStateLane final {
    TransportSpectrum beta;
    TransportSpectrum accumulated_radiance;
    TransportSpectrum wavelength_nanometers;
    TransportSpectrum wavelength_pdf_values;
    std::uint32_t diffuse_depth;
    std::uint32_t glossy_depth;
    std::uint32_t specular_depth;
    std::uint32_t transmission_depth;
    std::uint32_t volume_depth;
    std::uint32_t depth;
    float eta_scale;
    std::uint32_t current_medium;
    std::uint32_t delta_flags;
    std::uint8_t wavelength_pdf_measures[HostDeviceSpectrumLaneCount];
    std::uint32_t reserved[6U];
};

enum class LayoutValue : std::uint32_t {
    spectrum_size,
    spectrum_alignment,
    spectrum_values_offset,
    ray_size,
    ray_alignment,
    ray_origin_x_offset,
    ray_origin_y_offset,
    ray_origin_z_offset,
    ray_t_min_offset,
    ray_direction_x_offset,
    ray_direction_y_offset,
    ray_direction_z_offset,
    ray_t_max_offset,
    ray_time_offset,
    ray_visibility_mask_offset,
    ray_current_medium_offset,
    ray_reserved_offset,
    sample_stream_index_size,
    sample_stream_index_alignment,
    sample_stream_index_pixel_x_offset,
    sample_stream_index_pixel_y_offset,
    sample_stream_index_sample_index_offset,
    sample_stream_index_seed_offset,
    path_slot_size,
    path_slot_alignment,
    path_slot_value_offset,
    queue_header_size,
    queue_header_alignment,
    queue_header_abi_major_offset,
    queue_header_abi_minor_offset,
    queue_header_struct_size_offset,
    queue_header_kind_offset,
    queue_header_capacity_offset,
    queue_header_size_offset,
    queue_header_overflow_count_offset,
    queue_header_rejected_count_offset,
    queue_header_reserved_offset,
    surface_identifiers_size,
    surface_identifiers_alignment,
    surface_identifiers_object_offset,
    surface_identifiers_instance_offset,
    surface_identifiers_geometry_offset,
    surface_identifiers_primitive_offset,
    surface_identifiers_material_offset,
    surface_identifiers_reserved_offset,
    closest_hit_size,
    closest_hit_alignment,
    closest_hit_parameter_offset,
    closest_hit_barycentric_vertex0_offset,
    closest_hit_barycentric_vertex1_offset,
    closest_hit_barycentric_vertex2_offset,
    closest_hit_geometric_normal_x_offset,
    closest_hit_geometric_normal_y_offset,
    closest_hit_geometric_normal_z_offset,
    closest_hit_reserved_offset,
    closest_hit_identifiers_offset,
    path_state_size,
    path_state_alignment,
    path_state_beta_offset,
    path_state_accumulated_radiance_offset,
    path_state_wavelength_nanometers_offset,
    path_state_wavelength_pdf_values_offset,
    path_state_diffuse_depth_offset,
    path_state_glossy_depth_offset,
    path_state_specular_depth_offset,
    path_state_transmission_depth_offset,
    path_state_volume_depth_offset,
    path_state_depth_offset,
    path_state_eta_scale_offset,
    path_state_current_medium_offset,
    path_state_delta_flags_offset,
    path_state_wavelength_pdf_measures_offset,
    path_state_reserved_offset,
    count,
};

inline constexpr auto HostDeviceLayoutValueCount = static_cast<std::uint32_t>(LayoutValue::count);

// A CUDA probe fills this manifest from device code. The host computes the same indexed values
// independently, then compares every word. Explicit tail storage keeps the manifest itself free of
// compiler-owned padding.
struct alignas(16) LayoutManifest final {
    std::uint16_t abi_major;
    std::uint16_t abi_minor;
    std::uint32_t device_cxx_standard;
    std::uint32_t value_count;
    std::uint32_t reserved_header;
    std::uint32_t values[HostDeviceLayoutValueCount];
    std::uint32_t reserved_tail[3U];
};

[[nodiscard]] constexpr std::uint32_t layout_value_index(const LayoutValue value) noexcept {
    return static_cast<std::uint32_t>(value);
}

// This host-side diagnostic is deliberately expressed in the shared C++20 subset. CUDA device
// code computes its own manifest in a kernel rather than returning this host result.
[[nodiscard]] constexpr LayoutManifest
host_layout_manifest(const std::uint32_t cxx_standard) noexcept {
    auto manifest = LayoutManifest{};
    manifest.abi_major = HostDeviceTransportAbiMajor;
    manifest.abi_minor = HostDeviceTransportAbiMinor;
    manifest.device_cxx_standard = cxx_standard;
    manifest.value_count = HostDeviceLayoutValueCount;

#define BLACKFRAME_SET_HOST_LAYOUT_VALUE(name, expression)                                         \
    manifest.values[layout_value_index(LayoutValue::name)] = static_cast<std::uint32_t>(expression)

    BLACKFRAME_SET_HOST_LAYOUT_VALUE(spectrum_size, sizeof(TransportSpectrum));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(spectrum_alignment, alignof(TransportSpectrum));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(spectrum_values_offset, offsetof(TransportSpectrum, values));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(ray_size, sizeof(TransportRay));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(ray_alignment, alignof(TransportRay));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(ray_origin_x_offset, offsetof(TransportRay, origin_x));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(ray_origin_y_offset, offsetof(TransportRay, origin_y));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(ray_origin_z_offset, offsetof(TransportRay, origin_z));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(ray_t_min_offset, offsetof(TransportRay, t_min));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(ray_direction_x_offset, offsetof(TransportRay, direction_x));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(ray_direction_y_offset, offsetof(TransportRay, direction_y));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(ray_direction_z_offset, offsetof(TransportRay, direction_z));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(ray_t_max_offset, offsetof(TransportRay, t_max));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(ray_time_offset, offsetof(TransportRay, time));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(ray_visibility_mask_offset,
                                     offsetof(TransportRay, visibility_mask));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(ray_current_medium_offset,
                                     offsetof(TransportRay, current_medium));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(ray_reserved_offset, offsetof(TransportRay, reserved));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(sample_stream_index_size, sizeof(SampleStreamIndex));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(sample_stream_index_alignment, alignof(SampleStreamIndex));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(sample_stream_index_pixel_x_offset,
                                     offsetof(SampleStreamIndex, pixel_x));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(sample_stream_index_pixel_y_offset,
                                     offsetof(SampleStreamIndex, pixel_y));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(sample_stream_index_sample_index_offset,
                                     offsetof(SampleStreamIndex, sample_index));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(sample_stream_index_seed_offset,
                                     offsetof(SampleStreamIndex, seed));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_slot_size, sizeof(PathSlot));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_slot_alignment, alignof(PathSlot));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_slot_value_offset, offsetof(PathSlot, value));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(queue_header_size, sizeof(QueueHeader));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(queue_header_alignment, alignof(QueueHeader));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(queue_header_abi_major_offset,
                                     offsetof(QueueHeader, abi_major));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(queue_header_abi_minor_offset,
                                     offsetof(QueueHeader, abi_minor));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(queue_header_struct_size_offset,
                                     offsetof(QueueHeader, struct_size));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(queue_header_kind_offset, offsetof(QueueHeader, queue_kind));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(queue_header_capacity_offset, offsetof(QueueHeader, capacity));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(queue_header_size_offset, offsetof(QueueHeader, size));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(queue_header_overflow_count_offset,
                                     offsetof(QueueHeader, overflow_count));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(queue_header_rejected_count_offset,
                                     offsetof(QueueHeader, rejected_count));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(queue_header_reserved_offset, offsetof(QueueHeader, reserved));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(surface_identifiers_size, sizeof(SurfaceIdentifiers));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(surface_identifiers_alignment, alignof(SurfaceIdentifiers));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(surface_identifiers_object_offset,
                                     offsetof(SurfaceIdentifiers, object));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(surface_identifiers_instance_offset,
                                     offsetof(SurfaceIdentifiers, instance));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(surface_identifiers_geometry_offset,
                                     offsetof(SurfaceIdentifiers, geometry));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(surface_identifiers_primitive_offset,
                                     offsetof(SurfaceIdentifiers, primitive));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(surface_identifiers_material_offset,
                                     offsetof(SurfaceIdentifiers, material));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(surface_identifiers_reserved_offset,
                                     offsetof(SurfaceIdentifiers, reserved));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(closest_hit_size, sizeof(ClosestHit));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(closest_hit_alignment, alignof(ClosestHit));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(closest_hit_parameter_offset, offsetof(ClosestHit, parameter));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(closest_hit_barycentric_vertex0_offset,
                                     offsetof(ClosestHit, barycentric_vertex0));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(closest_hit_barycentric_vertex1_offset,
                                     offsetof(ClosestHit, barycentric_vertex1));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(closest_hit_barycentric_vertex2_offset,
                                     offsetof(ClosestHit, barycentric_vertex2));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(closest_hit_geometric_normal_x_offset,
                                     offsetof(ClosestHit, geometric_normal_x));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(closest_hit_geometric_normal_y_offset,
                                     offsetof(ClosestHit, geometric_normal_y));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(closest_hit_geometric_normal_z_offset,
                                     offsetof(ClosestHit, geometric_normal_z));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(closest_hit_reserved_offset, offsetof(ClosestHit, reserved));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(closest_hit_identifiers_offset,
                                     offsetof(ClosestHit, identifiers));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_state_size, sizeof(TransportPathStateLane));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_state_alignment, alignof(TransportPathStateLane));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_state_beta_offset,
                                     offsetof(TransportPathStateLane, beta));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_state_accumulated_radiance_offset,
                                     offsetof(TransportPathStateLane, accumulated_radiance));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_state_wavelength_nanometers_offset,
                                     offsetof(TransportPathStateLane, wavelength_nanometers));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_state_wavelength_pdf_values_offset,
                                     offsetof(TransportPathStateLane, wavelength_pdf_values));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_state_diffuse_depth_offset,
                                     offsetof(TransportPathStateLane, diffuse_depth));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_state_glossy_depth_offset,
                                     offsetof(TransportPathStateLane, glossy_depth));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_state_specular_depth_offset,
                                     offsetof(TransportPathStateLane, specular_depth));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_state_transmission_depth_offset,
                                     offsetof(TransportPathStateLane, transmission_depth));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_state_volume_depth_offset,
                                     offsetof(TransportPathStateLane, volume_depth));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_state_depth_offset,
                                     offsetof(TransportPathStateLane, depth));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_state_eta_scale_offset,
                                     offsetof(TransportPathStateLane, eta_scale));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_state_current_medium_offset,
                                     offsetof(TransportPathStateLane, current_medium));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_state_delta_flags_offset,
                                     offsetof(TransportPathStateLane, delta_flags));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_state_wavelength_pdf_measures_offset,
                                     offsetof(TransportPathStateLane, wavelength_pdf_measures));
    BLACKFRAME_SET_HOST_LAYOUT_VALUE(path_state_reserved_offset,
                                     offsetof(TransportPathStateLane, reserved));

#undef BLACKFRAME_SET_HOST_LAYOUT_VALUE

    return manifest;
}

static_assert(sizeof(float) == 4U);
static_assert(sizeof(std::uint8_t) == 1U);
static_assert(sizeof(std::uint16_t) == 2U);
static_assert(sizeof(std::uint32_t) == 4U);
static_assert(sizeof(std::uint64_t) == 8U);
static_assert(sizeof(QueueHeaderValidationStatus) == 1U);

#define BLACKFRAME_ASSERT_HOST_DEVICE_RECORD(record)                                               \
    static_assert(std::is_standard_layout_v<record>);                                              \
    static_assert(std::is_trivially_copyable_v<record>);                                           \
    static_assert(std::is_trivially_destructible_v<record>)

BLACKFRAME_ASSERT_HOST_DEVICE_RECORD(TransportSpectrum);
BLACKFRAME_ASSERT_HOST_DEVICE_RECORD(TransportRay);
BLACKFRAME_ASSERT_HOST_DEVICE_RECORD(SampleStreamIndex);
BLACKFRAME_ASSERT_HOST_DEVICE_RECORD(PathSlot);
BLACKFRAME_ASSERT_HOST_DEVICE_RECORD(QueueHeader);
BLACKFRAME_ASSERT_HOST_DEVICE_RECORD(SurfaceIdentifiers);
BLACKFRAME_ASSERT_HOST_DEVICE_RECORD(ClosestHit);
BLACKFRAME_ASSERT_HOST_DEVICE_RECORD(TransportPathStateLane);
BLACKFRAME_ASSERT_HOST_DEVICE_RECORD(LayoutManifest);

#undef BLACKFRAME_ASSERT_HOST_DEVICE_RECORD

static_assert(sizeof(TransportSpectrum) == 16U);
static_assert(alignof(TransportSpectrum) == 16U);
static_assert(offsetof(TransportSpectrum, values) == 0U);

static_assert(sizeof(TransportRay) == 48U);
static_assert(alignof(TransportRay) == 16U);
static_assert(offsetof(TransportRay, origin_x) == 0U);
static_assert(offsetof(TransportRay, origin_y) == 4U);
static_assert(offsetof(TransportRay, origin_z) == 8U);
static_assert(offsetof(TransportRay, t_min) == 12U);
static_assert(offsetof(TransportRay, direction_x) == 16U);
static_assert(offsetof(TransportRay, direction_y) == 20U);
static_assert(offsetof(TransportRay, direction_z) == 24U);
static_assert(offsetof(TransportRay, t_max) == 28U);
static_assert(offsetof(TransportRay, time) == 32U);
static_assert(offsetof(TransportRay, visibility_mask) == 36U);
static_assert(offsetof(TransportRay, current_medium) == 40U);
static_assert(offsetof(TransportRay, reserved) == 44U);

static_assert(sizeof(SampleStreamIndex) == 24U);
static_assert(alignof(SampleStreamIndex) == 8U);
static_assert(offsetof(SampleStreamIndex, pixel_x) == 0U);
static_assert(offsetof(SampleStreamIndex, pixel_y) == 4U);
static_assert(offsetof(SampleStreamIndex, sample_index) == 8U);
static_assert(offsetof(SampleStreamIndex, seed) == 16U);

static_assert(sizeof(PathSlot) == 4U);
static_assert(alignof(PathSlot) == 4U);
static_assert(offsetof(PathSlot, value) == 0U);

static_assert(sizeof(QueueHeader) == 32U);
static_assert(alignof(QueueHeader) == 16U);
static_assert(offsetof(QueueHeader, abi_major) == 0U);
static_assert(offsetof(QueueHeader, abi_minor) == 2U);
static_assert(offsetof(QueueHeader, struct_size) == 4U);
static_assert(offsetof(QueueHeader, queue_kind) == 8U);
static_assert(offsetof(QueueHeader, capacity) == 12U);
static_assert(offsetof(QueueHeader, size) == 16U);
static_assert(offsetof(QueueHeader, overflow_count) == 20U);
static_assert(offsetof(QueueHeader, rejected_count) == 24U);
static_assert(offsetof(QueueHeader, reserved) == 28U);

static_assert(sizeof(SurfaceIdentifiers) == 32U);
static_assert(alignof(SurfaceIdentifiers) == 16U);
static_assert(offsetof(SurfaceIdentifiers, object) == 0U);
static_assert(offsetof(SurfaceIdentifiers, instance) == 4U);
static_assert(offsetof(SurfaceIdentifiers, geometry) == 8U);
static_assert(offsetof(SurfaceIdentifiers, primitive) == 12U);
static_assert(offsetof(SurfaceIdentifiers, material) == 16U);
static_assert(offsetof(SurfaceIdentifiers, reserved) == 20U);

static_assert(sizeof(ClosestHit) == 64U);
static_assert(alignof(ClosestHit) == 16U);
static_assert(offsetof(ClosestHit, parameter) == 0U);
static_assert(offsetof(ClosestHit, barycentric_vertex0) == 4U);
static_assert(offsetof(ClosestHit, barycentric_vertex1) == 8U);
static_assert(offsetof(ClosestHit, barycentric_vertex2) == 12U);
static_assert(offsetof(ClosestHit, geometric_normal_x) == 16U);
static_assert(offsetof(ClosestHit, geometric_normal_y) == 20U);
static_assert(offsetof(ClosestHit, geometric_normal_z) == 24U);
static_assert(offsetof(ClosestHit, reserved) == 28U);
static_assert(offsetof(ClosestHit, identifiers) == 32U);

static_assert(sizeof(TransportPathStateLane) == 128U);
static_assert(alignof(TransportPathStateLane) == 16U);
static_assert(offsetof(TransportPathStateLane, beta) == 0U);
static_assert(offsetof(TransportPathStateLane, accumulated_radiance) == 16U);
static_assert(offsetof(TransportPathStateLane, wavelength_nanometers) == 32U);
static_assert(offsetof(TransportPathStateLane, wavelength_pdf_values) == 48U);
static_assert(offsetof(TransportPathStateLane, diffuse_depth) == 64U);
static_assert(offsetof(TransportPathStateLane, glossy_depth) == 68U);
static_assert(offsetof(TransportPathStateLane, specular_depth) == 72U);
static_assert(offsetof(TransportPathStateLane, transmission_depth) == 76U);
static_assert(offsetof(TransportPathStateLane, volume_depth) == 80U);
static_assert(offsetof(TransportPathStateLane, depth) == 84U);
static_assert(offsetof(TransportPathStateLane, eta_scale) == 88U);
static_assert(offsetof(TransportPathStateLane, current_medium) == 92U);
static_assert(offsetof(TransportPathStateLane, delta_flags) == 96U);
static_assert(offsetof(TransportPathStateLane, wavelength_pdf_measures) == 100U);
static_assert(offsetof(TransportPathStateLane, reserved) == 104U);

static_assert(HostDeviceLayoutValueCount == 73U);
static_assert(sizeof(LayoutManifest) == 320U);
static_assert(alignof(LayoutManifest) == 16U);
static_assert(offsetof(LayoutManifest, abi_major) == 0U);
static_assert(offsetof(LayoutManifest, abi_minor) == 2U);
static_assert(offsetof(LayoutManifest, device_cxx_standard) == 4U);
static_assert(offsetof(LayoutManifest, value_count) == 8U);
static_assert(offsetof(LayoutManifest, reserved_header) == 12U);
static_assert(offsetof(LayoutManifest, values) == 16U);
static_assert(offsetof(LayoutManifest, reserved_tail) == 308U);

} // namespace blackframe::xpu::shared
