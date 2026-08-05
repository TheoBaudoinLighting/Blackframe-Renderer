#include <Blackframe/XPU/CUDA/WavefrontQueueDevice.cuh>
#include <Blackframe/XPU/CUDA/WavefrontStageKernel.hpp>
#include <Blackframe/XPU/Shared/SceneSoaAbi.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

#if !defined(__CUDACC__)
#error "The CUDA wavefront stage kernels must be compiled by the CUDA compiler."
#endif

static_assert(__cplusplus == 202002L);

#if defined(__cpp_pack_indexing)
#error "C++26 features are forbidden in CUDA wavefront stage code."
#endif

namespace {

namespace cuda = blackframe::xpu::cuda;
namespace shared = blackframe::xpu::shared;
namespace scene_column = blackframe::xpu::shared::scene_soa_column;

using cuda::WavefrontCameraInputDeviceSoa;
using cuda::WavefrontLaneControl;
using cuda::WavefrontLanePhase;
using cuda::WavefrontPendingShadow;
using cuda::WavefrontQueueDevicePushStatus;
using cuda::WavefrontQueueDeviceSoa;
using cuda::WavefrontStageDeviceSoa;
using cuda::WavefrontStageOutcome;
using cuda::WavefrontStageRoute;
using cuda::WavefrontStageStatus;
using cuda::WavefrontTermination;
using shared::ClosestHit;
using shared::PathSlot;
using shared::QueueHeader;
using shared::SampleStreamIndex;
using shared::SceneClosestHitResult;
using shared::SceneClosestHitStatus;
using shared::SceneOcclusionResult;
using shared::SceneOcclusionStatus;
using shared::SceneSoaColumnDescriptor;
using shared::SceneSoaHeader;
using shared::TransportPathStateLane;
using shared::TransportRay;
using shared::TransportSpectrum;

constexpr auto ThreadsPerBlock = std::uint32_t{256U};
constexpr auto CameraQueue = std::uint32_t{0U};
constexpr auto RayQueue = std::uint32_t{1U};
constexpr auto HitQueue = std::uint32_t{2U};
constexpr auto MissQueue = std::uint32_t{3U};
constexpr auto ShadeQueue = std::uint32_t{4U};
constexpr auto ShadowQueue = std::uint32_t{5U};
constexpr auto ContinuationQueue = std::uint32_t{6U};
constexpr auto WavelengthMeasure = std::uint8_t{5U};
constexpr auto FirstBounceDimension = std::uint64_t{4U};
constexpr auto DimensionsPerBounce = std::uint64_t{10U};
constexpr auto Pi = 3.14159265358979323846F;
constexpr auto InversePi = 0.31830988618379067154F;
constexpr auto MaximumUniformLightCount = std::uint64_t{1U} << 24U;
constexpr auto MaximumU64 = ~std::uint64_t{0U};
constexpr auto FloatEpsilon = 0x1p-23F;
constexpr auto FloatDenormMinimum = 0x1p-149F;
constexpr auto Gamma7 = (7.0F * FloatEpsilon) / (1.0F - 7.0F * FloatEpsilon);
constexpr auto Gamma7Double =
    (7.0 * static_cast<double>(FloatEpsilon)) / (1.0 - 7.0 * static_cast<double>(FloatEpsilon));
constexpr auto MaximumFloat = 0x1.fffffeP+127F;

struct Vector3 final {
    float x{};
    float y{};
    float z{};
};

struct SurfaceData final {
    Vector3 position{};
    Vector3 position_error{};
    Vector3 geometric_normal{};
    Vector3 shading_normal{};
    std::uint32_t material_index{};
};

struct PositionWithError final {
    Vector3 position{};
    Vector3 absolute_error{};
};

struct LightEndpoint final {
    TransportSpectrum radiance{};
    Vector3 position{};
    Vector3 position_error{};
    Vector3 geometric_normal{};
    float area_density{};
    std::uint32_t visibility_mask{};
    std::uint32_t reserved[3U]{};
};

struct ScaledSegment final {
    Vector3 scaled{};
    Vector3 direction{};
    float scale{};
    float scaled_distance_cubed{};
    float length{};
};

enum class SceneDeviceStatus : std::uint32_t {
    valid = 0U,
    invalid_scene = 1U,
    unsupported_transport = 2U,
    numerical_failure = 3U,
};

enum class ShadeFailureDetail : std::uint32_t {
    surface_data = 1U,
    material_spectrum = 2U,
    emitted_radiance = 3U,
    light_sample = 4U,
    light_distance = 5U,
    light_direction = 6U,
    light_weight = 7U,
    light_contribution = 8U,
    endpoint_offset = 9U,
    shadow_segment = 10U,
    shadow_ray = 11U,
    bsdf_sample = 12U,
    throughput = 13U,
    continuation_ray = 14U,
    continuation_offset = 15U,
};

[[nodiscard]] __device__ constexpr std::uint32_t value(const WavefrontStageStatus status) noexcept {
    return static_cast<std::uint32_t>(status);
}

[[nodiscard]] __device__ constexpr std::uint32_t value(const WavefrontStageRoute route) noexcept {
    return static_cast<std::uint32_t>(route);
}

[[nodiscard]] __device__ constexpr std::uint32_t value(const WavefrontLanePhase phase) noexcept {
    return static_cast<std::uint32_t>(phase);
}

[[nodiscard]] __device__ constexpr std::uint32_t
value(const WavefrontTermination termination) noexcept {
    return static_cast<std::uint32_t>(termination);
}

[[nodiscard]] __device__ constexpr std::uint32_t value(const ShadeFailureDetail detail) noexcept {
    return static_cast<std::uint32_t>(detail);
}

[[nodiscard]] __device__ WavefrontStageOutcome outcome(const WavefrontStageStatus status,
                                                       const WavefrontStageRoute route,
                                                       const std::uint32_t path_slot,
                                                       const std::uint32_t detail = 0U) noexcept {
    return WavefrontStageOutcome{
        .status = value(status),
        .route = value(route),
        .path_slot = path_slot,
        .detail = detail,
    };
}

[[nodiscard]] __device__ bool finite_vector(const Vector3 vector) noexcept {
    return isfinite(vector.x) && isfinite(vector.y) && isfinite(vector.z);
}

[[nodiscard]] __device__ Vector3 add(const Vector3 left, const Vector3 right) noexcept {
    return Vector3{.x = left.x + right.x, .y = left.y + right.y, .z = left.z + right.z};
}

[[nodiscard]] __device__ Vector3 subtract(const Vector3 left, const Vector3 right) noexcept {
    return Vector3{.x = left.x - right.x, .y = left.y - right.y, .z = left.z - right.z};
}

[[nodiscard]] __device__ Vector3 multiply(const Vector3 vector, const float scalar) noexcept {
    return Vector3{.x = vector.x * scalar, .y = vector.y * scalar, .z = vector.z * scalar};
}

[[nodiscard]] __device__ float dot(const Vector3 left, const Vector3 right) noexcept {
    return fmaf(left.x, right.x, fmaf(left.y, right.y, left.z * right.z));
}

[[nodiscard]] __device__ Vector3 cross(const Vector3 left, const Vector3 right) noexcept {
    return Vector3{
        .x = fmaf(left.y, right.z, -left.z * right.y),
        .y = fmaf(left.z, right.x, -left.x * right.z),
        .z = fmaf(left.x, right.y, -left.y * right.x),
    };
}

[[nodiscard]] __device__ bool normalize(const Vector3 vector, Vector3& normalized) noexcept {
    const auto squared_length = dot(vector, vector);
    if (!isfinite(squared_length) || !(squared_length > 0.0F)) {
        return false;
    }
    const auto inverse_length = rsqrtf(squared_length);
    normalized = multiply(vector, inverse_length);
    return finite_vector(normalized);
}

[[nodiscard]] __device__ bool scaled_segment(const Vector3 displacement,
                                             ScaledSegment& segment) noexcept {
    if (!finite_vector(displacement)) {
        return false;
    }
    const auto scale =
        fmaxf(fabsf(displacement.x), fmaxf(fabsf(displacement.y), fabsf(displacement.z)));
    if (!isfinite(scale) || !(scale > 0.0F)) {
        return false;
    }
    const auto scaled = Vector3{
        .x = displacement.x / scale,
        .y = displacement.y / scale,
        .z = displacement.z / scale,
    };
    const auto squared_length = dot(scaled, scaled);
    const auto scaled_length = sqrtf(squared_length);
    const auto scaled_distance_cubed = squared_length * scaled_length;
    const auto length = scale * scaled_length;
    if (!finite_vector(scaled) || !isfinite(squared_length) || !(squared_length > 0.0F) ||
        !isfinite(scaled_length) || !(scaled_length > 0.0F) || !isfinite(scaled_distance_cubed) ||
        !(scaled_distance_cubed > 0.0F) || !isfinite(length) || !(length > 0.0F)) {
        return false;
    }
    const auto direction = multiply(scaled, 1.0F / scaled_length);
    if (!finite_vector(direction)) {
        return false;
    }
    segment = ScaledSegment{
        .scaled = scaled,
        .direction = direction,
        .scale = scale,
        .scaled_distance_cubed = scaled_distance_cubed,
        .length = length,
    };
    return true;
}

[[nodiscard]] __device__ bool light_surface_alignment(const Vector3 geometric_normal,
                                                      const Vector3 scaled_direction_to_light,
                                                      float& absolute_alignment,
                                                      bool& supported) noexcept {
    const auto outgoing_scaled = multiply(scaled_direction_to_light, -1.0F);
    const auto signed_alignment = dot(geometric_normal, outgoing_scaled);
    const auto componentwise_orthogonal =
        (geometric_normal.x == 0.0F || outgoing_scaled.x == 0.0F) &&
        (geometric_normal.y == 0.0F || outgoing_scaled.y == 0.0F) &&
        (geometric_normal.z == 0.0F || outgoing_scaled.z == 0.0F);
    if (signed_alignment == 0.0F && componentwise_orthogonal) {
        absolute_alignment = 0.0F;
        supported = false;
        return true;
    }
    const auto absolute_term_sum = fmaf(fabsf(geometric_normal.x), fabsf(outgoing_scaled.x),
                                        fmaf(fabsf(geometric_normal.y), fabsf(outgoing_scaled.y),
                                             fabsf(geometric_normal.z) * fabsf(outgoing_scaled.z)));
    constexpr auto rounding_factor = 4.0F * FloatEpsilon;
    constexpr auto underflow_allowance = 4.0F * FloatDenormMinimum;
    const auto uncertainty = fmaf(rounding_factor, absolute_term_sum, underflow_allowance);
    absolute_alignment = fabsf(signed_alignment);
    if (!isfinite(signed_alignment) || !isfinite(absolute_alignment) ||
        !isfinite(absolute_term_sum) || !isfinite(uncertainty) ||
        absolute_alignment <= uncertainty) {
        return false;
    }
    supported = signed_alignment > 0.0F;
    return true;
}

[[nodiscard]] __device__ bool finite_spectrum(const TransportSpectrum& spectrum) noexcept {
    for (auto lane = std::uint32_t{0U}; lane < shared::HostDeviceSpectrumLaneCount; ++lane) {
        if (!isfinite(spectrum.values[lane])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] __device__ bool nonnegative_spectrum(const TransportSpectrum& spectrum) noexcept {
    if (!finite_spectrum(spectrum)) {
        return false;
    }
    for (auto lane = std::uint32_t{0U}; lane < shared::HostDeviceSpectrumLaneCount; ++lane) {
        if (spectrum.values[lane] < 0.0F) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] __device__ bool zero_spectrum(const TransportSpectrum& spectrum) noexcept {
    for (auto lane = std::uint32_t{0U}; lane < shared::HostDeviceSpectrumLaneCount; ++lane) {
        if (spectrum.values[lane] != 0.0F) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] __device__ bool valid_ray(const TransportRay& ray) noexcept {
    const auto direction = Vector3{
        .x = ray.direction_x,
        .y = ray.direction_y,
        .z = ray.direction_z,
    };
    return isfinite(ray.origin_x) && isfinite(ray.origin_y) && isfinite(ray.origin_z) &&
           isfinite(ray.t_min) && ray.t_min >= 0.0F && !isnan(ray.t_max) &&
           ray.t_max >= ray.t_min && isfinite(ray.time) && ray.time >= 0.0F && ray.time <= 1.0F &&
           finite_vector(direction) && dot(direction, direction) > 0.0F && ray.reserved == 0U;
}

[[nodiscard]] __device__ bool valid_path_state(const TransportPathStateLane& state) noexcept {
    if (!nonnegative_spectrum(state.beta) || !nonnegative_spectrum(state.accumulated_radiance) ||
        !isfinite(state.eta_scale) || !(state.eta_scale > 0.0F)) {
        return false;
    }
    for (auto lane = std::uint32_t{0U}; lane < shared::HostDeviceSpectrumLaneCount; ++lane) {
        if (!isfinite(state.wavelength_nanometers.values[lane]) ||
            state.wavelength_nanometers.values[lane] < 360.0F ||
            state.wavelength_nanometers.values[lane] > 830.0F ||
            !isfinite(state.wavelength_pdf_values.values[lane]) ||
            !(state.wavelength_pdf_values.values[lane] > 0.0F) ||
            state.wavelength_pdf_measures[lane] != WavelengthMeasure) {
            return false;
        }
    }
    for (auto word = std::uint32_t{0U}; word < 6U; ++word) {
        if (state.reserved[word] != 0U) {
            return false;
        }
    }
    const auto total_depth = static_cast<std::uint64_t>(state.diffuse_depth) +
                             static_cast<std::uint64_t>(state.glossy_depth) +
                             static_cast<std::uint64_t>(state.specular_depth) +
                             static_cast<std::uint64_t>(state.transmission_depth) +
                             static_cast<std::uint64_t>(state.volume_depth);
    if (total_depth != state.depth || (state.delta_flags & ~3U) != 0U) {
        return false;
    }
    const auto previous_was_delta = (state.delta_flags & 1U) != 0U;
    const auto has_non_delta_history = (state.delta_flags & 2U) != 0U;
    const auto non_delta_depth = static_cast<std::uint64_t>(state.diffuse_depth) +
                                 static_cast<std::uint64_t>(state.glossy_depth) +
                                 static_cast<std::uint64_t>(state.volume_depth);
    if ((state.depth == 0U && state.delta_flags != 0U) ||
        (state.depth == 1U && previous_was_delta && has_non_delta_history) ||
        (state.depth != 0U && !previous_was_delta && !has_non_delta_history) ||
        ((non_delta_depth != 0U) != has_non_delta_history) ||
        (previous_was_delta && state.specular_depth == 0U) ||
        (state.depth == 1U && previous_was_delta && state.specular_depth != 1U) ||
        (state.depth == 1U && !previous_was_delta && state.specular_depth != 0U)) {
        return false;
    }
    return true;
}

[[nodiscard]] __device__ bool valid_queue_view(const WavefrontQueueDeviceSoa queues) noexcept {
    return queues.headers != nullptr && queues.queue_count == cuda::CudaWavefrontQueueCount &&
           (queues.slot_stride == 0U || queues.path_slots != nullptr);
}

[[nodiscard]] __device__ bool valid_stream_view(const WavefrontStageDeviceSoa streams) noexcept {
    return streams.reserved[0] == 0U && streams.reserved[1] == 0U && streams.reserved[2] == 0U &&
           (streams.capacity == 0U ||
            (streams.sample_streams != nullptr && streams.rays != nullptr &&
             streams.path_states != nullptr && streams.hits != nullptr &&
             streams.pending_shadows != nullptr && streams.controls != nullptr));
}

[[nodiscard]] __device__ WavefrontStageStatus queue_slot(const WavefrontQueueDeviceSoa queues,
                                                         const std::uint32_t queue_kind,
                                                         const std::uint32_t work_index,
                                                         const std::uint32_t work_count,
                                                         const std::uint32_t path_capacity,
                                                         PathSlot& slot) noexcept {
    if (!valid_queue_view(queues) || queue_kind >= cuda::CudaWavefrontQueueCount) {
        return WavefrontStageStatus::invalid_contract;
    }
    const auto& header = queues.headers[queue_kind];
    if (!cuda::wavefront_queue_device_detail::immutable_header_contract_is_valid(
            header, queue_kind, queues.slot_stride) ||
        header.size != work_count || header.size > header.capacity || work_index >= work_count) {
        return WavefrontStageStatus::invalid_contract;
    }
    if (header.overflow_count != 0U || header.rejected_count != 0U) {
        return WavefrontStageStatus::queue_overflow;
    }
    const auto source = static_cast<std::uint64_t>(queue_kind) * queues.slot_stride + work_index;
    slot = queues.path_slots[source];
    return slot.value < path_capacity ? WavefrontStageStatus::success
                                      : WavefrontStageStatus::invalid_path_slot;
}

[[nodiscard]] __device__ bool claim_phase(WavefrontLaneControl& control,
                                          const WavefrontLanePhase expected) noexcept {
    if (control.reserved != 0U) {
        return false;
    }
    return atomicCAS(&control.phase, value(expected), value(WavefrontLanePhase::processing)) ==
           value(expected);
}

__device__ void
finish_phase(WavefrontLaneControl& control, const WavefrontLanePhase phase,
             const WavefrontTermination termination = WavefrontTermination::none) noexcept {
    control.termination = value(termination);
    __threadfence();
    atomicExch(&control.phase, value(phase));
}

[[nodiscard]] __device__ WavefrontStageStatus push_route(const WavefrontQueueDeviceSoa queues,
                                                         const std::uint32_t queue_kind,
                                                         const PathSlot slot) noexcept {
    const auto pushed = cuda::try_push_wavefront_queue(queues, queue_kind, slot);
    if (pushed == WavefrontQueueDevicePushStatus::pushed) {
        return WavefrontStageStatus::success;
    }
    return pushed == WavefrontQueueDevicePushStatus::capacity_exhausted
               ? WavefrontStageStatus::queue_overflow
               : WavefrontStageStatus::invalid_contract;
}

[[nodiscard]] __device__ std::uint64_t mix_bits(std::uint64_t bits) noexcept {
    bits ^= bits >> 30U;
    bits *= 0xBF58476D1CE4E5B9ULL;
    bits ^= bits >> 27U;
    bits *= 0x94D049BB133111EBULL;
    return bits ^ (bits >> 31U);
}

[[nodiscard]] __device__ float sample_1d(const SampleStreamIndex& index,
                                         const std::uint64_t dimension) noexcept {
    const auto packed_pixel = (static_cast<std::uint64_t>(index.pixel_x) << 32U) |
                              static_cast<std::uint64_t>(index.pixel_y);
    auto state = mix_bits(index.seed ^ 0x9E3779B97F4A7C15ULL);
    state = mix_bits(state ^ packed_pixel ^ 0xD1B54A32D192ED03ULL);
    state = mix_bits(state ^ index.sample_index ^ 0x8CB92BA72F3D8DD7ULL);
    return static_cast<float>(mix_bits(state ^ dimension) >> 40U) * 0x1p-24F;
}

[[nodiscard]] __device__ constexpr std::uint64_t align_up(const std::uint64_t input,
                                                          const std::uint64_t alignment) noexcept {
    const auto remainder = input % alignment;
    return remainder == 0U ? input : input + (alignment - remainder);
}

[[nodiscard]] __device__ const SceneSoaColumnDescriptor&
scene_descriptor(const SceneSoaHeader* const header, const std::uint32_t column) noexcept {
    const auto* const descriptors = reinterpret_cast<const SceneSoaColumnDescriptor*>(
        reinterpret_cast<const std::uint8_t*>(header) + offsetof(SceneSoaHeader, columns));
    return descriptors[column];
}

[[nodiscard]] __device__ std::uint64_t scene_column_count(const SceneSoaHeader& header,
                                                          const std::uint32_t column) noexcept {
    if (column == scene_column::object_id) {
        return header.object_count;
    }
    if (column >= scene_column::geometry_id && column <= scene_column::geometry_triangle_count) {
        return header.geometry_count;
    }
    if (column >= scene_column::position_x && column <= scene_column::texture_coordinate_y) {
        return header.vertex_count;
    }
    if (column >= scene_column::triangle_vertex_0 && column <= scene_column::triangle_vertex_2) {
        return header.triangle_count;
    }
    if (column >= scene_column::material_id && column < scene_column::instance_id) {
        return header.material_count;
    }
    if (column >= scene_column::instance_id && column < scene_column::punctual_kind) {
        return header.instance_count;
    }
    if (column >= scene_column::punctual_kind &&
        column < scene_column::mesh_area_light_instance_id) {
        return header.punctual_light_count;
    }
    if (column == scene_column::mesh_area_light_instance_id) {
        return header.mesh_area_light_count;
    }
    if (column >= scene_column::environment_wavelength_nanometers && column < scene_column::count) {
        return header.environment_count;
    }
    return 0U;
}

[[nodiscard]] __device__ std::uint32_t
scene_column_element_size(const std::uint32_t column) noexcept {
    if (column >= scene_column::count) {
        return 0U;
    }
    if (column >= scene_column::geometry_vertex_offset &&
        column <= scene_column::geometry_triangle_count) {
        return sizeof(std::uint64_t);
    }
    if (column == scene_column::material_spectral_present ||
        (column >= scene_column::material_wavelength_measure &&
         column < scene_column::material_reflectance) ||
        column == scene_column::instance_parent_present ||
        (column >= scene_column::environment_wavelength_measure &&
         column < scene_column::environment_radiance)) {
        return sizeof(std::uint8_t);
    }
    if ((column >= scene_column::position_x && column <= scene_column::texture_coordinate_y) ||
        (column >= scene_column::material_wavelength_nanometers &&
         column < scene_column::material_wavelength_measure) ||
        (column >= scene_column::material_reflectance && column < scene_column::instance_id) ||
        (column >= scene_column::instance_local_to_parent &&
         column < scene_column::punctual_kind) ||
        (column >= scene_column::punctual_position_x &&
         column < scene_column::mesh_area_light_instance_id) ||
        (column >= scene_column::environment_wavelength_nanometers &&
         column < scene_column::environment_wavelength_measure) ||
        (column >= scene_column::environment_radiance && column < scene_column::count)) {
        return sizeof(float);
    }
    return sizeof(std::uint32_t);
}

[[nodiscard]] __device__ SceneDeviceStatus validate_scene(const std::uint8_t* const bytes,
                                                          const std::size_t byte_count) noexcept {
    if (bytes == nullptr || byte_count < sizeof(SceneSoaHeader)) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto* const header = reinterpret_cast<const SceneSoaHeader*>(bytes);
    if (header->magic != shared::SceneSoaMagic || header->abi_major != shared::SceneSoaAbiMajor ||
        header->abi_minor != shared::SceneSoaAbiMinor ||
        header->header_size != sizeof(SceneSoaHeader) ||
        header->column_count != scene_column::count ||
        header->hash_algorithm != shared::SceneSoaHashAlgorithmFnv1a64 ||
        header->environment_count > 1U || header->total_size_bytes != byte_count ||
        header->total_size_bytes < sizeof(SceneSoaHeader) || header->object_count > 0xFFFFFFFFULL ||
        header->geometry_count > 0xFFFFFFFFULL || header->vertex_count > 0xFFFFFFFFULL ||
        header->triangle_count > 0xFFFFFFFFULL || header->material_count > 0xFFFFFFFFULL ||
        header->instance_count > 0xFFFFFFFFULL || header->mesh_area_light_count > 0xFFFFFFFFULL ||
        header->punctual_light_count > 0xFFFFFFFFULL) {
        return SceneDeviceStatus::invalid_scene;
    }
    if (header->mesh_area_light_count > MaximumUniformLightCount) {
        return SceneDeviceStatus::unsupported_transport;
    }
    const auto* const reserved =
        reinterpret_cast<const std::uint64_t*>(bytes + offsetof(SceneSoaHeader, reserved));
    for (auto index = std::uint32_t{0U}; index < 5U; ++index) {
        if (reserved[index] != 0U) {
            return SceneDeviceStatus::invalid_scene;
        }
    }
    auto cursor = align_up(sizeof(SceneSoaHeader), shared::SceneSoaColumnAlignment);
    for (auto column = std::uint32_t{0U}; column < scene_column::count; ++column) {
        const auto& descriptor = scene_descriptor(header, column);
        const auto expected_count = scene_column_count(*header, column);
        const auto expected_size = scene_column_element_size(column);
        if (descriptor.element_count != expected_count ||
            descriptor.element_size != expected_size || descriptor.reserved != 0U) {
            return SceneDeviceStatus::invalid_scene;
        }
        if (expected_count == 0U) {
            if (descriptor.offset_bytes != 0U) {
                return SceneDeviceStatus::invalid_scene;
            }
            continue;
        }
        if (cursor > MaximumU64 - (shared::SceneSoaColumnAlignment - 1U)) {
            return SceneDeviceStatus::invalid_scene;
        }
        cursor = align_up(cursor, shared::SceneSoaColumnAlignment);
        if (descriptor.offset_bytes != cursor || expected_count > MaximumU64 / expected_size) {
            return SceneDeviceStatus::invalid_scene;
        }
        const auto column_bytes = expected_count * expected_size;
        if (cursor > MaximumU64 - column_bytes) {
            return SceneDeviceStatus::invalid_scene;
        }
        cursor += column_bytes;
    }
    return cursor == header->total_size_bytes ? SceneDeviceStatus::valid
                                              : SceneDeviceStatus::invalid_scene;
}

template <class Element>
[[nodiscard]] __device__ const Element* scene_values(const std::uint8_t* const bytes,
                                                     const SceneSoaHeader& header,
                                                     const std::uint32_t column) noexcept {
    return reinterpret_cast<const Element*>(bytes + scene_descriptor(&header, column).offset_bytes);
}

[[nodiscard]] __device__ bool find_id(const std::uint32_t* const ids, const std::uint32_t count,
                                      const std::uint32_t id, std::uint32_t& index) noexcept {
    for (auto candidate = std::uint32_t{0U}; candidate < count; ++candidate) {
        if (ids[candidate] == id) {
            index = candidate;
            return true;
        }
    }
    return false;
}

[[nodiscard]] __device__ float matrix_value(const std::uint8_t* const bytes,
                                            const SceneSoaHeader& header,
                                            const std::uint32_t first_column,
                                            const std::uint32_t instance,
                                            const std::uint32_t element) noexcept {
    return scene_values<float>(bytes, header, first_column + element)[instance];
}

[[nodiscard]] __device__ bool load_matrix(const std::uint8_t* const bytes,
                                          const SceneSoaHeader& header,
                                          const std::uint32_t first_column,
                                          const std::uint32_t instance,
                                          float* const matrix) noexcept {
    for (auto element = std::uint32_t{0U}; element < shared::SceneSoaMatrixElementCount;
         ++element) {
        matrix[element] = matrix_value(bytes, header, first_column, instance, element);
        if (!isfinite(matrix[element])) {
            return false;
        }
    }
    return matrix[12] == 0.0F && matrix[13] == 0.0F && matrix[14] == 0.0F && matrix[15] == 1.0F;
}

[[nodiscard]] __device__ Vector3 transform_point(const float* const matrix,
                                                 const Vector3 point) noexcept {
    return Vector3{
        .x =
            fmaf(matrix[0], point.x, fmaf(matrix[1], point.y, fmaf(matrix[2], point.z, matrix[3]))),
        .y =
            fmaf(matrix[4], point.x, fmaf(matrix[5], point.y, fmaf(matrix[6], point.z, matrix[7]))),
        .z = fmaf(matrix[8], point.x,
                  fmaf(matrix[9], point.y, fmaf(matrix[10], point.z, matrix[11]))),
    };
}

[[nodiscard]] __device__ float vector_component(const Vector3 vector,
                                                const std::uint32_t axis) noexcept {
    return axis == 0U ? vector.x : (axis == 1U ? vector.y : vector.z);
}

__device__ void set_vector_component(Vector3& vector, const std::uint32_t axis,
                                     const float value) noexcept {
    if (axis == 0U) {
        vector.x = value;
    } else if (axis == 1U) {
        vector.y = value;
    } else {
        vector.z = value;
    }
}

[[nodiscard]] __device__ bool
interpolate_position_with_error(const Vector3* const vertices, const Vector3 barycentrics,
                                PositionWithError& positioned) noexcept {
    auto triangle_scale = 0.0F;
    for (auto axis = std::uint32_t{0U}; axis < 3U; ++axis) {
        const auto first = vector_component(vertices[0], axis);
        triangle_scale =
            fmaxf(triangle_scale, fmaxf(fabsf(vector_component(vertices[1], axis) - first),
                                        fabsf(vector_component(vertices[2], axis) - first)));
    }
    const auto representable_floor = FloatEpsilon * triangle_scale;
    if (!isfinite(triangle_scale) || !isfinite(representable_floor) ||
        !(representable_floor > 0.0F)) {
        return false;
    }

    for (auto axis = std::uint32_t{0U}; axis < 3U; ++axis) {
        const auto product0 = barycentrics.x * vector_component(vertices[0], axis);
        const auto product1 = barycentrics.y * vector_component(vertices[1], axis);
        const auto product2 = barycentrics.z * vector_component(vertices[2], axis);
        const auto position =
            fmaf(barycentrics.x, vector_component(vertices[0], axis),
                 fmaf(barycentrics.y, vector_component(vertices[1], axis), product2));
        const auto magnitude = fabsf(product0) + fabsf(product1) + fabsf(product2);
        const auto error = fmaxf(Gamma7 * magnitude, representable_floor);
        if (!isfinite(position) || !isfinite(magnitude) || !isfinite(error) || error < 0.0F) {
            return false;
        }
        set_vector_component(positioned.position, axis, position);
        set_vector_component(positioned.absolute_error, axis, error);
    }
    return true;
}

[[nodiscard]] __device__ bool
transform_position_with_error(const float* const matrix, const PositionWithError& object_position,
                              const Vector3 traversal_position,
                              PositionWithError& positioned) noexcept {
    positioned.position = transform_point(matrix, object_position.position);
    if (!finite_vector(positioned.position) || !finite_vector(traversal_position)) {
        return false;
    }
    for (auto row = std::uint32_t{0U}; row < 3U; ++row) {
        auto transformed_magnitude = fabs(static_cast<double>(matrix[row * 4U + 3U]));
        auto propagated_error = 0.0;
        for (auto column = std::uint32_t{0U}; column < 3U; ++column) {
            const auto coefficient = fabs(static_cast<double>(matrix[row * 4U + column]));
            transformed_magnitude +=
                coefficient *
                fabs(static_cast<double>(vector_component(object_position.position, column)));
            propagated_error +=
                coefficient *
                static_cast<double>(vector_component(object_position.absolute_error, column));
        }
        const auto discrepancy =
            fabs(static_cast<double>(vector_component(positioned.position, row)) -
                 static_cast<double>(vector_component(traversal_position, row)));
        const auto bound = Gamma7Double * transformed_magnitude +
                           (1.0 + Gamma7Double) * propagated_error + discrepancy;
        if (!isfinite(bound) || bound < 0.0 || bound > static_cast<double>(MaximumFloat)) {
            return false;
        }
        auto rounded = static_cast<float>(bound);
        if (static_cast<double>(rounded) < bound) {
            rounded = nextafterf(rounded, __int_as_float(0x7F800000));
        }
        if (!isfinite(rounded) || rounded < 0.0F || (bound > 0.0 && rounded == 0.0F)) {
            return false;
        }
        set_vector_component(positioned.absolute_error, row, rounded);
    }
    return true;
}

[[nodiscard]] __device__ Vector3 transform_normal_from_inverse(const float* const inverse,
                                                               const Vector3 normal) noexcept {
    return Vector3{
        .x = fmaf(inverse[0], normal.x, fmaf(inverse[4], normal.y, inverse[8] * normal.z)),
        .y = fmaf(inverse[1], normal.x, fmaf(inverse[5], normal.y, inverse[9] * normal.z)),
        .z = fmaf(inverse[2], normal.x, fmaf(inverse[6], normal.y, inverse[10] * normal.z)),
    };
}

[[nodiscard]] __device__ Vector3 ray_point(const TransportRay& ray,
                                           const float parameter) noexcept {
    return Vector3{
        .x = fmaf(parameter, ray.direction_x, ray.origin_x),
        .y = fmaf(parameter, ray.direction_y, ray.origin_y),
        .z = fmaf(parameter, ray.direction_z, ray.origin_z),
    };
}

[[nodiscard]] __device__ bool
spectra_match(const TransportPathStateLane& state, const std::uint8_t* const bytes,
              const SceneSoaHeader& scene, const std::uint32_t wavelength_column,
              const std::uint32_t pdf_column, const std::uint32_t measure_column,
              const std::uint32_t index) noexcept {
    for (auto lane = std::uint32_t{0U}; lane < shared::HostDeviceSpectrumLaneCount; ++lane) {
        const auto wavelength = scene_values<float>(bytes, scene, wavelength_column + lane)[index];
        const auto pdf = scene_values<float>(bytes, scene, pdf_column + lane)[index];
        const auto measure = scene_values<std::uint8_t>(bytes, scene, measure_column + lane)[index];
        if (!isfinite(wavelength) || !isfinite(pdf) || !(wavelength > 0.0F) || !(pdf > 0.0F) ||
            wavelength != state.wavelength_nanometers.values[lane] ||
            pdf != state.wavelength_pdf_values.values[lane] ||
            measure != state.wavelength_pdf_measures[lane] || measure != WavelengthMeasure) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] __device__ bool load_spectrum(const std::uint8_t* const bytes,
                                            const SceneSoaHeader& scene,
                                            const std::uint32_t first_column,
                                            const std::uint32_t index,
                                            TransportSpectrum& spectrum) noexcept {
    for (auto lane = std::uint32_t{0U}; lane < shared::HostDeviceSpectrumLaneCount; ++lane) {
        spectrum.values[lane] = scene_values<float>(bytes, scene, first_column + lane)[index];
    }
    return nonnegative_spectrum(spectrum);
}

[[nodiscard]] __device__ bool checked_product(const float left, const float right,
                                              float& product) noexcept {
    if (!isfinite(left) || left < 0.0F || !isfinite(right) || right < 0.0F) {
        return false;
    }
    product = left * right;
    return isfinite(product) && product >= 0.0F &&
           !(left != 0.0F && right != 0.0F && product == 0.0F);
}

template <std::size_t NumeratorCount, std::size_t DenominatorCount>
[[nodiscard]] __device__ bool
checked_product_quotient(const float (&numerators)[NumeratorCount],
                         const float (&denominators)[DenominatorCount], float& result) noexcept {
    auto significand = 1.0F;
    auto exponent = 0;
    for (const auto numerator : numerators) {
        if (!isfinite(numerator) || numerator < 0.0F) {
            return false;
        }
        if (numerator == 0.0F) {
            result = 0.0F;
            return true;
        }
        auto factor_exponent = 0;
        significand *= frexpf(numerator, &factor_exponent);
        exponent += factor_exponent;
    }
    for (const auto denominator : denominators) {
        if (!isfinite(denominator) || !(denominator > 0.0F)) {
            return false;
        }
        auto factor_exponent = 0;
        significand /= frexpf(denominator, &factor_exponent);
        exponent -= factor_exponent;
    }
    if (!isfinite(significand) || !(significand > 0.0F)) {
        return false;
    }
    auto normalization_exponent = 0;
    significand = frexpf(significand, &normalization_exponent);
    exponent += normalization_exponent;
    result = scalbnf(significand, exponent);
    return isfinite(result) && result > 0.0F;
}

[[nodiscard]] __device__ bool add_product(TransportSpectrum& destination,
                                          const TransportSpectrum& first,
                                          const TransportSpectrum& second) noexcept {
    auto updated = destination;
    for (auto lane = std::uint32_t{0U}; lane < shared::HostDeviceSpectrumLaneCount; ++lane) {
        auto contribution = 0.0F;
        if (!checked_product(first.values[lane], second.values[lane], contribution)) {
            return false;
        }
        updated.values[lane] += contribution;
        if (!isfinite(updated.values[lane]) || updated.values[lane] < 0.0F) {
            return false;
        }
    }
    destination = updated;
    return true;
}

[[nodiscard]] __device__ bool add_spectrum(TransportSpectrum& destination,
                                           const TransportSpectrum& contribution) noexcept {
    for (auto lane = std::uint32_t{0U}; lane < shared::HostDeviceSpectrumLaneCount; ++lane) {
        destination.values[lane] += contribution.values[lane];
        if (!isfinite(destination.values[lane]) || destination.values[lane] < 0.0F) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] __device__ bool valid_hit(const ClosestHit& hit) noexcept {
    if (!isfinite(hit.parameter) || !(hit.parameter >= 0.0F) ||
        !isfinite(hit.barycentric_vertex0) || !isfinite(hit.barycentric_vertex1) ||
        !isfinite(hit.barycentric_vertex2) || hit.barycentric_vertex0 < 0.0F ||
        hit.barycentric_vertex1 < 0.0F || hit.barycentric_vertex2 < 0.0F || hit.reserved != 0U) {
        return false;
    }
    const auto barycentric_sum =
        hit.barycentric_vertex0 + hit.barycentric_vertex1 + hit.barycentric_vertex2;
    if (!isfinite(barycentric_sum) || fabsf(barycentric_sum - 1.0F) > 2.0e-4F ||
        !finite_vector(Vector3{.x = hit.geometric_normal_x,
                               .y = hit.geometric_normal_y,
                               .z = hit.geometric_normal_z})) {
        return false;
    }
    for (auto word = std::uint32_t{0U}; word < 3U; ++word) {
        if (hit.identifiers.reserved[word] != 0U) {
            return false;
        }
    }
    return true;
}

__global__ void seed_camera_kernel(const WavefrontQueueDeviceSoa queues,
                                   const WavefrontStageDeviceSoa streams,
                                   const std::uint32_t first_path_slot,
                                   const std::uint32_t path_count,
                                   WavefrontStageOutcome* const outcomes) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= path_count) {
        return;
    }
    const auto wide_slot = static_cast<std::uint64_t>(first_path_slot) + index;
    const auto diagnostic_slot =
        wide_slot <= 0xFFFFFFFFULL ? static_cast<std::uint32_t>(wide_slot) : 0xFFFFFFFFU;
    if (!valid_stream_view(streams) || wide_slot >= streams.capacity) {
        outcomes[index] = outcome(WavefrontStageStatus::invalid_path_slot,
                                  WavefrontStageRoute::none, diagnostic_slot);
        return;
    }
    const auto slot = PathSlot{.value = diagnostic_slot};
    auto& control = streams.controls[slot.value];
    if (!claim_phase(control, WavefrontLanePhase::empty)) {
        outcomes[index] = outcome(WavefrontStageStatus::invalid_lane_state,
                                  WavefrontStageRoute::none, slot.value, control.phase);
        return;
    }
    control.flags = 0U;
    const auto routed = push_route(queues, CameraQueue, slot);
    if (routed != WavefrontStageStatus::success) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(routed, WavefrontStageRoute::none, slot.value, CameraQueue);
        return;
    }
    finish_phase(control, WavefrontLanePhase::camera);
    outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::none, slot.value);
}

__global__ void clear_queue_kernel(const WavefrontQueueDeviceSoa queues,
                                   const std::uint32_t queue_kind,
                                   const std::uint32_t acknowledge_overflow,
                                   std::uint32_t* const device_status) {
    auto status = WavefrontStageStatus::success;
    if (!valid_queue_view(queues) || queue_kind >= cuda::CudaWavefrontQueueCount ||
        acknowledge_overflow > 1U) {
        status = WavefrontStageStatus::invalid_contract;
    } else {
        auto& header = queues.headers[queue_kind];
        if (!cuda::wavefront_queue_device_detail::immutable_header_contract_is_valid(
                header, queue_kind, queues.slot_stride) ||
            header.size > header.capacity) {
            status = WavefrontStageStatus::invalid_contract;
        } else if (acknowledge_overflow == 0U &&
                   (header.overflow_count != 0U || header.rejected_count != 0U)) {
            status = WavefrontStageStatus::queue_overflow;
        } else {
            header.size = 0U;
            header.overflow_count = 0U;
            header.rejected_count = 0U;
        }
    }
    *device_status = value(status);
}

__global__ void camera_stage_kernel(const WavefrontQueueDeviceSoa queues,
                                    const WavefrontCameraInputDeviceSoa inputs,
                                    const WavefrontStageDeviceSoa streams,
                                    const std::uint32_t work_count,
                                    WavefrontStageOutcome* const outcomes) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= work_count) {
        return;
    }
    auto slot = PathSlot{};
    const auto queue_status = queue_slot(queues, CameraQueue, static_cast<std::uint32_t>(index),
                                         work_count, streams.capacity, slot);
    if (queue_status != WavefrontStageStatus::success) {
        outcomes[index] = outcome(queue_status, WavefrontStageRoute::none, slot.value, CameraQueue);
        return;
    }
    auto& control = streams.controls[slot.value];
    if (!claim_phase(control, WavefrontLanePhase::camera)) {
        outcomes[index] = outcome(WavefrontStageStatus::invalid_lane_state,
                                  WavefrontStageRoute::none, slot.value, control.phase);
        return;
    }
    if (inputs.reserved != 0U || inputs.sample_streams == nullptr || inputs.rays == nullptr ||
        inputs.path_states == nullptr || slot.value >= inputs.count) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::invalid_contract, WavefrontStageRoute::none, slot.value);
        return;
    }
    const auto ray = inputs.rays[slot.value];
    const auto state = inputs.path_states[slot.value];
    if (!valid_ray(ray) || !valid_path_state(state) || ray.current_medium != state.current_medium) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(!valid_ray(ray) ? WavefrontStageStatus::invalid_ray
                                                  : WavefrontStageStatus::invalid_lane_state,
                                  WavefrontStageRoute::none, slot.value);
        return;
    }
    if (state.current_medium != 0U) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(WavefrontStageStatus::unsupported_transport,
                                  WavefrontStageRoute::none, slot.value, state.current_medium);
        return;
    }
    streams.sample_streams[slot.value] = inputs.sample_streams[slot.value];
    streams.rays[slot.value] = ray;
    streams.path_states[slot.value] = state;
    streams.hits[slot.value] = ClosestHit{};
    streams.pending_shadows[slot.value] = WavefrontPendingShadow{};
    control.flags = 0U;
    const auto routed = push_route(queues, RayQueue, slot);
    if (routed != WavefrontStageStatus::success) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(routed, WavefrontStageRoute::none, slot.value, RayQueue);
        return;
    }
    finish_phase(control, WavefrontLanePhase::ray);
    outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::ray, slot.value);
}

__global__ void
gather_rays_kernel(const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
                   const std::uint32_t work_count, PathSlot* const compact_path_slots,
                   TransportRay* const compact_rays, WavefrontStageOutcome* const outcomes) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= work_count) {
        return;
    }
    auto slot = PathSlot{};
    const auto queue_status = queue_slot(queues, RayQueue, static_cast<std::uint32_t>(index),
                                         work_count, streams.capacity, slot);
    if (queue_status != WavefrontStageStatus::success) {
        outcomes[index] = outcome(queue_status, WavefrontStageRoute::none, slot.value, RayQueue);
        return;
    }
    if (streams.controls[slot.value].phase != value(WavefrontLanePhase::ray) ||
        !valid_ray(streams.rays[slot.value])) {
        outcomes[index] =
            outcome(WavefrontStageStatus::invalid_lane_state, WavefrontStageRoute::none, slot.value,
                    streams.controls[slot.value].phase);
        return;
    }
    compact_path_slots[index] = slot;
    compact_rays[index] = streams.rays[slot.value];
    outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::ray, slot.value);
}

__global__ void classify_closest_hit_kernel(const WavefrontQueueDeviceSoa queues,
                                            const WavefrontStageDeviceSoa streams,
                                            const PathSlot* const compact_path_slots,
                                            const SceneClosestHitResult* const compact_results,
                                            const std::uint32_t work_count,
                                            WavefrontStageOutcome* const outcomes) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= work_count) {
        return;
    }
    auto queue_path = PathSlot{};
    const auto queue_status = queue_slot(queues, RayQueue, static_cast<std::uint32_t>(index),
                                         work_count, streams.capacity, queue_path);
    const auto compact_path = compact_path_slots[index];
    if (queue_status != WavefrontStageStatus::success || compact_path.value != queue_path.value) {
        outcomes[index] = outcome(queue_status == WavefrontStageStatus::success
                                      ? WavefrontStageStatus::invalid_contract
                                      : queue_status,
                                  WavefrontStageRoute::none, queue_path.value, RayQueue);
        return;
    }
    auto& control = streams.controls[queue_path.value];
    if (!claim_phase(control, WavefrontLanePhase::ray)) {
        outcomes[index] = outcome(WavefrontStageStatus::invalid_lane_state,
                                  WavefrontStageRoute::none, queue_path.value, control.phase);
        return;
    }
    const auto result = compact_results[index];
    const auto* const result_reserved = reinterpret_cast<const std::uint32_t*>(
        reinterpret_cast<const std::uint8_t*>(&result) + offsetof(SceneClosestHitResult, reserved));
    for (auto word = std::uint32_t{0U}; word < 3U; ++word) {
        if (result_reserved[word] != 0U) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] = outcome(WavefrontStageStatus::invalid_contract,
                                      WavefrontStageRoute::none, queue_path.value);
            return;
        }
    }
    auto destination = MissQueue;
    auto route = WavefrontStageRoute::miss;
    auto phase = WavefrontLanePhase::miss;
    if (result.status == static_cast<std::uint32_t>(SceneClosestHitStatus::hit)) {
        if (!valid_hit(result.hit)) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] = outcome(WavefrontStageStatus::numerical_failure,
                                      WavefrontStageRoute::none, queue_path.value);
            return;
        }
        streams.hits[queue_path.value] = result.hit;
        destination = HitQueue;
        route = WavefrontStageRoute::hit;
        phase = WavefrontLanePhase::hit;
    } else if (result.status != static_cast<std::uint32_t>(SceneClosestHitStatus::miss)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(WavefrontStageStatus::traversal_error, WavefrontStageRoute::none,
                                  queue_path.value, result.status);
        return;
    }
    const auto routed = push_route(queues, destination, queue_path);
    if (routed != WavefrontStageStatus::success) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(routed, WavefrontStageRoute::none, queue_path.value, destination);
        return;
    }
    finish_phase(control, phase);
    outcomes[index] = outcome(WavefrontStageStatus::success, route, queue_path.value);
}

__global__ void hit_stage_kernel(const WavefrontQueueDeviceSoa queues,
                                 const WavefrontStageDeviceSoa streams,
                                 const std::uint32_t work_count,
                                 WavefrontStageOutcome* const outcomes) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= work_count) {
        return;
    }
    auto slot = PathSlot{};
    const auto queue_status = queue_slot(queues, HitQueue, static_cast<std::uint32_t>(index),
                                         work_count, streams.capacity, slot);
    if (queue_status != WavefrontStageStatus::success) {
        outcomes[index] = outcome(queue_status, WavefrontStageRoute::none, slot.value, HitQueue);
        return;
    }
    auto& control = streams.controls[slot.value];
    if (!claim_phase(control, WavefrontLanePhase::hit) || !valid_hit(streams.hits[slot.value])) {
        if (control.phase == value(WavefrontLanePhase::processing)) {
            finish_phase(control, WavefrontLanePhase::terminated);
        }
        outcomes[index] = outcome(WavefrontStageStatus::invalid_lane_state,
                                  WavefrontStageRoute::none, slot.value, control.phase);
        return;
    }
    const auto routed = push_route(queues, ShadeQueue, slot);
    if (routed != WavefrontStageStatus::success) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(routed, WavefrontStageRoute::none, slot.value, ShadeQueue);
        return;
    }
    finish_phase(control, WavefrontLanePhase::shade);
    outcomes[index] =
        outcome(WavefrontStageStatus::success, WavefrontStageRoute::shade, slot.value);
}

[[nodiscard]] __device__ SceneDeviceStatus surface_data(
    const std::uint8_t* const bytes, const SceneSoaHeader& scene, const TransportRay& ray,
    const ClosestHit& hit, const TransportPathStateLane& state, SurfaceData& surface) noexcept {
    std::uint32_t instance = 0U;
    std::uint32_t geometry = 0U;
    std::uint32_t material = 0U;
    if (!find_id(scene_values<std::uint32_t>(bytes, scene, scene_column::instance_id),
                 static_cast<std::uint32_t>(scene.instance_count), hit.identifiers.instance,
                 instance) ||
        !find_id(scene_values<std::uint32_t>(bytes, scene, scene_column::geometry_id),
                 static_cast<std::uint32_t>(scene.geometry_count), hit.identifiers.geometry,
                 geometry) ||
        !find_id(scene_values<std::uint32_t>(bytes, scene, scene_column::material_id),
                 static_cast<std::uint32_t>(scene.material_count), hit.identifiers.material,
                 material)) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto* const instance_geometry =
        scene_values<std::uint32_t>(bytes, scene, scene_column::instance_geometry_id);
    const auto* const instance_material =
        scene_values<std::uint32_t>(bytes, scene, scene_column::instance_material_id);
    const auto* const instance_object =
        scene_values<std::uint32_t>(bytes, scene, scene_column::instance_object_id);
    if (instance_geometry[instance] != hit.identifiers.geometry ||
        instance_material[instance] != hit.identifiers.material ||
        instance_object[instance] != hit.identifiers.object ||
        scene_values<std::uint8_t>(bytes, scene,
                                   scene_column::material_spectral_present)[material] != 1U ||
        !spectra_match(state, bytes, scene, scene_column::material_wavelength_nanometers,
                       scene_column::material_wavelength_pdf,
                       scene_column::material_wavelength_measure, material)) {
        return SceneDeviceStatus::unsupported_transport;
    }
    const auto vertex_offset =
        scene_values<std::uint64_t>(bytes, scene, scene_column::geometry_vertex_offset)[geometry];
    const auto vertex_count =
        scene_values<std::uint64_t>(bytes, scene, scene_column::geometry_vertex_count)[geometry];
    const auto triangle_offset =
        scene_values<std::uint64_t>(bytes, scene, scene_column::geometry_triangle_offset)[geometry];
    const auto triangle_count =
        scene_values<std::uint64_t>(bytes, scene, scene_column::geometry_triangle_count)[geometry];
    if (hit.identifiers.primitive >= triangle_count || vertex_offset > scene.vertex_count ||
        vertex_count > scene.vertex_count - vertex_offset ||
        triangle_offset > scene.triangle_count ||
        triangle_count > scene.triangle_count - triangle_offset) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto triangle = triangle_offset + hit.identifiers.primitive;
    Vector3 local_normal{};
    const auto barycentrics = Vector3{
        .x = hit.barycentric_vertex0, .y = hit.barycentric_vertex1, .z = hit.barycentric_vertex2};
    Vector3 local_vertices[3U]{};
    for (auto corner = std::uint32_t{0U}; corner < 3U; ++corner) {
        const auto local_vertex = scene_values<std::uint32_t>(
            bytes, scene, scene_column::triangle_vertex_0 + corner)[triangle];
        if (local_vertex >= vertex_count) {
            return SceneDeviceStatus::invalid_scene;
        }
        const auto global_vertex = vertex_offset + local_vertex;
        local_vertices[corner] = Vector3{
            .x = scene_values<float>(bytes, scene, scene_column::position_x)[global_vertex],
            .y = scene_values<float>(bytes, scene, scene_column::position_y)[global_vertex],
            .z = scene_values<float>(bytes, scene, scene_column::position_z)[global_vertex],
        };
        const auto weight =
            corner == 0U ? barycentrics.x : (corner == 1U ? barycentrics.y : barycentrics.z);
        const auto vertex_normal =
            Vector3{.x = scene_values<float>(bytes, scene, scene_column::normal_x)[global_vertex],
                    .y = scene_values<float>(bytes, scene, scene_column::normal_y)[global_vertex],
                    .z = scene_values<float>(bytes, scene, scene_column::normal_z)[global_vertex]};
        if (!finite_vector(local_vertices[corner]) || !finite_vector(vertex_normal)) {
            return SceneDeviceStatus::numerical_failure;
        }
        local_normal = add(local_normal, multiply(vertex_normal, weight));
    }
    float inverse[shared::SceneSoaMatrixElementCount]{};
    float matrix[shared::SceneSoaMatrixElementCount]{};
    if (!load_matrix(bytes, scene, scene_column::instance_world_to_local, instance, inverse) ||
        !load_matrix(bytes, scene, scene_column::instance_local_to_world, instance, matrix)) {
        return SceneDeviceStatus::invalid_scene;
    }
    auto shading_normal = Vector3{};
    auto geometric_normal = Vector3{
        .x = hit.geometric_normal_x,
        .y = hit.geometric_normal_y,
        .z = hit.geometric_normal_z,
    };
    if (!normalize(transform_normal_from_inverse(inverse, local_normal), shading_normal) ||
        !normalize(geometric_normal, geometric_normal)) {
        return SceneDeviceStatus::numerical_failure;
    }
    if (dot(shading_normal, geometric_normal) < 0.0F) {
        shading_normal = multiply(shading_normal, -1.0F);
    }
    if (!(dot(shading_normal, geometric_normal) > 0.0F)) {
        return SceneDeviceStatus::invalid_scene;
    }
    auto object_position = PositionWithError{};
    auto world_position = PositionWithError{};
    if (!interpolate_position_with_error(local_vertices, barycentrics, object_position) ||
        !transform_position_with_error(matrix, object_position, ray_point(ray, hit.parameter),
                                       world_position)) {
        return SceneDeviceStatus::numerical_failure;
    }
    surface = SurfaceData{
        .position = world_position.position,
        .position_error = world_position.absolute_error,
        .geometric_normal = geometric_normal,
        .shading_normal = shading_normal,
        .material_index = material,
    };
    return finite_vector(surface.position) && finite_vector(surface.position_error)
               ? SceneDeviceStatus::valid
               : SceneDeviceStatus::numerical_failure;
}

[[nodiscard]] __device__ bool load_triangle_local(const std::uint8_t* const bytes,
                                                  const SceneSoaHeader& scene,
                                                  const std::uint32_t geometry,
                                                  const std::uint32_t local_triangle,
                                                  Vector3* const vertices) noexcept {
    const auto vertex_offset =
        scene_values<std::uint64_t>(bytes, scene, scene_column::geometry_vertex_offset)[geometry];
    const auto vertex_count =
        scene_values<std::uint64_t>(bytes, scene, scene_column::geometry_vertex_count)[geometry];
    const auto triangle_offset =
        scene_values<std::uint64_t>(bytes, scene, scene_column::geometry_triangle_offset)[geometry];
    const auto triangle_count =
        scene_values<std::uint64_t>(bytes, scene, scene_column::geometry_triangle_count)[geometry];
    if (local_triangle >= triangle_count || vertex_offset > scene.vertex_count ||
        vertex_count > scene.vertex_count - vertex_offset ||
        triangle_offset > scene.triangle_count ||
        triangle_count > scene.triangle_count - triangle_offset) {
        return false;
    }
    const auto triangle = triangle_offset + local_triangle;
    for (auto corner = std::uint32_t{0U}; corner < 3U; ++corner) {
        const auto local_vertex = scene_values<std::uint32_t>(
            bytes, scene, scene_column::triangle_vertex_0 + corner)[triangle];
        if (local_vertex >= vertex_count) {
            return false;
        }
        const auto global_vertex = vertex_offset + local_vertex;
        vertices[corner] = Vector3{
            .x = scene_values<float>(bytes, scene, scene_column::position_x)[global_vertex],
            .y = scene_values<float>(bytes, scene, scene_column::position_y)[global_vertex],
            .z = scene_values<float>(bytes, scene, scene_column::position_z)[global_vertex]};
        if (!finite_vector(vertices[corner])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] __device__ bool
load_triangle_world(const std::uint8_t* const bytes, const SceneSoaHeader& scene,
                    const std::uint32_t geometry, const std::uint32_t instance,
                    const std::uint32_t local_triangle, Vector3* const vertices) noexcept {
    Vector3 local_vertices[3U]{};
    float matrix[shared::SceneSoaMatrixElementCount]{};
    if (!load_triangle_local(bytes, scene, geometry, local_triangle, local_vertices) ||
        !load_matrix(bytes, scene, scene_column::instance_local_to_world, instance, matrix)) {
        return false;
    }
    for (auto corner = std::uint32_t{0U}; corner < 3U; ++corner) {
        vertices[corner] = transform_point(matrix, local_vertices[corner]);
        if (!finite_vector(vertices[corner])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] __device__ bool triangle_area(const Vector3* const vertices, float& area,
                                            Vector3* const normal = nullptr) noexcept {
    const auto crossed =
        cross(subtract(vertices[1], vertices[0]), subtract(vertices[2], vertices[0]));
    const auto squared_length = dot(crossed, crossed);
    if (!isfinite(squared_length) || !(squared_length > 0.0F)) {
        return false;
    }
    const auto length = sqrtf(squared_length);
    area = 0.5F * length;
    if (!isfinite(area) || !(area > 0.0F)) {
        return false;
    }
    if (normal != nullptr) {
        *normal = multiply(crossed, 1.0F / length);
    }
    return true;
}

[[nodiscard]] __device__ SceneDeviceStatus
sample_mesh_area_light(const std::uint8_t* const bytes, const SceneSoaHeader& scene,
                       const TransportPathStateLane& state, const SampleStreamIndex& sample_index,
                       const std::uint32_t bounce, LightEndpoint& endpoint) noexcept {
    if (scene.punctual_light_count != 0U) {
        return SceneDeviceStatus::unsupported_transport;
    }
    if (scene.mesh_area_light_count == 0U) {
        endpoint = LightEndpoint{};
        return SceneDeviceStatus::valid;
    }
    const auto dimension =
        FirstBounceDimension + static_cast<std::uint64_t>(bounce) * DimensionsPerBounce;
    const auto light_selection = sample_1d(sample_index, dimension);
    const auto area_sample = sample_1d(sample_index, dimension + 1U);
    const auto triangle_sample = sample_1d(sample_index, dimension + 2U);
    const auto light_count = static_cast<std::uint32_t>(scene.mesh_area_light_count);
    const auto light_index =
        static_cast<std::uint32_t>(light_selection * static_cast<float>(light_count));
    if (light_index >= light_count) {
        return SceneDeviceStatus::numerical_failure;
    }
    const auto light_instance_id = scene_values<std::uint32_t>(
        bytes, scene, scene_column::mesh_area_light_instance_id)[light_index];
    std::uint32_t instance = 0U;
    if (!find_id(scene_values<std::uint32_t>(bytes, scene, scene_column::instance_id),
                 static_cast<std::uint32_t>(scene.instance_count), light_instance_id, instance)) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto geometry_id =
        scene_values<std::uint32_t>(bytes, scene, scene_column::instance_geometry_id)[instance];
    const auto material_id =
        scene_values<std::uint32_t>(bytes, scene, scene_column::instance_material_id)[instance];
    std::uint32_t geometry = 0U;
    std::uint32_t material = 0U;
    if (!find_id(scene_values<std::uint32_t>(bytes, scene, scene_column::geometry_id),
                 static_cast<std::uint32_t>(scene.geometry_count), geometry_id, geometry) ||
        !find_id(scene_values<std::uint32_t>(bytes, scene, scene_column::material_id),
                 static_cast<std::uint32_t>(scene.material_count), material_id, material)) {
        return SceneDeviceStatus::invalid_scene;
    }
    if (scene_values<std::uint8_t>(bytes, scene,
                                   scene_column::material_spectral_present)[material] != 1U ||
        !spectra_match(state, bytes, scene, scene_column::material_wavelength_nanometers,
                       scene_column::material_wavelength_pdf,
                       scene_column::material_wavelength_measure, material)) {
        return SceneDeviceStatus::unsupported_transport;
    }
    const auto triangle_count = static_cast<std::uint32_t>(
        scene_values<std::uint64_t>(bytes, scene, scene_column::geometry_triangle_count)[geometry]);
    if (triangle_count == 0U) {
        return SceneDeviceStatus::invalid_scene;
    }
    auto area_sum = 0.0F;
    auto area_compensation = 0.0F;
    for (auto triangle = std::uint32_t{0U}; triangle < triangle_count; ++triangle) {
        Vector3 vertices[3U]{};
        auto area = 0.0F;
        if (!load_triangle_world(bytes, scene, geometry, instance, triangle, vertices) ||
            !triangle_area(vertices, area)) {
            return SceneDeviceStatus::numerical_failure;
        }
        const auto next_area = area_sum + area;
        if (!isfinite(next_area)) {
            return SceneDeviceStatus::numerical_failure;
        }
        area_compensation += fabsf(area_sum) >= fabsf(area) ? (area_sum - next_area) + area
                                                            : (area - next_area) + area_sum;
        if (!isfinite(area_compensation)) {
            return SceneDeviceStatus::numerical_failure;
        }
        area_sum = next_area;
    }
    const auto total_area = area_sum + area_compensation;
    const auto inverse_total_area = 1.0F / total_area;
    if (!isfinite(total_area) || !(total_area > 0.0F) || !isfinite(inverse_total_area) ||
        !(inverse_total_area > 0.0F)) {
        return SceneDeviceStatus::numerical_failure;
    }

    auto cumulative_probability = 0.0F;
    auto selected_previous_probability = 0.0F;
    auto selected_probability = 0.0F;
    auto selected = triangle_count;
    auto selected_area = 0.0F;
    Vector3 selected_vertices[3U]{};
    for (auto triangle = std::uint32_t{0U}; triangle < triangle_count; ++triangle) {
        Vector3 vertices[3U]{};
        auto area = 0.0F;
        if (!load_triangle_world(bytes, scene, geometry, instance, triangle, vertices) ||
            !triangle_area(vertices, area)) {
            return SceneDeviceStatus::numerical_failure;
        }
        auto probability = 1.0F - cumulative_probability;
        auto next_probability = 1.0F;
        if (triangle + 1U != triangle_count) {
            probability = area / total_area;
            next_probability = cumulative_probability + probability;
            if (!isfinite(probability) || !(probability > 0.0F) || !isfinite(next_probability) ||
                !(next_probability > cumulative_probability) || !(next_probability < 1.0F)) {
                return SceneDeviceStatus::numerical_failure;
            }
        } else if (!isfinite(probability) || !(probability > 0.0F) || probability > 1.0F) {
            return SceneDeviceStatus::numerical_failure;
        }
        if (selected == triangle_count && area_sample < next_probability) {
            selected = triangle;
            selected_area = area;
            selected_previous_probability = cumulative_probability;
            selected_probability = probability;
            selected_vertices[0] = vertices[0];
            selected_vertices[1] = vertices[1];
            selected_vertices[2] = vertices[2];
        }
        cumulative_probability = next_probability;
    }
    if (selected >= triangle_count || !isfinite(selected_area) || !(selected_area > 0.0F) ||
        !isfinite(selected_probability) || !(selected_probability > 0.0F)) {
        return SceneDeviceStatus::numerical_failure;
    }
    const auto remapped = (area_sample - selected_previous_probability) / selected_probability;
    if (!isfinite(remapped) || remapped < 0.0F || !(remapped < 1.0F)) {
        return SceneDeviceStatus::numerical_failure;
    }
    const auto root = sqrtf(remapped);
    const auto b0 = 1.0F - root;
    const auto b1 = root * (1.0F - triangle_sample);
    const auto b2 = root * triangle_sample;
    const auto barycentrics = Vector3{.x = b0, .y = b1, .z = b2};
    auto normal = Vector3{};
    auto ignored_area = 0.0F;
    if (!triangle_area(selected_vertices, ignored_area, &normal)) {
        return SceneDeviceStatus::numerical_failure;
    }
    const auto area_density = selected_probability / selected_area;
    if (!isfinite(area_density) || !(area_density > 0.0F)) {
        return SceneDeviceStatus::numerical_failure;
    }
    auto radiance = TransportSpectrum{};
    if (!load_spectrum(bytes, scene, scene_column::material_emitted_radiance, material, radiance)) {
        return SceneDeviceStatus::numerical_failure;
    }
    const auto traversal_position =
        add(add(multiply(selected_vertices[0], b0), multiply(selected_vertices[1], b1)),
            multiply(selected_vertices[2], b2));
    Vector3 local_vertices[3U]{};
    float matrix[shared::SceneSoaMatrixElementCount]{};
    auto object_position = PositionWithError{};
    auto world_position = PositionWithError{};
    if (!load_triangle_local(bytes, scene, geometry, selected, local_vertices) ||
        !load_matrix(bytes, scene, scene_column::instance_local_to_world, instance, matrix) ||
        !interpolate_position_with_error(local_vertices, barycentrics, object_position) ||
        !transform_position_with_error(matrix, object_position, traversal_position,
                                       world_position)) {
        return SceneDeviceStatus::numerical_failure;
    }
    endpoint = LightEndpoint{
        .radiance = radiance,
        .position = world_position.position,
        .position_error = world_position.absolute_error,
        .geometric_normal = normal,
        .area_density = area_density,
        .visibility_mask = scene_values<std::uint32_t>(
            bytes, scene, scene_column::instance_visibility_mask)[instance],
        .reserved = {0U, 0U, 0U},
    };
    return finite_vector(endpoint.position) && finite_vector(endpoint.position_error)
               ? SceneDeviceStatus::valid
               : SceneDeviceStatus::numerical_failure;
}

[[nodiscard]] __device__ bool offset_point(const Vector3 point, const Vector3 position_error,
                                           const Vector3 normal, const Vector3 outgoing,
                                           Vector3& offset) noexcept {
    if (!finite_vector(point) || !finite_vector(position_error) || position_error.x < 0.0F ||
        position_error.y < 0.0F || position_error.z < 0.0F || !finite_vector(normal) ||
        !finite_vector(outgoing)) {
        return false;
    }
    const auto oriented = dot(normal, outgoing) >= 0.0F ? normal : multiply(normal, -1.0F);
    const auto offset_distance = fabsf(normal.x) * position_error.x +
                                 fabsf(normal.y) * position_error.y +
                                 fabsf(normal.z) * position_error.z;
    if (!isfinite(offset_distance) || !(offset_distance > 0.0F)) {
        return false;
    }
    const auto displacement = multiply(oriented, offset_distance);
    const auto positive_infinity = __int_as_float(0x7F800000);
    const auto negative_infinity = __int_as_float(0xFF800000);
    offset = add(point, displacement);
    if (displacement.x != 0.0F) {
        offset.x =
            nextafterf(offset.x, displacement.x > 0.0F ? positive_infinity : negative_infinity);
    }
    if (displacement.y != 0.0F) {
        offset.y =
            nextafterf(offset.y, displacement.y > 0.0F ? positive_infinity : negative_infinity);
    }
    if (displacement.z != 0.0F) {
        offset.z =
            nextafterf(offset.z, displacement.z > 0.0F ? positive_infinity : negative_infinity);
    }
    return finite_vector(offset) &&
           (offset.x != point.x || offset.y != point.y || offset.z != point.z);
}

[[nodiscard]] __device__ bool cosine_direction(const Vector3 normal, const float u, const float v,
                                               float& local_cosine, Vector3& direction) noexcept {
    const auto offset_x = 2.0F * u - 1.0F;
    const auto offset_y = 2.0F * v - 1.0F;
    auto signed_radius = 0.0F;
    auto azimuth = 0.0F;
    constexpr auto quarter_pi = Pi * 0.25F;
    if (offset_x != 0.0F || offset_y != 0.0F) {
        if (fabsf(offset_x) > fabsf(offset_y)) {
            signed_radius = offset_x;
            azimuth = quarter_pi * (offset_y / offset_x);
        } else {
            signed_radius = offset_y;
            azimuth = 2.0F * quarter_pi - quarter_pi * (offset_x / offset_y);
        }
    }
    const auto radius = fabsf(signed_radius);
    const auto local_x = signed_radius * cosf(azimuth);
    const auto local_y = signed_radius * sinf(azimuth);
    local_cosine = sqrtf(fmaxf(0.0F, (1.0F - radius) * (1.0F + radius)));

    const auto sign = copysignf(1.0F, normal.z);
    const auto coefficient = -1.0F / (sign + normal.z);
    const auto product = normal.x * normal.y * coefficient;
    auto tangent_seed = Vector3{
        .x = 1.0F + sign * normal.x * normal.x * coefficient,
        .y = sign * product,
        .z = -sign * normal.x,
    };
    auto tangent = Vector3{};
    if (!normalize(tangent_seed, tangent)) {
        return false;
    }
    tangent_seed = subtract(tangent, multiply(normal, dot(tangent, normal)));
    if (!normalize(tangent_seed, tangent)) {
        return false;
    }
    auto bitangent = Vector3{};
    if (!normalize(cross(normal, tangent), bitangent) ||
        !normalize(cross(bitangent, normal), tangent)) {
        return false;
    }
    return normalize(add(add(multiply(tangent, local_x), multiply(bitangent, local_y)),
                         multiply(normal, local_cosine)),
                     direction);
}

[[nodiscard]] __device__ WavefrontStageStatus
scene_status(const SceneDeviceStatus status) noexcept {
    switch (status) {
    case SceneDeviceStatus::valid:
        return WavefrontStageStatus::success;
    case SceneDeviceStatus::invalid_scene:
        return WavefrontStageStatus::invalid_scene;
    case SceneDeviceStatus::unsupported_transport:
        return WavefrontStageStatus::unsupported_transport;
    case SceneDeviceStatus::numerical_failure:
        return WavefrontStageStatus::numerical_failure;
    }
    return WavefrontStageStatus::invalid_scene;
}

__global__ void
miss_stage_kernel(const std::uint8_t* const scene_bytes, const std::size_t scene_size,
                  const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
                  const std::uint32_t work_count, WavefrontStageOutcome* const outcomes) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= work_count) {
        return;
    }
    auto slot = PathSlot{};
    const auto queue_status = queue_slot(queues, MissQueue, static_cast<std::uint32_t>(index),
                                         work_count, streams.capacity, slot);
    if (queue_status != WavefrontStageStatus::success) {
        outcomes[index] = outcome(queue_status, WavefrontStageRoute::none, slot.value, MissQueue);
        return;
    }
    auto& control = streams.controls[slot.value];
    if (!claim_phase(control, WavefrontLanePhase::miss)) {
        outcomes[index] = outcome(WavefrontStageStatus::invalid_lane_state,
                                  WavefrontStageRoute::none, slot.value, control.phase);
        return;
    }
    const auto validation = validate_scene(scene_bytes, scene_size);
    if (validation != SceneDeviceStatus::valid) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(scene_status(validation), WavefrontStageRoute::none, slot.value);
        return;
    }
    const auto& scene = *reinterpret_cast<const SceneSoaHeader*>(scene_bytes);
    auto& state = streams.path_states[slot.value];
    if (!valid_path_state(state)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(WavefrontStageStatus::invalid_lane_state,
                                  WavefrontStageRoute::none, slot.value);
        return;
    }
    if (scene.environment_count == 1U) {
        if (!spectra_match(state, scene_bytes, scene,
                           scene_column::environment_wavelength_nanometers,
                           scene_column::environment_wavelength_pdf,
                           scene_column::environment_wavelength_measure, 0U)) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] = outcome(WavefrontStageStatus::unsupported_transport,
                                      WavefrontStageRoute::none, slot.value);
            return;
        }
        auto radiance = TransportSpectrum{};
        if (!load_spectrum(scene_bytes, scene, scene_column::environment_radiance, 0U, radiance) ||
            !add_product(state.accumulated_radiance, state.beta, radiance)) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] = outcome(WavefrontStageStatus::numerical_failure,
                                      WavefrontStageRoute::none, slot.value);
            return;
        }
    }
    control.flags = 0U;
    finish_phase(control, WavefrontLanePhase::terminated,
                 WavefrontTermination::escaped_environment);
    outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::terminated,
                              slot.value, value(WavefrontTermination::escaped_environment));
}

__global__ void
shade_stage_kernel(const std::uint8_t* const scene_bytes, const std::size_t scene_size,
                   const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
                   const std::uint32_t max_diffuse_depth, const std::uint32_t work_count,
                   WavefrontStageOutcome* const outcomes) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= work_count) {
        return;
    }
    auto slot = PathSlot{};
    const auto queue_status = queue_slot(queues, ShadeQueue, static_cast<std::uint32_t>(index),
                                         work_count, streams.capacity, slot);
    if (queue_status != WavefrontStageStatus::success) {
        outcomes[index] = outcome(queue_status, WavefrontStageRoute::none, slot.value, ShadeQueue);
        return;
    }
    auto& control = streams.controls[slot.value];
    if (!claim_phase(control, WavefrontLanePhase::shade)) {
        outcomes[index] = outcome(WavefrontStageStatus::invalid_lane_state,
                                  WavefrontStageRoute::none, slot.value, control.phase);
        return;
    }
    const auto validation = validate_scene(scene_bytes, scene_size);
    if (validation != SceneDeviceStatus::valid) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(scene_status(validation), WavefrontStageRoute::none, slot.value);
        return;
    }
    const auto& scene = *reinterpret_cast<const SceneSoaHeader*>(scene_bytes);
    auto& state = streams.path_states[slot.value];
    auto& ray = streams.rays[slot.value];
    const auto& hit = streams.hits[slot.value];
    if (!valid_path_state(state) || !valid_ray(ray) || !valid_hit(hit) ||
        ray.current_medium != state.current_medium) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(WavefrontStageStatus::invalid_lane_state,
                                  WavefrontStageRoute::none, slot.value);
        return;
    }
    auto surface = SurfaceData{};
    const auto loaded = surface_data(scene_bytes, scene, ray, hit, state, surface);
    if (loaded != SceneDeviceStatus::valid) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(scene_status(loaded), WavefrontStageRoute::none, slot.value,
                                  value(ShadeFailureDetail::surface_data));
        return;
    }
    const auto incoming = Vector3{
        .x = -ray.direction_x,
        .y = -ray.direction_y,
        .z = -ray.direction_z,
    };
    const auto front_facing = dot(surface.geometric_normal, incoming) > 0.0F;
    auto emitted = TransportSpectrum{};
    auto reflectance = TransportSpectrum{};
    if (!load_spectrum(scene_bytes, scene, scene_column::material_emitted_radiance,
                       surface.material_index, emitted) ||
        !load_spectrum(scene_bytes, scene, scene_column::material_reflectance,
                       surface.material_index, reflectance)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::material_spectrum));
        return;
    }
    // Mesh emitters are sampled explicitly below. Until complementary MIS weights are ported,
    // only camera-visible emission is accumulated here so the same light path is never counted
    // through both NEE and a diffuse continuation.
    if (front_facing && state.depth == 0U &&
        !add_product(state.accumulated_radiance, state.beta, emitted)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::emitted_radiance));
        return;
    }
    if (state.diffuse_depth >= max_diffuse_depth) {
        control.flags = 0U;
        finish_phase(control, WavefrontLanePhase::terminated,
                     WavefrontTermination::diffuse_depth_limit);
        outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::terminated,
                                  slot.value, value(WavefrontTermination::diffuse_depth_limit));
        return;
    }
    if (state.diffuse_depth == 0xFFFFFFFFU || state.depth == 0xFFFFFFFFU) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(WavefrontStageStatus::invalid_lane_state,
                                  WavefrontStageRoute::none, slot.value);
        return;
    }
    if (zero_spectrum(state.beta)) {
        control.flags = 0U;
        finish_phase(control, WavefrontLanePhase::terminated,
                     WavefrontTermination::zero_throughput);
        outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::terminated,
                                  slot.value, value(WavefrontTermination::zero_throughput));
        return;
    }
    if (!front_facing || !(dot(surface.shading_normal, incoming) > 0.0F)) {
        control.flags = 0U;
        finish_phase(control, WavefrontLanePhase::terminated,
                     WavefrontTermination::outside_bsdf_support);
        outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::terminated,
                                  slot.value, value(WavefrontTermination::outside_bsdf_support));
        return;
    }

    auto endpoint = LightEndpoint{};
    auto sampled_light = SceneDeviceStatus::valid;
    auto shade_detail = std::uint32_t{};
    if (scene.mesh_area_light_count != 0U && !zero_spectrum(state.beta) &&
        !zero_spectrum(reflectance)) {
        shade_detail = cuda::WavefrontShadeDetailLightSampled;
        sampled_light = sample_mesh_area_light(
            scene_bytes, scene, state, streams.sample_streams[slot.value], state.depth, endpoint);
    }
    if (sampled_light != SceneDeviceStatus::valid) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(scene_status(sampled_light), WavefrontStageRoute::none,
                                  slot.value, value(ShadeFailureDetail::light_sample));
        return;
    }

    auto pending = WavefrontPendingShadow{};
    auto has_shadow = false;
    if (scene.mesh_area_light_count != 0U &&
        (ray.visibility_mask & endpoint.visibility_mask) != 0U) {
        auto light_segment = ScaledSegment{};
        if (!scaled_segment(subtract(endpoint.position, surface.position), light_segment)) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] =
                outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                        slot.value, value(ShadeFailureDetail::light_distance));
            return;
        }
        const auto direction_to_light = light_segment.direction;
        auto absolute_light_alignment = 0.0F;
        auto light_supported = false;
        if (!light_surface_alignment(endpoint.geometric_normal, light_segment.scaled,
                                     absolute_light_alignment, light_supported)) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] =
                outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                        slot.value, value(ShadeFailureDetail::light_weight));
            return;
        }
        if (light_supported) {
            const auto light_count = static_cast<float>(scene.mesh_area_light_count);
            const auto selection_probability = 1.0F / light_count;
            const float pdf_numerators[]{endpoint.area_density, light_segment.scale,
                                         light_segment.scale, light_segment.scaled_distance_cubed};
            const float pdf_denominators[]{absolute_light_alignment};
            auto solid_angle_pdf = 0.0F;
            if (!isfinite(selection_probability) || !(selection_probability > 0.0F) ||
                !checked_product_quotient(pdf_numerators, pdf_denominators, solid_angle_pdf)) {
                finish_phase(control, WavefrontLanePhase::terminated);
                outcomes[index] =
                    outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                            slot.value, value(ShadeFailureDetail::light_weight));
                return;
            }
            const auto receiver_cosine = dot(surface.shading_normal, direction_to_light);
            const auto geometric_receiver_cosine =
                dot(surface.geometric_normal, direction_to_light);
            if (receiver_cosine > 0.0F && geometric_receiver_cosine > 0.0F) {
                for (auto lane = std::uint32_t{0U}; lane < shared::HostDeviceSpectrumLaneCount;
                     ++lane) {
                    const float contribution_numerators[]{
                        state.beta.values[lane], reflectance.values[lane], InversePi,
                        endpoint.radiance.values[lane], receiver_cosine};
                    const float contribution_denominators[]{selection_probability, solid_angle_pdf};
                    if (!checked_product_quotient(contribution_numerators,
                                                  contribution_denominators,
                                                  pending.visible_contribution.values[lane])) {
                        finish_phase(control, WavefrontLanePhase::terminated);
                        outcomes[index] = outcome(WavefrontStageStatus::numerical_failure,
                                                  WavefrontStageRoute::none, slot.value,
                                                  value(ShadeFailureDetail::light_contribution));
                        return;
                    }
                }
                if (!zero_spectrum(pending.visible_contribution)) {
                    auto source_offset = Vector3{};
                    auto light_offset = Vector3{};
                    if (!offset_point(surface.position, surface.position_error,
                                      surface.geometric_normal, direction_to_light,
                                      source_offset) ||
                        !offset_point(endpoint.position, endpoint.position_error,
                                      endpoint.geometric_normal,
                                      multiply(direction_to_light, -1.0F), light_offset)) {
                        finish_phase(control, WavefrontLanePhase::terminated);
                        outcomes[index] = outcome(WavefrontStageStatus::numerical_failure,
                                                  WavefrontStageRoute::none, slot.value,
                                                  value(ShadeFailureDetail::endpoint_offset));
                        return;
                    }
                    auto shadow_segment = ScaledSegment{};
                    if (!scaled_segment(subtract(light_offset, source_offset), shadow_segment) ||
                        !(dot(shadow_segment.direction, direction_to_light) > 0.0F)) {
                        finish_phase(control, WavefrontLanePhase::terminated);
                        outcomes[index] = outcome(WavefrontStageStatus::numerical_failure,
                                                  WavefrontStageRoute::none, slot.value,
                                                  value(ShadeFailureDetail::shadow_segment));
                        return;
                    }
                    const auto maximum = nextafterf(shadow_segment.length, 0.0F);
                    pending.ray = TransportRay{
                        .origin_x = source_offset.x,
                        .origin_y = source_offset.y,
                        .origin_z = source_offset.z,
                        .t_min = 0.0F,
                        .direction_x = shadow_segment.direction.x,
                        .direction_y = shadow_segment.direction.y,
                        .direction_z = shadow_segment.direction.z,
                        .t_max = maximum,
                        .time = ray.time,
                        .visibility_mask = ray.visibility_mask,
                        .current_medium = state.current_medium,
                        .reserved = 0U,
                    };
                    pending.continuation_pending = 0U;
                    pending.termination = value(WavefrontTermination::none);
                    if (!valid_ray(pending.ray) || !(maximum > 0.0F)) {
                        finish_phase(control, WavefrontLanePhase::terminated);
                        outcomes[index] = outcome(WavefrontStageStatus::numerical_failure,
                                                  WavefrontStageRoute::none, slot.value,
                                                  value(ShadeFailureDetail::shadow_ray));
                        return;
                    }
                    has_shadow = true;
                }
            }
        }
    }

    const auto bsdf_dimension =
        FirstBounceDimension + static_cast<std::uint64_t>(state.depth) * DimensionsPerBounce;
    const auto bsdf_u = sample_1d(streams.sample_streams[slot.value], bsdf_dimension + 4U);
    const auto bsdf_v = sample_1d(streams.sample_streams[slot.value], bsdf_dimension + 5U);
    auto outgoing = Vector3{};
    auto local_cosine = 0.0F;
    auto next_origin = Vector3{};
    if (!cosine_direction(surface.shading_normal, bsdf_u, bsdf_v, local_cosine, outgoing)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::bsdf_sample));
        return;
    }
    if (!(local_cosine > 0.0F) || !(dot(surface.geometric_normal, outgoing) > 0.0F)) {
        if (!has_shadow) {
            streams.pending_shadows[slot.value] = WavefrontPendingShadow{};
            control.flags = 0U;
            finish_phase(control, WavefrontLanePhase::terminated,
                         WavefrontTermination::outside_bsdf_support);
            outcomes[index] =
                outcome(WavefrontStageStatus::success, WavefrontStageRoute::terminated, slot.value,
                        value(WavefrontTermination::outside_bsdf_support) | shade_detail);
            return;
        }
        pending.continuation_pending = 0U;
        pending.termination = value(WavefrontTermination::outside_bsdf_support);
        streams.pending_shadows[slot.value] = pending;
        control.flags = cuda::WavefrontLaneShadowPending;
        const auto routed = push_route(queues, ShadowQueue, slot);
        if (routed != WavefrontStageStatus::success) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] = outcome(routed, WavefrontStageRoute::none, slot.value, ShadowQueue);
            return;
        }
        finish_phase(control, WavefrontLanePhase::shadow);
        outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::shadow,
                                  slot.value, shade_detail);
        return;
    }
    if (!offset_point(surface.position, surface.position_error, surface.geometric_normal, outgoing,
                      next_origin)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::continuation_offset));
        return;
    }
    auto updated_beta = state.beta;
    for (auto lane = std::uint32_t{0U}; lane < shared::HostDeviceSpectrumLaneCount; ++lane) {
        if (!checked_product(state.beta.values[lane], reflectance.values[lane],
                             updated_beta.values[lane])) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] =
                outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                        slot.value, value(ShadeFailureDetail::throughput));
            return;
        }
    }
    state.beta = updated_beta;
    state.diffuse_depth += 1U;
    state.depth += 1U;
    // A Lambertian event clears only the previous-delta bit and preserves non-delta history.
    state.delta_flags = (state.delta_flags | 2U) & ~1U;
    ray = TransportRay{
        .origin_x = next_origin.x,
        .origin_y = next_origin.y,
        .origin_z = next_origin.z,
        .t_min = 0.0F,
        .direction_x = outgoing.x,
        .direction_y = outgoing.y,
        .direction_z = outgoing.z,
        .t_max = __int_as_float(0x7F800000),
        .time = ray.time,
        .visibility_mask = ray.visibility_mask,
        .current_medium = state.current_medium,
        .reserved = 0U,
    };
    if (!valid_ray(ray)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::continuation_ray));
        return;
    }
    if (zero_spectrum(state.beta)) {
        control.flags = 0U;
        finish_phase(control, WavefrontLanePhase::terminated,
                     WavefrontTermination::zero_throughput);
        outcomes[index] =
            outcome(WavefrontStageStatus::success, WavefrontStageRoute::terminated, slot.value,
                    value(WavefrontTermination::zero_throughput) | shade_detail);
        return;
    }

    auto destination = ContinuationQueue;
    auto route = WavefrontStageRoute::continuation;
    auto phase = WavefrontLanePhase::continuation;
    if (has_shadow) {
        pending.continuation_pending = 1U;
        pending.termination = value(WavefrontTermination::none);
        streams.pending_shadows[slot.value] = pending;
        control.flags = cuda::WavefrontLaneShadowPending | cuda::WavefrontLaneContinuationPending;
        destination = ShadowQueue;
        route = WavefrontStageRoute::shadow;
        phase = WavefrontLanePhase::shadow;
    } else {
        streams.pending_shadows[slot.value] = WavefrontPendingShadow{};
        control.flags = 0U;
    }
    const auto routed = push_route(queues, destination, slot);
    if (routed != WavefrontStageStatus::success) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(routed, WavefrontStageRoute::none, slot.value, destination);
        return;
    }
    finish_phase(control, phase);
    outcomes[index] = outcome(WavefrontStageStatus::success, route, slot.value, shade_detail);
}

__global__ void gather_shadow_rays_kernel(const WavefrontQueueDeviceSoa queues,
                                          const WavefrontStageDeviceSoa streams,
                                          const std::uint32_t work_count,
                                          PathSlot* const compact_path_slots,
                                          TransportRay* const compact_rays,
                                          WavefrontStageOutcome* const outcomes) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= work_count) {
        return;
    }
    auto slot = PathSlot{};
    const auto queue_status = queue_slot(queues, ShadowQueue, static_cast<std::uint32_t>(index),
                                         work_count, streams.capacity, slot);
    if (queue_status != WavefrontStageStatus::success) {
        outcomes[index] = outcome(queue_status, WavefrontStageRoute::none, slot.value, ShadowQueue);
        return;
    }
    const auto& control = streams.controls[slot.value];
    const auto& pending = streams.pending_shadows[slot.value];
    const auto known_flags =
        cuda::WavefrontLaneShadowPending | cuda::WavefrontLaneContinuationPending;
    const auto continuation_pending = pending.continuation_pending == 1U;
    if (control.phase != value(WavefrontLanePhase::shadow) || control.reserved != 0U ||
        (control.flags & cuda::WavefrontLaneShadowPending) == 0U ||
        (control.flags & ~known_flags) != 0U || pending.continuation_pending > 1U ||
        ((control.flags & cuda::WavefrontLaneContinuationPending) != 0U) != continuation_pending ||
        (continuation_pending && pending.termination != value(WavefrontTermination::none)) ||
        (!continuation_pending &&
         pending.termination != value(WavefrontTermination::outside_bsdf_support)) ||
        pending.reserved[0] != 0U || pending.reserved[1] != 0U || !valid_ray(pending.ray) ||
        !nonnegative_spectrum(pending.visible_contribution) ||
        zero_spectrum(pending.visible_contribution)) {
        outcomes[index] = outcome(WavefrontStageStatus::invalid_lane_state,
                                  WavefrontStageRoute::none, slot.value, control.phase);
        return;
    }
    compact_path_slots[index] = slot;
    compact_rays[index] = pending.ray;
    outcomes[index] =
        outcome(WavefrontStageStatus::success, WavefrontStageRoute::shadow, slot.value);
}

__global__ void process_shadow_kernel(const WavefrontQueueDeviceSoa queues,
                                      const WavefrontStageDeviceSoa streams,
                                      const PathSlot* const compact_path_slots,
                                      const SceneOcclusionResult* const compact_results,
                                      const std::uint32_t work_count,
                                      WavefrontStageOutcome* const outcomes) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= work_count) {
        return;
    }
    auto queue_path = PathSlot{};
    const auto queue_status = queue_slot(queues, ShadowQueue, static_cast<std::uint32_t>(index),
                                         work_count, streams.capacity, queue_path);
    const auto compact_path = compact_path_slots[index];
    if (queue_status != WavefrontStageStatus::success || compact_path.value != queue_path.value) {
        outcomes[index] = outcome(queue_status == WavefrontStageStatus::success
                                      ? WavefrontStageStatus::invalid_contract
                                      : queue_status,
                                  WavefrontStageRoute::none, queue_path.value, ShadowQueue);
        return;
    }
    auto& control = streams.controls[queue_path.value];
    if (!claim_phase(control, WavefrontLanePhase::shadow)) {
        outcomes[index] = outcome(WavefrontStageStatus::invalid_lane_state,
                                  WavefrontStageRoute::none, queue_path.value, control.phase);
        return;
    }
    const auto result = compact_results[index];
    const auto* const result_reserved = reinterpret_cast<const std::uint32_t*>(
        reinterpret_cast<const std::uint8_t*>(&result) + offsetof(SceneOcclusionResult, reserved));
    if (result_reserved[0] != 0U || result_reserved[1] != 0U || result_reserved[2] != 0U) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(WavefrontStageStatus::invalid_contract, WavefrontStageRoute::none,
                                  queue_path.value);
        return;
    }
    const auto visible = result.status == static_cast<std::uint32_t>(SceneOcclusionStatus::visible);
    const auto occluded =
        result.status == static_cast<std::uint32_t>(SceneOcclusionStatus::occluded);
    if (!visible && !occluded) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(WavefrontStageStatus::traversal_error, WavefrontStageRoute::none,
                                  queue_path.value, result.status);
        return;
    }
    auto& pending = streams.pending_shadows[queue_path.value];
    const auto known_flags =
        cuda::WavefrontLaneShadowPending | cuda::WavefrontLaneContinuationPending;
    const auto continuation_pending = pending.continuation_pending == 1U;
    if ((control.flags & cuda::WavefrontLaneShadowPending) == 0U ||
        (control.flags & ~known_flags) != 0U || pending.continuation_pending > 1U ||
        ((control.flags & cuda::WavefrontLaneContinuationPending) != 0U) != continuation_pending ||
        (continuation_pending && pending.termination != value(WavefrontTermination::none)) ||
        (!continuation_pending &&
         pending.termination != value(WavefrontTermination::outside_bsdf_support)) ||
        pending.reserved[0] != 0U || pending.reserved[1] != 0U || !valid_ray(pending.ray) ||
        !nonnegative_spectrum(pending.visible_contribution) ||
        zero_spectrum(pending.visible_contribution)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(WavefrontStageStatus::invalid_lane_state,
                                  WavefrontStageRoute::none, queue_path.value);
        return;
    }
    if (visible && !add_spectrum(streams.path_states[queue_path.value].accumulated_radiance,
                                 pending.visible_contribution)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(WavefrontStageStatus::numerical_failure,
                                  WavefrontStageRoute::none, queue_path.value);
        return;
    }
    const auto pending_termination = pending.termination;
    pending = WavefrontPendingShadow{};
    control.flags = 0U;
    if (!continuation_pending) {
        if (pending_termination > value(WavefrontTermination::outside_bsdf_support)) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] =
                outcome(WavefrontStageStatus::invalid_lane_state, WavefrontStageRoute::none,
                        queue_path.value, pending_termination);
            return;
        }
        control.termination = pending_termination;
        finish_phase(control, WavefrontLanePhase::terminated,
                     static_cast<WavefrontTermination>(pending_termination));
        outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::terminated,
                                  queue_path.value, pending_termination);
        return;
    }
    const auto routed = push_route(queues, ContinuationQueue, queue_path);
    if (routed != WavefrontStageStatus::success) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(routed, WavefrontStageRoute::none, queue_path.value, ContinuationQueue);
        return;
    }
    finish_phase(control, WavefrontLanePhase::continuation);
    outcomes[index] =
        outcome(WavefrontStageStatus::success, WavefrontStageRoute::continuation, queue_path.value);
}

__global__ void continuation_stage_kernel(const WavefrontQueueDeviceSoa queues,
                                          const WavefrontStageDeviceSoa streams,
                                          const std::uint32_t work_count,
                                          WavefrontStageOutcome* const outcomes) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= work_count) {
        return;
    }
    auto slot = PathSlot{};
    const auto queue_status =
        queue_slot(queues, ContinuationQueue, static_cast<std::uint32_t>(index), work_count,
                   streams.capacity, slot);
    if (queue_status != WavefrontStageStatus::success) {
        outcomes[index] =
            outcome(queue_status, WavefrontStageRoute::none, slot.value, ContinuationQueue);
        return;
    }
    auto& control = streams.controls[slot.value];
    if (!claim_phase(control, WavefrontLanePhase::continuation) ||
        !valid_ray(streams.rays[slot.value]) ||
        !valid_path_state(streams.path_states[slot.value])) {
        if (control.phase == value(WavefrontLanePhase::processing)) {
            finish_phase(control, WavefrontLanePhase::terminated);
        }
        outcomes[index] = outcome(WavefrontStageStatus::invalid_lane_state,
                                  WavefrontStageRoute::none, slot.value, control.phase);
        return;
    }
    const auto routed = push_route(queues, RayQueue, slot);
    if (routed != WavefrontStageStatus::success) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(routed, WavefrontStageRoute::none, slot.value, RayQueue);
        return;
    }
    finish_phase(control, WavefrontLanePhase::ray);
    outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::ray, slot.value);
}

[[nodiscard]] bool host_queue_view_is_valid(const WavefrontQueueDeviceSoa queues) noexcept {
    return queues.headers != nullptr && queues.queue_count == cuda::CudaWavefrontQueueCount &&
           (queues.slot_stride == 0U || queues.path_slots != nullptr);
}

[[nodiscard]] bool host_stream_view_is_valid(const WavefrontStageDeviceSoa streams) noexcept {
    return streams.reserved[0] == 0U && streams.reserved[1] == 0U && streams.reserved[2] == 0U &&
           (streams.capacity == 0U ||
            (streams.sample_streams != nullptr && streams.rays != nullptr &&
             streams.path_states != nullptr && streams.hits != nullptr &&
             streams.pending_shadows != nullptr && streams.controls != nullptr));
}

[[nodiscard]] bool host_views_are_valid(const WavefrontQueueDeviceSoa queues,
                                        const WavefrontStageDeviceSoa streams) noexcept {
    return host_queue_view_is_valid(queues) && host_stream_view_is_valid(streams) &&
           queues.slot_stride == streams.capacity;
}

[[nodiscard]] std::uint32_t block_count(const std::uint32_t work_count) noexcept {
    return static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(work_count) + ThreadsPerBlock - 1U) / ThreadsPerBlock);
}

} // namespace

extern "C" int blackframe_cuda_launch_wavefront_seed_camera(
    const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
    const std::uint32_t first_path_slot, const std::uint32_t path_count,
    WavefrontStageOutcome* const outcomes) noexcept {
    if (!host_views_are_valid(queues, streams) || (path_count != 0U && outcomes == nullptr) ||
        first_path_slot > streams.capacity || path_count > streams.capacity - first_path_slot) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (path_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    seed_camera_kernel<<<block_count(path_count), ThreadsPerBlock>>>(
        queues, streams, first_path_slot, path_count, outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_clear_queue(
    const WavefrontQueueDeviceSoa queues, const std::uint32_t queue_kind,
    const std::uint32_t acknowledge_overflow, std::uint32_t* const device_status) noexcept {
    if (!host_queue_view_is_valid(queues) || queue_kind >= cuda::CudaWavefrontQueueCount ||
        acknowledge_overflow > 1U || device_status == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    clear_queue_kernel<<<1U, 1U>>>(queues, queue_kind, acknowledge_overflow, device_status);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_camera_stage(
    const WavefrontQueueDeviceSoa queues, const WavefrontCameraInputDeviceSoa inputs,
    const WavefrontStageDeviceSoa streams, const std::uint32_t work_count,
    WavefrontStageOutcome* const outcomes) noexcept {
    if (!host_views_are_valid(queues, streams) || inputs.reserved != 0U ||
        inputs.count > streams.capacity ||
        (inputs.count != 0U && (inputs.sample_streams == nullptr || inputs.rays == nullptr ||
                                inputs.path_states == nullptr)) ||
        (work_count != 0U && outcomes == nullptr)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    camera_stage_kernel<<<block_count(work_count), ThreadsPerBlock>>>(queues, inputs, streams,
                                                                      work_count, outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_gather_rays(
    const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
    const std::uint32_t work_count, PathSlot* const compact_path_slots,
    TransportRay* const compact_rays, WavefrontStageOutcome* const outcomes) noexcept {
    if (!host_views_are_valid(queues, streams) ||
        (work_count != 0U &&
         (compact_path_slots == nullptr || compact_rays == nullptr || outcomes == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    gather_rays_kernel<<<block_count(work_count), ThreadsPerBlock>>>(
        queues, streams, work_count, compact_path_slots, compact_rays, outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_classify_closest_hit(
    const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
    const PathSlot* const compact_path_slots, const SceneClosestHitResult* const compact_results,
    const std::uint32_t work_count, WavefrontStageOutcome* const outcomes) noexcept {
    if (!host_views_are_valid(queues, streams) ||
        (work_count != 0U &&
         (compact_path_slots == nullptr || compact_results == nullptr || outcomes == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    classify_closest_hit_kernel<<<block_count(work_count), ThreadsPerBlock>>>(
        queues, streams, compact_path_slots, compact_results, work_count, outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_hit_stage(
    const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
    const std::uint32_t work_count, WavefrontStageOutcome* const outcomes) noexcept {
    if (!host_views_are_valid(queues, streams) || (work_count != 0U && outcomes == nullptr)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    hit_stage_kernel<<<block_count(work_count), ThreadsPerBlock>>>(queues, streams, work_count,
                                                                   outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_miss_stage(
    const std::uint8_t* const scene_bytes, const std::size_t scene_size,
    const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
    const std::uint32_t work_count, WavefrontStageOutcome* const outcomes) noexcept {
    if (!host_views_are_valid(queues, streams) ||
        (work_count != 0U &&
         (scene_bytes == nullptr || scene_size < sizeof(SceneSoaHeader) || outcomes == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    miss_stage_kernel<<<block_count(work_count), ThreadsPerBlock>>>(scene_bytes, scene_size, queues,
                                                                    streams, work_count, outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_shade_stage(
    const std::uint8_t* const scene_bytes, const std::size_t scene_size,
    const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
    const std::uint32_t max_diffuse_depth, const std::uint32_t work_count,
    WavefrontStageOutcome* const outcomes) noexcept {
    if (!host_views_are_valid(queues, streams) || max_diffuse_depth == 0xFFFFFFFFU ||
        (work_count != 0U &&
         (scene_bytes == nullptr || scene_size < sizeof(SceneSoaHeader) || outcomes == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    shade_stage_kernel<<<block_count(work_count), ThreadsPerBlock>>>(
        scene_bytes, scene_size, queues, streams, max_diffuse_depth, work_count, outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_gather_shadow_rays(
    const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
    const std::uint32_t work_count, PathSlot* const compact_path_slots,
    TransportRay* const compact_rays, WavefrontStageOutcome* const outcomes) noexcept {
    if (!host_views_are_valid(queues, streams) ||
        (work_count != 0U &&
         (compact_path_slots == nullptr || compact_rays == nullptr || outcomes == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    gather_shadow_rays_kernel<<<block_count(work_count), ThreadsPerBlock>>>(
        queues, streams, work_count, compact_path_slots, compact_rays, outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_process_shadow(
    const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
    const PathSlot* const compact_path_slots, const SceneOcclusionResult* const compact_results,
    const std::uint32_t work_count, WavefrontStageOutcome* const outcomes) noexcept {
    if (!host_views_are_valid(queues, streams) ||
        (work_count != 0U &&
         (compact_path_slots == nullptr || compact_results == nullptr || outcomes == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    process_shadow_kernel<<<block_count(work_count), ThreadsPerBlock>>>(
        queues, streams, compact_path_slots, compact_results, work_count, outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_continuation_stage(
    const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
    const std::uint32_t work_count, WavefrontStageOutcome* const outcomes) noexcept {
    if (!host_views_are_valid(queues, streams) || (work_count != 0U && outcomes == nullptr)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    continuation_stage_kernel<<<block_count(work_count), ThreadsPerBlock>>>(queues, streams,
                                                                            work_count, outcomes);
    return static_cast<int>(cudaGetLastError());
}
