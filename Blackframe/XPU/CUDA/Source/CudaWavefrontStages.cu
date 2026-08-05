#include <Blackframe/XPU/CUDA/SampleStreamDevice.cuh>
#include <Blackframe/XPU/CUDA/TransportDevice.cuh>
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
namespace transport = blackframe::xpu::cuda::transport_device;
namespace shared = blackframe::xpu::shared;
namespace scene_column = blackframe::xpu::shared::scene_soa_column;

using cuda::SampleStreamBounceDimensions;
using cuda::SampleStreamDumpStatus;
using cuda::WavefrontCameraInputDeviceSoa;
using cuda::WavefrontLaneControl;
using cuda::WavefrontLanePhase;
using cuda::WavefrontPendingShadow;
using cuda::WavefrontPreviousBsdfSample;
using cuda::WavefrontQueueDevicePushStatus;
using cuda::WavefrontQueueDeviceSoa;
using cuda::WavefrontStageDeviceSoa;
using cuda::WavefrontStageOutcome;
using cuda::WavefrontStageRoute;
using cuda::WavefrontStageStatus;
using cuda::WavefrontTermination;
using cuda::WavefrontTransportConfig;
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
constexpr auto DiscreteMeasure = std::uint32_t{0U};
constexpr auto SolidAngleMeasure = std::uint32_t{1U};
constexpr auto UniformLightSampling = std::uint32_t{0U};
constexpr auto BalanceMisHeuristic = std::uint32_t{0U};
constexpr auto PowerMisHeuristic = std::uint32_t{1U};
constexpr auto DisabledRussianRoulette = std::uint32_t{0U};
constexpr auto EnabledRussianRoulette = std::uint32_t{1U};
constexpr auto Pi = 3.14159265358979323846F;
constexpr auto HalfPi = Pi / 2.0F;
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

enum class IncidentLightKind : std::uint32_t {
    none = 0U,
    finite_point = 1U,
    finite_surface = 2U,
    infinite = 3U,
};

struct IncidentLight final {
    TransportSpectrum radiance{};
    Vector3 direction_to_light{};
    Vector3 endpoint_position{};
    Vector3 endpoint_position_error{};
    Vector3 endpoint_geometric_normal{};
    float distance{};
    float conditional_probability{};
    float selection_probability{};
    std::uint32_t probability_measure{};
    std::uint32_t kind{};
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
    sample_dimensions = 16U,
    transport_config = 17U,
    light_registry = 18U,
    emission_mis = 19U,
    russian_roulette = 20U,
    emission_orientation = 21U,
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
    if (!finite_vector(vector)) {
        return false;
    }
    const auto maximum_component = fmaxf(fabsf(vector.x), fmaxf(fabsf(vector.y), fabsf(vector.z)));
    if (!isfinite(maximum_component) || maximum_component == 0.0F) {
        return false;
    }
    const auto scaled = Vector3{
        .x = vector.x / maximum_component,
        .y = vector.y / maximum_component,
        .z = vector.z / maximum_component,
    };
    const auto magnitude = sqrtf(dot(scaled, scaled));
    if (!isfinite(magnitude) || !(magnitude > 0.0F)) {
        return false;
    }
    normalized = Vector3{
        .x = scaled.x / magnitude,
        .y = scaled.y / magnitude,
        .z = scaled.z / magnitude,
    };
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
    if (!finite_vector(direction) || !transport::unit_vector(transport::Vector3{
                                         .x = direction.x, .y = direction.y, .z = direction.z})) {
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

[[nodiscard]] __device__ bool one_sided_emission_support(const Vector3 geometric_normal,
                                                         const Vector3 outgoing_direction,
                                                         bool& supported) noexcept {
    const auto squared_normal_length = dot(geometric_normal, geometric_normal);
    constexpr auto unit_tolerance = 128.0F * FloatEpsilon;
    if (!finite_vector(geometric_normal) || !isfinite(squared_normal_length) ||
        fabsf(squared_normal_length - 1.0F) > unit_tolerance ||
        !finite_vector(outgoing_direction)) {
        return false;
    }
    const auto maximum_component =
        fmaxf(fabsf(outgoing_direction.x),
              fmaxf(fabsf(outgoing_direction.y), fabsf(outgoing_direction.z)));
    if (!isfinite(maximum_component) || maximum_component == 0.0F) {
        return false;
    }
    const auto scaled_outgoing = Vector3{
        .x = outgoing_direction.x / maximum_component,
        .y = outgoing_direction.y / maximum_component,
        .z = outgoing_direction.z / maximum_component,
    };
    const auto alignment = dot(geometric_normal, scaled_outgoing);
    if (!isfinite(alignment)) {
        return false;
    }
    const auto componentwise_orthogonal =
        (geometric_normal.x == 0.0F || outgoing_direction.x == 0.0F) &&
        (geometric_normal.y == 0.0F || outgoing_direction.y == 0.0F) &&
        (geometric_normal.z == 0.0F || outgoing_direction.z == 0.0F);
    if (alignment == 0.0F && componentwise_orthogonal) {
        supported = false;
        return true;
    }
    const auto absolute_sum = fmaf(fabsf(geometric_normal.x), fabsf(scaled_outgoing.x),
                                   fmaf(fabsf(geometric_normal.y), fabsf(scaled_outgoing.y),
                                        fabsf(geometric_normal.z) * fabsf(scaled_outgoing.z)));
    constexpr auto rounding_factor = 4.0F * FloatEpsilon;
    constexpr auto underflow_allowance = 4.0F * FloatDenormMinimum;
    const auto uncertainty = fmaf(rounding_factor, absolute_sum, underflow_allowance);
    if (!isfinite(absolute_sum) || !isfinite(uncertainty) || fabsf(alignment) <= uncertainty) {
        return false;
    }
    supported = alignment > 0.0F;
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

[[nodiscard]] __device__ bool
valid_pending_shadow_radiometry(const WavefrontPendingShadow& pending) noexcept {
    return nonnegative_spectrum(pending.beta) && !zero_spectrum(pending.beta) &&
           nonnegative_spectrum(pending.reflectance) && !zero_spectrum(pending.reflectance) &&
           nonnegative_spectrum(pending.incident_radiance) &&
           !zero_spectrum(pending.incident_radiance) && isfinite(pending.receiver_cosine) &&
           pending.receiver_cosine > 0.0F && isfinite(pending.estimator_weight) &&
           pending.estimator_weight >= 0.0F && pending.estimator_weight <= 1.0F &&
           isfinite(pending.selection_probability) && pending.selection_probability > 0.0F &&
           pending.selection_probability <= 1.0F && isfinite(pending.conditional_probability) &&
           pending.conditional_probability > 0.0F;
}

[[nodiscard]] __device__ float exact_special_cosine(const float angle) noexcept {
    if (angle == 0.0F) {
        return 1.0F;
    }
    if (angle == HalfPi) {
        return 0.0F;
    }
    if (angle == Pi) {
        return -1.0F;
    }
    return cosf(angle);
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

[[nodiscard]] __device__ bool valid_transport_config(const WavefrontTransportConfig& config,
                                                     const SceneSoaHeader& scene) noexcept {
    const auto light_count = scene.punctual_light_count + scene.mesh_area_light_count;
    if (config.abi_major != cuda::WavefrontTransportConfigAbiMajor ||
        config.abi_minor != cuda::WavefrontTransportConfigAbiMinor ||
        config.struct_size != sizeof(WavefrontTransportConfig) ||
        config.light_sampling_strategy != UniformLightSampling ||
        config.light_count != light_count || light_count > MaximumUniformLightCount ||
        (config.mis_heuristic != BalanceMisHeuristic &&
         config.mis_heuristic != PowerMisHeuristic) ||
        (config.russian_roulette_mode != DisabledRussianRoulette &&
         config.russian_roulette_mode != EnabledRussianRoulette) ||
        config.reserved[0] != 0U || config.reserved[1] != 0U) {
        return false;
    }
    if (config.russian_roulette_mode == DisabledRussianRoulette) {
        return config.russian_roulette_first_depth == 0U &&
               config.russian_roulette_minimum_probability == 0.0F &&
               config.russian_roulette_maximum_probability == 0.0F;
    }
    return config.russian_roulette_first_depth != 0U &&
           isfinite(config.russian_roulette_minimum_probability) &&
           isfinite(config.russian_roulette_maximum_probability) &&
           config.russian_roulette_minimum_probability > 0.0F &&
           config.russian_roulette_minimum_probability < 1.0F &&
           config.russian_roulette_maximum_probability >=
               config.russian_roulette_minimum_probability &&
           config.russian_roulette_maximum_probability <= 1.0F;
}

[[nodiscard]] __device__ bool
valid_previous_bsdf_sample(const WavefrontPreviousBsdfSample& previous,
                           const TransportRay& ray) noexcept {
    if (previous.valid == 0U) {
        return previous.context_x == 0.0F && previous.context_y == 0.0F &&
               previous.context_z == 0.0F && previous.context_time == 0.0F &&
               previous.incoming_x == 0.0F && previous.incoming_y == 0.0F &&
               previous.incoming_z == 0.0F && previous.probability_value == 0.0F &&
               previous.probability_measure == DiscreteMeasure && previous.reserved[0] == 0U &&
               previous.reserved[1] == 0U;
    }
    return previous.valid == 1U && isfinite(previous.context_x) && isfinite(previous.context_y) &&
           isfinite(previous.context_z) && isfinite(previous.context_time) &&
           previous.context_time == ray.time && previous.context_time >= 0.0F &&
           previous.context_time <= 1.0F && isfinite(previous.incoming_x) &&
           isfinite(previous.incoming_y) && isfinite(previous.incoming_z) &&
           previous.incoming_x == ray.direction_x && previous.incoming_y == ray.direction_y &&
           previous.incoming_z == ray.direction_z && isfinite(previous.probability_value) &&
           previous.probability_value > 0.0F && previous.probability_measure == SolidAngleMeasure &&
           previous.reserved[0] == 0U && previous.reserved[1] == 0U;
}

[[nodiscard]] __device__ bool valid_queue_view(const WavefrontQueueDeviceSoa queues) noexcept {
    return queues.headers != nullptr && queues.queue_count == cuda::CudaWavefrontQueueCount &&
           (queues.slot_stride == 0U || queues.path_slots != nullptr);
}

[[nodiscard]] __device__ bool valid_stream_view(const WavefrontStageDeviceSoa streams) noexcept {
    return streams.reserved == 0U &&
           (streams.capacity == 0U ||
            (streams.sample_streams != nullptr && streams.rays != nullptr &&
             streams.path_states != nullptr && streams.hits != nullptr &&
             streams.pending_shadows != nullptr && streams.previous_bsdf_samples != nullptr &&
             streams.controls != nullptr));
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
    if (control.blocked_depth_limits != 0U) {
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
    streams.previous_bsdf_samples[slot.value] = WavefrontPreviousBsdfSample{};
    control.flags = 0U;
    control.blocked_depth_limits = 0U;
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

template <std::size_t Capacity> struct TriangleExpansion final {
    float components[Capacity]{};
    std::size_t size{};
};

[[nodiscard]] __device__ bool triangle_normal_float(const float value) noexcept {
    return isfinite(value) && fabsf(value) >= 0x1p-126F;
}

template <std::size_t Capacity>
[[nodiscard]] __device__ bool triangle_add_component(TriangleExpansion<Capacity>& expansion,
                                                     const float component) noexcept {
    if (!isfinite(component)) {
        return false;
    }

    auto accumulated = component;
    auto output_index = std::size_t{0U};
    const auto input_size = expansion.size;
    for (auto input_index = std::size_t{0U}; input_index < input_size; ++input_index) {
        const auto existing = expansion.components[input_index];
        const auto rounded_sum = accumulated + existing;
        const auto existing_from_sum = rounded_sum - accumulated;
        const auto accumulated_from_sum = rounded_sum - existing_from_sum;
        const auto existing_remainder = existing - existing_from_sum;
        const auto accumulated_remainder = accumulated - accumulated_from_sum;
        const auto error = existing_remainder + accumulated_remainder;
        if (!isfinite(rounded_sum) || !isfinite(error)) {
            return false;
        }
        if (error != 0.0F) {
            if (output_index == Capacity) {
                return false;
            }
            expansion.components[output_index++] = error;
        }
        accumulated = rounded_sum;
    }

    if (accumulated != 0.0F || output_index == 0U) {
        if (output_index == Capacity) {
            return false;
        }
        expansion.components[output_index++] = accumulated;
    }
    expansion.size = output_index;
    return true;
}

template <std::size_t Capacity>
[[nodiscard]] __device__ bool triangle_add_product(TriangleExpansion<Capacity>& expansion,
                                                   const float left, const float right,
                                                   const float sign = 1.0F) noexcept {
    const auto rounded_product = left * right;
    const auto product_error = fmaf(left, right, -rounded_product);
    if (!isfinite(rounded_product) || !isfinite(product_error) ||
        (left != 0.0F && right != 0.0F && !triangle_normal_float(rounded_product)) ||
        (product_error != 0.0F && !triangle_normal_float(product_error))) {
        return false;
    }
    return triangle_add_component(expansion, sign * product_error) &&
           triangle_add_component(expansion, sign * rounded_product);
}

template <std::size_t DestinationCapacity, std::size_t SourceCapacity>
[[nodiscard]] __device__ bool
triangle_add_expansion(TriangleExpansion<DestinationCapacity>& destination,
                       const TriangleExpansion<SourceCapacity>& source,
                       const float sign = 1.0F) noexcept {
    for (auto index = std::size_t{0U}; index < source.size; ++index) {
        if (!triangle_add_component(destination, sign * source.components[index])) {
            return false;
        }
    }
    return true;
}

template <std::size_t OutputCapacity, std::size_t LeftCapacity, std::size_t RightCapacity>
[[nodiscard]] __device__ bool
triangle_multiply_expansions(const TriangleExpansion<LeftCapacity>& left,
                             const TriangleExpansion<RightCapacity>& right,
                             TriangleExpansion<OutputCapacity>& product) noexcept {
    product = {};
    for (auto left_index = std::size_t{0U}; left_index < left.size; ++left_index) {
        for (auto right_index = std::size_t{0U}; right_index < right.size; ++right_index) {
            if (!triangle_add_product(product, left.components[left_index],
                                      right.components[right_index])) {
                return false;
            }
        }
    }
    return true;
}

template <std::size_t Capacity>
[[nodiscard]] __device__ int
triangle_expansion_sign(const TriangleExpansion<Capacity>& expansion) noexcept {
    for (auto index = expansion.size; index > 0U; --index) {
        const auto component = expansion.components[index - 1U];
        if (component != 0.0F) {
            return component > 0.0F ? 1 : -1;
        }
    }
    return 0;
}

struct TriangleExactValue final {
    int sign{};
    float value{};
};

template <std::size_t Capacity>
[[nodiscard]] __device__ bool
triangle_resolve_expansion(const TriangleExpansion<Capacity>& expansion,
                           TriangleExactValue& result) noexcept {
    const auto exact_sign = triangle_expansion_sign(expansion);
    if (exact_sign == 0) {
        result = {};
        return true;
    }

    auto resolved = 0.0F;
    for (auto index = std::size_t{0U}; index < expansion.size; ++index) {
        resolved += expansion.components[index];
    }
    if (!isfinite(resolved) || resolved == 0.0F || (resolved > 0.0F ? 1 : -1) != exact_sign) {
        return false;
    }
    result = TriangleExactValue{.sign = exact_sign, .value = resolved};
    return true;
}

using TriangleCoordinateExpansion = TriangleExpansion<2U>;
using TrianglePairProductExpansion = TriangleExpansion<16U>;
using TriangleCrossComponentExpansion = TriangleExpansion<32U>;

struct TriangleExpansionVector final {
    TriangleCoordinateExpansion x{};
    TriangleCoordinateExpansion y{};
    TriangleCoordinateExpansion z{};
};

struct TriangleAxisExponents final {
    int x{};
    int y{};
    int z{};
};

[[nodiscard]] __device__ bool
triangle_exact_difference(const float left, const float right,
                          TriangleCoordinateExpansion& difference) noexcept {
    difference = {};
    return triangle_add_component(difference, left) && triangle_add_component(difference, -right);
}

[[nodiscard]] __device__ bool
triangle_exact_relative_vector(const Vector3 point, const Vector3 origin,
                               TriangleExpansionVector& result) noexcept {
    result = {};
    return triangle_exact_difference(point.x, origin.x, result.x) &&
           triangle_exact_difference(point.y, origin.y, result.y) &&
           triangle_exact_difference(point.z, origin.z, result.z);
}

[[nodiscard]] __device__ bool triangle_resolved_difference_of_products(
    const TriangleCoordinateExpansion& first_left, const TriangleCoordinateExpansion& first_right,
    const TriangleCoordinateExpansion& second_left, const TriangleCoordinateExpansion& second_right,
    TriangleExactValue& result) noexcept {
    auto first_product = TrianglePairProductExpansion{};
    auto second_product = TrianglePairProductExpansion{};
    auto difference = TriangleCrossComponentExpansion{};
    return triangle_multiply_expansions(first_left, first_right, first_product) &&
           triangle_multiply_expansions(second_left, second_right, second_product) &&
           triangle_add_expansion(difference, first_product) &&
           triangle_add_expansion(difference, second_product, -1.0F) &&
           triangle_resolve_expansion(difference, result);
}

[[nodiscard]] __device__ bool triangle_scaling_preserves(const float original,
                                                         const float scaled) noexcept {
    return original == 0.0F || triangle_normal_float(scaled);
}

[[nodiscard]] __device__ bool triangle_scaling_preserves(const Vector3 original,
                                                         const Vector3 scaled) noexcept {
    return triangle_scaling_preserves(original.x, scaled.x) &&
           triangle_scaling_preserves(original.y, scaled.y) &&
           triangle_scaling_preserves(original.z, scaled.z);
}

[[nodiscard]] __device__ int triangle_scaling_exponent(const Vector3* const vertices,
                                                       const std::uint32_t axis) noexcept {
    auto maximum = 0.0F;
    for (auto vertex = std::uint32_t{0U}; vertex < 3U; ++vertex) {
        maximum = fmaxf(maximum, fabsf(vector_component(vertices[vertex], axis)));
    }
    auto exponent = 0;
    static_cast<void>(frexpf(maximum, &exponent));
    return exponent;
}

[[nodiscard]] __device__ Vector3
triangle_scaled_point(const Vector3 point, const TriangleAxisExponents exponents) noexcept {
    return Vector3{
        .x = scalbnf(point.x, -exponents.x),
        .y = scalbnf(point.y, -exponents.y),
        .z = scalbnf(point.z, -exponents.z),
    };
}

[[nodiscard]] __device__ bool triangle_area(const Vector3* const vertices, float& area,
                                            Vector3* const normal = nullptr) noexcept {
    const auto spatial_exponents = TriangleAxisExponents{
        .x = triangle_scaling_exponent(vertices, 0U),
        .y = triangle_scaling_exponent(vertices, 1U),
        .z = triangle_scaling_exponent(vertices, 2U),
    };
    const auto scaled_vertex0 = triangle_scaled_point(vertices[0], spatial_exponents);
    const auto scaled_vertex1 = triangle_scaled_point(vertices[1], spatial_exponents);
    const auto scaled_vertex2 = triangle_scaled_point(vertices[2], spatial_exponents);
    if (!triangle_scaling_preserves(vertices[0], scaled_vertex0) ||
        !triangle_scaling_preserves(vertices[1], scaled_vertex1) ||
        !triangle_scaling_preserves(vertices[2], scaled_vertex2)) {
        return false;
    }

    auto first = TriangleExpansionVector{};
    auto second = TriangleExpansionVector{};
    if (!triangle_exact_relative_vector(scaled_vertex1, scaled_vertex0, first) ||
        !triangle_exact_relative_vector(scaled_vertex2, scaled_vertex0, second)) {
        return false;
    }

    auto x = TriangleExactValue{};
    auto y = TriangleExactValue{};
    auto z = TriangleExactValue{};
    if (!triangle_resolved_difference_of_products(first.y, second.z, first.z, second.y, x) ||
        !triangle_resolved_difference_of_products(first.z, second.x, first.x, second.z, y) ||
        !triangle_resolved_difference_of_products(first.x, second.y, first.y, second.x, z)) {
        return false;
    }

    const auto cross_exponents = TriangleAxisExponents{
        .x = spatial_exponents.y + spatial_exponents.z,
        .y = spatial_exponents.z + spatial_exponents.x,
        .z = spatial_exponents.x + spatial_exponents.y,
    };
    auto magnitude_exponent = (-2147483647 - 1);
    if (x.sign != 0) {
        auto exponent = 0;
        static_cast<void>(frexpf(fabsf(x.value), &exponent));
        const auto cross_exponent = exponent + cross_exponents.x;
        magnitude_exponent =
            cross_exponent > magnitude_exponent ? cross_exponent : magnitude_exponent;
    }
    if (y.sign != 0) {
        auto exponent = 0;
        static_cast<void>(frexpf(fabsf(y.value), &exponent));
        const auto cross_exponent = exponent + cross_exponents.y;
        magnitude_exponent =
            cross_exponent > magnitude_exponent ? cross_exponent : magnitude_exponent;
    }
    if (z.sign != 0) {
        auto exponent = 0;
        static_cast<void>(frexpf(fabsf(z.value), &exponent));
        const auto cross_exponent = exponent + cross_exponents.z;
        magnitude_exponent =
            cross_exponent > magnitude_exponent ? cross_exponent : magnitude_exponent;
    }
    if (magnitude_exponent == (-2147483647 - 1)) {
        return false;
    }

    const auto adjusted_x =
        x.sign == 0 ? 0.0F : scalbnf(x.value, cross_exponents.x - magnitude_exponent);
    const auto adjusted_y =
        y.sign == 0 ? 0.0F : scalbnf(y.value, cross_exponents.y - magnitude_exponent);
    const auto adjusted_z =
        z.sign == 0 ? 0.0F : scalbnf(z.value, cross_exponents.z - magnitude_exponent);
    if ((x.sign != 0 && adjusted_x == 0.0F) || (y.sign != 0 && adjusted_y == 0.0F) ||
        (z.sign != 0 && adjusted_z == 0.0F) || !isfinite(adjusted_x) || !isfinite(adjusted_y) ||
        !isfinite(adjusted_z)) {
        return false;
    }

    const auto maximum_component =
        fmaxf(fabsf(adjusted_x), fmaxf(fabsf(adjusted_y), fabsf(adjusted_z)));
    if (!isfinite(maximum_component) || !(maximum_component > 0.0F)) {
        return false;
    }
    const auto scaled_x = adjusted_x / maximum_component;
    const auto scaled_y = adjusted_y / maximum_component;
    const auto scaled_z = adjusted_z / maximum_component;
    const auto squared_length =
        fmaf(scaled_x, scaled_x, fmaf(scaled_y, scaled_y, scaled_z * scaled_z));
    const auto scaled_length = sqrtf(squared_length);
    const auto adjusted_length = maximum_component * scaled_length;
    area = scalbnf(adjusted_length * 0.5F, magnitude_exponent);
    if (!isfinite(scaled_length) || !(scaled_length > 0.0F) || !isfinite(adjusted_length) ||
        !(adjusted_length > 0.0F) || !isfinite(area) || !(area > 0.0F)) {
        return false;
    }
    if (normal != nullptr) {
        if (!triangle_scaling_preserves(x.value, adjusted_x) ||
            !triangle_scaling_preserves(y.value, adjusted_y) ||
            !triangle_scaling_preserves(z.value, adjusted_z)) {
            return false;
        }
        *normal = Vector3{
            .x = scaled_x / scaled_length,
            .y = scaled_y / scaled_length,
            .z = scaled_z / scaled_length,
        };
        if (!finite_vector(*normal)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] __device__ SceneDeviceStatus sample_mesh_area_light(
    const std::uint8_t* const bytes, const SceneSoaHeader& scene,
    const TransportPathStateLane& state, const std::uint32_t area_light_index,
    const float area_sample, const float triangle_sample, LightEndpoint& endpoint) noexcept {
    const auto light_count = static_cast<std::uint32_t>(scene.mesh_area_light_count);
    if (area_light_index >= light_count || !isfinite(area_sample) || area_sample < 0.0F ||
        !(area_sample < 1.0F) || !isfinite(triangle_sample) || triangle_sample < 0.0F ||
        !(triangle_sample < 1.0F)) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto light_instance_id = scene_values<std::uint32_t>(
        bytes, scene, scene_column::mesh_area_light_instance_id)[area_light_index];
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

[[nodiscard]] __device__ bool uniform_light_selection(const std::uint32_t light_count,
                                                      const float canonical_sample,
                                                      std::uint32_t& light_index,
                                                      float& probability) noexcept {
    const auto selected = transport::uniform_light_selection(light_count, canonical_sample);
    if (!transport::succeeded(selected.status)) {
        return false;
    }
    light_index = selected.index;
    probability = selected.interval_probability;
    return true;
}

[[nodiscard]] __device__ bool uniform_light_probability(const std::uint32_t light_count,
                                                        const std::uint32_t light_index,
                                                        float& probability) noexcept {
    if (light_count == 0U || light_count > MaximumUniformLightCount || light_index >= light_count) {
        return false;
    }
    const auto scalar_count = static_cast<float>(light_count);
    const auto lower = light_index == 0U ? 0.0F : static_cast<float>(light_index) / scalar_count;
    const auto upper = light_index + 1U == light_count
                           ? 1.0F
                           : static_cast<float>(light_index + 1U) / scalar_count;
    probability = upper - lower;
    return isfinite(lower) && isfinite(upper) && isfinite(probability) && probability > 0.0F &&
           probability <= 1.0F;
}

[[nodiscard]] __device__ bool punctual_spectrum(const std::uint8_t* const bytes,
                                                const SceneSoaHeader& scene,
                                                const std::uint32_t punctual_index,
                                                TransportSpectrum& spectrum) noexcept {
    return punctual_index < scene.punctual_light_count &&
           load_spectrum(bytes, scene, scene_column::punctual_spectrum, punctual_index, spectrum);
}

[[nodiscard]] __device__ SceneDeviceStatus sample_punctual_light(const std::uint8_t* const bytes,
                                                                 const SceneSoaHeader& scene,
                                                                 const std::uint32_t punctual_index,
                                                                 const Vector3 reference_position,
                                                                 const float selection_probability,
                                                                 IncidentLight& incident) noexcept {
    if (punctual_index >= scene.punctual_light_count || !finite_vector(reference_position) ||
        !isfinite(selection_probability) || !(selection_probability > 0.0F) ||
        selection_probability > 1.0F) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto kind =
        scene_values<std::uint32_t>(bytes, scene, scene_column::punctual_kind)[punctual_index];
    auto spectrum = TransportSpectrum{};
    if (!punctual_spectrum(bytes, scene, punctual_index, spectrum)) {
        return SceneDeviceStatus::numerical_failure;
    }
    const auto direction = Vector3{
        .x = scene_values<float>(bytes, scene, scene_column::punctual_direction_x)[punctual_index],
        .y = scene_values<float>(bytes, scene, scene_column::punctual_direction_y)[punctual_index],
        .z = scene_values<float>(bytes, scene, scene_column::punctual_direction_z)[punctual_index],
    };
    if (kind == static_cast<std::uint32_t>(shared::SceneSoaPunctualKind::directional)) {
        if (zero_spectrum(spectrum)) {
            incident = IncidentLight{};
            return SceneDeviceStatus::valid;
        }
        if (!transport::unit_vector(transport::Vector3{
                .x = direction.x,
                .y = direction.y,
                .z = direction.z,
            })) {
            return SceneDeviceStatus::invalid_scene;
        }
        incident = IncidentLight{
            .radiance = spectrum,
            .direction_to_light = multiply(direction, -1.0F),
            .distance = __int_as_float(0x7F800000),
            .conditional_probability = 1.0F,
            .selection_probability = selection_probability,
            .probability_measure = DiscreteMeasure,
            .kind = static_cast<std::uint32_t>(IncidentLightKind::infinite),
        };
        return SceneDeviceStatus::valid;
    }
    if (kind != static_cast<std::uint32_t>(shared::SceneSoaPunctualKind::point) &&
        kind != static_cast<std::uint32_t>(shared::SceneSoaPunctualKind::spot)) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto position = Vector3{
        .x = scene_values<float>(bytes, scene, scene_column::punctual_position_x)[punctual_index],
        .y = scene_values<float>(bytes, scene, scene_column::punctual_position_y)[punctual_index],
        .z = scene_values<float>(bytes, scene, scene_column::punctual_position_z)[punctual_index],
    };
    const auto position_error = Vector3{
        .x = scene_values<float>(bytes, scene,
                                 scene_column::punctual_position_error_x)[punctual_index],
        .y = scene_values<float>(bytes, scene,
                                 scene_column::punctual_position_error_y)[punctual_index],
        .z = scene_values<float>(bytes, scene,
                                 scene_column::punctual_position_error_z)[punctual_index],
    };
    auto segment = ScaledSegment{};
    if (!finite_vector(position) || !finite_vector(position_error) || position_error.x < 0.0F ||
        position_error.y < 0.0F || position_error.z < 0.0F ||
        !scaled_segment(subtract(position, reference_position), segment)) {
        return SceneDeviceStatus::numerical_failure;
    }
    const auto scaled_squared_distance = dot(segment.scaled, segment.scaled);
    if (!isfinite(scaled_squared_distance) || !(scaled_squared_distance > 0.0F)) {
        return SceneDeviceStatus::numerical_failure;
    }
    auto falloff_transition = 1.0F;
    auto falloff_smooth_factor = 1.0F;
    if (kind == static_cast<std::uint32_t>(shared::SceneSoaPunctualKind::spot)) {
        if (!transport::unit_vector(transport::Vector3{
                .x = direction.x,
                .y = direction.y,
                .z = direction.z,
            })) {
            return SceneDeviceStatus::invalid_scene;
        }
        const auto inner_angle = scene_values<float>(
            bytes, scene, scene_column::punctual_inner_half_angle)[punctual_index];
        const auto outer_angle = scene_values<float>(
            bytes, scene, scene_column::punctual_outer_half_angle)[punctual_index];
        if (!isfinite(inner_angle) || !isfinite(outer_angle) || inner_angle < 0.0F ||
            inner_angle > outer_angle || !(outer_angle > 0.0F) || !(outer_angle < Pi)) {
            return SceneDeviceStatus::invalid_scene;
        }
        const auto cosine_inner = exact_special_cosine(inner_angle);
        const auto cosine_outer = exact_special_cosine(outer_angle);
        const auto cone_cosine = dot(direction, multiply(segment.direction, -1.0F));
        if (!isfinite(cosine_inner) || !isfinite(cosine_outer) || !isfinite(cone_cosine) ||
            cosine_inner < cosine_outer) {
            return SceneDeviceStatus::numerical_failure;
        }
        if (cone_cosine <= cosine_outer) {
            incident = IncidentLight{};
            return SceneDeviceStatus::valid;
        }
        if (inner_angle != outer_angle && cone_cosine < cosine_inner) {
            falloff_transition = (cone_cosine - cosine_outer) / (cosine_inner - cosine_outer);
            falloff_smooth_factor = 3.0F - 2.0F * falloff_transition;
            if (!isfinite(falloff_transition) || !(falloff_transition > 0.0F) ||
                !(falloff_transition < 1.0F) || !isfinite(falloff_smooth_factor) ||
                !(falloff_smooth_factor > 0.0F)) {
                return SceneDeviceStatus::numerical_failure;
            }
        }
    }
    auto radiance = TransportSpectrum{};
    for (auto lane = std::uint32_t{}; lane < shared::HostDeviceSpectrumLaneCount; ++lane) {
        const float numerators[]{spectrum.values[lane], falloff_transition, falloff_transition,
                                 falloff_smooth_factor};
        const float denominators[]{segment.scale, segment.scale, scaled_squared_distance};
        if (!checked_product_quotient(numerators, denominators, radiance.values[lane])) {
            return SceneDeviceStatus::numerical_failure;
        }
    }
    if (zero_spectrum(spectrum)) {
        incident = IncidentLight{};
        return SceneDeviceStatus::valid;
    }
    incident = IncidentLight{
        .radiance = radiance,
        .direction_to_light = segment.direction,
        .endpoint_position = position,
        .endpoint_position_error = position_error,
        .distance = segment.length,
        .conditional_probability = 1.0F,
        .selection_probability = selection_probability,
        .probability_measure = DiscreteMeasure,
        .kind = static_cast<std::uint32_t>(IncidentLightKind::finite_point),
    };
    return SceneDeviceStatus::valid;
}

[[nodiscard]] __device__ SceneDeviceStatus
sample_registered_light(const std::uint8_t* const bytes, const SceneSoaHeader& scene,
                        const TransportPathStateLane& state, const SampleStreamIndex& sample_index,
                        const SampleStreamBounceDimensions& dimensions, const SurfaceData& surface,
                        const std::uint32_t visibility_mask, IncidentLight& incident) noexcept {
    const auto wide_count = scene.punctual_light_count + scene.mesh_area_light_count;
    if (wide_count == 0U || wide_count > MaximumUniformLightCount || wide_count > 0xFFFFFFFFULL) {
        return SceneDeviceStatus::unsupported_transport;
    }
    const auto light_count = static_cast<std::uint32_t>(wide_count);
    const auto canonical_selection =
        cuda::sample_stream::sample_1d(sample_index, dimensions.light_selection);
    auto light_index = std::uint32_t{};
    auto selection_probability = 0.0F;
    if (!uniform_light_selection(light_count, canonical_selection, light_index,
                                 selection_probability)) {
        return SceneDeviceStatus::numerical_failure;
    }
    if (light_index < scene.punctual_light_count) {
        return sample_punctual_light(bytes, scene, light_index, surface.position,
                                     selection_probability, incident);
    }
    const auto area_index = light_index - static_cast<std::uint32_t>(scene.punctual_light_count);
    const auto instance_id = scene_values<std::uint32_t>(
        bytes, scene, scene_column::mesh_area_light_instance_id)[area_index];
    auto instance = std::uint32_t{};
    if (!find_id(scene_values<std::uint32_t>(bytes, scene, scene_column::instance_id),
                 static_cast<std::uint32_t>(scene.instance_count), instance_id, instance)) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto instance_mask =
        scene_values<std::uint32_t>(bytes, scene, scene_column::instance_visibility_mask)[instance];
    if ((instance_mask & visibility_mask) == 0U) {
        incident = IncidentLight{};
        return SceneDeviceStatus::valid;
    }
    auto endpoint = LightEndpoint{};
    const auto sampled = sample_mesh_area_light(
        bytes, scene, state, area_index,
        cuda::sample_stream::sample_1d(sample_index, dimensions.light_u),
        cuda::sample_stream::sample_1d(sample_index, dimensions.light_v), endpoint);
    if (sampled != SceneDeviceStatus::valid) {
        return sampled;
    }
    if (zero_spectrum(endpoint.radiance)) {
        incident = IncidentLight{};
        return SceneDeviceStatus::valid;
    }
    auto segment = ScaledSegment{};
    if (!scaled_segment(subtract(endpoint.position, surface.position), segment)) {
        return SceneDeviceStatus::numerical_failure;
    }
    auto absolute_alignment = 0.0F;
    auto supported = false;
    if (!light_surface_alignment(endpoint.geometric_normal, segment.scaled, absolute_alignment,
                                 supported)) {
        return SceneDeviceStatus::numerical_failure;
    }
    if (!supported) {
        incident = IncidentLight{};
        return SceneDeviceStatus::valid;
    }
    const float numerators[]{endpoint.area_density, segment.scale, segment.scale,
                             segment.scaled_distance_cubed};
    const float denominators[]{absolute_alignment};
    auto solid_angle_probability = 0.0F;
    if (!checked_product_quotient(numerators, denominators, solid_angle_probability)) {
        return SceneDeviceStatus::numerical_failure;
    }
    incident = IncidentLight{
        .radiance = endpoint.radiance,
        .direction_to_light = segment.direction,
        .endpoint_position = endpoint.position,
        .endpoint_position_error = endpoint.position_error,
        .endpoint_geometric_normal = endpoint.geometric_normal,
        .distance = segment.length,
        .conditional_probability = solid_angle_probability,
        .selection_probability = selection_probability,
        .probability_measure = SolidAngleMeasure,
        .kind = static_cast<std::uint32_t>(IncidentLightKind::finite_surface),
    };
    return SceneDeviceStatus::valid;
}

[[nodiscard]] __device__ SceneDeviceStatus
mesh_area_light_inverse_area(const std::uint8_t* const bytes, const SceneSoaHeader& scene,
                             const std::uint32_t area_light_index, float& inverse_area) noexcept {
    if (area_light_index >= scene.mesh_area_light_count) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto instance_id = scene_values<std::uint32_t>(
        bytes, scene, scene_column::mesh_area_light_instance_id)[area_light_index];
    auto instance = std::uint32_t{};
    if (!find_id(scene_values<std::uint32_t>(bytes, scene, scene_column::instance_id),
                 static_cast<std::uint32_t>(scene.instance_count), instance_id, instance)) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto geometry_id =
        scene_values<std::uint32_t>(bytes, scene, scene_column::instance_geometry_id)[instance];
    auto geometry = std::uint32_t{};
    if (!find_id(scene_values<std::uint32_t>(bytes, scene, scene_column::geometry_id),
                 static_cast<std::uint32_t>(scene.geometry_count), geometry_id, geometry)) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto triangle_count = static_cast<std::uint32_t>(
        scene_values<std::uint64_t>(bytes, scene, scene_column::geometry_triangle_count)[geometry]);
    if (triangle_count == 0U) {
        return SceneDeviceStatus::invalid_scene;
    }
    auto sum = 0.0F;
    auto compensation = 0.0F;
    for (auto triangle = std::uint32_t{}; triangle < triangle_count; ++triangle) {
        Vector3 vertices[3U]{};
        auto area = 0.0F;
        if (!load_triangle_world(bytes, scene, geometry, instance, triangle, vertices) ||
            !triangle_area(vertices, area)) {
            return SceneDeviceStatus::numerical_failure;
        }
        const auto next = sum + area;
        if (!isfinite(next)) {
            return SceneDeviceStatus::numerical_failure;
        }
        compensation += fabsf(sum) >= fabsf(area) ? (sum - next) + area : (area - next) + sum;
        if (!isfinite(compensation)) {
            return SceneDeviceStatus::numerical_failure;
        }
        sum = next;
    }
    const auto total = sum + compensation;
    inverse_area = 1.0F / total;
    return isfinite(total) && total > 0.0F && isfinite(inverse_area) && inverse_area > 0.0F
               ? SceneDeviceStatus::valid
               : SceneDeviceStatus::numerical_failure;
}

[[nodiscard]] __device__ SceneDeviceStatus
emissive_hit_mis_weight(const std::uint8_t* const bytes, const SceneSoaHeader& scene,
                        const ClosestHit& hit, const SurfaceData& surface, const TransportRay& ray,
                        const WavefrontPreviousBsdfSample& previous, const std::uint32_t heuristic,
                        float& weight) noexcept {
    weight = 1.0F;
    if (previous.valid == 0U) {
        return SceneDeviceStatus::valid;
    }
    if (!valid_previous_bsdf_sample(previous, ray)) {
        return SceneDeviceStatus::invalid_scene;
    }
    auto area_index = static_cast<std::uint32_t>(scene.mesh_area_light_count);
    const auto* const instance_ids =
        scene_values<std::uint32_t>(bytes, scene, scene_column::mesh_area_light_instance_id);
    for (auto candidate = std::uint32_t{}; candidate < scene.mesh_area_light_count; ++candidate) {
        if (instance_ids[candidate] == hit.identifiers.instance) {
            area_index = candidate;
            break;
        }
    }
    if (area_index >= scene.mesh_area_light_count) {
        return SceneDeviceStatus::unsupported_transport;
    }
    const auto wide_light_count = scene.punctual_light_count + scene.mesh_area_light_count;
    if (wide_light_count == 0U || wide_light_count > MaximumUniformLightCount ||
        wide_light_count > 0xFFFFFFFFULL) {
        return SceneDeviceStatus::unsupported_transport;
    }
    const auto global_light_index =
        static_cast<std::uint32_t>(scene.punctual_light_count) + area_index;
    auto selection_probability = 0.0F;
    if (!uniform_light_probability(static_cast<std::uint32_t>(wide_light_count), global_light_index,
                                   selection_probability)) {
        return SceneDeviceStatus::numerical_failure;
    }
    auto inverse_area = 0.0F;
    const auto area_status = mesh_area_light_inverse_area(bytes, scene, area_index, inverse_area);
    if (area_status != SceneDeviceStatus::valid) {
        return area_status;
    }
    const auto context =
        Vector3{.x = previous.context_x, .y = previous.context_y, .z = previous.context_z};
    auto segment = ScaledSegment{};
    if (!scaled_segment(subtract(surface.position, context), segment)) {
        return SceneDeviceStatus::numerical_failure;
    }
    auto absolute_alignment = 0.0F;
    auto supported = false;
    if (!light_surface_alignment(surface.geometric_normal, segment.scaled, absolute_alignment,
                                 supported)) {
        return SceneDeviceStatus::numerical_failure;
    }
    if (!supported) {
        return SceneDeviceStatus::unsupported_transport;
    }
    const float pdf_numerators[]{inverse_area, segment.scale, segment.scale,
                                 segment.scaled_distance_cubed};
    const float pdf_denominators[]{absolute_alignment};
    auto conditional_probability = 0.0F;
    if (!checked_product_quotient(pdf_numerators, pdf_denominators, conditional_probability)) {
        return SceneDeviceStatus::numerical_failure;
    }
    const auto joint = transport::joint_light_pdf(
        transport::probability_density(selection_probability,
                                       transport::ProbabilityMeasure::discrete),
        transport::probability_density(conditional_probability,
                                       transport::ProbabilityMeasure::solid_angle));
    if (!transport::succeeded(joint.status)) {
        return SceneDeviceStatus::numerical_failure;
    }
    const auto mis = transport::mis_weight(
        static_cast<transport::MisHeuristic>(heuristic),
        transport::probability_density(previous.probability_value,
                                       transport::ProbabilityMeasure::solid_angle),
        joint.value);
    if (!transport::succeeded(mis.status)) {
        return SceneDeviceStatus::numerical_failure;
    }
    weight = mis.value;
    return SceneDeviceStatus::valid;
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

[[nodiscard]] __device__ bool accumulate_point_endpoint_support(const float direction,
                                                                const float error,
                                                                float& contraction) noexcept {
    const auto candidate = fmaf(fabsf(direction), error, contraction);
    if (!isfinite(candidate) || candidate < contraction) {
        return false;
    }
    if (direction != 0.0F && error != 0.0F) {
        contraction = nextafterf(candidate, __int_as_float(0x7F800000));
        return isfinite(contraction);
    }
    contraction = candidate;
    return true;
}

[[nodiscard]] __device__ bool contract_point_endpoint(const Vector3 point,
                                                      const Vector3 position_error,
                                                      const Vector3 direction_to_light,
                                                      Vector3& contracted) noexcept {
    if (!finite_vector(point) || !finite_vector(position_error) || position_error.x < 0.0F ||
        position_error.y < 0.0F || position_error.z < 0.0F || !finite_vector(direction_to_light)) {
        return false;
    }
    const auto positive_infinity = __int_as_float(0x7F800000);
    const auto negative_infinity = __int_as_float(0xFF800000);
    auto contraction = 0.0F;
    if (!accumulate_point_endpoint_support(direction_to_light.x, position_error.x, contraction) ||
        !accumulate_point_endpoint_support(direction_to_light.y, position_error.y, contraction) ||
        !accumulate_point_endpoint_support(direction_to_light.z, position_error.z, contraction)) {
        return false;
    }
    contracted = point;
    if (contraction == 0.0F) {
        return true;
    }
    contracted = Vector3{
        .x = fmaf(-direction_to_light.x, contraction, point.x),
        .y = fmaf(-direction_to_light.y, contraction, point.y),
        .z = fmaf(-direction_to_light.z, contraction, point.z),
    };
    if (direction_to_light.x != 0.0F) {
        contracted.x = nextafterf(contracted.x, direction_to_light.x > 0.0F ? negative_infinity
                                                                            : positive_infinity);
    }
    if (direction_to_light.y != 0.0F) {
        contracted.y = nextafterf(contracted.y, direction_to_light.y > 0.0F ? negative_infinity
                                                                            : positive_infinity);
    }
    if (direction_to_light.z != 0.0F) {
        contracted.z = nextafterf(contracted.z, direction_to_light.z > 0.0F ? negative_infinity
                                                                            : positive_infinity);
    }
    return finite_vector(contracted);
}

struct LocalFrame final {
    Vector3 tangent{};
    Vector3 bitangent{};
    Vector3 normal{};
};

[[nodiscard]] __device__ bool orthonormal_frame(const LocalFrame& frame) noexcept {
    constexpr auto tolerance = 128.0F * FloatEpsilon;
    return finite_vector(frame.tangent) && finite_vector(frame.bitangent) &&
           finite_vector(frame.normal) &&
           fabsf(dot(frame.tangent, frame.tangent) - 1.0F) <= tolerance &&
           fabsf(dot(frame.bitangent, frame.bitangent) - 1.0F) <= tolerance &&
           fabsf(dot(frame.normal, frame.normal) - 1.0F) <= tolerance &&
           fabsf(dot(frame.tangent, frame.bitangent)) <= tolerance &&
           fabsf(dot(frame.tangent, frame.normal)) <= tolerance &&
           fabsf(dot(frame.bitangent, frame.normal)) <= tolerance &&
           fabsf(dot(cross(frame.tangent, frame.bitangent), frame.normal) - 1.0F) <= tolerance;
}

[[nodiscard]] __device__ bool make_local_frame(const Vector3 normal, LocalFrame& frame) noexcept {
    auto unit_normal = Vector3{};
    if (!normalize(normal, unit_normal)) {
        return false;
    }
    const auto sign = copysignf(1.0F, unit_normal.z);
    const auto coefficient = -1.0F / (sign + unit_normal.z);
    const auto product = unit_normal.x * unit_normal.y * coefficient;
    auto tangent_seed = Vector3{
        .x = 1.0F + sign * unit_normal.x * unit_normal.x * coefficient,
        .y = sign * product,
        .z = -sign * unit_normal.x,
    };
    auto tangent = Vector3{};
    if (!normalize(tangent_seed, tangent)) {
        return false;
    }
    tangent_seed = subtract(tangent, multiply(unit_normal, dot(tangent, unit_normal)));
    if (!normalize(tangent_seed, tangent)) {
        return false;
    }
    auto bitangent = Vector3{};
    if (!normalize(cross(unit_normal, tangent), bitangent) ||
        !normalize(cross(bitangent, unit_normal), tangent)) {
        return false;
    }
    const auto candidate =
        LocalFrame{.tangent = tangent, .bitangent = bitangent, .normal = unit_normal};
    if (!orthonormal_frame(candidate)) {
        return false;
    }
    frame = candidate;
    return true;
}

[[nodiscard]] __device__ Vector3 to_local(const LocalFrame& frame,
                                          const Vector3 direction) noexcept {
    return Vector3{.x = dot(direction, frame.tangent),
                   .y = dot(direction, frame.bitangent),
                   .z = dot(direction, frame.normal)};
}

[[nodiscard]] __device__ bool to_world(const LocalFrame& frame, const transport::Vector3 local,
                                       Vector3& direction) noexcept {
    return normalize(add(add(multiply(frame.tangent, local.x), multiply(frame.bitangent, local.y)),
                         multiply(frame.normal, local.z)),
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
    const auto& ray = streams.rays[slot.value];
    const auto& previous = streams.previous_bsdf_samples[slot.value];
    if (!valid_path_state(state) || !valid_ray(ray) ||
        (state.depth == 0U ? previous.valid != 0U : !valid_previous_bsdf_sample(previous, ray))) {
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
        if (!load_spectrum(scene_bytes, scene, scene_column::environment_radiance, 0U, radiance)) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] = outcome(WavefrontStageStatus::numerical_failure,
                                      WavefrontStageRoute::none, slot.value);
            return;
        }
        const auto accumulated =
            transport::checked_accumulate_product(state.accumulated_radiance, state.beta, radiance);
        if (!transport::succeeded(accumulated.status)) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] = outcome(WavefrontStageStatus::numerical_failure,
                                      WavefrontStageRoute::none, slot.value);
            return;
        }
        state.accumulated_radiance = accumulated.value;
    }
    control.flags = 0U;
    control.blocked_depth_limits = 0U;
    finish_phase(control, WavefrontLanePhase::terminated,
                 WavefrontTermination::escaped_environment);
    outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::terminated,
                              slot.value, value(WavefrontTermination::escaped_environment));
}

__global__ void
shade_stage_kernel(const std::uint8_t* const scene_bytes, const std::size_t scene_size,
                   const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
                   const WavefrontTransportConfig config, const std::uint32_t work_count,
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
    if (!valid_transport_config(config, scene)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(WavefrontStageStatus::invalid_contract, WavefrontStageRoute::none,
                                  slot.value, value(ShadeFailureDetail::transport_config));
        return;
    }
    auto& state = streams.path_states[slot.value];
    auto& ray = streams.rays[slot.value];
    const auto& hit = streams.hits[slot.value];
    auto& previous = streams.previous_bsdf_samples[slot.value];
    if (!valid_path_state(state) || !valid_ray(ray) || !valid_hit(hit) ||
        ray.current_medium != state.current_medium ||
        (state.depth == 0U ? previous.valid != 0U : !valid_previous_bsdf_sample(previous, ray))) {
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
    auto outgoing_world = Vector3{
        .x = -ray.direction_x,
        .y = -ray.direction_y,
        .z = -ray.direction_z,
    };
    if (!normalize(outgoing_world, outgoing_world)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::surface_data));
        return;
    }
    auto front_facing = false;
    if (!one_sided_emission_support(surface.geometric_normal, outgoing_world, front_facing)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::emission_orientation));
        return;
    }
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
    if (front_facing && !zero_spectrum(emitted)) {
        auto emission_weight = 1.0F;
        const auto weighted = emissive_hit_mis_weight(
            scene_bytes, scene, hit, surface, ray, previous, config.mis_heuristic, emission_weight);
        if (weighted != SceneDeviceStatus::valid) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] = outcome(scene_status(weighted), WavefrontStageRoute::none, slot.value,
                                      value(ShadeFailureDetail::emission_mis));
            return;
        }
        auto accumulated = state.accumulated_radiance;
        for (auto lane = std::uint32_t{}; lane < shared::HostDeviceSpectrumLaneCount; ++lane) {
            if (state.beta.values[lane] == 0.0F || emitted.values[lane] == 0.0F ||
                emission_weight == 0.0F) {
                continue;
            }
            const float numerators[]{state.beta.values[lane], emitted.values[lane],
                                     emission_weight};
            const float denominators[]{1.0F};
            const auto contribution = transport::checked_product_quotient(numerators, denominators);
            if (!transport::succeeded(contribution.status)) {
                finish_phase(control, WavefrontLanePhase::terminated);
                outcomes[index] =
                    outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                            slot.value, value(ShadeFailureDetail::emitted_radiance));
                return;
            }
            accumulated.values[lane] += contribution.value;
            if (!isfinite(accumulated.values[lane])) {
                finish_phase(control, WavefrontLanePhase::terminated);
                outcomes[index] =
                    outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                            slot.value, value(ShadeFailureDetail::emitted_radiance));
                return;
            }
        }
        state.accumulated_radiance = accumulated;
    }
    if (state.diffuse_depth >= config.diffuse_depth_limit) {
        control.flags = 0U;
        control.blocked_depth_limits = 0x00000001U;
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
    if (!front_facing || !(dot(surface.shading_normal, outgoing_world) > 0.0F)) {
        control.flags = 0U;
        control.blocked_depth_limits = 0U;
        finish_phase(control, WavefrontLanePhase::terminated,
                     WavefrontTermination::outside_bsdf_support);
        outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::terminated,
                                  slot.value, value(WavefrontTermination::outside_bsdf_support));
        return;
    }

    auto sample_dimensions = SampleStreamBounceDimensions{};
    const auto sample_dimension_status = cuda::sample_stream::dimensions_for_bounce(
        static_cast<std::uint64_t>(state.depth), sample_dimensions);
    if (sample_dimension_status != SampleStreamDumpStatus::success) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::invalid_lane_state, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::sample_dimensions));
        return;
    }

    auto shade_detail = std::uint32_t{};
    auto incident = IncidentLight{};
    if (config.light_count != 0U && !zero_spectrum(state.beta) && !zero_spectrum(reflectance)) {
        shade_detail = cuda::WavefrontShadeDetailLightSampled;
        const auto sampled_light =
            sample_registered_light(scene_bytes, scene, state, streams.sample_streams[slot.value],
                                    sample_dimensions, surface, ray.visibility_mask, incident);
        if (sampled_light != SceneDeviceStatus::valid) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] = outcome(scene_status(sampled_light), WavefrontStageRoute::none,
                                      slot.value, value(ShadeFailureDetail::light_sample));
            return;
        }
    }

    auto pending = WavefrontPendingShadow{};
    auto has_shadow = false;
    if (incident.kind != static_cast<std::uint32_t>(IncidentLightKind::none)) {
        const auto receiver_cosine = dot(surface.shading_normal, incident.direction_to_light);
        const auto geometric_receiver_cosine =
            dot(surface.geometric_normal, incident.direction_to_light);
        if (receiver_cosine > 0.0F && geometric_receiver_cosine > 0.0F) {
            auto estimator_weight = 1.0F;
            if (incident.probability_measure == SolidAngleMeasure) {
                const auto joint = transport::joint_light_pdf(
                    transport::probability_density(incident.selection_probability,
                                                   transport::ProbabilityMeasure::discrete),
                    transport::probability_density(incident.conditional_probability,
                                                   transport::ProbabilityMeasure::solid_angle));
                const auto bsdf = transport::probability_density(
                    receiver_cosine * InversePi, transport::ProbabilityMeasure::solid_angle);
                const auto mis = transport::succeeded(joint.status)
                                     ? transport::mis_weight(static_cast<transport::MisHeuristic>(
                                                                 config.mis_heuristic),
                                                             joint.value, bsdf)
                                     : transport::ScalarResult{};
                if (!transport::succeeded(joint.status) || !transport::succeeded(mis.status)) {
                    finish_phase(control, WavefrontLanePhase::terminated);
                    outcomes[index] =
                        outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                                slot.value, value(ShadeFailureDetail::light_weight));
                    return;
                }
                estimator_weight = mis.value;
            } else if (incident.probability_measure != DiscreteMeasure) {
                finish_phase(control, WavefrontLanePhase::terminated);
                outcomes[index] =
                    outcome(WavefrontStageStatus::unsupported_transport, WavefrontStageRoute::none,
                            slot.value, value(ShadeFailureDetail::light_weight));
                return;
            }
            pending.beta = state.beta;
            pending.reflectance = reflectance;
            pending.incident_radiance = incident.radiance;
            pending.receiver_cosine = receiver_cosine;
            pending.estimator_weight = estimator_weight;
            pending.selection_probability = incident.selection_probability;
            pending.conditional_probability = incident.conditional_probability;
            auto source_offset = Vector3{};
            if (!offset_point(surface.position, surface.position_error, surface.geometric_normal,
                              incident.direction_to_light, source_offset)) {
                finish_phase(control, WavefrontLanePhase::terminated);
                outcomes[index] =
                    outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                            slot.value, value(ShadeFailureDetail::endpoint_offset));
                return;
            }
            if (incident.kind == static_cast<std::uint32_t>(IncidentLightKind::infinite)) {
                pending.ray = TransportRay{
                    .origin_x = source_offset.x,
                    .origin_y = source_offset.y,
                    .origin_z = source_offset.z,
                    .t_min = 0.0F,
                    .direction_x = incident.direction_to_light.x,
                    .direction_y = incident.direction_to_light.y,
                    .direction_z = incident.direction_to_light.z,
                    .t_max = __int_as_float(0x7F800000),
                    .time = ray.time,
                    .visibility_mask = ray.visibility_mask,
                    .current_medium = state.current_medium,
                    .reserved = 0U,
                };
            } else {
                auto endpoint = incident.endpoint_position;
                if (incident.kind ==
                    static_cast<std::uint32_t>(IncidentLightKind::finite_surface)) {
                    if (!offset_point(incident.endpoint_position, incident.endpoint_position_error,
                                      incident.endpoint_geometric_normal,
                                      multiply(incident.direction_to_light, -1.0F), endpoint)) {
                        finish_phase(control, WavefrontLanePhase::terminated);
                        outcomes[index] = outcome(WavefrontStageStatus::numerical_failure,
                                                  WavefrontStageRoute::none, slot.value,
                                                  value(ShadeFailureDetail::endpoint_offset));
                        return;
                    }
                } else if (!contract_point_endpoint(incident.endpoint_position,
                                                    incident.endpoint_position_error,
                                                    incident.direction_to_light, endpoint)) {
                    finish_phase(control, WavefrontLanePhase::terminated);
                    outcomes[index] =
                        outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                                slot.value, value(ShadeFailureDetail::endpoint_offset));
                    return;
                }
                auto segment = ScaledSegment{};
                if (!scaled_segment(subtract(endpoint, source_offset), segment) ||
                    !(dot(segment.direction, incident.direction_to_light) > 0.0F)) {
                    finish_phase(control, WavefrontLanePhase::terminated);
                    outcomes[index] =
                        outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                                slot.value, value(ShadeFailureDetail::shadow_segment));
                    return;
                }
                const auto maximum = nextafterf(segment.length, 0.0F);
                pending.ray = TransportRay{
                    .origin_x = source_offset.x,
                    .origin_y = source_offset.y,
                    .origin_z = source_offset.z,
                    .t_min = 0.0F,
                    .direction_x = segment.direction.x,
                    .direction_y = segment.direction.y,
                    .direction_z = segment.direction.z,
                    .t_max = maximum,
                    .time = ray.time,
                    .visibility_mask = ray.visibility_mask,
                    .current_medium = state.current_medium,
                    .reserved = 0U,
                };
            }
            if (!valid_ray(pending.ray) || !(pending.ray.t_max > 0.0F)) {
                finish_phase(control, WavefrontLanePhase::terminated);
                outcomes[index] =
                    outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                            slot.value, value(ShadeFailureDetail::shadow_ray));
                return;
            }
            has_shadow = true;
        }
    }

    shade_detail |= cuda::WavefrontShadeDetailClosureSampled;
    const auto bsdf_component = cuda::sample_stream::sample_1d(streams.sample_streams[slot.value],
                                                               sample_dimensions.bsdf_component);
    const auto bsdf_u = cuda::sample_stream::sample_1d(streams.sample_streams[slot.value],
                                                       sample_dimensions.bsdf_u);
    const auto bsdf_v = cuda::sample_stream::sample_1d(streams.sample_streams[slot.value],
                                                       sample_dimensions.bsdf_v);
    auto frame = LocalFrame{};
    if (!isfinite(bsdf_component) || bsdf_component < 0.0F || !(bsdf_component < 1.0F) ||
        !make_local_frame(surface.shading_normal, frame)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::bsdf_sample));
        return;
    }
    const auto outgoing_local_world = to_local(frame, outgoing_world);
    const auto outgoing_local = transport::Vector3{
        .x = outgoing_local_world.x, .y = outgoing_local_world.y, .z = outgoing_local_world.z};
    const auto bsdf_sample = transport::sample_lambert(reflectance, outgoing_local, bsdf_u, bsdf_v);
    auto outgoing = Vector3{};
    auto next_origin = Vector3{};
    if (bsdf_sample.status != transport::Status::success ||
        !to_world(frame, bsdf_sample.incoming_local, outgoing)) {
        if (bsdf_sample.status == transport::Status::outside_support) {
            pending.continuation_pending = 0U;
            pending.termination = value(WavefrontTermination::outside_bsdf_support);
            if (has_shadow) {
                streams.pending_shadows[slot.value] = pending;
                control.flags = cuda::WavefrontLaneShadowPending;
                control.blocked_depth_limits = 0U;
                const auto routed = push_route(queues, ShadowQueue, slot);
                if (routed != WavefrontStageStatus::success) {
                    finish_phase(control, WavefrontLanePhase::terminated);
                    outcomes[index] =
                        outcome(routed, WavefrontStageRoute::none, slot.value, ShadowQueue);
                    return;
                }
                finish_phase(control, WavefrontLanePhase::shadow);
                outcomes[index] = outcome(WavefrontStageStatus::success,
                                          WavefrontStageRoute::shadow, slot.value, shade_detail);
                return;
            }
            control.flags = 0U;
            control.blocked_depth_limits = 0U;
            finish_phase(control, WavefrontLanePhase::terminated,
                         WavefrontTermination::outside_bsdf_support);
            outcomes[index] =
                outcome(WavefrontStageStatus::success, WavefrontStageRoute::terminated, slot.value,
                        value(WavefrontTermination::outside_bsdf_support) | shade_detail);
            return;
        }
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::bsdf_sample));
        return;
    }
    if (!(bsdf_sample.incoming_local.z > 0.0F) ||
        !(dot(surface.geometric_normal, outgoing) > 0.0F)) {
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
    previous = WavefrontPreviousBsdfSample{
        .context_x = surface.position.x,
        .context_y = surface.position.y,
        .context_z = surface.position.z,
        .context_time = ray.time,
        .incoming_x = outgoing.x,
        .incoming_y = outgoing.y,
        .incoming_z = outgoing.z,
        .probability_value = bsdf_sample.probability.value,
        .probability_measure = SolidAngleMeasure,
        .valid = 1U,
        .reserved = {0U, 0U},
    };

    auto continuation_pending = true;
    auto pending_termination = WavefrontTermination::none;
    if (zero_spectrum(state.beta)) {
        continuation_pending = false;
        pending_termination = WavefrontTermination::zero_throughput;
    } else if (config.russian_roulette_mode == EnabledRussianRoulette &&
               state.depth >= config.russian_roulette_first_depth) {
        const auto roulette = transport::evaluate_russian_roulette(
            state.beta, state.eta_scale, state.depth,
            cuda::sample_stream::sample_1d(streams.sample_streams[slot.value],
                                           sample_dimensions.russian_roulette),
            transport::RussianRoulettePolicy{
                .mode = transport::RussianRouletteMode::enabled,
                .reserved = {0U, 0U, 0U},
                .first_eligible_depth = config.russian_roulette_first_depth,
                .minimum_survival_probability = config.russian_roulette_minimum_probability,
                .maximum_survival_probability = config.russian_roulette_maximum_probability,
            });
        if (!transport::succeeded(roulette.status)) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] =
                outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                        slot.value, value(ShadeFailureDetail::russian_roulette));
            return;
        }
        if (roulette.outcome == transport::RussianRouletteOutcome::survived) {
            state.beta = roulette.throughput;
        } else if (roulette.outcome == transport::RussianRouletteOutcome::terminated) {
            continuation_pending = false;
            pending_termination = WavefrontTermination::russian_roulette;
        } else {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] =
                outcome(WavefrontStageStatus::invalid_lane_state, WavefrontStageRoute::none,
                        slot.value, value(ShadeFailureDetail::russian_roulette));
            return;
        }
    }

    if (has_shadow) {
        pending.continuation_pending = continuation_pending ? 1U : 0U;
        pending.termination = value(pending_termination);
        streams.pending_shadows[slot.value] = pending;
        control.flags = cuda::WavefrontLaneShadowPending |
                        (continuation_pending ? cuda::WavefrontLaneContinuationPending : 0U);
        control.blocked_depth_limits = 0U;
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
    streams.pending_shadows[slot.value] = WavefrontPendingShadow{};
    control.flags = 0U;
    control.blocked_depth_limits = 0U;
    if (!continuation_pending) {
        finish_phase(control, WavefrontLanePhase::terminated, pending_termination);
        outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::terminated,
                                  slot.value, value(pending_termination) | shade_detail);
        return;
    }
    const auto routed = push_route(queues, ContinuationQueue, slot);
    if (routed != WavefrontStageStatus::success) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(routed, WavefrontStageRoute::none, slot.value, ContinuationQueue);
        return;
    }
    finish_phase(control, WavefrontLanePhase::continuation);
    outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::continuation,
                              slot.value, shade_detail);
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
    const auto terminal_reason =
        pending.termination == value(WavefrontTermination::zero_throughput) ||
        pending.termination == value(WavefrontTermination::outside_bsdf_support) ||
        pending.termination == value(WavefrontTermination::russian_roulette);
    if (control.phase != value(WavefrontLanePhase::shadow) || control.blocked_depth_limits != 0U ||
        (control.flags & cuda::WavefrontLaneShadowPending) == 0U ||
        (control.flags & ~known_flags) != 0U || pending.continuation_pending > 1U ||
        ((control.flags & cuda::WavefrontLaneContinuationPending) != 0U) != continuation_pending ||
        (continuation_pending && pending.termination != value(WavefrontTermination::none)) ||
        (!continuation_pending && !terminal_reason) || pending.reserved[0] != 0U ||
        pending.reserved[1] != 0U || !valid_ray(pending.ray) ||
        !valid_pending_shadow_radiometry(pending)) {
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
    const auto terminal_reason =
        pending.termination == value(WavefrontTermination::zero_throughput) ||
        pending.termination == value(WavefrontTermination::outside_bsdf_support) ||
        pending.termination == value(WavefrontTermination::russian_roulette);
    if ((control.flags & cuda::WavefrontLaneShadowPending) == 0U ||
        (control.flags & ~known_flags) != 0U || pending.continuation_pending > 1U ||
        ((control.flags & cuda::WavefrontLaneContinuationPending) != 0U) != continuation_pending ||
        (continuation_pending && pending.termination != value(WavefrontTermination::none)) ||
        (!continuation_pending && !terminal_reason) || control.blocked_depth_limits != 0U ||
        pending.reserved[0] != 0U || pending.reserved[1] != 0U || !valid_ray(pending.ray) ||
        !valid_pending_shadow_radiometry(pending)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(WavefrontStageStatus::invalid_lane_state,
                                  WavefrontStageRoute::none, queue_path.value);
        return;
    }
    if (visible) {
        auto contribution = TransportSpectrum{};
        for (auto lane = std::uint32_t{}; lane < shared::HostDeviceSpectrumLaneCount; ++lane) {
            const float denominators[]{pending.selection_probability,
                                       pending.conditional_probability};
            if (pending.estimator_weight == 1.0F) {
                const float numerators[]{pending.beta.values[lane],
                                         pending.reflectance.values[lane],
                                         InversePi,
                                         pending.incident_radiance.values[lane],
                                         pending.receiver_cosine,
                                         1.0F};
                if (!checked_product_quotient(numerators, denominators,
                                              contribution.values[lane])) {
                    finish_phase(control, WavefrontLanePhase::terminated);
                    outcomes[index] =
                        outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                                queue_path.value, value(ShadeFailureDetail::light_contribution));
                    return;
                }
            } else {
                const float numerators[]{pending.beta.values[lane],
                                         pending.reflectance.values[lane],
                                         InversePi,
                                         pending.incident_radiance.values[lane],
                                         pending.receiver_cosine,
                                         1.0F,
                                         pending.estimator_weight};
                if (!checked_product_quotient(numerators, denominators,
                                              contribution.values[lane])) {
                    finish_phase(control, WavefrontLanePhase::terminated);
                    outcomes[index] =
                        outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                                queue_path.value, value(ShadeFailureDetail::light_contribution));
                    return;
                }
            }
        }
        auto& radiance = streams.path_states[queue_path.value].accumulated_radiance;
        const auto accumulated = transport::checked_accumulate(radiance, contribution);
        if (!transport::succeeded(accumulated.status)) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] = outcome(WavefrontStageStatus::numerical_failure,
                                      WavefrontStageRoute::none, queue_path.value);
            return;
        }
        radiance = accumulated.value;
    }
    const auto pending_termination = pending.termination;
    pending = WavefrontPendingShadow{};
    control.flags = 0U;
    if (!continuation_pending) {
        if (!terminal_reason) {
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
        !valid_path_state(streams.path_states[slot.value]) ||
        !valid_previous_bsdf_sample(streams.previous_bsdf_samples[slot.value],
                                    streams.rays[slot.value]) ||
        control.blocked_depth_limits != 0U) {
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
    return streams.reserved == 0U &&
           (streams.capacity == 0U ||
            (streams.sample_streams != nullptr && streams.rays != nullptr &&
             streams.path_states != nullptr && streams.hits != nullptr &&
             streams.pending_shadows != nullptr && streams.previous_bsdf_samples != nullptr &&
             streams.controls != nullptr));
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
    const WavefrontTransportConfig config, const std::uint32_t work_count,
    WavefrontStageOutcome* const outcomes) noexcept {
    if (!host_views_are_valid(queues, streams) ||
        (work_count != 0U &&
         (scene_bytes == nullptr || scene_size < sizeof(SceneSoaHeader) || outcomes == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    shade_stage_kernel<<<block_count(work_count), ThreadsPerBlock>>>(
        scene_bytes, scene_size, queues, streams, config, work_count, outcomes);
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
