#include <Blackframe/XPU/CUDA/SampleStreamDevice.cuh>
#include <Blackframe/XPU/CUDA/TransportDevice.cuh>
#include <Blackframe/XPU/CUDA/TransportLobesDevice.cuh>
#include <Blackframe/XPU/CUDA/WavefrontQueueDevice.cuh>
#include <Blackframe/XPU/CUDA/WavefrontStageKernel.hpp>
#include <Blackframe/XPU/Shared/SceneSoaAbi.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <type_traits>

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
using cuda::WavefrontPendingBsdfEncoding;
using cuda::WavefrontPendingShadow;
using cuda::WavefrontPreviousBsdfSample;
using cuda::WavefrontQueueDeviceSoa;
using cuda::WavefrontRayCone;
using cuda::WavefrontStageAudit;
using cuda::WavefrontStageDeviceSoa;
using cuda::WavefrontStageKind;
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
constexpr auto MaximumMaterialClosureCount =
    static_cast<std::uint64_t>(transport::MaximumClosureCount);
constexpr auto MaximumU64 = ~std::uint64_t{0U};
constexpr auto FloatEpsilon = 0x1p-23F;
constexpr auto FloatDenormMinimum = 0x1p-149F;
constexpr auto Gamma7 = (7.0F * FloatEpsilon) / (1.0F - 7.0F * FloatEpsilon);
constexpr auto Gamma7Double =
    (7.0 * static_cast<double>(FloatEpsilon)) / (1.0 - 7.0 * static_cast<double>(FloatEpsilon));
constexpr auto MaximumFloat = 0x1.fffffeP+127F;
constexpr auto ExactFloatTexelLimit = 8388608.0F;
constexpr auto MaximumEwaAnisotropy = std::uint32_t{64U};
constexpr auto DataTextureColorSpace = std::uint32_t{0U};
constexpr auto MaximumTextureWrapMode = std::uint32_t{3U};
constexpr auto MaximumNormalYConvention = std::uint32_t{1U};

struct Vector3 final {
    float x{};
    float y{};
    float z{};
};

struct Vector2 final {
    float x{};
    float y{};
};

struct SurfaceData final {
    Vector3 position{};
    Vector3 position_error{};
    Vector3 geometric_normal{};
    Vector3 shading_normal{};
    Vector2 uv{};
    Vector3 dpdu{};
    Vector3 dpdv{};
    std::uint32_t material_index{};
};

struct TextureCoordinateDifferentials final {
    float dudx{};
    float dvdx{};
    float dudy{};
    float dvdy{};
};

struct SceneImageTexture final {
    std::uint64_t mip_offset{};
    std::uint64_t mip_count{};
    std::uint32_t channel_count{};
};

struct SceneImageMip final {
    std::uint64_t texel_offset{};
    std::uint64_t texel_count{};
    std::uint32_t width{};
    std::uint32_t height{};
};

struct EwaFootprint final {
    float major_x{1.0F};
    float major_y{};
    float minor_x{};
    float minor_y{1.0F};
    float major_radius{1.0F};
    float minor_radius{1.0F};
    float minor_length{};
};

struct EwaEllipse final {
    float center_x{};
    float center_y{};
    float major_x{1.0F};
    float major_y{};
    float minor_x{};
    float minor_y{1.0F};
    float inverse_major_radius{1.0F};
    float inverse_minor_radius{1.0F};
    std::int64_t minimum_x{};
    std::int64_t maximum_x{};
    std::int64_t minimum_y{};
    std::int64_t maximum_y{};
    std::uint64_t visit_count{};
};

struct SceneMaterialClosureRange final {
    std::uint64_t offset;
    std::uint64_t count;
    float tangent_rotation_radians;
    std::uint8_t frame_mode;
    std::uint8_t spectral_present;
    std::uint8_t reserved[2U];
};

static_assert(std::is_trivial_v<SceneMaterialClosureRange>);
static_assert(std::is_standard_layout_v<SceneMaterialClosureRange>);
static_assert(sizeof(SceneMaterialClosureRange) == 24U);
static_assert(shared::SceneSoaSpectrumLaneCount == shared::HostDeviceSpectrumLaneCount);
static_assert(shared::SceneSoaClosureParameterScalarCount ==
              transport::ClosureParameterScalarCount);

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
    closure_mixture = 22U,
    closure_frame = 23U,
    depth_event = 24U,
    shading_normal_correction = 25U,
    probability_measure = 26U,
    eta_scale = 27U,
    geometric_support = 28U,
    ray_cone_advance = 29U,
    ray_cone_scattering = 30U,
    surface_maps = 31U,
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
    const auto encoding = static_cast<WavefrontPendingBsdfEncoding>(pending.bsdf_encoding);
    const auto known_encoding = encoding == WavefrontPendingBsdfEncoding::value ||
                                encoding == WavefrontPendingBsdfEncoding::lambertian_coefficient;
    return nonnegative_spectrum(pending.beta) && !zero_spectrum(pending.beta) &&
           nonnegative_spectrum(pending.bsdf_factor) && !zero_spectrum(pending.bsdf_factor) &&
           nonnegative_spectrum(pending.incident_radiance) &&
           !zero_spectrum(pending.incident_radiance) &&
           isfinite(pending.absolute_incoming_cosine) && pending.absolute_incoming_cosine > 0.0F &&
           pending.absolute_incoming_cosine <= 1.0F && isfinite(pending.estimator_weight) &&
           pending.estimator_weight >= 0.0F && pending.estimator_weight <= 1.0F &&
           isfinite(pending.selection_probability) && pending.selection_probability > 0.0F &&
           pending.selection_probability <= 1.0F && isfinite(pending.conditional_probability) &&
           pending.conditional_probability > 0.0F && isfinite(pending.shading_normal_correction) &&
           pending.shading_normal_correction > 0.0F && known_encoding;
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

[[nodiscard]] __device__ bool valid_ray_cone(const WavefrontRayCone cone) noexcept {
    return isfinite(cone.width) && cone.width >= 0.0F && isfinite(cone.spread) &&
           cone.spread >= 0.0F;
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
                             static_cast<std::uint64_t>(state.volume_depth);
    const auto surface_depth = static_cast<std::uint64_t>(state.diffuse_depth) +
                               static_cast<std::uint64_t>(state.glossy_depth) +
                               static_cast<std::uint64_t>(state.specular_depth);
    if (total_depth != state.depth || state.transmission_depth > surface_depth ||
        (state.delta_flags & ~3U) != 0U) {
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
    return streams.reserved == 0U && streams.reserved_tail[0U] == 0U &&
           streams.reserved_tail[1U] == 0U &&
           (streams.capacity == 0U ||
            (streams.sample_streams != nullptr && streams.rays != nullptr &&
             streams.ray_cones != nullptr && streams.path_states != nullptr &&
             streams.hits != nullptr && streams.pending_shadows != nullptr &&
             streams.previous_bsdf_samples != nullptr && streams.controls != nullptr));
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
    if (column >= scene_column::material_id && column < scene_column::closure_kind) {
        return header.material_count;
    }
    if (column >= scene_column::closure_kind && column < scene_column::instance_id) {
        return header.closure_count;
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
    if (column >= scene_column::environment_wavelength_nanometers &&
        column < scene_column::texture_id) {
        return header.environment_count;
    }
    if (column >= scene_column::texture_id && column < scene_column::material_normal_map_present) {
        return header.texture_count;
    }
    if (column >= scene_column::material_normal_map_present &&
        column < scene_column::image_texture_id) {
        return header.material_count;
    }
    if (column >= scene_column::image_texture_id && column < scene_column::image_mip_width) {
        return header.image_texture_count;
    }
    if (column >= scene_column::image_mip_width && column < scene_column::image_texel_value) {
        return header.image_mip_count;
    }
    if (column == scene_column::image_texel_value) {
        return header.image_texel_count;
    }
    return 0U;
}

[[nodiscard]] __device__ std::uint32_t
scene_column_element_size(const std::uint32_t column) noexcept {
    if (column >= scene_column::count) {
        return 0U;
    }
    if ((column >= scene_column::geometry_vertex_offset &&
         column <= scene_column::geometry_triangle_count) ||
        column == scene_column::material_closure_offset ||
        column == scene_column::material_closure_count) {
        return sizeof(std::uint64_t);
    }
    if (column == scene_column::material_spectral_present ||
        (column >= scene_column::material_wavelength_measure &&
         column < scene_column::material_closure_offset) ||
        column == scene_column::material_closure_frame_mode ||
        column == scene_column::instance_parent_present ||
        column == scene_column::material_normal_map_present ||
        column == scene_column::material_bump_map_present ||
        (column >= scene_column::environment_wavelength_measure &&
         column < scene_column::environment_radiance)) {
        return sizeof(std::uint8_t);
    }
    if ((column >= scene_column::position_x && column <= scene_column::texture_coordinate_y) ||
        (column >= scene_column::material_wavelength_nanometers &&
         column < scene_column::material_wavelength_measure) ||
        column == scene_column::material_closure_tangent_rotation_radians ||
        (column >= scene_column::material_emitted_radiance &&
         column < scene_column::closure_kind) ||
        (column >= scene_column::closure_weight && column < scene_column::instance_id) ||
        (column >= scene_column::instance_local_to_parent &&
         column < scene_column::punctual_kind) ||
        (column >= scene_column::punctual_position_x &&
         column < scene_column::mesh_area_light_instance_id) ||
        (column >= scene_column::environment_wavelength_nanometers &&
         column < scene_column::environment_wavelength_measure) ||
        (column >= scene_column::environment_radiance && column < scene_column::texture_id) ||
        (column >= scene_column::texture_value &&
         column < scene_column::material_normal_map_present) ||
        column == scene_column::material_bump_map_scale ||
        column == scene_column::image_texel_value) {
        return sizeof(float);
    }
    if (column == scene_column::image_texture_mip_offset ||
        column == scene_column::image_texture_mip_count ||
        column == scene_column::image_mip_texel_offset ||
        column == scene_column::image_mip_texel_count) {
        return sizeof(std::uint64_t);
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
        header->closure_count > 0xFFFFFFFFULL || header->instance_count > 0xFFFFFFFFULL ||
        header->mesh_area_light_count > 0xFFFFFFFFULL ||
        header->punctual_light_count > 0xFFFFFFFFULL || header->texture_count > 0xFFFFFFFFULL ||
        header->image_texture_count > 0xFFFFFFFFULL || header->image_mip_count > 0xFFFFFFFFULL) {
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

[[nodiscard]] __device__ SceneDeviceStatus load_material_closure_range(
    const std::uint8_t* const bytes, const SceneSoaHeader& scene,
    const std::uint32_t material_index, SceneMaterialClosureRange& range) noexcept {
    if (material_index >= scene.material_count) {
        return SceneDeviceStatus::invalid_scene;
    }

    auto loaded = SceneMaterialClosureRange{};
    loaded.offset = scene_values<std::uint64_t>(
        bytes, scene, scene_column::material_closure_offset)[material_index];
    loaded.count = scene_values<std::uint64_t>(
        bytes, scene, scene_column::material_closure_count)[material_index];
    loaded.frame_mode = scene_values<std::uint8_t>(
        bytes, scene, scene_column::material_closure_frame_mode)[material_index];
    loaded.spectral_present = scene_values<std::uint8_t>(
        bytes, scene, scene_column::material_spectral_present)[material_index];
    loaded.tangent_rotation_radians = scene_values<float>(
        bytes, scene, scene_column::material_closure_tangent_rotation_radians)[material_index];

    if (loaded.spectral_present > 1U || loaded.frame_mode > 1U ||
        !isfinite(loaded.tangent_rotation_radians) || loaded.tangent_rotation_radians < -Pi ||
        !(loaded.tangent_rotation_radians < Pi) || loaded.count > MaximumMaterialClosureCount ||
        loaded.offset > scene.closure_count || loaded.count > scene.closure_count - loaded.offset) {
        return SceneDeviceStatus::invalid_scene;
    }
    if (loaded.spectral_present == 0U && (loaded.count != 0U || loaded.frame_mode != 0U ||
                                          loaded.tangent_rotation_radians != 0.0F)) {
        return SceneDeviceStatus::invalid_scene;
    }

    range = loaded;
    return SceneDeviceStatus::valid;
}

[[nodiscard]] __device__ SceneDeviceStatus load_scene_closure_record(
    const std::uint8_t* const bytes, const SceneSoaHeader& scene, const std::uint64_t closure_index,
    transport::ClosureRecord& record, float& probability) noexcept {
    if (closure_index >= scene.closure_count) {
        return SceneDeviceStatus::invalid_scene;
    }

    auto loaded = transport::ClosureRecord{};
    loaded.kind = static_cast<transport::ClosureKind>(
        scene_values<std::uint32_t>(bytes, scene, scene_column::closure_kind)[closure_index]);
    loaded.lobes = static_cast<transport::ScatteringLobe>(
        scene_values<std::uint32_t>(bytes, scene, scene_column::closure_lobes)[closure_index]);
    for (auto lane = std::uint32_t{0U}; lane < shared::SceneSoaSpectrumLaneCount; ++lane) {
        loaded.weight[lane] =
            scene_values<float>(bytes, scene, scene_column::closure_weight + lane)[closure_index];
    }
    for (auto parameter = std::uint32_t{0U};
         parameter < shared::SceneSoaClosureParameterScalarCount; ++parameter) {
        loaded.parameters[parameter] = scene_values<float>(
            bytes, scene, scene_column::closure_parameters + parameter)[closure_index];
    }
    const auto loaded_probability =
        scene_values<float>(bytes, scene, scene_column::closure_probability)[closure_index];
    if (!transport::valid_closure_record(loaded) || !isfinite(loaded_probability) ||
        !(loaded_probability > 0.0F) || loaded_probability > 1.0F) {
        return SceneDeviceStatus::invalid_scene;
    }

    record = loaded;
    probability = loaded_probability;
    return SceneDeviceStatus::valid;
}

[[nodiscard]] __device__ SceneDeviceStatus
load_material_closure_mixture(const std::uint8_t* const bytes, const SceneSoaHeader& scene,
                              const std::uint32_t material_index, SceneMaterialClosureRange& range,
                              transport::ClosureMixtureRecord& mixture) noexcept {
    auto loaded_range = SceneMaterialClosureRange{};
    const auto range_status =
        load_material_closure_range(bytes, scene, material_index, loaded_range);
    if (range_status != SceneDeviceStatus::valid) {
        return range_status;
    }
    if (loaded_range.spectral_present != 1U) {
        return SceneDeviceStatus::unsupported_transport;
    }

    auto loaded = transport::ClosureMixtureRecord{};
    loaded.active_count = static_cast<std::uint32_t>(loaded_range.count);
    for (auto index = std::uint32_t{0U}; index < loaded.active_count; ++index) {
        auto probability = 0.0F;
        const auto closure_status = load_scene_closure_record(
            bytes, scene, loaded_range.offset + index, loaded.closures[index], probability);
        if (closure_status != SceneDeviceStatus::valid) {
            return closure_status;
        }
        loaded.probabilities[index] = probability;
        loaded.cdf[index + 1U] = loaded.cdf[index] + probability;
    }
    if (!transport::valid_closure_mixture_record(loaded)) {
        return SceneDeviceStatus::invalid_scene;
    }

    range = loaded_range;
    mixture = loaded;
    return SceneDeviceStatus::valid;
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

[[nodiscard]] __device__ bool surface_derivatives(const Vector3* const positions,
                                                  const Vector2* const coordinates, Vector3& dpdu,
                                                  Vector3& dpdv) noexcept {
    const auto edge1 = subtract(positions[1U], positions[0U]);
    const auto edge2 = subtract(positions[2U], positions[0U]);
    const auto du1 = coordinates[1U].x - coordinates[0U].x;
    const auto dv1 = coordinates[1U].y - coordinates[0U].y;
    const auto du2 = coordinates[2U].x - coordinates[0U].x;
    const auto dv2 = coordinates[2U].y - coordinates[0U].y;
    const auto determinant = fmaf(du1, dv2, -dv1 * du2);
    if (!finite_vector(edge1) || !finite_vector(edge2) || !isfinite(du1) || !isfinite(dv1) ||
        !isfinite(du2) || !isfinite(dv2) || !isfinite(determinant)) {
        return false;
    }
    if (determinant == 0.0F) {
        dpdu = Vector3{};
        dpdv = Vector3{};
        return true;
    }

    const auto reciprocal = 1.0F / determinant;
    dpdu = multiply(subtract(multiply(edge1, dv2), multiply(edge2, dv1)), reciprocal);
    dpdv = multiply(subtract(multiply(edge2, du1), multiply(edge1, du2)), reciprocal);
    return isfinite(reciprocal) && finite_vector(dpdu) && finite_vector(dpdv);
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
    const auto& camera_header = queues.headers[CameraQueue];
    if (!cuda::wavefront_queue_device_detail::immutable_header_contract_is_valid(
            camera_header, CameraQueue, queues.slot_stride) ||
        camera_header.size != 0U || camera_header.overflow_count != 0U ||
        camera_header.rejected_count != 0U) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(WavefrontStageStatus::invalid_contract, WavefrontStageRoute::none,
                                  slot.value, CameraQueue);
        return;
    }
    const auto destination = static_cast<std::uint64_t>(CameraQueue) * queues.slot_stride + index;
    queues.path_slots[destination] = slot;
    finish_phase(control, WavefrontLanePhase::camera);
    outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::none, slot.value);
}

__global__ void publish_seed_camera_header_kernel(const WavefrontQueueDeviceSoa queues,
                                                  const std::uint32_t path_count) {
    auto& header = queues.headers[CameraQueue];
    if (cuda::wavefront_queue_device_detail::immutable_header_contract_is_valid(
            header, CameraQueue, queues.slot_stride) &&
        header.size == 0U && header.overflow_count == 0U && header.rejected_count == 0U &&
        path_count <= header.capacity) {
        header.size = path_count;
    }
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
    if (inputs.reserved != 0U || inputs.reserved_tail[0U] != 0U || inputs.reserved_tail[1U] != 0U ||
        inputs.sample_streams == nullptr || inputs.rays == nullptr || inputs.ray_cones == nullptr ||
        inputs.path_states == nullptr || slot.value >= inputs.count) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::invalid_contract, WavefrontStageRoute::none, slot.value);
        return;
    }
    const auto ray = inputs.rays[slot.value];
    const auto ray_cone = inputs.ray_cones[slot.value];
    const auto state = inputs.path_states[slot.value];
    if (!valid_ray(ray) || !valid_ray_cone(ray_cone) || !valid_path_state(state) ||
        ray.current_medium != state.current_medium) {
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
    streams.ray_cones[slot.value] = ray_cone;
    streams.path_states[slot.value] = state;
    streams.hits[slot.value] = ClosestHit{};
    streams.pending_shadows[slot.value] = WavefrontPendingShadow{};
    streams.previous_bsdf_samples[slot.value] = WavefrontPreviousBsdfSample{};
    control.flags = 0U;
    control.blocked_depth_limits = 0U;
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
        !valid_ray(streams.rays[slot.value]) || !valid_ray_cone(streams.ray_cones[slot.value])) {
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
        route = WavefrontStageRoute::hit;
        phase = WavefrontLanePhase::hit;
    } else if (result.status != static_cast<std::uint32_t>(SceneClosestHitStatus::miss)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(WavefrontStageStatus::traversal_error, WavefrontStageRoute::none,
                                  queue_path.value, result.status);
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
    Vector2 texture_coordinates[3U]{};
    auto uv = Vector2{};
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
        texture_coordinates[corner] = Vector2{
            .x = scene_values<float>(bytes, scene,
                                     scene_column::texture_coordinate_x)[global_vertex],
            .y = scene_values<float>(bytes, scene,
                                     scene_column::texture_coordinate_y)[global_vertex],
        };
        const auto weight =
            corner == 0U ? barycentrics.x : (corner == 1U ? barycentrics.y : barycentrics.z);
        const auto vertex_normal =
            Vector3{.x = scene_values<float>(bytes, scene, scene_column::normal_x)[global_vertex],
                    .y = scene_values<float>(bytes, scene, scene_column::normal_y)[global_vertex],
                    .z = scene_values<float>(bytes, scene, scene_column::normal_z)[global_vertex]};
        if (!finite_vector(local_vertices[corner]) || !finite_vector(vertex_normal) ||
            !isfinite(texture_coordinates[corner].x) || !isfinite(texture_coordinates[corner].y)) {
            return SceneDeviceStatus::numerical_failure;
        }
        local_normal = add(local_normal, multiply(vertex_normal, weight));
        uv.x = fmaf(texture_coordinates[corner].x, weight, uv.x);
        uv.y = fmaf(texture_coordinates[corner].y, weight, uv.y);
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
    Vector3 world_vertices[3U]{};
    for (auto corner = std::uint32_t{0U}; corner < 3U; ++corner) {
        world_vertices[corner] = transform_point(matrix, local_vertices[corner]);
        if (!finite_vector(world_vertices[corner])) {
            return SceneDeviceStatus::numerical_failure;
        }
    }
    auto dpdu = Vector3{};
    auto dpdv = Vector3{};
    if (!surface_derivatives(world_vertices, texture_coordinates, dpdu, dpdv)) {
        return SceneDeviceStatus::numerical_failure;
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
        .uv = uv,
        .dpdu = dpdu,
        .dpdv = dpdv,
        .material_index = material,
    };
    return finite_vector(surface.position) && finite_vector(surface.position_error) &&
                   isfinite(surface.uv.x) && isfinite(surface.uv.y) &&
                   finite_vector(surface.dpdu) && finite_vector(surface.dpdv)
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

[[nodiscard]] __device__ bool make_local_frame_from_tangent(const Vector3 normal,
                                                            Vector3 tangent_seed,
                                                            LocalFrame& frame) noexcept {
    auto unit_normal = Vector3{};
    if (!normalize(normal, unit_normal)) {
        return false;
    }
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

[[nodiscard]] __device__ bool make_local_frame(const Vector3 normal, LocalFrame& frame) noexcept {
    auto unit_normal = Vector3{};
    if (!normalize(normal, unit_normal)) {
        return false;
    }
    const auto sign = copysignf(1.0F, unit_normal.z);
    const auto coefficient = -1.0F / (sign + unit_normal.z);
    const auto product = unit_normal.x * unit_normal.y * coefficient;
    const auto tangent_seed = Vector3{
        .x = 1.0F + sign * unit_normal.x * unit_normal.x * coefficient,
        .y = sign * product,
        .z = -sign * unit_normal.x,
    };
    return make_local_frame_from_tangent(normal, tangent_seed, frame);
}

[[nodiscard]] __device__ bool rotate_local_frame(const LocalFrame& source,
                                                 const float angle_radians,
                                                 LocalFrame& rotated) noexcept {
    if (!orthonormal_frame(source) || !isfinite(angle_radians) || angle_radians < -Pi ||
        !(angle_radians < Pi)) {
        return false;
    }
    if (angle_radians == 0.0F) {
        rotated = source;
        return true;
    }
    const auto cosine = cosf(angle_radians);
    const auto sine = sinf(angle_radians);
    const auto tangent = add(multiply(source.tangent, cosine), multiply(source.bitangent, sine));
    return isfinite(cosine) && isfinite(sine) &&
           make_local_frame_from_tangent(source.normal, tangent, rotated);
}

[[nodiscard]] __device__ bool make_closure_frame(const SurfaceData& surface,
                                                 const SceneMaterialClosureRange& material,
                                                 LocalFrame& frame) noexcept {
    auto unrotated = LocalFrame{};
    const auto created =
        material.frame_mode == 0U
            ? make_local_frame(surface.shading_normal, unrotated)
            : make_local_frame_from_tangent(surface.shading_normal, surface.dpdu, unrotated);
    return created && rotate_local_frame(unrotated, material.tangent_rotation_radians, frame);
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

[[nodiscard]] __device__ constexpr std::uint32_t
lobe_bits(const transport::ScatteringLobe lobes) noexcept {
    return static_cast<std::uint32_t>(lobes);
}

[[nodiscard]] __device__ constexpr bool has_lobe(const transport::ScatteringLobe lobes,
                                                 const transport::ScatteringLobe lobe) noexcept {
    return (lobe_bits(lobes) & lobe_bits(lobe)) == lobe_bits(lobe);
}

[[nodiscard]] __device__ constexpr bool
valid_surface_event(const transport::ScatteringLobe lobes) noexcept {
    constexpr auto family_mask = std::uint32_t{0x00000007U};
    constexpr auto direction_mask = std::uint32_t{0x00000018U};
    constexpr auto known_mask = std::uint32_t{0x0000003FU};
    const auto bits = lobe_bits(lobes);
    const auto family = bits & family_mask;
    const auto direction = bits & direction_mask;
    return bits != 0U && (bits & ~known_mask) == 0U && (bits & 0x00000020U) == 0U &&
           (family == 0x00000001U || family == 0x00000002U || family == 0x00000004U) &&
           (direction == 0x00000008U || direction == 0x00000010U);
}

[[nodiscard]] __device__ bool checked_ray_cone_product(const float left, const float right,
                                                       float& product) noexcept {
    product = left * right;
    return isfinite(product) && !(left != 0.0F && right != 0.0F && product == 0.0F);
}

[[nodiscard]] __device__ bool checked_ray_cone_ratio(const float numerator, const float denominator,
                                                     float& ratio) noexcept {
    if (!isfinite(numerator) || !(numerator > 0.0F) || !isfinite(denominator) ||
        !(denominator > 0.0F)) {
        return false;
    }
    ratio = numerator / denominator;
    return isfinite(ratio) && ratio > 0.0F;
}

[[nodiscard]] __device__ bool advance_ray_cone_to_hit(const WavefrontRayCone cone,
                                                      const TransportRay& ray,
                                                      const float parameter,
                                                      WavefrontRayCone& advanced) noexcept {
    if (!valid_ray_cone(cone) || !isfinite(parameter) || parameter < 0.0F) {
        return false;
    }
    const auto direction = Vector3{
        .x = ray.direction_x,
        .y = ray.direction_y,
        .z = ray.direction_z,
    };
    const auto scale = fmaxf(fabsf(direction.x), fmaxf(fabsf(direction.y), fabsf(direction.z)));
    if (!isfinite(scale) || !(scale > 0.0F)) {
        return false;
    }
    const auto scaled = multiply(direction, 1.0F / scale);
    const auto normalized_length = sqrtf(dot(scaled, scaled));
    auto direction_length = 0.0F;
    if (!isfinite(normalized_length) || !(normalized_length > 0.0F) ||
        !checked_ray_cone_product(scale, normalized_length, direction_length) ||
        !(direction_length > 0.0F)) {
        return false;
    }
    auto distance = 0.0F;
    if (!checked_ray_cone_product(parameter, direction_length, distance) || distance < 0.0F) {
        return false;
    }
    auto growth = 0.0F;
    if (!checked_ray_cone_product(cone.spread, distance, growth)) {
        return false;
    }
    const auto width = fmaf(cone.spread, distance, cone.width);
    if (!isfinite(width) || width < cone.width || (growth != 0.0F && width == cone.width)) {
        return false;
    }
    advanced = WavefrontRayCone{.width = width, .spread = cone.spread};
    return valid_ray_cone(advanced);
}

[[nodiscard]] __device__ bool known_texture_wrap(const std::uint32_t mode) noexcept {
    return mode <= MaximumTextureWrapMode;
}

[[nodiscard]] __device__ SceneDeviceStatus load_image_texture(const std::uint8_t* const bytes,
                                                              const SceneSoaHeader& scene,
                                                              const std::uint32_t texture_id,
                                                              SceneImageTexture& texture) noexcept {
    if (scene.image_texture_count == 0U) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto* const ids =
        scene_values<std::uint32_t>(bytes, scene, scene_column::image_texture_id);
    const auto* const mip_offsets =
        scene_values<std::uint64_t>(bytes, scene, scene_column::image_texture_mip_offset);
    const auto* const mip_counts =
        scene_values<std::uint64_t>(bytes, scene, scene_column::image_texture_mip_count);
    const auto* const channel_counts =
        scene_values<std::uint32_t>(bytes, scene, scene_column::image_texture_channel_count);
    const auto* const color_spaces =
        scene_values<std::uint32_t>(bytes, scene, scene_column::image_texture_storage_color_space);
    auto found = false;
    auto previous_id = std::uint32_t{};
    auto expected_mip_offset = std::uint64_t{};
    for (auto index = std::uint64_t{}; index < scene.image_texture_count; ++index) {
        if ((index != 0U && ids[index] <= previous_id) || mip_counts[index] == 0U ||
            mip_offsets[index] != expected_mip_offset ||
            mip_offsets[index] > scene.image_mip_count ||
            mip_counts[index] > scene.image_mip_count - mip_offsets[index] ||
            channel_counts[index] == 0U) {
            return SceneDeviceStatus::invalid_scene;
        }
        expected_mip_offset += mip_counts[index];
        previous_id = ids[index];
        if (ids[index] == texture_id) {
            if (found || color_spaces[index] != DataTextureColorSpace) {
                return SceneDeviceStatus::invalid_scene;
            }
            found = true;
            texture = SceneImageTexture{
                .mip_offset = mip_offsets[index],
                .mip_count = mip_counts[index],
                .channel_count = channel_counts[index],
            };
        }
    }
    return found && expected_mip_offset == scene.image_mip_count ? SceneDeviceStatus::valid
                                                                 : SceneDeviceStatus::invalid_scene;
}

[[nodiscard]] __device__ SceneDeviceStatus load_image_mip(const std::uint8_t* const bytes,
                                                          const SceneSoaHeader& scene,
                                                          const SceneImageTexture texture,
                                                          const std::uint64_t local_level,
                                                          SceneImageMip& mip) noexcept {
    if (local_level >= texture.mip_count ||
        texture.mip_offset > scene.image_mip_count - local_level) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto mip_index = texture.mip_offset + local_level;
    if (mip_index >= scene.image_mip_count) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto width =
        scene_values<std::uint32_t>(bytes, scene, scene_column::image_mip_width)[mip_index];
    const auto height =
        scene_values<std::uint32_t>(bytes, scene, scene_column::image_mip_height)[mip_index];
    const auto texel_offset =
        scene_values<std::uint64_t>(bytes, scene, scene_column::image_mip_texel_offset)[mip_index];
    const auto texel_count =
        scene_values<std::uint64_t>(bytes, scene, scene_column::image_mip_texel_count)[mip_index];
    if (width == 0U || height == 0U || texture.channel_count == 0U ||
        static_cast<float>(width) != static_cast<double>(width) ||
        static_cast<float>(height) != static_cast<double>(height)) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto pixels = static_cast<std::uint64_t>(width) * height;
    if (pixels / width != height ||
        pixels > MaximumU64 / static_cast<std::uint64_t>(texture.channel_count) ||
        texel_count != pixels * texture.channel_count || texel_offset > scene.image_texel_count ||
        texel_count > scene.image_texel_count - texel_offset) {
        return SceneDeviceStatus::invalid_scene;
    }
    auto expected_texel_offset = std::uint64_t{};
    if (mip_index != 0U) {
        const auto previous_offset = scene_values<std::uint64_t>(
            bytes, scene, scene_column::image_mip_texel_offset)[mip_index - 1U];
        const auto previous_count = scene_values<std::uint64_t>(
            bytes, scene, scene_column::image_mip_texel_count)[mip_index - 1U];
        if (previous_offset > scene.image_texel_count ||
            previous_count > scene.image_texel_count - previous_offset) {
            return SceneDeviceStatus::invalid_scene;
        }
        expected_texel_offset = previous_offset + previous_count;
    }
    if (texel_offset != expected_texel_offset ||
        (mip_index + 1U == scene.image_mip_count &&
         texel_count != scene.image_texel_count - texel_offset)) {
        return SceneDeviceStatus::invalid_scene;
    }
    if (local_level != 0U) {
        const auto previous_index = mip_index - 1U;
        const auto previous_width = scene_values<std::uint32_t>(
            bytes, scene, scene_column::image_mip_width)[previous_index];
        const auto previous_height = scene_values<std::uint32_t>(
            bytes, scene, scene_column::image_mip_height)[previous_index];
        if (previous_width == 0U || previous_height == 0U ||
            width != (previous_width > 1U ? previous_width / 2U : 1U) ||
            height != (previous_height > 1U ? previous_height / 2U : 1U)) {
            return SceneDeviceStatus::invalid_scene;
        }
    }
    mip = SceneImageMip{
        .texel_offset = texel_offset,
        .texel_count = texel_count,
        .width = width,
        .height = height,
    };
    return SceneDeviceStatus::valid;
}

[[nodiscard]] __device__ bool wrap_texture_index(const std::int64_t index,
                                                 const std::uint32_t extent,
                                                 const std::uint32_t mode, bool& present,
                                                 std::uint32_t& wrapped) noexcept {
    if (extent == 0U || !known_texture_wrap(mode)) {
        return false;
    }
    present = true;
    switch (mode) {
    case 0U: {
        const auto modulus = static_cast<std::int64_t>(extent);
        auto remainder = index % modulus;
        if (remainder < 0) {
            remainder += modulus;
        }
        wrapped = static_cast<std::uint32_t>(remainder);
        return true;
    }
    case 1U:
        wrapped = index < 0 ? 0U
                            : (index >= static_cast<std::int64_t>(extent)
                                   ? extent - 1U
                                   : static_cast<std::uint32_t>(index));
        return true;
    case 2U: {
        const auto period = static_cast<std::int64_t>(extent) * 2;
        auto phase = index % period;
        if (phase < 0) {
            phase += period;
        }
        const auto local = phase < static_cast<std::int64_t>(extent) ? phase : period - 1 - phase;
        wrapped = static_cast<std::uint32_t>(local);
        return true;
    }
    case 3U:
        if (index < 0 || index >= static_cast<std::int64_t>(extent)) {
            present = false;
            wrapped = 0U;
        } else {
            wrapped = static_cast<std::uint32_t>(index);
        }
        return true;
    default:
        return false;
    }
}

[[nodiscard]] __device__ bool
cone_texture_differentials(const WavefrontRayCone cone, const TransportRay& ray,
                           const SurfaceData& surface,
                           TextureCoordinateDifferentials& differentials) noexcept {
    auto direction = Vector3{.x = ray.direction_x, .y = ray.direction_y, .z = ray.direction_z};
    if (!valid_ray_cone(cone) || !normalize(direction, direction)) {
        return false;
    }
    const auto cosine = fabsf(dot(direction, surface.geometric_normal));
    if (!isfinite(cosine) || !(cosine > 0.0F)) {
        return false;
    }
    const auto radius = cone.width / cosine;
    const auto metric_uu = dot(surface.dpdu, surface.dpdu);
    const auto metric_uv = dot(surface.dpdu, surface.dpdv);
    const auto metric_vv = dot(surface.dpdv, surface.dpdv);
    const auto determinant = fmaf(metric_uu, metric_vv, -metric_uv * metric_uv);
    if (!isfinite(radius) || radius < 0.0F || !isfinite(metric_uu) || !isfinite(metric_uv) ||
        !isfinite(metric_vv) || !(metric_uu > 0.0F) || !(metric_vv > 0.0F) ||
        !isfinite(determinant) || !(determinant > 0.0F)) {
        return false;
    }
    if (radius == 0.0F) {
        return false;
    }
    const auto radius_squared = radius * radius;
    const auto covariance_uu = radius_squared * metric_vv / determinant;
    const auto covariance_uv = -radius_squared * metric_uv / determinant;
    const auto covariance_vv = radius_squared * metric_uu / determinant;
    if (!isfinite(radius_squared) || !(radius_squared > 0.0F) || !isfinite(covariance_uu) ||
        !isfinite(covariance_uv) || !isfinite(covariance_vv) || !(covariance_uu > 0.0F) ||
        !(covariance_vv > 0.0F)) {
        return false;
    }
    const auto dudx = sqrtf(covariance_uu);
    const auto dvdx = covariance_uv / dudx;
    auto remaining = fmaf(-dvdx, dvdx, covariance_vv);
    const auto covariance_tolerance = 64.0F * FloatEpsilon * covariance_vv;
    if (remaining < 0.0F && remaining >= -covariance_tolerance) {
        remaining = 0.0F;
    }
    const auto dvdy = sqrtf(remaining);
    differentials = TextureCoordinateDifferentials{
        .dudx = dudx,
        .dvdx = dvdx,
        .dudy = 0.0F,
        .dvdy = dvdy,
    };
    return isfinite(dudx) && dudx > 0.0F && isfinite(dvdx) && isfinite(dvdy) && dvdy > 0.0F;
}

[[nodiscard]] __device__ bool make_ewa_footprint(const SceneImageMip mip,
                                                 const TextureCoordinateDifferentials differentials,
                                                 const std::uint32_t maximum_anisotropy,
                                                 EwaFootprint& footprint) noexcept {
    if (maximum_anisotropy == 0U || maximum_anisotropy > MaximumEwaAnisotropy ||
        !isfinite(differentials.dudx) || !isfinite(differentials.dvdx) ||
        !isfinite(differentials.dudy) || !isfinite(differentials.dvdy)) {
        return false;
    }
    const auto width = static_cast<float>(mip.width);
    const auto height = static_cast<float>(mip.height);
    const auto dudx = differentials.dudx * width;
    const auto dvdx = differentials.dvdx * height;
    const auto dudy = differentials.dudy * width;
    const auto dvdy = differentials.dvdy * height;
    if (!isfinite(dudx) || !isfinite(dvdx) || !isfinite(dudy) || !isfinite(dvdy) ||
        fabsf(dudx) >= ExactFloatTexelLimit || fabsf(dvdx) >= ExactFloatTexelLimit ||
        fabsf(dudy) >= ExactFloatTexelLimit || fabsf(dvdy) >= ExactFloatTexelLimit) {
        return false;
    }
    const auto covariance_xx = fmaf(dudx, dudx, dudy * dudy);
    const auto covariance_xy = fmaf(dudx, dvdx, dudy * dvdy);
    const auto covariance_yy = fmaf(dvdx, dvdx, dvdy * dvdy);
    const auto trace = covariance_xx + covariance_yy;
    const auto discriminant = hypotf(covariance_xx - covariance_yy, 2.0F * covariance_xy);
    const auto maximum_eigenvalue = 0.5F * (trace + discriminant);
    const auto jacobian_determinant = fmaf(dudx, dvdy, -dudy * dvdx);
    const auto determinant = jacobian_determinant * jacobian_determinant;
    auto minimum_eigenvalue = maximum_eigenvalue > 0.0F ? determinant / maximum_eigenvalue : 0.0F;
    const auto anisotropy = static_cast<float>(maximum_anisotropy);
    const auto minimum_allowed = maximum_eigenvalue / (anisotropy * anisotropy);
    minimum_eigenvalue = fmaxf(minimum_eigenvalue, minimum_allowed);
    const auto angle = maximum_eigenvalue > 0.0F
                           ? 0.5F * atan2f(2.0F * covariance_xy, covariance_xx - covariance_yy)
                           : 0.0F;
    const auto major_x = cosf(angle);
    const auto major_y = sinf(angle);
    const auto major_length = sqrtf(maximum_eigenvalue);
    const auto minor_length = sqrtf(minimum_eigenvalue);
    footprint = EwaFootprint{
        .major_x = major_x,
        .major_y = major_y,
        .minor_x = -major_y,
        .minor_y = major_x,
        .major_radius = hypotf(1.0F, major_length),
        .minor_radius = hypotf(1.0F, minor_length),
        .minor_length = minor_length,
    };
    return isfinite(covariance_xx) && isfinite(covariance_xy) && isfinite(covariance_yy) &&
           isfinite(maximum_eigenvalue) && maximum_eigenvalue >= 0.0F &&
           isfinite(minimum_eigenvalue) && minimum_eigenvalue >= 0.0F &&
           isfinite(footprint.major_x) && isfinite(footprint.major_y) &&
           isfinite(footprint.major_radius) && isfinite(footprint.minor_radius) &&
           isfinite(footprint.minor_length);
}

[[nodiscard]] __device__ bool make_ewa_ellipse(const SceneImageMip mip, const Vector2 uv,
                                               const EwaFootprint footprint,
                                               EwaEllipse& ellipse) noexcept {
    const auto scaled_x = uv.x * static_cast<float>(mip.width);
    const auto scaled_y = uv.y * static_cast<float>(mip.height);
    if (!isfinite(uv.x) || !isfinite(uv.y) || !isfinite(scaled_x) || !isfinite(scaled_y) ||
        scaled_x <= -ExactFloatTexelLimit || scaled_x >= ExactFloatTexelLimit ||
        scaled_y <= -ExactFloatTexelLimit || scaled_y >= ExactFloatTexelLimit) {
        return false;
    }
    ellipse = EwaEllipse{
        .center_x = scaled_x - 0.5F,
        .center_y = scaled_y - 0.5F,
        .major_x = footprint.major_x,
        .major_y = footprint.major_y,
        .minor_x = footprint.minor_x,
        .minor_y = footprint.minor_y,
        .inverse_major_radius = 1.0F / footprint.major_radius,
        .inverse_minor_radius = 1.0F / footprint.minor_radius,
    };
    const auto radius_x = hypotf(footprint.major_radius * footprint.major_x,
                                 footprint.minor_radius * footprint.minor_x);
    const auto radius_y = hypotf(footprint.major_radius * footprint.major_y,
                                 footprint.minor_radius * footprint.minor_y);
    const auto minimum_x = ceilf(ellipse.center_x - radius_x);
    const auto maximum_x = floorf(ellipse.center_x + radius_x);
    const auto minimum_y = ceilf(ellipse.center_y - radius_y);
    const auto maximum_y = floorf(ellipse.center_y + radius_y);
    if (!isfinite(radius_x) || !isfinite(radius_y) || !isfinite(minimum_x) ||
        !isfinite(maximum_x) || !isfinite(minimum_y) || !isfinite(maximum_y) ||
        minimum_x <= -ExactFloatTexelLimit || maximum_x >= ExactFloatTexelLimit ||
        minimum_y <= -ExactFloatTexelLimit || maximum_y >= ExactFloatTexelLimit) {
        return false;
    }
    ellipse.minimum_x = static_cast<std::int64_t>(minimum_x);
    ellipse.maximum_x = static_cast<std::int64_t>(maximum_x);
    ellipse.minimum_y = static_cast<std::int64_t>(minimum_y);
    ellipse.maximum_y = static_cast<std::int64_t>(maximum_y);
    if (ellipse.maximum_x < ellipse.minimum_x || ellipse.maximum_y < ellipse.minimum_y) {
        return false;
    }
    const auto width = static_cast<std::uint64_t>(ellipse.maximum_x - ellipse.minimum_x) + 1U;
    const auto height = static_cast<std::uint64_t>(ellipse.maximum_y - ellipse.minimum_y) + 1U;
    if (height != 0U && width > MaximumU64 / height) {
        return false;
    }
    ellipse.visit_count = width * height;
    return true;
}

[[nodiscard]] __device__ SceneDeviceStatus sample_ewa_ellipse(
    const std::uint8_t* const bytes, const SceneSoaHeader& scene, const SceneImageTexture texture,
    const SceneImageMip mip, const EwaEllipse ellipse, const std::uint32_t channel,
    const std::uint32_t u_wrap, const std::uint32_t v_wrap, float& filtered) noexcept {
    if (channel >= texture.channel_count || !known_texture_wrap(u_wrap) ||
        !known_texture_wrap(v_wrap)) {
        return SceneDeviceStatus::invalid_scene;
    }
    constexpr auto gaussian_alpha = 2.0F;
    const auto edge_weight = expf(-gaussian_alpha);
    auto accumulated_weight = 0.0F;
    filtered = 0.0F;
    const auto* const texels = scene_values<float>(bytes, scene, scene_column::image_texel_value);
    for (auto y = ellipse.minimum_y; y <= ellipse.maximum_y; ++y) {
        auto y_present = false;
        auto wrapped_y = std::uint32_t{};
        if (!wrap_texture_index(y, mip.height, v_wrap, y_present, wrapped_y)) {
            return SceneDeviceStatus::invalid_scene;
        }
        const auto offset_y = static_cast<float>(y) - ellipse.center_y;
        for (auto x = ellipse.minimum_x; x <= ellipse.maximum_x; ++x) {
            const auto offset_x = static_cast<float>(x) - ellipse.center_x;
            const auto major_distance =
                fmaf(offset_x, ellipse.major_x, offset_y * ellipse.major_y) *
                ellipse.inverse_major_radius;
            const auto minor_distance =
                fmaf(offset_x, ellipse.minor_x, offset_y * ellipse.minor_y) *
                ellipse.inverse_minor_radius;
            const auto radius_squared =
                fmaf(major_distance, major_distance, minor_distance * minor_distance);
            if (!isfinite(radius_squared) || radius_squared < 0.0F) {
                return SceneDeviceStatus::numerical_failure;
            }
            if (radius_squared >= 1.0F) {
                continue;
            }
            const auto weight = edge_weight * expm1f(gaussian_alpha * (1.0F - radius_squared));
            if (!isfinite(weight) || !(weight > 0.0F)) {
                return SceneDeviceStatus::numerical_failure;
            }
            auto x_present = false;
            auto wrapped_x = std::uint32_t{};
            if (!wrap_texture_index(x, mip.width, u_wrap, x_present, wrapped_x)) {
                return SceneDeviceStatus::invalid_scene;
            }
            auto value = 0.0F;
            if (x_present && y_present) {
                const auto pixel = static_cast<std::uint64_t>(wrapped_y) * mip.width + wrapped_x;
                if (pixel > MaximumU64 / texture.channel_count) {
                    return SceneDeviceStatus::invalid_scene;
                }
                const auto local_element = pixel * texture.channel_count + channel;
                if (local_element >= mip.texel_count ||
                    mip.texel_offset > scene.image_texel_count - local_element - 1U) {
                    return SceneDeviceStatus::invalid_scene;
                }
                value = texels[mip.texel_offset + local_element];
                if (!isfinite(value)) {
                    return SceneDeviceStatus::numerical_failure;
                }
            }
            const auto next_weight = accumulated_weight + weight;
            if (!isfinite(next_weight) || !(next_weight > 0.0F)) {
                return SceneDeviceStatus::numerical_failure;
            }
            if (accumulated_weight == 0.0F) {
                filtered = value;
            } else {
                filtered += (value - filtered) * (weight / next_weight);
            }
            accumulated_weight = next_weight;
        }
    }
    return accumulated_weight > 0.0F && isfinite(filtered) ? SceneDeviceStatus::valid
                                                           : SceneDeviceStatus::numerical_failure;
}

[[nodiscard]] __device__ SceneDeviceStatus filter_image_ewa(
    const std::uint8_t* const bytes, const SceneSoaHeader& scene, const SceneImageTexture texture,
    const Vector2 uv, const TextureCoordinateDifferentials differentials,
    const std::uint32_t channel, const std::uint32_t u_wrap, const std::uint32_t v_wrap,
    const std::uint32_t maximum_anisotropy, const std::uint32_t maximum_texel_visits,
    float& filtered) noexcept {
    if (texture.mip_count == 0U || channel >= texture.channel_count ||
        !known_texture_wrap(u_wrap) || !known_texture_wrap(v_wrap) || maximum_anisotropy == 0U ||
        maximum_anisotropy > MaximumEwaAnisotropy || maximum_texel_visits == 0U ||
        !isfinite(uv.x) || !isfinite(uv.y)) {
        return SceneDeviceStatus::invalid_scene;
    }
    auto lower_mip = SceneImageMip{};
    auto status = load_image_mip(bytes, scene, texture, 0U, lower_mip);
    if (status != SceneDeviceStatus::valid) {
        return status;
    }
    auto lower_footprint = EwaFootprint{};
    if (!make_ewa_footprint(lower_mip, differentials, maximum_anisotropy, lower_footprint)) {
        return SceneDeviceStatus::numerical_failure;
    }
    auto upper_mip = SceneImageMip{};
    auto upper_footprint = EwaFootprint{};
    auto has_upper = false;
    auto fraction = 0.0F;
    if (lower_footprint.minor_length > 1.0F) {
        for (auto level = std::uint64_t{1U}; level < texture.mip_count; ++level) {
            auto candidate_mip = SceneImageMip{};
            status = load_image_mip(bytes, scene, texture, level, candidate_mip);
            if (status != SceneDeviceStatus::valid) {
                return status;
            }
            auto candidate_footprint = EwaFootprint{};
            if (!make_ewa_footprint(candidate_mip, differentials, maximum_anisotropy,
                                    candidate_footprint)) {
                return SceneDeviceStatus::numerical_failure;
            }
            if (candidate_footprint.minor_length <= 1.0F) {
                const auto lower_log = logf(lower_footprint.minor_length);
                const auto upper_log = logf(candidate_footprint.minor_length);
                const auto denominator = lower_log - upper_log;
                const auto blend = lower_log / denominator;
                if (!isfinite(blend) || !(denominator > 0.0F) || blend < 0.0F || blend > 1.0F) {
                    return SceneDeviceStatus::numerical_failure;
                }
                if (blend == 1.0F) {
                    lower_mip = candidate_mip;
                    lower_footprint = candidate_footprint;
                } else {
                    upper_mip = candidate_mip;
                    upper_footprint = candidate_footprint;
                    has_upper = true;
                    fraction = blend;
                }
                break;
            }
            lower_mip = candidate_mip;
            lower_footprint = candidate_footprint;
        }
    }
    if (!has_upper && lower_footprint.minor_length > 1.0F) {
        lower_footprint = EwaFootprint{};
    }

    auto lower_ellipse = EwaEllipse{};
    if (!make_ewa_ellipse(lower_mip, uv, lower_footprint, lower_ellipse)) {
        return SceneDeviceStatus::numerical_failure;
    }
    auto upper_ellipse = EwaEllipse{};
    auto total_visits = lower_ellipse.visit_count;
    if (has_upper) {
        if (!make_ewa_ellipse(upper_mip, uv, upper_footprint, upper_ellipse) ||
            total_visits > MaximumU64 - upper_ellipse.visit_count) {
            return SceneDeviceStatus::numerical_failure;
        }
        total_visits += upper_ellipse.visit_count;
    }
    if (total_visits > maximum_texel_visits) {
        return SceneDeviceStatus::unsupported_transport;
    }

    auto lower = 0.0F;
    status = sample_ewa_ellipse(bytes, scene, texture, lower_mip, lower_ellipse, channel, u_wrap,
                                v_wrap, lower);
    if (status != SceneDeviceStatus::valid) {
        return status;
    }
    if (!has_upper) {
        filtered = lower;
        return SceneDeviceStatus::valid;
    }
    auto upper = 0.0F;
    status = sample_ewa_ellipse(bytes, scene, texture, upper_mip, upper_ellipse, channel, u_wrap,
                                v_wrap, upper);
    if (status != SceneDeviceStatus::valid) {
        return status;
    }
    filtered = lower + fraction * (upper - lower);
    return isfinite(filtered) ? SceneDeviceStatus::valid : SceneDeviceStatus::numerical_failure;
}

[[nodiscard]] __device__ bool projected_tangent_frame(const SurfaceData& surface, Vector3& tangent,
                                                      Vector3& bitangent) noexcept {
    const auto tangent_alignment = dot(surface.shading_normal, surface.dpdu);
    const auto projected = Vector3{
        .x = fmaf(-tangent_alignment, surface.shading_normal.x, surface.dpdu.x),
        .y = fmaf(-tangent_alignment, surface.shading_normal.y, surface.dpdu.y),
        .z = fmaf(-tangent_alignment, surface.shading_normal.z, surface.dpdu.z),
    };
    auto base_bitangent = Vector3{};
    if (!normalize(projected, tangent) ||
        !normalize(cross(surface.shading_normal, tangent), base_bitangent)) {
        return false;
    }
    const auto dpdv_scale =
        fmaxf(fabsf(surface.dpdv.x), fmaxf(fabsf(surface.dpdv.y), fabsf(surface.dpdv.z)));
    if (!isfinite(dpdv_scale) || dpdv_scale == 0.0F) {
        return false;
    }
    const auto handedness = dot(base_bitangent, multiply(surface.dpdv, 1.0F / dpdv_scale));
    if (!isfinite(handedness) || handedness == 0.0F) {
        return false;
    }
    bitangent = handedness > 0.0F ? base_bitangent : multiply(base_bitangent, -1.0F);
    return true;
}

[[nodiscard]] __device__ bool normalized_cross(const Vector3 left, const Vector3 right,
                                               Vector3& result) noexcept {
    const auto left_scale = fmaxf(fabsf(left.x), fmaxf(fabsf(left.y), fabsf(left.z)));
    const auto right_scale = fmaxf(fabsf(right.x), fmaxf(fabsf(right.y), fabsf(right.z)));
    return isfinite(left_scale) && isfinite(right_scale) && left_scale > 0.0F &&
           right_scale > 0.0F &&
           normalize(cross(multiply(left, 1.0F / left_scale), multiply(right, 1.0F / right_scale)),
                     result);
}

[[nodiscard]] __device__ SceneDeviceStatus apply_normal_map(
    const std::uint8_t* const bytes, const SceneSoaHeader& scene,
    const TextureCoordinateDifferentials differentials, SurfaceData& surface) noexcept {
    const auto material = surface.material_index;
    const auto present = scene_values<std::uint8_t>(
        bytes, scene, scene_column::material_normal_map_present)[material];
    if (present == 0U) {
        const auto canonical =
            scene_values<std::uint32_t>(
                bytes, scene, scene_column::material_normal_map_texture_id)[material] == 0U &&
            scene_values<std::uint32_t>(
                bytes, scene, scene_column::material_normal_map_red_channel)[material] == 0U &&
            scene_values<std::uint32_t>(
                bytes, scene, scene_column::material_normal_map_green_channel)[material] == 0U &&
            scene_values<std::uint32_t>(
                bytes, scene, scene_column::material_normal_map_blue_channel)[material] == 0U &&
            scene_values<std::uint32_t>(bytes, scene,
                                        scene_column::material_normal_map_u_wrap)[material] == 0U &&
            scene_values<std::uint32_t>(bytes, scene,
                                        scene_column::material_normal_map_v_wrap)[material] == 0U &&
            scene_values<std::uint32_t>(
                bytes, scene, scene_column::material_normal_map_maximum_anisotropy)[material] ==
                0U &&
            scene_values<std::uint32_t>(
                bytes, scene, scene_column::material_normal_map_maximum_texel_visits)[material] ==
                0U &&
            scene_values<std::uint32_t>(
                bytes, scene, scene_column::material_normal_map_y_convention)[material] == 0U;
        return canonical ? SceneDeviceStatus::valid : SceneDeviceStatus::invalid_scene;
    }
    if (present != 1U) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto texture_id = scene_values<std::uint32_t>(
        bytes, scene, scene_column::material_normal_map_texture_id)[material];
    const auto red_channel = scene_values<std::uint32_t>(
        bytes, scene, scene_column::material_normal_map_red_channel)[material];
    const auto green_channel = scene_values<std::uint32_t>(
        bytes, scene, scene_column::material_normal_map_green_channel)[material];
    const auto blue_channel = scene_values<std::uint32_t>(
        bytes, scene, scene_column::material_normal_map_blue_channel)[material];
    const auto u_wrap = scene_values<std::uint32_t>(
        bytes, scene, scene_column::material_normal_map_u_wrap)[material];
    const auto v_wrap = scene_values<std::uint32_t>(
        bytes, scene, scene_column::material_normal_map_v_wrap)[material];
    const auto maximum_anisotropy = scene_values<std::uint32_t>(
        bytes, scene, scene_column::material_normal_map_maximum_anisotropy)[material];
    const auto maximum_texel_visits = scene_values<std::uint32_t>(
        bytes, scene, scene_column::material_normal_map_maximum_texel_visits)[material];
    const auto y_convention = scene_values<std::uint32_t>(
        bytes, scene, scene_column::material_normal_map_y_convention)[material];
    auto texture = SceneImageTexture{};
    auto status = load_image_texture(bytes, scene, texture_id, texture);
    if (status != SceneDeviceStatus::valid) {
        return status;
    }
    if (red_channel >= texture.channel_count || green_channel >= texture.channel_count ||
        blue_channel >= texture.channel_count || red_channel == green_channel ||
        red_channel == blue_channel || green_channel == blue_channel ||
        y_convention > MaximumNormalYConvention) {
        return SceneDeviceStatus::invalid_scene;
    }
    float encoded[3U]{};
    const std::uint32_t channels[]{red_channel, green_channel, blue_channel};
    for (auto component = std::uint32_t{}; component < 3U; ++component) {
        status = filter_image_ewa(bytes, scene, texture, surface.uv, differentials,
                                  channels[component], u_wrap, v_wrap, maximum_anisotropy,
                                  maximum_texel_visits, encoded[component]);
        if (status != SceneDeviceStatus::valid) {
            return status;
        }
        if (encoded[component] < 0.0F || encoded[component] > 1.0F) {
            return SceneDeviceStatus::numerical_failure;
        }
    }
    auto mapped = Vector3{
        .x = fmaf(2.0F, encoded[0U], -1.0F),
        .y = fmaf(2.0F, encoded[1U], -1.0F),
        .z = fmaf(2.0F, encoded[2U], -1.0F),
    };
    if (!normalize(mapped, mapped) || !(mapped.z > 0.0F)) {
        return SceneDeviceStatus::numerical_failure;
    }
    auto tangent = Vector3{};
    auto bitangent = Vector3{};
    if (!projected_tangent_frame(surface, tangent, bitangent)) {
        return SceneDeviceStatus::numerical_failure;
    }
    const auto mapped_y = y_convention == 0U ? mapped.y : -mapped.y;
    auto shading_normal = Vector3{
        .x = fmaf(tangent.x, mapped.x,
                  fmaf(bitangent.x, mapped_y, surface.shading_normal.x * mapped.z)),
        .y = fmaf(tangent.y, mapped.x,
                  fmaf(bitangent.y, mapped_y, surface.shading_normal.y * mapped.z)),
        .z = fmaf(tangent.z, mapped.x,
                  fmaf(bitangent.z, mapped_y, surface.shading_normal.z * mapped.z)),
    };
    if (!normalize(shading_normal, shading_normal) ||
        !(dot(surface.geometric_normal, shading_normal) > 0.0F)) {
        return SceneDeviceStatus::numerical_failure;
    }
    surface.shading_normal = shading_normal;
    return SceneDeviceStatus::valid;
}

[[nodiscard]] __device__ SceneDeviceStatus
apply_bump_map(const std::uint8_t* const bytes, const SceneSoaHeader& scene,
               const TextureCoordinateDifferentials differentials, SurfaceData& surface) noexcept {
    const auto material = surface.material_index;
    const auto present =
        scene_values<std::uint8_t>(bytes, scene, scene_column::material_bump_map_present)[material];
    if (present == 0U) {
        const auto canonical =
            scene_values<std::uint32_t>(
                bytes, scene, scene_column::material_bump_map_texture_id)[material] == 0U &&
            scene_values<std::uint32_t>(bytes, scene,
                                        scene_column::material_bump_map_channel)[material] == 0U &&
            __float_as_uint(scene_values<float>(
                bytes, scene, scene_column::material_bump_map_scale)[material]) == 0U &&
            scene_values<std::uint32_t>(bytes, scene,
                                        scene_column::material_bump_map_u_wrap)[material] == 0U &&
            scene_values<std::uint32_t>(bytes, scene,
                                        scene_column::material_bump_map_v_wrap)[material] == 0U &&
            scene_values<std::uint32_t>(
                bytes, scene, scene_column::material_bump_map_maximum_anisotropy)[material] == 0U &&
            scene_values<std::uint32_t>(
                bytes, scene, scene_column::material_bump_map_maximum_texel_visits)[material] == 0U;
        return canonical ? SceneDeviceStatus::valid : SceneDeviceStatus::invalid_scene;
    }
    if (present != 1U) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto texture_id = scene_values<std::uint32_t>(
        bytes, scene, scene_column::material_bump_map_texture_id)[material];
    const auto channel = scene_values<std::uint32_t>(
        bytes, scene, scene_column::material_bump_map_channel)[material];
    const auto scale =
        scene_values<float>(bytes, scene, scene_column::material_bump_map_scale)[material];
    const auto u_wrap =
        scene_values<std::uint32_t>(bytes, scene, scene_column::material_bump_map_u_wrap)[material];
    const auto v_wrap =
        scene_values<std::uint32_t>(bytes, scene, scene_column::material_bump_map_v_wrap)[material];
    const auto maximum_anisotropy = scene_values<std::uint32_t>(
        bytes, scene, scene_column::material_bump_map_maximum_anisotropy)[material];
    const auto maximum_texel_visits = scene_values<std::uint32_t>(
        bytes, scene, scene_column::material_bump_map_maximum_texel_visits)[material];
    auto texture = SceneImageTexture{};
    auto status = load_image_texture(bytes, scene, texture_id, texture);
    if (status != SceneDeviceStatus::valid) {
        return status;
    }
    auto base = SceneImageMip{};
    status = load_image_mip(bytes, scene, texture, 0U, base);
    if (status != SceneDeviceStatus::valid) {
        return status;
    }
    if (channel >= texture.channel_count || !isfinite(scale)) {
        return SceneDeviceStatus::invalid_scene;
    }
    const auto dpdu_alignment = dot(surface.shading_normal, surface.dpdu);
    const auto dpdv_alignment = dot(surface.shading_normal, surface.dpdv);
    const auto projected_dpdu = Vector3{
        .x = fmaf(-dpdu_alignment, surface.shading_normal.x, surface.dpdu.x),
        .y = fmaf(-dpdu_alignment, surface.shading_normal.y, surface.dpdu.y),
        .z = fmaf(-dpdu_alignment, surface.shading_normal.z, surface.dpdu.z),
    };
    const auto projected_dpdv = Vector3{
        .x = fmaf(-dpdv_alignment, surface.shading_normal.x, surface.dpdv.x),
        .y = fmaf(-dpdv_alignment, surface.shading_normal.y, surface.dpdv.y),
        .z = fmaf(-dpdv_alignment, surface.shading_normal.z, surface.dpdv.z),
    };
    auto projected_orientation = Vector3{};
    if (!isfinite(dpdu_alignment) || !isfinite(dpdv_alignment) || !finite_vector(projected_dpdu) ||
        !finite_vector(projected_dpdv) ||
        !normalized_cross(projected_dpdu, projected_dpdv, projected_orientation)) {
        return SceneDeviceStatus::numerical_failure;
    }
    const auto projected_handedness = dot(surface.shading_normal, projected_orientation);
    if (!isfinite(projected_handedness) || projected_handedness == 0.0F) {
        return SceneDeviceStatus::numerical_failure;
    }
    const auto du = fmaxf(1.0F / static_cast<float>(base.width),
                          hypotf(differentials.dudx, differentials.dudy));
    const auto dv = fmaxf(1.0F / static_cast<float>(base.height),
                          hypotf(differentials.dvdx, differentials.dvdy));
    if (!isfinite(du) || !(du > 0.0F) || !isfinite(dv) || !(dv > 0.0F)) {
        return SceneDeviceStatus::numerical_failure;
    }
    const Vector2 coordinates[]{
        Vector2{.x = surface.uv.x + du, .y = surface.uv.y},
        Vector2{.x = surface.uv.x - du, .y = surface.uv.y},
        Vector2{.x = surface.uv.x, .y = surface.uv.y + dv},
        Vector2{.x = surface.uv.x, .y = surface.uv.y - dv},
    };
    float heights[4U]{};
    for (auto sample = std::uint32_t{}; sample < 4U; ++sample) {
        status = filter_image_ewa(bytes, scene, texture, coordinates[sample], differentials,
                                  channel, u_wrap, v_wrap, maximum_anisotropy, maximum_texel_visits,
                                  heights[sample]);
        if (status != SceneDeviceStatus::valid) {
            return status;
        }
    }
    const auto dhdu = (heights[0U] - heights[1U]) / (2.0F * du);
    const auto dhdv = (heights[2U] - heights[3U]) / (2.0F * dv);
    const auto scaled_dhdu = scale * dhdu;
    const auto scaled_dhdv = scale * dhdv;
    const auto tangent_u = Vector3{
        .x = fmaf(surface.shading_normal.x, scaled_dhdu, projected_dpdu.x),
        .y = fmaf(surface.shading_normal.y, scaled_dhdu, projected_dpdu.y),
        .z = fmaf(surface.shading_normal.z, scaled_dhdu, projected_dpdu.z),
    };
    const auto tangent_v = Vector3{
        .x = fmaf(surface.shading_normal.x, scaled_dhdv, projected_dpdv.x),
        .y = fmaf(surface.shading_normal.y, scaled_dhdv, projected_dpdv.y),
        .z = fmaf(surface.shading_normal.z, scaled_dhdv, projected_dpdv.z),
    };
    auto shading_normal = Vector3{};
    if (!isfinite(dhdu) || !isfinite(dhdv) || !isfinite(scaled_dhdu) || !isfinite(scaled_dhdv) ||
        !finite_vector(tangent_u) || !finite_vector(tangent_v) ||
        !normalized_cross(tangent_u, tangent_v, shading_normal)) {
        return SceneDeviceStatus::numerical_failure;
    }
    if (dot(shading_normal, surface.shading_normal) < 0.0F) {
        shading_normal = multiply(shading_normal, -1.0F);
    }
    if (!(dot(shading_normal, surface.shading_normal) > 0.0F) ||
        !(dot(surface.geometric_normal, shading_normal) > 0.0F)) {
        return SceneDeviceStatus::numerical_failure;
    }
    surface.shading_normal = shading_normal;
    return SceneDeviceStatus::valid;
}

[[nodiscard]] __device__ SceneDeviceStatus apply_surface_maps(const std::uint8_t* const bytes,
                                                              const SceneSoaHeader& scene,
                                                              const WavefrontRayCone cone,
                                                              const TransportRay& ray,
                                                              SurfaceData& surface) noexcept {
    const auto normal_present = scene_values<std::uint8_t>(
        bytes, scene, scene_column::material_normal_map_present)[surface.material_index];
    const auto bump_present = scene_values<std::uint8_t>(
        bytes, scene, scene_column::material_bump_map_present)[surface.material_index];
    if (normal_present == 0U && bump_present == 0U) {
        const auto empty_differentials = TextureCoordinateDifferentials{};
        const auto normal_status = apply_normal_map(bytes, scene, empty_differentials, surface);
        return normal_status == SceneDeviceStatus::valid
                   ? apply_bump_map(bytes, scene, empty_differentials, surface)
                   : normal_status;
    }
    if (normal_present > 1U || bump_present > 1U) {
        return SceneDeviceStatus::invalid_scene;
    }
    auto differentials = TextureCoordinateDifferentials{};
    if (!cone_texture_differentials(cone, ray, surface, differentials)) {
        return SceneDeviceStatus::numerical_failure;
    }
    auto status = apply_normal_map(bytes, scene, differentials, surface);
    if (status != SceneDeviceStatus::valid) {
        return status;
    }
    return apply_bump_map(bytes, scene, differentials, surface);
}

[[nodiscard]] __device__ bool propagate_ray_cone_scattering(const WavefrontRayCone cone,
                                                            const transport::ClosureRecord& closure,
                                                            const transport::ScatteringLobe event,
                                                            const transport::Vector3 outgoing_local,
                                                            const transport::Vector3 incoming_local,
                                                            WavefrontRayCone& propagated) noexcept {
    if (!valid_ray_cone(cone) || !valid_surface_event(event) ||
        !transport::unit_vector(outgoing_local) || !transport::unit_vector(incoming_local) ||
        outgoing_local.z == 0.0F || incoming_local.z == 0.0F) {
        return false;
    }

    const auto reflection = has_lobe(event, transport::ScatteringLobe::reflection);
    const auto transmission = has_lobe(event, transport::ScatteringLobe::transmission);
    const auto same_hemisphere = (outgoing_local.z > 0.0F) == (incoming_local.z > 0.0F);
    if ((reflection && !same_hemisphere) || (transmission && same_hemisphere)) {
        return false;
    }
    auto spread_floor = 0.0F;
    auto transmission_scale = 1.0F;
    auto width_scale = 1.0F;
    switch (closure.kind) {
    case transport::ClosureKind::lambertian_reflection:
    case transport::ClosureKind::rough_diffuse_reflection:
        if (!reflection || !has_lobe(event, transport::ScatteringLobe::diffuse)) {
            return false;
        }
        spread_floor = 1.0F;
        break;
    case transport::ClosureKind::rough_conductor_reflection:
        if (!reflection || !has_lobe(event, transport::ScatteringLobe::glossy)) {
            return false;
        }
        if (!isfinite(closure.parameters[8U]) || closure.parameters[8U] < 0.0F ||
            !isfinite(closure.parameters[9U]) || closure.parameters[9U] < 0.0F) {
            return false;
        }
        spread_floor = fmaxf(closure.parameters[8U], closure.parameters[9U]);
        break;
    case transport::ClosureKind::rough_dielectric:
        if (!has_lobe(event, transport::ScatteringLobe::glossy)) {
            return false;
        }
        if (!isfinite(closure.parameters[2U]) || closure.parameters[2U] < 0.0F ||
            !isfinite(closure.parameters[3U]) || closure.parameters[3U] < 0.0F) {
            return false;
        }
        spread_floor = fmaxf(closure.parameters[2U], closure.parameters[3U]);
        break;
    case transport::ClosureKind::specular_reflection:
        if (!reflection || !has_lobe(event, transport::ScatteringLobe::specular)) {
            return false;
        }
        break;
    case transport::ClosureKind::specular_transmission:
        if (!transmission || !has_lobe(event, transport::ScatteringLobe::specular)) {
            return false;
        }
        break;
    case transport::ClosureKind::none:
        return false;
    default:
        return false;
    }

    if (!isfinite(spread_floor) || spread_floor < 0.0F) {
        return false;
    }
    if (transmission) {
        if (closure.kind != transport::ClosureKind::rough_dielectric &&
            closure.kind != transport::ClosureKind::specular_transmission) {
            return false;
        }
        const auto exterior_eta = closure.parameters[0U];
        const auto interior_eta = closure.parameters[1U];
        const auto incident_eta = outgoing_local.z > 0.0F ? exterior_eta : interior_eta;
        const auto transmitted_eta = outgoing_local.z > 0.0F ? interior_eta : exterior_eta;
        auto eta_ratio = 0.0F;
        auto cosine_ratio = 0.0F;
        auto inverse_cosine_ratio = 0.0F;
        if (!checked_ray_cone_ratio(incident_eta, transmitted_eta, eta_ratio) ||
            !checked_ray_cone_ratio(fabsf(outgoing_local.z), fabsf(incoming_local.z),
                                    cosine_ratio) ||
            !checked_ray_cone_ratio(fabsf(incoming_local.z), fabsf(outgoing_local.z),
                                    inverse_cosine_ratio)) {
            return false;
        }
        auto meridional_scale = 0.0F;
        if (!checked_ray_cone_product(eta_ratio, cosine_ratio, meridional_scale)) {
            return false;
        }
        transmission_scale = fmaxf(eta_ratio, meridional_scale);
        width_scale = fmaxf(1.0F, inverse_cosine_ratio);
    }

    auto transported_width = 0.0F;
    auto transported_spread = 0.0F;
    if (!checked_ray_cone_product(cone.width, width_scale, transported_width) ||
        !checked_ray_cone_product(cone.spread, transmission_scale, transported_spread)) {
        return false;
    }
    propagated = WavefrontRayCone{
        .width = transported_width,
        .spread = fmaxf(transported_spread, spread_floor),
    };
    return valid_ray_cone(propagated);
}

[[nodiscard]] __device__ constexpr bool
delta_surface_event(const transport::ScatteringLobe lobes) noexcept {
    return valid_surface_event(lobes) && has_lobe(lobes, transport::ScatteringLobe::specular);
}

[[nodiscard]] __device__ bool shading_normal_correction_radiance(const Vector3 geometric_normal,
                                                                 const Vector3 shading_normal,
                                                                 const Vector3 outgoing_world,
                                                                 const Vector3 incoming_world,
                                                                 float& correction) noexcept {
    const auto outgoing_geometric = dot(geometric_normal, outgoing_world);
    const auto outgoing_shading = dot(shading_normal, outgoing_world);
    const auto incoming_geometric = dot(geometric_normal, incoming_world);
    const auto incoming_shading = dot(shading_normal, incoming_world);
    if (!transport::unit_vector(transport::Vector3{
            .x = geometric_normal.x, .y = geometric_normal.y, .z = geometric_normal.z}) ||
        !transport::unit_vector(transport::Vector3{
            .x = shading_normal.x, .y = shading_normal.y, .z = shading_normal.z}) ||
        !transport::unit_vector(transport::Vector3{
            .x = outgoing_world.x, .y = outgoing_world.y, .z = outgoing_world.z}) ||
        !transport::unit_vector(transport::Vector3{
            .x = incoming_world.x, .y = incoming_world.y, .z = incoming_world.z}) ||
        !(dot(geometric_normal, shading_normal) > 0.0F) || !isfinite(outgoing_geometric) ||
        !isfinite(outgoing_shading) || !isfinite(incoming_geometric) ||
        !isfinite(incoming_shading)) {
        return false;
    }
    if (outgoing_geometric == 0.0F || outgoing_shading == 0.0F || incoming_geometric == 0.0F ||
        incoming_shading == 0.0F || signbit(outgoing_geometric) != signbit(outgoing_shading) ||
        signbit(incoming_geometric) != signbit(incoming_shading)) {
        correction = 0.0F;
        return true;
    }
    correction = 1.0F;
    return true;
}

[[nodiscard]] __device__ bool geometric_event_support(const Vector3 geometric_normal,
                                                      const Vector3 outgoing_world,
                                                      const Vector3 incoming_world,
                                                      const transport::ScatteringLobe lobes,
                                                      bool& supported) noexcept {
    if (!valid_surface_event(lobes)) {
        return false;
    }
    const auto outgoing = dot(geometric_normal, outgoing_world);
    const auto incoming = dot(geometric_normal, incoming_world);
    if (!isfinite(outgoing) || !isfinite(incoming)) {
        return false;
    }
    if (outgoing == 0.0F || incoming == 0.0F) {
        supported = false;
        return true;
    }
    const auto same_side = signbit(outgoing) == signbit(incoming);
    supported = has_lobe(lobes, transport::ScatteringLobe::reflection) ? same_side : !same_side;
    return true;
}

[[nodiscard]] __device__ bool valid_depth_state(const WavefrontTransportConfig& config,
                                                const TransportPathStateLane& state) noexcept {
    return state.diffuse_depth <= config.diffuse_depth_limit &&
           state.glossy_depth <= config.glossy_depth_limit &&
           state.specular_depth <= config.specular_depth_limit &&
           state.transmission_depth <= config.transmission_depth_limit &&
           state.volume_depth <= config.volume_depth_limit;
}

[[nodiscard]] __device__ bool advance_depth_event(const transport::ScatteringLobe lobes,
                                                  TransportPathStateLane& state) noexcept {
    if (!valid_surface_event(lobes) || state.depth == 0xFFFFFFFFU) {
        return false;
    }
    if (has_lobe(lobes, transport::ScatteringLobe::diffuse)) {
        if (state.diffuse_depth == 0xFFFFFFFFU) {
            return false;
        }
        ++state.diffuse_depth;
    } else if (has_lobe(lobes, transport::ScatteringLobe::glossy)) {
        if (state.glossy_depth == 0xFFFFFFFFU) {
            return false;
        }
        ++state.glossy_depth;
    } else {
        if (state.specular_depth == 0xFFFFFFFFU) {
            return false;
        }
        ++state.specular_depth;
    }
    if (has_lobe(lobes, transport::ScatteringLobe::transmission)) {
        if (state.transmission_depth == 0xFFFFFFFFU) {
            return false;
        }
        ++state.transmission_depth;
    }
    ++state.depth;
    return true;
}

[[nodiscard]] __device__ bool
mixture_has_nonblack_weight(const transport::ClosureMixtureRecord& mixture) noexcept {
    for (auto closure = std::uint32_t{0U}; closure < mixture.active_count; ++closure) {
        for (auto lane = std::uint32_t{0U}; lane < shared::HostDeviceSpectrumLaneCount; ++lane) {
            if (mixture.closures[closure].weight[lane] != 0.0F) {
                return true;
            }
        }
    }
    return false;
}

[[nodiscard]] __device__ bool update_closure_throughput(
    const TransportSpectrum& beta, const transport::ClosureMixtureSampleResult& sample,
    const float shading_normal_correction, const transport::ClosureMixtureRecord& mixture,
    TransportSpectrum& updated) noexcept {
    if (!valid_surface_event(sample.lobes) || !nonnegative_spectrum(sample.value) ||
        !isfinite(sample.probability.value) || !(sample.probability.value > 0.0F) ||
        !isfinite(shading_normal_correction) || !(shading_normal_correction > 0.0F)) {
        return false;
    }
    const auto delta = delta_surface_event(sample.lobes);
    const auto expected_measure = delta ? transport::ProbabilityMeasure::discrete
                                        : transport::ProbabilityMeasure::solid_angle;
    if (sample.probability.measure != expected_measure ||
        (delta && sample.probability.value > 1.0F)) {
        return false;
    }
    const auto cosine = fabsf(sample.incoming_local.z);
    if (!isfinite(cosine) || !(cosine > 0.0F)) {
        return false;
    }

    if (mixture.active_count == 1U &&
        mixture.closures[0U].kind == transport::ClosureKind::lambertian_reflection &&
        shading_normal_correction == 1.0F) {
        for (auto lane = std::uint32_t{0U}; lane < shared::HostDeviceSpectrumLaneCount; ++lane) {
            if (!checked_product(beta.values[lane], mixture.closures[0U].weight[lane],
                                 updated.values[lane])) {
                return false;
            }
        }
        return true;
    }

    for (auto lane = std::uint32_t{0U}; lane < shared::HostDeviceSpectrumLaneCount; ++lane) {
        const float numerators[]{beta.values[lane], sample.value.values[lane], cosine,
                                 shading_normal_correction};
        const float denominators[]{sample.probability.value};
        const auto value = transport::checked_product_quotient(numerators, denominators);
        if (!transport::succeeded(value.status)) {
            return false;
        }
        updated.values[lane] = value.value;
    }
    return true;
}

[[nodiscard]] __device__ constexpr std::uint32_t
updated_delta_flags(const std::uint32_t current, const transport::ScatteringLobe event) noexcept {
    auto flags = (current & 2U) != 0U ? 2U : 0U;
    flags |= delta_surface_event(event) ? 1U : 2U;
    return flags;
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
    __shared__ std::uint32_t scene_validation_status;
    if (threadIdx.x == 0U) {
        scene_validation_status =
            static_cast<std::uint32_t>(validate_scene(scene_bytes, scene_size));
    }
    __syncthreads();

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
    const auto validation = static_cast<SceneDeviceStatus>(scene_validation_status);
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

[[nodiscard]] __device__ SceneDeviceStatus prepare_shadow_ray(
    const SurfaceData& surface, const IncidentLight& incident, const TransportRay& path_ray,
    const TransportPathStateLane& state, WavefrontPendingShadow& pending) noexcept {
    auto source_offset = Vector3{};
    if (!offset_point(surface.position, surface.position_error, surface.geometric_normal,
                      incident.direction_to_light, source_offset)) {
        return SceneDeviceStatus::numerical_failure;
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
            .time = path_ray.time,
            .visibility_mask = path_ray.visibility_mask,
            .current_medium = state.current_medium,
            .reserved = 0U,
        };
    } else {
        auto endpoint = incident.endpoint_position;
        if (incident.kind == static_cast<std::uint32_t>(IncidentLightKind::finite_surface)) {
            if (!offset_point(incident.endpoint_position, incident.endpoint_position_error,
                              incident.endpoint_geometric_normal,
                              multiply(incident.direction_to_light, -1.0F), endpoint)) {
                return SceneDeviceStatus::numerical_failure;
            }
        } else if (incident.kind == static_cast<std::uint32_t>(IncidentLightKind::finite_point)) {
            if (!contract_point_endpoint(incident.endpoint_position,
                                         incident.endpoint_position_error,
                                         incident.direction_to_light, endpoint)) {
                return SceneDeviceStatus::numerical_failure;
            }
        } else {
            return SceneDeviceStatus::invalid_scene;
        }
        auto segment = ScaledSegment{};
        if (!scaled_segment(subtract(endpoint, source_offset), segment) ||
            !(dot(segment.direction, incident.direction_to_light) > 0.0F)) {
            return SceneDeviceStatus::numerical_failure;
        }
        pending.ray = TransportRay{
            .origin_x = source_offset.x,
            .origin_y = source_offset.y,
            .origin_z = source_offset.z,
            .t_min = 0.0F,
            .direction_x = segment.direction.x,
            .direction_y = segment.direction.y,
            .direction_z = segment.direction.z,
            .t_max = nextafterf(segment.length, 0.0F),
            .time = path_ray.time,
            .visibility_mask = path_ray.visibility_mask,
            .current_medium = state.current_medium,
            .reserved = 0U,
        };
    }
    return valid_ray(pending.ray) && pending.ray.t_max > 0.0F
               ? SceneDeviceStatus::valid
               : SceneDeviceStatus::numerical_failure;
}

__global__ void
shade_stage_kernel(const std::uint8_t* const scene_bytes, const std::size_t scene_size,
                   const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
                   const WavefrontTransportConfig config, const std::uint32_t work_count,
                   WavefrontStageOutcome* const outcomes) {
    __shared__ std::uint32_t scene_validation_status;
    if (threadIdx.x == 0U) {
        scene_validation_status =
            static_cast<std::uint32_t>(validate_scene(scene_bytes, scene_size));
    }
    __syncthreads();

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
    const auto validation = static_cast<SceneDeviceStatus>(scene_validation_status);
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
    auto& ray_cone = streams.ray_cones[slot.value];
    const auto& hit = streams.hits[slot.value];
    auto& previous = streams.previous_bsdf_samples[slot.value];
    if (!valid_path_state(state) || !valid_ray(ray) || !valid_ray_cone(ray_cone) ||
        !valid_hit(hit) || ray.current_medium != state.current_medium ||
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
    const auto normal_map_present = scene_values<std::uint8_t>(
        scene_bytes, scene, scene_column::material_normal_map_present)[surface.material_index];
    const auto bump_map_present = scene_values<std::uint8_t>(
        scene_bytes, scene, scene_column::material_bump_map_present)[surface.material_index];
    const auto mapped_surface = normal_map_present != 0U || bump_map_present != 0U;
    auto advanced_cone = WavefrontRayCone{};
    auto cone_advanced = false;
    if (mapped_surface) {
        if (!advance_ray_cone_to_hit(ray_cone, ray, hit.parameter, advanced_cone)) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] =
                outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                        slot.value, value(ShadeFailureDetail::ray_cone_advance));
            return;
        }
        cone_advanced = true;
    }
    const auto maps_applied = apply_surface_maps(
        scene_bytes, scene, cone_advanced ? advanced_cone : ray_cone, ray, surface);
    if (maps_applied != SceneDeviceStatus::valid) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(scene_status(maps_applied), WavefrontStageRoute::none, slot.value,
                                  value(ShadeFailureDetail::surface_maps));
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
    auto material_range = SceneMaterialClosureRange{};
    auto closures = transport::ClosureMixtureRecord{};
    if (!load_spectrum(scene_bytes, scene, scene_column::material_emitted_radiance,
                       surface.material_index, emitted)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::material_spectrum));
        return;
    }
    const auto closure_loaded = load_material_closure_mixture(
        scene_bytes, scene, surface.material_index, material_range, closures);
    if (closure_loaded != SceneDeviceStatus::valid) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(scene_status(closure_loaded), WavefrontStageRoute::none,
                                  slot.value, value(ShadeFailureDetail::closure_mixture));
        return;
    }
    auto frame = LocalFrame{};
    if (!make_closure_frame(surface, material_range, frame)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::closure_frame));
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
    if (zero_spectrum(state.beta)) {
        control.flags = 0U;
        control.blocked_depth_limits = 0U;
        finish_phase(control, WavefrontLanePhase::terminated,
                     WavefrontTermination::zero_throughput);
        outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::terminated,
                                  slot.value, value(WavefrontTermination::zero_throughput));
        return;
    }
    if (!valid_depth_state(config, state)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::invalid_lane_state, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::depth_event));
        return;
    }

    const auto filtered_closures = transport::filter_closure_mixture_by_depth(
        closures,
        transport::PathDepthLimitsRecord{
            .diffuse = config.diffuse_depth_limit,
            .glossy = config.glossy_depth_limit,
            .specular = config.specular_depth_limit,
            .transmission = config.transmission_depth_limit,
            .volume = config.volume_depth_limit,
        },
        transport::PathDepthCountersRecord{
            .diffuse = state.diffuse_depth,
            .glossy = state.glossy_depth,
            .specular = state.specular_depth,
            .transmission = state.transmission_depth,
            .volume = state.volume_depth,
        },
        state.depth);
    if (!transport::succeeded(filtered_closures.status)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(filtered_closures.status == transport::Status::invalid_argument
                        ? WavefrontStageStatus::invalid_lane_state
                        : WavefrontStageStatus::numerical_failure,
                    WavefrontStageRoute::none, slot.value, value(ShadeFailureDetail::depth_event));
        return;
    }
    if (closures.active_count == 0U && filtered_closures.source_count != 0U) {
        control.flags = 0U;
        control.blocked_depth_limits =
            transport::scattering_lobe_bits(filtered_closures.blocked_lobes);
        finish_phase(control, WavefrontLanePhase::terminated, WavefrontTermination::depth_limit);
        outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::terminated,
                                  slot.value, value(WavefrontTermination::depth_limit));
        return;
    }

    const auto outgoing_local_world = to_local(frame, outgoing_world);
    const auto outgoing_local = transport::Vector3{
        .x = outgoing_local_world.x, .y = outgoing_local_world.y, .z = outgoing_local_world.z};

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

    auto shade_detail = cuda::WavefrontShadeDetailClosureSampled;
    const auto bsdf_component = cuda::sample_stream::sample_1d(streams.sample_streams[slot.value],
                                                               sample_dimensions.bsdf_component);
    const auto bsdf_u = cuda::sample_stream::sample_1d(streams.sample_streams[slot.value],
                                                       sample_dimensions.bsdf_u);
    const auto bsdf_v = cuda::sample_stream::sample_1d(streams.sample_streams[slot.value],
                                                       sample_dimensions.bsdf_v);
    const auto bsdf_sample = transport::sample_depth_filtered_closure_mixture(
        closures, filtered_closures, outgoing_local, bsdf_component, bsdf_u, bsdf_v,
        transport::TransportMode::radiance);
    const auto has_continuation_sample = bsdf_sample.status == transport::Status::success;
    if (!has_continuation_sample && bsdf_sample.status != transport::Status::outside_support) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::bsdf_sample));
        return;
    }
    if (has_continuation_sample) {
        const auto delta = delta_surface_event(bsdf_sample.lobes);
        const auto expected_measure = delta ? transport::ProbabilityMeasure::discrete
                                            : transport::ProbabilityMeasure::solid_angle;
        if (!valid_surface_event(bsdf_sample.lobes) ||
            bsdf_sample.probability.measure != expected_measure ||
            !isfinite(bsdf_sample.probability.value) || !(bsdf_sample.probability.value > 0.0F) ||
            (delta && bsdf_sample.probability.value > 1.0F)) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] =
                outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                        slot.value, value(ShadeFailureDetail::probability_measure));
            return;
        }
    }

    auto incident = IncidentLight{};
    if (config.light_count != 0U && mixture_has_nonblack_weight(closures)) {
        shade_detail |= cuda::WavefrontShadeDetailLightSampled;
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
        const auto incoming_local_world = to_local(frame, incident.direction_to_light);
        const auto incoming_local = transport::Vector3{
            .x = incoming_local_world.x, .y = incoming_local_world.y, .z = incoming_local_world.z};
        if (outgoing_local.z != 0.0F && incoming_local.z != 0.0F) {
            const auto singleton_lambertian =
                closures.active_count == 1U &&
                closures.closures[0U].kind == transport::ClosureKind::lambertian_reflection;
            auto bsdf_factor = TransportSpectrum{};
            if (singleton_lambertian) {
                if (outgoing_local.z > 0.0F && incoming_local.z > 0.0F) {
                    bsdf_factor = transport::record_weight(closures.closures[0U]);
                }
            } else {
                const auto evaluated = transport::depth_filtered_closure_mixture_eval(
                    closures, filtered_closures, outgoing_local, incoming_local,
                    transport::TransportMode::radiance);
                if (!transport::succeeded(evaluated.status)) {
                    finish_phase(control, WavefrontLanePhase::terminated);
                    outcomes[index] =
                        outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                                slot.value, value(ShadeFailureDetail::light_contribution));
                    return;
                }
                bsdf_factor = evaluated.value;
            }
            if (!zero_spectrum(bsdf_factor)) {
                auto shading_correction = 0.0F;
                if (!shading_normal_correction_radiance(
                        surface.geometric_normal, surface.shading_normal, outgoing_world,
                        incident.direction_to_light, shading_correction)) {
                    finish_phase(control, WavefrontLanePhase::terminated);
                    outcomes[index] =
                        outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                                slot.value, value(ShadeFailureDetail::shading_normal_correction));
                    return;
                }
                if (shading_correction > 0.0F) {
                    auto estimator_weight = 1.0F;
                    if (incident.probability_measure == SolidAngleMeasure) {
                        const auto joint = transport::joint_light_pdf(
                            transport::probability_density(incident.selection_probability,
                                                           transport::ProbabilityMeasure::discrete),
                            transport::probability_density(
                                incident.conditional_probability,
                                transport::ProbabilityMeasure::solid_angle));
                        const auto bsdf = transport::depth_filtered_closure_mixture_pdf(
                            closures, filtered_closures, outgoing_local, incoming_local,
                            transport::TransportMode::radiance);
                        const auto mis =
                            transport::succeeded(joint.status) && transport::succeeded(bsdf.status)
                                ? transport::mis_weight(
                                      static_cast<transport::MisHeuristic>(config.mis_heuristic),
                                      joint.value, bsdf.value)
                                : transport::ScalarResult{};
                        if (!transport::succeeded(joint.status) ||
                            !transport::succeeded(bsdf.status) ||
                            !transport::succeeded(mis.status)) {
                            finish_phase(control, WavefrontLanePhase::terminated);
                            outcomes[index] = outcome(WavefrontStageStatus::numerical_failure,
                                                      WavefrontStageRoute::none, slot.value,
                                                      value(ShadeFailureDetail::light_weight));
                            return;
                        }
                        estimator_weight = mis.value;
                    } else if (incident.probability_measure != DiscreteMeasure) {
                        finish_phase(control, WavefrontLanePhase::terminated);
                        outcomes[index] = outcome(WavefrontStageStatus::unsupported_transport,
                                                  WavefrontStageRoute::none, slot.value,
                                                  value(ShadeFailureDetail::light_weight));
                        return;
                    }
                    pending.beta = state.beta;
                    pending.bsdf_factor = bsdf_factor;
                    pending.incident_radiance = incident.radiance;
                    pending.absolute_incoming_cosine = fabsf(incoming_local.z);
                    pending.estimator_weight = estimator_weight;
                    pending.selection_probability = incident.selection_probability;
                    pending.conditional_probability = incident.conditional_probability;
                    pending.shading_normal_correction = shading_correction;
                    pending.bsdf_encoding = static_cast<std::uint32_t>(
                        singleton_lambertian ? WavefrontPendingBsdfEncoding::lambertian_coefficient
                                             : WavefrontPendingBsdfEncoding::value);
                    const auto shadow_status =
                        prepare_shadow_ray(surface, incident, ray, state, pending);
                    if (shadow_status != SceneDeviceStatus::valid) {
                        finish_phase(control, WavefrontLanePhase::terminated);
                        outcomes[index] =
                            outcome(scene_status(shadow_status), WavefrontStageRoute::none,
                                    slot.value, value(ShadeFailureDetail::shadow_ray));
                        return;
                    }
                    has_shadow = true;
                }
            }
        }
    }
    auto outgoing = Vector3{};
    auto continuation_supported = has_continuation_sample;
    auto shading_correction = 0.0F;
    if (continuation_supported) {
        if (!to_world(frame, bsdf_sample.incoming_local, outgoing)) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] =
                outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                        slot.value, value(ShadeFailureDetail::bsdf_sample));
            return;
        }
        auto geometric_supported = false;
        if (!shading_normal_correction_radiance(surface.geometric_normal, surface.shading_normal,
                                                outgoing_world, outgoing, shading_correction) ||
            !geometric_event_support(surface.geometric_normal, outgoing_world, outgoing,
                                     bsdf_sample.lobes, geometric_supported)) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] =
                outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                        slot.value, value(ShadeFailureDetail::geometric_support));
            return;
        }
        continuation_supported = shading_correction > 0.0F && geometric_supported;
    }
    if (!continuation_supported) {
        pending.continuation_pending = 0U;
        pending.termination = value(WavefrontTermination::outside_bsdf_support);
        if (has_shadow) {
            streams.pending_shadows[slot.value] = pending;
            control.flags = cuda::WavefrontLaneShadowPending;
            control.blocked_depth_limits = 0U;
            finish_phase(control, WavefrontLanePhase::shadow);
            outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::shadow,
                                      slot.value, shade_detail);
            return;
        }
        streams.pending_shadows[slot.value] = WavefrontPendingShadow{};
        control.flags = 0U;
        control.blocked_depth_limits = 0U;
        finish_phase(control, WavefrontLanePhase::terminated,
                     WavefrontTermination::outside_bsdf_support);
        outcomes[index] =
            outcome(WavefrontStageStatus::success, WavefrontStageRoute::terminated, slot.value,
                    value(WavefrontTermination::outside_bsdf_support) | shade_detail);
        return;
    }

    auto updated_beta = TransportSpectrum{};
    if (!update_closure_throughput(state.beta, bsdf_sample, shading_correction, closures,
                                   updated_beta)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::throughput));
        return;
    }
    const auto next_eta_scale = state.eta_scale * bsdf_sample.eta_scale_multiplier;
    if (!isfinite(bsdf_sample.eta_scale_multiplier) || !(bsdf_sample.eta_scale_multiplier > 0.0F) ||
        !isfinite(next_eta_scale) || !(next_eta_scale > 0.0F)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::eta_scale));
        return;
    }

    auto next_origin = Vector3{};
    if (!offset_point(surface.position, surface.position_error, surface.geometric_normal, outgoing,
                      next_origin)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::continuation_offset));
        return;
    }
    const auto next_ray = TransportRay{
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
    if (!valid_ray(next_ray)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::continuation_ray));
        return;
    }
    auto compact_closure_index = closures.active_count;
    auto compact_closure_matches = std::uint32_t{};
    for (auto index = std::uint32_t{}; index < closures.active_count; ++index) {
        if (filtered_closures.source_indices[index] == bsdf_sample.selected_closure) {
            compact_closure_index = index;
            ++compact_closure_matches;
        }
    }
    if (compact_closure_matches != 1U || compact_closure_index >= closures.active_count) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::invalid_lane_state, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::ray_cone_scattering));
        return;
    }
    if (!cone_advanced) {
        if (!advance_ray_cone_to_hit(ray_cone, ray, hit.parameter, advanced_cone)) {
            finish_phase(control, WavefrontLanePhase::terminated);
            outcomes[index] =
                outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none,
                        slot.value, value(ShadeFailureDetail::ray_cone_advance));
            return;
        }
    }
    auto next_cone = WavefrontRayCone{};
    if (!propagate_ray_cone_scattering(advanced_cone, closures.closures[compact_closure_index],
                                       bsdf_sample.lobes, outgoing_local,
                                       bsdf_sample.incoming_local, next_cone)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::numerical_failure, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::ray_cone_scattering));
        return;
    }
    auto next_state = state;
    next_state.beta = updated_beta;
    next_state.eta_scale = next_eta_scale;
    next_state.delta_flags = updated_delta_flags(state.delta_flags, bsdf_sample.lobes);
    if (!advance_depth_event(bsdf_sample.lobes, next_state) || !valid_path_state(next_state)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::invalid_lane_state, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::depth_event));
        return;
    }
    auto next_previous = WavefrontPreviousBsdfSample{};
    if (!delta_surface_event(bsdf_sample.lobes)) {
        next_previous = WavefrontPreviousBsdfSample{
            .context_x = surface.position.x,
            .context_y = surface.position.y,
            .context_z = surface.position.z,
            .context_time = next_ray.time,
            .incoming_x = outgoing.x,
            .incoming_y = outgoing.y,
            .incoming_z = outgoing.z,
            .probability_value = bsdf_sample.probability.value,
            .probability_measure = SolidAngleMeasure,
            .valid = 1U,
            .reserved = {0U, 0U},
        };
    }
    if (!valid_previous_bsdf_sample(next_previous, next_ray)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] =
            outcome(WavefrontStageStatus::invalid_lane_state, WavefrontStageRoute::none, slot.value,
                    value(ShadeFailureDetail::probability_measure));
        return;
    }
    state = next_state;
    ray = next_ray;
    ray_cone = next_cone;
    previous = next_previous;

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
        (!continuation_pending && !terminal_reason) || !valid_ray(pending.ray) ||
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
        !valid_ray(pending.ray) || !valid_pending_shadow_radiometry(pending)) {
        finish_phase(control, WavefrontLanePhase::terminated);
        outcomes[index] = outcome(WavefrontStageStatus::invalid_lane_state,
                                  WavefrontStageRoute::none, queue_path.value);
        return;
    }
    if (visible) {
        auto contribution = TransportSpectrum{};
        const auto encoding = static_cast<WavefrontPendingBsdfEncoding>(pending.bsdf_encoding);
        const auto lambertian_coefficient =
            encoding == WavefrontPendingBsdfEncoding::lambertian_coefficient;
        for (auto lane = std::uint32_t{}; lane < shared::HostDeviceSpectrumLaneCount; ++lane) {
            const float denominators[]{pending.selection_probability,
                                       pending.conditional_probability};
            if (pending.estimator_weight == 1.0F) {
                const float numerators[]{pending.beta.values[lane],
                                         pending.bsdf_factor.values[lane],
                                         lambertian_coefficient ? InversePi : 1.0F,
                                         pending.incident_radiance.values[lane],
                                         pending.absolute_incoming_cosine,
                                         pending.shading_normal_correction};
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
                                         pending.bsdf_factor.values[lane],
                                         lambertian_coefficient ? InversePi : 1.0F,
                                         pending.incident_radiance.values[lane],
                                         pending.absolute_incoming_cosine,
                                         pending.shading_normal_correction,
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
    finish_phase(control, WavefrontLanePhase::ray);
    outcomes[index] = outcome(WavefrontStageStatus::success, WavefrontStageRoute::ray, slot.value);
}

[[nodiscard]] __device__ bool stage_outcome_failed(const WavefrontStageOutcome stage_outcome,
                                                   const std::uint32_t allowed_route_mask,
                                                   const std::uint32_t path_capacity) noexcept {
    if (stage_outcome.status != value(WavefrontStageStatus::success)) {
        return true;
    }
    constexpr auto LastKnownRoute = static_cast<std::uint32_t>(WavefrontStageRoute::terminated);
    if (stage_outcome.route > LastKnownRoute) {
        return true;
    }
    const auto route_bit = std::uint32_t{1U} << stage_outcome.route;
    return (allowed_route_mask & route_bit) == 0U || stage_outcome.path_slot >= path_capacity;
}

__global__ void initialize_stage_audit_kernel(const std::uint32_t work_count,
                                              const std::uint32_t stage_kind,
                                              WavefrontStageAudit* const audit) {
    if (threadIdx.x == 0U) {
        *audit = WavefrontStageAudit{
            .abi_major = cuda::WavefrontStageAuditAbiMajor,
            .abi_minor = cuda::WavefrontStageAuditAbiMinor,
            .struct_size = sizeof(WavefrontStageAudit),
            .stage_kind = stage_kind,
            .expected_work_count = work_count,
            .inspected_work_count = 0U,
            .first_failure_work_index = 0xFFFFFFFFU,
        };
    }
}

__global__ void
stage_audit_kernel(const WavefrontStageOutcome* const outcomes, const std::uint32_t work_count,
                   const std::uint32_t allowed_route_mask, const std::uint32_t path_capacity,
                   const std::uint32_t stage_kind, WavefrontStageAudit* const audit) {
    __shared__ std::uint32_t first_failure_indices[ThreadsPerBlock];
    __shared__ std::uint32_t closure_sample_counts[ThreadsPerBlock];
    __shared__ std::uint32_t light_sample_counts[ThreadsPerBlock];

    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto first_failure = std::uint32_t{0xFFFFFFFFU};
    auto closure_samples = std::uint32_t{};
    auto light_samples = std::uint32_t{};
    const auto shade_stage = stage_kind == static_cast<std::uint32_t>(WavefrontStageKind::shade);
    if (index < work_count) {
        const auto stage_outcome = outcomes[index];
        if (stage_outcome_failed(stage_outcome, allowed_route_mask, path_capacity)) {
            first_failure = static_cast<std::uint32_t>(index);
        }
        if (shade_stage) {
            closure_samples +=
                (stage_outcome.detail & cuda::WavefrontShadeDetailClosureSampled) != 0U ? 1U : 0U;
            light_samples +=
                (stage_outcome.detail & cuda::WavefrontShadeDetailLightSampled) != 0U ? 1U : 0U;
        }
    }

    first_failure_indices[threadIdx.x] = first_failure;
    closure_sample_counts[threadIdx.x] = closure_samples;
    light_sample_counts[threadIdx.x] = light_samples;
    __syncthreads();
    for (auto stride = ThreadsPerBlock / 2U; stride != 0U; stride /= 2U) {
        if (threadIdx.x < stride) {
            const auto right_failure = first_failure_indices[threadIdx.x + stride];
            first_failure_indices[threadIdx.x] = right_failure < first_failure_indices[threadIdx.x]
                                                     ? right_failure
                                                     : first_failure_indices[threadIdx.x];
            closure_sample_counts[threadIdx.x] += closure_sample_counts[threadIdx.x + stride];
            light_sample_counts[threadIdx.x] += light_sample_counts[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0U) {
        atomicMin(&audit->first_failure_work_index, first_failure_indices[0U]);
        if (shade_stage) {
            atomicAdd(&audit->closure_samples, closure_sample_counts[0U]);
            atomicAdd(&audit->light_samples, light_sample_counts[0U]);
        }
        __threadfence();
        const auto completed_block = atomicAdd(&audit->reserved[0U], 1U);
        if (completed_block + 1U == gridDim.x) {
            audit->inspected_work_count = work_count;
            if (audit->first_failure_work_index < work_count) {
                audit->first_failure = outcomes[audit->first_failure_work_index];
            }
            audit->reserved[0U] = 0U;
        }
    }
}

[[nodiscard]] bool host_queue_view_is_valid(const WavefrontQueueDeviceSoa queues) noexcept {
    return queues.headers != nullptr && queues.queue_count == cuda::CudaWavefrontQueueCount &&
           (queues.slot_stride == 0U || queues.path_slots != nullptr);
}

[[nodiscard]] bool host_stream_view_is_valid(const WavefrontStageDeviceSoa streams) noexcept {
    return streams.reserved == 0U && streams.reserved_tail[0U] == 0U &&
           streams.reserved_tail[1U] == 0U &&
           (streams.capacity == 0U ||
            (streams.sample_streams != nullptr && streams.rays != nullptr &&
             streams.ray_cones != nullptr && streams.path_states != nullptr &&
             streams.hits != nullptr && streams.pending_shadows != nullptr &&
             streams.previous_bsdf_samples != nullptr && streams.controls != nullptr));
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

[[nodiscard]] cudaStream_t cuda_stream(void* const stream_handle) noexcept {
    return reinterpret_cast<cudaStream_t>(stream_handle);
}

} // namespace

extern "C" int blackframe_cuda_launch_wavefront_seed_camera(const WavefrontQueueDeviceSoa queues,
                                                            const WavefrontStageDeviceSoa streams,
                                                            const std::uint32_t first_path_slot,
                                                            const std::uint32_t path_count,
                                                            WavefrontStageOutcome* const outcomes,
                                                            void* const stream_handle) noexcept {
    if (!host_views_are_valid(queues, streams) || (path_count != 0U && outcomes == nullptr) ||
        first_path_slot > streams.capacity || path_count > streams.capacity - first_path_slot) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (path_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    const auto stream = cuda_stream(stream_handle);
    seed_camera_kernel<<<block_count(path_count), ThreadsPerBlock, 0U, stream>>>(
        queues, streams, first_path_slot, path_count, outcomes);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    publish_seed_camera_header_kernel<<<1U, 1U, 0U, stream>>>(queues, path_count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_clear_queue(
    const WavefrontQueueDeviceSoa queues, const std::uint32_t queue_kind,
    const std::uint32_t acknowledge_overflow, std::uint32_t* const device_status,
    void* const stream_handle) noexcept {
    if (!host_queue_view_is_valid(queues) || queue_kind >= cuda::CudaWavefrontQueueCount ||
        acknowledge_overflow > 1U || device_status == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    clear_queue_kernel<<<1U, 1U, 0U, cuda_stream(stream_handle)>>>(
        queues, queue_kind, acknowledge_overflow, device_status);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_camera_stage(
    const WavefrontQueueDeviceSoa queues, const WavefrontCameraInputDeviceSoa inputs,
    const WavefrontStageDeviceSoa streams, const std::uint32_t work_count,
    WavefrontStageOutcome* const outcomes, void* const stream_handle) noexcept {
    if (!host_views_are_valid(queues, streams) || inputs.reserved != 0U ||
        inputs.reserved_tail[0U] != 0U || inputs.reserved_tail[1U] != 0U ||
        inputs.count > streams.capacity ||
        (inputs.count != 0U && (inputs.sample_streams == nullptr || inputs.rays == nullptr ||
                                inputs.ray_cones == nullptr || inputs.path_states == nullptr)) ||
        (work_count != 0U && outcomes == nullptr)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    camera_stage_kernel<<<block_count(work_count), ThreadsPerBlock, 0U,
                          cuda_stream(stream_handle)>>>(queues, inputs, streams, work_count,
                                                        outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_gather_rays(const WavefrontQueueDeviceSoa queues,
                                                            const WavefrontStageDeviceSoa streams,
                                                            const std::uint32_t work_count,
                                                            PathSlot* const compact_path_slots,
                                                            TransportRay* const compact_rays,
                                                            WavefrontStageOutcome* const outcomes,
                                                            void* const stream_handle) noexcept {
    if (!host_views_are_valid(queues, streams) ||
        (work_count != 0U &&
         (compact_path_slots == nullptr || compact_rays == nullptr || outcomes == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    gather_rays_kernel<<<block_count(work_count), ThreadsPerBlock, 0U,
                         cuda_stream(stream_handle)>>>(queues, streams, work_count,
                                                       compact_path_slots, compact_rays, outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_classify_closest_hit(
    const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
    const PathSlot* const compact_path_slots, const SceneClosestHitResult* const compact_results,
    const std::uint32_t work_count, WavefrontStageOutcome* const outcomes,
    void* const stream_handle) noexcept {
    if (!host_views_are_valid(queues, streams) ||
        (work_count != 0U &&
         (compact_path_slots == nullptr || compact_results == nullptr || outcomes == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    classify_closest_hit_kernel<<<block_count(work_count), ThreadsPerBlock, 0U,
                                  cuda_stream(stream_handle)>>>(
        queues, streams, compact_path_slots, compact_results, work_count, outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_hit_stage(const WavefrontQueueDeviceSoa queues,
                                                          const WavefrontStageDeviceSoa streams,
                                                          const std::uint32_t work_count,
                                                          WavefrontStageOutcome* const outcomes,
                                                          void* const stream_handle) noexcept {
    if (!host_views_are_valid(queues, streams) || (work_count != 0U && outcomes == nullptr)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    hit_stage_kernel<<<block_count(work_count), ThreadsPerBlock, 0U, cuda_stream(stream_handle)>>>(
        queues, streams, work_count, outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_miss_stage(const std::uint8_t* const scene_bytes,
                                                           const std::size_t scene_size,
                                                           const WavefrontQueueDeviceSoa queues,
                                                           const WavefrontStageDeviceSoa streams,
                                                           const std::uint32_t work_count,
                                                           WavefrontStageOutcome* const outcomes,
                                                           void* const stream_handle) noexcept {
    if (!host_views_are_valid(queues, streams) ||
        (work_count != 0U &&
         (scene_bytes == nullptr || scene_size < sizeof(SceneSoaHeader) || outcomes == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    miss_stage_kernel<<<block_count(work_count), ThreadsPerBlock, 0U, cuda_stream(stream_handle)>>>(
        scene_bytes, scene_size, queues, streams, work_count, outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_shade_stage(
    const std::uint8_t* const scene_bytes, const std::size_t scene_size,
    const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
    const WavefrontTransportConfig config, const std::uint32_t work_count,
    WavefrontStageOutcome* const outcomes, void* const stream_handle) noexcept {
    if (!host_views_are_valid(queues, streams) ||
        (work_count != 0U &&
         (scene_bytes == nullptr || scene_size < sizeof(SceneSoaHeader) || outcomes == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    shade_stage_kernel<<<block_count(work_count), ThreadsPerBlock, 0U,
                         cuda_stream(stream_handle)>>>(scene_bytes, scene_size, queues, streams,
                                                       config, work_count, outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_gather_shadow_rays(
    const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
    const std::uint32_t work_count, PathSlot* const compact_path_slots,
    TransportRay* const compact_rays, WavefrontStageOutcome* const outcomes,
    void* const stream_handle) noexcept {
    if (!host_views_are_valid(queues, streams) ||
        (work_count != 0U &&
         (compact_path_slots == nullptr || compact_rays == nullptr || outcomes == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    gather_shadow_rays_kernel<<<block_count(work_count), ThreadsPerBlock, 0U,
                                cuda_stream(stream_handle)>>>(
        queues, streams, work_count, compact_path_slots, compact_rays, outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_process_shadow(
    const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
    const PathSlot* const compact_path_slots, const SceneOcclusionResult* const compact_results,
    const std::uint32_t work_count, WavefrontStageOutcome* const outcomes,
    void* const stream_handle) noexcept {
    if (!host_views_are_valid(queues, streams) ||
        (work_count != 0U &&
         (compact_path_slots == nullptr || compact_results == nullptr || outcomes == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    process_shadow_kernel<<<block_count(work_count), ThreadsPerBlock, 0U,
                            cuda_stream(stream_handle)>>>(queues, streams, compact_path_slots,
                                                          compact_results, work_count, outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_continuation_stage(
    const WavefrontQueueDeviceSoa queues, const WavefrontStageDeviceSoa streams,
    const std::uint32_t work_count, WavefrontStageOutcome* const outcomes,
    void* const stream_handle) noexcept {
    if (!host_views_are_valid(queues, streams) || (work_count != 0U && outcomes == nullptr)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (work_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    continuation_stage_kernel<<<block_count(work_count), ThreadsPerBlock, 0U,
                                cuda_stream(stream_handle)>>>(queues, streams, work_count,
                                                              outcomes);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blackframe_cuda_launch_wavefront_audit_stage(
    const WavefrontStageOutcome* const outcomes, const std::uint32_t work_count,
    const std::uint32_t allowed_route_mask, const std::uint32_t path_capacity,
    const std::uint32_t stage_kind, WavefrontStageAudit* const audit,
    void* const stream_handle) noexcept {
    if (audit == nullptr || (work_count != 0U && outcomes == nullptr) ||
        stage_kind > static_cast<std::uint32_t>(WavefrontStageKind::continuation)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const auto stream = cuda_stream(stream_handle);
    initialize_stage_audit_kernel<<<1U, 1U, 0U, stream>>>(work_count, stage_kind, audit);
    auto status = cudaGetLastError();
    if (status != cudaSuccess || work_count == 0U) {
        return static_cast<int>(status);
    }
    stage_audit_kernel<<<block_count(work_count), ThreadsPerBlock, 0U, stream>>>(
        outcomes, work_count, allowed_route_mask, path_capacity, stage_kind, audit);
    return static_cast<int>(cudaGetLastError());
}
