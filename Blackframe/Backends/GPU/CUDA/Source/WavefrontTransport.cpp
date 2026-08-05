#include "SceneQuery.hpp"

#include <Blackframe/Backends/GPU/CUDA/WavefrontQueues.hpp>
#include <Blackframe/Backends/GPU/CUDA/WavefrontTransport.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <Blackframe/XPU/CUDA/SceneClosestHit.hpp>
#include <Blackframe/XPU/CUDA/SceneOcclusion.hpp>
#include <Blackframe/XPU/CUDA/WavefrontStageKernel.hpp>
#include <Blackframe/XPU/Shared/SceneTraversalAbi.hpp>
#include <Blackframe/XPU/Shared/TransportAbi.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

using xpu::cuda::WavefrontCameraInputDeviceSoa;
using xpu::cuda::WavefrontLaneControl;
using xpu::cuda::WavefrontLanePhase;
using xpu::cuda::WavefrontPendingShadow;
using xpu::cuda::WavefrontPreviousBsdfSample;
using xpu::cuda::WavefrontStageDeviceSoa;
using xpu::cuda::WavefrontStageOutcome;
using xpu::cuda::WavefrontStageRoute;
using xpu::cuda::WavefrontStageStatus;
using xpu::cuda::WavefrontTermination;
using xpu::cuda::WavefrontTransportConfig;
using xpu::shared::ClosestHit;
using xpu::shared::PathSlot;
using xpu::shared::QueueHeader;
using xpu::shared::SampleStreamIndex;
using xpu::shared::SceneClosestHitResult;
using xpu::shared::SceneOcclusionResult;
using xpu::shared::TransportPathStateLane;
using xpu::shared::TransportRay;

static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::camera) == 0U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::ray) == 1U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::hit) == 2U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::miss) == 3U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::shade) == 4U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::shadow) == 5U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::continuation) == 6U);
static_assert(static_cast<std::uint32_t>(renderer::ProbabilityMeasure::discrete) == 0U);
static_assert(static_cast<std::uint32_t>(renderer::ProbabilityMeasure::solid_angle) == 1U);
static_assert(static_cast<std::uint32_t>(renderer::ProbabilityMeasure::area) == 2U);
static_assert(static_cast<std::uint32_t>(renderer::ProbabilityMeasure::distance) == 3U);
static_assert(static_cast<std::uint32_t>(renderer::ProbabilityMeasure::volume) == 4U);
static_assert(static_cast<std::uint32_t>(renderer::ProbabilityMeasure::wavelength) == 5U);
static_assert(static_cast<std::uint32_t>(renderer::MisHeuristic::balance) == 0U);
static_assert(static_cast<std::uint32_t>(renderer::MisHeuristic::power) == 1U);
static_assert(static_cast<std::uint32_t>(renderer::LightSamplingStrategy::uniform) == 0U);
static_assert(static_cast<std::uint32_t>(renderer::LightSamplingStrategy::power_weighted) == 1U);
static_assert(static_cast<std::uint32_t>(renderer::LightSamplingStrategy::spatial_tree) == 2U);
static_assert(static_cast<std::uint32_t>(renderer::RussianRouletteMode::disabled) == 0U);
static_assert(static_cast<std::uint32_t>(renderer::RussianRouletteMode::enabled) == 1U);
static_assert(static_cast<std::uint32_t>(renderer::ScatteringLobe::diffuse) == 0x00000001U);
static_assert(static_cast<std::uint32_t>(renderer::PathDeltaFlags::previous_bounce_was_delta) ==
              1U);
static_assert(static_cast<std::uint32_t>(renderer::PathDeltaFlags::any_non_delta_bounces) == 2U);

[[nodiscard]] core::Error transport_error(const core::StatusCode code, std::string message) {
    return core::Error{.code = code, .message = std::move(message)};
}

[[nodiscard]] core::Error cuda_transport_error(const cudaError_t status,
                                               const std::string_view operation,
                                               const std::size_t byte_count = 0U) {
    auto message = std::string{"CUDA wavefront transport "} + std::string{operation} + " failed";
    if (byte_count != 0U) {
        message += " for " + std::to_string(byte_count) + " bytes";
    }
    message += ": ";
    message += cudaGetErrorName(status);
    message += " (";
    message += cudaGetErrorString(status);
    message += ").";
    return transport_error(xpu::cuda::cuda_memory_status_code(static_cast<std::int32_t>(status)),
                           std::move(message));
}

[[nodiscard]] constexpr std::uint32_t
queue_index(const renderer::WavefrontQueueKind kind) noexcept {
    return static_cast<std::uint32_t>(kind);
}

[[nodiscard]] core::Status
validate_options_and_inputs(const std::span<const CudaWavefrontPathInput> inputs,
                            const CudaWavefrontTransportOptions options) {
    if (inputs.empty()) {
        return std::unexpected(
            transport_error(core::StatusCode::invalid_argument,
                            "CUDA wavefront transport requires at least one explicit path input."));
    }
    if (inputs.size() > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront transport path count exceeds its fixed 32-bit device domain."));
    }
    switch (options.heuristic) {
    case renderer::MisHeuristic::balance:
    case renderer::MisHeuristic::power:
        break;
    default:
        return std::unexpected(transport_error(
            core::StatusCode::invalid_argument,
            "CUDA wavefront transport requires the balance or power MIS heuristic."));
    }
    if (auto status = renderer::validate_russian_roulette_policy(options.roulette_policy);
        !status) {
        return std::unexpected(transport_error(
            status.error().code, "CUDA wavefront transport rejected its Russian roulette policy: " +
                                     status.error().message));
    }
    for (auto index = std::size_t{}; index < inputs.size(); ++index) {
        const auto& input = inputs[index];
        if (input.primary_ray.current_medium() != input.initial_state.current_medium()) {
            return std::unexpected(transport_error(
                core::StatusCode::incompatible,
                "CUDA wavefront transport path and ray medium identities disagree at path " +
                    std::to_string(index) + "."));
        }
        if (input.initial_state.current_medium() != renderer::VacuumMedium) {
            return std::unexpected(transport_error(
                core::StatusCode::invalid_argument,
                "CUDA wavefront transport currently requires vacuum rays and path states; path " +
                    std::to_string(index) + " selected another medium."));
        }
        if (input.initial_state.depth() != 0U) {
            return std::unexpected(transport_error(
                core::StatusCode::incompatible,
                "CUDA wavefront transport must begin at primary path depth zero; path " +
                    std::to_string(index) + " supplied a resumed state."));
        }
        if (auto status = renderer::validate_path_depth_state(options.depth_limits,
                                                              input.initial_state.depth_counters(),
                                                              input.initial_state.depth());
            !status) {
            return std::unexpected(transport_error(
                status.error().code, "CUDA wavefront transport rejected depth state at path " +
                                         std::to_string(index) + ": " + status.error().message));
        }
        const auto finite_nonnegative = [](const renderer::TransportSpectrum& spectrum) {
            return std::ranges::all_of(spectrum.values, [](const auto value) {
                return std::isfinite(value) && value >= 0.0F;
            });
        };
        if (!finite_nonnegative(input.initial_state.beta()) ||
            !finite_nonnegative(input.initial_state.accumulated_radiance())) {
            return std::unexpected(transport_error(
                core::StatusCode::invalid_argument,
                "CUDA wavefront transport requires finite non-negative path spectra at path " +
                    std::to_string(index) + "."));
        }
    }
    return {};
}

struct ValidatedLightSampler final {
    std::uint32_t light_count{};
    bool present{};
    renderer::LightSamplingStrategy strategy{renderer::LightSamplingStrategy::uniform};
};

[[nodiscard]] core::Result<ValidatedLightSampler>
validate_light_sampler(const xpu::shared::SceneSoaHeader& scene,
                       const CudaWavefrontLightSampler light_sampler) {
    if (scene.punctual_light_count >
        std::numeric_limits<std::uint64_t>::max() - scene.mesh_area_light_count) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront light-registry size overflowed its serialized domain."));
    }
    const auto expected_count = scene.punctual_light_count + scene.mesh_area_light_count;
    if (expected_count == 0U) {
        if (light_sampler.has_value()) {
            return std::unexpected(transport_error(
                core::StatusCode::incompatible,
                "CUDA wavefront transport requires the light sampler to be explicitly absent "
                "when the serialized scene light registry is empty."));
        }
        return ValidatedLightSampler{};
    }
    if (expected_count > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront light registry exceeds its fixed 32-bit device domain."));
    }
    const auto count = static_cast<std::uint32_t>(expected_count);
    if (!light_sampler.has_value()) {
        return std::unexpected(transport_error(
            core::StatusCode::incompatible,
            "CUDA wavefront transport requires an explicit light sampler for a non-empty "
            "serialized scene light registry."));
    }
    const auto& sampler = light_sampler->get();
    if (sampler.strategy() != renderer::LightSamplingStrategy::uniform) {
        return std::unexpected(transport_error(
            core::StatusCode::unavailable,
            "CUDA wavefront transport currently requires an explicit uniform light sampler; "
            "the requested strategy is not ported and will not be substituted."));
    }
    if (sampler.light_count() != count) {
        return std::unexpected(transport_error(
            core::StatusCode::incompatible,
            "CUDA wavefront light sampler count does not match the serialized punctual and "
            "mesh-area registry."));
    }
    return ValidatedLightSampler{
        .light_count = count,
        .present = true,
        .strategy = sampler.strategy(),
    };
}

[[nodiscard]] WavefrontTransportConfig
transport_config(const CudaWavefrontTransportOptions& options, const std::uint32_t light_count,
                 const renderer::LightSamplingStrategy strategy) noexcept {
    return WavefrontTransportConfig{
        .abi_major = xpu::cuda::WavefrontTransportConfigAbiMajor,
        .abi_minor = xpu::cuda::WavefrontTransportConfigAbiMinor,
        .struct_size = sizeof(WavefrontTransportConfig),
        .mis_heuristic = static_cast<std::uint32_t>(options.heuristic),
        .light_sampling_strategy = static_cast<std::uint32_t>(strategy),
        .light_count = light_count,
        .diffuse_depth_limit = options.depth_limits.diffuse,
        .glossy_depth_limit = options.depth_limits.glossy,
        .specular_depth_limit = options.depth_limits.specular,
        .transmission_depth_limit = options.depth_limits.transmission,
        .volume_depth_limit = options.depth_limits.volume,
        .russian_roulette_mode = static_cast<std::uint32_t>(options.roulette_policy.mode()),
        .russian_roulette_first_depth = options.roulette_policy.first_eligible_depth(),
        .russian_roulette_minimum_probability =
            options.roulette_policy.minimum_survival_probability(),
        .russian_roulette_maximum_probability =
            options.roulette_policy.maximum_survival_probability(),
        .reserved = {0U, 0U},
    };
}

[[nodiscard]] TransportPathStateLane transport_path_state(const renderer::PathState& state) {
    auto result = TransportPathStateLane{};
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        result.beta.values[lane] = state.beta()[lane];
        result.accumulated_radiance.values[lane] = state.accumulated_radiance()[lane];
        result.wavelength_nanometers.values[lane] = state.wavelengths()[lane].nanometers;
        result.wavelength_pdf_values.values[lane] = state.wavelengths()[lane].probability.value;
        result.wavelength_pdf_measures[lane] =
            static_cast<std::uint8_t>(state.wavelengths()[lane].probability.measure);
    }
    const auto& counters = state.depth_counters();
    result.diffuse_depth = counters.diffuse;
    result.glossy_depth = counters.glossy;
    result.specular_depth = counters.specular;
    result.transmission_depth = counters.transmission;
    result.volume_depth = counters.volume;
    result.depth = state.depth();
    result.eta_scale = state.eta_scale();
    result.current_medium = state.current_medium().value;
    result.delta_flags = static_cast<std::uint32_t>(state.delta_flags());
    return result;
}

[[nodiscard]] SampleStreamIndex
transport_sample_stream_index(const renderer::SampleStreamIndex index) noexcept {
    return SampleStreamIndex{
        .pixel_x = index.pixel_x,
        .pixel_y = index.pixel_y,
        .sample_index = index.sample_index,
        .seed = index.seed,
    };
}

[[nodiscard]] core::Result<renderer::PathState>
renderer_path_state(const TransportPathStateLane& state, const std::size_t path_index) {
    if (std::ranges::any_of(state.reserved, [](const auto value) { return value != 0U; })) {
        return std::unexpected(
            transport_error(core::StatusCode::internal_error,
                            "CUDA wavefront path state returned non-zero reserved data at path " +
                                std::to_string(path_index) + "."));
    }
    auto beta = renderer::TransportSpectrum{};
    auto radiance = renderer::TransportSpectrum{};
    auto wavelengths = renderer::SampledWavelengths{};
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        beta[lane] = state.beta.values[lane];
        radiance[lane] = state.accumulated_radiance.values[lane];
        const auto measure =
            static_cast<renderer::ProbabilityMeasure>(state.wavelength_pdf_measures[lane]);
        if (measure != renderer::ProbabilityMeasure::wavelength) {
            return std::unexpected(transport_error(
                core::StatusCode::internal_error,
                "CUDA wavefront path returned a non-wavelength spectral PDF measure."));
        }
        wavelengths[lane] = renderer::WavelengthSample{
            .nanometers = state.wavelength_nanometers.values[lane],
            .probability =
                renderer::ProbabilityDensity{
                    .value = state.wavelength_pdf_values.values[lane],
                    .measure = measure,
                },
        };
    }
    const auto delta_bits = state.delta_flags;
    if ((delta_bits & ~std::uint32_t{0x3U}) != 0U) {
        return std::unexpected(
            transport_error(core::StatusCode::internal_error,
                            "CUDA wavefront path returned unsupported delta-history bits."));
    }
    auto converted = renderer::PathState::create(beta, radiance,
                                                 renderer::PathDepthCounters{
                                                     .diffuse = state.diffuse_depth,
                                                     .glossy = state.glossy_depth,
                                                     .specular = state.specular_depth,
                                                     .transmission = state.transmission_depth,
                                                     .volume = state.volume_depth,
                                                 },
                                                 state.eta_scale, wavelengths,
                                                 static_cast<renderer::PathDeltaFlags>(delta_bits),
                                                 renderer::MediumId{.value = state.current_medium});
    if (!converted) {
        return std::unexpected(
            transport_error(core::StatusCode::internal_error,
                            "CUDA wavefront path state " + std::to_string(path_index) +
                                " violated the renderer contract: " + converted.error().message));
    }
    if (converted->depth() != state.depth) {
        return std::unexpected(
            transport_error(core::StatusCode::internal_error,
                            "CUDA wavefront path returned an inconsistent total depth."));
    }
    return converted;
}

[[nodiscard]] core::Result<renderer::Ray> renderer_ray(const TransportRay& ray,
                                                       const std::size_t path_index) {
    if (ray.reserved != 0U) {
        return std::unexpected(
            transport_error(core::StatusCode::internal_error,
                            "CUDA wavefront terminal ray returned non-zero reserved data at path " +
                                std::to_string(path_index) + "."));
    }
    auto converted = renderer::Ray::create(
        renderer::Point3{.x = ray.origin_x, .y = ray.origin_y, .z = ray.origin_z},
        renderer::Vector3{
            .x = ray.direction_x,
            .y = ray.direction_y,
            .z = ray.direction_z,
        },
        ray.t_min, ray.t_max, ray.time, ray.visibility_mask,
        renderer::MediumId{.value = ray.current_medium});
    if (!converted) {
        return std::unexpected(
            transport_error(core::StatusCode::internal_error,
                            "CUDA wavefront terminal ray " + std::to_string(path_index) +
                                " violated the renderer contract: " + converted.error().message));
    }
    return converted;
}

struct RendererTermination final {
    CudaWavefrontPathTermination reason{};
    renderer::ScatteringLobe blocked_depth_limits{renderer::ScatteringLobe::none};
};

[[nodiscard]] core::Result<RendererTermination>
renderer_termination(const WavefrontLaneControl& control, const std::size_t path_index) {
    if (control.phase != static_cast<std::uint32_t>(WavefrontLanePhase::terminated) ||
        control.flags != 0U) {
        return std::unexpected(transport_error(core::StatusCode::internal_error,
                                               "CUDA wavefront path " + std::to_string(path_index) +
                                                   " did not finish in canonical terminal state."));
    }
    constexpr auto depth_category_mask =
        renderer::ScatteringLobe::diffuse | renderer::ScatteringLobe::glossy |
        renderer::ScatteringLobe::specular | renderer::ScatteringLobe::transmission |
        renderer::ScatteringLobe::volume;
    const auto blocked = static_cast<renderer::ScatteringLobe>(control.blocked_depth_limits);
    if (!renderer::is_known_scattering_lobe_mask(blocked) ||
        (blocked & depth_category_mask) != blocked) {
        return std::unexpected(transport_error(core::StatusCode::internal_error,
                                               "CUDA wavefront path " + std::to_string(path_index) +
                                                   " returned invalid blocked depth-limit bits."));
    }
    const auto termination = static_cast<WavefrontTermination>(control.termination);
    if ((termination == WavefrontTermination::diffuse_depth_limit &&
         blocked != renderer::ScatteringLobe::diffuse) ||
        (termination != WavefrontTermination::diffuse_depth_limit &&
         blocked != renderer::ScatteringLobe::none)) {
        return std::unexpected(
            transport_error(core::StatusCode::internal_error,
                            "CUDA wavefront path " + std::to_string(path_index) +
                                " returned blocked depth-limit bits incompatible with its "
                                "termination."));
    }
    switch (termination) {
    case WavefrontTermination::escaped_environment:
        return RendererTermination{.reason = CudaWavefrontPathTermination::escaped_environment};
    case WavefrontTermination::diffuse_depth_limit:
        return RendererTermination{
            .reason = CudaWavefrontPathTermination::depth_limit,
            .blocked_depth_limits = blocked,
        };
    case WavefrontTermination::zero_throughput:
        return RendererTermination{.reason = CudaWavefrontPathTermination::zero_throughput};
    case WavefrontTermination::outside_bsdf_support:
        return RendererTermination{.reason = CudaWavefrontPathTermination::outside_bsdf_support};
    case WavefrontTermination::russian_roulette:
        return RendererTermination{.reason = CudaWavefrontPathTermination::russian_roulette};
    case WavefrontTermination::none:
        break;
    }
    return std::unexpected(
        transport_error(core::StatusCode::internal_error,
                        "CUDA wavefront path " + std::to_string(path_index) +
                            " returned an unknown or missing termination reason."));
}

[[nodiscard]] constexpr bool known_stage_route(const WavefrontStageRoute route) noexcept {
    switch (route) {
    case WavefrontStageRoute::none:
    case WavefrontStageRoute::ray:
    case WavefrontStageRoute::hit:
    case WavefrontStageRoute::miss:
    case WavefrontStageRoute::shade:
    case WavefrontStageRoute::shadow:
    case WavefrontStageRoute::continuation:
    case WavefrontStageRoute::terminated:
        return true;
    }
    return false;
}

using StageRouteMask = std::uint32_t;

[[nodiscard]] constexpr StageRouteMask route_mask(const WavefrontStageRoute route) noexcept {
    return StageRouteMask{1U} << static_cast<std::uint32_t>(route);
}

[[nodiscard]] core::Status stage_failure(const WavefrontStageOutcome& outcome,
                                         const std::string_view stage_name,
                                         const std::size_t work_index,
                                         const std::uint32_t path_capacity,
                                         const StageRouteMask allowed_routes) {
    const auto status = static_cast<WavefrontStageStatus>(outcome.status);
    if (status == WavefrontStageStatus::success) {
        const auto route = static_cast<WavefrontStageRoute>(outcome.route);
        if (!known_stage_route(route) || (allowed_routes & route_mask(route)) == 0U ||
            outcome.path_slot >= path_capacity) {
            return std::unexpected(
                transport_error(core::StatusCode::internal_error,
                                "CUDA wavefront " + std::string{stage_name} +
                                    " returned a non-canonical outcome at work item " +
                                    std::to_string(work_index) + "."));
        }
        return {};
    }

    auto code = core::StatusCode::internal_error;
    switch (status) {
    case WavefrontStageStatus::invalid_contract:
    case WavefrontStageStatus::invalid_path_slot:
    case WavefrontStageStatus::invalid_lane_state:
    case WavefrontStageStatus::invalid_ray:
        code = core::StatusCode::incompatible;
        break;
    case WavefrontStageStatus::invalid_scene:
    case WavefrontStageStatus::traversal_error:
        code = core::StatusCode::incompatible;
        break;
    case WavefrontStageStatus::unsupported_transport:
        code = core::StatusCode::unavailable;
        break;
    case WavefrontStageStatus::queue_overflow:
        code = core::StatusCode::resource_exhausted;
        break;
    case WavefrontStageStatus::numerical_failure:
        code = core::StatusCode::internal_error;
        break;
    case WavefrontStageStatus::success:
        break;
    }
    return std::unexpected(transport_error(
        code, "CUDA wavefront " + std::string{stage_name} + " failed at work item " +
                  std::to_string(work_index) + " (path slot " + std::to_string(outcome.path_slot) +
                  ", status " + std::to_string(outcome.status) + ", detail " +
                  std::to_string(outcome.detail) + ")."));
}

template <typename Launcher>
[[nodiscard]] core::Status
execute_stage(const std::string_view stage_name, const std::uint32_t work_count,
              const StageRouteMask allowed_routes, const std::uint32_t path_capacity,
              WavefrontStageOutcome* const device_outcomes,
              std::vector<WavefrontStageOutcome>& host_outcomes, Launcher&& launcher) {
    if (work_count == 0U) {
        return {};
    }
    const auto byte_count = static_cast<std::size_t>(work_count) * sizeof(WavefrontStageOutcome);
    auto cuda_status = cudaMemset(device_outcomes, 0xFF, byte_count);
    if (cuda_status != cudaSuccess) {
        return std::unexpected(
            cuda_transport_error(cuda_status, "outcome initialization", byte_count));
    }
    cuda_status = static_cast<cudaError_t>(launcher());
    if (cuda_status != cudaSuccess) {
        return std::unexpected(
            cuda_transport_error(cuda_status, std::string{stage_name} + " kernel launch"));
    }
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        return std::unexpected(
            cuda_transport_error(cuda_status, std::string{stage_name} + " kernel synchronization"));
    }
    host_outcomes.resize(work_count);
    cuda_status =
        cudaMemcpy(host_outcomes.data(), device_outcomes, byte_count, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        return std::unexpected(cuda_transport_error(cuda_status, "outcome download", byte_count));
    }
    for (auto index = std::size_t{}; index < host_outcomes.size(); ++index) {
        if (auto status = stage_failure(host_outcomes[index], stage_name, index, path_capacity,
                                        allowed_routes);
            !status) {
            return status;
        }
    }
    return {};
}

[[nodiscard]] core::Result<QueueHeader>
queue_header(const xpu::cuda::WavefrontQueueDeviceSoa queues,
             const renderer::WavefrontQueueKind kind, const std::uint32_t capacity) {
    auto header = QueueHeader{};
    const auto status = cudaMemcpy(&header, queues.headers + queue_index(kind), sizeof(header),
                                   cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        return std::unexpected(
            cuda_transport_error(status, "queue-header download", sizeof(header)));
    }
    if (xpu::shared::validate_queue_header(header) !=
            xpu::shared::QueueHeaderValidationStatus::valid ||
        header.queue_kind != queue_index(kind) || header.capacity != capacity ||
        header.overflow_count != header.rejected_count) {
        return std::unexpected(transport_error(
            core::StatusCode::incompatible,
            "CUDA wavefront queue counters violated their fixed host/device contract."));
    }
    if (header.overflow_count != 0U) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront queue exhausted its fixed capacity without publishing a partial "
            "fallback result."));
    }
    return header;
}

[[nodiscard]] core::Status clear_queue(const xpu::cuda::WavefrontQueueDeviceSoa queues,
                                       const renderer::WavefrontQueueKind kind,
                                       std::uint32_t* const device_status) {
    constexpr auto sentinel = std::numeric_limits<std::uint32_t>::max();
    auto status = cudaMemcpy(device_status, &sentinel, sizeof(sentinel), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        return std::unexpected(
            cuda_transport_error(status, "queue-clear status initialization", sizeof(sentinel)));
    }
    status = static_cast<cudaError_t>(
        blackframe_cuda_launch_wavefront_clear_queue(queues, queue_index(kind), 0U, device_status));
    if (status != cudaSuccess) {
        return std::unexpected(cuda_transport_error(status, "queue-clear kernel launch"));
    }
    auto host_status = sentinel;
    status = cudaMemcpy(&host_status, device_status, sizeof(host_status), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        return std::unexpected(
            cuda_transport_error(status, "queue-clear status download", sizeof(host_status)));
    }
    if (host_status != static_cast<std::uint32_t>(WavefrontStageStatus::success)) {
        return std::unexpected(transport_error(
            host_status == static_cast<std::uint32_t>(WavefrontStageStatus::queue_overflow)
                ? core::StatusCode::resource_exhausted
                : core::StatusCode::incompatible,
            "CUDA wavefront queue clear refused a non-canonical consumed queue."));
    }
    return {};
}

[[nodiscard]] core::Result<std::size_t>
scratch_capacity(const std::size_t path_count, const xpu::cuda::DeviceMemoryBudget memory_budget) {
    constexpr auto column_element_bytes =
        sizeof(SampleStreamIndex) + sizeof(TransportRay) + sizeof(TransportPathStateLane) +
        sizeof(SampleStreamIndex) + sizeof(TransportRay) + sizeof(TransportPathStateLane) +
        sizeof(ClosestHit) + sizeof(WavefrontPendingShadow) + sizeof(WavefrontLaneControl) +
        sizeof(WavefrontPreviousBsdfSample) + sizeof(PathSlot) + sizeof(TransportRay) +
        sizeof(SceneClosestHitResult) + sizeof(SceneOcclusionResult) +
        sizeof(WavefrontStageOutcome);
    constexpr auto column_alignment_slack =
        (alignof(SampleStreamIndex) - 1U) + (alignof(TransportRay) - 1U) +
        (alignof(TransportPathStateLane) - 1U) + (alignof(SampleStreamIndex) - 1U) +
        (alignof(TransportRay) - 1U) + (alignof(TransportPathStateLane) - 1U) +
        (alignof(ClosestHit) - 1U) + (alignof(WavefrontPendingShadow) - 1U) +
        (alignof(WavefrontPreviousBsdfSample) - 1U) + (alignof(WavefrontLaneControl) - 1U) +
        (alignof(PathSlot) - 1U) + (alignof(TransportRay) - 1U) +
        (alignof(SceneClosestHitResult) - 1U) + (alignof(SceneOcclusionResult) - 1U) +
        (alignof(WavefrontStageOutcome) - 1U) + (alignof(std::uint32_t) - 1U);
    constexpr auto fixed_bytes = sizeof(std::uint32_t);
    if (path_count >
        (std::numeric_limits<std::size_t>::max() - column_alignment_slack - fixed_bytes) /
            column_element_bytes) {
        return std::unexpected(
            transport_error(core::StatusCode::resource_exhausted,
                            "CUDA wavefront scratch byte count overflowed its host size domain."));
    }
    const auto scratch_bytes =
        path_count * column_element_bytes + column_alignment_slack + fixed_bytes;
    constexpr auto queue_header_bytes =
        static_cast<std::size_t>(xpu::cuda::CudaWavefrontQueueCount) * sizeof(QueueHeader);
    constexpr auto queue_slot_bytes_per_path =
        static_cast<std::size_t>(xpu::cuda::CudaWavefrontQueueCount) * sizeof(PathSlot);
    if (path_count > (std::numeric_limits<std::size_t>::max() - queue_header_bytes) /
                         queue_slot_bytes_per_path) {
        return std::unexpected(transport_error(core::StatusCode::resource_exhausted,
                                               "CUDA wavefront queue byte count overflowed."));
    }
    const auto queue_bytes = queue_header_bytes + path_count * queue_slot_bytes_per_path;
    if (scratch_bytes > std::numeric_limits<std::size_t>::max() - queue_bytes ||
        scratch_bytes + queue_bytes > memory_budget.maximum_bytes) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront transport exceeds its explicit aggregate device-memory budget."));
    }
    return scratch_bytes;
}

} // namespace

core::Result<CudaWavefrontTransportBatch>
trace_cuda_wavefront_transport(const CudaSceneSoA& scene, const CudaSceneBvh& bvh,
                               const std::span<const CudaWavefrontPathInput> inputs,
                               const CudaWavefrontLightSampler light_sampler,
                               const CudaWavefrontTransportOptions options) try {
    if (auto status = cuda_scene_query_detail::validate_binding(scene, bvh, "wavefront transport");
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    if (auto status = validate_options_and_inputs(inputs, options); !status) {
        return std::unexpected(std::move(status.error()));
    }
    const auto validated_light_sampler = validate_light_sampler(scene.header(), light_sampler);
    if (!validated_light_sampler) {
        return std::unexpected(validated_light_sampler.error());
    }
    const auto config = transport_config(options, validated_light_sampler->light_count,
                                         validated_light_sampler->strategy);

    const auto path_count = inputs.size();
    const auto device_path_count = static_cast<std::uint32_t>(path_count);
    const auto required_scratch = scratch_capacity(path_count, options.device_memory_budget);
    if (!required_scratch) {
        return std::unexpected(required_scratch.error());
    }
    auto scratch_result =
        xpu::cuda::DeviceScratchBuffer::create(*required_scratch, options.device_memory_budget);
    if (!scratch_result) {
        return std::unexpected(std::move(scratch_result.error()));
    }
    auto scratch = std::move(*scratch_result);
    auto queues_result = CudaWavefrontQueues::create(
        path_count,
        CudaWavefrontQueueCreateOptions{.device_memory_budget = options.device_memory_budget});
    if (!queues_result) {
        return std::unexpected(std::move(queues_result.error()));
    }
    auto queues = std::move(*queues_result);

    auto allocate_column = [&]<typename Value>(const std::size_t count) -> core::Result<Value*> {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(Value)) {
            return std::unexpected(
                transport_error(core::StatusCode::resource_exhausted,
                                "CUDA wavefront scratch column byte count overflowed."));
        }
        auto slice = scratch.allocate(count * sizeof(Value), alignof(Value));
        if (!slice) {
            return std::unexpected(std::move(slice.error()));
        }
        return static_cast<Value*>(slice->data);
    };

    auto input_samples_result = allocate_column.template operator()<SampleStreamIndex>(path_count);
    auto input_rays_result = allocate_column.template operator()<TransportRay>(path_count);
    auto input_states_result =
        allocate_column.template operator()<TransportPathStateLane>(path_count);
    auto stream_samples_result = allocate_column.template operator()<SampleStreamIndex>(path_count);
    auto stream_rays_result = allocate_column.template operator()<TransportRay>(path_count);
    auto stream_states_result =
        allocate_column.template operator()<TransportPathStateLane>(path_count);
    auto hits_result = allocate_column.template operator()<ClosestHit>(path_count);
    auto pending_shadows_result =
        allocate_column.template operator()<WavefrontPendingShadow>(path_count);
    auto previous_bsdf_samples_result =
        allocate_column.template operator()<WavefrontPreviousBsdfSample>(path_count);
    auto controls_result = allocate_column.template operator()<WavefrontLaneControl>(path_count);
    auto compact_slots_result = allocate_column.template operator()<PathSlot>(path_count);
    auto compact_rays_result = allocate_column.template operator()<TransportRay>(path_count);
    auto closest_results_result =
        allocate_column.template operator()<SceneClosestHitResult>(path_count);
    auto occlusion_results_result =
        allocate_column.template operator()<SceneOcclusionResult>(path_count);
    auto outcomes_result = allocate_column.template operator()<WavefrontStageOutcome>(path_count);
    auto clear_status_result = allocate_column.template operator()<std::uint32_t>(1U);
    const auto allocation_error = [&]() -> const core::Error* {
        if (!input_samples_result)
            return &input_samples_result.error();
        if (!input_rays_result)
            return &input_rays_result.error();
        if (!input_states_result)
            return &input_states_result.error();
        if (!stream_samples_result)
            return &stream_samples_result.error();
        if (!stream_rays_result)
            return &stream_rays_result.error();
        if (!stream_states_result)
            return &stream_states_result.error();
        if (!hits_result)
            return &hits_result.error();
        if (!pending_shadows_result)
            return &pending_shadows_result.error();
        if (!previous_bsdf_samples_result)
            return &previous_bsdf_samples_result.error();
        if (!controls_result)
            return &controls_result.error();
        if (!compact_slots_result)
            return &compact_slots_result.error();
        if (!compact_rays_result)
            return &compact_rays_result.error();
        if (!closest_results_result)
            return &closest_results_result.error();
        if (!occlusion_results_result)
            return &occlusion_results_result.error();
        if (!outcomes_result)
            return &outcomes_result.error();
        if (!clear_status_result)
            return &clear_status_result.error();
        return nullptr;
    }();
    if (allocation_error != nullptr) {
        return std::unexpected(*allocation_error);
    }

    auto* const input_samples = *input_samples_result;
    auto* const input_rays = *input_rays_result;
    auto* const input_states = *input_states_result;
    auto* const stream_samples = *stream_samples_result;
    auto* const stream_rays = *stream_rays_result;
    auto* const stream_states = *stream_states_result;
    auto* const hits = *hits_result;
    auto* const pending_shadows = *pending_shadows_result;
    auto* const previous_bsdf_samples = *previous_bsdf_samples_result;
    auto* const controls = *controls_result;
    auto* const compact_slots = *compact_slots_result;
    auto* const compact_rays = *compact_rays_result;
    auto* const closest_results = *closest_results_result;
    auto* const occlusion_results = *occlusion_results_result;
    auto* const outcomes = *outcomes_result;
    auto* const clear_status = *clear_status_result;

    auto host_samples = std::vector<SampleStreamIndex>{};
    auto host_rays = std::vector<TransportRay>{};
    auto host_states = std::vector<TransportPathStateLane>{};
    host_samples.reserve(path_count);
    host_rays.reserve(path_count);
    host_states.reserve(path_count);
    for (auto index = std::size_t{}; index < path_count; ++index) {
        host_samples.push_back(transport_sample_stream_index(inputs[index].sample));
        auto converted_ray = cuda_scene_query_detail::transport_ray(inputs[index].primary_ray,
                                                                    index, "wavefront transport");
        if (!converted_ray) {
            return std::unexpected(std::move(converted_ray.error()));
        }
        host_rays.push_back(*converted_ray);
        host_states.push_back(transport_path_state(inputs[index].initial_state));
    }

    auto cuda_status = cudaMemset(scratch.data(), 0, scratch.capacity_bytes());
    if (cuda_status != cudaSuccess) {
        return std::unexpected(
            cuda_transport_error(cuda_status, "scratch initialization", scratch.capacity_bytes()));
    }
    const auto sample_bytes = path_count * sizeof(SampleStreamIndex);
    const auto ray_bytes = path_count * sizeof(TransportRay);
    const auto state_bytes = path_count * sizeof(TransportPathStateLane);
    cuda_status =
        cudaMemcpy(input_samples, host_samples.data(), sample_bytes, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        return std::unexpected(cuda_transport_error(cuda_status, "sample upload", sample_bytes));
    }
    cuda_status = cudaMemcpy(input_rays, host_rays.data(), ray_bytes, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        return std::unexpected(cuda_transport_error(cuda_status, "ray upload", ray_bytes));
    }
    cuda_status = cudaMemcpy(input_states, host_states.data(), state_bytes, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        return std::unexpected(cuda_transport_error(cuda_status, "path-state upload", state_bytes));
    }

    const auto queue_view = queues.device_view();
    const auto stream_view = WavefrontStageDeviceSoa{
        .sample_streams = stream_samples,
        .rays = stream_rays,
        .path_states = stream_states,
        .hits = hits,
        .pending_shadows = pending_shadows,
        .previous_bsdf_samples = previous_bsdf_samples,
        .controls = controls,
        .capacity = device_path_count,
        .reserved = 0U,
    };
    const auto camera_inputs = WavefrontCameraInputDeviceSoa{
        .sample_streams = input_samples,
        .rays = input_rays,
        .path_states = input_states,
        .count = device_path_count,
        .reserved = 0U,
    };
    auto host_outcomes = std::vector<WavefrontStageOutcome>{};
    auto report = CudaWavefrontTransportReport{
        .schema_version = CurrentCudaWavefrontTransportReportSchemaVersion,
        .has_light_sampler = validated_light_sampler->present,
        .registered_light_count = validated_light_sampler->light_count,
        .heuristic = options.heuristic,
        .light_sampling_strategy = validated_light_sampler->strategy,
        .depth_limits = options.depth_limits,
        .roulette_policy = options.roulette_policy,
        .path_count = path_count,
    };

    if (auto status =
            execute_stage("camera seed", device_path_count, route_mask(WavefrontStageRoute::none),
                          device_path_count, outcomes, host_outcomes,
                          [&] {
                              return blackframe_cuda_launch_wavefront_seed_camera(
                                  queue_view, stream_view, 0U, device_path_count, outcomes);
                          });
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    const auto camera_header =
        queue_header(queue_view, renderer::WavefrontQueueKind::camera, device_path_count);
    if (!camera_header || camera_header->size != device_path_count) {
        return std::unexpected(camera_header
                                   ? transport_error(core::StatusCode::internal_error,
                                                     "CUDA wavefront camera seed lost path lanes.")
                                   : camera_header.error());
    }
    if (auto status = execute_stage(
            "camera", device_path_count, route_mask(WavefrontStageRoute::ray), device_path_count,
            outcomes, host_outcomes,
            [&] {
                return blackframe_cuda_launch_wavefront_camera_stage(
                    queue_view, camera_inputs, stream_view, device_path_count, outcomes);
            });
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    report.stage_lanes.camera = device_path_count;
    if (auto status = clear_queue(queue_view, renderer::WavefrontQueueKind::camera, clear_status);
        !status) {
        return std::unexpected(std::move(status.error()));
    }

    auto ray_header =
        queue_header(queue_view, renderer::WavefrontQueueKind::ray, device_path_count);
    if (!ray_header) {
        return std::unexpected(ray_header.error());
    }
    auto iteration = std::uint64_t{};
    const auto maximum_iterations = static_cast<std::uint64_t>(options.depth_limits.diffuse) + 2U;
    while (ray_header->size != 0U) {
        if (iteration++ >= maximum_iterations) {
            return std::unexpected(transport_error(
                core::StatusCode::internal_error,
                "CUDA wavefront transport exceeded its depth-derived dispatch bound."));
        }
        const auto ray_count = ray_header->size;
        report.stage_lanes.intersection += ray_count;
        if (auto status = execute_stage(
                "intersection gather", ray_count, route_mask(WavefrontStageRoute::ray),
                device_path_count, outcomes, host_outcomes,
                [&] {
                    return blackframe_cuda_launch_wavefront_gather_rays(
                        queue_view, stream_view, ray_count, compact_slots, compact_rays, outcomes);
                });
            !status) {
            return std::unexpected(std::move(status.error()));
        }
        cuda_status = static_cast<cudaError_t>(blackframe_cuda_launch_scene_closest_hit(
            scene.device_data(), scene.size_bytes(), bvh.device_data(), bvh.size_bytes(),
            compact_rays, ray_count, closest_results));
        if (cuda_status != cudaSuccess) {
            return std::unexpected(cuda_transport_error(cuda_status, "closest-hit kernel launch"));
        }
        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            return std::unexpected(
                cuda_transport_error(cuda_status, "closest-hit synchronization"));
        }
        if (auto status = execute_stage(
                "intersection classify", ray_count,
                route_mask(WavefrontStageRoute::hit) | route_mask(WavefrontStageRoute::miss),
                device_path_count, outcomes, host_outcomes,
                [&] {
                    return blackframe_cuda_launch_wavefront_classify_closest_hit(
                        queue_view, stream_view, compact_slots, closest_results, ray_count,
                        outcomes);
                });
            !status) {
            return std::unexpected(std::move(status.error()));
        }
        if (auto status = clear_queue(queue_view, renderer::WavefrontQueueKind::ray, clear_status);
            !status) {
            return std::unexpected(std::move(status.error()));
        }

        auto miss_header =
            queue_header(queue_view, renderer::WavefrontQueueKind::miss, device_path_count);
        auto hit_header =
            queue_header(queue_view, renderer::WavefrontQueueKind::hit, device_path_count);
        if (!miss_header)
            return std::unexpected(miss_header.error());
        if (!hit_header)
            return std::unexpected(hit_header.error());
        if (static_cast<std::uint64_t>(miss_header->size) + hit_header->size != ray_count) {
            return std::unexpected(transport_error(
                core::StatusCode::internal_error,
                "CUDA closest-hit classification did not partition every ray exactly once."));
        }
        report.stage_lanes.miss += miss_header->size;
        report.stage_lanes.hit += hit_header->size;

        if (miss_header->size != 0U) {
            if (auto status = execute_stage(
                    "miss", miss_header->size, route_mask(WavefrontStageRoute::terminated),
                    device_path_count, outcomes, host_outcomes,
                    [&] {
                        return blackframe_cuda_launch_wavefront_miss_stage(
                            scene.device_data(), scene.size_bytes(), queue_view, stream_view,
                            miss_header->size, outcomes);
                    });
                !status) {
                return std::unexpected(std::move(status.error()));
            }
            if (auto status =
                    clear_queue(queue_view, renderer::WavefrontQueueKind::miss, clear_status);
                !status) {
                return std::unexpected(std::move(status.error()));
            }
        }

        if (hit_header->size != 0U) {
            if (auto status =
                    execute_stage("hit", hit_header->size, route_mask(WavefrontStageRoute::shade),
                                  device_path_count, outcomes, host_outcomes,
                                  [&] {
                                      return blackframe_cuda_launch_wavefront_hit_stage(
                                          queue_view, stream_view, hit_header->size, outcomes);
                                  });
                !status) {
                return std::unexpected(std::move(status.error()));
            }
            if (auto status =
                    clear_queue(queue_view, renderer::WavefrontQueueKind::hit, clear_status);
                !status) {
                return std::unexpected(std::move(status.error()));
            }
        }

        auto shade_header =
            queue_header(queue_view, renderer::WavefrontQueueKind::shade, device_path_count);
        if (!shade_header)
            return std::unexpected(shade_header.error());
        if (shade_header->size != hit_header->size) {
            return std::unexpected(
                transport_error(core::StatusCode::internal_error,
                                "CUDA hit stage did not route every surface lane to shading."));
        }
        report.stage_lanes.shade += shade_header->size;
        if (shade_header->size != 0U) {
            if (auto status = execute_stage("shade", shade_header->size,
                                            route_mask(WavefrontStageRoute::shadow) |
                                                route_mask(WavefrontStageRoute::continuation) |
                                                route_mask(WavefrontStageRoute::terminated),
                                            device_path_count, outcomes, host_outcomes,
                                            [&] {
                                                return blackframe_cuda_launch_wavefront_shade_stage(
                                                    scene.device_data(), scene.size_bytes(),
                                                    queue_view, stream_view, config,
                                                    shade_header->size, outcomes);
                                            });
                !status) {
                return std::unexpected(std::move(status.error()));
            }
            report.closure_samples += static_cast<std::uint64_t>(
                std::ranges::count_if(host_outcomes, [](const WavefrontStageOutcome outcome) {
                    return (outcome.detail & xpu::cuda::WavefrontShadeDetailClosureSampled) != 0U;
                }));
            report.light_samples += static_cast<std::uint64_t>(
                std::ranges::count_if(host_outcomes, [](const WavefrontStageOutcome outcome) {
                    return (outcome.detail & xpu::cuda::WavefrontShadeDetailLightSampled) != 0U;
                }));
            if (auto status =
                    clear_queue(queue_view, renderer::WavefrontQueueKind::shade, clear_status);
                !status) {
                return std::unexpected(std::move(status.error()));
            }
        }

        auto shadow_header =
            queue_header(queue_view, renderer::WavefrontQueueKind::shadow, device_path_count);
        if (!shadow_header)
            return std::unexpected(shadow_header.error());
        report.stage_lanes.shadow += shadow_header->size;
        report.shadow_queries += shadow_header->size;
        if (shadow_header->size != 0U) {
            if (auto status = execute_stage(
                    "shadow gather", shadow_header->size, route_mask(WavefrontStageRoute::shadow),
                    device_path_count, outcomes, host_outcomes,
                    [&] {
                        return blackframe_cuda_launch_wavefront_gather_shadow_rays(
                            queue_view, stream_view, shadow_header->size, compact_slots,
                            compact_rays, outcomes);
                    });
                !status) {
                return std::unexpected(std::move(status.error()));
            }
            cuda_status = static_cast<cudaError_t>(blackframe_cuda_launch_scene_occlusion(
                scene.device_data(), scene.size_bytes(), bvh.device_data(), bvh.size_bytes(),
                compact_rays, shadow_header->size, occlusion_results));
            if (cuda_status != cudaSuccess) {
                return std::unexpected(
                    cuda_transport_error(cuda_status, "shadow any-hit kernel launch"));
            }
            cuda_status = cudaDeviceSynchronize();
            if (cuda_status != cudaSuccess) {
                return std::unexpected(
                    cuda_transport_error(cuda_status, "shadow any-hit synchronization"));
            }
            if (auto status =
                    execute_stage("shadow", shadow_header->size,
                                  route_mask(WavefrontStageRoute::continuation) |
                                      route_mask(WavefrontStageRoute::terminated),
                                  device_path_count, outcomes, host_outcomes,
                                  [&] {
                                      return blackframe_cuda_launch_wavefront_process_shadow(
                                          queue_view, stream_view, compact_slots, occlusion_results,
                                          shadow_header->size, outcomes);
                                  });
                !status) {
                return std::unexpected(std::move(status.error()));
            }
            if (auto status =
                    clear_queue(queue_view, renderer::WavefrontQueueKind::shadow, clear_status);
                !status) {
                return std::unexpected(std::move(status.error()));
            }
        }

        auto continuation_header =
            queue_header(queue_view, renderer::WavefrontQueueKind::continuation, device_path_count);
        if (!continuation_header)
            return std::unexpected(continuation_header.error());
        report.stage_lanes.continuation += continuation_header->size;
        if (continuation_header->size != 0U) {
            if (auto status = execute_stage(
                    "continuation", continuation_header->size, route_mask(WavefrontStageRoute::ray),
                    device_path_count, outcomes, host_outcomes,
                    [&] {
                        return blackframe_cuda_launch_wavefront_continuation_stage(
                            queue_view, stream_view, continuation_header->size, outcomes);
                    });
                !status) {
                return std::unexpected(std::move(status.error()));
            }
            if (auto status = clear_queue(queue_view, renderer::WavefrontQueueKind::continuation,
                                          clear_status);
                !status) {
                return std::unexpected(std::move(status.error()));
            }
        }
        ray_header = queue_header(queue_view, renderer::WavefrontQueueKind::ray, device_path_count);
        if (!ray_header)
            return std::unexpected(ray_header.error());
    }

    auto final_states = std::vector<TransportPathStateLane>(path_count);
    auto final_rays = std::vector<TransportRay>(path_count);
    auto final_controls = std::vector<WavefrontLaneControl>(path_count);
    cuda_status =
        cudaMemcpy(final_states.data(), stream_states, state_bytes, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        return std::unexpected(
            cuda_transport_error(cuda_status, "final state download", state_bytes));
    }
    cuda_status = cudaMemcpy(final_rays.data(), stream_rays, ray_bytes, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        return std::unexpected(
            cuda_transport_error(cuda_status, "terminal ray download", ray_bytes));
    }
    const auto control_bytes = path_count * sizeof(WavefrontLaneControl);
    cuda_status =
        cudaMemcpy(final_controls.data(), controls, control_bytes, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        return std::unexpected(
            cuda_transport_error(cuda_status, "lane-control download", control_bytes));
    }

    auto paths = std::vector<CudaWavefrontPathResult>{};
    paths.reserve(path_count);
    for (auto index = std::size_t{}; index < path_count; ++index) {
        auto state = renderer_path_state(final_states[index], index);
        auto ray = renderer_ray(final_rays[index], index);
        auto termination = renderer_termination(final_controls[index], index);
        if (!state)
            return std::unexpected(std::move(state.error()));
        if (!ray)
            return std::unexpected(std::move(ray.error()));
        if (!termination)
            return std::unexpected(std::move(termination.error()));
        if (state->current_medium() != renderer::VacuumMedium ||
            ray->current_medium() != state->current_medium() ||
            state->wavelengths() != inputs[index].initial_state.wavelengths()) {
            return std::unexpected(transport_error(
                core::StatusCode::internal_error,
                "CUDA wavefront path " + std::to_string(index) +
                    " returned terminal transport identities inconsistent with its input."));
        }
        paths.push_back(CudaWavefrontPathResult{
            .state = *state,
            .terminal_ray = *ray,
            .termination = termination->reason,
            .blocked_depth_limits = termination->blocked_depth_limits,
        });
    }
    report.terminated_paths = path_count;

    auto queue_close = queues.close();
    auto scratch_close = scratch.close();
    if (!queue_close) {
        return std::unexpected(std::move(queue_close.error()));
    }
    if (!scratch_close) {
        return std::unexpected(std::move(scratch_close.error()));
    }
    return CudaWavefrontTransportBatch{.paths = std::move(paths), .report = report};
} catch (const std::bad_alloc&) {
    return std::unexpected(transport_error(core::StatusCode::resource_exhausted,
                                           "CUDA wavefront transport exhausted host memory."));
} catch (const std::length_error&) {
    return std::unexpected(
        transport_error(core::StatusCode::resource_exhausted,
                        "CUDA wavefront transport exceeded a host container length limit."));
}

} // namespace blackframe::engine
