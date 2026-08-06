#include "SceneQuery.hpp"

#include <Blackframe/Backends/GPU/CUDA/WavefrontQueues.hpp>
#include <Blackframe/Backends/GPU/CUDA/WavefrontTransport.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <Blackframe/XPU/CUDA/SceneClosestHit.hpp>
#include <Blackframe/XPU/CUDA/SceneOcclusion.hpp>
#include <Blackframe/XPU/CUDA/WavefrontQueueCompaction.hpp>
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
#include <memory>
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
using xpu::cuda::WavefrontQueueCompactionResult;
using xpu::cuda::WavefrontQueueCompactionStatus;
using xpu::cuda::WavefrontStageAudit;
using xpu::cuda::WavefrontStageDeviceSoa;
using xpu::cuda::WavefrontStageKind;
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
static_assert(static_cast<std::uint32_t>(WavefrontStageRoute::ray) ==
              static_cast<std::uint32_t>(renderer::WavefrontQueueKind::ray));
static_assert(static_cast<std::uint32_t>(WavefrontStageRoute::hit) ==
              static_cast<std::uint32_t>(renderer::WavefrontQueueKind::hit));
static_assert(static_cast<std::uint32_t>(WavefrontStageRoute::miss) ==
              static_cast<std::uint32_t>(renderer::WavefrontQueueKind::miss));
static_assert(static_cast<std::uint32_t>(WavefrontStageRoute::shade) ==
              static_cast<std::uint32_t>(renderer::WavefrontQueueKind::shade));
static_assert(static_cast<std::uint32_t>(WavefrontStageRoute::shadow) ==
              static_cast<std::uint32_t>(renderer::WavefrontQueueKind::shadow));
static_assert(static_cast<std::uint32_t>(WavefrontStageRoute::continuation) ==
              static_cast<std::uint32_t>(renderer::WavefrontQueueKind::continuation));
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
    if ((termination == WavefrontTermination::depth_limit &&
         blocked == renderer::ScatteringLobe::none) ||
        (termination != WavefrontTermination::depth_limit &&
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
    case WavefrontTermination::depth_limit:
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
[[nodiscard]] core::Result<WavefrontStageAudit>
execute_stage(const WavefrontStageKind stage_kind, const std::string_view stage_name,
              const std::uint32_t work_count, const StageRouteMask allowed_routes,
              const std::uint32_t path_capacity, WavefrontStageOutcome* const device_outcomes,
              WavefrontStageAudit* const device_audit, Launcher&& launcher) {
    if (work_count == 0U) {
        return WavefrontStageAudit{
            .abi_major = xpu::cuda::WavefrontStageAuditAbiMajor,
            .abi_minor = xpu::cuda::WavefrontStageAuditAbiMinor,
            .struct_size = sizeof(WavefrontStageAudit),
            .stage_kind = static_cast<std::uint32_t>(stage_kind),
            .expected_work_count = 0U,
            .inspected_work_count = 0U,
            .first_failure_work_index = std::numeric_limits<std::uint32_t>::max(),
        };
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
    cuda_status = static_cast<cudaError_t>(blackframe_cuda_launch_wavefront_audit_stage(
        device_outcomes, work_count, allowed_routes, path_capacity,
        static_cast<std::uint32_t>(stage_kind), device_audit));
    if (cuda_status != cudaSuccess) {
        return std::unexpected(cuda_transport_error(
            cuda_status, std::string{stage_name} + " outcome-audit kernel launch"));
    }
    auto audit = WavefrontStageAudit{};
    cuda_status = cudaMemcpy(&audit, device_audit, sizeof(audit), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        return std::unexpected(
            cuda_transport_error(cuda_status, "outcome-audit download", sizeof(audit)));
    }
    const auto expected_stage_kind = static_cast<std::uint32_t>(stage_kind);
    const auto no_failure = std::numeric_limits<std::uint32_t>::max();
    const auto canonical_empty_failure =
        audit.first_failure.status == 0U && audit.first_failure.route == 0U &&
        audit.first_failure.path_slot == 0U && audit.first_failure.detail == 0U;
    const auto counts_are_canonical = audit.closure_samples <= work_count &&
                                      audit.light_samples <= work_count &&
                                      (stage_kind == WavefrontStageKind::shade ||
                                       (audit.closure_samples == 0U && audit.light_samples == 0U));
    if (audit.abi_major != xpu::cuda::WavefrontStageAuditAbiMajor ||
        audit.abi_minor != xpu::cuda::WavefrontStageAuditAbiMinor ||
        audit.struct_size != sizeof(WavefrontStageAudit) ||
        audit.stage_kind != expected_stage_kind || audit.expected_work_count != work_count ||
        audit.inspected_work_count != work_count || !counts_are_canonical ||
        audit.reserved[0U] != 0U || audit.reserved[1U] != 0U || audit.reserved[2U] != 0U ||
        audit.reserved[3U] != 0U ||
        (audit.first_failure_work_index == no_failure && !canonical_empty_failure) ||
        (audit.first_failure_work_index != no_failure &&
         audit.first_failure_work_index >= work_count)) {
        return std::unexpected(transport_error(core::StatusCode::internal_error,
                                               "CUDA wavefront " + std::string{stage_name} +
                                                   " returned a non-canonical device audit."));
    }
    if (audit.first_failure_work_index != no_failure) {
        if (auto status =
                stage_failure(audit.first_failure, stage_name, audit.first_failure_work_index,
                              path_capacity, allowed_routes);
            !status) {
            return std::unexpected(std::move(status.error()));
        }
        return std::unexpected(transport_error(core::StatusCode::internal_error,
                                               "CUDA wavefront " + std::string{stage_name} +
                                                   " audit reported a non-failing outcome."));
    }
    return audit;
}

[[nodiscard]] core::Result<WavefrontQueueCompactionResult>
compact_route(const xpu::cuda::WavefrontQueueDeviceSoa queues,
              const WavefrontStageOutcome* const device_outcomes, const std::uint32_t work_count,
              const WavefrontStageRoute route, void* const device_scratch,
              const std::size_t scratch_bytes,
              WavefrontQueueCompactionResult* const device_result) {
    const auto route_value = static_cast<std::uint32_t>(route);
    auto cuda_status = static_cast<cudaError_t>(xpu::cuda::launch_wavefront_queue_compaction(
        queues, device_outcomes, work_count, route_value, device_scratch, scratch_bytes,
        device_result));
    if (cuda_status != cudaSuccess) {
        return std::unexpected(cuda_transport_error(cuda_status, "queue-compaction kernel launch"));
    }

    auto result = WavefrontQueueCompactionResult{};
    cuda_status = cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        return std::unexpected(
            cuda_transport_error(cuda_status, "queue-compaction result download", sizeof(result)));
    }

    const auto status = static_cast<WavefrontQueueCompactionStatus>(result.status);
    const auto reserved_is_zero = result.reserved[0U] == 0U && result.reserved[1U] == 0U &&
                                  result.reserved[2U] == 0U && result.reserved[3U] == 0U;
    const auto identity_is_canonical = result.queue_kind == route_value &&
                                       result.route == route_value &&
                                       result.input_count == work_count;
    const auto counts_are_bounded = result.initial_size <= queues.slot_stride &&
                                    result.selected_count <= work_count &&
                                    result.published_count <= result.selected_count &&
                                    result.rejected_count <= result.selected_count;
    if (!reserved_is_zero || !identity_is_canonical || !counts_are_bounded) {
        return std::unexpected(transport_error(
            core::StatusCode::incompatible,
            "CUDA wavefront queue compaction returned a non-canonical device result."));
    }

    switch (status) {
    case WavefrontQueueCompactionStatus::success:
        if (result.published_count != result.selected_count || result.rejected_count != 0U ||
            result.published_count > queues.slot_stride - result.initial_size) {
            return std::unexpected(transport_error(
                core::StatusCode::incompatible,
                "CUDA wavefront queue compaction returned inconsistent success counters."));
        }
        return result;
    case WavefrontQueueCompactionStatus::capacity_exhausted:
        if (result.selected_count == 0U || result.published_count != 0U ||
            result.rejected_count != result.selected_count ||
            result.selected_count <= queues.slot_stride - result.initial_size) {
            return std::unexpected(transport_error(
                core::StatusCode::incompatible,
                "CUDA wavefront queue compaction returned inconsistent overflow counters."));
        }
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront queue compaction rejected a complete append that exceeded its fixed "
            "capacity."));
    case WavefrontQueueCompactionStatus::invalid_contract:
        return std::unexpected(
            transport_error(core::StatusCode::incompatible,
                            "CUDA wavefront queue compaction rejected its device contract."));
    }
    return std::unexpected(
        transport_error(core::StatusCode::incompatible,
                        "CUDA wavefront queue compaction returned an unknown status."));
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

struct CudaWavefrontWorkspaceColumns final {
    SampleStreamIndex* input_samples{};
    TransportRay* input_rays{};
    TransportPathStateLane* input_states{};
    SampleStreamIndex* stream_samples{};
    TransportRay* stream_rays{};
    TransportPathStateLane* stream_states{};
    ClosestHit* hits{};
    WavefrontPendingShadow* pending_shadows{};
    WavefrontPreviousBsdfSample* previous_bsdf_samples{};
    WavefrontLaneControl* controls{};
    PathSlot* compact_slots{};
    TransportRay* compact_rays{};
    SceneClosestHitResult* closest_results{};
    SceneOcclusionResult* occlusion_results{};
    WavefrontStageOutcome* outcomes{};
    WavefrontStageAudit* stage_audit{};
    WavefrontQueueCompactionResult* compaction_result{};
    void* compaction_scratch{};
    std::size_t compaction_scratch_bytes{};
    std::uint32_t* clear_status{};
};

[[nodiscard]] core::Result<CudaWavefrontWorkspaceColumns>
allocate_workspace_columns(xpu::cuda::DeviceScratchBuffer& scratch, const std::size_t capacity,
                           const std::size_t compaction_scratch_bytes) {
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

    auto input_samples = allocate_column.template operator()<SampleStreamIndex>(capacity);
    auto input_rays = allocate_column.template operator()<TransportRay>(capacity);
    auto input_states = allocate_column.template operator()<TransportPathStateLane>(capacity);
    auto stream_samples = allocate_column.template operator()<SampleStreamIndex>(capacity);
    auto stream_rays = allocate_column.template operator()<TransportRay>(capacity);
    auto stream_states = allocate_column.template operator()<TransportPathStateLane>(capacity);
    auto hits = allocate_column.template operator()<ClosestHit>(capacity);
    auto pending_shadows = allocate_column.template operator()<WavefrontPendingShadow>(capacity);
    auto previous_bsdf_samples =
        allocate_column.template operator()<WavefrontPreviousBsdfSample>(capacity);
    auto controls = allocate_column.template operator()<WavefrontLaneControl>(capacity);
    auto compact_slots = allocate_column.template operator()<PathSlot>(capacity);
    auto compact_rays = allocate_column.template operator()<TransportRay>(capacity);
    auto closest_results = allocate_column.template operator()<SceneClosestHitResult>(capacity);
    auto occlusion_results = allocate_column.template operator()<SceneOcclusionResult>(capacity);
    auto outcomes = allocate_column.template operator()<WavefrontStageOutcome>(capacity);
    auto stage_audit = allocate_column.template operator()<WavefrontStageAudit>(1U);
    auto compaction_result =
        allocate_column.template operator()<WavefrontQueueCompactionResult>(1U);
    auto compaction_scratch = scratch.allocate(compaction_scratch_bytes,
                                               xpu::cuda::WavefrontQueueCompactionScratchAlignment);
    auto clear_status = allocate_column.template operator()<std::uint32_t>(1U);
    const auto allocation_error = [&]() -> const core::Error* {
        if (!input_samples)
            return &input_samples.error();
        if (!input_rays)
            return &input_rays.error();
        if (!input_states)
            return &input_states.error();
        if (!stream_samples)
            return &stream_samples.error();
        if (!stream_rays)
            return &stream_rays.error();
        if (!stream_states)
            return &stream_states.error();
        if (!hits)
            return &hits.error();
        if (!pending_shadows)
            return &pending_shadows.error();
        if (!previous_bsdf_samples)
            return &previous_bsdf_samples.error();
        if (!controls)
            return &controls.error();
        if (!compact_slots)
            return &compact_slots.error();
        if (!compact_rays)
            return &compact_rays.error();
        if (!closest_results)
            return &closest_results.error();
        if (!occlusion_results)
            return &occlusion_results.error();
        if (!outcomes)
            return &outcomes.error();
        if (!stage_audit)
            return &stage_audit.error();
        if (!compaction_result)
            return &compaction_result.error();
        if (!compaction_scratch)
            return &compaction_scratch.error();
        if (!clear_status)
            return &clear_status.error();
        return nullptr;
    }();
    if (allocation_error != nullptr) {
        return std::unexpected(*allocation_error);
    }

    return CudaWavefrontWorkspaceColumns{
        .input_samples = *input_samples,
        .input_rays = *input_rays,
        .input_states = *input_states,
        .stream_samples = *stream_samples,
        .stream_rays = *stream_rays,
        .stream_states = *stream_states,
        .hits = *hits,
        .pending_shadows = *pending_shadows,
        .previous_bsdf_samples = *previous_bsdf_samples,
        .controls = *controls,
        .compact_slots = *compact_slots,
        .compact_rays = *compact_rays,
        .closest_results = *closest_results,
        .occlusion_results = *occlusion_results,
        .outcomes = *outcomes,
        .stage_audit = *stage_audit,
        .compaction_result = *compaction_result,
        .compaction_scratch = compaction_scratch->data,
        .compaction_scratch_bytes = compaction_scratch->size_bytes,
        .clear_status = *clear_status,
    };
}

struct CudaWavefrontScratchRequirement final {
    std::size_t total_bytes{};
    std::size_t compaction_bytes{};
};

[[nodiscard]] core::Result<CudaWavefrontScratchRequirement>
scratch_capacity(const std::size_t path_count, const xpu::cuda::DeviceMemoryBudget memory_budget) {
    auto compaction_bytes = std::size_t{};
    const auto query_status =
        static_cast<cudaError_t>(xpu::cuda::query_wavefront_queue_compaction_scratch_bytes(
            static_cast<std::uint32_t>(path_count), &compaction_bytes));
    if (query_status != cudaSuccess) {
        return std::unexpected(
            cuda_transport_error(query_status, "queue-compaction scratch query"));
    }
    if (path_count != 0U && compaction_bytes == 0U) {
        return std::unexpected(transport_error(
            core::StatusCode::internal_error,
            "CUDA wavefront queue compaction reported zero scratch for non-zero work."));
    }

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
        (alignof(WavefrontStageOutcome) - 1U) + (alignof(WavefrontStageAudit) - 1U) +
        (alignof(WavefrontQueueCompactionResult) - 1U) +
        (xpu::cuda::WavefrontQueueCompactionScratchAlignment - 1U) + (alignof(std::uint32_t) - 1U);
    constexpr auto fixed_bytes = sizeof(WavefrontStageAudit) +
                                 sizeof(WavefrontQueueCompactionResult) + sizeof(std::uint32_t);
    if (path_count >
        (std::numeric_limits<std::size_t>::max() - column_alignment_slack - fixed_bytes) /
            column_element_bytes) {
        return std::unexpected(
            transport_error(core::StatusCode::resource_exhausted,
                            "CUDA wavefront scratch byte count overflowed its host size domain."));
    }
    const auto column_bytes =
        path_count * column_element_bytes + column_alignment_slack + fixed_bytes;
    if (compaction_bytes > std::numeric_limits<std::size_t>::max() - column_bytes) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront compaction scratch byte count overflowed its host size domain."));
    }
    const auto scratch_bytes = column_bytes + compaction_bytes;
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
    return CudaWavefrontScratchRequirement{
        .total_bytes = scratch_bytes,
        .compaction_bytes = compaction_bytes,
    };
}

[[nodiscard]] constexpr std::size_t queue_capacity_bytes(const std::size_t capacity) noexcept {
    return static_cast<std::size_t>(xpu::cuda::CudaWavefrontQueueCount) * sizeof(QueueHeader) +
           capacity * static_cast<std::size_t>(xpu::cuda::CudaWavefrontQueueCount) *
               sizeof(PathSlot);
}

} // namespace

struct CudaWavefrontTransportWorkspace::Storage final {
    Storage(xpu::cuda::DeviceScratchBuffer scratch_storage, CudaWavefrontQueues queue_storage,
            CudaWavefrontWorkspaceColumns column_views, const std::size_t path_capacity,
            const std::size_t allocated_device_bytes)
        : scratch(std::move(scratch_storage)), queues(std::move(queue_storage)),
          columns(column_views), capacity(path_capacity), device_size_bytes(allocated_device_bytes),
          host_samples(path_capacity), host_rays(path_capacity), host_states(path_capacity),
          final_states(path_capacity), final_rays(path_capacity), final_controls(path_capacity) {}

    xpu::cuda::DeviceScratchBuffer scratch;
    CudaWavefrontQueues queues;
    CudaWavefrontWorkspaceColumns columns{};
    std::size_t capacity{};
    std::size_t device_size_bytes{};
    std::vector<SampleStreamIndex> host_samples;
    std::vector<TransportRay> host_rays;
    std::vector<TransportPathStateLane> host_states;
    std::vector<TransportPathStateLane> final_states;
    std::vector<TransportRay> final_rays;
    std::vector<WavefrontLaneControl> final_controls;
    bool queues_dirty{};
};

CudaWavefrontTransportWorkspace::CudaWavefrontTransportWorkspace(
    std::unique_ptr<Storage> storage) noexcept
    : storage_(std::move(storage)) {}

CudaWavefrontTransportWorkspace::CudaWavefrontTransportWorkspace(
    CudaWavefrontTransportWorkspace&& other) noexcept = default;

CudaWavefrontTransportWorkspace::~CudaWavefrontTransportWorkspace() noexcept = default;

core::Result<CudaWavefrontTransportWorkspace> CudaWavefrontTransportWorkspace::create(
    const std::size_t capacity, const xpu::cuda::DeviceMemoryBudget device_memory_budget) try {
    if (capacity == 0U) {
        return std::unexpected(
            transport_error(core::StatusCode::invalid_argument,
                            "CUDA wavefront transport workspace capacity must be non-zero."));
    }
    if (capacity > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront transport workspace capacity exceeds its 32-bit device domain."));
    }
    const auto required_scratch = scratch_capacity(capacity, device_memory_budget);
    if (!required_scratch) {
        return std::unexpected(required_scratch.error());
    }
    const auto required_queues = queue_capacity_bytes(capacity);
    if (required_scratch->total_bytes > std::numeric_limits<std::size_t>::max() - required_queues ||
        required_scratch->total_bytes + required_queues > device_memory_budget.maximum_bytes) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront workspace exceeds its aggregate device-memory budget."));
    }
    auto scratch =
        xpu::cuda::DeviceScratchBuffer::create(required_scratch->total_bytes, device_memory_budget);
    if (!scratch) {
        return std::unexpected(std::move(scratch.error()));
    }
    auto columns =
        allocate_workspace_columns(*scratch, capacity, required_scratch->compaction_bytes);
    if (!columns) {
        return std::unexpected(std::move(columns.error()));
    }
    auto queues = CudaWavefrontQueues::create(
        capacity, CudaWavefrontQueueCreateOptions{.device_memory_budget = device_memory_budget});
    if (!queues) {
        return std::unexpected(std::move(queues.error()));
    }
    if (scratch->device_ordinal() != queues->device_ordinal()) {
        return std::unexpected(transport_error(
            core::StatusCode::internal_error,
            "CUDA wavefront workspace allocations unexpectedly landed on different devices."));
    }
    const auto allocated_device_bytes = required_scratch->total_bytes + required_queues;
    auto storage = std::make_unique<Storage>(std::move(*scratch), std::move(*queues), *columns,
                                             capacity, allocated_device_bytes);
    return CudaWavefrontTransportWorkspace{std::move(storage)};
} catch (const std::bad_alloc&) {
    return std::unexpected(
        transport_error(core::StatusCode::resource_exhausted,
                        "CUDA wavefront transport workspace exhausted host memory."));
} catch (const std::length_error&) {
    return std::unexpected(transport_error(
        core::StatusCode::resource_exhausted,
        "CUDA wavefront transport workspace exceeded a host container length limit."));
}

std::size_t CudaWavefrontTransportWorkspace::capacity() const noexcept {
    return storage_ != nullptr ? storage_->capacity : 0U;
}

std::size_t CudaWavefrontTransportWorkspace::device_size_bytes() const noexcept {
    return storage_ != nullptr ? storage_->device_size_bytes : 0U;
}

std::int32_t CudaWavefrontTransportWorkspace::device_ordinal() const noexcept {
    return storage_ != nullptr ? storage_->scratch.device_ordinal() : -1;
}

CudaWavefrontTransportWorkspace::operator bool() const noexcept {
    return storage_ != nullptr && storage_->capacity != 0U && !storage_->scratch.empty() &&
           static_cast<bool>(storage_->queues) &&
           storage_->scratch.device_ordinal() == storage_->queues.device_ordinal();
}

core::Status CudaWavefrontTransportWorkspace::close() {
    if (storage_ == nullptr) {
        return {};
    }
    auto first_error = core::Status{};
    auto queue_status = storage_->queues.close();
    if (!queue_status) {
        first_error = std::unexpected(std::move(queue_status.error()));
    }
    auto scratch_status = storage_->scratch.close();
    if (!scratch_status && first_error) {
        first_error = std::unexpected(std::move(scratch_status.error()));
    }
    if (first_error) {
        storage_.reset();
    }
    return first_error;
}

core::Result<CudaWavefrontTransportBatch>
trace_cuda_wavefront_transport(CudaWavefrontTransportWorkspace& workspace,
                               const CudaSceneSoA& scene, const CudaSceneBvh& bvh,
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
    if (!workspace) {
        return std::unexpected(
            transport_error(core::StatusCode::invalid_argument,
                            "CUDA wavefront transport requires an open reusable workspace."));
    }
    auto& storage = *workspace.storage_;
    if (path_count > storage.capacity) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront transport path count exceeds the reusable workspace capacity."));
    }
    if (storage.device_size_bytes > options.device_memory_budget.maximum_bytes) {
        return std::unexpected(transport_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront transport workspace exceeds the requested device-memory budget."));
    }
    if (storage.scratch.device_ordinal() != scene.device_ordinal()) {
        return std::unexpected(transport_error(
            core::StatusCode::incompatible,
            "CUDA wavefront transport workspace and scene must reside on the same device."));
    }

    const auto device_capacity = static_cast<std::uint32_t>(storage.capacity);
    const auto& columns = storage.columns;
    auto* const stream_rays = columns.stream_rays;
    auto* const stream_states = columns.stream_states;
    auto* const controls = columns.controls;
    auto* const compact_slots = columns.compact_slots;
    auto* const compact_rays = columns.compact_rays;
    auto* const closest_results = columns.closest_results;
    auto* const occlusion_results = columns.occlusion_results;
    auto* const outcomes = columns.outcomes;
    auto* const stage_audit = columns.stage_audit;
    auto* const compaction_result = columns.compaction_result;
    auto* const compaction_scratch = columns.compaction_scratch;
    const auto compaction_scratch_bytes = columns.compaction_scratch_bytes;
    auto* const clear_status = columns.clear_status;
    for (auto index = std::size_t{}; index < path_count; ++index) {
        storage.host_samples[index] = transport_sample_stream_index(inputs[index].sample);
        auto converted_ray = cuda_scene_query_detail::transport_ray(inputs[index].primary_ray,
                                                                    index, "wavefront transport");
        if (!converted_ray) {
            return std::unexpected(std::move(converted_ray.error()));
        }
        storage.host_rays[index] = *converted_ray;
        storage.host_states[index] = transport_path_state(inputs[index].initial_state);
    }
    if (storage.queues_dirty) {
        // The preceding trace already returned the device-stage failure to the caller. Reset may
        // therefore acknowledge its queue diagnostics while still validating every counter before
        // the workspace is reused.
        auto reset_status =
            storage.queues.reset(CudaWavefrontQueueResetPolicy::acknowledge_overflow);
        if (!reset_status) {
            return std::unexpected(std::move(reset_status.error()));
        }
        storage.queues_dirty = false;
    }

    auto cuda_status = cudaMemset(storage.scratch.data(), 0, storage.scratch.capacity_bytes());
    if (cuda_status != cudaSuccess) {
        return std::unexpected(cuda_transport_error(cuda_status, "scratch initialization",
                                                    storage.scratch.capacity_bytes()));
    }
    const auto sample_bytes = path_count * sizeof(SampleStreamIndex);
    const auto ray_bytes = path_count * sizeof(TransportRay);
    const auto state_bytes = path_count * sizeof(TransportPathStateLane);
    cuda_status = cudaMemcpy(columns.input_samples, storage.host_samples.data(), sample_bytes,
                             cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        return std::unexpected(cuda_transport_error(cuda_status, "sample upload", sample_bytes));
    }
    cuda_status =
        cudaMemcpy(columns.input_rays, storage.host_rays.data(), ray_bytes, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        return std::unexpected(cuda_transport_error(cuda_status, "ray upload", ray_bytes));
    }
    cuda_status = cudaMemcpy(columns.input_states, storage.host_states.data(), state_bytes,
                             cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        return std::unexpected(cuda_transport_error(cuda_status, "path-state upload", state_bytes));
    }

    const auto queue_view = storage.queues.device_view();
    const auto stream_view = WavefrontStageDeviceSoa{
        .sample_streams = columns.stream_samples,
        .rays = columns.stream_rays,
        .path_states = columns.stream_states,
        .hits = columns.hits,
        .pending_shadows = columns.pending_shadows,
        .previous_bsdf_samples = columns.previous_bsdf_samples,
        .controls = columns.controls,
        .capacity = device_capacity,
        .reserved = 0U,
    };
    const auto camera_inputs = WavefrontCameraInputDeviceSoa{
        .sample_streams = columns.input_samples,
        .rays = columns.input_rays,
        .path_states = columns.input_states,
        .count = device_path_count,
        .reserved = 0U,
    };
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

    storage.queues_dirty = true;
    if (auto status = execute_stage(
            WavefrontStageKind::camera_seed, "camera seed", device_path_count,
            route_mask(WavefrontStageRoute::none), device_path_count, outcomes, stage_audit,
            [&] {
                return blackframe_cuda_launch_wavefront_seed_camera(queue_view, stream_view, 0U,
                                                                    device_path_count, outcomes);
            });
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    const auto camera_header =
        queue_header(queue_view, renderer::WavefrontQueueKind::camera, device_capacity);
    if (!camera_header || camera_header->size != device_path_count) {
        return std::unexpected(camera_header
                                   ? transport_error(core::StatusCode::internal_error,
                                                     "CUDA wavefront camera seed lost path lanes.")
                                   : camera_header.error());
    }
    if (auto status = execute_stage(
            WavefrontStageKind::camera, "camera", device_path_count,
            route_mask(WavefrontStageRoute::ray), device_path_count, outcomes, stage_audit,
            [&] {
                return blackframe_cuda_launch_wavefront_camera_stage(
                    queue_view, camera_inputs, stream_view, device_path_count, outcomes);
            });
        !status) {
        return std::unexpected(std::move(status.error()));
    }
    if (auto compacted =
            compact_route(queue_view, outcomes, device_path_count, WavefrontStageRoute::ray,
                          compaction_scratch, compaction_scratch_bytes, compaction_result);
        !compacted) {
        return std::unexpected(std::move(compacted.error()));
    }
    report.stage_lanes.camera = device_path_count;
    if (auto status = clear_queue(queue_view, renderer::WavefrontQueueKind::camera, clear_status);
        !status) {
        return std::unexpected(std::move(status.error()));
    }

    auto ray_header = queue_header(queue_view, renderer::WavefrontQueueKind::ray, device_capacity);
    if (!ray_header) {
        return std::unexpected(ray_header.error());
    }
    auto iteration = std::uint64_t{};
    // Transmission is an orthogonal counter attached to a surface-family event, so it does not
    // add another bounce to the dispatch bound. The four primary families do.
    const auto maximum_iterations = static_cast<std::uint64_t>(options.depth_limits.diffuse) +
                                    static_cast<std::uint64_t>(options.depth_limits.glossy) +
                                    static_cast<std::uint64_t>(options.depth_limits.specular) +
                                    static_cast<std::uint64_t>(options.depth_limits.volume) + 2U;
    while (ray_header->size != 0U) {
        if (iteration++ >= maximum_iterations) {
            return std::unexpected(transport_error(
                core::StatusCode::internal_error,
                "CUDA wavefront transport exceeded its depth-derived dispatch bound."));
        }
        const auto ray_count = ray_header->size;
        report.stage_lanes.intersection += ray_count;
        if (auto status = execute_stage(
                WavefrontStageKind::intersection_gather, "intersection gather", ray_count,
                route_mask(WavefrontStageRoute::ray), device_path_count, outcomes, stage_audit,
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
        // Classification uses the same stream. Its compact stage audit is the synchronization
        // boundary for both kernels, so an intermediate device-wide barrier is redundant.
        if (auto status = execute_stage(
                WavefrontStageKind::intersection_classify, "intersection classify", ray_count,
                route_mask(WavefrontStageRoute::hit) | route_mask(WavefrontStageRoute::miss),
                device_path_count, outcomes, stage_audit,
                [&] {
                    return blackframe_cuda_launch_wavefront_classify_closest_hit(
                        queue_view, stream_view, compact_slots, closest_results, ray_count,
                        outcomes);
                });
            !status) {
            return std::unexpected(std::move(status.error()));
        }
        if (auto compacted =
                compact_route(queue_view, outcomes, ray_count, WavefrontStageRoute::hit,
                              compaction_scratch, compaction_scratch_bytes, compaction_result);
            !compacted) {
            return std::unexpected(std::move(compacted.error()));
        }
        if (auto compacted =
                compact_route(queue_view, outcomes, ray_count, WavefrontStageRoute::miss,
                              compaction_scratch, compaction_scratch_bytes, compaction_result);
            !compacted) {
            return std::unexpected(std::move(compacted.error()));
        }
        if (auto status = clear_queue(queue_view, renderer::WavefrontQueueKind::ray, clear_status);
            !status) {
            return std::unexpected(std::move(status.error()));
        }

        auto miss_header =
            queue_header(queue_view, renderer::WavefrontQueueKind::miss, device_capacity);
        auto hit_header =
            queue_header(queue_view, renderer::WavefrontQueueKind::hit, device_capacity);
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
            if (auto status = execute_stage(WavefrontStageKind::miss, "miss", miss_header->size,
                                            route_mask(WavefrontStageRoute::terminated),
                                            device_path_count, outcomes, stage_audit,
                                            [&] {
                                                return blackframe_cuda_launch_wavefront_miss_stage(
                                                    scene.device_data(), scene.size_bytes(),
                                                    queue_view, stream_view, miss_header->size,
                                                    outcomes);
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
            if (auto status = execute_stage(WavefrontStageKind::hit, "hit", hit_header->size,
                                            route_mask(WavefrontStageRoute::shade),
                                            device_path_count, outcomes, stage_audit,
                                            [&] {
                                                return blackframe_cuda_launch_wavefront_hit_stage(
                                                    queue_view, stream_view, hit_header->size,
                                                    outcomes);
                                            });
                !status) {
                return std::unexpected(std::move(status.error()));
            }
            if (auto compacted = compact_route(queue_view, outcomes, hit_header->size,
                                               WavefrontStageRoute::shade, compaction_scratch,
                                               compaction_scratch_bytes, compaction_result);
                !compacted) {
                return std::unexpected(std::move(compacted.error()));
            }
            if (auto status =
                    clear_queue(queue_view, renderer::WavefrontQueueKind::hit, clear_status);
                !status) {
                return std::unexpected(std::move(status.error()));
            }
        }

        auto shade_header =
            queue_header(queue_view, renderer::WavefrontQueueKind::shade, device_capacity);
        if (!shade_header)
            return std::unexpected(shade_header.error());
        if (shade_header->size != hit_header->size) {
            return std::unexpected(
                transport_error(core::StatusCode::internal_error,
                                "CUDA hit stage did not route every surface lane to shading."));
        }
        report.stage_lanes.shade += shade_header->size;
        if (shade_header->size != 0U) {
            const auto audit =
                execute_stage(WavefrontStageKind::shade, "shade", shade_header->size,
                              route_mask(WavefrontStageRoute::shadow) |
                                  route_mask(WavefrontStageRoute::continuation) |
                                  route_mask(WavefrontStageRoute::terminated),
                              device_path_count, outcomes, stage_audit, [&] {
                                  return blackframe_cuda_launch_wavefront_shade_stage(
                                      scene.device_data(), scene.size_bytes(), queue_view,
                                      stream_view, config, shade_header->size, outcomes);
                              });
            if (!audit) {
                return std::unexpected(std::move(audit.error()));
            }
            report.closure_samples += audit->closure_samples;
            report.light_samples += audit->light_samples;
            if (auto compacted = compact_route(queue_view, outcomes, shade_header->size,
                                               WavefrontStageRoute::shadow, compaction_scratch,
                                               compaction_scratch_bytes, compaction_result);
                !compacted) {
                return std::unexpected(std::move(compacted.error()));
            }
            if (auto compacted = compact_route(
                    queue_view, outcomes, shade_header->size, WavefrontStageRoute::continuation,
                    compaction_scratch, compaction_scratch_bytes, compaction_result);
                !compacted) {
                return std::unexpected(std::move(compacted.error()));
            }
            if (auto status =
                    clear_queue(queue_view, renderer::WavefrontQueueKind::shade, clear_status);
                !status) {
                return std::unexpected(std::move(status.error()));
            }
        }

        auto shadow_header =
            queue_header(queue_view, renderer::WavefrontQueueKind::shadow, device_capacity);
        if (!shadow_header)
            return std::unexpected(shadow_header.error());
        report.stage_lanes.shadow += shadow_header->size;
        report.shadow_queries += shadow_header->size;
        if (shadow_header->size != 0U) {
            if (auto status =
                    execute_stage(WavefrontStageKind::shadow_gather, "shadow gather",
                                  shadow_header->size, route_mask(WavefrontStageRoute::shadow),
                                  device_path_count, outcomes, stage_audit,
                                  [&] {
                                      return blackframe_cuda_launch_wavefront_gather_shadow_rays(
                                          queue_view, stream_view, shadow_header->size,
                                          compact_slots, compact_rays, outcomes);
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
            // Shadow processing is ordered on the same stream and the following stage audit
            // provides the required completion/error boundary.
            if (auto status =
                    execute_stage(WavefrontStageKind::shadow_process, "shadow", shadow_header->size,
                                  route_mask(WavefrontStageRoute::continuation) |
                                      route_mask(WavefrontStageRoute::terminated),
                                  device_path_count, outcomes, stage_audit,
                                  [&] {
                                      return blackframe_cuda_launch_wavefront_process_shadow(
                                          queue_view, stream_view, compact_slots, occlusion_results,
                                          shadow_header->size, outcomes);
                                  });
                !status) {
                return std::unexpected(std::move(status.error()));
            }
            if (auto compacted = compact_route(
                    queue_view, outcomes, shadow_header->size, WavefrontStageRoute::continuation,
                    compaction_scratch, compaction_scratch_bytes, compaction_result);
                !compacted) {
                return std::unexpected(std::move(compacted.error()));
            }
            if (auto status =
                    clear_queue(queue_view, renderer::WavefrontQueueKind::shadow, clear_status);
                !status) {
                return std::unexpected(std::move(status.error()));
            }
        }

        auto continuation_header =
            queue_header(queue_view, renderer::WavefrontQueueKind::continuation, device_capacity);
        if (!continuation_header)
            return std::unexpected(continuation_header.error());
        report.stage_lanes.continuation += continuation_header->size;
        if (continuation_header->size != 0U) {
            if (auto status = execute_stage(
                    WavefrontStageKind::continuation, "continuation", continuation_header->size,
                    route_mask(WavefrontStageRoute::ray), device_path_count, outcomes, stage_audit,
                    [&] {
                        return blackframe_cuda_launch_wavefront_continuation_stage(
                            queue_view, stream_view, continuation_header->size, outcomes);
                    });
                !status) {
                return std::unexpected(std::move(status.error()));
            }
            if (auto compacted = compact_route(queue_view, outcomes, continuation_header->size,
                                               WavefrontStageRoute::ray, compaction_scratch,
                                               compaction_scratch_bytes, compaction_result);
                !compacted) {
                return std::unexpected(std::move(compacted.error()));
            }
            if (auto status = clear_queue(queue_view, renderer::WavefrontQueueKind::continuation,
                                          clear_status);
                !status) {
                return std::unexpected(std::move(status.error()));
            }
        }
        ray_header = queue_header(queue_view, renderer::WavefrontQueueKind::ray, device_capacity);
        if (!ray_header)
            return std::unexpected(ray_header.error());
    }

    cuda_status =
        cudaMemcpy(storage.final_states.data(), stream_states, state_bytes, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        return std::unexpected(
            cuda_transport_error(cuda_status, "final state download", state_bytes));
    }
    cuda_status =
        cudaMemcpy(storage.final_rays.data(), stream_rays, ray_bytes, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        return std::unexpected(
            cuda_transport_error(cuda_status, "terminal ray download", ray_bytes));
    }
    const auto control_bytes = path_count * sizeof(WavefrontLaneControl);
    cuda_status =
        cudaMemcpy(storage.final_controls.data(), controls, control_bytes, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        return std::unexpected(
            cuda_transport_error(cuda_status, "lane-control download", control_bytes));
    }

    auto paths = std::vector<CudaWavefrontPathResult>{};
    paths.reserve(path_count);
    for (auto index = std::size_t{}; index < path_count; ++index) {
        auto state = renderer_path_state(storage.final_states[index], index);
        auto ray = renderer_ray(storage.final_rays[index], index);
        auto termination = renderer_termination(storage.final_controls[index], index);
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
    storage.queues_dirty = false;
    return CudaWavefrontTransportBatch{.paths = std::move(paths), .report = report};
} catch (const std::bad_alloc&) {
    return std::unexpected(transport_error(core::StatusCode::resource_exhausted,
                                           "CUDA wavefront transport exhausted host memory."));
} catch (const std::length_error&) {
    return std::unexpected(
        transport_error(core::StatusCode::resource_exhausted,
                        "CUDA wavefront transport exceeded a host container length limit."));
}

core::Result<CudaWavefrontTransportBatch>
trace_cuda_wavefront_transport(const CudaSceneSoA& scene, const CudaSceneBvh& bvh,
                               const std::span<const CudaWavefrontPathInput> inputs,
                               const CudaWavefrontLightSampler light_sampler,
                               const CudaWavefrontTransportOptions options) try {
    // Preserve the original contract that rejects the scene and inputs before allocating any
    // per-call transport storage. The workspace overload repeats this inexpensive preflight so it
    // remains independently safe for direct callers.
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

    auto workspace =
        CudaWavefrontTransportWorkspace::create(inputs.size(), options.device_memory_budget);
    if (!workspace) {
        return std::unexpected(std::move(workspace.error()));
    }
    auto traced =
        trace_cuda_wavefront_transport(*workspace, scene, bvh, inputs, light_sampler, options);
    if (!traced) {
        return std::unexpected(std::move(traced.error()));
    }
    auto close_status = workspace->close();
    if (!close_status) {
        return std::unexpected(std::move(close_status.error()));
    }
    return traced;
} catch (const std::bad_alloc&) {
    return std::unexpected(transport_error(core::StatusCode::resource_exhausted,
                                           "CUDA wavefront transport exhausted host memory."));
} catch (const std::length_error&) {
    return std::unexpected(
        transport_error(core::StatusCode::resource_exhausted,
                        "CUDA wavefront transport exceeded a host container length limit."));
}

} // namespace blackframe::engine
