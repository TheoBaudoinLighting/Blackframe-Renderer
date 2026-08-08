#include "../../Tests/Backends/CPU/Embree/CornellWavefrontScene.hpp"

#include <Blackframe/Backends/GPU/CUDA/SceneBvh.hpp>
#include <Blackframe/Backends/GPU/CUDA/SceneSoA.hpp>
#include <Blackframe/Backends/GPU/CUDA/WavefrontTransport.hpp>
#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/PixelJitter.hpp>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

inline constexpr auto CornellExtent = renderer::RenderExtent{.width = 64U, .height = 64U};
inline constexpr auto CornellPixelCount = std::size_t{4'096U};
inline constexpr auto CornellLargePathCount = std::size_t{65'536U};
inline constexpr auto CornellSeed = std::uint64_t{0x243F6A8885A308D3ULL};
inline constexpr auto CornellPathTime = renderer::TransportScalar{0.5F};

[[nodiscard]] core::Result<std::vector<CudaWavefrontPathInput>>
make_cornell_inputs(const renderer::PinholeCamera& camera,
                    const renderer::SampledWavelengths& wavelengths, const std::size_t path_count) {
    if (path_count == 0U || path_count % CornellPixelCount != 0U) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "The CUDA Cornell benchmark path count must contain complete images.",
        });
    }
    const auto initial_state =
        renderer::PathState::create_initial(wavelengths, renderer::VacuumMedium);
    if (!initial_state) {
        return std::unexpected(initial_state.error());
    }

    const auto sampler = renderer::IndependentSampler{CornellSeed};
    auto inputs = std::vector<CudaWavefrontPathInput>{};
    inputs.reserve(path_count);
    const auto samples_per_pixel = path_count / CornellPixelCount;
    for (auto sample_index = std::size_t{}; sample_index < samples_per_pixel; ++sample_index) {
        for (auto pixel_y = std::uint32_t{}; pixel_y < CornellExtent.height; ++pixel_y) {
            for (auto pixel_x = std::uint32_t{}; pixel_x < CornellExtent.width; ++pixel_x) {
                const auto pixel_sample = renderer::PixelSampleIndex{
                    .pixel_x = pixel_x,
                    .pixel_y = pixel_y,
                    .sample_index = sample_index,
                    .seed = CornellSeed,
                };
                const auto primary = camera.generate_primary_ray(
                    pixel_sample, renderer::PixelJitterMode::uniform, CornellPathTime);
                if (!primary) {
                    return std::unexpected(primary.error());
                }
                const auto primary_cone = camera.generate_primary_ray_cone(
                    pixel_sample, renderer::PixelJitterMode::uniform, CornellPathTime);
                if (!primary_cone) {
                    return std::unexpected(primary_cone.error());
                }
                inputs.push_back(CudaWavefrontPathInput{
                    .primary_ray = *primary,
                    .primary_cone = *primary_cone,
                    .initial_state = *initial_state,
                    .sample = sampler.make_stream(pixel_x, pixel_y, sample_index).index(),
                });
            }
        }
    }
    return inputs;
}

[[nodiscard]] double transport_checksum(const CudaWavefrontTransportBatch& batch) noexcept {
    auto checksum = double{};
    for (auto path_index = std::size_t{}; path_index < batch.paths.size(); ++path_index) {
        const auto& spectrum = batch.paths[path_index].state.accumulated_radiance();
        const auto path_weight = static_cast<double>((path_index % 17U) + 1U);
        for (auto lane = std::size_t{}; lane < spectrum.values.size(); ++lane) {
            checksum += static_cast<double>(spectrum.values[lane]) * path_weight *
                        static_cast<double>(lane + 1U);
        }
        checksum += static_cast<double>(batch.terminal_cones[path_index].width()) * path_weight;
        checksum += static_cast<double>(batch.terminal_cones[path_index].spread()) * path_weight;
    }
    return checksum;
}

[[nodiscard]] bool transport_paths_exact(const CudaWavefrontPathResult& expected,
                                         const CudaWavefrontPathResult& actual) noexcept {
    return actual.termination == expected.termination &&
           actual.blocked_depth_limits == expected.blocked_depth_limits &&
           actual.state.beta() == expected.state.beta() &&
           actual.state.accumulated_radiance() == expected.state.accumulated_radiance() &&
           actual.state.depth() == expected.state.depth() &&
           actual.state.depth_counters() == expected.state.depth_counters() &&
           actual.state.eta_scale() == expected.state.eta_scale() &&
           actual.state.wavelengths() == expected.state.wavelengths() &&
           actual.state.delta_flags() == expected.state.delta_flags() &&
           actual.state.current_medium() == expected.state.current_medium() &&
           actual.terminal_ray.origin() == expected.terminal_ray.origin() &&
           actual.terminal_ray.direction() == expected.terminal_ray.direction() &&
           actual.terminal_ray.t_min() == expected.terminal_ray.t_min() &&
           actual.terminal_ray.t_max() == expected.terminal_ray.t_max() &&
           actual.terminal_ray.time() == expected.terminal_ray.time() &&
           actual.terminal_ray.mask() == expected.terminal_ray.mask() &&
           actual.terminal_ray.current_medium() == expected.terminal_ray.current_medium();
}

[[nodiscard]] bool
transport_reports_physically_exact(const CudaWavefrontTransportReport& expected,
                                   const CudaWavefrontTransportReport& actual) noexcept {
    return actual.schema_version == expected.schema_version &&
           actual.has_light_sampler == expected.has_light_sampler &&
           actual.registered_light_count == expected.registered_light_count &&
           actual.heuristic == expected.heuristic &&
           actual.light_sampling_strategy == expected.light_sampling_strategy &&
           actual.depth_limits == expected.depth_limits &&
           actual.roulette_policy == expected.roulette_policy &&
           actual.path_count == expected.path_count && actual.stage_lanes == expected.stage_lanes &&
           actual.closure_samples == expected.closure_samples &&
           actual.light_samples == expected.light_samples &&
           actual.shadow_queries == expected.shadow_queries &&
           actual.terminated_paths == expected.terminated_paths &&
           actual.queue_overflow_attempts == expected.queue_overflow_attempts &&
           actual.queue_rejected_lanes == expected.queue_rejected_lanes;
}

[[nodiscard]] bool
transport_batches_physically_exact(const CudaWavefrontTransportBatch& expected,
                                   const CudaWavefrontTransportBatch& actual) noexcept {
    if (!transport_reports_physically_exact(expected.report, actual.report) ||
        actual.paths.size() != expected.paths.size() ||
        actual.terminal_cones.size() != expected.terminal_cones.size() ||
        actual.terminal_cones.size() != actual.paths.size()) {
        return false;
    }
    for (auto index = std::size_t{}; index < expected.paths.size(); ++index) {
        if (!transport_paths_exact(expected.paths[index], actual.paths[index]) ||
            actual.terminal_cones[index] != expected.terminal_cones[index]) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool transport_report_matches_mode(const CudaWavefrontTransportReport& report,
                                                 const CudaWavefrontTransferMode mode) noexcept {
    if (report.transfer_mode != mode) {
        return false;
    }
    if (mode == CudaWavefrontTransferMode::asynchronous) {
        return report.asynchronous_upload_bytes > 0U && report.asynchronous_download_bytes > 0U &&
               report.cross_stream_event_dependencies > 0U;
    }
    return report.asynchronous_upload_bytes == 0U && report.asynchronous_download_bytes == 0U &&
           report.cross_stream_event_dependencies == 0U;
}

void run_cuda_cornell_wavefront_transport(benchmark::State& state, const bool reuse_workspace,
                                          const CudaWavefrontTransferMode transfer_mode) {
    const auto path_count = static_cast<std::size_t>(state.range(0));
    if (path_count != CornellPixelCount && path_count != CornellLargePathCount) {
        state.SkipWithError("The CUDA Cornell benchmark supports 4096 or 65536 paths.");
        return;
    }

    const auto scene = cornell_wavefront_test::make_cornell_scene();
    if (!scene) {
        state.SkipWithError(scene.error().message);
        return;
    }
    const auto light_sampler = renderer::LightSampler::create_uniform(
        (*scene)->punctual_lights().size() + (*scene)->mesh_area_lights().size());
    if (!light_sampler) {
        state.SkipWithError(light_sampler.error().message);
        return;
    }
    auto uploaded = CudaSceneSoA::upload(**scene);
    if (!uploaded) {
        state.SkipWithError(uploaded.error().message);
        return;
    }
    auto bvh = CudaSceneBvh::build(*uploaded);
    if (!bvh) {
        state.SkipWithError(bvh.error().message);
        return;
    }
    const auto camera = cornell_wavefront_test::make_camera(CornellExtent);
    if (!camera) {
        state.SkipWithError(camera.error().message);
        return;
    }
    const auto inputs =
        make_cornell_inputs(*camera, (*scene)->spectral_environment()->wavelengths, path_count);
    if (!inputs) {
        state.SkipWithError(inputs.error().message);
        return;
    }

    const auto options = CudaWavefrontTransportOptions{
        .heuristic = renderer::MisHeuristic::power,
        .depth_limits = renderer::PathDepthLimits{.diffuse = 5U},
        .roulette_policy = renderer::RussianRoulettePolicy::disabled(),
        .transfer_mode = transfer_mode,
    };
    const auto light_sampler_reference = std::cref(*light_sampler);
    auto workspace = std::optional<CudaWavefrontTransportWorkspace>{};
    if (reuse_workspace) {
        auto created_workspace = CudaWavefrontTransportWorkspace::create(
            path_count, xpu::cuda::DeviceMemoryBudget{}, transfer_mode);
        if (!created_workspace) {
            state.SkipWithError(created_workspace.error().message);
            return;
        }
        workspace.emplace(std::move(*created_workspace));
    }

    const auto trace = [&]() -> core::Result<CudaWavefrontTransportBatch> {
        if (workspace) {
            return trace_cuda_wavefront_transport(*workspace, *uploaded, *bvh, *inputs,
                                                  light_sampler_reference, options);
        }
        return trace_cuda_wavefront_transport(*uploaded, *bvh, *inputs, light_sampler_reference,
                                              options);
    };

    auto synchronous_baseline = std::optional<CudaWavefrontTransportBatch>{};
    auto synchronous_checksum = double{};
    if (transfer_mode == CudaWavefrontTransferMode::asynchronous) {
        auto synchronous_workspace = CudaWavefrontTransportWorkspace::create(
            path_count, xpu::cuda::DeviceMemoryBudget{}, CudaWavefrontTransferMode::synchronous);
        if (!synchronous_workspace) {
            state.SkipWithError(synchronous_workspace.error().message);
            return;
        }
        auto synchronous_options = options;
        synchronous_options.transfer_mode = CudaWavefrontTransferMode::synchronous;
        auto baseline =
            trace_cuda_wavefront_transport(*synchronous_workspace, *uploaded, *bvh, *inputs,
                                           light_sampler_reference, synchronous_options);
        if (!baseline) {
            state.SkipWithError(baseline.error().message);
            return;
        }
        if (!transport_report_matches_mode(baseline->report,
                                           CudaWavefrontTransferMode::synchronous)) {
            state.SkipWithError(
                "The CUDA Cornell synchronous baseline reported an inconsistent transfer mode.");
            return;
        }
        synchronous_checksum = transport_checksum(*baseline);
        if (!std::isfinite(synchronous_checksum)) {
            state.SkipWithError("The CUDA Cornell synchronous baseline checksum is not finite.");
            return;
        }
        synchronous_baseline.emplace(std::move(*baseline));
        const auto close_status = synchronous_workspace->close();
        if (!close_status) {
            state.SkipWithError(close_status.error().message);
            return;
        }
    }

    const auto warmup = trace();
    if (!warmup) {
        state.SkipWithError(warmup.error().message);
        return;
    }
    if (warmup->paths.size() != inputs->size() ||
        warmup->report.terminated_paths != inputs->size() ||
        warmup->report.queue_overflow_attempts != 0U || warmup->report.queue_rejected_lanes != 0U) {
        state.SkipWithError("The CUDA Cornell warmup returned an inconsistent transport batch.");
        return;
    }
    if (!transport_report_matches_mode(warmup->report, transfer_mode)) {
        state.SkipWithError("The CUDA Cornell warmup reported an inconsistent transfer mode.");
        return;
    }
    auto warmup_checksum = transport_checksum(*warmup);
    if (!std::isfinite(warmup_checksum)) {
        state.SkipWithError("The CUDA Cornell warmup checksum is not finite.");
        return;
    }
    if (synchronous_baseline &&
        (warmup_checksum != synchronous_checksum ||
         !transport_batches_physically_exact(*synchronous_baseline, *warmup))) {
        state.SkipWithError(
            "The asynchronous CUDA Cornell warmup differs from its synchronous baseline.");
        return;
    }
    benchmark::DoNotOptimize(warmup_checksum);

    auto latest = std::optional<CudaWavefrontTransportBatch>{};
    for (auto _ : state) {
        static_cast<void>(_);
        auto traced = trace();
        if (!traced) {
            state.SkipWithError(traced.error().message);
            return;
        }
        latest.emplace(std::move(*traced));
        benchmark::DoNotOptimize(latest->paths.data());
        benchmark::ClobberMemory();
    }

    if (!latest) {
        state.SkipWithError("The CUDA Cornell benchmark produced no measured transport batch.");
        return;
    }
    const auto checksum = transport_checksum(*latest);
    if (!std::isfinite(checksum)) {
        state.SkipWithError("The CUDA Cornell measured checksum is not finite.");
        return;
    }
    if (checksum != warmup_checksum) {
        state.SkipWithError("The CUDA Cornell benchmark replay changed its transport checksum.");
        return;
    }

    const auto& report = latest->report;
    state.counters["checksum"] = checksum;
    state.counters["warmup_checksum"] = warmup_checksum;
    state.counters["synchronous_checksum"] =
        synchronous_baseline ? synchronous_checksum : warmup_checksum;
    state.counters["path_count"] = static_cast<double>(report.path_count);
    state.counters["terminated_paths"] = static_cast<double>(report.terminated_paths);
    state.counters["closure_samples"] = static_cast<double>(report.closure_samples);
    state.counters["light_samples"] = static_cast<double>(report.light_samples);
    state.counters["shadow_queries"] = static_cast<double>(report.shadow_queries);
    state.counters["intersection_lanes"] = static_cast<double>(report.stage_lanes.intersection);
    state.counters["shade_lanes"] = static_cast<double>(report.stage_lanes.shade);
    state.counters["scene_bytes"] = static_cast<double>(uploaded->size_bytes());
    state.counters["bvh_bytes"] = static_cast<double>(bvh->size_bytes());
    state.counters["workspace_reused"] = reuse_workspace ? 1.0 : 0.0;
    state.counters["workspace_device_bytes"] =
        workspace ? static_cast<double>(workspace->device_size_bytes()) : 0.0;
    state.counters["transfer_mode"] =
        transfer_mode == CudaWavefrontTransferMode::asynchronous ? 1.0 : 0.0;
    state.counters["asynchronous_upload_bytes"] =
        static_cast<double>(report.asynchronous_upload_bytes);
    state.counters["asynchronous_download_bytes"] =
        static_cast<double>(report.asynchronous_download_bytes);
    state.counters["cross_stream_event_dependencies"] =
        static_cast<double>(report.cross_stream_event_dependencies);
    state.counters["synchronous_parity_exact"] = 1.0;
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(inputs->size()));

    if (workspace) {
        const auto workspace_close = workspace->close();
        if (!workspace_close) {
            state.SkipWithError(workspace_close.error().message);
            return;
        }
    }
    const auto bvh_close = bvh->close();
    if (!bvh_close) {
        state.SkipWithError(bvh_close.error().message);
        return;
    }
    const auto scene_close = uploaded->close();
    if (!scene_close) {
        state.SkipWithError(scene_close.error().message);
    }
}

void cuda_cornell_wavefront_transport(benchmark::State& state) {
    run_cuda_cornell_wavefront_transport(state, false, CudaWavefrontTransferMode::synchronous);
}

void cuda_cornell_wavefront_transport_reused_workspace(benchmark::State& state) {
    run_cuda_cornell_wavefront_transport(state, true, CudaWavefrontTransferMode::synchronous);
}

void cuda_cornell_wavefront_transport_reused_workspace_asynchronous(benchmark::State& state) {
    run_cuda_cornell_wavefront_transport(state, true, CudaWavefrontTransferMode::asynchronous);
}

BENCHMARK(cuda_cornell_wavefront_transport)->Arg(4'096)->Arg(65'536)->UseRealTime();
BENCHMARK(cuda_cornell_wavefront_transport_reused_workspace)
    ->Arg(4'096)
    ->Arg(65'536)
    ->UseRealTime();
BENCHMARK(cuda_cornell_wavefront_transport_reused_workspace_asynchronous)
    ->Arg(4'096)
    ->Arg(65'536)
    ->UseRealTime();

} // namespace
} // namespace blackframe::engine
