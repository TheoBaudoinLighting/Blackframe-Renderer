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
                inputs.push_back(CudaWavefrontPathInput{
                    .primary_ray = *primary,
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
    }
    return checksum;
}

void run_cuda_cornell_wavefront_transport(benchmark::State& state, const bool reuse_workspace) {
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
    };
    const auto light_sampler_reference = std::cref(*light_sampler);
    auto workspace = std::optional<CudaWavefrontTransportWorkspace>{};
    if (reuse_workspace) {
        auto created_workspace = CudaWavefrontTransportWorkspace::create(path_count);
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
    auto warmup_checksum = transport_checksum(*warmup);
    if (!std::isfinite(warmup_checksum)) {
        state.SkipWithError("The CUDA Cornell warmup checksum is not finite.");
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
    run_cuda_cornell_wavefront_transport(state, false);
}

void cuda_cornell_wavefront_transport_reused_workspace(benchmark::State& state) {
    run_cuda_cornell_wavefront_transport(state, true);
}

BENCHMARK(cuda_cornell_wavefront_transport)->Arg(4'096)->Arg(65'536)->UseRealTime();
BENCHMARK(cuda_cornell_wavefront_transport_reused_workspace)
    ->Arg(4'096)
    ->Arg(65'536)
    ->UseRealTime();

} // namespace
} // namespace blackframe::engine
