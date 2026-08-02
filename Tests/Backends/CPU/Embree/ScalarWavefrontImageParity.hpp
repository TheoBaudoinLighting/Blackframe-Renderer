#pragma once

#include <Blackframe/Backends/CPU/Embree/AccelBackend.hpp>
#include <Blackframe/Backends/CPU/Embree/WavefrontMisTransport.hpp>
#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/SceneMisPathLoop.hpp>
#include <Blackframe/Renderer/Cie1931Sensor.hpp>
#include <Blackframe/Renderer/CpuWavefrontScheduler.hpp>
#include <Blackframe/Renderer/DisplayPsnr.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/LinearMetrics.hpp>
#include <Blackframe/Renderer/MisHeuristics.hpp>
#include <Blackframe/Renderer/PathDepthLimits.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <Blackframe/Renderer/PixelJitter.hpp>
#include <Blackframe/Renderer/RussianRoulette.hpp>
#include <Blackframe/Renderer/SampleStream.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace blackframe::engine::scalar_wavefront_parity_test {

inline constexpr auto CurrentReportSchemaVersion = std::uint32_t{1U};

struct Thresholds final {
    renderer::ReferenceScalar maximum_linear_mse;
    renderer::ReferenceScalar maximum_linear_rmse;
    renderer::ReferenceScalar maximum_linear_absolute_error;
    renderer::ReferenceScalar maximum_path_radiance_absolute_error;
    renderer::ReferenceScalar minimum_display_psnr;
};

inline constexpr auto StrictThresholds = Thresholds{
    .maximum_linear_mse = 1.0e-10,
    .maximum_linear_rmse = 1.0e-5,
    .maximum_linear_absolute_error = 1.0e-4,
    .maximum_path_radiance_absolute_error = 1.0e-4,
    .minimum_display_psnr = 80.0,
};

struct Configuration final {
    std::string_view scene_name;
    renderer::RenderExtent extent;
    std::uint32_t samples_per_pixel;
    std::uint64_t seed;
    renderer::MisHeuristic heuristic;
    renderer::PathDepthLimits depth_limits;
    renderer::RussianRoulettePolicy roulette_policy;
    std::uint32_t worker_count;
    Thresholds thresholds;
};

struct Result final {
    renderer::LinearMetrics linear;
    renderer::ReferenceScalar maximum_path_radiance_absolute_error;
    renderer::ReferenceScalar display_psnr;
    CpuWavefrontMisReport wavefront_report;
};

[[nodiscard]] inline core::Error parity_error(const core::StatusCode code, std::string message) {
    return core::Error{
        .code = code,
        .message = std::move(message),
    };
}

[[nodiscard]] inline core::Result<std::size_t>
expected_path_count(const renderer::RenderExtent extent, const std::uint32_t samples_per_pixel) {
    if (extent.width == 0U || extent.height == 0U || samples_per_pixel == 0U) {
        return std::unexpected(parity_error(
            core::StatusCode::invalid_argument,
            "Scalar/wavefront image parity requires non-zero dimensions and sample count."));
    }
    const auto width = static_cast<std::size_t>(extent.width);
    const auto height = static_cast<std::size_t>(extent.height);
    const auto samples = static_cast<std::size_t>(samples_per_pixel);
    if (height > std::numeric_limits<std::size_t>::max() / width ||
        samples > std::numeric_limits<std::size_t>::max() / (width * height)) {
        return std::unexpected(
            parity_error(core::StatusCode::resource_exhausted,
                         "Scalar/wavefront image parity path count is not representable."));
    }
    return width * height * samples;
}

template <typename GenerateRay>
[[nodiscard]] core::Result<std::vector<CpuWavefrontMisPathInput>>
make_inputs(const renderer::RenderExtent extent, const std::uint32_t samples_per_pixel,
            const std::uint64_t seed, const renderer::SampledWavelengths& wavelengths,
            GenerateRay&& generate_ray) {
    const auto count = expected_path_count(extent, samples_per_pixel);
    if (!count) {
        return std::unexpected(count.error());
    }
    const auto initial_state =
        renderer::PathState::create_initial(wavelengths, renderer::VacuumMedium);
    if (!initial_state) {
        return std::unexpected(initial_state.error());
    }

    const auto sampler = renderer::IndependentSampler{seed};
    auto inputs = std::vector<CpuWavefrontMisPathInput>{};
    inputs.reserve(*count);
    for (auto sample_index = std::uint64_t{}; sample_index < samples_per_pixel; ++sample_index) {
        for (auto pixel_y = std::uint32_t{}; pixel_y < extent.height; ++pixel_y) {
            for (auto pixel_x = std::uint32_t{}; pixel_x < extent.width; ++pixel_x) {
                const auto index = renderer::PixelSampleIndex{
                    .pixel_x = pixel_x,
                    .pixel_y = pixel_y,
                    .sample_index = sample_index,
                    .seed = seed,
                };
                const auto stream = sampler.make_stream(pixel_x, pixel_y, sample_index);
                const auto ray = generate_ray(index, stream);
                if (!ray) {
                    return std::unexpected(ray.error());
                }
                inputs.push_back(CpuWavefrontMisPathInput{
                    .primary_ray = *ray,
                    .initial_state = *initial_state,
                    .sample = stream.index(),
                });
            }
        }
    }
    return inputs;
}

template <renderer::AccumulationPrecision Precision>
[[nodiscard]] core::Status accumulate_path(renderer::FilmT<Precision>& film,
                                           const CpuWavefrontMisPathInput& input,
                                           const renderer::BsdfOnlyPathResult& path) {
    const auto xyz = renderer::cie_1931_spectrum_to_xyz(path.state.accumulated_radiance(),
                                                        input.initial_state.wavelengths());
    if (!xyz) {
        return std::unexpected(xyz.error());
    }
    const auto rgb = renderer::xyz_to_linear_rgb(*xyz);
    if (!rgb) {
        return std::unexpected(rgb.error());
    }
    using Film = renderer::FilmT<Precision>;
    return film.add_sample(input.sample.pixel_x, input.sample.pixel_y,
                           typename Film::Color{
                               .red = static_cast<typename Film::Scalar>(rgb->red),
                               .green = static_cast<typename Film::Scalar>(rgb->green),
                               .blue = static_cast<typename Film::Scalar>(rgb->blue),
                           },
                           typename Film::Scalar{1});
}

[[nodiscard]] inline core::Status validate_queue_report(const CpuWavefrontMisReport& report,
                                                        const std::size_t path_count,
                                                        const std::uint32_t worker_count) {
    if (report.schema_version != CurrentCpuWavefrontMisReportSchemaVersion ||
        report.path_count != path_count || report.configured_workers != worker_count) {
        return std::unexpected(
            parity_error(core::StatusCode::internal_error,
                         "CPU wavefront parity received an inconsistent versioned batch report."));
    }
    const auto queues = std::array{
        &report.queue_statistics.camera,       &report.queue_statistics.ray,
        &report.queue_statistics.hit,          &report.queue_statistics.miss,
        &report.queue_statistics.shade,        &report.queue_statistics.shadow,
        &report.queue_statistics.continuation,
    };
    const auto expected_kinds = std::array{
        renderer::WavefrontQueueKind::camera,       renderer::WavefrontQueueKind::ray,
        renderer::WavefrontQueueKind::hit,          renderer::WavefrontQueueKind::miss,
        renderer::WavefrontQueueKind::shade,        renderer::WavefrontQueueKind::shadow,
        renderer::WavefrontQueueKind::continuation,
    };
    const auto stage_lanes = std::array{
        report.stage_lanes.camera,       report.stage_lanes.ray,   report.stage_lanes.hit,
        report.stage_lanes.miss,         report.stage_lanes.shade, report.stage_lanes.shadow,
        report.stage_lanes.continuation,
    };
    for (auto index = std::size_t{}; index < queues.size(); ++index) {
        const auto* const queue = queues[index];
        const auto peak_occupancy = queue->peak_occupancy();
        const auto mean_occupancy = queue->mean_occupancy();
        if (queue->kind != expected_kinds[index] || queue->capacity != path_count ||
            queue->peak_size > queue->capacity || queue->input_lanes != stage_lanes[index] ||
            ((queue->dispatch_count == 0U) != (queue->input_lanes == 0U)) ||
            queue->overflow_attempts != 0U || queue->rejected_lanes != 0U ||
            !std::isfinite(peak_occupancy) || peak_occupancy < 0.0 || peak_occupancy > 1.0 ||
            !std::isfinite(mean_occupancy) || mean_occupancy < 0.0 || mean_occupancy > 1.0) {
            return std::unexpected(parity_error(
                core::StatusCode::internal_error,
                "CPU wavefront parity observed invalid or overflowing queue statistics."));
        }
    }
    return {};
}

[[nodiscard]] inline core::Result<renderer::ReferenceScalar>
maximum_path_radiance_error(const std::span<const renderer::BsdfOnlyPathResult> scalar_paths,
                            const std::span<const renderer::BsdfOnlyPathResult> wavefront_paths) {
    if (scalar_paths.size() != wavefront_paths.size()) {
        return std::unexpected(
            parity_error(core::StatusCode::incompatible,
                         "Scalar/wavefront path radiance comparison requires equal path counts."));
    }
    auto maximum = renderer::ReferenceScalar{};
    for (auto path_index = std::size_t{}; path_index < scalar_paths.size(); ++path_index) {
        for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
            const auto difference =
                std::abs(static_cast<renderer::ReferenceScalar>(
                             wavefront_paths[path_index].state.accumulated_radiance()[lane]) -
                         static_cast<renderer::ReferenceScalar>(
                             scalar_paths[path_index].state.accumulated_radiance()[lane]));
            if (!std::isfinite(difference)) {
                return std::unexpected(parity_error(
                    core::StatusCode::internal_error,
                    "Scalar/wavefront path radiance comparison produced an invalid value."));
            }
            maximum = std::max(maximum, difference);
        }
    }
    return maximum;
}

[[nodiscard]] inline core::Result<Result>
compare(const FrameSceneHandle& scene, const std::span<const CpuWavefrontMisPathInput> inputs,
        const Configuration& configuration) {
    if (!scene || configuration.scene_name.empty()) {
        return std::unexpected(parity_error(
            core::StatusCode::invalid_argument,
            "Scalar/wavefront image parity requires an explicit scene and scene name."));
    }
    if (!std::isfinite(configuration.thresholds.maximum_linear_mse) ||
        configuration.thresholds.maximum_linear_mse < 0.0 ||
        !std::isfinite(configuration.thresholds.maximum_linear_rmse) ||
        configuration.thresholds.maximum_linear_rmse < 0.0 ||
        !std::isfinite(configuration.thresholds.maximum_linear_absolute_error) ||
        configuration.thresholds.maximum_linear_absolute_error < 0.0 ||
        !std::isfinite(configuration.thresholds.maximum_path_radiance_absolute_error) ||
        configuration.thresholds.maximum_path_radiance_absolute_error < 0.0 ||
        !std::isfinite(configuration.thresholds.minimum_display_psnr) ||
        configuration.thresholds.minimum_display_psnr < 0.0) {
        return std::unexpected(parity_error(
            core::StatusCode::invalid_argument,
            "Scalar/wavefront image parity requires explicit finite non-negative thresholds."));
    }
    const auto count = expected_path_count(configuration.extent, configuration.samples_per_pixel);
    if (!count) {
        return std::unexpected(count.error());
    }
    if (inputs.size() != *count) {
        return std::unexpected(parity_error(
            core::StatusCode::incompatible,
            "Scalar/wavefront image parity input count does not match the image domain."));
    }
    if (scene->punctual_lights().size() >
        std::numeric_limits<std::size_t>::max() - scene->mesh_area_lights().size()) {
        return std::unexpected(parity_error(
            core::StatusCode::resource_exhausted,
            "Scalar/wavefront image parity light registry size is not representable."));
    }
    const auto light_count = scene->punctual_lights().size() + scene->mesh_area_lights().size();
    const auto light_sampler = renderer::LightSampler::create_uniform(light_count);
    const auto scalar_backend = create_analytic_accel_backend(scene);
    const auto wavefront_backend = create_embree_accel_backend(scene);
    const auto scheduler = renderer::CpuWavefrontScheduler::create(configuration.worker_count);
    auto scalar_film = renderer::ReferenceFilm::create(configuration.extent);
    auto wavefront_film = renderer::Film::create(configuration.extent);
    if (!light_sampler) {
        return std::unexpected(light_sampler.error());
    }
    if (!scalar_backend) {
        return std::unexpected(scalar_backend.error());
    }
    if (!wavefront_backend) {
        return std::unexpected(wavefront_backend.error());
    }
    if (!scheduler) {
        return std::unexpected(scheduler.error());
    }
    if (!scalar_film) {
        return std::unexpected(scalar_film.error());
    }
    if (!wavefront_film) {
        return std::unexpected(wavefront_film.error());
    }
    if ((*scalar_backend)->kind() != AccelBackendKind::analytic_reference ||
        (*wavefront_backend)->kind() != AccelBackendKind::embree) {
        return std::unexpected(parity_error(
            core::StatusCode::incompatible,
            "Scalar/wavefront image parity did not receive the explicitly selected backends."));
    }

    auto scalar_paths = std::vector<renderer::BsdfOnlyPathResult>{};
    scalar_paths.reserve(inputs.size());
    for (auto input_index = std::size_t{}; input_index < inputs.size(); ++input_index) {
        const auto& input = inputs[input_index];
        const auto scalar = trace_scene_mis(
            input.primary_ray, input.initial_state, renderer::SampleStream{input.sample},
            **scalar_backend, *light_sampler, configuration.heuristic, configuration.depth_limits,
            configuration.roulette_policy);
        if (!scalar) {
            return std::unexpected(parity_error(
                scalar.error().code, "Scalar reference path " + std::to_string(input_index) +
                                         " failed: " + scalar.error().message));
        }
        if (auto status = accumulate_path(*scalar_film, input, *scalar); !status) {
            return std::unexpected(status.error());
        }
        scalar_paths.push_back(*scalar);
    }

    const auto wavefront = trace_cpu_wavefront_mis(
        inputs, *scheduler, **wavefront_backend, *light_sampler, configuration.heuristic,
        configuration.depth_limits, configuration.roulette_policy);
    if (!wavefront) {
        return std::unexpected(wavefront.error());
    }
    if (wavefront->paths.size() != inputs.size()) {
        return std::unexpected(parity_error(
            core::StatusCode::internal_error,
            "CPU wavefront parity returned a different number of paths than it received."));
    }
    if (auto status =
            validate_queue_report(wavefront->report, inputs.size(), configuration.worker_count);
        !status) {
        return std::unexpected(status.error());
    }
    for (auto index = std::size_t{}; index < inputs.size(); ++index) {
        if (wavefront->paths[index].termination != scalar_paths[index].termination ||
            wavefront->paths[index].blocked_depth_limits !=
                scalar_paths[index].blocked_depth_limits) {
            return std::unexpected(parity_error(
                core::StatusCode::incompatible,
                "Scalar and CPU wavefront paths disagree on terminal transport state."));
        }
        if (auto status = accumulate_path(*wavefront_film, inputs[index], wavefront->paths[index]);
            !status) {
            return std::unexpected(status.error());
        }
    }

    for (auto pixel_y = std::uint32_t{}; pixel_y < configuration.extent.height; ++pixel_y) {
        for (auto pixel_x = std::uint32_t{}; pixel_x < configuration.extent.width; ++pixel_x) {
            const auto scalar_pixel = scalar_film->pixel(pixel_x, pixel_y);
            const auto wavefront_pixel = wavefront_film->pixel(pixel_x, pixel_y);
            if (!scalar_pixel || !wavefront_pixel ||
                scalar_pixel->sample_count != configuration.samples_per_pixel ||
                wavefront_pixel->sample_count != configuration.samples_per_pixel) {
                return std::unexpected(parity_error(
                    core::StatusCode::internal_error,
                    "Scalar/wavefront image parity did not cover every pixel exactly."));
            }
        }
    }

    const auto linear = renderer::compute_linear_metrics(*wavefront_film, *scalar_film);
    const auto display = renderer::compute_display_psnr(*wavefront_film, *scalar_film);
    const auto maximum_path_error = maximum_path_radiance_error(scalar_paths, wavefront->paths);
    if (!linear) {
        return std::unexpected(linear.error());
    }
    if (!display) {
        return std::unexpected(display.error());
    }
    if (!maximum_path_error) {
        return std::unexpected(maximum_path_error.error());
    }
    if (!std::isfinite(linear->mse) || !std::isfinite(linear->rmse) ||
        !std::isfinite(linear->mean_bias) || !std::isfinite(linear->maximum_absolute_error) ||
        std::isnan(display->psnr) || (std::isinf(display->psnr) && display->psnr < 0.0)) {
        return std::unexpected(
            parity_error(core::StatusCode::internal_error,
                         "Scalar/wavefront image parity produced an invalid metric."));
    }
    return Result{
        .linear = *linear,
        .maximum_path_radiance_absolute_error = *maximum_path_error,
        .display_psnr = display->psnr,
        .wavefront_report = wavefront->report,
    };
}

[[nodiscard]] inline std::string metric_text(const renderer::ReferenceScalar value) {
    auto buffer = std::array<char, 64U>{};
    const auto [end, error] = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::general,
        std::numeric_limits<renderer::ReferenceScalar>::max_digits10);
    if (error != std::errc{}) {
        return "unrepresentable";
    }
    return std::string{buffer.data(), static_cast<std::size_t>(end - buffer.data())};
}

[[nodiscard]] inline constexpr std::string_view
heuristic_name(const renderer::MisHeuristic heuristic) noexcept {
    switch (heuristic) {
    case renderer::MisHeuristic::balance:
        return "balance";
    case renderer::MisHeuristic::power:
        return "power";
    }
    return "unknown";
}

inline void record_and_expect(const Configuration& configuration, const Result& result) {
    testing::Test::RecordProperty("parity_schema_version",
                                  static_cast<int>(CurrentReportSchemaVersion));
    testing::Test::RecordProperty("wavefront_report_schema_version",
                                  static_cast<int>(result.wavefront_report.schema_version));
    testing::Test::RecordProperty("scene", std::string{configuration.scene_name});
    testing::Test::RecordProperty("scalar_backend", "scalar_ref");
    testing::Test::RecordProperty("scalar_acceleration", "analytic_reference");
    testing::Test::RecordProperty("scalar_transport_precision", "float");
    testing::Test::RecordProperty("scalar_accumulation_precision", "double");
    testing::Test::RecordProperty("wavefront_backend", "cpu_wavefront");
    testing::Test::RecordProperty("wavefront_acceleration", "embree");
    testing::Test::RecordProperty("wavefront_transport_precision", "float");
    testing::Test::RecordProperty("wavefront_accumulation_precision", "float");
    testing::Test::RecordProperty("sampler", "independent_indexed");
    testing::Test::RecordProperty("mis_heuristic",
                                  std::string{heuristic_name(configuration.heuristic)});
    testing::Test::RecordProperty("width", static_cast<int>(configuration.extent.width));
    testing::Test::RecordProperty("height", static_cast<int>(configuration.extent.height));
    testing::Test::RecordProperty("samples_per_pixel",
                                  static_cast<int>(configuration.samples_per_pixel));
    testing::Test::RecordProperty("seed", std::to_string(configuration.seed));
    testing::Test::RecordProperty("workers", static_cast<int>(configuration.worker_count));
    testing::Test::RecordProperty("path_count", std::to_string(result.wavefront_report.path_count));
    testing::Test::RecordProperty("closure_samples",
                                  std::to_string(result.wavefront_report.closure_samples));
    testing::Test::RecordProperty("light_samples",
                                  std::to_string(result.wavefront_report.light_samples));
    testing::Test::RecordProperty("shadow_queries",
                                  std::to_string(result.wavefront_report.shadow_queries));
    testing::Test::RecordProperty("queue_overflow_attempts", "0");
    testing::Test::RecordProperty("queue_rejected_lanes", "0");
    testing::Test::RecordProperty("maximum_linear_mse",
                                  metric_text(configuration.thresholds.maximum_linear_mse));
    testing::Test::RecordProperty("maximum_linear_rmse",
                                  metric_text(configuration.thresholds.maximum_linear_rmse));
    testing::Test::RecordProperty(
        "maximum_linear_absolute_error",
        metric_text(configuration.thresholds.maximum_linear_absolute_error));
    testing::Test::RecordProperty(
        "maximum_path_radiance_absolute_error",
        metric_text(configuration.thresholds.maximum_path_radiance_absolute_error));
    testing::Test::RecordProperty("minimum_display_psnr",
                                  metric_text(configuration.thresholds.minimum_display_psnr));
    testing::Test::RecordProperty("mse_linear", metric_text(result.linear.mse));
    testing::Test::RecordProperty("rmse_linear", metric_text(result.linear.rmse));
    testing::Test::RecordProperty("bias_mean", metric_text(result.linear.mean_bias));
    testing::Test::RecordProperty("max_abs", metric_text(result.linear.maximum_absolute_error));
    testing::Test::RecordProperty("path_radiance_max_abs",
                                  metric_text(result.maximum_path_radiance_absolute_error));
    testing::Test::RecordProperty("psnr_display", metric_text(result.display_psnr));

    EXPECT_LE(result.linear.mse, configuration.thresholds.maximum_linear_mse);
    EXPECT_LE(result.linear.rmse, configuration.thresholds.maximum_linear_rmse);
    EXPECT_LE(result.linear.maximum_absolute_error,
              configuration.thresholds.maximum_linear_absolute_error);
    EXPECT_LE(result.maximum_path_radiance_absolute_error,
              configuration.thresholds.maximum_path_radiance_absolute_error);
    EXPECT_TRUE(std::isinf(result.display_psnr) ||
                result.display_psnr >= configuration.thresholds.minimum_display_psnr);
}

} // namespace blackframe::engine::scalar_wavefront_parity_test
