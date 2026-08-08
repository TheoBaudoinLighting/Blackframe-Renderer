#include "../../Tests/Backends/CPU/Embree/CornellWavefrontScene.hpp"

#include <Blackframe/Backends/CPU/Embree/AccelBackend.hpp>
#include <Blackframe/Backends/CPU/Embree/WavefrontMisTransport.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/SceneMisPathLoop.hpp>
#include <Blackframe/Renderer/Cie1931Sensor.hpp>
#include <Blackframe/Renderer/ConvergenceMetrics.hpp>
#include <Blackframe/Renderer/DisplayPsnr.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/LinearMetrics.hpp>
#include <Blackframe/Renderer/PathDepthLimits.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <Blackframe/Renderer/PinholeCamera.hpp>
#include <Blackframe/Renderer/PngWriter.hpp>
#include <Blackframe/Renderer/RussianRoulette.hpp>
#include <algorithm>
#include <array>
#include <benchmark/benchmark.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef BLACKFRAME_CORNELL_BENCHMARK_ARTIFACT_PATH
#error "The Cornell benchmark artifact path must be supplied by CMake."
#endif

namespace blackframe::engine {
namespace {

using Clock = std::chrono::steady_clock;

inline constexpr auto ImageExtent = renderer::RenderExtent{.width = 128U, .height = 128U};
inline constexpr auto ImageSamplesPerPixel = std::uint64_t{256U};
inline constexpr auto ReferenceSamplesPerPixel = std::uint64_t{512U};
inline constexpr auto ImageSeeds = std::array<std::uint64_t, 8U>{
    0x243F6A8885A308D3ULL, 0x85A308D313198A2EULL, 0x03707344A4093822ULL, 0x299F31D0082EFA98ULL,
    0xEC4E6C89452821E6ULL, 0x38D01377BE5466CFULL, 0x34E90C6CC0AC29B7ULL, 0xC97C50DD3F84D5B5ULL,
};
inline constexpr auto ReferenceSeed = std::uint64_t{0x13198A2E03707344ULL};
inline constexpr auto PathTime = renderer::TransportScalar{0.5F};
inline constexpr auto CpuWavefrontWorkerCount = std::uint32_t{4U};
inline constexpr auto CheckpointSamples =
    std::array<std::uint64_t, 5U>{1U, 4U, 16U, 64U, ImageSamplesPerPixel};
inline constexpr auto QualityTargets = renderer::QualityThresholds{
    .maximum_linear_mse = 0.003,
    .minimum_display_psnr = 25.0,
};

[[nodiscard]] core::Error benchmark_error(std::string message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = std::move(message),
    };
}

template <renderer::AccumulationPrecision Precision>
[[nodiscard]] core::Status render_scalar_reference_sample_range(
    renderer::FilmT<Precision>& film, const std::uint64_t first_sample,
    const std::uint64_t sample_end, const std::uint64_t seed, const renderer::PinholeCamera& camera,
    const renderer::SampledWavelengths& wavelengths, const AccelBackend& acceleration,
    const renderer::LightSampler& light_sampler) {
    if (first_sample >= sample_end) {
        return std::unexpected(benchmark_error("A Cornell render batch must be non-empty."));
    }
    const auto initial_state =
        renderer::PathState::create_initial(wavelengths, renderer::VacuumMedium);
    if (!initial_state) {
        return std::unexpected(initial_state.error());
    }
    const auto independent_sampler = renderer::IndependentSampler{seed};
    using Film = renderer::FilmT<Precision>;

    for (auto sample_index = first_sample; sample_index < sample_end; ++sample_index) {
        for (auto y = std::uint32_t{}; y < ImageExtent.height; ++y) {
            for (auto x = std::uint32_t{}; x < ImageExtent.width; ++x) {
                const auto index = renderer::PixelSampleIndex{
                    .pixel_x = x,
                    .pixel_y = y,
                    .sample_index = sample_index,
                    .seed = seed,
                };
                const auto ray = camera.generate_primary_ray(
                    index, renderer::PixelJitterMode::uniform, PathTime);
                if (!ray) {
                    return std::unexpected(ray.error());
                }
                const auto stream = independent_sampler.make_stream(x, y, sample_index);
                const auto traced = trace_scene_mis(*ray, *initial_state, stream, acceleration,
                                                    light_sampler, renderer::MisHeuristic::power,
                                                    renderer::PathDepthLimits{.diffuse = 5U},
                                                    renderer::RussianRoulettePolicy::disabled());
                if (!traced) {
                    return std::unexpected(traced.error());
                }
                const auto xyz = renderer::cie_1931_spectrum_to_xyz(
                    traced->state.accumulated_radiance(), wavelengths);
                if (!xyz) {
                    return std::unexpected(xyz.error());
                }
                const auto rgb = renderer::xyz_to_linear_rgb(*xyz);
                if (!rgb) {
                    return std::unexpected(rgb.error());
                }
                const auto color = typename Film::Color{
                    .red = static_cast<typename Film::Scalar>(rgb->red),
                    .green = static_cast<typename Film::Scalar>(rgb->green),
                    .blue = static_cast<typename Film::Scalar>(rgb->blue),
                };
                const auto status =
                    film.add_sample(x, y, color, static_cast<typename Film::Scalar>(1));
                if (!status) {
                    return std::unexpected(status.error());
                }
            }
        }
    }
    return {};
}

[[nodiscard]] core::Status render_cpu_wavefront_sample_range(
    renderer::Film& film, const std::uint64_t first_sample, const std::uint64_t sample_end,
    const std::uint64_t seed, const renderer::PinholeCamera& camera,
    const renderer::SampledWavelengths& wavelengths,
    const renderer::CpuWavefrontScheduler& scheduler, const AccelBackend& acceleration,
    const renderer::LightSampler& light_sampler) {
    if (first_sample >= sample_end) {
        return std::unexpected(benchmark_error("A Cornell render batch must be non-empty."));
    }
    const auto initial_state =
        renderer::PathState::create_initial(wavelengths, renderer::VacuumMedium);
    if (!initial_state) {
        return std::unexpected(initial_state.error());
    }
    const auto independent_sampler = renderer::IndependentSampler{seed};
    const auto path_count =
        static_cast<std::size_t>(ImageExtent.width) * static_cast<std::size_t>(ImageExtent.height);
    auto inputs = std::vector<CpuWavefrontMisPathInput>{};
    inputs.reserve(path_count);

    for (auto sample_index = first_sample; sample_index < sample_end; ++sample_index) {
        inputs.clear();
        for (auto y = std::uint32_t{}; y < ImageExtent.height; ++y) {
            for (auto x = std::uint32_t{}; x < ImageExtent.width; ++x) {
                const auto index = renderer::PixelSampleIndex{
                    .pixel_x = x,
                    .pixel_y = y,
                    .sample_index = sample_index,
                    .seed = seed,
                };
                const auto ray = camera.generate_primary_ray(
                    index, renderer::PixelJitterMode::uniform, PathTime);
                const auto cone = camera.generate_primary_ray_cone(
                    index, renderer::PixelJitterMode::uniform, PathTime);
                if (!ray) {
                    return std::unexpected(ray.error());
                }
                if (!cone) {
                    return std::unexpected(cone.error());
                }
                inputs.push_back(CpuWavefrontMisPathInput{
                    .primary_ray = *ray,
                    .primary_cone = *cone,
                    .initial_state = *initial_state,
                    .sample = independent_sampler.make_stream(x, y, sample_index).index(),
                });
            }
        }

        const auto traced = trace_cpu_wavefront_mis(
            inputs, scheduler, acceleration, light_sampler, renderer::MisHeuristic::power,
            renderer::PathDepthLimits{.diffuse = 5U}, renderer::RussianRoulettePolicy::disabled());
        if (!traced) {
            return std::unexpected(traced.error());
        }
        if (traced->paths.size() != inputs.size() ||
            traced->terminal_cones.size() != inputs.size() ||
            traced->report.path_count != inputs.size() ||
            traced->report.schema_version != CurrentCpuWavefrontMisReportSchemaVersion ||
            traced->report.configured_workers != scheduler.worker_count()) {
            return std::unexpected(benchmark_error(
                "CPU wavefront Cornell transport returned an inconsistent batch report."));
        }

        for (auto lane = std::size_t{}; lane < traced->paths.size(); ++lane) {
            const auto xyz = renderer::cie_1931_spectrum_to_xyz(
                traced->paths[lane].state.accumulated_radiance(), wavelengths);
            if (!xyz) {
                return std::unexpected(xyz.error());
            }
            const auto rgb = renderer::xyz_to_linear_rgb(*xyz);
            if (!rgb) {
                return std::unexpected(rgb.error());
            }
            const auto x = static_cast<std::uint32_t>(lane % ImageExtent.width);
            const auto y = static_cast<std::uint32_t>(lane / ImageExtent.width);
            const auto status = film.add_sample(x, y,
                                                renderer::Film::Color{
                                                    .red = rgb->red,
                                                    .green = rgb->green,
                                                    .blue = rgb->blue,
                                                },
                                                1.0F);
            if (!status) {
                return std::unexpected(status.error());
            }
        }
    }
    return {};
}

void record_checkpoint_counter(benchmark::State& state, const std::uint64_t samples_per_pixel,
                               const double median_mse, const double mse_mad,
                               const double median_psnr, const double psnr_mad) {
    const auto suffix = std::to_string(samples_per_pixel) + "_spp";
    state.counters["mse_at_" + suffix] = median_mse;
    state.counters["mse_at_" + suffix + "_mad"] = mse_mad;
    state.counters["psnr_at_" + suffix] = median_psnr;
    state.counters["psnr_at_" + suffix + "_mad"] = psnr_mad;
}

struct SeedQualityResult final {
    std::chrono::nanoseconds time_to_mse{};
    std::chrono::nanoseconds time_to_psnr{};
    std::uint64_t samples_to_mse{};
    std::uint64_t samples_to_psnr{};
    double observed_mse{};
    double observed_psnr{};
    renderer::LinearMetrics final_linear{};
    double final_psnr{};
    std::array<double, CheckpointSamples.size()> checkpoint_mse{};
    std::array<double, CheckpointSamples.size()> checkpoint_psnr{};
};

struct ScalarDistribution final {
    double median{};
    double median_absolute_deviation{};
    double minimum{};
    double maximum{};
};

template <std::size_t Size> [[nodiscard]] double median(std::array<double, Size> values) noexcept {
    static_assert(Size > 0U);
    std::sort(values.begin(), values.end());
    constexpr auto middle = Size / 2U;
    if constexpr ((Size % 2U) == 0U) {
        return values[middle - 1U] + (values[middle] - values[middle - 1U]) * 0.5;
    } else {
        return values[middle];
    }
}

template <std::size_t Size>
[[nodiscard]] ScalarDistribution summarize(std::array<double, Size> values) noexcept {
    static_assert(Size > 0U);
    const auto distribution_median = median(values);
    auto deviations = std::array<double, Size>{};
    for (auto index = std::size_t{}; index < Size; ++index) {
        deviations[index] = std::abs(values[index] - distribution_median);
    }
    const auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
    return ScalarDistribution{
        .median = distribution_median,
        .median_absolute_deviation = median(deviations),
        .minimum = *minimum,
        .maximum = *maximum,
    };
}

template <std::size_t Size, typename Projection>
[[nodiscard]] ScalarDistribution summarize(const std::array<SeedQualityResult, Size>& results,
                                           Projection projection) noexcept {
    auto values = std::array<double, Size>{};
    for (auto index = std::size_t{}; index < Size; ++index) {
        values[index] = projection(results[index]);
    }
    return summarize(values);
}

template <std::size_t Size, typename Projection>
[[nodiscard]] std::uint64_t upper_median_samples(const std::array<SeedQualityResult, Size>& results,
                                                 Projection projection) noexcept {
    static_assert(Size > 0U);
    auto values = std::array<std::uint64_t, Size>{};
    for (auto index = std::size_t{}; index < Size; ++index) {
        values[index] = projection(results[index]);
    }
    std::sort(values.begin(), values.end());
    return values[Size / 2U];
}

struct PixelRegion final {
    std::uint32_t minimum_x{};
    std::uint32_t minimum_y{};
    std::uint32_t maximum_x{};
    std::uint32_t maximum_y{};
};

[[nodiscard]] constexpr std::uint32_t
scale_preview_coordinate(const std::uint32_t coordinate, const std::uint32_t extent) noexcept {
    constexpr auto canonical_extent = std::uint64_t{128U};
    return static_cast<std::uint32_t>(static_cast<std::uint64_t>(coordinate) * extent /
                                      canonical_extent);
}

[[nodiscard]] constexpr PixelRegion scaled_preview_region(const std::uint32_t minimum_x,
                                                          const std::uint32_t minimum_y,
                                                          const std::uint32_t maximum_x,
                                                          const std::uint32_t maximum_y) noexcept {
    return PixelRegion{
        .minimum_x = scale_preview_coordinate(minimum_x, ImageExtent.width),
        .minimum_y = scale_preview_coordinate(minimum_y, ImageExtent.height),
        .maximum_x = scale_preview_coordinate(maximum_x, ImageExtent.width),
        .maximum_y = scale_preview_coordinate(maximum_y, ImageExtent.height),
    };
}

struct RegionMean final {
    double red{};
    double green{};
    double blue{};
};

struct PreviewSemantics final {
    double mean_luminance{};
    double left_wall_red_ratio{};
    double right_wall_green_ratio{};
    double neutral_enclosure_channel_ratio{};
    double left_sphere_luminance_ratio{};
    double right_sphere_luminance_ratio{};
};

[[nodiscard]] core::Result<RegionMean> mean_region(const renderer::Film& film,
                                                   const PixelRegion region) {
    const auto extent = film.extent();
    if (region.minimum_x >= region.maximum_x || region.minimum_y >= region.maximum_y ||
        region.maximum_x > extent.width || region.maximum_y > extent.height) {
        return std::unexpected(benchmark_error("A Cornell semantic image region is invalid."));
    }

    auto result = RegionMean{};
    for (auto y = region.minimum_y; y < region.maximum_y; ++y) {
        for (auto x = region.minimum_x; x < region.maximum_x; ++x) {
            const auto pixel = film.resolved_pixel(x, y);
            if (!pixel) {
                return std::unexpected(pixel.error());
            }
            result.red += static_cast<double>(pixel->red);
            result.green += static_cast<double>(pixel->green);
            result.blue += static_cast<double>(pixel->blue);
        }
    }
    const auto pixel_count = static_cast<double>(region.maximum_x - region.minimum_x) *
                             static_cast<double>(region.maximum_y - region.minimum_y);
    result.red /= pixel_count;
    result.green /= pixel_count;
    result.blue /= pixel_count;
    return result;
}

[[nodiscard]] core::Result<PreviewSemantics>
validate_preview_semantics(const renderer::Film& film) {
    const auto whole_image = mean_region(
        film, PixelRegion{.maximum_x = ImageExtent.width, .maximum_y = ImageExtent.height});
    if (!whole_image) {
        return std::unexpected(whole_image.error());
    }
    const auto left_wall = mean_region(film, scaled_preview_region(4U, 28U, 20U, 85U));
    if (!left_wall) {
        return std::unexpected(left_wall.error());
    }
    const auto right_wall = mean_region(film, scaled_preview_region(108U, 28U, 124U, 85U));
    if (!right_wall) {
        return std::unexpected(right_wall.error());
    }
    const auto neutral_enclosure = mean_region(film, scaled_preview_region(48U, 24U, 80U, 50U));
    if (!neutral_enclosure) {
        return std::unexpected(neutral_enclosure.error());
    }
    const auto left_sphere = mean_region(film, scaled_preview_region(31U, 85U, 46U, 101U));
    if (!left_sphere) {
        return std::unexpected(left_sphere.error());
    }
    const auto right_sphere = mean_region(film, scaled_preview_region(80U, 78U, 96U, 96U));
    if (!right_sphere) {
        return std::unexpected(right_sphere.error());
    }

    const auto mean_luminance =
        0.2126 * whole_image->red + 0.7152 * whole_image->green + 0.0722 * whole_image->blue;
    const auto left_wall_red_ratio = left_wall->red / std::max(left_wall->green, left_wall->blue);
    const auto right_wall_green_ratio =
        right_wall->green / std::max(right_wall->red, right_wall->blue);
    const auto neutral_minimum =
        std::min({neutral_enclosure->red, neutral_enclosure->green, neutral_enclosure->blue});
    const auto neutral_maximum =
        std::max({neutral_enclosure->red, neutral_enclosure->green, neutral_enclosure->blue});
    const auto neutral_luminance = 0.2126 * neutral_enclosure->red +
                                   0.7152 * neutral_enclosure->green +
                                   0.0722 * neutral_enclosure->blue;
    const auto left_sphere_luminance =
        0.2126 * left_sphere->red + 0.7152 * left_sphere->green + 0.0722 * left_sphere->blue;
    const auto right_sphere_luminance =
        0.2126 * right_sphere->red + 0.7152 * right_sphere->green + 0.0722 * right_sphere->blue;
    if (!std::isfinite(mean_luminance) || !std::isfinite(left_wall_red_ratio) ||
        !std::isfinite(right_wall_green_ratio) || !std::isfinite(neutral_minimum) ||
        !std::isfinite(neutral_maximum) || !std::isfinite(neutral_luminance) ||
        !std::isfinite(left_sphere_luminance) || !std::isfinite(right_sphere_luminance)) {
        return std::unexpected(
            benchmark_error("The Cornell semantic image measurements must be finite."));
    }
    if (!(mean_luminance > 0.01)) {
        return std::unexpected(benchmark_error("The Cornell preview is unexpectedly black."));
    }
    if (!(left_wall->red > 0.02) || !(left_wall_red_ratio >= 2.0)) {
        return std::unexpected(
            benchmark_error("The Cornell preview does not contain the expected red left wall."));
    }
    if (!(right_wall->green > 0.02) || !(right_wall_green_ratio >= 1.5)) {
        return std::unexpected(
            benchmark_error("The Cornell preview does not contain the expected green right wall."));
    }
    if (!(neutral_minimum > 0.05) || !(neutral_maximum / neutral_minimum <= 1.5)) {
        return std::unexpected(benchmark_error(
            "The Cornell enclosure is not sufficiently neutral in the fixed control region."));
    }
    if (!(left_sphere_luminance < neutral_luminance * 0.65) ||
        !(right_sphere_luminance < neutral_luminance * 0.65)) {
        return std::unexpected(benchmark_error(
            "The Cornell preview does not contain both expected sphere silhouettes."));
    }

    return PreviewSemantics{
        .mean_luminance = mean_luminance,
        .left_wall_red_ratio = left_wall_red_ratio,
        .right_wall_green_ratio = right_wall_green_ratio,
        .neutral_enclosure_channel_ratio = neutral_maximum / neutral_minimum,
        .left_sphere_luminance_ratio = left_sphere_luminance / neutral_luminance,
        .right_sphere_luminance_ratio = right_sphere_luminance / neutral_luminance,
    };
}

[[nodiscard]] core::Status validate_cornell_scene_contract(const FrameScene& scene) {
    if (scene.instances().size() != 8U || scene.mesh_area_lights().size() != 1U) {
        return std::unexpected(benchmark_error(
            "The canonical Cornell scene must contain five walls, one emitter, and two spheres."));
    }
    const auto sphere_count = std::count_if(
        scene.instances().begin(), scene.instances().end(), [](const SceneInstance& instance) {
            return instance.geometry.value == 17U && instance.material.value == 21U;
        });
    if (sphere_count != 2) {
        return std::unexpected(benchmark_error(
            "The canonical Cornell scene must contain exactly two shared white sphere instances."));
    }
    const auto first_sphere = scene.instance(renderer::InstanceId{.value = 37U});
    const auto second_sphere = scene.instance(renderer::InstanceId{.value = 38U});
    if (!first_sphere) {
        return std::unexpected(first_sphere.error());
    }
    if (!second_sphere) {
        return std::unexpected(second_sphere.error());
    }
    const auto sphere_binding_is_canonical = [](const SceneInstance& instance) {
        return !instance.parent && instance.object.value == 7U && instance.geometry.value == 17U &&
               instance.material.value == 21U;
    };
    if (!sphere_binding_is_canonical(first_sphere->get()) ||
        !sphere_binding_is_canonical(second_sphere->get()) ||
        first_sphere->get().local_to_parent == second_sphere->get().local_to_parent) {
        return std::unexpected(benchmark_error(
            "The canonical Cornell spheres must be two distinct instances of one shared mesh."));
    }
    const auto sphere_geometry = scene.geometry(renderer::GeometryId{.value = 17U});
    if (!sphere_geometry) {
        return std::unexpected(sphere_geometry.error());
    }
    if (!sphere_geometry->get().mesh || sphere_geometry->get().mesh->positions().size() != 2017U ||
        sphere_geometry->get().mesh->triangles().size() != 3968U) {
        return std::unexpected(
            benchmark_error("The canonical Cornell sphere mesh topology is not exact."));
    }
    const auto sphere_material = scene.material(renderer::MaterialId{.value = 21U});
    if (!sphere_material) {
        return std::unexpected(sphere_material.error());
    }
    if (!sphere_material->get().spectral) {
        return std::unexpected(
            benchmark_error("The canonical Cornell sphere material must be spectral."));
    }
    const auto& sphere_closure_mixture = sphere_material->get().spectral->closure_mixture;
    const auto sphere_closures = sphere_closure_mixture.closures.closures();
    const auto sphere_component_probabilities =
        sphere_closure_mixture.active_component_probabilities();
    if (sphere_closures.size() != 1U || sphere_component_probabilities.size() != 1U ||
        sphere_closures.front().kind != renderer::ClosureKind::lambertian_reflection ||
        sphere_component_probabilities.front() != 1.0F) {
        return std::unexpected(benchmark_error(
            "The two canonical Cornell spheres must use one explicit Lambertian closure."));
    }
    for (const auto reflectance : sphere_closures.front().weight.values) {
        if (reflectance != 0.72F) {
            return std::unexpected(benchmark_error(
                "The two canonical Cornell spheres must use the same white Lambertian closure."));
        }
    }
    for (const auto emission : sphere_material->get().spectral->emitted_radiance.values) {
        if (emission != 0.0F) {
            return std::unexpected(
                benchmark_error("The two canonical Cornell spheres must not emit light."));
        }
    }
    return {};
}

void record_queue_statistics_counter(benchmark::State& state, const char* const prefix,
                                     const CpuWavefrontMisQueueStatistics& statistics) {
    const auto name = std::string{prefix};
    state.counters[name + "_capacity"] = static_cast<double>(statistics.capacity);
    state.counters[name + "_peak_size"] = static_cast<double>(statistics.peak_size);
    state.counters[name + "_dispatch_count"] = static_cast<double>(statistics.dispatch_count);
    state.counters[name + "_input_lanes"] = static_cast<double>(statistics.input_lanes);
    state.counters[name + "_peak_occupancy"] = statistics.peak_occupancy();
    state.counters[name + "_mean_occupancy"] = statistics.mean_occupancy();
    state.counters[name + "_overflow_attempts"] = static_cast<double>(statistics.overflow_attempts);
    state.counters[name + "_rejected_lanes"] = static_cast<double>(statistics.rejected_lanes);
    state.counters[name + "_stage_wall_nanoseconds"] =
        static_cast<double>(statistics.stage_wall_nanoseconds);
}

void cornell_wavefront_queue_statistics(benchmark::State& state) {
    const auto scene = cornell_wavefront_test::make_cornell_scene();
    if (!scene) {
        state.SkipWithError(scene.error().message);
        return;
    }
    const auto scene_status = validate_cornell_scene_contract(**scene);
    if (!scene_status) {
        state.SkipWithError(scene_status.error().message);
        return;
    }
    const auto acceleration = create_embree_accel_backend(*scene);
    if (!acceleration) {
        state.SkipWithError(acceleration.error().message);
        return;
    }
    const auto light_sampler =
        renderer::LightSampler::create_uniform((*scene)->mesh_area_lights().size());
    if (!light_sampler) {
        state.SkipWithError(light_sampler.error().message);
        return;
    }
    const auto camera = cornell_wavefront_test::make_camera(ImageExtent);
    if (!camera) {
        state.SkipWithError(camera.error().message);
        return;
    }
    const auto scheduler = renderer::CpuWavefrontScheduler::create(CpuWavefrontWorkerCount);
    if (!scheduler) {
        state.SkipWithError(scheduler.error().message);
        return;
    }
    const auto initial_state = renderer::PathState::create_initial(
        (*scene)->spectral_environment()->wavelengths, renderer::VacuumMedium);
    if (!initial_state) {
        state.SkipWithError(initial_state.error().message);
        return;
    }
    const auto independent_sampler = renderer::IndependentSampler{ImageSeeds.front()};
    const auto path_count =
        static_cast<std::size_t>(ImageExtent.width) * static_cast<std::size_t>(ImageExtent.height);
    auto inputs = std::vector<CpuWavefrontMisPathInput>{};
    inputs.reserve(path_count);
    for (auto y = std::uint32_t{}; y < ImageExtent.height; ++y) {
        for (auto x = std::uint32_t{}; x < ImageExtent.width; ++x) {
            const auto index = renderer::PixelSampleIndex{
                .pixel_x = x,
                .pixel_y = y,
                .sample_index = 0U,
                .seed = ImageSeeds.front(),
            };
            const auto ray =
                camera->generate_primary_ray(index, renderer::PixelJitterMode::uniform, PathTime);
            const auto cone = camera->generate_primary_ray_cone(
                index, renderer::PixelJitterMode::uniform, PathTime);
            if (!ray) {
                state.SkipWithError(ray.error().message);
                return;
            }
            if (!cone) {
                state.SkipWithError(cone.error().message);
                return;
            }
            inputs.push_back(CpuWavefrontMisPathInput{
                .primary_ray = *ray,
                .primary_cone = *cone,
                .initial_state = *initial_state,
                .sample = independent_sampler.make_stream(x, y, 0U).index(),
            });
        }
    }

    auto latest_report = std::optional<CpuWavefrontMisReport>{};
    for (auto _ : state) {
        static_cast<void>(_);
        const auto traced = trace_cpu_wavefront_mis(
            inputs, *scheduler, **acceleration, *light_sampler, renderer::MisHeuristic::power,
            renderer::PathDepthLimits{.diffuse = 5U}, renderer::RussianRoulettePolicy::disabled());
        if (!traced) {
            state.SkipWithError(traced.error().message);
            return;
        }
        if (traced->paths.size() != inputs.size() ||
            traced->terminal_cones.size() != inputs.size() ||
            traced->report.path_count != inputs.size() ||
            traced->report.schema_version != CurrentCpuWavefrontMisReportSchemaVersion ||
            traced->report.configured_workers != scheduler->worker_count()) {
            state.SkipWithError(
                "CPU wavefront queue statistics received an inconsistent transport report.");
            return;
        }
        latest_report = traced->report;
        benchmark::DoNotOptimize(traced->paths.data());
        benchmark::ClobberMemory();
    }
    if (!latest_report) {
        state.SkipWithError("CPU wavefront queue statistics produced no report.");
        return;
    }

    const auto statistics = std::array{
        &latest_report->queue_statistics.camera,       &latest_report->queue_statistics.ray,
        &latest_report->queue_statistics.hit,          &latest_report->queue_statistics.miss,
        &latest_report->queue_statistics.shade,        &latest_report->queue_statistics.shadow,
        &latest_report->queue_statistics.continuation,
    };
    auto total_stage_wall_nanoseconds = std::uint64_t{};
    for (const auto* const queue : statistics) {
        if (queue->stage_wall_nanoseconds >
            std::numeric_limits<std::uint64_t>::max() - total_stage_wall_nanoseconds) {
            state.SkipWithError("CPU wavefront benchmark stage-time sum is not representable.");
            return;
        }
        total_stage_wall_nanoseconds += queue->stage_wall_nanoseconds;
    }

    state.counters["report_schema_version"] = static_cast<double>(latest_report->schema_version);
    state.counters["configured_workers"] = static_cast<double>(latest_report->configured_workers);
    state.counters["path_count"] = static_cast<double>(latest_report->path_count);
    state.counters["closure_samples"] = static_cast<double>(latest_report->closure_samples);
    state.counters["light_samples"] = static_cast<double>(latest_report->light_samples);
    state.counters["shadow_queries"] = static_cast<double>(latest_report->shadow_queries);
    state.counters["total_stage_wall_nanoseconds"] =
        static_cast<double>(total_stage_wall_nanoseconds);
    record_queue_statistics_counter(state, "camera", latest_report->queue_statistics.camera);
    record_queue_statistics_counter(state, "ray", latest_report->queue_statistics.ray);
    record_queue_statistics_counter(state, "hit", latest_report->queue_statistics.hit);
    record_queue_statistics_counter(state, "miss", latest_report->queue_statistics.miss);
    record_queue_statistics_counter(state, "shade", latest_report->queue_statistics.shade);
    record_queue_statistics_counter(state, "shadow", latest_report->queue_statistics.shadow);
    record_queue_statistics_counter(state, "continuation",
                                    latest_report->queue_statistics.continuation);
}

void cornell_power_mis_convergence(benchmark::State& state) {
    const auto scene = cornell_wavefront_test::make_cornell_scene();
    if (!scene) {
        state.SkipWithError(scene.error().message);
        return;
    }
    const auto scene_status = validate_cornell_scene_contract(**scene);
    if (!scene_status) {
        state.SkipWithError(scene_status.error().message);
        return;
    }
    const auto acceleration = create_embree_accel_backend(*scene);
    if (!acceleration) {
        state.SkipWithError(acceleration.error().message);
        return;
    }
    const auto light_sampler =
        renderer::LightSampler::create_uniform((*scene)->mesh_area_lights().size());
    if (!light_sampler) {
        state.SkipWithError(light_sampler.error().message);
        return;
    }
    const auto camera = cornell_wavefront_test::make_camera(ImageExtent);
    if (!camera) {
        state.SkipWithError(camera.error().message);
        return;
    }
    const auto wavelengths = (*scene)->spectral_environment()->wavelengths;
    const auto scheduler = renderer::CpuWavefrontScheduler::create(CpuWavefrontWorkerCount);
    if (!scheduler) {
        state.SkipWithError(scheduler.error().message);
        return;
    }
    auto reference = renderer::ReferenceFilm::create(ImageExtent);
    if (!reference) {
        state.SkipWithError(reference.error().message);
        return;
    }
    const auto reference_status = render_scalar_reference_sample_range(
        *reference, 0U, ReferenceSamplesPerPixel, ReferenceSeed, *camera, wavelengths,
        **acceleration, *light_sampler);
    if (!reference_status) {
        state.SkipWithError(reference_status.error().message);
        return;
    }

    for (auto _ : state) {
        static_cast<void>(_);
        state.PauseTiming();
        const auto fail_while_paused = [&state](const std::string& message) {
            state.ResumeTiming();
            state.SkipWithError(message);
        };
        auto results = std::array<SeedQualityResult, ImageSeeds.size()>{};
        auto preview = std::optional<renderer::Film>{};
        for (auto seed_index = std::size_t{}; seed_index < ImageSeeds.size(); ++seed_index) {
            auto evaluated = renderer::Film::create(ImageExtent);
            if (!evaluated) {
                fail_while_paused(evaluated.error().message);
                return;
            }

            auto checkpoints = std::vector<renderer::QualityCheckpoint>{};
            checkpoints.reserve(CheckpointSamples.size());
            auto cumulative_render_time = std::chrono::nanoseconds{};
            auto previous_samples_per_pixel = std::uint64_t{};
            auto final_linear = renderer::LinearMetrics{};
            auto final_psnr = double{};

            for (auto checkpoint_index = std::size_t{}; checkpoint_index < CheckpointSamples.size();
                 ++checkpoint_index) {
                const auto samples_per_pixel = CheckpointSamples[checkpoint_index];
                state.ResumeTiming();
                const auto render_start = Clock::now();
                const auto render_status = render_cpu_wavefront_sample_range(
                    *evaluated, previous_samples_per_pixel, samples_per_pixel,
                    ImageSeeds[seed_index], *camera, wavelengths, *scheduler, **acceleration,
                    *light_sampler);
                const auto render_end = Clock::now();
                state.PauseTiming();
                cumulative_render_time +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(render_end - render_start);
                if (!render_status) {
                    fail_while_paused(render_status.error().message);
                    return;
                }

                const auto linear = renderer::compute_linear_metrics(*evaluated, *reference);
                if (!linear) {
                    fail_while_paused(linear.error().message);
                    return;
                }
                const auto display = renderer::compute_display_psnr(*evaluated, *reference);
                if (!display) {
                    fail_while_paused(display.error().message);
                    return;
                }
                if (!std::isfinite(linear->mse) || !std::isfinite(display->psnr)) {
                    fail_while_paused(
                        "Cornell convergence metrics must remain finite for every image seed.");
                    return;
                }
                checkpoints.push_back(renderer::QualityCheckpoint{
                    .cumulative_samples_per_pixel = samples_per_pixel,
                    .cumulative_render_time = cumulative_render_time,
                    .linear_mse = linear->mse,
                    .display_psnr = display->psnr,
                });
                results[seed_index].checkpoint_mse[checkpoint_index] = linear->mse;
                results[seed_index].checkpoint_psnr[checkpoint_index] = display->psnr;
                final_linear = *linear;
                final_psnr = display->psnr;
                previous_samples_per_pixel = samples_per_pixel;
            }

            const auto time_to_quality =
                renderer::compute_time_to_quality(checkpoints, QualityTargets);
            if (!time_to_quality) {
                fail_while_paused(time_to_quality.error().message);
                return;
            }
            if (!time_to_quality->linear_mse || !time_to_quality->display_psnr) {
                fail_while_paused(
                    "Every Cornell image seed must reach both fixed quality targets.");
                return;
            }
            if (final_linear.mse > QualityTargets.maximum_linear_mse ||
                final_psnr < QualityTargets.minimum_display_psnr) {
                fail_while_paused(
                    "Every final Cornell image must remain beyond both quality targets.");
                return;
            }

            results[seed_index].time_to_mse = time_to_quality->linear_mse->cumulative_render_time;
            results[seed_index].time_to_psnr =
                time_to_quality->display_psnr->cumulative_render_time;
            results[seed_index].samples_to_mse =
                time_to_quality->linear_mse->cumulative_samples_per_pixel;
            results[seed_index].samples_to_psnr =
                time_to_quality->display_psnr->cumulative_samples_per_pixel;
            results[seed_index].observed_mse = time_to_quality->linear_mse->observed_value;
            results[seed_index].observed_psnr = time_to_quality->display_psnr->observed_value;
            results[seed_index].final_linear = final_linear;
            results[seed_index].final_psnr = final_psnr;
            if (seed_index == 0U) {
                preview.emplace(std::move(*evaluated));
            }
        }

        if (!preview) {
            fail_while_paused("The Cornell benchmark did not retain its canonical preview film.");
            return;
        }
        const auto preview_semantics = validate_preview_semantics(*preview);
        if (!preview_semantics) {
            fail_while_paused(preview_semantics.error().message);
            return;
        }

        const auto output_path = std::filesystem::path{BLACKFRAME_CORNELL_BENCHMARK_ARTIFACT_PATH};
        auto filesystem_error = std::error_code{};
        std::filesystem::create_directories(output_path.parent_path(), filesystem_error);
        if (filesystem_error) {
            fail_while_paused("The Cornell preview directory could not be created.");
            return;
        }
        const auto png_status = renderer::write_png_preview(*preview, output_path);
        if (!png_status) {
            fail_while_paused(png_status.error().message);
            return;
        }

        const auto time_to_mse = summarize(results, [](const SeedQualityResult& result) {
            return std::chrono::duration<double>(result.time_to_mse).count();
        });
        const auto time_to_psnr = summarize(results, [](const SeedQualityResult& result) {
            return std::chrono::duration<double>(result.time_to_psnr).count();
        });
        const auto samples_to_mse = summarize(results, [](const SeedQualityResult& result) {
            return static_cast<double>(result.samples_to_mse);
        });
        const auto samples_to_psnr = summarize(results, [](const SeedQualityResult& result) {
            return static_cast<double>(result.samples_to_psnr);
        });
        const auto observed_mse =
            summarize(results, [](const SeedQualityResult& result) { return result.observed_mse; });
        const auto observed_psnr = summarize(
            results, [](const SeedQualityResult& result) { return result.observed_psnr; });
        const auto final_mse = summarize(
            results, [](const SeedQualityResult& result) { return result.final_linear.mse; });
        const auto final_rmse = summarize(
            results, [](const SeedQualityResult& result) { return result.final_linear.rmse; });
        const auto final_bias = summarize(
            results, [](const SeedQualityResult& result) { return result.final_linear.mean_bias; });
        const auto final_maximum_error = summarize(results, [](const SeedQualityResult& result) {
            return result.final_linear.maximum_absolute_error;
        });
        const auto final_psnr =
            summarize(results, [](const SeedQualityResult& result) { return result.final_psnr; });

        for (auto checkpoint_index = std::size_t{}; checkpoint_index < CheckpointSamples.size();
             ++checkpoint_index) {
            const auto checkpoint_mse =
                summarize(results, [checkpoint_index](const SeedQualityResult& result) {
                    return result.checkpoint_mse[checkpoint_index];
                });
            const auto checkpoint_psnr =
                summarize(results, [checkpoint_index](const SeedQualityResult& result) {
                    return result.checkpoint_psnr[checkpoint_index];
                });
            record_checkpoint_counter(
                state, CheckpointSamples[checkpoint_index], checkpoint_mse.median,
                checkpoint_mse.median_absolute_deviation, checkpoint_psnr.median,
                checkpoint_psnr.median_absolute_deviation);
        }

        state.counters["time_to_mse_seconds"] = time_to_mse.median;
        state.counters["time_to_mse_mad_seconds"] = time_to_mse.median_absolute_deviation;
        state.counters["time_to_psnr_seconds"] = time_to_psnr.median;
        state.counters["time_to_psnr_mad_seconds"] = time_to_psnr.median_absolute_deviation;
        state.counters["samples_to_mse"] = static_cast<double>(upper_median_samples(
            results, [](const SeedQualityResult& result) { return result.samples_to_mse; }));
        state.counters["samples_to_mse_min"] = samples_to_mse.minimum;
        state.counters["samples_to_mse_max"] = samples_to_mse.maximum;
        state.counters["samples_to_psnr"] = static_cast<double>(upper_median_samples(
            results, [](const SeedQualityResult& result) { return result.samples_to_psnr; }));
        state.counters["samples_to_psnr_min"] = samples_to_psnr.minimum;
        state.counters["samples_to_psnr_max"] = samples_to_psnr.maximum;
        state.counters["target_mse"] = QualityTargets.maximum_linear_mse;
        state.counters["target_psnr"] = QualityTargets.minimum_display_psnr;
        state.counters["observed_mse_at_threshold"] = observed_mse.median;
        state.counters["observed_psnr_at_threshold"] = observed_psnr.median;
        state.counters["final_mse"] = final_mse.median;
        state.counters["final_mse_mad"] = final_mse.median_absolute_deviation;
        state.counters["final_mse_max"] = final_mse.maximum;
        state.counters["final_rmse"] = final_rmse.median;
        state.counters["final_rmse_mad"] = final_rmse.median_absolute_deviation;
        state.counters["final_bias_mean"] = final_bias.median;
        state.counters["final_bias_mean_mad"] = final_bias.median_absolute_deviation;
        state.counters["final_max_abs"] = final_maximum_error.median;
        state.counters["final_max_abs_mad"] = final_maximum_error.median_absolute_deviation;
        state.counters["final_psnr"] = final_psnr.median;
        state.counters["final_psnr_mad"] = final_psnr.median_absolute_deviation;
        state.counters["final_psnr_min"] = final_psnr.minimum;
        state.counters["reference_spp"] = static_cast<double>(ReferenceSamplesPerPixel);
        state.counters["image_spp"] = static_cast<double>(ImageSamplesPerPixel);
        state.counters["seed_count"] = static_cast<double>(ImageSeeds.size());
        state.counters["cpu_wavefront_workers"] = static_cast<double>(scheduler->worker_count());
        state.counters["scene_instance_count"] = static_cast<double>((*scene)->instances().size());
        state.counters["sphere_instance_count"] = 2.0;
        state.counters["image_mean_luminance"] = preview_semantics->mean_luminance;
        state.counters["left_wall_red_ratio"] = preview_semantics->left_wall_red_ratio;
        state.counters["right_wall_green_ratio"] = preview_semantics->right_wall_green_ratio;
        state.counters["neutral_enclosure_channel_ratio"] =
            preview_semantics->neutral_enclosure_channel_ratio;
        state.counters["left_sphere_luminance_ratio"] =
            preview_semantics->left_sphere_luminance_ratio;
        state.counters["right_sphere_luminance_ratio"] =
            preview_semantics->right_sphere_luminance_ratio;
        state.SetItemsProcessed(static_cast<std::int64_t>(ImageExtent.width) *
                                static_cast<std::int64_t>(ImageExtent.height) *
                                static_cast<std::int64_t>(ImageSamplesPerPixel) *
                                static_cast<std::int64_t>(ImageSeeds.size()));
        state.ResumeTiming();
    }
}

BENCHMARK(cornell_power_mis_convergence);
BENCHMARK(cornell_wavefront_queue_statistics);

} // namespace
} // namespace blackframe::engine
