#include "../../CPU/Embree/CornellWavefrontScene.hpp"

#include <Blackframe/Backends/GPU/CUDA/SceneBvh.hpp>
#include <Blackframe/Backends/GPU/CUDA/SceneSoA.hpp>
#include <Blackframe/Backends/GPU/CUDA/WavefrontTransport.hpp>
#include <Blackframe/Renderer/Cie1931Sensor.hpp>
#include <Blackframe/Renderer/Color.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <Blackframe/Renderer/PixelJitter.hpp>
#if defined(BLACKFRAME_CUDA_CORNELL_PNG)
#include <Blackframe/Renderer/PngWriter.hpp>
#endif
#include <Blackframe/Renderer/Ray.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#if !defined(BLACKFRAME_CUDA_TEST_BINARY_DIR)
#error "The CUDA Cornell smoke test requires its configured build directory."
#endif

namespace blackframe::engine {
namespace {

constexpr auto CornellExtent = renderer::RenderExtent{.width = 64U, .height = 64U};
constexpr auto CornellSamplesPerPixel = std::uint32_t{4U};
constexpr auto CornellSeed = std::uint64_t{0x243F6A8885A308D3ULL};
constexpr auto CornellPathTime = renderer::TransportScalar{0.5F};
constexpr auto CornellMaximumDiffuseDepth = std::uint32_t{5U};
constexpr auto DiagnosticCornellExtent = renderer::RenderExtent{.width = 800U, .height = 800U};
constexpr auto DiagnosticCornellSamplesPerPixel = std::uint32_t{16U};

[[nodiscard]] testing::AssertionResult select_test_device() {
    auto device_count = int{};
    const auto count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess) {
        return testing::AssertionFailure()
               << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_status);
    }
    if (device_count <= 0) {
        return testing::AssertionFailure() << "No CUDA device is available.";
    }
    const auto select_status = cudaSetDevice(0);
    if (select_status != cudaSuccess) {
        return testing::AssertionFailure()
               << "cudaSetDevice failed: " << cudaGetErrorString(select_status);
    }
    return testing::AssertionSuccess();
}

[[nodiscard]] core::Result<std::vector<CudaWavefrontPathInput>>
make_inputs(const renderer::PinholeCamera& camera, const renderer::SampledWavelengths& wavelengths,
            const renderer::RenderExtent extent, const std::uint64_t first_sample_index,
            const std::uint32_t samples_in_batch,
            const renderer::MediumId medium = renderer::VacuumMedium) {
    const auto initial_state = renderer::PathState::create_initial(wavelengths, medium);
    if (!initial_state) {
        return std::unexpected(initial_state.error());
    }

    const auto sampler = renderer::IndependentSampler{CornellSeed};
    auto inputs = std::vector<CudaWavefrontPathInput>{};
    inputs.reserve(static_cast<std::size_t>(extent.width) * extent.height * samples_in_batch);
    for (auto sample_offset = std::uint32_t{}; sample_offset < samples_in_batch; ++sample_offset) {
        const auto sample_index = first_sample_index + sample_offset;
        for (auto pixel_y = std::uint32_t{}; pixel_y < extent.height; ++pixel_y) {
            for (auto pixel_x = std::uint32_t{}; pixel_x < extent.width; ++pixel_x) {
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
                const auto ray = renderer::Ray::create(primary->origin(), primary->direction(),
                                                       primary->t_min(), primary->t_max(),
                                                       primary->time(), primary->mask(), medium);
                if (!ray) {
                    return std::unexpected(ray.error());
                }
                inputs.push_back(CudaWavefrontPathInput{
                    .primary_ray = *ray,
                    .initial_state = *initial_state,
                    .sample = sampler.make_stream(pixel_x, pixel_y, sample_index).index(),
                });
            }
        }
    }
    return inputs;
}

[[nodiscard]] core::Result<std::vector<CudaWavefrontPathInput>>
make_pixel_inputs(const renderer::PinholeCamera& camera,
                  const renderer::SampledWavelengths& wavelengths, const std::uint32_t pixel_x,
                  const std::uint32_t pixel_y, const std::uint64_t first_sample_index,
                  const std::uint32_t sample_count) {
    const auto initial_state =
        renderer::PathState::create_initial(wavelengths, renderer::VacuumMedium);
    if (!initial_state) {
        return std::unexpected(initial_state.error());
    }
    const auto sampler = renderer::IndependentSampler{CornellSeed};
    auto inputs = std::vector<CudaWavefrontPathInput>{};
    inputs.reserve(sample_count);
    for (auto sample_offset = std::uint32_t{}; sample_offset < sample_count; ++sample_offset) {
        const auto sample_index = first_sample_index + sample_offset;
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
        const auto ray = renderer::Ray::create(primary->origin(), primary->direction(),
                                               primary->t_min(), primary->t_max(), primary->time(),
                                               primary->mask(), renderer::VacuumMedium);
        if (!ray) {
            return std::unexpected(ray.error());
        }
        inputs.push_back(CudaWavefrontPathInput{
            .primary_ray = *ray,
            .initial_state = *initial_state,
            .sample = sampler.make_stream(pixel_x, pixel_y, sample_index).index(),
        });
    }
    return inputs;
}

[[nodiscard]] std::optional<std::filesystem::path> diagnostic_output_path() {
#if defined(_WIN32)
    auto* value = static_cast<char*>(nullptr);
    auto value_size = std::size_t{};
    if (_dupenv_s(&value, &value_size, "BLACKFRAME_CUDA_CORNELL_OUTPUT") != 0 || value == nullptr) {
        return std::nullopt;
    }
    const auto path = value_size > 1U ? std::optional{std::filesystem::path{value}} : std::nullopt;
    std::free(value);
    return path;
#else
    const auto* const value = std::getenv("BLACKFRAME_CUDA_CORNELL_OUTPUT");
    if (value == nullptr || *value == '\0') {
        return std::nullopt;
    }
    return std::filesystem::path{value};
#endif
}

[[nodiscard]] bool diagnostic_path_is_below_build(const std::filesystem::path& path) {
    auto error = std::error_code{};
    const auto build_root =
        std::filesystem::weakly_canonical(BLACKFRAME_CUDA_TEST_BINARY_DIR, error);
    if (error) {
        return false;
    }
    const auto output = std::filesystem::weakly_canonical(path, error);
    if (error) {
        return false;
    }
    auto root_component = build_root.begin();
    auto output_component = output.begin();
    for (; root_component != build_root.end() && output_component != output.end();
         ++root_component, ++output_component) {
        if (*root_component != *output_component) {
            return false;
        }
    }
    return root_component == build_root.end() && output_component != output.end();
}

struct RegionSum final {
    double red{};
    double green{};
    double blue{};
    std::size_t count{};

    void add(const renderer::LinearRGB color) noexcept {
        red += color.red;
        green += color.green;
        blue += color.blue;
        ++count;
    }

    [[nodiscard]] double luminance() const noexcept {
        return (0.2126 * red + 0.7152 * green + 0.0722 * blue) / static_cast<double>(count);
    }
};

void add_report(CudaWavefrontTransportReport& destination,
                const CudaWavefrontTransportReport& source) noexcept {
    destination.path_count += source.path_count;
    destination.stage_lanes.camera += source.stage_lanes.camera;
    destination.stage_lanes.intersection += source.stage_lanes.intersection;
    destination.stage_lanes.hit += source.stage_lanes.hit;
    destination.stage_lanes.miss += source.stage_lanes.miss;
    destination.stage_lanes.shade += source.stage_lanes.shade;
    destination.stage_lanes.shadow += source.stage_lanes.shadow;
    destination.stage_lanes.continuation += source.stage_lanes.continuation;
    destination.light_samples += source.light_samples;
    destination.shadow_queries += source.shadow_queries;
    destination.terminated_paths += source.terminated_paths;
    destination.queue_overflow_attempts += source.queue_overflow_attempts;
    destination.queue_rejected_lanes += source.queue_rejected_lanes;
}

[[nodiscard]] bool finite_non_negative(const renderer::TransportSpectrum& spectrum) noexcept {
    return std::ranges::all_of(spectrum.values,
                               [](const auto value) { return std::isfinite(value) && value >= 0; });
}

class CudaCornellWavefrontSmokeTest : public testing::Test {
  protected:
    void SetUp() override {
        ASSERT_TRUE(select_test_device());

        const auto created_scene = cornell_wavefront_test::make_cornell_scene();
        ASSERT_TRUE(created_scene.has_value()) << created_scene.error().message;
        scene_ = *created_scene;

        auto uploaded = CudaSceneSoA::upload(*scene_);
        ASSERT_TRUE(uploaded.has_value()) << uploaded.error().message;
        scene_soa_.emplace(std::move(*uploaded));

        auto built = CudaSceneBvh::build(*scene_soa_);
        ASSERT_TRUE(built.has_value()) << built.error().message;
        scene_bvh_.emplace(std::move(*built));
    }

    void TearDown() override {
        if (scene_bvh_) {
            const auto status = scene_bvh_->close();
            EXPECT_TRUE(status.has_value()) << status.error().message;
        }
        if (scene_soa_) {
            const auto status = scene_soa_->close();
            EXPECT_TRUE(status.has_value()) << status.error().message;
        }
    }

    [[nodiscard]] const renderer::SampledWavelengths& wavelengths() const {
        return scene_->spectral_environment()->wavelengths;
    }

    FrameSceneHandle scene_{};
    std::optional<CudaSceneSoA> scene_soa_{};
    std::optional<CudaSceneBvh> scene_bvh_{};
};

TEST_F(CudaCornellWavefrontSmokeTest, RendersCanonicalCornellThroughCudaWavefront) {
    ASSERT_TRUE(scene_->spectral_environment().has_value());
    const auto diagnostic_output = diagnostic_output_path();
#if !defined(BLACKFRAME_CUDA_CORNELL_PNG)
    ASSERT_FALSE(diagnostic_output.has_value())
        << "BLACKFRAME_CUDA_CORNELL_OUTPUT requires the explicit PNG capability.";
#endif
    if (diagnostic_output) {
        ASSERT_TRUE(diagnostic_output->is_absolute())
            << "BLACKFRAME_CUDA_CORNELL_OUTPUT must name an absolute path.";
        ASSERT_TRUE(diagnostic_path_is_below_build(*diagnostic_output))
            << "BLACKFRAME_CUDA_CORNELL_OUTPUT must remain below the configured build tree.";
    }
    const auto extent = diagnostic_output ? DiagnosticCornellExtent : CornellExtent;
    const auto samples_per_pixel =
        diagnostic_output ? DiagnosticCornellSamplesPerPixel : CornellSamplesPerPixel;
    const auto samples_per_batch = diagnostic_output ? std::uint32_t{1U} : samples_per_pixel;

    const auto camera = cornell_wavefront_test::make_camera(extent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    auto film = renderer::Film::create(extent);
    ASSERT_TRUE(film.has_value()) << film.error().message;

    auto aggregate_report = CudaWavefrontTransportReport{
        .schema_version = CurrentCudaWavefrontTransportReportSchemaVersion,
        .maximum_diffuse_depth = CornellMaximumDiffuseDepth,
    };
    auto non_black_spectral_paths = std::size_t{};
    for (auto first_sample = std::uint32_t{}; first_sample < samples_per_pixel;
         first_sample += samples_per_batch) {
        const auto batch_sample_count =
            std::min(samples_per_batch, samples_per_pixel - first_sample);
        const auto inputs =
            make_inputs(*camera, wavelengths(), extent, first_sample, batch_sample_count);
        ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

        const auto traced = trace_cuda_wavefront_transport(
            *scene_soa_, *scene_bvh_, *inputs,
            CudaWavefrontTransportOptions{.maximum_diffuse_depth = CornellMaximumDiffuseDepth});
        ASSERT_TRUE(traced.has_value()) << traced.error().message;
        ASSERT_EQ(traced->paths.size(), inputs->size());

        const auto& report = traced->report;
        EXPECT_EQ(report.schema_version, CurrentCudaWavefrontTransportReportSchemaVersion);
        EXPECT_EQ(report.maximum_diffuse_depth, CornellMaximumDiffuseDepth);
        EXPECT_EQ(report.path_count, inputs->size());
        EXPECT_EQ(report.terminated_paths, inputs->size());
        EXPECT_EQ(report.queue_overflow_attempts, 0U);
        EXPECT_EQ(report.queue_rejected_lanes, 0U);
        EXPECT_EQ(report.stage_lanes.camera, inputs->size());
        EXPECT_EQ(report.stage_lanes.hit + report.stage_lanes.miss,
                  report.stage_lanes.intersection);
        add_report(aggregate_report, report);

        for (auto path_index = std::size_t{}; path_index < traced->paths.size(); ++path_index) {
            const auto& result = traced->paths[path_index];
            EXPECT_TRUE(finite_non_negative(result.state.beta()));
            EXPECT_TRUE(finite_non_negative(result.state.accumulated_radiance()));
            EXPECT_EQ(result.state.current_medium(), renderer::VacuumMedium);
            EXPECT_EQ(result.terminal_ray.current_medium(), renderer::VacuumMedium);
            EXPECT_EQ(result.state.wavelengths(), wavelengths());

            if (std::ranges::any_of(result.state.accumulated_radiance().values,
                                    [](const auto value) { return value > 0; })) {
                ++non_black_spectral_paths;
            }
            const auto xyz = renderer::cie_1931_spectrum_to_xyz(result.state.accumulated_radiance(),
                                                                result.state.wavelengths());
            ASSERT_TRUE(xyz.has_value()) << xyz.error().message;
            const auto rgb = renderer::xyz_to_linear_rgb(*xyz);
            ASSERT_TRUE(rgb.has_value()) << rgb.error().message;
            ASSERT_TRUE(std::isfinite(rgb->red));
            ASSERT_TRUE(std::isfinite(rgb->green));
            ASSERT_TRUE(std::isfinite(rgb->blue));

            const auto& sample = (*inputs)[path_index].sample;
            const auto accumulated = film->add_sample(sample.pixel_x, sample.pixel_y, *rgb,
                                                      renderer::TransportScalar{1});
            ASSERT_TRUE(accumulated.has_value()) << accumulated.error().message;
        }
    }

    const auto expected_path_count =
        static_cast<std::size_t>(extent.width) * extent.height * samples_per_pixel;
    EXPECT_EQ(aggregate_report.path_count, expected_path_count);
    EXPECT_EQ(aggregate_report.terminated_paths, expected_path_count);
    EXPECT_EQ(aggregate_report.queue_overflow_attempts, 0U);
    EXPECT_EQ(aggregate_report.queue_rejected_lanes, 0U);
    EXPECT_EQ(aggregate_report.stage_lanes.camera, expected_path_count);
    EXPECT_GT(aggregate_report.stage_lanes.intersection, 0U);
    EXPECT_GT(aggregate_report.stage_lanes.hit, 0U);
    EXPECT_GT(aggregate_report.stage_lanes.miss, 0U);
    EXPECT_GT(aggregate_report.stage_lanes.shade, 0U);
    EXPECT_GT(aggregate_report.stage_lanes.shadow, 0U);
    EXPECT_GT(aggregate_report.stage_lanes.continuation, 0U);
    EXPECT_EQ(aggregate_report.stage_lanes.hit + aggregate_report.stage_lanes.miss,
              aggregate_report.stage_lanes.intersection);
    EXPECT_GT(aggregate_report.light_samples, 0U);
    EXPECT_GT(aggregate_report.shadow_queries, 0U);
    EXPECT_GT(aggregate_report.light_samples, aggregate_report.shadow_queries);
    EXPECT_GT(non_black_spectral_paths, 0U);

    auto positive_image_energy = renderer::ReferenceScalar{};
    auto left_wall = RegionSum{};
    auto right_wall = RegionSum{};
    auto ceiling_emitter = RegionSum{};
    auto back_wall = RegionSum{};
    for (auto pixel_y = std::uint32_t{}; pixel_y < extent.height; ++pixel_y) {
        for (auto pixel_x = std::uint32_t{}; pixel_x < extent.width; ++pixel_x) {
            const auto stored = film->pixel(pixel_x, pixel_y);
            ASSERT_TRUE(stored.has_value()) << stored.error().message;
            EXPECT_EQ(stored->sample_count, samples_per_pixel);
            const auto resolved = film->resolved_pixel(pixel_x, pixel_y);
            ASSERT_TRUE(resolved.has_value()) << resolved.error().message;
            ASSERT_TRUE(std::isfinite(resolved->red));
            ASSERT_TRUE(std::isfinite(resolved->green));
            ASSERT_TRUE(std::isfinite(resolved->blue));
            positive_image_energy += std::max(
                renderer::ReferenceScalar{}, static_cast<renderer::ReferenceScalar>(resolved->red));
            positive_image_energy +=
                std::max(renderer::ReferenceScalar{},
                         static_cast<renderer::ReferenceScalar>(resolved->green));
            positive_image_energy +=
                std::max(renderer::ReferenceScalar{},
                         static_cast<renderer::ReferenceScalar>(resolved->blue));
            const auto in_region = [&](const std::uint32_t x0, const std::uint32_t y0,
                                       const std::uint32_t x1, const std::uint32_t y1) {
                return pixel_x >= extent.width * x0 / 64U && pixel_x < extent.width * x1 / 64U &&
                       pixel_y >= extent.height * y0 / 64U && pixel_y < extent.height * y1 / 64U;
            };
            if (in_region(2U, 16U, 10U, 44U)) {
                left_wall.add(*resolved);
            }
            if (in_region(54U, 16U, 62U, 44U)) {
                right_wall.add(*resolved);
            }
            if (in_region(24U, 1U, 40U, 5U)) {
                ceiling_emitter.add(*resolved);
            }
            if (in_region(24U, 10U, 40U, 18U)) {
                back_wall.add(*resolved);
            }
        }
    }
    EXPECT_TRUE(std::isfinite(positive_image_energy));
    EXPECT_GT(positive_image_energy, renderer::ReferenceScalar{});
    ASSERT_GT(left_wall.count, 0U);
    ASSERT_GT(right_wall.count, 0U);
    ASSERT_GT(ceiling_emitter.count, 0U);
    ASSERT_GT(back_wall.count, 0U);
    EXPECT_GT(left_wall.red, 1.5 * left_wall.green);
    EXPECT_GT(left_wall.red, 1.5 * left_wall.blue);
    EXPECT_GT(right_wall.green, 1.2 * right_wall.red);
    EXPECT_GT(right_wall.green, 1.5 * right_wall.blue);
    EXPECT_GT(ceiling_emitter.luminance(), 3.0 * back_wall.luminance());

#if defined(BLACKFRAME_CUDA_CORNELL_PNG)
    if (diagnostic_output) {
        const auto written = renderer::write_png_preview(*film, *diagnostic_output);
        ASSERT_TRUE(written.has_value()) << written.error().message;
        testing::Test::RecordProperty("cuda_cornell_output", diagnostic_output->string());
    }
#endif
}

TEST_F(CudaCornellWavefrontSmokeTest, AccumulatesConstantEnvironmentOnPrimaryMiss) {
    const auto environment = renderer::TransportSpectrum{.values = {0.25F, 0.5F, 0.75F, 1.0F}};
    const auto frame = FrameScene::create(FrameSceneDescription{
        .objects = {},
        .geometries = {},
        .materials = {},
        .instances = {},
        .punctual_lights = {},
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = wavelengths(),
                .radiance = environment,
            },
    });
    ASSERT_TRUE(frame.has_value()) << frame.error().message;
    auto uploaded_result = CudaSceneSoA::upload(**frame);
    ASSERT_TRUE(uploaded_result.has_value()) << uploaded_result.error().message;
    auto uploaded = std::move(*uploaded_result);
    auto bvh_result = CudaSceneBvh::build(uploaded);
    ASSERT_TRUE(bvh_result.has_value()) << bvh_result.error().message;
    auto bvh = std::move(*bvh_result);

    const auto camera = cornell_wavefront_test::make_camera(CornellExtent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    const auto inputs = make_pixel_inputs(*camera, wavelengths(), 32U, 32U, 0U, 1U);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    const auto traced = trace_cuda_wavefront_transport(
        uploaded, bvh, *inputs,
        CudaWavefrontTransportOptions{.maximum_diffuse_depth = CornellMaximumDiffuseDepth});
    ASSERT_TRUE(traced.has_value()) << traced.error().message;
    ASSERT_EQ(traced->paths.size(), 1U);
    EXPECT_EQ(traced->paths.front().termination, CudaWavefrontPathTermination::escaped_environment);
    EXPECT_EQ(traced->paths.front().state.accumulated_radiance(), environment);
    EXPECT_EQ(traced->paths.front().state.depth(), 0U);
    EXPECT_EQ(traced->report.stage_lanes.intersection, 1U);
    EXPECT_EQ(traced->report.stage_lanes.hit, 0U);
    EXPECT_EQ(traced->report.stage_lanes.miss, 1U);
    EXPECT_EQ(traced->report.stage_lanes.shade, 0U);
    EXPECT_EQ(traced->report.light_samples, 0U);
    EXPECT_EQ(traced->report.shadow_queries, 0U);
    EXPECT_TRUE(bvh.close().has_value());
    EXPECT_TRUE(uploaded.close().has_value());
}

TEST_F(CudaCornellWavefrontSmokeTest, RejectsEmptyInputWithoutFallback) {
    const auto rejected = trace_cuda_wavefront_transport(
        *scene_soa_, *scene_bvh_, std::span<const CudaWavefrontPathInput>{},
        CudaWavefrontTransportOptions{.maximum_diffuse_depth = CornellMaximumDiffuseDepth});
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(rejected.error().message.empty());
}

TEST_F(CudaCornellWavefrontSmokeTest, TerminatesSmoothNormalSupportMissAfterPendingShadow) {
    const auto camera = cornell_wavefront_test::make_camera(DiagnosticCornellExtent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    constexpr auto pixel_x = std::uint32_t{535U};
    constexpr auto pixel_y = std::uint32_t{383U};
    constexpr auto candidate_count = std::uint32_t{4096U};
    auto inputs = make_pixel_inputs(*camera, wavelengths(), pixel_x, pixel_y, 0U, candidate_count);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    for (auto& input : *inputs) {
        input.primary_ray = inputs->front().primary_ray;
    }

    const auto candidates = trace_cuda_wavefront_transport(
        *scene_soa_, *scene_bvh_, *inputs,
        CudaWavefrontTransportOptions{.maximum_diffuse_depth = CornellMaximumDiffuseDepth});
    ASSERT_TRUE(candidates.has_value()) << candidates.error().message;
    ASSERT_EQ(candidates->paths.size(), candidate_count);

    auto verified = false;
    for (auto index = std::size_t{}; index < candidates->paths.size() && !verified; ++index) {
        if (candidates->paths[index].termination !=
                CudaWavefrontPathTermination::outside_bsdf_support ||
            candidates->paths[index].state.depth() != 0U) {
            continue;
        }
        const auto input =
            std::span<const CudaWavefrontPathInput>{inputs->data() + index, std::size_t{1U}};
        const auto traced = trace_cuda_wavefront_transport(
            *scene_soa_, *scene_bvh_, input,
            CudaWavefrontTransportOptions{.maximum_diffuse_depth = CornellMaximumDiffuseDepth});
        ASSERT_TRUE(traced.has_value()) << traced.error().message;
        ASSERT_EQ(traced->paths.size(), 1U);
        verified =
            traced->paths.front().termination ==
                CudaWavefrontPathTermination::outside_bsdf_support &&
            traced->report.stage_lanes.camera == 1U &&
            traced->report.stage_lanes.intersection == 1U && traced->report.stage_lanes.hit == 1U &&
            traced->report.stage_lanes.shade == 1U && traced->report.stage_lanes.shadow == 1U &&
            traced->report.stage_lanes.continuation == 0U && traced->report.terminated_paths == 1U;
    }
    EXPECT_TRUE(verified);
}

TEST_F(CudaCornellWavefrontSmokeTest, KeepsCoplanarBackWallShadowRaysVisible) {
    const auto camera = cornell_wavefront_test::make_camera(DiagnosticCornellExtent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    constexpr auto sample_count = std::uint32_t{16U};
    const auto inputs = make_pixel_inputs(*camera, wavelengths(), 400U, 383U, 0U, sample_count);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    const auto traced =
        trace_cuda_wavefront_transport(*scene_soa_, *scene_bvh_, *inputs,
                                       CudaWavefrontTransportOptions{.maximum_diffuse_depth = 1U});
    ASSERT_TRUE(traced.has_value()) << traced.error().message;
    ASSERT_EQ(traced->paths.size(), sample_count);
    EXPECT_EQ(traced->report.stage_lanes.camera, sample_count);
    EXPECT_EQ(traced->report.stage_lanes.shadow, sample_count);
    EXPECT_EQ(traced->report.shadow_queries, sample_count);
    EXPECT_EQ(traced->report.queue_overflow_attempts, 0U);
    EXPECT_EQ(traced->report.queue_rejected_lanes, 0U);
    for (const auto& path : traced->paths) {
        EXPECT_TRUE(std::ranges::any_of(path.state.accumulated_radiance().values,
                                        [](const auto value) { return value > 0.0F; }));
    }
}

TEST_F(CudaCornellWavefrontSmokeTest, RejectsNonVacuumPathWithoutFallback) {
    constexpr auto unsupported_medium = renderer::MediumId{.value = 7U};
    const auto camera = cornell_wavefront_test::make_camera(CornellExtent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    const auto inputs = make_inputs(*camera, wavelengths(), CornellExtent, 0U,
                                    CornellSamplesPerPixel, unsupported_medium);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    const auto rejected = trace_cuda_wavefront_transport(
        *scene_soa_, *scene_bvh_, *inputs,
        CudaWavefrontTransportOptions{.maximum_diffuse_depth = CornellMaximumDiffuseDepth});
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error().code, core::StatusCode::invalid_argument);
    EXPECT_NE(rejected.error().message.find("vacuum"), std::string_view::npos);
}

TEST_F(CudaCornellWavefrontSmokeTest, RejectsResumedPathWithoutFallback) {
    const auto camera = cornell_wavefront_test::make_camera(CornellExtent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    auto inputs = make_pixel_inputs(*camera, wavelengths(), 32U, 32U, 0U, 1U);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    const auto initial = renderer::PathState::create_initial(wavelengths(), renderer::VacuumMedium);
    ASSERT_TRUE(initial.has_value()) << initial.error().message;
    const auto resumed = renderer::PathState::create(
        initial->beta(), initial->accumulated_radiance(),
        renderer::PathDepthCounters{.diffuse = 1U}, renderer::TransportScalar{1}, wavelengths(),
        renderer::PathDeltaFlags::any_non_delta_bounces, renderer::VacuumMedium);
    ASSERT_TRUE(resumed.has_value()) << resumed.error().message;
    inputs->front().initial_state = *resumed;

    const auto rejected = trace_cuda_wavefront_transport(
        *scene_soa_, *scene_bvh_, *inputs,
        CudaWavefrontTransportOptions{.maximum_diffuse_depth = CornellMaximumDiffuseDepth});
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error().code, core::StatusCode::incompatible);
    EXPECT_NE(rejected.error().message.find("primary path depth zero"), std::string_view::npos);
}

TEST_F(CudaCornellWavefrontSmokeTest, TerminatesZeroThroughputBeforeSampling) {
    const auto camera = cornell_wavefront_test::make_camera(CornellExtent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    auto inputs = make_pixel_inputs(*camera, wavelengths(), 32U, 32U, 0U, 1U);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    const auto zero_state = renderer::PathState::create(
        renderer::TransportSpectrum{}, renderer::TransportSpectrum{}, renderer::PathDepthCounters{},
        renderer::TransportScalar{1}, wavelengths(), renderer::PathDeltaFlags::none,
        renderer::VacuumMedium);
    ASSERT_TRUE(zero_state.has_value()) << zero_state.error().message;
    inputs->front().initial_state = *zero_state;

    const auto traced = trace_cuda_wavefront_transport(
        *scene_soa_, *scene_bvh_, *inputs,
        CudaWavefrontTransportOptions{.maximum_diffuse_depth = CornellMaximumDiffuseDepth});
    ASSERT_TRUE(traced.has_value()) << traced.error().message;
    ASSERT_EQ(traced->paths.size(), 1U);
    EXPECT_EQ(traced->paths.front().termination, CudaWavefrontPathTermination::zero_throughput);
    EXPECT_EQ(traced->paths.front().state.depth(), 0U);
    EXPECT_EQ(traced->paths.front().state.depth_counters(), renderer::PathDepthCounters{});
    EXPECT_EQ(traced->report.light_samples, 0U);
    EXPECT_EQ(traced->report.shadow_queries, 0U);
    EXPECT_EQ(traced->report.stage_lanes.continuation, 0U);
}

TEST_F(CudaCornellWavefrontSmokeTest, RejectsInsufficientDeviceBudgetWithoutFallback) {
    const auto camera = cornell_wavefront_test::make_camera(CornellExtent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    const auto inputs = make_pixel_inputs(*camera, wavelengths(), 32U, 32U, 0U, 1U);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    const auto rejected = trace_cuda_wavefront_transport(
        *scene_soa_, *scene_bvh_, *inputs,
        CudaWavefrontTransportOptions{
            .maximum_diffuse_depth = CornellMaximumDiffuseDepth,
            .device_memory_budget = xpu::cuda::DeviceMemoryBudget{.maximum_bytes = 1U},
        });
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error().code, core::StatusCode::resource_exhausted);
    EXPECT_FALSE(rejected.error().message.empty());
}

} // namespace
} // namespace blackframe::engine
