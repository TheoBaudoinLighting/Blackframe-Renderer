#include "../../CPU/Embree/CornellWavefrontScene.hpp"

#include <Blackframe/Backends/GPU/CUDA/SceneBvh.hpp>
#include <Blackframe/Backends/GPU/CUDA/SceneSoA.hpp>
#include <Blackframe/Backends/GPU/CUDA/WavefrontTransport.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Renderer/Cie1931Sensor.hpp>
#include <Blackframe/Renderer/Color.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/PixelJitter.hpp>
#if defined(BLACKFRAME_CUDA_CORNELL_PNG)
#include <Blackframe/Renderer/PngWriter.hpp>
#endif
#include <Blackframe/Renderer/Ray.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
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

[[nodiscard]] CudaWavefrontTransportOptions
transport_options(const std::uint32_t maximum_diffuse_depth = CornellMaximumDiffuseDepth) {
    return CudaWavefrontTransportOptions{
        .heuristic = renderer::MisHeuristic::power,
        .depth_limits = renderer::PathDepthLimits{.diffuse = maximum_diffuse_depth},
        .roulette_policy = renderer::RussianRoulettePolicy::disabled(),
    };
}

[[nodiscard]] SceneClosureMixture
require_lambertian_scene_closure(const renderer::TransportSpectrum reflectance) {
    return SceneClosureMixture::create_lambertian(reflectance).value();
}

[[nodiscard]] core::Result<renderer::LightSampler>
make_uniform_light_sampler(const FrameScene& scene) {
    return renderer::LightSampler::create_uniform(scene.punctual_lights().size() +
                                                  scene.mesh_area_lights().size());
}

[[nodiscard]] core::Result<FrameSceneHandle>
make_environment_receiver_scene(const renderer::SampledWavelengths& sampled_wavelengths) {
    auto mesh = TriangleMesh::create(
        {
            renderer::Point3{.x = -2.0F, .y = -2.0F},
            renderer::Point3{.x = 2.0F, .y = -2.0F},
            renderer::Point3{.x = 2.0F, .y = 2.0F},
            renderer::Point3{.x = -2.0F, .y = 2.0F},
        },
        std::vector<renderer::Normal3>(4U, renderer::Normal3{.z = 1.0F}),
        {
            renderer::Point2{},
            renderer::Point2{.x = 1.0F},
            renderer::Point2{.x = 1.0F, .y = 1.0F},
            renderer::Point2{.y = 1.0F},
        },
        {
            TriangleVertexIndices{.vertices = {0U, 1U, 2U}},
            TriangleVertexIndices{.vertices = {0U, 2U, 3U}},
        });
    if (!mesh) {
        return std::unexpected(mesh.error());
    }
    auto shared_mesh = std::make_shared<const TriangleMesh>(std::move(*mesh));
    return FrameScene::create(FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 1U}}},
        .geometries = {SceneGeometry{.id = {.value = 11U}, .mesh = std::move(shared_mesh)}},
        .materials =
            {
                SceneMaterial{
                    .id = {.value = 21U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = sampled_wavelengths,
                            .closure_mixture = require_lambertian_scene_closure(
                                {.values = {0.25F, 0.5F, 0.75F, 1.0F}}),
                            .emitted_radiance = {},
                        },
                },
            },
        .instances =
            {
                SceneInstance{
                    .id = {.value = 31U},
                    .parent = std::nullopt,
                    .object = {.value = 1U},
                    .geometry = {.value = 11U},
                    .material = {.value = 21U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
            },
        .punctual_lights = {},
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = sampled_wavelengths,
                .radiance = {.values = {2.0F, 2.0F, 2.0F, 2.0F}},
            },
    });
}

[[nodiscard]] core::Result<FrameSceneHandle>
make_specular_dispatch_scene(const renderer::SampledWavelengths& sampled_wavelengths) {
    auto mesh = TriangleMesh::create(
        {
            renderer::Point3{.x = 0.0F, .y = -2.0F, .z = 0.0F},
            renderer::Point3{.x = 2.0F, .y = -2.0F, .z = 0.0F},
            renderer::Point3{.x = 2.0F, .y = 2.0F, .z = 0.0F},
            renderer::Point3{.x = 0.0F, .y = 2.0F, .z = 0.0F},
            renderer::Point3{.x = 0.0F, .y = -2.0F, .z = 1.0F},
            renderer::Point3{.x = 0.0F, .y = 2.0F, .z = 1.0F},
            renderer::Point3{.x = 2.0F, .y = 2.0F, .z = 1.0F},
            renderer::Point3{.x = 2.0F, .y = -2.0F, .z = 1.0F},
        },
        {
            renderer::Normal3{.z = 1.0F},
            renderer::Normal3{.z = 1.0F},
            renderer::Normal3{.z = 1.0F},
            renderer::Normal3{.z = 1.0F},
            renderer::Normal3{.z = -1.0F},
            renderer::Normal3{.z = -1.0F},
            renderer::Normal3{.z = -1.0F},
            renderer::Normal3{.z = -1.0F},
        },
        {
            renderer::Point2{},
            renderer::Point2{.x = 1.0F},
            renderer::Point2{.x = 1.0F, .y = 1.0F},
            renderer::Point2{.y = 1.0F},
            renderer::Point2{},
            renderer::Point2{.y = 1.0F},
            renderer::Point2{.x = 1.0F, .y = 1.0F},
            renderer::Point2{.x = 1.0F},
        },
        {
            TriangleVertexIndices{.vertices = {0U, 1U, 2U}},
            TriangleVertexIndices{.vertices = {0U, 2U, 3U}},
            TriangleVertexIndices{.vertices = {4U, 5U, 6U}},
            TriangleVertexIndices{.vertices = {4U, 6U, 7U}},
        });
    if (!mesh) {
        return std::unexpected(mesh.error());
    }

    auto closures = renderer::ClosureSet{};
    const auto appended = closures.append_specular_reflection(
        renderer::TransportSpectrum{.values = {0.8F, 0.8F, 0.8F, 0.8F}});
    if (appended != renderer::ClosureAppendStatus::appended) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::internal_error,
            .message = "The specular dispatch fixture could not create its mirror closure.",
        });
    }
    constexpr auto probabilities = std::array{renderer::TransportScalar{1.0F}};
    auto closure_mixture = SceneClosureMixture::create(std::move(closures), probabilities);
    if (!closure_mixture) {
        return std::unexpected(closure_mixture.error());
    }

    return FrameScene::create(FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 1U}}},
        .geometries = {SceneGeometry{
            .id = {.value = 11U},
            .mesh = std::make_shared<const TriangleMesh>(std::move(*mesh)),
        }},
        .materials = {SceneMaterial{
            .id = {.value = 21U},
            .spectral =
                SceneSpectralMaterial{
                    .wavelengths = sampled_wavelengths,
                    .closure_mixture = std::move(*closure_mixture),
                    .emitted_radiance = {},
                },
        }},
        .instances = {SceneInstance{
            .id = {.value = 31U},
            .parent = std::nullopt,
            .object = {.value = 1U},
            .geometry = {.value = 11U},
            .material = {.value = 21U},
            .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
        }},
        .punctual_lights = {},
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = sampled_wavelengths,
                .radiance = {.values = {0.25F, 0.25F, 0.25F, 0.25F}},
            },
    });
}

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
                const auto primary_cone = camera.generate_primary_ray_cone(
                    pixel_sample, renderer::PixelJitterMode::uniform, CornellPathTime);
                if (!primary_cone) {
                    return std::unexpected(primary_cone.error());
                }
                const auto ray = renderer::Ray::create(primary->origin(), primary->direction(),
                                                       primary->t_min(), primary->t_max(),
                                                       primary->time(), primary->mask(), medium);
                if (!ray) {
                    return std::unexpected(ray.error());
                }
                inputs.push_back(CudaWavefrontPathInput{
                    .primary_ray = *ray,
                    .primary_cone = *primary_cone,
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
        const auto primary_cone = camera.generate_primary_ray_cone(
            pixel_sample, renderer::PixelJitterMode::uniform, CornellPathTime);
        if (!primary_cone) {
            return std::unexpected(primary_cone.error());
        }
        const auto ray = renderer::Ray::create(primary->origin(), primary->direction(),
                                               primary->t_min(), primary->t_max(), primary->time(),
                                               primary->mask(), renderer::VacuumMedium);
        if (!ray) {
            return std::unexpected(ray.error());
        }
        inputs.push_back(CudaWavefrontPathInput{
            .primary_ray = *ray,
            .primary_cone = *primary_cone,
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
    const auto add_metric = [](CudaWavefrontStageMetric& target,
                               const CudaWavefrontStageMetric& value) {
        target.kernel_dispatches += value.kernel_dispatches;
        target.kernel_lanes += value.kernel_lanes;
        target.gpu_elapsed_nanoseconds += value.gpu_elapsed_nanoseconds;
    };
    destination.path_count += source.path_count;
    destination.asynchronous_upload_bytes += source.asynchronous_upload_bytes;
    destination.asynchronous_download_bytes += source.asynchronous_download_bytes;
    destination.cross_stream_event_dependencies += source.cross_stream_event_dependencies;
    destination.stage_lanes.camera += source.stage_lanes.camera;
    destination.stage_lanes.intersection += source.stage_lanes.intersection;
    destination.stage_lanes.hit += source.stage_lanes.hit;
    destination.stage_lanes.miss += source.stage_lanes.miss;
    destination.stage_lanes.shade += source.stage_lanes.shade;
    destination.stage_lanes.shadow += source.stage_lanes.shadow;
    destination.stage_lanes.continuation += source.stage_lanes.continuation;
    add_metric(destination.stage_metrics.camera, source.stage_metrics.camera);
    add_metric(destination.stage_metrics.intersection, source.stage_metrics.intersection);
    add_metric(destination.stage_metrics.hit, source.stage_metrics.hit);
    add_metric(destination.stage_metrics.miss, source.stage_metrics.miss);
    add_metric(destination.stage_metrics.shade, source.stage_metrics.shade);
    add_metric(destination.stage_metrics.shadow, source.stage_metrics.shadow);
    add_metric(destination.stage_metrics.continuation, source.stage_metrics.continuation);
    destination.closure_samples += source.closure_samples;
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

void expect_transport_paths_exact(const CudaWavefrontTransportBatch& expected,
                                  const CudaWavefrontTransportBatch& actual) {
    ASSERT_EQ(actual.paths.size(), expected.paths.size());
    ASSERT_EQ(actual.terminal_cones.size(), actual.paths.size());
    ASSERT_EQ(expected.terminal_cones.size(), expected.paths.size());
    for (auto index = std::size_t{}; index < expected.paths.size(); ++index) {
        SCOPED_TRACE(testing::Message{} << "path " << index);
        const auto& expected_path = expected.paths[index];
        const auto& actual_path = actual.paths[index];
        EXPECT_EQ(actual_path.termination, expected_path.termination);
        EXPECT_EQ(actual_path.blocked_depth_limits, expected_path.blocked_depth_limits);
        EXPECT_EQ(actual_path.state.beta(), expected_path.state.beta());
        EXPECT_EQ(actual_path.state.accumulated_radiance(),
                  expected_path.state.accumulated_radiance());
        EXPECT_EQ(actual_path.state.depth(), expected_path.state.depth());
        EXPECT_EQ(actual_path.state.depth_counters(), expected_path.state.depth_counters());
        EXPECT_EQ(actual_path.state.eta_scale(), expected_path.state.eta_scale());
        EXPECT_EQ(actual_path.state.wavelengths(), expected_path.state.wavelengths());
        EXPECT_EQ(actual_path.state.delta_flags(), expected_path.state.delta_flags());
        EXPECT_EQ(actual_path.state.current_medium(), expected_path.state.current_medium());
        EXPECT_EQ(actual_path.terminal_ray.origin(), expected_path.terminal_ray.origin());
        EXPECT_EQ(actual_path.terminal_ray.direction(), expected_path.terminal_ray.direction());
        EXPECT_EQ(actual_path.terminal_ray.t_min(), expected_path.terminal_ray.t_min());
        EXPECT_EQ(actual_path.terminal_ray.t_max(), expected_path.terminal_ray.t_max());
        EXPECT_EQ(actual_path.terminal_ray.time(), expected_path.terminal_ray.time());
        EXPECT_EQ(actual_path.terminal_ray.mask(), expected_path.terminal_ray.mask());
        EXPECT_EQ(actual_path.terminal_ray.current_medium(),
                  expected_path.terminal_ray.current_medium());
        EXPECT_EQ(actual.terminal_cones[index], expected.terminal_cones[index]);
    }
}

void expect_transport_batches_exact(const CudaWavefrontTransportBatch& expected,
                                    const CudaWavefrontTransportBatch& actual) {
    EXPECT_EQ(actual.report, expected.report);
    expect_transport_paths_exact(expected, actual);
}

void expect_transport_batches_physically_exact(const CudaWavefrontTransportBatch& expected,
                                               const CudaWavefrontTransportBatch& actual) {
    EXPECT_EQ(actual.report.schema_version, expected.report.schema_version);
    EXPECT_EQ(actual.report.has_light_sampler, expected.report.has_light_sampler);
    EXPECT_EQ(actual.report.registered_light_count, expected.report.registered_light_count);
    EXPECT_EQ(actual.report.heuristic, expected.report.heuristic);
    EXPECT_EQ(actual.report.light_sampling_strategy, expected.report.light_sampling_strategy);
    EXPECT_EQ(actual.report.depth_limits, expected.report.depth_limits);
    EXPECT_EQ(actual.report.roulette_policy, expected.report.roulette_policy);
    EXPECT_EQ(actual.report.path_count, expected.report.path_count);
    EXPECT_EQ(actual.report.stage_lanes, expected.report.stage_lanes);
    EXPECT_EQ(actual.report.closure_samples, expected.report.closure_samples);
    EXPECT_EQ(actual.report.light_samples, expected.report.light_samples);
    EXPECT_EQ(actual.report.shadow_queries, expected.report.shadow_queries);
    EXPECT_EQ(actual.report.terminated_paths, expected.report.terminated_paths);
    EXPECT_EQ(actual.report.queue_overflow_attempts, expected.report.queue_overflow_attempts);
    EXPECT_EQ(actual.report.queue_rejected_lanes, expected.report.queue_rejected_lanes);
    expect_transport_paths_exact(expected, actual);
}

void expect_synchronous_transfer_report(const CudaWavefrontTransportReport& report) {
    EXPECT_EQ(report.transfer_mode, CudaWavefrontTransferMode::synchronous);
    EXPECT_EQ(report.asynchronous_upload_bytes, 0U);
    EXPECT_EQ(report.asynchronous_download_bytes, 0U);
    EXPECT_EQ(report.cross_stream_event_dependencies, 0U);
}

void expect_asynchronous_transfer_report(const CudaWavefrontTransportReport& report) {
    EXPECT_EQ(report.transfer_mode, CudaWavefrontTransferMode::asynchronous);
    EXPECT_GT(report.asynchronous_upload_bytes, 0U);
    EXPECT_GT(report.asynchronous_download_bytes, 0U);
    EXPECT_GT(report.cross_stream_event_dependencies, 0U);
}

class CudaCornellWavefrontSmokeTest : public testing::Test {
  protected:
    void SetUp() override {
        ASSERT_TRUE(select_test_device());

        const auto created_scene = cornell_wavefront_test::make_cornell_scene();
        ASSERT_TRUE(created_scene.has_value()) << created_scene.error().message;
        scene_ = *created_scene;

        auto light_sampler = make_uniform_light_sampler(*scene_);
        ASSERT_TRUE(light_sampler.has_value()) << light_sampler.error().message;
        light_sampler_.emplace(std::move(*light_sampler));

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
    std::optional<renderer::LightSampler> light_sampler_{};
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
    const auto transfer_mode = diagnostic_output ? CudaWavefrontTransferMode::asynchronous
                                                 : CudaWavefrontTransferMode::synchronous;
    auto options = transport_options();
    options.transfer_mode = transfer_mode;

    const auto camera = cornell_wavefront_test::make_camera(extent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    auto film = renderer::Film::create(extent);
    ASSERT_TRUE(film.has_value()) << film.error().message;
    const auto workspace_capacity = static_cast<std::size_t>(extent.width) * extent.height *
                                    static_cast<std::size_t>(samples_per_batch);
    auto workspace = CudaWavefrontTransportWorkspace::create(
        workspace_capacity, xpu::cuda::DeviceMemoryBudget{}, transfer_mode);
    ASSERT_TRUE(workspace.has_value()) << workspace.error().message;

    auto aggregate_report = CudaWavefrontTransportReport{
        .schema_version = CurrentCudaWavefrontTransportReportSchemaVersion,
        .has_light_sampler = true,
        .registered_light_count = light_sampler_->light_count(),
        .heuristic = renderer::MisHeuristic::power,
        .light_sampling_strategy = renderer::LightSamplingStrategy::uniform,
        .depth_limits = renderer::PathDepthLimits{.diffuse = CornellMaximumDiffuseDepth},
        .roulette_policy = renderer::RussianRoulettePolicy::disabled(),
        .transfer_mode = transfer_mode,
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
            *workspace, *scene_soa_, *scene_bvh_, *inputs, std::cref(*light_sampler_), options);
        ASSERT_TRUE(traced.has_value()) << traced.error().message;
        ASSERT_EQ(traced->paths.size(), inputs->size());
        ASSERT_EQ(traced->terminal_cones.size(), traced->paths.size());

        const auto& report = traced->report;
        EXPECT_EQ(report.schema_version, CurrentCudaWavefrontTransportReportSchemaVersion);
        EXPECT_TRUE(report.has_light_sampler);
        EXPECT_EQ(report.registered_light_count, light_sampler_->light_count());
        EXPECT_EQ(report.heuristic, renderer::MisHeuristic::power);
        EXPECT_EQ(report.light_sampling_strategy, renderer::LightSamplingStrategy::uniform);
        EXPECT_EQ(report.depth_limits,
                  renderer::PathDepthLimits{.diffuse = CornellMaximumDiffuseDepth});
        EXPECT_EQ(report.roulette_policy, renderer::RussianRoulettePolicy::disabled());
        EXPECT_EQ(report.path_count, inputs->size());
        EXPECT_EQ(report.transfer_mode, transfer_mode);
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
    EXPECT_GT(aggregate_report.closure_samples, 0U);
    EXPECT_GT(aggregate_report.light_samples, 0U);
    EXPECT_GT(aggregate_report.shadow_queries, 0U);
    EXPECT_GT(aggregate_report.light_samples, aggregate_report.shadow_queries);
    if (transfer_mode == CudaWavefrontTransferMode::asynchronous) {
        EXPECT_GT(aggregate_report.asynchronous_upload_bytes, 0U);
        EXPECT_GT(aggregate_report.asynchronous_download_bytes, 0U);
        EXPECT_GT(aggregate_report.cross_stream_event_dependencies, 0U);
    } else {
        EXPECT_EQ(aggregate_report.asynchronous_upload_bytes, 0U);
        EXPECT_EQ(aggregate_report.asynchronous_download_bytes, 0U);
        EXPECT_EQ(aggregate_report.cross_stream_event_dependencies, 0U);
    }
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
    const auto workspace_close = workspace->close();
    EXPECT_TRUE(workspace_close.has_value()) << workspace_close.error().message;
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
    const auto traced =
        trace_cuda_wavefront_transport(uploaded, bvh, *inputs, std::nullopt, transport_options());
    ASSERT_TRUE(traced.has_value()) << traced.error().message;
    ASSERT_EQ(traced->paths.size(), 1U);
    EXPECT_EQ(traced->paths.front().termination, CudaWavefrontPathTermination::escaped_environment);
    EXPECT_EQ(traced->paths.front().state.accumulated_radiance(), environment);
    EXPECT_EQ(traced->paths.front().state.depth(), 0U);
    EXPECT_EQ(traced->report.stage_lanes.intersection, 1U);
    EXPECT_EQ(traced->report.stage_lanes.hit, 0U);
    EXPECT_EQ(traced->report.stage_lanes.miss, 1U);
    EXPECT_EQ(traced->report.stage_lanes.shade, 0U);
    EXPECT_FALSE(traced->report.has_light_sampler);
    EXPECT_EQ(traced->report.registered_light_count, 0U);
    EXPECT_EQ(traced->report.closure_samples, 0U);
    EXPECT_EQ(traced->report.light_samples, 0U);
    EXPECT_EQ(traced->report.shadow_queries, 0U);

    const auto extraneous_sampler = renderer::LightSampler::create_uniform(1U);
    ASSERT_TRUE(extraneous_sampler.has_value()) << extraneous_sampler.error().message;
    const auto rejected = trace_cuda_wavefront_transport(
        uploaded, bvh, *inputs, std::cref(*extraneous_sampler), transport_options());
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error().code, core::StatusCode::incompatible);
    EXPECT_NE(rejected.error().message.find("explicitly absent"), std::string::npos);

    EXPECT_TRUE(bvh.close().has_value());
    EXPECT_TRUE(uploaded.close().has_value());
}

TEST_F(CudaCornellWavefrontSmokeTest, SamplesLambertThenEnvironmentWithoutRegisteredLights) {
    const auto frame = make_environment_receiver_scene(wavelengths());
    ASSERT_TRUE(frame.has_value()) << frame.error().message;
    auto uploaded_result = CudaSceneSoA::upload(**frame);
    ASSERT_TRUE(uploaded_result.has_value()) << uploaded_result.error().message;
    auto uploaded = std::move(*uploaded_result);
    auto bvh_result = CudaSceneBvh::build(uploaded);
    ASSERT_TRUE(bvh_result.has_value()) << bvh_result.error().message;
    auto bvh = std::move(*bvh_result);

    const auto initial_state =
        renderer::PathState::create_initial(wavelengths(), renderer::VacuumMedium);
    ASSERT_TRUE(initial_state.has_value()) << initial_state.error().message;
    const auto primary =
        renderer::Ray::create(renderer::Point3{.z = 1.0F}, renderer::Vector3{.z = -1.0F}, 0.0F,
                              std::numeric_limits<renderer::TransportScalar>::infinity(),
                              CornellPathTime, renderer::AllRayVisibility, renderer::VacuumMedium);
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    const auto sampler = renderer::IndependentSampler{CornellSeed};
    const auto primary_cone = renderer::RayCone::create(0.125F, 0.25F);
    ASSERT_TRUE(primary_cone.has_value()) << primary_cone.error().message;
    const auto inputs = std::vector{
        CudaWavefrontPathInput{
            .primary_ray = *primary,
            .primary_cone = *primary_cone,
            .initial_state = *initial_state,
            .sample = sampler.make_stream(0U, 0U, 0U).index(),
        },
    };

    const auto traced =
        trace_cuda_wavefront_transport(uploaded, bvh, inputs, std::nullopt, transport_options(1U));
    ASSERT_TRUE(traced.has_value()) << traced.error().message;
    ASSERT_EQ(traced->paths.size(), 1U);
    ASSERT_EQ(traced->terminal_cones.size(), traced->paths.size());
    const auto& path = traced->paths.front();
    EXPECT_EQ(path.termination, CudaWavefrontPathTermination::escaped_environment);
    EXPECT_EQ(path.state.depth(), 1U);
    EXPECT_EQ(path.state.depth_counters(), renderer::PathDepthCounters{.diffuse = 1U});
    const auto expected_beta = renderer::TransportSpectrum{
        .values = {0.25F, 0.5F, 0.75F, 1.0F},
    };
    const auto expected_radiance = renderer::TransportSpectrum{
        .values = {0.5F, 1.0F, 1.5F, 2.0F},
    };
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(path.state.beta()[lane], expected_beta[lane], 1.0e-6F);
        EXPECT_NEAR(path.state.accumulated_radiance()[lane], expected_radiance[lane], 2.0e-6F);
    }
    EXPECT_FALSE(traced->report.has_light_sampler);
    EXPECT_EQ(traced->report.registered_light_count, 0U);
    EXPECT_EQ(traced->report.closure_samples, 1U);
    EXPECT_EQ(traced->report.light_samples, 0U);
    EXPECT_EQ(traced->report.shadow_queries, 0U);
    EXPECT_EQ(traced->report.stage_lanes.hit, 1U);
    EXPECT_EQ(traced->report.stage_lanes.miss, 1U);
    EXPECT_EQ(traced->report.stage_lanes.shade, 1U);
    EXPECT_EQ(traced->report.stage_lanes.continuation, 1U);

    const auto advanced = renderer::advance_ray_cone(*primary_cone, *primary, 1.0F);
    ASSERT_TRUE(advanced.has_value()) << advanced.error().message;
    auto closures = renderer::ClosureSet{};
    ASSERT_EQ(closures.append_lambertian_reflection(expected_beta),
              renderer::ClosureAppendStatus::appended);
    const auto incoming_local = renderer::normalized(path.terminal_ray.direction());
    ASSERT_TRUE(incoming_local.has_value()) << incoming_local.error().message;
    const auto reference_cone = renderer::propagate_ray_cone_scattering(
        *advanced, closures.closures().front(),
        renderer::ScatteringLobe::diffuse | renderer::ScatteringLobe::reflection,
        renderer::Vector3{.z = 1.0F}, *incoming_local);
    ASSERT_TRUE(reference_cone.has_value()) << reference_cone.error().message;
    EXPECT_EQ(traced->terminal_cones.front(), *reference_cone);

    EXPECT_TRUE(bvh.close().has_value());
    EXPECT_TRUE(uploaded.close().has_value());
}

TEST(CudaWavefrontDepthDispatchTest, AllowsSpecularBouncesBeyondDiffuseLimit) {
    ASSERT_TRUE(select_test_device());
    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.25F);
    ASSERT_TRUE(wavelengths.has_value()) << wavelengths.error().message;
    const auto frame = make_specular_dispatch_scene(*wavelengths);
    ASSERT_TRUE(frame.has_value()) << frame.error().message;
    auto uploaded_result = CudaSceneSoA::upload(**frame);
    ASSERT_TRUE(uploaded_result.has_value()) << uploaded_result.error().message;
    auto uploaded = std::move(*uploaded_result);
    auto bvh_result = CudaSceneBvh::build(uploaded);
    ASSERT_TRUE(bvh_result.has_value()) << bvh_result.error().message;
    auto bvh = std::move(*bvh_result);

    const auto initial_state =
        renderer::PathState::create_initial(*wavelengths, renderer::VacuumMedium);
    ASSERT_TRUE(initial_state.has_value()) << initial_state.error().message;
    const auto primary = renderer::Ray::create(
        renderer::Point3{.x = 0.5F, .z = 0.5F}, renderer::Vector3{.x = 0.6F, .z = -0.8F}, 0.0F,
        std::numeric_limits<renderer::TransportScalar>::infinity(), CornellPathTime,
        renderer::AllRayVisibility, renderer::VacuumMedium);
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    const auto sampler = renderer::IndependentSampler{CornellSeed};
    const auto primary_cone = renderer::RayCone::create(0.0F, 0.0F);
    ASSERT_TRUE(primary_cone.has_value()) << primary_cone.error().message;
    const auto inputs = std::array{
        CudaWavefrontPathInput{
            .primary_ray = *primary,
            .primary_cone = *primary_cone,
            .initial_state = *initial_state,
            .sample = sampler.make_stream(0U, 0U, 0U).index(),
        },
    };
    const auto options = CudaWavefrontTransportOptions{
        .heuristic = renderer::MisHeuristic::power,
        .depth_limits = renderer::PathDepthLimits{.specular = 2U},
        .roulette_policy = renderer::RussianRoulettePolicy::disabled(),
    };
    const auto traced =
        trace_cuda_wavefront_transport(uploaded, bvh, inputs, std::nullopt, options);
    ASSERT_TRUE(traced.has_value()) << traced.error().message;
    ASSERT_EQ(traced->paths.size(), 1U);
    const auto& path = traced->paths.front();
    EXPECT_EQ(path.termination, CudaWavefrontPathTermination::escaped_environment);
    EXPECT_EQ(path.state.depth_counters(), renderer::PathDepthCounters{.specular = 2U});
    EXPECT_EQ(path.state.delta_flags(), renderer::PathDeltaFlags::previous_bounce_was_delta);
    EXPECT_EQ(traced->report.stage_lanes.intersection, 3U);
    EXPECT_EQ(traced->report.stage_lanes.hit, 2U);
    EXPECT_EQ(traced->report.stage_lanes.miss, 1U);
    EXPECT_EQ(traced->report.stage_lanes.shade, 2U);
    EXPECT_EQ(traced->report.stage_lanes.continuation, 2U);
    EXPECT_EQ(traced->report.closure_samples, 2U);
    EXPECT_EQ(traced->report.light_samples, 0U);
    EXPECT_EQ(traced->report.shadow_queries, 0U);
    for (const auto value : path.state.beta().values) {
        EXPECT_NEAR(value, 0.64F, 1.0e-6F);
    }
    for (const auto value : path.state.accumulated_radiance().values) {
        EXPECT_NEAR(value, 0.16F, 1.0e-6F);
    }

    EXPECT_TRUE(bvh.close().has_value());
    EXPECT_TRUE(uploaded.close().has_value());
}

TEST_F(CudaCornellWavefrontSmokeTest, RejectsEmptyInputWithoutFallback) {
    const auto rejected = trace_cuda_wavefront_transport(
        *scene_soa_, *scene_bvh_, std::span<const CudaWavefrontPathInput>{},
        std::cref(*light_sampler_), transport_options());
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
        *scene_soa_, *scene_bvh_, *inputs, std::cref(*light_sampler_), transport_options());
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
            *scene_soa_, *scene_bvh_, input, std::cref(*light_sampler_), transport_options());
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

    const auto traced = trace_cuda_wavefront_transport(
        *scene_soa_, *scene_bvh_, *inputs, std::cref(*light_sampler_), transport_options(1U));
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
        *scene_soa_, *scene_bvh_, *inputs, std::cref(*light_sampler_), transport_options());
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
        *scene_soa_, *scene_bvh_, *inputs, std::cref(*light_sampler_), transport_options());
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
        *scene_soa_, *scene_bvh_, *inputs, std::cref(*light_sampler_), transport_options());
    ASSERT_TRUE(traced.has_value()) << traced.error().message;
    ASSERT_EQ(traced->paths.size(), 1U);
    EXPECT_EQ(traced->paths.front().termination, CudaWavefrontPathTermination::zero_throughput);
    EXPECT_EQ(traced->paths.front().state.depth(), 0U);
    EXPECT_EQ(traced->paths.front().state.depth_counters(), renderer::PathDepthCounters{});
    EXPECT_EQ(traced->report.light_samples, 0U);
    EXPECT_EQ(traced->report.shadow_queries, 0U);
    EXPECT_EQ(traced->report.stage_lanes.continuation, 0U);
}

TEST_F(CudaCornellWavefrontSmokeTest, NsightInstrumentationExportsStageMetrics) {
    const auto camera = cornell_wavefront_test::make_camera(CornellExtent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    const auto inputs = make_inputs(*camera, wavelengths(), CornellExtent, 0U, 1U);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    auto baseline_options = transport_options();
    baseline_options.transfer_mode = CudaWavefrontTransferMode::synchronous;
    const auto baseline = trace_cuda_wavefront_transport(
        *scene_soa_, *scene_bvh_, *inputs, std::cref(*light_sampler_), baseline_options);
    ASSERT_TRUE(baseline.has_value()) << baseline.error().message;
    EXPECT_EQ(baseline->report.instrumentation_mode, CudaWavefrontInstrumentationMode::disabled);
    EXPECT_EQ(baseline->report.stage_metrics, CudaWavefrontStageMetrics{});

    auto profiled_options = baseline_options;
    profiled_options.transfer_mode = CudaWavefrontTransferMode::asynchronous;
    profiled_options.instrumentation_mode = CudaWavefrontInstrumentationMode::nsight;
    auto workspace = CudaWavefrontTransportWorkspace::create(
        inputs->size(), xpu::cuda::DeviceMemoryBudget{}, profiled_options.transfer_mode,
        profiled_options.instrumentation_mode);
    ASSERT_TRUE(workspace.has_value()) << workspace.error().message;
    EXPECT_EQ(workspace->instrumentation_mode(), CudaWavefrontInstrumentationMode::nsight);

    const auto profiled =
        trace_cuda_wavefront_transport(*workspace, *scene_soa_, *scene_bvh_, *inputs,
                                       std::cref(*light_sampler_), profiled_options);
    ASSERT_TRUE(profiled.has_value()) << profiled.error().message;
    expect_transport_batches_physically_exact(*baseline, *profiled);
    EXPECT_EQ(profiled->report.instrumentation_mode, CudaWavefrontInstrumentationMode::nsight);

    const auto& lanes = profiled->report.stage_lanes;
    const auto& metrics = profiled->report.stage_metrics;
    EXPECT_EQ(metrics.camera.kernel_dispatches, 1U);
    EXPECT_EQ(metrics.camera.kernel_lanes, lanes.camera);
    EXPECT_EQ(metrics.intersection.kernel_lanes, lanes.intersection * 3U);
    EXPECT_EQ(metrics.hit.kernel_lanes, lanes.hit);
    EXPECT_EQ(metrics.miss.kernel_lanes, lanes.miss);
    EXPECT_EQ(metrics.shade.kernel_lanes, lanes.shade);
    EXPECT_EQ(metrics.shadow.kernel_lanes, lanes.shadow * 3U);
    EXPECT_EQ(metrics.continuation.kernel_lanes, lanes.continuation);
    EXPECT_EQ(metrics.intersection.kernel_dispatches % 3U, 0U);
    EXPECT_EQ(metrics.shadow.kernel_dispatches % 3U, 0U);

    const auto stage_metrics = std::array{
        std::cref(metrics.camera),       std::cref(metrics.intersection), std::cref(metrics.hit),
        std::cref(metrics.miss),         std::cref(metrics.shade),        std::cref(metrics.shadow),
        std::cref(metrics.continuation),
    };
    for (const auto metric : stage_metrics) {
        EXPECT_GT(metric.get().kernel_dispatches, 0U);
        EXPECT_GT(metric.get().kernel_lanes, 0U);
        EXPECT_GT(metric.get().gpu_elapsed_nanoseconds, 0U);
    }

    const auto replay =
        trace_cuda_wavefront_transport(*workspace, *scene_soa_, *scene_bvh_, *inputs,
                                       std::cref(*light_sampler_), profiled_options);
    ASSERT_TRUE(replay.has_value()) << replay.error().message;
    expect_transport_paths_exact(*profiled, *replay);
    EXPECT_EQ(replay->report, profiled->report)
        << "Observational GPU durations must not make deterministic reports unequal.";

    const auto close_status = workspace->close();
    ASSERT_TRUE(close_status) << close_status.error().message;
}

TEST_F(CudaCornellWavefrontSmokeTest, SynchronousAndAsynchronousTransfersMatchExactly) {
    const auto camera = cornell_wavefront_test::make_camera(CornellExtent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    const auto full_inputs = make_pixel_inputs(*camera, wavelengths(), 32U, 32U, 0U, 8U);
    ASSERT_TRUE(full_inputs.has_value()) << full_inputs.error().message;
    const auto smaller_inputs = make_pixel_inputs(*camera, wavelengths(), 31U, 33U, 97U, 3U);
    ASSERT_TRUE(smaller_inputs.has_value()) << smaller_inputs.error().message;
    auto synchronous_options = transport_options();
    synchronous_options.transfer_mode = CudaWavefrontTransferMode::synchronous;
    auto asynchronous_options = synchronous_options;
    asynchronous_options.transfer_mode = CudaWavefrontTransferMode::asynchronous;

    auto synchronous_workspace = CudaWavefrontTransportWorkspace::create(
        8U, xpu::cuda::DeviceMemoryBudget{}, CudaWavefrontTransferMode::synchronous);
    ASSERT_TRUE(synchronous_workspace.has_value()) << synchronous_workspace.error().message;
    auto asynchronous_workspace = CudaWavefrontTransportWorkspace::create(
        8U, xpu::cuda::DeviceMemoryBudget{}, CudaWavefrontTransferMode::asynchronous);
    ASSERT_TRUE(asynchronous_workspace.has_value()) << asynchronous_workspace.error().message;
    EXPECT_EQ(synchronous_workspace->transfer_mode(), CudaWavefrontTransferMode::synchronous);
    EXPECT_EQ(asynchronous_workspace->transfer_mode(), CudaWavefrontTransferMode::asynchronous);
    EXPECT_EQ(synchronous_workspace->instrumentation_mode(),
              CudaWavefrontInstrumentationMode::disabled);
    EXPECT_EQ(asynchronous_workspace->instrumentation_mode(),
              CudaWavefrontInstrumentationMode::disabled);
    EXPECT_EQ(synchronous_workspace->capacity(), 8U);
    EXPECT_EQ(asynchronous_workspace->capacity(), 8U);
    EXPECT_EQ(synchronous_workspace->device_ordinal(), scene_soa_->device_ordinal());
    EXPECT_EQ(asynchronous_workspace->device_ordinal(), scene_soa_->device_ordinal());

    const auto synchronous_full_first = trace_cuda_wavefront_transport(
        *synchronous_workspace, *scene_soa_, *scene_bvh_, *full_inputs, std::cref(*light_sampler_),
        synchronous_options);
    ASSERT_TRUE(synchronous_full_first.has_value()) << synchronous_full_first.error().message;
    const auto asynchronous_full_first = trace_cuda_wavefront_transport(
        *asynchronous_workspace, *scene_soa_, *scene_bvh_, *full_inputs, std::cref(*light_sampler_),
        asynchronous_options);
    ASSERT_TRUE(asynchronous_full_first.has_value()) << asynchronous_full_first.error().message;
    expect_transport_batches_physically_exact(*synchronous_full_first, *asynchronous_full_first);
    EXPECT_TRUE(std::ranges::any_of(synchronous_full_first->terminal_cones, [](const auto cone) {
        return cone.width() > 0.0F && cone.spread() > 0.0F;
    }));
    EXPECT_TRUE(std::ranges::any_of(asynchronous_full_first->terminal_cones, [](const auto cone) {
        return cone.width() > 0.0F && cone.spread() > 0.0F;
    }));
    expect_synchronous_transfer_report(synchronous_full_first->report);
    expect_asynchronous_transfer_report(asynchronous_full_first->report);

    const auto synchronous_smaller = trace_cuda_wavefront_transport(
        *synchronous_workspace, *scene_soa_, *scene_bvh_, *smaller_inputs,
        std::cref(*light_sampler_), synchronous_options);
    ASSERT_TRUE(synchronous_smaller.has_value()) << synchronous_smaller.error().message;
    const auto asynchronous_smaller = trace_cuda_wavefront_transport(
        *asynchronous_workspace, *scene_soa_, *scene_bvh_, *smaller_inputs,
        std::cref(*light_sampler_), asynchronous_options);
    ASSERT_TRUE(asynchronous_smaller.has_value()) << asynchronous_smaller.error().message;
    expect_transport_batches_physically_exact(*synchronous_smaller, *asynchronous_smaller);
    expect_synchronous_transfer_report(synchronous_smaller->report);
    expect_asynchronous_transfer_report(asynchronous_smaller->report);

    const auto synchronous_full_second = trace_cuda_wavefront_transport(
        *synchronous_workspace, *scene_soa_, *scene_bvh_, *full_inputs, std::cref(*light_sampler_),
        synchronous_options);
    ASSERT_TRUE(synchronous_full_second.has_value()) << synchronous_full_second.error().message;
    const auto asynchronous_full_second = trace_cuda_wavefront_transport(
        *asynchronous_workspace, *scene_soa_, *scene_bvh_, *full_inputs, std::cref(*light_sampler_),
        asynchronous_options);
    ASSERT_TRUE(asynchronous_full_second.has_value()) << asynchronous_full_second.error().message;
    expect_transport_batches_physically_exact(*synchronous_full_second, *asynchronous_full_second);
    expect_transport_batches_exact(*synchronous_full_first, *synchronous_full_second);
    expect_transport_batches_exact(*asynchronous_full_first, *asynchronous_full_second);
    expect_synchronous_transfer_report(synchronous_full_second->report);
    expect_asynchronous_transfer_report(asynchronous_full_second->report);

    EXPECT_TRUE(synchronous_workspace->close().has_value());
    EXPECT_TRUE(asynchronous_workspace->close().has_value());
}

TEST_F(CudaCornellWavefrontSmokeTest,
       RejectsInvalidOrMismatchedRuntimeModesWithoutPoisoningWorkspace) {
    const auto camera = cornell_wavefront_test::make_camera(CornellExtent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    const auto inputs = make_pixel_inputs(*camera, wavelengths(), 32U, 32U, 0U, 3U);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    constexpr auto invalid_mode = static_cast<CudaWavefrontTransferMode>(0xFFU);
    const auto invalid_workspace =
        CudaWavefrontTransportWorkspace::create(3U, xpu::cuda::DeviceMemoryBudget{}, invalid_mode);
    ASSERT_FALSE(invalid_workspace.has_value());
    EXPECT_EQ(invalid_workspace.error().code, core::StatusCode::invalid_argument);
    constexpr auto invalid_instrumentation = static_cast<CudaWavefrontInstrumentationMode>(0xFFU);
    const auto invalid_instrumented_workspace = CudaWavefrontTransportWorkspace::create(
        3U, xpu::cuda::DeviceMemoryBudget{}, CudaWavefrontTransferMode::synchronous,
        invalid_instrumentation);
    ASSERT_FALSE(invalid_instrumented_workspace.has_value());
    EXPECT_EQ(invalid_instrumented_workspace.error().code, core::StatusCode::invalid_argument);

    auto synchronous_options = transport_options();
    synchronous_options.transfer_mode = CudaWavefrontTransferMode::synchronous;
    auto asynchronous_options = synchronous_options;
    asynchronous_options.transfer_mode = CudaWavefrontTransferMode::asynchronous;
    auto invalid_options = synchronous_options;
    invalid_options.transfer_mode = invalid_mode;
    auto invalid_instrumentation_options = synchronous_options;
    invalid_instrumentation_options.instrumentation_mode = invalid_instrumentation;
    auto nsight_options = synchronous_options;
    nsight_options.instrumentation_mode = CudaWavefrontInstrumentationMode::nsight;

    const auto baseline = trace_cuda_wavefront_transport(
        *scene_soa_, *scene_bvh_, *inputs, std::cref(*light_sampler_), synchronous_options);
    ASSERT_TRUE(baseline.has_value()) << baseline.error().message;
    expect_synchronous_transfer_report(baseline->report);

    auto synchronous_workspace = CudaWavefrontTransportWorkspace::create(
        3U, xpu::cuda::DeviceMemoryBudget{}, CudaWavefrontTransferMode::synchronous);
    ASSERT_TRUE(synchronous_workspace.has_value()) << synchronous_workspace.error().message;
    const auto invalid_trace =
        trace_cuda_wavefront_transport(*synchronous_workspace, *scene_soa_, *scene_bvh_, *inputs,
                                       std::cref(*light_sampler_), invalid_options);
    ASSERT_FALSE(invalid_trace.has_value());
    EXPECT_EQ(invalid_trace.error().code, core::StatusCode::invalid_argument);
    const auto invalid_instrumentation_trace =
        trace_cuda_wavefront_transport(*synchronous_workspace, *scene_soa_, *scene_bvh_, *inputs,
                                       std::cref(*light_sampler_), invalid_instrumentation_options);
    ASSERT_FALSE(invalid_instrumentation_trace.has_value());
    EXPECT_EQ(invalid_instrumentation_trace.error().code, core::StatusCode::invalid_argument);
    const auto asynchronous_mismatch =
        trace_cuda_wavefront_transport(*synchronous_workspace, *scene_soa_, *scene_bvh_, *inputs,
                                       std::cref(*light_sampler_), asynchronous_options);
    ASSERT_FALSE(asynchronous_mismatch.has_value());
    EXPECT_EQ(asynchronous_mismatch.error().code, core::StatusCode::incompatible);
    const auto instrumentation_mismatch =
        trace_cuda_wavefront_transport(*synchronous_workspace, *scene_soa_, *scene_bvh_, *inputs,
                                       std::cref(*light_sampler_), nsight_options);
    ASSERT_FALSE(instrumentation_mismatch.has_value());
    EXPECT_EQ(instrumentation_mismatch.error().code, core::StatusCode::incompatible);
    const auto synchronous_recovered =
        trace_cuda_wavefront_transport(*synchronous_workspace, *scene_soa_, *scene_bvh_, *inputs,
                                       std::cref(*light_sampler_), synchronous_options);
    ASSERT_TRUE(synchronous_recovered.has_value()) << synchronous_recovered.error().message;
    expect_transport_batches_exact(*baseline, *synchronous_recovered);

    auto asynchronous_workspace = CudaWavefrontTransportWorkspace::create(
        3U, xpu::cuda::DeviceMemoryBudget{}, CudaWavefrontTransferMode::asynchronous);
    ASSERT_TRUE(asynchronous_workspace.has_value()) << asynchronous_workspace.error().message;
    const auto synchronous_mismatch =
        trace_cuda_wavefront_transport(*asynchronous_workspace, *scene_soa_, *scene_bvh_, *inputs,
                                       std::cref(*light_sampler_), synchronous_options);
    ASSERT_FALSE(synchronous_mismatch.has_value());
    EXPECT_EQ(synchronous_mismatch.error().code, core::StatusCode::incompatible);
    const auto asynchronous_recovered =
        trace_cuda_wavefront_transport(*asynchronous_workspace, *scene_soa_, *scene_bvh_, *inputs,
                                       std::cref(*light_sampler_), asynchronous_options);
    ASSERT_TRUE(asynchronous_recovered.has_value()) << asynchronous_recovered.error().message;
    expect_transport_batches_physically_exact(*baseline, *asynchronous_recovered);
    expect_asynchronous_transfer_report(asynchronous_recovered->report);

    EXPECT_TRUE(synchronous_workspace->close().has_value());
    EXPECT_TRUE(asynchronous_workspace->close().has_value());
}

TEST_F(CudaCornellWavefrontSmokeTest, CapacityAndBudgetRejectionsDoNotPoisonWorkspace) {
    const auto camera = cornell_wavefront_test::make_camera(CornellExtent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    const auto inputs = make_pixel_inputs(*camera, wavelengths(), 32U, 32U, 0U, 5U);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    const auto valid_inputs =
        std::span<const CudaWavefrontPathInput>{inputs->data(), std::size_t{4U}};
    const auto options = transport_options();
    const auto baseline = trace_cuda_wavefront_transport(*scene_soa_, *scene_bvh_, valid_inputs,
                                                         std::cref(*light_sampler_), options);
    ASSERT_TRUE(baseline.has_value()) << baseline.error().message;

    auto workspace = CudaWavefrontTransportWorkspace::create(4U);
    ASSERT_TRUE(workspace.has_value()) << workspace.error().message;
    const auto initial_device_bytes = workspace->device_size_bytes();
    ASSERT_GT(initial_device_bytes, 0U);
    const auto over_capacity = trace_cuda_wavefront_transport(
        *workspace, *scene_soa_, *scene_bvh_, *inputs, std::cref(*light_sampler_), options);
    ASSERT_FALSE(over_capacity.has_value());
    EXPECT_EQ(over_capacity.error().code, core::StatusCode::resource_exhausted);

    auto insufficient_budget = options;
    insufficient_budget.device_memory_budget =
        xpu::cuda::DeviceMemoryBudget{.maximum_bytes = initial_device_bytes - 1U};
    const auto budget_rejected =
        trace_cuda_wavefront_transport(*workspace, *scene_soa_, *scene_bvh_, valid_inputs,
                                       std::cref(*light_sampler_), insufficient_budget);
    ASSERT_FALSE(budget_rejected.has_value());
    EXPECT_EQ(budget_rejected.error().code, core::StatusCode::resource_exhausted);

    const auto recovered = trace_cuda_wavefront_transport(
        *workspace, *scene_soa_, *scene_bvh_, valid_inputs, std::cref(*light_sampler_), options);
    ASSERT_TRUE(recovered.has_value()) << recovered.error().message;
    expect_transport_batches_exact(*baseline, *recovered);
    EXPECT_EQ(workspace->capacity(), 4U);
    EXPECT_EQ(workspace->device_size_bytes(), initial_device_bytes);
    EXPECT_EQ(workspace->device_ordinal(), scene_soa_->device_ordinal());
    EXPECT_TRUE(workspace->close().has_value());
}

TEST_F(CudaCornellWavefrontSmokeTest, CloseIsIdempotentAndClosedWorkspaceRefusesTrace) {
    const auto camera = cornell_wavefront_test::make_camera(CornellExtent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    const auto inputs = make_pixel_inputs(*camera, wavelengths(), 32U, 32U, 0U, 2U);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    auto workspace = CudaWavefrontTransportWorkspace::create(2U);
    ASSERT_TRUE(workspace.has_value()) << workspace.error().message;
    const auto traced =
        trace_cuda_wavefront_transport(*workspace, *scene_soa_, *scene_bvh_, *inputs,
                                       std::cref(*light_sampler_), transport_options());
    ASSERT_TRUE(traced.has_value()) << traced.error().message;

    EXPECT_TRUE(workspace->close().has_value());
    EXPECT_TRUE(workspace->close().has_value());
    EXPECT_FALSE(static_cast<bool>(*workspace));
    EXPECT_EQ(workspace->capacity(), 0U);
    EXPECT_EQ(workspace->device_size_bytes(), 0U);
    EXPECT_EQ(workspace->device_ordinal(), -1);
    const auto rejected =
        trace_cuda_wavefront_transport(*workspace, *scene_soa_, *scene_bvh_, *inputs,
                                       std::cref(*light_sampler_), transport_options());
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error().code, core::StatusCode::invalid_argument);
}

TEST_F(CudaCornellWavefrontSmokeTest, RejectsInsufficientDeviceBudgetWithoutFallback) {
    const auto camera = cornell_wavefront_test::make_camera(CornellExtent);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;
    const auto inputs = make_pixel_inputs(*camera, wavelengths(), 32U, 32U, 0U, 1U);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    auto options = transport_options();
    options.device_memory_budget = xpu::cuda::DeviceMemoryBudget{.maximum_bytes = 1U};
    const auto rejected = trace_cuda_wavefront_transport(*scene_soa_, *scene_bvh_, *inputs,
                                                         std::cref(*light_sampler_), options);
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error().code, core::StatusCode::resource_exhausted);
    EXPECT_FALSE(rejected.error().message.empty());

    const auto workspace_rejected = CudaWavefrontTransportWorkspace::create(
        1U, xpu::cuda::DeviceMemoryBudget{.maximum_bytes = 1U});
    ASSERT_FALSE(workspace_rejected.has_value());
    EXPECT_EQ(workspace_rejected.error().code, core::StatusCode::resource_exhausted);

    auto sized_workspace = CudaWavefrontTransportWorkspace::create(1U);
    ASSERT_TRUE(sized_workspace.has_value()) << sized_workspace.error().message;
    const auto exact_workspace_budget = sized_workspace->device_size_bytes();
    ASSERT_GT(exact_workspace_budget, 1U);
    ASSERT_TRUE(sized_workspace->close().has_value());

    const auto aggregate_rejected = CudaWavefrontTransportWorkspace::create(
        1U, xpu::cuda::DeviceMemoryBudget{.maximum_bytes = exact_workspace_budget - 1U});
    ASSERT_FALSE(aggregate_rejected.has_value());
    EXPECT_EQ(aggregate_rejected.error().code, core::StatusCode::resource_exhausted);
    EXPECT_NE(aggregate_rejected.error().message.find("aggregate device-memory budget"),
              std::string::npos);

    auto exact_workspace = CudaWavefrontTransportWorkspace::create(
        1U, xpu::cuda::DeviceMemoryBudget{.maximum_bytes = exact_workspace_budget});
    ASSERT_TRUE(exact_workspace.has_value()) << exact_workspace.error().message;
    EXPECT_EQ(exact_workspace->device_size_bytes(), exact_workspace_budget);
    EXPECT_TRUE(exact_workspace->close().has_value());
}

} // namespace
} // namespace blackframe::engine
