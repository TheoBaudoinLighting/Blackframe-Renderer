#include <Blackframe/Backends/CPU/Embree/AccelBackend.hpp>
#include <Blackframe/Backends/CPU/Embree/WavefrontMisTransport.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/SceneMisPathLoop.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Renderer/CpuWavefrontScheduler.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/PathDepthLimits.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <Blackframe/Renderer/RussianRoulette.hpp>
#include <Blackframe/Renderer/SampleDimensionMap.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

inline constexpr auto PathCount = std::size_t{24U};
inline constexpr auto PathTime = renderer::TransportScalar{0.375F};
inline constexpr auto ScalarParityTolerance = renderer::TransportScalar{1.0e-5F};
inline constexpr auto OneDiffuseBounce = renderer::PathDepthLimits{.diffuse = 1U};
inline constexpr auto EvaluationSeed = std::uint64_t{0x243F6A8885A308D3ULL};

[[nodiscard]] renderer::TransportSpectrum constant_spectrum(const float value) {
    auto spectrum = renderer::TransportSpectrum{};
    spectrum.values.fill(value);
    return spectrum;
}

[[nodiscard]] core::Result<std::shared_ptr<const TriangleMesh>>
horizontal_quad(const float half_extent, const float height, const bool faces_up) {
    const auto normal = faces_up ? renderer::Normal3{.z = 1.0F} : renderer::Normal3{.z = -1.0F};
    auto mesh = TriangleMesh::create(
        {
            renderer::Point3{.x = -half_extent, .y = -half_extent, .z = height},
            renderer::Point3{.x = half_extent, .y = -half_extent, .z = height},
            renderer::Point3{.x = half_extent, .y = half_extent, .z = height},
            renderer::Point3{.x = -half_extent, .y = half_extent, .z = height},
        },
        std::vector(4U, normal),
        {
            renderer::Point2{},
            renderer::Point2{.x = 1.0F},
            renderer::Point2{.x = 1.0F, .y = 1.0F},
            renderer::Point2{.y = 1.0F},
        },
        faces_up ? std::vector{
                       TriangleVertexIndices{.vertices = {0U, 1U, 2U}},
                       TriangleVertexIndices{.vertices = {0U, 2U, 3U}},
                   }
                 : std::vector{
                       TriangleVertexIndices{.vertices = {0U, 2U, 1U}},
                       TriangleVertexIndices{.vertices = {0U, 3U, 2U}},
                   });
    if (!mesh) {
        return std::unexpected(mesh.error());
    }
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

[[nodiscard]] core::Result<FrameSceneHandle>
make_wavefront_scene(const float environment_radiance = 0.0F,
                     const float receiver_reflectance = 0.65F) {
    auto receiver = horizontal_quad(4.0F, 0.0F, true);
    // This finite ceiling is deliberately much wider than the fixed sample set. Every sampled
    // continuation reaches it, which makes all seven queue counters analytically predictable.
    auto emitter = horizontal_quad(2'048.0F, 2.0F, false);
    if (!receiver) {
        return std::unexpected(receiver.error());
    }
    if (!emitter) {
        return std::unexpected(emitter.error());
    }

    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.25F);
    if (!wavelengths) {
        return std::unexpected(wavelengths.error());
    }
    return FrameScene::create(FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 1U}}, SceneObject{.id = {.value = 2U}}},
        .geometries =
            {
                SceneGeometry{.id = {.value = 11U}, .mesh = std::move(*receiver)},
                SceneGeometry{.id = {.value = 12U}, .mesh = std::move(*emitter)},
            },
        .materials =
            {
                SceneMaterial{
                    .id = {.value = 21U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .reflectance = constant_spectrum(receiver_reflectance),
                            .emitted_radiance = {},
                        },
                },
                SceneMaterial{
                    .id = {.value = 22U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .reflectance = {},
                            .emitted_radiance = constant_spectrum(4.0F),
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
                SceneInstance{
                    .id = {.value = 32U},
                    .parent = std::nullopt,
                    .object = {.value = 2U},
                    .geometry = {.value = 12U},
                    .material = {.value = 22U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
            },
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = *wavelengths,
                .radiance = constant_spectrum(environment_radiance),
            },
    });
}

[[nodiscard]] core::Result<FrameSceneHandle> make_occluded_overflow_scene() {
    auto receiver = horizontal_quad(4.0F, 0.0F, true);
    auto blocker = horizontal_quad(2'048.0F, 1.5F, false);
    if (!receiver) {
        return std::unexpected(receiver.error());
    }
    if (!blocker) {
        return std::unexpected(blocker.error());
    }
    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.25F);
    if (!wavelengths) {
        return std::unexpected(wavelengths.error());
    }

    return FrameScene::create(FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 1U}}, SceneObject{.id = {.value = 2U}}},
        .geometries =
            {
                SceneGeometry{.id = {.value = 11U}, .mesh = std::move(*receiver)},
                SceneGeometry{.id = {.value = 12U}, .mesh = std::move(*blocker)},
            },
        .materials =
            {
                SceneMaterial{
                    .id = {.value = 21U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .reflectance = constant_spectrum(0.5F),
                            .emitted_radiance = {},
                        },
                },
                SceneMaterial{
                    .id = {.value = 22U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .reflectance = {},
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
                SceneInstance{
                    .id = {.value = 32U},
                    .parent = std::nullopt,
                    .object = {.value = 2U},
                    .geometry = {.value = 12U},
                    .material = {.value = 22U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
            },
        .punctual_lights =
            {
                ScenePointLight{
                    .position = renderer::Point3{.z = 2.0F},
                    .absolute_position_error = {},
                    .spectral_radiant_intensity = constant_spectrum(0x1p120F),
                },
            },
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = *wavelengths,
                .radiance = {},
            },
    });
}

[[nodiscard]] core::Result<std::vector<CpuWavefrontMisPathInput>>
make_inputs(const FrameSceneHandle& scene) {
    const auto ray =
        renderer::Ray::create(renderer::Point3{.z = 1.0F}, renderer::Vector3{.z = -1.0F}, 0.0F,
                              std::numeric_limits<renderer::TransportScalar>::infinity(), PathTime,
                              renderer::AllRayVisibility, renderer::VacuumMedium);
    if (!ray) {
        return std::unexpected(ray.error());
    }
    const auto state = renderer::PathState::create_initial(
        scene->spectral_environment()->wavelengths, renderer::VacuumMedium);
    if (!state) {
        return std::unexpected(state.error());
    }

    auto inputs = std::vector<CpuWavefrontMisPathInput>{};
    inputs.reserve(PathCount);
    for (auto index = std::size_t{}; index < PathCount; ++index) {
        inputs.push_back(CpuWavefrontMisPathInput{
            .primary_ray = *ray,
            .initial_state = *state,
            .sample =
                renderer::SampleStreamIndex{
                    .pixel_x = static_cast<std::uint32_t>(index % 6U),
                    .pixel_y = static_cast<std::uint32_t>(index / 6U),
                    .sample_index = static_cast<std::uint64_t>(index),
                    .seed = EvaluationSeed,
                },
        });
    }
    return inputs;
}

void expect_finite_near(const float expected, const float actual, const float tolerance) {
    if (std::isfinite(expected)) {
        EXPECT_NEAR(actual, expected, tolerance);
    } else {
        EXPECT_EQ(actual, expected);
    }
}

void expect_path_near(const renderer::BsdfOnlyPathResult& expected,
                      const renderer::BsdfOnlyPathResult& actual, const float tolerance) {
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(actual.state.beta()[lane], expected.state.beta()[lane], tolerance)
            << "spectral lane " << lane;
        EXPECT_NEAR(actual.state.accumulated_radiance()[lane],
                    expected.state.accumulated_radiance()[lane], tolerance)
            << "spectral lane " << lane;
    }
    EXPECT_EQ(actual.state.depth(), expected.state.depth());
    EXPECT_EQ(actual.state.depth_counters(), expected.state.depth_counters());
    EXPECT_NEAR(actual.state.eta_scale(), expected.state.eta_scale(), tolerance);
    EXPECT_EQ(actual.state.wavelengths(), expected.state.wavelengths());
    EXPECT_EQ(actual.state.delta_flags(), expected.state.delta_flags());
    EXPECT_EQ(actual.state.current_medium(), expected.state.current_medium());

    expect_finite_near(expected.terminal_ray.origin().x, actual.terminal_ray.origin().x, tolerance);
    expect_finite_near(expected.terminal_ray.origin().y, actual.terminal_ray.origin().y, tolerance);
    expect_finite_near(expected.terminal_ray.origin().z, actual.terminal_ray.origin().z, tolerance);
    expect_finite_near(expected.terminal_ray.direction().x, actual.terminal_ray.direction().x,
                       tolerance);
    expect_finite_near(expected.terminal_ray.direction().y, actual.terminal_ray.direction().y,
                       tolerance);
    expect_finite_near(expected.terminal_ray.direction().z, actual.terminal_ray.direction().z,
                       tolerance);
    expect_finite_near(expected.terminal_ray.t_min(), actual.terminal_ray.t_min(), tolerance);
    expect_finite_near(expected.terminal_ray.t_max(), actual.terminal_ray.t_max(), tolerance);
    expect_finite_near(expected.terminal_ray.time(), actual.terminal_ray.time(), tolerance);
    EXPECT_EQ(actual.terminal_ray.mask(), expected.terminal_ray.mask());
    EXPECT_EQ(actual.terminal_ray.current_medium(), expected.terminal_ray.current_medium());
    EXPECT_EQ(actual.termination, expected.termination);
    EXPECT_EQ(actual.blocked_depth_limits, expected.blocked_depth_limits);
}

void expect_path_exact(const renderer::BsdfOnlyPathResult& expected,
                       const renderer::BsdfOnlyPathResult& actual) {
    EXPECT_EQ(actual.state.beta(), expected.state.beta());
    EXPECT_EQ(actual.state.accumulated_radiance(), expected.state.accumulated_radiance());
    EXPECT_EQ(actual.state.depth(), expected.state.depth());
    EXPECT_EQ(actual.state.depth_counters(), expected.state.depth_counters());
    EXPECT_EQ(actual.state.eta_scale(), expected.state.eta_scale());
    EXPECT_EQ(actual.state.wavelengths(), expected.state.wavelengths());
    EXPECT_EQ(actual.state.delta_flags(), expected.state.delta_flags());
    EXPECT_EQ(actual.state.current_medium(), expected.state.current_medium());
    EXPECT_EQ(actual.terminal_ray.origin(), expected.terminal_ray.origin());
    EXPECT_EQ(actual.terminal_ray.direction(), expected.terminal_ray.direction());
    EXPECT_EQ(actual.terminal_ray.t_min(), expected.terminal_ray.t_min());
    EXPECT_EQ(actual.terminal_ray.t_max(), expected.terminal_ray.t_max());
    EXPECT_EQ(actual.terminal_ray.time(), expected.terminal_ray.time());
    EXPECT_EQ(actual.terminal_ray.mask(), expected.terminal_ray.mask());
    EXPECT_EQ(actual.terminal_ray.current_medium(), expected.terminal_ray.current_medium());
    EXPECT_EQ(actual.termination, expected.termination);
    EXPECT_EQ(actual.blocked_depth_limits, expected.blocked_depth_limits);
}

[[nodiscard]] core::Result<std::vector<renderer::BsdfOnlyPathResult>>
trace_scalar_batch(const std::span<const CpuWavefrontMisPathInput> inputs,
                   const AccelBackend& acceleration, const renderer::LightSampler& sampler,
                   const renderer::MisHeuristic heuristic,
                   const renderer::PathDepthLimits& depth_limits,
                   const renderer::RussianRoulettePolicy& roulette_policy) {
    auto paths = std::vector<renderer::BsdfOnlyPathResult>{};
    paths.reserve(inputs.size());
    for (const auto& input : inputs) {
        const auto traced = trace_scene_mis(input.primary_ray, input.initial_state,
                                            renderer::SampleStream{input.sample}, acceleration,
                                            sampler, heuristic, depth_limits, roulette_policy);
        if (!traced) {
            return std::unexpected(traced.error());
        }
        paths.push_back(*traced);
    }
    return paths;
}

class WavefrontMisTransportTest : public testing::TestWithParam<renderer::MisHeuristic> {};

TEST_P(WavefrontMisTransportTest, MatchesAnalyticScalarAndIsIdenticalWithOneOrManyThreads) {
    const auto scene = make_wavefront_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    ASSERT_EQ((*scene)->mesh_area_lights().size(), 1U);
    const auto inputs = make_inputs(*scene);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    const auto sampler = renderer::LightSampler::create_uniform(1U);
    ASSERT_TRUE(sampler.has_value()) << sampler.error().message;

    const auto scalar_backend = create_analytic_accel_backend(*scene);
    ASSERT_TRUE(scalar_backend.has_value()) << scalar_backend.error().message;
    const auto scalar =
        trace_scalar_batch(*inputs, **scalar_backend, *sampler, GetParam(), OneDiffuseBounce,
                           renderer::RussianRoulettePolicy::disabled());
    ASSERT_TRUE(scalar.has_value()) << scalar.error().message;

    const auto single_backend = create_embree_accel_backend(*scene);
    const auto parallel_backend = create_embree_accel_backend(*scene);
    ASSERT_TRUE(single_backend.has_value()) << single_backend.error().message;
    ASSERT_TRUE(parallel_backend.has_value()) << parallel_backend.error().message;
    const auto single_scheduler = renderer::CpuWavefrontScheduler::create(1U);
    const auto parallel_scheduler = renderer::CpuWavefrontScheduler::create(4U);
    ASSERT_TRUE(single_scheduler.has_value()) << single_scheduler.error().message;
    ASSERT_TRUE(parallel_scheduler.has_value()) << parallel_scheduler.error().message;

    const auto single =
        trace_cpu_wavefront_mis(*inputs, *single_scheduler, **single_backend, *sampler, GetParam(),
                                OneDiffuseBounce, renderer::RussianRoulettePolicy::disabled());
    const auto parallel = trace_cpu_wavefront_mis(*inputs, *parallel_scheduler, **parallel_backend,
                                                  *sampler, GetParam(), OneDiffuseBounce,
                                                  renderer::RussianRoulettePolicy::disabled());
    ASSERT_TRUE(single.has_value()) << single.error().message;
    ASSERT_TRUE(parallel.has_value()) << parallel.error().message;
    ASSERT_EQ(single->paths.size(), PathCount);
    ASSERT_EQ(parallel->paths.size(), PathCount);
    ASSERT_EQ(scalar->size(), PathCount);

    for (auto index = std::size_t{}; index < PathCount; ++index) {
        SCOPED_TRACE(index);
        expect_path_near((*scalar)[index], single->paths[index], ScalarParityTolerance);
        expect_path_near((*scalar)[index], parallel->paths[index], ScalarParityTolerance);
        expect_path_exact(single->paths[index], parallel->paths[index]);
        EXPECT_EQ(single->paths[index].termination, renderer::BsdfOnlyPathTermination::depth_limit);
    }

    const auto lane_count = static_cast<std::uint64_t>(PathCount);
    const auto expected_stages = CpuWavefrontMisStageLaneCounts{
        .camera = lane_count,
        .ray = lane_count * 2U,
        .hit = lane_count * 2U,
        .miss = 0U,
        .shade = lane_count * 2U,
        .shadow = lane_count,
        .continuation = lane_count,
    };
    for (const auto* const report : std::array{&single->report, &parallel->report}) {
        EXPECT_EQ(report->schema_version, CurrentCpuWavefrontMisReportSchemaVersion);
        EXPECT_EQ(report->path_count, PathCount);
        EXPECT_EQ(report->stage_lanes, expected_stages);
        EXPECT_EQ(report->closure_samples, lane_count);
        EXPECT_EQ(report->light_samples, lane_count);
        EXPECT_EQ(report->shadow_queries, lane_count);
    }
    EXPECT_EQ(single->report.configured_workers, 1U);
    EXPECT_EQ(parallel->report.configured_workers, 4U);
    EXPECT_EQ(single->report.stage_lanes, parallel->report.stage_lanes);
    EXPECT_EQ(single->report.closure_samples, parallel->report.closure_samples);
    EXPECT_EQ(single->report.light_samples, parallel->report.light_samples);
    EXPECT_EQ(single->report.shadow_queries, parallel->report.shadow_queries);
}

INSTANTIATE_TEST_SUITE_P(BalanceAndPower, WavefrontMisTransportTest,
                         testing::Values(renderer::MisHeuristic::balance,
                                         renderer::MisHeuristic::power),
                         [](const testing::TestParamInfo<renderer::MisHeuristic>& parameter) {
                             return parameter.param == renderer::MisHeuristic::balance ? "Balance"
                                                                                       : "Power";
                         });

TEST(WavefrontMisTransportQueueTest, PreservesPrimaryMissEnvironmentParityWithOneOrManyThreads) {
    constexpr auto environment_radiance = renderer::TransportScalar{0.75F};
    const auto scene = make_wavefront_scene(environment_radiance);
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    auto inputs = make_inputs(*scene);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    const auto miss_ray =
        renderer::Ray::create(renderer::Point3{.z = 1.0F}, renderer::Vector3{.x = 1.0F}, 0.0F,
                              std::numeric_limits<renderer::TransportScalar>::infinity(), PathTime,
                              renderer::AllRayVisibility, renderer::VacuumMedium);
    ASSERT_TRUE(miss_ray.has_value()) << miss_ray.error().message;
    inputs->front().primary_ray = *miss_ray;

    const auto sampler = renderer::LightSampler::create_uniform(1U);
    const auto scalar_backend = create_analytic_accel_backend(*scene);
    const auto single_backend = create_embree_accel_backend(*scene);
    const auto parallel_backend = create_embree_accel_backend(*scene);
    const auto single_scheduler = renderer::CpuWavefrontScheduler::create(1U);
    const auto parallel_scheduler = renderer::CpuWavefrontScheduler::create(4U);
    ASSERT_TRUE(sampler.has_value()) << sampler.error().message;
    ASSERT_TRUE(scalar_backend.has_value()) << scalar_backend.error().message;
    ASSERT_TRUE(single_backend.has_value()) << single_backend.error().message;
    ASSERT_TRUE(parallel_backend.has_value()) << parallel_backend.error().message;
    ASSERT_TRUE(single_scheduler.has_value()) << single_scheduler.error().message;
    ASSERT_TRUE(parallel_scheduler.has_value()) << parallel_scheduler.error().message;

    constexpr auto heuristic = renderer::MisHeuristic::power;
    const auto scalar =
        trace_scalar_batch(*inputs, **scalar_backend, *sampler, heuristic, OneDiffuseBounce,
                           renderer::RussianRoulettePolicy::disabled());
    const auto single =
        trace_cpu_wavefront_mis(*inputs, *single_scheduler, **single_backend, *sampler, heuristic,
                                OneDiffuseBounce, renderer::RussianRoulettePolicy::disabled());
    const auto parallel = trace_cpu_wavefront_mis(*inputs, *parallel_scheduler, **parallel_backend,
                                                  *sampler, heuristic, OneDiffuseBounce,
                                                  renderer::RussianRoulettePolicy::disabled());
    ASSERT_TRUE(scalar.has_value()) << scalar.error().message;
    ASSERT_TRUE(single.has_value()) << single.error().message;
    ASSERT_TRUE(parallel.has_value()) << parallel.error().message;
    ASSERT_EQ(scalar->size(), PathCount);
    ASSERT_EQ(single->paths.size(), PathCount);
    ASSERT_EQ(parallel->paths.size(), PathCount);

    for (auto index = std::size_t{}; index < PathCount; ++index) {
        SCOPED_TRACE(index);
        expect_path_near((*scalar)[index], single->paths[index], ScalarParityTolerance);
        expect_path_near((*scalar)[index], parallel->paths[index], ScalarParityTolerance);
        expect_path_exact(single->paths[index], parallel->paths[index]);
    }
    EXPECT_EQ(single->paths.front().termination,
              renderer::BsdfOnlyPathTermination::escaped_environment);
    EXPECT_EQ(single->paths.front().state.depth(), 0U);
    for (const auto lane : single->paths.front().state.accumulated_radiance().values) {
        EXPECT_EQ(lane, environment_radiance);
    }

    const auto lane_count = static_cast<std::uint64_t>(PathCount);
    const auto surface_path_count = lane_count - 1U;
    const auto expected_stages = CpuWavefrontMisStageLaneCounts{
        .camera = lane_count,
        .ray = lane_count + surface_path_count,
        .hit = surface_path_count * 2U,
        .miss = 1U,
        .shade = surface_path_count * 2U,
        .shadow = surface_path_count,
        .continuation = surface_path_count,
    };
    for (const auto* const report : std::array{&single->report, &parallel->report}) {
        EXPECT_EQ(report->stage_lanes, expected_stages);
        EXPECT_EQ(report->closure_samples, surface_path_count);
        EXPECT_EQ(report->light_samples, surface_path_count);
        EXPECT_EQ(report->shadow_queries, surface_path_count);
    }
}

TEST(WavefrontMisTransportRussianRouletteTest,
     MatchesAnalyticScalarAndCompactsFirstBounceTerminationsDeterministically) {
    constexpr auto survival_probability = renderer::TransportScalar{0.5F};
    const auto roulette = renderer::RussianRoulettePolicy::create_enabled(1U, survival_probability,
                                                                          survival_probability);
    ASSERT_TRUE(roulette.has_value()) << roulette.error().message;
    constexpr auto depth_limits = renderer::PathDepthLimits{.diffuse = 2U};

    const auto scene = make_wavefront_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto inputs = make_inputs(*scene);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    const auto dimensions = renderer::sample_dimensions_for_bounce(0U);
    ASSERT_TRUE(dimensions.has_value()) << dimensions.error().message;
    auto expected_survivors = std::uint64_t{};
    for (const auto& input : *inputs) {
        const auto sample =
            renderer::SampleStream{input.sample}.sample_1d(dimensions->russian_roulette);
        expected_survivors += sample < survival_probability ? 1U : 0U;
    }
    const auto lane_count = static_cast<std::uint64_t>(inputs->size());
    const auto expected_terminated = lane_count - expected_survivors;
    ASSERT_GT(expected_survivors, 0U);
    ASSERT_GT(expected_terminated, 0U);

    const auto sampler = renderer::LightSampler::create_uniform(1U);
    const auto scalar_backend = create_analytic_accel_backend(*scene);
    const auto single_backend = create_embree_accel_backend(*scene);
    const auto parallel_backend = create_embree_accel_backend(*scene);
    const auto single_scheduler = renderer::CpuWavefrontScheduler::create(1U);
    const auto parallel_scheduler = renderer::CpuWavefrontScheduler::create(4U);
    ASSERT_TRUE(sampler.has_value()) << sampler.error().message;
    ASSERT_TRUE(scalar_backend.has_value()) << scalar_backend.error().message;
    ASSERT_TRUE(single_backend.has_value()) << single_backend.error().message;
    ASSERT_TRUE(parallel_backend.has_value()) << parallel_backend.error().message;
    ASSERT_TRUE(single_scheduler.has_value()) << single_scheduler.error().message;
    ASSERT_TRUE(parallel_scheduler.has_value()) << parallel_scheduler.error().message;

    constexpr auto heuristic = renderer::MisHeuristic::power;
    const auto scalar =
        trace_scalar_batch(*inputs, **scalar_backend, *sampler, heuristic, depth_limits, *roulette);
    const auto single = trace_cpu_wavefront_mis(*inputs, *single_scheduler, **single_backend,
                                                *sampler, heuristic, depth_limits, *roulette);
    const auto parallel = trace_cpu_wavefront_mis(*inputs, *parallel_scheduler, **parallel_backend,
                                                  *sampler, heuristic, depth_limits, *roulette);
    ASSERT_TRUE(scalar.has_value()) << scalar.error().message;
    ASSERT_TRUE(single.has_value()) << single.error().message;
    ASSERT_TRUE(parallel.has_value()) << parallel.error().message;
    ASSERT_EQ(scalar->size(), inputs->size());
    ASSERT_EQ(single->paths.size(), inputs->size());
    ASSERT_EQ(parallel->paths.size(), inputs->size());

    auto terminated = std::uint64_t{};
    auto survived_first_bounce = std::uint64_t{};
    constexpr auto terminal_geometry_tolerance = renderer::TransportScalar{5.0e-4F};
    for (auto index = std::size_t{}; index < inputs->size(); ++index) {
        SCOPED_TRACE(index);
        expect_path_near((*scalar)[index], single->paths[index], terminal_geometry_tolerance);
        expect_path_near((*scalar)[index], parallel->paths[index], terminal_geometry_tolerance);
        expect_path_exact(single->paths[index], parallel->paths[index]);
        for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
            EXPECT_NEAR(single->paths[index].state.beta()[lane],
                        (*scalar)[index].state.beta()[lane], ScalarParityTolerance);
            EXPECT_NEAR(single->paths[index].state.accumulated_radiance()[lane],
                        (*scalar)[index].state.accumulated_radiance()[lane], ScalarParityTolerance);
        }
        const auto& result = single->paths[index];
        switch (result.termination) {
        case renderer::BsdfOnlyPathTermination::russian_roulette:
            ++terminated;
            EXPECT_EQ(result.state.depth(), 1U);
            EXPECT_EQ(result.state.depth_counters().diffuse, 1U);
            EXPECT_EQ(result.blocked_depth_limits, renderer::ScatteringLobe::none);
            break;
        case renderer::BsdfOnlyPathTermination::zero_throughput:
            ++survived_first_bounce;
            EXPECT_GE(result.state.depth(), 2U);
            EXPECT_GE(result.state.depth_counters().diffuse, 2U);
            EXPECT_EQ(result.blocked_depth_limits, renderer::ScatteringLobe::none);
            break;
        default:
            ADD_FAILURE() << "Unexpected roulette-batch termination "
                          << static_cast<unsigned>(result.termination);
            break;
        }
    }
    EXPECT_EQ(terminated, expected_terminated);
    EXPECT_EQ(survived_first_bounce, expected_survivors);

    const auto expected_stages = CpuWavefrontMisStageLaneCounts{
        .camera = lane_count,
        .ray = lane_count + expected_survivors,
        .hit = lane_count + expected_survivors,
        .miss = 0U,
        .shade = lane_count + expected_survivors,
        .shadow = lane_count,
        .continuation = expected_survivors,
    };
    for (const auto* const report : std::array{&single->report, &parallel->report}) {
        EXPECT_EQ(report->stage_lanes, expected_stages);
        EXPECT_EQ(report->closure_samples, lane_count + expected_survivors);
        EXPECT_EQ(report->light_samples, lane_count);
        EXPECT_EQ(report->shadow_queries, lane_count);
    }
    EXPECT_EQ(single->report.configured_workers, 1U);
    EXPECT_EQ(parallel->report.configured_workers, 4U);
}

TEST(WavefrontMisTransportNumericTest,
     PreservesARepresentableHighBetaTimesSubnormalLambertReflectance) {
    constexpr auto initial_beta_value = renderer::TransportScalar{0x1p120F};
    const auto reflectance = std::numeric_limits<renderer::TransportScalar>::denorm_min();
    ASSERT_EQ(std::fpclassify(reflectance), FP_SUBNORMAL);
    const auto expected_throughput = initial_beta_value * reflectance;
    ASSERT_TRUE(std::isnormal(expected_throughput));

    const auto scene = make_wavefront_scene(0.0F, reflectance);
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    auto inputs = make_inputs(*scene);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    inputs->erase(std::next(inputs->begin()), inputs->end());
    const auto high_beta_state = renderer::PathState::create(
        constant_spectrum(initial_beta_value), renderer::TransportSpectrum{},
        renderer::PathDepthCounters{}, 1.0F, (*scene)->spectral_environment()->wavelengths,
        renderer::PathDeltaFlags::none, renderer::VacuumMedium);
    ASSERT_TRUE(high_beta_state.has_value()) << high_beta_state.error().message;
    inputs->front().initial_state = *high_beta_state;

    const auto sampler = renderer::LightSampler::create_uniform(1U);
    const auto scalar_backend = create_analytic_accel_backend(*scene);
    const auto wavefront_backend = create_embree_accel_backend(*scene);
    const auto scheduler = renderer::CpuWavefrontScheduler::create(4U);
    ASSERT_TRUE(sampler.has_value()) << sampler.error().message;
    ASSERT_TRUE(scalar_backend.has_value()) << scalar_backend.error().message;
    ASSERT_TRUE(wavefront_backend.has_value()) << wavefront_backend.error().message;
    ASSERT_TRUE(scheduler.has_value()) << scheduler.error().message;

    constexpr auto heuristic = renderer::MisHeuristic::balance;
    const auto scalar =
        trace_scalar_batch(*inputs, **scalar_backend, *sampler, heuristic, OneDiffuseBounce,
                           renderer::RussianRoulettePolicy::disabled());
    const auto wavefront =
        trace_cpu_wavefront_mis(*inputs, *scheduler, **wavefront_backend, *sampler, heuristic,
                                OneDiffuseBounce, renderer::RussianRoulettePolicy::disabled());
    ASSERT_TRUE(scalar.has_value()) << scalar.error().message;
    ASSERT_TRUE(wavefront.has_value()) << wavefront.error().message;
    ASSERT_EQ(scalar->size(), 1U);
    ASSERT_EQ(wavefront->paths.size(), 1U);
    expect_path_near(scalar->front(), wavefront->paths.front(),
                     std::numeric_limits<renderer::TransportScalar>::denorm_min());

    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        EXPECT_EQ(scalar->front().state.beta()[lane], expected_throughput);
        EXPECT_EQ(wavefront->paths.front().state.beta()[lane], expected_throughput);
        EXPECT_GT(scalar->front().state.accumulated_radiance()[lane], 0.0F);
        EXPECT_GT(wavefront->paths.front().state.accumulated_radiance()[lane], 0.0F);
        EXPECT_FLOAT_EQ(wavefront->paths.front().state.accumulated_radiance()[lane],
                        scalar->front().state.accumulated_radiance()[lane]);
    }
    EXPECT_EQ(wavefront->paths.front().termination, renderer::BsdfOnlyPathTermination::depth_limit);
    EXPECT_EQ(wavefront->report.closure_samples, 1U);
    EXPECT_EQ(wavefront->report.light_samples, 1U);
    EXPECT_EQ(wavefront->report.shadow_queries, 1U);
}

TEST(WavefrontMisTransportNumericTest, SkipsUnrepresentableDirectRadiometryWhenTheLightIsOccluded) {
    constexpr auto initial_beta_value = renderer::TransportScalar{0x1p120F};
    constexpr auto light_intensity = renderer::TransportScalar{0x1p120F};
    const auto hypothetical_visible_scale = static_cast<double>(initial_beta_value) *
                                            static_cast<double>(light_intensity) * 0.5 /
                                            (4.0 * std::numbers::pi_v<double>);
    ASSERT_GT(hypothetical_visible_scale,
              static_cast<double>(std::numeric_limits<renderer::TransportScalar>::max()));

    const auto scene = make_occluded_overflow_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    ASSERT_EQ((*scene)->punctual_lights().size(), 1U);
    ASSERT_EQ((*scene)->mesh_area_lights().size(), 0U);
    auto inputs = make_inputs(*scene);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    inputs->erase(std::next(inputs->begin()), inputs->end());
    const auto high_beta_state = renderer::PathState::create(
        constant_spectrum(initial_beta_value), renderer::TransportSpectrum{},
        renderer::PathDepthCounters{}, 1.0F, (*scene)->spectral_environment()->wavelengths,
        renderer::PathDeltaFlags::none, renderer::VacuumMedium);
    ASSERT_TRUE(high_beta_state.has_value()) << high_beta_state.error().message;
    inputs->front().initial_state = *high_beta_state;

    const auto sampler = renderer::LightSampler::create_uniform(1U);
    const auto scalar_backend = create_analytic_accel_backend(*scene);
    const auto wavefront_backend = create_embree_accel_backend(*scene);
    const auto scheduler = renderer::CpuWavefrontScheduler::create(4U);
    ASSERT_TRUE(sampler.has_value()) << sampler.error().message;
    ASSERT_TRUE(scalar_backend.has_value()) << scalar_backend.error().message;
    ASSERT_TRUE(wavefront_backend.has_value()) << wavefront_backend.error().message;
    ASSERT_TRUE(scheduler.has_value()) << scheduler.error().message;

    constexpr auto heuristic = renderer::MisHeuristic::power;
    const auto scalar =
        trace_scalar_batch(*inputs, **scalar_backend, *sampler, heuristic, OneDiffuseBounce,
                           renderer::RussianRoulettePolicy::disabled());
    const auto wavefront =
        trace_cpu_wavefront_mis(*inputs, *scheduler, **wavefront_backend, *sampler, heuristic,
                                OneDiffuseBounce, renderer::RussianRoulettePolicy::disabled());
    ASSERT_TRUE(scalar.has_value()) << scalar.error().message;
    ASSERT_TRUE(wavefront.has_value()) << wavefront.error().message;
    ASSERT_EQ(scalar->size(), 1U);
    ASSERT_EQ(wavefront->paths.size(), 1U);
    expect_path_near(scalar->front(), wavefront->paths.front(), ScalarParityTolerance);

    EXPECT_EQ(scalar->front().state.accumulated_radiance(), renderer::TransportSpectrum{});
    EXPECT_EQ(wavefront->paths.front().state.accumulated_radiance(), renderer::TransportSpectrum{});
    for (const auto lane : wavefront->paths.front().state.beta().values) {
        EXPECT_EQ(lane, initial_beta_value * 0.5F);
        EXPECT_GT(lane, 0.0F);
    }
    EXPECT_EQ(wavefront->paths.front().termination, renderer::BsdfOnlyPathTermination::depth_limit);
    EXPECT_EQ(wavefront->report.closure_samples, 1U);
    EXPECT_EQ(wavefront->report.light_samples, 1U);
    EXPECT_EQ(wavefront->report.shadow_queries, 1U);
    EXPECT_EQ(wavefront->report.stage_lanes.shadow, 1U);
}

TEST(WavefrontMisTransportBackendTest, RejectsTheAnalyticOracleBeforeAnyFallbackCanRun) {
    const auto scene = make_wavefront_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto inputs = make_inputs(*scene);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    const auto sampler = renderer::LightSampler::create_uniform(1U);
    const auto scheduler = renderer::CpuWavefrontScheduler::create(4U);
    const auto analytic = create_analytic_accel_backend(*scene);
    ASSERT_TRUE(sampler.has_value()) << sampler.error().message;
    ASSERT_TRUE(scheduler.has_value()) << scheduler.error().message;
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
    ASSERT_EQ((*analytic)->kind(), AccelBackendKind::analytic_reference);

    const auto result = trace_cpu_wavefront_mis(*inputs, *scheduler, **analytic, *sampler,
                                                renderer::MisHeuristic::power, OneDiffuseBounce,
                                                renderer::RussianRoulettePolicy::disabled());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::unavailable);
    EXPECT_NE(result.error().message.find("Embree"), std::string::npos);
}

} // namespace
} // namespace blackframe::engine
