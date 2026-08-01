#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/SceneMisPathLoop.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/PathDepthLimits.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <Blackframe/Renderer/RussianRoulette.hpp>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

inline constexpr auto ReceiverMask = renderer::RayMask{1U << 0U};
inline constexpr auto LightMask = renderer::RayMask{1U << 1U};
inline constexpr auto OneDiffuseBounce = renderer::PathDepthLimits{.diffuse = 1U};

[[nodiscard]] renderer::TransportSpectrum constant_spectrum(const float value) {
    auto result = renderer::TransportSpectrum{};
    result.values.fill(value);
    return result;
}

[[nodiscard]] std::shared_ptr<const TriangleMesh>
make_quad(const float half_extent, const float height, const bool faces_up) {
    auto positions = std::vector{
        renderer::Point3{.x = -half_extent, .y = -half_extent, .z = height},
        renderer::Point3{.x = half_extent, .y = -half_extent, .z = height},
        renderer::Point3{.x = half_extent, .y = half_extent, .z = height},
        renderer::Point3{.x = -half_extent, .y = half_extent, .z = height},
    };
    auto triangles = std::vector<TriangleVertexIndices>{};
    if (faces_up) {
        triangles = {
            TriangleVertexIndices{.vertices = {0U, 1U, 2U}},
            TriangleVertexIndices{.vertices = {0U, 2U, 3U}},
        };
    } else {
        triangles = {
            TriangleVertexIndices{.vertices = {0U, 2U, 1U}},
            TriangleVertexIndices{.vertices = {0U, 3U, 2U}},
        };
    }
    const auto normal = faces_up ? renderer::Normal3{.z = 1.0F} : renderer::Normal3{.z = -1.0F};
    return std::make_shared<const TriangleMesh>(
        TriangleMesh::create(std::move(positions), std::vector(4U, normal),
                             {
                                 renderer::Point2{},
                                 renderer::Point2{.x = 1.0F},
                                 renderer::Point2{.x = 1.0F, .y = 1.0F},
                                 renderer::Point2{.y = 1.0F},
                             },
                             std::move(triangles))
            .value());
}

[[nodiscard]] FrameSceneHandle make_mis_scene(const renderer::RayMask emitter_mask) {
    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.25F).value();
    return FrameScene::create(FrameSceneDescription{
                                  .objects =
                                      {
                                          SceneObject{.id = {.value = 1U}},
                                          SceneObject{.id = {.value = 2U}},
                                      },
                                  .geometries =
                                      {
                                          SceneGeometry{
                                              .id = {.value = 11U},
                                              .mesh = make_quad(2.0F, 0.0F, true),
                                          },
                                          SceneGeometry{
                                              .id = {.value = 12U},
                                              .mesh = make_quad(0.75F, 1.0F, false),
                                          },
                                      },
                                  .materials =
                                      {
                                          SceneMaterial{
                                              .id = {.value = 21U},
                                              .spectral =
                                                  SceneSpectralMaterial{
                                                      .wavelengths = wavelengths,
                                                      .reflectance = constant_spectrum(0.8F),
                                                      .emitted_radiance = {},
                                                  },
                                          },
                                          SceneMaterial{
                                              .id = {.value = 22U},
                                              .spectral =
                                                  SceneSpectralMaterial{
                                                      .wavelengths = wavelengths,
                                                      .reflectance = {},
                                                      .emitted_radiance = constant_spectrum(1.5F),
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
                                              .local_to_parent = renderer::identity_matrix<float>(),
                                              .visibility_mask = ReceiverMask,
                                          },
                                          SceneInstance{
                                              .id = {.value = 32U},
                                              .parent = std::nullopt,
                                              .object = {.value = 2U},
                                              .geometry = {.value = 12U},
                                              .material = {.value = 22U},
                                              .local_to_parent = renderer::identity_matrix<float>(),
                                              .visibility_mask = emitter_mask,
                                          },
                                      },
                                  .spectral_environment =
                                      SceneSpectralEnvironment{
                                          .wavelengths = wavelengths,
                                          .radiance = {},
                                      },
                              })
        .value();
}

[[nodiscard]] renderer::Ray primary_ray(const renderer::RayMask mask) {
    return renderer::Ray::create(renderer::Point3{.z = 0.5F}, renderer::Vector3{.z = -1.0F}, 0.0F,
                                 std::numeric_limits<float>::infinity(), 0.5F, mask,
                                 renderer::VacuumMedium)
        .value();
}

[[nodiscard]] core::Result<renderer::BsdfOnlyPathResult>
trace(const AccelBackend& acceleration, const renderer::Ray& ray,
      const renderer::SampleStream& stream, const renderer::MisHeuristic heuristic,
      const renderer::LightSampler& sampler, const renderer::PathState& state) {
    return trace_scene_mis(ray, state, stream, acceleration, sampler, heuristic, OneDiffuseBounce,
                           renderer::RussianRoulettePolicy::disabled());
}

TEST(SceneMisPathLoopTest, ReplaysBothHeuristicsAndExercisesBothSamplingTechniques) {
    const auto scene = make_mis_scene(ReceiverMask);
    ASSERT_EQ(scene->mesh_area_lights().size(), 1U);
    ASSERT_EQ(scene->mesh_area_light_instance_ids().front(), (renderer::InstanceId{.value = 32U}));
    const auto acceleration = create_analytic_accel_backend(scene).value();
    const auto sampler = renderer::LightSampler::create_uniform(1U).value();
    const auto state = renderer::PathState::create_initial(
                           scene->spectral_environment()->wavelengths, renderer::VacuumMedium)
                           .value();
    auto saw_bsdf_light_hit = false;
    auto saw_bsdf_miss = false;

    for (const auto heuristic : {renderer::MisHeuristic::balance, renderer::MisHeuristic::power}) {
        for (auto sample_index = std::uint64_t{}; sample_index < 128U; ++sample_index) {
            const auto stream = renderer::IndependentSampler{0xA4093822299F31D0ULL}.make_stream(
                0U, 0U, sample_index);
            const auto first =
                trace(*acceleration, primary_ray(ReceiverMask), stream, heuristic, sampler, state);
            const auto replay =
                trace(*acceleration, primary_ray(ReceiverMask), stream, heuristic, sampler, state);
            ASSERT_TRUE(first.has_value()) << first.error().message;
            ASSERT_TRUE(replay.has_value()) << replay.error().message;
            EXPECT_EQ(first->state.accumulated_radiance(), replay->state.accumulated_radiance());
            for (const auto lane : first->state.accumulated_radiance().values) {
                EXPECT_GT(lane, 0.0F);
            }
            saw_bsdf_light_hit =
                saw_bsdf_light_hit ||
                first->termination == renderer::BsdfOnlyPathTermination::depth_limit;
            saw_bsdf_miss =
                saw_bsdf_miss ||
                first->termination == renderer::BsdfOnlyPathTermination::escaped_environment;
        }
    }
    EXPECT_TRUE(saw_bsdf_light_hit);
    EXPECT_TRUE(saw_bsdf_miss);
}

TEST(SceneMisPathLoopTest, HonorsEmitterVisibilityWithoutLeakingDirectRadiance) {
    const auto scene = make_mis_scene(LightMask);
    const auto acceleration = create_analytic_accel_backend(scene).value();
    const auto sampler = renderer::LightSampler::create_uniform(1U).value();
    const auto state = renderer::PathState::create_initial(
                           scene->spectral_environment()->wavelengths, renderer::VacuumMedium)
                           .value();
    const auto stream = renderer::IndependentSampler{0x082EFA98EC4E6C89ULL}.make_stream(0U, 0U, 0U);
    const auto result = trace(*acceleration, primary_ray(ReceiverMask), stream,
                              renderer::MisHeuristic::power, sampler, state);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_EQ(result->state.accumulated_radiance(), renderer::TransportSpectrum{});
}

TEST(SceneMisPathLoopTest, RejectsUnknownHeuristicsRegistryMismatchAndMissingPriorPdfs) {
    const auto scene = make_mis_scene(ReceiverMask);
    const auto acceleration = create_analytic_accel_backend(scene).value();
    const auto sampler = renderer::LightSampler::create_uniform(1U).value();
    const auto wrong_sampler = renderer::LightSampler::create_uniform(2U).value();
    const auto wavelengths = scene->spectral_environment()->wavelengths;
    const auto initial_state =
        renderer::PathState::create_initial(wavelengths, renderer::VacuumMedium).value();
    const auto resumed_state =
        renderer::PathState::create(constant_spectrum(1.0F), renderer::TransportSpectrum{},
                                    renderer::PathDepthCounters{.diffuse = 1U}, 1.0F, wavelengths,
                                    renderer::PathDeltaFlags::any_non_delta_bounces,
                                    renderer::VacuumMedium)
            .value();
    const auto stream = renderer::IndependentSampler{0x452821E638D01377ULL}.make_stream(0U, 0U, 0U);

    const auto unknown = trace(*acceleration, primary_ray(ReceiverMask), stream,
                               static_cast<renderer::MisHeuristic>(255U), sampler, initial_state);
    const auto mismatch = trace(*acceleration, primary_ray(ReceiverMask), stream,
                                renderer::MisHeuristic::balance, wrong_sampler, initial_state);
    const auto resumed = trace(*acceleration, primary_ray(ReceiverMask), stream,
                               renderer::MisHeuristic::power, sampler, resumed_state);
    for (const auto* const result : {&unknown, &mismatch, &resumed}) {
        ASSERT_FALSE(result->has_value());
        EXPECT_FALSE(result->error().message.empty());
    }
    EXPECT_EQ(unknown.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(mismatch.error().code, core::StatusCode::incompatible);
    EXPECT_EQ(resumed.error().code, core::StatusCode::incompatible);
}

} // namespace
} // namespace blackframe::engine
