#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/SceneBsdfOnlyPathLoop.hpp>
#include <Blackframe/Engine/SceneNeePathLoop.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/MatrixOperations.hpp>
#include <Blackframe/Renderer/SampleDimensionMap.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

inline constexpr auto SurfaceVisibility = renderer::RayMask{1U << 0U};
inline constexpr auto BlockerVisibility = renderer::RayMask{1U << 1U};
inline constexpr auto TestDepthLimits = renderer::PathDepthLimits{
    .diffuse = 1U,
};

[[nodiscard]] renderer::SampledWavelengths test_wavelengths() {
    return renderer::sample_uniform_visible_wavelengths(0.25F).value();
}

[[nodiscard]] renderer::TransportSpectrum spectrum(const std::array<float, 4> values) {
    return renderer::TransportSpectrum{.values = values};
}

[[nodiscard]] renderer::TransportSpectrum unit_spectrum() {
    auto result = renderer::TransportSpectrum{};
    result.values.fill(1.0F);
    return result;
}

[[nodiscard]] SceneClosureMixture
require_lambertian_scene_closure(const renderer::TransportSpectrum reflectance) {
    return SceneClosureMixture::create_lambertian(reflectance).value();
}

[[nodiscard]] SceneClosureMixture
continuous_mixed_family_scene_closure(const renderer::TransportSpectrum reflectance) {
    auto closures = renderer::ClosureSet{};
    if (closures.append_lambertian_reflection(reflectance) !=
            renderer::ClosureAppendStatus::appended ||
        closures.append_rough_conductor_reflection(
            spectrum({0.8F, 0.8F, 0.8F, 0.8F}), spectrum({0.5F, 0.5F, 0.5F, 0.5F}),
            spectrum({2.5F, 2.5F, 2.5F, 2.5F}), 0.35F) != renderer::ClosureAppendStatus::appended) {
        throw std::runtime_error{"The NEE test closure mixture could not be constructed."};
    }
    constexpr auto probabilities = std::array{0.25F, 0.75F};
    return SceneClosureMixture::create(std::move(closures), probabilities).value();
}

[[nodiscard]] std::shared_ptr<const TriangleMesh> horizontal_triangle(const float height,
                                                                      const float extent) {
    auto mesh = TriangleMesh::create(
        {
            renderer::Point3{.x = -extent, .y = -extent, .z = height},
            renderer::Point3{.x = extent, .y = -extent, .z = height},
            renderer::Point3{.x = 0.0F, .y = extent, .z = height},
        },
        std::vector(3U, renderer::Normal3{.z = 1.0F}),
        {
            renderer::Point2{},
            renderer::Point2{.x = 1.0F},
            renderer::Point2{.x = 0.5F, .y = 1.0F},
        },
        {
            TriangleVertexIndices{.vertices = {0U, 1U, 2U}},
        });
    return std::make_shared<const TriangleMesh>(std::move(mesh).value());
}

[[nodiscard]] std::shared_ptr<const TriangleMesh>
horizontal_quad(const float height, const float half_extent, const bool faces_up) {
    const auto normal = faces_up ? renderer::Normal3{.z = 1.0F} : renderer::Normal3{.z = -1.0F};
    const auto triangles = faces_up
                               ? std::vector{
                                     TriangleVertexIndices{.vertices = {0U, 1U, 2U}},
                                     TriangleVertexIndices{.vertices = {0U, 2U, 3U}},
                                 }
                               : std::vector{
                                     TriangleVertexIndices{.vertices = {0U, 2U, 1U}},
                                     TriangleVertexIndices{.vertices = {0U, 3U, 2U}},
                                 };
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
        triangles);
    return std::make_shared<const TriangleMesh>(std::move(mesh).value());
}

struct NeeSceneOptions final {
    std::vector<renderer::TransportSpectrum> point_intensities;
    std::vector<ScenePunctualLight> additional_lights;
    bool add_blocker{};
};

[[nodiscard]] FrameSceneHandle
make_nee_scene(const NeeSceneOptions& options,
               const std::optional<SceneClosureMixture>& receiver_closure = std::nullopt) {
    const auto wavelengths = test_wavelengths();
    const auto reflectance = spectrum({0.25F, 0.5F, 0.75F, 1.0F});
    auto description = FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 1U}}},
        .geometries =
            {
                SceneGeometry{
                    .id = {.value = 11U},
                    .mesh = horizontal_triangle(0.0F, 2.0F),
                },
            },
        .materials =
            {
                SceneMaterial{
                    .id = {.value = 21U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = wavelengths,
                            .closure_mixture = receiver_closure.value_or(
                                require_lambertian_scene_closure(reflectance)),
                            .emitted_radiance = renderer::TransportSpectrum{},
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
                    .visibility_mask = SurfaceVisibility,
                },
            },
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = wavelengths,
                .radiance = renderer::TransportSpectrum{},
            },
    };

    description.punctual_lights.reserve(options.point_intensities.size());
    for (const auto& intensity : options.point_intensities) {
        description.punctual_lights.emplace_back(ScenePointLight{
            .position = renderer::Point3{.z = 2.0F},
            .absolute_position_error = renderer::Vector3{},
            .spectral_radiant_intensity = intensity,
        });
    }
    description.punctual_lights.insert(description.punctual_lights.end(),
                                       options.additional_lights.begin(),
                                       options.additional_lights.end());

    if (options.add_blocker) {
        description.objects.push_back(SceneObject{.id = {.value = 2U}});
        description.geometries.push_back(SceneGeometry{
            .id = {.value = 12U},
            .mesh = horizontal_triangle(1.0F, 0.5F),
        });
        description.instances.push_back(SceneInstance{
            .id = {.value = 32U},
            .parent = std::nullopt,
            .object = {.value = 2U},
            .geometry = {.value = 12U},
            .material = {.value = 21U},
            .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
            .visibility_mask = BlockerVisibility,
        });
    }

    return FrameScene::create(std::move(description)).value();
}

[[nodiscard]] FrameSceneHandle
make_two_bounce_scene(const renderer::TransportSpectrum& first_reflectance,
                      const renderer::TransportSpectrum& second_reflectance,
                      const renderer::TransportSpectrum& directional_irradiance) {
    const auto wavelengths = test_wavelengths();
    return FrameScene::create(
               FrameSceneDescription{
                   .objects =
                       {
                           SceneObject{.id = {.value = 1U}},
                           SceneObject{.id = {.value = 2U}},
                       },
                   .geometries =
                       {
                           SceneGeometry{
                               .id = {.value = 11U},
                               .mesh = horizontal_quad(0.0F, 0.01F, true),
                           },
                           SceneGeometry{
                               .id = {.value = 12U},
                               .mesh = horizontal_quad(1.0F, 100.0F, false),
                           },
                       },
                   .materials =
                       {
                           SceneMaterial{
                               .id = {.value = 21U},
                               .spectral =
                                   SceneSpectralMaterial{
                                       .wavelengths = wavelengths,
                                       .closure_mixture =
                                           require_lambertian_scene_closure(first_reflectance),
                                       .emitted_radiance = {},
                                   },
                           },
                           SceneMaterial{
                               .id = {.value = 22U},
                               .spectral =
                                   SceneSpectralMaterial{
                                       .wavelengths = wavelengths,
                                       .closure_mixture =
                                           require_lambertian_scene_closure(second_reflectance),
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
                               .local_to_parent =
                                   renderer::identity_matrix<renderer::TransportScalar>(),
                               .visibility_mask = SurfaceVisibility,
                           },
                           SceneInstance{
                               .id = {.value = 32U},
                               .parent = std::nullopt,
                               .object = {.value = 2U},
                               .geometry = {.value = 12U},
                               .material = {.value = 22U},
                               .local_to_parent =
                                   renderer::identity_matrix<renderer::TransportScalar>(),
                               .visibility_mask = SurfaceVisibility,
                           },
                       },
                   .punctual_lights =
                       {
                           SceneDirectionalLight{
                               .propagation_direction = {.z = 1.0F},
                               .spectral_irradiance = directional_irradiance,
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

[[nodiscard]] renderer::Ray
make_primary_ray(const renderer::RayMask mask = SurfaceVisibility,
                 const renderer::MediumId medium = renderer::VacuumMedium) {
    return renderer::Ray::create(renderer::Point3{.z = 0.5F}, renderer::Vector3{.z = -1.0F}, 0.0F,
                                 std::numeric_limits<float>::infinity(), 0.5F, mask, medium)
        .value();
}

[[nodiscard]] renderer::PathState
make_path_state(const renderer::TransportSpectrum& beta = unit_spectrum(),
                const renderer::MediumId medium = renderer::VacuumMedium) {
    return renderer::PathState::create(beta, renderer::TransportSpectrum{},
                                       renderer::PathDepthCounters{}, 1.0F, test_wavelengths(),
                                       renderer::PathDeltaFlags::none, medium)
        .value();
}

[[nodiscard]] renderer::SampleStream
sample_stream(const std::uint64_t sample_index = 0U,
              const std::uint64_t seed = 0x123456789ABCDEF0ULL) {
    return renderer::SampleStream{renderer::SampleStreamIndex{
        .pixel_x = 7U,
        .pixel_y = 11U,
        .sample_index = sample_index,
        .seed = seed,
    }};
}

[[nodiscard]] std::unique_ptr<AccelBackend> analytic_backend(const FrameSceneHandle& scene) {
    return std::move(create_analytic_accel_backend(scene).value());
}

[[nodiscard]] core::Result<renderer::BsdfOnlyPathResult>
trace_nee(const renderer::Ray& ray, const renderer::PathState& state,
          const renderer::SampleStream& stream, const AccelBackend& acceleration,
          const renderer::LightSampler& sampler) {
    return trace_scene_nee(ray, state, stream, acceleration, sampler, TestDepthLimits,
                           renderer::RussianRoulettePolicy::disabled());
}

[[nodiscard]] renderer::TransportSpectrum
expected_point_contribution(const renderer::TransportSpectrum& beta,
                            const renderer::TransportSpectrum& intensity,
                            const float selection_probability) {
    constexpr auto inverse_squared_distance = 0.25;
    constexpr auto reflectance = std::array{0.25, 0.5, 0.75, 1.0};
    auto expected = renderer::TransportSpectrum{};
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        const auto value = static_cast<double>(beta[lane]) * reflectance[lane] *
                           static_cast<double>(intensity[lane]) * inverse_squared_distance *
                           std::numbers::inv_pi_v<double> /
                           static_cast<double>(selection_probability);
        expected[lane] = static_cast<float>(value);
    }
    return expected;
}

[[nodiscard]] renderer::TransportSpectrum
expected_directional_contribution(const renderer::TransportSpectrum& beta,
                                  const renderer::TransportSpectrum& irradiance,
                                  const float selection_probability) {
    constexpr auto reflectance = std::array{0.25, 0.5, 0.75, 1.0};
    auto expected = renderer::TransportSpectrum{};
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        const auto value = static_cast<double>(beta[lane]) * reflectance[lane] *
                           static_cast<double>(irradiance[lane]) * std::numbers::inv_pi_v<double> /
                           static_cast<double>(selection_probability);
        expected[lane] = static_cast<float>(value);
    }
    return expected;
}

struct TwoBounceExpectation final {
    renderer::TransportSpectrum final_beta;
    renderer::TransportSpectrum direct_radiance;
};

[[nodiscard]] TwoBounceExpectation
expected_two_bounce_directional(const renderer::TransportSpectrum& initial_beta,
                                const renderer::TransportSpectrum& first_reflectance,
                                const renderer::TransportSpectrum& second_reflectance,
                                const renderer::TransportSpectrum& irradiance) {
    auto expectation = TwoBounceExpectation{};
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        const auto beta_after_first = initial_beta[lane] * first_reflectance[lane];
        expectation.final_beta[lane] = beta_after_first * second_reflectance[lane];
        expectation.direct_radiance[lane] = static_cast<float>(
            static_cast<double>(beta_after_first) * static_cast<double>(second_reflectance[lane]) *
            static_cast<double>(irradiance[lane]) * std::numbers::inv_pi_v<double>);
    }
    return expectation;
}

void expect_spectrum_near(const renderer::TransportSpectrum& actual,
                          const renderer::TransportSpectrum& expected) {
    for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
        const auto tolerance = std::max(2.0e-6F * std::abs(expected[lane]), 1.0e-7F);
        EXPECT_NEAR(actual[lane], expected[lane], tolerance) << "spectral lane " << lane;
    }
}

void expect_same_result(const renderer::BsdfOnlyPathResult& first,
                        const renderer::BsdfOnlyPathResult& second) {
    EXPECT_EQ(first.state.beta(), second.state.beta());
    EXPECT_EQ(first.state.accumulated_radiance(), second.state.accumulated_radiance());
    EXPECT_EQ(first.state.depth(), second.state.depth());
    EXPECT_EQ(first.state.depth_counters(), second.state.depth_counters());
    EXPECT_EQ(first.state.eta_scale(), second.state.eta_scale());
    EXPECT_EQ(first.state.wavelengths(), second.state.wavelengths());
    EXPECT_EQ(first.state.delta_flags(), second.state.delta_flags());
    EXPECT_EQ(first.state.current_medium(), second.state.current_medium());
    EXPECT_EQ(first.terminal_ray.origin(), second.terminal_ray.origin());
    EXPECT_EQ(first.terminal_ray.direction(), second.terminal_ray.direction());
    EXPECT_EQ(first.terminal_ray.t_min(), second.terminal_ray.t_min());
    EXPECT_EQ(first.terminal_ray.t_max(), second.terminal_ray.t_max());
    EXPECT_EQ(first.terminal_ray.time(), second.terminal_ray.time());
    EXPECT_EQ(first.terminal_ray.mask(), second.terminal_ray.mask());
    EXPECT_EQ(first.terminal_ray.current_medium(), second.terminal_ray.current_medium());
    EXPECT_EQ(first.termination, second.termination);
    EXPECT_EQ(first.blocked_depth_limits, second.blocked_depth_limits);
}

TEST(SceneNeePathLoopTest, MatchesTheAxialPointLightEstimator) {
    const auto intensity = spectrum({4.0F, 8.0F, 12.0F, 16.0F});
    const auto beta = spectrum({0.5F, 1.0F, 1.5F, 2.0F});
    const auto scene = make_nee_scene({.point_intensities = {intensity}, .additional_lights = {}});
    const auto acceleration = analytic_backend(scene);
    const auto sampler = renderer::LightSampler::create_uniform(1U).value();

    const auto result = trace_nee(make_primary_ray(), make_path_state(beta), sample_stream(),
                                  *acceleration, sampler);

    ASSERT_TRUE(result.has_value()) << result.error().message;
    expect_spectrum_near(result->state.accumulated_radiance(),
                         expected_point_contribution(beta, intensity, 1.0F));
    EXPECT_EQ(result->termination, renderer::BsdfOnlyPathTermination::escaped_environment);
    EXPECT_EQ(result->state.depth(), 1U);
}

TEST(SceneNeePathLoopTest,
     FiltersBlockedContinuousLobesBeforeDirectLightingAndContinuationSampling) {
    const auto intensity = spectrum({4.0F, 8.0F, 12.0F, 16.0F});
    const auto beta = spectrum({0.5F, 1.0F, 1.5F, 2.0F});
    const auto reflectance = spectrum({0.25F, 0.5F, 0.75F, 1.0F});
    const auto scene = make_nee_scene(
        {
            .point_intensities = {intensity},
            .additional_lights = {},
        },
        continuous_mixed_family_scene_closure(reflectance));
    const auto acceleration = analytic_backend(scene);
    const auto sampler = renderer::LightSampler::create_uniform(1U).value();

    const auto result = trace_nee(make_primary_ray(), make_path_state(beta), sample_stream(),
                                  *acceleration, sampler);

    ASSERT_TRUE(result.has_value()) << result.error().message;
    expect_spectrum_near(result->state.accumulated_radiance(),
                         expected_point_contribution(beta, intensity, 1.0F));
    EXPECT_EQ(result->state.beta(), beta * reflectance);
    EXPECT_EQ(result->state.depth_counters(), (renderer::PathDepthCounters{.diffuse = 1U}));
    EXPECT_EQ(result->termination, renderer::BsdfOnlyPathTermination::escaped_environment);
}

TEST(SceneNeePathLoopTest, SamplesDirectionalAndSpotRecordsWithTheirExactRadiometry) {
    const auto incident = spectrum({4.0F, 8.0F, 12.0F, 16.0F});
    const auto beta = spectrum({0.5F, 1.0F, 1.5F, 2.0F});
    const auto sampler = renderer::LightSampler::create_uniform(1U).value();

    const auto directional_scene = make_nee_scene({
        .point_intensities = {},
        .additional_lights =
            {
                SceneDirectionalLight{
                    .propagation_direction = {.z = -1.0F},
                    .spectral_irradiance = incident,
                },
            },
    });
    const auto directional_acceleration = analytic_backend(directional_scene);
    const auto directional = trace_nee(make_primary_ray(), make_path_state(beta), sample_stream(),
                                       *directional_acceleration, sampler);
    ASSERT_TRUE(directional.has_value()) << directional.error().message;
    expect_spectrum_near(directional->state.accumulated_radiance(),
                         expected_directional_contribution(beta, incident, 1.0F));

    const auto spot_scene = make_nee_scene({
        .point_intensities = {},
        .additional_lights =
            {
                SceneSpotLight{
                    .position = {.z = 2.0F},
                    .absolute_position_error = {},
                    .emission_direction = {.z = -1.0F},
                    .inner_half_angle_radians = 0.25F,
                    .outer_half_angle_radians = 0.5F,
                    .on_axis_spectral_radiant_intensity = incident,
                },
            },
    });
    const auto spot_acceleration = analytic_backend(spot_scene);
    const auto spot = trace_nee(make_primary_ray(), make_path_state(beta), sample_stream(),
                                *spot_acceleration, sampler);
    ASSERT_TRUE(spot.has_value()) << spot.error().message;
    expect_spectrum_near(spot->state.accumulated_radiance(),
                         expected_point_contribution(beta, incident, 1.0F));
}

TEST(SceneNeePathLoopTest, EstimatesTheSecondVertexWithPropagatedThroughput) {
    const auto initial_beta = spectrum({0.5F, 1.0F, 1.5F, 2.0F});
    const auto first_reflectance = spectrum({0.2F, 0.4F, 0.6F, 0.8F});
    const auto second_reflectance = spectrum({0.3F, 0.5F, 0.7F, 0.9F});
    const auto irradiance = spectrum({2.0F, 3.0F, 4.0F, 5.0F});
    const auto scene = make_two_bounce_scene(first_reflectance, second_reflectance, irradiance);
    const auto acceleration = analytic_backend(scene);
    const auto sampler = renderer::LightSampler::create_uniform(1U).value();
    constexpr auto two_diffuse_bounces = renderer::PathDepthLimits{.diffuse = 2U};

    const auto result = trace_scene_nee(
        make_primary_ray(), make_path_state(initial_beta), sample_stream(), *acceleration, sampler,
        two_diffuse_bounces, renderer::RussianRoulettePolicy::disabled());

    ASSERT_TRUE(result.has_value()) << result.error().message;
    const auto expected = expected_two_bounce_directional(initial_beta, first_reflectance,
                                                          second_reflectance, irradiance);
    expect_spectrum_near(result->state.beta(), expected.final_beta);
    expect_spectrum_near(result->state.accumulated_radiance(), expected.direct_radiance);
    EXPECT_EQ(result->state.depth(), 2U);
    EXPECT_EQ(result->state.depth_counters(), (renderer::PathDepthCounters{.diffuse = 2U}));
}

TEST(SceneNeePathLoopTest, AppliesUniformSelectionAndReplaysEachSelectedSlotExactly) {
    const auto first_intensity = spectrum({1.0F, 2.0F, 3.0F, 4.0F});
    const auto second_intensity = spectrum({5.0F, 6.0F, 7.0F, 8.0F});
    const auto beta = spectrum({0.5F, 1.0F, 1.5F, 2.0F});
    const auto scene = make_nee_scene(
        {.point_intensities = {first_intensity, second_intensity}, .additional_lights = {}});
    const auto acceleration = analytic_backend(scene);
    const auto sampler = renderer::LightSampler::create_uniform(2U).value();
    const auto dimensions = renderer::sample_dimensions_for_bounce(0U).value();
    auto observed_slots = std::array<bool, 2>{};

    for (auto sample_index = std::uint64_t{}; sample_index < 64U; ++sample_index) {
        const auto stream = sample_stream(sample_index);
        const auto selection = sampler.sample(stream.sample_1d(dimensions.light_selection)).value();
        observed_slots[selection.light_index()] = true;
        const auto& selected_intensity =
            selection.light_index() == 0U ? first_intensity : second_intensity;

        const auto first =
            trace_nee(make_primary_ray(), make_path_state(beta), stream, *acceleration, sampler);
        const auto replay =
            trace_nee(make_primary_ray(), make_path_state(beta), stream, *acceleration, sampler);
        ASSERT_TRUE(first.has_value()) << first.error().message;
        ASSERT_TRUE(replay.has_value()) << replay.error().message;
        expect_spectrum_near(first->state.accumulated_radiance(),
                             expected_point_contribution(beta, selected_intensity, 0.5F));
        expect_same_result(*first, *replay);
        if (observed_slots[0] && observed_slots[1]) {
            break;
        }
    }
    EXPECT_TRUE(observed_slots[0]);
    EXPECT_TRUE(observed_slots[1]);
}

TEST(SceneNeePathLoopTest, AppliesShadowVisibilityAndTheExactRayMask) {
    const auto intensity = spectrum({4.0F, 8.0F, 12.0F, 16.0F});
    const auto beta = unit_spectrum();
    const auto scene = make_nee_scene(
        {.point_intensities = {intensity}, .additional_lights = {}, .add_blocker = true});
    const auto acceleration = analytic_backend(scene);
    const auto sampler = renderer::LightSampler::create_uniform(1U).value();

    const auto visible = trace_nee(make_primary_ray(SurfaceVisibility), make_path_state(beta),
                                   sample_stream(), *acceleration, sampler);
    const auto blocked = trace_nee(make_primary_ray(SurfaceVisibility | BlockerVisibility),
                                   make_path_state(beta), sample_stream(), *acceleration, sampler);

    ASSERT_TRUE(visible.has_value()) << visible.error().message;
    ASSERT_TRUE(blocked.has_value()) << blocked.error().message;
    expect_spectrum_near(visible->state.accumulated_radiance(),
                         expected_point_contribution(beta, intensity, 1.0F));
    EXPECT_EQ(blocked->state.accumulated_radiance(), renderer::TransportSpectrum{});
}

TEST(SceneNeePathLoopTest, RejectsRegistryAndMediumMismatchesWithoutFallback) {
    const auto intensity = spectrum({1.0F, 2.0F, 3.0F, 4.0F});
    const auto one_light_scene =
        make_nee_scene({.point_intensities = {intensity}, .additional_lights = {}});
    const auto one_light_acceleration = analytic_backend(one_light_scene);
    const auto wrong_sampler = renderer::LightSampler::create_uniform(2U).value();
    const auto mismatched = trace_nee(make_primary_ray(), make_path_state(), sample_stream(),
                                      *one_light_acceleration, wrong_sampler);
    ASSERT_FALSE(mismatched.has_value());
    EXPECT_EQ(mismatched.error().code, core::StatusCode::incompatible);
    EXPECT_FALSE(mismatched.error().message.empty());

    const auto two_light_scene =
        make_nee_scene({.point_intensities = {intensity, intensity}, .additional_lights = {}});
    const auto two_light_acceleration = analytic_backend(two_light_scene);
    const auto support_weights = std::array{
        unit_spectrum(),
        renderer::TransportSpectrum{},
    };
    const auto incomplete_sampler = renderer::LightSampler::create_power_weighted(support_weights);
    ASSERT_TRUE(incomplete_sampler.has_value()) << incomplete_sampler.error().message;
    const auto unsupported = trace_nee(make_primary_ray(), make_path_state(), sample_stream(),
                                       *two_light_acceleration, *incomplete_sampler);
    ASSERT_FALSE(unsupported.has_value());
    EXPECT_EQ(unsupported.error().code, core::StatusCode::incompatible);
    EXPECT_FALSE(unsupported.error().message.empty());

    const auto no_light_scene = make_nee_scene({});
    const auto no_light_acceleration = analytic_backend(no_light_scene);
    const auto one_light_sampler = renderer::LightSampler::create_uniform(1U).value();
    const auto unavailable = trace_nee(make_primary_ray(), make_path_state(), sample_stream(),
                                       *no_light_acceleration, one_light_sampler);
    ASSERT_FALSE(unavailable.has_value());
    EXPECT_EQ(unavailable.error().code, core::StatusCode::unavailable);
    EXPECT_FALSE(unavailable.error().message.empty());

    constexpr auto fog = renderer::MediumId{.value = 9U};
    const auto medium =
        trace_nee(make_primary_ray(SurfaceVisibility, fog), make_path_state({}, fog),
                  sample_stream(), *one_light_acceleration, one_light_sampler);
    ASSERT_FALSE(medium.has_value());
    EXPECT_EQ(medium.error().code, core::StatusCode::unavailable);
    EXPECT_FALSE(medium.error().message.empty());
}

TEST(SceneNeePathLoopTest, LeavesTheHistoricalBsdfOnlyOracleIndependentOfPunctualLights) {
    const auto intensity = spectrum({4.0F, 8.0F, 12.0F, 16.0F});
    const auto scene_without_light = make_nee_scene({});
    const auto scene_with_light =
        make_nee_scene({.point_intensities = {intensity}, .additional_lights = {}});
    const auto acceleration_without_light = analytic_backend(scene_without_light);
    const auto acceleration_with_light = analytic_backend(scene_with_light);
    const auto ray = make_primary_ray();
    const auto state = make_path_state();
    const auto stream = sample_stream();

    const auto without_light =
        trace_scene_bsdf_only(ray, state, stream, *acceleration_without_light, TestDepthLimits,
                              renderer::RussianRoulettePolicy::disabled());
    const auto with_light =
        trace_scene_bsdf_only(ray, state, stream, *acceleration_with_light, TestDepthLimits,
                              renderer::RussianRoulettePolicy::disabled());

    ASSERT_TRUE(without_light.has_value()) << without_light.error().message;
    ASSERT_TRUE(with_light.has_value()) << with_light.error().message;
    expect_same_result(*without_light, *with_light);
    EXPECT_EQ(with_light->state.accumulated_radiance(), renderer::TransportSpectrum{});
}

} // namespace
} // namespace blackframe::engine
