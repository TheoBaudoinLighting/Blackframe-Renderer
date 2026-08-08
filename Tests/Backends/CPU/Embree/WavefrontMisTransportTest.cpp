#include "ScalarWavefrontImageParity.hpp"

#include <Blackframe/Backends/CPU/Embree/AccelBackend.hpp>
#include <Blackframe/Backends/CPU/Embree/WavefrontMisTransport.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/SceneMisPathLoop.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Renderer/CpuWavefrontScheduler.hpp>
#include <Blackframe/Renderer/Fresnel.hpp>
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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

inline constexpr auto PathCount = std::size_t{24U};
inline constexpr auto PathTime = renderer::TransportScalar{0.375F};
inline constexpr auto ScalarParityTolerance = renderer::TransportScalar{1.0e-5F};
inline constexpr auto OneDiffuseBounce = renderer::PathDepthLimits{.diffuse = 1U};
inline constexpr auto OneSurfaceBounce = renderer::PathDepthLimits{
    .diffuse = 1U,
    .glossy = 1U,
    .specular = 1U,
    .transmission = 1U,
};
inline constexpr auto EvaluationSeed = std::uint64_t{0x243F6A8885A308D3ULL};

[[nodiscard]] renderer::TransportSpectrum constant_spectrum(const float value) {
    auto spectrum = renderer::TransportSpectrum{};
    spectrum.values.fill(value);
    return spectrum;
}

[[nodiscard]] SceneClosureMixture
require_lambertian_scene_closure(const renderer::TransportSpectrum reflectance) {
    return SceneClosureMixture::create_lambertian(reflectance).value();
}

void require_closure_append(const renderer::ClosureAppendStatus status) {
    if (status != renderer::ClosureAppendStatus::appended) {
        throw std::runtime_error{"A material transport test closure could not be appended."};
    }
}

[[nodiscard]] SceneClosureMixture require_scene_closure_mixture(
    renderer::ClosureSet closures,
    const std::span<const renderer::TransportScalar> component_probabilities,
    const renderer::TransportScalar tangent_rotation_radians = 0.0F) {
    auto result = SceneClosureMixture::create(std::move(closures), component_probabilities,
                                              SceneClosureFrameMode::surface_tangent,
                                              tangent_rotation_radians);
    if (!result) {
        throw std::runtime_error{result.error().message};
    }
    return std::move(*result);
}

[[nodiscard]] SceneClosureMixture rough_diffuse_scene_closure() {
    auto closures = renderer::ClosureSet{};
    require_closure_append(
        closures.append_rough_diffuse_reflection(constant_spectrum(0.72F), 0.65F));
    constexpr auto probabilities = std::array{renderer::TransportScalar{1.0F}};
    return require_scene_closure_mixture(std::move(closures), probabilities);
}

[[nodiscard]] SceneClosureMixture
rough_conductor_scene_closure(const renderer::TransportScalar alpha_x,
                              const renderer::TransportScalar alpha_y,
                              const renderer::TransportScalar tangent_rotation_radians = 0.0F) {
    auto closures = renderer::ClosureSet{};
    require_closure_append(closures.append_rough_conductor_reflection(
        constant_spectrum(0.85F),
        renderer::TransportSpectrum{.values = {0.25F, 0.45F, 0.75F, 1.10F}},
        renderer::TransportSpectrum{.values = {3.2F, 2.7F, 2.2F, 1.8F}}, alpha_x, alpha_y));
    constexpr auto probabilities = std::array{renderer::TransportScalar{1.0F}};
    return require_scene_closure_mixture(std::move(closures), probabilities,
                                         tangent_rotation_radians);
}

[[nodiscard]] SceneClosureMixture
rough_dielectric_scene_closure(const renderer::TransportScalar alpha_x = 0.25F,
                               const renderer::TransportScalar alpha_y = 0.25F) {
    auto closures = renderer::ClosureSet{};
    require_closure_append(
        closures.append_rough_dielectric(constant_spectrum(0.9F), 1.0F, 1.5F, alpha_x, alpha_y));
    constexpr auto probabilities = std::array{renderer::TransportScalar{1.0F}};
    return require_scene_closure_mixture(std::move(closures), probabilities);
}

[[nodiscard]] SceneClosureMixture mirror_scene_closure() {
    auto closures = renderer::ClosureSet{};
    require_closure_append(closures.append_specular_reflection(constant_spectrum(0.85F)));
    constexpr auto probabilities = std::array{renderer::TransportScalar{1.0F}};
    return require_scene_closure_mixture(std::move(closures), probabilities);
}

[[nodiscard]] SceneClosureMixture transmission_scene_closure() {
    auto closures = renderer::ClosureSet{};
    require_closure_append(
        closures.append_specular_transmission(constant_spectrum(0.9F), 1.0F, 1.5F));
    constexpr auto probabilities = std::array{renderer::TransportScalar{1.0F}};
    return require_scene_closure_mixture(std::move(closures), probabilities);
}

[[nodiscard]] SceneClosureMixture continuous_mixture_scene_closure() {
    auto closures = renderer::ClosureSet{};
    require_closure_append(closures.append_lambertian_reflection(constant_spectrum(0.25F)));
    require_closure_append(
        closures.append_rough_diffuse_reflection(constant_spectrum(0.75F), 0.7F));
    constexpr auto probabilities =
        std::array{renderer::TransportScalar{0.35F}, renderer::TransportScalar{0.65F}};
    return require_scene_closure_mixture(std::move(closures), probabilities);
}

[[nodiscard]] SceneClosureMixture depth_filtered_continuous_mixture_scene_closure() {
    auto closures = renderer::ClosureSet{};
    require_closure_append(closures.append_lambertian_reflection(constant_spectrum(0.25F)));
    require_closure_append(closures.append_rough_conductor_reflection(
        constant_spectrum(0.85F),
        renderer::TransportSpectrum{.values = {0.25F, 0.45F, 0.75F, 1.10F}},
        renderer::TransportSpectrum{.values = {3.2F, 2.7F, 2.2F, 1.8F}}, 0.35F));
    constexpr auto probabilities =
        std::array{renderer::TransportScalar{0.35F}, renderer::TransportScalar{0.65F}};
    return require_scene_closure_mixture(std::move(closures), probabilities);
}

[[nodiscard]] SceneClosureMixture mixed_measure_scene_closure() {
    auto closures = renderer::ClosureSet{};
    require_closure_append(closures.append_lambertian_reflection(constant_spectrum(0.3F)));
    require_closure_append(closures.append_specular_reflection(constant_spectrum(0.8F)));
    constexpr auto probabilities =
        std::array{renderer::TransportScalar{0.4F}, renderer::TransportScalar{0.6F}};
    return require_scene_closure_mixture(std::move(closures), probabilities);
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
                            .closure_mixture = require_lambertian_scene_closure(
                                constant_spectrum(receiver_reflectance)),
                            .emitted_radiance = {},
                        },
                },
                SceneMaterial{
                    .id = {.value = 22U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .closure_mixture = require_lambertian_scene_closure({}),
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

[[nodiscard]] core::Error material_test_error(std::string message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = std::move(message),
    };
}

[[nodiscard]] core::Result<FrameSceneHandle>
make_material_scene(SceneClosureMixture closure_mixture) {
    auto receiver = horizontal_quad(16.0F, 0.0F, true);
    if (!receiver) {
        return std::unexpected(receiver.error());
    }
    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.25F);
    if (!wavelengths) {
        return std::unexpected(wavelengths.error());
    }
    return FrameScene::create(FrameSceneDescription{
        .objects = {SceneObject{.id = {.value = 1U}}},
        .geometries = {SceneGeometry{.id = {.value = 11U}, .mesh = std::move(*receiver)}},
        .materials =
            {
                SceneMaterial{
                    .id = {.value = 21U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .closure_mixture = std::move(closure_mixture),
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
        .punctual_lights =
            {
                ScenePointLight{
                    .position = renderer::Point3{.z = 3.0F},
                    .absolute_position_error = {},
                    .spectral_radiant_intensity = constant_spectrum(0.5F),
                },
            },
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = *wavelengths,
                .radiance = constant_spectrum(0.35F),
            },
    });
}

[[nodiscard]] core::Result<renderer::Ray>
material_primary_ray(const renderer::Vector3 outgoing_world) {
    return renderer::Ray::create(
        renderer::Point3{.x = outgoing_world.x, .y = outgoing_world.y, .z = outgoing_world.z},
        -outgoing_world, 0.0F, std::numeric_limits<renderer::TransportScalar>::infinity(), PathTime,
        renderer::AllRayVisibility, renderer::VacuumMedium);
}

struct ComponentSampleBand final {
    renderer::TransportScalar lower{};
    renderer::TransportScalar upper{};
    std::size_t count{};
};

[[nodiscard]] core::Result<std::vector<CpuWavefrontMisPathInput>>
make_material_inputs(const FrameSceneHandle& scene, const renderer::Ray& ray,
                     const std::span<const ComponentSampleBand> bands) {
    const auto state = renderer::PathState::create_initial(
        scene->spectral_environment()->wavelengths, renderer::VacuumMedium);
    if (!state) {
        return std::unexpected(state.error());
    }
    const auto dimensions = renderer::sample_dimensions_for_bounce(0U);
    if (!dimensions) {
        return std::unexpected(dimensions.error());
    }
    const auto primary_cone = renderer::RayCone::create(0.0F, 0.0F);
    if (!primary_cone) {
        return std::unexpected(primary_cone.error());
    }

    auto requested_count = std::size_t{};
    for (const auto& band : bands) {
        if (!std::isfinite(band.lower) || !std::isfinite(band.upper) || band.lower < 0.0F ||
            !(band.lower < band.upper) || band.upper > 1.0F || band.count == 0U ||
            requested_count > std::numeric_limits<std::uint32_t>::max() - band.count) {
            return std::unexpected(material_test_error(
                "Material parity sample bands must be finite, canonical, and representable."));
        }
        requested_count += band.count;
    }
    auto inputs = std::vector<CpuWavefrontMisPathInput>{};
    inputs.reserve(requested_count);
    for (const auto& band : bands) {
        for (auto band_index = std::size_t{}; band_index < band.count; ++band_index) {
            const auto pixel_x = static_cast<std::uint32_t>(inputs.size());
            auto found = false;
            for (auto candidate = std::uint64_t{}; candidate < 100'000U; ++candidate) {
                const auto sample = renderer::SampleStreamIndex{
                    .pixel_x = pixel_x,
                    .pixel_y = 0U,
                    .sample_index = candidate,
                    .seed = EvaluationSeed,
                };
                const auto component =
                    renderer::SampleStream{sample}.sample_1d(dimensions->bsdf_component);
                if (component < band.lower || component >= band.upper) {
                    continue;
                }
                inputs.push_back(CpuWavefrontMisPathInput{
                    .primary_ray = ray,
                    .primary_cone = *primary_cone,
                    .initial_state = *state,
                    .sample = sample,
                });
                found = true;
                break;
            }
            if (!found) {
                return std::unexpected(material_test_error(
                    "A deterministic material parity sample band could not be populated."));
            }
        }
    }
    return inputs;
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
                            .closure_mixture =
                                require_lambertian_scene_closure(constant_spectrum(0.5F)),
                            .emitted_radiance = {},
                        },
                },
                SceneMaterial{
                    .id = {.value = 22U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .closure_mixture = require_lambertian_scene_closure({}),
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
    const auto primary_cone = renderer::RayCone::create(0.0F, 0.0F);
    if (!primary_cone) {
        return std::unexpected(primary_cone.error());
    }

    auto inputs = std::vector<CpuWavefrontMisPathInput>{};
    inputs.reserve(PathCount);
    for (auto index = std::size_t{}; index < PathCount; ++index) {
        inputs.push_back(CpuWavefrontMisPathInput{
            .primary_ray = *ray,
            .primary_cone = *primary_cone,
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

void expect_image_metrics(const std::string_view scene_name, const renderer::RenderExtent extent,
                          const std::uint32_t samples_per_pixel,
                          const renderer::MisHeuristic heuristic,
                          const renderer::PathDepthLimits& depth_limits,
                          const renderer::RussianRoulettePolicy& roulette_policy,
                          const std::span<const CpuWavefrontMisPathInput> inputs,
                          const std::span<const renderer::BsdfOnlyPathResult> scalar_paths,
                          const CpuWavefrontMisBatch& wavefront,
                          const bool require_shadow_queries = true) {
    ASSERT_EQ(inputs.size(), scalar_paths.size());
    ASSERT_EQ(inputs.size(), wavefront.paths.size());
    ASSERT_EQ(inputs.size(), wavefront.terminal_cones.size());
    constexpr auto expected_worker_count = std::uint32_t{4U};
    const auto queue_status = scalar_wavefront_parity_test::validate_queue_report(
        wavefront.report, inputs.size(), expected_worker_count);
    ASSERT_TRUE(queue_status.has_value()) << queue_status.error().message;
    ASSERT_GT(wavefront.report.closure_samples, 0U);
    ASSERT_GT(wavefront.report.light_samples, 0U);
    if (require_shadow_queries) {
        ASSERT_GT(wavefront.report.shadow_queries, 0U);
    }
    auto scalar_film = renderer::ReferenceFilm::create(extent);
    auto wavefront_film = renderer::Film::create(extent);
    ASSERT_TRUE(scalar_film.has_value()) << scalar_film.error().message;
    ASSERT_TRUE(wavefront_film.has_value()) << wavefront_film.error().message;
    for (auto index = std::size_t{}; index < inputs.size(); ++index) {
        const auto scalar_status = scalar_wavefront_parity_test::accumulate_path(
            *scalar_film, inputs[index], scalar_paths[index]);
        const auto wavefront_status = scalar_wavefront_parity_test::accumulate_path(
            *wavefront_film, inputs[index], wavefront.paths[index]);
        ASSERT_TRUE(scalar_status.has_value()) << scalar_status.error().message;
        ASSERT_TRUE(wavefront_status.has_value()) << wavefront_status.error().message;
    }
    const auto linear = renderer::compute_linear_metrics(*wavefront_film, *scalar_film);
    const auto display = renderer::compute_display_psnr(*wavefront_film, *scalar_film);
    const auto maximum_path_error =
        scalar_wavefront_parity_test::maximum_path_radiance_error(scalar_paths, wavefront.paths);
    ASSERT_TRUE(linear.has_value()) << linear.error().message;
    ASSERT_TRUE(display.has_value()) << display.error().message;
    ASSERT_TRUE(maximum_path_error.has_value()) << maximum_path_error.error().message;
    scalar_wavefront_parity_test::record_and_expect(
        scalar_wavefront_parity_test::Configuration{
            .scene_name = scene_name,
            .extent = extent,
            .samples_per_pixel = samples_per_pixel,
            .seed = EvaluationSeed,
            .heuristic = heuristic,
            .depth_limits = depth_limits,
            .roulette_policy = roulette_policy,
            .worker_count = expected_worker_count,
            .thresholds = scalar_wavefront_parity_test::StrictThresholds,
        },
        scalar_wavefront_parity_test::Result{
            .linear = *linear,
            .maximum_path_radiance_absolute_error = *maximum_path_error,
            .display_psnr = display->psnr,
            .wavefront_report = wavefront.report,
        });
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

struct MaterialParityResult final {
    std::vector<renderer::BsdfOnlyPathResult> scalar;
    CpuWavefrontMisBatch wavefront;
};

[[nodiscard]] core::Result<MaterialParityResult>
trace_material_parity(const FrameSceneHandle& scene,
                      const std::span<const CpuWavefrontMisPathInput> inputs,
                      const renderer::PathDepthLimits& depth_limits = OneSurfaceBounce,
                      const renderer::MisHeuristic heuristic = renderer::MisHeuristic::power) {
    const auto light_count = scene->punctual_lights().size() + scene->mesh_area_lights().size();
    const auto sampler = renderer::LightSampler::create_uniform(light_count);
    const auto scalar_backend = create_analytic_accel_backend(scene);
    const auto wavefront_backend = create_embree_accel_backend(scene);
    const auto scheduler = renderer::CpuWavefrontScheduler::create(4U);
    if (!sampler) {
        return std::unexpected(sampler.error());
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
    auto scalar = trace_scalar_batch(inputs, **scalar_backend, *sampler, heuristic, depth_limits,
                                     renderer::RussianRoulettePolicy::disabled());
    if (!scalar) {
        return std::unexpected(scalar.error());
    }
    auto wavefront =
        trace_cpu_wavefront_mis(inputs, *scheduler, **wavefront_backend, *sampler, heuristic,
                                depth_limits, renderer::RussianRoulettePolicy::disabled());
    if (!wavefront) {
        return std::unexpected(wavefront.error());
    }
    return MaterialParityResult{
        .scalar = std::move(*scalar),
        .wavefront = std::move(*wavefront),
    };
}

void expect_material_parity(const std::string_view scene_name,
                            const std::span<const CpuWavefrontMisPathInput> inputs,
                            const MaterialParityResult& parity,
                            const bool require_shadow_queries = true,
                            const renderer::PathDepthLimits& depth_limits = OneSurfaceBounce) {
    ASSERT_EQ(parity.scalar.size(), inputs.size());
    ASSERT_EQ(parity.wavefront.paths.size(), inputs.size());
    ASSERT_EQ(parity.wavefront.terminal_cones.size(), inputs.size());
    for (auto index = std::size_t{}; index < inputs.size(); ++index) {
        SCOPED_TRACE(index);
        expect_path_near(parity.scalar[index], parity.wavefront.paths[index],
                         ScalarParityTolerance);
    }
    ASSERT_LE(inputs.size(), std::numeric_limits<std::uint32_t>::max());
    expect_image_metrics(
        scene_name,
        renderer::RenderExtent{.width = static_cast<std::uint32_t>(inputs.size()), .height = 1U},
        1U, renderer::MisHeuristic::power, depth_limits,
        renderer::RussianRoulettePolicy::disabled(), inputs, parity.scalar, parity.wavefront,
        require_shadow_queries);
}

struct ExpectedQueueStatistics final {
    renderer::WavefrontQueueKind kind;
    std::uint64_t capacity;
    std::uint64_t peak_size;
    std::uint64_t dispatch_count;
    std::uint64_t input_lanes;
};

[[nodiscard]] constexpr std::array<const CpuWavefrontMisQueueStatistics*, 7U>
queue_statistics_in_stage_order(const CpuWavefrontMisQueueStatisticsSet& statistics) noexcept {
    return {
        &statistics.camera, &statistics.ray,    &statistics.hit,          &statistics.miss,
        &statistics.shade,  &statistics.shadow, &statistics.continuation,
    };
}

[[nodiscard]] constexpr std::array<std::uint64_t, 7U>
stage_lanes_in_stage_order(const CpuWavefrontMisStageLaneCounts& stage_lanes) noexcept {
    return {
        stage_lanes.camera, stage_lanes.ray,    stage_lanes.hit,          stage_lanes.miss,
        stage_lanes.shade,  stage_lanes.shadow, stage_lanes.continuation,
    };
}

void expect_queue_statistics(const CpuWavefrontMisReport& report,
                             const std::array<ExpectedQueueStatistics, 7U>& expected) {
    const auto actual = queue_statistics_in_stage_order(report.queue_statistics);
    const auto stage_lanes = stage_lanes_in_stage_order(report.stage_lanes);
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
        SCOPED_TRACE(std::string{renderer::wavefront_queue_kind_name(expected[index].kind)});
        EXPECT_EQ(actual[index]->kind, expected[index].kind);
        EXPECT_EQ(actual[index]->capacity, expected[index].capacity);
        EXPECT_EQ(actual[index]->peak_size, expected[index].peak_size);
        EXPECT_EQ(actual[index]->dispatch_count, expected[index].dispatch_count);
        EXPECT_EQ(actual[index]->input_lanes, expected[index].input_lanes);
        EXPECT_EQ(actual[index]->input_lanes, stage_lanes[index]);
        EXPECT_EQ(actual[index]->overflow_attempts, 0U);
        EXPECT_EQ(actual[index]->rejected_lanes, 0U);

        const auto peak_occupancy = actual[index]->peak_occupancy();
        const auto mean_occupancy = actual[index]->mean_occupancy();
        EXPECT_TRUE(std::isfinite(peak_occupancy));
        EXPECT_GE(peak_occupancy, 0.0);
        EXPECT_LE(peak_occupancy, 1.0);
        EXPECT_TRUE(std::isfinite(mean_occupancy));
        EXPECT_GE(mean_occupancy, 0.0);
        EXPECT_LE(mean_occupancy, 1.0);
    }
}

void expect_queue_statistics_structurally_equal(const CpuWavefrontMisQueueStatisticsSet& expected,
                                                const CpuWavefrontMisQueueStatisticsSet& actual) {
    const auto expected_stages = queue_statistics_in_stage_order(expected);
    const auto actual_stages = queue_statistics_in_stage_order(actual);
    for (auto index = std::size_t{}; index < expected_stages.size(); ++index) {
        SCOPED_TRACE(
            std::string{renderer::wavefront_queue_kind_name(expected_stages[index]->kind)});
        EXPECT_EQ(actual_stages[index]->kind, expected_stages[index]->kind);
        EXPECT_EQ(actual_stages[index]->capacity, expected_stages[index]->capacity);
        EXPECT_EQ(actual_stages[index]->peak_size, expected_stages[index]->peak_size);
        EXPECT_EQ(actual_stages[index]->dispatch_count, expected_stages[index]->dispatch_count);
        EXPECT_EQ(actual_stages[index]->input_lanes, expected_stages[index]->input_lanes);
        EXPECT_EQ(actual_stages[index]->overflow_attempts,
                  expected_stages[index]->overflow_attempts);
        EXPECT_EQ(actual_stages[index]->rejected_lanes, expected_stages[index]->rejected_lanes);
        EXPECT_EQ(actual_stages[index]->peak_occupancy(), expected_stages[index]->peak_occupancy());
        EXPECT_EQ(actual_stages[index]->mean_occupancy(), expected_stages[index]->mean_occupancy());
    }
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
    ASSERT_EQ(single->terminal_cones.size(), PathCount);
    ASSERT_EQ(parallel->terminal_cones.size(), PathCount);
    ASSERT_EQ(scalar->size(), PathCount);

    for (auto index = std::size_t{}; index < PathCount; ++index) {
        SCOPED_TRACE(index);
        expect_path_near((*scalar)[index], single->paths[index], ScalarParityTolerance);
        expect_path_near((*scalar)[index], parallel->paths[index], ScalarParityTolerance);
        expect_path_exact(single->paths[index], parallel->paths[index]);
        EXPECT_EQ(single->terminal_cones[index], parallel->terminal_cones[index]);
        EXPECT_EQ(single->paths[index].termination, renderer::BsdfOnlyPathTermination::depth_limit);
    }
    expect_image_metrics(
        GetParam() == renderer::MisHeuristic::balance ? "AreaEmitterBalance" : "AreaEmitterPower",
        renderer::RenderExtent{.width = 6U, .height = 4U}, 1U, GetParam(), OneDiffuseBounce,
        renderer::RussianRoulettePolicy::disabled(), *inputs, *scalar, *parallel);

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
    const auto expected_queue_statistics = std::array{
        ExpectedQueueStatistics{renderer::WavefrontQueueKind::camera, lane_count, lane_count, 1U,
                                expected_stages.camera},
        ExpectedQueueStatistics{renderer::WavefrontQueueKind::ray, lane_count, lane_count, 2U,
                                expected_stages.ray},
        ExpectedQueueStatistics{renderer::WavefrontQueueKind::hit, lane_count, lane_count, 2U,
                                expected_stages.hit},
        ExpectedQueueStatistics{renderer::WavefrontQueueKind::miss, lane_count, 0U, 0U,
                                expected_stages.miss},
        ExpectedQueueStatistics{renderer::WavefrontQueueKind::shade, lane_count, lane_count, 2U,
                                expected_stages.shade},
        ExpectedQueueStatistics{renderer::WavefrontQueueKind::shadow, lane_count, lane_count, 1U,
                                expected_stages.shadow},
        ExpectedQueueStatistics{renderer::WavefrontQueueKind::continuation, lane_count, lane_count,
                                1U, expected_stages.continuation},
    };
    for (const auto* const report : std::array{&single->report, &parallel->report}) {
        EXPECT_EQ(report->schema_version, 2U);
        EXPECT_EQ(report->schema_version, CurrentCpuWavefrontMisReportSchemaVersion);
        EXPECT_EQ(report->path_count, PathCount);
        EXPECT_EQ(report->stage_lanes, expected_stages);
        EXPECT_EQ(report->closure_samples, lane_count);
        EXPECT_EQ(report->light_samples, lane_count);
        EXPECT_EQ(report->shadow_queries, lane_count);
        expect_queue_statistics(*report, expected_queue_statistics);
    }
    EXPECT_EQ(single->report.configured_workers, 1U);
    EXPECT_EQ(parallel->report.configured_workers, 4U);
    EXPECT_EQ(single->report.stage_lanes, parallel->report.stage_lanes);
    EXPECT_EQ(single->report.closure_samples, parallel->report.closure_samples);
    EXPECT_EQ(single->report.light_samples, parallel->report.light_samples);
    EXPECT_EQ(single->report.shadow_queries, parallel->report.shadow_queries);
    expect_queue_statistics_structurally_equal(single->report.queue_statistics,
                                               parallel->report.queue_statistics);
}

INSTANTIATE_TEST_SUITE_P(BalanceAndPower, WavefrontMisTransportTest,
                         testing::Values(renderer::MisHeuristic::balance,
                                         renderer::MisHeuristic::power),
                         [](const testing::TestParamInfo<renderer::MisHeuristic>& parameter) {
                             return parameter.param == renderer::MisHeuristic::balance ? "Balance"
                                                                                       : "Power";
                         });

TEST(WavefrontMisTransportQueueTest, ReportsCanonicalZeroStatisticsForAnEmptyBatch) {
    const auto scene = make_wavefront_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto sampler = renderer::LightSampler::create_uniform(1U);
    const auto backend = create_embree_accel_backend(*scene);
    const auto scheduler = renderer::CpuWavefrontScheduler::create(4U);
    ASSERT_TRUE(sampler.has_value()) << sampler.error().message;
    ASSERT_TRUE(backend.has_value()) << backend.error().message;
    ASSERT_TRUE(scheduler.has_value()) << scheduler.error().message;

    const auto batch =
        trace_cpu_wavefront_mis(std::span<const CpuWavefrontMisPathInput>{}, *scheduler, **backend,
                                *sampler, renderer::MisHeuristic::power, OneDiffuseBounce,
                                renderer::RussianRoulettePolicy::disabled());
    ASSERT_TRUE(batch.has_value()) << batch.error().message;
    EXPECT_TRUE(batch->paths.empty());
    EXPECT_TRUE(batch->terminal_cones.empty());
    EXPECT_EQ(batch->report.schema_version, 2U);
    EXPECT_EQ(batch->report.schema_version, CurrentCpuWavefrontMisReportSchemaVersion);
    EXPECT_EQ(batch->report.configured_workers, 4U);
    EXPECT_EQ(batch->report.path_count, 0U);
    EXPECT_EQ(batch->report.stage_lanes, CpuWavefrontMisStageLaneCounts{});
    EXPECT_EQ(batch->report.closure_samples, 0U);
    EXPECT_EQ(batch->report.light_samples, 0U);
    EXPECT_EQ(batch->report.shadow_queries, 0U);

    constexpr auto expected_kinds = std::array{
        renderer::WavefrontQueueKind::camera,       renderer::WavefrontQueueKind::ray,
        renderer::WavefrontQueueKind::hit,          renderer::WavefrontQueueKind::miss,
        renderer::WavefrontQueueKind::shade,        renderer::WavefrontQueueKind::shadow,
        renderer::WavefrontQueueKind::continuation,
    };
    const auto statistics = queue_statistics_in_stage_order(batch->report.queue_statistics);
    for (auto index = std::size_t{}; index < statistics.size(); ++index) {
        SCOPED_TRACE(std::string{renderer::wavefront_queue_kind_name(expected_kinds[index])});
        EXPECT_EQ(statistics[index]->kind, expected_kinds[index]);
        EXPECT_EQ(statistics[index]->capacity, 0U);
        EXPECT_EQ(statistics[index]->peak_size, 0U);
        EXPECT_EQ(statistics[index]->dispatch_count, 0U);
        EXPECT_EQ(statistics[index]->input_lanes, 0U);
        EXPECT_EQ(statistics[index]->overflow_attempts, 0U);
        EXPECT_EQ(statistics[index]->rejected_lanes, 0U);
        EXPECT_EQ(statistics[index]->stage_wall_nanoseconds, 0U);
        EXPECT_EQ(statistics[index]->peak_occupancy(), 0.0);
        EXPECT_EQ(statistics[index]->mean_occupancy(), 0.0);
    }
}

TEST(WavefrontMisTransportQueueTest, ExcludesOnlyStageTimingFromDeterministicReportEquality) {
    auto first = CpuWavefrontMisReport{
        .schema_version = CurrentCpuWavefrontMisReportSchemaVersion,
        .configured_workers = 4U,
        .path_count = 8U,
        .queue_statistics =
            CpuWavefrontMisQueueStatisticsSet{
                .ray =
                    CpuWavefrontMisQueueStatistics{
                        .kind = renderer::WavefrontQueueKind::ray,
                        .capacity = 8U,
                        .peak_size = 7U,
                        .dispatch_count = 2U,
                        .input_lanes = 11U,
                        .stage_wall_nanoseconds = 100U,
                    },
            },
    };
    auto replay = first;
    replay.queue_statistics.ray.stage_wall_nanoseconds = 900U;
    EXPECT_EQ(replay, first);

    replay.queue_statistics.ray.rejected_lanes = 1U;
    EXPECT_NE(replay, first);
}

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
    expect_image_metrics("EnvironmentMiss", renderer::RenderExtent{.width = 6U, .height = 4U}, 1U,
                         heuristic, OneDiffuseBounce, renderer::RussianRoulettePolicy::disabled(),
                         *inputs, *scalar, *parallel);
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
    expect_image_metrics("RussianRoulette", renderer::RenderExtent{.width = 6U, .height = 4U}, 1U,
                         heuristic, depth_limits, *roulette, *inputs, *scalar, *parallel);

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
    expect_image_metrics("SubnormalReflectance", renderer::RenderExtent{.width = 1U, .height = 1U},
                         1U, heuristic, OneDiffuseBounce,
                         renderer::RussianRoulettePolicy::disabled(), *inputs, *scalar, *wavefront);
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
    expect_image_metrics("OccludedPointLight", renderer::RenderExtent{.width = 1U, .height = 1U},
                         1U, heuristic, OneDiffuseBounce,
                         renderer::RussianRoulettePolicy::disabled(), *inputs, *scalar, *wavefront);
}

TEST(WavefrontMisMaterialParityTest, MatchesRoughDiffuseAndIsotropicConductorEvents) {
    const auto primary = material_primary_ray(renderer::Vector3{.z = 1.0F});
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    constexpr auto bands =
        std::array{ComponentSampleBand{.lower = 0.0F, .upper = 1.0F, .count = 16U}};

    const auto verify = [&](const std::string_view name, SceneClosureMixture closure,
                            const renderer::ScatteringLobe expected_family) {
        SCOPED_TRACE(name);
        const auto scene = make_material_scene(std::move(closure));
        ASSERT_TRUE(scene.has_value()) << scene.error().message;
        const auto inputs = make_material_inputs(*scene, *primary, bands);
        ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
        const auto parity = trace_material_parity(*scene, *inputs);
        ASSERT_TRUE(parity.has_value()) << parity.error().message;

        auto accepted_events = std::size_t{};
        for (const auto& path : parity->scalar) {
            if (path.termination == renderer::BsdfOnlyPathTermination::outside_bsdf_support) {
                EXPECT_EQ(path.state.depth(), 0U);
                EXPECT_EQ(path.state.depth_counters(), renderer::PathDepthCounters{});
                EXPECT_EQ(path.state.delta_flags(), renderer::PathDeltaFlags::none);
                continue;
            }
            ++accepted_events;
            EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::escaped_environment);
            EXPECT_EQ(path.state.depth(), 1U);
            EXPECT_EQ(path.state.eta_scale(), 1.0F);
            EXPECT_EQ(path.state.delta_flags(), renderer::PathDeltaFlags::any_non_delta_bounces);
            EXPECT_GT(path.terminal_ray.direction().z, 0.0F);
            if (expected_family == renderer::ScatteringLobe::diffuse) {
                EXPECT_EQ(path.state.depth_counters(),
                          (renderer::PathDepthCounters{.diffuse = 1U}));
            } else {
                EXPECT_EQ(path.state.depth_counters(), (renderer::PathDepthCounters{.glossy = 1U}));
            }
            for (const auto radiance : path.state.accumulated_radiance().values) {
                EXPECT_TRUE(std::isfinite(radiance));
                EXPECT_GT(radiance, 0.0F);
            }
        }
        EXPECT_GT(accepted_events, 0U);
        expect_material_parity(name, *inputs, *parity);
    };

    verify("RoughDiffuse", rough_diffuse_scene_closure(), renderer::ScatteringLobe::diffuse);
    verify("RoughConductorIsotropic", rough_conductor_scene_closure(0.35F, 0.35F),
           renderer::ScatteringLobe::glossy);
}

TEST(WavefrontMisMaterialParityTest, RotatesAnisotropicConductorFrameCoherently) {
    const auto unrotated_scene = make_material_scene(rough_conductor_scene_closure(0.15F, 0.6F));
    const auto rotated_scene = make_material_scene(
        rough_conductor_scene_closure(0.15F, 0.6F, std::numbers::pi_v<float> / 2.0F));
    const auto primary = material_primary_ray(renderer::Vector3{.z = 1.0F});
    ASSERT_TRUE(unrotated_scene.has_value()) << unrotated_scene.error().message;
    ASSERT_TRUE(rotated_scene.has_value()) << rotated_scene.error().message;
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    constexpr auto bands =
        std::array{ComponentSampleBand{.lower = 0.0F, .upper = 1.0F, .count = 16U}};
    const auto inputs = make_material_inputs(*unrotated_scene, *primary, bands);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    const auto unrotated = trace_material_parity(*unrotated_scene, *inputs);
    const auto rotated = trace_material_parity(*rotated_scene, *inputs);
    ASSERT_TRUE(unrotated.has_value()) << unrotated.error().message;
    ASSERT_TRUE(rotated.has_value()) << rotated.error().message;
    ASSERT_EQ(unrotated->scalar.size(), rotated->scalar.size());

    auto observed_azimuthal_sample = false;
    auto accepted_events = std::size_t{};
    for (auto index = std::size_t{}; index < unrotated->scalar.size(); ++index) {
        SCOPED_TRACE(index);
        const auto& base = unrotated->scalar[index];
        const auto& quarter_turn = rotated->scalar[index];
        EXPECT_EQ(quarter_turn.termination, base.termination);
        if (base.termination == renderer::BsdfOnlyPathTermination::outside_bsdf_support) {
            EXPECT_EQ(base.state.depth_counters(), renderer::PathDepthCounters{});
            EXPECT_EQ(quarter_turn.state.depth_counters(), renderer::PathDepthCounters{});
            continue;
        }
        ++accepted_events;
        EXPECT_EQ(base.state.depth_counters(), (renderer::PathDepthCounters{.glossy = 1U}));
        EXPECT_EQ(quarter_turn.state.depth_counters(), base.state.depth_counters());
        EXPECT_NEAR(quarter_turn.terminal_ray.direction().x, -base.terminal_ray.direction().y,
                    ScalarParityTolerance);
        EXPECT_NEAR(quarter_turn.terminal_ray.direction().y, base.terminal_ray.direction().x,
                    ScalarParityTolerance);
        EXPECT_NEAR(quarter_turn.terminal_ray.direction().z, base.terminal_ray.direction().z,
                    ScalarParityTolerance);
        EXPECT_EQ(quarter_turn.state.beta(), base.state.beta());
        for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
            EXPECT_NEAR(quarter_turn.state.accumulated_radiance()[lane],
                        base.state.accumulated_radiance()[lane], ScalarParityTolerance);
        }
        observed_azimuthal_sample = observed_azimuthal_sample ||
                                    std::abs(base.terminal_ray.direction().x) > 1.0e-3F ||
                                    std::abs(base.terminal_ray.direction().y) > 1.0e-3F;
    }
    EXPECT_GT(accepted_events, 0U);
    EXPECT_TRUE(observed_azimuthal_sample);
    expect_material_parity("RoughConductorAnisotropic", *inputs, *unrotated);
    expect_material_parity("RoughConductorAnisotropicRotated", *inputs, *rotated);
}

TEST(WavefrontMisMaterialParityTest, MatchesRoughDielectricReflectionTransmissionAndTir) {
    const auto scene = make_material_scene(rough_dielectric_scene_closure());
    const auto primary = material_primary_ray(renderer::Vector3{.z = 1.0F});
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    constexpr auto branch_bands = std::array{
        ComponentSampleBand{.lower = 0.0F, .upper = 0.01F, .count = 8U},
        ComponentSampleBand{.lower = 0.5F, .upper = 0.95F, .count = 8U},
    };
    const auto inputs = make_material_inputs(*scene, *primary, branch_bands);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    const auto parity = trace_material_parity(*scene, *inputs);
    ASSERT_TRUE(parity.has_value()) << parity.error().message;

    auto reflected = std::size_t{};
    auto transmitted = std::size_t{};
    auto supported_events = std::size_t{};
    for (const auto& path : parity->scalar) {
        if (path.termination == renderer::BsdfOnlyPathTermination::outside_bsdf_support) {
            EXPECT_EQ(path.state.depth(), 0U);
            EXPECT_EQ(path.state.depth_counters(), renderer::PathDepthCounters{});
            continue;
        }
        ++supported_events;
        EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::escaped_environment);
        EXPECT_EQ(path.state.depth(), 1U);
        EXPECT_EQ(path.state.depth_counters().glossy, 1U);
        EXPECT_EQ(path.state.delta_flags(), renderer::PathDeltaFlags::any_non_delta_bounces);
        if (path.terminal_ray.direction().z > 0.0F) {
            ++reflected;
            EXPECT_EQ(path.state.depth_counters().transmission, 0U);
            EXPECT_EQ(path.state.eta_scale(), 1.0F);
        } else {
            ++transmitted;
            EXPECT_EQ(path.state.depth_counters().transmission, 1U);
            EXPECT_NEAR(path.state.eta_scale(), 2.25F, ScalarParityTolerance);
        }
        for (const auto radiance : path.state.accumulated_radiance().values) {
            EXPECT_TRUE(std::isfinite(radiance));
            EXPECT_GT(radiance, 0.0F);
        }
    }
    EXPECT_GT(supported_events, 0U);
    EXPECT_GT(reflected, 0U);
    EXPECT_GT(transmitted, 0U);
    expect_material_parity("RoughDielectricBranches", *inputs, *parity);

    const auto tir_scene = make_material_scene(rough_dielectric_scene_closure(0.05F, 0.2F));
    constexpr auto inside_outgoing = renderer::Vector3{.x = 0.8F, .z = -0.6F};
    const auto tir_primary = material_primary_ray(inside_outgoing);
    ASSERT_TRUE(tir_scene.has_value()) << tir_scene.error().message;
    ASSERT_TRUE(tir_primary.has_value()) << tir_primary.error().message;
    constexpr auto tir_band =
        std::array{ComponentSampleBand{.lower = 0.95F, .upper = 1.0F, .count = 64U}};
    const auto tir_inputs = make_material_inputs(*tir_scene, *tir_primary, tir_band);
    ASSERT_TRUE(tir_inputs.has_value()) << tir_inputs.error().message;
    const auto tir_parity = trace_material_parity(*tir_scene, *tir_inputs);
    ASSERT_TRUE(tir_parity.has_value()) << tir_parity.error().message;

    auto total_internal_reflections = std::size_t{};
    for (const auto& path : tir_parity->scalar) {
        if (path.state.depth_counters().glossy != 1U ||
            path.state.depth_counters().transmission != 0U ||
            path.terminal_ray.direction().z >= 0.0F) {
            continue;
        }
        auto microfacet = inside_outgoing + path.terminal_ray.direction();
        const auto length = std::hypot(std::hypot(microfacet.x, microfacet.y), microfacet.z);
        ASSERT_TRUE(std::isfinite(length));
        ASSERT_GT(length, 0.0F);
        microfacet = microfacet / length;
        if (renderer::dot(inside_outgoing, microfacet) < 0.0F) {
            microfacet = microfacet * -1.0F;
        }
        const auto fresnel = renderer::dielectric_fresnel(
            std::abs(renderer::dot(inside_outgoing, microfacet)), 1.5F, 1.0F);
        ASSERT_TRUE(fresnel.has_value()) << fresnel.error().message;
        if (*fresnel == 1.0F) {
            ++total_internal_reflections;
            EXPECT_EQ(path.state.eta_scale(), 1.0F);
            EXPECT_EQ(path.state.delta_flags(), renderer::PathDeltaFlags::any_non_delta_bounces);
        }
    }
    EXPECT_GT(total_internal_reflections, 0U);
    expect_material_parity("RoughDielectricTir", *tir_inputs, *tir_parity, false);
}

TEST(WavefrontMisMaterialParityTest, MatchesSpecularDeltaEntryExitAndTir) {
    constexpr auto bands =
        std::array{ComponentSampleBand{.lower = 0.0F, .upper = 1.0F, .count = 4U}};
    {
        const auto scene = make_material_scene(mirror_scene_closure());
        const auto primary = material_primary_ray(renderer::Vector3{.x = 0.6F, .z = 0.8F});
        ASSERT_TRUE(scene.has_value()) << scene.error().message;
        ASSERT_TRUE(primary.has_value()) << primary.error().message;
        const auto inputs = make_material_inputs(*scene, *primary, bands);
        ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
        const auto parity = trace_material_parity(*scene, *inputs);
        ASSERT_TRUE(parity.has_value()) << parity.error().message;
        for (const auto& path : parity->scalar) {
            EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::escaped_environment);
            EXPECT_EQ(path.state.depth_counters(), (renderer::PathDepthCounters{.specular = 1U}));
            EXPECT_EQ(path.state.delta_flags(),
                      renderer::PathDeltaFlags::previous_bounce_was_delta);
            EXPECT_EQ(path.state.eta_scale(), 1.0F);
            EXPECT_NEAR(path.terminal_ray.direction().x, -0.6F, ScalarParityTolerance);
            EXPECT_NEAR(path.terminal_ray.direction().z, 0.8F, ScalarParityTolerance);
        }
        expect_material_parity("SpecularMirror", *inputs, *parity, false);
    }
    {
        const auto scene = make_material_scene(transmission_scene_closure());
        const auto primary = material_primary_ray(renderer::Vector3{.x = 0.6F, .z = 0.8F});
        ASSERT_TRUE(scene.has_value()) << scene.error().message;
        ASSERT_TRUE(primary.has_value()) << primary.error().message;
        const auto inputs = make_material_inputs(*scene, *primary, bands);
        ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
        const auto parity = trace_material_parity(*scene, *inputs);
        ASSERT_TRUE(parity.has_value()) << parity.error().message;
        for (const auto& path : parity->scalar) {
            EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::escaped_environment);
            EXPECT_EQ(path.state.depth_counters(),
                      (renderer::PathDepthCounters{.specular = 1U, .transmission = 1U}));
            EXPECT_EQ(path.state.delta_flags(),
                      renderer::PathDeltaFlags::previous_bounce_was_delta);
            EXPECT_NEAR(path.state.eta_scale(), 2.25F, ScalarParityTolerance);
            EXPECT_NEAR(path.terminal_ray.direction().x, -0.4F, ScalarParityTolerance);
            EXPECT_NEAR(path.terminal_ray.direction().z, -std::sqrt(0.84F), ScalarParityTolerance);
        }
        expect_material_parity("SpecularTransmissionEntry", *inputs, *parity, false);
    }
    {
        const auto scene = make_material_scene(transmission_scene_closure());
        const auto primary = material_primary_ray(renderer::Vector3{.x = 0.6F, .z = -0.8F});
        ASSERT_TRUE(scene.has_value()) << scene.error().message;
        ASSERT_TRUE(primary.has_value()) << primary.error().message;
        const auto inputs = make_material_inputs(*scene, *primary, bands);
        ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
        const auto parity = trace_material_parity(*scene, *inputs);
        ASSERT_TRUE(parity.has_value()) << parity.error().message;
        for (const auto& path : parity->scalar) {
            EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::escaped_environment);
            EXPECT_EQ(path.state.depth_counters(),
                      (renderer::PathDepthCounters{.specular = 1U, .transmission = 1U}));
            EXPECT_EQ(path.state.delta_flags(),
                      renderer::PathDeltaFlags::previous_bounce_was_delta);
            EXPECT_NEAR(path.state.eta_scale(), 4.0F / 9.0F, ScalarParityTolerance);
            EXPECT_NEAR(path.terminal_ray.direction().x, -0.9F, ScalarParityTolerance);
            EXPECT_NEAR(path.terminal_ray.direction().z, std::sqrt(0.19F), ScalarParityTolerance);
        }
        expect_material_parity("SpecularTransmissionExit", *inputs, *parity, false);
    }
    {
        const auto scene = make_material_scene(transmission_scene_closure());
        const auto primary = material_primary_ray(renderer::Vector3{.x = 0.8F, .z = -0.6F});
        ASSERT_TRUE(scene.has_value()) << scene.error().message;
        ASSERT_TRUE(primary.has_value()) << primary.error().message;
        const auto inputs = make_material_inputs(*scene, *primary, bands);
        ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
        const auto parity = trace_material_parity(*scene, *inputs);
        ASSERT_TRUE(parity.has_value()) << parity.error().message;
        for (auto index = std::size_t{}; index < parity->scalar.size(); ++index) {
            const auto& path = parity->scalar[index];
            EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::outside_bsdf_support);
            EXPECT_EQ(path.state.depth_counters(), renderer::PathDepthCounters{});
            EXPECT_EQ(path.state.delta_flags(), renderer::PathDeltaFlags::none);
            EXPECT_EQ(path.state.eta_scale(), 1.0F);
            EXPECT_EQ(path.state.accumulated_radiance(), renderer::TransportSpectrum{});
            EXPECT_EQ(path.terminal_ray.direction(), inputs->at(index).primary_ray.direction());
        }
        expect_material_parity("SpecularTransmissionTir", *inputs, *parity, false);
    }
}

TEST(WavefrontMisRayConeTest, PropagatesOnlyWhenAContinuationRayIsCreated) {
    constexpr auto bands =
        std::array{ComponentSampleBand{.lower = 0.0F, .upper = 1.0F, .count = 1U}};
    const auto scene = make_material_scene(mirror_scene_closure());
    const auto primary = material_primary_ray(renderer::Vector3{.x = 0.6F, .z = 0.8F});
    const auto cone = renderer::RayCone::create(0.125F, 0.25F);
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    ASSERT_TRUE(cone.has_value()) << cone.error().message;
    auto inputs = make_material_inputs(*scene, *primary, bands);
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    inputs->front().primary_cone = *cone;

    const auto continued = trace_material_parity(*scene, *inputs);
    ASSERT_TRUE(continued.has_value()) << continued.error().message;
    ASSERT_EQ(continued->wavefront.paths.size(), 1U);
    ASSERT_EQ(continued->wavefront.terminal_cones.size(), 1U);
    EXPECT_EQ(continued->wavefront.paths.front().termination,
              renderer::BsdfOnlyPathTermination::escaped_environment);
    EXPECT_NE(continued->wavefront.paths.front().terminal_ray.origin(),
              inputs->front().primary_ray.origin());
    EXPECT_FLOAT_EQ(continued->wavefront.terminal_cones.front().width(), 0.375F);
    EXPECT_FLOAT_EQ(continued->wavefront.terminal_cones.front().spread(), 0.25F);

    const auto terminal_hit = trace_material_parity(*scene, *inputs, renderer::PathDepthLimits{},
                                                    renderer::MisHeuristic::power);
    ASSERT_TRUE(terminal_hit.has_value()) << terminal_hit.error().message;
    ASSERT_EQ(terminal_hit->wavefront.paths.size(), 1U);
    ASSERT_EQ(terminal_hit->wavefront.terminal_cones.size(), 1U);
    EXPECT_EQ(terminal_hit->wavefront.paths.front().termination,
              renderer::BsdfOnlyPathTermination::depth_limit);
    EXPECT_EQ(terminal_hit->wavefront.paths.front().terminal_ray.origin(),
              inputs->front().primary_ray.origin());
    EXPECT_EQ(terminal_hit->wavefront.paths.front().terminal_ray.direction(),
              inputs->front().primary_ray.direction());
    EXPECT_EQ(terminal_hit->wavefront.paths.front().terminal_ray.t_min(),
              inputs->front().primary_ray.t_min());
    EXPECT_EQ(terminal_hit->wavefront.paths.front().terminal_ray.t_max(),
              inputs->front().primary_ray.t_max());
    EXPECT_EQ(terminal_hit->wavefront.terminal_cones.front(), inputs->front().primary_cone);
}

TEST(WavefrontMisMaterialParityTest, MatchesContinuousAndMixedMeasureMixtures) {
    const auto primary = material_primary_ray(renderer::Vector3{.z = 1.0F});
    const auto dimensions = renderer::sample_dimensions_for_bounce(0U);
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    ASSERT_TRUE(dimensions.has_value()) << dimensions.error().message;
    {
        const auto mixture = continuous_mixture_scene_closure();
        const auto first_probability = mixture.active_component_probabilities().front();
        const auto scene = make_material_scene(mixture);
        ASSERT_TRUE(scene.has_value()) << scene.error().message;
        const auto bands = std::array{
            ComponentSampleBand{.lower = 0.0F, .upper = first_probability * 0.9F, .count = 8U},
            ComponentSampleBand{.lower = first_probability + 0.05F, .upper = 1.0F, .count = 8U},
        };
        const auto inputs = make_material_inputs(*scene, *primary, bands);
        ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
        const auto parity = trace_material_parity(*scene, *inputs);
        ASSERT_TRUE(parity.has_value()) << parity.error().message;
        auto first_selected = std::size_t{};
        auto second_selected = std::size_t{};
        for (auto index = std::size_t{}; index < parity->scalar.size(); ++index) {
            const auto component = renderer::SampleStream{inputs->at(index).sample}.sample_1d(
                dimensions->bsdf_component);
            component < first_probability ? ++first_selected : ++second_selected;
            const auto& path = parity->scalar[index];
            EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::escaped_environment);
            EXPECT_EQ(path.state.depth_counters(), (renderer::PathDepthCounters{.diffuse = 1U}));
            EXPECT_EQ(path.state.delta_flags(), renderer::PathDeltaFlags::any_non_delta_bounces);
            EXPECT_EQ(path.state.eta_scale(), 1.0F);
        }
        EXPECT_GT(first_selected, 0U);
        EXPECT_GT(second_selected, 0U);
        expect_material_parity("ContinuousClosureMixture", *inputs, *parity);
    }
    {
        const auto mixture = mixed_measure_scene_closure();
        const auto first_probability = mixture.active_component_probabilities().front();
        const auto scene = make_material_scene(mixture);
        ASSERT_TRUE(scene.has_value()) << scene.error().message;
        const auto bands = std::array{
            ComponentSampleBand{.lower = 0.0F, .upper = first_probability * 0.9F, .count = 8U},
            ComponentSampleBand{.lower = first_probability + 0.05F, .upper = 1.0F, .count = 8U},
        };
        const auto inputs = make_material_inputs(*scene, *primary, bands);
        ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
        const auto parity = trace_material_parity(*scene, *inputs);
        ASSERT_TRUE(parity.has_value()) << parity.error().message;
        auto diffuse_selected = std::size_t{};
        auto mirror_selected = std::size_t{};
        for (auto index = std::size_t{}; index < parity->scalar.size(); ++index) {
            const auto component = renderer::SampleStream{inputs->at(index).sample}.sample_1d(
                dimensions->bsdf_component);
            const auto& path = parity->scalar[index];
            EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::escaped_environment);
            EXPECT_EQ(path.state.eta_scale(), 1.0F);
            if (component < first_probability) {
                ++diffuse_selected;
                EXPECT_EQ(path.state.depth_counters(),
                          (renderer::PathDepthCounters{.diffuse = 1U}));
                EXPECT_EQ(path.state.delta_flags(),
                          renderer::PathDeltaFlags::any_non_delta_bounces);
            } else {
                ++mirror_selected;
                EXPECT_EQ(path.state.depth_counters(),
                          (renderer::PathDepthCounters{.specular = 1U}));
                EXPECT_EQ(path.state.delta_flags(),
                          renderer::PathDeltaFlags::previous_bounce_was_delta);
                EXPECT_NEAR(path.terminal_ray.direction().x, 0.0F, ScalarParityTolerance);
                EXPECT_NEAR(path.terminal_ray.direction().y, 0.0F, ScalarParityTolerance);
                EXPECT_NEAR(path.terminal_ray.direction().z, 1.0F, ScalarParityTolerance);
            }
        }
        EXPECT_GT(diffuse_selected, 0U);
        EXPECT_GT(mirror_selected, 0U);
        expect_material_parity("MixedMeasureClosureMixture", *inputs, *parity);
    }
}

TEST(WavefrontMisMaterialParityTest,
     MatchesDepthFilteredContinuousMixturesAndReflectionOnlyRoughDielectric) {
    const auto primary = material_primary_ray(renderer::Vector3{.z = 1.0F});
    ASSERT_TRUE(primary.has_value()) << primary.error().message;
    constexpr auto bands = std::array{
        ComponentSampleBand{.lower = 0.0F, .upper = 0.3F, .count = 8U},
        ComponentSampleBand{.lower = 0.4F, .upper = 1.0F, .count = 8U},
    };

    {
        constexpr auto diffuse_only = renderer::PathDepthLimits{
            .diffuse = 1U,
            .glossy = 0U,
        };
        const auto scene = make_material_scene(depth_filtered_continuous_mixture_scene_closure());
        ASSERT_TRUE(scene.has_value()) << scene.error().message;
        const auto inputs = make_material_inputs(*scene, *primary, bands);
        ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
        const auto parity = trace_material_parity(*scene, *inputs, diffuse_only);
        ASSERT_TRUE(parity.has_value()) << parity.error().message;
        for (const auto& path : parity->scalar) {
            EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::escaped_environment);
            EXPECT_EQ(path.state.depth_counters(), (renderer::PathDepthCounters{.diffuse = 1U}));
            EXPECT_EQ(path.state.beta(), constant_spectrum(0.25F));
            EXPECT_EQ(path.state.delta_flags(), renderer::PathDeltaFlags::any_non_delta_bounces);
        }
        expect_material_parity("DepthFilteredContinuousClosureMixture", *inputs, *parity, true,
                               diffuse_only);
    }

    {
        constexpr auto reflection_only = renderer::PathDepthLimits{
            .glossy = 1U,
            .transmission = 0U,
        };
        const auto scene = make_material_scene(rough_dielectric_scene_closure());
        ASSERT_TRUE(scene.has_value()) << scene.error().message;
        const auto inputs = make_material_inputs(*scene, *primary, bands);
        ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
        const auto parity = trace_material_parity(*scene, *inputs, reflection_only);
        ASSERT_TRUE(parity.has_value()) << parity.error().message;
        auto reflected = std::size_t{};
        for (const auto& path : parity->scalar) {
            if (path.termination == renderer::BsdfOnlyPathTermination::outside_bsdf_support) {
                EXPECT_EQ(path.state.depth_counters(), renderer::PathDepthCounters{});
                continue;
            }
            ++reflected;
            EXPECT_EQ(path.termination, renderer::BsdfOnlyPathTermination::escaped_environment);
            EXPECT_EQ(path.state.depth_counters(), (renderer::PathDepthCounters{.glossy = 1U}));
            EXPECT_GT(path.terminal_ray.direction().z, 0.0F);
            EXPECT_EQ(path.state.eta_scale(), 1.0F);
            EXPECT_EQ(path.state.delta_flags(), renderer::PathDeltaFlags::any_non_delta_bounces);
        }
        EXPECT_GT(reflected, 0U);
        expect_material_parity("ReflectionOnlyRoughDielectric", *inputs, *parity, true,
                               reflection_only);
    }
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
