#include "ScalarWavefrontImageParity.hpp"

#include <Blackframe/Backends/CPU/Embree/AccelBackend.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/SceneMisPathLoop.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Renderer/Cie1931Sensor.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/LocalFrame.hpp>
#include <Blackframe/Renderer/PathDepthLimits.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <Blackframe/Renderer/PinholeCamera.hpp>
#include <Blackframe/Renderer/PngWriter.hpp>
#include <Blackframe/Renderer/RussianRoulette.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if BLACKFRAME_HAS_PNG_PREVIEW
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#endif

namespace blackframe::engine {
namespace {

inline constexpr auto EnergySampleCount = std::uint32_t{32'768U};
inline constexpr auto ReplaySampleCount = std::uint32_t{2'048U};
inline constexpr auto ImageExtent = renderer::RenderExtent{.width = 64U, .height = 64U};
inline constexpr auto ImageSamplesPerPixel = std::uint32_t{128U};
inline constexpr auto EvaluationSeed = std::uint64_t{0x6A09E667F3BCC909ULL};
inline constexpr auto ReceiverReflectance = renderer::TransportScalar{0.65F};
inline constexpr auto EnergyEmitterHeight = renderer::TransportScalar{2.0F};
inline constexpr auto EnergyEmitterHalfExtent = renderer::TransportScalar{1.0F};
inline constexpr auto PathTime = renderer::TransportScalar{0.375F};
inline constexpr auto ReceiverInstance = renderer::InstanceId{.value = 31U};
inline constexpr auto EnergyEmitterInstance = renderer::InstanceId{.value = 32U};

[[nodiscard]] core::Error test_error(std::string message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = std::move(message),
    };
}

[[nodiscard]] renderer::TransportSpectrum constant_spectrum(const float value) {
    auto result = renderer::TransportSpectrum{};
    result.values.fill(value);
    return result;
}

[[nodiscard]] SceneClosureMixture
require_lambertian_scene_closure(const renderer::TransportSpectrum reflectance) {
    return SceneClosureMixture::create_lambertian(reflectance).value();
}

// At the fixed packet below these non-negative lane weights map to equal
// scene-linear sRGB components through Blackframe's CIE 1931 sensor. Keeping
// all diffuse reflectances spectrally constant preserves that neutrality at
// every path depth instead of correcting color after transport.
[[nodiscard]] renderer::TransportSpectrum neutral_emission(const float scale = 1.0F) {
    return renderer::TransportSpectrum{
        .values =
            {
                7.28694487F * scale,
                0.952785134F * scale,
                1.53811407F * scale,
                0.0F,
            },
    };
}

[[nodiscard]] renderer::SampledWavelengths test_wavelengths() {
    return renderer::sample_uniform_visible_wavelengths(0.1F).value();
}

[[nodiscard]] core::Result<std::shared_ptr<const TriangleMesh>>
horizontal_quad(const renderer::Point3 center, const float half_width, const float half_depth,
                const bool faces_down) {
    const auto normal = faces_down ? renderer::Normal3{.z = -1.0F} : renderer::Normal3{.z = 1.0F};
    auto mesh = TriangleMesh::create(
        {
            renderer::Point3{.x = center.x - half_width,
                             .y = center.y - half_depth,
                             .z = center.z},
            renderer::Point3{.x = center.x + half_width,
                             .y = center.y - half_depth,
                             .z = center.z},
            renderer::Point3{.x = center.x + half_width,
                             .y = center.y + half_depth,
                             .z = center.z},
            renderer::Point3{.x = center.x - half_width,
                             .y = center.y + half_depth,
                             .z = center.z},
        },
        std::vector(4U, normal),
        {
            renderer::Point2{},
            renderer::Point2{.x = 1.0F},
            renderer::Point2{.x = 1.0F, .y = 1.0F},
            renderer::Point2{.y = 1.0F},
        },
        faces_down ? std::vector{
                         TriangleVertexIndices{.vertices = {0U, 2U, 1U}},
                         TriangleVertexIndices{.vertices = {0U, 3U, 2U}},
                     }
                   : std::vector{
                         TriangleVertexIndices{.vertices = {0U, 1U, 2U}},
                         TriangleVertexIndices{.vertices = {0U, 2U, 3U}},
                     });
    if (!mesh) {
        return std::unexpected(mesh.error());
    }
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

[[nodiscard]] core::Result<std::shared_ptr<const TriangleMesh>>
back_wall_quad(const float center_x, const float y, const float half_width, const float height) {
    auto mesh = TriangleMesh::create(
        {
            renderer::Point3{.x = center_x - half_width, .y = y},
            renderer::Point3{.x = center_x + half_width, .y = y},
            renderer::Point3{.x = center_x + half_width, .y = y, .z = height},
            renderer::Point3{.x = center_x - half_width, .y = y, .z = height},
        },
        std::vector(4U, renderer::Normal3{.y = -1.0F}),
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
    return std::make_shared<const TriangleMesh>(std::move(*mesh));
}

[[nodiscard]] core::Result<FrameSceneHandle> make_energy_scene() {
    auto receiver = horizontal_quad(renderer::Point3{}, 4.0F, 4.0F, false);
    auto emitter = horizontal_quad(renderer::Point3{.z = EnergyEmitterHeight},
                                   EnergyEmitterHalfExtent, EnergyEmitterHalfExtent, true);
    if (!receiver) {
        return std::unexpected(receiver.error());
    }
    if (!emitter) {
        return std::unexpected(emitter.error());
    }
    const auto wavelengths = test_wavelengths();
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
                            .wavelengths = wavelengths,
                            .closure_mixture = require_lambertian_scene_closure(
                                constant_spectrum(ReceiverReflectance)),
                            .emitted_radiance = {},
                        },
                },
                SceneMaterial{
                    .id = {.value = 22U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = wavelengths,
                            .closure_mixture = require_lambertian_scene_closure({}),
                            .emitted_radiance = neutral_emission(),
                        },
                },
            },
        .instances =
            {
                SceneInstance{
                    .id = ReceiverInstance,
                    .parent = std::nullopt,
                    .object = {.value = 1U},
                    .geometry = {.value = 11U},
                    .material = {.value = 21U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
                SceneInstance{
                    .id = EnergyEmitterInstance,
                    .parent = std::nullopt,
                    .object = {.value = 2U},
                    .geometry = {.value = 12U},
                    .material = {.value = 22U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
            },
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = wavelengths,
                .radiance = {},
            },
    });
}

[[nodiscard]] core::Result<FrameSceneHandle> make_image_scene() {
    auto floor = horizontal_quad(renderer::Point3{}, 4.0F, 4.0F, false);
    auto wall = back_wall_quad(0.0F, 3.0F, 4.0F, 4.0F);
    if (!floor) {
        return std::unexpected(floor.error());
    }
    if (!wall) {
        return std::unexpected(wall.error());
    }

    constexpr auto half_extents = std::array{0.15F, 0.30F, 0.55F, 0.85F};
    constexpr auto positions_x = std::array{-2.35F, -0.85F, 0.75F, 2.35F};
    constexpr auto equal_power_scale = 2.0F;
    const auto wavelengths = test_wavelengths();
    auto description = FrameSceneDescription{};
    description.objects = {SceneObject{.id = {.value = 1U}}, SceneObject{.id = {.value = 2U}}};
    description.geometries = {
        SceneGeometry{.id = {.value = 11U}, .mesh = std::move(*floor)},
        SceneGeometry{.id = {.value = 12U}, .mesh = std::move(*wall)},
    };
    description.materials = {
        SceneMaterial{
            .id = {.value = 21U},
            .spectral =
                SceneSpectralMaterial{
                    .wavelengths = wavelengths,
                    .closure_mixture = require_lambertian_scene_closure(constant_spectrum(0.68F)),
                    .emitted_radiance = {},
                },
        },
    };
    description.instances = {
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
            .material = {.value = 21U},
            .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
        },
    };

    for (auto index = std::size_t{}; index < half_extents.size(); ++index) {
        const auto half_extent = half_extents[index];
        auto emitter =
            horizontal_quad(renderer::Point3{.x = positions_x[index], .y = 0.65F, .z = 2.65F},
                            half_extent, half_extent, true);
        if (!emitter) {
            return std::unexpected(emitter.error());
        }
        const auto wide_index = static_cast<std::uint32_t>(index);
        const auto object = renderer::ObjectId{.value = 100U + wide_index};
        const auto geometry = renderer::GeometryId{.value = 200U + wide_index};
        const auto material = renderer::MaterialId{.value = 300U + wide_index};
        const auto instance = renderer::InstanceId{.value = 400U + wide_index};
        const auto area = 4.0F * half_extent * half_extent;
        description.objects.push_back(SceneObject{.id = object});
        description.geometries.push_back(
            SceneGeometry{.id = geometry, .mesh = std::move(*emitter)});
        description.materials.push_back(SceneMaterial{
            .id = material,
            .spectral =
                SceneSpectralMaterial{
                    .wavelengths = wavelengths,
                    .closure_mixture = require_lambertian_scene_closure({}),
                    .emitted_radiance = neutral_emission(equal_power_scale / area),
                },
        });
        description.instances.push_back(SceneInstance{
            .id = instance,
            .parent = std::nullopt,
            .object = object,
            .geometry = geometry,
            .material = material,
            .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
        });
    }
    description.spectral_environment = SceneSpectralEnvironment{
        .wavelengths = wavelengths,
        .radiance = {},
    };
    return FrameScene::create(std::move(description));
}

class CountingAccelBackend final : public AccelBackend {
  public:
    CountingAccelBackend(std::unique_ptr<AccelBackend> backend,
                         std::vector<renderer::InstanceId> emitter_instances)
        : AccelBackend{backend->frame_scene()}, backend_{std::move(backend)},
          emitter_instances_{std::move(emitter_instances)} {}

    [[nodiscard]] AccelBackendKind kind() const noexcept override {
        return backend_->kind();
    }

    [[nodiscard]] core::Result<std::optional<AccelHit>>
    closest_hit(const renderer::Ray& ray) const override {
        ++closest_hit_queries_;
        auto hit = backend_->closest_hit(ray);
        if (hit && *hit &&
            std::ranges::find(emitter_instances_, (**hit).identifiers.instance) !=
                emitter_instances_.end()) {
            ++emitter_hits_;
        }
        return hit;
    }

    [[nodiscard]] core::Result<bool> occluded(const renderer::Ray& ray) const override {
        ++occlusion_queries_;
        auto result = backend_->occluded(ray);
        if (result && *result) {
            ++blocked_queries_;
        }
        return result;
    }

    [[nodiscard]] core::Status rebuild(FrameSceneHandle scene) override {
        const auto retained = scene;
        auto status = backend_->rebuild(std::move(scene));
        if (status) {
            publish_rebuild(retained);
        }
        return status;
    }

    [[nodiscard]] core::Status refit(FrameSceneHandle scene) override {
        const auto retained = scene;
        auto status = backend_->refit(std::move(scene));
        if (status) {
            publish_refit(retained);
        }
        return status;
    }

    [[nodiscard]] AccelBuildStatistics build_statistics() const noexcept override {
        return backend_->build_statistics();
    }

    [[nodiscard]] std::uint64_t closest_hit_queries() const noexcept {
        return closest_hit_queries_;
    }
    [[nodiscard]] std::uint64_t emitter_hits() const noexcept {
        return emitter_hits_;
    }
    [[nodiscard]] std::uint64_t occlusion_queries() const noexcept {
        return occlusion_queries_;
    }
    [[nodiscard]] std::uint64_t blocked_queries() const noexcept {
        return blocked_queries_;
    }

  private:
    std::unique_ptr<AccelBackend> backend_;
    std::vector<renderer::InstanceId> emitter_instances_;
    mutable std::uint64_t closest_hit_queries_{};
    mutable std::uint64_t emitter_hits_{};
    mutable std::uint64_t occlusion_queries_{};
    mutable std::uint64_t blocked_queries_{};
};

struct EnergyEstimate final {
    std::array<long double, renderer::TransportSpectrumSampleCount> sums{};
    std::uint64_t closest_hit_queries{};
    std::uint64_t emitter_hits{};
    std::uint64_t occlusion_queries{};
    std::uint64_t blocked_queries{};
    std::uint64_t escaped_paths{};
    std::uint64_t emitter_paths{};

    [[nodiscard]] bool operator==(const EnergyEstimate&) const noexcept = default;
};

[[nodiscard]] core::Result<EnergyEstimate> estimate_center(const FrameSceneHandle& scene,
                                                           const renderer::MisHeuristic heuristic,
                                                           const std::uint32_t sample_count,
                                                           const std::uint64_t seed) {
    auto embree = create_embree_accel_backend(scene);
    if (!embree) {
        return std::unexpected(embree.error());
    }
    auto acceleration = CountingAccelBackend{std::move(*embree), {EnergyEmitterInstance}};
    const auto sampler = renderer::LightSampler::create_uniform(1U);
    if (!sampler) {
        return std::unexpected(sampler.error());
    }
    const auto ray =
        renderer::Ray::create(renderer::Point3{.z = 1.0F}, renderer::Vector3{.z = -1.0F}, 0.0F,
                              std::numeric_limits<renderer::TransportScalar>::infinity(), PathTime,
                              renderer::AllRayVisibility, renderer::VacuumMedium);
    const auto initial_state =
        renderer::PathState::create_initial(test_wavelengths(), renderer::VacuumMedium);
    if (!ray) {
        return std::unexpected(ray.error());
    }
    if (!initial_state) {
        return std::unexpected(initial_state.error());
    }

    auto estimate = EnergyEstimate{};
    for (auto sample_index = std::uint64_t{}; sample_index < sample_count; ++sample_index) {
        const auto stream = renderer::IndependentSampler{seed}.make_stream(0U, 0U, sample_index);
        const auto traced = trace_scene_mis(*ray, *initial_state, stream, acceleration, *sampler,
                                            heuristic, renderer::PathDepthLimits{.diffuse = 1U},
                                            renderer::RussianRoulettePolicy::disabled());
        if (!traced) {
            return std::unexpected(traced.error());
        }
        for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
            estimate.sums[lane] +=
                static_cast<long double>(traced->state.accumulated_radiance()[lane]);
        }
        if (traced->termination == renderer::BsdfOnlyPathTermination::escaped_environment) {
            ++estimate.escaped_paths;
        } else if (traced->termination == renderer::BsdfOnlyPathTermination::depth_limit) {
            ++estimate.emitter_paths;
        } else {
            return std::unexpected(
                test_error("The one-bounce MIS energy path terminated unexpectedly."));
        }
    }
    estimate.closest_hit_queries = acceleration.closest_hit_queries();
    estimate.emitter_hits = acceleration.emitter_hits();
    estimate.occlusion_queries = acceleration.occlusion_queries();
    estimate.blocked_queries = acceleration.blocked_queries();
    return estimate;
}

[[nodiscard]] long double rectangle_cosine_integral() {
    // Midpoint quadrature over one symmetric quadrant. This integrates
    // cos(theta_receiver) cos(theta_light) / r^2 in double-reference space.
    constexpr auto subdivisions = std::uint32_t{1'024U};
    constexpr auto height = static_cast<long double>(EnergyEmitterHeight);
    constexpr auto half_extent = static_cast<long double>(EnergyEmitterHalfExtent);
    constexpr auto step = half_extent / static_cast<long double>(subdivisions);
    auto sum = static_cast<long double>(0);
    for (auto y_index = std::uint32_t{}; y_index < subdivisions; ++y_index) {
        const auto y = (static_cast<long double>(y_index) + 0.5L) * step;
        for (auto x_index = std::uint32_t{}; x_index < subdivisions; ++x_index) {
            const auto x = (static_cast<long double>(x_index) + 0.5L) * step;
            const auto squared_distance = height * height + x * x + y * y;
            sum += height * height / (squared_distance * squared_distance);
        }
    }
    return 4.0L * sum * step * step;
}

[[nodiscard]] std::array<long double, renderer::TransportSpectrumSampleCount>
expected_center_radiance() {
    const auto geometry = rectangle_cosine_integral();
    const auto emission = neutral_emission();
    auto expected = std::array<long double, renderer::TransportSpectrumSampleCount>{};
    for (auto lane = std::size_t{}; lane < expected.size(); ++lane) {
        expected[lane] = static_cast<long double>(ReceiverReflectance) *
                         static_cast<long double>(emission[lane]) * geometry /
                         std::numbers::pi_v<long double>;
    }
    return expected;
}

[[nodiscard]] long double mean_lane(const EnergyEstimate& estimate, const std::size_t lane,
                                    const std::uint32_t sample_count) {
    return estimate.sums[lane] / static_cast<long double>(sample_count);
}

[[nodiscard]] core::Result<renderer::OrthonormalFrame> image_camera_frame() {
    const auto origin = renderer::Point3{.y = -7.0F, .z = 2.0F};
    const auto target = renderer::Point3{.y = 0.8F, .z = 1.0F};
    const auto forward_seed = target - origin;
    const auto forward_length = std::sqrt(renderer::length_squared(forward_seed));
    if (!std::isfinite(forward_length) || !(forward_length > 0.0F)) {
        return std::unexpected(test_error("The Veach MIS camera direction is invalid."));
    }
    const auto forward = forward_seed / forward_length;
    return renderer::OrthonormalFrame::from_normal_and_tangent(
        renderer::Normal3{.x = -forward.x, .y = -forward.y, .z = -forward.z},
        renderer::Vector3{.x = 1.0F});
}

struct RenderedImage final {
    renderer::Film film;
    std::uint64_t closest_hit_queries{};
    std::uint64_t emitter_hits{};
    std::uint64_t occlusion_queries{};
};

[[nodiscard]] core::Result<RenderedImage> render_image(const FrameSceneHandle& scene) {
    auto embree = create_embree_accel_backend(scene);
    if (!embree) {
        return std::unexpected(embree.error());
    }
    auto acceleration = CountingAccelBackend{
        std::move(*embree),
        std::vector<renderer::InstanceId>{scene->mesh_area_light_instance_ids().begin(),
                                          scene->mesh_area_light_instance_ids().end()},
    };
    const auto sampler = renderer::LightSampler::create_uniform(scene->mesh_area_lights().size());
    auto film = renderer::Film::create(ImageExtent);
    const auto frame = image_camera_frame();
    if (!sampler) {
        return std::unexpected(sampler.error());
    }
    if (!film) {
        return std::unexpected(film.error());
    }
    if (!frame) {
        return std::unexpected(frame.error());
    }
    const auto camera = renderer::PinholeCamera::create(
        renderer::Point3{.y = -7.0F, .z = 2.0F}, *frame, ImageExtent, 0.82F, 0.0F,
        std::numeric_limits<renderer::TransportScalar>::infinity(), renderer::AllRayVisibility,
        renderer::VacuumMedium);
    if (!camera) {
        return std::unexpected(camera.error());
    }
    const auto wavelengths = test_wavelengths();

    for (auto y = std::uint32_t{}; y < ImageExtent.height; ++y) {
        for (auto x = std::uint32_t{}; x < ImageExtent.width; ++x) {
            for (auto sample_index = std::uint64_t{}; sample_index < ImageSamplesPerPixel;
                 ++sample_index) {
                const auto index = renderer::PixelSampleIndex{
                    .pixel_x = x,
                    .pixel_y = y,
                    .sample_index = sample_index,
                    .seed = EvaluationSeed,
                };
                const auto ray = camera->generate_primary_ray(
                    index, renderer::PixelJitterMode::uniform, PathTime);
                const auto state =
                    renderer::PathState::create_initial(wavelengths, renderer::VacuumMedium);
                if (!ray) {
                    return std::unexpected(ray.error());
                }
                if (!state) {
                    return std::unexpected(state.error());
                }
                const auto stream =
                    renderer::IndependentSampler{EvaluationSeed}.make_stream(x, y, sample_index);
                const auto traced = trace_scene_mis(*ray, *state, stream, acceleration, *sampler,
                                                    renderer::MisHeuristic::power,
                                                    renderer::PathDepthLimits{.diffuse = 2U},
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
                const auto accumulated = film->add_sample(x, y, *rgb, 1.0F);
                if (!accumulated) {
                    return std::unexpected(accumulated.error());
                }
            }
        }
    }
    return RenderedImage{
        .film = std::move(*film),
        .closest_hit_queries = acceleration.closest_hit_queries(),
        .emitter_hits = acceleration.emitter_hits(),
        .occlusion_queries = acceleration.occlusion_queries(),
    };
}

[[nodiscard]] std::optional<std::filesystem::path> checksum_output_path() {
#if defined(_WIN32)
    auto* value = static_cast<char*>(nullptr);
    auto value_size = std::size_t{};
    if (_dupenv_s(&value, &value_size, "BLACKFRAME_PNG_CHECKSUM_OUTPUT") != 0 || value == nullptr) {
        return std::nullopt;
    }
    const auto path = value_size > 1U ? std::optional{std::filesystem::path{value}} : std::nullopt;
    std::free(value);
    return path;
#else
    const auto* const value = std::getenv("BLACKFRAME_PNG_CHECKSUM_OUTPUT");
    if (value == nullptr || *value == '\0') {
        return std::nullopt;
    }
    return std::filesystem::path{value};
#endif
}

[[nodiscard]] std::filesystem::path image_output_path() {
    if (const auto checksum_output = checksum_output_path(); checksum_output) {
        return *checksum_output;
    }
    return std::filesystem::path{BLACKFRAME_EMBREE_MIS_TEST_OUTPUT_DIR} /
           "veach-mis-power-128spp.png";
}

TEST(VeachMisIntegrationTest, CombinesNeeAndBsdfHitsWithoutDoublingEnergy) {
    const auto scene = make_energy_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    ASSERT_EQ((*scene)->punctual_lights().size(), 0U);
    ASSERT_EQ((*scene)->mesh_area_lights().size(), 1U);
    ASSERT_EQ((*scene)->mesh_area_light_instance_ids().size(), 1U);
    EXPECT_EQ((*scene)->mesh_area_light_instance_ids().front(), EnergyEmitterInstance);

    const auto balance =
        estimate_center(*scene, renderer::MisHeuristic::balance, EnergySampleCount, EvaluationSeed);
    const auto power =
        estimate_center(*scene, renderer::MisHeuristic::power, EnergySampleCount, EvaluationSeed);
    ASSERT_TRUE(balance.has_value()) << balance.error().message;
    ASSERT_TRUE(power.has_value()) << power.error().message;
    const auto expected = expected_center_radiance();

    for (auto lane = std::size_t{}; lane < expected.size(); ++lane) {
        if (expected[lane] == 0.0L) {
            EXPECT_EQ(mean_lane(*balance, lane, EnergySampleCount), 0.0L);
            EXPECT_EQ(mean_lane(*power, lane, EnergySampleCount), 0.0L);
            continue;
        }
        const auto balance_mean = mean_lane(*balance, lane, EnergySampleCount);
        const auto power_mean = mean_lane(*power, lane, EnergySampleCount);
        const auto tolerance = expected[lane] * 0.025L;
        EXPECT_NEAR(static_cast<double>(balance_mean), static_cast<double>(expected[lane]),
                    static_cast<double>(tolerance))
            << "spectral lane " << lane;
        EXPECT_NEAR(static_cast<double>(power_mean), static_cast<double>(expected[lane]),
                    static_cast<double>(tolerance))
            << "spectral lane " << lane;
        EXPECT_NEAR(static_cast<double>(balance_mean), static_cast<double>(power_mean),
                    static_cast<double>(expected[lane] * 0.02L))
            << "spectral lane " << lane;
        EXPECT_LT(std::abs(power_mean - expected[lane]),
                  std::abs(power_mean - 2.0L * expected[lane]) * 0.10L)
            << "spectral lane " << lane;
    }

    for (const auto* estimate : std::array{&*balance, &*power}) {
        EXPECT_EQ(estimate->closest_hit_queries,
                  static_cast<std::uint64_t>(EnergySampleCount) * 2U);
        EXPECT_EQ(estimate->occlusion_queries, EnergySampleCount);
        EXPECT_EQ(estimate->blocked_queries, 0U);
        EXPECT_EQ(estimate->emitter_hits, estimate->emitter_paths);
        EXPECT_GT(estimate->emitter_paths, EnergySampleCount / 20U);
        EXPECT_GT(estimate->escaped_paths, EnergySampleCount / 2U);
        EXPECT_EQ(estimate->escaped_paths + estimate->emitter_paths, EnergySampleCount);
    }

    testing::Test::RecordProperty("reference_lane0",
                                  std::to_string(static_cast<double>(expected[0])));
    testing::Test::RecordProperty(
        "balance_lane0",
        std::to_string(static_cast<double>(mean_lane(*balance, 0U, EnergySampleCount))));
    testing::Test::RecordProperty("power_lane0", std::to_string(static_cast<double>(
                                                     mean_lane(*power, 0U, EnergySampleCount))));
}

TEST(VeachMisIntegrationTest, ReplaysExactlyAndRejectsIncompleteDispatchState) {
    const auto scene = make_energy_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto first =
        estimate_center(*scene, renderer::MisHeuristic::power, ReplaySampleCount, EvaluationSeed);
    const auto replay =
        estimate_center(*scene, renderer::MisHeuristic::power, ReplaySampleCount, EvaluationSeed);
    ASSERT_TRUE(first.has_value()) << first.error().message;
    ASSERT_TRUE(replay.has_value()) << replay.error().message;
    EXPECT_EQ(*first, *replay);

    auto embree = create_embree_accel_backend(*scene);
    ASSERT_TRUE(embree.has_value()) << embree.error().message;
    const auto ray =
        renderer::Ray::create(renderer::Point3{.z = 1.0F}, renderer::Vector3{.z = -1.0F}, 0.0F,
                              std::numeric_limits<renderer::TransportScalar>::infinity(), PathTime,
                              renderer::AllRayVisibility, renderer::VacuumMedium);
    const auto initial =
        renderer::PathState::create_initial(test_wavelengths(), renderer::VacuumMedium);
    const auto stream = renderer::IndependentSampler{EvaluationSeed}.make_stream(0U, 0U, 0U);
    ASSERT_TRUE(ray.has_value()) << ray.error().message;
    ASSERT_TRUE(initial.has_value()) << initial.error().message;

    const auto wrong_sampler = renderer::LightSampler::create_uniform(2U);
    ASSERT_TRUE(wrong_sampler.has_value()) << wrong_sampler.error().message;
    const auto wrong_registry = trace_scene_mis(
        *ray, *initial, stream, **embree, *wrong_sampler, renderer::MisHeuristic::power,
        renderer::PathDepthLimits{.diffuse = 1U}, renderer::RussianRoulettePolicy::disabled());
    ASSERT_FALSE(wrong_registry.has_value());
    EXPECT_EQ(wrong_registry.error().code, core::StatusCode::incompatible);
    EXPECT_FALSE(wrong_registry.error().message.empty());

    const auto sampler = renderer::LightSampler::create_uniform(1U);
    ASSERT_TRUE(sampler.has_value()) << sampler.error().message;
    const auto invalid_heuristic = trace_scene_mis(
        *ray, *initial, stream, **embree, *sampler, static_cast<renderer::MisHeuristic>(255U),
        renderer::PathDepthLimits{.diffuse = 1U}, renderer::RussianRoulettePolicy::disabled());
    ASSERT_FALSE(invalid_heuristic.has_value());
    EXPECT_EQ(invalid_heuristic.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(invalid_heuristic.error().message.empty());

    const auto resumed = renderer::PathState::create(
        constant_spectrum(1.0F), renderer::TransportSpectrum{},
        renderer::PathDepthCounters{.diffuse = 1U}, 1.0F, test_wavelengths(),
        renderer::PathDeltaFlags::any_non_delta_bounces, renderer::VacuumMedium);
    ASSERT_TRUE(resumed.has_value()) << resumed.error().message;
    const auto missing_previous_pdf = trace_scene_mis(
        *ray, *resumed, stream, **embree, *sampler, renderer::MisHeuristic::balance,
        renderer::PathDepthLimits{.diffuse = 2U}, renderer::RussianRoulettePolicy::disabled());
    ASSERT_FALSE(missing_previous_pdf.has_value());
    EXPECT_EQ(missing_previous_pdf.error().code, core::StatusCode::incompatible);
    EXPECT_FALSE(missing_previous_pdf.error().message.empty());
}

TEST(VeachMisParityTest, MatchesScalarReferenceThroughCpuWavefront) {
    constexpr auto extent = renderer::RenderExtent{.width = 8U, .height = 8U};
    constexpr auto samples_per_pixel = std::uint32_t{4U};

    const auto scene = make_image_scene();
    const auto frame = image_camera_frame();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    ASSERT_TRUE(frame.has_value()) << frame.error().message;
    ASSERT_EQ((*scene)->mesh_area_lights().size(), 4U);

    const auto camera = renderer::PinholeCamera::create(
        renderer::Point3{.y = -7.0F, .z = 2.0F}, *frame, extent, 0.82F, 0.0F,
        std::numeric_limits<renderer::TransportScalar>::infinity(), renderer::AllRayVisibility,
        renderer::VacuumMedium);
    ASSERT_TRUE(camera.has_value()) << camera.error().message;

    const auto inputs = scalar_wavefront_parity_test::make_inputs(
        extent, samples_per_pixel, EvaluationSeed, test_wavelengths(),
        [&camera](const renderer::PixelSampleIndex& index, const renderer::SampleStream&) {
            return scalar_wavefront_parity_test::camera_primary_ray(
                *camera, index, renderer::PixelJitterMode::uniform, PathTime);
        });
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    const auto configuration = scalar_wavefront_parity_test::Configuration{
        .scene_name = "VeachMIS",
        .extent = extent,
        .samples_per_pixel = samples_per_pixel,
        .seed = EvaluationSeed,
        .heuristic = renderer::MisHeuristic::power,
        .depth_limits = renderer::PathDepthLimits{.diffuse = 2U},
        .roulette_policy = renderer::RussianRoulettePolicy::disabled(),
        .worker_count = 4U,
        .thresholds = scalar_wavefront_parity_test::StrictThresholds,
    };
    const auto parity = scalar_wavefront_parity_test::compare(*scene, *inputs, configuration);
    ASSERT_TRUE(parity.has_value()) << parity.error().message;
    EXPECT_GT(parity->wavefront_report.closure_samples, 0U);
    EXPECT_GT(parity->wavefront_report.light_samples, 0U);
    EXPECT_GT(parity->wavefront_report.shadow_queries, 0U);
    scalar_wavefront_parity_test::record_and_expect(configuration, *parity);
}

TEST(VeachMisImageTest, WritesNeutralStablePowerHeuristicPreview) {
#if BLACKFRAME_HAS_PNG_PREVIEW
    const auto scene = make_image_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    ASSERT_EQ((*scene)->punctual_lights().size(), 0U);
    ASSERT_EQ((*scene)->mesh_area_lights().size(), 4U);
    const auto rendered = render_image(*scene);
    ASSERT_TRUE(rendered.has_value()) << rendered.error().message;
    EXPECT_GT(rendered->closest_hit_queries, static_cast<std::uint64_t>(ImageExtent.width) *
                                                 ImageExtent.height * ImageSamplesPerPixel);
    EXPECT_GT(rendered->occlusion_queries, 0U);
    EXPECT_GT(rendered->emitter_hits, 0U);

    const auto output_path = image_output_path();
    std::error_code cleanup_error;
    std::filesystem::remove(output_path, cleanup_error);
    ASSERT_FALSE(cleanup_error) << "Cannot replace '" << output_path.string()
                                << "': " << cleanup_error.message();
    const auto write_status = renderer::write_png_preview(rendered->film, output_path);
    ASSERT_TRUE(write_status.has_value()) << write_status.error().message;
    ASSERT_TRUE(std::filesystem::is_regular_file(output_path));

    auto width = int{};
    auto height = int{};
    auto components = int{};
    auto* const decoded = stbi_load(output_path.string().c_str(), &width, &height, &components, 3);
    ASSERT_NE(decoded, nullptr) << "Cannot decode '" << output_path.string()
                                << "': " << stbi_failure_reason();
    EXPECT_EQ(width, static_cast<int>(ImageExtent.width));
    EXPECT_EQ(height, static_cast<int>(ImageExtent.height));
    EXPECT_EQ(components, 3);

    auto channel_sums = std::array<std::uint64_t, 3>{};
    auto minimum_channel = std::numeric_limits<std::uint8_t>::max();
    auto maximum_channel = std::uint8_t{};
    auto non_black_pixels = std::uint32_t{};
    const auto pixel_count = static_cast<std::size_t>(ImageExtent.width) * ImageExtent.height;
    for (auto pixel = std::size_t{}; pixel < pixel_count; ++pixel) {
        auto non_black = false;
        for (auto channel = std::size_t{}; channel < channel_sums.size(); ++channel) {
            const auto value = decoded[pixel * 3U + channel];
            channel_sums[channel] += value;
            minimum_channel = std::min(minimum_channel, value);
            maximum_channel = std::max(maximum_channel, value);
            non_black = non_black || value != 0U;
        }
        non_black_pixels += non_black ? 1U : 0U;
    }
    stbi_image_free(decoded);

    EXPECT_GT(non_black_pixels, ImageExtent.width * ImageExtent.height / 3U);
    EXPECT_LT(minimum_channel, maximum_channel);
    const auto minimum_sum = *std::ranges::min_element(channel_sums);
    const auto maximum_sum = *std::ranges::max_element(channel_sums);
    EXPECT_GT(minimum_sum, 0U);
    EXPECT_LT(maximum_sum - minimum_sum, maximum_sum / 25U);

    testing::Test::RecordProperty("closest_hit_queries",
                                  std::to_string(rendered->closest_hit_queries));
    testing::Test::RecordProperty("shadow_queries", std::to_string(rendered->occlusion_queries));
    testing::Test::RecordProperty("bsdf_emitter_hits", std::to_string(rendered->emitter_hits));
    testing::Test::RecordProperty("red_sum", std::to_string(channel_sums[0]));
    testing::Test::RecordProperty("green_sum", std::to_string(channel_sums[1]));
    testing::Test::RecordProperty("blue_sum", std::to_string(channel_sums[2]));
#else
    GTEST_SKIP() << "PNG preview support is disabled explicitly.";
#endif
}

} // namespace
} // namespace blackframe::engine
