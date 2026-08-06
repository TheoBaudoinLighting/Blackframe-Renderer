#include "ScalarWavefrontImageParity.hpp"

#include <Blackframe/Backends/CPU/Embree/AccelBackend.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/SceneNeePathLoop.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Renderer/Cie1931Sensor.hpp>
#include <Blackframe/Renderer/DisplayPsnr.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/LinearMetrics.hpp>
#include <Blackframe/Renderer/PathDepthLimits.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <Blackframe/Renderer/PngWriter.hpp>
#include <Blackframe/Renderer/RussianRoulette.hpp>
#include <Blackframe/Renderer/SampleDimensionMap.hpp>
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
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#if BLACKFRAME_HAS_PNG_PREVIEW
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#endif

namespace blackframe::engine {
namespace {

inline constexpr auto ImageExtent = renderer::RenderExtent{.width = 64U, .height = 64U};
inline constexpr auto ConvergenceExtent = renderer::RenderExtent{.width = 32U, .height = 32U};
inline constexpr auto ReplayExtent = renderer::RenderExtent{.width = 8U, .height = 8U};
inline constexpr auto ImageSamplesPerPixel = std::uint32_t{64U};
inline constexpr auto ReceiverReflectance = renderer::TransportScalar{0.72F};
inline constexpr auto ReceiverHalfExtent = renderer::TransportScalar{2.0F};
inline constexpr auto CameraSpan = renderer::TransportScalar{3.0F};
inline constexpr auto CameraHeight = renderer::TransportScalar{3.0F};
inline constexpr auto WallHalfWidth = renderer::TransportScalar{0.75F};
inline constexpr auto WallHeight = renderer::TransportScalar{1.45F};
inline constexpr auto PathTime = renderer::TransportScalar{0.375F};
inline constexpr auto EvaluationSeed = std::uint64_t{0xA4093822299F31D0ULL};

[[nodiscard]] core::Error image_error(std::string message) {
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

[[nodiscard]] core::Result<std::shared_ptr<const TriangleMesh>> make_receiver_mesh() {
    auto mesh = TriangleMesh::create(
        {
            renderer::Point3{.x = -ReceiverHalfExtent, .y = -ReceiverHalfExtent},
            renderer::Point3{.x = ReceiverHalfExtent, .y = -ReceiverHalfExtent},
            renderer::Point3{.x = ReceiverHalfExtent, .y = ReceiverHalfExtent},
            renderer::Point3{.x = -ReceiverHalfExtent, .y = ReceiverHalfExtent},
        },
        std::vector(4U, renderer::Normal3{.z = 1.0F}),
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

[[nodiscard]] core::Result<std::shared_ptr<const TriangleMesh>> make_blocker_mesh() {
    auto mesh = TriangleMesh::create(
        {
            renderer::Point3{.y = -WallHalfWidth},
            renderer::Point3{.y = WallHalfWidth},
            renderer::Point3{.y = WallHalfWidth, .z = WallHeight},
            renderer::Point3{.y = -WallHalfWidth, .z = WallHeight},
        },
        std::vector(4U, renderer::Normal3{.x = 1.0F}),
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

[[nodiscard]] ScenePunctualLight point_light(const renderer::Point3 position,
                                             const float intensity) {
    return ScenePunctualLight{ScenePointLight{
        .position = position,
        .absolute_position_error = {},
        .spectral_radiant_intensity = constant_spectrum(intensity),
    }};
}

[[nodiscard]] core::Result<FrameSceneHandle> make_point_light_scene() {
    auto receiver = make_receiver_mesh();
    auto blocker = make_blocker_mesh();
    const auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.25F);
    if (!receiver) {
        return std::unexpected(receiver.error());
    }
    if (!blocker) {
        return std::unexpected(blocker.error());
    }
    if (!wavelengths) {
        return std::unexpected(wavelengths.error());
    }

    return FrameScene::create(FrameSceneDescription{
        .objects =
            {
                SceneObject{.id = {.value = 1U}},
                SceneObject{.id = {.value = 2U}},
            },
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
                            .closure_mixture = require_lambertian_scene_closure(
                                constant_spectrum(ReceiverReflectance)),
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
                    .material = {.value = 21U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                },
            },
        .punctual_lights =
            {
                point_light(renderer::Point3{.x = -1.25F, .y = -1.15F, .z = 2.5F}, 2.0F),
                point_light(renderer::Point3{.x = 1.25F, .y = -1.15F, .z = 2.5F}, 3.0F),
                point_light(renderer::Point3{.x = -1.25F, .y = 1.15F, .z = 2.5F}, 4.0F),
                point_light(renderer::Point3{.x = 1.25F, .y = 1.15F, .z = 2.5F}, 5.0F),
            },
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = *wavelengths,
                .radiance = {},
            },
    });
}

class CountingAccelBackend final : public AccelBackend {
  public:
    explicit CountingAccelBackend(std::unique_ptr<AccelBackend> backend)
        : AccelBackend{backend->frame_scene()}, backend_{std::move(backend)} {}

    [[nodiscard]] AccelBackendKind kind() const noexcept override {
        return backend_->kind();
    }

    [[nodiscard]] core::Result<std::optional<AccelHit>>
    closest_hit(const renderer::Ray& ray) const override {
        ++closest_hit_queries_;
        return backend_->closest_hit(ray);
    }

    [[nodiscard]] core::Result<bool> occluded(const renderer::Ray& ray) const override {
        ++occlusion_queries_;
        const auto result = backend_->occluded(ray);
        if (result) {
            if (*result) {
                ++blocked_queries_;
            } else {
                ++visible_queries_;
            }
        }
        return result;
    }

    [[nodiscard]] core::Status rebuild(FrameSceneHandle scene) override {
        const auto retained_scene = scene;
        auto status = backend_->rebuild(std::move(scene));
        if (status) {
            publish_rebuild(retained_scene);
        }
        return status;
    }

    [[nodiscard]] core::Status refit(FrameSceneHandle scene) override {
        const auto retained_scene = scene;
        auto status = backend_->refit(std::move(scene));
        if (status) {
            publish_refit(retained_scene);
        }
        return status;
    }

    [[nodiscard]] AccelBuildStatistics build_statistics() const noexcept override {
        return backend_->build_statistics();
    }

    [[nodiscard]] std::uint64_t closest_hit_queries() const noexcept {
        return closest_hit_queries_;
    }

    [[nodiscard]] std::uint64_t occlusion_queries() const noexcept {
        return occlusion_queries_;
    }

    [[nodiscard]] std::uint64_t blocked_queries() const noexcept {
        return blocked_queries_;
    }

    [[nodiscard]] std::uint64_t visible_queries() const noexcept {
        return visible_queries_;
    }

  private:
    std::unique_ptr<AccelBackend> backend_;
    mutable std::uint64_t closest_hit_queries_{};
    mutable std::uint64_t occlusion_queries_{};
    mutable std::uint64_t blocked_queries_{};
    mutable std::uint64_t visible_queries_{};
};

struct RenderStatistics final {
    std::uint64_t closest_hit_queries{};
    std::uint64_t occlusion_queries{};
    std::uint64_t blocked_queries{};
    std::uint64_t visible_queries{};
};

struct RenderedImage final {
    renderer::Film film;
    RenderStatistics statistics;
};

[[nodiscard]] renderer::Point3 primary_position(const renderer::RenderExtent extent,
                                                const std::uint32_t pixel_x,
                                                const std::uint32_t pixel_y,
                                                const renderer::SampleStream& stream) noexcept {
    const auto jitter_x = stream.sample_1d(renderer::PrimarySampleDimensionMap.camera_raster_x);
    const auto jitter_y = stream.sample_1d(renderer::PrimarySampleDimensionMap.camera_raster_y);
    return renderer::Point3{
        .x = -CameraSpan / 2.0F + (static_cast<float>(pixel_x) + jitter_x) * CameraSpan /
                                      static_cast<float>(extent.width),
        .y = CameraSpan / 2.0F - (static_cast<float>(pixel_y) + jitter_y) * CameraSpan /
                                     static_cast<float>(extent.height),
    };
}

[[nodiscard]] core::Result<RenderedImage> render_nee(const FrameSceneHandle& scene,
                                                     const renderer::RenderExtent extent,
                                                     const std::uint32_t samples_per_pixel,
                                                     const std::uint64_t seed) {
    auto backend = create_embree_accel_backend(scene);
    if (!backend) {
        return std::unexpected(backend.error());
    }
    auto acceleration = CountingAccelBackend{std::move(*backend)};
    const auto sampler = renderer::LightSampler::create_uniform(scene->punctual_lights().size());
    auto film = renderer::Film::create(extent);
    if (!sampler) {
        return std::unexpected(sampler.error());
    }
    if (!film) {
        return std::unexpected(film.error());
    }
    if (!scene->spectral_environment()) {
        return std::unexpected(image_error("The NEE image scene lost its spectral packet."));
    }
    const auto wavelengths = scene->spectral_environment()->wavelengths;

    for (auto pixel_y = std::uint32_t{}; pixel_y < extent.height; ++pixel_y) {
        for (auto pixel_x = std::uint32_t{}; pixel_x < extent.width; ++pixel_x) {
            for (auto sample_index = std::uint64_t{}; sample_index < samples_per_pixel;
                 ++sample_index) {
                const auto stream =
                    renderer::IndependentSampler{seed}.make_stream(pixel_x, pixel_y, sample_index);
                const auto position = primary_position(extent, pixel_x, pixel_y, stream);
                const auto ray = renderer::Ray::create(
                    position + renderer::Vector3{.z = CameraHeight}, renderer::Vector3{.z = -1.0F},
                    0.0F, std::numeric_limits<renderer::TransportScalar>::infinity(), PathTime,
                    renderer::AllRayVisibility, renderer::VacuumMedium);
                const auto state =
                    renderer::PathState::create_initial(wavelengths, renderer::VacuumMedium);
                if (!ray) {
                    return std::unexpected(ray.error());
                }
                if (!state) {
                    return std::unexpected(state.error());
                }

                const auto traced = trace_scene_nee(*ray, *state, stream, acceleration, *sampler,
                                                    renderer::PathDepthLimits{.diffuse = 1U},
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
                if (!std::isfinite(rgb->red) || !std::isfinite(rgb->green) ||
                    !std::isfinite(rgb->blue)) {
                    return std::unexpected(
                        image_error("The Embree NEE image produced a non-finite pixel sample."));
                }
                const auto accumulated = film->add_sample(pixel_x, pixel_y, *rgb, 1.0F);
                if (!accumulated) {
                    return std::unexpected(accumulated.error());
                }
            }
        }
    }

    return RenderedImage{
        .film = std::move(*film),
        .statistics =
            RenderStatistics{
                .closest_hit_queries = acceleration.closest_hit_queries(),
                .occlusion_queries = acceleration.occlusion_queries(),
                .blocked_queries = acceleration.blocked_queries(),
                .visible_queries = acceleration.visible_queries(),
            },
    };
}

[[nodiscard]] bool wall_occludes(const renderer::ReferencePoint3 source,
                                 const ScenePointLight& light) noexcept {
    const auto light_x = static_cast<double>(light.position.x);
    const auto denominator = light_x - source.x;
    if (denominator == 0.0) {
        return false;
    }
    const auto parameter = -source.x / denominator;
    if (!(parameter > 0.0) || !(parameter < 1.0)) {
        return false;
    }
    const auto intersection_y =
        std::fma(parameter, static_cast<double>(light.position.y) - source.y, source.y);
    const auto intersection_z = parameter * static_cast<double>(light.position.z);
    return intersection_y >= -static_cast<double>(WallHalfWidth) &&
           intersection_y <= static_cast<double>(WallHalfWidth) && intersection_z >= 0.0 &&
           intersection_z <= static_cast<double>(WallHeight);
}

[[nodiscard]] core::Result<renderer::ReferenceSpectrum>
enumerate_point_lights(const FrameScene& scene, const renderer::ReferencePoint3 source) {
    auto radiance = renderer::ReferenceSpectrum{};
    for (const auto& variant : scene.punctual_lights()) {
        const auto* const light = std::get_if<ScenePointLight>(&variant);
        if (light == nullptr) {
            return std::unexpected(
                image_error("The point-light oracle received a non-point registry slot."));
        }
        if (wall_occludes(source, *light)) {
            continue;
        }
        const auto displacement = renderer::ReferenceVector3{
            .x = static_cast<double>(light->position.x) - source.x,
            .y = static_cast<double>(light->position.y) - source.y,
            .z = static_cast<double>(light->position.z),
        };
        const auto squared_distance = renderer::length_squared(displacement);
        const auto distance = std::sqrt(squared_distance);
        if (!std::isfinite(squared_distance) || !(squared_distance > 0.0) ||
            !std::isfinite(distance) || !(distance > 0.0)) {
            return std::unexpected(
                image_error("The point-light oracle found invalid finite-light geometry."));
        }
        const auto cosine = displacement.z / distance;
        const auto geometric_scale = static_cast<double>(ReceiverReflectance) * cosine /
                                     (std::numbers::pi_v<double> * squared_distance);
        for (auto lane = std::size_t{}; lane < renderer::TransportSpectrumSampleCount; ++lane) {
            radiance[lane] +=
                geometric_scale * static_cast<double>(light->spectral_radiant_intensity[lane]);
        }
    }
    return radiance;
}

[[nodiscard]] core::Result<renderer::ReferenceFilm>
render_enumerated_oracle(const FrameSceneHandle& scene, const renderer::RenderExtent extent,
                         const std::uint32_t samples_per_pixel, const std::uint64_t seed) {
    auto film = renderer::ReferenceFilm::create(extent);
    const auto wavelengths =
        renderer::sample_uniform_visible_wavelengths(renderer::ReferenceScalar{0.25});
    if (!film) {
        return std::unexpected(film.error());
    }
    if (!wavelengths) {
        return std::unexpected(wavelengths.error());
    }

    for (auto pixel_y = std::uint32_t{}; pixel_y < extent.height; ++pixel_y) {
        for (auto pixel_x = std::uint32_t{}; pixel_x < extent.width; ++pixel_x) {
            for (auto sample_index = std::uint64_t{}; sample_index < samples_per_pixel;
                 ++sample_index) {
                const auto stream =
                    renderer::IndependentSampler{seed}.make_stream(pixel_x, pixel_y, sample_index);
                const auto transport_position = primary_position(extent, pixel_x, pixel_y, stream);
                const auto source = renderer::ReferencePoint3{
                    .x = static_cast<double>(transport_position.x),
                    .y = static_cast<double>(transport_position.y),
                };
                const auto spectrum = enumerate_point_lights(*scene, source);
                if (!spectrum) {
                    return std::unexpected(spectrum.error());
                }
                const auto xyz = renderer::cie_1931_spectrum_to_xyz(*spectrum, *wavelengths);
                if (!xyz) {
                    return std::unexpected(xyz.error());
                }
                const auto rgb = renderer::xyz_to_linear_rgb(*xyz);
                if (!rgb) {
                    return std::unexpected(rgb.error());
                }
                const auto accumulated = film->add_sample(pixel_x, pixel_y, *rgb, 1.0);
                if (!accumulated) {
                    return std::unexpected(accumulated.error());
                }
            }
        }
    }
    return std::move(*film);
}

[[nodiscard]] double median(std::array<double, 8> values) {
    std::ranges::sort(values);
    return (values[3] + values[4]) / 2.0;
}

void expect_query_counts(const RenderedImage& rendered, const renderer::RenderExtent extent,
                         const std::uint32_t samples_per_pixel) {
    const auto sample_count =
        static_cast<std::uint64_t>(extent.width) * extent.height * samples_per_pixel;
    EXPECT_EQ(rendered.statistics.occlusion_queries, sample_count);
    EXPECT_EQ(rendered.statistics.closest_hit_queries, 2U * sample_count);
    EXPECT_EQ(rendered.statistics.blocked_queries + rendered.statistics.visible_queries,
              sample_count);
    EXPECT_GT(rendered.statistics.blocked_queries, 0U);
    EXPECT_GT(rendered.statistics.visible_queries, 0U);
}

void expect_replay(const renderer::Film& first, const renderer::Film& second) {
    ASSERT_EQ(first.extent().width, second.extent().width);
    ASSERT_EQ(first.extent().height, second.extent().height);
    for (auto y = std::uint32_t{}; y < first.extent().height; ++y) {
        for (auto x = std::uint32_t{}; x < first.extent().width; ++x) {
            const auto first_pixel = first.pixel(x, y);
            const auto second_pixel = second.pixel(x, y);
            ASSERT_TRUE(first_pixel.has_value()) << x << ", " << y;
            ASSERT_TRUE(second_pixel.has_value()) << x << ", " << y;
            EXPECT_EQ(*first_pixel, *second_pixel) << x << ", " << y;
        }
    }
}

[[nodiscard]] std::optional<std::filesystem::path> checksum_output_path() {
#if defined(_WIN32)
    auto* value = static_cast<char*>(nullptr);
    auto value_size = std::size_t{};
    if (_dupenv_s(&value, &value_size, "BLACKFRAME_PNG_CHECKSUM_OUTPUT") != 0 || value == nullptr) {
        return std::nullopt;
    }
    const auto path = value_size > 1 ? std::optional{std::filesystem::path{value}} : std::nullopt;
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
    return std::filesystem::path{BLACKFRAME_EMBREE_NEE_TEST_OUTPUT_DIR} /
           "point-light-nee-64spp.png";
}

TEST(NextEventEstimationIntegrationTest, ReplaysAndConvergesAgainstEnumeratedPointLights) {
    const auto scene = make_point_light_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    constexpr auto seeds = std::array{
        EvaluationSeed,
        std::uint64_t{0x082EFA98EC4E6C89ULL},
        std::uint64_t{0x452821E638D01377ULL},
        std::uint64_t{0xBE5466CF34E90C6CULL},
        std::uint64_t{0xC0AC29B7C97C50DDULL},
        std::uint64_t{0x3F84D5B5B5470917ULL},
        std::uint64_t{0x9216D5D98979FB1BULL},
        std::uint64_t{0xD1310BA698DFB5ACULL},
    };
    auto mse_1 = std::array<double, seeds.size()>{};
    auto mse_4 = std::array<double, seeds.size()>{};
    auto mse_16 = std::array<double, seeds.size()>{};
    auto psnr_1 = std::array<double, seeds.size()>{};
    auto psnr_4 = std::array<double, seeds.size()>{};
    auto psnr_16 = std::array<double, seeds.size()>{};

    for (auto index = std::size_t{}; index < seeds.size(); ++index) {
        const auto evaluated_1 = render_nee(*scene, ConvergenceExtent, 1U, seeds[index]);
        const auto evaluated_4 = render_nee(*scene, ConvergenceExtent, 4U, seeds[index]);
        const auto evaluated_16 = render_nee(*scene, ConvergenceExtent, 16U, seeds[index]);
        const auto oracle_1 = render_enumerated_oracle(*scene, ConvergenceExtent, 1U, seeds[index]);
        const auto oracle_4 = render_enumerated_oracle(*scene, ConvergenceExtent, 4U, seeds[index]);
        const auto oracle_16 =
            render_enumerated_oracle(*scene, ConvergenceExtent, 16U, seeds[index]);
        ASSERT_TRUE(evaluated_1.has_value()) << evaluated_1.error().message;
        ASSERT_TRUE(evaluated_4.has_value()) << evaluated_4.error().message;
        ASSERT_TRUE(evaluated_16.has_value()) << evaluated_16.error().message;
        ASSERT_TRUE(oracle_1.has_value()) << oracle_1.error().message;
        ASSERT_TRUE(oracle_4.has_value()) << oracle_4.error().message;
        ASSERT_TRUE(oracle_16.has_value()) << oracle_16.error().message;
        expect_query_counts(*evaluated_1, ConvergenceExtent, 1U);
        expect_query_counts(*evaluated_4, ConvergenceExtent, 4U);
        expect_query_counts(*evaluated_16, ConvergenceExtent, 16U);

        const auto metrics_1 = renderer::compute_linear_metrics(evaluated_1->film, *oracle_1);
        const auto metrics_4 = renderer::compute_linear_metrics(evaluated_4->film, *oracle_4);
        const auto metrics_16 = renderer::compute_linear_metrics(evaluated_16->film, *oracle_16);
        const auto display_1 = renderer::compute_display_psnr(evaluated_1->film, *oracle_1);
        const auto display_4 = renderer::compute_display_psnr(evaluated_4->film, *oracle_4);
        const auto display_16 = renderer::compute_display_psnr(evaluated_16->film, *oracle_16);
        ASSERT_TRUE(metrics_1.has_value()) << metrics_1.error().message;
        ASSERT_TRUE(metrics_4.has_value()) << metrics_4.error().message;
        ASSERT_TRUE(metrics_16.has_value()) << metrics_16.error().message;
        ASSERT_TRUE(display_1.has_value()) << display_1.error().message;
        ASSERT_TRUE(display_4.has_value()) << display_4.error().message;
        ASSERT_TRUE(display_16.has_value()) << display_16.error().message;
        mse_1[index] = metrics_1->mse;
        mse_4[index] = metrics_4->mse;
        mse_16[index] = metrics_16->mse;
        psnr_1[index] = display_1->psnr;
        psnr_4[index] = display_4->psnr;
        psnr_16[index] = display_16->psnr;
    }

    const auto median_mse_1 = median(mse_1);
    const auto median_mse_4 = median(mse_4);
    const auto median_mse_16 = median(mse_16);
    const auto median_psnr_1 = median(psnr_1);
    const auto median_psnr_4 = median(psnr_4);
    const auto median_psnr_16 = median(psnr_16);
    testing::Test::RecordProperty("median_mse_1spp", std::to_string(median_mse_1));
    testing::Test::RecordProperty("median_mse_4spp", std::to_string(median_mse_4));
    testing::Test::RecordProperty("median_mse_16spp", std::to_string(median_mse_16));
    testing::Test::RecordProperty("median_psnr_1spp", std::to_string(median_psnr_1));
    testing::Test::RecordProperty("median_psnr_4spp", std::to_string(median_psnr_4));
    testing::Test::RecordProperty("median_psnr_16spp", std::to_string(median_psnr_16));
    EXPECT_GT(median_mse_1, 0.0);
    EXPECT_LT(median_mse_4, median_mse_1 * 0.35);
    EXPECT_LT(median_mse_16, median_mse_4 * 0.35);
    EXPECT_GT(median_psnr_4, median_psnr_1 + 5.0);
    EXPECT_GT(median_psnr_16, median_psnr_4 + 5.0);

    const auto replay_first = render_nee(*scene, ReplayExtent, 4U, EvaluationSeed);
    const auto replay_second = render_nee(*scene, ReplayExtent, 4U, EvaluationSeed);
    ASSERT_TRUE(replay_first.has_value()) << replay_first.error().message;
    ASSERT_TRUE(replay_second.has_value()) << replay_second.error().message;
    expect_replay(replay_first->film, replay_second->film);
    EXPECT_EQ(replay_first->statistics.closest_hit_queries,
              replay_second->statistics.closest_hit_queries);
    EXPECT_EQ(replay_first->statistics.occlusion_queries,
              replay_second->statistics.occlusion_queries);
    EXPECT_EQ(replay_first->statistics.blocked_queries, replay_second->statistics.blocked_queries);
}

TEST(NextEventEstimationParityTest, MatchesScalarReferenceThroughCpuWavefront) {
    const auto scene = make_point_light_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    ASSERT_TRUE((*scene)->spectral_environment().has_value());
    ASSERT_EQ((*scene)->punctual_lights().size(), 4U);

    constexpr auto samples_per_pixel = std::uint32_t{4U};
    const auto inputs = scalar_wavefront_parity_test::make_inputs(
        ReplayExtent, samples_per_pixel, EvaluationSeed,
        (*scene)->spectral_environment()->wavelengths,
        [](const renderer::PixelSampleIndex& index,
           const renderer::SampleStream& stream) -> core::Result<renderer::Ray> {
            const auto position =
                primary_position(ReplayExtent, index.pixel_x, index.pixel_y, stream);
            return renderer::Ray::create(
                position + renderer::Vector3{.z = CameraHeight}, renderer::Vector3{.z = -1.0F},
                0.0F, std::numeric_limits<renderer::TransportScalar>::infinity(), PathTime,
                renderer::AllRayVisibility, renderer::VacuumMedium);
        });
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;

    const auto configuration = scalar_wavefront_parity_test::Configuration{
        .scene_name = "PointLightNee",
        .extent = ReplayExtent,
        .samples_per_pixel = samples_per_pixel,
        .seed = EvaluationSeed,
        .heuristic = renderer::MisHeuristic::power,
        .depth_limits = renderer::PathDepthLimits{.diffuse = 1U},
        .roulette_policy = renderer::RussianRoulettePolicy::disabled(),
        .worker_count = 4U,
        .thresholds = scalar_wavefront_parity_test::StrictThresholds,
    };
    const auto result = scalar_wavefront_parity_test::compare(*scene, *inputs, configuration);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_GT(result->wavefront_report.closure_samples, 0U);
    EXPECT_GT(result->wavefront_report.light_samples, 0U);
    EXPECT_GT(result->wavefront_report.shadow_queries, 0U);
    scalar_wavefront_parity_test::record_and_expect(configuration, *result);
}

TEST(NextEventEstimationImageTest, WritesStableEmbreePointLightImageAt64Spp) {
#if BLACKFRAME_HAS_PNG_PREVIEW
    const auto scene = make_point_light_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    const auto rendered = render_nee(*scene, ImageExtent, ImageSamplesPerPixel, EvaluationSeed);
    const auto oracle =
        render_enumerated_oracle(*scene, ImageExtent, ImageSamplesPerPixel, EvaluationSeed);
    ASSERT_TRUE(rendered.has_value()) << rendered.error().message;
    ASSERT_TRUE(oracle.has_value()) << oracle.error().message;
    expect_query_counts(*rendered, ImageExtent, ImageSamplesPerPixel);

    const auto metrics = renderer::compute_linear_metrics(rendered->film, *oracle);
    const auto display = renderer::compute_display_psnr(rendered->film, *oracle);
    ASSERT_TRUE(metrics.has_value()) << metrics.error().message;
    ASSERT_TRUE(display.has_value()) << display.error().message;
    testing::Test::RecordProperty("mse_linear", std::to_string(metrics->mse));
    testing::Test::RecordProperty("rmse_linear", std::to_string(metrics->rmse));
    testing::Test::RecordProperty("psnr_display", std::to_string(display->psnr));
    testing::Test::RecordProperty("closest_hit_queries",
                                  std::to_string(rendered->statistics.closest_hit_queries));
    testing::Test::RecordProperty("shadow_queries",
                                  std::to_string(rendered->statistics.occlusion_queries));
    EXPECT_TRUE(std::isfinite(metrics->mse));
    EXPECT_TRUE(std::isfinite(metrics->rmse));
    EXPECT_TRUE(std::isfinite(display->psnr));
    EXPECT_LT(metrics->mse, 0.0015);
    EXPECT_LT(metrics->rmse, 0.04);
    EXPECT_GT(display->psnr, 29.0);

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
    auto minimum_channel = std::numeric_limits<std::uint8_t>::max();
    auto maximum_channel = std::uint8_t{};
    auto non_black_pixels = std::uint32_t{};
    for (auto pixel = std::size_t{};
         pixel < static_cast<std::size_t>(ImageExtent.width) * ImageExtent.height; ++pixel) {
        auto non_black = false;
        for (auto channel = std::size_t{}; channel < 3U; ++channel) {
            const auto value = decoded[pixel * 3U + channel];
            minimum_channel = std::min(minimum_channel, value);
            maximum_channel = std::max(maximum_channel, value);
            non_black = non_black || value != 0U;
        }
        if (non_black) {
            ++non_black_pixels;
        }
    }
    stbi_image_free(decoded);
    EXPECT_GT(non_black_pixels, ImageExtent.width * ImageExtent.height / 2U);
    EXPECT_LT(minimum_channel, maximum_channel);
#else
    GTEST_SKIP() << "PNG preview support is disabled explicitly.";
#endif
}

} // namespace
} // namespace blackframe::engine
