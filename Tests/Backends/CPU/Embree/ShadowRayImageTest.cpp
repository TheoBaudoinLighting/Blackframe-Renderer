#include <Blackframe/Backends/CPU/Embree/AccelBackend.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Engine/FrameScene.hpp>
#include <Blackframe/Engine/SceneSurfaceInteraction.hpp>
#include <Blackframe/Engine/SceneVisibility.hpp>
#include <Blackframe/Engine/TriangleMesh.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/Light.hpp>
#include <Blackframe/Renderer/PngWriter.hpp>
#include <Blackframe/Renderer/ShadowRay.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
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

#if BLACKFRAME_HAS_PNG_PREVIEW

inline constexpr auto PanelExtent = renderer::RenderExtent{.width = 64U, .height = 64U};
inline constexpr auto AtlasExtent = renderer::RenderExtent{.width = 128U, .height = 64U};
inline constexpr auto ExpectedShadowPixelCount = std::uint32_t{24U * 24U};
inline constexpr auto ExpectedLitPixelCount =
    PanelExtent.width * PanelExtent.height - ExpectedShadowPixelCount;
inline constexpr auto PrimaryVisibility = renderer::RayMask{1U << 0U};
inline constexpr auto ShadowVisibility = renderer::RayMask{1U << 1U};

struct PanelStatistics final {
    std::uint32_t lit_pixels{};
    std::uint32_t shadow_pixels{};
    std::vector<std::uint8_t> visibility;
};

[[nodiscard]] core::Error image_error(std::string message) {
    return core::Error{
        .code = core::StatusCode::invalid_argument,
        .message = std::move(message),
    };
}

[[nodiscard]] renderer::TransportSpectrum constant_spectrum(const float value) {
    auto spectrum = renderer::TransportSpectrum{};
    spectrum.values.fill(value);
    return spectrum;
}

[[nodiscard]] core::Result<std::shared_ptr<const TriangleMesh>>
make_horizontal_quad(const float height, const float half_extent, const bool faces_down = false) {
    const auto normal = faces_down ? renderer::Normal3{.z = -1.0F} : renderer::Normal3{.z = 1.0F};
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

[[nodiscard]] core::Result<FrameSceneHandle> make_shadow_scene() {
    auto receiver = make_horizontal_quad(0.0F, 2.0F);
    auto blocker = make_horizontal_quad(1.0F, 0.375F);
    auto emitter = make_horizontal_quad(3.0F, 0.5F, true);
    auto wavelengths = renderer::sample_uniform_visible_wavelengths(0.25F);
    if (!receiver) {
        return std::unexpected(receiver.error());
    }
    if (!blocker) {
        return std::unexpected(blocker.error());
    }
    if (!emitter) {
        return std::unexpected(emitter.error());
    }
    if (!wavelengths) {
        return std::unexpected(wavelengths.error());
    }

    return FrameScene::create(FrameSceneDescription{
        .objects =
            {
                SceneObject{.id = {.value = 1U}},
                SceneObject{.id = {.value = 2U}},
                SceneObject{.id = {.value = 3U}},
            },
        .geometries =
            {
                SceneGeometry{.id = {.value = 11U}, .mesh = std::move(*receiver)},
                SceneGeometry{.id = {.value = 12U}, .mesh = std::move(*blocker)},
                SceneGeometry{.id = {.value = 13U}, .mesh = std::move(*emitter)},
            },
        .materials =
            {
                SceneMaterial{
                    .id = {.value = 21U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .reflectance = constant_spectrum(0.5F),
                            .emitted_radiance = constant_spectrum(0.0F),
                        },
                },
                SceneMaterial{
                    .id = {.value = 22U},
                    .spectral =
                        SceneSpectralMaterial{
                            .wavelengths = *wavelengths,
                            .reflectance = constant_spectrum(0.0F),
                            .emitted_radiance = constant_spectrum(1.0F),
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
                    .visibility_mask = PrimaryVisibility | ShadowVisibility,
                },
                SceneInstance{
                    .id = {.value = 32U},
                    .parent = std::nullopt,
                    .object = {.value = 2U},
                    .geometry = {.value = 12U},
                    .material = {.value = 21U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                    .visibility_mask = ShadowVisibility,
                },
                SceneInstance{
                    .id = {.value = 33U},
                    .parent = std::nullopt,
                    .object = {.value = 3U},
                    .geometry = {.value = 13U},
                    .material = {.value = 22U},
                    .local_to_parent = renderer::identity_matrix<renderer::TransportScalar>(),
                    .visibility_mask = ShadowVisibility,
                },
            },
        .spectral_environment =
            SceneSpectralEnvironment{
                .wavelengths = *wavelengths,
                .radiance = constant_spectrum(0.0F),
            },
    });
}

[[nodiscard]] renderer::Point3 receiver_pixel_position(const std::uint32_t x,
                                                       const std::uint32_t y) noexcept {
    constexpr auto span = renderer::TransportScalar{3};
    const auto world_x = renderer::TransportScalar{-1.5F} +
                         (static_cast<renderer::TransportScalar>(x) + 0.5F) * span /
                             static_cast<renderer::TransportScalar>(PanelExtent.width);
    const auto world_y = renderer::TransportScalar{1.5F} -
                         (static_cast<renderer::TransportScalar>(y) + 0.5F) * span /
                             static_cast<renderer::TransportScalar>(PanelExtent.height);
    return renderer::Point3{.x = world_x, .y = world_y, .z = 0.0F};
}

[[nodiscard]] core::Result<PanelStatistics>
render_visibility_panel(const AccelBackend& acceleration, renderer::Film& film,
                        const std::uint32_t atlas_x_offset,
                        const renderer::LightSampleEndpoint& light_endpoint) {
    auto statistics = PanelStatistics{};
    statistics.visibility.reserve(static_cast<std::size_t>(PanelExtent.width) * PanelExtent.height);

    for (auto y = std::uint32_t{}; y < PanelExtent.height; ++y) {
        for (auto x = std::uint32_t{}; x < PanelExtent.width; ++x) {
            const auto receiver_position = receiver_pixel_position(x, y);
            const auto primary_ray = renderer::Ray::create(
                receiver_position + renderer::Vector3{.z = -1.0F}, renderer::Vector3{.z = 1.0F},
                0.0F, 4.0F, 0.375F, PrimaryVisibility, renderer::VacuumMedium);
            if (!primary_ray) {
                return std::unexpected(primary_ray.error());
            }

            const auto source = resolve_scene_surface(acceleration, *primary_ray);
            if (!source) {
                return std::unexpected(source.error());
            }
            if (!*source) {
                return std::unexpected(
                    image_error("A sharp-shadow atlas primary ray missed the receiver."));
            }
            if ((*source)->interaction.identifiers().instance !=
                renderer::InstanceId{.value = 31U}) {
                return std::unexpected(
                    image_error("A sharp-shadow atlas primary ray resolved the wrong instance."));
            }

            const auto context = renderer::LightSampleContext::create(
                (*source)->interaction.position(), (*source)->interaction.time());
            if (!context) {
                return std::unexpected(context.error());
            }
            const auto light_sample = renderer::IncidentLightSample::create_finite(
                *context, light_endpoint, constant_spectrum(1.0F),
                renderer::ProbabilityDensity{
                    .value = 1.0F,
                    .measure = light_endpoint.is_surface()
                                   ? renderer::ProbabilityMeasure::solid_angle
                                   : renderer::ProbabilityMeasure::discrete,
                });
            if (!light_sample) {
                return std::unexpected(light_sample.error());
            }

            const auto shadow_ray =
                renderer::make_shadow_ray((*source)->interaction, (*source)->position_error,
                                          *light_sample, ShadowVisibility, renderer::VacuumMedium);
            if (!shadow_ray) {
                return std::unexpected(shadow_ray.error());
            }
            if (shadow_ray->time() != primary_ray->time() ||
                shadow_ray->mask() != ShadowVisibility ||
                shadow_ray->current_medium() != renderer::VacuumMedium ||
                shadow_ray->t_min() != 0.0F || !std::isfinite(shadow_ray->t_max()) ||
                !(shadow_ray->t_max() > shadow_ray->t_min())) {
                return std::unexpected(image_error(
                    "A finite sharp-shadow ray did not preserve its transport metadata."));
            }

            const auto transmittance = trace_vacuum_visibility(acceleration, *shadow_ray);
            if (!transmittance) {
                return std::unexpected(transmittance.error());
            }
            const auto visible = transmittance->values[0] == 1.0F;
            const auto expected_lane = visible ? 1.0F : 0.0F;
            for (const auto lane : transmittance->values) {
                if (lane != expected_lane) {
                    return std::unexpected(image_error(
                        "Vacuum visibility did not return an exact binary spectral packet."));
                }
            }

            statistics.visibility.push_back(visible ? std::uint8_t{1} : std::uint8_t{0});
            if (visible) {
                ++statistics.lit_pixels;
            } else {
                ++statistics.shadow_pixels;
            }
            const auto gray = renderer::LinearRGB{
                .red = expected_lane,
                .green = expected_lane,
                .blue = expected_lane,
            };
            const auto accumulated = film.add_sample(atlas_x_offset + x, y, gray, 1.0F);
            if (!accumulated) {
                return std::unexpected(accumulated.error());
            }
        }
    }
    return statistics;
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

[[nodiscard]] std::filesystem::path atlas_output_path() {
    if (const auto checksum_output = checksum_output_path(); checksum_output) {
        return *checksum_output;
    }
    return std::filesystem::path{BLACKFRAME_EMBREE_TEST_OUTPUT_DIR} /
           "shadow-ray-analytic-embree.png";
}

TEST(ShadowRayImageTest, WritesStableAnalyticAndEmbreeSharpShadowAtlas) {
    const auto scene = make_shadow_scene();
    ASSERT_TRUE(scene.has_value()) << scene.error().message;
    auto analytic = create_analytic_accel_backend(*scene);
    auto embree = create_embree_accel_backend(*scene);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
    ASSERT_TRUE(embree.has_value()) << embree.error().message;

    const auto light_endpoint = renderer::LightSampleEndpoint::create_surface(
        renderer::Point3{.x = 0.1F, .y = -0.2F, .z = 3.0F}, renderer::Vector3{},
        renderer::Normal3{.z = -1.0F});
    ASSERT_TRUE(light_endpoint.has_value()) << light_endpoint.error().message;
    auto film = renderer::Film::create(AtlasExtent);
    ASSERT_TRUE(film.has_value()) << film.error().message;

    const auto analytic_panel = render_visibility_panel(**analytic, *film, 0U, *light_endpoint);
    const auto embree_panel =
        render_visibility_panel(**embree, *film, PanelExtent.width, *light_endpoint);
    ASSERT_TRUE(analytic_panel.has_value()) << analytic_panel.error().message;
    ASSERT_TRUE(embree_panel.has_value()) << embree_panel.error().message;

    EXPECT_EQ(analytic_panel->visibility, embree_panel->visibility);
    EXPECT_EQ(analytic_panel->shadow_pixels, ExpectedShadowPixelCount);
    EXPECT_EQ(embree_panel->shadow_pixels, ExpectedShadowPixelCount);
    EXPECT_EQ(analytic_panel->lit_pixels, ExpectedLitPixelCount);
    EXPECT_EQ(embree_panel->lit_pixels, ExpectedLitPixelCount);

    const auto output_path = atlas_output_path();
    std::error_code cleanup_error;
    std::filesystem::remove(output_path, cleanup_error);
    ASSERT_FALSE(cleanup_error) << "Cannot replace '" << output_path.string()
                                << "': " << cleanup_error.message();
    const auto write_status = renderer::write_png_preview(*film, output_path);
    ASSERT_TRUE(write_status.has_value()) << write_status.error().message;
    ASSERT_TRUE(std::filesystem::is_regular_file(output_path));

    auto width = int{};
    auto height = int{};
    auto components = int{};
    auto* const decoded = stbi_load(output_path.string().c_str(), &width, &height, &components, 3);
    ASSERT_NE(decoded, nullptr) << "Cannot decode '" << output_path.string()
                                << "': " << stbi_failure_reason();
    EXPECT_EQ(width, static_cast<int>(AtlasExtent.width));
    EXPECT_EQ(height, static_cast<int>(AtlasExtent.height));
    EXPECT_EQ(components, 3);

    auto decoded_black_pixels = std::uint32_t{};
    auto decoded_white_pixels = std::uint32_t{};
    for (auto pixel = std::size_t{};
         pixel < static_cast<std::size_t>(AtlasExtent.width) * AtlasExtent.height; ++pixel) {
        const auto red = decoded[pixel * 3U];
        const auto green = decoded[pixel * 3U + 1U];
        const auto blue = decoded[pixel * 3U + 2U];
        EXPECT_EQ(red, green) << "pixel " << pixel;
        EXPECT_EQ(red, blue) << "pixel " << pixel;
        EXPECT_TRUE(red == 0U || red == std::numeric_limits<std::uint8_t>::max())
            << "pixel " << pixel;
        if (red == 0U) {
            ++decoded_black_pixels;
        } else {
            ++decoded_white_pixels;
        }
    }
    stbi_image_free(decoded);
    EXPECT_EQ(decoded_black_pixels, 2U * ExpectedShadowPixelCount);
    EXPECT_EQ(decoded_white_pixels, 2U * ExpectedLitPixelCount);
}

#else

TEST(ShadowRayImageTest, WritesStableAnalyticAndEmbreeSharpShadowAtlas) {
    GTEST_SKIP() << "PNG preview support is disabled explicitly.";
}

#endif

} // namespace
} // namespace blackframe::engine
