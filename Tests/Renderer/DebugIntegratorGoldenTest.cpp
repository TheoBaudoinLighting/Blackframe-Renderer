#include <Blackframe/Renderer/DebugIntegrators.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/PinholeCamera.hpp>
#include <Blackframe/Renderer/PngWriter.hpp>
#include <Blackframe/Renderer/Triangle.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#if BLACKFRAME_HAS_PNG_PREVIEW
#include <stb_image.h>
#endif

namespace blackframe::renderer {
namespace {

#if BLACKFRAME_HAS_PNG_PREVIEW

inline constexpr auto DebugGoldenExtent = RenderExtent{.width = 64, .height = 64};
inline constexpr auto DebugGoldenTime = TransportScalar{0.25F};

enum class DebugGoldenMode {
    normal,
    depth,
    uv,
    barycentrics,
    identifiers,
};

struct DiagnosticTriangle final {
    Triangle triangle;
    Vector3 dpdu;
    Vector3 dpdv;
    SurfaceIdentifiers identifiers;
};

struct DiagnosticHit final {
    SurfaceInteraction interaction;
    TransportScalar parameter;
    TriangleBarycentrics barycentrics;
};

[[nodiscard]] core::Error golden_error(std::string message) {
    return core::Error{
        .code = core::StatusCode::internal_error,
        .message = std::move(message),
    };
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

[[nodiscard]] std::filesystem::path artifact_path(const char* const filename) {
    if (const auto checksum_output = checksum_output_path(); checksum_output.has_value()) {
        return *checksum_output;
    }
    return std::filesystem::path{BLACKFRAME_RENDERER_TEST_OUTPUT_DIR} / filename;
}

[[nodiscard]] core::Result<std::array<DiagnosticTriangle, 2>> make_diagnostic_scene() {
    const auto left = Triangle::create(Point3{.x = -0.25F, .y = -0.22F, .z = -0.55F},
                                       Point3{.x = -0.02F, .y = -0.22F, .z = -0.45F},
                                       Point3{.x = -0.25F, .y = 0.22F, .z = -0.55F});
    const auto right = Triangle::create(Point3{.x = 0.02F, .y = -0.22F, .z = -0.45F},
                                        Point3{.x = 0.25F, .y = -0.22F, .z = -0.55F},
                                        Point3{.x = 0.25F, .y = 0.22F, .z = -0.55F});
    if (!left.has_value()) {
        return std::unexpected(left.error());
    }
    if (!right.has_value()) {
        return std::unexpected(right.error());
    }

    const auto left_dpdu = left->vertices()[1] - left->vertices()[0];
    const auto left_dpdv = left->vertices()[2] - left->vertices()[0];
    const auto right_dpdu = right->vertices()[1] - right->vertices()[0];
    const auto right_dpdv = right->vertices()[2] - right->vertices()[0];
    return std::array{
        DiagnosticTriangle{
            .triangle = *left,
            .dpdu = left_dpdu,
            .dpdv = left_dpdv,
            .identifiers =
                {
                    .instance = {.value = 11},
                    .geometry = {.value = 101},
                    .primitive = {.value = 2047},
                    .material = {.value = 1001},
                },
        },
        DiagnosticTriangle{
            .triangle = *right,
            .dpdu = right_dpdu,
            .dpdv = right_dpdv,
            .identifiers =
                {
                    .instance = {.value = 12},
                    .geometry = {.value = 102},
                    .primitive = {.value = std::uint32_t{2047} << 11U},
                    .material = {.value = 1002},
                },
        },
    };
}

[[nodiscard]] core::Result<std::optional<DiagnosticHit>>
trace_diagnostic_scene(const Ray& ray, const std::array<DiagnosticTriangle, 2>& scene) {
    auto closest = std::optional<DiagnosticHit>{};
    for (const auto& diagnostic : scene) {
        const auto intersection = diagnostic.triangle.intersect(ray);
        if (!intersection.has_value()) {
            return std::unexpected(intersection.error());
        }
        if (!intersection->has_value()) {
            continue;
        }

        const auto& hit = **intersection;
        if (closest.has_value() && hit.parameter >= closest->parameter) {
            continue;
        }
        const auto interaction = SurfaceInteraction::create(
            hit.position, hit.geometric_normal, hit.geometric_normal,
            Point2{.x = hit.barycentrics.vertex1, .y = hit.barycentrics.vertex2}, diagnostic.dpdu,
            diagnostic.dpdv, diagnostic.identifiers, ray.time());
        if (!interaction.has_value()) {
            return std::unexpected(interaction.error());
        }
        closest.emplace(DiagnosticHit{
            .interaction = *interaction,
            .parameter = hit.parameter,
            .barycentrics = hit.barycentrics,
        });
    }
    return closest;
}

[[nodiscard]] core::Result<LinearRGB> evaluate_debug_hit(const DebugGoldenMode mode,
                                                         const DiagnosticHit& hit) {
    switch (mode) {
    case DebugGoldenMode::normal:
        return debug_normal_color(hit.interaction);
    case DebugGoldenMode::depth:
        return debug_depth_color(hit.parameter);
    case DebugGoldenMode::uv:
        return debug_uv_color(hit.interaction);
    case DebugGoldenMode::barycentrics:
        return debug_barycentric_color(hit.barycentrics);
    case DebugGoldenMode::identifiers:
        return debug_identifier_color<TransportScalar>(hit.interaction.identifiers(),
                                                       DebugIdentifierKind::primitive);
    }
    return std::unexpected(golden_error("Unsupported debug golden mode."));
}

[[nodiscard]] core::Result<Film> render_debug_golden(const DebugGoldenMode mode) {
    const auto frame =
        OrthonormalFrame::from_normal_and_tangent(Normal3{.z = 1.0F}, Vector3{.x = 1.0F});
    if (!frame.has_value()) {
        return std::unexpected(frame.error());
    }
    const auto camera = PinholeCamera::create(Point3{}, *frame, DebugGoldenExtent,
                                              std::numbers::pi_v<TransportScalar> / 3.0F, 0.0F,
                                              2.0F, AllRayVisibility, VacuumMedium);
    if (!camera.has_value()) {
        return std::unexpected(camera.error());
    }
    const auto scene = make_diagnostic_scene();
    if (!scene.has_value()) {
        return std::unexpected(scene.error());
    }
    auto film = Film::create(DebugGoldenExtent);
    if (!film.has_value()) {
        return std::unexpected(film.error());
    }

    for (auto y = std::uint32_t{}; y < DebugGoldenExtent.height; ++y) {
        for (auto x = std::uint32_t{}; x < DebugGoldenExtent.width; ++x) {
            const auto ray = camera->generate_primary_ray(
                PixelSampleIndex{
                    .pixel_x = x,
                    .pixel_y = y,
                    .sample_index = 0,
                    .seed = 0,
                },
                PixelJitterMode::center, DebugGoldenTime);
            if (!ray.has_value()) {
                return std::unexpected(ray.error());
            }
            const auto hit = trace_diagnostic_scene(*ray, *scene);
            if (!hit.has_value()) {
                return std::unexpected(hit.error());
            }

            // A primary miss has the explicit diagnostic background color
            // black. It is not represented by any reserved surface ID.
            auto color = LinearRGB{};
            if (hit->has_value()) {
                const auto evaluated = evaluate_debug_hit(mode, **hit);
                if (!evaluated.has_value()) {
                    return std::unexpected(evaluated.error());
                }
                color = *evaluated;
            }
            const auto accumulated = film->add_sample(x, y, color, 1.0F);
            if (!accumulated.has_value()) {
                return std::unexpected(accumulated.error());
            }
        }
    }
    return std::move(*film);
}

void verify_complete_film(const Film& film) {
    EXPECT_EQ(film.extent().width, DebugGoldenExtent.width);
    EXPECT_EQ(film.extent().height, DebugGoldenExtent.height);
    EXPECT_EQ(film.crop(), full_film_crop(DebugGoldenExtent));
    EXPECT_EQ(film.pixel_count(), std::size_t{4096});
    for (auto y = std::uint32_t{}; y < DebugGoldenExtent.height; ++y) {
        for (auto x = std::uint32_t{}; x < DebugGoldenExtent.width; ++x) {
            const auto pixel = film.pixel(x, y);
            ASSERT_TRUE(pixel.has_value()) << x << ", " << y;
            EXPECT_FLOAT_EQ(pixel->weight_sum, 1.0F);
            EXPECT_EQ(pixel->sample_count, std::uint64_t{1});
            const auto resolved = film.resolved_pixel(x, y);
            ASSERT_TRUE(resolved.has_value()) << x << ", " << y;
            EXPECT_TRUE(std::isfinite(resolved->red));
            EXPECT_TRUE(std::isfinite(resolved->green));
            EXPECT_TRUE(std::isfinite(resolved->blue));
        }
    }
}

[[nodiscard]] LinearRGB resolved(const Film& film, const std::uint32_t x, const std::uint32_t y) {
    const auto color = film.resolved_pixel(x, y);
    EXPECT_TRUE(color.has_value());
    return color.value_or(LinearRGB{});
}

void verify_sentinels(const Film& film, const DebugGoldenMode mode) {
    EXPECT_EQ(resolved(film, 0, 0), LinearRGB{});
    const auto left = resolved(film, 16, 32);
    const auto right = resolved(film, 47, 32);
    EXPECT_NE(left, LinearRGB{});
    EXPECT_NE(right, LinearRGB{});

    switch (mode) {
    case DebugGoldenMode::normal:
        EXPECT_LT(left.red, 0.5F);
        EXPECT_GT(right.red, 0.5F);
        EXPECT_FLOAT_EQ(left.green, 0.5F);
        EXPECT_FLOAT_EQ(right.green, 0.5F);
        EXPECT_GT(left.blue, 0.9F);
        EXPECT_GT(right.blue, 0.9F);
        break;
    case DebugGoldenMode::depth:
        EXPECT_FLOAT_EQ(left.red, left.green);
        EXPECT_FLOAT_EQ(left.red, left.blue);
        EXPECT_FLOAT_EQ(right.red, right.green);
        EXPECT_FLOAT_EQ(right.red, right.blue);
        EXPECT_GT(left.red, 0.0F);
        EXPECT_LT(left.red, 1.0F);
        EXPECT_GT(right.red, 0.0F);
        EXPECT_LT(right.red, 1.0F);
        break;
    case DebugGoldenMode::uv:
        EXPECT_GE(left.red, 0.0F);
        EXPECT_LE(left.red, 1.0F);
        EXPECT_GE(left.green, 0.0F);
        EXPECT_LE(left.green, 1.0F);
        EXPECT_FLOAT_EQ(left.blue, 0.0F);
        EXPECT_FLOAT_EQ(right.blue, 0.0F);
        break;
    case DebugGoldenMode::barycentrics:
        EXPECT_FLOAT_EQ(left.red + left.green + left.blue, 1.0F);
        EXPECT_FLOAT_EQ(right.red + right.green + right.blue, 1.0F);
        break;
    case DebugGoldenMode::identifiers:
        EXPECT_EQ(left, (LinearRGB{.red = 1.0F, .green = 1.0F / 2048.0F, .blue = 1.0F / 1024.0F}));
        EXPECT_EQ(right, (LinearRGB{.red = 1.0F / 2048.0F, .green = 1.0F, .blue = 1.0F / 1024.0F}));
        break;
    }
}

void verify_written_png(const std::filesystem::path& output_path) {
    ASSERT_TRUE(std::filesystem::is_regular_file(output_path));
    auto width = int{};
    auto height = int{};
    auto components = int{};
    auto* const decoded = stbi_load(output_path.string().c_str(), &width, &height, &components, 3);
    ASSERT_NE(decoded, nullptr);
    EXPECT_EQ(width, 64);
    EXPECT_EQ(height, 64);
    EXPECT_EQ(components, 3);
    stbi_image_free(decoded);
}

void verify_debug_golden(const DebugGoldenMode mode, const char* const filename) {
    auto film = render_debug_golden(mode);
    ASSERT_TRUE(film.has_value());
    verify_complete_film(*film);
    verify_sentinels(*film, mode);

    const auto output_path = artifact_path(filename);
    std::error_code cleanup_error;
    std::filesystem::remove(output_path, cleanup_error);
    ASSERT_TRUE(write_png_preview(*film, output_path).has_value());
    verify_written_png(output_path);
    if (!checksum_output_path().has_value()) {
        EXPECT_TRUE(std::filesystem::remove(output_path));
    }
}

#endif

TEST(DebugIntegratorGoldenTest, Normal64x64) {
#if BLACKFRAME_HAS_PNG_PREVIEW
    verify_debug_golden(DebugGoldenMode::normal, "debug-normal-64x64.png");
#else
    GTEST_SKIP() << "PNG preview support is disabled explicitly.";
#endif
}

TEST(DebugIntegratorGoldenTest, Depth64x64) {
#if BLACKFRAME_HAS_PNG_PREVIEW
    verify_debug_golden(DebugGoldenMode::depth, "debug-depth-64x64.png");
#else
    GTEST_SKIP() << "PNG preview support is disabled explicitly.";
#endif
}

TEST(DebugIntegratorGoldenTest, Uv64x64) {
#if BLACKFRAME_HAS_PNG_PREVIEW
    verify_debug_golden(DebugGoldenMode::uv, "debug-uv-64x64.png");
#else
    GTEST_SKIP() << "PNG preview support is disabled explicitly.";
#endif
}

TEST(DebugIntegratorGoldenTest, Barycentrics64x64) {
#if BLACKFRAME_HAS_PNG_PREVIEW
    verify_debug_golden(DebugGoldenMode::barycentrics, "debug-barycentrics-64x64.png");
#else
    GTEST_SKIP() << "PNG preview support is disabled explicitly.";
#endif
}

TEST(DebugIntegratorGoldenTest, Identifiers64x64) {
#if BLACKFRAME_HAS_PNG_PREVIEW
    verify_debug_golden(DebugGoldenMode::identifiers, "debug-identifiers-64x64.png");
#else
    GTEST_SKIP() << "PNG preview support is disabled explicitly.";
#endif
}

} // namespace
} // namespace blackframe::renderer
