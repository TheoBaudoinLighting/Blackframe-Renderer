#include <Blackframe/Renderer/EnvironmentLights.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/PngWriter.hpp>
#include <Blackframe/Renderer/Transforms.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

#if BLACKFRAME_HAS_PNG_PREVIEW
#include <stb_image.h>
#endif

namespace blackframe::renderer {
namespace {

#if BLACKFRAME_HAS_PNG_PREVIEW

inline constexpr auto EnvironmentAtlasExtent = RenderExtent{.width = 256U, .height = 128U};
inline constexpr auto EnvironmentPanelWidth = std::uint32_t{128U};
inline constexpr auto EnvironmentPanelHeight = std::uint32_t{64U};
inline constexpr auto EnvironmentSourceWidth = std::uint32_t{16U};
inline constexpr auto EnvironmentSourceHeight = std::uint32_t{8U};

[[nodiscard]] std::optional<std::filesystem::path> environment_checksum_output_path() {
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

[[nodiscard]] std::filesystem::path environment_atlas_output_path() {
    if (const auto checksum_output = environment_checksum_output_path();
        checksum_output.has_value()) {
        return *checksum_output;
    }
    return std::filesystem::path{BLACKFRAME_LIGHT_TEST_OUTPUT_DIR} /
           "environment-light-radiance-pdf.png";
}

[[nodiscard]] TransportSpectrum atlas_spectrum(const TransportScalar lane0,
                                               const TransportScalar lane1,
                                               const TransportScalar lane2,
                                               const TransportScalar lane3) {
    return TransportSpectrum{.values = {lane0, lane1, lane2, lane3}};
}

[[nodiscard]] std::vector<TransportSpectrum> environment_atlas_source() {
    auto source = std::vector<TransportSpectrum>{};
    source.reserve(static_cast<std::size_t>(EnvironmentSourceWidth) * EnvironmentSourceHeight);
    for (auto y = std::uint32_t{}; y < EnvironmentSourceHeight; ++y) {
        for (auto x = std::uint32_t{}; x < EnvironmentSourceWidth; ++x) {
            const auto longitude = static_cast<TransportScalar>(x) /
                                   static_cast<TransportScalar>(EnvironmentSourceWidth - 1U);
            const auto latitude = static_cast<TransportScalar>(y) /
                                  static_cast<TransportScalar>(EnvironmentSourceHeight - 1U);
            auto value = atlas_spectrum(0.015F + 0.035F * longitude, 0.02F + 0.03F * latitude,
                                        0.025F + 0.02F * (1.0F - longitude),
                                        0.03F + 0.015F * (1.0F - latitude));
            if (y == 3U || y == 4U) {
                value = atlas_spectrum(0.12F + 0.08F * longitude, 0.09F + 0.04F * longitude, 0.055F,
                                       0.035F);
            }
            if (x == 2U && y == 2U) {
                value = atlas_spectrum(0.95F, 0.42F, 0.11F, 0.035F);
            }
            if (x == 11U && y == 5U) {
                value = atlas_spectrum(0.06F, 0.2F, 0.72F, 0.48F);
            }
            source.push_back(value);
        }
    }
    return source;
}

[[nodiscard]] Vector3 environment_panel_direction(const std::uint32_t x, const std::uint32_t y) {
    const auto phi = 2.0F * std::numbers::pi_v<TransportScalar> *
                     (static_cast<TransportScalar>(x) + 0.5F) /
                     static_cast<TransportScalar>(EnvironmentPanelWidth);
    const auto theta = std::numbers::pi_v<TransportScalar> *
                       (static_cast<TransportScalar>(y) + 0.5F) /
                       static_cast<TransportScalar>(EnvironmentPanelHeight);
    const auto sine_theta = std::sin(theta);
    return {
        .x = sine_theta * std::cos(phi),
        .y = std::cos(theta),
        .z = sine_theta * std::sin(phi),
    };
}

[[nodiscard]] core::Result<Ray> environment_atlas_ray(const Vector3 direction) {
    return Ray::create(Point3{}, direction, 0.0F, std::numeric_limits<TransportScalar>::infinity(),
                       0.0F, AllRayVisibility, VacuumMedium);
}

[[nodiscard]] LinearRGB diagnostic_radiance_rgb(const TransportSpectrum& spectrum) noexcept {
    return {
        .red = spectrum[0],
        .green = spectrum[1],
        .blue = 0.5F * (spectrum[2] + spectrum[3]),
    };
}

[[nodiscard]] LinearRGB diagnostic_pdf_rgb(const TransportScalar pdf) noexcept {
    const auto relative_to_uniform = pdf * (4.0F * std::numbers::pi_v<TransportScalar>);
    const auto gray = relative_to_uniform / (1.0F + relative_to_uniform);
    return {.red = gray, .green = gray, .blue = gray};
}

TEST(EnvironmentLightImageTest, WritesStableRadianceAndPdfAtlas) {
    const auto wavelengths = sample_uniform_visible_wavelengths(0.25F);
    const auto rotation =
        quaternion_from_axis_angle(Vector3{.y = 1.0F}, std::numbers::pi_v<TransportScalar> / 2.0F);
    ASSERT_TRUE(wavelengths.has_value()) << wavelengths.error().message;
    ASSERT_TRUE(rotation.has_value()) << rotation.error().message;

    const auto source = environment_atlas_source();
    const auto identity = EnvironmentMapLightT<TransportScalar>::create(
        EnvironmentSourceWidth, EnvironmentSourceHeight, source, *wavelengths, Quaternion{});
    const auto rotated = EnvironmentMapLightT<TransportScalar>::create(
        EnvironmentSourceWidth, EnvironmentSourceHeight, source, *wavelengths, *rotation);
    const auto context = LightSampleContext::create(Point3{}, 0.0F);
    ASSERT_TRUE(identity.has_value()) << identity.error().message;
    ASSERT_TRUE(rotated.has_value()) << rotated.error().message;
    ASSERT_TRUE(context.has_value()) << context.error().message;

    auto film = Film::create(EnvironmentAtlasExtent);
    ASSERT_TRUE(film.has_value()) << film.error().message;
    for (auto y = std::uint32_t{}; y < EnvironmentAtlasExtent.height; ++y) {
        for (auto x = std::uint32_t{}; x < EnvironmentAtlasExtent.width; ++x) {
            const auto& light = x < EnvironmentPanelWidth ? *identity : *rotated;
            const auto panel_x = x % EnvironmentPanelWidth;
            const auto panel_y = y % EnvironmentPanelHeight;
            const auto direction = environment_panel_direction(panel_x, panel_y);
            auto diagnostic = LinearRGB{};
            if (y < EnvironmentPanelHeight) {
                const auto escaped_ray = environment_atlas_ray(direction);
                ASSERT_TRUE(escaped_ray.has_value()) << escaped_ray.error().message;
                const auto radiance = light.le(*escaped_ray, *wavelengths);
                ASSERT_TRUE(radiance.has_value())
                    << "pixel (" << x << ", " << y << "): " << radiance.error().message;
                diagnostic = diagnostic_radiance_rgb(*radiance);
            } else {
                const auto pdf = light.pdf_li(*context, direction, *wavelengths);
                ASSERT_TRUE(pdf.has_value())
                    << "pixel (" << x << ", " << y << "): " << pdf.error().message;
                diagnostic = diagnostic_pdf_rgb(pdf->value());
            }
            const auto accumulated = film->add_sample(x, y, diagnostic, 1.0F);
            ASSERT_TRUE(accumulated.has_value())
                << "pixel (" << x << ", " << y << "): " << accumulated.error().message;
        }
    }

    const auto output_path = environment_atlas_output_path();
    std::error_code cleanup_error;
    std::filesystem::remove(output_path, cleanup_error);
    ASSERT_FALSE(cleanup_error) << "Cannot replace '" << output_path.string()
                                << "': " << cleanup_error.message();
    const auto write_status = write_png_preview(*film, output_path);
    ASSERT_TRUE(write_status.has_value()) << write_status.error().message;
    ASSERT_TRUE(std::filesystem::is_regular_file(output_path));

    auto width = int{};
    auto height = int{};
    auto components = int{};
    auto* const decoded = stbi_load(output_path.string().c_str(), &width, &height, &components, 3);
    ASSERT_NE(decoded, nullptr) << "Cannot decode '" << output_path.string()
                                << "': " << stbi_failure_reason();
    EXPECT_EQ(width, static_cast<int>(EnvironmentAtlasExtent.width));
    EXPECT_EQ(height, static_cast<int>(EnvironmentAtlasExtent.height));
    EXPECT_EQ(components, 3);
    stbi_image_free(decoded);
}

#else

TEST(EnvironmentLightImageTest, WritesStableRadianceAndPdfAtlas) {
    GTEST_SKIP() << "PNG preview support is disabled explicitly.";
}

#endif

} // namespace
} // namespace blackframe::renderer
