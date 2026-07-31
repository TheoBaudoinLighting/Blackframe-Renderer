#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/PngWriter.hpp>
#include <Blackframe/Renderer/PunctualLights.hpp>
#include <Blackframe/Renderer/WavelengthSampling.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <gtest/gtest.h>
#include <numbers>
#include <optional>
#include <system_error>

#if BLACKFRAME_HAS_PNG_PREVIEW
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#endif

namespace blackframe::renderer {
namespace {

#if BLACKFRAME_HAS_PNG_PREVIEW

inline constexpr auto PunctualLightAtlasExtent = RenderExtent{.width = 192U, .height = 64U};
inline constexpr auto PunctualLightPanelWidth = std::uint32_t{64U};

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
    if (const auto checksum_output = checksum_output_path(); checksum_output.has_value()) {
        return *checksum_output;
    }
    return std::filesystem::path{BLACKFRAME_LIGHT_TEST_OUTPUT_DIR} / "punctual-light-response.png";
}

[[nodiscard]] TransportSpectrum constant_spectrum(const TransportScalar value) {
    auto spectrum = TransportSpectrum{};
    spectrum.values.fill(value);
    return spectrum;
}

[[nodiscard]] TransportScalar mean_spectral_lanes(const TransportSpectrum& spectrum) noexcept {
    auto sum = TransportScalar{};
    for (const auto value : spectrum.values) {
        sum += value;
    }
    return sum / static_cast<TransportScalar>(TransportSpectrumSampleCount);
}

template <typename Light>
[[nodiscard]] core::Result<TransportScalar>
sample_mean_radiance(const Light& light, const Point3 position,
                     const SampledWavelengths& wavelengths) {
    const auto context = LightSampleContext::create(position, 0.0F);
    if (!context.has_value()) {
        return std::unexpected(context.error());
    }

    const auto sampled = light.sample_li(*context, Point2{.x = 0.5F, .y = 0.5F}, wavelengths);
    if (!sampled.has_value()) {
        return std::unexpected(sampled.error());
    }
    // A successful empty delta-light sample is the contract's explicit zero-support result.
    // Only that physical result becomes black; implementation errors remain errors above.
    if (!sampled->has_value()) {
        return TransportScalar{};
    }
    return mean_spectral_lanes((**sampled).incident_radiance());
}

[[nodiscard]] Point3 panel_position(const std::uint32_t x, const std::uint32_t y) noexcept {
    const auto panel_x = x % PunctualLightPanelWidth;
    return Point3{
        .x = 2.0F * (static_cast<TransportScalar>(panel_x) + 0.5F) /
                 static_cast<TransportScalar>(PunctualLightPanelWidth) -
             1.0F,
        .y = 1.0F - 2.0F * (static_cast<TransportScalar>(y) + 0.5F) /
                        static_cast<TransportScalar>(PunctualLightAtlasExtent.height),
        .z = 0.0F,
    };
}

TEST(PunctualLightImageTest, WritesStableAnalyticResponseAtlas) {
    const auto wavelengths = sample_uniform_visible_wavelengths(0.25F);
    ASSERT_TRUE(wavelengths.has_value()) << wavelengths.error().message;

    const auto point =
        PointLight::create(Point3{.z = 1.0F}, Vector3{}, *wavelengths, constant_spectrum(0.8F));
    const auto directional =
        DirectionalLight::create(Vector3{.z = -1.0F}, *wavelengths, constant_spectrum(0.35F));
    const auto spot = SpotLight::create(Point3{.z = 1.0F}, Vector3{}, Vector3{.z = -1.0F},
                                        std::numbers::pi_v<TransportScalar> / 12.0F,
                                        std::numbers::pi_v<TransportScalar> / 4.0F, *wavelengths,
                                        constant_spectrum(0.8F));
    ASSERT_TRUE(point.has_value()) << point.error().message;
    ASSERT_TRUE(directional.has_value()) << directional.error().message;
    ASSERT_TRUE(spot.has_value()) << spot.error().message;

    const auto point_center = sample_mean_radiance(*point, Point3{}, *wavelengths);
    const auto point_corner =
        sample_mean_radiance(*point, Point3{.x = 1.0F, .y = 1.0F}, *wavelengths);
    const auto directional_center = sample_mean_radiance(*directional, Point3{}, *wavelengths);
    const auto directional_corner =
        sample_mean_radiance(*directional, Point3{.x = 1.0F, .y = 1.0F}, *wavelengths);
    const auto spot_center = sample_mean_radiance(*spot, Point3{}, *wavelengths);
    const auto spot_transition = sample_mean_radiance(*spot, Point3{.x = 0.5F}, *wavelengths);
    const auto spot_outside = sample_mean_radiance(*spot, Point3{.x = 1.5F}, *wavelengths);
    ASSERT_TRUE(point_center.has_value()) << point_center.error().message;
    ASSERT_TRUE(point_corner.has_value()) << point_corner.error().message;
    ASSERT_TRUE(directional_center.has_value()) << directional_center.error().message;
    ASSERT_TRUE(directional_corner.has_value()) << directional_corner.error().message;
    ASSERT_TRUE(spot_center.has_value()) << spot_center.error().message;
    ASSERT_TRUE(spot_transition.has_value()) << spot_transition.error().message;
    ASSERT_TRUE(spot_outside.has_value()) << spot_outside.error().message;
    EXPECT_GT(*point_center, *point_corner);
    EXPECT_FLOAT_EQ(*directional_center, *directional_corner);
    EXPECT_GT(*spot_center, *spot_transition);
    EXPECT_GT(*spot_transition, 0.0F);
    EXPECT_FLOAT_EQ(*spot_outside, 0.0F);

    auto film = Film::create(PunctualLightAtlasExtent);
    ASSERT_TRUE(film.has_value()) << film.error().message;
    for (auto y = std::uint32_t{}; y < PunctualLightAtlasExtent.height; ++y) {
        for (auto x = std::uint32_t{}; x < PunctualLightAtlasExtent.width; ++x) {
            const auto position = panel_position(x, y);
            auto response = core::Result<TransportScalar>{};
            if (x < PunctualLightPanelWidth) {
                response = sample_mean_radiance(*point, position, *wavelengths);
            } else if (x < 2U * PunctualLightPanelWidth) {
                response = sample_mean_radiance(*directional, position, *wavelengths);
            } else {
                response = sample_mean_radiance(*spot, position, *wavelengths);
            }
            ASSERT_TRUE(response.has_value()) << response.error().message;
            const auto gray = LinearRGB{
                .red = *response,
                .green = *response,
                .blue = *response,
            };
            const auto accumulated = film->add_sample(x, y, gray, 1.0F);
            ASSERT_TRUE(accumulated.has_value())
                << "pixel (" << x << ", " << y << "): " << accumulated.error().message;
        }
    }

    const auto output_path = atlas_output_path();
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
    EXPECT_EQ(width, static_cast<int>(PunctualLightAtlasExtent.width));
    EXPECT_EQ(height, static_cast<int>(PunctualLightAtlasExtent.height));
    EXPECT_EQ(components, 3);
    stbi_image_free(decoded);
}

#else

TEST(PunctualLightImageTest, WritesStableAnalyticResponseAtlas) {
    GTEST_SKIP() << "PNG preview support is disabled explicitly.";
}

#endif

} // namespace
} // namespace blackframe::renderer
