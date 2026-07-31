#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/PngWriter.hpp>
#include <Blackframe/Renderer/SampleDimensionMap.hpp>
#include <Blackframe/Renderer/SampleStream.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <optional>
#include <span>
#include <system_error>
#include <utility>

#if BLACKFRAME_HAS_PNG_PREVIEW
#include <stb_image.h>
#endif

namespace blackframe::renderer {
namespace {

#if BLACKFRAME_HAS_PNG_PREVIEW

inline constexpr auto LightSamplerAtlasExtent = RenderExtent{.width = 256U, .height = 128U};
inline constexpr auto LightSamplerPanelWidth = std::uint32_t{128U};
inline constexpr auto LightSamplerMosaicHeight = std::uint32_t{64U};
inline constexpr auto LightSamplerMosaicBlockSide = std::uint32_t{2U};
inline constexpr auto LightSamplerMosaicPixelsPerSample =
    LightSamplerMosaicBlockSide * LightSamplerMosaicBlockSide;
inline constexpr auto LightSamplerMosaicSampleCount =
    (LightSamplerPanelWidth / LightSamplerMosaicBlockSide) *
    (LightSamplerMosaicHeight / LightSamplerMosaicBlockSide);

[[nodiscard]] std::optional<std::filesystem::path> light_sampler_checksum_output_path() {
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

[[nodiscard]] std::filesystem::path light_sampler_atlas_output_path() {
    if (const auto checksum_output = light_sampler_checksum_output_path();
        checksum_output.has_value()) {
        return *checksum_output;
    }
    return std::filesystem::path{BLACKFRAME_LIGHT_TEST_OUTPUT_DIR} / "light-sampler-selection.png";
}

[[nodiscard]] TransportSpectrum light_sampler_spectrum(const TransportScalar lane0,
                                                       const TransportScalar lane1,
                                                       const TransportScalar lane2,
                                                       const TransportScalar lane3) {
    return TransportSpectrum{.values = {lane0, lane1, lane2, lane3}};
}

[[nodiscard]] std::array<TransportSpectrum, 4> light_sampler_atlas_powers() {
    return {
        light_sampler_spectrum(4.0F, 0.0F, 0.0F, 0.0F),
        light_sampler_spectrum(0.0F, 4.0F, 0.0F, 0.0F),
        light_sampler_spectrum(0.0F, 0.0F, 8.0F, 0.0F),
        light_sampler_spectrum(0.0F, 0.0F, 0.0F, 16.0F),
    };
}

[[nodiscard]] LinearRGB light_sampler_color(const std::uint32_t index,
                                            const TransportScalar scale) noexcept {
    constexpr auto colors = std::array{
        LinearRGB{.red = 0.80F, .green = 0.035F, .blue = 0.02F},
        LinearRGB{.red = 0.035F, .green = 0.70F, .blue = 0.06F},
        LinearRGB{.red = 0.025F, .green = 0.07F, .blue = 0.85F},
        LinearRGB{.red = 0.82F, .green = 0.58F, .blue = 0.025F},
    };
    const auto color = colors[index];
    return {
        .red = color.red * scale,
        .green = color.green * scale,
        .blue = color.blue * scale,
    };
}

TEST(LightSamplerImageTest, WritesStableUniformAndPowerSelectionAtlas) {
    const auto powers = light_sampler_atlas_powers();
    const auto uniform = LightSampler::create_uniform(powers.size());
    const auto weighted =
        LightSampler::create_power_weighted(std::span<const TransportSpectrum>{powers});
    ASSERT_TRUE(uniform.has_value()) << uniform.error().message;
    ASSERT_TRUE(weighted.has_value()) << weighted.error().message;
    const auto dimensions = sample_dimensions_for_bounce(0U);
    ASSERT_TRUE(dimensions.has_value()) << dimensions.error().message;

    auto film = Film::create(LightSamplerAtlasExtent);
    ASSERT_TRUE(film.has_value()) << film.error().message;
    auto uniform_counts = std::array<std::uint32_t, 4>{};
    auto weighted_counts = std::array<std::uint32_t, 4>{};
    for (auto y = std::uint32_t{}; y < LightSamplerAtlasExtent.height; ++y) {
        for (auto x = std::uint32_t{}; x < LightSamplerAtlasExtent.width; ++x) {
            const auto in_weighted_panel = x >= LightSamplerPanelWidth;
            const auto panel_x = x % LightSamplerPanelWidth;
            const auto& sampler = in_weighted_panel ? *weighted : *uniform;

            auto canonical = TransportScalar{};
            auto color_scale = TransportScalar{1};
            if (y < LightSamplerMosaicHeight) {
                const auto stream = SampleStream{SampleStreamIndex{
                    .pixel_x = panel_x / LightSamplerMosaicBlockSide,
                    .pixel_y = y / LightSamplerMosaicBlockSide,
                    .sample_index = 0U,
                    .seed = 0x243F6A8885A308D3ULL,
                }};
                canonical = stream.sample_1d(dimensions->light_selection);
            } else {
                canonical = (static_cast<TransportScalar>(panel_x) + 0.5F) /
                            static_cast<TransportScalar>(LightSamplerPanelWidth);
                color_scale = 0.55F;
            }

            const auto selection = sampler.sample(canonical);
            ASSERT_TRUE(selection.has_value())
                << "pixel (" << x << ", " << y << "): " << selection.error().message;
            if (y < LightSamplerMosaicHeight) {
                auto& counts = in_weighted_panel ? weighted_counts : uniform_counts;
                ++counts[selection->light_index()];
            }
            const auto accumulated = film->add_sample(
                x, y, light_sampler_color(selection->light_index(), color_scale), 1.0F);
            ASSERT_TRUE(accumulated.has_value())
                << "pixel (" << x << ", " << y << "): " << accumulated.error().message;
        }
    }
    constexpr auto uniform_probabilities = std::array{0.25, 0.25, 0.25, 0.25};
    constexpr auto weighted_probabilities = std::array{0.125, 0.125, 0.25, 0.5};
    for (auto index = std::size_t{}; index < uniform_counts.size(); ++index) {
        for (const auto observation :
             {std::pair{uniform_counts[index], uniform_probabilities[index]},
              std::pair{weighted_counts[index], weighted_probabilities[index]}}) {
            ASSERT_EQ(observation.first % LightSamplerMosaicPixelsPerSample, 0U);
            const auto independent_count = observation.first / LightSamplerMosaicPixelsPerSample;
            const auto expected =
                static_cast<ReferenceScalar>(LightSamplerMosaicSampleCount) * observation.second;
            const auto sigma =
                std::sqrt(static_cast<ReferenceScalar>(LightSamplerMosaicSampleCount) *
                          observation.second * (1.0 - observation.second));
            EXPECT_LE(std::abs(static_cast<ReferenceScalar>(independent_count) - expected),
                      8.0 * sigma + 1.0);
        }
    }

    const auto output_path = light_sampler_atlas_output_path();
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
    EXPECT_EQ(width, static_cast<int>(LightSamplerAtlasExtent.width));
    EXPECT_EQ(height, static_cast<int>(LightSamplerAtlasExtent.height));
    EXPECT_EQ(components, 3);
    stbi_image_free(decoded);
}

#else

TEST(LightSamplerImageTest, WritesStableUniformAndPowerSelectionAtlas) {
    GTEST_SKIP() << "PNG preview support is disabled explicitly.";
}

#endif

} // namespace
} // namespace blackframe::renderer
