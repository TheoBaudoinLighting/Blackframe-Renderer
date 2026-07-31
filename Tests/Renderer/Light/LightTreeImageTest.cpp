#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/PngWriter.hpp>
#include <Blackframe/Renderer/SampleDimensionMap.hpp>
#include <Blackframe/Renderer/SampleStream.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <span>
#include <system_error>
#include <utility>
#include <vector>

#if BLACKFRAME_HAS_PNG_PREVIEW
#include <stb_image.h>
#endif

namespace blackframe::renderer {
namespace {

#if BLACKFRAME_HAS_PNG_PREVIEW

inline constexpr auto LightTreeAtlasExtent = RenderExtent{.width = 256U, .height = 128U};
inline constexpr auto LightTreePanelWidth = std::uint32_t{128U};
inline constexpr auto LightTreePanelHeight = std::uint32_t{64U};
inline constexpr auto LightTreeGridSide = std::uint32_t{16U};
inline constexpr auto LightTreeCellWidth = LightTreePanelWidth / LightTreeGridSide;
inline constexpr auto LightTreeCellHeight = LightTreePanelHeight / LightTreeGridSide;
inline constexpr auto LightTreeLightCount = LightTreeGridSide * LightTreeGridSide;
inline constexpr auto LightTreeFrequencySamples = std::uint32_t{65'536};

template <typename Value> [[nodiscard]] Value require_tree_image_value(core::Result<Value> result) {
    if (!result) {
        ADD_FAILURE() << result.error().message;
        std::abort();
    }
    return std::move(*result);
}

[[nodiscard]] std::optional<std::filesystem::path> light_tree_checksum_output_path() {
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

[[nodiscard]] std::filesystem::path light_tree_atlas_output_path() {
    if (const auto checksum_output = light_tree_checksum_output_path();
        checksum_output.has_value()) {
        return *checksum_output;
    }
    return std::filesystem::path{BLACKFRAME_LIGHT_TEST_OUTPUT_DIR} / "light-tree-distribution.png";
}

[[nodiscard]] TransportSpectrum light_tree_atlas_power(const std::uint32_t index) {
    if (index % 17U == 0U) {
        return {};
    }
    auto power = TransportSpectrum{};
    const auto lane = static_cast<std::size_t>(index % power.values.size());
    const auto grid_x = index % LightTreeGridSide;
    const auto grid_y = index / LightTreeGridSide;
    const auto value = 0.5F + 0.125F * static_cast<TransportScalar>((grid_x + 2U * grid_y) % 9U);
    power[lane] = value;
    return power;
}

[[nodiscard]] LinearRGB light_tree_atlas_color(const std::uint32_t panel,
                                               const TransportScalar value) noexcept {
    const auto intensity = std::clamp(value, 0.0F, 1.0F);
    if (panel == 0U) {
        return {.red = 0.06F + 0.74F * intensity,
                .green = 0.015F + 0.10F * intensity,
                .blue = 0.10F + 0.78F * intensity};
    }
    if (panel == 1U) {
        return {.red = 0.02F + 0.82F * intensity,
                .green = 0.04F + 0.70F * intensity,
                .blue = 0.025F + 0.08F * intensity};
    }
    if (panel == 2U) {
        return {.red = 0.02F + 0.06F * intensity,
                .green = 0.04F + 0.68F * intensity,
                .blue = 0.08F + 0.82F * intensity};
    }
    return {.red = 0.025F + 0.90F * intensity,
            .green = 0.015F + 0.18F * intensity,
            .blue = 0.015F + 0.03F * intensity};
}

TEST(LightTreeImageTest, WritesStableNormalizedDistributionAtlas) {
    auto inputs = std::vector<LightTreeInput>{};
    auto powers = std::vector<TransportSpectrum>{};
    inputs.reserve(LightTreeLightCount);
    powers.reserve(LightTreeLightCount);
    for (auto grid_y = std::uint32_t{0}; grid_y < LightTreeGridSide; ++grid_y) {
        for (auto grid_x = std::uint32_t{0}; grid_x < LightTreeGridSide; ++grid_x) {
            const auto center_x = static_cast<TransportScalar>(grid_x) - 7.5F;
            const auto center_y = static_cast<TransportScalar>(grid_y) - 7.5F;
            const auto bounds = Bounds3::from_minimum_maximum(
                Point3{.x = center_x - 0.2F, .y = center_y - 0.2F, .z = -0.2F},
                Point3{.x = center_x + 0.2F, .y = center_y + 0.2F, .z = 0.2F});
            ASSERT_TRUE(bounds.has_value()) << bounds.error().message;
            const auto power = light_tree_atlas_power(grid_y * LightTreeGridSide + grid_x);
            inputs.push_back(LightTreeInput{.bounds = *bounds, .spectral_power = power});
            powers.push_back(power);
        }
    }

    const auto flat = LightSampler::create_power_weighted(powers);
    const auto tree = LightSampler::create_spatial_tree(inputs);
    const auto context =
        LightSampleContext::create(Point3{.x = -6.5F, .y = -5.5F, .z = 1.25F}, 0.0F);
    const auto dimensions = sample_dimensions_for_bounce(0U);
    ASSERT_TRUE(flat.has_value()) << flat.error().message;
    ASSERT_TRUE(tree.has_value()) << tree.error().message;
    ASSERT_TRUE(context.has_value()) << context.error().message;
    ASSERT_TRUE(dimensions.has_value()) << dimensions.error().message;

    auto flat_probabilities = std::array<TransportScalar, LightTreeLightCount>{};
    auto tree_probabilities = std::array<TransportScalar, LightTreeLightCount>{};
    auto frequencies = std::array<TransportScalar, LightTreeLightCount>{};
    auto errors = std::array<TransportScalar, LightTreeLightCount>{};
    auto counts = std::array<std::uint32_t, LightTreeLightCount>{};
    auto probability_sum = ReferenceScalar{0};
    for (auto index = std::uint32_t{0}; index < LightTreeLightCount; ++index) {
        flat_probabilities[index] = require_tree_image_value(flat->probability(index)).value();
        tree_probabilities[index] =
            require_tree_image_value(tree->probability(*context, index)).value();
        probability_sum += static_cast<ReferenceScalar>(tree_probabilities[index]);
    }
    EXPECT_NEAR(probability_sum, 1.0,
                1024.0 *
                    static_cast<ReferenceScalar>(std::numeric_limits<TransportScalar>::epsilon()));
    for (auto sample_index = std::uint32_t{0}; sample_index < LightTreeFrequencySamples;
         ++sample_index) {
        const auto stream = SampleStream{SampleStreamIndex{
            .pixel_x = 31U,
            .pixel_y = 47U,
            .sample_index = sample_index,
            .seed = 0xD1B54A32D192ED03ULL,
        }};
        const auto selection =
            tree->sample(*context, stream.sample_1d(dimensions->light_selection));
        ASSERT_TRUE(selection.has_value()) << selection.error().message;
        ++counts[selection->light_index()];
    }
    for (auto index = std::size_t{0}; index < counts.size(); ++index) {
        const auto probability = static_cast<ReferenceScalar>(tree_probabilities[index]);
        const auto expected = static_cast<ReferenceScalar>(LightTreeFrequencySamples) * probability;
        const auto sigma = std::sqrt(static_cast<ReferenceScalar>(LightTreeFrequencySamples) *
                                     probability * (1.0 - probability));
        EXPECT_LE(std::abs(static_cast<ReferenceScalar>(counts[index]) - expected),
                  8.0 * sigma + 1.0);
        frequencies[index] = static_cast<TransportScalar>(counts[index]) /
                             static_cast<TransportScalar>(LightTreeFrequencySamples);
        errors[index] = std::abs(frequencies[index] - tree_probabilities[index]);
    }

    const auto flat_peak = *std::max_element(flat_probabilities.begin(), flat_probabilities.end());
    const auto tree_peak = *std::max_element(tree_probabilities.begin(), tree_probabilities.end());
    const auto frequency_peak = *std::max_element(frequencies.begin(), frequencies.end());
    const auto error_peak = *std::max_element(errors.begin(), errors.end());
    ASSERT_GT(flat_peak, 0.0F);
    ASSERT_GT(tree_peak, 0.0F);
    ASSERT_GT(frequency_peak, 0.0F);
    ASSERT_GT(error_peak, 0.0F);

    auto film = Film::create(LightTreeAtlasExtent);
    ASSERT_TRUE(film.has_value()) << film.error().message;
    for (auto y = std::uint32_t{0}; y < LightTreeAtlasExtent.height; ++y) {
        for (auto x = std::uint32_t{0}; x < LightTreeAtlasExtent.width; ++x) {
            const auto panel_x = x / LightTreePanelWidth;
            const auto panel_y = y / LightTreePanelHeight;
            const auto panel = panel_y * 2U + panel_x;
            const auto local_x = x % LightTreePanelWidth;
            const auto local_y = y % LightTreePanelHeight;
            const auto light_index =
                (local_y / LightTreeCellHeight) * LightTreeGridSide + local_x / LightTreeCellWidth;
            auto normalized = TransportScalar{0};
            if (panel == 0U) {
                normalized = flat_probabilities[light_index] / flat_peak;
            } else if (panel == 1U) {
                normalized = tree_probabilities[light_index] / tree_peak;
            } else if (panel == 2U) {
                normalized = frequencies[light_index] / frequency_peak;
            } else {
                normalized = errors[light_index] / error_peak;
            }
            const auto accumulated =
                film->add_sample(x, y, light_tree_atlas_color(panel, normalized), 1.0F);
            ASSERT_TRUE(accumulated.has_value())
                << "pixel (" << x << ", " << y << "): " << accumulated.error().message;
        }
    }

    const auto output_path = light_tree_atlas_output_path();
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
    EXPECT_EQ(width, static_cast<int>(LightTreeAtlasExtent.width));
    EXPECT_EQ(height, static_cast<int>(LightTreeAtlasExtent.height));
    EXPECT_EQ(components, 3);
    stbi_image_free(decoded);
}

#else

TEST(LightTreeImageTest, WritesStableNormalizedDistributionAtlas) {
    GTEST_SKIP() << "PNG preview support is disabled explicitly.";
}

#endif

} // namespace
} // namespace blackframe::renderer
