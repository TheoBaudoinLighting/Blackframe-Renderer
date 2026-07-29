#include <Blackframe/Renderer/PngWriter.hpp>
#include <gtest/gtest.h>
#define STB_IMAGE_IMPLEMENTATION
#include <array>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stb_image.h>
#include <string>
#include <system_error>

namespace blackframe::renderer {
namespace {

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

[[nodiscard]] bool is_checksum_invocation() {
    return checksum_output_path().has_value();
}

TEST(PngWriterTest, WritesFixedDisplayTransformForSyntheticCrop) {
#if BLACKFRAME_HAS_PNG_PREVIEW
    const auto crop = FilmCrop{.minimum_x = 1, .minimum_y = 1, .maximum_x = 4, .maximum_y = 3};
    auto film = Film::create(RenderExtent{.width = 4, .height = 3}, crop);
    ASSERT_TRUE(film.has_value());

    constexpr auto source = std::array{
        LinearRGB{.red = -1.0F, .green = 0.0F, .blue = 0.0031308F},
        LinearRGB{.red = 0.18F, .green = 0.5F, .blue = 1.0F},
        LinearRGB{.red = 2.0F, .green = 0.25F, .blue = 0.75F},
        LinearRGB{.red = 0.01F, .green = 0.04F, .blue = 0.09F},
        LinearRGB{.red = 0.16F, .green = 0.36F, .blue = 0.64F},
        LinearRGB{.red = 0.81F, .green = 1.5F, .blue = -0.5F},
    };
    auto source_index = std::size_t{};
    for (auto y = crop.minimum_y; y < crop.maximum_y; ++y) {
        for (auto x = crop.minimum_x; x < crop.maximum_x; ++x) {
            ASSERT_TRUE(film->add_sample(x, y, source[source_index++], 1.0F).has_value());
        }
    }

    const auto output_path = artifact_path("fixed-display-preview.png");
    std::error_code cleanup_error;
    std::filesystem::remove(output_path, cleanup_error);
    ASSERT_TRUE(write_png_preview(*film, output_path).has_value());
    ASSERT_TRUE(std::filesystem::is_regular_file(output_path));

    auto width = int{};
    auto height = int{};
    auto components = int{};
    auto* const decoded = stbi_load(output_path.string().c_str(), &width, &height, &components, 3);
    ASSERT_NE(decoded, nullptr);
    EXPECT_EQ(width, 3);
    EXPECT_EQ(height, 2);
    EXPECT_EQ(components, 3);

    constexpr auto expected = std::array<std::uint8_t, 18>{
        0, 0, 10, 118, 188, 255, 255, 137, 225, 25, 56, 85, 111, 162, 209, 232, 255, 0,
    };
    for (auto index = std::size_t{}; index < expected.size(); ++index) {
        EXPECT_EQ(decoded[index], expected[index]) << "channel byte " << index;
    }
    stbi_image_free(decoded);

    if (!is_checksum_invocation()) {
        EXPECT_TRUE(std::filesystem::remove(output_path));
    }
#else
    GTEST_SKIP() << "PNG preview support is disabled explicitly.";
#endif
}

TEST(PngWriterTest, RejectsIncompleteFilmAndNonPngPathsWithoutFallback) {
#if BLACKFRAME_HAS_PNG_PREVIEW
    auto film = Film::create(RenderExtent{.width = 1, .height = 1});
    ASSERT_TRUE(film.has_value());

    const auto output_path = artifact_path("rejected-preview.png");
    const auto wrong_extension_path = artifact_path("rejected-preview.exr");
    std::error_code cleanup_error;
    std::filesystem::remove(output_path, cleanup_error);
    std::filesystem::remove(wrong_extension_path, cleanup_error);

    const auto unresolved = write_png_preview(*film, output_path);
    ASSERT_FALSE(unresolved.has_value());
    EXPECT_EQ(unresolved.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(output_path));

    ASSERT_TRUE(film->add_sample(0, 0, LinearRGB{.red = 1.0F}, 1.0F).has_value());
    const auto wrong_extension = write_png_preview(*film, wrong_extension_path);
    ASSERT_FALSE(wrong_extension.has_value());
    EXPECT_EQ(wrong_extension.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(wrong_extension_path));

    const auto relative_path =
        write_png_preview(*film, std::filesystem::path{"relative-preview.png"});
    ASSERT_FALSE(relative_path.has_value());
    EXPECT_EQ(relative_path.error().code, core::StatusCode::invalid_argument);
#else
    GTEST_SKIP() << "PNG preview support is disabled explicitly.";
#endif
}

} // namespace
} // namespace blackframe::renderer
