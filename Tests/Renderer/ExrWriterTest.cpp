#include <Blackframe/Renderer/ExrWriter.hpp>
#include <IexBaseExc.h>
#include <ImathBox.h>
#include <ImfChannelList.h>
#include <ImfChromaticities.h>
#include <ImfChromaticitiesAttribute.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfInputFile.h>
#include <ImfStringAttribute.h>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace blackframe::renderer {
namespace {

[[nodiscard]] std::filesystem::path artifact_path(const std::string_view name) {
    return std::filesystem::path{BLACKFRAME_RENDERER_TEST_OUTPUT_DIR} / name;
}

[[nodiscard]] std::string utf8_path(const std::filesystem::path& path) {
    const auto encoded = path.u8string();
    return std::string{reinterpret_cast<const char*>(encoded.data()), encoded.size()};
}

[[nodiscard]] ExrRunMetadata test_metadata() {
    return ExrRunMetadata{
        .scene = "scenes/S00_ImagesSynthetic.json",
        .seed = std::numeric_limits<std::uint64_t>::max(),
        .commit = "0123456789abcdef",
        .options = "spp=8;depth=4;crop=1,1,4,3",
        .backend = "scalar_ref",
        .capabilities = "film_float32,exr_openexr",
        .asset_hashes = "scene=4df6c03f",
    };
}

TEST(ExrWriterTest, ReopensExactSceneLinearFloatPixelsAndStableRunMetadata) {
    constexpr auto extent = RenderExtent{.width = 4, .height = 3};
    constexpr auto crop = FilmCrop{.minimum_x = 1, .minimum_y = 1, .maximum_x = 4, .maximum_y = 3};
    auto film = Film::create(extent, crop);
    ASSERT_TRUE(film.has_value());

    const auto expected = std::vector{
        LinearRGB{.red = 0.1F, .green = -0.25F, .blue = 1.0F},
        LinearRGB{.red = 1.0F / 3.0F, .green = 42.5F, .blue = -8.0F},
        LinearRGB{.red = 100000.25F, .green = 0.00001F, .blue = 3.1415927F},
        LinearRGB{.red = -1024.5F, .green = 65536.5F, .blue = 0.0F},
        LinearRGB{.red = 2.0F, .green = 4.0F, .blue = 8.0F},
        LinearRGB{.red = 0.0078125F, .green = -0.03125F, .blue = 16.0F},
    };

    auto index = std::size_t{};
    for (auto y = crop.minimum_y; y < crop.maximum_y; ++y) {
        for (auto x = crop.minimum_x; x < crop.maximum_x; ++x) {
            ASSERT_TRUE(film->add_sample(x, y, expected[index], 1.0F).has_value());
            ++index;
        }
    }

    const auto output_path = artifact_path("scene-linear-roundtrip.exr");
    std::error_code cleanup_error;
    std::filesystem::remove(output_path, cleanup_error);
    ASSERT_TRUE(write_scene_linear_exr(*film, output_path, test_metadata()).has_value());
    ASSERT_TRUE(std::filesystem::is_regular_file(output_path));
    ASSERT_GT(std::filesystem::file_size(output_path), 0U);

    auto input = OPENEXR_IMF_NAMESPACE::InputFile{utf8_path(output_path).c_str()};
    const auto& header = input.header();
    const auto expected_display_window =
        IMATH_NAMESPACE::Box2i{IMATH_NAMESPACE::V2i{0, 0}, IMATH_NAMESPACE::V2i{3, 2}};
    const auto expected_data_window =
        IMATH_NAMESPACE::Box2i{IMATH_NAMESPACE::V2i{1, 1}, IMATH_NAMESPACE::V2i{3, 2}};
    EXPECT_EQ(header.compression(), OPENEXR_IMF_NAMESPACE::ZIP_COMPRESSION);
    EXPECT_EQ(header.lineOrder(), OPENEXR_IMF_NAMESPACE::INCREASING_Y);
    EXPECT_TRUE(header.displayWindow() == expected_display_window);
    EXPECT_TRUE(header.dataWindow() == expected_data_window);

    auto channel_count = std::size_t{};
    for (auto channel = header.channels().begin(); channel != header.channels().end(); ++channel) {
        ++channel_count;
    }
    EXPECT_EQ(channel_count, 3U);
    EXPECT_EQ(header.channels().findChannel("A"), nullptr);
    for (const auto* const name : {"B", "G", "R"}) {
        const auto* const channel = header.channels().findChannel(name);
        ASSERT_NE(channel, nullptr);
        EXPECT_EQ(channel->type, OPENEXR_IMF_NAMESPACE::FLOAT);
        EXPECT_EQ(channel->xSampling, 1);
        EXPECT_EQ(channel->ySampling, 1);
    }

    const auto& chromaticities =
        header.typedAttribute<OPENEXR_IMF_NAMESPACE::ChromaticitiesAttribute>("chromaticities")
            .value();
    EXPECT_TRUE(chromaticities == OPENEXR_IMF_NAMESPACE::Chromaticities{});

    const auto expect_string_attribute = [&header](const char* const name,
                                                   const std::string_view expected_value) {
        const auto& actual =
            header.typedAttribute<OPENEXR_IMF_NAMESPACE::StringAttribute>(name).value();
        EXPECT_EQ(actual, expected_value);
    };
    expect_string_attribute("blackframe.run_metadata_version", "1");
    expect_string_attribute("blackframe.color_space", "scene-linear-srgb");
    expect_string_attribute("blackframe.scene", "scenes/S00_ImagesSynthetic.json");
    expect_string_attribute("blackframe.seed", "18446744073709551615");
    expect_string_attribute("blackframe.commit", "0123456789abcdef");
    expect_string_attribute("blackframe.options", "spp=8;depth=4;crop=1,1,4,3");
    expect_string_attribute("blackframe.backend", "scalar_ref");
    expect_string_attribute("blackframe.capabilities", "film_float32,exr_openexr");
    expect_string_attribute("blackframe.asset_hashes", "scene=4df6c03f");

    auto blue = std::vector<TransportScalar>(expected.size());
    auto green = std::vector<TransportScalar>(expected.size());
    auto red = std::vector<TransportScalar>(expected.size());
    auto frame_buffer = OPENEXR_IMF_NAMESPACE::FrameBuffer{};
    frame_buffer.insert("B", OPENEXR_IMF_NAMESPACE::Slice::Make(OPENEXR_IMF_NAMESPACE::FLOAT,
                                                                blue.data(), header.dataWindow()));
    frame_buffer.insert("G", OPENEXR_IMF_NAMESPACE::Slice::Make(OPENEXR_IMF_NAMESPACE::FLOAT,
                                                                green.data(), header.dataWindow()));
    frame_buffer.insert("R", OPENEXR_IMF_NAMESPACE::Slice::Make(OPENEXR_IMF_NAMESPACE::FLOAT,
                                                                red.data(), header.dataWindow()));
    input.setFrameBuffer(frame_buffer);
    input.readPixels(header.dataWindow().min.y, header.dataWindow().max.y);

    for (auto pixel = std::size_t{}; pixel < expected.size(); ++pixel) {
        EXPECT_FLOAT_EQ(red[pixel], expected[pixel].red);
        EXPECT_FLOAT_EQ(green[pixel], expected[pixel].green);
        EXPECT_FLOAT_EQ(blue[pixel], expected[pixel].blue);
    }

    EXPECT_TRUE(std::filesystem::remove(output_path));
}

TEST(ExrWriterTest, RejectsIncompleteInputMetadataAndNonExrPathsWithoutFallback) {
    auto film = Film::create(RenderExtent{.width = 1, .height = 1});
    ASSERT_TRUE(film.has_value());

    const auto output_path = artifact_path("rejected-output.exr");
    const auto wrong_extension_path = artifact_path("rejected-output.png");
    std::error_code cleanup_error;
    std::filesystem::remove(output_path, cleanup_error);
    std::filesystem::remove(wrong_extension_path, cleanup_error);

    const auto unresolved = write_scene_linear_exr(*film, output_path, test_metadata());
    ASSERT_FALSE(unresolved.has_value());
    EXPECT_EQ(unresolved.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(output_path));

    ASSERT_TRUE(film->add_sample(0, 0, LinearRGB{.red = 1.0F}, 1.0F).has_value());
    auto missing_metadata = test_metadata();
    missing_metadata.scene.clear();
    const auto incomplete_metadata = write_scene_linear_exr(*film, output_path, missing_metadata);
    ASSERT_FALSE(incomplete_metadata.has_value());
    EXPECT_EQ(incomplete_metadata.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(output_path));

    const auto wrong_extension =
        write_scene_linear_exr(*film, wrong_extension_path, test_metadata());
    ASSERT_FALSE(wrong_extension.has_value());
    EXPECT_EQ(wrong_extension.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(wrong_extension_path));

    const auto relative_path =
        write_scene_linear_exr(*film, std::filesystem::path{"relative.exr"}, test_metadata());
    ASSERT_FALSE(relative_path.has_value());
    EXPECT_EQ(relative_path.error().code, core::StatusCode::invalid_argument);
}

} // namespace
} // namespace blackframe::renderer
