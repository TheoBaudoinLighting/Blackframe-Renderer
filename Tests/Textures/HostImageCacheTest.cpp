#include <Blackframe/Renderer/ExrWriter.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/HostImageCache.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace blackframe::renderer {
namespace {

inline constexpr auto DataTextureSpace = TextureColorSpace::data;
inline constexpr auto SrgbTextureSpace = TextureColorSpace::srgb;
inline constexpr auto WorkingTextureSpace = TextureColorSpace::scene_linear_srgb;

[[nodiscard]] std::filesystem::path fixture_path() {
    return std::filesystem::path{BLACKFRAME_HOST_IMAGE_TEST_FIXTURE};
}

[[nodiscard]] std::filesystem::path artifact_path(const std::string_view name) {
    return std::filesystem::path{BLACKFRAME_HOST_IMAGE_TEST_OUTPUT_DIR} / name;
}

[[nodiscard]] ExrRunMetadata test_metadata() {
    return ExrRunMetadata{
        .scene = "scenes/S00_ImagesSynthetic.json",
        .seed = 7U,
        .commit = "0123456789abcdef",
        .options = "host-image-cache-test",
        .backend = "scalar_ref",
        .capabilities = "openimageio_host_cache",
        .asset_hashes = "synthetic=fixture",
    };
}

[[nodiscard]] std::filesystem::path write_exr_fixture() {
    const auto output = artifact_path("host-image-rgb32f.exr");
    auto film = Film::create(RenderExtent{.width = 3U, .height = 2U});
    EXPECT_TRUE(film.has_value());
    if (!film) {
        return output;
    }
    constexpr auto pixels = std::array{
        LinearRGB{.red = -0.25F, .green = 0.0F, .blue = 2.0F},
        LinearRGB{.red = 1.0F, .green = 4.0F, .blue = 8.0F},
        LinearRGB{.red = 0.125F, .green = -16.0F, .blue = 32.0F},
        LinearRGB{.red = 64.0F, .green = 0.5F, .blue = -0.5F},
        LinearRGB{.red = 3.0F, .green = 5.0F, .blue = 7.0F},
        LinearRGB{.red = -8.0F, .green = 0.25F, .blue = 0.0625F},
    };
    auto index = std::size_t{};
    for (auto y = std::uint32_t{}; y < 2U; ++y) {
        for (auto x = std::uint32_t{}; x < 3U; ++x) {
            EXPECT_TRUE(film->add_sample(x, y, pixels[index++], 1.0F).has_value());
        }
    }
    std::error_code cleanup_error;
    std::filesystem::remove(output, cleanup_error);
    EXPECT_TRUE(write_scene_linear_exr(*film, output, test_metadata()).has_value());
    return output;
}

[[nodiscard]] std::filesystem::path write_png_fixture() {
    const auto output = artifact_path("host-image-rgba8.png");
    constexpr auto encoded = std::array<std::uint8_t, 95>{
        0x89U, 0x50U, 0x4EU, 0x47U, 0x0DU, 0x0AU, 0x1AU, 0x0AU, 0x00U, 0x00U, 0x00U, 0x0DU,
        0x49U, 0x48U, 0x44U, 0x52U, 0x00U, 0x00U, 0x00U, 0x02U, 0x00U, 0x00U, 0x00U, 0x01U,
        0x08U, 0x06U, 0x00U, 0x00U, 0x00U, 0xF4U, 0x22U, 0x7FU, 0x8AU, 0x00U, 0x00U, 0x00U,
        0x09U, 0x70U, 0x48U, 0x59U, 0x73U, 0x00U, 0x00U, 0x00U, 0x01U, 0x00U, 0x00U, 0x00U,
        0x01U, 0x00U, 0x4FU, 0x25U, 0xC4U, 0xD6U, 0x00U, 0x00U, 0x00U, 0x11U, 0x49U, 0x44U,
        0x41U, 0x54U, 0x78U, 0x9CU, 0x63U, 0x3CU, 0x91U, 0x62U, 0x54U, 0xCFU, 0xC0U, 0xC0U,
        0xC0U, 0x00U, 0x00U, 0x0CU, 0xB5U, 0x01U, 0xDFU, 0xF8U, 0x81U, 0x50U, 0x79U, 0x00U,
        0x00U, 0x00U, 0x00U, 0x49U, 0x45U, 0x4EU, 0x44U, 0xAEU, 0x42U, 0x60U, 0x82U,
    };
    auto stream = std::ofstream{output, std::ios::binary | std::ios::trunc};
    EXPECT_TRUE(stream.is_open());
    stream.write(reinterpret_cast<const char*>(encoded.data()),
                 static_cast<std::streamsize>(encoded.size()));
    EXPECT_TRUE(stream.good());
    return output;
}

[[nodiscard]] std::filesystem::path write_second_pnm_fixture() {
    const auto output = artifact_path("host-image-second.ppm");
    auto stream = std::ofstream{output, std::ios::binary | std::ios::trunc};
    EXPECT_TRUE(stream.is_open());
    stream << "P3\n2 2\n255\n0 0 0 255 0 255\n0 255 255 255 255 0\n";
    EXPECT_TRUE(stream.good());
    return output;
}

[[nodiscard]] std::filesystem::path write_known_srgb_pnm_fixture() {
    const auto output = artifact_path("host-image-known-srgb.ppm");
    auto stream = std::ofstream{output, std::ios::binary | std::ios::trunc};
    EXPECT_TRUE(stream.is_open());
    stream << "P3\n2 1\n255\n0 10 11 128 255 50\n";
    EXPECT_TRUE(stream.good());
    return output;
}

[[nodiscard]] std::filesystem::path write_grayscale_pnm_fixture() {
    const auto output = artifact_path("host-image-grayscale.pgm");
    auto stream = std::ofstream{output, std::ios::binary | std::ios::trunc};
    EXPECT_TRUE(stream.is_open());
    stream << "P2\n1 1\n255\n128\n";
    EXPECT_TRUE(stream.good());
    return output;
}

[[nodiscard]] std::filesystem::path write_pfm_fixture() {
    static_assert(std::endian::native == std::endian::little ||
                  std::endian::native == std::endian::big);
    const auto output = artifact_path("host-image-rgb32f.pfm");
    auto stream = std::ofstream{output, std::ios::binary | std::ios::trunc};
    EXPECT_TRUE(stream.is_open());
    if constexpr (std::endian::native == std::endian::little) {
        stream << "PF\n2 1\n-1.0\n";
    } else {
        stream << "PF\n2 1\n1.0\n";
    }
    constexpr auto pixels = std::array{-0.25F, 0.5F, 2.0F, 4.0F, 8.0F, 16.0F};
    stream.write(reinterpret_cast<const char*>(pixels.data()),
                 static_cast<std::streamsize>(sizeof(pixels)));
    EXPECT_TRUE(stream.good());
    return output;
}

[[nodiscard]] std::filesystem::path write_unrepresentable_srgb_pfm_fixture() {
    const auto output = artifact_path("host-image-srgb-overflow.pfm");
    auto stream = std::ofstream{output, std::ios::binary | std::ios::trunc};
    EXPECT_TRUE(stream.is_open());
    if constexpr (std::endian::native == std::endian::little) {
        stream << "PF\n1 1\n-1.0\n";
    } else {
        stream << "PF\n1 1\n1.0\n";
    }
    constexpr auto pixels = std::array{std::numeric_limits<TransportScalar>::max(), 0.0F, 0.0F};
    stream.write(reinterpret_cast<const char*>(pixels.data()),
                 static_cast<std::streamsize>(sizeof(pixels)));
    EXPECT_TRUE(stream.good());
    return output;
}

[[nodiscard]] std::filesystem::path write_snapshot_pnm_fixture(const bool alternate) {
    const auto output = artifact_path("host-image-snapshot.ppm");
    auto stream = std::ofstream{output, std::ios::binary | std::ios::trunc};
    EXPECT_TRUE(stream.is_open());
    stream << (alternate ? "P3\n1 1\n255\n0 0 0\n" : "P3\n1 1\n255\n255 64 16\n");
    EXPECT_TRUE(stream.good());
    return output;
}

TEST(HostImageCacheTest, LoadsPnmPixelsAndCanonicalAliasesOnce) {
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value()) << cache.error().message;

    const auto first = cache->load(fixture_path(), DataTextureSpace);
    ASSERT_TRUE(first.has_value()) << first.error().message;
    EXPECT_EQ((*first)->format_name(), "pnm");
    EXPECT_EQ((*first)->source_path(), std::filesystem::canonical(fixture_path()));
    EXPECT_EQ((*first)->source_color_space(), DataTextureSpace);
    EXPECT_EQ((*first)->storage_color_space(), DataTextureSpace);
    EXPECT_EQ((*first)->origin_x(), 0);
    EXPECT_EQ((*first)->origin_y(), 0);
    EXPECT_EQ((*first)->width(), 2U);
    EXPECT_EQ((*first)->height(), 2U);
    EXPECT_EQ((*first)->channel_count(), 3U);
    ASSERT_EQ((*first)->channel_names().size(), 3U);
    EXPECT_EQ((*first)->channel_names()[0], "R");
    EXPECT_EQ((*first)->channel_names()[1], "G");
    EXPECT_EQ((*first)->channel_names()[2], "B");

    constexpr auto bytes = std::array<std::uint8_t, 12>{
        255U, 0U, 0U, 0U, 128U, 255U, 64U, 32U, 16U, 255U, 255U, 255U,
    };
    ASSERT_EQ((*first)->pixels().size(), bytes.size());
    for (auto index = std::size_t{}; index < bytes.size(); ++index) {
        EXPECT_FLOAT_EQ((*first)->pixels()[index], static_cast<float>(bytes[index]) / 255.0F);
    }

    const auto alias = fixture_path().parent_path() / "." / fixture_path().filename();
    const auto second = cache->load(alias, DataTextureSpace);
    ASSERT_TRUE(second.has_value()) << second.error().message;
    EXPECT_EQ(first->get(), second->get());
    const auto entry_count = cache->entry_count();
    ASSERT_TRUE(entry_count.has_value());
    EXPECT_EQ(*entry_count, 1U);
    const auto resident_pixel_bytes = cache->resident_pixel_bytes();
    ASSERT_TRUE(resident_pixel_bytes.has_value());
    EXPECT_EQ(*resident_pixel_bytes, 12U * sizeof(TransportScalar));
}

TEST(HostImageCacheTest, LoadsSceneLinearExrWithoutClampingOrColorConversion) {
    const auto exr_path = write_exr_fixture();
    ASSERT_TRUE(std::filesystem::is_regular_file(exr_path));
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = cache->load(exr_path, WorkingTextureSpace);
    ASSERT_TRUE(image.has_value()) << image.error().message;
    EXPECT_EQ((*image)->format_name(), "openexr");
    EXPECT_EQ((*image)->source_color_space(), WorkingTextureSpace);
    EXPECT_EQ((*image)->storage_color_space(), WorkingTextureSpace);
    EXPECT_EQ((*image)->width(), 3U);
    EXPECT_EQ((*image)->height(), 2U);
    EXPECT_EQ((*image)->channel_count(), 3U);
    ASSERT_EQ((*image)->channel_names().size(), 3U);
    EXPECT_EQ((*image)->channel_names()[0], "R");
    EXPECT_EQ((*image)->channel_names()[1], "G");
    EXPECT_EQ((*image)->channel_names()[2], "B");

    constexpr auto expected = std::array{
        -0.25F, 0.0F, 2.0F,  1.0F, 4.0F, 8.0F, 0.125F, -16.0F, 32.0F,
        64.0F,  0.5F, -0.5F, 3.0F, 5.0F, 7.0F, -8.0F,  0.25F,  0.0625F,
    };
    ASSERT_EQ((*image)->pixels().size(), expected.size());
    for (auto index = std::size_t{}; index < expected.size(); ++index) {
        EXPECT_EQ(std::bit_cast<std::uint32_t>((*image)->pixels()[index]),
                  std::bit_cast<std::uint32_t>(expected[index]))
            << index;
    }
}

TEST(HostImageCacheTest, LoadsUnassociatedRawPngChannelsWithoutDisplayConversion) {
    const auto png_path = write_png_fixture();
    ASSERT_TRUE(std::filesystem::is_regular_file(png_path));
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = cache->load(png_path, DataTextureSpace);
    ASSERT_TRUE(image.has_value()) << image.error().message;
    EXPECT_EQ((*image)->format_name(), "png");
    EXPECT_EQ((*image)->source_color_space(), DataTextureSpace);
    EXPECT_EQ((*image)->storage_color_space(), DataTextureSpace);
    EXPECT_EQ((*image)->width(), 2U);
    EXPECT_EQ((*image)->height(), 1U);
    EXPECT_EQ((*image)->channel_count(), 4U);
    ASSERT_EQ((*image)->channel_names().size(), 4U);
    EXPECT_EQ((*image)->channel_names()[0], "R");
    EXPECT_EQ((*image)->channel_names()[1], "G");
    EXPECT_EQ((*image)->channel_names()[2], "B");
    EXPECT_EQ((*image)->channel_names()[3], "A");

    constexpr auto expected = std::array<std::uint8_t, 8>{
        200U, 100U, 50U, 127U, 200U, 100U, 50U, 127U,
    };
    ASSERT_EQ((*image)->pixels().size(), expected.size());
    for (auto index = std::size_t{}; index < expected.size(); ++index) {
        EXPECT_FLOAT_EQ((*image)->pixels()[index], static_cast<float>(expected[index]) / 255.0F)
            << index;
    }
    EXPECT_FLOAT_EQ((*image)->pixels()[0], 200.0F / 255.0F);
}

TEST(HostImageCacheTest, ConvertsKnownSrgbTextureToWorkingSpaceAndPreservesAlpha) {
    const auto png_path = write_png_fixture();
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());

    const auto raw = cache->load(png_path, DataTextureSpace);
    const auto converted = cache->load(png_path, SrgbTextureSpace);
    ASSERT_TRUE(raw.has_value()) << raw.error().message;
    ASSERT_TRUE(converted.has_value()) << converted.error().message;
    EXPECT_NE(raw->get(), converted->get());
    EXPECT_EQ((*converted)->source_color_space(), SrgbTextureSpace);
    EXPECT_EQ((*converted)->storage_color_space(), WorkingTextureSpace);
    ASSERT_EQ((*converted)->pixels().size(), 8U);

    constexpr auto expected_rgb =
        std::array{0.5775804404296506, 0.12743768043564743, 0.03189603307301153};
    for (auto pixel = std::size_t{}; pixel < 2U; ++pixel) {
        for (auto channel = std::size_t{}; channel < expected_rgb.size(); ++channel) {
            EXPECT_NEAR((*converted)->pixels()[pixel * 4U + channel], expected_rgb[channel], 2.0e-7)
                << pixel << ':' << channel;
        }
        EXPECT_EQ(std::bit_cast<std::uint32_t>((*converted)->pixels()[pixel * 4U + 3U]),
                  std::bit_cast<std::uint32_t>((*raw)->pixels()[pixel * 4U + 3U]));
    }

    const auto repeated = cache->load(png_path, SrgbTextureSpace);
    ASSERT_TRUE(repeated.has_value());
    EXPECT_EQ(repeated->get(), converted->get());
    const auto entry_count = cache->entry_count();
    const auto resident_bytes = cache->resident_pixel_bytes();
    ASSERT_TRUE(entry_count.has_value());
    ASSERT_TRUE(resident_bytes.has_value());
    EXPECT_EQ(*entry_count, 2U);
    EXPECT_EQ(*resident_bytes, 16U * sizeof(TransportScalar));
}

TEST(HostImageCacheTest, UsesBothBranchesOfTheFixedSrgbTransferFunction) {
    const auto srgb_path = write_known_srgb_pnm_fixture();
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = cache->load(srgb_path, SrgbTextureSpace);
    ASSERT_TRUE(image.has_value()) << image.error().message;

    constexpr auto expected =
        std::array{0.0, 0.003035269835488375, 0.003346535763899161, 0.21586050011389926,
                   1.0, 0.03189603307301153};
    ASSERT_EQ((*image)->pixels().size(), expected.size());
    for (auto index = std::size_t{}; index < expected.size(); ++index) {
        EXPECT_NEAR((*image)->pixels()[index], expected[index], 1.0e-7) << index;
    }
}

TEST(HostImageCacheTest, LoadsFloatingPointPfmValuesWithoutClamping) {
    const auto pfm_path = write_pfm_fixture();
    ASSERT_TRUE(std::filesystem::is_regular_file(pfm_path));
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = cache->load(pfm_path, DataTextureSpace);
    ASSERT_TRUE(image.has_value()) << image.error().message;
    EXPECT_EQ((*image)->format_name(), "pnm");
    EXPECT_EQ((*image)->source_color_space(), DataTextureSpace);
    EXPECT_EQ((*image)->storage_color_space(), DataTextureSpace);
    EXPECT_EQ((*image)->width(), 2U);
    EXPECT_EQ((*image)->height(), 1U);
    EXPECT_EQ((*image)->channel_count(), 3U);

    constexpr auto expected = std::array{-0.25F, 0.5F, 2.0F, 4.0F, 8.0F, 16.0F};
    ASSERT_EQ((*image)->pixels().size(), expected.size());
    for (auto index = std::size_t{}; index < expected.size(); ++index) {
        EXPECT_EQ(std::bit_cast<std::uint32_t>((*image)->pixels()[index]),
                  std::bit_cast<std::uint32_t>(expected[index]))
            << index;
    }
}

TEST(HostImageCacheTest, RejectsInvalidOrAmbiguousColorTagsWithoutCaching) {
    const auto grayscale_path = write_grayscale_pnm_fixture();
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());

    const auto invalid = cache->load(
        fixture_path(), static_cast<TextureColorSpace>(std::numeric_limits<std::uint32_t>::max()));
    ASSERT_FALSE(invalid.has_value());
    EXPECT_EQ(invalid.error().code, core::StatusCode::invalid_argument);

    const auto srgb_grayscale = cache->load(grayscale_path, SrgbTextureSpace);
    ASSERT_FALSE(srgb_grayscale.has_value());
    EXPECT_EQ(srgb_grayscale.error().code, core::StatusCode::incompatible);
    const auto linear_grayscale = cache->load(grayscale_path, WorkingTextureSpace);
    ASSERT_FALSE(linear_grayscale.has_value());
    EXPECT_EQ(linear_grayscale.error().code, core::StatusCode::incompatible);

    const auto entry_count = cache->entry_count();
    ASSERT_TRUE(entry_count.has_value());
    EXPECT_EQ(*entry_count, 0U);

    const auto data_grayscale = cache->load(grayscale_path, DataTextureSpace);
    ASSERT_TRUE(data_grayscale.has_value()) << data_grayscale.error().message;
    EXPECT_EQ((*data_grayscale)->source_color_space(), DataTextureSpace);
    EXPECT_EQ((*data_grayscale)->storage_color_space(), DataTextureSpace);
}

TEST(HostImageCacheTest, RejectsUnrepresentableSrgbConversionWithoutClamping) {
    const auto overflow_path = write_unrepresentable_srgb_pfm_fixture();
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());

    const auto converted = cache->load(overflow_path, SrgbTextureSpace);
    ASSERT_FALSE(converted.has_value());
    EXPECT_EQ(converted.error().code, core::StatusCode::incompatible);
    const auto entry_count = cache->entry_count();
    ASSERT_TRUE(entry_count.has_value());
    EXPECT_EQ(*entry_count, 0U);

    const auto linear = cache->load(overflow_path, WorkingTextureSpace);
    ASSERT_TRUE(linear.has_value()) << linear.error().message;
    EXPECT_EQ(std::bit_cast<std::uint32_t>((*linear)->pixels()[0]),
              std::bit_cast<std::uint32_t>(std::numeric_limits<TransportScalar>::max()));
}

TEST(HostImageCacheTest, RetainsCachedImagesAfterCallerHandlesAreReleased) {
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());

    const HostImage* cached_address = nullptr;
    {
        const auto loaded = cache->load(fixture_path(), DataTextureSpace);
        ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
        cached_address = loaded->get();
    }
    ASSERT_NE(cached_address, nullptr);
    const auto entry_count = cache->entry_count();
    ASSERT_TRUE(entry_count.has_value());
    EXPECT_EQ(*entry_count, 1U);

    const auto reloaded = cache->load(fixture_path(), DataTextureSpace);
    ASSERT_TRUE(reloaded.has_value()) << reloaded.error().message;
    EXPECT_EQ(reloaded->get(), cached_address);
}

TEST(HostImageCacheTest, KeepsSnapshotAccessibleAfterSourceChangesAndDisappears) {
    const auto source_path = write_snapshot_pnm_fixture(false);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto initial = cache->load(source_path, DataTextureSpace);
    ASSERT_TRUE(initial.has_value()) << initial.error().message;
    constexpr auto expected = std::array{1.0F, 64.0F / 255.0F, 16.0F / 255.0F};
    ASSERT_EQ((*initial)->pixels().size(), expected.size());

    ASSERT_EQ(write_snapshot_pnm_fixture(true), source_path);
    const auto after_change = cache->load(source_path, DataTextureSpace);
    ASSERT_TRUE(after_change.has_value()) << after_change.error().message;
    EXPECT_EQ(after_change->get(), initial->get());
    EXPECT_TRUE(std::ranges::equal((*after_change)->pixels(), expected));

    const auto srgb_after_change = cache->load(source_path, SrgbTextureSpace);
    ASSERT_TRUE(srgb_after_change.has_value()) << srgb_after_change.error().message;
    constexpr auto expected_srgb = std::array{1.0, 0.05126945837404324, 0.005181516702338386};
    ASSERT_EQ((*srgb_after_change)->pixels().size(), expected_srgb.size());
    for (auto index = std::size_t{}; index < expected_srgb.size(); ++index) {
        EXPECT_NEAR((*srgb_after_change)->pixels()[index], expected_srgb[index], 1.0e-7) << index;
    }

    std::error_code remove_error;
    ASSERT_TRUE(std::filesystem::remove(source_path, remove_error)) << remove_error.message();
    const auto after_removal = cache->load(source_path, DataTextureSpace);
    ASSERT_TRUE(after_removal.has_value()) << after_removal.error().message;
    EXPECT_EQ(after_removal->get(), initial->get());

    const auto linear_after_removal = cache->load(source_path, WorkingTextureSpace);
    ASSERT_TRUE(linear_after_removal.has_value()) << linear_after_removal.error().message;
    ASSERT_EQ((*linear_after_removal)->pixels().size(), expected.size());
    for (auto index = std::size_t{}; index < expected.size(); ++index) {
        EXPECT_EQ(std::bit_cast<std::uint32_t>((*linear_after_removal)->pixels()[index]),
                  std::bit_cast<std::uint32_t>(expected[index]));
    }
    const auto cached_entries = cache->entry_count();
    const auto cached_bytes = cache->resident_pixel_bytes();
    ASSERT_TRUE(cached_entries.has_value());
    ASSERT_TRUE(cached_bytes.has_value());
    EXPECT_EQ(*cached_entries, 3U);
    EXPECT_EQ(*cached_bytes, 6U * sizeof(TransportScalar));

    auto fresh_cache = HostImageCache::create();
    ASSERT_TRUE(fresh_cache.has_value());
    const auto missing = fresh_cache->load(source_path, DataTextureSpace);
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, core::StatusCode::not_found);
}

TEST(HostImageCacheTest, ReturnsOneImmutableHandleAcrossConcurrentLoads) {
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    auto futures = std::vector<std::future<core::Result<HostImageHandle>>>{};
    for (auto index = 0U; index < 8U; ++index) {
        futures.push_back(std::async(std::launch::async, [&cache] {
            return cache->load(fixture_path(), DataTextureSpace);
        }));
    }

    auto first = HostImageHandle{};
    for (auto& future : futures) {
        auto loaded = future.get();
        ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
        if (!first) {
            first = *loaded;
        }
        EXPECT_EQ(first.get(), loaded->get());
    }
    const auto entry_count = cache->entry_count();
    ASSERT_TRUE(entry_count.has_value());
    EXPECT_EQ(*entry_count, 1U);
}

TEST(HostImageCacheTest, RejectsInvalidSourcesAndExhaustedBudgetsWithoutSubstitution) {
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());

    const auto relative = cache->load(std::filesystem::path{"relative.ppm"}, DataTextureSpace);
    ASSERT_FALSE(relative.has_value());
    EXPECT_EQ(relative.error().code, core::StatusCode::invalid_argument);

    const auto missing = cache->load(artifact_path("missing.ppm"), DataTextureSpace);
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, core::StatusCode::not_found);

    const auto directory =
        cache->load(std::filesystem::path{BLACKFRAME_HOST_IMAGE_TEST_OUTPUT_DIR}, DataTextureSpace);
    ASSERT_FALSE(directory.has_value());
    EXPECT_EQ(directory.error().code, core::StatusCode::invalid_argument);

    const auto corrupt_path = artifact_path("corrupt-image.bin");
    {
        auto corrupt = std::ofstream{corrupt_path, std::ios::binary | std::ios::trunc};
        ASSERT_TRUE(corrupt.is_open());
        corrupt << "this is not an image";
    }
    const auto corrupt = cache->load(corrupt_path, DataTextureSpace);
    ASSERT_FALSE(corrupt.has_value());
    EXPECT_EQ(corrupt.error().code, core::StatusCode::incompatible);
    const auto entry_count = cache->entry_count();
    ASSERT_TRUE(entry_count.has_value());
    EXPECT_EQ(*entry_count, 0U);

    auto limits = HostImageCacheLimits{};
    limits.maximum_image_pixel_bytes = 8U;
    auto bounded = HostImageCache::create(limits);
    ASSERT_TRUE(bounded.has_value());
    const auto oversized = bounded->load(fixture_path(), DataTextureSpace);
    ASSERT_FALSE(oversized.has_value());
    EXPECT_EQ(oversized.error().code, core::StatusCode::resource_exhausted);
    const auto bounded_entry_count = bounded->entry_count();
    ASSERT_TRUE(bounded_entry_count.has_value());
    EXPECT_EQ(*bounded_entry_count, 0U);
}

TEST(HostImageCacheTest, RejectsInvalidCacheLimitsBeforeAnyLoad) {
    auto limits = HostImageCacheLimits{};
    limits.maximum_entries = 0U;
    const auto zero_entries = HostImageCache::create(limits);
    ASSERT_FALSE(zero_entries.has_value());
    EXPECT_EQ(zero_entries.error().code, core::StatusCode::invalid_argument);

    limits = HostImageCacheLimits{};
    limits.maximum_image_pixel_bytes = limits.maximum_resident_pixel_bytes + 1U;
    const auto inverted_bytes = HostImageCache::create(limits);
    ASSERT_FALSE(inverted_bytes.has_value());
    EXPECT_EQ(inverted_bytes.error().code, core::StatusCode::invalid_argument);
}

TEST(HostImageCacheTest, RejectsUnsupportedSuffixesAndMismatchedPayloads) {
    const auto unsupported_path = artifact_path("valid-pnm-payload.bin");
    const auto mismatched_path = artifact_path("pnm-payload-with-png-suffix.png");
    std::error_code copy_error;
    ASSERT_TRUE(std::filesystem::copy_file(fixture_path(), unsupported_path,
                                           std::filesystem::copy_options::overwrite_existing,
                                           copy_error))
        << copy_error.message();
    copy_error.clear();
    ASSERT_TRUE(std::filesystem::copy_file(fixture_path(), mismatched_path,
                                           std::filesystem::copy_options::overwrite_existing,
                                           copy_error))
        << copy_error.message();

    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto unsupported = cache->load(unsupported_path, DataTextureSpace);
    ASSERT_FALSE(unsupported.has_value());
    EXPECT_EQ(unsupported.error().code, core::StatusCode::incompatible);
    const auto mismatched = cache->load(mismatched_path, DataTextureSpace);
    ASSERT_FALSE(mismatched.has_value());
    EXPECT_EQ(mismatched.error().code, core::StatusCode::incompatible);

    const auto entry_count = cache->entry_count();
    ASSERT_TRUE(entry_count.has_value());
    EXPECT_EQ(*entry_count, 0U);
}

TEST(HostImageCacheTest, ReportsUseAfterMoveExplicitly) {
    auto source = HostImageCache::create();
    ASSERT_TRUE(source.has_value());
    auto destination = std::move(*source);

    const auto load = source->load(fixture_path(), DataTextureSpace);
    ASSERT_FALSE(load.has_value());
    EXPECT_EQ(load.error().code, core::StatusCode::incompatible);
    const auto count = source->entry_count();
    ASSERT_FALSE(count.has_value());
    EXPECT_EQ(count.error().code, core::StatusCode::incompatible);
    const auto bytes = source->resident_pixel_bytes();
    ASSERT_FALSE(bytes.has_value());
    EXPECT_EQ(bytes.error().code, core::StatusCode::incompatible);
    const auto cache_limits = source->limits();
    ASSERT_FALSE(cache_limits.has_value());
    EXPECT_EQ(cache_limits.error().code, core::StatusCode::incompatible);

    const auto loaded = destination.load(fixture_path(), DataTextureSpace);
    ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
}

TEST(HostImageCacheTest, EnforcesEntryAndResidentByteLimits) {
    const auto second_fixture = write_second_pnm_fixture();
    ASSERT_TRUE(std::filesystem::is_regular_file(second_fixture));

    auto entry_limits = HostImageCacheLimits{};
    entry_limits.maximum_entries = 1U;
    auto entry_bounded = HostImageCache::create(entry_limits);
    ASSERT_TRUE(entry_bounded.has_value());
    ASSERT_TRUE(entry_bounded->load(fixture_path(), DataTextureSpace).has_value());
    const auto entry_overflow = entry_bounded->load(second_fixture, DataTextureSpace);
    ASSERT_FALSE(entry_overflow.has_value());
    EXPECT_EQ(entry_overflow.error().code, core::StatusCode::resource_exhausted);
    const auto entry_count = entry_bounded->entry_count();
    ASSERT_TRUE(entry_count.has_value());
    EXPECT_EQ(*entry_count, 1U);

    auto resident_limits = HostImageCacheLimits{};
    resident_limits.maximum_image_pixel_bytes = 48U;
    resident_limits.maximum_resident_pixel_bytes = 64U;
    auto resident_bounded = HostImageCache::create(resident_limits);
    ASSERT_TRUE(resident_bounded.has_value());
    ASSERT_TRUE(resident_bounded->load(fixture_path(), DataTextureSpace).has_value());
    const auto resident_overflow = resident_bounded->load(second_fixture, DataTextureSpace);
    ASSERT_FALSE(resident_overflow.has_value());
    EXPECT_EQ(resident_overflow.error().code, core::StatusCode::resource_exhausted);
    const auto resident_entry_count = resident_bounded->entry_count();
    ASSERT_TRUE(resident_entry_count.has_value());
    EXPECT_EQ(*resident_entry_count, 1U);
    const auto resident_pixel_bytes = resident_bounded->resident_pixel_bytes();
    ASSERT_TRUE(resident_pixel_bytes.has_value());
    EXPECT_EQ(*resident_pixel_bytes, 48U);
}

} // namespace
} // namespace blackframe::renderer
