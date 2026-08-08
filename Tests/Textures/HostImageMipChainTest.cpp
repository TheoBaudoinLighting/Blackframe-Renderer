#include <Blackframe/Renderer/HostImageCache.hpp>
#include <Blackframe/Renderer/HostImageMipChain.hpp>
#if defined(BLACKFRAME_HOST_IMAGE_FILTER_PNG)
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/PngWriter.hpp>
#endif
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace blackframe::renderer {
namespace {

[[nodiscard]] std::filesystem::path mip_artifact_path(const std::string_view name) {
    return std::filesystem::path{BLACKFRAME_HOST_IMAGE_TEST_OUTPUT_DIR} / name;
}

[[nodiscard]] std::filesystem::path
write_pfm_fixture(const std::string_view name, const std::span<const TransportScalar> values) {
    const auto output = mip_artifact_path(name);
    auto stream = std::ofstream{output, std::ios::binary | std::ios::trunc};
    EXPECT_TRUE(stream.is_open());
    if constexpr (std::endian::native == std::endian::little) {
        stream << "Pf\n" << values.size() << " 1\n-1.0\n";
    } else {
        stream << "Pf\n" << values.size() << " 1\n1.0\n";
    }
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size_bytes()));
    EXPECT_TRUE(stream.good());
    return output;
}

[[nodiscard]] std::filesystem::path
write_ppm_fixture(const std::string_view name, const std::uint32_t width,
                  const std::uint32_t height,
                  const std::span<const std::array<std::uint32_t, 3>> pixels) {
    const auto output = mip_artifact_path(name);
    auto stream = std::ofstream{output, std::ios::binary | std::ios::trunc};
    EXPECT_TRUE(stream.is_open());
    EXPECT_EQ(pixels.size(), static_cast<std::size_t>(width) * height);
    stream << "P3\n" << width << ' ' << height << "\n255\n";
    for (const auto& pixel : pixels) {
        stream << pixel[0] << ' ' << pixel[1] << ' ' << pixel[2] << '\n';
    }
    EXPECT_TRUE(stream.good());
    return output;
}

[[nodiscard]] std::filesystem::path write_pgm_fixture(const std::string_view name,
                                                      const std::uint32_t width,
                                                      const std::uint32_t height,
                                                      const std::span<const std::uint32_t> pixels) {
    const auto output = mip_artifact_path(name);
    auto stream = std::ofstream{output, std::ios::binary | std::ios::trunc};
    EXPECT_TRUE(stream.is_open());
    EXPECT_EQ(pixels.size(), static_cast<std::size_t>(width) * height);
    stream << "P2\n" << width << ' ' << height << "\n255\n";
    for (const auto pixel : pixels) {
        stream << pixel << '\n';
    }
    EXPECT_TRUE(stream.good());
    return output;
}

[[nodiscard]] HostImageHandle
load_image(HostImageCache& cache, const std::filesystem::path& path,
           const TextureColorSpace color_space = TextureColorSpace::data) {
    const auto loaded = cache.load(path, color_space);
    EXPECT_TRUE(loaded.has_value()) << loaded.error().message;
    return loaded ? *loaded : HostImageHandle{};
}

[[nodiscard]] std::vector<std::array<std::uint32_t, 3>> affine_rgb_pixels() {
    auto pixels = std::vector<std::array<std::uint32_t, 3>>{};
    pixels.reserve(15U);
    for (auto y = std::uint32_t{}; y < 5U; ++y) {
        for (auto x = std::uint32_t{}; x < 3U; ++x) {
            const auto value = x + 10U * y;
            pixels.push_back({value, value, value});
        }
    }
    return pixels;
}

[[nodiscard]] std::vector<std::uint32_t> checker_pixels(const std::uint32_t width,
                                                        const std::uint32_t height) {
    auto pixels = std::vector<std::uint32_t>{};
    pixels.reserve(static_cast<std::size_t>(width) * height);
    for (auto y = std::uint32_t{}; y < height; ++y) {
        for (auto x = std::uint32_t{}; x < width; ++x) {
            pixels.push_back((x + y) % 2U == 0U ? 0U : 255U);
        }
    }
    return pixels;
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar>
sample_trilinear(const HostImageMipChain& chain, const Point2T<Scalar> uv, const Scalar lod,
                 const std::uint32_t channel, const TextureWrapMode u_wrap = TextureWrapMode::clamp,
                 const TextureWrapMode v_wrap = TextureWrapMode::clamp) {
    return filter_host_image_trilinear_channel(chain, uv, lod, channel, u_wrap, v_wrap);
}

TEST(HostImageMipChainTest, BuildsOddAffineLevelsWithoutDroppingEdges) {
    const auto pixels = affine_rgb_pixels();
    const auto path = write_ppm_fixture("texture-mip-affine-3x5.ppm", 3U, 5U, pixels);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value()) << cache.error().message;
    const auto image = load_image(*cache, path);
    ASSERT_TRUE(image);

    const auto generated = HostImageMipChain::generate(image);
    ASSERT_TRUE(generated.has_value()) << generated.error().message;
    EXPECT_EQ((*generated)->source_image().get(), image.get());
    EXPECT_EQ((*generated)->level_count(), 3U);

    const auto level0 = (*generated)->level(0U);
    const auto level1 = (*generated)->level(1U);
    const auto level2 = (*generated)->level(2U);
    ASSERT_TRUE(level0.has_value()) << level0.error().message;
    ASSERT_TRUE(level1.has_value()) << level1.error().message;
    ASSERT_TRUE(level2.has_value()) << level2.error().message;
    ASSERT_TRUE(*level0);
    ASSERT_TRUE(*level1);
    ASSERT_TRUE(*level2);
    EXPECT_EQ(level0->get(), image.get());
    EXPECT_EQ((*level1)->width(), 1U);
    EXPECT_EQ((*level1)->height(), 2U);
    EXPECT_EQ((*level2)->width(), 1U);
    EXPECT_EQ((*level2)->height(), 1U);
    EXPECT_EQ((*level1)->channel_count(), 3U);
    EXPECT_EQ((*level2)->channel_count(), 3U);

    constexpr auto expected_level1 = std::array{9.0F / 255.0F, 33.0F / 255.0F};
    constexpr auto expected_top = 21.0F / 255.0F;
    constexpr auto tolerance = 32.0F * std::numeric_limits<TransportScalar>::epsilon();
    ASSERT_EQ((*level1)->pixels().size(), 6U);
    ASSERT_EQ((*level2)->pixels().size(), 3U);
    for (auto row = std::size_t{}; row < expected_level1.size(); ++row) {
        for (auto channel = std::size_t{}; channel < 3U; ++channel) {
            EXPECT_NEAR((*level1)->pixels()[row * 3U + channel], expected_level1[row], tolerance);
        }
    }
    for (auto channel = std::size_t{}; channel < 3U; ++channel) {
        EXPECT_NEAR((*level2)->pixels()[channel], expected_top, tolerance);
    }

    EXPECT_EQ((*generated)->generated_pixel_bytes(), 36U);
    EXPECT_EQ((*generated)->total_pixel_bytes(), image->pixel_byte_size() + 36U);
}

TEST(HostImageMipChainTest, UsesExactAreaWeightsForOddOneDimensionalLevels) {
    constexpr auto values = std::array<TransportScalar, 5>{0.0F, 1.0F, 2.0F, 3.0F, 4.0F};
    const auto path = write_pfm_fixture("texture-mip-ramp-5x1.pfm", values);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = load_image(*cache, path);
    ASSERT_TRUE(image);

    const auto generated = HostImageMipChain::generate(image);
    ASSERT_TRUE(generated.has_value()) << generated.error().message;
    ASSERT_EQ((*generated)->level_count(), 3U);
    const auto level1 = (*generated)->level(1U);
    const auto level2 = (*generated)->level(2U);
    ASSERT_TRUE(level1.has_value()) << level1.error().message;
    ASSERT_TRUE(level2.has_value()) << level2.error().message;
    ASSERT_EQ((*level1)->pixels().size(), 2U);
    ASSERT_EQ((*level2)->pixels().size(), 1U);
    constexpr auto tolerance = 16.0F * std::numeric_limits<TransportScalar>::epsilon();
    EXPECT_NEAR((*level1)->pixels()[0], 0.8F, tolerance);
    EXPECT_NEAR((*level1)->pixels()[1], 3.2F, tolerance);
    EXPECT_NEAR((*level2)->pixels()[0], 2.0F, tolerance);
    EXPECT_EQ((*generated)->generated_pixel_bytes(), 3U * sizeof(TransportScalar));
    EXPECT_EQ((*generated)->total_pixel_bytes(), 8U * sizeof(TransportScalar));
}

TEST(HostImageMipChainTest, TrilinearMinificationConvergesOnCheckerAverage) {
    const auto pixels = checker_pixels(8U, 8U);
    const auto path = write_pgm_fixture("texture-mip-checker-8x8.pgm", 8U, 8U, pixels);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = load_image(*cache, path);
    ASSERT_TRUE(image);
    const auto generated = HostImageMipChain::generate(image);
    ASSERT_TRUE(generated.has_value()) << generated.error().message;
    ASSERT_EQ((*generated)->level_count(), 4U);

    constexpr auto uv_transport = Point2{.x = 0.0625F, .y = 0.0625F};
    constexpr auto uv_reference = ReferencePoint2{.x = 0.0625, .y = 0.0625};
    const auto base = sample_trilinear(**generated, uv_transport, 0.0F, 0U);
    const auto half = sample_trilinear(**generated, uv_transport, 0.5F, 0U);
    const auto first_mip = sample_trilinear(**generated, uv_transport, 1.0F, 0U);
    const auto magnified = sample_trilinear(**generated, uv_transport, -8.0F, 0U);
    const auto top = sample_trilinear(**generated, uv_transport, 100.0F, 0U);
    const auto reference_half = sample_trilinear(**generated, uv_reference, 0.5, 0U);
    ASSERT_TRUE(base.has_value()) << base.error().message;
    ASSERT_TRUE(half.has_value()) << half.error().message;
    ASSERT_TRUE(first_mip.has_value()) << first_mip.error().message;
    ASSERT_TRUE(magnified.has_value()) << magnified.error().message;
    ASSERT_TRUE(top.has_value()) << top.error().message;
    ASSERT_TRUE(reference_half.has_value()) << reference_half.error().message;
    EXPECT_EQ(*base, 0.0F);
    EXPECT_EQ(*half, 0.25F);
    EXPECT_EQ(*first_mip, 0.5F);
    EXPECT_EQ(*magnified, *base);
    EXPECT_EQ(*top, 0.5F);
    EXPECT_EQ(*reference_half, 0.25);

    for (auto level_index = std::uint32_t{1U}; level_index < (*generated)->level_count();
         ++level_index) {
        const auto level = (*generated)->level(level_index);
        ASSERT_TRUE(level.has_value()) << level.error().message;
        for (const auto value : (*level)->pixels()) {
            EXPECT_EQ(value, 0.5F);
        }
    }
}

TEST(HostImageMipChainTest, PreservesAndCancelsFiniteHdrExtremes) {
    constexpr auto maximum = std::numeric_limits<TransportScalar>::max();
    constexpr auto positive = std::array{maximum, maximum, maximum, maximum};
    constexpr auto negative = std::array{-maximum, -maximum, -maximum, -maximum};
    constexpr auto alternating = std::array{maximum, -maximum, maximum, -maximum};
    const auto positive_path = write_pfm_fixture("texture-mip-positive-maximum.pfm", positive);
    const auto negative_path = write_pfm_fixture("texture-mip-negative-maximum.pfm", negative);
    const auto alternating_path =
        write_pfm_fixture("texture-mip-alternating-maximum.pfm", alternating);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto positive_image = load_image(*cache, positive_path);
    const auto negative_image = load_image(*cache, negative_path);
    const auto alternating_image = load_image(*cache, alternating_path);
    ASSERT_TRUE(positive_image);
    ASSERT_TRUE(negative_image);
    ASSERT_TRUE(alternating_image);

    const auto positive_chain = HostImageMipChain::generate(positive_image);
    const auto negative_chain = HostImageMipChain::generate(negative_image);
    const auto alternating_chain = HostImageMipChain::generate(alternating_image);
    ASSERT_TRUE(positive_chain.has_value()) << positive_chain.error().message;
    ASSERT_TRUE(negative_chain.has_value()) << negative_chain.error().message;
    ASSERT_TRUE(alternating_chain.has_value()) << alternating_chain.error().message;
    for (auto level_index = std::uint32_t{1U}; level_index < (*positive_chain)->level_count();
         ++level_index) {
        const auto positive_level = (*positive_chain)->level(level_index);
        const auto negative_level = (*negative_chain)->level(level_index);
        ASSERT_TRUE(positive_level.has_value()) << positive_level.error().message;
        ASSERT_TRUE(negative_level.has_value()) << negative_level.error().message;
        for (const auto value : (*positive_level)->pixels()) {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(value), std::bit_cast<std::uint32_t>(maximum));
        }
        for (const auto value : (*negative_level)->pixels()) {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(value), std::bit_cast<std::uint32_t>(-maximum));
        }
    }
    for (auto level_index = std::uint32_t{1U}; level_index < (*alternating_chain)->level_count();
         ++level_index) {
        const auto level = (*alternating_chain)->level(level_index);
        ASSERT_TRUE(level.has_value()) << level.error().message;
        for (const auto value : (*level)->pixels()) {
            EXPECT_EQ(value, 0.0F);
        }
    }
}

TEST(HostImageMipChainTest, GeneratesLevelsAfterSrgbDecoding) {
    constexpr auto pixels = std::array{
        std::array<std::uint32_t, 3>{0U, 0U, 0U},
        std::array<std::uint32_t, 3>{255U, 255U, 255U},
    };
    const auto path = write_ppm_fixture("texture-mip-srgb-black-white.ppm", 2U, 1U, pixels);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = load_image(*cache, path, TextureColorSpace::srgb);
    ASSERT_TRUE(image);
    const auto generated = HostImageMipChain::generate(image);
    ASSERT_TRUE(generated.has_value()) << generated.error().message;
    ASSERT_EQ((*generated)->level_count(), 2U);
    const auto top = (*generated)->level(1U);
    ASSERT_TRUE(top.has_value()) << top.error().message;
    ASSERT_EQ((*top)->pixels().size(), 3U);
    EXPECT_EQ((*top)->source_color_space(), TextureColorSpace::srgb);
    EXPECT_EQ((*top)->storage_color_space(), TextureWorkingColorSpace);
    for (const auto value : (*top)->pixels()) {
        EXPECT_EQ(value, 0.5F);
    }

    for (auto channel = std::uint32_t{}; channel < 3U; ++channel) {
        const auto filtered =
            sample_trilinear(**generated, ReferencePoint2{.x = 0.5, .y = 0.5}, 1.0, channel);
        ASSERT_TRUE(filtered.has_value()) << filtered.error().message;
        EXPECT_EQ(*filtered, 0.5);
    }
}

TEST(HostImageMipChainTest, EnforcesBudgetsAndRejectsInvalidAccess) {
    constexpr auto values = std::array<TransportScalar, 5>{0.0F, 1.0F, 2.0F, 3.0F, 4.0F};
    const auto path = write_pfm_fixture("texture-mip-budget-5x1.pfm", values);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = load_image(*cache, path);
    ASSERT_TRUE(image);

    const auto exact = HostImageMipChain::generate(
        image, HostImageMipChainLimits{.maximum_generated_pixel_bytes = 12U});
    ASSERT_TRUE(exact.has_value()) << exact.error().message;
    const auto exhausted = HostImageMipChain::generate(
        image, HostImageMipChainLimits{.maximum_generated_pixel_bytes = 11U});
    ASSERT_FALSE(exhausted.has_value());
    EXPECT_EQ(exhausted.error().code, core::StatusCode::resource_exhausted);

    const auto zero_budget = HostImageMipChain::generate(
        image, HostImageMipChainLimits{.maximum_generated_pixel_bytes = 0U});
    ASSERT_FALSE(zero_budget.has_value());
    EXPECT_EQ(zero_budget.error().code, core::StatusCode::invalid_argument);
    const auto null_source = HostImageMipChain::generate(HostImageHandle{});
    ASSERT_FALSE(null_source.has_value());
    EXPECT_EQ(null_source.error().code, core::StatusCode::invalid_argument);

    const auto invalid_level = (*exact)->level((*exact)->level_count());
    ASSERT_FALSE(invalid_level.has_value());
    EXPECT_EQ(invalid_level.error().code, core::StatusCode::invalid_argument);
}

TEST(HostImageMipChainTest, IsDeterministicAndOwnsItsSourceBeyondCacheLifetime) {
    constexpr auto values = std::array<TransportScalar, 5>{0.0F, 1.0F, 2.0F, 3.0F, 4.0F};
    const auto path = write_pfm_fixture("texture-mip-lifetime-5x1.pfm", values);

    const auto surviving = [&]() -> core::Result<HostImageMipChainHandle> {
        auto cache = HostImageCache::create();
        if (!cache) {
            return std::unexpected(cache.error());
        }
        const auto image = cache->load(path, TextureColorSpace::data);
        if (!image) {
            return std::unexpected(image.error());
        }
        return HostImageMipChain::generate(*image);
    }();
    ASSERT_TRUE(surviving.has_value()) << surviving.error().message;
    ASSERT_TRUE((*surviving)->source_image());
    const auto top = (*surviving)->level((*surviving)->level_count() - 1U);
    ASSERT_TRUE(top.has_value()) << top.error().message;
    ASSERT_EQ((*top)->pixels().size(), 1U);
    EXPECT_NEAR((*top)->pixels()[0], 2.0F, 16.0F * std::numeric_limits<TransportScalar>::epsilon());

    const auto replay = HostImageMipChain::generate((*surviving)->source_image());
    ASSERT_TRUE(replay.has_value()) << replay.error().message;
    ASSERT_EQ((*replay)->level_count(), (*surviving)->level_count());
    ASSERT_EQ((*replay)->generated_pixel_bytes(), (*surviving)->generated_pixel_bytes());
    for (auto level_index = std::uint32_t{}; level_index < (*replay)->level_count();
         ++level_index) {
        const auto first = (*surviving)->level(level_index);
        const auto second = (*replay)->level(level_index);
        ASSERT_TRUE(first.has_value()) << first.error().message;
        ASSERT_TRUE(second.has_value()) << second.error().message;
        ASSERT_EQ((*first)->pixels().size(), (*second)->pixels().size());
        for (auto element = std::size_t{}; element < (*first)->pixels().size(); ++element) {
            EXPECT_EQ(std::bit_cast<std::uint32_t>((*first)->pixels()[element]),
                      std::bit_cast<std::uint32_t>((*second)->pixels()[element]));
        }
    }
}

TEST(HostImageMipChainTest, RejectsInvalidTrilinearRequestsWithoutFallback) {
    const auto pixels = checker_pixels(8U, 8U);
    const auto path = write_pgm_fixture("texture-mip-errors-8x8.pgm", 8U, 8U, pixels);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = load_image(*cache, path);
    ASSERT_TRUE(image);
    const auto generated = HostImageMipChain::generate(image);
    ASSERT_TRUE(generated.has_value()) << generated.error().message;

    constexpr auto invalid_wrap = static_cast<TextureWrapMode>(0xFFFFFFFFU);
    const auto bad_channel =
        sample_trilinear(**generated, Point2{.x = 0.5F, .y = 0.5F}, 0.0F, image->channel_count());
    const auto bad_u_wrap = sample_trilinear(**generated, Point2{.x = 0.5F, .y = 0.5F}, 0.0F, 0U,
                                             invalid_wrap, TextureWrapMode::black);
    const auto bad_v_wrap = sample_trilinear(**generated, Point2{.x = -1.0F, .y = 0.5F}, 0.0F, 0U,
                                             TextureWrapMode::black, invalid_wrap);
    for (const auto* result : {&bad_channel, &bad_u_wrap, &bad_v_wrap}) {
        ASSERT_FALSE(result->has_value());
        EXPECT_EQ(result->error().code, core::StatusCode::invalid_argument);
    }

    constexpr auto invalid_lods = std::array{
        std::numeric_limits<ReferenceScalar>::quiet_NaN(),
        std::numeric_limits<ReferenceScalar>::infinity(),
        -std::numeric_limits<ReferenceScalar>::infinity(),
    };
    for (const auto lod : invalid_lods) {
        const auto result =
            sample_trilinear(**generated, ReferencePoint2{.x = 0.5, .y = 0.5}, lod, 0U);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    }

    const auto non_finite_uv = sample_trilinear(
        **generated, Point2{.x = std::numeric_limits<TransportScalar>::quiet_NaN(), .y = 0.5F},
        0.0F, 0U);
    ASSERT_FALSE(non_finite_uv.has_value());
    EXPECT_EQ(non_finite_uv.error().code, core::StatusCode::invalid_argument);
}

#if defined(_WIN32)
[[nodiscard]] bool mip_preview_requested() {
    auto* value = static_cast<char*>(nullptr);
    auto value_size = std::size_t{};
    if (_dupenv_s(&value, &value_size, "BLACKFRAME_WRITE_TEXTURE_MIP_PREVIEWS") != 0 ||
        value == nullptr) {
        return false;
    }
    const auto requested = value_size == 2U && value[0] == '1';
    std::free(value);
    return requested;
}
#else
[[nodiscard]] bool mip_preview_requested() {
    const auto* const value = std::getenv("BLACKFRAME_WRITE_TEXTURE_MIP_PREVIEWS");
    return value != nullptr && std::string_view{value} == "1";
}
#endif

#if defined(BLACKFRAME_HOST_IMAGE_FILTER_PNG)
[[nodiscard]] std::filesystem::path write_preview_checker() {
    constexpr auto width = std::uint32_t{64U};
    constexpr auto height = std::uint32_t{64U};
    auto pixels = std::vector<std::array<std::uint32_t, 3>>{};
    pixels.reserve(static_cast<std::size_t>(width) * height);
    for (auto y = std::uint32_t{}; y < height; ++y) {
        for (auto x = std::uint32_t{}; x < width; ++x) {
            const auto value = (x + y) % 2U == 0U ? 0U : 255U;
            pixels.push_back({value, value, value});
        }
    }
    return write_ppm_fixture("texture-mip-preview-checker.ppm", width, height, pixels);
}

[[nodiscard]] core::Status write_mip_preview(const HostImageMipChain& chain,
                                             const std::filesystem::path& output) {
    constexpr auto extent = RenderExtent{.width = 800U, .height = 800U};
    constexpr auto texture_repeat_x = TransportScalar{32};
    constexpr auto texture_repeat_y = TransportScalar{16};
    const auto minification_lod = std::log2(TransportScalar{64} * texture_repeat_x /
                                            static_cast<TransportScalar>(extent.width / 2U));
    auto film = Film::create(extent);
    if (!film) {
        return std::unexpected(film.error());
    }
    for (auto y = std::uint32_t{}; y < extent.height; ++y) {
        for (auto x = std::uint32_t{}; x < extent.width; ++x) {
            const auto local_x = x < extent.width / 2U ? x : x - extent.width / 2U;
            const auto uv = Point2{
                .x = ((static_cast<TransportScalar>(local_x) + 0.5F) /
                      static_cast<TransportScalar>(extent.width / 2U)) *
                     texture_repeat_x,
                .y = ((static_cast<TransportScalar>(y) + 0.5F) /
                      static_cast<TransportScalar>(extent.height)) *
                     texture_repeat_y,
            };
            const auto lod = x < extent.width / 2U ? 0.0F : minification_lod;
            const auto value = sample_trilinear(chain, uv, lod, 0U, TextureWrapMode::repeat,
                                                TextureWrapMode::repeat);
            if (!value) {
                return std::unexpected(value.error());
            }
            const auto accumulated =
                film->add_sample(x, y, LinearRGB{.red = *value, .green = *value, .blue = *value},
                                 TransportScalar{1});
            if (!accumulated) {
                return accumulated;
            }
        }
    }
    return write_png_preview(*film, output);
}
#endif

TEST(HostImageMipChainTest, WritesExplicitMinificationPreviewWhenRequested) {
    if (!mip_preview_requested()) {
        return;
    }
#if !defined(BLACKFRAME_HOST_IMAGE_FILTER_PNG)
    FAIL() << "Texture mip previews require the explicit PNG capability.";
#else
    const auto source_path = write_preview_checker();
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = load_image(*cache, source_path, TextureColorSpace::srgb);
    ASSERT_TRUE(image);
    const auto generated = HostImageMipChain::generate(image);
    ASSERT_TRUE(generated.has_value()) << generated.error().message;
    const auto output = mip_artifact_path("texture-mip-trilinear-800x800.png");
    const auto written = write_mip_preview(**generated, output);
    ASSERT_TRUE(written.has_value()) << written.error().message;
    ASSERT_TRUE(std::filesystem::is_regular_file(output));
    testing::Test::RecordProperty("texture-mip-trilinear-800x800.png", output.string());
#endif
}

} // namespace
} // namespace blackframe::renderer
