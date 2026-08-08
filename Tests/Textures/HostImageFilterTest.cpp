#include <Blackframe/Renderer/HostImageCache.hpp>
#include <Blackframe/Renderer/HostImageFilter.hpp>
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
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {
namespace {

[[nodiscard]] std::filesystem::path filter_fixture_path() {
    return std::filesystem::path{BLACKFRAME_HOST_IMAGE_TEST_FIXTURE};
}

[[nodiscard]] std::filesystem::path filter_artifact_path(const std::string_view name) {
    return std::filesystem::path{BLACKFRAME_HOST_IMAGE_TEST_OUTPUT_DIR} / name;
}

[[nodiscard]] std::filesystem::path
write_pfm_fixture_2d(const std::string_view name, const std::span<const TransportScalar> values,
                     const std::size_t width, const std::size_t height) {
    const auto output = filter_artifact_path(name);
    auto stream = std::ofstream{output, std::ios::binary | std::ios::trunc};
    EXPECT_TRUE(stream.is_open());
    EXPECT_EQ(values.size(), width * height);
    if constexpr (std::endian::native == std::endian::little) {
        stream << "Pf\n" << width << ' ' << height << "\n-1.0\n";
    } else {
        stream << "Pf\n" << width << ' ' << height << "\n1.0\n";
    }
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size_bytes()));
    EXPECT_TRUE(stream.good());
    return output;
}

[[nodiscard]] std::filesystem::path
write_pfm_fixture(const std::string_view name, const std::span<const TransportScalar> values) {
    return write_pfm_fixture_2d(name, values, values.size(), 1U);
}

[[nodiscard]] HostImageHandle load_data_image(HostImageCache& cache,
                                              const std::filesystem::path& path) {
    const auto image = cache.load(path, TextureColorSpace::data);
    EXPECT_TRUE(image.has_value()) << image.error().message;
    return image ? *image : HostImageHandle{};
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar>
sample(const HostImage& image, const Point2T<Scalar> uv, const std::uint32_t channel,
       const TextureFilterMode filter, const TextureWrapMode u_wrap = TextureWrapMode::clamp,
       const TextureWrapMode v_wrap = TextureWrapMode::clamp) {
    return filter_host_image_channel(image, uv, channel, filter, u_wrap, v_wrap);
}

TEST(HostImageFilterTest, KeepsFilterModesDistinctAndStable) {
    static_assert(sizeof(TextureFilterMode) == sizeof(std::uint32_t));
    static_assert(std::is_trivially_copyable_v<TextureFilterMode>);
    static_assert(is_valid_texture_filter_mode(TextureFilterMode::nearest));
    static_assert(is_valid_texture_filter_mode(TextureFilterMode::bilinear));
    static_assert(is_valid_texture_filter_mode(TextureFilterMode::bicubic));
    static_assert(!is_valid_texture_filter_mode(static_cast<TextureFilterMode>(0xFFFFFFFFU)));
}

TEST(HostImageFilterTest, ReconstructsEveryTexelCenterWithoutFlippingScanlines) {
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value()) << cache.error().message;
    const auto image = load_data_image(*cache, filter_fixture_path());
    ASSERT_TRUE(image);
    ASSERT_EQ(image->width(), 2U);
    ASSERT_EQ(image->height(), 2U);
    ASSERT_EQ(image->channel_count(), 3U);

    constexpr auto centers = std::array{
        Point2{.x = 0.25F, .y = 0.25F},
        Point2{.x = 0.75F, .y = 0.25F},
        Point2{.x = 0.25F, .y = 0.75F},
        Point2{.x = 0.75F, .y = 0.75F},
    };
    constexpr auto filters = std::array{TextureFilterMode::nearest, TextureFilterMode::bilinear,
                                        TextureFilterMode::bicubic};
    for (const auto filter : filters) {
        for (auto pixel = std::size_t{}; pixel < centers.size(); ++pixel) {
            for (auto channel = std::uint32_t{}; channel < image->channel_count(); ++channel) {
                const auto expected = image->pixels()[pixel * image->channel_count() + channel];
                const auto transport = sample(*image, centers[pixel], channel, filter);
                ASSERT_TRUE(transport.has_value()) << transport.error().message;
                EXPECT_EQ(std::bit_cast<std::uint32_t>(*transport),
                          std::bit_cast<std::uint32_t>(expected));

                const auto reference =
                    sample(*image, ReferencePoint2{.x = centers[pixel].x, .y = centers[pixel].y},
                           channel, filter);
                ASSERT_TRUE(reference.has_value()) << reference.error().message;
                EXPECT_EQ(*reference, static_cast<ReferenceScalar>(expected));
            }
        }
    }
}

TEST(HostImageFilterTest, MatchesNearestAndBilinearAnalyticValues) {
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = load_data_image(*cache, filter_fixture_path());
    ASSERT_TRUE(image);

    for (auto channel = std::uint32_t{}; channel < image->channel_count(); ++channel) {
        const auto nearest = sample(*image, ReferencePoint2{.x = 0.5, .y = 0.5}, channel,
                                    TextureFilterMode::nearest);
        ASSERT_TRUE(nearest.has_value()) << nearest.error().message;
        EXPECT_EQ(*nearest, static_cast<ReferenceScalar>(image->pixels()[9U + channel]));

        const auto expected = (static_cast<ReferenceScalar>(image->pixels()[channel]) +
                               static_cast<ReferenceScalar>(image->pixels()[3U + channel]) +
                               static_cast<ReferenceScalar>(image->pixels()[6U + channel]) +
                               static_cast<ReferenceScalar>(image->pixels()[9U + channel])) /
                              ReferenceScalar{4};
        const auto reference = sample(*image, ReferencePoint2{.x = 0.5, .y = 0.5}, channel,
                                      TextureFilterMode::bilinear);
        ASSERT_TRUE(reference.has_value()) << reference.error().message;
        EXPECT_EQ(*reference, expected);
        const auto transport =
            sample(*image, Point2{.x = 0.5F, .y = 0.5F}, channel, TextureFilterMode::bilinear);
        ASSERT_TRUE(transport.has_value()) << transport.error().message;
        EXPECT_FLOAT_EQ(*transport, static_cast<TransportScalar>(expected));
    }
}

TEST(HostImageFilterTest, ResolvesNearestCoordinatesAtAndBeyondTheUvBoundary) {
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = load_data_image(*cache, filter_fixture_path());
    ASSERT_TRUE(image);

    const auto last_inside =
        sample(*image, ReferencePoint2{.x = std::nextafter(1.0, 0.0), .y = 0.25}, 2U,
               TextureFilterMode::nearest, TextureWrapMode::black, TextureWrapMode::black);
    const auto upper_black =
        sample(*image, ReferencePoint2{.x = 1.0, .y = 0.25}, 2U, TextureFilterMode::nearest,
               TextureWrapMode::black, TextureWrapMode::black);
    const auto upper_clamp =
        sample(*image, ReferencePoint2{.x = 1.0, .y = 0.25}, 2U, TextureFilterMode::nearest,
               TextureWrapMode::clamp, TextureWrapMode::black);
    const auto upper_mirror =
        sample(*image, ReferencePoint2{.x = 1.0, .y = 0.25}, 2U, TextureFilterMode::nearest,
               TextureWrapMode::mirror, TextureWrapMode::black);
    ASSERT_TRUE(last_inside.has_value());
    ASSERT_TRUE(upper_black.has_value());
    ASSERT_TRUE(upper_clamp.has_value());
    ASSERT_TRUE(upper_mirror.has_value());
    EXPECT_EQ(*last_inside, 1.0);
    EXPECT_EQ(*upper_black, 0.0);
    EXPECT_EQ(*upper_clamp, 1.0);
    EXPECT_EQ(*upper_mirror, 1.0);

    const auto negative_denormal =
        sample(*image,
               ReferencePoint2{.x = -std::numeric_limits<ReferenceScalar>::denorm_min(), .y = 0.25},
               0U, TextureFilterMode::nearest, TextureWrapMode::black, TextureWrapMode::black);
    const auto lower_edge =
        sample(*image, ReferencePoint2{.x = 0.0, .y = 0.25}, 0U, TextureFilterMode::nearest,
               TextureWrapMode::black, TextureWrapMode::black);
    const auto transport_negative_denormal =
        sample(*image, Point2{.x = -std::numeric_limits<TransportScalar>::denorm_min(), .y = 0.25F},
               0U, TextureFilterMode::nearest, TextureWrapMode::black, TextureWrapMode::black);
    ASSERT_TRUE(negative_denormal.has_value());
    ASSERT_TRUE(lower_edge.has_value());
    ASSERT_TRUE(transport_negative_denormal.has_value());
    EXPECT_EQ(*negative_denormal, 0.0);
    EXPECT_EQ(*lower_edge, 1.0);
    EXPECT_EQ(*transport_negative_denormal, 0.0F);
}

TEST(HostImageFilterTest, UsesTheDocumentedCatmullRomKernel) {
    constexpr auto values = std::array<TransportScalar, 4>{0.0F, 1.0F, 4.0F, 9.0F};
    const auto path = write_pfm_fixture("texture-filter-quadratic.pfm", values);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = load_data_image(*cache, path);
    ASSERT_TRUE(image);
    ASSERT_EQ(image->pixels().size(), 4U);

    const auto nearest =
        sample(*image, ReferencePoint2{.x = 0.5, .y = 0.5}, 0U, TextureFilterMode::nearest);
    const auto bilinear =
        sample(*image, ReferencePoint2{.x = 0.5, .y = 0.5}, 0U, TextureFilterMode::bilinear);
    const auto bicubic =
        sample(*image, ReferencePoint2{.x = 0.5, .y = 0.5}, 0U, TextureFilterMode::bicubic);
    ASSERT_TRUE(nearest.has_value()) << nearest.error().message;
    ASSERT_TRUE(bilinear.has_value()) << bilinear.error().message;
    ASSERT_TRUE(bicubic.has_value()) << bicubic.error().message;
    EXPECT_EQ(*nearest, 4.0);
    EXPECT_EQ(*bilinear, 2.5);
    EXPECT_EQ(*bicubic, 2.25);
}

TEST(HostImageFilterTest, ReproducesAnAsymmetricAffinePatternInTwoDimensions) {
    const auto path = filter_artifact_path("texture-filter-affine.ppm");
    auto stream = std::ofstream{path, std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(stream.is_open());
    stream << "P3\n4 4\n255\n";
    for (auto y = std::uint32_t{}; y < 4U; ++y) {
        for (auto x = std::uint32_t{}; x < 4U; ++x) {
            stream << 10U + 20U * x + 30U * y << ' ' << 5U + 30U * x + 10U * y << ' '
                   << 12U + 5U * x + 40U * y << '\n';
        }
    }
    stream.close();
    ASSERT_TRUE(stream.good());

    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = load_data_image(*cache, path);
    ASSERT_TRUE(image);
    constexpr auto uv = ReferencePoint2{.x = 0.4375, .y = 0.5};
    constexpr auto expected = std::array{80.0 / 255.0, 57.5 / 255.0, 78.25 / 255.0};
    constexpr auto tolerance = 64.0 * std::numeric_limits<TransportScalar>::epsilon();
    for (const auto filter : {TextureFilterMode::bilinear, TextureFilterMode::bicubic}) {
        for (auto channel = std::uint32_t{}; channel < expected.size(); ++channel) {
            const auto reference = sample(*image, uv, channel, filter);
            const auto transport = sample(*image,
                                          Point2{.x = static_cast<TransportScalar>(uv.x),
                                                 .y = static_cast<TransportScalar>(uv.y)},
                                          channel, filter);
            ASSERT_TRUE(reference.has_value()) << reference.error().message;
            ASSERT_TRUE(transport.has_value()) << transport.error().message;
            EXPECT_NEAR(*reference, expected[channel], tolerance);
            EXPECT_NEAR(*transport, expected[channel], tolerance);
        }
    }
}

TEST(HostImageFilterTest, AppliesEveryWrapModePerBilinearTap) {
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = load_data_image(*cache, filter_fixture_path());
    ASSERT_TRUE(image);

    const auto at_left = [&](const TextureWrapMode mode) {
        return sample(*image, ReferencePoint2{.x = 0.0, .y = 0.25}, 0U, TextureFilterMode::bilinear,
                      mode, TextureWrapMode::clamp);
    };
    const auto repeat_left = at_left(TextureWrapMode::repeat);
    const auto clamp_left = at_left(TextureWrapMode::clamp);
    const auto mirror_left = at_left(TextureWrapMode::mirror);
    const auto black_left = at_left(TextureWrapMode::black);
    ASSERT_TRUE(repeat_left.has_value());
    ASSERT_TRUE(clamp_left.has_value());
    ASSERT_TRUE(mirror_left.has_value());
    ASSERT_TRUE(black_left.has_value());
    EXPECT_EQ(*repeat_left, 0.5);
    EXPECT_EQ(*clamp_left, 1.0);
    EXPECT_EQ(*mirror_left, 1.0);
    EXPECT_EQ(*black_left, 0.5);

    const auto at_right = [&](const TextureWrapMode mode) {
        return sample(*image, ReferencePoint2{.x = 1.0, .y = 0.25}, 0U, TextureFilterMode::bilinear,
                      mode, TextureWrapMode::clamp);
    };
    const auto repeat_right = at_right(TextureWrapMode::repeat);
    const auto clamp_right = at_right(TextureWrapMode::clamp);
    const auto mirror_right = at_right(TextureWrapMode::mirror);
    const auto black_right = at_right(TextureWrapMode::black);
    ASSERT_TRUE(repeat_right.has_value());
    ASSERT_TRUE(clamp_right.has_value());
    ASSERT_TRUE(mirror_right.has_value());
    ASSERT_TRUE(black_right.has_value());
    EXPECT_EQ(*repeat_right, 0.5);
    EXPECT_EQ(*clamp_right, 0.0);
    EXPECT_EQ(*mirror_right, 0.0);
    EXPECT_EQ(*black_right, 0.0);

    const auto black_corner = sample(*image, ReferencePoint2{}, 0U, TextureFilterMode::bilinear,
                                     TextureWrapMode::black, TextureWrapMode::black);
    ASSERT_TRUE(black_corner.has_value());
    EXPECT_EQ(*black_corner, 0.25);

    const auto bicubic_black_left =
        sample(*image, ReferencePoint2{.x = 0.0, .y = 0.25}, 0U, TextureFilterMode::bicubic,
               TextureWrapMode::black, TextureWrapMode::clamp);
    ASSERT_TRUE(bicubic_black_left.has_value());
    EXPECT_EQ(*bicubic_black_left, 9.0 / 16.0);

    const auto top = static_cast<ReferenceScalar>(image->pixels()[0U]);
    const auto bottom = static_cast<ReferenceScalar>(image->pixels()[6U]);
    const auto at_top = [&](const TextureWrapMode mode) {
        return sample(*image, ReferencePoint2{.x = 0.25, .y = 0.0}, 0U, TextureFilterMode::bilinear,
                      TextureWrapMode::clamp, mode);
    };
    const auto at_bottom = [&](const TextureWrapMode mode) {
        return sample(*image, ReferencePoint2{.x = 0.25, .y = 1.0}, 0U, TextureFilterMode::bilinear,
                      TextureWrapMode::clamp, mode);
    };
    const auto top_repeat = at_top(TextureWrapMode::repeat);
    const auto top_clamp = at_top(TextureWrapMode::clamp);
    const auto top_mirror = at_top(TextureWrapMode::mirror);
    const auto top_black = at_top(TextureWrapMode::black);
    const auto bottom_repeat = at_bottom(TextureWrapMode::repeat);
    const auto bottom_clamp = at_bottom(TextureWrapMode::clamp);
    const auto bottom_mirror = at_bottom(TextureWrapMode::mirror);
    const auto bottom_black = at_bottom(TextureWrapMode::black);
    ASSERT_TRUE(top_repeat.has_value());
    ASSERT_TRUE(top_clamp.has_value());
    ASSERT_TRUE(top_mirror.has_value());
    ASSERT_TRUE(top_black.has_value());
    ASSERT_TRUE(bottom_repeat.has_value());
    ASSERT_TRUE(bottom_clamp.has_value());
    ASSERT_TRUE(bottom_mirror.has_value());
    ASSERT_TRUE(bottom_black.has_value());
    EXPECT_EQ(*top_repeat, (top + bottom) * 0.5);
    EXPECT_EQ(*top_clamp, top);
    EXPECT_EQ(*top_mirror, top);
    EXPECT_EQ(*top_black, top * 0.5);
    EXPECT_EQ(*bottom_repeat, (top + bottom) * 0.5);
    EXPECT_EQ(*bottom_clamp, bottom);
    EXPECT_EQ(*bottom_mirror, bottom);
    EXPECT_EQ(*bottom_black, bottom * 0.5);
}

TEST(HostImageFilterTest, PreservesConstantExtremeValuesWithoutClamping) {
    constexpr auto maximum = std::numeric_limits<TransportScalar>::max();
    constexpr auto constant = std::array{maximum};
    const auto path = write_pfm_fixture("texture-filter-maximum.pfm", constant);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = load_data_image(*cache, path);
    ASSERT_TRUE(image);

    constexpr auto filters = std::array{TextureFilterMode::nearest, TextureFilterMode::bilinear,
                                        TextureFilterMode::bicubic};
    constexpr auto wraps =
        std::array{TextureWrapMode::repeat, TextureWrapMode::clamp, TextureWrapMode::mirror};
    for (const auto filter : filters) {
        for (const auto wrap : wraps) {
            const auto value =
                sample(*image, Point2{.x = -17.25F, .y = 8.75F}, 0U, filter, wrap, wrap);
            ASSERT_TRUE(value.has_value()) << value.error().message;
            EXPECT_EQ(std::bit_cast<std::uint32_t>(*value), std::bit_cast<std::uint32_t>(maximum));
        }
    }
}

TEST(HostImageFilterTest, ReconstructsAndCancelsOppositeTransportExtremes) {
    constexpr auto maximum = std::numeric_limits<TransportScalar>::max();
    constexpr auto denormal = std::numeric_limits<TransportScalar>::denorm_min();
    constexpr auto center_values = std::array{-maximum, maximum, maximum, maximum};
    constexpr auto denormal_center_values = std::array{maximum, denormal, maximum, maximum};
    constexpr auto cancellation_values = std::array{-maximum, maximum, -maximum, maximum};
    constexpr auto cancellation_2d_values = std::array{
        -maximum, maximum, -maximum, maximum, -maximum, maximum, -maximum, maximum,
        -maximum, maximum, -maximum, maximum, -maximum, maximum, -maximum, maximum,
    };
    const auto center_path = write_pfm_fixture("texture-filter-extreme-center.pfm", center_values);
    const auto denormal_center_path =
        write_pfm_fixture("texture-filter-extreme-denormal-center.pfm", denormal_center_values);
    const auto cancellation_path =
        write_pfm_fixture("texture-filter-extreme-cancellation.pfm", cancellation_values);
    const auto cancellation_2d_path = write_pfm_fixture_2d(
        "texture-filter-extreme-cancellation-2d.pfm", cancellation_2d_values, 4U, 4U);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto center_image = load_data_image(*cache, center_path);
    const auto denormal_center_image = load_data_image(*cache, denormal_center_path);
    const auto cancellation_image = load_data_image(*cache, cancellation_path);
    const auto cancellation_2d_image = load_data_image(*cache, cancellation_2d_path);
    ASSERT_TRUE(center_image);
    ASSERT_TRUE(denormal_center_image);
    ASSERT_TRUE(cancellation_image);
    ASSERT_TRUE(cancellation_2d_image);

    const auto center_transport =
        sample(*center_image, Point2{.x = 0.375F, .y = 0.5F}, 0U, TextureFilterMode::bicubic);
    const auto center_reference = sample(*center_image, ReferencePoint2{.x = 0.375, .y = 0.5}, 0U,
                                         TextureFilterMode::bicubic);
    ASSERT_TRUE(center_transport.has_value()) << center_transport.error().message;
    ASSERT_TRUE(center_reference.has_value()) << center_reference.error().message;
    EXPECT_EQ(std::bit_cast<std::uint32_t>(*center_transport),
              std::bit_cast<std::uint32_t>(maximum));
    EXPECT_EQ(*center_reference, static_cast<ReferenceScalar>(maximum));

    const auto denormal_center_transport = sample(
        *denormal_center_image, Point2{.x = 0.375F, .y = 0.5F}, 0U, TextureFilterMode::bicubic);
    const auto denormal_center_reference =
        sample(*denormal_center_image, ReferencePoint2{.x = 0.375, .y = 0.5}, 0U,
               TextureFilterMode::bicubic);
    ASSERT_TRUE(denormal_center_transport.has_value()) << denormal_center_transport.error().message;
    ASSERT_TRUE(denormal_center_reference.has_value()) << denormal_center_reference.error().message;
    EXPECT_EQ(std::bit_cast<std::uint32_t>(*denormal_center_transport),
              std::bit_cast<std::uint32_t>(denormal));
    EXPECT_EQ(*denormal_center_reference, static_cast<ReferenceScalar>(denormal));

    const auto cancellation_transport =
        sample(*cancellation_image, Point2{.x = 0.5F, .y = 0.5F}, 0U, TextureFilterMode::bicubic);
    const auto cancellation_reference = sample(
        *cancellation_image, ReferencePoint2{.x = 0.5, .y = 0.5}, 0U, TextureFilterMode::bicubic);
    ASSERT_TRUE(cancellation_transport.has_value()) << cancellation_transport.error().message;
    ASSERT_TRUE(cancellation_reference.has_value()) << cancellation_reference.error().message;
    EXPECT_EQ(*cancellation_transport, 0.0F);
    EXPECT_EQ(*cancellation_reference, 0.0);

    const auto cancellation_2d_transport = sample(
        *cancellation_2d_image, Point2{.x = 0.5F, .y = 0.4375F}, 0U, TextureFilterMode::bicubic);
    const auto cancellation_2d_reference =
        sample(*cancellation_2d_image, ReferencePoint2{.x = 0.5, .y = 0.4375}, 0U,
               TextureFilterMode::bicubic);
    ASSERT_TRUE(cancellation_2d_transport.has_value()) << cancellation_2d_transport.error().message;
    ASSERT_TRUE(cancellation_2d_reference.has_value()) << cancellation_2d_reference.error().message;
    EXPECT_EQ(*cancellation_2d_transport, 0.0F);
    EXPECT_EQ(*cancellation_2d_reference, 0.0);
}

TEST(HostImageFilterTest, ReportsUnrepresentableBicubicOvershootInsteadOfClamping) {
    constexpr auto maximum = std::numeric_limits<TransportScalar>::max();
    constexpr auto values = std::array{-maximum, maximum, maximum, -maximum};
    const auto path = write_pfm_fixture("texture-filter-overshoot.pfm", values);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = load_data_image(*cache, path);
    ASSERT_TRUE(image);

    const auto reference =
        sample(*image, ReferencePoint2{.x = 0.5, .y = 0.5}, 0U, TextureFilterMode::bicubic);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;
    EXPECT_GT(*reference, static_cast<ReferenceScalar>(maximum));
    const auto transport =
        sample(*image, Point2{.x = 0.5F, .y = 0.5F}, 0U, TextureFilterMode::bicubic);
    ASSERT_FALSE(transport.has_value());
    EXPECT_EQ(transport.error().code, core::StatusCode::invalid_argument);
}

TEST(HostImageFilterTest, FiltersAfterSrgbDecoding) {
    constexpr auto values = std::array<std::uint32_t, 6>{0U, 0U, 0U, 255U, 255U, 255U};
    const auto path = filter_artifact_path("texture-filter-srgb.ppm");
    auto stream = std::ofstream{path, std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(stream.is_open());
    stream << "P3\n2 1\n255\n";
    for (const auto value : values) {
        stream << value << '\n';
    }
    stream.close();
    ASSERT_TRUE(stream.good());

    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto loaded = cache->load(path, TextureColorSpace::srgb);
    ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
    for (auto channel = std::uint32_t{}; channel < 3U; ++channel) {
        const auto filtered = sample(**loaded, ReferencePoint2{.x = 0.5, .y = 0.5}, channel,
                                     TextureFilterMode::bilinear);
        ASSERT_TRUE(filtered.has_value()) << filtered.error().message;
        EXPECT_EQ(*filtered, 0.5);
    }
}

TEST(HostImageFilterTest, RejectsInvalidRequestsWithoutAHiddenMode) {
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = load_data_image(*cache, filter_fixture_path());
    ASSERT_TRUE(image);
    constexpr auto invalid_filter = static_cast<TextureFilterMode>(0xFFFFFFFFU);
    constexpr auto invalid_wrap = static_cast<TextureWrapMode>(0xFFFFFFFFU);

    const auto bad_filter = sample(*image, Point2{.x = 0.5F, .y = 0.5F}, 0U, invalid_filter);
    const auto bad_u_wrap =
        sample(*image, Point2{.x = 0.5F, .y = 0.5F}, 0U, TextureFilterMode::nearest, invalid_wrap,
               TextureWrapMode::black);
    const auto bad_v_wrap =
        sample(*image, Point2{.x = -1.0F, .y = 0.5F}, 0U, TextureFilterMode::nearest,
               TextureWrapMode::black, invalid_wrap);
    const auto bad_channel = sample(*image, Point2{.x = 0.5F, .y = 0.5F}, image->channel_count(),
                                    TextureFilterMode::nearest);
    for (const auto* result : {&bad_filter, &bad_u_wrap, &bad_v_wrap, &bad_channel}) {
        ASSERT_FALSE(result->has_value());
        EXPECT_EQ(result->error().code, core::StatusCode::invalid_argument);
    }

    constexpr auto invalid_coordinates = std::array{
        std::numeric_limits<ReferenceScalar>::quiet_NaN(),
        std::numeric_limits<ReferenceScalar>::infinity(),
        -std::numeric_limits<ReferenceScalar>::infinity(),
        std::numeric_limits<ReferenceScalar>::max(),
    };
    for (const auto coordinate : invalid_coordinates) {
        const auto bad_u =
            sample(*image, ReferencePoint2{.x = coordinate, .y = 0.5}, 0U,
                   TextureFilterMode::bicubic, TextureWrapMode::black, TextureWrapMode::repeat);
        const auto bad_v =
            sample(*image, ReferencePoint2{.x = -1.0, .y = coordinate}, 0U,
                   TextureFilterMode::bilinear, TextureWrapMode::black, TextureWrapMode::repeat);
        ASSERT_FALSE(bad_u.has_value());
        ASSERT_FALSE(bad_v.has_value());
        EXPECT_EQ(bad_u.error().code, core::StatusCode::invalid_argument);
        EXPECT_EQ(bad_v.error().code, core::StatusCode::invalid_argument);
    }

    const auto phase_limit = std::ldexp(ReferenceScalar{1}, 52) / image->width();
    const auto insufficient_phase =
        sample(*image, ReferencePoint2{.x = phase_limit, .y = 0.5}, 0U, TextureFilterMode::bilinear,
               TextureWrapMode::repeat, TextureWrapMode::repeat);
    ASSERT_FALSE(insufficient_phase.has_value());
    EXPECT_EQ(insufficient_phase.error().code, core::StatusCode::invalid_argument);

    const auto transport_phase_limit =
        std::ldexp(TransportScalar{1}, std::numeric_limits<TransportScalar>::digits - 1) /
        static_cast<TransportScalar>(image->width());
    const auto insufficient_transport_phase =
        sample(*image, Point2{.x = transport_phase_limit, .y = 0.5F}, 0U,
               TextureFilterMode::bilinear, TextureWrapMode::repeat, TextureWrapMode::repeat);
    ASSERT_FALSE(insufficient_transport_phase.has_value());
    EXPECT_EQ(insufficient_transport_phase.error().code, core::StatusCode::invalid_argument);
}

#if defined(_WIN32)
[[nodiscard]] bool filter_preview_requested() {
    auto* value = static_cast<char*>(nullptr);
    auto value_size = std::size_t{};
    if (_dupenv_s(&value, &value_size, "BLACKFRAME_WRITE_TEXTURE_FILTER_PREVIEWS") != 0 ||
        value == nullptr) {
        return false;
    }
    const auto requested = value_size == 2U && value[0] == '1';
    std::free(value);
    return requested;
}
#else
[[nodiscard]] bool filter_preview_requested() {
    const auto* const value = std::getenv("BLACKFRAME_WRITE_TEXTURE_FILTER_PREVIEWS");
    return value != nullptr && std::string_view{value} == "1";
}
#endif

#if defined(BLACKFRAME_HOST_IMAGE_FILTER_PNG)
[[nodiscard]] std::filesystem::path write_filter_preview_source() {
    const auto output = filter_artifact_path("texture-filter-preview-source.ppm");
    auto stream = std::ofstream{output, std::ios::binary | std::ios::trunc};
    EXPECT_TRUE(stream.is_open());
    stream << "P3\n8 8\n255\n";
    for (auto y = std::uint32_t{}; y < 8U; ++y) {
        for (auto x = std::uint32_t{}; x < 8U; ++x) {
            auto red = (x + y) % 2U == 0U ? 235U : 20U;
            auto green = x * 255U / 7U;
            auto blue = y * 255U / 7U;
            if (x == 3U && y == 3U) {
                red = green = blue = 255U;
            } else if (x == 4U && y == 4U) {
                red = green = blue = 0U;
            }
            stream << red << ' ' << green << ' ' << blue << '\n';
        }
    }
    EXPECT_TRUE(stream.good());
    return output;
}

[[nodiscard]] core::Status write_filter_preview(const HostImage& image,
                                                const TextureFilterMode filter,
                                                const std::filesystem::path& output) {
    constexpr auto extent = RenderExtent{.width = 800U, .height = 800U};
    auto film = Film::create(extent);
    if (!film) {
        return std::unexpected(film.error());
    }
    for (auto y = std::uint32_t{}; y < extent.height; ++y) {
        for (auto x = std::uint32_t{}; x < extent.width; ++x) {
            const auto uv = Point2{
                .x = (static_cast<TransportScalar>(x) + 0.5F) /
                     static_cast<TransportScalar>(extent.width),
                .y = (static_cast<TransportScalar>(y) + 0.5F) /
                     static_cast<TransportScalar>(extent.height),
            };
            auto color = LinearRGB{};
            const auto red = sample(image, uv, 0U, filter);
            const auto green = sample(image, uv, 1U, filter);
            const auto blue = sample(image, uv, 2U, filter);
            if (!red) {
                return std::unexpected(red.error());
            }
            if (!green) {
                return std::unexpected(green.error());
            }
            if (!blue) {
                return std::unexpected(blue.error());
            }
            color = LinearRGB{.red = *red, .green = *green, .blue = *blue};
            const auto accumulated = film->add_sample(x, y, color, TransportScalar{1});
            if (!accumulated) {
                return accumulated;
            }
        }
    }
    return write_png_preview(*film, output);
}
#endif

TEST(HostImageFilterTest, WritesAnalyticFilterPreviewsWhenRequested) {
    if (!filter_preview_requested()) {
        return;
    }
#if !defined(BLACKFRAME_HOST_IMAGE_FILTER_PNG)
    FAIL() << "Texture filter previews require the explicit PNG capability.";
#else
    const auto source_path = write_filter_preview_source();
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto image = load_data_image(*cache, source_path);
    ASSERT_TRUE(image);

    constexpr auto outputs = std::array{
        std::pair{TextureFilterMode::nearest,
                  std::string_view{"texture-filter-nearest-800x800.png"}},
        std::pair{TextureFilterMode::bilinear,
                  std::string_view{"texture-filter-bilinear-800x800.png"}},
        std::pair{TextureFilterMode::bicubic,
                  std::string_view{"texture-filter-bicubic-800x800.png"}},
    };
    for (const auto& [filter, name] : outputs) {
        const auto output = filter_artifact_path(name);
        const auto written = write_filter_preview(*image, filter, output);
        ASSERT_TRUE(written.has_value()) << written.error().message;
        ASSERT_TRUE(std::filesystem::is_regular_file(output));
        testing::Test::RecordProperty(name.data(), output.string());
    }
#endif
}

} // namespace
} // namespace blackframe::renderer
