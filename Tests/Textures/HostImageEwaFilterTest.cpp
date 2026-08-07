#include <Blackframe/Renderer/HostImageFilter.hpp>
#include <Blackframe/Renderer/HostImageMipChain.hpp>
#if defined(BLACKFRAME_HOST_IMAGE_FILTER_PNG)
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/PngWriter.hpp>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace blackframe::renderer {
namespace {

[[nodiscard]] std::filesystem::path ewa_artifact_path(const std::string_view name) {
    return std::filesystem::path{BLACKFRAME_HOST_IMAGE_TEST_OUTPUT_DIR} / name;
}

[[nodiscard]] std::filesystem::path
write_pfm_fixture(const std::string_view name, const std::uint32_t width,
                  const std::uint32_t height, const std::span<const TransportScalar> pixels) {
    const auto output = ewa_artifact_path(name);
    auto stream = std::ofstream{output, std::ios::binary | std::ios::trunc};
    EXPECT_TRUE(stream.is_open());
    EXPECT_EQ(pixels.size(), static_cast<std::size_t>(width) * height);
    stream << "Pf\n"
           << width << ' ' << height << '\n'
           << (std::endian::native == std::endian::little ? "-1.0\n" : "1.0\n");
    stream.write(reinterpret_cast<const char*>(pixels.data()),
                 static_cast<std::streamsize>(pixels.size_bytes()));
    EXPECT_TRUE(stream.good());
    return output;
}

[[nodiscard]] std::filesystem::path write_pgm_fixture(const std::string_view name,
                                                      const std::uint32_t width,
                                                      const std::uint32_t height,
                                                      const std::span<const std::uint8_t> pixels) {
    const auto output = ewa_artifact_path(name);
    auto stream = std::ofstream{output, std::ios::binary | std::ios::trunc};
    EXPECT_TRUE(stream.is_open());
    EXPECT_EQ(pixels.size(), static_cast<std::size_t>(width) * height);
    stream << "P5\n" << width << ' ' << height << "\n255\n";
    stream.write(reinterpret_cast<const char*>(pixels.data()),
                 static_cast<std::streamsize>(pixels.size()));
    EXPECT_TRUE(stream.good());
    return output;
}

[[nodiscard]] HostImageMipChainHandle load_mip_chain(HostImageCache& cache,
                                                     const std::filesystem::path& path) {
    const auto image = cache.load(path, TextureColorSpace::data);
    EXPECT_TRUE(image.has_value()) << image.error().message;
    if (!image) {
        return {};
    }
    const auto chain = HostImageMipChain::generate(*image);
    EXPECT_TRUE(chain.has_value()) << chain.error().message;
    return chain ? *chain : HostImageMipChainHandle{};
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar>
sample_ewa(const HostImageMipChain& chain, const Point2T<Scalar> uv,
           const TextureCoordinateDifferentialsT<Scalar> differentials,
           const std::uint32_t channel = 0U, const TextureWrapMode u_wrap = TextureWrapMode::repeat,
           const TextureWrapMode v_wrap = TextureWrapMode::repeat,
           const HostImageEwaLimits limits = {}) {
    return filter_host_image_ewa_channel(chain, uv, differentials, channel, u_wrap, v_wrap, limits);
}

[[nodiscard]] std::vector<std::uint8_t> checker_pixels(const std::uint32_t width,
                                                       const std::uint32_t height,
                                                       const std::uint32_t cell_size = 1U) {
    auto pixels = std::vector<std::uint8_t>{};
    pixels.reserve(static_cast<std::size_t>(width) * height);
    for (auto y = std::uint32_t{}; y < height; ++y) {
        for (auto x = std::uint32_t{}; x < width; ++x) {
            pixels.push_back(((x / cell_size) + (y / cell_size)) % 2U == 0U ? 0U : 255U);
        }
    }
    return pixels;
}

[[nodiscard]] std::string metric_string(const ReferenceScalar value) {
    auto stream = std::ostringstream{};
    stream << std::scientific
           << std::setprecision(std::numeric_limits<ReferenceScalar>::max_digits10) << value;
    return stream.str();
}

TEST(HostImageEwaFilterTest, KeepsPublicFootprintAndLimitsStable) {
    static_assert(std::is_trivially_copyable_v<TextureCoordinateDifferentials>);
    static_assert(std::is_trivially_copyable_v<ReferenceTextureCoordinateDifferentials>);
    static_assert(std::is_trivially_copyable_v<HostImageEwaLimits>);
    static_assert(std::is_standard_layout_v<TextureCoordinateDifferentials>);
    static_assert(std::is_standard_layout_v<ReferenceTextureCoordinateDifferentials>);
    static_assert(std::is_standard_layout_v<HostImageEwaLimits>);
    EXPECT_EQ(sizeof(TextureCoordinateDifferentials), 4U * sizeof(TransportScalar));
    EXPECT_EQ(sizeof(ReferenceTextureCoordinateDifferentials), 4U * sizeof(ReferenceScalar));
    EXPECT_EQ(sizeof(HostImageEwaLimits), 2U * sizeof(std::uint32_t));
    EXPECT_EQ(HostImageEwaLimits{}.maximum_anisotropy, 16U);
    EXPECT_EQ(HostImageEwaLimits{}.maximum_texel_visits, 8'192U);
    EXPECT_EQ(HostImageEwaMaximumAnisotropy, 64U);
}

TEST(HostImageEwaFilterTest, MatchesCenteredImpulseAnalyticallyAndNeedsNoZeroFootprintFallback) {
    constexpr auto pixels = std::array<TransportScalar, 9>{
        0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F,
    };
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value()) << cache.error().message;
    const auto chain =
        load_mip_chain(*cache, write_pfm_fixture("texture-ewa-impulse.pfm", 3U, 3U, pixels));
    ASSERT_TRUE(chain);

    constexpr auto transport_footprint = TextureCoordinateDifferentials{
        .dudx = 1.0F / 3.0F,
        .dvdy = 1.0F / 3.0F,
    };
    constexpr auto reference_footprint = ReferenceTextureCoordinateDifferentials{
        .dudx = 1.0 / 3.0,
        .dvdy = 1.0 / 3.0,
    };
    const auto transport = sample_ewa(*chain, Point2{.x = 0.5F, .y = 0.5F}, transport_footprint);
    const auto reference =
        sample_ewa(*chain, ReferencePoint2{.x = 0.5, .y = 0.5}, reference_footprint);
    ASSERT_TRUE(transport.has_value()) << transport.error().message;
    ASSERT_TRUE(reference.has_value()) << reference.error().message;
    const auto edge = std::exp(-2.0);
    const auto expected = (1.0 - edge) / (1.0 + 4.0 * std::exp(-1.0) - 5.0 * edge);
    EXPECT_NEAR(*transport, static_cast<TransportScalar>(expected), 2.0e-6F);
    EXPECT_NEAR(*reference, expected, 2.0e-15);

    const auto zero =
        sample_ewa(*chain, Point2{.x = 0.5F, .y = 0.5F}, TextureCoordinateDifferentials{});
    ASSERT_TRUE(zero.has_value()) << zero.error().message;
    EXPECT_EQ(*zero, 1.0F);
}

TEST(HostImageEwaFilterTest, PreservesHdrConstantsAndBlackBorderWeight) {
    constexpr auto ones = std::array<TransportScalar, 9>{
        1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F,
    };
    constexpr auto maximum = std::numeric_limits<TransportScalar>::max();
    constexpr auto positive = std::array{maximum, maximum, maximum, maximum};
    constexpr auto negative = std::array{-maximum, -maximum, -maximum, -maximum};
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto one_chain =
        load_mip_chain(*cache, write_pfm_fixture("texture-ewa-ones.pfm", 3U, 3U, ones));
    const auto positive_chain =
        load_mip_chain(*cache, write_pfm_fixture("texture-ewa-positive.pfm", 2U, 2U, positive));
    const auto negative_chain =
        load_mip_chain(*cache, write_pfm_fixture("texture-ewa-negative.pfm", 2U, 2U, negative));
    ASSERT_TRUE(one_chain);
    ASSERT_TRUE(positive_chain);
    ASSERT_TRUE(negative_chain);

    constexpr auto rotated = TextureCoordinateDifferentials{
        .dudx = 2.0F,
        .dvdx = 2.0F,
        .dudy = 0.125F,
        .dvdy = -0.125F,
    };
    const auto positive_value =
        sample_ewa(*positive_chain, Point2{.x = 0.37F, .y = 0.61F}, rotated);
    const auto negative_value =
        sample_ewa(*negative_chain, Point2{.x = 0.37F, .y = 0.61F}, rotated);
    ASSERT_TRUE(positive_value.has_value()) << positive_value.error().message;
    ASSERT_TRUE(negative_value.has_value()) << negative_value.error().message;
    EXPECT_EQ(*positive_value, maximum);
    EXPECT_EQ(*negative_value, -maximum);

    constexpr auto edge_uv = Point2{.x = 0.0F, .y = 0.5F};
    constexpr auto zero = TextureCoordinateDifferentials{};
    const auto black =
        sample_ewa(*one_chain, edge_uv, zero, 0U, TextureWrapMode::black, TextureWrapMode::black);
    const auto repeat =
        sample_ewa(*one_chain, edge_uv, zero, 0U, TextureWrapMode::repeat, TextureWrapMode::repeat);
    const auto clamp =
        sample_ewa(*one_chain, edge_uv, zero, 0U, TextureWrapMode::clamp, TextureWrapMode::clamp);
    const auto mirror =
        sample_ewa(*one_chain, edge_uv, zero, 0U, TextureWrapMode::mirror, TextureWrapMode::mirror);
    ASSERT_TRUE(black.has_value()) << black.error().message;
    ASSERT_TRUE(repeat.has_value()) << repeat.error().message;
    ASSERT_TRUE(clamp.has_value()) << clamp.error().message;
    ASSERT_TRUE(mirror.has_value()) << mirror.error().message;
    EXPECT_NEAR(*black, 0.5F, 2.0e-6F);
    EXPECT_EQ(*repeat, 1.0F);
    EXPECT_EQ(*clamp, 1.0F);
    EXPECT_EQ(*mirror, 1.0F);
}

TEST(HostImageEwaFilterTest, ReducesObliqueCheckerAliasingAndMatchesDoubleReference) {
    constexpr auto texture_extent = std::uint32_t{256U};
    const auto pixels = checker_pixels(texture_extent, texture_extent);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto chain =
        load_mip_chain(*cache, write_pgm_fixture("texture-ewa-checker.pgm", texture_extent,
                                                 texture_extent, pixels));
    ASSERT_TRUE(chain);

    constexpr auto transport_footprint = TextureCoordinateDifferentials{
        .dudx = 20.0F / static_cast<TransportScalar>(texture_extent),
        .dvdx = 10.0F / static_cast<TransportScalar>(texture_extent),
        .dudy = -0.25F / static_cast<TransportScalar>(texture_extent),
        .dvdy = 0.5F / static_cast<TransportScalar>(texture_extent),
    };
    constexpr auto reference_footprint = ReferenceTextureCoordinateDifferentials{
        .dudx = 20.0 / static_cast<ReferenceScalar>(texture_extent),
        .dvdx = 10.0 / static_cast<ReferenceScalar>(texture_extent),
        .dudy = -0.25 / static_cast<ReferenceScalar>(texture_extent),
        .dvdy = 0.5 / static_cast<ReferenceScalar>(texture_extent),
    };
    auto baseline_squared_error = ReferenceScalar{};
    auto ewa_squared_error = ReferenceScalar{};
    auto reference_difference_squared = ReferenceScalar{};
    constexpr auto measurement_extent = std::uint32_t{64U};
    for (auto y = std::uint32_t{}; y < measurement_extent; ++y) {
        for (auto x = std::uint32_t{}; x < measurement_extent; ++x) {
            const auto transport_uv = Point2{
                .x = 0.137F + static_cast<TransportScalar>(x) * transport_footprint.dudx +
                     static_cast<TransportScalar>(y) * transport_footprint.dudy,
                .y = 0.421F + static_cast<TransportScalar>(x) * transport_footprint.dvdx +
                     static_cast<TransportScalar>(y) * transport_footprint.dvdy,
            };
            const auto reference_uv = ReferencePoint2{
                .x = 0.137 + static_cast<ReferenceScalar>(x) * reference_footprint.dudx +
                     static_cast<ReferenceScalar>(y) * reference_footprint.dudy,
                .y = 0.421 + static_cast<ReferenceScalar>(x) * reference_footprint.dvdx +
                     static_cast<ReferenceScalar>(y) * reference_footprint.dvdy,
            };
            const auto baseline = filter_host_image_trilinear_channel(
                *chain, transport_uv, 0.0F, 0U, TextureWrapMode::repeat, TextureWrapMode::repeat);
            const auto filtered = sample_ewa(*chain, transport_uv, transport_footprint);
            const auto reference = sample_ewa(*chain, reference_uv, reference_footprint);
            ASSERT_TRUE(baseline.has_value()) << baseline.error().message;
            ASSERT_TRUE(filtered.has_value()) << filtered.error().message;
            ASSERT_TRUE(reference.has_value()) << reference.error().message;
            const auto baseline_error = static_cast<ReferenceScalar>(*baseline) - 0.5;
            const auto ewa_error = static_cast<ReferenceScalar>(*filtered) - 0.5;
            const auto parity_error = static_cast<ReferenceScalar>(*filtered) - *reference;
            baseline_squared_error += baseline_error * baseline_error;
            ewa_squared_error += ewa_error * ewa_error;
            reference_difference_squared += parity_error * parity_error;
        }
    }
    const auto sample_count = static_cast<ReferenceScalar>(measurement_extent) *
                              static_cast<ReferenceScalar>(measurement_extent);
    const auto baseline_mse = baseline_squared_error / sample_count;
    const auto ewa_mse = ewa_squared_error / sample_count;
    const auto parity_mse = reference_difference_squared / sample_count;
    testing::Test::RecordProperty("baseline_mse", metric_string(baseline_mse));
    testing::Test::RecordProperty("ewa_mse", metric_string(ewa_mse));
    testing::Test::RecordProperty("float_reference_mse", metric_string(parity_mse));
    testing::Test::RecordProperty("ewa_to_baseline_mse", metric_string(ewa_mse / baseline_mse));
    EXPECT_GT(baseline_mse, 0.015);
    EXPECT_LT(ewa_mse, baseline_mse * 0.1);
    EXPECT_LT(ewa_mse, 2.0e-3);
    EXPECT_LT(parity_mse, 1.0e-10);
}

TEST(HostImageEwaFilterTest, RejectsInvalidOrUnboundedRequestsWithoutFallback) {
    constexpr auto one = std::array<TransportScalar, 1>{1.0F};
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto chain =
        load_mip_chain(*cache, write_pfm_fixture("texture-ewa-errors.pfm", 1U, 1U, one));
    ASSERT_TRUE(chain);

    constexpr auto uv = Point2{.x = 0.5F, .y = 0.5F};
    constexpr auto footprint = TextureCoordinateDifferentials{};
    constexpr auto invalid_wrap = static_cast<TextureWrapMode>(0xFFFFFFFFU);
    const auto bad_channel = sample_ewa(*chain, uv, footprint, 1U);
    const auto bad_wrap =
        sample_ewa(*chain, uv, footprint, 0U, invalid_wrap, TextureWrapMode::repeat);
    const auto bad_anisotropy =
        sample_ewa(*chain, uv, footprint, 0U, TextureWrapMode::repeat, TextureWrapMode::repeat,
                   HostImageEwaLimits{.maximum_anisotropy = 0U, .maximum_texel_visits = 4'096U});
    const auto excessive_anisotropy =
        sample_ewa(*chain, uv, footprint, 0U, TextureWrapMode::repeat, TextureWrapMode::repeat,
                   HostImageEwaLimits{.maximum_anisotropy = HostImageEwaMaximumAnisotropy + 1U,
                                      .maximum_texel_visits = 4'096U});
    const auto zero_budget =
        sample_ewa(*chain, uv, footprint, 0U, TextureWrapMode::repeat, TextureWrapMode::repeat,
                   HostImageEwaLimits{.maximum_anisotropy = 16U, .maximum_texel_visits = 0U});
    for (const auto* result :
         {&bad_channel, &bad_wrap, &bad_anisotropy, &excessive_anisotropy, &zero_budget}) {
        ASSERT_FALSE(result->has_value());
        EXPECT_EQ(result->error().code, core::StatusCode::invalid_argument);
    }

    const auto bad_uv = sample_ewa(
        *chain, Point2{.x = std::numeric_limits<TransportScalar>::quiet_NaN(), .y = 0.5F},
        footprint);
    const auto bad_derivative =
        sample_ewa(*chain, uv,
                   TextureCoordinateDifferentials{
                       .dudx = std::numeric_limits<TransportScalar>::infinity(),
                   });
    ASSERT_FALSE(bad_uv.has_value());
    ASSERT_FALSE(bad_derivative.has_value());
    EXPECT_EQ(bad_uv.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(bad_derivative.error().code, core::StatusCode::invalid_argument);

    constexpr auto huge = TextureCoordinateDifferentials{
        .dudx = 1'000.0F,
        .dvdy = 1'000.0F,
    };
    const auto terminal =
        sample_ewa(*chain, uv, huge, 0U, TextureWrapMode::repeat, TextureWrapMode::repeat,
                   HostImageEwaLimits{.maximum_anisotropy = 16U, .maximum_texel_visits = 16U});
    ASSERT_TRUE(terminal.has_value()) << terminal.error().message;
    EXPECT_EQ(*terminal, 1.0F);

    const auto exhausted =
        sample_ewa(*chain, uv, footprint, 0U, TextureWrapMode::repeat, TextureWrapMode::repeat,
                   HostImageEwaLimits{.maximum_anisotropy = 16U, .maximum_texel_visits = 8U});
    const auto exact_budget =
        sample_ewa(*chain, uv, footprint, 0U, TextureWrapMode::repeat, TextureWrapMode::repeat,
                   HostImageEwaLimits{.maximum_anisotropy = 16U, .maximum_texel_visits = 9U});
    ASSERT_FALSE(exhausted.has_value());
    EXPECT_EQ(exhausted.error().code, core::StatusCode::resource_exhausted);
    ASSERT_TRUE(exact_budget.has_value()) << exact_budget.error().message;
    EXPECT_EQ(*exact_budget, 1.0F);
}

#if defined(_WIN32)
[[nodiscard]] std::filesystem::path checksum_output_path() {
    auto* value = static_cast<char*>(nullptr);
    auto value_size = std::size_t{};
    if (_dupenv_s(&value, &value_size, "BLACKFRAME_PNG_CHECKSUM_OUTPUT") != 0 || value == nullptr) {
        return {};
    }
    auto output = std::filesystem::path{value};
    std::free(value);
    return output;
}
#else
[[nodiscard]] std::filesystem::path checksum_output_path() {
    const auto* const value = std::getenv("BLACKFRAME_PNG_CHECKSUM_OUTPUT");
    return value == nullptr ? std::filesystem::path{} : std::filesystem::path{value};
}
#endif

[[nodiscard]] std::vector<std::uint8_t> preview_texture_pixels(const std::uint32_t extent) {
    auto pixels = std::vector<std::uint8_t>{};
    pixels.reserve(static_cast<std::size_t>(extent) * extent);
    for (auto y = std::uint32_t{}; y < extent; ++y) {
        for (auto x = std::uint32_t{}; x < extent; ++x) {
            const auto macro = ((x / 32U) + (y / 32U)) % 2U;
            const auto micro = (x + y) % 2U;
            const auto value =
                macro == 0U ? (micro == 0U ? 24U : 88U) : (micro == 0U ? 168U : 236U);
            pixels.push_back(static_cast<std::uint8_t>(value));
        }
    }
    return pixels;
}

template <GeometryScalar Scalar>
[[nodiscard]] Scalar repeat_point_sample(const HostImage& image, const Point2T<Scalar> uv) {
    const auto scaled_x =
        static_cast<std::int64_t>(std::floor(uv.x * static_cast<Scalar>(image.width())));
    const auto scaled_y =
        static_cast<std::int64_t>(std::floor(uv.y * static_cast<Scalar>(image.height())));
    const auto x =
        wrap_texture_index(scaled_x, image.origin_x(), image.width(), TextureWrapMode::repeat);
    const auto y =
        wrap_texture_index(scaled_y, image.origin_y(), image.height(), TextureWrapMode::repeat);
    if (!x || !y || !*x || !*y) {
        return std::numeric_limits<Scalar>::quiet_NaN();
    }
    const auto column = static_cast<std::size_t>(**x - image.origin_x());
    const auto row = static_cast<std::size_t>(**y - image.origin_y());
    const auto pixel = row * static_cast<std::size_t>(image.width()) + column;
    const auto element = pixel * static_cast<std::size_t>(image.channel_count());
    return static_cast<Scalar>(image.pixels()[element]);
}

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<Scalar>
supersampled_box(const HostImage& image, const Point2T<Scalar> uv,
                 const TextureCoordinateDifferentialsT<Scalar> differentials,
                 const std::uint32_t strata = 16U) {
    if (strata == 0U) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "The EWA box oracle requires a non-zero stratum count.",
        });
    }
    auto sum = ReferenceScalar{};
    for (auto sample_y = std::uint32_t{}; sample_y < strata; ++sample_y) {
        for (auto sample_x = std::uint32_t{}; sample_x < strata; ++sample_x) {
            const auto offset_x =
                (static_cast<Scalar>(sample_x) + Scalar{0.5}) / static_cast<Scalar>(strata) -
                Scalar{0.5};
            const auto offset_y =
                (static_cast<Scalar>(sample_y) + Scalar{0.5}) / static_cast<Scalar>(strata) -
                Scalar{0.5};
            const auto sample_uv = Point2T<Scalar>{
                .x = uv.x + offset_x * differentials.dudx + offset_y * differentials.dudy,
                .y = uv.y + offset_x * differentials.dvdx + offset_y * differentials.dvdy,
            };
            const auto value = repeat_point_sample(image, sample_uv);
            if (!std::isfinite(value)) {
                return std::unexpected(core::Error{
                    .code = core::StatusCode::internal_error,
                    .message = "The EWA preview oracle could not address its source texture.",
                });
            }
            sum += static_cast<ReferenceScalar>(value);
        }
    }
    const auto sample_count =
        static_cast<ReferenceScalar>(strata) * static_cast<ReferenceScalar>(strata);
    return static_cast<Scalar>(sum / sample_count);
}

[[nodiscard]] ReferenceScalar
axis_aligned_ewa_level_reference(const HostImage& image, const ReferencePoint2 uv,
                                 const ReferenceTextureCoordinateDifferentials differentials) {
    const auto sigma_x = std::abs(differentials.dudx * image.width());
    const auto sigma_y = std::abs(differentials.dvdy * image.height());
    const auto radius_x = std::hypot(1.0, sigma_x);
    const auto radius_y = std::hypot(1.0, sigma_y);
    const auto center_x = uv.x * image.width() - 0.5;
    const auto center_y = uv.y * image.height() - 0.5;
    const auto minimum_x = static_cast<std::int64_t>(std::ceil(center_x - radius_x));
    const auto maximum_x = static_cast<std::int64_t>(std::floor(center_x + radius_x));
    const auto minimum_y = static_cast<std::int64_t>(std::ceil(center_y - radius_y));
    const auto maximum_y = static_cast<std::int64_t>(std::floor(center_y + radius_y));
    const auto edge_weight = std::exp(-2.0);
    auto weighted_sum = ReferenceScalar{};
    auto weight_sum = ReferenceScalar{};
    for (auto y = minimum_y; y <= maximum_y; ++y) {
        for (auto x = minimum_x; x <= maximum_x; ++x) {
            const auto normalized_x = (static_cast<ReferenceScalar>(x) - center_x) / radius_x;
            const auto normalized_y = (static_cast<ReferenceScalar>(y) - center_y) / radius_y;
            const auto radius_squared =
                std::fma(normalized_x, normalized_x, normalized_y * normalized_y);
            if (radius_squared >= 1.0) {
                continue;
            }
            const auto weight = edge_weight * std::expm1(2.0 * (1.0 - radius_squared));
            const auto wrapped_x =
                wrap_texture_index(x, image.origin_x(), image.width(), TextureWrapMode::repeat);
            const auto wrapped_y =
                wrap_texture_index(y, image.origin_y(), image.height(), TextureWrapMode::repeat);
            EXPECT_TRUE(wrapped_x.has_value());
            EXPECT_TRUE(wrapped_y.has_value());
            EXPECT_TRUE(wrapped_x && *wrapped_x);
            EXPECT_TRUE(wrapped_y && *wrapped_y);
            if (!wrapped_x || !*wrapped_x || !wrapped_y || !*wrapped_y) {
                return std::numeric_limits<ReferenceScalar>::quiet_NaN();
            }
            const auto column = static_cast<std::size_t>(**wrapped_x - image.origin_x());
            const auto row = static_cast<std::size_t>(**wrapped_y - image.origin_y());
            weighted_sum +=
                weight * static_cast<ReferenceScalar>(image.pixels()[row * image.width() + column]);
            weight_sum += weight;
        }
    }
    EXPECT_GT(weight_sum, 0.0);
    return weighted_sum / weight_sum;
}

TEST(HostImageEwaFilterTest, UsesActualOddMipExtentsForLevelSelection) {
    constexpr auto pixels = std::array<TransportScalar, 15>{
        0.00F, 0.10F, 0.20F, 0.80F, 1.00F, 0.30F, 0.90F, 0.40F,
        0.05F, 0.70F, 1.00F, 0.20F, 0.60F, 0.35F, 0.15F,
    };
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto chain =
        load_mip_chain(*cache, write_pfm_fixture("texture-ewa-odd-5x3.pfm", 5U, 3U, pixels));
    ASSERT_TRUE(chain);
    ASSERT_EQ(chain->level_count(), 3U);
    const auto lower_image = chain->level(0U);
    const auto upper_image = chain->level(1U);
    ASSERT_TRUE(lower_image.has_value());
    ASSERT_TRUE(upper_image.has_value());
    ASSERT_EQ((*lower_image)->width(), 5U);
    ASSERT_EQ((*lower_image)->height(), 3U);
    ASSERT_EQ((*upper_image)->width(), 2U);
    ASSERT_EQ((*upper_image)->height(), 1U);

    constexpr auto uv = ReferencePoint2{.x = 0.37, .y = 0.61};
    constexpr auto footprint = ReferenceTextureCoordinateDifferentials{
        .dudx = 2.0 / 5.0,
        .dvdy = 2.0 / 3.0,
    };
    const auto filtered = sample_ewa(*chain, uv, footprint);
    ASSERT_TRUE(filtered.has_value()) << filtered.error().message;

    const auto lower = axis_aligned_ewa_level_reference(**lower_image, uv, footprint);
    const auto upper = axis_aligned_ewa_level_reference(**upper_image, uv, footprint);
    const auto lower_minor = std::min(std::abs(footprint.dudx * (*lower_image)->width()),
                                      std::abs(footprint.dvdy * (*lower_image)->height()));
    const auto upper_minor = std::min(std::abs(footprint.dudx * (*upper_image)->width()),
                                      std::abs(footprint.dvdy * (*upper_image)->height()));
    const auto fraction = std::log(lower_minor) / (std::log(lower_minor) - std::log(upper_minor));
    const auto expected = std::lerp(lower, upper, fraction);
    EXPECT_NEAR(*filtered, expected, 2.0e-15);

    const auto assumed_half_extent = std::lerp(lower, upper, 1.0);
    EXPECT_GT(std::abs(*filtered - assumed_half_extent), 1.0e-3);
}

TEST(HostImageEwaFilterTest, TracksTheSharedObliqueBoxOracle) {
    constexpr auto texture_extent = std::uint32_t{256U};
    constexpr auto footprint = ReferenceTextureCoordinateDifferentials{
        .dudx = 20.0 / texture_extent,
        .dvdx = 10.0 / texture_extent,
        .dudy = -0.25 / texture_extent,
        .dvdy = 0.5 / texture_extent,
    };
    constexpr auto transport_footprint = TextureCoordinateDifferentials{
        .dudx = static_cast<TransportScalar>(footprint.dudx),
        .dvdx = static_cast<TransportScalar>(footprint.dvdx),
        .dudy = static_cast<TransportScalar>(footprint.dudy),
        .dvdy = static_cast<TransportScalar>(footprint.dvdy),
    };
    const auto pixels = preview_texture_pixels(texture_extent);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto chain =
        load_mip_chain(*cache, write_pgm_fixture("texture-ewa-oracle-source.pgm", texture_extent,
                                                 texture_extent, pixels));
    ASSERT_TRUE(chain);
    const auto base = chain->level(0U);
    ASSERT_TRUE(base.has_value());
    const auto major_lod = std::log2(std::hypot(20.0, 10.0));

    auto bilinear_squared_error = ReferenceScalar{};
    auto trilinear_squared_error = ReferenceScalar{};
    auto ewa_squared_error = ReferenceScalar{};
    auto ewa_bias = ReferenceScalar{};
    auto float_reference_squared_error = ReferenceScalar{};
    auto float_reference_max_abs = ReferenceScalar{};
    constexpr auto measurement_extent = std::uint32_t{32U};
    for (auto y = std::uint32_t{}; y < measurement_extent; ++y) {
        for (auto x = std::uint32_t{}; x < measurement_extent; ++x) {
            const auto uv = ReferencePoint2{
                .x = 0.137 + static_cast<ReferenceScalar>(x) * footprint.dudx +
                     static_cast<ReferenceScalar>(y) * footprint.dudy,
                .y = 0.421 + static_cast<ReferenceScalar>(x) * footprint.dvdx +
                     static_cast<ReferenceScalar>(y) * footprint.dvdy,
            };
            const auto transport_uv = Point2{.x = static_cast<TransportScalar>(uv.x),
                                             .y = static_cast<TransportScalar>(uv.y)};
            const auto bilinear =
                filter_host_image_channel(**base, uv, 0U, TextureFilterMode::bilinear,
                                          TextureWrapMode::repeat, TextureWrapMode::repeat);
            const auto trilinear = filter_host_image_trilinear_channel(
                *chain, uv, major_lod, 0U, TextureWrapMode::repeat, TextureWrapMode::repeat);
            const auto ewa = sample_ewa(*chain, uv, footprint);
            const auto ewa_transport = sample_ewa(*chain, transport_uv, transport_footprint);
            const auto oracle = supersampled_box(**base, uv, footprint, 16U);
            ASSERT_TRUE(bilinear.has_value()) << bilinear.error().message;
            ASSERT_TRUE(trilinear.has_value()) << trilinear.error().message;
            ASSERT_TRUE(ewa.has_value()) << ewa.error().message;
            ASSERT_TRUE(ewa_transport.has_value()) << ewa_transport.error().message;
            ASSERT_TRUE(oracle.has_value()) << oracle.error().message;
            const auto bilinear_error = *bilinear - *oracle;
            const auto trilinear_error = *trilinear - *oracle;
            const auto ewa_error = *ewa - *oracle;
            const auto parity_error = static_cast<ReferenceScalar>(*ewa_transport) - *ewa;
            bilinear_squared_error += bilinear_error * bilinear_error;
            trilinear_squared_error += trilinear_error * trilinear_error;
            ewa_squared_error += ewa_error * ewa_error;
            ewa_bias += ewa_error;
            float_reference_squared_error += parity_error * parity_error;
            float_reference_max_abs = std::max(float_reference_max_abs, std::abs(parity_error));
        }
    }
    const auto sample_count = static_cast<ReferenceScalar>(measurement_extent) *
                              static_cast<ReferenceScalar>(measurement_extent);
    const auto bilinear_mse = bilinear_squared_error / sample_count;
    const auto trilinear_mse = trilinear_squared_error / sample_count;
    const auto ewa_mse = ewa_squared_error / sample_count;
    const auto parity_mse = float_reference_squared_error / sample_count;
    const auto mean_bias = ewa_bias / sample_count;
    testing::Test::RecordProperty("bilinear_oracle_mse", metric_string(bilinear_mse));
    testing::Test::RecordProperty("trilinear_oracle_mse", metric_string(trilinear_mse));
    testing::Test::RecordProperty("ewa_oracle_mse", metric_string(ewa_mse));
    testing::Test::RecordProperty("ewa_oracle_rmse", metric_string(std::sqrt(ewa_mse)));
    testing::Test::RecordProperty("ewa_oracle_bias", metric_string(mean_bias));
    testing::Test::RecordProperty("float_reference_mse", metric_string(parity_mse));
    testing::Test::RecordProperty("float_reference_max_abs",
                                  metric_string(float_reference_max_abs));
    EXPECT_LT(ewa_mse, bilinear_mse * 0.15);
    EXPECT_LT(ewa_mse, trilinear_mse * 0.4);
    EXPECT_LT(std::abs(mean_bias), 0.03);
    EXPECT_LT(parity_mse, 1.0e-10);
    EXPECT_LT(float_reference_max_abs, 1.0e-4);

    auto oracle_16_to_64_squared_error = ReferenceScalar{};
    auto oracle_32_to_64_squared_error = ReferenceScalar{};
    constexpr auto convergence_extent = std::uint32_t{8U};
    for (auto y = std::uint32_t{}; y < convergence_extent; ++y) {
        for (auto x = std::uint32_t{}; x < convergence_extent; ++x) {
            const auto uv = ReferencePoint2{
                .x = 0.173 + static_cast<ReferenceScalar>(x) * footprint.dudx +
                     static_cast<ReferenceScalar>(y) * footprint.dudy,
                .y = 0.397 + static_cast<ReferenceScalar>(x) * footprint.dvdx +
                     static_cast<ReferenceScalar>(y) * footprint.dvdy,
            };
            const auto oracle_16 = supersampled_box(**base, uv, footprint, 16U);
            const auto oracle_32 = supersampled_box(**base, uv, footprint, 32U);
            const auto oracle_64 = supersampled_box(**base, uv, footprint, 64U);
            ASSERT_TRUE(oracle_16.has_value());
            ASSERT_TRUE(oracle_32.has_value());
            ASSERT_TRUE(oracle_64.has_value());
            const auto error_16 = *oracle_16 - *oracle_64;
            const auto error_32 = *oracle_32 - *oracle_64;
            oracle_16_to_64_squared_error += error_16 * error_16;
            oracle_32_to_64_squared_error += error_32 * error_32;
        }
    }
    const auto convergence_count = static_cast<ReferenceScalar>(convergence_extent) *
                                   static_cast<ReferenceScalar>(convergence_extent);
    const auto oracle_16_to_64_mse = oracle_16_to_64_squared_error / convergence_count;
    const auto oracle_32_to_64_mse = oracle_32_to_64_squared_error / convergence_count;
    testing::Test::RecordProperty("oracle_16_to_64_mse", metric_string(oracle_16_to_64_mse));
    testing::Test::RecordProperty("oracle_32_to_64_mse", metric_string(oracle_32_to_64_mse));
    EXPECT_LT(oracle_32_to_64_mse, oracle_16_to_64_mse);
    EXPECT_LT(oracle_32_to_64_mse, 5.0e-4);
}

#if defined(BLACKFRAME_HOST_IMAGE_FILTER_PNG)
[[nodiscard]] core::Status write_ewa_preview(const HostImageMipChain& chain,
                                             const std::filesystem::path& output) {
    constexpr auto output_extent = std::uint32_t{800U};
    constexpr auto panel_extent = output_extent / 2U;
    constexpr auto texture_extent = TransportScalar{256};
    constexpr auto differentials = TextureCoordinateDifferentials{
        .dudx = 20.0F / texture_extent,
        .dvdx = 10.0F / texture_extent,
        .dudy = -0.25F / texture_extent,
        .dvdy = 0.5F / texture_extent,
    };
    const auto major_lod = std::log2(std::hypot(20.0F, 10.0F));
    const auto base = chain.level(0U);
    if (!base) {
        return std::unexpected(base.error());
    }
    auto film = Film::create(RenderExtent{.width = output_extent, .height = output_extent});
    if (!film) {
        return std::unexpected(film.error());
    }
    for (auto y = std::uint32_t{}; y < output_extent; ++y) {
        for (auto x = std::uint32_t{}; x < output_extent; ++x) {
            const auto local_x = x % panel_extent;
            const auto local_y = y % panel_extent;
            auto value = TransportScalar{0.02F};
            if (local_x > 2U && local_y > 2U) {
                const auto uv = Point2{
                    .x = 0.137F + static_cast<TransportScalar>(local_x) * differentials.dudx +
                         static_cast<TransportScalar>(local_y) * differentials.dudy,
                    .y = 0.421F + static_cast<TransportScalar>(local_x) * differentials.dvdx +
                         static_cast<TransportScalar>(local_y) * differentials.dvdy,
                };
                auto filtered = core::Result<TransportScalar>{};
                if (x < panel_extent && y < panel_extent) {
                    filtered =
                        filter_host_image_channel(**base, uv, 0U, TextureFilterMode::bilinear,
                                                  TextureWrapMode::repeat, TextureWrapMode::repeat);
                } else if (x >= panel_extent && y < panel_extent) {
                    filtered = filter_host_image_trilinear_channel(
                        chain, uv, major_lod, 0U, TextureWrapMode::repeat, TextureWrapMode::repeat);
                } else if (x < panel_extent) {
                    filtered = sample_ewa(chain, uv, differentials);
                } else {
                    filtered = supersampled_box(**base, uv, differentials);
                }
                if (!filtered) {
                    return std::unexpected(filtered.error());
                }
                value = *filtered;
            }
            const auto status = film->add_sample(
                x, y, LinearRGB{.red = value, .green = value, .blue = value}, 1.0F);
            if (!status) {
                return status;
            }
        }
    }
    return write_png_preview(*film, output);
}
#endif

TEST(HostImageEwaFilterTest, WritesStableObliqueFilteringAtlas) {
    const auto output = checksum_output_path();
    if (output.empty()) {
        GTEST_SKIP() << "The explicit PNG checksum output path was not supplied.";
    }
#if !defined(BLACKFRAME_HOST_IMAGE_FILTER_PNG)
    FAIL() << "The EWA filtering atlas requires the explicit PNG capability.";
#else
    constexpr auto texture_extent = std::uint32_t{256U};
    const auto pixels = preview_texture_pixels(texture_extent);
    auto cache = HostImageCache::create();
    ASSERT_TRUE(cache.has_value());
    const auto chain =
        load_mip_chain(*cache, write_pgm_fixture("texture-ewa-preview-source.pgm", texture_extent,
                                                 texture_extent, pixels));
    ASSERT_TRUE(chain);
    const auto status = write_ewa_preview(*chain, output);
    ASSERT_TRUE(status.has_value()) << status.error().message;
    ASSERT_TRUE(std::filesystem::is_regular_file(output));
    testing::Test::RecordProperty(
        "atlas_layout",
        "top-left=bilinear;top-right=isotropic-trilinear;bottom-left=EWA;bottom-right=16x16-box-"
        "oracle");
#endif
}

} // namespace
} // namespace blackframe::renderer
