#include <Blackframe/Renderer/IndependentSampler.hpp>
#include <Blackframe/Renderer/SampleDimensionMap.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {
namespace {

struct KnownIndependentSample final {
    SampleStreamIndex index;
    SampleDimension dimension;
    std::uint32_t transport_numerator;
    std::uint64_t reference_numerator;
};

constexpr auto known_samples = std::array{
    KnownIndependentSample{
        .index = {},
        .dimension = 0,
        .transport_numerator = 0x2F37B1U,
        .reference_numerator = 0x05E6F6339B5611ULL,
    },
    KnownIndependentSample{
        .index = {.pixel_x = 17, .pixel_y = 29, .sample_index = 0, .seed = 0x0123456789ABCDEFULL},
        .dimension = 0,
        .transport_numerator = 0xDE27E5U,
        .reference_numerator = 0x1BC4FCA4264791ULL,
    },
    KnownIndependentSample{
        .index =
            {.pixel_x = 17, .pixel_y = 29, .sample_index = 4095, .seed = 0x0123456789ABCDEFULL},
        .dimension = 1,
        .transport_numerator = 0xDC4661U,
        .reference_numerator = 0x1B88CC28476D40ULL,
    },
    KnownIndependentSample{
        .index = {.pixel_x = std::numeric_limits<std::uint32_t>::max(),
                  .pixel_y = std::numeric_limits<std::uint32_t>::max(),
                  .sample_index = std::numeric_limits<std::uint64_t>::max(),
                  .seed = std::numeric_limits<std::uint64_t>::max()},
        .dimension = std::numeric_limits<SampleDimension>::max(),
        .transport_numerator = 0x8F4FD2U,
        .reference_numerator = 0x11E9FA55578958ULL,
    },
};

template <GeometryScalar Scalar> void expect_acceptable_independent_uniformity() {
    constexpr auto bin_count = std::size_t{16};
    constexpr auto sample_count = std::size_t{65'536};
    constexpr auto expected_per_bin =
        static_cast<double>(sample_count) / static_cast<double>(bin_count * bin_count);
    constexpr auto sampler_seed = std::uint64_t{0xA5A5F00D12345678ULL};
    const auto bounce_dimensions = sample_dimensions_for_bounce(0);
    ASSERT_TRUE(bounce_dimensions.has_value());

    const auto sampler = IndependentSamplerT<Scalar>{sampler_seed};
    std::array<std::uint32_t, bin_count * bin_count> histogram{};
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_x_squared = 0.0;
    double sum_y_squared = 0.0;
    double sum_xy = 0.0;

    for (std::size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
        const auto stream = sampler.make_stream(113, 47, static_cast<std::uint64_t>(sample_index));
        const auto x = stream.sample_1d(PrimarySampleDimensionMap.lens_u);
        const auto y = stream.sample_1d(bounce_dimensions->russian_roulette);
        ASSERT_GE(x, Scalar{0});
        ASSERT_LT(x, Scalar{1});
        ASSERT_GE(y, Scalar{0});
        ASSERT_LT(y, Scalar{1});

        const auto bin_x = static_cast<std::size_t>(x * static_cast<Scalar>(bin_count));
        const auto bin_y = static_cast<std::size_t>(y * static_cast<Scalar>(bin_count));
        ASSERT_LT(bin_x, bin_count);
        ASSERT_LT(bin_y, bin_count);
        ++histogram[bin_y * bin_count + bin_x];

        sum_x += x;
        sum_y += y;
        sum_x_squared += x * x;
        sum_y_squared += y * y;
        sum_xy += x * y;
    }

    const auto inverse_count = 1.0 / static_cast<double>(sample_count);
    const auto mean_x = sum_x * inverse_count;
    const auto mean_y = sum_y * inverse_count;
    const auto covariance = sum_xy * inverse_count - mean_x * mean_y;
    double chi_squared = 0.0;
    for (const auto observed : histogram) {
        const auto difference = static_cast<double>(observed) - expected_per_bin;
        chi_squared += difference * difference / expected_per_bin;
    }

    EXPECT_NEAR(mean_x, 0.5, 0.005);
    EXPECT_NEAR(mean_y, 0.5, 0.005);
    EXPECT_NEAR(sum_x_squared * inverse_count, 1.0 / 3.0, 0.006);
    EXPECT_NEAR(sum_y_squared * inverse_count, 1.0 / 3.0, 0.006);
    EXPECT_NEAR(covariance, 0.0, 0.003);
    EXPECT_LT(chi_squared, 400.0);
}

TEST(IndependentSamplerTest, BuildsTheExactStreamAddressWithoutADefaultSeed) {
    static_assert(std::is_same_v<IndependentSampler::value_type, TransportScalar>);
    static_assert(std::is_same_v<ReferenceIndependentSampler::value_type, ReferenceScalar>);
    static_assert(!std::is_default_constructible_v<IndependentSampler>);
    static_assert(!std::is_default_constructible_v<ReferenceIndependentSampler>);
    static_assert(std::is_standard_layout_v<IndependentSampler>);
    static_assert(std::is_trivially_copyable_v<IndependentSampler>);
    static_assert(std::is_standard_layout_v<ReferenceIndependentSampler>);
    static_assert(std::is_trivially_copyable_v<ReferenceIndependentSampler>);

    constexpr auto sampler = IndependentSampler{std::numeric_limits<std::uint64_t>::max()};
    constexpr auto stream = sampler.make_stream(std::numeric_limits<std::uint32_t>::max(),
                                                std::numeric_limits<std::uint32_t>::max(),
                                                std::numeric_limits<std::uint64_t>::max());
    constexpr auto expected = SampleStreamIndex{
        .pixel_x = std::numeric_limits<std::uint32_t>::max(),
        .pixel_y = std::numeric_limits<std::uint32_t>::max(),
        .sample_index = std::numeric_limits<std::uint64_t>::max(),
        .seed = std::numeric_limits<std::uint64_t>::max(),
    };
    static_assert(sampler.seed() == expected.seed);
    static_assert(stream.index() == expected);
    EXPECT_EQ(sampler.seed(), expected.seed);
    EXPECT_EQ(stream.index(), expected);
    EXPECT_GE(stream.sample_1d(std::numeric_limits<SampleDimension>::max()), 0.0F);
    EXPECT_LT(stream.sample_1d(std::numeric_limits<SampleDimension>::max()), 1.0F);
}

TEST(IndependentSamplerTest, MatchesTheFrozenHashOracleInBothPrecisions) {
    for (const auto& expected : known_samples) {
        const auto transport_sampler = IndependentSampler{expected.index.seed};
        const auto reference_sampler = ReferenceIndependentSampler{expected.index.seed};
        const auto transport = transport_sampler
                                   .make_stream(expected.index.pixel_x, expected.index.pixel_y,
                                                expected.index.sample_index)
                                   .sample_1d(expected.dimension);
        const auto reference = reference_sampler
                                   .make_stream(expected.index.pixel_x, expected.index.pixel_y,
                                                expected.index.sample_index)
                                   .sample_1d(expected.dimension);

        EXPECT_EQ(transport, static_cast<TransportScalar>(expected.transport_numerator) * 0x1p-24F);
        EXPECT_EQ(reference, static_cast<ReferenceScalar>(expected.reference_numerator) * 0x1p-53);
    }
}

TEST(IndependentSamplerTest, CoversTheExactClosedOpenConversionEndpoints) {
    constexpr auto zero_hash_dimension = SampleDimension{0xD9DFEE5D0039B834ULL};
    constexpr auto maximum_hash_dimension = SampleDimension{0x1645EAF2FA5215F4ULL};
    constexpr auto expected_transport_maximum = static_cast<TransportScalar>(0xFFFFFFU) * 0x1p-24F;
    constexpr auto expected_reference_maximum =
        static_cast<ReferenceScalar>(0x1FFFFFFFFFFFFFULL) * 0x1p-53;

    constexpr auto transport = IndependentSampler{0}.make_stream(0, 0, 0);
    constexpr auto reference = ReferenceIndependentSampler{0}.make_stream(0, 0, 0);
    static_assert(transport.sample_1d(zero_hash_dimension) == 0.0F);
    static_assert(reference.sample_1d(zero_hash_dimension) == 0.0);
    static_assert(transport.sample_1d(maximum_hash_dimension) == expected_transport_maximum);
    static_assert(reference.sample_1d(maximum_hash_dimension) == expected_reference_maximum);

    EXPECT_EQ(transport.sample_1d(zero_hash_dimension), 0.0F);
    EXPECT_EQ(reference.sample_1d(zero_hash_dimension), 0.0);
    EXPECT_EQ(transport.sample_1d(maximum_hash_dimension), expected_transport_maximum);
    EXPECT_EQ(reference.sample_1d(maximum_hash_dimension), expected_reference_maximum);
    EXPECT_LT(expected_transport_maximum, 1.0F);
    EXPECT_LT(expected_reference_maximum, 1.0);
}

TEST(IndependentSamplerTest, IsStableAcrossOrderAndInterleavedInstances) {
    constexpr auto sample_count = std::size_t{2048};
    constexpr auto first = IndependentSampler{0x0123456789ABCDEFULL};
    constexpr auto second = IndependentSampler{0xFEDCBA9876543210ULL};
    std::array<TransportScalar, sample_count> recorded{};

    for (std::size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
        const auto dimension = static_cast<SampleDimension>(sample_index * 4051U + 17U);
        recorded[sample_index] = first.make_stream(17, 29, static_cast<std::uint64_t>(sample_index))
                                     .sample_1d(dimension);
        static_cast<void>(
            second.make_stream(31, 43, static_cast<std::uint64_t>(sample_count - sample_index))
                .sample_1d(dimension ^ 0x8000000000000000ULL));
    }

    for (std::size_t reverse = sample_count; reverse > 0; --reverse) {
        const auto sample_index = reverse - 1;
        const auto dimension = static_cast<SampleDimension>(sample_index * 4051U + 17U);
        EXPECT_EQ(first.make_stream(17, 29, static_cast<std::uint64_t>(sample_index))
                      .sample_1d(dimension),
                  recorded[sample_index]);
    }
}

TEST(IndependentSamplerTest, HasAcceptableTransportUniformityAndIndependence) {
    expect_acceptable_independent_uniformity<TransportScalar>();
}

TEST(IndependentSamplerTest, HasAcceptableReferenceUniformityAndIndependence) {
    expect_acceptable_independent_uniformity<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
