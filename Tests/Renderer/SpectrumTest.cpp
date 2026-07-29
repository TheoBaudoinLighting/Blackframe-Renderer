#include <Blackframe/Renderer/Color.hpp>
#include <Blackframe/Renderer/Spectrum.hpp>
#include <cmath>
#include <concepts>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {
namespace {

template <std::size_t SampleCount>
concept SupportedSpectrumExtent = requires { typename SampledSpectrum<SampleCount>; };

TEST(SampledSpectrumTest, FixesTransportToFourFloatSamples) {
    static_assert(SupportedSpectrumExtent<4>);
    static_assert(!SupportedSpectrumExtent<3>);
    static_assert(!std::same_as<TransportSpectrum, ReferenceSpectrum>);
    static_assert(std::is_trivially_copyable_v<TransportSpectrum>);
    static_assert(std::is_trivially_copyable_v<ReferenceSpectrum>);

    EXPECT_EQ(TransportSpectrumSampleCount, 4U);
    EXPECT_EQ(sizeof(TransportSpectrum), 4U * sizeof(TransportScalar));
    EXPECT_EQ(sizeof(ReferenceSpectrum), 4U * sizeof(ReferenceScalar));
}

TEST(SampledSpectrumTest, AppliesLaneWiseArithmeticWithoutChangingPrecision) {
    constexpr auto left = SampledSpectrum<4>{.values = {1.0F, 2.0F, 3.0F, 4.0F}};
    constexpr auto right = SampledSpectrum<4>{.values = {0.5F, -1.0F, 2.0F, 8.0F}};

    constexpr auto sum = left + right;
    constexpr auto difference = left - right;
    constexpr auto product = left * right;
    constexpr auto scaled = 2.0F * left;
    constexpr auto divided = left / 2.0F;

    static_assert(std::same_as<decltype(sum), const SampledSpectrum<4>>);
    EXPECT_EQ(sum, (SampledSpectrum<4>{.values = {1.5F, 1.0F, 5.0F, 12.0F}}));
    EXPECT_EQ(difference, (SampledSpectrum<4>{.values = {0.5F, 3.0F, 1.0F, -4.0F}}));
    EXPECT_EQ(product, (SampledSpectrum<4>{.values = {0.5F, -2.0F, 6.0F, 32.0F}}));
    EXPECT_EQ(scaled, (SampledSpectrum<4>{.values = {2.0F, 4.0F, 6.0F, 8.0F}}));
    EXPECT_EQ(divided, (SampledSpectrum<4>{.values = {0.5F, 1.0F, 1.5F, 2.0F}}));
    EXPECT_EQ(-left, (SampledSpectrum<4>{.values = {-1.0F, -2.0F, -3.0F, -4.0F}}));
}

TEST(SampledSpectrumMaskTest, CombinesReducesAndSelectsLanes) {
    constexpr auto left = SampledSpectrum<4>{.values = {1.0F, 4.0F, 3.0F, 8.0F}};
    constexpr auto right = SampledSpectrum<4>{.values = {2.0F, 2.0F, 3.0F, 9.0F}};
    constexpr auto less = left < right;
    constexpr auto greater_or_equal = left >= right;

    static_assert(less == SampledSpectrumMask<4>{.lanes = {true, false, false, true}});
    static_assert(greater_or_equal == SampledSpectrumMask<4>{.lanes = {false, true, true, false}});
    EXPECT_TRUE(any(less));
    EXPECT_FALSE(all(less));
    EXPECT_FALSE(none(less));
    EXPECT_TRUE(all(less | greater_or_equal));
    EXPECT_TRUE(none(less & greater_or_equal));
    EXPECT_EQ(less ^ greater_or_equal, (SampledSpectrumMask<4>{.lanes = {true, true, true, true}}));
    EXPECT_EQ(~less, (SampledSpectrumMask<4>{.lanes = {false, true, true, false}}));

    constexpr auto selected = select(less, left, right);
    EXPECT_EQ(selected, (SampledSpectrum<4>{.values = {1.0F, 2.0F, 3.0F, 8.0F}}));
}

TEST(ColorConversionTest, ConvertsLinearSrgbPrimariesToCieXyz) {
    const auto red = linear_rgb_to_xyz(LinearRGB{.red = 1.0F});
    const auto green = linear_rgb_to_xyz(LinearRGB{.green = 1.0F});
    const auto blue = linear_rgb_to_xyz(LinearRGB{.blue = 1.0F});

    ASSERT_TRUE(red.has_value());
    ASSERT_TRUE(green.has_value());
    ASSERT_TRUE(blue.has_value());
    EXPECT_NEAR(red->x, 0.4124564F, 1.0E-7F);
    EXPECT_NEAR(red->y, 0.2126729F, 1.0E-7F);
    EXPECT_NEAR(red->z, 0.0193339F, 1.0E-7F);
    EXPECT_NEAR(green->x, 0.3575761F, 1.0E-7F);
    EXPECT_NEAR(green->y, 0.7151522F, 1.0E-7F);
    EXPECT_NEAR(green->z, 0.1191920F, 1.0E-7F);
    EXPECT_NEAR(blue->x, 0.1804375F, 1.0E-7F);
    EXPECT_NEAR(blue->y, 0.0721750F, 1.0E-7F);
    EXPECT_NEAR(blue->z, 0.9503041F, 1.0E-7F);
}

TEST(ColorConversionTest, RoundTripsLinearRgbWithoutClampingSignedValues) {
    constexpr auto input = ReferenceLinearRGB{.red = 1.25, .green = -0.1, .blue = 0.5};
    const auto xyz = linear_rgb_to_xyz(input);
    ASSERT_TRUE(xyz.has_value());
    const auto round_trip = xyz_to_linear_rgb(*xyz);
    ASSERT_TRUE(round_trip.has_value());

    EXPECT_NEAR(round_trip->red, input.red, 1.0E-6);
    EXPECT_NEAR(round_trip->green, input.green, 1.0E-6);
    EXPECT_NEAR(round_trip->blue, input.blue, 1.0E-6);
    EXPECT_LT(xyz->y, 1.0);

    const auto outside_gamut = xyz_to_linear_rgb(ReferenceXYZ{.x = 0.0, .y = 1.0, .z = 0.0});
    ASSERT_TRUE(outside_gamut.has_value());
    EXPECT_LT(outside_gamut->red, 0.0);
    EXPECT_GT(outside_gamut->green, 1.0);
    EXPECT_LT(outside_gamut->blue, 0.0);
}

TEST(ColorConversionTest, RejectsNonFiniteInputsAndOverflowWithoutFallback) {
    const auto invalid_rgb = linear_rgb_to_xyz(LinearRGB{
        .red = std::numeric_limits<TransportScalar>::infinity(),
    });
    ASSERT_FALSE(invalid_rgb.has_value());
    EXPECT_EQ(invalid_rgb.error().code, core::StatusCode::invalid_argument);

    const auto overflow = xyz_to_linear_rgb(XYZ{
        .x = std::numeric_limits<TransportScalar>::max(),
        .y = -std::numeric_limits<TransportScalar>::max(),
        .z = std::numeric_limits<TransportScalar>::max(),
    });
    ASSERT_FALSE(overflow.has_value());
    EXPECT_EQ(overflow.error().code, core::StatusCode::invalid_argument);
}

} // namespace
} // namespace blackframe::renderer
