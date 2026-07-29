#include <Blackframe/Renderer/NumericConversion.hpp>
#include <Blackframe/Renderer/NumericPrecision.hpp>
#include <Blackframe/Renderer/TransportConventions.hpp>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {
namespace {

TEST(NumericPrecisionTest, FixesScalarRolesAndSizes) {
    static_assert(std::is_same_v<TransportScalar, float>);
    static_assert(std::is_same_v<ReferenceScalar, double>);
    static_assert(
        std::is_same_v<AccumulationScalar<AccumulationPrecision::float32>, TransportScalar>);
    static_assert(
        std::is_same_v<AccumulationScalar<AccumulationPrecision::float64>, ReferenceScalar>);

    EXPECT_EQ(sizeof(TransportScalar), 4U);
    EXPECT_EQ(sizeof(ReferenceScalar), 8U);
    EXPECT_TRUE(std::numeric_limits<TransportScalar>::is_iec559);
    EXPECT_TRUE(std::numeric_limits<ReferenceScalar>::is_iec559);
    EXPECT_EQ(std::numeric_limits<TransportScalar>::digits, 24);
    EXPECT_EQ(std::numeric_limits<ReferenceScalar>::digits, 53);
    EXPECT_EQ(sizeof(AccumulationScalar<AccumulationPrecision::float32>), 4U);
    EXPECT_EQ(sizeof(AccumulationScalar<AccumulationPrecision::float64>), 8U);
    EXPECT_EQ(sizeof(ProbabilityDensity::value), sizeof(TransportScalar));
    EXPECT_EQ(sizeof(WavelengthSample::nanometers), sizeof(TransportScalar));
}

TEST(NumericPrecisionTest, WidensEveryFiniteTransportBoundaryExactly) {
    constexpr auto values = std::array{
        TransportScalar{0},
        -TransportScalar{0},
        TransportScalar{1},
        std::numeric_limits<TransportScalar>::denorm_min(),
        std::numeric_limits<TransportScalar>::lowest(),
        std::numeric_limits<TransportScalar>::max(),
    };

    for (const auto value : values) {
        const auto widened = to_reference_scalar(value);
        EXPECT_EQ(static_cast<TransportScalar>(widened), value);
        EXPECT_EQ(std::signbit(widened), std::signbit(value));
    }
}

TEST(NumericPrecisionTest, NarrowsFiniteReferenceValuesWithExplicitRounding) {
    constexpr auto exactly_representable = ReferenceScalar{16.5};
    const auto exact = to_transport_scalar(exactly_representable);
    ASSERT_TRUE(exact.has_value());
    EXPECT_EQ(*exact, TransportScalar{16.5F});

    constexpr auto rounded_input = ReferenceScalar{0.1};
    const auto rounded = to_transport_scalar(rounded_input);
    ASSERT_TRUE(rounded.has_value());
    EXPECT_EQ(*rounded, static_cast<TransportScalar>(rounded_input));

    constexpr auto boundaries = std::array{
        static_cast<ReferenceScalar>(std::numeric_limits<TransportScalar>::denorm_min()),
        static_cast<ReferenceScalar>(std::numeric_limits<TransportScalar>::lowest()),
        static_cast<ReferenceScalar>(std::numeric_limits<TransportScalar>::max()),
    };
    for (const auto value : boundaries) {
        const auto converted = to_transport_scalar(value);
        ASSERT_TRUE(converted.has_value());
        EXPECT_EQ(*converted, static_cast<TransportScalar>(value));
    }

    const auto negative_zero = to_transport_scalar(-ReferenceScalar{0});
    ASSERT_TRUE(negative_zero.has_value());
    EXPECT_TRUE(std::signbit(*negative_zero));
}

TEST(NumericPrecisionTest, RejectsInvalidNarrowingWithoutAReplacementValue) {
    const auto overflow =
        std::nextafter(static_cast<ReferenceScalar>(std::numeric_limits<TransportScalar>::max()),
                       std::numeric_limits<ReferenceScalar>::infinity());
    const auto underflow =
        static_cast<ReferenceScalar>(std::numeric_limits<TransportScalar>::denorm_min()) / 4.0;

    const auto invalid_values = std::array{
        std::numeric_limits<ReferenceScalar>::infinity(),
        -std::numeric_limits<ReferenceScalar>::infinity(),
        std::numeric_limits<ReferenceScalar>::quiet_NaN(),
        overflow,
        -overflow,
        underflow,
        -underflow,
    };
    for (const auto value : invalid_values) {
        const auto conversion = to_transport_scalar(value);
        ASSERT_FALSE(conversion.has_value());
        EXPECT_EQ(conversion.error().code, core::StatusCode::invalid_argument);
    }
}

} // namespace
} // namespace blackframe::renderer
