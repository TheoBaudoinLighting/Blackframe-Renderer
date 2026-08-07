#include <Blackframe/Renderer/TextureWrap.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <type_traits>

namespace blackframe::renderer {
namespace {

void expect_index(const std::int64_t input, const std::int64_t origin, const std::uint32_t extent,
                  const TextureWrapMode mode, const std::optional<std::int64_t> expected) {
    const auto result = wrap_texture_index(input, origin, extent, mode);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_EQ(*result, expected) << input;
}

[[nodiscard]] std::int64_t point_tap_from_uv(const double coordinate, const std::uint32_t extent) {
    return static_cast<std::int64_t>(std::floor(coordinate * static_cast<double>(extent)));
}

TEST(TextureWrapTest, KeepsModesDistinctAndStable) {
    static_assert(sizeof(TextureWrapMode) == sizeof(std::uint32_t));
    static_assert(std::is_trivially_copyable_v<TextureWrapMode>);
    static_assert(is_valid_texture_wrap_mode(TextureWrapMode::repeat));
    static_assert(is_valid_texture_wrap_mode(TextureWrapMode::clamp));
    static_assert(is_valid_texture_wrap_mode(TextureWrapMode::mirror));
    static_assert(is_valid_texture_wrap_mode(TextureWrapMode::black));
    static_assert(!is_valid_texture_wrap_mode(static_cast<TextureWrapMode>(0xFFFFFFFFU)));
}

TEST(TextureWrapTest, ResolvesTexelTapsDerivedFromOutOfDomainUv) {
    constexpr auto extent = 4U;
    const auto below = point_tap_from_uv(-0.25, extent);
    const auto lower_edge = point_tap_from_uv(0.0, extent);
    const auto interior = point_tap_from_uv(0.75, extent);
    const auto upper_edge = point_tap_from_uv(1.0, extent);
    const auto above = point_tap_from_uv(1.25, extent);
    ASSERT_EQ(below, -1);
    ASSERT_EQ(lower_edge, 0);
    ASSERT_EQ(interior, 3);
    ASSERT_EQ(upper_edge, 4);
    ASSERT_EQ(above, 5);

    expect_index(below, 0, extent, TextureWrapMode::repeat, 3);
    expect_index(upper_edge, 0, extent, TextureWrapMode::repeat, 0);
    expect_index(above, 0, extent, TextureWrapMode::repeat, 1);

    expect_index(below, 0, extent, TextureWrapMode::clamp, 0);
    expect_index(upper_edge, 0, extent, TextureWrapMode::clamp, 3);
    expect_index(above, 0, extent, TextureWrapMode::clamp, 3);

    expect_index(below, 0, extent, TextureWrapMode::mirror, 0);
    expect_index(upper_edge, 0, extent, TextureWrapMode::mirror, 3);
    expect_index(above, 0, extent, TextureWrapMode::mirror, 2);

    expect_index(below, 0, extent, TextureWrapMode::black, std::nullopt);
    expect_index(lower_edge, 0, extent, TextureWrapMode::black, 0);
    expect_index(interior, 0, extent, TextureWrapMode::black, 3);
    expect_index(upper_edge, 0, extent, TextureWrapMode::black, std::nullopt);
    expect_index(above, 0, extent, TextureWrapMode::black, std::nullopt);
}

TEST(TextureWrapTest, AppliesModesIndependentlyToUAndVTapIndices) {
    constexpr auto extent = 4U;
    const auto u_tap = point_tap_from_uv(-0.25, extent);
    const auto v_tap = point_tap_from_uv(1.25, extent);
    expect_index(u_tap, 0, extent, TextureWrapMode::repeat, 3);
    expect_index(v_tap, 0, extent, TextureWrapMode::mirror, 2);
    expect_index(u_tap, 0, extent, TextureWrapMode::black, std::nullopt);
    expect_index(v_tap, 0, extent, TextureWrapMode::clamp, 3);
}

TEST(TextureWrapTest, RepeatsAndMirrorsIntegerTapsWithoutOverflow) {
    for (const auto& [input, expected] : std::array<std::array<std::int64_t, 2>, 9>{
             {{-9, 3}, {-4, 0}, {-1, 3}, {0, 0}, {3, 3}, {4, 0}, {7, 3}, {8, 0}, {9, 1}}}) {
        expect_index(input, 0, 4U, TextureWrapMode::repeat, expected);
    }
    expect_index(std::numeric_limits<std::int64_t>::min(), 0, 4U, TextureWrapMode::repeat, 0);
    expect_index(std::numeric_limits<std::int64_t>::max(), 0, 4U, TextureWrapMode::repeat, 3);

    for (const auto& [input, expected] : std::array<std::array<std::int64_t, 2>, 12>{{
             {-9, 0},
             {-5, 3},
             {-4, 3},
             {-3, 2},
             {-2, 1},
             {-1, 0},
             {0, 0},
             {3, 3},
             {4, 3},
             {5, 2},
             {7, 0},
             {8, 0},
         }}) {
        expect_index(input, 0, 4U, TextureWrapMode::mirror, expected);
    }
    expect_index(std::numeric_limits<std::int64_t>::min(), 0, 4U, TextureWrapMode::mirror, 0);
    expect_index(std::numeric_limits<std::int64_t>::max(), 0, 4U, TextureWrapMode::mirror, 0);

    expect_index(std::numeric_limits<std::int64_t>::min(), 10, 4U, TextureWrapMode::repeat, 12);
    expect_index(std::numeric_limits<std::int64_t>::max(), 10, 4U, TextureWrapMode::mirror, 12);
}

TEST(TextureWrapTest, AcceptsTheLargestRepresentableTexelIntervals) {
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
    expect_index(maximum, maximum - 3, 4U, TextureWrapMode::repeat, maximum);
    expect_index(minimum, maximum - 3, 4U, TextureWrapMode::repeat, maximum - 3);
    expect_index(maximum, maximum - 3, 4U, TextureWrapMode::mirror, maximum);
    expect_index(minimum, maximum - 3, 4U, TextureWrapMode::mirror, maximum);
    expect_index(minimum, minimum, std::numeric_limits<std::uint32_t>::max(),
                 TextureWrapMode::black, minimum);
    expect_index(maximum, maximum, 1U, TextureWrapMode::clamp, maximum);
}

TEST(TextureWrapTest, ClampsOrReturnsBlackForTexelTaps) {
    expect_index(9, 10, 4U, TextureWrapMode::clamp, 10);
    expect_index(10, 10, 4U, TextureWrapMode::clamp, 10);
    expect_index(13, 10, 4U, TextureWrapMode::clamp, 13);
    expect_index(14, 10, 4U, TextureWrapMode::clamp, 13);
    expect_index(std::numeric_limits<std::int64_t>::min(), 10, 4U, TextureWrapMode::black,
                 std::nullopt);
    expect_index(10, 10, 4U, TextureWrapMode::black, 10);
    expect_index(13, 10, 4U, TextureWrapMode::black, 13);
    expect_index(std::numeric_limits<std::int64_t>::max(), 10, 4U, TextureWrapMode::black,
                 std::nullopt);

    for (const auto mode :
         {TextureWrapMode::repeat, TextureWrapMode::clamp, TextureWrapMode::mirror}) {
        expect_index(std::numeric_limits<std::int64_t>::min(), -17, 1U, mode, -17);
        expect_index(std::numeric_limits<std::int64_t>::max(), -17, 1U, mode, -17);
    }
}

TEST(TextureWrapTest, RejectsInvalidIntervalsAndModes) {
    const auto zero_extent = wrap_texture_index(0, 0, 0U, TextureWrapMode::repeat);
    ASSERT_FALSE(zero_extent.has_value());
    EXPECT_EQ(zero_extent.error().code, core::StatusCode::invalid_argument);

    const auto overflowing_interval =
        wrap_texture_index(std::numeric_limits<std::int64_t>::max(),
                           std::numeric_limits<std::int64_t>::max(), 2U, TextureWrapMode::clamp);
    ASSERT_FALSE(overflowing_interval.has_value());
    EXPECT_EQ(overflowing_interval.error().code, core::StatusCode::invalid_argument);

    const auto invalid_mode =
        wrap_texture_index(0, 0, 4U, static_cast<TextureWrapMode>(0xFFFFFFFFU));
    ASSERT_FALSE(invalid_mode.has_value());
    EXPECT_EQ(invalid_mode.error().code, core::StatusCode::invalid_argument);
}

} // namespace
} // namespace blackframe::renderer
