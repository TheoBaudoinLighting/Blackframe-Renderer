#include <Blackframe/Renderer/ConvergenceMetrics.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <type_traits>

namespace blackframe::renderer {
namespace {

using namespace std::chrono_literals;

[[nodiscard]] QualityCheckpoint checkpoint(const std::uint64_t samples_per_pixel,
                                           const std::chrono::nanoseconds render_time,
                                           const ReferenceScalar mse, const ReferenceScalar psnr) {
    return QualityCheckpoint{
        .cumulative_samples_per_pixel = samples_per_pixel,
        .cumulative_render_time = render_time,
        .linear_mse = mse,
        .display_psnr = psnr,
    };
}

void expect_invalid(const std::span<const QualityCheckpoint> checkpoints,
                    const QualityThresholds thresholds = {
                        .maximum_linear_mse = 0.1,
                        .minimum_display_psnr = 25.0,
                    }) {
    const auto result = compute_time_to_quality(checkpoints, thresholds);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(result.error().message.empty());
}

TEST(ConvergenceMetricsTest, FindsInclusiveFirstObservedCrossingsWithoutInterpolation) {
    static_assert(std::is_same_v<decltype(QualityThresholds::maximum_linear_mse), ReferenceScalar>);
    static_assert(
        std::is_same_v<decltype(QualityThresholds::minimum_display_psnr), ReferenceScalar>);
    static_assert(
        std::is_same_v<decltype(QualityCheckpoint::cumulative_samples_per_pixel), std::uint64_t>);
    static_assert(std::is_same_v<decltype(QualityCheckpoint::cumulative_render_time),
                                 std::chrono::nanoseconds>);

    const auto checkpoints = std::array{
        checkpoint(1U, 10ns, 0.25, 12.0),
        checkpoint(4U, 40ns, 0.0625, 20.0),
        checkpoint(16U, 130ns, 0.015625, 25.0),
    };

    const auto result = compute_time_to_quality(checkpoints, QualityThresholds{
                                                                 .maximum_linear_mse = 0.0625,
                                                                 .minimum_display_psnr = 25.0,
                                                             });

    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_TRUE(result->linear_mse.has_value());
    EXPECT_EQ(*result->linear_mse, (QualityThresholdHit{
                                       .cumulative_samples_per_pixel = 4U,
                                       .cumulative_render_time = 40ns,
                                       .observed_value = 0.0625,
                                   }));
    ASSERT_TRUE(result->display_psnr.has_value());
    EXPECT_EQ(*result->display_psnr, (QualityThresholdHit{
                                         .cumulative_samples_per_pixel = 16U,
                                         .cumulative_render_time = 130ns,
                                         .observed_value = 25.0,
                                     }));
}

TEST(ConvergenceMetricsTest, LeavesUnreachedTargetsExplicitlyEmpty) {
    const auto checkpoints = std::array{
        checkpoint(1U, 10ns, 0.4, 10.0),
        checkpoint(8U, 90ns, 0.2, 20.0),
    };

    const auto result = compute_time_to_quality(checkpoints, QualityThresholds{
                                                                 .maximum_linear_mse = 0.1,
                                                                 .minimum_display_psnr = 25.0,
                                                             });

    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_FALSE(result->linear_mse.has_value());
    EXPECT_FALSE(result->display_psnr.has_value());
}

TEST(ConvergenceMetricsTest, AllowsQualityRegressionAndKeepsTheFirstCrossing) {
    const auto checkpoints = std::array{
        checkpoint(1U, 10ns, 0.09, 26.0),
        checkpoint(2U, 20ns, 0.12, 24.0),
        checkpoint(4U, 50ns, 0.08, 27.0),
    };

    const auto result = compute_time_to_quality(checkpoints, QualityThresholds{
                                                                 .maximum_linear_mse = 0.1,
                                                                 .minimum_display_psnr = 25.0,
                                                             });

    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_TRUE(result->linear_mse.has_value());
    ASSERT_TRUE(result->display_psnr.has_value());
    EXPECT_EQ(result->linear_mse->cumulative_samples_per_pixel, 1U);
    EXPECT_EQ(result->display_psnr->cumulative_samples_per_pixel, 1U);
}

TEST(ConvergenceMetricsTest, AcceptsExactMatchPsnrAndEqualCumulativeTimestamps) {
    const auto infinity = std::numeric_limits<ReferenceScalar>::infinity();
    const auto checkpoints = std::array{
        checkpoint(1U, 5ns, 0.1, 20.0),
        checkpoint(2U, 5ns, 0.0, infinity),
    };

    const auto result = compute_time_to_quality(checkpoints, QualityThresholds{
                                                                 .maximum_linear_mse = 0.0,
                                                                 .minimum_display_psnr = 100.0,
                                                             });

    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_TRUE(result->linear_mse.has_value());
    ASSERT_TRUE(result->display_psnr.has_value());
    EXPECT_EQ(result->linear_mse->cumulative_render_time, 5ns);
    EXPECT_EQ(result->display_psnr->observed_value, infinity);
}

TEST(ConvergenceMetricsTest, RejectsInvalidThresholds) {
    const auto checkpoints = std::array{checkpoint(1U, 1ns, 0.1, 20.0)};
    const auto infinity = std::numeric_limits<ReferenceScalar>::infinity();
    const auto nan = std::numeric_limits<ReferenceScalar>::quiet_NaN();
    const auto thresholds = std::array{
        QualityThresholds{.maximum_linear_mse = -0.1, .minimum_display_psnr = 20.0},
        QualityThresholds{.maximum_linear_mse = infinity, .minimum_display_psnr = 20.0},
        QualityThresholds{.maximum_linear_mse = nan, .minimum_display_psnr = 20.0},
        QualityThresholds{.maximum_linear_mse = 0.1, .minimum_display_psnr = infinity},
        QualityThresholds{.maximum_linear_mse = 0.1, .minimum_display_psnr = -infinity},
        QualityThresholds{.maximum_linear_mse = 0.1, .minimum_display_psnr = -1.0},
        QualityThresholds{.maximum_linear_mse = 0.1, .minimum_display_psnr = nan},
    };

    for (const auto invalid_thresholds : thresholds) {
        expect_invalid(checkpoints, invalid_thresholds);
    }
}

TEST(ConvergenceMetricsTest, RejectsEmptyAndNonIncreasingSampleSchedules) {
    expect_invalid({});

    const auto zero = std::array{checkpoint(0U, 0ns, 0.1, 20.0)};
    expect_invalid(zero);

    const auto duplicate = std::array{
        checkpoint(1U, 1ns, 0.1, 20.0),
        checkpoint(1U, 2ns, 0.05, 25.0),
    };
    expect_invalid(duplicate);

    const auto decreasing = std::array{
        checkpoint(2U, 1ns, 0.1, 20.0),
        checkpoint(1U, 2ns, 0.05, 25.0),
    };
    expect_invalid(decreasing);
}

TEST(ConvergenceMetricsTest, RejectsNegativeOrDecreasingCumulativeTime) {
    const auto negative = std::array{checkpoint(1U, -1ns, 0.1, 20.0)};
    expect_invalid(negative);

    const auto decreasing = std::array{
        checkpoint(1U, 2ns, 0.1, 20.0),
        checkpoint(2U, 1ns, 0.05, 25.0),
    };
    expect_invalid(decreasing);
}

TEST(ConvergenceMetricsTest, RejectsInvalidMeasuredMetrics) {
    const auto infinity = std::numeric_limits<ReferenceScalar>::infinity();
    const auto nan = std::numeric_limits<ReferenceScalar>::quiet_NaN();
    const auto invalid_checkpoints = std::array{
        checkpoint(1U, 1ns, -0.1, 20.0), checkpoint(1U, 1ns, infinity, 20.0),
        checkpoint(1U, 1ns, nan, 20.0),  checkpoint(1U, 1ns, 0.1, -infinity),
        checkpoint(1U, 1ns, 0.1, -1.0),  checkpoint(1U, 1ns, 0.1, nan),
    };

    for (const auto invalid_checkpoint : invalid_checkpoints) {
        const auto timeline = std::array{invalid_checkpoint};
        expect_invalid(timeline);
    }
}

TEST(ConvergenceMetricsTest, ValidatesTrailingDataAfterBothTargetsWereReached) {
    const auto checkpoints = std::array{
        checkpoint(1U, 1ns, 0.01, 40.0),
        checkpoint(1U, 2ns, 0.005, 45.0),
    };

    expect_invalid(checkpoints, QualityThresholds{
                                    .maximum_linear_mse = 0.1,
                                    .minimum_display_psnr = 25.0,
                                });
}

TEST(ConvergenceMetricsTest, ReplaysBitIdenticalResults) {
    const auto checkpoints = std::array{
        checkpoint(1U, 10ns, 0.2, 15.0),
        checkpoint(4U, 40ns, 0.05, 30.0),
    };
    constexpr auto thresholds = QualityThresholds{
        .maximum_linear_mse = 0.1,
        .minimum_display_psnr = 25.0,
    };

    const auto first = compute_time_to_quality(checkpoints, thresholds);
    const auto replay = compute_time_to_quality(checkpoints, thresholds);

    ASSERT_TRUE(first.has_value()) << first.error().message;
    ASSERT_TRUE(replay.has_value()) << replay.error().message;
    EXPECT_EQ(*first, *replay);
}

} // namespace
} // namespace blackframe::renderer
