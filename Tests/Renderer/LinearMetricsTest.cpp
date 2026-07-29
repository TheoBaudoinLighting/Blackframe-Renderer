#include <Blackframe/Renderer/LinearMetrics.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {
namespace {

TEST(LinearMetricsTest, ComputesAnalyticComponentValuesExactly) {
    static_assert(std::is_same_v<decltype(LinearMetrics::mse), ReferenceScalar>);
    static_assert(std::is_same_v<decltype(LinearMetrics::rmse), ReferenceScalar>);
    static_assert(std::is_same_v<decltype(LinearMetrics::mean_bias), ReferenceScalar>);
    static_assert(std::is_same_v<decltype(LinearMetrics::maximum_absolute_error), ReferenceScalar>);

    auto evaluated = Film::create(RenderExtent{.width = 1, .height = 1});
    auto reference = ReferenceFilm::create(RenderExtent{.width = 1, .height = 1});
    ASSERT_TRUE(evaluated.has_value());
    ASSERT_TRUE(reference.has_value());
    ASSERT_TRUE(
        evaluated->add_sample(0, 0, LinearRGB{.red = 2.0F, .green = 4.0F, .blue = 8.0F}, 1.0F)
            .has_value());
    ASSERT_TRUE(
        reference->add_sample(0, 0, ReferenceLinearRGB{.red = 1.0, .green = 2.0, .blue = 5.0}, 1.0)
            .has_value());

    const auto metrics = compute_linear_metrics(*evaluated, *reference);
    ASSERT_TRUE(metrics.has_value());
    EXPECT_DOUBLE_EQ(metrics->mse, 14.0 / 3.0);
    EXPECT_DOUBLE_EQ(metrics->rmse, std::sqrt(14.0 / 3.0));
    EXPECT_DOUBLE_EQ(metrics->mean_bias, 2.0);
    EXPECT_DOUBLE_EQ(metrics->maximum_absolute_error, 3.0);
}

TEST(LinearMetricsTest, UsesResolvedCropPixelsAndPreservesSignedBias) {
    const auto crop = FilmCrop{.minimum_x = 1, .minimum_y = 0, .maximum_x = 3, .maximum_y = 1};
    auto evaluated = Film::create(RenderExtent{.width = 4, .height = 2}, crop);
    auto reference = Film::create(RenderExtent{.width = 4, .height = 2}, crop);
    ASSERT_TRUE(evaluated.has_value());
    ASSERT_TRUE(reference.has_value());

    ASSERT_TRUE(
        evaluated->add_sample(1, 0, LinearRGB{.red = 1.0F, .green = 2.0F, .blue = 3.0F}, 2.0F)
            .has_value());
    ASSERT_TRUE(
        reference->add_sample(1, 0, LinearRGB{.red = 2.0F, .green = 3.0F, .blue = 4.0F}, 5.0F)
            .has_value());
    ASSERT_TRUE(
        evaluated->add_sample(2, 0, LinearRGB{.red = 5.0F, .green = 6.0F, .blue = 7.0F}, 4.0F)
            .has_value());
    ASSERT_TRUE(
        reference->add_sample(2, 0, LinearRGB{.red = 6.0F, .green = 7.0F, .blue = 8.0F}, 3.0F)
            .has_value());

    const auto metrics = compute_linear_metrics(*evaluated, *reference);
    ASSERT_TRUE(metrics.has_value());
    EXPECT_DOUBLE_EQ(metrics->mse, 1.0);
    EXPECT_DOUBLE_EQ(metrics->rmse, 1.0);
    EXPECT_DOUBLE_EQ(metrics->mean_bias, -1.0);
    EXPECT_DOUBLE_EQ(metrics->maximum_absolute_error, 1.0);
}

TEST(LinearMetricsTest, ReturnsExactZeroForIdenticalReferenceFilms) {
    auto first = ReferenceFilm::create(RenderExtent{.width = 1, .height = 1});
    auto second = ReferenceFilm::create(RenderExtent{.width = 1, .height = 1});
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    constexpr auto sample = ReferenceLinearRGB{.red = -2.0, .green = 0.25, .blue = 16.0};
    ASSERT_TRUE(first->add_sample(0, 0, sample, 1.0).has_value());
    ASSERT_TRUE(second->add_sample(0, 0, sample, 1.0).has_value());

    const auto metrics = compute_linear_metrics(*first, *second);
    ASSERT_TRUE(metrics.has_value());
    EXPECT_EQ(*metrics, LinearMetrics{});
}

TEST(LinearMetricsTest, RejectsMismatchedOrUnresolvedInputsWithoutFallback) {
    auto evaluated = Film::create(RenderExtent{.width = 2, .height = 1});
    auto different_extent = Film::create(RenderExtent{.width = 1, .height = 1});
    auto different_crop = Film::create(RenderExtent{.width = 2, .height = 1},
                                       FilmCrop{.minimum_x = 1, .maximum_x = 2, .maximum_y = 1});
    ASSERT_TRUE(evaluated.has_value());
    ASSERT_TRUE(different_extent.has_value());
    ASSERT_TRUE(different_crop.has_value());

    const auto extent_mismatch = compute_linear_metrics(*evaluated, *different_extent);
    ASSERT_FALSE(extent_mismatch.has_value());
    EXPECT_EQ(extent_mismatch.error().code, core::StatusCode::invalid_argument);

    const auto crop_mismatch = compute_linear_metrics(*evaluated, *different_crop);
    ASSERT_FALSE(crop_mismatch.has_value());
    EXPECT_EQ(crop_mismatch.error().code, core::StatusCode::invalid_argument);

    ASSERT_TRUE(evaluated->add_sample(0, 0, LinearRGB{}, 1.0F).has_value());
    ASSERT_TRUE(evaluated->add_sample(1, 0, LinearRGB{}, 1.0F).has_value());
    auto unresolved_reference = Film::create(RenderExtent{.width = 2, .height = 1});
    ASSERT_TRUE(unresolved_reference.has_value());
    const auto unresolved = compute_linear_metrics(*evaluated, *unresolved_reference);
    ASSERT_FALSE(unresolved.has_value());
    EXPECT_EQ(unresolved.error().code, core::StatusCode::invalid_argument);
}

TEST(LinearMetricsTest, RejectsDoubleOverflowWithoutReplacementMetrics) {
    auto evaluated = ReferenceFilm::create(RenderExtent{.width = 1, .height = 1});
    auto reference = ReferenceFilm::create(RenderExtent{.width = 1, .height = 1});
    ASSERT_TRUE(evaluated.has_value());
    ASSERT_TRUE(reference.has_value());
    const auto maximum = std::numeric_limits<ReferenceScalar>::max();
    ASSERT_TRUE(evaluated->add_sample(0, 0, ReferenceLinearRGB{.red = maximum}, 1.0).has_value());
    ASSERT_TRUE(reference->add_sample(0, 0, ReferenceLinearRGB{.red = -maximum}, 1.0).has_value());

    const auto metrics = compute_linear_metrics(*evaluated, *reference);
    ASSERT_FALSE(metrics.has_value());
    EXPECT_EQ(metrics.error().code, core::StatusCode::invalid_argument);
}

} // namespace
} // namespace blackframe::renderer
