#include <Blackframe/Renderer/HdrMetrics.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {
namespace {

TEST(HdrMetricsTest, ComputesSyntheticRelativeMseAndLuminanceSmape) {
    static_assert(std::is_same_v<decltype(HdrMetrics::relative_mse), ReferenceScalar>);
    static_assert(std::is_same_v<decltype(HdrMetrics::luminance_smape), ReferenceScalar>);

    auto evaluated = Film::create(RenderExtent{.width = 2, .height = 1});
    auto reference = ReferenceFilm::create(RenderExtent{.width = 2, .height = 1});
    ASSERT_TRUE(evaluated.has_value());
    ASSERT_TRUE(reference.has_value());
    ASSERT_TRUE(evaluated->add_sample(0, 0, LinearRGB{.green = 2.0F}, 1.0F).has_value());
    ASSERT_TRUE(reference->add_sample(0, 0, ReferenceLinearRGB{.green = 1.0}, 1.0).has_value());
    ASSERT_TRUE(evaluated->add_sample(1, 0, LinearRGB{}, 1.0F).has_value());
    ASSERT_TRUE(reference->add_sample(1, 0, ReferenceLinearRGB{}, 1.0).has_value());

    const auto metrics = compute_hdr_metrics(*evaluated, *reference);
    ASSERT_TRUE(metrics.has_value());
    EXPECT_DOUBLE_EQ(metrics->relative_mse, 1.0);
    EXPECT_DOUBLE_EQ(metrics->luminance_smape, 1.0 / 3.0);
}

TEST(HdrMetricsTest, IsInvariantUnderSharedPositivePowerOfTwoScale) {
    auto evaluated = ReferenceFilm::create(RenderExtent{.width = 1, .height = 1});
    auto reference = ReferenceFilm::create(RenderExtent{.width = 1, .height = 1});
    auto scaled_evaluated = ReferenceFilm::create(RenderExtent{.width = 1, .height = 1});
    auto scaled_reference = ReferenceFilm::create(RenderExtent{.width = 1, .height = 1});
    ASSERT_TRUE(evaluated.has_value());
    ASSERT_TRUE(reference.has_value());
    ASSERT_TRUE(scaled_evaluated.has_value());
    ASSERT_TRUE(scaled_reference.has_value());

    constexpr auto evaluated_color = ReferenceLinearRGB{.red = 2.0, .green = 1.0, .blue = 8.0};
    constexpr auto reference_color = ReferenceLinearRGB{.red = 1.0, .green = 2.0, .blue = 4.0};
    constexpr auto scale = ReferenceScalar{8};
    ASSERT_TRUE(evaluated->add_sample(0, 0, evaluated_color, 1.0).has_value());
    ASSERT_TRUE(reference->add_sample(0, 0, reference_color, 1.0).has_value());
    ASSERT_TRUE(scaled_evaluated
                    ->add_sample(0, 0,
                                 ReferenceLinearRGB{
                                     .red = scale * evaluated_color.red,
                                     .green = scale * evaluated_color.green,
                                     .blue = scale * evaluated_color.blue,
                                 },
                                 1.0)
                    .has_value());
    ASSERT_TRUE(scaled_reference
                    ->add_sample(0, 0,
                                 ReferenceLinearRGB{
                                     .red = scale * reference_color.red,
                                     .green = scale * reference_color.green,
                                     .blue = scale * reference_color.blue,
                                 },
                                 1.0)
                    .has_value());

    const auto baseline = compute_hdr_metrics(*evaluated, *reference);
    const auto scaled = compute_hdr_metrics(*scaled_evaluated, *scaled_reference);
    ASSERT_TRUE(baseline.has_value());
    ASSERT_TRUE(scaled.has_value());
    EXPECT_EQ(*scaled, *baseline);
}

TEST(HdrMetricsTest, HandlesFiniteHdrEnergyWithoutSquaringOverflow) {
    auto evaluated = ReferenceFilm::create(RenderExtent{.width = 1, .height = 1});
    auto reference = ReferenceFilm::create(RenderExtent{.width = 1, .height = 1});
    ASSERT_TRUE(evaluated.has_value());
    ASSERT_TRUE(reference.has_value());
    const auto maximum = std::numeric_limits<ReferenceScalar>::max();
    const auto color = ReferenceLinearRGB{.red = maximum};
    ASSERT_TRUE(evaluated->add_sample(0, 0, color, 1.0).has_value());
    ASSERT_TRUE(reference->add_sample(0, 0, color, 1.0).has_value());

    const auto metrics = compute_hdr_metrics(*evaluated, *reference);
    ASSERT_TRUE(metrics.has_value());
    EXPECT_EQ(*metrics, HdrMetrics{});
}

TEST(HdrMetricsTest, RejectsZeroEnergyReferenceWithoutRelativeMseFallback) {
    auto evaluated = Film::create(RenderExtent{.width = 1, .height = 1});
    auto reference = Film::create(RenderExtent{.width = 1, .height = 1});
    ASSERT_TRUE(evaluated.has_value());
    ASSERT_TRUE(reference.has_value());
    ASSERT_TRUE(evaluated->add_sample(0, 0, LinearRGB{.red = 1.0F}, 1.0F).has_value());
    ASSERT_TRUE(reference->add_sample(0, 0, LinearRGB{}, 1.0F).has_value());

    const auto metrics = compute_hdr_metrics(*evaluated, *reference);
    ASSERT_FALSE(metrics.has_value());
    EXPECT_EQ(metrics.error().code, core::StatusCode::invalid_argument);
}

TEST(HdrMetricsTest, RejectsMismatchedAndUnresolvedFilmsWithoutFallback) {
    auto evaluated = Film::create(RenderExtent{.width = 2, .height = 1});
    auto different_extent = Film::create(RenderExtent{.width = 1, .height = 1});
    auto different_crop = Film::create(RenderExtent{.width = 2, .height = 1},
                                       FilmCrop{.minimum_x = 1, .maximum_x = 2, .maximum_y = 1});
    ASSERT_TRUE(evaluated.has_value());
    ASSERT_TRUE(different_extent.has_value());
    ASSERT_TRUE(different_crop.has_value());

    const auto extent_mismatch = compute_hdr_metrics(*evaluated, *different_extent);
    ASSERT_FALSE(extent_mismatch.has_value());
    EXPECT_EQ(extent_mismatch.error().code, core::StatusCode::invalid_argument);

    const auto crop_mismatch = compute_hdr_metrics(*evaluated, *different_crop);
    ASSERT_FALSE(crop_mismatch.has_value());
    EXPECT_EQ(crop_mismatch.error().code, core::StatusCode::invalid_argument);

    ASSERT_TRUE(evaluated->add_sample(0, 0, LinearRGB{}, 1.0F).has_value());
    ASSERT_TRUE(evaluated->add_sample(1, 0, LinearRGB{}, 1.0F).has_value());
    auto unresolved_reference = Film::create(RenderExtent{.width = 2, .height = 1});
    ASSERT_TRUE(unresolved_reference.has_value());
    const auto unresolved = compute_hdr_metrics(*evaluated, *unresolved_reference);
    ASSERT_FALSE(unresolved.has_value());
    EXPECT_EQ(unresolved.error().code, core::StatusCode::invalid_argument);
}

} // namespace
} // namespace blackframe::renderer
