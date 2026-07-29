#include <Blackframe/Renderer/DisplayPsnr.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {
namespace {

TEST(DisplayPsnrTest, ComputesAnalyticDisplayValueAndDifferenceHeatmap) {
    static_assert(std::is_same_v<decltype(DisplayPsnrResult::psnr), ReferenceScalar>);
    static_assert(
        std::is_same_v<typename decltype(DisplayPsnrResult::difference_heatmap)::value_type,
                       ReferenceScalar>);
    static_assert(FixedDisplayPeak == ReferenceScalar{1});

    const auto crop = FilmCrop{.minimum_x = 1, .minimum_y = 0, .maximum_x = 3, .maximum_y = 1};
    auto evaluated = ReferenceFilm::create(RenderExtent{.width = 4, .height = 2}, crop);
    auto reference = ReferenceFilm::create(RenderExtent{.width = 4, .height = 2}, crop);
    ASSERT_TRUE(evaluated.has_value());
    ASSERT_TRUE(reference.has_value());
    ASSERT_TRUE(evaluated->add_sample(1, 0, ReferenceLinearRGB{}, 1.0).has_value());
    ASSERT_TRUE(reference->add_sample(1, 0, ReferenceLinearRGB{}, 1.0).has_value());
    ASSERT_TRUE(evaluated->add_sample(2, 0, ReferenceLinearRGB{}, 1.0).has_value());
    ASSERT_TRUE(
        reference
            ->add_sample(2, 0, ReferenceLinearRGB{.red = 0.002, .green = 0.002, .blue = 0.002}, 1.0)
            .has_value());

    const auto result = compute_display_psnr(*evaluated, *reference);
    ASSERT_TRUE(result.has_value());
    constexpr auto display_difference = ReferenceScalar{12.92} * ReferenceScalar{0.002};
    EXPECT_DOUBLE_EQ(result->psnr, -ReferenceScalar{20} * std::log10(display_difference) +
                                       ReferenceScalar{10} * std::log10(ReferenceScalar{2}));
    EXPECT_EQ(result->crop, crop);
    ASSERT_EQ(result->difference_heatmap.size(), 2U);
    EXPECT_DOUBLE_EQ(result->difference_heatmap[0], 0.0);
    EXPECT_DOUBLE_EQ(result->difference_heatmap[1], display_difference);
}

TEST(DisplayPsnrTest, PreservesRepresentableDifferencesWhoseSquaresUnderflow) {
    auto evaluated = ReferenceFilm::create(RenderExtent{.width = 1, .height = 1});
    auto reference = ReferenceFilm::create(RenderExtent{.width = 1, .height = 1});
    ASSERT_TRUE(evaluated.has_value());
    ASSERT_TRUE(reference.has_value());
    const auto minimum_normal = std::numeric_limits<ReferenceScalar>::min();
    ASSERT_TRUE(evaluated->add_sample(0, 0, ReferenceLinearRGB{}, 1.0).has_value());
    ASSERT_TRUE(
        reference->add_sample(0, 0, ReferenceLinearRGB{.red = minimum_normal}, 1.0).has_value());

    const auto result = compute_display_psnr(*evaluated, *reference);
    ASSERT_TRUE(result.has_value());
    const auto display_difference = ReferenceScalar{12.92} * minimum_normal;
    EXPECT_TRUE(std::isfinite(result->psnr));
    EXPECT_DOUBLE_EQ(result->psnr, -ReferenceScalar{20} * std::log10(display_difference) +
                                       ReferenceScalar{10} * std::log10(ReferenceScalar{3}));
    ASSERT_EQ(result->difference_heatmap.size(), 1U);
    EXPECT_DOUBLE_EQ(result->difference_heatmap[0], display_difference);
}

TEST(DisplayPsnrTest, UsesFixedUnitPeakAfterHdrClipping) {
    auto evaluated = Film::create(RenderExtent{.width = 1, .height = 1});
    auto reference = ReferenceFilm::create(RenderExtent{.width = 1, .height = 1});
    ASSERT_TRUE(evaluated.has_value());
    ASSERT_TRUE(reference.has_value());
    ASSERT_TRUE(evaluated->add_sample(0, 0, LinearRGB{}, 1.0F).has_value());
    ASSERT_TRUE(
        reference
            ->add_sample(0, 0, ReferenceLinearRGB{.red = 10.0, .green = 10.0, .blue = 10.0}, 1.0)
            .has_value());

    const auto result = compute_display_psnr(*evaluated, *reference);
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(result->psnr, 0.0);
    ASSERT_EQ(result->difference_heatmap.size(), 1U);
    EXPECT_DOUBLE_EQ(result->difference_heatmap[0], 1.0);
}

TEST(DisplayPsnrTest, ReturnsPositiveInfinityForExactDisplayMatch) {
    auto evaluated = Film::create(RenderExtent{.width = 1, .height = 1});
    auto reference = Film::create(RenderExtent{.width = 1, .height = 1});
    ASSERT_TRUE(evaluated.has_value());
    ASSERT_TRUE(reference.has_value());
    constexpr auto color = LinearRGB{.red = 0.18F, .green = 0.5F, .blue = 2.0F};
    ASSERT_TRUE(evaluated->add_sample(0, 0, color, 1.0F).has_value());
    ASSERT_TRUE(reference->add_sample(0, 0, color, 1.0F).has_value());

    const auto result = compute_display_psnr(*evaluated, *reference);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(std::isinf(result->psnr));
    EXPECT_GT(result->psnr, 0.0);
    ASSERT_EQ(result->difference_heatmap.size(), 1U);
    EXPECT_DOUBLE_EQ(result->difference_heatmap[0], 0.0);
}

TEST(DisplayPsnrTest, RejectsMismatchedAndUnresolvedFilmsWithoutFallback) {
    auto evaluated = Film::create(RenderExtent{.width = 2, .height = 1});
    auto different_extent = Film::create(RenderExtent{.width = 1, .height = 1});
    auto different_crop = Film::create(RenderExtent{.width = 2, .height = 1},
                                       FilmCrop{.minimum_x = 1, .maximum_x = 2, .maximum_y = 1});
    ASSERT_TRUE(evaluated.has_value());
    ASSERT_TRUE(different_extent.has_value());
    ASSERT_TRUE(different_crop.has_value());

    const auto extent_mismatch = compute_display_psnr(*evaluated, *different_extent);
    ASSERT_FALSE(extent_mismatch.has_value());
    EXPECT_EQ(extent_mismatch.error().code, core::StatusCode::invalid_argument);

    const auto crop_mismatch = compute_display_psnr(*evaluated, *different_crop);
    ASSERT_FALSE(crop_mismatch.has_value());
    EXPECT_EQ(crop_mismatch.error().code, core::StatusCode::invalid_argument);

    ASSERT_TRUE(evaluated->add_sample(0, 0, LinearRGB{}, 1.0F).has_value());
    ASSERT_TRUE(evaluated->add_sample(1, 0, LinearRGB{}, 1.0F).has_value());
    auto unresolved_reference = Film::create(RenderExtent{.width = 2, .height = 1});
    ASSERT_TRUE(unresolved_reference.has_value());
    const auto unresolved = compute_display_psnr(*evaluated, *unresolved_reference);
    ASSERT_FALSE(unresolved.has_value());
    EXPECT_EQ(unresolved.error().code, core::StatusCode::invalid_argument);
}

} // namespace
} // namespace blackframe::renderer
