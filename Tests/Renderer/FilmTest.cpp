#include <Blackframe/Renderer/Film.hpp>
#include <concepts>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {
namespace {

TEST(FilmTest, StoresExactWeightedSumsWeightsAndSampleCountsPerPixel) {
    auto film = Film::create(RenderExtent{.width = 2, .height = 1});
    ASSERT_TRUE(film.has_value());

    EXPECT_TRUE(film->add_sample(0, 0, LinearRGB{.red = 1.0F, .green = 2.0F, .blue = 3.0F}, 0.5F)
                    .has_value());
    EXPECT_TRUE(film->add_sample(0, 0, LinearRGB{.red = 3.0F, .green = 4.0F, .blue = 5.0F}, 0.5F)
                    .has_value());
    EXPECT_TRUE(film->add_sample(1, 0, LinearRGB{.red = 8.0F, .green = 4.0F, .blue = 2.0F}, 0.25F)
                    .has_value());

    const auto first = film->pixel(0, 0);
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->weighted_sum, (LinearRGB{.red = 2.0F, .green = 3.0F, .blue = 4.0F}));
    EXPECT_FLOAT_EQ(first->weight_sum, 1.0F);
    EXPECT_EQ(first->sample_count, 2U);

    const auto second = film->pixel(1, 0);
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(second->weighted_sum, (LinearRGB{.red = 2.0F, .green = 1.0F, .blue = 0.5F}));
    EXPECT_FLOAT_EQ(second->weight_sum, 0.25F);
    EXPECT_EQ(second->sample_count, 1U);

    const auto resolved = film->resolved_pixel(0, 0);
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(*resolved, (LinearRGB{.red = 2.0F, .green = 3.0F, .blue = 4.0F}));
}

TEST(FilmTest, SupportsExplicitDoublePrecisionAccumulation) {
    static_assert(std::same_as<Film::Scalar, TransportScalar>);
    static_assert(std::same_as<ReferenceFilm::Scalar, ReferenceScalar>);
    static_assert(std::same_as<ReferenceFilm, DoubleAccumulationFilm>);
    static_assert(!std::same_as<Film, ReferenceFilm>);

    auto film = ReferenceFilm::create(RenderExtent{.width = 1, .height = 1});
    ASSERT_TRUE(film.has_value());
    EXPECT_TRUE(
        film->add_sample(0, 0, ReferenceLinearRGB{.red = 0.5, .green = 1.0, .blue = 1.5}, 2.0)
            .has_value());

    const auto pixel = film->pixel(0, 0);
    ASSERT_TRUE(pixel.has_value());
    EXPECT_EQ(pixel->weighted_sum, (ReferenceLinearRGB{.red = 1.0, .green = 2.0, .blue = 3.0}));
    EXPECT_DOUBLE_EQ(pixel->weight_sum, 2.0);
    EXPECT_EQ(pixel->sample_count, 1U);
}

TEST(FilmTest, ReferenceModeRecoversALongIllConditionedSum) {
    constexpr auto small_sample_count = std::uint64_t{65'536};
    constexpr auto cancellation_weight = ReferenceScalar{1.0E16};
    constexpr auto sample = ReferenceLinearRGB{.red = 1.0, .green = 2.0, .blue = -1.0};
    auto film = ReferenceFilm::create(RenderExtent{.width = 1, .height = 1});
    ASSERT_TRUE(film.has_value());

    ASSERT_TRUE(film->add_sample(0, 0, sample, cancellation_weight).has_value());
    for (std::uint64_t index = 0; index < small_sample_count; ++index) {
        ASSERT_TRUE(film->add_sample(0, 0, sample, 1.0).has_value());
    }
    ASSERT_TRUE(film->add_sample(0, 0, sample, -cancellation_weight).has_value());

    const auto pixel = film->pixel(0, 0);
    ASSERT_TRUE(pixel.has_value());
    EXPECT_DOUBLE_EQ(pixel->weighted_sum.red, static_cast<ReferenceScalar>(small_sample_count));
    EXPECT_DOUBLE_EQ(pixel->weighted_sum.green,
                     static_cast<ReferenceScalar>(2 * small_sample_count));
    EXPECT_DOUBLE_EQ(pixel->weighted_sum.blue, -static_cast<ReferenceScalar>(small_sample_count));
    EXPECT_DOUBLE_EQ(pixel->weight_sum, static_cast<ReferenceScalar>(small_sample_count));
    EXPECT_EQ(pixel->sample_count, small_sample_count + 2);

    const auto resolved = film->resolved_pixel(0, 0);
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(*resolved, sample);
}

TEST(FilmTest, InitializesEveryPixelIndependently) {
    const auto film = Film::create(RenderExtent{.width = 3, .height = 2});
    ASSERT_TRUE(film.has_value());
    EXPECT_EQ(film->extent().width, 3U);
    EXPECT_EQ(film->extent().height, 2U);
    EXPECT_EQ(film->pixel_count(), 6U);

    for (std::uint32_t y = 0; y < film->extent().height; ++y) {
        for (std::uint32_t x = 0; x < film->extent().width; ++x) {
            const auto pixel = film->pixel(x, y);
            ASSERT_TRUE(pixel.has_value());
            EXPECT_EQ(pixel->weighted_sum, LinearRGB{});
            EXPECT_FLOAT_EQ(pixel->weight_sum, 0.0F);
            EXPECT_EQ(pixel->sample_count, 0U);
        }
    }
    EXPECT_FALSE(film->resolved_pixel(0, 0).has_value());
}

TEST(FilmTest, RejectsInvalidExtentsAndCoordinatesBeforeAccess) {
    const auto zero_width = Film::create(RenderExtent{.width = 0, .height = 1});
    ASSERT_FALSE(zero_width.has_value());
    EXPECT_EQ(zero_width.error().code, core::StatusCode::invalid_argument);

    const auto excessive_dimension =
        Film::create(RenderExtent{.width = (1U << 20U) + 1U, .height = 1});
    ASSERT_FALSE(excessive_dimension.has_value());
    EXPECT_EQ(excessive_dimension.error().code, core::StatusCode::resource_exhausted);

    auto film = Film::create(RenderExtent{.width = 2, .height = 2});
    ASSERT_TRUE(film.has_value());
    EXPECT_FALSE(film->pixel(2, 0).has_value());
    EXPECT_FALSE(film->pixel(0, 2).has_value());
    EXPECT_FALSE(film->add_sample(2, 0, LinearRGB{}, 1.0F).has_value());
}

TEST(FilmTest, RejectsInvalidAccumulationWithoutMutatingThePixel) {
    auto film = Film::create(RenderExtent{.width = 1, .height = 1});
    ASSERT_TRUE(film.has_value());
    ASSERT_TRUE(film->add_sample(0, 0, LinearRGB{.red = 1.0F, .green = 2.0F, .blue = 3.0F}, 1.0F)
                    .has_value());
    const auto before = film->pixel(0, 0);
    ASSERT_TRUE(before.has_value());

    const auto infinity = std::numeric_limits<TransportScalar>::infinity();
    EXPECT_FALSE(film->add_sample(0, 0, LinearRGB{.red = infinity}, 1.0F).has_value());
    EXPECT_FALSE(film->add_sample(0, 0, LinearRGB{}, infinity).has_value());
    EXPECT_FALSE(film->add_sample(0, 0,
                                  LinearRGB{.red = std::numeric_limits<TransportScalar>::max(),
                                            .green = 0.0F,
                                            .blue = 0.0F},
                                  2.0F)
                     .has_value());

    const auto after = film->pixel(0, 0);
    ASSERT_TRUE(after.has_value());
    EXPECT_EQ(*after, *before);
}

} // namespace
} // namespace blackframe::renderer
