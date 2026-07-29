#include <Blackframe/Renderer/Bounds.hpp>
#include <Blackframe/Renderer/Interval.hpp>
#include <cmath>
#include <concepts>
#include <gtest/gtest.h>
#include <limits>

namespace blackframe::renderer {
namespace {

TEST(IntervalTest, RepresentsClosedEmptyAndInfiniteRanges) {
    const auto finite = Interval::closed(-2.0F, 3.0F);
    ASSERT_TRUE(finite.has_value());
    EXPECT_TRUE(finite->contains(-2.0F));
    EXPECT_TRUE(finite->contains(3.0F));
    EXPECT_FALSE(finite->contains(std::nextafter(3.0F, 4.0F)));

    constexpr auto unbounded = Interval::unbounded();
    EXPECT_TRUE(unbounded.contains(-std::numeric_limits<TransportScalar>::infinity()));
    EXPECT_TRUE(unbounded.contains(std::numeric_limits<TransportScalar>::infinity()));

    constexpr auto empty = Interval::empty();
    EXPECT_TRUE(empty.is_empty());
    EXPECT_FALSE(empty.contains(0.0F));
    EXPECT_FALSE(empty.overlaps(unbounded));
}

TEST(IntervalTest, IntersectsAtInclusiveLimits) {
    const auto left = Interval::closed(-1.0F, 2.0F);
    const auto right = Interval::closed(2.0F, 4.0F);
    ASSERT_TRUE(left.has_value());
    ASSERT_TRUE(right.has_value());

    const auto touching = left->intersection_with(*right);
    EXPECT_FALSE(touching.is_empty());
    EXPECT_FLOAT_EQ(touching.lower(), 2.0F);
    EXPECT_FLOAT_EQ(touching.upper(), 2.0F);

    const auto disjoint = Interval::closed(3.0F, 4.0F);
    ASSERT_TRUE(disjoint.has_value());
    EXPECT_TRUE(left->intersection_with(*disjoint).is_empty());
}

TEST(IntervalTest, RejectsReversedAndNanEndpoints) {
    const auto reversed = Interval::closed(2.0F, -1.0F);
    ASSERT_FALSE(reversed.has_value());
    EXPECT_EQ(reversed.error().code, core::StatusCode::invalid_argument);

    const auto with_nan = Interval::closed(0.0F, std::numeric_limits<TransportScalar>::quiet_NaN());
    ASSERT_FALSE(with_nan.has_value());
    EXPECT_EQ(with_nan.error().code, core::StatusCode::invalid_argument);
}

TEST(BoundsTest, BuildsTwoDimensionalBoundsFromUnorderedPoints) {
    static_assert(!std::same_as<Bounds2, ReferenceBounds2>);
    const auto bounds =
        Bounds2::from_points(Point2{.x = 4.0F, .y = -2.0F}, Point2{.x = -1.0F, .y = 3.0F});
    ASSERT_TRUE(bounds.has_value());

    EXPECT_EQ(bounds->minimum(), (Point2{.x = -1.0F, .y = -2.0F}));
    EXPECT_EQ(bounds->maximum(), (Point2{.x = 4.0F, .y = 3.0F}));
    EXPECT_TRUE(bounds->contains(Point2{.x = -1.0F, .y = 3.0F}));
    EXPECT_FALSE(bounds->contains(Point2{.x = 4.1F, .y = 0.0F}));
    EXPECT_FALSE(Bounds2::empty().contains(Point2{}));
}

TEST(BoundsTest, ValidatesOrderedCornersAndSupportsInfiniteBounds) {
    const auto reversed = Bounds3::from_minimum_maximum(Point3{.x = 1.0F, .y = 0.0F, .z = 0.0F},
                                                        Point3{.x = -1.0F, .y = 1.0F, .z = 1.0F});
    ASSERT_FALSE(reversed.has_value());
    EXPECT_EQ(reversed.error().code, core::StatusCode::invalid_argument);

    constexpr auto infinity = std::numeric_limits<TransportScalar>::infinity();
    constexpr auto unbounded = Bounds3::unbounded();
    EXPECT_TRUE(unbounded.contains(Point3{.x = -infinity, .y = 0.0F, .z = infinity}));

    const auto with_nan = Bounds2::from_points(
        Point2{}, Point2{.x = std::numeric_limits<TransportScalar>::quiet_NaN(), .y = 1.0F});
    ASSERT_FALSE(with_nan.has_value());
}

TEST(AabbIntersectionTest, ReturnsTheClosedRayParameterInterval) {
    const auto bounds = Bounds3::from_minimum_maximum(Point3{.x = -1.0F, .y = -1.0F, .z = -1.0F},
                                                      Point3{.x = 1.0F, .y = 1.0F, .z = 1.0F});
    ASSERT_TRUE(bounds.has_value());

    const auto hit = intersect_aabb(*bounds, Point3{.x = -2.0F, .y = 0.0F, .z = 0.0F},
                                    Vector3{.x = 1.0F, .y = 0.0F, .z = 0.0F});
    ASSERT_TRUE(hit.has_value());
    ASSERT_FALSE(hit->is_empty());
    EXPECT_FLOAT_EQ(hit->lower(), 1.0F);
    EXPECT_FLOAT_EQ(hit->upper(), 3.0F);

    const auto away = intersect_aabb(*bounds, Point3{.x = 2.0F, .y = 0.0F, .z = 0.0F},
                                     Vector3{.x = 1.0F, .y = 0.0F, .z = 0.0F});
    ASSERT_TRUE(away.has_value());
    EXPECT_TRUE(away->is_empty());
}

TEST(AabbIntersectionTest, HandlesParallelRaysWithoutReciprocalFallback) {
    const auto bounds = Bounds3::from_minimum_maximum(Point3{.x = -1.0F, .y = -1.0F, .z = -1.0F},
                                                      Point3{.x = 1.0F, .y = 1.0F, .z = 1.0F});
    ASSERT_TRUE(bounds.has_value());

    const auto inside_slabs = intersect_aabb(*bounds, Point3{.x = -2.0F, .y = 1.0F, .z = 0.0F},
                                             Vector3{.x = 1.0F, .y = -0.0F, .z = 0.0F});
    ASSERT_TRUE(inside_slabs.has_value());
    EXPECT_FALSE(inside_slabs->is_empty());

    const auto outside_parallel_slab =
        intersect_aabb(*bounds, Point3{.x = -2.0F, .y = 1.01F, .z = 0.0F},
                       Vector3{.x = 1.0F, .y = 0.0F, .z = 0.0F});
    ASSERT_TRUE(outside_parallel_slab.has_value());
    EXPECT_TRUE(outside_parallel_slab->is_empty());

    const auto zero_direction = intersect_aabb(*bounds, Point3{}, Vector3{});
    ASSERT_FALSE(zero_direction.has_value());
    EXPECT_EQ(zero_direction.error().code, core::StatusCode::invalid_argument);
}

TEST(AabbIntersectionTest, IncludesTangenciesAndClipsToTheRequestedRange) {
    const auto bounds = Bounds3::from_minimum_maximum(Point3{.x = -1.0F, .y = -1.0F, .z = -1.0F},
                                                      Point3{.x = 1.0F, .y = 1.0F, .z = 1.0F});
    ASSERT_TRUE(bounds.has_value());

    const auto tangent = intersect_aabb(*bounds, Point3{.x = -2.0F, .y = 1.0F, .z = 1.0F},
                                        Vector3{.x = 1.0F, .y = 0.0F, .z = 0.0F});
    ASSERT_TRUE(tangent.has_value());
    EXPECT_FALSE(tangent->is_empty());

    const auto short_range = Interval::closed(0.0F, 0.5F);
    ASSERT_TRUE(short_range.has_value());
    const auto clipped = intersect_aabb(*bounds, Point3{.x = -2.0F, .y = 0.0F, .z = 0.0F},
                                        Vector3{.x = 1.0F, .y = 0.0F, .z = 0.0F}, *short_range);
    ASSERT_TRUE(clipped.has_value());
    EXPECT_TRUE(clipped->is_empty());
}

TEST(AabbIntersectionTest, PropagatesInfiniteSlabsAndTinyDirections) {
    constexpr auto unbounded = Bounds3::unbounded();
    const auto infinite_hit =
        intersect_aabb(unbounded, Point3{}, Vector3{.x = 1.0F, .y = 0.0F, .z = 0.0F});
    ASSERT_TRUE(infinite_hit.has_value());
    EXPECT_FLOAT_EQ(infinite_hit->lower(), 0.0F);
    EXPECT_EQ(infinite_hit->upper(), std::numeric_limits<TransportScalar>::infinity());

    const auto finite_bounds = Bounds3::from_minimum_maximum(
        Point3{.x = -1.0F, .y = -1.0F, .z = -1.0F}, Point3{.x = 1.0F, .y = 1.0F, .z = 1.0F});
    ASSERT_TRUE(finite_bounds.has_value());
    const auto tiny_direction = intersect_aabb(
        *finite_bounds, Point3{},
        Vector3{.x = std::numeric_limits<TransportScalar>::denorm_min(), .y = 0.0F, .z = 0.0F});
    ASSERT_TRUE(tiny_direction.has_value());
    EXPECT_FALSE(tiny_direction->is_empty());
    EXPECT_TRUE(tiny_direction->contains(0.0F));
}

TEST(AabbIntersectionTest, RejectsNonFiniteRays) {
    constexpr auto bounds = Bounds3::unbounded();
    const auto invalid = intersect_aabb(
        bounds, Point3{.x = std::numeric_limits<TransportScalar>::infinity(), .y = 0.0F, .z = 0.0F},
        Vector3{.x = 1.0F, .y = 0.0F, .z = 0.0F});

    ASSERT_FALSE(invalid.has_value());
    EXPECT_EQ(invalid.error().code, core::StatusCode::invalid_argument);
}

} // namespace
} // namespace blackframe::renderer
