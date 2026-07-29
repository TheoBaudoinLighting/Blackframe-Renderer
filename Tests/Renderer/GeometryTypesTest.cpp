#include <Blackframe/Renderer/GeometryOperations.hpp>
#include <Blackframe/Renderer/GeometryTypes.hpp>
#include <cmath>
#include <concepts>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {
namespace {

template <typename Left, typename Right>
concept Addable = requires(const Left left, const Right right) { left + right; };

template <typename Left, typename Right>
concept Subtractable = requires(const Left left, const Right right) { left - right; };

template <typename Left, typename Right>
concept Dotable = requires(const Left left, const Right right) { dot(left, right); };

template <typename Value>
concept Normalizable = requires(const Value value) { normalized(value); };

TEST(GeometryTypesTest, KeepsAffineRolesAndPrecisionsDistinct) {
    static_assert(!std::same_as<Vector3, Point3>);
    static_assert(!std::same_as<Vector3, Normal3>);
    static_assert(!std::same_as<Point3, Normal3>);
    static_assert(!std::convertible_to<Vector3, Point3>);
    static_assert(!std::convertible_to<Point3, Vector3>);
    static_assert(!std::convertible_to<Vector3, Normal3>);
    static_assert(!std::convertible_to<Normal3, Vector3>);
    static_assert(!std::convertible_to<Vector3, ReferenceVector3>);
    static_assert(!Addable<Vector3, ReferenceVector3>);

    EXPECT_EQ(sizeof(Vector3), 12U);
    EXPECT_EQ(sizeof(Point3), 12U);
    EXPECT_EQ(sizeof(Normal3), 12U);
    EXPECT_EQ(sizeof(ReferenceVector3), 24U);
    EXPECT_EQ(sizeof(ReferencePoint3), 24U);
    EXPECT_EQ(sizeof(ReferenceNormal3), 24U);
}

TEST(GeometryTypesTest, AppliesVectorAlgebra) {
    constexpr auto left = Vector3{.x = 1.0F, .y = 2.0F, .z = 3.0F};
    constexpr auto right = Vector3{.x = -4.0F, .y = 5.0F, .z = 2.0F};

    EXPECT_EQ(left + right, (Vector3{.x = -3.0F, .y = 7.0F, .z = 5.0F}));
    EXPECT_EQ(left - right, (Vector3{.x = 5.0F, .y = -3.0F, .z = 1.0F}));
    EXPECT_EQ(-left, (Vector3{.x = -1.0F, .y = -2.0F, .z = -3.0F}));
    EXPECT_EQ(left * 2.0F, (Vector3{.x = 2.0F, .y = 4.0F, .z = 6.0F}));
    EXPECT_EQ(2.0F * left, left * 2.0F);
    EXPECT_EQ(left / 2.0F, (Vector3{.x = 0.5F, .y = 1.0F, .z = 1.5F}));
    EXPECT_FLOAT_EQ(dot(left, right), 12.0F);
    EXPECT_FLOAT_EQ(length_squared(left), 14.0F);
}

TEST(GeometryTypesTest, ComputesAnOrientedCrossProduct) {
    constexpr auto x_axis = Vector3{.x = 1.0F, .y = 0.0F, .z = 0.0F};
    constexpr auto y_axis = Vector3{.x = 0.0F, .y = 1.0F, .z = 0.0F};
    constexpr auto z_axis = cross(x_axis, y_axis);

    static_assert(z_axis == Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F});
    EXPECT_EQ(z_axis, (Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F}));
    EXPECT_FLOAT_EQ(dot(z_axis, x_axis), 0.0F);
    EXPECT_FLOAT_EQ(dot(z_axis, y_axis), 0.0F);
    EXPECT_EQ(cross(y_axis, x_axis), -z_axis);
}

TEST(GeometryTypesTest, AppliesAffinePointSemantics) {
    constexpr auto origin = Point3{.x = 1.0F, .y = 2.0F, .z = 3.0F};
    constexpr auto offset = Vector3{.x = 4.0F, .y = -2.0F, .z = 1.0F};
    constexpr auto translated = Point3{.x = 5.0F, .y = 0.0F, .z = 4.0F};

    static_assert(std::same_as<decltype(origin - translated), Vector3>);
    static_assert(std::same_as<decltype(origin + offset), Point3>);
    static_assert(std::same_as<decltype(origin - offset), Point3>);
    static_assert(!Addable<Point3, Point3>);
    static_assert(!Subtractable<Vector3, Point3>);
    static_assert(!Dotable<Point3, Vector3>);
    static_assert(!Normalizable<Point3>);

    EXPECT_EQ(origin + offset, translated);
    EXPECT_EQ(offset + origin, translated);
    EXPECT_EQ(translated - offset, origin);
    EXPECT_EQ(translated - origin, offset);
}

TEST(GeometryTypesTest, KeepsNormalAlgebraSeparateFromTranslation) {
    constexpr auto normal = Normal3{.x = 0.0F, .y = 2.0F, .z = 0.0F};
    constexpr auto direction = Vector3{.x = 1.0F, .y = 3.0F, .z = 4.0F};

    static_assert(!Addable<Point3, Normal3>);
    static_assert(!Addable<Vector3, Normal3>);
    static_assert(!Subtractable<Point3, Normal3>);
    static_assert(Dotable<Normal3, Vector3>);
    static_assert(Dotable<Vector3, Normal3>);

    EXPECT_EQ(normal * 0.5F, (Normal3{.x = 0.0F, .y = 1.0F, .z = 0.0F}));
    EXPECT_EQ(normal / 2.0F, (Normal3{.x = 0.0F, .y = 1.0F, .z = 0.0F}));
    EXPECT_EQ(normal + normal, (Normal3{.x = 0.0F, .y = 4.0F, .z = 0.0F}));
    EXPECT_EQ(normal - normal, Normal3{});
    EXPECT_EQ(-normal, (Normal3{.x = 0.0F, .y = -2.0F, .z = 0.0F}));
    EXPECT_FLOAT_EQ(dot(normal, direction), 6.0F);
    EXPECT_FLOAT_EQ(dot(direction, normal), 6.0F);
    EXPECT_FLOAT_EQ(length_squared(normal), 4.0F);
}

TEST(GeometryTypesTest, NormalizesVectorsAndNormalsInTheirOwnTypes) {
    const auto vector = normalized(Vector3{.x = 3.0F, .y = 0.0F, .z = 4.0F});
    ASSERT_TRUE(vector.has_value());
    static_assert(std::same_as<std::remove_cvref_t<decltype(*vector)>, Vector3>);
    EXPECT_FLOAT_EQ(length(*vector), 1.0F);
    EXPECT_FLOAT_EQ(vector->x, 0.6F);
    EXPECT_FLOAT_EQ(vector->z, 0.8F);

    const auto normal = normalized(ReferenceNormal3{.x = 0.0, .y = 0.0, .z = -2.0});
    ASSERT_TRUE(normal.has_value());
    static_assert(std::same_as<std::remove_cvref_t<decltype(*normal)>, ReferenceNormal3>);
    EXPECT_DOUBLE_EQ(length(*normal), 1.0);
    EXPECT_DOUBLE_EQ(normal->z, -1.0);
}

TEST(GeometryTypesTest, RejectsUndefinedNormalizationWithoutFallback) {
    const auto zero_vector = normalized(Vector3{});
    ASSERT_FALSE(zero_vector.has_value());
    EXPECT_EQ(zero_vector.error().code, core::StatusCode::invalid_argument);

    const auto non_finite_normal = normalized(Normal3{
        .x = std::numeric_limits<TransportScalar>::infinity(),
        .y = 0.0F,
        .z = 0.0F,
    });
    ASSERT_FALSE(non_finite_normal.has_value());
    EXPECT_EQ(non_finite_normal.error().code, core::StatusCode::invalid_argument);
}

} // namespace
} // namespace blackframe::renderer
