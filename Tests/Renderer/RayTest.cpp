#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/Renderer/RayDiagnostics.hpp>
#include <cmath>
#include <concepts>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {
namespace {

TEST(RayTest, StoresEveryExplicitTransportFieldWithoutNormalizing) {
    constexpr auto origin = Point3{.x = 1.0F, .y = -2.0F, .z = 3.0F};
    constexpr auto direction = Vector3{.x = 0.0F, .y = 0.0F, .z = 2.0F};
    constexpr auto medium = MediumId{.value = 17};
    const auto ray = Ray::create(origin, direction, 0.25F, 12.0F, 0.75F, 0x00FF00FFU, medium);

    ASSERT_TRUE(ray.has_value());
    EXPECT_EQ(ray->origin(), origin);
    EXPECT_EQ(ray->direction(), direction);
    EXPECT_FLOAT_EQ(ray->t_min(), 0.25F);
    EXPECT_FLOAT_EQ(ray->t_max(), 12.0F);
    EXPECT_FLOAT_EQ(ray->time(), 0.75F);
    EXPECT_EQ(ray->mask(), 0x00FF00FFU);
    EXPECT_EQ(ray->current_medium(), medium);
}

TEST(RayTest, AcceptsInfiniteTMaxAndExplicitEmptyVisibility) {
    const auto ray = Ray::create(Point3{}, Vector3{.x = 1.0F, .y = 0.0F, .z = 0.0F}, 0.0F,
                                 std::numeric_limits<TransportScalar>::infinity(), 0.0F, RayMask{0},
                                 VacuumMedium);

    ASSERT_TRUE(ray.has_value());
    EXPECT_EQ(ray->t_max(), std::numeric_limits<TransportScalar>::infinity());
    EXPECT_EQ(ray->mask(), RayMask{0});
    EXPECT_EQ(ray->current_medium(), VacuumMedium);
}

TEST(RayTest, RejectsInvalidGeometryIntervalsAndTime) {
    const auto valid_origin = Point3{};
    const auto valid_direction = Vector3{.x = 1.0F, .y = 0.0F, .z = 0.0F};
    const auto create = [&](const Point3 origin, const Vector3 direction,
                            const TransportScalar t_min, const TransportScalar t_max,
                            const TransportScalar time) {
        return Ray::create(origin, direction, t_min, t_max, time, AllRayVisibility, VacuumMedium);
    };

    EXPECT_FALSE(create(valid_origin, Vector3{}, 0.0F, 1.0F, 0.0F).has_value());
    EXPECT_FALSE(create(Point3{.x = std::numeric_limits<TransportScalar>::infinity()},
                        valid_direction, 0.0F, 1.0F, 0.0F)
                     .has_value());
    EXPECT_FALSE(create(valid_origin, valid_direction, -1.0F, 1.0F, 0.0F).has_value());
    EXPECT_FALSE(create(valid_origin, valid_direction, 2.0F, 1.0F, 0.0F).has_value());
    EXPECT_FALSE(create(valid_origin, valid_direction, 0.0F,
                        std::numeric_limits<TransportScalar>::quiet_NaN(), 0.0F)
                     .has_value());
    EXPECT_FALSE(create(valid_origin, valid_direction, 0.0F, 1.0F,
                        std::numeric_limits<TransportScalar>::infinity())
                     .has_value());
}

TEST(RayTest, EvaluatesOnlyFiniteParametersInsideItsBounds) {
    const auto ray = Ray::create(Point3{.x = 1.0F, .y = 2.0F, .z = 3.0F},
                                 Vector3{.x = 2.0F, .y = -1.0F, .z = 0.5F}, 1.0F, 3.0F, 0.0F,
                                 AllRayVisibility, VacuumMedium);
    ASSERT_TRUE(ray.has_value());

    const auto at_minimum = ray->at(1.0F);
    ASSERT_TRUE(at_minimum.has_value());
    EXPECT_EQ(*at_minimum, (Point3{.x = 3.0F, .y = 1.0F, .z = 3.5F}));

    const auto at_maximum = ray->at(3.0F);
    ASSERT_TRUE(at_maximum.has_value());
    EXPECT_EQ(*at_maximum, (Point3{.x = 7.0F, .y = -1.0F, .z = 4.5F}));

    EXPECT_FALSE(ray->at(std::nextafter(1.0F, 0.0F)).has_value());
    EXPECT_FALSE(ray->at(std::numeric_limits<TransportScalar>::infinity()).has_value());
}

TEST(RayDiagnosticTest, SerializesTransportRayWithStableExactBits) {
    const auto ray = Ray::create(
        Point3{.x = 1.0F, .y = -2.0F, .z = 0.0F}, Vector3{.x = -0.0F, .y = 0.5F, .z = -1.0F}, 0.25F,
        std::numeric_limits<TransportScalar>::infinity(), 0.5F, 0xA5A5A5A5U, MediumId{.value = 42});
    ASSERT_TRUE(ray.has_value());

    constexpr auto expected =
        R"({"schema_version":1,"precision":"float32","origin_bits":["0x3f800000","0xc0000000","0x00000000"],"direction_bits":["0x80000000","0x3f000000","0xbf800000"],"t_min_bits":"0x3e800000","t_max_bits":"0x7f800000","time_bits":"0x3f000000","mask":"0xa5a5a5a5","current_medium":"0x0000002a"})";
    EXPECT_EQ(serialize_ray_diagnostic(*ray), expected);
    EXPECT_EQ(serialize_ray_diagnostic(*ray), serialize_ray_diagnostic(*ray));
}

TEST(RayDiagnosticTest, SerializesReferenceRayWithStableExactBits) {
    const auto ray = ReferenceRay::create(ReferencePoint3{.x = 1.0, .y = -2.0, .z = 0.0},
                                          ReferenceVector3{.x = -0.0, .y = 0.5, .z = -1.0}, 0.25,
                                          4.0, 0.5, 0x0000000FU, MediumId{.value = 7});
    ASSERT_TRUE(ray.has_value());

    constexpr auto expected =
        R"({"schema_version":1,"precision":"float64","origin_bits":["0x3ff0000000000000","0xc000000000000000","0x0000000000000000"],"direction_bits":["0x8000000000000000","0x3fe0000000000000","0xbff0000000000000"],"t_min_bits":"0x3fd0000000000000","t_max_bits":"0x4010000000000000","time_bits":"0x3fe0000000000000","mask":"0x0000000f","current_medium":"0x00000007"})";
    EXPECT_EQ(serialize_ray_diagnostic(*ray), expected);
}

TEST(RayTest, KeepsTheRayAndMediumContractsTriviallyCopyable) {
    static_assert(!std::same_as<Ray, ReferenceRay>);
    static_assert(std::is_trivially_copyable_v<Ray>);
    static_assert(std::is_trivially_copyable_v<ReferenceRay>);
    static_assert(std::is_trivially_copyable_v<MediumId>);
    EXPECT_EQ(sizeof(MediumId), 4U);
}

} // namespace
} // namespace blackframe::renderer
