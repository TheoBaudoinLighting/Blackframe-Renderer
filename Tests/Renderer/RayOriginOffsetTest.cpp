#include <Blackframe/Renderer/Ray.hpp>
#include <Blackframe/Renderer/RayOriginOffset.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace blackframe::renderer {
namespace {

template <typename Scalar> [[nodiscard]] Scalar next_up(Scalar value, const int count) {
    for (int index = 0; index < count; ++index) {
        value = std::nextafter(value, std::numeric_limits<Scalar>::infinity());
    }
    return value;
}

template <typename Scalar>
[[nodiscard]] core::Result<Scalar> intersect_horizontal_plane(const RayT<Scalar>& ray,
                                                              const Scalar height) {
    if (ray.direction().z == Scalar{0}) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "The ray is parallel to the horizontal plane.",
        });
    }
    const auto parameter = (height - ray.origin().z) / ray.direction().z;
    if (!ray.contains_parameter(parameter)) {
        return std::unexpected(core::Error{
            .code = core::StatusCode::invalid_argument,
            .message = "The horizontal plane is outside the ray interval.",
        });
    }
    return parameter;
}

TEST(RayOriginOffsetTest, MovesToTheOutgoingSideUsingPropagatedError) {
    constexpr auto point = Point3{.x = 4.0F, .y = -2.0F, .z = 8.0F};
    constexpr auto error = Vector3{.x = 0.25F, .y = 0.5F, .z = 1.0F};
    constexpr auto normal = Normal3{.x = 0.0F, .y = 0.0F, .z = 7.0F};

    const auto above =
        offset_ray_origin(point, error, normal, Vector3{.x = 1.0F, .y = 0.0F, .z = 1.0F});
    const auto below =
        offset_ray_origin(point, error, normal, Vector3{.x = 1.0F, .y = 0.0F, .z = -1.0F});

    ASSERT_TRUE(above.has_value());
    ASSERT_TRUE(below.has_value());
    EXPECT_EQ(above->x, point.x);
    EXPECT_EQ(above->y, point.y);
    EXPECT_GT(above->z, point.z + error.z);
    EXPECT_EQ(below->x, point.x);
    EXPECT_EQ(below->y, point.y);
    EXPECT_LT(below->z, point.z - error.z);
}

TEST(RayOriginOffsetTest, RoundsOutwardWhenTheComputedDisplacementIsSubUlp) {
    constexpr auto point = Point3{.x = 16'777'216.0F, .y = 0.0F, .z = 0.0F};
    const auto shifted = offset_ray_origin(point, Vector3{.x = 0.25F, .y = 0.0F, .z = 0.0F},
                                           Normal3{.x = 1.0F, .y = 0.0F, .z = 0.0F},
                                           Vector3{.x = 1.0F, .y = 0.0F, .z = 0.0F});

    ASSERT_TRUE(shifted.has_value());
    EXPECT_EQ(shifted->x,
              std::nextafter(point.x, std::numeric_limits<TransportScalar>::infinity()));
}

TEST(RayOriginOffsetTest, HandlesExtremeFiniteNormalScales) {
    constexpr auto point = ReferencePoint3{.x = 1.0, .y = 2.0, .z = 3.0};
    constexpr auto error = ReferenceVector3{.x = 0.0, .y = 0.0, .z = 0.25};
    const auto tiny_normal =
        offset_ray_origin(point, error,
                          ReferenceNormal3{
                              .x = 0.0,
                              .y = 0.0,
                              .z = std::numeric_limits<ReferenceScalar>::denorm_min(),
                          },
                          ReferenceVector3{.x = 0.0, .y = 0.0, .z = 1.0});
    const auto huge_normal = offset_ray_origin(point, error,
                                               ReferenceNormal3{
                                                   .x = 0.0,
                                                   .y = 0.0,
                                                   .z = std::numeric_limits<ReferenceScalar>::max(),
                                               },
                                               ReferenceVector3{.x = 0.0, .y = 0.0, .z = 1.0});

    ASSERT_TRUE(tiny_normal.has_value());
    ASSERT_TRUE(huge_normal.has_value());
    EXPECT_EQ(*tiny_normal, *huge_normal);
}

TEST(RayOriginOffsetTest, RejectsInvalidInputsInsteadOfLeavingTheOriginUnchanged) {
    constexpr auto point = Point3{};
    constexpr auto error = Vector3{.x = 0.0F, .y = 0.0F, .z = 0.1F};
    constexpr auto normal = Normal3{.x = 0.0F, .y = 0.0F, .z = 1.0F};
    constexpr auto direction = Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F};

    EXPECT_FALSE(offset_ray_origin(point, Vector3{.x = -0.1F}, normal, direction).has_value());
    EXPECT_FALSE(offset_ray_origin(point, error, Normal3{}, direction).has_value());
    EXPECT_FALSE(offset_ray_origin(point, error, normal, Vector3{}).has_value());
    EXPECT_FALSE(offset_ray_origin(Point3{.x = std::numeric_limits<TransportScalar>::infinity()},
                                   error, normal, direction)
                     .has_value());
    EXPECT_FALSE(offset_ray_origin(Point3{.x = std::numeric_limits<TransportScalar>::max()},
                                   Vector3{.x = 0.1F}, Normal3{.x = 1.0F}, Vector3{.x = 1.0F})
                     .has_value());
}

TEST(RayOriginOffsetTest, CoplanarSceneAvoidsAcneWithoutSkippingNearbyGeometry) {
    constexpr auto surface_height = 1'024.0F;
    constexpr auto surface_point = Point3{.x = 0.0F, .y = 0.0F, .z = surface_height};
    const auto one_ulp = next_up(surface_height, 1) - surface_height;
    const auto nearby_height = next_up(surface_height, 16);

    const auto origin = offset_ray_origin(
        surface_point, Vector3{.x = 0.0F, .y = 0.0F, .z = one_ulp},
        Normal3{.x = 0.0F, .y = 0.0F, .z = 1.0F}, Vector3{.x = 1.0F, .y = 0.0F, .z = 1.0F});
    ASSERT_TRUE(origin.has_value());
    ASSERT_GT(origin->z, surface_height);
    ASSERT_LT(origin->z, nearby_height);

    const auto ray = Ray::create(*origin, Vector3{.x = 1.0F, .y = 0.0F, .z = 1.0F}, 0.0F,
                                 std::numeric_limits<TransportScalar>::infinity(), 0.0F,
                                 AllRayVisibility, VacuumMedium);
    ASSERT_TRUE(ray.has_value());

    EXPECT_FALSE(intersect_horizontal_plane(*ray, surface_height).has_value());
    const auto nearby_hit = intersect_horizontal_plane(*ray, nearby_height);
    ASSERT_TRUE(nearby_hit.has_value());
    EXPECT_GT(*nearby_hit, 0.0F);
}

} // namespace
} // namespace blackframe::renderer
