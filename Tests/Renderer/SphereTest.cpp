#include <Blackframe/Renderer/Sphere.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {
namespace {

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<RayT<Scalar>>
make_ray(const Point3T<Scalar> origin, const Vector3T<Scalar> direction,
         const Scalar t_min = Scalar{0},
         const Scalar t_max = std::numeric_limits<Scalar>::infinity()) {
    return RayT<Scalar>::create(origin, direction, t_min, t_max, Scalar{0}, AllRayVisibility,
                                VacuumMedium);
}

TEST(SphereTest, KeepsPrecisionExplicitAndRejectsInvalidSpheres) {
    static_assert(!std::same_as<Sphere, ReferenceSphere>);
    static_assert(!std::same_as<SphereHit, ReferenceSphereHit>);

    const auto sphere = Sphere::create(Point3{.x = 1.0F, .y = 2.0F, .z = 3.0F}, 4.0F);
    ASSERT_TRUE(sphere.has_value());
    EXPECT_EQ(sphere->center(), (Point3{.x = 1.0F, .y = 2.0F, .z = 3.0F}));
    EXPECT_FLOAT_EQ(sphere->radius(), 4.0F);

    const auto infinity = std::numeric_limits<TransportScalar>::infinity();
    const auto nan = std::numeric_limits<TransportScalar>::quiet_NaN();
    const auto zero_radius = Sphere::create(Point3{}, 0.0F);
    EXPECT_FALSE(Sphere::create(Point3{.x = infinity}, 1.0F).has_value());
    ASSERT_FALSE(zero_radius.has_value());
    EXPECT_EQ(zero_radius.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(Sphere::create(Point3{}, -1.0F).has_value());
    EXPECT_FALSE(Sphere::create(Point3{}, infinity).has_value());
    EXPECT_FALSE(Sphere::create(Point3{}, nan).has_value());
}

TEST(SphereIntersectionTest, SelectsNearAndFarExternalRootsWithinClosedBounds) {
    const auto sphere = Sphere::create(Point3{}, 1.0F);
    ASSERT_TRUE(sphere.has_value());

    const auto near_ray = make_ray(Point3{.x = 0.0F, .y = 0.0F, .z = -3.0F},
                                   Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F});
    ASSERT_TRUE(near_ray.has_value());
    const auto near_hit = sphere->intersect(*near_ray);
    ASSERT_TRUE(near_hit.has_value());
    ASSERT_TRUE(near_hit->has_value());
    EXPECT_FLOAT_EQ((*near_hit)->parameter, 2.0F);
    EXPECT_EQ((*near_hit)->position, (Point3{.x = 0.0F, .y = 0.0F, .z = -1.0F}));

    const auto far_ray = make_ray(Point3{.x = 0.0F, .y = 0.0F, .z = -3.0F},
                                  Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F}, 3.0F, 4.0F);
    ASSERT_TRUE(far_ray.has_value());
    const auto far_hit = sphere->intersect(*far_ray);
    ASSERT_TRUE(far_hit.has_value());
    ASSERT_TRUE(far_hit->has_value());
    EXPECT_FLOAT_EQ((*far_hit)->parameter, 4.0F);
    EXPECT_EQ((*far_hit)->position, (Point3{.x = 0.0F, .y = 0.0F, .z = 1.0F}));

    const auto near_boundary = make_ray(Point3{.x = 0.0F, .y = 0.0F, .z = -3.0F},
                                        Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F}, 2.0F, 2.0F);
    const auto far_boundary = make_ray(Point3{.x = 0.0F, .y = 0.0F, .z = -3.0F},
                                       Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F}, 4.0F, 4.0F);
    ASSERT_TRUE(near_boundary.has_value());
    ASSERT_TRUE(far_boundary.has_value());
    const auto near_boundary_hit = sphere->intersect(*near_boundary);
    const auto far_boundary_hit = sphere->intersect(*far_boundary);
    ASSERT_TRUE(near_boundary_hit.has_value());
    ASSERT_TRUE(far_boundary_hit.has_value());
    EXPECT_TRUE(near_boundary_hit->has_value());
    EXPECT_TRUE(far_boundary_hit->has_value());

    const auto excluded_roots =
        make_ray(Point3{.x = 0.0F, .y = 0.0F, .z = -3.0F}, Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F},
                 std::nextafter(2.0F, 3.0F), std::nextafter(4.0F, 3.0F));
    ASSERT_TRUE(excluded_roots.has_value());
    const auto excluded = sphere->intersect(*excluded_roots);
    ASSERT_TRUE(excluded.has_value());
    EXPECT_FALSE(excluded->has_value());
}

TEST(SphereIntersectionTest, SelectsThePositiveInternalRootForANonUnitDirection) {
    const auto sphere = Sphere::create(Point3{}, 1.0F);
    const auto ray = make_ray(Point3{}, Vector3{.x = 2.0F, .y = 0.0F, .z = 0.0F});
    ASSERT_TRUE(sphere.has_value());
    ASSERT_TRUE(ray.has_value());

    const auto intersection = sphere->intersect(*ray);
    ASSERT_TRUE(intersection.has_value());
    ASSERT_TRUE(intersection->has_value());
    const auto& hit = **intersection;
    EXPECT_FLOAT_EQ(hit.parameter, 0.5F);
    EXPECT_EQ(hit.position, (Point3{.x = 1.0F, .y = 0.0F, .z = 0.0F}));
    EXPECT_EQ(hit.geometric_normal, (Normal3{.x = 1.0F, .y = 0.0F, .z = 0.0F}));
    EXPECT_GT(dot(hit.geometric_normal, ray->direction()), 0.0F);
}

TEST(SphereIntersectionTest, IncludesTheSurfaceOriginAtTheClosedLowerBound) {
    const auto sphere = Sphere::create(Point3{}, 0.1F);
    const auto ray = make_ray(Point3{.x = 0.1F}, Vector3{.x = 1.0F}, 0.0F, 0.0F);
    ASSERT_TRUE(sphere.has_value());
    ASSERT_TRUE(ray.has_value());

    const auto intersection = sphere->intersect(*ray);
    ASSERT_TRUE(intersection.has_value());
    ASSERT_TRUE(intersection->has_value());
    const auto& hit = **intersection;
    EXPECT_FLOAT_EQ(hit.parameter, 0.0F);
    EXPECT_EQ(hit.position, (Point3{.x = 0.1F}));
    EXPECT_EQ(hit.geometric_normal, (Normal3{.x = 1.0F}));
}

TEST(SphereIntersectionTest, IncludesTheDoubleTangentRoot) {
    const auto sphere = Sphere::create(Point3{}, 1.0F);
    const auto ray = make_ray(Point3{.x = -2.0F, .y = 1.0F, .z = 0.0F},
                              Vector3{.x = 1.0F, .y = 0.0F, .z = 0.0F}, 2.0F, 2.0F);
    ASSERT_TRUE(sphere.has_value());
    ASSERT_TRUE(ray.has_value());

    const auto intersection = sphere->intersect(*ray);
    ASSERT_TRUE(intersection.has_value());
    ASSERT_TRUE(intersection->has_value());
    const auto& hit = **intersection;
    EXPECT_FLOAT_EQ(hit.parameter, 2.0F);
    EXPECT_EQ(hit.position, (Point3{.x = 0.0F, .y = 1.0F, .z = 0.0F}));
    EXPECT_EQ(hit.geometric_normal, (Normal3{.x = 0.0F, .y = 1.0F, .z = 0.0F}));
    EXPECT_EQ(hit.uv, (Point2{.x = 0.0F, .y = 0.0F}));
}

TEST(SphereIntersectionTest, PreservesAnObliqueDistantTangent) {
    const auto sphere = Sphere::create(Point3{}, 5.0F);
    const auto ray = make_ray(Point3{.x = -399'997.0F, .y = 300'004.0F, .z = 0.0F},
                              Vector3{.x = 4.0F, .y = -3.0F, .z = 0.0F});
    ASSERT_TRUE(sphere.has_value());
    ASSERT_TRUE(ray.has_value());

    const auto intersection = sphere->intersect(*ray);
    ASSERT_TRUE(intersection.has_value());
    ASSERT_TRUE(intersection->has_value());
    const auto& hit = **intersection;
    EXPECT_FLOAT_EQ(hit.parameter, 100'000.0F);
    EXPECT_EQ(hit.position, (Point3{.x = 3.0F, .y = 4.0F, .z = 0.0F}));
    EXPECT_FLOAT_EQ(hit.geometric_normal.x, 0.6F);
    EXPECT_FLOAT_EQ(hit.geometric_normal.y, 0.8F);
    EXPECT_FLOAT_EQ(hit.geometric_normal.z, 0.0F);
}

TEST(SphereIntersectionTest, PreservesATangentWhenSquaredProductsRoundInOppositeDirections) {
    const auto sphere = Sphere::create(Point3{}, 149.0F);
    const auto ray = make_ray(Point3{.x = -89.0F, .y = 191.0F, .z = 0.0F},
                              Vector3{.x = 140.0F, .y = -51.0F, .z = 0.0F}, 1.0F, 1.0F);
    ASSERT_TRUE(sphere.has_value());
    ASSERT_TRUE(ray.has_value());

    const auto intersection = sphere->intersect(*ray);
    ASSERT_TRUE(intersection.has_value());
    ASSERT_TRUE(intersection->has_value());
    const auto& hit = **intersection;
    EXPECT_FLOAT_EQ(hit.parameter, 1.0F);
    EXPECT_EQ(hit.position, (Point3{.x = 51.0F, .y = 140.0F, .z = 0.0F}));
    EXPECT_FLOAT_EQ(hit.geometric_normal.x, 51.0F / 149.0F);
    EXPECT_FLOAT_EQ(hit.geometric_normal.y, 140.0F / 149.0F);
    EXPECT_FLOAT_EQ(hit.geometric_normal.z, 0.0F);

    const auto outside_ray =
        make_ray(Point3{.x = -89.0F, .y = std::nextafter(191.0F, 192.0F), .z = 0.0F},
                 Vector3{.x = 140.0F, .y = -51.0F, .z = 0.0F});
    const auto inside_ray =
        make_ray(Point3{.x = -89.0F, .y = std::nextafter(191.0F, 190.0F), .z = 0.0F},
                 Vector3{.x = 140.0F, .y = -51.0F, .z = 0.0F});
    ASSERT_TRUE(outside_ray.has_value());
    ASSERT_TRUE(inside_ray.has_value());
    const auto outside = sphere->intersect(*outside_ray);
    const auto inside = sphere->intersect(*inside_ray);
    ASSERT_TRUE(outside.has_value());
    ASSERT_TRUE(inside.has_value());
    EXPECT_FALSE(outside->has_value());
    EXPECT_TRUE(inside->has_value());

    const auto reference_sphere = ReferenceSphere::create(ReferencePoint3{}, 149.0);
    const auto reference_ray =
        make_ray(ReferencePoint3{.x = -89.0, .y = 191.0, .z = 0.0},
                 ReferenceVector3{.x = 140.0, .y = -51.0, .z = 0.0}, 1.0, 1.0);
    ASSERT_TRUE(reference_sphere.has_value());
    ASSERT_TRUE(reference_ray.has_value());
    const auto reference_intersection = reference_sphere->intersect(*reference_ray);
    ASSERT_TRUE(reference_intersection.has_value());
    ASSERT_TRUE(reference_intersection->has_value());
    EXPECT_DOUBLE_EQ((**reference_intersection).parameter, 1.0);
    EXPECT_EQ((**reference_intersection).position,
              (ReferencePoint3{.x = 51.0, .y = 140.0, .z = 0.0}));
}

TEST(SphereIntersectionTest, PreservesARepresentableRootForATinyDirection) {
    const auto sphere = Sphere::create(Point3{}, 1.0F);
    const auto ray = make_ray(Point3{.x = 0.0F, .y = 0.0F, .z = -3.0F},
                              Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0e-22F});
    ASSERT_TRUE(sphere.has_value());
    ASSERT_TRUE(ray.has_value());

    const auto intersection = sphere->intersect(*ray);
    ASSERT_TRUE(intersection.has_value());
    ASSERT_TRUE(intersection->has_value());
    const auto& hit = **intersection;
    EXPECT_NEAR(hit.parameter, 2.0e22F, 2.0e15F);
    EXPECT_NEAR(hit.position.z, -1.0F, 2.0e-6F);
    EXPECT_NEAR(hit.geometric_normal.z, -1.0F, 2.0e-6F);
}

TEST(SphereIntersectionTest, ComputesAnOutwardNormalOnATranslatedSphere) {
    const auto sphere = ReferenceSphere::create(ReferencePoint3{.x = 1.0, .y = 2.0, .z = 3.0}, 5.0);
    const auto ray = make_ray(ReferencePoint3{.x = 7.0, .y = 10.0, .z = 3.0},
                              ReferenceVector3{.x = -3.0, .y = -4.0, .z = 0.0});
    ASSERT_TRUE(sphere.has_value());
    ASSERT_TRUE(ray.has_value());

    const auto intersection = sphere->intersect(*ray);
    ASSERT_TRUE(intersection.has_value());
    ASSERT_TRUE(intersection->has_value());
    const auto& hit = **intersection;
    EXPECT_DOUBLE_EQ(hit.parameter, 1.0);
    EXPECT_EQ(hit.position, (ReferencePoint3{.x = 4.0, .y = 6.0, .z = 3.0}));
    EXPECT_DOUBLE_EQ(hit.geometric_normal.x, 0.6);
    EXPECT_DOUBLE_EQ(hit.geometric_normal.y, 0.8);
    EXPECT_DOUBLE_EQ(hit.geometric_normal.z, 0.0);
    EXPECT_DOUBLE_EQ(hit.uv.x, 0.0);
    EXPECT_NEAR(hit.uv.y, std::atan2(0.6, 0.8) / std::numbers::pi, 1.0e-15);
}

TEST(SphereIntersectionTest, MapsCardinalNormalsToDocumentedUvCoordinates) {
    struct CardinalCase final {
        ReferencePoint3 origin;
        ReferenceVector3 direction;
        ReferenceNormal3 normal;
        ReferencePoint2 uv;
    };
    constexpr auto cases = std::array{
        CardinalCase{.origin = {.x = 3.0, .y = 0.0, .z = 0.0},
                     .direction = {.x = -1.0, .y = 0.0, .z = 0.0},
                     .normal = {.x = 1.0, .y = 0.0, .z = 0.0},
                     .uv = {.x = 0.0, .y = 0.5}},
        CardinalCase{.origin = {.x = 0.0, .y = 0.0, .z = 3.0},
                     .direction = {.x = 0.0, .y = 0.0, .z = -1.0},
                     .normal = {.x = 0.0, .y = 0.0, .z = 1.0},
                     .uv = {.x = 0.25, .y = 0.5}},
        CardinalCase{.origin = {.x = -3.0, .y = 0.0, .z = 0.0},
                     .direction = {.x = 1.0, .y = 0.0, .z = 0.0},
                     .normal = {.x = -1.0, .y = 0.0, .z = 0.0},
                     .uv = {.x = 0.5, .y = 0.5}},
        CardinalCase{.origin = {.x = 0.0, .y = 0.0, .z = -3.0},
                     .direction = {.x = 0.0, .y = 0.0, .z = 1.0},
                     .normal = {.x = 0.0, .y = 0.0, .z = -1.0},
                     .uv = {.x = 0.75, .y = 0.5}},
        CardinalCase{.origin = {.x = 0.0, .y = 3.0, .z = 0.0},
                     .direction = {.x = 0.0, .y = -1.0, .z = 0.0},
                     .normal = {.x = 0.0, .y = 1.0, .z = 0.0},
                     .uv = {.x = 0.0, .y = 0.0}},
        CardinalCase{.origin = {.x = 0.0, .y = -3.0, .z = 0.0},
                     .direction = {.x = 0.0, .y = 1.0, .z = 0.0},
                     .normal = {.x = 0.0, .y = -1.0, .z = 0.0},
                     .uv = {.x = 0.0, .y = 1.0}},
    };
    const auto sphere = ReferenceSphere::create(ReferencePoint3{}, 1.0);
    ASSERT_TRUE(sphere.has_value());

    for (const auto& cardinal : cases) {
        const auto ray = make_ray(cardinal.origin, cardinal.direction);
        ASSERT_TRUE(ray.has_value());
        const auto intersection = sphere->intersect(*ray);
        ASSERT_TRUE(intersection.has_value());
        ASSERT_TRUE(intersection->has_value());
        EXPECT_EQ((**intersection).geometric_normal, cardinal.normal);
        EXPECT_NEAR((**intersection).uv.x, cardinal.uv.x, 1.0e-15);
        EXPECT_NEAR((**intersection).uv.y, cardinal.uv.y, 1.0e-15);
    }
}

TEST(SphereIntersectionTest, DistinguishesMissesFromArithmeticErrors) {
    const auto sphere = Sphere::create(Point3{}, 1.0F);
    const auto miss_ray = make_ray(Point3{.x = 0.0F, .y = 2.0F, .z = -3.0F},
                                   Vector3{.x = 0.0F, .y = 0.0F, .z = 1.0F});
    ASSERT_TRUE(sphere.has_value());
    ASSERT_TRUE(miss_ray.has_value());
    const auto miss = sphere->intersect(*miss_ray);
    ASSERT_TRUE(miss.has_value());
    EXPECT_FALSE(miss->has_value());

    const auto maximum = std::numeric_limits<TransportScalar>::max();
    const auto extreme_sphere = Sphere::create(Point3{.x = maximum}, 1.0F);
    const auto extreme_ray =
        make_ray(Point3{.x = -maximum}, Vector3{.x = 1.0F, .y = 0.0F, .z = 0.0F});
    ASSERT_TRUE(extreme_sphere.has_value());
    ASSERT_TRUE(extreme_ray.has_value());
    const auto arithmetic_error = extreme_sphere->intersect(*extreme_ray);
    ASSERT_FALSE(arithmetic_error.has_value());
    EXPECT_EQ(arithmetic_error.error().code, core::StatusCode::invalid_argument);
}

} // namespace
} // namespace blackframe::renderer
