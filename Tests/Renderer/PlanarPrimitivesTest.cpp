#include <Blackframe/Renderer/Disk.hpp>
#include <Blackframe/Renderer/Plane.hpp>
#include <cmath>
#include <concepts>
#include <gtest/gtest.h>
#include <limits>

namespace blackframe::renderer {
namespace {

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<RayT<Scalar>>
make_planar_ray(const Point3T<Scalar> origin, const Vector3T<Scalar> direction,
                const Scalar t_min = Scalar{0},
                const Scalar t_max = std::numeric_limits<Scalar>::infinity()) {
    return RayT<Scalar>::create(origin, direction, t_min, t_max, Scalar{0}, AllRayVisibility,
                                VacuumMedium);
}

TEST(PlanarPrimitiveTest, KeepsPrecisionExplicitAndRejectsInvalidConstruction) {
    static_assert(!std::same_as<Plane, ReferencePlane>);
    static_assert(!std::same_as<PlaneHit, ReferencePlaneHit>);
    static_assert(!std::same_as<Disk, ReferenceDisk>);
    static_assert(!std::same_as<DiskHit, ReferenceDiskHit>);

    const auto plane = Plane::create(Point3{.x = 1.0F, .y = 2.0F, .z = 3.0F}, Normal3{.z = 4.0F});
    const auto disk =
        Disk::create(Point3{.x = 1.0F, .y = 2.0F, .z = 3.0F}, Normal3{.z = 4.0F}, 5.0F);
    ASSERT_TRUE(plane.has_value());
    ASSERT_TRUE(disk.has_value());
    EXPECT_EQ(plane->point(), (Point3{.x = 1.0F, .y = 2.0F, .z = 3.0F}));
    EXPECT_EQ(plane->orientation().normal(), (Normal3{.z = 1.0F}));
    EXPECT_EQ(disk->center(), plane->point());
    EXPECT_EQ(disk->orientation().normal(), plane->orientation().normal());
    EXPECT_FLOAT_EQ(disk->radius(), 5.0F);

    const auto infinity = std::numeric_limits<TransportScalar>::infinity();
    const auto nan = std::numeric_limits<TransportScalar>::quiet_NaN();
    EXPECT_FALSE(Plane::create(Point3{.x = infinity}, Normal3{.z = 1.0F}).has_value());
    EXPECT_FALSE(Plane::create(Point3{}, Normal3{}).has_value());
    EXPECT_FALSE(Plane::create(Point3{}, Normal3{.z = nan}).has_value());
    EXPECT_FALSE(Disk::create(Point3{}, Normal3{.z = 1.0F}, 0.0F).has_value());
    EXPECT_FALSE(Disk::create(Point3{}, Normal3{.z = 1.0F}, -1.0F).has_value());
    EXPECT_FALSE(Disk::create(Point3{}, Normal3{.z = 1.0F}, infinity).has_value());
    EXPECT_FALSE(Disk::create(Point3{.x = infinity}, Normal3{.z = 1.0F}, 1.0F).has_value());
    EXPECT_FALSE(Disk::create(Point3{}, Normal3{}, 1.0F).has_value());
    const auto invalid_disk = Disk::create(Point3{}, Normal3{.z = 1.0F}, nan);
    ASSERT_FALSE(invalid_disk.has_value());
    EXPECT_EQ(invalid_disk.error().code, core::StatusCode::invalid_argument);
}

TEST(PlaneIntersectionTest, PreservesDeclaredOrientationFromBothSides) {
    const auto plane =
        Plane::create(Point3{.x = 1.0F, .y = 2.0F, .z = 3.0F}, Normal3{.y = 3.0F, .z = 4.0F});
    const auto front_ray =
        make_planar_ray(Point3{.x = 1.0F, .y = 8.0F, .z = 11.0F}, Vector3{.y = -3.0F, .z = -4.0F});
    const auto back_ray =
        make_planar_ray(Point3{.x = 1.0F, .y = -1.0F, .z = -1.0F}, Vector3{.y = 3.0F, .z = 4.0F});
    ASSERT_TRUE(plane.has_value());
    ASSERT_TRUE(front_ray.has_value());
    ASSERT_TRUE(back_ray.has_value());

    const auto front = plane->intersect(*front_ray);
    const auto back = plane->intersect(*back_ray);
    ASSERT_TRUE(front.has_value());
    ASSERT_TRUE(back.has_value());
    ASSERT_TRUE(front->has_value());
    ASSERT_TRUE(back->has_value());
    EXPECT_FLOAT_EQ((**front).parameter, 2.0F);
    EXPECT_FLOAT_EQ((**back).parameter, 1.0F);
    EXPECT_EQ((**front).position, (Point3{.x = 1.0F, .y = 2.0F, .z = 3.0F}));
    EXPECT_EQ((**back).position, (**front).position);
    EXPECT_FLOAT_EQ((**front).geometric_normal.x, 0.0F);
    EXPECT_FLOAT_EQ((**front).geometric_normal.y, 0.6F);
    EXPECT_FLOAT_EQ((**front).geometric_normal.z, 0.8F);
    EXPECT_EQ((**back).geometric_normal, (**front).geometric_normal);
    EXPECT_LT(dot((**front).geometric_normal, front_ray->direction()), 0.0F);
    EXPECT_GT(dot((**back).geometric_normal, back_ray->direction()), 0.0F);
}

TEST(PlaneIntersectionTest, AppliesExactClosedRayClipping) {
    const auto plane = Plane::create(Point3{}, Normal3{.z = 1.0F});
    const auto included = make_planar_ray(Point3{.z = -2.0F}, Vector3{.z = 1.0F}, 2.0F, 2.0F);
    const auto clipped_before =
        make_planar_ray(Point3{.z = -2.0F}, Vector3{.z = 1.0F}, 0.0F, std::nextafter(2.0F, 1.0F));
    const auto clipped_after =
        make_planar_ray(Point3{.z = -2.0F}, Vector3{.z = 1.0F}, std::nextafter(2.0F, 3.0F), 3.0F);
    ASSERT_TRUE(plane.has_value());
    ASSERT_TRUE(included.has_value());
    ASSERT_TRUE(clipped_before.has_value());
    ASSERT_TRUE(clipped_after.has_value());

    const auto boundary = plane->intersect(*included);
    const auto before = plane->intersect(*clipped_before);
    const auto after = plane->intersect(*clipped_after);
    ASSERT_TRUE(boundary.has_value());
    ASSERT_TRUE(before.has_value());
    ASSERT_TRUE(after.has_value());
    EXPECT_TRUE(boundary->has_value());
    EXPECT_FALSE(before->has_value());
    EXPECT_FALSE(after->has_value());
}

TEST(PlaneIntersectionTest, RoundsAnObliqueRootBeforeApplyingClipping) {
    const auto plane =
        Plane::create(Point3{}, Normal3{.x = 841.4520264F, .y = -581.4399414F, .z = 683.4759521F});
    constexpr auto rounded_root = 3.412228107F;
    const auto ray = make_planar_ray(
        Point3{.x = -63.82965088F, .y = 865.0806885F, .z = -270.7992554F},
        Vector3{.x = 898.7358398F, .y = 190.1027832F, .z = -626.6774902F}, rounded_root);
    ASSERT_TRUE(plane.has_value());
    ASSERT_TRUE(ray.has_value());

    const auto intersection = plane->intersect(*ray);
    ASSERT_TRUE(intersection.has_value());
    ASSERT_TRUE(intersection->has_value());
    EXPECT_FLOAT_EQ((**intersection).parameter, rounded_root);
}

TEST(PlaneIntersectionTest, IgnoresRepresentableTangentialAnchorAndDirectionComponents) {
    const auto maximum = std::numeric_limits<TransportScalar>::max();
    const auto translated_plane = Plane::create(Point3{.x = maximum}, Normal3{.z = 1.0F});
    const auto translated_ray =
        make_planar_ray(Point3{.x = -maximum, .z = 1.0F}, Vector3{.z = -1.0F});
    const auto origin_plane = Plane::create(Point3{}, Normal3{.z = 1.0F});
    const auto tangent_ray = make_planar_ray(Point3{.z = 1.0F}, Vector3{.x = maximum, .z = -1.0F});
    ASSERT_TRUE(translated_plane.has_value());
    ASSERT_TRUE(translated_ray.has_value());
    ASSERT_TRUE(origin_plane.has_value());
    ASSERT_TRUE(tangent_ray.has_value());

    const auto translated_hit = translated_plane->intersect(*translated_ray);
    const auto tangent_hit = origin_plane->intersect(*tangent_ray);
    ASSERT_TRUE(translated_hit.has_value());
    ASSERT_TRUE(tangent_hit.has_value());
    ASSERT_TRUE(translated_hit->has_value());
    ASSERT_TRUE(tangent_hit->has_value());
    EXPECT_FLOAT_EQ((**translated_hit).parameter, 1.0F);
    EXPECT_EQ((**translated_hit).position, (Point3{.x = -maximum}));
    EXPECT_FLOAT_EQ((**tangent_hit).parameter, 1.0F);
    EXPECT_EQ((**tangent_hit).position, (Point3{.x = maximum}));
}

TEST(PlanarIntersectionTest, DistinguishesParallelMissFromCoplanarAmbiguity) {
    const auto plane = Plane::create(Point3{}, Normal3{.z = 1.0F});
    const auto disk = Disk::create(Point3{}, Normal3{.z = 1.0F}, 2.0F);
    const auto parallel = make_planar_ray(Point3{.z = 1.0F}, Vector3{.x = 1.0F, .y = 2.0F});
    const auto coplanar = make_planar_ray(Point3{}, Vector3{.x = 1.0F, .y = 2.0F});
    ASSERT_TRUE(plane.has_value());
    ASSERT_TRUE(disk.has_value());
    ASSERT_TRUE(parallel.has_value());
    ASSERT_TRUE(coplanar.has_value());

    const auto miss = plane->intersect(*parallel);
    const auto ambiguous = plane->intersect(*coplanar);
    const auto disk_ambiguous = disk->intersect(*coplanar);
    ASSERT_TRUE(miss.has_value());
    EXPECT_FALSE(miss->has_value());
    ASSERT_FALSE(ambiguous.has_value());
    ASSERT_FALSE(disk_ambiguous.has_value());
    EXPECT_EQ(ambiguous.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(disk_ambiguous.error().code, core::StatusCode::invalid_argument);
}

TEST(PlaneIntersectionTest, PreservesARepresentableRootForATinyDirection) {
    const auto plane = Plane::create(Point3{}, Normal3{.z = 1.0F});
    const auto ray = make_planar_ray(Point3{.z = -1.0F}, Vector3{.z = 1.0e-22F});
    ASSERT_TRUE(plane.has_value());
    ASSERT_TRUE(ray.has_value());

    const auto intersection = plane->intersect(*ray);
    ASSERT_TRUE(intersection.has_value());
    ASSERT_TRUE(intersection->has_value());
    EXPECT_NEAR((**intersection).parameter, 1.0e22F, 1.0e15F);
    EXPECT_NEAR((**intersection).position.z, 0.0F, 2.0e-6F);
}

TEST(DiskIntersectionTest, UsesAnInclusiveEuclideanRadius) {
    const auto disk =
        Disk::create(Point3{.x = 1.0F, .y = 2.0F, .z = 3.0F}, Normal3{.y = 2.0F}, 5.0F);
    const auto boundary =
        make_planar_ray(Point3{.x = 4.0F, .y = 8.0F, .z = 7.0F}, Vector3{.y = -2.0F});
    const auto inside = make_planar_ray(
        Point3{.x = 4.0F, .y = 8.0F, .z = std::nextafter(7.0F, 6.0F)}, Vector3{.y = -2.0F});
    const auto outside = make_planar_ray(
        Point3{.x = 4.0F, .y = 8.0F, .z = std::nextafter(7.0F, 8.0F)}, Vector3{.y = -2.0F});
    ASSERT_TRUE(disk.has_value());
    ASSERT_TRUE(boundary.has_value());
    ASSERT_TRUE(inside.has_value());
    ASSERT_TRUE(outside.has_value());

    const auto boundary_hit = disk->intersect(*boundary);
    const auto inside_hit = disk->intersect(*inside);
    const auto outside_hit = disk->intersect(*outside);
    ASSERT_TRUE(boundary_hit.has_value());
    ASSERT_TRUE(inside_hit.has_value());
    ASSERT_TRUE(outside_hit.has_value());
    ASSERT_TRUE(boundary_hit->has_value());
    EXPECT_TRUE(inside_hit->has_value());
    EXPECT_FALSE(outside_hit->has_value());
    EXPECT_EQ((**boundary_hit).position, (Point3{.x = 4.0F, .y = 2.0F, .z = 7.0F}));
    EXPECT_EQ((**boundary_hit).geometric_normal, (Normal3{.y = 1.0F}));
}

TEST(DiskIntersectionTest, PreservesOrientationAndClosedRayClipping) {
    const auto disk = ReferenceDisk::create(ReferencePoint3{}, ReferenceNormal3{.z = 2.0}, 2.0);
    const auto front =
        make_planar_ray(ReferencePoint3{.z = 2.0}, ReferenceVector3{.z = -2.0}, 1.0, 1.0);
    const auto back =
        make_planar_ray(ReferencePoint3{.z = -2.0}, ReferenceVector3{.z = 2.0}, 1.0, 1.0);
    const auto clipped = make_planar_ray(ReferencePoint3{.z = 2.0}, ReferenceVector3{.z = -2.0},
                                         0.0, std::nextafter(1.0, 0.0));
    const auto clipped_after = make_planar_ray(
        ReferencePoint3{.z = 2.0}, ReferenceVector3{.z = -2.0}, std::nextafter(1.0, 2.0), 2.0);
    ASSERT_TRUE(disk.has_value());
    ASSERT_TRUE(front.has_value());
    ASSERT_TRUE(back.has_value());
    ASSERT_TRUE(clipped.has_value());
    ASSERT_TRUE(clipped_after.has_value());

    const auto front_hit = disk->intersect(*front);
    const auto back_hit = disk->intersect(*back);
    const auto clipped_hit = disk->intersect(*clipped);
    const auto clipped_after_hit = disk->intersect(*clipped_after);
    ASSERT_TRUE(front_hit.has_value());
    ASSERT_TRUE(back_hit.has_value());
    ASSERT_TRUE(clipped_hit.has_value());
    ASSERT_TRUE(clipped_after_hit.has_value());
    ASSERT_TRUE(front_hit->has_value());
    ASSERT_TRUE(back_hit->has_value());
    EXPECT_DOUBLE_EQ((**front_hit).parameter, 1.0);
    EXPECT_DOUBLE_EQ((**back_hit).parameter, 1.0);
    EXPECT_EQ((**front_hit).geometric_normal, (ReferenceNormal3{.z = 1.0}));
    EXPECT_EQ((**back_hit).geometric_normal, (**front_hit).geometric_normal);
    EXPECT_FALSE(clipped_hit->has_value());
    EXPECT_FALSE(clipped_after_hit->has_value());
}

TEST(PlanarIntersectionTest, DoesNotHideUnrepresentableArithmeticAsAMiss) {
    const auto maximum = std::numeric_limits<TransportScalar>::max();
    const auto plane = Plane::create(Point3{.x = maximum}, Normal3{.x = 1.0F});
    const auto disk = Disk::create(Point3{.x = maximum}, Normal3{.x = 1.0F}, 1.0F);
    const auto ray = make_planar_ray(Point3{.x = -maximum}, Vector3{.x = 1.0F});
    const auto radial_disk = Disk::create(Point3{}, Normal3{.z = 1.0F}, 1.0F);
    const auto radial_ray = make_planar_ray(Point3{.x = maximum, .z = 1.0F}, Vector3{.z = -1.0F});
    ASSERT_TRUE(plane.has_value());
    ASSERT_TRUE(disk.has_value());
    ASSERT_TRUE(ray.has_value());
    ASSERT_TRUE(radial_disk.has_value());
    ASSERT_TRUE(radial_ray.has_value());

    const auto plane_error = plane->intersect(*ray);
    const auto disk_error = disk->intersect(*ray);
    const auto radial_error = radial_disk->intersect(*radial_ray);
    ASSERT_FALSE(plane_error.has_value());
    ASSERT_FALSE(disk_error.has_value());
    ASSERT_FALSE(radial_error.has_value());
    EXPECT_EQ(plane_error.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(disk_error.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(radial_error.error().code, core::StatusCode::invalid_argument);
}

} // namespace
} // namespace blackframe::renderer
