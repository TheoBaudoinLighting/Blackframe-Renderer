#include <Blackframe/Renderer/Triangle.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <gtest/gtest.h>
#include <limits>

namespace blackframe::renderer {
namespace {

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<RayT<Scalar>>
make_triangle_ray(const Point3T<Scalar> origin, const Vector3T<Scalar> direction,
                  const Scalar t_min = Scalar{0},
                  const Scalar t_max = std::numeric_limits<Scalar>::infinity()) {
    return RayT<Scalar>::create(origin, direction, t_min, t_max, Scalar{0}, AllRayVisibility,
                                VacuumMedium);
}

TEST(TriangleTest, KeepsPrecisionExplicitAndRejectsInvalidConstruction) {
    static_assert(!std::same_as<Triangle, ReferenceTriangle>);
    static_assert(!std::same_as<TriangleBarycentrics, ReferenceTriangleBarycentrics>);
    static_assert(!std::same_as<TriangleHit, ReferenceTriangleHit>);

    const auto triangle = Triangle::create(Point3{}, Point3{.x = 2.0F}, Point3{.y = 3.0F});
    ASSERT_TRUE(triangle.has_value());
    EXPECT_EQ(triangle->vertices()[0], Point3{});
    EXPECT_EQ(triangle->vertices()[1], (Point3{.x = 2.0F}));
    EXPECT_EQ(triangle->vertices()[2], (Point3{.y = 3.0F}));
    EXPECT_EQ(triangle->geometric_normal(), (Normal3{.z = 1.0F}));

    const auto infinity = std::numeric_limits<TransportScalar>::infinity();
    const auto nan = std::numeric_limits<TransportScalar>::quiet_NaN();
    EXPECT_FALSE(
        Triangle::create(Point3{.x = infinity}, Point3{.x = 1.0F}, Point3{.y = 1.0F}).has_value());
    EXPECT_FALSE(Triangle::create(Point3{}, Point3{.x = nan}, Point3{.y = 1.0F}).has_value());

    const auto duplicate = Triangle::create(Point3{}, Point3{}, Point3{.y = 1.0F});
    const auto collinear = Triangle::create(Point3{}, Point3{.x = 1.0F}, Point3{.x = 2.0F});
    ASSERT_FALSE(duplicate.has_value());
    ASSERT_FALSE(collinear.has_value());
    EXPECT_EQ(duplicate.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(collinear.error().code, core::StatusCode::invalid_argument);

    const auto thin =
        Triangle::create(Point3{}, Point3{.x = 1.0F}, Point3{.x = 1.0F, .y = 1.0e-20F});
    ASSERT_TRUE(thin.has_value());
    EXPECT_EQ(thin->geometric_normal(), (Normal3{.z = 1.0F}));
}

TEST(TriangleTest, RestoresAnAnisotropicNormalAfterAxisScaling) {
    const auto triangle =
        Triangle::create(Point3{}, Point3{.y = 1'048'576.0F}, Point3{.x = 0x1p-20F, .z = 1.0F});
    ASSERT_TRUE(triangle.has_value());
    EXPECT_EQ(triangle->geometric_normal(), (Normal3{.x = 1.0F, .z = -0x1p-20F}));
}

TEST(TriangleIntersectionTest, ComputesOrderedBarycentricsAndPosition) {
    const auto triangle = Triangle::create(Point3{}, Point3{.x = 2.0F}, Point3{.y = 2.0F});
    const auto ray =
        make_triangle_ray(Point3{.x = 0.5F, .y = 0.5F, .z = 2.0F}, Vector3{.z = -2.0F});
    ASSERT_TRUE(triangle.has_value());
    ASSERT_TRUE(ray.has_value());

    const auto intersection = triangle->intersect(*ray);
    ASSERT_TRUE(intersection.has_value());
    ASSERT_TRUE(intersection->has_value());
    const auto& hit = **intersection;
    EXPECT_FLOAT_EQ(hit.parameter, 1.0F);
    EXPECT_EQ(hit.position, (Point3{.x = 0.5F, .y = 0.5F}));
    EXPECT_EQ(hit.geometric_normal, (Normal3{.z = 1.0F}));
    EXPECT_FLOAT_EQ(hit.barycentrics.vertex0, 0.5F);
    EXPECT_FLOAT_EQ(hit.barycentrics.vertex1, 0.25F);
    EXPECT_FLOAT_EQ(hit.barycentrics.vertex2, 0.25F);
    EXPECT_FLOAT_EQ(hit.barycentrics.vertex0 + hit.barycentrics.vertex1 + hit.barycentrics.vertex2,
                    1.0F);

    const auto& vertices = triangle->vertices();
    const auto reconstructed = Point3{
        .x = hit.barycentrics.vertex0 * vertices[0].x + hit.barycentrics.vertex1 * vertices[1].x +
             hit.barycentrics.vertex2 * vertices[2].x,
        .y = hit.barycentrics.vertex0 * vertices[0].y + hit.barycentrics.vertex1 * vertices[1].y +
             hit.barycentrics.vertex2 * vertices[2].y,
        .z = hit.barycentrics.vertex0 * vertices[0].z + hit.barycentrics.vertex1 * vertices[1].z +
             hit.barycentrics.vertex2 * vertices[2].z,
    };
    EXPECT_EQ(reconstructed, hit.position);
}

TEST(TriangleIntersectionTest, RoundsNonDyadicBarycentricsBeforeClipping) {
    const auto triangle = Triangle::create(Point3{}, Point3{.x = 3.0F}, Point3{.y = 3.0F});
    constexpr auto rounded_third = 1.0F / 3.0F;
    const auto included = make_triangle_ray(Point3{.x = 1.0F, .y = 1.0F, .z = 1.0F},
                                            Vector3{.z = -3.0F}, rounded_third, rounded_third);
    const auto clipped_before =
        make_triangle_ray(Point3{.x = 1.0F, .y = 1.0F, .z = 1.0F}, Vector3{.z = -3.0F}, 0.0F,
                          std::nextafter(rounded_third, 0.0F));
    const auto clipped_after =
        make_triangle_ray(Point3{.x = 1.0F, .y = 1.0F, .z = 1.0F}, Vector3{.z = -3.0F},
                          std::nextafter(rounded_third, 1.0F), 1.0F);
    ASSERT_TRUE(triangle.has_value());
    ASSERT_TRUE(included.has_value());
    ASSERT_TRUE(clipped_before.has_value());
    ASSERT_TRUE(clipped_after.has_value());

    const auto hit = triangle->intersect(*included);
    const auto before = triangle->intersect(*clipped_before);
    const auto after = triangle->intersect(*clipped_after);
    ASSERT_TRUE(hit.has_value());
    ASSERT_TRUE(before.has_value());
    ASSERT_TRUE(after.has_value());
    ASSERT_TRUE(hit->has_value());
    EXPECT_FALSE(before->has_value());
    EXPECT_FALSE(after->has_value());
    EXPECT_FLOAT_EQ((**hit).parameter, rounded_third);
    EXPECT_FLOAT_EQ((**hit).barycentrics.vertex0, rounded_third);
    EXPECT_FLOAT_EQ((**hit).barycentrics.vertex1, rounded_third);
    EXPECT_FLOAT_EQ((**hit).barycentrics.vertex2, rounded_third);
    EXPECT_NEAR((**hit).barycentrics.vertex0 + (**hit).barycentrics.vertex1 +
                    (**hit).barycentrics.vertex2,
                1.0F, std::numeric_limits<TransportScalar>::epsilon());
}

TEST(TriangleIntersectionTest, RoundsAQuotientWithASubnormalExactResidual) {
    auto numerator = triangle_detail::FloatingExpansion<TransportScalar, 2>{};
    auto denominator = triangle_detail::FloatingExpansion<TransportScalar, 2>{};
    ASSERT_TRUE(triangle_detail::add_component(numerator, 1.0F));
    ASSERT_TRUE(triangle_detail::add_component(denominator,
                                               std::numeric_limits<TransportScalar>::denorm_min()));
    ASSERT_TRUE(triangle_detail::add_component(denominator, 1.0F));
    ASSERT_EQ(denominator.size, 2U);
    ASSERT_EQ(denominator.components[0], std::numeric_limits<TransportScalar>::denorm_min());
    ASSERT_EQ(denominator.components[1], 1.0F);

    const auto quotient = triangle_detail::correctly_rounded_quotient(
        numerator, denominator, "The synthetic triangle quotient is not representable.");
    ASSERT_TRUE(quotient.has_value()) << quotient.error().message;
    EXPECT_FLOAT_EQ(*quotient, 1.0F);
}

TEST(TriangleIntersectionTest, PreservesAHugeCommonAxisTranslationAndParameterScale) {
    const auto maximum = std::numeric_limits<TransportScalar>::max();
    const auto previous = std::nextafter(maximum, 0.0F);
    const auto step = maximum - previous;
    const auto triangle = Triangle::create(Point3{.x = maximum}, Point3{.x = maximum, .y = 2.0F},
                                           Point3{.x = maximum, .z = 2.0F});
    const auto ray = make_triangle_ray(Point3{.x = previous, .y = 0.5F, .z = 0.5F},
                                       Vector3{.x = step}, 1.0F, 1.0F);
    ASSERT_TRUE(triangle.has_value());
    ASSERT_TRUE(ray.has_value());
    EXPECT_EQ(triangle->geometric_normal(), (Normal3{.x = 1.0F}));

    const auto intersection = triangle->intersect(*ray);
    ASSERT_TRUE(intersection.has_value());
    ASSERT_TRUE(intersection->has_value());
    EXPECT_FLOAT_EQ((**intersection).parameter, 1.0F);
    EXPECT_EQ((**intersection).position, (Point3{.x = maximum, .y = 0.5F, .z = 0.5F}));
    EXPECT_FLOAT_EQ((**intersection).barycentrics.vertex0, 0.5F);
    EXPECT_FLOAT_EQ((**intersection).barycentrics.vertex1, 0.25F);
    EXPECT_FLOAT_EQ((**intersection).barycentrics.vertex2, 0.25F);
}

TEST(TriangleIntersectionTest, IncludesEveryVertexAndEdge) {
    const auto triangle = Triangle::create(Point3{}, Point3{.x = 2.0F}, Point3{.y = 2.0F});
    ASSERT_TRUE(triangle.has_value());

    struct BoundaryCase final {
        Point3 point;
        TriangleBarycentrics barycentrics;
    };
    constexpr auto cases = std::array{
        BoundaryCase{.point = {},
                     .barycentrics = {.vertex0 = 1.0F, .vertex1 = 0.0F, .vertex2 = 0.0F}},
        BoundaryCase{.point = {.x = 2.0F},
                     .barycentrics = {.vertex0 = 0.0F, .vertex1 = 1.0F, .vertex2 = 0.0F}},
        BoundaryCase{.point = {.y = 2.0F},
                     .barycentrics = {.vertex0 = 0.0F, .vertex1 = 0.0F, .vertex2 = 1.0F}},
        BoundaryCase{.point = {.x = 1.0F},
                     .barycentrics = {.vertex0 = 0.5F, .vertex1 = 0.5F, .vertex2 = 0.0F}},
        BoundaryCase{.point = {.x = 1.0F, .y = 1.0F},
                     .barycentrics = {.vertex0 = 0.0F, .vertex1 = 0.5F, .vertex2 = 0.5F}},
        BoundaryCase{.point = {.y = 1.0F},
                     .barycentrics = {.vertex0 = 0.5F, .vertex1 = 0.0F, .vertex2 = 0.5F}},
    };

    for (const auto& boundary : cases) {
        const auto ray =
            make_triangle_ray(boundary.point + Vector3{.z = 1.0F}, Vector3{.z = -1.0F});
        ASSERT_TRUE(ray.has_value());
        const auto intersection = triangle->intersect(*ray);
        ASSERT_TRUE(intersection.has_value());
        ASSERT_TRUE(intersection->has_value());
        EXPECT_EQ((**intersection).position, boundary.point);
        EXPECT_EQ((**intersection).barycentrics, boundary.barycentrics);
    }
}

TEST(TriangleIntersectionTest, PreservesWindingOrientationFromBothSides) {
    const auto triangle = Triangle::create(Point3{}, Point3{.x = 1.0F}, Point3{.y = 1.0F});
    const auto reversed = Triangle::create(Point3{}, Point3{.y = 1.0F}, Point3{.x = 1.0F});
    const auto front =
        make_triangle_ray(Point3{.x = 0.25F, .y = 0.25F, .z = 1.0F}, Vector3{.z = -1.0F});
    const auto back =
        make_triangle_ray(Point3{.x = 0.25F, .y = 0.25F, .z = -1.0F}, Vector3{.z = 1.0F});
    ASSERT_TRUE(triangle.has_value());
    ASSERT_TRUE(reversed.has_value());
    ASSERT_TRUE(front.has_value());
    ASSERT_TRUE(back.has_value());

    const auto front_hit = triangle->intersect(*front);
    const auto back_hit = triangle->intersect(*back);
    const auto reversed_hit = reversed->intersect(*front);
    ASSERT_TRUE(front_hit.has_value());
    ASSERT_TRUE(back_hit.has_value());
    ASSERT_TRUE(reversed_hit.has_value());
    ASSERT_TRUE(front_hit->has_value());
    ASSERT_TRUE(back_hit->has_value());
    ASSERT_TRUE(reversed_hit->has_value());
    EXPECT_EQ((**front_hit).geometric_normal, (Normal3{.z = 1.0F}));
    EXPECT_EQ((**back_hit).geometric_normal, (**front_hit).geometric_normal);
    EXPECT_EQ((**reversed_hit).geometric_normal, (Normal3{.z = -1.0F}));
    EXPECT_LT(dot((**front_hit).geometric_normal, front->direction()), 0.0F);
    EXPECT_GT(dot((**back_hit).geometric_normal, back->direction()), 0.0F);
}

TEST(TriangleIntersectionTest, AppliesExactClosedRayClipping) {
    const auto triangle = Triangle::create(Point3{}, Point3{.x = 1.0F}, Point3{.y = 1.0F});
    const auto included = make_triangle_ray(Point3{.x = 0.25F, .y = 0.25F, .z = 2.0F},
                                            Vector3{.z = -1.0F}, 2.0F, 2.0F);
    const auto clipped_before =
        make_triangle_ray(Point3{.x = 0.25F, .y = 0.25F, .z = 2.0F}, Vector3{.z = -1.0F}, 0.0F,
                          std::nextafter(2.0F, 1.0F));
    const auto clipped_after =
        make_triangle_ray(Point3{.x = 0.25F, .y = 0.25F, .z = 2.0F}, Vector3{.z = -1.0F},
                          std::nextafter(2.0F, 3.0F), 3.0F);
    ASSERT_TRUE(triangle.has_value());
    ASSERT_TRUE(included.has_value());
    ASSERT_TRUE(clipped_before.has_value());
    ASSERT_TRUE(clipped_after.has_value());

    const auto boundary = triangle->intersect(*included);
    const auto before = triangle->intersect(*clipped_before);
    const auto after = triangle->intersect(*clipped_after);
    ASSERT_TRUE(boundary.has_value());
    ASSERT_TRUE(before.has_value());
    ASSERT_TRUE(after.has_value());
    EXPECT_TRUE(boundary->has_value());
    EXPECT_FALSE(before->has_value());
    EXPECT_FALSE(after->has_value());
}

TEST(TriangleIntersectionTest, DistinguishesParallelMissFromCoplanarAmbiguity) {
    const auto triangle = Triangle::create(Point3{}, Point3{.x = 2.0F}, Point3{.y = 2.0F});
    const auto parallel = make_triangle_ray(Point3{.z = 1.0F}, Vector3{.x = 1.0F});
    const auto coplanar = make_triangle_ray(Point3{.x = 0.25F, .y = 0.25F}, Vector3{.x = 1.0F});
    ASSERT_TRUE(triangle.has_value());
    ASSERT_TRUE(parallel.has_value());
    ASSERT_TRUE(coplanar.has_value());

    const auto miss = triangle->intersect(*parallel);
    const auto ambiguous = triangle->intersect(*coplanar);
    const auto classified_ambiguity = triangle->intersect_classified(*coplanar);
    ASSERT_TRUE(miss.has_value());
    EXPECT_FALSE(miss->has_value());
    ASSERT_FALSE(ambiguous.has_value());
    EXPECT_EQ(ambiguous.error().code, core::StatusCode::invalid_argument);
    ASSERT_FALSE(classified_ambiguity.has_value());
    EXPECT_EQ(classified_ambiguity.error().kind, TriangleIntersectionErrorKind::coplanar_ambiguity);
    EXPECT_EQ(classified_ambiguity.error().diagnostic.code, core::StatusCode::invalid_argument);
}

TEST(TriangleWatertightTest, IncludesBothOwnersOfASharedEdge) {
    const auto triangle0 = Triangle::create(Point3{}, Point3{.x = 2.0F}, Point3{.y = 2.0F});
    const auto triangle1 =
        Triangle::create(Point3{.x = 2.0F, .y = 2.0F}, Point3{.y = 2.0F}, Point3{.x = 2.0F});
    ASSERT_TRUE(triangle0.has_value());
    ASSERT_TRUE(triangle1.has_value());

    constexpr auto parameters = std::array{0.0F, 0.125F, 0.25F, 0.5F, 0.75F, 0.875F, 1.0F};
    for (const auto parameter : parameters) {
        const auto point = Point3{.x = 2.0F * (1.0F - parameter), .y = 2.0F * parameter};
        for (const auto direction : std::array{Vector3{.z = -1.0F}, Vector3{.z = 1.0F}}) {
            const auto origin = point - direction;
            const auto ray = make_triangle_ray(origin, direction);
            ASSERT_TRUE(ray.has_value());

            const auto hit0 = triangle0->intersect(*ray);
            const auto hit1 = triangle1->intersect(*ray);
            ASSERT_TRUE(hit0.has_value());
            ASSERT_TRUE(hit1.has_value());
            ASSERT_TRUE(hit0->has_value());
            ASSERT_TRUE(hit1->has_value());
            EXPECT_FLOAT_EQ((**hit0).parameter, 1.0F);
            EXPECT_FLOAT_EQ((**hit1).parameter, 1.0F);
            EXPECT_FLOAT_EQ((**hit0).barycentrics.vertex0, 0.0F);
            EXPECT_FLOAT_EQ((**hit1).barycentrics.vertex0, 0.0F);
        }
    }
}

TEST(TriangleWatertightTest, SelectsEitherSideWithoutAGap) {
    const auto triangle0 = Triangle::create(Point3{}, Point3{.x = 2.0F}, Point3{.y = 2.0F});
    const auto triangle1 =
        Triangle::create(Point3{.x = 2.0F, .y = 2.0F}, Point3{.y = 2.0F}, Point3{.x = 2.0F});
    ASSERT_TRUE(triangle0.has_value());
    ASSERT_TRUE(triangle1.has_value());

    const auto below = make_triangle_ray(
        Point3{.x = 1.0F, .y = std::nextafter(1.0F, 0.0F), .z = 1.0F}, Vector3{.z = -1.0F});
    const auto above = make_triangle_ray(
        Point3{.x = 1.0F, .y = std::nextafter(1.0F, 2.0F), .z = 1.0F}, Vector3{.z = -1.0F});
    ASSERT_TRUE(below.has_value());
    ASSERT_TRUE(above.has_value());

    const auto below0 = triangle0->intersect(*below);
    const auto below1 = triangle1->intersect(*below);
    const auto above0 = triangle0->intersect(*above);
    const auto above1 = triangle1->intersect(*above);
    ASSERT_TRUE(below0.has_value());
    ASSERT_TRUE(below1.has_value());
    ASSERT_TRUE(above0.has_value());
    ASSERT_TRUE(above1.has_value());
    EXPECT_TRUE(below0->has_value());
    EXPECT_FALSE(below1->has_value());
    EXPECT_FALSE(above0->has_value());
    EXPECT_TRUE(above1->has_value());
}

TEST(TriangleWatertightTest, PreservesATranslatedObliqueSharedEdge) {
    constexpr auto base = Point3{.x = 1'048'576.0F, .y = -1'048'576.0F, .z = 1'048'576.0F};
    constexpr auto tangent0 = Vector3{.x = 16.0F, .z = 4.0F};
    constexpr auto tangent1 = Vector3{.y = 16.0F, .z = 8.0F};
    const auto vertex1 = base + tangent0;
    const auto vertex2 = base + tangent1;
    const auto opposite = base + tangent0 + tangent1;
    const auto triangle0 = Triangle::create(base, vertex1, vertex2);
    const auto triangle1 = Triangle::create(opposite, vertex2, vertex1);
    ASSERT_TRUE(triangle0.has_value());
    ASSERT_TRUE(triangle1.has_value());

    constexpr auto edge_parameters = std::array{0.0F, 0.25F, 0.5F, 0.75F, 1.0F};
    constexpr auto directions = std::array{
        Vector3{.x = 1.0F, .y = 2.0F, .z = -4.0F}, Vector3{.x = -3.0F, .y = 2.0F, .z = 5.0F},
        Vector3{.x = 8.0F, .y = 1.0F, .z = -1.0F}, Vector3{.x = 1.0F, .y = 8.0F, .z = -1.0F}};
    for (const auto edge_parameter : edge_parameters) {
        const auto edge_offset = (vertex2 - vertex1) * edge_parameter;
        const auto point = vertex1 + edge_offset;
        for (const auto direction : directions) {
            const auto origin = point - direction * 8.0F;
            const auto ray = make_triangle_ray(origin, direction);
            ASSERT_TRUE(ray.has_value());

            const auto hit0 = triangle0->intersect(*ray);
            const auto hit1 = triangle1->intersect(*ray);
            ASSERT_TRUE(hit0.has_value());
            ASSERT_TRUE(hit1.has_value());
            ASSERT_TRUE(hit0->has_value());
            ASSERT_TRUE(hit1->has_value());
            EXPECT_FLOAT_EQ((**hit0).parameter, 8.0F);
            EXPECT_FLOAT_EQ((**hit1).parameter, 8.0F);
            EXPECT_FLOAT_EQ((**hit0).barycentrics.vertex0, 0.0F);
            EXPECT_FLOAT_EQ((**hit1).barycentrics.vertex0, 0.0F);
            EXPECT_EQ((**hit0).position, point);
            EXPECT_EQ((**hit1).position, point);
        }
    }
}

TEST(TriangleIntersectionTest, SupportsReferencePrecision) {
    const auto triangle = ReferenceTriangle::create(ReferencePoint3{.x = 1.0, .y = 2.0, .z = 3.0},
                                                    ReferencePoint3{.x = 5.0, .y = 2.0, .z = 3.0},
                                                    ReferencePoint3{.x = 1.0, .y = 6.0, .z = 3.0});
    const auto ray = make_triangle_ray(ReferencePoint3{.x = 2.0, .y = 3.0, .z = 7.0},
                                       ReferenceVector3{.z = -2.0});
    ASSERT_TRUE(triangle.has_value());
    ASSERT_TRUE(ray.has_value());

    const auto intersection = triangle->intersect(*ray);
    ASSERT_TRUE(intersection.has_value());
    ASSERT_TRUE(intersection->has_value());
    EXPECT_DOUBLE_EQ((**intersection).parameter, 2.0);
    EXPECT_EQ((**intersection).position, (ReferencePoint3{.x = 2.0, .y = 3.0, .z = 3.0}));
    EXPECT_DOUBLE_EQ((**intersection).barycentrics.vertex0, 0.5);
    EXPECT_DOUBLE_EQ((**intersection).barycentrics.vertex1, 0.25);
    EXPECT_DOUBLE_EQ((**intersection).barycentrics.vertex2, 0.25);
    EXPECT_EQ((**intersection).geometric_normal, (ReferenceNormal3{.z = 1.0}));
}

TEST(TriangleIntersectionTest, DoesNotHideUnrepresentableArithmeticAsAMiss) {
    const auto maximum = std::numeric_limits<TransportScalar>::max();
    const auto minimum = std::numeric_limits<TransportScalar>::denorm_min();
    const auto triangle =
        Triangle::create(Point3{.x = maximum, .y = maximum}, Point3{.x = -maximum, .y = maximum},
                         Point3{.x = maximum, .y = -maximum});
    const auto ray = make_triangle_ray(Point3{.z = -maximum}, Vector3{.z = minimum});
    ASSERT_TRUE(triangle.has_value());
    ASSERT_TRUE(ray.has_value());

    const auto intersection = triangle->intersect(*ray);
    ASSERT_FALSE(intersection.has_value());
    EXPECT_EQ(intersection.error().code, core::StatusCode::invalid_argument);
}

} // namespace
} // namespace blackframe::renderer
