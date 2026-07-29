#include <Blackframe/Renderer/Sphere.hpp>
#include <Blackframe/Renderer/SurfaceInteraction.hpp>
#include <Blackframe/Renderer/Triangle.hpp>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <type_traits>

namespace blackframe::renderer {
namespace {

template <GeometryScalar Scalar>
[[nodiscard]] core::Result<RayT<Scalar>> make_surface_ray(const Point3T<Scalar> origin,
                                                          const Vector3T<Scalar> direction,
                                                          const Scalar time) {
    return RayT<Scalar>::create(origin, direction, Scalar{0},
                                std::numeric_limits<Scalar>::infinity(), time, AllRayVisibility,
                                VacuumMedium);
}

constexpr auto SyntheticIdentifiers = SurfaceIdentifiers{
    .instance = {.value = 17},
    .geometry = {.value = 23},
    .primitive = {.value = 31},
    .material = {.value = 47},
};

TEST(SurfaceInteractionTest, KeepsIdentifiersAndPrecisionTypesDistinct) {
    static_assert(!std::same_as<InstanceId, GeometryId>);
    static_assert(!std::same_as<InstanceId, PrimitiveId>);
    static_assert(!std::same_as<InstanceId, MaterialId>);
    static_assert(!std::same_as<GeometryId, PrimitiveId>);
    static_assert(!std::same_as<GeometryId, MaterialId>);
    static_assert(!std::same_as<PrimitiveId, MaterialId>);
    static_assert(!std::same_as<SurfaceInteraction, ReferenceSurfaceInteraction>);
    static_assert(!std::convertible_to<std::uint32_t, InstanceId>);
    static_assert(!std::convertible_to<std::uint32_t, GeometryId>);
    static_assert(!std::convertible_to<std::uint32_t, PrimitiveId>);
    static_assert(!std::convertible_to<std::uint32_t, MaterialId>);
    static_assert(std::is_trivially_copyable_v<SurfaceInteraction>);
    static_assert(std::is_trivially_copyable_v<ReferenceSurfaceInteraction>);

    EXPECT_EQ(SyntheticIdentifiers.instance, (InstanceId{.value = 17}));
    EXPECT_EQ(SyntheticIdentifiers.geometry, (GeometryId{.value = 23}));
    EXPECT_EQ(SyntheticIdentifiers.primitive, (PrimitiveId{.value = 31}));
    EXPECT_EQ(SyntheticIdentifiers.material, (MaterialId{.value = 47}));
}

TEST(SurfaceInteractionTest, StoresAnalyticSphereValuesExactly) {
    const auto sphere = Sphere::create(Point3{.x = 1.0F, .y = 2.0F, .z = 3.0F}, 2.0F);
    const auto ray =
        make_surface_ray(Point3{.x = 5.0F, .y = 2.0F, .z = 3.0F}, Vector3{.x = -1.0F}, 0.375F);
    ASSERT_TRUE(sphere.has_value());
    ASSERT_TRUE(ray.has_value());

    const auto sphere_result = sphere->intersect(*ray);
    ASSERT_TRUE(sphere_result.has_value());
    ASSERT_TRUE(sphere_result->has_value());
    const auto& hit = **sphere_result;
    ASSERT_EQ(hit.position, (Point3{.x = 3.0F, .y = 2.0F, .z = 3.0F}));
    ASSERT_EQ(hit.geometric_normal, (Normal3{.x = 1.0F}));
    ASSERT_EQ(hit.uv, (Point2{.x = 0.0F, .y = 0.5F}));

    const auto dpdu = Vector3{.z = 4.0F * std::numbers::pi_v<TransportScalar>};
    const auto dpdv = Vector3{.y = -2.0F * std::numbers::pi_v<TransportScalar>};
    const auto interaction =
        SurfaceInteraction::create(hit.position, hit.geometric_normal, hit.geometric_normal, hit.uv,
                                   dpdu, dpdv, SyntheticIdentifiers, ray->time());
    ASSERT_TRUE(interaction.has_value());

    EXPECT_EQ(interaction->position(), hit.position);
    EXPECT_EQ(interaction->geometric_normal(), hit.geometric_normal);
    EXPECT_EQ(interaction->shading_normal(), hit.geometric_normal);
    EXPECT_EQ(interaction->uv(), hit.uv);
    EXPECT_EQ(interaction->dpdu(), dpdu);
    EXPECT_EQ(interaction->dpdv(), dpdv);
    EXPECT_EQ(interaction->identifiers(), SyntheticIdentifiers);
    EXPECT_FLOAT_EQ(interaction->time(), 0.375F);
}

TEST(SurfaceInteractionTest, StoresSyntheticTriangleValuesAndDistinctNormals) {
    const auto triangle = Triangle::create(Point3{}, Point3{.x = 2.0F}, Point3{.y = 4.0F});
    const auto ray =
        make_surface_ray(Point3{.x = 0.5F, .y = 1.0F, .z = 2.0F}, Vector3{.z = -1.0F}, 1.25F);
    ASSERT_TRUE(triangle.has_value());
    ASSERT_TRUE(ray.has_value());

    const auto triangle_result = triangle->intersect(*ray);
    ASSERT_TRUE(triangle_result.has_value());
    ASSERT_TRUE(triangle_result->has_value());
    const auto& hit = **triangle_result;
    ASSERT_EQ(hit.position, (Point3{.x = 0.5F, .y = 1.0F}));
    ASSERT_EQ(hit.geometric_normal, (Normal3{.z = 1.0F}));
    ASSERT_FLOAT_EQ(hit.barycentrics.vertex0, 0.5F);
    ASSERT_FLOAT_EQ(hit.barycentrics.vertex1, 0.25F);
    ASSERT_FLOAT_EQ(hit.barycentrics.vertex2, 0.25F);

    constexpr auto uv = Point2{.x = 0.25F, .y = 0.25F};
    constexpr auto shading_normal = Normal3{.y = 0.6F, .z = 0.8F};
    constexpr auto dpdu = Vector3{.x = 2.0F};
    constexpr auto dpdv = Vector3{.y = 4.0F};
    const auto interaction =
        SurfaceInteraction::create(hit.position, hit.geometric_normal, shading_normal, uv, dpdu,
                                   dpdv, SyntheticIdentifiers, ray->time());
    ASSERT_TRUE(interaction.has_value());

    EXPECT_EQ(interaction->position(), hit.position);
    EXPECT_EQ(interaction->geometric_normal(), hit.geometric_normal);
    EXPECT_EQ(interaction->shading_normal(), shading_normal);
    EXPECT_EQ(interaction->uv(), uv);
    EXPECT_EQ(interaction->dpdu(), dpdu);
    EXPECT_EQ(interaction->dpdv(), dpdv);
    EXPECT_FLOAT_EQ(interaction->time(), 1.25F);

    constexpr auto mirrored_dpdu = Vector3{.x = -2.0F};
    const auto mirrored =
        SurfaceInteraction::create(hit.position, hit.geometric_normal, shading_normal, uv,
                                   mirrored_dpdu, dpdv, SyntheticIdentifiers, ray->time());
    ASSERT_TRUE(mirrored.has_value());
    EXPECT_EQ(mirrored->dpdu(), mirrored_dpdu);
    EXPECT_EQ(mirrored->dpdv(), dpdv);
}

TEST(SurfaceInteractionTest,
     PreservesUnboundedUvDegenerateDifferentialsTimesAndEveryIdentifierValue) {
    constexpr auto extreme_identifiers = SurfaceIdentifiers{
        .instance = {.value = 0},
        .geometry = {.value = std::numeric_limits<std::uint32_t>::max()},
        .primitive = {.value = 0},
        .material = {.value = std::numeric_limits<std::uint32_t>::max()},
    };
    constexpr auto unbounded_uv = Point2{.x = -3.0F, .y = 8.0F};
    constexpr auto zero_derivative = Vector3{};
    constexpr auto collinear_derivative = Vector3{.x = 2.0F};

    const auto singular =
        SurfaceInteraction::create(Point3{}, Normal3{.z = 1.0F}, Normal3{.z = 1.0F}, unbounded_uv,
                                   zero_derivative, zero_derivative, extreme_identifiers, -2.0F);
    ASSERT_TRUE(singular.has_value());
    EXPECT_EQ(singular->uv(), unbounded_uv);
    EXPECT_EQ(singular->dpdu(), zero_derivative);
    EXPECT_EQ(singular->dpdv(), zero_derivative);
    EXPECT_EQ(singular->identifiers(), extreme_identifiers);
    EXPECT_FLOAT_EQ(singular->time(), -2.0F);

    const auto collinear = SurfaceInteraction::create(
        Point3{}, Normal3{.z = 1.0F}, Normal3{.z = 1.0F}, Point2{}, Vector3{.x = 1.0F},
        collinear_derivative, SyntheticIdentifiers, 0.0F);
    ASSERT_TRUE(collinear.has_value());
    EXPECT_EQ(collinear->dpdu(), (Vector3{.x = 1.0F}));
    EXPECT_EQ(collinear->dpdv(), collinear_derivative);
}

TEST(SurfaceInteractionTest, RejectsInvalidValuesWithoutRepairingThem) {
    const auto expect_invalid = [](const auto& result) {
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    };
    const auto infinity = std::numeric_limits<TransportScalar>::infinity();
    const auto nan = std::numeric_limits<TransportScalar>::quiet_NaN();
    constexpr auto position = Point3{.x = 1.0F, .y = 2.0F, .z = 3.0F};
    constexpr auto normal = Normal3{.z = 1.0F};
    constexpr auto uv = Point2{.x = 0.25F, .y = 0.75F};
    constexpr auto dpdu = Vector3{.x = 1.0F};
    constexpr auto dpdv = Vector3{.y = 1.0F};

    const auto nearby_unit = Normal3{.z = std::nextafter(1.0F, 2.0F)};
    const auto preserved = SurfaceInteraction::create(position, normal, nearby_unit, uv, dpdu, dpdv,
                                                      SyntheticIdentifiers, 0.0F);
    ASSERT_TRUE(preserved.has_value());
    EXPECT_EQ(preserved->shading_normal(), nearby_unit);

    expect_invalid(SurfaceInteraction::create(Point3{.x = infinity}, normal, normal, uv, dpdu, dpdv,
                                              SyntheticIdentifiers, 0.0F));
    expect_invalid(SurfaceInteraction::create(position, Normal3{}, normal, uv, dpdu, dpdv,
                                              SyntheticIdentifiers, 0.0F));
    expect_invalid(SurfaceInteraction::create(position, Normal3{.z = 2.0F}, normal, uv, dpdu, dpdv,
                                              SyntheticIdentifiers, 0.0F));
    expect_invalid(SurfaceInteraction::create(position, normal, Normal3{.z = nan}, uv, dpdu, dpdv,
                                              SyntheticIdentifiers, 0.0F));
    expect_invalid(SurfaceInteraction::create(position, normal, Normal3{.x = 1.0F}, uv, dpdu, dpdv,
                                              SyntheticIdentifiers, 0.0F));
    expect_invalid(SurfaceInteraction::create(position, normal, Normal3{.z = -1.0F}, uv, dpdu, dpdv,
                                              SyntheticIdentifiers, 0.0F));
    expect_invalid(SurfaceInteraction::create(position, normal, normal, Point2{.x = nan}, dpdu,
                                              dpdv, SyntheticIdentifiers, 0.0F));
    expect_invalid(SurfaceInteraction::create(position, normal, normal, uv, Vector3{.x = infinity},
                                              dpdv, SyntheticIdentifiers, 0.0F));
    expect_invalid(SurfaceInteraction::create(position, normal, normal, uv, dpdu, Vector3{.y = nan},
                                              SyntheticIdentifiers, 0.0F));
    expect_invalid(SurfaceInteraction::create(position, normal, normal, uv,
                                              Vector3{.x = 1.0F, .z = 1.0F}, dpdv,
                                              SyntheticIdentifiers, 0.0F));
    expect_invalid(SurfaceInteraction::create(position, normal, normal, uv, dpdu,
                                              Vector3{.y = 1.0F, .z = -1.0F}, SyntheticIdentifiers,
                                              0.0F));
    expect_invalid(SurfaceInteraction::create(position, normal, normal, uv, dpdu, dpdv,
                                              SyntheticIdentifiers, infinity));
}

TEST(SurfaceInteractionTest, SupportsReferencePrecision) {
    const auto triangle = ReferenceTriangle::create(ReferencePoint3{.x = 1.0, .y = 2.0, .z = 3.0},
                                                    ReferencePoint3{.x = 5.0, .y = 2.0, .z = 3.0},
                                                    ReferencePoint3{.x = 1.0, .y = 6.0, .z = 3.0});
    const auto ray = make_surface_ray(ReferencePoint3{.x = 2.0, .y = 3.0, .z = 7.0},
                                      ReferenceVector3{.z = -2.0}, -3.5);
    ASSERT_TRUE(triangle.has_value());
    ASSERT_TRUE(ray.has_value());

    const auto triangle_result = triangle->intersect(*ray);
    ASSERT_TRUE(triangle_result.has_value());
    ASSERT_TRUE(triangle_result->has_value());
    const auto& hit = **triangle_result;
    ASSERT_EQ(hit.position, (ReferencePoint3{.x = 2.0, .y = 3.0, .z = 3.0}));
    ASSERT_EQ(hit.geometric_normal, (ReferenceNormal3{.z = 1.0}));

    constexpr auto shading_normal = ReferenceNormal3{.y = 0.6, .z = 0.8};
    constexpr auto uv = ReferencePoint2{.x = 0.25, .y = 0.25};
    constexpr auto dpdu = ReferenceVector3{.x = 4.0};
    constexpr auto dpdv = ReferenceVector3{.y = 4.0};

    const auto interaction =
        ReferenceSurfaceInteraction::create(hit.position, hit.geometric_normal, shading_normal, uv,
                                            dpdu, dpdv, SyntheticIdentifiers, ray->time());
    ASSERT_TRUE(interaction.has_value());
    EXPECT_EQ(interaction->position(), hit.position);
    EXPECT_EQ(interaction->geometric_normal(), hit.geometric_normal);
    EXPECT_EQ(interaction->shading_normal(), shading_normal);
    EXPECT_EQ(interaction->uv(), uv);
    EXPECT_EQ(interaction->dpdu(), dpdu);
    EXPECT_EQ(interaction->dpdv(), dpdv);
    EXPECT_EQ(interaction->identifiers(), SyntheticIdentifiers);
    EXPECT_DOUBLE_EQ(interaction->time(), -3.5);
}

} // namespace
} // namespace blackframe::renderer
