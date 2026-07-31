#include <Blackframe/Renderer/GeometryOperations.hpp>
#include <Blackframe/Renderer/ShadowRay.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <gtest/gtest.h>
#include <limits>

namespace blackframe::renderer {
namespace {

// Exercise the same visibility-segment contract in transport and reference precision.

template <SpectrumScalar Scalar>
[[nodiscard]] SurfaceInteractionT<Scalar>
make_surface(const Point3T<Scalar> position = {},
             const Normal3T<Scalar> normal = Normal3T<Scalar>{.z = Scalar{1}},
             const Scalar time = Scalar{0.5}) {
    auto dpdu = Vector3T<Scalar>{.x = Scalar{1}};
    auto dpdv = Vector3T<Scalar>{.y = Scalar{1}};
    if (normal.x != Scalar{0}) {
        dpdu = Vector3T<Scalar>{.y = Scalar{1}};
        dpdv = Vector3T<Scalar>{.z = Scalar{1}};
    } else if (normal.y != Scalar{0}) {
        dpdu = Vector3T<Scalar>{.x = Scalar{1}};
        dpdv = Vector3T<Scalar>{.z = Scalar{1}};
    }
    return SurfaceInteractionT<Scalar>::create(position, normal, normal, Point2T<Scalar>{}, dpdu,
                                               dpdv,
                                               SurfaceIdentifiers{
                                                   .instance = {.value = 1U},
                                                   .geometry = {.value = 2U},
                                                   .primitive = {.value = 3U},
                                                   .material = {.value = 4U},
                                               },
                                               time)
        .value();
}

template <SpectrumScalar Scalar> [[nodiscard]] LightSpectrumT<Scalar> unit_radiance() {
    auto radiance = LightSpectrumT<Scalar>{};
    radiance.values.fill(Scalar{1});
    return radiance;
}

template <SpectrumScalar Scalar>
[[nodiscard]] IncidentLightSampleT<Scalar>
make_point_sample(const Point3T<Scalar> source, const Point3T<Scalar> endpoint,
                  const Vector3T<Scalar> endpoint_error, const Scalar time = Scalar{0.5}) {
    const auto context = LightSampleContextT<Scalar>::create(source, time).value();
    const auto light_endpoint =
        LightSampleEndpointT<Scalar>::create_point(endpoint, endpoint_error).value();
    return IncidentLightSampleT<Scalar>::create_finite(context, light_endpoint,
                                                       unit_radiance<Scalar>(),
                                                       LightProbabilityDensityT<Scalar>{
                                                           .value = Scalar{1},
                                                           .measure = ProbabilityMeasure::discrete,
                                                       })
        .value();
}

template <SpectrumScalar Scalar>
[[nodiscard]] IncidentLightSampleT<Scalar>
make_surface_sample(const Point3T<Scalar> source, const Point3T<Scalar> endpoint,
                    const Vector3T<Scalar> endpoint_error, const Normal3T<Scalar> endpoint_normal,
                    const Scalar time = Scalar{0.5}) {
    const auto context = LightSampleContextT<Scalar>::create(source, time).value();
    const auto light_endpoint =
        LightSampleEndpointT<Scalar>::create_surface(endpoint, endpoint_error, endpoint_normal)
            .value();
    return IncidentLightSampleT<Scalar>::create_finite(
               context, light_endpoint, unit_radiance<Scalar>(),
               LightProbabilityDensityT<Scalar>{
                   .value = Scalar{0.25},
                   .measure = ProbabilityMeasure::solid_angle,
               })
        .value();
}

template <SpectrumScalar Scalar>
[[nodiscard]] IncidentLightSampleT<Scalar> make_infinite_sample(const Vector3T<Scalar> direction) {
    return IncidentLightSampleT<Scalar>::create_infinite(
               direction, unit_radiance<Scalar>(),
               LightProbabilityDensityT<Scalar>{
                   .value = Scalar{0.125},
                   .measure = ProbabilityMeasure::solid_angle,
               })
        .value();
}

template <SpectrumScalar Scalar> void expect_finite_point_segment() {
    const auto source = make_surface<Scalar>();
    const auto sample = make_point_sample<Scalar>(
        source.position(), Point3T<Scalar>{.z = Scalar{10}}, Vector3T<Scalar>{.z = Scalar{0.25}});
    constexpr auto mask = RayMask{0x12345678U};
    constexpr auto medium = MediumId{.value = 17U};

    static_assert(
        std::same_as<decltype(make_shadow_ray(source, Vector3T<Scalar>{}, sample, mask, medium)),
                     core::Result<RayT<Scalar>>>);
    const auto ray = make_shadow_ray(source, Vector3T<Scalar>{}, sample, mask, medium);
    ASSERT_TRUE(ray.has_value()) << ray.error().message;

    const auto source_z = std::nextafter(Scalar{0}, std::numeric_limits<Scalar>::infinity());
    const auto outward_error =
        std::nextafter(Scalar{0.25}, std::numeric_limits<Scalar>::infinity());
    const auto rounded_target = std::fma(Scalar{-1}, outward_error, Scalar{10});
    const auto contracted_z =
        std::nextafter(rounded_target, -std::numeric_limits<Scalar>::infinity());
    const auto segment_length = contracted_z - source_z;
    EXPECT_EQ(ray->origin(), (Point3T<Scalar>{.z = source_z}));
    EXPECT_EQ(ray->direction(), (Vector3T<Scalar>{.z = Scalar{1}}));
    EXPECT_EQ(ray->t_min(), Scalar{0});
    EXPECT_EQ(ray->t_max(), std::nextafter(segment_length, Scalar{0}));
    EXPECT_LT(ray->t_max(), segment_length);
    EXPECT_EQ(ray->time(), source.time());
    EXPECT_EQ(ray->mask(), mask);
    EXPECT_EQ(ray->current_medium(), medium);
}

template <SpectrumScalar Scalar> void expect_finite_surface_segment() {
    const auto source = make_surface<Scalar>();
    const auto sample =
        make_surface_sample<Scalar>(source.position(), Point3T<Scalar>{.z = Scalar{4}},
                                    Vector3T<Scalar>{}, Normal3T<Scalar>{.z = Scalar{-1}});
    const auto ray =
        make_shadow_ray(source, Vector3T<Scalar>{}, sample, AllRayVisibility, VacuumMedium);
    ASSERT_TRUE(ray.has_value()) << ray.error().message;

    const auto source_z = std::nextafter(Scalar{0}, std::numeric_limits<Scalar>::infinity());
    const auto target_z = std::nextafter(Scalar{4}, -std::numeric_limits<Scalar>::infinity());
    const auto segment_length = target_z - source_z;
    EXPECT_EQ(ray->origin().z, source_z);
    EXPECT_EQ(ray->direction(), (Vector3T<Scalar>{.z = Scalar{1}}));
    EXPECT_EQ(ray->t_max(), std::nextafter(segment_length, Scalar{0}));
    EXPECT_LT(ray->t_max(), target_z - ray->origin().z);
}

template <SpectrumScalar Scalar> void expect_oblique_point_error_contraction() {
    const auto source = make_surface<Scalar>(Point3T<Scalar>{}, Normal3T<Scalar>{.x = Scalar{1}});
    const auto endpoint = Point3T<Scalar>{.z = Scalar{10}};
    const auto sample =
        make_point_sample<Scalar>(source.position(), endpoint, Vector3T<Scalar>{.x = Scalar{1}});
    const auto ray = make_shadow_ray(source, Vector3T<Scalar>{.x = Scalar{0.5}}, sample,
                                     AllRayVisibility, VacuumMedium);
    ASSERT_TRUE(ray.has_value()) << ray.error().message;

    const auto nominal_displacement = endpoint - ray->origin();
    const auto nominal_length = std::sqrt(length_squared(nominal_displacement));
    const auto projected_endpoint_error = std::abs(ray->direction().x);
    EXPECT_LT(ray->direction().x, Scalar{0});
    EXPECT_GT(projected_endpoint_error, Scalar{0.04});
    EXPECT_LT(ray->t_max(), nominal_length - projected_endpoint_error);
}

template <SpectrumScalar Scalar> void expect_grazing_surface_endpoint_orientation() {
    const auto source = make_surface<Scalar>(Point3T<Scalar>{}, Normal3T<Scalar>{.x = Scalar{-1}});
    const auto sample = make_surface_sample<Scalar>(
        source.position(), Point3T<Scalar>{.z = Scalar{10}}, Vector3T<Scalar>{.x = Scalar{0.25}},
        Normal3T<Scalar>{.x = Scalar{1}});
    const auto ray = make_shadow_ray(source, Vector3T<Scalar>{.x = Scalar{0.5}}, sample,
                                     AllRayVisibility, VacuumMedium);
    ASSERT_TRUE(ray.has_value()) << ray.error().message;
    const auto last_representable_point = ray->at(ray->t_max());
    ASSERT_TRUE(last_representable_point.has_value()) << last_representable_point.error().message;

    EXPECT_LT(ray->origin().x, Scalar{0});
    EXPECT_LT(last_representable_point->x, Scalar{0});
    EXPECT_LT(last_representable_point->z, Scalar{10});
}

template <SpectrumScalar Scalar> void expect_infinite_segment() {
    const auto source = make_surface<Scalar>();
    const auto sample = make_infinite_sample<Scalar>(Vector3T<Scalar>{.x = Scalar{1}});
    const auto ray =
        make_shadow_ray(source, Vector3T<Scalar>{}, sample, RayMask{0U}, MediumId{.value = 23U});
    ASSERT_TRUE(ray.has_value()) << ray.error().message;

    EXPECT_EQ(ray->origin().z, std::nextafter(Scalar{0}, std::numeric_limits<Scalar>::infinity()));
    EXPECT_EQ(ray->direction(), sample.direction_to_light());
    EXPECT_EQ(ray->t_min(), Scalar{0});
    EXPECT_TRUE(std::isinf(ray->t_max()));
    EXPECT_EQ(ray->mask(), RayMask{0U});
    EXPECT_EQ(ray->current_medium(), (MediumId{.value = 23U}));
}

template <SpectrumScalar Scalar> void expect_strict_rejections() {
    const auto source = make_surface<Scalar>();
    const auto infinite = make_infinite_sample<Scalar>(Vector3T<Scalar>{.z = Scalar{1}});

    for (const auto invalid_error : {
             Vector3T<Scalar>{.x = Scalar{-1}},
             Vector3T<Scalar>{.y = std::numeric_limits<Scalar>::infinity()},
             Vector3T<Scalar>{.z = std::numeric_limits<Scalar>::quiet_NaN()},
         }) {
        const auto result =
            make_shadow_ray(source, invalid_error, infinite, AllRayVisibility, VacuumMedium);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
        EXPECT_FALSE(result.error().message.empty());
    }

    const auto foreign_sample = make_point_sample<Scalar>(
        Point3T<Scalar>{.x = Scalar{1}}, Point3T<Scalar>{.x = Scalar{1}, .z = Scalar{4}},
        Vector3T<Scalar>{});
    const auto mismatched =
        make_shadow_ray(source, Vector3T<Scalar>{}, foreign_sample, AllRayVisibility, VacuumMedium);
    ASSERT_FALSE(mismatched.has_value());
    EXPECT_EQ(mismatched.error().code, core::StatusCode::invalid_argument);

    const auto over_contracting = make_point_sample<Scalar>(
        source.position(), Point3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.z = Scalar{2}});
    const auto reversed = make_shadow_ray(source, Vector3T<Scalar>{}, over_contracting,
                                          AllRayVisibility, VacuumMedium);
    ASSERT_FALSE(reversed.has_value());
    EXPECT_EQ(reversed.error().code, core::StatusCode::invalid_argument);

    const auto invalid_time_source =
        make_surface<Scalar>(Point3T<Scalar>{}, Normal3T<Scalar>{.z = Scalar{1}}, Scalar{1.25});
    const auto invalid_time = make_shadow_ray(invalid_time_source, Vector3T<Scalar>{}, infinite,
                                              AllRayVisibility, VacuumMedium);
    ASSERT_FALSE(invalid_time.has_value());
    EXPECT_EQ(invalid_time.error().code, core::StatusCode::invalid_argument);

    const auto maximum_source = make_surface<Scalar>(
        Point3T<Scalar>{.x = std::numeric_limits<Scalar>::max()}, Normal3T<Scalar>{.x = Scalar{1}});
    const auto positive_x = make_infinite_sample<Scalar>(Vector3T<Scalar>{.x = Scalar{1}});
    const auto unrepresentable = make_shadow_ray(maximum_source, Vector3T<Scalar>{}, positive_x,
                                                 AllRayVisibility, VacuumMedium);
    ASSERT_FALSE(unrepresentable.has_value());
    EXPECT_EQ(unrepresentable.error().code, core::StatusCode::invalid_argument);
}

TEST(ShadowRayTest, BuildsFinitePointSegmentsInTransportAndReferencePrecision) {
    expect_finite_point_segment<TransportScalar>();
    expect_finite_point_segment<ReferenceScalar>();
}

TEST(ShadowRayTest, OffsetsBothSurfaceEndpointsWithoutAWorldScaleEpsilon) {
    expect_finite_surface_segment<TransportScalar>();
    expect_finite_surface_segment<ReferenceScalar>();
}

TEST(ShadowRayTest, ContractsPointErrorAlongThePostOffsetSegment) {
    expect_oblique_point_error_contraction<TransportScalar>();
    expect_oblique_point_error_contraction<ReferenceScalar>();
}

TEST(ShadowRayTest, OrientsAGrazingSurfaceEndpointTowardTheOffsetSource) {
    expect_grazing_surface_endpoint_orientation<TransportScalar>();
    expect_grazing_surface_endpoint_orientation<ReferenceScalar>();
}

TEST(ShadowRayTest, RoundsPointErrorSupportOutwardAcrossEveryAxis) {
    const auto direction = normalized(Vector3{.x = 1.0F, .y = 2.0F, .z = 3.0F});
    ASSERT_TRUE(direction.has_value()) << direction.error().message;
    constexpr auto errors = std::array{
        Vector3{.x = 1.0F,
                .y = std::numeric_limits<TransportScalar>::epsilon(),
                .z = std::numeric_limits<TransportScalar>::epsilon() *
                     std::numeric_limits<TransportScalar>::epsilon()},
        Vector3{.x = 1.0e10F, .y = 1.0F, .z = 1.0e-10F},
        Vector3{.x = 1.0e-10F, .y = 1.0e-20F, .z = 1.0e-30F},
        Vector3{.x = std::numeric_limits<TransportScalar>::denorm_min(),
                .y = std::numeric_limits<TransportScalar>::denorm_min(),
                .z = std::numeric_limits<TransportScalar>::denorm_min()},
    };
    for (const auto error : errors) {
        const auto contracted =
            shadow_ray_detail::contract_point_endpoint(Point3{}, error, *direction);
        ASSERT_TRUE(contracted.has_value()) << contracted.error().message;

        const auto exact_support = std::abs(static_cast<ReferenceScalar>(direction->x)) *
                                       static_cast<ReferenceScalar>(error.x) +
                                   std::abs(static_cast<ReferenceScalar>(direction->y)) *
                                       static_cast<ReferenceScalar>(error.y) +
                                   std::abs(static_cast<ReferenceScalar>(direction->z)) *
                                       static_cast<ReferenceScalar>(error.z);
        const auto represented_contraction = -static_cast<ReferenceScalar>(contracted->x) *
                                                 static_cast<ReferenceScalar>(direction->x) -
                                             static_cast<ReferenceScalar>(contracted->y) *
                                                 static_cast<ReferenceScalar>(direction->y) -
                                             static_cast<ReferenceScalar>(contracted->z) *
                                                 static_cast<ReferenceScalar>(direction->z);
        EXPECT_GE(represented_contraction, exact_support);
    }
}

TEST(ShadowRayTest, BuildsInfiniteSegmentsAndPreservesExplicitRayState) {
    expect_infinite_segment<TransportScalar>();
    expect_infinite_segment<ReferenceScalar>();
}

TEST(ShadowRayTest, RejectsInvalidOrIncoherentSegmentsWithoutFallback) {
    expect_strict_rejections<TransportScalar>();
    expect_strict_rejections<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
