#include <Blackframe/Renderer/AreaLights.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
inline constexpr auto AreaTolerance =
    std::same_as<Scalar, TransportScalar> ? ReferenceScalar{2.0e-5} : ReferenceScalar{2.0e-12};

template <typename Value> [[nodiscard]] Value require_value(core::Result<Value> result) {
    if (!result) {
        throw std::runtime_error{result.error().message};
    }
    return std::move(result).value();
}

template <typename Result> void expect_invalid(const Result& result) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
}

template <SpectrumScalar Scalar> [[nodiscard]] SampledWavelengthsT<Scalar> wavelengths() {
    return require_value(sample_uniform_visible_wavelengths(Scalar{0.25}));
}

template <SpectrumScalar Scalar>
[[nodiscard]] LightSpectrumT<Scalar> spectrum(const Scalar value = Scalar{2}) {
    auto result = LightSpectrumT<Scalar>{};
    result.values.fill(value);
    return result;
}

template <SpectrumScalar Scalar>
[[nodiscard]] LightSampleContextT<Scalar> context(const Point3T<Scalar> position) {
    return require_value(LightSampleContextT<Scalar>::create(position, Scalar{0.5}));
}

template <SpectrumScalar Scalar> [[nodiscard]] Bounds3T<Scalar> scene_bounds() {
    return require_value(Bounds3T<Scalar>::from_minimum_maximum(
        Point3T<Scalar>{.x = Scalar{-10}, .y = Scalar{-10}, .z = Scalar{-10}},
        Point3T<Scalar>{.x = Scalar{10}, .y = Scalar{10}, .z = Scalar{10}}));
}

template <SpectrumScalar Scalar> [[nodiscard]] RayT<Scalar> escaped_ray() {
    return require_value(RayT<Scalar>::create(Point3T<Scalar>{}, Vector3T<Scalar>{.z = Scalar{1}},
                                              Scalar{0}, std::numeric_limits<Scalar>::infinity(),
                                              Scalar{0.5}, AllRayVisibility, VacuumMedium));
}

template <SpectrumScalar Scalar>
void expect_near(const Scalar actual, const ReferenceScalar expected,
                 const ReferenceScalar scale = ReferenceScalar{1}) {
    EXPECT_NEAR(static_cast<ReferenceScalar>(actual), expected,
                AreaTolerance<Scalar> * std::max(scale, ReferenceScalar{1}));
}

template <SpectrumScalar Scalar>
void expect_power(const LightSpectrumT<Scalar>& actual, const ReferenceScalar expected) {
    for (const auto value : actual.values) {
        expect_near(value, expected, std::abs(expected));
    }
}

template <SpectrumScalar Scalar> [[nodiscard]] std::vector<Point3T<Scalar>> mesh_positions() {
    return {
        {.x = Scalar{0}, .y = Scalar{0}, .z = Scalar{0}},
        {.x = Scalar{2}, .y = Scalar{0}, .z = Scalar{0}},
        {.x = Scalar{0}, .y = Scalar{1}, .z = Scalar{0}},
        {.x = Scalar{3}, .y = Scalar{0}, .z = Scalar{0}},
        {.x = Scalar{6}, .y = Scalar{0}, .z = Scalar{0}},
        {.x = Scalar{3}, .y = Scalar{2}, .z = Scalar{0}},
    };
}

[[nodiscard]] std::vector<AreaLightTriangleIndices> mesh_triangles() {
    return {
        {.vertex0 = 0U, .vertex1 = 1U, .vertex2 = 2U},
        {.vertex0 = 3U, .vertex1 = 4U, .vertex2 = 5U},
    };
}

TEST(AreaLightsContractTest, ExposesFourShapesInTransportAndReferencePrecision) {
    static_assert(LightModelFor<RectangleAreaLight, TransportScalar>);
    static_assert(LightModelFor<ReferenceRectangleAreaLight, ReferenceScalar>);
    static_assert(LightModelFor<DiskAreaLight, TransportScalar>);
    static_assert(LightModelFor<ReferenceDiskAreaLight, ReferenceScalar>);
    static_assert(LightModelFor<SphereAreaLight, TransportScalar>);
    static_assert(LightModelFor<ReferenceSphereAreaLight, ReferenceScalar>);
    static_assert(LightModelFor<MeshAreaLight, TransportScalar>);
    static_assert(LightModelFor<ReferenceMeshAreaLight, ReferenceScalar>);
    static_assert(!LightModelFor<RectangleAreaLight, ReferenceScalar>);
    static_assert(!LightModelFor<ReferenceMeshAreaLight, TransportScalar>);
    static_assert(sizeof(AreaLightTriangleIndices) == 3U * sizeof(std::uint32_t));
}

template <SpectrumScalar Scalar> void check_rectangle() {
    const auto packet = wavelengths<Scalar>();
    const auto radiance = spectrum<Scalar>();
    const auto error = Vector3T<Scalar>{.x = Scalar{0.01}, .y = Scalar{0.02}, .z = Scalar{0.03}};
    const auto one_sided = RectangleAreaLightT<Scalar>::create(
        Point3T<Scalar>{}, Normal3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.x = Scalar{1}},
        Scalar{1}, Scalar{1}, error, AreaLightSidedness::one_sided, packet, radiance);
    const auto two_sided = RectangleAreaLightT<Scalar>::create(
        Point3T<Scalar>{}, Normal3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.x = Scalar{1}},
        Scalar{1}, Scalar{1}, error, AreaLightSidedness::two_sided, packet, radiance);
    ASSERT_TRUE(one_sided.has_value()) << one_sided.error().message;
    ASSERT_TRUE(two_sided.has_value()) << two_sided.error().message;
    EXPECT_EQ(one_sided->surface_area(), Scalar{4});

    const auto front = context<Scalar>(Point3T<Scalar>{.z = Scalar{2}});
    const auto sampled =
        one_sided->sample_li(front, Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}}, packet);
    ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
    ASSERT_TRUE(sampled->has_value());
    EXPECT_EQ((**sampled).endpoint().kind(), LightEndpointKind::finite_surface);
    EXPECT_EQ(*(**sampled).endpoint().position(), Point3T<Scalar>{});
    EXPECT_EQ(*(**sampled).endpoint().geometric_normal(), (Normal3T<Scalar>{.z = Scalar{1}}));
    EXPECT_EQ((**sampled).direction_to_light(), (Vector3T<Scalar>{.z = Scalar{-1}}));
    EXPECT_EQ((**sampled).distance(), Scalar{2});
    EXPECT_EQ((**sampled).probability().measure, ProbabilityMeasure::solid_angle);
    expect_near((**sampled).probability().value, 1.0);
    EXPECT_EQ((**sampled).incident_radiance(), radiance);

    const auto off_center =
        one_sided->sample_li(front, Point2T<Scalar>{.x = Scalar{0.75}, .y = Scalar{0.25}}, packet);
    ASSERT_TRUE(off_center.has_value()) << off_center.error().message;
    ASSERT_TRUE(off_center->has_value());
    EXPECT_EQ(*(**off_center).endpoint().position(),
              (Point3T<Scalar>{.x = Scalar{0.5}, .y = Scalar{-0.5}}));

    const auto queried = one_sided->pdf_li(front, Vector3T<Scalar>{.z = Scalar{-1}}, packet);
    ASSERT_TRUE(queried.has_value()) << queried.error().message;
    expect_near(queried->value(), 1.0);
    const auto miss = one_sided->pdf_li(front, Vector3T<Scalar>{.x = Scalar{1}}, packet);
    ASSERT_TRUE(miss.has_value()) << miss.error().message;
    EXPECT_EQ(miss->value(), Scalar{0});
    const auto inverse_sqrt_two = Scalar{1} / std::sqrt(Scalar{2});
    const auto clipped = one_sided->pdf_li(
        front, Vector3T<Scalar>{.x = inverse_sqrt_two, .z = -inverse_sqrt_two}, packet);
    ASSERT_TRUE(clipped.has_value()) << clipped.error().message;
    EXPECT_EQ(clipped->value(), Scalar{0});

    const auto back = context<Scalar>(Point3T<Scalar>{.z = Scalar{-2}});
    const auto rejected =
        one_sided->sample_li(back, Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}}, packet);
    ASSERT_TRUE(rejected.has_value()) << rejected.error().message;
    EXPECT_FALSE(rejected->has_value());
    const auto back_pdf = one_sided->pdf_li(back, Vector3T<Scalar>{.z = Scalar{1}}, packet);
    ASSERT_TRUE(back_pdf.has_value()) << back_pdf.error().message;
    EXPECT_EQ(back_pdf->value(), Scalar{0});
    const auto accepted =
        two_sided->sample_li(back, Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}}, packet);
    ASSERT_TRUE(accepted.has_value()) << accepted.error().message;
    ASSERT_TRUE(accepted->has_value());
    expect_near((**accepted).probability().value, 1.0);

    const auto one_power = one_sided->power(scene_bounds<Scalar>(), packet);
    const auto two_power = two_sided->power(scene_bounds<Scalar>(), packet);
    ASSERT_TRUE(one_power.has_value()) << one_power.error().message;
    ASSERT_TRUE(two_power.has_value()) << two_power.error().message;
    expect_power(*one_power, 8.0 * std::numbers::pi_v<ReferenceScalar>);
    expect_power(*two_power, 16.0 * std::numbers::pi_v<ReferenceScalar>);

    const auto bounds = one_sided->bounds();
    expect_near(bounds.minimum().x, -1.01);
    expect_near(bounds.minimum().y, -1.02);
    expect_near(bounds.minimum().z, -0.03);
    expect_near(bounds.maximum().x, 1.01);
    expect_near(bounds.maximum().y, 1.02);
    expect_near(bounds.maximum().z, 0.03);
}

TEST(RectangleAreaLightTest, HasAnalyticPdfPowerBoundsAndSidedness) {
    check_rectangle<TransportScalar>();
    check_rectangle<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void check_disk() {
    const auto packet = wavelengths<Scalar>();
    const auto radiance = spectrum<Scalar>();
    const auto one_sided = DiskAreaLightT<Scalar>::create(
        Point3T<Scalar>{}, Normal3T<Scalar>{.z = Scalar{1}}, Scalar{1}, Vector3T<Scalar>{},
        AreaLightSidedness::one_sided, packet, radiance);
    const auto two_sided = DiskAreaLightT<Scalar>::create(
        Point3T<Scalar>{}, Normal3T<Scalar>{.z = Scalar{1}}, Scalar{1}, Vector3T<Scalar>{},
        AreaLightSidedness::two_sided, packet, radiance);
    ASSERT_TRUE(one_sided.has_value()) << one_sided.error().message;
    ASSERT_TRUE(two_sided.has_value()) << two_sided.error().message;
    expect_near(one_sided->surface_area(), std::numbers::pi_v<ReferenceScalar>);

    const auto front = context<Scalar>(Point3T<Scalar>{.z = Scalar{2}});
    const auto sampled =
        one_sided->sample_li(front, Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}}, packet);
    ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
    ASSERT_TRUE(sampled->has_value());
    expect_near((**sampled).probability().value, 4.0 / std::numbers::pi_v<ReferenceScalar>);
    const auto off_center =
        one_sided->sample_li(front, Point2T<Scalar>{.x = Scalar{0.75}, .y = Scalar{0.5}}, packet);
    ASSERT_TRUE(off_center.has_value()) << off_center.error().message;
    ASSERT_TRUE(off_center->has_value());
    expect_near((*(**off_center).endpoint().position()).x, 0.5);
    expect_near((*(**off_center).endpoint().position()).y, 0.0);
    const auto queried = one_sided->pdf_li(front, Vector3T<Scalar>{.z = Scalar{-1}}, packet);
    ASSERT_TRUE(queried.has_value()) << queried.error().message;
    expect_near(queried->value(), 4.0 / std::numbers::pi_v<ReferenceScalar>);
    const auto outside_length = std::hypot(Scalar{1.1}, Scalar{2});
    const auto clipped = one_sided->pdf_li(
        front,
        Vector3T<Scalar>{.x = Scalar{1.1} / outside_length, .z = Scalar{-2} / outside_length},
        packet);
    ASSERT_TRUE(clipped.has_value()) << clipped.error().message;
    EXPECT_EQ(clipped->value(), Scalar{0});

    const auto back = context<Scalar>(Point3T<Scalar>{.z = Scalar{-2}});
    const auto rejected = one_sided->pdf_li(back, Vector3T<Scalar>{.z = Scalar{1}}, packet);
    const auto accepted = two_sided->pdf_li(back, Vector3T<Scalar>{.z = Scalar{1}}, packet);
    ASSERT_TRUE(rejected.has_value()) << rejected.error().message;
    ASSERT_TRUE(accepted.has_value()) << accepted.error().message;
    EXPECT_EQ(rejected->value(), Scalar{0});
    expect_near(accepted->value(), 4.0 / std::numbers::pi_v<ReferenceScalar>);

    const auto one_power = one_sided->power(scene_bounds<Scalar>(), packet);
    const auto two_power = two_sided->power(scene_bounds<Scalar>(), packet);
    ASSERT_TRUE(one_power.has_value()) << one_power.error().message;
    ASSERT_TRUE(two_power.has_value()) << two_power.error().message;
    const auto pi = std::numbers::pi_v<ReferenceScalar>;
    expect_power(*one_power, 2.0 * pi * pi);
    expect_power(*two_power, 4.0 * pi * pi);
}

TEST(DiskAreaLightTest, UsesUniformDiskAreaAndBothSidednessModes) {
    check_disk<TransportScalar>();
    check_disk<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void check_sphere() {
    const auto packet = wavelengths<Scalar>();
    const auto radiance = spectrum<Scalar>();
    const auto one_sided =
        SphereAreaLightT<Scalar>::create(Point3T<Scalar>{}, Scalar{1}, Vector3T<Scalar>{},
                                         AreaLightSidedness::one_sided, packet, radiance);
    const auto two_sided =
        SphereAreaLightT<Scalar>::create(Point3T<Scalar>{}, Scalar{1}, Vector3T<Scalar>{},
                                         AreaLightSidedness::two_sided, packet, radiance);
    ASSERT_TRUE(one_sided.has_value()) << one_sided.error().message;
    ASSERT_TRUE(two_sided.has_value()) << two_sided.error().message;
    const auto pi = std::numbers::pi_v<ReferenceScalar>;
    expect_near(one_sided->surface_area(), 4.0 * pi);

    const auto outside = context<Scalar>(Point3T<Scalar>{.z = Scalar{3}});
    const auto north = one_sided->sample_li(outside, Point2T<Scalar>{}, packet);
    ASSERT_TRUE(north.has_value()) << north.error().message;
    ASSERT_TRUE(north->has_value());
    EXPECT_EQ(*(**north).endpoint().position(), (Point3T<Scalar>{.z = Scalar{1}}));
    EXPECT_EQ(*(**north).endpoint().geometric_normal(), (Normal3T<Scalar>{.z = Scalar{1}}));
    expect_near((**north).probability().value, 1.0 / pi);
    const auto queried = one_sided->pdf_li(outside, Vector3T<Scalar>{.z = Scalar{-1}}, packet);
    ASSERT_TRUE(queried.has_value()) << queried.error().message;
    expect_near(queried->value(), 1.0 / pi);
    const auto two_sided_query =
        two_sided->pdf_li(outside, Vector3T<Scalar>{.z = Scalar{-1}}, packet);
    ASSERT_TRUE(two_sided_query.has_value()) << two_sided_query.error().message;
    expect_near(two_sided_query->value(), 1.0 / pi);
    const auto two_sided_north = two_sided->sample_li(outside, Point2T<Scalar>{}, packet);
    ASSERT_TRUE(two_sided_north.has_value()) << two_sided_north.error().message;
    ASSERT_TRUE(two_sided_north->has_value());
    expect_near((**two_sided_north).probability().value, 1.0 / pi);
    const auto two_sided_south = two_sided->sample_li(
        outside, Point2T<Scalar>{.x = std::nextafter(Scalar{1}, Scalar{0})}, packet);
    ASSERT_TRUE(two_sided_south.has_value()) << two_sided_south.error().message;
    ASSERT_TRUE(two_sided_south->has_value());
    expect_near((**two_sided_south).probability().value, 4.0 / pi);

    const auto inside = context<Scalar>(Point3T<Scalar>{});
    const auto rejected = one_sided->sample_li(inside, Point2T<Scalar>{}, packet);
    ASSERT_TRUE(rejected.has_value()) << rejected.error().message;
    EXPECT_FALSE(rejected->has_value());
    const auto one_inside_pdf = one_sided->pdf_li(inside, Vector3T<Scalar>{.z = Scalar{1}}, packet);
    const auto two_inside_pdf = two_sided->pdf_li(inside, Vector3T<Scalar>{.z = Scalar{1}}, packet);
    ASSERT_TRUE(one_inside_pdf.has_value()) << one_inside_pdf.error().message;
    ASSERT_TRUE(two_inside_pdf.has_value()) << two_inside_pdf.error().message;
    EXPECT_EQ(one_inside_pdf->value(), Scalar{0});
    expect_near(two_inside_pdf->value(), 1.0 / (4.0 * pi));

    const auto one_power = one_sided->power(scene_bounds<Scalar>(), packet);
    const auto two_power = two_sided->power(scene_bounds<Scalar>(), packet);
    ASSERT_TRUE(one_power.has_value()) << one_power.error().message;
    ASSERT_TRUE(two_power.has_value()) << two_power.error().message;
    expect_power(*one_power, 8.0 * pi * pi);
    expect_power(*two_power, 16.0 * pi * pi);
}

TEST(SphereAreaLightTest, UsesOutwardNormalsAndRejectsInteriorForOneSidedEmission) {
    check_sphere<TransportScalar>();
    check_sphere<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void check_mesh() {
    const auto packet = wavelengths<Scalar>();
    const auto radiance = spectrum<Scalar>();
    const auto error = Vector3T<Scalar>{.x = Scalar{0.01}, .y = Scalar{0.02}, .z = Scalar{0.03}};
    const auto one_sided =
        MeshAreaLightT<Scalar>::create(mesh_positions<Scalar>(), mesh_triangles(), error,
                                       AreaLightSidedness::one_sided, packet, radiance);
    const auto two_sided =
        MeshAreaLightT<Scalar>::create(mesh_positions<Scalar>(), mesh_triangles(), error,
                                       AreaLightSidedness::two_sided, packet, radiance);
    ASSERT_TRUE(one_sided.has_value()) << one_sided.error().message;
    ASSERT_TRUE(two_sided.has_value()) << two_sided.error().message;
    EXPECT_EQ(one_sided->positions().size(), 6U);
    EXPECT_EQ(one_sided->triangles().size(), 2U);
    expect_near(one_sided->surface_area(), 4.0);

    const auto above =
        context<Scalar>(Point3T<Scalar>{.x = Scalar{0.25}, .y = Scalar{0.25}, .z = Scalar{2}});
    const auto first =
        one_sided->sample_li(above, Point2T<Scalar>{.x = Scalar{0.125}, .y = Scalar{0.5}}, packet);
    const auto second =
        one_sided->sample_li(above, Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}}, packet);
    ASSERT_TRUE(first.has_value()) << first.error().message;
    ASSERT_TRUE(second.has_value()) << second.error().message;
    ASSERT_TRUE(first->has_value());
    ASSERT_TRUE(second->has_value());
    EXPECT_LT((*(**first).endpoint().position()).x, Scalar{3});
    EXPECT_GE((*(**second).endpoint().position()).x, Scalar{3});
    const auto expected_first_pdf = convert_area_pdf_to_solid_angle(
        LightProbabilityDensityT<Scalar>{.value = Scalar{0.25},
                                         .measure = ProbabilityMeasure::area},
        above.position(), *(**first).endpoint().position(),
        *(**first).endpoint().geometric_normal());
    ASSERT_TRUE(expected_first_pdf.has_value()) << expected_first_pdf.error().message;
    expect_near((**first).probability().value,
                static_cast<ReferenceScalar>(expected_first_pdf->value));

    const auto boundary =
        one_sided->sample_li(above, Point2T<Scalar>{.x = Scalar{0.25}, .y = Scalar{0.75}}, packet);
    ASSERT_TRUE(boundary.has_value()) << boundary.error().message;
    ASSERT_TRUE(boundary->has_value());
    EXPECT_EQ(*(**boundary).endpoint().position(), (Point3T<Scalar>{.x = Scalar{3}}));

    const auto queried = one_sided->pdf_li(above, Vector3T<Scalar>{.z = Scalar{-1}}, packet);
    ASSERT_TRUE(queried.has_value()) << queried.error().message;
    expect_near(queried->value(), 1.0);
    const auto below =
        context<Scalar>(Point3T<Scalar>{.x = Scalar{0.25}, .y = Scalar{0.25}, .z = Scalar{-2}});
    const auto rejected = one_sided->pdf_li(below, Vector3T<Scalar>{.z = Scalar{1}}, packet);
    const auto accepted = two_sided->pdf_li(below, Vector3T<Scalar>{.z = Scalar{1}}, packet);
    ASSERT_TRUE(rejected.has_value()) << rejected.error().message;
    ASSERT_TRUE(accepted.has_value()) << accepted.error().message;
    EXPECT_EQ(rejected->value(), Scalar{0});
    expect_near(accepted->value(), 1.0);

    const auto one_power = one_sided->power(scene_bounds<Scalar>(), packet);
    const auto two_power = two_sided->power(scene_bounds<Scalar>(), packet);
    ASSERT_TRUE(one_power.has_value()) << one_power.error().message;
    ASSERT_TRUE(two_power.has_value()) << two_power.error().message;
    expect_power(*one_power, 8.0 * std::numbers::pi_v<ReferenceScalar>);
    expect_power(*two_power, 16.0 * std::numbers::pi_v<ReferenceScalar>);

    const auto bounds = one_sided->bounds();
    EXPECT_LE(bounds.minimum().x, Scalar{-0.01});
    EXPECT_LE(bounds.minimum().y, Scalar{-0.02});
    EXPECT_LE(bounds.minimum().z, Scalar{-0.03});
    EXPECT_GE(bounds.maximum().x, Scalar{6.01});
    EXPECT_GE(bounds.maximum().y, Scalar{2.02});
    EXPECT_GE(bounds.maximum().z, Scalar{0.03});
}

TEST(MeshAreaLightTest, SamplesCompactIndexedTrianglesByTheirArea) {
    check_mesh<TransportScalar>();
    check_mesh<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void check_one_sided_tangent_support_precedes_jacobian() {
    const auto packet = wavelengths<Scalar>();
    const auto radiance = spectrum<Scalar>();
    const auto tangent_context = context<Scalar>(Point3T<Scalar>{.x = Scalar{3}});
    const auto normal = Normal3T<Scalar>{.z = Scalar{1}};

    const auto one_sided_rectangle = RectangleAreaLightT<Scalar>::create(
        Point3T<Scalar>{}, normal, Vector3T<Scalar>{.x = Scalar{1}}, Scalar{1}, Scalar{1},
        Vector3T<Scalar>{}, AreaLightSidedness::one_sided, packet, radiance);
    const auto two_sided_rectangle = RectangleAreaLightT<Scalar>::create(
        Point3T<Scalar>{}, normal, Vector3T<Scalar>{.x = Scalar{1}}, Scalar{1}, Scalar{1},
        Vector3T<Scalar>{}, AreaLightSidedness::two_sided, packet, radiance);
    ASSERT_TRUE(one_sided_rectangle.has_value()) << one_sided_rectangle.error().message;
    ASSERT_TRUE(two_sided_rectangle.has_value()) << two_sided_rectangle.error().message;

    const auto canonical = Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}};
    const auto unsupported = one_sided_rectangle->sample_li(tangent_context, canonical, packet);
    ASSERT_TRUE(unsupported.has_value()) << unsupported.error().message;
    EXPECT_FALSE(unsupported->has_value());
    const auto singular = two_sided_rectangle->sample_li(tangent_context, canonical, packet);
    expect_invalid(singular);
    EXPECT_EQ(singular.error().message,
              "Light PDF conversion Jacobian is singular or numerically ambiguous.");

    const auto one_sided_mesh = MeshAreaLightT<Scalar>::create(
        mesh_positions<Scalar>(), mesh_triangles(), Vector3T<Scalar>{},
        AreaLightSidedness::one_sided, packet, radiance);
    const auto two_sided_mesh = MeshAreaLightT<Scalar>::create(
        mesh_positions<Scalar>(), mesh_triangles(), Vector3T<Scalar>{},
        AreaLightSidedness::two_sided, packet, radiance);
    ASSERT_TRUE(one_sided_mesh.has_value()) << one_sided_mesh.error().message;
    ASSERT_TRUE(two_sided_mesh.has_value()) << two_sided_mesh.error().message;

    const auto zero_pdf =
        one_sided_mesh->pdf_li_at_surface(tangent_context, Point3T<Scalar>{}, normal, packet);
    ASSERT_TRUE(zero_pdf.has_value()) << zero_pdf.error().message;
    EXPECT_EQ(zero_pdf->value(), Scalar{0});
    const auto singular_pdf =
        two_sided_mesh->pdf_li_at_surface(tangent_context, Point3T<Scalar>{}, normal, packet);
    expect_invalid(singular_pdf);
    EXPECT_EQ(singular_pdf.error().message,
              "Light PDF conversion Jacobian is singular or numerically ambiguous.");
}

TEST(AreaLightSidednessTest, RejectsUnsupportedTangentsBeforeConvertingTheirPdf) {
    check_one_sided_tangent_support_precedes_jacobian<TransportScalar>();
    check_one_sided_tangent_support_precedes_jacobian<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void check_mesh_endpoint_conditioning() {
    const auto packet = wavelengths<Scalar>();
    const auto light = MeshAreaLightT<Scalar>::create(
        {
            {.x = Scalar{0}, .y = Scalar{0}, .z = Scalar{0}},
            {.x = Scalar{1}, .y = Scalar{0}, .z = Scalar{0}},
            {.x = Scalar{0}, .y = Scalar{1}, .z = Scalar{0}},
            {.x = Scalar{0}, .y = Scalar{0}, .z = Scalar{-1}},
            {.x = Scalar{1}, .y = Scalar{0}, .z = Scalar{-1}},
            {.x = Scalar{0}, .y = Scalar{1}, .z = Scalar{-1}},
        },
        mesh_triangles(), Vector3T<Scalar>{}, AreaLightSidedness::two_sided, packet,
        spectrum<Scalar>());
    ASSERT_TRUE(light.has_value()) << light.error().message;
    const auto above = context<Scalar>(
        Point3T<Scalar>{.x = Scalar{1} / Scalar{3}, .y = Scalar{1} / Scalar{3}, .z = Scalar{2}});
    const auto queried = light->pdf_li(above, Vector3T<Scalar>{.z = Scalar{-1}}, packet);
    ASSERT_TRUE(queried.has_value()) << queried.error().message;
    expect_near(queried->value(), 4.0);

    const auto cases = std::array{
        std::pair{Scalar{2} / Scalar{9}, ReferenceScalar{4}},
        std::pair{Scalar{13} / Scalar{18}, ReferenceScalar{9}},
    };
    for (const auto [canonical_x, expected_pdf] : cases) {
        const auto sampled =
            light->sample_li(above, Point2T<Scalar>{.x = canonical_x, .y = Scalar{0.5}}, packet);
        ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
        ASSERT_TRUE(sampled->has_value());
        expect_near((**sampled).probability().value, expected_pdf);
    }
}

TEST(MeshAreaLightTest, KeepsSamplePdfBoundToEndpointAndQueriesClosestSurface) {
    check_mesh_endpoint_conditioning<TransportScalar>();
    check_mesh_endpoint_conditioning<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void check_mesh_validated_hit_endpoint() {
    const auto packet = wavelengths<Scalar>();
    const auto light = MeshAreaLightT<Scalar>::create(
        {
            {.x = Scalar{0}, .y = Scalar{0}, .z = Scalar{0}},
            {.x = Scalar{1}, .y = Scalar{0}, .z = Scalar{0}},
            {.x = Scalar{0}, .y = Scalar{1}, .z = Scalar{0}},
        },
        {{.vertex0 = 0U, .vertex1 = 1U, .vertex2 = 2U}}, Vector3T<Scalar>{},
        AreaLightSidedness::one_sided, packet, spectrum<Scalar>());
    ASSERT_TRUE(light.has_value()) << light.error().message;

    const auto above =
        context<Scalar>(Point3T<Scalar>{.x = Scalar{2}, .y = Scalar{0.25}, .z = Scalar{1}});
    const auto missed_direction = light->pdf_li(above, Vector3T<Scalar>{.z = Scalar{-1}}, packet);
    ASSERT_TRUE(missed_direction.has_value()) << missed_direction.error().message;
    EXPECT_EQ(missed_direction->value(), Scalar{0});

    const auto endpoint = Point3T<Scalar>{.x = Scalar{0.25}, .y = Scalar{0.25}};
    const auto normal = Normal3T<Scalar>{.z = Scalar{1}};
    const auto endpoint_pdf = light->pdf_li_at_surface(above, endpoint, normal, packet);
    const auto expected = convert_area_pdf_to_solid_angle(
        LightProbabilityDensityT<Scalar>{
            .value = Scalar{1} / light->surface_area(),
            .measure = ProbabilityMeasure::area,
        },
        above.position(), endpoint, normal);
    ASSERT_TRUE(endpoint_pdf.has_value()) << endpoint_pdf.error().message;
    ASSERT_TRUE(expected.has_value()) << expected.error().message;
    expect_near(endpoint_pdf->value(), static_cast<ReferenceScalar>(expected->value));

    const auto below =
        context<Scalar>(Point3T<Scalar>{.x = Scalar{0.25}, .y = Scalar{0.25}, .z = Scalar{-1}});
    const auto unsupported = light->pdf_li_at_surface(below, endpoint, normal, packet);
    ASSERT_TRUE(unsupported.has_value()) << unsupported.error().message;
    EXPECT_EQ(unsupported->value(), Scalar{0});
}

TEST(MeshAreaLightTest, EvaluatesValidatedHitEndpointWithoutRetraversal) {
    check_mesh_validated_hit_endpoint<TransportScalar>();
    check_mesh_validated_hit_endpoint<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void check_mesh_closest_backface_occlusion() {
    const auto packet = wavelengths<Scalar>();
    const auto light = MeshAreaLightT<Scalar>::create(
        {
            {.x = Scalar{0}, .y = Scalar{0}, .z = Scalar{0}},
            {.x = Scalar{1}, .y = Scalar{0}, .z = Scalar{0}},
            {.x = Scalar{0}, .y = Scalar{1}, .z = Scalar{0}},
            {.x = Scalar{0}, .y = Scalar{0}, .z = Scalar{-1}},
            {.x = Scalar{1}, .y = Scalar{0}, .z = Scalar{-1}},
            {.x = Scalar{0}, .y = Scalar{1}, .z = Scalar{-1}},
        },
        {
            {.vertex0 = 0U, .vertex1 = 2U, .vertex2 = 1U},
            {.vertex0 = 3U, .vertex1 = 4U, .vertex2 = 5U},
        },
        Vector3T<Scalar>{}, AreaLightSidedness::one_sided, packet, spectrum<Scalar>());
    ASSERT_TRUE(light.has_value()) << light.error().message;
    const auto above = context<Scalar>(
        Point3T<Scalar>{.x = Scalar{1} / Scalar{3}, .y = Scalar{1} / Scalar{3}, .z = Scalar{2}});

    const auto queried = light->pdf_li(above, Vector3T<Scalar>{.z = Scalar{-1}}, packet);
    ASSERT_TRUE(queried.has_value()) << queried.error().message;
    EXPECT_EQ(queried->value(), Scalar{0});

    const auto sampled_far = light->sample_li(
        above, Point2T<Scalar>{.x = Scalar{13} / Scalar{18}, .y = Scalar{0.5}}, packet);
    ASSERT_TRUE(sampled_far.has_value()) << sampled_far.error().message;
    ASSERT_TRUE(sampled_far->has_value());
    expect_near((**sampled_far).probability().value, 9.0);
}

TEST(MeshAreaLightTest, ClosestBackfaceOccludesFartherSupportedSurface) {
    check_mesh_closest_backface_occlusion<TransportScalar>();
    check_mesh_closest_backface_occlusion<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void check_centered_sphere_pdf_normalization() {
    const auto packet = wavelengths<Scalar>();
    const auto light = require_value(SphereAreaLightT<Scalar>::create(
        Point3T<Scalar>{}, Scalar{1}, Vector3T<Scalar>{}, AreaLightSidedness::two_sided, packet,
        spectrum<Scalar>()));
    const auto inside = context<Scalar>(Point3T<Scalar>{});
    auto accumulated = ReferenceScalar{0};
    constexpr auto resolution = std::size_t{16};
    for (auto y = std::size_t{0}; y < resolution; ++y) {
        for (auto x = std::size_t{0}; x < resolution; ++x) {
            const auto canonical = Point2T<Scalar>{
                .x = (static_cast<Scalar>(x) + Scalar{0.5}) / static_cast<Scalar>(resolution),
                .y = (static_cast<Scalar>(y) + Scalar{0.5}) / static_cast<Scalar>(resolution),
            };
            const auto direction = map_uniform_sphere(canonical);
            ASSERT_TRUE(direction.has_value()) << direction.error().message;
            const auto pdf = light.pdf_li(inside, *direction, packet);
            ASSERT_TRUE(pdf.has_value()) << pdf.error().message;
            accumulated += static_cast<ReferenceScalar>(pdf->value());
        }
    }
    const auto mean = accumulated / static_cast<ReferenceScalar>(resolution * resolution);
    const auto integral = mean * Scalar{4} * std::numbers::pi_v<ReferenceScalar>;
    EXPECT_NEAR(integral, 1.0, AreaTolerance<Scalar> * 8.0);
}

TEST(SphereAreaLightTest, DirectionalPdfIntegratesToOneAtItsCenter) {
    check_centered_sphere_pdf_normalization<TransportScalar>();
    check_centered_sphere_pdf_normalization<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void check_failures_and_black() {
    const auto packet = wavelengths<Scalar>();
    const auto radiance = spectrum<Scalar>();
    const auto rectangle = require_value(RectangleAreaLightT<Scalar>::create(
        Point3T<Scalar>{}, Normal3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.x = Scalar{1}},
        Scalar{1}, Scalar{1}, Vector3T<Scalar>{}, AreaLightSidedness::two_sided, packet, radiance));
    expect_invalid(rectangle.sample_li(context<Scalar>(Point3T<Scalar>{.z = Scalar{2}}),
                                       Point2T<Scalar>{.x = Scalar{1}}, packet));

    auto different_packet = require_value(sample_uniform_visible_wavelengths(Scalar{0.5}));
    expect_invalid(rectangle.pdf_li(context<Scalar>(Point3T<Scalar>{.z = Scalar{2}}),
                                    Vector3T<Scalar>{.z = Scalar{-1}}, different_packet));
    expect_invalid(rectangle.sample_li(context<Scalar>(Point3T<Scalar>{.z = Scalar{2}}),
                                       Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}},
                                       different_packet));
    expect_invalid(rectangle.power(scene_bounds<Scalar>(), different_packet));
    expect_invalid(rectangle.le(escaped_ray<Scalar>(), different_packet));
    expect_invalid(rectangle.pdf_li(context<Scalar>(Point3T<Scalar>{.z = Scalar{2}}),
                                    Vector3T<Scalar>{.z = Scalar{-2}}, packet));
    expect_invalid(rectangle.sample_li(context<Scalar>(Point3T<Scalar>{.x = Scalar{2}}),
                                       Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}},
                                       packet));
    expect_invalid(rectangle.power(Bounds3T<Scalar>::empty(), packet));
    expect_invalid(rectangle.power(Bounds3T<Scalar>::unbounded(), packet));
    const auto escaped_non_environment = rectangle.le(escaped_ray<Scalar>(), packet);
    ASSERT_TRUE(escaped_non_environment.has_value()) << escaped_non_environment.error().message;
    EXPECT_EQ(*escaped_non_environment, LightSpectrumT<Scalar>{});

    auto malformed = radiance;
    malformed[0] = -Scalar{1};
    expect_invalid(DiskAreaLightT<Scalar>::create(
        Point3T<Scalar>{}, Normal3T<Scalar>{.z = Scalar{1}}, Scalar{1}, Vector3T<Scalar>{},
        AreaLightSidedness::one_sided, packet, malformed));
    expect_invalid(
        SphereAreaLightT<Scalar>::create(Point3T<Scalar>{}, Scalar{0}, Vector3T<Scalar>{},
                                         AreaLightSidedness::one_sided, packet, radiance));
    const auto invalid_sidedness = static_cast<AreaLightSidedness>(255U);
    expect_invalid(RectangleAreaLightT<Scalar>::create(
        Point3T<Scalar>{}, Normal3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.x = Scalar{1}},
        Scalar{1}, Scalar{1}, Vector3T<Scalar>{}, invalid_sidedness, packet, radiance));
    expect_invalid(MeshAreaLightT<Scalar>::create({}, {}, Vector3T<Scalar>{},
                                                  AreaLightSidedness::one_sided, packet, radiance));
    expect_invalid(MeshAreaLightT<Scalar>::create(mesh_positions<Scalar>(), mesh_triangles(),
                                                  Vector3T<Scalar>{}, invalid_sidedness, packet,
                                                  radiance));
    expect_invalid(MeshAreaLightT<Scalar>::create(
        mesh_positions<Scalar>(), {{.vertex0 = 0U, .vertex1 = 1U, .vertex2 = 99U}},
        Vector3T<Scalar>{}, AreaLightSidedness::one_sided, packet, radiance));
    expect_invalid(MeshAreaLightT<Scalar>::create(
        mesh_positions<Scalar>(), {{.vertex0 = 0U, .vertex1 = 0U, .vertex2 = 1U}},
        Vector3T<Scalar>{}, AreaLightSidedness::one_sided, packet, radiance));

    const auto small_edge = std::sqrt(std::numeric_limits<Scalar>::epsilon() / Scalar{2});
    auto collapsed_positions = std::vector<Point3T<Scalar>>{
        {.x = Scalar{0}, .y = Scalar{0}, .z = Scalar{0}},
        {.x = Scalar{2}, .y = Scalar{0}, .z = Scalar{0}},
        {.x = Scalar{0}, .y = Scalar{1}, .z = Scalar{0}},
        {.x = Scalar{0}, .y = Scalar{0}, .z = Scalar{1}},
        {.x = small_edge, .y = Scalar{0}, .z = Scalar{1}},
        {.x = Scalar{0}, .y = small_edge, .z = Scalar{1}},
    };
    expect_invalid(MeshAreaLightT<Scalar>::create(std::move(collapsed_positions), mesh_triangles(),
                                                  Vector3T<Scalar>{}, AreaLightSidedness::one_sided,
                                                  packet, radiance));

    const auto enormous_radius = std::sqrt(std::numeric_limits<Scalar>::max() / Scalar{2});
    expect_invalid(
        SphereAreaLightT<Scalar>::create(Point3T<Scalar>{}, enormous_radius, Vector3T<Scalar>{},
                                         AreaLightSidedness::one_sided, packet, radiance));
    const auto tiny_radius = std::sqrt(std::numeric_limits<Scalar>::denorm_min());
    expect_invalid(
        SphereAreaLightT<Scalar>::create(Point3T<Scalar>{}, tiny_radius, Vector3T<Scalar>{},
                                         AreaLightSidedness::one_sided, packet, radiance));

    const auto black = require_value(RectangleAreaLightT<Scalar>::create(
        Point3T<Scalar>{}, Normal3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.x = Scalar{1}},
        Scalar{1}, Scalar{1}, Vector3T<Scalar>{}, AreaLightSidedness::one_sided, packet,
        LightSpectrumT<Scalar>{}));
    const auto black_sample =
        black.sample_li(context<Scalar>(Point3T<Scalar>{.z = Scalar{2}}),
                        Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}}, packet);
    ASSERT_TRUE(black_sample.has_value()) << black_sample.error().message;
    EXPECT_FALSE(black_sample->has_value());
    const auto black_pdf = black.pdf_li(context<Scalar>(Point3T<Scalar>{.z = Scalar{2}}),
                                        Vector3T<Scalar>{.z = Scalar{-1}}, packet);
    ASSERT_TRUE(black_pdf.has_value()) << black_pdf.error().message;
    EXPECT_EQ(black_pdf->value(), Scalar{0});
    const auto black_power = black.power(scene_bounds<Scalar>(), packet);
    ASSERT_TRUE(black_power.has_value()) << black_power.error().message;
    EXPECT_EQ(*black_power, LightSpectrumT<Scalar>{});
    const auto escaped = black.le(escaped_ray<Scalar>(), packet);
    ASSERT_TRUE(escaped.has_value()) << escaped.error().message;
    EXPECT_EQ(*escaped, LightSpectrumT<Scalar>{});
    expect_invalid(black.sample_li(context<Scalar>(Point3T<Scalar>{.z = Scalar{2}}),
                                   Point2T<Scalar>{.x = Scalar{1}, .y = Scalar{0.5}}, packet));
}

TEST(AreaLightsValidationTest, RejectsInvalidInputsWithoutBlackFallbacks) {
    check_failures_and_black<TransportScalar>();
    check_failures_and_black<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
