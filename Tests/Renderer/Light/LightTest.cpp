#include <Blackframe/Renderer/Light.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <string_view>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar> using ProbabilityFor = LightProbabilityDensityT<Scalar>;

template <SpectrumScalar Scalar> using DirectionalPdfFor = DirectionalLightPdfT<Scalar>;

template <SpectrumScalar Scalar> using SpectrumFor = LightSpectrumT<Scalar>;

template <SpectrumScalar Scalar>
inline constexpr auto AnalyticTolerance =
    std::same_as<Scalar, TransportScalar> ? ReferenceScalar{3.0e-6} : ReferenceScalar{3.0e-13};

template <typename Result>
void expect_invalid(const Result& result, const std::string_view expected_message) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(result.error().message, expected_message);
}

template <SpectrumScalar Scalar> [[nodiscard]] SampledWavelengthsT<Scalar> test_wavelengths() {
    const auto wavelengths = sample_uniform_visible_wavelengths(Scalar{0.25});
    EXPECT_TRUE(wavelengths.has_value());
    return *wavelengths;
}

template <SpectrumScalar Scalar> [[nodiscard]] RayT<Scalar> test_ray() {
    const auto ray = RayT<Scalar>::create(
        Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{2}, .z = Scalar{3}},
        Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0}, std::numeric_limits<Scalar>::infinity(),
        Scalar{0.5}, AllRayVisibility, VacuumMedium);
    EXPECT_TRUE(ray.has_value());
    return *ray;
}

template <SpectrumScalar Scalar> [[nodiscard]] Bounds3T<Scalar> test_scene_bounds() {
    const auto bounds = Bounds3T<Scalar>::from_minimum_maximum(
        Point3T<Scalar>{.x = Scalar{-2}, .y = Scalar{-3}, .z = Scalar{-4}},
        Point3T<Scalar>{.x = Scalar{5}, .y = Scalar{6}, .z = Scalar{7}});
    EXPECT_TRUE(bounds.has_value());
    return *bounds;
}

template <SpectrumScalar Scalar> class ContinuousContractLight final {
  public:
    [[nodiscard]] core::Result<std::optional<IncidentLightSampleT<Scalar>>>
    sample_li(const LightSampleContextT<Scalar>& context, const Point2T<Scalar> canonical_sample,
              const SampledWavelengthsT<Scalar>&) const {
        if (canonical_sample == Point2T<Scalar>{}) {
            return std::optional<IncidentLightSampleT<Scalar>>{};
        }

        constexpr auto direction = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
        const auto endpoint = LightSampleEndpointT<Scalar>::create_surface(
            context.position() + direction * Scalar{5}, Vector3T<Scalar>{},
            Normal3T<Scalar>{.x = Scalar{-0.6}, .z = Scalar{-0.8}});
        if (!endpoint) {
            return std::unexpected(endpoint.error());
        }
        const auto sample = IncidentLightSampleT<Scalar>::create_finite(
            context, *endpoint,
            SpectrumFor<Scalar>{.values = {Scalar{1}, Scalar{2}, Scalar{3}, Scalar{4}}},
            ProbabilityFor<Scalar>{
                .value = Scalar{0.125},
                .measure = ProbabilityMeasure::solid_angle,
            });
        if (!sample) {
            return std::unexpected(sample.error());
        }
        return std::optional<IncidentLightSampleT<Scalar>>{*sample};
    }

    [[nodiscard]] core::Result<DirectionalPdfFor<Scalar>>
    pdf_li(const LightSampleContextT<Scalar>&, const Vector3T<Scalar>,
           const SampledWavelengthsT<Scalar>&) const {
        return DirectionalPdfFor<Scalar>::create(Scalar{0.125});
    }

    [[nodiscard]] core::Result<SpectrumFor<Scalar>> le(const RayT<Scalar>&,
                                                       const SampledWavelengthsT<Scalar>&) const {
        return SpectrumFor<Scalar>{
            .values = {Scalar{5}, Scalar{6}, Scalar{7}, Scalar{8}},
        };
    }

    [[nodiscard]] core::Result<SpectrumFor<Scalar>>
    power(const Bounds3T<Scalar>&, const SampledWavelengthsT<Scalar>&) const {
        return SpectrumFor<Scalar>{
            .values = {Scalar{11}, Scalar{12}, Scalar{13}, Scalar{14}},
        };
    }

    [[nodiscard]] Bounds3T<Scalar> bounds() const {
        const auto result = Bounds3T<Scalar>::from_minimum_maximum(
            Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{-2}, .z = Scalar{-3}},
            Point3T<Scalar>{.x = Scalar{4}, .y = Scalar{5}, .z = Scalar{6}});
        return *result;
    }
};

template <SpectrumScalar Scalar> class ErrorContractLight final {
  public:
    [[nodiscard]] core::Result<std::optional<IncidentLightSampleT<Scalar>>>
    sample_li(const LightSampleContextT<Scalar>&, const Point2T<Scalar>,
              const SampledWavelengthsT<Scalar>&) const {
        return std::unexpected(error());
    }

    [[nodiscard]] core::Result<DirectionalPdfFor<Scalar>>
    pdf_li(const LightSampleContextT<Scalar>&, const Vector3T<Scalar>,
           const SampledWavelengthsT<Scalar>&) const {
        return std::unexpected(error());
    }

    [[nodiscard]] core::Result<SpectrumFor<Scalar>> le(const RayT<Scalar>&,
                                                       const SampledWavelengthsT<Scalar>&) const {
        return std::unexpected(error());
    }

    [[nodiscard]] core::Result<SpectrumFor<Scalar>>
    power(const Bounds3T<Scalar>&, const SampledWavelengthsT<Scalar>&) const {
        return std::unexpected(error());
    }

    [[nodiscard]] static constexpr Bounds3T<Scalar> bounds() noexcept {
        return Bounds3T<Scalar>::unbounded();
    }

  private:
    [[nodiscard]] static core::Error error() {
        return core::Error{
            .code = core::StatusCode::unavailable,
            .message = "Synthetic light operation is unavailable.",
        };
    }
};

template <SpectrumScalar Scalar> class DeltaContractLight final {
  public:
    [[nodiscard]] core::Result<std::optional<IncidentLightSampleT<Scalar>>>
    sample_li(const LightSampleContextT<Scalar>&, const Point2T<Scalar>,
              const SampledWavelengthsT<Scalar>&) const {
        const auto sample = IncidentLightSampleT<Scalar>::create_infinite(
            Vector3T<Scalar>{.x = Scalar{1}},
            SpectrumFor<Scalar>{.values = {Scalar{4}, Scalar{3}, Scalar{2}, Scalar{1}}},
            ProbabilityFor<Scalar>{
                .value = Scalar{1},
                .measure = ProbabilityMeasure::discrete,
            });
        if (!sample) {
            return std::unexpected(sample.error());
        }
        return std::optional<IncidentLightSampleT<Scalar>>{*sample};
    }

    [[nodiscard]] core::Result<DirectionalPdfFor<Scalar>>
    pdf_li(const LightSampleContextT<Scalar>&, const Vector3T<Scalar>,
           const SampledWavelengthsT<Scalar>&) const {
        return DirectionalPdfFor<Scalar>::create(Scalar{0});
    }

    [[nodiscard]] core::Result<SpectrumFor<Scalar>> le(const RayT<Scalar>&,
                                                       const SampledWavelengthsT<Scalar>&) const {
        return SpectrumFor<Scalar>{};
    }

    [[nodiscard]] core::Result<SpectrumFor<Scalar>>
    power(const Bounds3T<Scalar>&, const SampledWavelengthsT<Scalar>&) const {
        return SpectrumFor<Scalar>{
            .values = {Scalar{8}, Scalar{8}, Scalar{8}, Scalar{8}},
        };
    }

    [[nodiscard]] static constexpr Bounds3T<Scalar> bounds() noexcept {
        return Bounds3T<Scalar>::unbounded();
    }
};

template <SpectrumScalar Scalar> class MissingPdfLight final {
  public:
    [[nodiscard]] core::Result<std::optional<IncidentLightSampleT<Scalar>>>
    sample_li(const LightSampleContextT<Scalar>&, const Point2T<Scalar>,
              const SampledWavelengthsT<Scalar>&) const;
    [[nodiscard]] core::Result<SpectrumFor<Scalar>> le(const RayT<Scalar>&,
                                                       const SampledWavelengthsT<Scalar>&) const;
    [[nodiscard]] core::Result<SpectrumFor<Scalar>> power(const Bounds3T<Scalar>&,
                                                          const SampledWavelengthsT<Scalar>&) const;
    [[nodiscard]] Bounds3T<Scalar> bounds() const;
};

template <typename Probability, typename Point, typename Normal>
concept AreaPdfConvertible = requires(Probability probability, Point point, Normal normal) {
    convert_area_pdf_to_solid_angle(probability, point, point, normal);
};

TEST(LightContractTest, ExposesDistinctTransportAndReferenceStaticInterfaces) {
    static_assert(LightModelFor<ContinuousContractLight<TransportScalar>, TransportScalar>);
    static_assert(LightModelFor<ContinuousContractLight<ReferenceScalar>, ReferenceScalar>);
    static_assert(LightModelFor<ErrorContractLight<TransportScalar>, TransportScalar>);
    static_assert(LightModelFor<ErrorContractLight<ReferenceScalar>, ReferenceScalar>);
    static_assert(LightModelFor<DeltaContractLight<TransportScalar>, TransportScalar>);
    static_assert(LightModelFor<DeltaContractLight<ReferenceScalar>, ReferenceScalar>);
    static_assert(!LightModelFor<MissingPdfLight<TransportScalar>, TransportScalar>);
    static_assert(!LightModelFor<MissingPdfLight<ReferenceScalar>, ReferenceScalar>);
    static_assert(!LightModelFor<ContinuousContractLight<TransportScalar>, ReferenceScalar>);
    static_assert(!LightModelFor<ContinuousContractLight<ReferenceScalar>, TransportScalar>);
    static_assert(!AreaPdfConvertible<ProbabilityDensity, ReferencePoint3, ReferenceNormal3>);
    static_assert(!AreaPdfConvertible<ReferenceProbabilityDensity, Point3, Normal3>);
    static_assert(!std::same_as<LightSampleContext, ReferenceLightSampleContext>);
    static_assert(!std::same_as<IncidentLightSample, ReferenceIncidentLightSample>);
    static_assert(!std::same_as<DirectionalLightPdf, ReferenceDirectionalLightPdf>);

    static_assert(
        std::same_as<
            decltype(std::declval<const ContinuousContractLight<TransportScalar>&>().sample_li(
                std::declval<const LightSampleContext&>(), std::declval<Point2>(),
                std::declval<const SampledWavelengths&>())),
            core::Result<std::optional<IncidentLightSample>>>);
    static_assert(
        std::same_as<decltype(std::declval<const ContinuousContractLight<ReferenceScalar>&>()
                                  .pdf_li(std::declval<const ReferenceLightSampleContext&>(),
                                          std::declval<ReferenceVector3>(),
                                          std::declval<const ReferenceSampledWavelengths&>())),
                     core::Result<ReferenceDirectionalLightPdf>>);
}

template <SpectrumScalar Scalar> void expect_contract_dispatch() {
    const auto context =
        LightSampleContextT<Scalar>::create(Point3T<Scalar>{.x = Scalar{9}}, Scalar{0.25});
    ASSERT_TRUE(context.has_value());
    const auto wavelengths = test_wavelengths<Scalar>();
    const auto ray = test_ray<Scalar>();
    const auto scene_bounds = test_scene_bounds<Scalar>();
    const auto light = ContinuousContractLight<Scalar>{};

    const auto sampled = light.sample_li(
        *context, Point2T<Scalar>{.x = Scalar{0.25}, .y = Scalar{0.75}}, wavelengths);
    const auto probability =
        light.pdf_li(*context, Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}}, wavelengths);
    const auto emitted = light.le(ray, wavelengths);
    const auto flux = light.power(scene_bounds, wavelengths);
    ASSERT_TRUE(sampled.has_value());
    ASSERT_TRUE(sampled->has_value());
    ASSERT_TRUE(probability.has_value());
    ASSERT_TRUE(emitted.has_value());
    ASSERT_TRUE(flux.has_value());

    const auto& incident = **sampled;
    EXPECT_EQ(incident.direction_to_light(),
              (Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}}));
    EXPECT_EQ(incident.distance(), Scalar{5});
    EXPECT_EQ(incident.endpoint().kind(), LightEndpointKind::finite_surface);
    ASSERT_TRUE(incident.endpoint().position().has_value());
    EXPECT_EQ(*incident.endpoint().position(), (Point3T<Scalar>{.x = Scalar{12}, .z = Scalar{4}}));
    EXPECT_EQ(incident.incident_radiance(),
              (SpectrumFor<Scalar>{.values = {Scalar{1}, Scalar{2}, Scalar{3}, Scalar{4}}}));
    EXPECT_EQ(incident.probability().value, probability->value());
    EXPECT_EQ(incident.probability().measure, probability->measure());
    EXPECT_EQ(*emitted,
              (SpectrumFor<Scalar>{.values = {Scalar{5}, Scalar{6}, Scalar{7}, Scalar{8}}}));
    EXPECT_EQ(*flux,
              (SpectrumFor<Scalar>{.values = {Scalar{11}, Scalar{12}, Scalar{13}, Scalar{14}}}));

    const auto bounds = light.bounds();
    EXPECT_FALSE(bounds.is_empty());
    EXPECT_EQ(bounds.minimum(),
              (Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{-2}, .z = Scalar{-3}}));
    EXPECT_EQ(bounds.maximum(), (Point3T<Scalar>{.x = Scalar{4}, .y = Scalar{5}, .z = Scalar{6}}));
}

TEST(LightContractTest, DispatchesAllFiveOperationsWithoutSubstitution) {
    expect_contract_dispatch<TransportScalar>();
    expect_contract_dispatch<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_empty_sample_is_explicit() {
    const auto context = LightSampleContextT<Scalar>::create(Point3T<Scalar>{}, Scalar{0});
    ASSERT_TRUE(context.has_value());
    const auto sampled = ContinuousContractLight<Scalar>{}.sample_li(*context, Point2T<Scalar>{},
                                                                     test_wavelengths<Scalar>());
    ASSERT_TRUE(sampled.has_value());
    EXPECT_FALSE(sampled->has_value());
}

TEST(LightContractTest, RepresentsPhysicalSamplingFailureWithoutFallback) {
    expect_empty_sample_is_explicit<TransportScalar>();
    expect_empty_sample_is_explicit<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_delta_measure_contract() {
    const auto context = LightSampleContextT<Scalar>::create(Point3T<Scalar>{}, Scalar{0});
    ASSERT_TRUE(context.has_value());
    const auto wavelengths = test_wavelengths<Scalar>();
    const auto light = DeltaContractLight<Scalar>{};
    const auto sampled = light.sample_li(*context, Point2T<Scalar>{}, wavelengths);
    const auto directional_pdf =
        light.pdf_li(*context, Vector3T<Scalar>{.x = Scalar{1}}, wavelengths);
    ASSERT_TRUE(sampled.has_value());
    ASSERT_TRUE(sampled->has_value());
    ASSERT_TRUE(directional_pdf.has_value());
    EXPECT_EQ((**sampled).probability().value, Scalar{1});
    EXPECT_EQ((**sampled).probability().measure, ProbabilityMeasure::discrete);
    EXPECT_TRUE(std::isinf((**sampled).distance()));
    EXPECT_EQ((**sampled).endpoint().kind(), LightEndpointKind::infinite);
    EXPECT_EQ(directional_pdf->value(), Scalar{0});
    EXPECT_FALSE(std::signbit(directional_pdf->value()));
    EXPECT_EQ(directional_pdf->measure(), ProbabilityMeasure::solid_angle);
}

TEST(LightContractTest, KeepsDeltaSamplesOutOfSolidAngleDensity) {
    expect_delta_measure_contract<TransportScalar>();
    expect_delta_measure_contract<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_error_propagation() {
    const auto context = LightSampleContextT<Scalar>::create(Point3T<Scalar>{}, Scalar{0});
    ASSERT_TRUE(context.has_value());
    const auto wavelengths = test_wavelengths<Scalar>();
    const auto ray = test_ray<Scalar>();
    const auto scene_bounds = test_scene_bounds<Scalar>();
    const auto light = ErrorContractLight<Scalar>{};

    const auto sampled = light.sample_li(*context, Point2T<Scalar>{}, wavelengths);
    const auto probability = light.pdf_li(*context, Vector3T<Scalar>{.z = Scalar{1}}, wavelengths);
    const auto emitted = light.le(ray, wavelengths);
    const auto flux = light.power(scene_bounds, wavelengths);
    ASSERT_FALSE(sampled.has_value());
    ASSERT_FALSE(probability.has_value());
    ASSERT_FALSE(emitted.has_value());
    ASSERT_FALSE(flux.has_value());
    for (const auto* const result :
         std::array{&sampled.error(), &probability.error(), &emitted.error(), &flux.error()}) {
        EXPECT_EQ(result->code, core::StatusCode::unavailable);
        EXPECT_EQ(result->message, "Synthetic light operation is unavailable.");
    }
    const auto bounds = light.bounds();
    EXPECT_FALSE(bounds.is_empty());
    EXPECT_TRUE(std::isinf(bounds.minimum().x));
    EXPECT_TRUE(std::isinf(bounds.maximum().x));
}

TEST(LightContractTest, PreservesExplicitOperationErrors) {
    expect_error_propagation<TransportScalar>();
    expect_error_propagation<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_context_validation() {
    const auto context = LightSampleContextT<Scalar>::create(
        Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{2}, .z = Scalar{3}}, Scalar{0.75});
    ASSERT_TRUE(context.has_value());
    EXPECT_EQ(context->position(),
              (Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{2}, .z = Scalar{3}}));
    EXPECT_EQ(context->time(), Scalar{0.75});

    const auto infinity = std::numeric_limits<Scalar>::infinity();
    const auto nan = std::numeric_limits<Scalar>::quiet_NaN();
    for (const auto value : std::array{nan, infinity, -infinity}) {
        expect_invalid(LightSampleContextT<Scalar>::create(Point3T<Scalar>{.x = value}, Scalar{0}),
                       "A light sample context requires a finite world-space position.");
        expect_invalid(LightSampleContextT<Scalar>::create(Point3T<Scalar>{}, value),
                       "A light sample context requires a finite time.");
    }
}

TEST(LightContractTest, ValidatesWorldSpaceContextInBothPrecisions) {
    expect_context_validation<TransportScalar>();
    expect_context_validation<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_endpoint_validation() {
    const auto position = Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{2}, .z = Scalar{3}};
    const auto position_error =
        Vector3T<Scalar>{.x = Scalar{0.01}, .y = Scalar{0.02}, .z = Scalar{0.03}};
    const auto normal = Normal3T<Scalar>{.z = Scalar{1}};

    const auto point = LightSampleEndpointT<Scalar>::create_point(position, position_error);
    ASSERT_TRUE(point.has_value());
    EXPECT_EQ(point->kind(), LightEndpointKind::finite_point);
    EXPECT_TRUE(point->is_finite());
    EXPECT_FALSE(point->is_surface());
    ASSERT_TRUE(point->position().has_value());
    ASSERT_TRUE(point->absolute_position_error().has_value());
    EXPECT_EQ(*point->position(), position);
    EXPECT_EQ(*point->absolute_position_error(), position_error);
    EXPECT_FALSE(point->geometric_normal().has_value());

    const auto surface =
        LightSampleEndpointT<Scalar>::create_surface(position, position_error, normal);
    ASSERT_TRUE(surface.has_value());
    EXPECT_EQ(surface->kind(), LightEndpointKind::finite_surface);
    EXPECT_TRUE(surface->is_finite());
    EXPECT_TRUE(surface->is_surface());
    ASSERT_TRUE(surface->geometric_normal().has_value());
    EXPECT_EQ(*surface->geometric_normal(), normal);

    const auto infinite = LightSampleEndpointT<Scalar>::infinite();
    EXPECT_EQ(infinite.kind(), LightEndpointKind::infinite);
    EXPECT_FALSE(infinite.is_finite());
    EXPECT_FALSE(infinite.is_surface());
    EXPECT_FALSE(infinite.position().has_value());
    EXPECT_FALSE(infinite.absolute_position_error().has_value());
    EXPECT_FALSE(infinite.geometric_normal().has_value());

    const auto infinity = std::numeric_limits<Scalar>::infinity();
    const auto nan = std::numeric_limits<Scalar>::quiet_NaN();
    for (const auto value : std::array{nan, infinity, -infinity}) {
        expect_invalid(LightSampleEndpointT<Scalar>::create_point(Point3T<Scalar>{.x = value},
                                                                  Vector3T<Scalar>{}),
                       "A finite light endpoint requires a finite world-space position.");
    }
    for (const auto value : std::array{Scalar{-1}, nan, infinity, -infinity}) {
        expect_invalid(
            LightSampleEndpointT<Scalar>::create_point(position, Vector3T<Scalar>{.y = value}),
            "A finite light endpoint requires finite non-negative position error.");
    }
    for (const auto malformed_normal :
         std::array{Normal3T<Scalar>{}, Normal3T<Scalar>{.x = Scalar{2}},
                    Normal3T<Scalar>{.x = nan}, Normal3T<Scalar>{.x = infinity}}) {
        expect_invalid(LightSampleEndpointT<Scalar>::create_surface(position, position_error,
                                                                    malformed_normal),
                       "A surface light endpoint requires a finite unit geometric normal.");
    }
}

TEST(LightContractTest, ValidatesFiniteAndInfiniteSampleEndpoints) {
    expect_endpoint_validation<TransportScalar>();
    expect_endpoint_validation<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_directional_pdf_validation() {
    const auto zero = DirectionalPdfFor<Scalar>::create(-Scalar{0});
    ASSERT_TRUE(zero.has_value());
    EXPECT_EQ(zero->value(), Scalar{0});
    EXPECT_FALSE(std::signbit(zero->value()));
    EXPECT_EQ(zero->measure(), ProbabilityMeasure::solid_angle);
    const auto zero_density = zero->probability_density();
    EXPECT_EQ(zero_density.value, Scalar{0});
    EXPECT_EQ(zero_density.measure, ProbabilityMeasure::solid_angle);

    const auto positive = DirectionalPdfFor<Scalar>::create(Scalar{2.5});
    ASSERT_TRUE(positive.has_value());
    EXPECT_EQ(positive->value(), Scalar{2.5});

    for (const auto invalid : std::array{Scalar{-1}, std::numeric_limits<Scalar>::quiet_NaN(),
                                         std::numeric_limits<Scalar>::infinity(),
                                         -std::numeric_limits<Scalar>::infinity()}) {
        expect_invalid(
            DirectionalPdfFor<Scalar>::create(invalid),
            "A directional light PDF requires a finite non-negative solid-angle density.");
    }
}

TEST(LightContractTest, FixesDirectionalPdfToSolidAngleMeasure) {
    expect_directional_pdf_validation<TransportScalar>();
    expect_directional_pdf_validation<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_incident_sample_validation() {
    const auto radiance =
        SpectrumFor<Scalar>{.values = {Scalar{0}, Scalar{1}, Scalar{2}, Scalar{3}}};
    const auto context = LightSampleContextT<Scalar>::create(Point3T<Scalar>{}, Scalar{0});
    ASSERT_TRUE(context.has_value());
    const auto surface_endpoint = LightSampleEndpointT<Scalar>::create_surface(
        Point3T<Scalar>{.z = Scalar{2}}, Vector3T<Scalar>{}, Normal3T<Scalar>{.z = Scalar{-1}});
    ASSERT_TRUE(surface_endpoint.has_value());

    const auto continuous =
        IncidentLightSampleT<Scalar>::create_finite(*context, *surface_endpoint, radiance,
                                                    ProbabilityFor<Scalar>{
                                                        .value = Scalar{0.25},
                                                        .measure = ProbabilityMeasure::solid_angle,
                                                    });
    ASSERT_TRUE(continuous.has_value());
    EXPECT_EQ(continuous->endpoint(), *surface_endpoint);
    EXPECT_EQ(continuous->direction_to_light(), (Vector3T<Scalar>{.z = Scalar{1}}));
    EXPECT_EQ(continuous->distance(), Scalar{2});
    EXPECT_EQ(continuous->probability().measure, ProbabilityMeasure::solid_angle);

    const auto point_endpoint = LightSampleEndpointT<Scalar>::create_point(
        Point3T<Scalar>{.x = Scalar{3}}, Vector3T<Scalar>{});
    ASSERT_TRUE(point_endpoint.has_value());
    const auto point_sample =
        IncidentLightSampleT<Scalar>::create_finite(*context, *point_endpoint, radiance,
                                                    ProbabilityFor<Scalar>{
                                                        .value = Scalar{0.5},
                                                        .measure = ProbabilityMeasure::discrete,
                                                    });
    ASSERT_TRUE(point_sample.has_value());
    EXPECT_EQ(point_sample->direction_to_light(), (Vector3T<Scalar>{.x = Scalar{1}}));
    EXPECT_EQ(point_sample->distance(), Scalar{3});

    auto hdr = SpectrumFor<Scalar>{};
    hdr.values.fill(std::numeric_limits<Scalar>::max());
    const auto delta =
        IncidentLightSampleT<Scalar>::create_infinite(Vector3T<Scalar>{.x = Scalar{1}}, hdr,
                                                      ProbabilityFor<Scalar>{
                                                          .value = Scalar{1},
                                                          .measure = ProbabilityMeasure::discrete,
                                                      });
    ASSERT_TRUE(delta.has_value());
    EXPECT_EQ(delta->endpoint().kind(), LightEndpointKind::infinite);
    EXPECT_TRUE(std::isinf(delta->distance()));
    EXPECT_EQ(delta->incident_radiance(), hdr);

    const auto environment = IncidentLightSampleT<Scalar>::create_infinite(
        Vector3T<Scalar>{.z = Scalar{1}}, radiance,
        ProbabilityFor<Scalar>{
            .value = Scalar{0.25},
            .measure = ProbabilityMeasure::solid_angle,
        });
    ASSERT_TRUE(environment.has_value());
    EXPECT_EQ(environment->probability().measure, ProbabilityMeasure::solid_angle);

    for (const auto direction : std::array{
             Vector3T<Scalar>{},
             Vector3T<Scalar>{.x = Scalar{2}},
             Vector3T<Scalar>{.x = std::numeric_limits<Scalar>::quiet_NaN()},
             Vector3T<Scalar>{.x = std::numeric_limits<Scalar>::infinity()},
         }) {
        expect_invalid(IncidentLightSampleT<Scalar>::create_infinite(
                           direction, radiance,
                           ProbabilityFor<Scalar>{
                               .value = Scalar{1},
                               .measure = ProbabilityMeasure::discrete,
                           }),
                       "An incident light sample requires a finite unit direction.");
    }

    for (const auto invalid : std::array{Scalar{-1}, std::numeric_limits<Scalar>::quiet_NaN(),
                                         std::numeric_limits<Scalar>::infinity()}) {
        auto malformed = radiance;
        malformed[2] = invalid;
        expect_invalid(IncidentLightSampleT<Scalar>::create_infinite(
                           Vector3T<Scalar>{.z = Scalar{1}}, malformed,
                           ProbabilityFor<Scalar>{
                               .value = Scalar{1},
                               .measure = ProbabilityMeasure::discrete,
                           }),
                       "Incident light radiance requires finite non-negative spectral lanes.");
    }

    for (const auto probability : std::array{
             ProbabilityFor<Scalar>{
                 .value = Scalar{0},
                 .measure = ProbabilityMeasure::solid_angle,
             },
             ProbabilityFor<Scalar>{
                 .value = Scalar{-1},
                 .measure = ProbabilityMeasure::solid_angle,
             },
             ProbabilityFor<Scalar>{
                 .value = std::numeric_limits<Scalar>::quiet_NaN(),
                 .measure = ProbabilityMeasure::solid_angle,
             },
             ProbabilityFor<Scalar>{
                 .value = std::numeric_limits<Scalar>::infinity(),
                 .measure = ProbabilityMeasure::solid_angle,
             },
             ProbabilityFor<Scalar>{
                 .value = Scalar{1},
                 .measure = ProbabilityMeasure::area,
             },
             ProbabilityFor<Scalar>{
                 .value = Scalar{1.01},
                 .measure = ProbabilityMeasure::discrete,
             },
             ProbabilityFor<Scalar>{
                 .value = Scalar{0},
                 .measure = ProbabilityMeasure::discrete,
             },
             ProbabilityFor<Scalar>{
                 .value = Scalar{-1},
                 .measure = ProbabilityMeasure::discrete,
             },
             ProbabilityFor<Scalar>{
                 .value = std::numeric_limits<Scalar>::quiet_NaN(),
                 .measure = ProbabilityMeasure::discrete,
             },
             ProbabilityFor<Scalar>{
                 .value = std::numeric_limits<Scalar>::infinity(),
                 .measure = ProbabilityMeasure::discrete,
             },
         }) {
        expect_invalid(IncidentLightSampleT<Scalar>::create_infinite(
                           Vector3T<Scalar>{.z = Scalar{1}}, radiance, probability),
                       "An incident light sample requires a finite positive solid-angle density or "
                       "a discrete probability in (0, 1].");
    }

    expect_invalid(
        IncidentLightSampleT<Scalar>::create_finite(*context, *surface_endpoint, radiance,
                                                    ProbabilityFor<Scalar>{
                                                        .value = Scalar{1},
                                                        .measure = ProbabilityMeasure::discrete,
                                                    }),
        "A finite surface light sample requires a solid-angle density.");
    expect_invalid(
        IncidentLightSampleT<Scalar>::create_finite(*context, *point_endpoint, radiance,
                                                    ProbabilityFor<Scalar>{
                                                        .value = Scalar{1},
                                                        .measure = ProbabilityMeasure::solid_angle,
                                                    }),
        "A finite point light sample requires a discrete probability.");
    expect_invalid(IncidentLightSampleT<Scalar>::create_finite(
                       *context, LightSampleEndpointT<Scalar>::infinite(), radiance,
                       ProbabilityFor<Scalar>{
                           .value = Scalar{1},
                           .measure = ProbabilityMeasure::discrete,
                       }),
                   "A finite incident light sample requires a finite endpoint.");

    const auto coincident_endpoint =
        LightSampleEndpointT<Scalar>::create_point(context->position(), Vector3T<Scalar>{});
    ASSERT_TRUE(coincident_endpoint.has_value());
    expect_invalid(
        IncidentLightSampleT<Scalar>::create_finite(*context, *coincident_endpoint, radiance,
                                                    ProbabilityFor<Scalar>{
                                                        .value = Scalar{1},
                                                        .measure = ProbabilityMeasure::discrete,
                                                    }),
        "A finite incident light sample requires distinct context and endpoint positions.");

    const auto maximum = std::numeric_limits<Scalar>::max();
    const auto negative_context =
        LightSampleContextT<Scalar>::create(Point3T<Scalar>{.x = -maximum}, Scalar{0});
    const auto positive_endpoint = LightSampleEndpointT<Scalar>::create_point(
        Point3T<Scalar>{.x = maximum}, Vector3T<Scalar>{});
    ASSERT_TRUE(negative_context.has_value());
    ASSERT_TRUE(positive_endpoint.has_value());
    expect_invalid(
        IncidentLightSampleT<Scalar>::create_finite(*negative_context, *positive_endpoint, radiance,
                                                    ProbabilityFor<Scalar>{
                                                        .value = Scalar{1},
                                                        .measure = ProbabilityMeasure::discrete,
                                                    }),
        "Finite incident light sample separation is not representable in the requested precision.");

    const auto overflowing_distance_endpoint = LightSampleEndpointT<Scalar>::create_point(
        Point3T<Scalar>{.x = maximum, .y = maximum}, Vector3T<Scalar>{});
    ASSERT_TRUE(overflowing_distance_endpoint.has_value());
    expect_invalid(
        IncidentLightSampleT<Scalar>::create_finite(*context, *overflowing_distance_endpoint,
                                                    radiance,
                                                    ProbabilityFor<Scalar>{
                                                        .value = Scalar{1},
                                                        .measure = ProbabilityMeasure::discrete,
                                                    }),
        "Finite incident light sample distance is not representable in the requested precision.");
}

TEST(LightContractTest, ValidatesContinuousAndDeltaIncidentSamples) {
    expect_incident_sample_validation<TransportScalar>();
    expect_incident_sample_validation<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_analytic_pdf_conversion() {
    const auto reference = Point3T<Scalar>{};
    const auto surface = Point3T<Scalar>{.x = Scalar{3}, .y = Scalar{0}, .z = Scalar{4}};
    const auto normal = Normal3T<Scalar>{.z = Scalar{-1}};
    const auto flipped_normal = -normal;
    const auto area = ProbabilityFor<Scalar>{
        .value = Scalar{0.04},
        .measure = ProbabilityMeasure::area,
    };

    const auto solid = convert_area_pdf_to_solid_angle(area, reference, surface, normal);
    const auto flipped = convert_area_pdf_to_solid_angle(area, reference, surface, flipped_normal);
    ASSERT_TRUE(solid.has_value());
    ASSERT_TRUE(flipped.has_value());
    EXPECT_EQ(solid->measure, ProbabilityMeasure::solid_angle);
    EXPECT_EQ(flipped->measure, ProbabilityMeasure::solid_angle);
    EXPECT_NEAR(static_cast<ReferenceScalar>(solid->value), 1.25, AnalyticTolerance<Scalar>);
    EXPECT_EQ(flipped->value, solid->value);

    const auto round_trip = convert_solid_angle_pdf_to_area(*solid, reference, surface, normal);
    ASSERT_TRUE(round_trip.has_value());
    EXPECT_EQ(round_trip->measure, ProbabilityMeasure::area);
    EXPECT_NEAR(static_cast<ReferenceScalar>(round_trip->value), 0.04, AnalyticTolerance<Scalar>);

    const auto axial = convert_area_pdf_to_solid_angle(
        ProbabilityFor<Scalar>{
            .value = Scalar{0.25},
            .measure = ProbabilityMeasure::area,
        },
        reference, Point3T<Scalar>{.z = Scalar{2}}, Normal3T<Scalar>{.z = Scalar{-1}});
    ASSERT_TRUE(axial.has_value());
    EXPECT_EQ(axial->value, Scalar{1});

    constexpr auto radius = Scalar{3};
    const auto sphere_area_density =
        Scalar{1} / (Scalar{4} * std::numbers::pi_v<Scalar> * radius * radius);
    const auto sphere = convert_area_pdf_to_solid_angle(
        ProbabilityFor<Scalar>{
            .value = sphere_area_density,
            .measure = ProbabilityMeasure::area,
        },
        reference, Point3T<Scalar>{.z = radius}, Normal3T<Scalar>{.z = Scalar{1}});
    ASSERT_TRUE(sphere.has_value());
    EXPECT_NEAR(static_cast<ReferenceScalar>(sphere->value),
                1.0 / (4.0 * std::numbers::pi_v<ReferenceScalar>), AnalyticTolerance<Scalar>);
}

TEST(LightPdfConversionTest, MatchesAnalyticJacobiansAndRoundTrips) {
    expect_analytic_pdf_conversion<TransportScalar>();
    expect_analytic_pdf_conversion<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_zero_and_invalid_pdf_conversion() {
    const auto reference = Point3T<Scalar>{};
    const auto surface = Point3T<Scalar>{.z = Scalar{2}};
    const auto normal = Normal3T<Scalar>{.z = Scalar{-1}};
    const auto signed_zero = convert_area_pdf_to_solid_angle(
        ProbabilityFor<Scalar>{
            .value = -Scalar{0},
            .measure = ProbabilityMeasure::area,
        },
        reference, surface, normal);
    ASSERT_TRUE(signed_zero.has_value());
    EXPECT_EQ(signed_zero->value, Scalar{0});
    EXPECT_FALSE(std::signbit(signed_zero->value));
    EXPECT_EQ(signed_zero->measure, ProbabilityMeasure::solid_angle);

    const auto inverse_signed_zero = convert_solid_angle_pdf_to_area(
        ProbabilityFor<Scalar>{
            .value = -Scalar{0},
            .measure = ProbabilityMeasure::solid_angle,
        },
        reference, surface, normal);
    ASSERT_TRUE(inverse_signed_zero.has_value());
    EXPECT_EQ(inverse_signed_zero->value, Scalar{0});
    EXPECT_FALSE(std::signbit(inverse_signed_zero->value));
    EXPECT_EQ(inverse_signed_zero->measure, ProbabilityMeasure::area);

    expect_invalid(convert_area_pdf_to_solid_angle(
                       ProbabilityFor<Scalar>{
                           .value = Scalar{1},
                           .measure = ProbabilityMeasure::solid_angle,
                       },
                       reference, surface, normal),
                   "Light PDF conversion source measure does not match the requested conversion.");
    expect_invalid(convert_solid_angle_pdf_to_area(
                       ProbabilityFor<Scalar>{
                           .value = Scalar{1},
                           .measure = ProbabilityMeasure::area,
                       },
                       reference, surface, normal),
                   "Light PDF conversion source measure does not match the requested conversion.");

    for (const auto invalid : std::array{Scalar{-1}, std::numeric_limits<Scalar>::quiet_NaN(),
                                         std::numeric_limits<Scalar>::infinity(),
                                         -std::numeric_limits<Scalar>::infinity()}) {
        expect_invalid(convert_area_pdf_to_solid_angle(
                           ProbabilityFor<Scalar>{
                               .value = invalid,
                               .measure = ProbabilityMeasure::area,
                           },
                           reference, surface, normal),
                       "Light PDF conversion requires a finite non-negative density.");
        expect_invalid(convert_solid_angle_pdf_to_area(
                           ProbabilityFor<Scalar>{
                               .value = invalid,
                               .measure = ProbabilityMeasure::solid_angle,
                           },
                           reference, surface, normal),
                       "Light PDF conversion requires a finite non-negative density.");
    }

    const auto infinity = std::numeric_limits<Scalar>::infinity();
    const auto nan = std::numeric_limits<Scalar>::quiet_NaN();
    for (const auto point : std::array{Point3T<Scalar>{.x = nan}, Point3T<Scalar>{.x = infinity}}) {
        expect_invalid(convert_area_pdf_to_solid_angle(
                           ProbabilityFor<Scalar>{
                               .value = Scalar{0},
                               .measure = ProbabilityMeasure::area,
                           },
                           point, surface, normal),
                       "Light PDF conversion requires finite reference and surface positions.");
        expect_invalid(convert_solid_angle_pdf_to_area(
                           ProbabilityFor<Scalar>{
                               .value = Scalar{0},
                               .measure = ProbabilityMeasure::solid_angle,
                           },
                           reference, point, normal),
                       "Light PDF conversion requires finite reference and surface positions.");
    }

    expect_invalid(convert_area_pdf_to_solid_angle(
                       ProbabilityFor<Scalar>{
                           .value = Scalar{0},
                           .measure = ProbabilityMeasure::area,
                       },
                       reference, reference, normal),
                   "Light PDF conversion requires two distinct positions.");
    expect_invalid(convert_area_pdf_to_solid_angle(
                       ProbabilityFor<Scalar>{
                           .value = Scalar{1},
                           .measure = ProbabilityMeasure::area,
                       },
                       reference, surface, Normal3T<Scalar>{}),
                   "Light PDF conversion requires a finite unit surface geometric normal.");
    expect_invalid(
        convert_area_pdf_to_solid_angle(
            ProbabilityFor<Scalar>{
                .value = Scalar{1},
                .measure = ProbabilityMeasure::area,
            },
            Point3T<Scalar>{.x = -std::numeric_limits<Scalar>::max()},
            Point3T<Scalar>{.x = std::numeric_limits<Scalar>::max()},
            Normal3T<Scalar>{.x = Scalar{-1}}),
        "Light PDF conversion separation is not representable in the requested precision.");
    expect_invalid(convert_area_pdf_to_solid_angle(
                       ProbabilityFor<Scalar>{
                           .value = Scalar{1},
                           .measure = ProbabilityMeasure::area,
                       },
                       reference, surface, Normal3T<Scalar>{.z = infinity}),
                   "Light PDF conversion requires a finite unit surface geometric normal.");

    const auto tangent_surface = Point3T<Scalar>{.x = Scalar{1}};
    const auto tangent_normal = Normal3T<Scalar>{.z = Scalar{1}};
    expect_invalid(convert_area_pdf_to_solid_angle(
                       ProbabilityFor<Scalar>{
                           .value = Scalar{1},
                           .measure = ProbabilityMeasure::area,
                       },
                       reference, tangent_surface, tangent_normal),
                   "Light PDF conversion Jacobian is singular or numerically ambiguous.");
    expect_invalid(convert_solid_angle_pdf_to_area(
                       ProbabilityFor<Scalar>{
                           .value = Scalar{0},
                           .measure = ProbabilityMeasure::solid_angle,
                       },
                       reference, tangent_surface, tangent_normal),
                   "Light PDF conversion Jacobian is singular or numerically ambiguous.");

    const auto diagonal = Scalar{1} / std::sqrt(Scalar{2});
    const auto near_tangent_normal = Normal3T<Scalar>{.x = diagonal, .z = diagonal};
    const auto last = std::nextafter(Scalar{1}, Scalar{0});
    expect_invalid(convert_area_pdf_to_solid_angle(
                       ProbabilityFor<Scalar>{
                           .value = Scalar{1},
                           .measure = ProbabilityMeasure::area,
                       },
                       reference, Point3T<Scalar>{.x = Scalar{1}, .z = -last}, near_tangent_normal),
                   "Light PDF conversion Jacobian is singular or numerically ambiguous.");

    const auto resolvable_alignment = Scalar{64} * std::numeric_limits<Scalar>::epsilon();
    const auto resolvable_normal = Normal3T<Scalar>{
        .x = resolvable_alignment,
        .z = std::sqrt(Scalar{1} - resolvable_alignment * resolvable_alignment),
    };
    const auto resolvable = convert_area_pdf_to_solid_angle(
        ProbabilityFor<Scalar>{
            .value = Scalar{1},
            .measure = ProbabilityMeasure::area,
        },
        reference, tangent_surface, resolvable_normal);
    ASSERT_TRUE(resolvable.has_value());
    EXPECT_NEAR(static_cast<ReferenceScalar>(resolvable->value * resolvable_alignment), 1.0,
                AnalyticTolerance<Scalar>);
    const auto resolvable_round_trip =
        convert_solid_angle_pdf_to_area(*resolvable, reference, tangent_surface, resolvable_normal);
    ASSERT_TRUE(resolvable_round_trip.has_value());
    EXPECT_NEAR(static_cast<ReferenceScalar>(resolvable_round_trip->value), 1.0,
                AnalyticTolerance<Scalar>);
}

TEST(LightPdfConversionTest, PreservesZeroAndRejectsInvalidDomains) {
    expect_zero_and_invalid_pdf_conversion<TransportScalar>();
    expect_zero_and_invalid_pdf_conversion<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_scaled_pdf_conversion() {
    constexpr auto exponent = std::same_as<Scalar, TransportScalar> ? 80 : 600;
    const auto far_distance = std::scalbn(Scalar{1}, exponent);
    const auto reciprocal_scale = std::scalbn(Scalar{1}, -exponent);
    const auto far = convert_area_pdf_to_solid_angle(
        ProbabilityFor<Scalar>{
            .value = reciprocal_scale,
            .measure = ProbabilityMeasure::area,
        },
        Point3T<Scalar>{}, Point3T<Scalar>{.x = far_distance}, Normal3T<Scalar>{.x = Scalar{-1}});
    ASSERT_TRUE(far.has_value());
    EXPECT_EQ(far->value, far_distance);
    const auto far_round_trip =
        convert_solid_angle_pdf_to_area(*far, Point3T<Scalar>{}, Point3T<Scalar>{.x = far_distance},
                                        Normal3T<Scalar>{.x = Scalar{-1}});
    ASSERT_TRUE(far_round_trip.has_value());
    EXPECT_EQ(far_round_trip->value, reciprocal_scale);

    const auto near_distance = reciprocal_scale;
    const auto large_density = far_distance;
    const auto near = convert_area_pdf_to_solid_angle(
        ProbabilityFor<Scalar>{
            .value = large_density,
            .measure = ProbabilityMeasure::area,
        },
        Point3T<Scalar>{}, Point3T<Scalar>{.x = near_distance}, Normal3T<Scalar>{.x = Scalar{-1}});
    ASSERT_TRUE(near.has_value());
    EXPECT_EQ(near->value, reciprocal_scale);

    const auto denormal = std::numeric_limits<Scalar>::denorm_min();
    const auto preserved_subnormal = convert_area_pdf_to_solid_angle(
        ProbabilityFor<Scalar>{
            .value = denormal,
            .measure = ProbabilityMeasure::area,
        },
        Point3T<Scalar>{}, Point3T<Scalar>{.x = Scalar{1}}, Normal3T<Scalar>{.x = Scalar{-1}});
    ASSERT_TRUE(preserved_subnormal.has_value());
    EXPECT_EQ(preserved_subnormal->value, denormal);

    const auto minimum_distance = std::numeric_limits<Scalar>::min();
    const auto maximum_density = std::numeric_limits<Scalar>::max();
    const auto minimum_distance_conversion = convert_area_pdf_to_solid_angle(
        ProbabilityFor<Scalar>{
            .value = maximum_density,
            .measure = ProbabilityMeasure::area,
        },
        Point3T<Scalar>{}, Point3T<Scalar>{.x = minimum_distance},
        Normal3T<Scalar>{.x = Scalar{-1}});
    ASSERT_TRUE(minimum_distance_conversion.has_value());
    const auto expected_minimum_distance_conversion =
        std::scalbn(maximum_density, 2 * (std::numeric_limits<Scalar>::min_exponent - 1));
    EXPECT_EQ(minimum_distance_conversion->value, expected_minimum_distance_conversion);

    const auto minimum_distance_inverse = convert_solid_angle_pdf_to_area(
        ProbabilityFor<Scalar>{
            .value = denormal,
            .measure = ProbabilityMeasure::solid_angle,
        },
        Point3T<Scalar>{}, Point3T<Scalar>{.x = minimum_distance},
        Normal3T<Scalar>{.x = Scalar{-1}});
    ASSERT_TRUE(minimum_distance_inverse.has_value());
    const auto expected_minimum_distance_inverse =
        std::scalbn(denormal, -2 * (std::numeric_limits<Scalar>::min_exponent - 1));
    EXPECT_EQ(minimum_distance_inverse->value, expected_minimum_distance_inverse);

    const auto maximum_distance = std::numeric_limits<Scalar>::max();
    const auto maximum_distance_conversion = convert_area_pdf_to_solid_angle(
        ProbabilityFor<Scalar>{
            .value = denormal,
            .measure = ProbabilityMeasure::area,
        },
        Point3T<Scalar>{}, Point3T<Scalar>{.x = maximum_distance},
        Normal3T<Scalar>{.x = Scalar{-1}});
    ASSERT_TRUE(maximum_distance_conversion.has_value());
    EXPECT_TRUE(std::isfinite(maximum_distance_conversion->value));
    EXPECT_GT(maximum_distance_conversion->value, Scalar{0});
    const auto maximum_distance_round_trip = convert_solid_angle_pdf_to_area(
        *maximum_distance_conversion, Point3T<Scalar>{}, Point3T<Scalar>{.x = maximum_distance},
        Normal3T<Scalar>{.x = Scalar{-1}});
    ASSERT_TRUE(maximum_distance_round_trip.has_value());
    EXPECT_EQ(maximum_distance_round_trip->value, denormal);

    expect_invalid(convert_area_pdf_to_solid_angle(
                       ProbabilityFor<Scalar>{
                           .value = std::numeric_limits<Scalar>::max(),
                           .measure = ProbabilityMeasure::area,
                       },
                       Point3T<Scalar>{}, Point3T<Scalar>{.x = Scalar{2}},
                       Normal3T<Scalar>{.x = Scalar{-1}}),
                   "Converted light PDF is not representable in the requested precision.");
    expect_invalid(convert_area_pdf_to_solid_angle(
                       ProbabilityFor<Scalar>{
                           .value = denormal,
                           .measure = ProbabilityMeasure::area,
                       },
                       Point3T<Scalar>{}, Point3T<Scalar>{.x = Scalar{0.5}},
                       Normal3T<Scalar>{.x = Scalar{-1}}),
                   "Converted light PDF is not representable in the requested precision.");
    expect_invalid(convert_solid_angle_pdf_to_area(
                       ProbabilityFor<Scalar>{
                           .value = std::numeric_limits<Scalar>::max(),
                           .measure = ProbabilityMeasure::solid_angle,
                       },
                       Point3T<Scalar>{}, Point3T<Scalar>{.x = Scalar{0.5}},
                       Normal3T<Scalar>{.x = Scalar{-1}}),
                   "Converted light PDF is not representable in the requested precision.");
    expect_invalid(convert_solid_angle_pdf_to_area(
                       ProbabilityFor<Scalar>{
                           .value = denormal,
                           .measure = ProbabilityMeasure::solid_angle,
                       },
                       Point3T<Scalar>{}, Point3T<Scalar>{.x = Scalar{2}},
                       Normal3T<Scalar>{.x = Scalar{-1}}),
                   "Converted light PDF is not representable in the requested precision.");
}

TEST(LightPdfConversionTest, AvoidsIntermediateOverflowAndUnderflow) {
    expect_scaled_pdf_conversion<TransportScalar>();
    expect_scaled_pdf_conversion<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
