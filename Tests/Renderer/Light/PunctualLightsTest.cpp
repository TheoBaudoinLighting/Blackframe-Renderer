#include <Blackframe/Renderer/PunctualLights.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar> using SpectrumFor = LightSpectrumT<Scalar>;
template <SpectrumScalar Scalar> using PointLightFor = PointLightT<Scalar>;
template <SpectrumScalar Scalar> using DirectionalLightFor = DirectionalLightT<Scalar>;
template <SpectrumScalar Scalar> using SpotLightFor = SpotLightT<Scalar>;

template <SpectrumScalar Scalar>
inline constexpr auto AnalyticTolerance =
    std::same_as<Scalar, TransportScalar> ? ReferenceScalar{8.0e-6} : ReferenceScalar{8.0e-13};

template <typename Result> void expect_invalid(const Result& result) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
}

template <typename Value> [[nodiscard]] Value require_test_value(core::Result<Value> result) {
    if (!result.has_value()) {
        throw std::runtime_error{result.error().message};
    }
    return std::move(result).value();
}

template <SpectrumScalar Scalar> [[nodiscard]] SampledWavelengthsT<Scalar> test_wavelengths() {
    return require_test_value(sample_uniform_visible_wavelengths(Scalar{0.25}));
}

template <SpectrumScalar Scalar>
[[nodiscard]] LightSampleContextT<Scalar> test_context(const Point3T<Scalar> position) {
    return require_test_value(LightSampleContextT<Scalar>::create(position, Scalar{0.5}));
}

template <SpectrumScalar Scalar> [[nodiscard]] RayT<Scalar> test_ray() {
    return require_test_value(RayT<Scalar>::create(
        Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{2}, .z = Scalar{3}},
        Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0}, std::numeric_limits<Scalar>::infinity(),
        Scalar{0.5}, AllRayVisibility, VacuumMedium));
}

template <SpectrumScalar Scalar> [[nodiscard]] Bounds3T<Scalar> unit_scene_bounds() {
    return require_test_value(Bounds3T<Scalar>::from_minimum_maximum(
        Point3T<Scalar>{.x = Scalar{-1}, .y = Scalar{-1}, .z = Scalar{-1}},
        Point3T<Scalar>{.x = Scalar{1}, .y = Scalar{1}, .z = Scalar{1}}));
}

template <SpectrumScalar Scalar>
void expect_spectrum_near(const SpectrumFor<Scalar>& actual, const SpectrumFor<Scalar>& expected,
                          const ReferenceScalar tolerance = AnalyticTolerance<Scalar>) {
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(static_cast<ReferenceScalar>(actual[lane]),
                    static_cast<ReferenceScalar>(expected[lane]), tolerance);
    }
}

template <SpectrumScalar Scalar> void expect_zero_spectrum(const SpectrumFor<Scalar>& spectrum) {
    for (const auto value : spectrum.values) {
        EXPECT_EQ(value, Scalar{0});
        EXPECT_FALSE(std::signbit(value));
    }
}

template <SpectrumScalar Scalar>
[[nodiscard]] SpectrumFor<Scalar> filled_spectrum(const Scalar value) {
    auto spectrum = SpectrumFor<Scalar>{};
    spectrum.values.fill(value);
    return spectrum;
}

template <SpectrumScalar Scalar, typename Light>
void expect_delta_queries(const Light& light, const LightSampleContextT<Scalar>& context,
                          const SampledWavelengthsT<Scalar>& wavelengths) {
    const auto probability = light.pdf_li(context, Vector3T<Scalar>{.z = Scalar{1}}, wavelengths);
    ASSERT_TRUE(probability.has_value());
    EXPECT_EQ(probability->value(), Scalar{0});
    EXPECT_EQ(probability->measure(), ProbabilityMeasure::solid_angle);

    const auto emitted = light.le(test_ray<Scalar>(), wavelengths);
    ASSERT_TRUE(emitted.has_value());
    expect_zero_spectrum(*emitted);
}

TEST(PunctualLightsContractTest, ExposesBothPrecisionsThroughTheLightContract) {
    static_assert(LightModelFor<PointLight, TransportScalar>);
    static_assert(LightModelFor<ReferencePointLight, ReferenceScalar>);
    static_assert(LightModelFor<DirectionalLight, TransportScalar>);
    static_assert(LightModelFor<ReferenceDirectionalLight, ReferenceScalar>);
    static_assert(LightModelFor<SpotLight, TransportScalar>);
    static_assert(LightModelFor<ReferenceSpotLight, ReferenceScalar>);
    static_assert(!LightModelFor<PointLight, ReferenceScalar>);
    static_assert(!LightModelFor<ReferencePointLight, TransportScalar>);

    static_assert(std::same_as<decltype(PointLight::create(
                                   Point3{}, Vector3{}, SampledWavelengths{}, TransportSpectrum{})),
                               core::Result<PointLight>>);
    static_assert(
        std::same_as<decltype(ReferenceDirectionalLight::create(
                         ReferenceVector3{}, ReferenceSampledWavelengths{}, ReferenceSpectrum{})),
                     core::Result<ReferenceDirectionalLight>>);
    static_assert(std::same_as<decltype(SpotLight::create(
                                   Point3{}, Vector3{}, Vector3{}, TransportScalar{},
                                   TransportScalar{}, SampledWavelengths{}, TransportSpectrum{})),
                               core::Result<SpotLight>>);
}

template <SpectrumScalar Scalar> void expect_point_analytic_illumination() {
    const auto wavelengths = test_wavelengths<Scalar>();
    const auto intensity = SpectrumFor<Scalar>{
        .values = {Scalar{4}, Scalar{8}, Scalar{12}, Scalar{16}},
    };
    const auto position_error =
        Vector3T<Scalar>{.x = Scalar{0.125}, .y = Scalar{0.25}, .z = Scalar{0.5}};
    const auto light = PointLightFor<Scalar>::create(Point3T<Scalar>{.z = Scalar{2}},
                                                     position_error, wavelengths, intensity);
    ASSERT_TRUE(light.has_value());

    const auto context = test_context<Scalar>(Point3T<Scalar>{});
    const auto sampled = light->sample_li(
        context, Point2T<Scalar>{.x = Scalar{0.25}, .y = Scalar{0.75}}, wavelengths);
    ASSERT_TRUE(sampled.has_value());
    ASSERT_TRUE(sampled->has_value());
    const auto& sample = **sampled;
    EXPECT_EQ(sample.direction_to_light(), (Vector3T<Scalar>{.z = Scalar{1}}));
    EXPECT_EQ(sample.distance(), Scalar{2});
    expect_spectrum_near(sample.incident_radiance(),
                         SpectrumFor<Scalar>{
                             .values = {Scalar{1}, Scalar{2}, Scalar{3}, Scalar{4}},
                         });
    EXPECT_EQ(sample.probability().measure, ProbabilityMeasure::discrete);
    EXPECT_EQ(sample.probability().value, Scalar{1});
    EXPECT_EQ(sample.endpoint().kind(), LightEndpointKind::finite_point);
    ASSERT_TRUE(sample.endpoint().position().has_value());
    EXPECT_EQ(*sample.endpoint().position(), (Point3T<Scalar>{.z = Scalar{2}}));
    ASSERT_TRUE(sample.endpoint().absolute_position_error().has_value());
    EXPECT_EQ(*sample.endpoint().absolute_position_error(), position_error);

    // Receiver orientation belongs to the scattering estimator, not sample_li.
    const auto receiver_normal = Normal3T<Scalar>{.x = std::sqrt(Scalar{0.75}), .z = Scalar{0.5}};
    const auto receiver_cosine = dot(receiver_normal, sample.direction_to_light());
    EXPECT_NEAR(static_cast<ReferenceScalar>(receiver_cosine), 0.5, AnalyticTolerance<Scalar>);
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        const auto irradiance = sample.incident_radiance()[lane] * receiver_cosine;
        EXPECT_NEAR(static_cast<ReferenceScalar>(irradiance),
                    static_cast<ReferenceScalar>(intensity[lane]) / 8.0, AnalyticTolerance<Scalar>);
    }

    const auto farther_context = test_context<Scalar>(Point3T<Scalar>{.z = Scalar{-2}});
    const auto farther = light->sample_li(farther_context, Point2T<Scalar>{}, wavelengths);
    ASSERT_TRUE(farther.has_value());
    ASSERT_TRUE(farther->has_value());
    EXPECT_EQ((**farther).distance(), Scalar{4});
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(static_cast<ReferenceScalar>((**farther).incident_radiance()[lane]),
                    static_cast<ReferenceScalar>(intensity[lane]) / 16.0,
                    AnalyticTolerance<Scalar>);
        EXPECT_NEAR(static_cast<ReferenceScalar>(sample.incident_radiance()[lane] /
                                                 (**farther).incident_radiance()[lane]),
                    4.0, AnalyticTolerance<Scalar>);
    }

    expect_delta_queries<Scalar>(*light, context, wavelengths);
    const auto bounds = light->bounds();
    EXPECT_FALSE(bounds.is_empty());
    EXPECT_EQ(bounds.minimum(), (Point3T<Scalar>{
                                    .x = -position_error.x,
                                    .y = -position_error.y,
                                    .z = Scalar{2} - position_error.z,
                                }));
    EXPECT_EQ(bounds.maximum(), (Point3T<Scalar>{
                                    .x = position_error.x,
                                    .y = position_error.y,
                                    .z = Scalar{2} + position_error.z,
                                }));

    const auto power = light->power(unit_scene_bounds<Scalar>(), wavelengths);
    ASSERT_TRUE(power.has_value());
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(static_cast<ReferenceScalar>((*power)[lane]),
                    Scalar{4} * std::numbers::pi_v<ReferenceScalar> *
                        static_cast<ReferenceScalar>(intensity[lane]),
                    Scalar{16} * AnalyticTolerance<Scalar>);
    }
}

TEST(PointLightTest, FollowsInverseSquareAndLeavesReceiverCosineToTransport) {
    expect_point_analytic_illumination<TransportScalar>();
    expect_point_analytic_illumination<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_directional_analytic_illumination() {
    const auto wavelengths = test_wavelengths<Scalar>();
    const auto irradiance = SpectrumFor<Scalar>{
        .values = {Scalar{1}, Scalar{2}, Scalar{3}, Scalar{4}},
    };
    const auto light = DirectionalLightFor<Scalar>::create(Vector3T<Scalar>{.y = Scalar{-1}},
                                                           wavelengths, irradiance);
    ASSERT_TRUE(light.has_value());

    for (const auto position : std::array{
             Point3T<Scalar>{},
             Point3T<Scalar>{.x = Scalar{100}, .y = Scalar{-40}, .z = Scalar{7}},
         }) {
        const auto context = test_context<Scalar>(position);
        const auto sampled = light->sample_li(
            context, Point2T<Scalar>{.x = Scalar{0.125}, .y = Scalar{0.875}}, wavelengths);
        ASSERT_TRUE(sampled.has_value());
        ASSERT_TRUE(sampled->has_value());
        EXPECT_EQ((**sampled).direction_to_light(), (Vector3T<Scalar>{.y = Scalar{1}}));
        EXPECT_TRUE(std::isinf((**sampled).distance()));
        EXPECT_EQ((**sampled).endpoint().kind(), LightEndpointKind::infinite);
        EXPECT_EQ((**sampled).incident_radiance(), irradiance);
        EXPECT_EQ((**sampled).probability().measure, ProbabilityMeasure::discrete);
        EXPECT_EQ((**sampled).probability().value, Scalar{1});
    }

    const auto context = test_context<Scalar>(Point3T<Scalar>{});
    const auto receiver_normal = Normal3T<Scalar>{.x = std::sqrt(Scalar{0.75}), .y = Scalar{0.5}};
    const auto receiver_cosine = dot(receiver_normal, Vector3T<Scalar>{.y = Scalar{1}});
    EXPECT_NEAR(static_cast<ReferenceScalar>(receiver_cosine), 0.5, AnalyticTolerance<Scalar>);
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(static_cast<ReferenceScalar>(irradiance[lane] * receiver_cosine),
                    static_cast<ReferenceScalar>(irradiance[lane]) * 0.5,
                    AnalyticTolerance<Scalar>);
    }

    expect_delta_queries<Scalar>(*light, context, wavelengths);
    const auto bounds = light->bounds();
    EXPECT_FALSE(bounds.is_empty());
    EXPECT_TRUE(std::isinf(bounds.minimum().x));
    EXPECT_LT(bounds.minimum().x, Scalar{0});
    EXPECT_TRUE(std::isinf(bounds.maximum().x));
    EXPECT_GT(bounds.maximum().x, Scalar{0});

    const auto power = light->power(unit_scene_bounds<Scalar>(), wavelengths);
    ASSERT_TRUE(power.has_value());
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(static_cast<ReferenceScalar>((*power)[lane]),
                    Scalar{3} * std::numbers::pi_v<ReferenceScalar> *
                        static_cast<ReferenceScalar>(irradiance[lane]),
                    Scalar{16} * AnalyticTolerance<Scalar>);
    }
}

TEST(DirectionalLightTest, IsTranslationInvariantAndUsesBoundingSpherePower) {
    expect_directional_analytic_illumination<TransportScalar>();
    expect_directional_analytic_illumination<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_spot_analytic_illumination() {
    const auto wavelengths = test_wavelengths<Scalar>();
    const auto intensity = filled_spectrum<Scalar>(Scalar{8});
    const auto inner_angle = Scalar{0};
    const auto outer_angle = std::numbers::pi_v<Scalar> / Scalar{2};
    const auto position_error =
        Vector3T<Scalar>{.x = Scalar{0.125}, .y = Scalar{0.25}, .z = Scalar{0.5}};
    const auto light = SpotLightFor<Scalar>::create(Point3T<Scalar>{}, position_error,
                                                    Vector3T<Scalar>{.z = Scalar{1}}, inner_angle,
                                                    outer_angle, wavelengths, intensity);
    ASSERT_TRUE(light.has_value());

    const auto midpoint_context =
        test_context<Scalar>(Point3T<Scalar>{.x = std::sqrt(Scalar{3}), .z = Scalar{1}});
    const auto midpoint = light->sample_li(midpoint_context, Point2T<Scalar>{}, wavelengths);
    ASSERT_TRUE(midpoint.has_value());
    ASSERT_TRUE(midpoint->has_value());
    EXPECT_NEAR(static_cast<ReferenceScalar>((**midpoint).distance()), 2.0,
                AnalyticTolerance<Scalar>);
    expect_spectrum_near((**midpoint).incident_radiance(), filled_spectrum<Scalar>(Scalar{1}),
                         Scalar{8} * AnalyticTolerance<Scalar>);
    EXPECT_EQ((**midpoint).probability().measure, ProbabilityMeasure::discrete);
    EXPECT_EQ((**midpoint).probability().value, Scalar{1});
    ASSERT_TRUE((**midpoint).endpoint().absolute_position_error().has_value());
    EXPECT_EQ(*(**midpoint).endpoint().absolute_position_error(), position_error);

    const auto azimuth_context =
        test_context<Scalar>(Point3T<Scalar>{.y = std::sqrt(Scalar{3}), .z = Scalar{1}});
    const auto azimuth = light->sample_li(azimuth_context, Point2T<Scalar>{}, wavelengths);
    ASSERT_TRUE(azimuth.has_value());
    ASSERT_TRUE(azimuth->has_value());
    expect_spectrum_near((**azimuth).incident_radiance(), (**midpoint).incident_radiance());

    const auto axial_context = test_context<Scalar>(Point3T<Scalar>{.z = Scalar{2}});
    const auto axial = light->sample_li(axial_context, Point2T<Scalar>{}, wavelengths);
    ASSERT_TRUE(axial.has_value());
    ASSERT_TRUE(axial->has_value());
    expect_spectrum_near((**axial).incident_radiance(), filled_spectrum<Scalar>(Scalar{2}));

    const auto outside_context = test_context<Scalar>(Point3T<Scalar>{.z = Scalar{-2}});
    const auto outside = light->sample_li(outside_context, Point2T<Scalar>{}, wavelengths);
    ASSERT_TRUE(outside.has_value());
    EXPECT_FALSE(outside->has_value());

    expect_delta_queries<Scalar>(*light, axial_context, wavelengths);
    const auto bounds = light->bounds();
    EXPECT_FALSE(bounds.is_empty());
    EXPECT_EQ(bounds.minimum(), (Point3T<Scalar>{
                                    .x = -position_error.x,
                                    .y = -position_error.y,
                                    .z = -position_error.z,
                                }));
    EXPECT_EQ(bounds.maximum(), (Point3T<Scalar>{
                                    .x = position_error.x,
                                    .y = position_error.y,
                                    .z = position_error.z,
                                }));

    const auto power = light->power(unit_scene_bounds<Scalar>(), wavelengths);
    ASSERT_TRUE(power.has_value());
    const auto cosine_inner = std::cos(inner_angle);
    const auto cosine_outer = std::cos(outer_angle);
    const auto effective_solid_angle =
        Scalar{2} * std::numbers::pi_v<Scalar> *
        ((Scalar{1} - cosine_inner) + (cosine_inner - cosine_outer) / Scalar{2});
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(static_cast<ReferenceScalar>((*power)[lane]),
                    static_cast<ReferenceScalar>(effective_solid_angle * intensity[lane]),
                    Scalar{16} * AnalyticTolerance<Scalar>);
    }
}

TEST(SpotLightTest, AppliesSmoothConeFalloffBeforeInverseSquare) {
    expect_spot_analytic_illumination<TransportScalar>();
    expect_spot_analytic_illumination<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_creation_validation() {
    const auto wavelengths = test_wavelengths<Scalar>();
    const auto intensity = filled_spectrum<Scalar>(Scalar{1});
    const auto infinity = std::numeric_limits<Scalar>::infinity();
    const auto nan = std::numeric_limits<Scalar>::quiet_NaN();

    for (const auto invalid : std::array{nan, infinity, -infinity}) {
        expect_invalid(PointLightFor<Scalar>::create(Point3T<Scalar>{.x = invalid},
                                                     Vector3T<Scalar>{}, wavelengths, intensity));
        expect_invalid(SpotLightFor<Scalar>::create(
            Point3T<Scalar>{.x = invalid}, Vector3T<Scalar>{}, Vector3T<Scalar>{.z = Scalar{1}},
            Scalar{0}, std::numbers::pi_v<Scalar> / Scalar{2}, wavelengths, intensity));
    }

    for (const auto error : std::array{
             Vector3T<Scalar>{.x = Scalar{-1}},
             Vector3T<Scalar>{.y = nan},
             Vector3T<Scalar>{.z = infinity},
         }) {
        expect_invalid(
            PointLightFor<Scalar>::create(Point3T<Scalar>{}, error, wavelengths, intensity));
        expect_invalid(SpotLightFor<Scalar>::create(
            Point3T<Scalar>{}, error, Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0},
            std::numbers::pi_v<Scalar> / Scalar{2}, wavelengths, intensity));
    }

    for (const auto direction : std::array{
             Vector3T<Scalar>{},
             Vector3T<Scalar>{.z = Scalar{2}},
             Vector3T<Scalar>{.x = nan},
             Vector3T<Scalar>{.y = infinity},
         }) {
        expect_invalid(DirectionalLightFor<Scalar>::create(direction, wavelengths, intensity));
        expect_invalid(SpotLightFor<Scalar>::create(
            Point3T<Scalar>{}, Vector3T<Scalar>{}, direction, Scalar{0},
            std::numbers::pi_v<Scalar> / Scalar{2}, wavelengths, intensity));
    }

    for (const auto invalid_value : std::array{Scalar{-1}, nan, infinity}) {
        auto malformed = intensity;
        malformed[2] = invalid_value;
        expect_invalid(PointLightFor<Scalar>::create(Point3T<Scalar>{}, Vector3T<Scalar>{},
                                                     wavelengths, malformed));
        expect_invalid(DirectionalLightFor<Scalar>::create(Vector3T<Scalar>{.z = Scalar{1}},
                                                           wavelengths, malformed));
        expect_invalid(SpotLightFor<Scalar>::create(
            Point3T<Scalar>{}, Vector3T<Scalar>{}, Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0},
            std::numbers::pi_v<Scalar> / Scalar{2}, wavelengths, malformed));
    }

    auto invalid_wavelengths = wavelengths;
    invalid_wavelengths[0].probability.value = Scalar{0};
    expect_invalid(PointLightFor<Scalar>::create(Point3T<Scalar>{}, Vector3T<Scalar>{},
                                                 invalid_wavelengths, intensity));
    expect_invalid(DirectionalLightFor<Scalar>::create(Vector3T<Scalar>{.z = Scalar{1}},
                                                       invalid_wavelengths, intensity));
    expect_invalid(SpotLightFor<Scalar>::create(
        Point3T<Scalar>{}, Vector3T<Scalar>{}, Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0},
        std::numbers::pi_v<Scalar> / Scalar{2}, invalid_wavelengths, intensity));

    const auto half_pi = std::numbers::pi_v<Scalar> / Scalar{2};
    for (const auto angles : std::array{
             std::pair{Scalar{-0.1}, half_pi},
             std::pair{Scalar{0.75}, Scalar{0.5}},
             std::pair{Scalar{0}, Scalar{0}},
             std::pair{Scalar{0}, std::numbers::pi_v<Scalar>},
             std::pair{nan, half_pi},
             std::pair{Scalar{0}, nan},
             std::pair{Scalar{0}, infinity},
         }) {
        expect_invalid(SpotLightFor<Scalar>::create(Point3T<Scalar>{}, Vector3T<Scalar>{},
                                                    Vector3T<Scalar>{.z = Scalar{1}}, angles.first,
                                                    angles.second, wavelengths, intensity));
    }

    const auto unrepresentable_positive_outer =
        std::nextafter(Scalar{0}, std::numeric_limits<Scalar>::infinity());
    expect_invalid(SpotLightFor<Scalar>::create(
        Point3T<Scalar>{}, Vector3T<Scalar>{}, Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0},
        unrepresentable_positive_outer, wavelengths, intensity));

    const auto positive_inner_below_cosine_resolution =
        std::nextafter(Scalar{0}, std::numeric_limits<Scalar>::infinity());
    ASSERT_EQ(std::cos(positive_inner_below_cosine_resolution), Scalar{1});
    expect_invalid(SpotLightFor<Scalar>::create(
        Point3T<Scalar>{}, Vector3T<Scalar>{}, Vector3T<Scalar>{.z = Scalar{1}},
        positive_inner_below_cosine_resolution, half_pi, wavelengths, intensity));

    const auto outer_below_pi =
        std::nextafter(std::numbers::pi_v<Scalar>, -std::numeric_limits<Scalar>::infinity());
    ASSERT_EQ(std::cos(outer_below_pi), Scalar{-1});
    expect_invalid(SpotLightFor<Scalar>::create(Point3T<Scalar>{}, Vector3T<Scalar>{},
                                                Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0},
                                                outer_below_pi, wavelengths, intensity));
}

TEST(PunctualLightsValidationTest, RejectsMalformedConstructionWithoutRepair) {
    expect_creation_validation<TransportScalar>();
    expect_creation_validation<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_conservative_emitter_bounds() {
    const auto wavelengths = test_wavelengths<Scalar>();
    const auto intensity = filled_spectrum<Scalar>(Scalar{1});
    const auto tiny_error = std::numeric_limits<Scalar>::denorm_min();
    const auto position = Point3T<Scalar>{.x = Scalar{1}};
    const auto error = Vector3T<Scalar>{.x = tiny_error};
    const auto point = PointLightFor<Scalar>::create(position, error, wavelengths, intensity);
    const auto spot = SpotLightFor<Scalar>::create(
        position, error, Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0},
        std::numbers::pi_v<Scalar> / Scalar{2}, wavelengths, intensity);
    ASSERT_TRUE(point.has_value());
    ASSERT_TRUE(spot.has_value());
    EXPECT_LT(point->bounds().minimum().x, position.x);
    EXPECT_GT(point->bounds().maximum().x, position.x);
    EXPECT_EQ(spot->bounds().minimum(), point->bounds().minimum());
    EXPECT_EQ(spot->bounds().maximum(), point->bounds().maximum());

    const auto maximum = std::numeric_limits<Scalar>::max();
    const auto maximum_error = Vector3T<Scalar>{.x = maximum};
    const auto exact_extreme =
        PointLightFor<Scalar>::create(Point3T<Scalar>{}, maximum_error, wavelengths, intensity);
    ASSERT_TRUE(exact_extreme.has_value());
    EXPECT_EQ(exact_extreme->bounds().minimum().x, -maximum);
    EXPECT_EQ(exact_extreme->bounds().maximum().x, maximum);

    expect_invalid(PointLightFor<Scalar>::create(Point3T<Scalar>{.x = maximum}, error, wavelengths,
                                                 intensity));
    expect_invalid(SpotLightFor<Scalar>::create(
        Point3T<Scalar>{.x = maximum}, error, Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0},
        std::numbers::pi_v<Scalar> / Scalar{2}, wavelengths, intensity));
}

TEST(PunctualLightsValidationTest, ExpandsPositivePositionErrorOutward) {
    expect_conservative_emitter_bounds<TransportScalar>();
    expect_conservative_emitter_bounds<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_query_validation() {
    const auto wavelengths = test_wavelengths<Scalar>();
    const auto intensity = filled_spectrum<Scalar>(Scalar{1});
    const auto point = PointLightFor<Scalar>::create(Point3T<Scalar>{.z = Scalar{2}},
                                                     Vector3T<Scalar>{}, wavelengths, intensity);
    const auto directional = DirectionalLightFor<Scalar>::create(Vector3T<Scalar>{.z = Scalar{-1}},
                                                                 wavelengths, intensity);
    const auto spot = SpotLightFor<Scalar>::create(
        Point3T<Scalar>{}, Vector3T<Scalar>{}, Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0},
        std::numbers::pi_v<Scalar> / Scalar{2}, wavelengths, intensity);
    ASSERT_TRUE(point.has_value());
    ASSERT_TRUE(directional.has_value());
    ASSERT_TRUE(spot.has_value());

    const auto context = test_context<Scalar>(Point3T<Scalar>{.z = Scalar{1}});
    for (const auto sample : std::array{
             Point2T<Scalar>{.x = Scalar{-0.1}, .y = Scalar{0.5}},
             Point2T<Scalar>{.x = Scalar{1}, .y = Scalar{0.5}},
             Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{1}},
             Point2T<Scalar>{
                 .x = std::numeric_limits<Scalar>::quiet_NaN(),
                 .y = Scalar{0.5},
             },
             Point2T<Scalar>{
                 .x = std::numeric_limits<Scalar>::infinity(),
                 .y = Scalar{0.5},
             },
         }) {
        expect_invalid(point->sample_li(context, sample, wavelengths));
        expect_invalid(directional->sample_li(context, sample, wavelengths));
        expect_invalid(spot->sample_li(context, sample, wavelengths));
    }

    auto mismatched_wavelengths = wavelengths;
    mismatched_wavelengths[0].nanometers = std::nextafter(mismatched_wavelengths[0].nanometers,
                                                          std::numeric_limits<Scalar>::infinity());
    expect_invalid(point->sample_li(context, Point2T<Scalar>{}, mismatched_wavelengths));
    expect_invalid(
        point->pdf_li(context, Vector3T<Scalar>{.z = Scalar{1}}, mismatched_wavelengths));
    expect_invalid(point->le(test_ray<Scalar>(), mismatched_wavelengths));
    expect_invalid(point->power(unit_scene_bounds<Scalar>(), mismatched_wavelengths));
    expect_invalid(directional->sample_li(context, Point2T<Scalar>{}, mismatched_wavelengths));
    expect_invalid(
        directional->pdf_li(context, Vector3T<Scalar>{.z = Scalar{1}}, mismatched_wavelengths));
    expect_invalid(directional->le(test_ray<Scalar>(), mismatched_wavelengths));
    expect_invalid(directional->power(unit_scene_bounds<Scalar>(), mismatched_wavelengths));
    expect_invalid(spot->sample_li(context, Point2T<Scalar>{}, mismatched_wavelengths));
    expect_invalid(spot->pdf_li(context, Vector3T<Scalar>{.z = Scalar{1}}, mismatched_wavelengths));
    expect_invalid(spot->le(test_ray<Scalar>(), mismatched_wavelengths));
    expect_invalid(spot->power(unit_scene_bounds<Scalar>(), mismatched_wavelengths));

    for (const auto direction : std::array{
             Vector3T<Scalar>{},
             Vector3T<Scalar>{.z = Scalar{2}},
             Vector3T<Scalar>{.x = std::numeric_limits<Scalar>::quiet_NaN()},
         }) {
        expect_invalid(point->pdf_li(context, direction, wavelengths));
        expect_invalid(directional->pdf_li(context, direction, wavelengths));
        expect_invalid(spot->pdf_li(context, direction, wavelengths));
    }

    for (const auto bounds : std::array{Bounds3T<Scalar>::empty(), Bounds3T<Scalar>::unbounded()}) {
        expect_invalid(point->power(bounds, wavelengths));
        expect_invalid(directional->power(bounds, wavelengths));
        expect_invalid(spot->power(bounds, wavelengths));
    }
}

TEST(PunctualLightsValidationTest, ValidatesSamplesWavelengthsDirectionsAndSceneBounds) {
    expect_query_validation<TransportScalar>();
    expect_query_validation<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_black_lights() {
    const auto wavelengths = test_wavelengths<Scalar>();
    const auto black = SpectrumFor<Scalar>{};
    const auto point = PointLightFor<Scalar>::create(Point3T<Scalar>{.z = Scalar{2}},
                                                     Vector3T<Scalar>{}, wavelengths, black);
    const auto directional =
        DirectionalLightFor<Scalar>::create(Vector3T<Scalar>{.z = Scalar{-1}}, wavelengths, black);
    const auto spot = SpotLightFor<Scalar>::create(
        Point3T<Scalar>{}, Vector3T<Scalar>{}, Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0},
        std::numbers::pi_v<Scalar> / Scalar{2}, wavelengths, black);
    ASSERT_TRUE(point.has_value());
    ASSERT_TRUE(directional.has_value());
    ASSERT_TRUE(spot.has_value());

    const auto point_context = test_context<Scalar>(Point3T<Scalar>{});
    const auto spot_context = test_context<Scalar>(Point3T<Scalar>{.z = Scalar{2}});
    const auto point_sample = point->sample_li(point_context, Point2T<Scalar>{}, wavelengths);
    const auto directional_sample =
        directional->sample_li(point_context, Point2T<Scalar>{}, wavelengths);
    const auto spot_sample = spot->sample_li(spot_context, Point2T<Scalar>{}, wavelengths);
    ASSERT_TRUE(point_sample.has_value());
    ASSERT_TRUE(directional_sample.has_value());
    ASSERT_TRUE(spot_sample.has_value());
    EXPECT_FALSE(point_sample->has_value());
    EXPECT_FALSE(directional_sample->has_value());
    EXPECT_FALSE(spot_sample->has_value());

    const auto point_power = point->power(unit_scene_bounds<Scalar>(), wavelengths);
    const auto directional_power = directional->power(unit_scene_bounds<Scalar>(), wavelengths);
    const auto spot_power = spot->power(unit_scene_bounds<Scalar>(), wavelengths);
    ASSERT_TRUE(point_power.has_value());
    ASSERT_TRUE(directional_power.has_value());
    ASSERT_TRUE(spot_power.has_value());
    expect_zero_spectrum(*point_power);
    expect_zero_spectrum(*directional_power);
    expect_zero_spectrum(*spot_power);

    // Black must not hide invalid geometry or an invalid canonical sample.
    const auto coincident = test_context<Scalar>(Point3T<Scalar>{.z = Scalar{2}});
    expect_invalid(point->sample_li(coincident, Point2T<Scalar>{}, wavelengths));
    expect_invalid(point->sample_li(point_context, Point2T<Scalar>{.x = Scalar{1}}, wavelengths));
    expect_invalid(
        directional->sample_li(point_context, Point2T<Scalar>{.x = Scalar{1}}, wavelengths));
    expect_invalid(spot->sample_li(spot_context, Point2T<Scalar>{.x = Scalar{1}}, wavelengths));
    expect_invalid(point->power(Bounds3T<Scalar>::empty(), wavelengths));
    expect_invalid(directional->power(Bounds3T<Scalar>::empty(), wavelengths));
    expect_invalid(spot->power(Bounds3T<Scalar>::empty(), wavelengths));
}

TEST(PunctualLightsTest, TreatsBlackAsPhysicalZeroWithoutMaskingInvalidInputs) {
    expect_black_lights<TransportScalar>();
    expect_black_lights<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_finite_light_singularities() {
    const auto wavelengths = test_wavelengths<Scalar>();
    const auto intensity = filled_spectrum<Scalar>(Scalar{1});
    const auto point = PointLightFor<Scalar>::create(Point3T<Scalar>{}, Vector3T<Scalar>{},
                                                     wavelengths, intensity);
    const auto spot = SpotLightFor<Scalar>::create(
        Point3T<Scalar>{}, Vector3T<Scalar>{}, Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0},
        std::numbers::pi_v<Scalar> / Scalar{2}, wavelengths, intensity);
    ASSERT_TRUE(point.has_value());
    ASSERT_TRUE(spot.has_value());
    const auto coincident = test_context<Scalar>(Point3T<Scalar>{});
    expect_invalid(point->sample_li(coincident, Point2T<Scalar>{}, wavelengths));
    expect_invalid(spot->sample_li(coincident, Point2T<Scalar>{}, wavelengths));

    const auto maximum = std::numeric_limits<Scalar>::max();
    const auto far_point = PointLightFor<Scalar>::create(
        Point3T<Scalar>{.x = maximum}, Vector3T<Scalar>{}, wavelengths, intensity);
    const auto far_spot = SpotLightFor<Scalar>::create(
        Point3T<Scalar>{.x = maximum}, Vector3T<Scalar>{}, Vector3T<Scalar>{.x = Scalar{-1}},
        Scalar{0}, std::numbers::pi_v<Scalar> / Scalar{2}, wavelengths, intensity);
    ASSERT_TRUE(far_point.has_value());
    ASSERT_TRUE(far_spot.has_value());
    const auto opposite = test_context<Scalar>(Point3T<Scalar>{.x = -maximum});
    expect_invalid(far_point->sample_li(opposite, Point2T<Scalar>{}, wavelengths));
    expect_invalid(far_spot->sample_li(opposite, Point2T<Scalar>{}, wavelengths));

    const auto diagonal_point = PointLightFor<Scalar>::create(
        Point3T<Scalar>{.x = maximum, .y = maximum}, Vector3T<Scalar>{}, wavelengths, intensity);
    const auto diagonal_spot = SpotLightFor<Scalar>::create(
        Point3T<Scalar>{.x = maximum, .y = maximum}, Vector3T<Scalar>{},
        Vector3T<Scalar>{.x = Scalar{-1}}, Scalar{0}, std::numbers::pi_v<Scalar> / Scalar{2},
        wavelengths, intensity);
    ASSERT_TRUE(diagonal_point.has_value());
    ASSERT_TRUE(diagonal_spot.has_value());
    const auto origin = test_context<Scalar>(Point3T<Scalar>{});
    expect_invalid(diagonal_point->sample_li(origin, Point2T<Scalar>{}, wavelengths));
    expect_invalid(diagonal_spot->sample_li(origin, Point2T<Scalar>{}, wavelengths));
}

TEST(PunctualLightsTest, RejectsCoincidentAndUnrepresentableFiniteSegments) {
    expect_finite_light_singularities<TransportScalar>();
    expect_finite_light_singularities<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_inverse_square_extremes_for_point() {
    const auto wavelengths = test_wavelengths<Scalar>();
    constexpr auto exponent = std::same_as<Scalar, TransportScalar> ? 80 : 600;
    const auto far = std::scalbn(Scalar{1}, exponent);
    const auto near = std::scalbn(Scalar{1}, -exponent);
    const auto origin = test_context<Scalar>(Point3T<Scalar>{});

    const auto far_light = PointLightFor<Scalar>::create(
        Point3T<Scalar>{.x = far}, Vector3T<Scalar>{}, wavelengths, filled_spectrum<Scalar>(far));
    ASSERT_TRUE(far_light.has_value());
    const auto far_sample = far_light->sample_li(origin, Point2T<Scalar>{}, wavelengths);
    ASSERT_TRUE(far_sample.has_value());
    ASSERT_TRUE(far_sample->has_value());
    expect_spectrum_near((**far_sample).incident_radiance(), filled_spectrum<Scalar>(near),
                         Scalar{8} * AnalyticTolerance<Scalar>);

    const auto near_light = PointLightFor<Scalar>::create(
        Point3T<Scalar>{.x = near}, Vector3T<Scalar>{}, wavelengths, filled_spectrum<Scalar>(near));
    ASSERT_TRUE(near_light.has_value());
    const auto near_sample = near_light->sample_li(origin, Point2T<Scalar>{}, wavelengths);
    ASSERT_TRUE(near_sample.has_value());
    ASSERT_TRUE(near_sample->has_value());
    expect_spectrum_near((**near_sample).incident_radiance(), filled_spectrum<Scalar>(far),
                         static_cast<ReferenceScalar>(far) * Scalar{8} *
                             std::numeric_limits<Scalar>::epsilon());

    const auto minimum_distance = std::numeric_limits<Scalar>::min();
    const auto denormal = std::numeric_limits<Scalar>::denorm_min();
    const auto minimum_light =
        PointLightFor<Scalar>::create(Point3T<Scalar>{.x = minimum_distance}, Vector3T<Scalar>{},
                                      wavelengths, filled_spectrum<Scalar>(denormal));
    ASSERT_TRUE(minimum_light.has_value());
    const auto minimum_sample = minimum_light->sample_li(origin, Point2T<Scalar>{}, wavelengths);
    ASSERT_TRUE(minimum_sample.has_value());
    ASSERT_TRUE(minimum_sample->has_value());
    const auto expected_minimum =
        std::scalbn(denormal, -2 * (std::numeric_limits<Scalar>::min_exponent - 1));
    EXPECT_EQ((**minimum_sample).incident_radiance()[0], expected_minimum);

    const auto maximum_distance = std::numeric_limits<Scalar>::max();
    const auto maximum_light =
        PointLightFor<Scalar>::create(Point3T<Scalar>{.x = maximum_distance}, Vector3T<Scalar>{},
                                      wavelengths, filled_spectrum<Scalar>(maximum_distance));
    ASSERT_TRUE(maximum_light.has_value());
    const auto maximum_sample = maximum_light->sample_li(origin, Point2T<Scalar>{}, wavelengths);
    ASSERT_TRUE(maximum_sample.has_value());
    ASSERT_TRUE(maximum_sample->has_value());
    const auto maximum_result = (**maximum_sample).incident_radiance()[0];
    EXPECT_TRUE(std::isfinite(maximum_result));
    EXPECT_GT(maximum_result, Scalar{0});
    EXPECT_EQ(std::fpclassify(maximum_result), FP_SUBNORMAL);
    EXPECT_NEAR(static_cast<ReferenceScalar>(maximum_result * maximum_distance), 1.0,
                Scalar{8} * AnalyticTolerance<Scalar>);

    const auto overflowing = PointLightFor<Scalar>::create(
        Point3T<Scalar>{.x = Scalar{0.5}}, Vector3T<Scalar>{}, wavelengths,
        filled_spectrum<Scalar>(std::numeric_limits<Scalar>::max()));
    const auto underflowing =
        PointLightFor<Scalar>::create(Point3T<Scalar>{.x = Scalar{2}}, Vector3T<Scalar>{},
                                      wavelengths, filled_spectrum<Scalar>(denormal));
    ASSERT_TRUE(overflowing.has_value());
    ASSERT_TRUE(underflowing.has_value());
    expect_invalid(overflowing->sample_li(origin, Point2T<Scalar>{}, wavelengths));
    expect_invalid(overflowing->power(unit_scene_bounds<Scalar>(), wavelengths));
    expect_invalid(underflowing->sample_li(origin, Point2T<Scalar>{}, wavelengths));
}

TEST(PointLightTest, KeepsCompensatedInverseSquareResultsRepresentable) {
    expect_inverse_square_extremes_for_point<TransportScalar>();
    expect_inverse_square_extremes_for_point<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_directional_power_extremes() {
    const auto wavelengths = test_wavelengths<Scalar>();
    constexpr auto exponent = std::same_as<Scalar, TransportScalar> ? 80 : 600;
    const auto large = std::scalbn(Scalar{1}, exponent);
    const auto small = std::scalbn(Scalar{1}, -exponent);

    const auto small_bounds = Bounds3T<Scalar>::from_minimum_maximum(Point3T<Scalar>{.x = -small},
                                                                     Point3T<Scalar>{.x = small});
    const auto large_bounds = Bounds3T<Scalar>::from_minimum_maximum(Point3T<Scalar>{.x = -large},
                                                                     Point3T<Scalar>{.x = large});
    ASSERT_TRUE(small_bounds.has_value());
    ASSERT_TRUE(large_bounds.has_value());

    const auto small_scene_light = DirectionalLightFor<Scalar>::create(
        Vector3T<Scalar>{.z = Scalar{-1}}, wavelengths, filled_spectrum<Scalar>(large));
    const auto large_scene_light = DirectionalLightFor<Scalar>::create(
        Vector3T<Scalar>{.z = Scalar{-1}}, wavelengths, filled_spectrum<Scalar>(small));
    ASSERT_TRUE(small_scene_light.has_value());
    ASSERT_TRUE(large_scene_light.has_value());

    const auto small_power = small_scene_light->power(*small_bounds, wavelengths);
    const auto large_power = large_scene_light->power(*large_bounds, wavelengths);
    ASSERT_TRUE(small_power.has_value());
    ASSERT_TRUE(large_power.has_value());
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_TRUE(std::isfinite((*small_power)[lane]));
        EXPECT_TRUE(std::isfinite((*large_power)[lane]));
        EXPECT_GT((*small_power)[lane], Scalar{0});
        EXPECT_GT((*large_power)[lane], Scalar{0});
        EXPECT_NEAR(static_cast<ReferenceScalar>((*small_power)[lane] / small),
                    std::numbers::pi_v<ReferenceScalar>, Scalar{16} * AnalyticTolerance<Scalar>);
        EXPECT_NEAR(static_cast<ReferenceScalar>((*large_power)[lane] / large),
                    std::numbers::pi_v<ReferenceScalar>, Scalar{16} * AnalyticTolerance<Scalar>);
    }
    expect_invalid(small_scene_light->power(*large_bounds, wavelengths));
    expect_invalid(large_scene_light->power(*small_bounds, wavelengths));

    const auto scaled_bounds = Bounds3T<Scalar>::from_minimum_maximum(
        Point3T<Scalar>{.x = Scalar{-2}, .y = Scalar{-2}, .z = Scalar{-2}},
        Point3T<Scalar>{.x = Scalar{2}, .y = Scalar{2}, .z = Scalar{2}});
    const auto translated_bounds = Bounds3T<Scalar>::from_minimum_maximum(
        Point3T<Scalar>{.x = Scalar{9}, .y = Scalar{-7}, .z = Scalar{3}},
        Point3T<Scalar>{.x = Scalar{11}, .y = Scalar{-5}, .z = Scalar{5}});
    ASSERT_TRUE(scaled_bounds.has_value());
    ASSERT_TRUE(translated_bounds.has_value());
    const auto unit_light = DirectionalLightFor<Scalar>::create(
        Vector3T<Scalar>{.z = Scalar{-1}}, wavelengths, filled_spectrum<Scalar>(Scalar{1}));
    ASSERT_TRUE(unit_light.has_value());
    const auto unit_power = unit_light->power(unit_scene_bounds<Scalar>(), wavelengths);
    const auto scaled_power = unit_light->power(*scaled_bounds, wavelengths);
    const auto translated_power = unit_light->power(*translated_bounds, wavelengths);
    ASSERT_TRUE(unit_power.has_value());
    ASSERT_TRUE(scaled_power.has_value());
    ASSERT_TRUE(translated_power.has_value());
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(static_cast<ReferenceScalar>((*scaled_power)[lane] / (*unit_power)[lane]), 4.0,
                    AnalyticTolerance<Scalar>);
        EXPECT_EQ((*translated_power)[lane], (*unit_power)[lane]);
    }

    const auto point_bounds = Bounds3T<Scalar>::from_minimum_maximum(
        Point3T<Scalar>{.x = Scalar{7}, .y = Scalar{-3}, .z = Scalar{2}},
        Point3T<Scalar>{.x = Scalar{7}, .y = Scalar{-3}, .z = Scalar{2}});
    ASSERT_TRUE(point_bounds.has_value());
    const auto zero_power = unit_light->power(*point_bounds, wavelengths);
    ASSERT_TRUE(zero_power.has_value());
    expect_zero_spectrum(*zero_power);

    const auto denormal = std::numeric_limits<Scalar>::denorm_min();
    const auto subnormal_bounds =
        Bounds3T<Scalar>::from_minimum_maximum(Point3T<Scalar>{}, Point3T<Scalar>{.x = denormal});
    ASSERT_TRUE(subnormal_bounds.has_value());
    expect_invalid(unit_light->power(*subnormal_bounds, wavelengths));

    const auto maximum = std::numeric_limits<Scalar>::max();
    const auto maximum_bounds = Bounds3T<Scalar>::from_minimum_maximum(
        Point3T<Scalar>{.x = -maximum}, Point3T<Scalar>{.x = maximum});
    const auto compensated_light = DirectionalLightFor<Scalar>::create(
        Vector3T<Scalar>{.z = Scalar{-1}}, wavelengths, filled_spectrum<Scalar>(denormal));
    ASSERT_TRUE(maximum_bounds.has_value());
    ASSERT_TRUE(compensated_light.has_value());
    const auto compensated_power = compensated_light->power(*maximum_bounds, wavelengths);
    ASSERT_TRUE(compensated_power.has_value());
    for (const auto value : compensated_power->values) {
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_GT(value, Scalar{0});
    }
}

TEST(DirectionalLightTest, ComposesSpectralIrradianceWithExtremeSceneRadiusSquared) {
    expect_directional_power_extremes<TransportScalar>();
    expect_directional_power_extremes<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_spot_falloff_extremes() {
    const auto wavelengths = test_wavelengths<Scalar>();
    constexpr auto exponent = std::same_as<Scalar, TransportScalar> ? 80 : 600;
    const auto large = std::scalbn(Scalar{1}, exponent);
    const auto transition = std::scalbn(Scalar{1}, -exponent);
    const auto light = SpotLightFor<Scalar>::create(
        Point3T<Scalar>{}, Vector3T<Scalar>{}, Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0},
        std::numbers::pi_v<Scalar> / Scalar{2}, wavelengths, filled_spectrum<Scalar>(large));
    ASSERT_TRUE(light.has_value());

    const auto context = test_context<Scalar>(Point3T<Scalar>{.x = Scalar{1}, .z = transition});
    const auto sampled = light->sample_li(context, Point2T<Scalar>{}, wavelengths);
    ASSERT_TRUE(sampled.has_value());
    ASSERT_TRUE(sampled->has_value());
    const auto expected = transition * (Scalar{3} - Scalar{2} * transition);
    for (const auto value : (**sampled).incident_radiance().values) {
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_GT(value, Scalar{0});
        EXPECT_NEAR(static_cast<ReferenceScalar>(value / expected), 1.0,
                    Scalar{32} * AnalyticTolerance<Scalar>);
    }

    const auto far_light = SpotLightFor<Scalar>::create(
        Point3T<Scalar>{.z = large}, Vector3T<Scalar>{}, Vector3T<Scalar>{.z = Scalar{-1}},
        Scalar{0}, std::numbers::pi_v<Scalar> / Scalar{2}, wavelengths,
        filled_spectrum<Scalar>(large));
    ASSERT_TRUE(far_light.has_value());
    const auto origin = test_context<Scalar>(Point3T<Scalar>{});
    const auto far_sample = far_light->sample_li(origin, Point2T<Scalar>{}, wavelengths);
    ASSERT_TRUE(far_sample.has_value());
    ASSERT_TRUE(far_sample->has_value());
    expect_spectrum_near((**far_sample).incident_radiance(), filled_spectrum<Scalar>(transition),
                         Scalar{16} * AnalyticTolerance<Scalar>);

    const auto overflowing = SpotLightFor<Scalar>::create(
        Point3T<Scalar>{}, Vector3T<Scalar>{}, Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0},
        std::numbers::pi_v<Scalar> / Scalar{2}, wavelengths,
        filled_spectrum<Scalar>(std::numeric_limits<Scalar>::max()));
    const auto underflowing = SpotLightFor<Scalar>::create(
        Point3T<Scalar>{}, Vector3T<Scalar>{}, Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0},
        std::numbers::pi_v<Scalar> / Scalar{2}, wavelengths,
        filled_spectrum<Scalar>(std::numeric_limits<Scalar>::denorm_min()));
    ASSERT_TRUE(overflowing.has_value());
    ASSERT_TRUE(underflowing.has_value());
    expect_invalid(overflowing->sample_li(test_context<Scalar>(Point3T<Scalar>{.z = Scalar{0.5}}),
                                          Point2T<Scalar>{}, wavelengths));
    expect_invalid(underflowing->sample_li(test_context<Scalar>(Point3T<Scalar>{.z = Scalar{2}}),
                                           Point2T<Scalar>{}, wavelengths));
    expect_invalid(overflowing->power(unit_scene_bounds<Scalar>(), wavelengths));

    const auto hard_angle = std::numbers::pi_v<Scalar> / Scalar{3};
    const auto hard = SpotLightFor<Scalar>::create(
        Point3T<Scalar>{}, Vector3T<Scalar>{}, Vector3T<Scalar>{.z = Scalar{1}}, hard_angle,
        hard_angle, wavelengths, filled_spectrum<Scalar>(Scalar{1}));
    ASSERT_TRUE(hard.has_value());
    const auto hard_power = hard->power(unit_scene_bounds<Scalar>(), wavelengths);
    ASSERT_TRUE(hard_power.has_value());
    const auto expected_hard_power =
        Scalar{2} * std::numbers::pi_v<Scalar> * (Scalar{1} - std::cos(hard_angle));
    for (const auto value : hard_power->values) {
        EXPECT_NEAR(static_cast<ReferenceScalar>(value),
                    static_cast<ReferenceScalar>(expected_hard_power),
                    Scalar{16} * AnalyticTolerance<Scalar>);
    }

    // This cone makes cos(outer) the representable value immediately below one on the
    // supported implementations.  Computing 1 - (cos(inner) + cos(outer)) / 2 loses the
    // complete solid angle, while the difference form remains positive.
    const auto narrow_angle = std::sqrt(Scalar{1.25} * std::numeric_limits<Scalar>::epsilon());
    const auto narrow_cosine = std::cos(narrow_angle);
    ASSERT_LT(narrow_cosine, Scalar{1});
    const auto narrow = SpotLightFor<Scalar>::create(
        Point3T<Scalar>{}, Vector3T<Scalar>{}, Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0},
        narrow_angle, wavelengths, filled_spectrum<Scalar>(Scalar{1}));
    const auto narrow_underflowing = SpotLightFor<Scalar>::create(
        Point3T<Scalar>{}, Vector3T<Scalar>{}, Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0},
        narrow_angle, wavelengths,
        filled_spectrum<Scalar>(std::numeric_limits<Scalar>::denorm_min()));
    ASSERT_TRUE(narrow.has_value());
    ASSERT_TRUE(narrow_underflowing.has_value());
    const auto narrow_power = narrow->power(unit_scene_bounds<Scalar>(), wavelengths);
    ASSERT_TRUE(narrow_power.has_value());
    const auto expected_narrow_power = std::numbers::pi_v<Scalar> * (Scalar{1} - narrow_cosine);
    for (const auto value : narrow_power->values) {
        EXPECT_GT(value, Scalar{0});
        EXPECT_NEAR(static_cast<ReferenceScalar>(value / expected_narrow_power), 1.0,
                    Scalar{16} * AnalyticTolerance<Scalar>);
    }
    expect_invalid(narrow_underflowing->power(unit_scene_bounds<Scalar>(), wavelengths));
}

TEST(SpotLightTest, PreservesCompensatedSmoothFalloffAndInverseSquareTerms) {
    expect_spot_falloff_extremes<TransportScalar>();
    expect_spot_falloff_extremes<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
