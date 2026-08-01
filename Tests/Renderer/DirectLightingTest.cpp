#include <Blackframe/Renderer/DirectLighting.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
using SpectrumFor = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

template <SpectrumScalar Scalar> [[nodiscard]] OrthonormalFrameT<Scalar> upward_frame() {
    return OrthonormalFrameT<Scalar>::from_normal(Normal3T<Scalar>{.z = Scalar{1}}).value();
}

template <SpectrumScalar Scalar>
[[nodiscard]] LightSelectionProbabilityT<Scalar>
selection_probability(const std::size_t light_count, const Scalar canonical = Scalar{0}) {
    const auto sampler = LightSamplerT<Scalar>::create_uniform(light_count).value();
    return sampler.sample(canonical).value().probability();
}

template <SpectrumScalar Scalar>
[[nodiscard]] IncidentLightSampleT<Scalar>
point_sample(const Point3T<Scalar> endpoint, const SpectrumFor<Scalar>& radiance,
             const Scalar conditional_probability = Scalar{1}) {
    const auto context =
        LightSampleContextT<Scalar>::create(Point3T<Scalar>{}, Scalar{0.5}).value();
    const auto light_endpoint =
        LightSampleEndpointT<Scalar>::create_point(endpoint, Vector3T<Scalar>{}).value();
    return IncidentLightSampleT<Scalar>::create_finite(context, light_endpoint, radiance,
                                                       LightProbabilityDensityT<Scalar>{
                                                           .value = conditional_probability,
                                                           .measure = ProbabilityMeasure::discrete,
                                                       })
        .value();
}

template <SpectrumScalar Scalar> [[nodiscard]] SpectrumFor<Scalar> unit_spectrum() {
    auto result = SpectrumFor<Scalar>{};
    result.values.fill(Scalar{1});
    return result;
}

template <SpectrumScalar Scalar> void expect_axial_point_estimator() {
    const auto beta = SpectrumFor<Scalar>{
        .values = {Scalar{1}, Scalar{2}, Scalar{0.5}, Scalar{0.25}},
    };
    const auto reflectance = SpectrumFor<Scalar>{
        .values = {Scalar{0.25}, Scalar{0.5}, Scalar{0.75}, Scalar{1}},
    };
    const auto radiance = SpectrumFor<Scalar>{
        .values = {Scalar{2}, Scalar{4}, Scalar{8}, Scalar{16}},
    };
    const auto reflection = LambertianReflectionT<Scalar>::create(reflectance).value();
    const auto incident = point_sample<Scalar>(Point3T<Scalar>{.z = Scalar{2}}, radiance);

    static_assert(
        std::same_as<decltype(evaluate_lambertian_direct_lighting(
                         beta, reflection, upward_frame<Scalar>(), Vector3T<Scalar>{.z = Scalar{1}},
                         selection_probability<Scalar>(1U), incident, unit_spectrum<Scalar>())),
                     core::Result<SpectrumFor<Scalar>>>);
    const auto evaluated = evaluate_lambertian_direct_lighting(
        beta, reflection, upward_frame<Scalar>(), Vector3T<Scalar>{.z = Scalar{1}},
        selection_probability<Scalar>(1U), incident, unit_spectrum<Scalar>());
    ASSERT_TRUE(evaluated.has_value()) << evaluated.error().message;

    constexpr auto relative_tolerance =
        std::same_as<Scalar, TransportScalar> ? ReferenceScalar{2.0e-6} : ReferenceScalar{2.0e-14};
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        const auto expected = static_cast<ReferenceScalar>(beta[lane]) *
                              static_cast<ReferenceScalar>(reflectance[lane]) *
                              static_cast<ReferenceScalar>(radiance[lane]) *
                              std::numbers::inv_pi_v<ReferenceScalar>;
        EXPECT_NEAR(static_cast<ReferenceScalar>((*evaluated)[lane]), expected,
                    std::abs(expected) * relative_tolerance);
    }
}

template <SpectrumScalar Scalar> void expect_selection_factor() {
    const auto reflection = LambertianReflectionT<Scalar>::create(unit_spectrum<Scalar>()).value();
    const auto incident =
        point_sample<Scalar>(Point3T<Scalar>{.z = Scalar{1}}, unit_spectrum<Scalar>());
    const auto one_light = evaluate_lambertian_direct_lighting(
        unit_spectrum<Scalar>(), reflection, upward_frame<Scalar>(),
        Vector3T<Scalar>{.z = Scalar{1}}, selection_probability<Scalar>(1U), incident,
        unit_spectrum<Scalar>());
    const auto four_lights = evaluate_lambertian_direct_lighting(
        unit_spectrum<Scalar>(), reflection, upward_frame<Scalar>(),
        Vector3T<Scalar>{.z = Scalar{1}}, selection_probability<Scalar>(4U), incident,
        unit_spectrum<Scalar>());
    ASSERT_TRUE(one_light.has_value()) << one_light.error().message;
    ASSERT_TRUE(four_lights.has_value()) << four_lights.error().message;
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_EQ((*four_lights)[lane], Scalar{4} * (*one_light)[lane]);
    }
}

template <SpectrumScalar Scalar> void expect_zero_support() {
    const auto reflection = LambertianReflectionT<Scalar>::create(unit_spectrum<Scalar>()).value();
    const auto below =
        point_sample<Scalar>(Point3T<Scalar>{.z = Scalar{-1}}, unit_spectrum<Scalar>());
    const auto opposite = evaluate_lambertian_direct_lighting(
        unit_spectrum<Scalar>(), reflection, upward_frame<Scalar>(),
        Vector3T<Scalar>{.z = Scalar{1}}, selection_probability<Scalar>(1U), below,
        unit_spectrum<Scalar>());
    ASSERT_TRUE(opposite.has_value()) << opposite.error().message;
    EXPECT_EQ(*opposite, SpectrumFor<Scalar>{});

    const auto above =
        point_sample<Scalar>(Point3T<Scalar>{.z = Scalar{1}}, unit_spectrum<Scalar>());
    const auto blocked = evaluate_lambertian_direct_lighting(
        unit_spectrum<Scalar>(), reflection, upward_frame<Scalar>(),
        Vector3T<Scalar>{.z = Scalar{1}}, selection_probability<Scalar>(1U), above,
        SpectrumFor<Scalar>{});
    ASSERT_TRUE(blocked.has_value()) << blocked.error().message;
    EXPECT_EQ(*blocked, SpectrumFor<Scalar>{});
}

TEST(DirectLightingTest, MatchesTheAxialPointEstimatorInBothPrecisions) {
    expect_axial_point_estimator<TransportScalar>();
    expect_axial_point_estimator<ReferenceScalar>();
}

TEST(DirectLightingTest, AppliesTheDiscreteLightSelectionProbabilityExactlyOnce) {
    expect_selection_factor<TransportScalar>();
    expect_selection_factor<ReferenceScalar>();
}

TEST(DirectLightingTest, ReturnsExactZeroOutsideSupportOrThroughZeroTransmittance) {
    expect_zero_support<TransportScalar>();
    expect_zero_support<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_estimator_weight_applied_exactly_once() {
    const auto reflection = LambertianReflectionT<Scalar>::create(unit_spectrum<Scalar>()).value();
    const auto incident =
        point_sample<Scalar>(Point3T<Scalar>{.z = Scalar{1}}, unit_spectrum<Scalar>());
    const auto unweighted = evaluate_lambertian_direct_lighting(
        unit_spectrum<Scalar>(), reflection, upward_frame<Scalar>(),
        Vector3T<Scalar>{.z = Scalar{1}}, selection_probability<Scalar>(1U), incident,
        unit_spectrum<Scalar>());
    const auto weighted = evaluate_lambertian_direct_lighting(
        unit_spectrum<Scalar>(), reflection, upward_frame<Scalar>(),
        Vector3T<Scalar>{.z = Scalar{1}}, selection_probability<Scalar>(1U), incident,
        unit_spectrum<Scalar>(), Scalar{0.25});
    ASSERT_TRUE(unweighted.has_value()) << unweighted.error().message;
    ASSERT_TRUE(weighted.has_value()) << weighted.error().message;
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_EQ((*weighted)[lane], Scalar{0.25} * (*unweighted)[lane]);
    }
}

TEST(DirectLightingTest, AppliesAnEstimatorWeightExactlyOnceInBothPrecisions) {
    expect_estimator_weight_applied_exactly_once<TransportScalar>();
    expect_estimator_weight_applied_exactly_once<ReferenceScalar>();
}

TEST(DirectLightingTest, RejectsInvalidSpectraAndDirections) {
    const auto reflection = LambertianReflection::create(unit_spectrum<TransportScalar>()).value();
    const auto incident =
        point_sample<TransportScalar>(Point3{.z = 1.0F}, unit_spectrum<TransportScalar>());
    const auto probability = selection_probability<TransportScalar>(1U);
    const auto frame = upward_frame<TransportScalar>();

    auto invalid_beta = unit_spectrum<TransportScalar>();
    invalid_beta[1] = -1.0F;
    const auto negative = evaluate_lambertian_direct_lighting(
        invalid_beta, reflection, frame, Vector3{.z = 1.0F}, probability, incident,
        unit_spectrum<TransportScalar>());
    ASSERT_FALSE(negative.has_value());
    EXPECT_EQ(negative.error().code, core::StatusCode::invalid_argument);

    auto invalid_transmittance = unit_spectrum<TransportScalar>();
    invalid_transmittance[2] = std::numeric_limits<TransportScalar>::infinity();
    const auto infinite = evaluate_lambertian_direct_lighting(
        unit_spectrum<TransportScalar>(), reflection, frame, Vector3{.z = 1.0F}, probability,
        incident, invalid_transmittance);
    ASSERT_FALSE(infinite.has_value());
    EXPECT_EQ(infinite.error().code, core::StatusCode::invalid_argument);

    auto amplified_transmittance = unit_spectrum<TransportScalar>();
    amplified_transmittance[0] = 1.0001F;
    const auto amplified = evaluate_lambertian_direct_lighting(
        unit_spectrum<TransportScalar>(), reflection, frame, Vector3{.z = 1.0F}, probability,
        incident, amplified_transmittance);
    ASSERT_FALSE(amplified.has_value());
    EXPECT_EQ(amplified.error().code, core::StatusCode::invalid_argument);

    const auto direction = evaluate_lambertian_direct_lighting(
        unit_spectrum<TransportScalar>(), reflection, frame, Vector3{}, probability, incident,
        unit_spectrum<TransportScalar>());
    ASSERT_FALSE(direction.has_value());
    EXPECT_EQ(direction.error().code, core::StatusCode::invalid_argument);

    const auto invalid_weight = evaluate_lambertian_direct_lighting(
        unit_spectrum<TransportScalar>(), reflection, frame, Vector3{.z = 1.0F}, probability,
        incident, unit_spectrum<TransportScalar>(), -0.25F);
    ASSERT_FALSE(invalid_weight.has_value());
    EXPECT_EQ(invalid_weight.error().code, core::StatusCode::invalid_argument);
}

TEST(DirectLightingTest, ReportsAnUnrepresentableContributionInsteadOfOverflowing) {
    auto radiance = unit_spectrum<TransportScalar>();
    radiance.values.fill(std::numeric_limits<TransportScalar>::max());
    const auto incident = point_sample<TransportScalar>(
        Point3{.z = 1.0F}, radiance, std::numeric_limits<TransportScalar>::denorm_min());
    const auto reflection = LambertianReflection::create(unit_spectrum<TransportScalar>()).value();

    const auto result = evaluate_lambertian_direct_lighting(
        unit_spectrum<TransportScalar>(), reflection, upward_frame<TransportScalar>(),
        Vector3{.z = 1.0F}, selection_probability<TransportScalar>(1U), incident,
        unit_spectrum<TransportScalar>());

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(result.error().message.empty());
}

template <SpectrumScalar Scalar> void expect_representable_result_survives_opposed_extremes() {
    auto reflectance = unit_spectrum<Scalar>();
    reflectance.values.fill(std::numeric_limits<Scalar>::denorm_min());
    auto radiance = unit_spectrum<Scalar>();
    radiance.values.fill(std::numeric_limits<Scalar>::max());
    const auto reflection = LambertianReflectionT<Scalar>::create(reflectance).value();
    const auto incident = point_sample<Scalar>(Point3T<Scalar>{.z = Scalar{1}}, radiance);

    const auto result = evaluate_lambertian_direct_lighting(
        unit_spectrum<Scalar>(), reflection, upward_frame<Scalar>(),
        Vector3T<Scalar>{.z = Scalar{1}}, selection_probability<Scalar>(1U), incident,
        unit_spectrum<Scalar>());

    ASSERT_TRUE(result.has_value()) << result.error().message;
    const auto expected = static_cast<long double>(std::numeric_limits<Scalar>::denorm_min()) *
                          static_cast<long double>(std::numeric_limits<Scalar>::max()) /
                          std::numbers::pi_v<long double>;
    ASSERT_GT(expected, 0.0L);
    for (const auto lane : result->values) {
        EXPECT_GT(lane, Scalar{0});
        EXPECT_NEAR(static_cast<long double>(lane), expected,
                    expected * (std::same_as<Scalar, TransportScalar> ? 2.0e-6L : 2.0e-14L));
    }
}

TEST(DirectLightingTest, PreservesRepresentableResultsAcrossOpposedFactorExtremes) {
    expect_representable_result_survives_opposed_extremes<TransportScalar>();
    expect_representable_result_survives_opposed_extremes<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
