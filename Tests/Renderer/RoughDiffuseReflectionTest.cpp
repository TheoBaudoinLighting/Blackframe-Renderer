#include <Blackframe/Renderer/LambertianReflection.hpp>
#include <Blackframe/Renderer/RoughDiffuseReflection.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <type_traits>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
using ReflectionFor = std::conditional_t<std::same_as<Scalar, TransportScalar>,
                                         RoughDiffuseReflection, ReferenceRoughDiffuseReflection>;

template <SpectrumScalar Scalar>
using LambertianFor = std::conditional_t<std::same_as<Scalar, TransportScalar>,
                                         LambertianReflection, ReferenceLambertianReflection>;

template <SpectrumScalar Scalar>
using SpectrumFor = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

template <SpectrumScalar Scalar>
inline constexpr auto AnalyticTolerance = std::same_as<Scalar, TransportScalar> ? 2.0e-6 : 3.0e-14;

template <typename Result> void expect_invalid(const Result& result) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(result.error().message.empty());
}

template <SpectrumScalar Scalar> [[nodiscard]] SpectrumFor<Scalar> test_spectrum() {
    return SpectrumFor<Scalar>{
        .values = {Scalar{0}, Scalar{0.25}, Scalar{0.625}, Scalar{1}},
    };
}

template <SpectrumScalar Scalar>
[[nodiscard]] SpectrumFor<Scalar> constant_spectrum(const Scalar value) {
    auto spectrum = SpectrumFor<Scalar>{};
    spectrum.values.fill(value);
    return spectrum;
}

template <SpectrumScalar Scalar>
void expect_spectrum_near(const SpectrumFor<Scalar>& actual, const SpectrumFor<Scalar>& expected,
                          const double tolerance = AnalyticTolerance<Scalar>) {
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(static_cast<double>(actual[lane]), static_cast<double>(expected[lane]),
                    tolerance);
    }
}

template <SpectrumScalar Scalar> void expect_creation_contract() {
    const auto spectrum = test_spectrum<Scalar>();
    for (const auto roughness : std::array{Scalar{0}, Scalar{0.25}, Scalar{1}}) {
        const auto reflection = ReflectionFor<Scalar>::create(spectrum, roughness);
        ASSERT_TRUE(reflection.has_value());
        EXPECT_EQ(reflection->reflectance(), spectrum);
        EXPECT_EQ(reflection->roughness(), roughness);
    }

    for (const auto invalid : std::array{
             std::numeric_limits<Scalar>::quiet_NaN(),
             std::numeric_limits<Scalar>::infinity(),
             -std::numeric_limits<Scalar>::denorm_min(),
             std::nextafter(Scalar{1}, std::numeric_limits<Scalar>::infinity()),
         }) {
        expect_invalid(ReflectionFor<Scalar>::create(spectrum, invalid));
    }

    for (const auto invalid : std::array{
             std::numeric_limits<Scalar>::quiet_NaN(),
             std::numeric_limits<Scalar>::infinity(),
             -std::numeric_limits<Scalar>::denorm_min(),
             std::nextafter(Scalar{1}, std::numeric_limits<Scalar>::infinity()),
         }) {
        auto malformed = spectrum;
        malformed[2] = invalid;
        expect_invalid(ReflectionFor<Scalar>::create(malformed, Scalar{0.5}));
    }
}

TEST(RoughDiffuseReflectionTest, CreatesOnlyFiniteBoundedSpectralParameters) {
    static_assert(std::same_as<decltype(RoughDiffuseReflection::create(TransportSpectrum{},
                                                                       TransportScalar{})),
                               core::Result<RoughDiffuseReflection>>);
    static_assert(std::same_as<decltype(ReferenceRoughDiffuseReflection::create(ReferenceSpectrum{},
                                                                                ReferenceScalar{})),
                               core::Result<ReferenceRoughDiffuseReflection>>);
    static_assert(!std::same_as<RoughDiffuseReflection, ReferenceRoughDiffuseReflection>);
    static_assert(!std::same_as<RoughDiffuseSample, ReferenceRoughDiffuseSample>);

    expect_creation_contract<TransportScalar>();
    expect_creation_contract<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_lambertian_limit() {
    const auto spectrum = test_spectrum<Scalar>();
    const auto rough = ReflectionFor<Scalar>::create(spectrum, Scalar{0});
    const auto lambert = LambertianFor<Scalar>::create(spectrum);
    ASSERT_TRUE(rough.has_value());
    ASSERT_TRUE(lambert.has_value());

    const auto outgoing = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto incoming = Vector3T<Scalar>{
        .x = Scalar{-0.3}, .y = Scalar{0.4}, .z = static_cast<Scalar>(std::sqrt(0.75L))};
    const auto rough_value = rough->eval(outgoing, incoming);
    const auto lambert_value = lambert->eval(outgoing, incoming);
    const auto rough_pdf = rough->pdf(outgoing, incoming);
    const auto lambert_pdf = lambert->pdf(outgoing, incoming);
    ASSERT_TRUE(rough_value.has_value());
    ASSERT_TRUE(lambert_value.has_value());
    ASSERT_TRUE(rough_pdf.has_value());
    ASSERT_TRUE(lambert_pdf.has_value());
    EXPECT_EQ(*rough_value, *lambert_value);
    EXPECT_EQ(rough_pdf->value, lambert_pdf->value);
    EXPECT_EQ(rough_pdf->measure, lambert_pdf->measure);

    const auto canonical = Point2T<Scalar>{.x = Scalar{0.625}, .y = Scalar{0.25}};
    const auto rough_sample = rough->sample(outgoing, canonical);
    const auto lambert_sample = lambert->sample(outgoing, canonical);
    ASSERT_TRUE(rough_sample.has_value());
    ASSERT_TRUE(rough_sample->has_value());
    ASSERT_TRUE(lambert_sample.has_value());
    ASSERT_TRUE(lambert_sample->has_value());
    EXPECT_EQ((**rough_sample).incoming_local, (**lambert_sample).incoming_local);
    EXPECT_EQ((**rough_sample).value, (**lambert_sample).value);
    EXPECT_EQ((**rough_sample).probability.value, (**lambert_sample).probability.value);
    EXPECT_EQ((**rough_sample).probability.measure, (**lambert_sample).probability.measure);
}

TEST(RoughDiffuseReflectionTest, ReducesExactlyToLambertAtZeroRoughness) {
    expect_lambertian_limit<TransportScalar>();
    expect_lambertian_limit<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_reciprocal_analytic_value() {
    const auto spectrum = test_spectrum<Scalar>();
    const auto reflection = ReflectionFor<Scalar>::create(spectrum, Scalar{1});
    ASSERT_TRUE(reflection.has_value());

    const auto outgoing = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto incoming = Vector3T<Scalar>{
        .x = Scalar{-0.3}, .y = Scalar{0.4}, .z = static_cast<Scalar>(std::sqrt(0.75L))};
    const auto forward = reflection->eval(outgoing, incoming);
    const auto reverse = reflection->eval(incoming, outgoing);
    ASSERT_TRUE(forward.has_value());
    ASSERT_TRUE(reverse.has_value());
    expect_spectrum_near(*forward, *reverse);

    constexpr auto c1 = Scalar{0.5} - Scalar{2} * std::numbers::inv_pi_v<Scalar> / Scalar{3};
    constexpr auto c2 =
        Scalar{2} / Scalar{3} - Scalar{28} * std::numbers::inv_pi_v<Scalar> / Scalar{15};
    const auto a = Scalar{1} / (Scalar{1} + c1);
    const auto average_albedo = a * (Scalar{1} + c2);
    const auto average_loss = Scalar{1} - average_albedo;
    const auto normal_loss = Scalar{1} - a;
    const auto normal = Vector3T<Scalar>{.z = Scalar{1}};
    const auto actual_normal = reflection->eval(normal, normal);
    ASSERT_TRUE(actual_normal.has_value());
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        const auto rho = spectrum[lane];
        const auto multiple_scatter_albedo =
            rho * rho * average_albedo / (Scalar{1} - rho * average_loss);
        const auto expected =
            std::numbers::inv_pi_v<Scalar> *
            (rho * a + multiple_scatter_albedo * normal_loss * normal_loss / average_loss);
        EXPECT_NEAR(static_cast<double>((*actual_normal)[lane]), static_cast<double>(expected),
                    AnalyticTolerance<Scalar>);
    }
}

TEST(RoughDiffuseReflectionTest, EvaluatesAReciprocalEnergyCompensatedBrdf) {
    expect_reciprocal_analytic_value<TransportScalar>();
    expect_reciprocal_analytic_value<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_sample_and_pdf_contract() {
    const auto reflection = ReflectionFor<Scalar>::create(test_spectrum<Scalar>(), Scalar{0.75});
    ASSERT_TRUE(reflection.has_value());
    const auto outgoing = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto canonical = Point2T<Scalar>{.x = Scalar{0.75}, .y = Scalar{0.125}};
    const auto expected_direction = map_cosine_hemisphere(canonical);
    const auto sampled = reflection->sample(outgoing, canonical);
    ASSERT_TRUE(expected_direction.has_value());
    ASSERT_TRUE(sampled.has_value());
    ASSERT_TRUE(sampled->has_value());
    EXPECT_EQ((**sampled).incoming_local, *expected_direction);
    EXPECT_EQ((**sampled).probability.measure, ProbabilityMeasure::solid_angle);
    EXPECT_EQ((**sampled).probability.value,
              expected_direction->z * std::numbers::inv_pi_v<Scalar>);

    const auto queried_value = reflection->eval(outgoing, *expected_direction);
    const auto queried_pdf = reflection->pdf(outgoing, *expected_direction);
    ASSERT_TRUE(queried_value.has_value());
    ASSERT_TRUE(queried_pdf.has_value());
    EXPECT_EQ((**sampled).value, *queried_value);
    EXPECT_EQ((**sampled).probability.value, queried_pdf->value);

    const auto replay = reflection->sample(outgoing, canonical);
    ASSERT_TRUE(replay.has_value());
    ASSERT_TRUE(replay->has_value());
    EXPECT_EQ((**sampled).incoming_local, (**replay).incoming_local);
    EXPECT_EQ((**sampled).value, (**replay).value);
    EXPECT_EQ((**sampled).probability.value, (**replay).probability.value);
}

TEST(RoughDiffuseReflectionTest, SamplesTheDeclaredCosineSolidAngleDistribution) {
    expect_sample_and_pdf_contract<TransportScalar>();
    expect_sample_and_pdf_contract<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_pdf_integrates_to_one() {
    const auto reflection = ReflectionFor<Scalar>::create(test_spectrum<Scalar>(), Scalar{0.75});
    ASSERT_TRUE(reflection.has_value());
    const auto outgoing = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    constexpr auto cosine_steps = std::size_t{128};
    constexpr auto azimuth_steps = std::size_t{32};
    constexpr auto delta_cosine = 2.0L / static_cast<long double>(cosine_steps);
    constexpr auto delta_azimuth =
        2.0L * std::numbers::pi_v<long double> / static_cast<long double>(azimuth_steps);
    auto integral = 0.0L;
    for (auto cosine_index = std::size_t{}; cosine_index < cosine_steps; ++cosine_index) {
        const auto cosine = -1.0L + (static_cast<long double>(cosine_index) + 0.5L) * delta_cosine;
        const auto sine = std::sqrt((1.0L - cosine) * (1.0L + cosine));
        for (auto azimuth_index = std::size_t{}; azimuth_index < azimuth_steps; ++azimuth_index) {
            const auto azimuth = (static_cast<long double>(azimuth_index) + 0.5L) * delta_azimuth;
            const auto incoming = Vector3T<Scalar>{
                .x = static_cast<Scalar>(sine * std::cos(azimuth)),
                .y = static_cast<Scalar>(sine * std::sin(azimuth)),
                .z = static_cast<Scalar>(cosine),
            };
            const auto probability = reflection->pdf(outgoing, incoming);
            ASSERT_TRUE(probability.has_value());
            EXPECT_EQ(probability->measure, ProbabilityMeasure::solid_angle);
            integral += static_cast<long double>(probability->value) * delta_cosine * delta_azimuth;
        }
    }
    const auto tolerance = std::same_as<Scalar, TransportScalar> ? 8.0e-6L : 2.0e-12L;
    EXPECT_NEAR(integral, 1.0L, tolerance);
}

TEST(RoughDiffuseReflectionTest, IntegratesItsCompleteDirectionalPdfToOne) {
    expect_pdf_integrates_to_one<TransportScalar>();
    expect_pdf_integrates_to_one<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_support_and_invalid_inputs() {
    const auto reflection = ReflectionFor<Scalar>::create(test_spectrum<Scalar>(), Scalar{0.5});
    ASSERT_TRUE(reflection.has_value());
    const auto normal = Vector3T<Scalar>{.z = Scalar{1}};
    const auto below = Vector3T<Scalar>{.z = Scalar{-1}};
    const auto horizon = Vector3T<Scalar>{.x = Scalar{1}};

    for (const auto outside : std::array{below, horizon}) {
        const auto value = reflection->eval(normal, outside);
        const auto probability = reflection->pdf(normal, outside);
        ASSERT_TRUE(value.has_value());
        ASSERT_TRUE(probability.has_value());
        EXPECT_EQ(*value, SpectrumFor<Scalar>{});
        EXPECT_EQ(probability->value, Scalar{0});
        EXPECT_EQ(probability->measure, ProbabilityMeasure::solid_angle);
    }
    const auto absent =
        reflection->sample(below, Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}});
    ASSERT_TRUE(absent.has_value());
    EXPECT_FALSE(absent->has_value());

    for (const auto malformed : std::array{
             Vector3T<Scalar>{},
             Vector3T<Scalar>{.x = Scalar{2}},
             Vector3T<Scalar>{.x = std::numeric_limits<Scalar>::quiet_NaN(), .z = Scalar{1}},
             Vector3T<Scalar>{.x = std::numeric_limits<Scalar>::infinity(), .z = Scalar{1}},
         }) {
        expect_invalid(reflection->eval(malformed, normal));
        expect_invalid(reflection->eval(normal, malformed));
        expect_invalid(reflection->pdf(malformed, normal));
        expect_invalid(reflection->pdf(normal, malformed));
        expect_invalid(
            reflection->sample(malformed, Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}}));
    }

    for (const auto malformed : std::array{
             Point2T<Scalar>{.x = Scalar{1}, .y = Scalar{0.5}},
             Point2T<Scalar>{.x = -std::numeric_limits<Scalar>::denorm_min(), .y = Scalar{0.5}},
             Point2T<Scalar>{.x = std::numeric_limits<Scalar>::quiet_NaN(), .y = Scalar{0.5}},
             Point2T<Scalar>{.x = Scalar{0.5}, .y = std::numeric_limits<Scalar>::infinity()},
         }) {
        expect_invalid(reflection->sample(normal, malformed));
    }

    const auto near_horizon = Vector3T<Scalar>{
        .x = Scalar{1},
        .z = std::numeric_limits<Scalar>::denorm_min(),
    };
    const auto unrepresentable = reflection->eval(near_horizon, near_horizon);
    expect_invalid(unrepresentable);
    const auto black =
        ReflectionFor<Scalar>::create(constant_spectrum<Scalar>(Scalar{0}), Scalar{1});
    ASSERT_TRUE(black.has_value());
    const auto black_value = black->eval(near_horizon, near_horizon);
    ASSERT_TRUE(black_value.has_value());
    EXPECT_EQ(*black_value, SpectrumFor<Scalar>{});
}

TEST(RoughDiffuseReflectionTest, KeepsOneSidedSupportAndRejectsMalformedInputs) {
    expect_support_and_invalid_inputs<TransportScalar>();
    expect_support_and_invalid_inputs<ReferenceScalar>();
}

template <SpectrumScalar Scalar>
[[nodiscard]] std::array<long double, TransportSpectrumSampleCount>
integrate_hemispherical_reflectance(const ReflectionFor<Scalar>& reflection,
                                    const Vector3T<Scalar> outgoing) {
    constexpr auto cosine_steps = std::size_t{160};
    constexpr auto azimuth_steps = std::size_t{320};
    constexpr auto delta_cosine = 1.0L / static_cast<long double>(cosine_steps);
    constexpr auto delta_azimuth =
        2.0L * std::numbers::pi_v<long double> / static_cast<long double>(azimuth_steps);
    auto integral = std::array<long double, TransportSpectrumSampleCount>{};

    for (auto cosine_index = std::size_t{}; cosine_index < cosine_steps; ++cosine_index) {
        const auto cosine = (static_cast<long double>(cosine_index) + 0.5L) * delta_cosine;
        const auto sine = std::sqrt((1.0L - cosine) * (1.0L + cosine));
        for (auto azimuth_index = std::size_t{}; azimuth_index < azimuth_steps; ++azimuth_index) {
            const auto azimuth = (static_cast<long double>(azimuth_index) + 0.5L) * delta_azimuth;
            const auto incoming = Vector3T<Scalar>{
                .x = static_cast<Scalar>(sine * std::cos(azimuth)),
                .y = static_cast<Scalar>(sine * std::sin(azimuth)),
                .z = static_cast<Scalar>(cosine),
            };
            const auto value = reflection.eval(outgoing, incoming);
            EXPECT_TRUE(value.has_value());
            if (!value.has_value()) {
                return {};
            }
            for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
                integral[lane] += static_cast<long double>((*value)[lane]) * cosine * delta_cosine *
                                  delta_azimuth;
            }
        }
    }
    return integral;
}

template <SpectrumScalar Scalar> void expect_white_furnace() {
    const auto white = constant_spectrum<Scalar>(Scalar{1});
    const auto outgoing_directions = std::array{
        Vector3T<Scalar>{.z = Scalar{1}},
        Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}},
        Vector3T<Scalar>{.x = static_cast<Scalar>(std::sqrt(0.96L)), .z = Scalar{0.2}},
    };
    for (const auto roughness : std::array{Scalar{0.25}, Scalar{1}}) {
        const auto reflection = ReflectionFor<Scalar>::create(white, roughness);
        ASSERT_TRUE(reflection.has_value());
        for (const auto outgoing : outgoing_directions) {
            const auto integral = integrate_hemispherical_reflectance(*reflection, outgoing);
            for (const auto lane : integral) {
                EXPECT_NEAR(lane, 1.0L, 1.5e-4L);
            }
        }
    }
}

TEST(RoughDiffuseReflectionTest, PreservesWhiteEnergyAcrossRoughnessAndViewAngle) {
    expect_white_furnace<TransportScalar>();
    expect_white_furnace<ReferenceScalar>();
}

static_assert(std::is_standard_layout_v<RoughDiffuseSample>);
static_assert(std::is_trivially_copyable_v<RoughDiffuseSample>);
static_assert(std::is_standard_layout_v<ReferenceRoughDiffuseSample>);
static_assert(std::is_trivially_copyable_v<ReferenceRoughDiffuseSample>);
static_assert(std::is_standard_layout_v<RoughDiffuseReflection>);
static_assert(std::is_trivially_copyable_v<RoughDiffuseReflection>);
static_assert(std::is_standard_layout_v<ReferenceRoughDiffuseReflection>);
static_assert(std::is_trivially_copyable_v<ReferenceRoughDiffuseReflection>);

} // namespace
} // namespace blackframe::renderer
