#include <Blackframe/Renderer/LambertianReflection.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
using ReflectionFor = std::conditional_t<std::is_same_v<Scalar, TransportScalar>,
                                         LambertianReflection, ReferenceLambertianReflection>;

template <SpectrumScalar Scalar>
using SpectrumFor = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

inline constexpr std::string_view ReflectanceError =
    "Lambertian reflectance requires every spectral lane to be finite and in [0, 1].";
inline constexpr std::string_view DirectionError =
    "Lambertian directions must be finite unit vectors.";
inline constexpr std::string_view PdfRepresentationError =
    "Lambertian PDF is not representable for the supplied direction.";

template <SpectrumScalar Scalar>
inline constexpr auto AnalyticTolerance =
    std::is_same_v<Scalar, TransportScalar> ? ReferenceScalar{2.0e-6} : ReferenceScalar{2.0e-13};

template <SpectrumScalar Scalar> [[nodiscard]] SpectrumFor<Scalar> mixed_reflectance() {
    return {
        .values =
            {
                Scalar{0},
                Scalar{0.25},
                Scalar{0.5},
                Scalar{1},
            },
    };
}

template <typename Result>
void expect_invalid(const Result& result, const std::string_view expected_message) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_EQ(result.error().message, expected_message);
}

template <SpectrumScalar Scalar> void expect_bounded_reflectance_creation() {
    const auto reflectance = mixed_reflectance<Scalar>();
    const auto lambertian = ReflectionFor<Scalar>::create(reflectance);
    ASSERT_TRUE(lambertian.has_value());
    EXPECT_EQ(lambertian->reflectance(), reflectance);

    const auto signed_zero = ReflectionFor<Scalar>::create(SpectrumFor<Scalar>{
        .values = {-Scalar{0}, Scalar{0}, Scalar{1}, Scalar{1}},
    });
    ASSERT_TRUE(signed_zero.has_value());
    EXPECT_EQ(signed_zero->reflectance()[1], Scalar{0});
    EXPECT_EQ(signed_zero->reflectance()[2], Scalar{1});

    const auto infinity = std::numeric_limits<Scalar>::infinity();
    const auto below_zero = std::nextafter(Scalar{0}, -infinity);
    const auto above_one = std::nextafter(Scalar{1}, infinity);
    for (const auto invalid :
         std::array{below_zero, above_one, std::numeric_limits<Scalar>::quiet_NaN(), infinity,
                    -infinity}) {
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            auto malformed = reflectance;
            malformed[lane] = invalid;
            expect_invalid(ReflectionFor<Scalar>::create(malformed), ReflectanceError);
        }
    }
}

TEST(LambertianReflectionTest, CreatesBoundedSpectralReflectanceWithoutClamping) {
    static_assert(std::same_as<decltype(LambertianReflection::create(TransportSpectrum{})),
                               core::Result<LambertianReflection>>);
    static_assert(std::same_as<decltype(ReferenceLambertianReflection::create(ReferenceSpectrum{})),
                               core::Result<ReferenceLambertianReflection>>);
    static_assert(std::same_as<decltype(std::declval<const LambertianReflection&>().eval(
                                   std::declval<Vector3>(), std::declval<Vector3>())),
                               core::Result<TransportSpectrum>>);
    static_assert(
        std::same_as<decltype(std::declval<const ReferenceLambertianReflection&>().pdf(
                         std::declval<ReferenceVector3>(), std::declval<ReferenceVector3>())),
                     core::Result<ReferenceProbabilityDensity>>);
    static_assert(std::same_as<decltype(std::declval<const LambertianReflection&>().sample(
                                   std::declval<Vector3>(), std::declval<Point2>())),
                               core::Result<std::optional<LambertianSample>>>);
    static_assert(!std::same_as<LambertianReflection, ReferenceLambertianReflection>);
    static_assert(!std::same_as<LambertianSample, ReferenceLambertianSample>);

    expect_bounded_reflectance_creation<TransportScalar>();
    expect_bounded_reflectance_creation<ReferenceScalar>();
}

template <SpectrumScalar Scalar>
void expect_spectrum_near(const SpectrumFor<Scalar>& actual, const SpectrumFor<Scalar>& expected) {
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(static_cast<ReferenceScalar>(actual[lane]),
                    static_cast<ReferenceScalar>(expected[lane]), AnalyticTolerance<Scalar>);
    }
}

template <SpectrumScalar Scalar> void expect_eval_and_pdf_contract() {
    const auto reflectance = mixed_reflectance<Scalar>();
    const auto lambertian = ReflectionFor<Scalar>::create(reflectance);
    ASSERT_TRUE(lambertian.has_value());

    const auto normal = Vector3T<Scalar>{.z = Scalar{1}};
    const auto oblique = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto azimuthal = Vector3T<Scalar>{
        .x = Scalar{-0.3},
        .y = Scalar{0.4},
        .z = std::sqrt(Scalar{0.75}),
    };
    const auto expected = reflectance * std::numbers::inv_pi_v<Scalar>;
    for (const auto incoming : std::array{normal, oblique, azimuthal}) {
        const auto evaluated = lambertian->eval(normal, incoming);
        ASSERT_TRUE(evaluated.has_value());
        expect_spectrum_near(*evaluated, expected);

        const auto reciprocal = lambertian->eval(incoming, normal);
        ASSERT_TRUE(reciprocal.has_value());
        EXPECT_EQ(*reciprocal, *evaluated);

        const auto probability = lambertian->pdf(normal, incoming);
        ASSERT_TRUE(probability.has_value());
        EXPECT_EQ(probability->measure, ProbabilityMeasure::solid_angle);
        EXPECT_NEAR(static_cast<ReferenceScalar>(probability->value),
                    static_cast<ReferenceScalar>(incoming.z) / std::numbers::pi_v<ReferenceScalar>,
                    AnalyticTolerance<Scalar>);
    }

    const auto black = ReflectionFor<Scalar>::create(SpectrumFor<Scalar>{});
    ASSERT_TRUE(black.has_value());
    const auto black_pdf = black->pdf(normal, oblique);
    const auto colored_pdf = lambertian->pdf(normal, oblique);
    ASSERT_TRUE(black_pdf.has_value());
    ASSERT_TRUE(colored_pdf.has_value());
    EXPECT_EQ(black_pdf->value, colored_pdf->value);

    const auto rounded_axis = Vector3T<Scalar>{
        .z = std::nextafter(Scalar{1}, std::numeric_limits<Scalar>::infinity()),
    };
    const auto rounded_eval = lambertian->eval(rounded_axis, rounded_axis);
    const auto rounded_pdf = lambertian->pdf(rounded_axis, rounded_axis);
    ASSERT_TRUE(rounded_eval.has_value());
    ASSERT_TRUE(rounded_pdf.has_value());
    EXPECT_EQ(*rounded_eval, expected);
    EXPECT_GT(rounded_pdf->value, Scalar{0});

    const auto unrepresentable_pdf_direction = Vector3T<Scalar>{
        .x = Scalar{1},
        .z = std::numeric_limits<Scalar>::denorm_min(),
    };
    expect_invalid(lambertian->pdf(normal, unrepresentable_pdf_direction), PdfRepresentationError);

    const auto horizon = Vector3T<Scalar>{.x = Scalar{1}};
    const auto below = Vector3T<Scalar>{.z = Scalar{-1}};
    for (const auto directions : std::array{
             std::pair{normal, horizon},
             std::pair{horizon, normal},
             std::pair{normal, below},
             std::pair{below, normal},
             std::pair{below, below},
         }) {
        const auto evaluated = lambertian->eval(directions.first, directions.second);
        const auto probability = lambertian->pdf(directions.first, directions.second);
        ASSERT_TRUE(evaluated.has_value());
        ASSERT_TRUE(probability.has_value());
        EXPECT_EQ(*evaluated, SpectrumFor<Scalar>{});
        EXPECT_EQ(probability->value, Scalar{0});
        EXPECT_FALSE(std::signbit(probability->value));
        EXPECT_EQ(probability->measure, ProbabilityMeasure::solid_angle);
    }
}

TEST(LambertianReflectionTest, EvaluatesReciprocalRhoOverPiAndSolidAnglePdf) {
    expect_eval_and_pdf_contract<TransportScalar>();
    expect_eval_and_pdf_contract<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_sampling_contract() {
    const auto reflectance = mixed_reflectance<Scalar>();
    const auto lambertian = ReflectionFor<Scalar>::create(reflectance);
    ASSERT_TRUE(lambertian.has_value());
    const auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};

    for (const auto canonical_sample : std::array{
             Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}},
             Point2T<Scalar>{.x = Scalar{0.75}, .y = Scalar{0.5}},
         }) {
        const auto sampled = lambertian->sample(outgoing, canonical_sample);
        ASSERT_TRUE(sampled.has_value());
        ASSERT_TRUE(sampled->has_value());
        const auto& event = **sampled;

        const auto mapped = map_cosine_hemisphere(canonical_sample);
        const auto evaluated = lambertian->eval(outgoing, event.incoming_local);
        const auto probability = lambertian->pdf(outgoing, event.incoming_local);
        ASSERT_TRUE(mapped.has_value());
        ASSERT_TRUE(evaluated.has_value());
        ASSERT_TRUE(probability.has_value());
        EXPECT_EQ(event.incoming_local, *mapped);
        EXPECT_EQ(event.value, *evaluated);
        EXPECT_EQ(event.probability.value, probability->value);
        EXPECT_EQ(event.probability.measure, ProbabilityMeasure::solid_angle);
        EXPECT_GT(event.incoming_local.z, Scalar{0});
        EXPECT_GT(event.probability.value, Scalar{0});
        EXPECT_NEAR(static_cast<ReferenceScalar>(length_squared(event.incoming_local)), 1.0,
                    AnalyticTolerance<Scalar>);
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            const auto throughput =
                event.value[lane] * event.incoming_local.z / event.probability.value;
            EXPECT_NEAR(static_cast<ReferenceScalar>(throughput),
                        static_cast<ReferenceScalar>(reflectance[lane]),
                        Scalar{32} * std::numeric_limits<Scalar>::epsilon());
        }
    }

    constexpr auto grid_size = std::size_t{32};
    for (auto y = std::size_t{0}; y < grid_size; ++y) {
        for (auto x = std::size_t{0}; x < grid_size; ++x) {
            const auto canonical_sample = Point2T<Scalar>{
                .x = (static_cast<Scalar>(x) + Scalar{0.5}) / static_cast<Scalar>(grid_size),
                .y = (static_cast<Scalar>(y) + Scalar{0.5}) / static_cast<Scalar>(grid_size),
            };
            const auto sampled = lambertian->sample(outgoing, canonical_sample);
            ASSERT_TRUE(sampled.has_value());
            ASSERT_TRUE(sampled->has_value());
            EXPECT_GT((**sampled).incoming_local.z, Scalar{0});
            EXPECT_GT((**sampled).probability.value, Scalar{0});
        }
    }

    const auto horizon_sample = lambertian->sample(outgoing, Point2T<Scalar>{});
    ASSERT_TRUE(horizon_sample.has_value());
    EXPECT_FALSE(horizon_sample->has_value());

    const auto below_sample = lambertian->sample(
        Vector3T<Scalar>{.z = Scalar{-1}}, Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}});
    ASSERT_TRUE(below_sample.has_value());
    EXPECT_FALSE(below_sample->has_value());

    const auto horizon_outgoing_sample = lambertian->sample(
        Vector3T<Scalar>{.x = Scalar{1}}, Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}});
    ASSERT_TRUE(horizon_outgoing_sample.has_value());
    EXPECT_FALSE(horizon_outgoing_sample->has_value());

    const auto last = std::nextafter(Scalar{1}, Scalar{0});
    const auto last_sample = lambertian->sample(outgoing, Point2T<Scalar>{.x = last, .y = last});
    ASSERT_TRUE(last_sample.has_value());
    ASSERT_TRUE(last_sample->has_value());
}

TEST(LambertianReflectionTest, SamplesCosineHemisphereWithMatchingEvalAndPdf) {
    expect_sampling_contract<TransportScalar>();
    expect_sampling_contract<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_invalid_inputs_rejected() {
    const auto lambertian = ReflectionFor<Scalar>::create(mixed_reflectance<Scalar>());
    ASSERT_TRUE(lambertian.has_value());
    const auto valid = Vector3T<Scalar>{.z = Scalar{1}};
    const auto infinity = std::numeric_limits<Scalar>::infinity();
    const auto invalid_directions = std::array{
        Vector3T<Scalar>{},
        Vector3T<Scalar>{.z = Scalar{2}},
        Vector3T<Scalar>{
            .z = Scalar{1} + Scalar{256} * std::numeric_limits<Scalar>::epsilon(),
        },
        Vector3T<Scalar>{.x = std::numeric_limits<Scalar>::quiet_NaN(), .z = Scalar{1}},
        Vector3T<Scalar>{.y = infinity, .z = Scalar{1}},
        Vector3T<Scalar>{.x = -infinity, .z = Scalar{1}},
    };
    for (const auto invalid : invalid_directions) {
        expect_invalid(lambertian->eval(invalid, valid), DirectionError);
        expect_invalid(lambertian->eval(valid, invalid), DirectionError);
        expect_invalid(lambertian->pdf(invalid, valid), DirectionError);
        expect_invalid(lambertian->pdf(valid, invalid), DirectionError);
        expect_invalid(
            lambertian->sample(invalid, Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{0.5}}),
            DirectionError);
    }

    const auto invalid_samples = std::array{
        Point2T<Scalar>{.x = Scalar{-0.25}, .y = Scalar{0.5}},
        Point2T<Scalar>{.x = Scalar{1}, .y = Scalar{0.5}},
        Point2T<Scalar>{.x = Scalar{0.5}, .y = Scalar{1}},
        Point2T<Scalar>{
            .x = std::numeric_limits<Scalar>::quiet_NaN(),
            .y = Scalar{0.5},
        },
        Point2T<Scalar>{.x = Scalar{0.5}, .y = infinity},
        Point2T<Scalar>{.x = Scalar{0.5}, .y = -infinity},
    };
    for (const auto invalid : invalid_samples) {
        const auto sampled = lambertian->sample(valid, invalid);
        ASSERT_FALSE(sampled.has_value());
        EXPECT_EQ(sampled.error().code, core::StatusCode::invalid_argument);
        EXPECT_EQ(sampled.error().message,
                  "Cosine hemisphere mapping requires finite coordinates in the half-open unit "
                  "square [0, 1).");
    }

    const auto invalid_without_support =
        lambertian->sample(Vector3T<Scalar>{.z = Scalar{-1}}, invalid_samples.front());
    ASSERT_FALSE(invalid_without_support.has_value());
    EXPECT_EQ(invalid_without_support.error().code, core::StatusCode::invalid_argument);
}

TEST(LambertianReflectionTest, RejectsMalformedInputsWithoutSubstitution) {
    expect_invalid_inputs_rejected<TransportScalar>();
    expect_invalid_inputs_rejected<ReferenceScalar>();
}

template <SpectrumScalar Scalar>
void integrate_lambertian(const SpectrumFor<Scalar>& reflectance, const Vector3T<Scalar> outgoing) {
    const auto lambertian = ReflectionFor<Scalar>::create(reflectance);
    ASSERT_TRUE(lambertian.has_value());

    constexpr auto cosine_steps = std::size_t{128};
    constexpr auto azimuth_steps = std::size_t{32};
    constexpr auto delta_cosine = 1.0L / static_cast<long double>(cosine_steps);
    constexpr auto delta_azimuth =
        2.0L * std::numbers::pi_v<long double> / static_cast<long double>(azimuth_steps);
    auto integral = std::array<long double, TransportSpectrumSampleCount>{};

    for (auto cosine_index = std::size_t{0}; cosine_index < cosine_steps; ++cosine_index) {
        const auto cosine = (static_cast<long double>(cosine_index) + 0.5L) * delta_cosine;
        const auto radial = std::sqrt((1.0L - cosine) * (1.0L + cosine));
        for (auto azimuth_index = std::size_t{0}; azimuth_index < azimuth_steps; ++azimuth_index) {
            const auto azimuth = (static_cast<long double>(azimuth_index) + 0.5L) * delta_azimuth;
            const auto incoming = Vector3T<Scalar>{
                .x = static_cast<Scalar>(radial * std::cos(azimuth)),
                .y = static_cast<Scalar>(radial * std::sin(azimuth)),
                .z = static_cast<Scalar>(cosine),
            };
            const auto evaluated = lambertian->eval(outgoing, incoming);
            ASSERT_TRUE(evaluated.has_value());
            for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
                integral[lane] += static_cast<long double>((*evaluated)[lane]) * cosine *
                                  delta_cosine * delta_azimuth;
            }
        }
    }

    const auto tolerance = std::is_same_v<Scalar, TransportScalar> ? 2.0e-6L : 2.0e-12L;
    for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(integral[lane], static_cast<long double>(reflectance[lane]), tolerance);
    }
}

template <SpectrumScalar Scalar> void expect_pdf_integrates_to_one() {
    const auto lambertian = ReflectionFor<Scalar>::create(mixed_reflectance<Scalar>());
    ASSERT_TRUE(lambertian.has_value());
    const auto outgoing = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};

    constexpr auto cosine_steps = std::size_t{256};
    constexpr auto azimuth_steps = std::size_t{32};
    constexpr auto delta_cosine = 2.0L / static_cast<long double>(cosine_steps);
    constexpr auto delta_azimuth =
        2.0L * std::numbers::pi_v<long double> / static_cast<long double>(azimuth_steps);
    auto integral = 0.0L;
    for (auto cosine_index = std::size_t{0}; cosine_index < cosine_steps; ++cosine_index) {
        const auto cosine = -1.0L + (static_cast<long double>(cosine_index) + 0.5L) * delta_cosine;
        const auto radial = std::sqrt((1.0L - cosine) * (1.0L + cosine));
        for (auto azimuth_index = std::size_t{0}; azimuth_index < azimuth_steps; ++azimuth_index) {
            const auto azimuth = (static_cast<long double>(azimuth_index) + 0.5L) * delta_azimuth;
            const auto incoming = Vector3T<Scalar>{
                .x = static_cast<Scalar>(radial * std::cos(azimuth)),
                .y = static_cast<Scalar>(radial * std::sin(azimuth)),
                .z = static_cast<Scalar>(cosine),
            };
            const auto probability = lambertian->pdf(outgoing, incoming);
            ASSERT_TRUE(probability.has_value());
            EXPECT_EQ(probability->measure, ProbabilityMeasure::solid_angle);
            integral += static_cast<long double>(probability->value) * delta_cosine * delta_azimuth;
        }
    }

    const auto tolerance = std::is_same_v<Scalar, TransportScalar> ? 2.0e-6L : 2.0e-12L;
    EXPECT_NEAR(integral, 1.0L, tolerance);
}

template <SpectrumScalar Scalar> void expect_analytic_integral_and_white_furnace() {
    const auto normal = Vector3T<Scalar>{.z = Scalar{1}};
    const auto oblique = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    for (const auto outgoing : std::array{normal, oblique}) {
        integrate_lambertian(SpectrumFor<Scalar>{}, outgoing);
        integrate_lambertian(mixed_reflectance<Scalar>(), outgoing);
        integrate_lambertian(
            SpectrumFor<Scalar>{
                .values = {Scalar{1}, Scalar{1}, Scalar{1}, Scalar{1}},
            },
            outgoing);
    }
    expect_pdf_integrates_to_one<Scalar>();
}

TEST(LambertianReflectionTest, MatchesAnalyticIntegralAndWhiteFurnaceInBothPrecisions) {
    expect_analytic_integral_and_white_furnace<TransportScalar>();
    expect_analytic_integral_and_white_furnace<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
