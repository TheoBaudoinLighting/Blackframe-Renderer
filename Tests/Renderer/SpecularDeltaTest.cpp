#include <Blackframe/Renderer/SpecularDelta.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
using SpectrumFor = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

template <SpectrumScalar Scalar>
inline constexpr auto AnalyticTolerance =
    std::same_as<Scalar, TransportScalar> ? ReferenceScalar{3.0e-6} : ReferenceScalar{3.0e-13};

template <SpectrumScalar Scalar> [[nodiscard]] SpectrumFor<Scalar> test_spectrum() {
    return {
        .values = {Scalar{0.125}, Scalar{0.25}, Scalar{0.5}, Scalar{1}},
    };
}

template <SpectrumScalar Scalar> [[nodiscard]] SpectrumFor<Scalar> white_spectrum() {
    return {
        .values = {Scalar{1}, Scalar{1}, Scalar{1}, Scalar{1}},
    };
}

template <typename Result> void expect_invalid(const Result& result) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(result.error().message.empty());
}

template <SpectrumScalar Scalar>
void expect_scalar_near(const Scalar actual, const ReferenceScalar expected) {
    EXPECT_NEAR(static_cast<ReferenceScalar>(actual), expected, AnalyticTolerance<Scalar>);
}

template <SpectrumScalar Scalar>
void expect_vector_near(const Vector3T<Scalar> actual, const Vector3T<Scalar> expected) {
    expect_scalar_near(actual.x, static_cast<ReferenceScalar>(expected.x));
    expect_scalar_near(actual.y, static_cast<ReferenceScalar>(expected.y));
    expect_scalar_near(actual.z, static_cast<ReferenceScalar>(expected.z));
}

template <SpectrumScalar Scalar>
void expect_spectrum_near(const SpectrumFor<Scalar>& actual, const SpectrumFor<Scalar>& expected) {
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        expect_scalar_near(actual[lane], static_cast<ReferenceScalar>(expected[lane]));
    }
}

template <SpectrumScalar Scalar, typename Sample>
[[nodiscard]] SpectrumFor<Scalar> sample_throughput(const Sample& sample) {
    auto throughput = SpectrumFor<Scalar>{};
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        throughput[lane] =
            sample.value[lane] * std::abs(sample.incoming_local.z) / sample.probability.value;
    }
    return throughput;
}

template <SpectrumScalar Scalar> void expect_directional_delta_queries_are_zero() {
    const auto reflection = SpecularReflectionT<Scalar>::create(test_spectrum<Scalar>());
    const auto transmission =
        SpecularTransmissionT<Scalar>::create(test_spectrum<Scalar>(), Scalar{1}, Scalar{1.5});
    ASSERT_TRUE(reflection.has_value()) << reflection.error().message;
    ASSERT_TRUE(transmission.has_value()) << transmission.error().message;

    const auto outgoing = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto reflected = Vector3T<Scalar>{.x = Scalar{-0.6}, .z = Scalar{0.8}};
    const auto transmitted = Vector3T<Scalar>{
        .x = Scalar{-0.4},
        .z = -std::sqrt(Scalar{0.84}),
    };
    const auto arbitrary = Vector3T<Scalar>{.z = Scalar{1}};

    for (const auto incoming : std::array{reflected, arbitrary}) {
        const auto value = reflection->eval(outgoing, incoming);
        const auto probability = reflection->pdf(outgoing, incoming);
        ASSERT_TRUE(value.has_value()) << value.error().message;
        ASSERT_TRUE(probability.has_value()) << probability.error().message;
        EXPECT_EQ(*value, SpectrumFor<Scalar>{});
        EXPECT_EQ(probability->value, Scalar{0});
        EXPECT_FALSE(std::signbit(probability->value));
        EXPECT_EQ(probability->measure, ProbabilityMeasure::solid_angle);
    }

    for (const auto mode : std::array{TransportMode::radiance, TransportMode::importance}) {
        for (const auto incoming : std::array{transmitted, arbitrary}) {
            const auto value = transmission->eval(outgoing, incoming, mode);
            const auto probability = transmission->pdf(outgoing, incoming, mode);
            ASSERT_TRUE(value.has_value()) << value.error().message;
            ASSERT_TRUE(probability.has_value()) << probability.error().message;
            EXPECT_EQ(*value, SpectrumFor<Scalar>{});
            EXPECT_EQ(probability->value, Scalar{0});
            EXPECT_FALSE(std::signbit(probability->value));
            EXPECT_EQ(probability->measure, ProbabilityMeasure::solid_angle);
        }
    }
}

TEST(SpecularDeltaTest, DirectionalEvalAndPdfExcludeDiracSupportInBothPrecisions) {
    expect_directional_delta_queries_are_zero<TransportScalar>();
    expect_directional_delta_queries_are_zero<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_mirror_sampling_contract() {
    const auto spectrum = test_spectrum<Scalar>();
    const auto reflection = SpecularReflectionT<Scalar>::create(spectrum);
    ASSERT_TRUE(reflection.has_value()) << reflection.error().message;

    for (const auto outgoing : std::array{
             Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}},
             Vector3T<Scalar>{.x = Scalar{-0.3}, .y = Scalar{0.4}, .z = -std::sqrt(Scalar{0.75})},
         }) {
        const auto sampled = reflection->sample(outgoing);
        ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
        ASSERT_TRUE(sampled->has_value());
        const auto& event = **sampled;
        const auto expected_direction = Vector3T<Scalar>{
            .x = -outgoing.x,
            .y = -outgoing.y,
            .z = outgoing.z,
        };

        expect_vector_near(event.incoming_local, expected_direction);
        EXPECT_EQ(event.probability.value, Scalar{1});
        EXPECT_EQ(event.probability.measure, ProbabilityMeasure::discrete);
        EXPECT_EQ(event.lobes, ScatteringLobe::specular | ScatteringLobe::reflection);
        EXPECT_EQ(event.eta_scale_multiplier, Scalar{1});
        expect_spectrum_near(sample_throughput<Scalar>(event), spectrum);
        for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
            expect_scalar_near(event.value[lane],
                               static_cast<ReferenceScalar>(spectrum[lane]) / std::abs(outgoing.z));
        }
    }

    const auto tangent = reflection->sample(Vector3T<Scalar>{.x = Scalar{1}});
    ASSERT_TRUE(tangent.has_value()) << tangent.error().message;
    EXPECT_FALSE(tangent->has_value());
}

TEST(SpecularDeltaTest, MirrorSamplesExactEnergyPreservingDiscreteReflection) {
    expect_mirror_sampling_contract<TransportScalar>();
    expect_mirror_sampling_contract<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_snell_entry_and_exit() {
    const auto spectrum = test_spectrum<Scalar>();
    const auto transmission =
        SpecularTransmissionT<Scalar>::create(spectrum, Scalar{1}, Scalar{1.5});
    ASSERT_TRUE(transmission.has_value()) << transmission.error().message;

    const auto outgoing = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto entering = transmission->sample(outgoing, TransportMode::radiance);
    ASSERT_TRUE(entering.has_value()) << entering.error().message;
    ASSERT_TRUE(entering->has_value());
    const auto& entering_event = **entering;
    const auto transmitted_cosine = std::sqrt(Scalar{0.84});
    expect_vector_near(entering_event.incoming_local,
                       Vector3T<Scalar>{.x = Scalar{-0.4}, .z = -transmitted_cosine});
    EXPECT_EQ(entering_event.probability.value, Scalar{1});
    EXPECT_EQ(entering_event.probability.measure, ProbabilityMeasure::discrete);
    EXPECT_EQ(entering_event.lobes, ScatteringLobe::specular | ScatteringLobe::transmission);
    expect_scalar_near(Scalar{1} * std::sqrt(Scalar{1} - outgoing.z * outgoing.z),
                       Scalar{1.5} * std::sqrt(Scalar{1} - entering_event.incoming_local.z *
                                                               entering_event.incoming_local.z));
    expect_scalar_near(entering_event.eta_scale_multiplier, ReferenceScalar{2.25});

    const auto leaving =
        transmission->sample(entering_event.incoming_local, TransportMode::radiance);
    ASSERT_TRUE(leaving.has_value()) << leaving.error().message;
    ASSERT_TRUE(leaving->has_value());
    const auto& leaving_event = **leaving;
    expect_vector_near(leaving_event.incoming_local, outgoing);
    expect_scalar_near(leaving_event.eta_scale_multiplier, ReferenceScalar{4} / ReferenceScalar{9});
    expect_scalar_near(Scalar{1.5} * std::sqrt(Scalar{1} - entering_event.incoming_local.z *
                                                               entering_event.incoming_local.z),
                       Scalar{1} * std::sqrt(Scalar{1} - leaving_event.incoming_local.z *
                                                             leaving_event.incoming_local.z));
}

TEST(SpecularDeltaTest, TransmissionObeysSnellOnEntryAndExitInBothPrecisions) {
    expect_snell_entry_and_exit<TransportScalar>();
    expect_snell_entry_and_exit<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_equal_indices_transmit_exactly_near_tangent() {
    const auto spectrum = test_spectrum<Scalar>();
    const auto transmission = SpecularTransmissionT<Scalar>::create(spectrum, Scalar{1}, Scalar{1});
    ASSERT_TRUE(transmission.has_value()) << transmission.error().message;

    const auto outgoing = Vector3T<Scalar>{
        .x = Scalar{1},
        .z = std::sqrt(Scalar{64} * std::numeric_limits<Scalar>::epsilon()),
    };
    for (const auto mode : std::array{TransportMode::radiance, TransportMode::importance}) {
        const auto sampled = transmission->sample(outgoing, mode);
        ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
        ASSERT_TRUE(sampled->has_value());
        const auto& event = **sampled;
        EXPECT_EQ(event.incoming_local, (Vector3T<Scalar>{.x = -outgoing.x, .z = -outgoing.z}));
        EXPECT_EQ(event.probability.value, Scalar{1});
        EXPECT_EQ(event.probability.measure, ProbabilityMeasure::discrete);
        EXPECT_EQ(event.lobes, ScatteringLobe::specular | ScatteringLobe::transmission);
        EXPECT_EQ(event.eta_scale_multiplier, Scalar{1});
        expect_spectrum_near(sample_throughput<Scalar>(event), spectrum);
    }
}

TEST(SpecularDeltaTest, EqualIndicesTransmitExactlyNearTangentInBothPrecisions) {
    expect_equal_indices_transmit_exactly_near_tangent<TransportScalar>();
    expect_equal_indices_transmit_exactly_near_tangent<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_local_frame_roundoff_preserves_snell_support() {
    const auto spectrum = test_spectrum<Scalar>();
    const auto transmission =
        SpecularTransmissionT<Scalar>::create(spectrum, Scalar{1}, Scalar{1.5});
    ASSERT_TRUE(transmission.has_value()) << transmission.error().message;

    const auto outgoing = Vector3T<Scalar>{
        .x = std::nextafter(Scalar{1}, std::numeric_limits<Scalar>::infinity()),
        .z = std::sqrt(Scalar{64} * std::numeric_limits<Scalar>::epsilon()),
    };
    ASSERT_GT(std::hypot(outgoing.x, outgoing.y), Scalar{1});
    const auto sampled = transmission->sample(outgoing, TransportMode::radiance);
    ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
    ASSERT_TRUE(sampled->has_value());
    const auto& event = **sampled;
    const auto incident_sine =
        std::sqrt((Scalar{1} - std::abs(outgoing.z)) * (Scalar{1} + std::abs(outgoing.z)));
    const auto expected_transmitted_sine = incident_sine / Scalar{1.5};
    expect_scalar_near(std::hypot(event.incoming_local.x, event.incoming_local.y),
                       static_cast<ReferenceScalar>(expected_transmitted_sine));
    expect_scalar_near(event.incoming_local.z, -static_cast<ReferenceScalar>(std::sqrt(
                                                   (Scalar{1} - expected_transmitted_sine) *
                                                   (Scalar{1} + expected_transmitted_sine))));
    expect_spectrum_near(sample_throughput<Scalar>(event), spectrum * (Scalar{4} / Scalar{9}));
}

TEST(SpecularDeltaTest, LocalFrameRoundoffPreservesSnellSupportInBothPrecisions) {
    expect_local_frame_roundoff_preserves_snell_support<TransportScalar>();
    expect_local_frame_roundoff_preserves_snell_support<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_transport_mode_contract() {
    const auto spectrum = test_spectrum<Scalar>();
    const auto transmission =
        SpecularTransmissionT<Scalar>::create(spectrum, Scalar{1}, Scalar{1.5});
    ASSERT_TRUE(transmission.has_value()) << transmission.error().message;
    const auto outgoing = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};

    const auto radiance = transmission->sample(outgoing, TransportMode::radiance);
    const auto importance = transmission->sample(outgoing, TransportMode::importance);
    ASSERT_TRUE(radiance.has_value()) << radiance.error().message;
    ASSERT_TRUE(importance.has_value()) << importance.error().message;
    ASSERT_TRUE(radiance->has_value());
    ASSERT_TRUE(importance->has_value());
    const auto& radiance_event = **radiance;
    const auto& importance_event = **importance;

    EXPECT_EQ(radiance_event.incoming_local, importance_event.incoming_local);
    EXPECT_EQ(radiance_event.probability.value, importance_event.probability.value);
    EXPECT_EQ(radiance_event.probability.measure, importance_event.probability.measure);
    EXPECT_EQ(radiance_event.lobes, importance_event.lobes);
    expect_scalar_near(radiance_event.eta_scale_multiplier, ReferenceScalar{2.25});
    expect_scalar_near(importance_event.eta_scale_multiplier, ReferenceScalar{1});

    const auto radiance_throughput = sample_throughput<Scalar>(radiance_event);
    const auto importance_throughput = sample_throughput<Scalar>(importance_event);
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        expect_scalar_near(radiance_throughput[lane],
                           static_cast<ReferenceScalar>(spectrum[lane]) *
                               (ReferenceScalar{4} / ReferenceScalar{9}));
        expect_scalar_near(importance_throughput[lane],
                           static_cast<ReferenceScalar>(spectrum[lane]));
        expect_scalar_near(radiance_throughput[lane] * radiance_event.eta_scale_multiplier,
                           static_cast<ReferenceScalar>(spectrum[lane]));
    }
}

TEST(SpecularDeltaTest, TransmissionSeparatesRadianceAdjointFromImportance) {
    expect_transport_mode_contract<TransportScalar>();
    expect_transport_mode_contract<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_round_trip_compensation() {
    const auto transmission =
        SpecularTransmissionT<Scalar>::create(white_spectrum<Scalar>(), Scalar{1}, Scalar{1.5});
    ASSERT_TRUE(transmission.has_value()) << transmission.error().message;
    const auto outgoing = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto entering = transmission->sample(outgoing, TransportMode::radiance);
    ASSERT_TRUE(entering.has_value()) << entering.error().message;
    ASSERT_TRUE(entering->has_value());
    const auto leaving = transmission->sample((**entering).incoming_local, TransportMode::radiance);
    ASSERT_TRUE(leaving.has_value()) << leaving.error().message;
    ASSERT_TRUE(leaving->has_value());

    const auto entering_throughput = sample_throughput<Scalar>(**entering);
    const auto leaving_throughput = sample_throughput<Scalar>(**leaving);
    const auto eta_scale = (**entering).eta_scale_multiplier * (**leaving).eta_scale_multiplier;
    expect_scalar_near(eta_scale, ReferenceScalar{1});
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        const auto beta = entering_throughput[lane] * leaving_throughput[lane];
        expect_scalar_near(beta, ReferenceScalar{1});
        expect_scalar_near(beta * eta_scale, ReferenceScalar{1});
    }
}

TEST(SpecularDeltaTest, RadianceBetaAndEtaScaleRoundTripExactlyCompensate) {
    expect_round_trip_compensation<TransportScalar>();
    expect_round_trip_compensation<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_total_internal_reflection_has_no_fallback() {
    const auto white = white_spectrum<Scalar>();
    const auto dense_to_thin = SpecularTransmissionT<Scalar>::create(white, Scalar{1.5}, Scalar{1});
    ASSERT_TRUE(dense_to_thin.has_value()) << dense_to_thin.error().message;
    const auto total_internal_reflection = dense_to_thin->sample(
        Vector3T<Scalar>{.x = Scalar{0.8}, .z = Scalar{0.6}}, TransportMode::radiance);
    ASSERT_TRUE(total_internal_reflection.has_value()) << total_internal_reflection.error().message;
    EXPECT_FALSE(total_internal_reflection->has_value());

    const auto critical_outgoing =
        Vector3T<Scalar>{.x = std::sqrt(Scalar{0.4375}), .z = Scalar{0.75}};
    const auto critical_transmitted_eta = critical_outgoing.x;
    const auto critical =
        SpecularTransmissionT<Scalar>::create(white, Scalar{1}, critical_transmitted_eta);
    ASSERT_TRUE(critical.has_value()) << critical.error().message;
    for (const auto mode : std::array{TransportMode::radiance, TransportMode::importance}) {
        const auto sampled = critical->sample(critical_outgoing, mode);
        ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
        EXPECT_FALSE(sampled->has_value());
    }
}

TEST(SpecularDeltaTest, TransmissionReturnsNoEventAtOrBeyondCriticalAngle) {
    expect_total_internal_reflection_has_no_fallback<TransportScalar>();
    expect_total_internal_reflection_has_no_fallback<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_subcritical_rounding_is_an_explicit_error() {
    const auto incident_eta = [] {
        if constexpr (std::same_as<Scalar, TransportScalar>) {
            return Scalar{0.0651621073F};
        }
        return Scalar{1.4873998303129339};
    }();
    const auto transmitted_eta = [] {
        if constexpr (std::same_as<Scalar, TransportScalar>) {
            return Scalar{0.065099515F};
        }
        return Scalar{1.4830340531642678};
    }();
    const auto incident_sine = [] {
        if constexpr (std::same_as<Scalar, TransportScalar>) {
            return Scalar{0.999039352F};
        }
        return Scalar{0.99706482610815705};
    }();
    ASSERT_LT(incident_sine, transmitted_eta / incident_eta);
    ASSERT_GE((incident_eta / transmitted_eta) * incident_sine, Scalar{1});
    const auto outgoing = Vector3T<Scalar>{
        .x = incident_sine,
        .z = std::sqrt((Scalar{1} - incident_sine) * (Scalar{1} + incident_sine)),
    };
    const auto transmission = SpecularTransmissionT<Scalar>::create(white_spectrum<Scalar>(),
                                                                    incident_eta, transmitted_eta);
    ASSERT_TRUE(transmission.has_value()) << transmission.error().message;

    const auto sampled = transmission->sample(outgoing, TransportMode::radiance);
    expect_invalid(sampled);
}

TEST(SpecularDeltaTest, SubcriticalUnrepresentableDirectionFailsExplicitlyInBothPrecisions) {
    expect_subcritical_rounding_is_an_explicit_error<TransportScalar>();
    expect_subcritical_rounding_is_an_explicit_error<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_invalid_spectra_and_indices_rejected() {
    const auto spectrum = test_spectrum<Scalar>();
    const auto infinity = std::numeric_limits<Scalar>::infinity();
    const auto below_zero = std::nextafter(Scalar{0}, -infinity);
    const auto above_one = std::nextafter(Scalar{1}, infinity);
    for (const auto invalid :
         std::array{below_zero, above_one, std::numeric_limits<Scalar>::quiet_NaN(), infinity,
                    -infinity}) {
        for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
            auto malformed = spectrum;
            malformed[lane] = invalid;
            expect_invalid(SpecularReflectionT<Scalar>::create(malformed));
            expect_invalid(
                SpecularTransmissionT<Scalar>::create(malformed, Scalar{1}, Scalar{1.5}));
        }
    }

    for (const auto invalid_eta :
         std::array{Scalar{0}, below_zero, std::numeric_limits<Scalar>::quiet_NaN(), infinity,
                    -infinity}) {
        expect_invalid(SpecularTransmissionT<Scalar>::create(spectrum, invalid_eta, Scalar{1.5}));
        expect_invalid(SpecularTransmissionT<Scalar>::create(spectrum, Scalar{1}, invalid_eta));
    }
}

template <SpectrumScalar Scalar> void expect_invalid_queries_rejected() {
    const auto spectrum = test_spectrum<Scalar>();
    const auto reflection = SpecularReflectionT<Scalar>::create(spectrum);
    const auto transmission =
        SpecularTransmissionT<Scalar>::create(spectrum, Scalar{1}, Scalar{1.5});
    ASSERT_TRUE(reflection.has_value()) << reflection.error().message;
    ASSERT_TRUE(transmission.has_value()) << transmission.error().message;
    const auto valid = Vector3T<Scalar>{.z = Scalar{1}};
    const auto infinity = std::numeric_limits<Scalar>::infinity();
    const auto invalid_directions = std::array{
        Vector3T<Scalar>{},
        Vector3T<Scalar>{.z = Scalar{2}},
        Vector3T<Scalar>{.z = Scalar{1} + Scalar{256} * std::numeric_limits<Scalar>::epsilon()},
        Vector3T<Scalar>{.x = std::numeric_limits<Scalar>::quiet_NaN(), .z = Scalar{1}},
        Vector3T<Scalar>{.y = infinity, .z = Scalar{1}},
        Vector3T<Scalar>{.x = -infinity, .z = Scalar{1}},
    };
    for (const auto invalid : invalid_directions) {
        expect_invalid(reflection->eval(invalid, valid));
        expect_invalid(reflection->eval(valid, invalid));
        expect_invalid(reflection->pdf(invalid, valid));
        expect_invalid(reflection->pdf(valid, invalid));
        expect_invalid(reflection->sample(invalid));
        for (const auto mode : std::array{TransportMode::radiance, TransportMode::importance}) {
            expect_invalid(transmission->eval(invalid, valid, mode));
            expect_invalid(transmission->eval(valid, invalid, mode));
            expect_invalid(transmission->pdf(invalid, valid, mode));
            expect_invalid(transmission->pdf(valid, invalid, mode));
            expect_invalid(transmission->sample(invalid, mode));
        }
    }

    const auto invalid_mode = static_cast<TransportMode>(0xffU);
    expect_invalid(transmission->eval(valid, valid, invalid_mode));
    expect_invalid(transmission->pdf(valid, valid, invalid_mode));
    expect_invalid(transmission->sample(valid, invalid_mode));
}

template <SpectrumScalar Scalar> void expect_nonrepresentable_events_rejected() {
    const auto reflection = SpecularReflectionT<Scalar>::create(white_spectrum<Scalar>());
    ASSERT_TRUE(reflection.has_value()) << reflection.error().message;
    const auto nearly_tangent = Vector3T<Scalar>{
        .x = Scalar{1},
        .z = std::numeric_limits<Scalar>::denorm_min(),
    };
    expect_invalid(reflection->sample(nearly_tangent));

    const auto extreme = SpecularTransmissionT<Scalar>::create(
        white_spectrum<Scalar>(), std::numeric_limits<Scalar>::max(),
        std::numeric_limits<Scalar>::denorm_min());
    if (extreme.has_value()) {
        expect_invalid(extreme->sample(Vector3T<Scalar>{.z = Scalar{1}}, TransportMode::radiance));
    } else {
        EXPECT_EQ(extreme.error().code, core::StatusCode::invalid_argument);
        EXPECT_FALSE(extreme.error().message.empty());
    }
}

TEST(SpecularDeltaTest, RejectsInvalidAndUnrepresentableInputsWithoutSubstitution) {
    expect_invalid_spectra_and_indices_rejected<TransportScalar>();
    expect_invalid_spectra_and_indices_rejected<ReferenceScalar>();
    expect_invalid_queries_rejected<TransportScalar>();
    expect_invalid_queries_rejected<ReferenceScalar>();
    expect_nonrepresentable_events_rejected<TransportScalar>();
    expect_nonrepresentable_events_rejected<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
