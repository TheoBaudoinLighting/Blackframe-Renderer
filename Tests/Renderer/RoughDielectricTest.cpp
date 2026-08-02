#include <Blackframe/Renderer/Fresnel.hpp>
#include <Blackframe/Renderer/GgxMicrofacet.hpp>
#include <Blackframe/Renderer/RoughDielectric.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
using DielectricFor = std::conditional_t<std::same_as<Scalar, TransportScalar>, RoughDielectric,
                                         ReferenceRoughDielectric>;

template <SpectrumScalar Scalar>
using GgxFor = std::conditional_t<std::same_as<Scalar, TransportScalar>, GgxMicrofacet,
                                  ReferenceGgxMicrofacet>;

template <SpectrumScalar Scalar>
using SpectrumFor = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

template <SpectrumScalar Scalar>
inline constexpr auto AnalyticTolerance =
    std::same_as<Scalar, TransportScalar> ? ReferenceScalar{1.2e-5} : ReferenceScalar{3.0e-12};

template <typename Result> void expect_invalid(const Result& result) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(result.error().message.empty());
}

template <SpectrumScalar Scalar> [[nodiscard]] SpectrumFor<Scalar> test_coefficient() {
    return {.values = {Scalar{0.125}, Scalar{0.25}, Scalar{0.5}, Scalar{1}}};
}

template <SpectrumScalar Scalar>
[[nodiscard]] SpectrumFor<Scalar> constant_spectrum(const Scalar value) {
    auto spectrum = SpectrumFor<Scalar>{};
    spectrum.values.fill(value);
    return spectrum;
}

template <SpectrumScalar Scalar>
void expect_scalar_near(const Scalar actual, const ReferenceScalar expected,
                        const ReferenceScalar scale = ReferenceScalar{1}) {
    EXPECT_NEAR(static_cast<ReferenceScalar>(actual), expected,
                AnalyticTolerance<Scalar> * std::max(ReferenceScalar{1}, std::abs(scale)));
}

template <SpectrumScalar Scalar>
void expect_spectrum_near(const SpectrumFor<Scalar>& actual, const SpectrumFor<Scalar>& expected,
                          const ReferenceScalar scale = ReferenceScalar{1}) {
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        expect_scalar_near(actual[lane], static_cast<ReferenceScalar>(expected[lane]), scale);
    }
}

template <SpectrumScalar Scalar>
[[nodiscard]] Scalar vector_dot(const Vector3T<Scalar> left, const Vector3T<Scalar> right) {
    return std::fma(left.x, right.x, std::fma(left.y, right.y, left.z * right.z));
}

template <SpectrumScalar Scalar>
[[nodiscard]] Vector3T<Scalar> scaled_vector(const Vector3T<Scalar> value, const Scalar scale) {
    return {.x = value.x * scale, .y = value.y * scale, .z = value.z * scale};
}

template <SpectrumScalar Scalar>
[[nodiscard]] Vector3T<Scalar> normalized_vector(const Vector3T<Scalar> value) {
    const auto length = std::hypot(std::hypot(value.x, value.y), value.z);
    EXPECT_TRUE(std::isfinite(length));
    EXPECT_GT(length, Scalar{0});
    return {.x = value.x / length, .y = value.y / length, .z = value.z / length};
}

template <SpectrumScalar Scalar>
[[nodiscard]] Vector3T<Scalar>
oriented_microfacet_normal(const Vector3T<Scalar> outgoing, const Vector3T<Scalar> incoming,
                           const Scalar incident_eta, const Scalar transmitted_eta,
                           const bool reflection) {
    const auto unnormalized =
        reflection ? Vector3T<Scalar>{.x = outgoing.x + incoming.x,
                                      .y = outgoing.y + incoming.y,
                                      .z = outgoing.z + incoming.z}
                   : Vector3T<Scalar>{
                         .x = incident_eta * outgoing.x + transmitted_eta * incoming.x,
                         .y = incident_eta * outgoing.y + transmitted_eta * incoming.y,
                         .z = incident_eta * outgoing.z + transmitted_eta * incoming.z,
                     };
    auto normal = normalized_vector(unnormalized);
    if (vector_dot(outgoing, normal) < Scalar{0}) {
        normal = scaled_vector(normal, Scalar{-1});
    }
    return normal;
}

template <SpectrumScalar Scalar> void expect_creation_contract() {
    const auto coefficient = test_coefficient<Scalar>();
    for (const auto alpha : std::array{Scalar{0.2}, Scalar{0.6}, Scalar{1}, Scalar{2}}) {
        const auto dielectric =
            RoughDielectricT<Scalar>::create(coefficient, Scalar{1}, Scalar{1.5}, alpha);
        ASSERT_TRUE(dielectric.has_value()) << dielectric.error().message;
        EXPECT_EQ(dielectric->alpha(), alpha);
        EXPECT_EQ(dielectric->alpha_x(), alpha);
        EXPECT_EQ(dielectric->alpha_y(), alpha);
    }
    const auto anisotropic = RoughDielectricT<Scalar>::create(coefficient, Scalar{1}, Scalar{1.5},
                                                              Scalar{0.25}, Scalar{0.75});
    ASSERT_TRUE(anisotropic.has_value()) << anisotropic.error().message;
    EXPECT_EQ(anisotropic->alpha_x(), Scalar{0.25});
    EXPECT_EQ(anisotropic->alpha_y(), Scalar{0.75});

    const auto infinity = std::numeric_limits<Scalar>::infinity();
    const auto invalid_values = std::array{
        -std::numeric_limits<Scalar>::denorm_min(),
        std::nextafter(Scalar{1}, infinity),
        std::numeric_limits<Scalar>::quiet_NaN(),
        infinity,
        -infinity,
    };
    for (const auto invalid : invalid_values) {
        auto malformed = coefficient;
        malformed[2] = invalid;
        expect_invalid(
            RoughDielectricT<Scalar>::create(malformed, Scalar{1}, Scalar{1.5}, Scalar{0.5}));
    }
    for (const auto invalid_eta :
         std::array{Scalar{0}, Scalar{-0.0}, -std::numeric_limits<Scalar>::denorm_min(),
                    std::numeric_limits<Scalar>::quiet_NaN(), infinity, -infinity}) {
        expect_invalid(
            RoughDielectricT<Scalar>::create(coefficient, invalid_eta, Scalar{1.5}, Scalar{0.5}));
        expect_invalid(
            RoughDielectricT<Scalar>::create(coefficient, Scalar{1}, invalid_eta, Scalar{0.5}));
    }
    for (const auto invalid_alpha :
         std::array{Scalar{0}, Scalar{-0.0}, -std::numeric_limits<Scalar>::denorm_min(),
                    std::numeric_limits<Scalar>::quiet_NaN(), infinity, -infinity}) {
        expect_invalid(
            RoughDielectricT<Scalar>::create(coefficient, Scalar{1}, Scalar{1.5}, invalid_alpha));
    }
    for (const auto invalid_alpha :
         std::array{Scalar{0}, Scalar{-0.0}, -std::numeric_limits<Scalar>::denorm_min(),
                    std::numeric_limits<Scalar>::quiet_NaN(), infinity, -infinity}) {
        expect_invalid(RoughDielectricT<Scalar>::create(coefficient, Scalar{1}, Scalar{1.5},
                                                        invalid_alpha, Scalar{0.5}));
        expect_invalid(RoughDielectricT<Scalar>::create(coefficient, Scalar{1}, Scalar{1.5},
                                                        Scalar{0.5}, invalid_alpha));
    }
    expect_invalid(
        RoughDielectricT<Scalar>::create(coefficient, Scalar{1.5}, Scalar{1.5}, Scalar{0.5}));
}

TEST(RoughDielectricTest, CreatesOnlyFinitePhysicalParametersAndPositiveWidth) {
    static_assert(
        std::same_as<decltype(RoughDielectric::create(TransportSpectrum{}, TransportScalar{},
                                                      TransportScalar{}, TransportScalar{})),
                     core::Result<RoughDielectric>>);
    static_assert(std::same_as<decltype(ReferenceRoughDielectric::create(
                                   ReferenceSpectrum{}, ReferenceScalar{}, ReferenceScalar{},
                                   ReferenceScalar{})),
                               core::Result<ReferenceRoughDielectric>>);
    static_assert(std::same_as<decltype(RoughDielectric::create(
                                   TransportSpectrum{}, TransportScalar{}, TransportScalar{},
                                   TransportScalar{}, TransportScalar{})),
                               core::Result<RoughDielectric>>);
    static_assert(std::same_as<decltype(ReferenceRoughDielectric::create(
                                   ReferenceSpectrum{}, ReferenceScalar{}, ReferenceScalar{},
                                   ReferenceScalar{}, ReferenceScalar{})),
                               core::Result<ReferenceRoughDielectric>>);
    static_assert(!std::same_as<RoughDielectric, ReferenceRoughDielectric>);
    expect_creation_contract<TransportScalar>();
    expect_creation_contract<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_invalid_queries_and_explicit_support() {
    const auto dielectric = RoughDielectricT<Scalar>::create(test_coefficient<Scalar>(), Scalar{1},
                                                             Scalar{1.5}, Scalar{0.5});
    ASSERT_TRUE(dielectric.has_value()) << dielectric.error().message;
    const auto normal = Vector3T<Scalar>{.z = Scalar{1}};
    const auto tangent = Vector3T<Scalar>{.x = Scalar{1}};
    for (const auto mode : std::array{TransportMode::radiance, TransportMode::importance}) {
        const auto tangent_value = dielectric->eval(normal, tangent, mode);
        const auto tangent_probability = dielectric->pdf(normal, tangent, mode);
        ASSERT_TRUE(tangent_value.has_value()) << tangent_value.error().message;
        ASSERT_TRUE(tangent_probability.has_value()) << tangent_probability.error().message;
        EXPECT_EQ(*tangent_value, SpectrumFor<Scalar>{});
        EXPECT_EQ(tangent_probability->value, Scalar{0});
        EXPECT_FALSE(std::signbit(tangent_probability->value));
        EXPECT_EQ(tangent_probability->measure, ProbabilityMeasure::solid_angle);

        const auto tangent_sample =
            dielectric->sample(tangent, Scalar{0.5}, Point2T<Scalar>{}, mode);
        ASSERT_TRUE(tangent_sample.has_value()) << tangent_sample.error().message;
        EXPECT_FALSE(tangent_sample->has_value());

        const auto rejected_reflection = dielectric->sample(
            normal, Scalar{0}, Point2T<Scalar>{.x = Scalar{0.99}, .y = Scalar{0.25}}, mode);
        ASSERT_TRUE(rejected_reflection.has_value()) << rejected_reflection.error().message;
        EXPECT_FALSE(rejected_reflection->has_value());
    }

    const auto malformed_directions = std::array{
        Vector3T<Scalar>{},
        Vector3T<Scalar>{.z = Scalar{2}},
        Vector3T<Scalar>{.x = std::numeric_limits<Scalar>::quiet_NaN(), .z = Scalar{1}},
    };
    for (const auto mode : std::array{TransportMode::radiance, TransportMode::importance}) {
        for (const auto malformed : malformed_directions) {
            expect_invalid(dielectric->eval(normal, malformed, mode));
            expect_invalid(dielectric->pdf(malformed, normal, mode));
            expect_invalid(dielectric->sample(malformed, Scalar{0.5}, Point2T<Scalar>{}, mode));
        }
        expect_invalid(dielectric->sample(normal, Scalar{-0.1}, Point2T<Scalar>{}, mode));
        expect_invalid(dielectric->sample(normal, Scalar{1}, Point2T<Scalar>{}, mode));
        expect_invalid(dielectric->sample(normal, Scalar{0.5},
                                          Point2T<Scalar>{.x = Scalar{1}, .y = Scalar{0.5}}, mode));
    }
    const auto invalid_mode = static_cast<TransportMode>(0xffU);
    expect_invalid(dielectric->eval(normal, normal, invalid_mode));
    expect_invalid(dielectric->pdf(normal, normal, invalid_mode));
    expect_invalid(dielectric->sample(normal, Scalar{0.5}, Point2T<Scalar>{}, invalid_mode));
}

TEST(RoughDielectricTest, KeepsInvalidInputsAndUnsupportedDirectionsExplicit) {
    expect_invalid_queries_and_explicit_support<TransportScalar>();
    expect_invalid_queries_and_explicit_support<ReferenceScalar>();
}

template <SpectrumScalar Scalar>
void expect_sample_replay_and_jacobian(const Scalar alpha_x, const Scalar alpha_y,
                                       const bool reflection) {
    constexpr auto exterior_eta = Scalar{1};
    constexpr auto interior_eta = Scalar{1.5};
    const auto dielectric = RoughDielectricT<Scalar>::create(
        test_coefficient<Scalar>(), exterior_eta, interior_eta, alpha_x, alpha_y);
    const auto distribution = GgxFor<Scalar>::create(alpha_x, alpha_y);
    ASSERT_TRUE(dielectric.has_value()) << dielectric.error().message;
    ASSERT_TRUE(distribution.has_value()) << distribution.error().message;

    const auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};
    const auto event_sample = reflection ? Scalar{0} : Scalar{0.5};
    const auto canonical = Point2T<Scalar>{.x = Scalar{0.1}, .y = Scalar{0.375}};
    const auto sampled =
        dielectric->sample(outgoing, event_sample, canonical, TransportMode::importance);
    const auto replay =
        dielectric->sample(outgoing, event_sample, canonical, TransportMode::importance);
    ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
    ASSERT_TRUE(sampled->has_value());
    ASSERT_TRUE(replay.has_value()) << replay.error().message;
    ASSERT_TRUE(replay->has_value());
    const auto& event = **sampled;
    EXPECT_EQ(event.incoming_local, (**replay).incoming_local);
    EXPECT_EQ(event.value, (**replay).value);
    EXPECT_EQ(event.probability.value, (**replay).probability.value);
    EXPECT_EQ(event.lobes, (**replay).lobes);
    EXPECT_EQ(event.eta_scale_multiplier, (**replay).eta_scale_multiplier);
    EXPECT_EQ(event.lobes, ScatteringLobe::glossy | (reflection ? ScatteringLobe::reflection
                                                                : ScatteringLobe::transmission));
    EXPECT_EQ(event.probability.measure, ProbabilityMeasure::solid_angle);
    EXPECT_GT(event.probability.value, Scalar{0});

    const auto evaluated =
        dielectric->eval(outgoing, event.incoming_local, TransportMode::importance);
    const auto probability =
        dielectric->pdf(outgoing, event.incoming_local, TransportMode::importance);
    ASSERT_TRUE(evaluated.has_value()) << evaluated.error().message;
    ASSERT_TRUE(probability.has_value()) << probability.error().message;
    EXPECT_EQ(event.value, *evaluated);
    EXPECT_EQ(event.probability.value, probability->value);

    const auto microfacet = oriented_microfacet_normal(outgoing, event.incoming_local, exterior_eta,
                                                       interior_eta, reflection);
    const auto oriented_microfacet = Normal3T<Scalar>{
        .x = microfacet.x,
        .y = microfacet.y,
        .z = microfacet.z,
    };
    const auto visible_probability =
        distribution->visible_normal_pdf(outgoing, oriented_microfacet);
    ASSERT_TRUE(visible_probability.has_value()) << visible_probability.error().message;
    const auto outgoing_dot = vector_dot(outgoing, microfacet);
    const auto incoming_dot = vector_dot(event.incoming_local, microfacet);
    const auto fresnel = dielectric_fresnel(std::abs(outgoing_dot), exterior_eta, interior_eta);
    ASSERT_TRUE(fresnel.has_value()) << fresnel.error().message;
    auto expected_probability = Scalar{};
    if (reflection) {
        expected_probability =
            *fresnel * visible_probability->value / (Scalar{4} * std::abs(outgoing_dot));
    } else {
        const auto denominator = exterior_eta * outgoing_dot + interior_eta * incoming_dot;
        const auto jacobian =
            interior_eta * interior_eta * std::abs(incoming_dot) / (denominator * denominator);
        expected_probability = (Scalar{1} - *fresnel) * visible_probability->value * jacobian;
    }
    expect_scalar_near(probability->value, static_cast<ReferenceScalar>(expected_probability),
                       static_cast<ReferenceScalar>(expected_probability));
}

TEST(RoughDielectricTest, SamplesReflectionAndTransmissionWithExactVndfJacobians) {
    for (const auto alpha : std::array{TransportScalar{0.45F}, TransportScalar{2}}) {
        expect_sample_replay_and_jacobian<TransportScalar>(alpha, alpha, true);
        expect_sample_replay_and_jacobian<TransportScalar>(alpha, alpha, false);
    }
    expect_sample_replay_and_jacobian<TransportScalar>(TransportScalar{0.2F}, TransportScalar{0.7F},
                                                       true);
    expect_sample_replay_and_jacobian<TransportScalar>(TransportScalar{0.2F}, TransportScalar{0.7F},
                                                       false);
    for (const auto alpha : std::array{ReferenceScalar{0.45}, ReferenceScalar{2}}) {
        expect_sample_replay_and_jacobian<ReferenceScalar>(alpha, alpha, true);
        expect_sample_replay_and_jacobian<ReferenceScalar>(alpha, alpha, false);
    }
    expect_sample_replay_and_jacobian<ReferenceScalar>(ReferenceScalar{0.2}, ReferenceScalar{0.7},
                                                       true);
    expect_sample_replay_and_jacobian<ReferenceScalar>(ReferenceScalar{0.2}, ReferenceScalar{0.7},
                                                       false);
}

template <SpectrumScalar Scalar> void expect_axis_rotation_covariance() {
    const auto x_rough = RoughDielectricT<Scalar>::create(test_coefficient<Scalar>(), Scalar{1},
                                                          Scalar{1.5}, Scalar{0.2}, Scalar{0.7});
    const auto y_rough = RoughDielectricT<Scalar>::create(test_coefficient<Scalar>(), Scalar{1},
                                                          Scalar{1.5}, Scalar{0.7}, Scalar{0.2});
    ASSERT_TRUE(x_rough.has_value()) << x_rough.error().message;
    ASSERT_TRUE(y_rough.has_value()) << y_rough.error().message;
    const auto outgoing = Vector3T<Scalar>{
        .x = Scalar{0.3}, .y = Scalar{0.4}, .z = static_cast<Scalar>(std::sqrt(0.75L))};
    const auto rotate_quarter_turn = [](const Vector3T<Scalar> value) {
        return Vector3T<Scalar>{.x = -value.y, .y = value.x, .z = value.z};
    };
    const auto rotated_outgoing = rotate_quarter_turn(outgoing);
    constexpr auto canonical = Point2T<Scalar>{.x = Scalar{0.1}, .y = Scalar{0.375}};
    for (const auto event_sample : std::array{Scalar{0}, Scalar{0.5}}) {
        const auto x_sample =
            x_rough->sample(outgoing, event_sample, canonical, TransportMode::importance);
        const auto y_sample =
            y_rough->sample(rotated_outgoing, event_sample, canonical, TransportMode::importance);
        ASSERT_TRUE(x_sample.has_value()) << x_sample.error().message;
        ASSERT_TRUE(x_sample->has_value());
        ASSERT_TRUE(y_sample.has_value()) << y_sample.error().message;
        ASSERT_TRUE(y_sample->has_value());
        const auto& x_event = **x_sample;
        const auto& y_event = **y_sample;
        const auto expected_incoming = rotate_quarter_turn(x_event.incoming_local);
        expect_scalar_near(y_event.incoming_local.x,
                           static_cast<ReferenceScalar>(expected_incoming.x));
        expect_scalar_near(y_event.incoming_local.y,
                           static_cast<ReferenceScalar>(expected_incoming.y));
        expect_scalar_near(y_event.incoming_local.z,
                           static_cast<ReferenceScalar>(expected_incoming.z));
        expect_spectrum_near(x_event.value, y_event.value);
        expect_scalar_near(x_event.probability.value,
                           static_cast<ReferenceScalar>(y_event.probability.value),
                           static_cast<ReferenceScalar>(x_event.probability.value));
        EXPECT_EQ(x_event.probability.measure, y_event.probability.measure);
        EXPECT_EQ(x_event.lobes, y_event.lobes);
        EXPECT_EQ(x_event.lobes, ScatteringLobe::glossy |
                                     (event_sample == Scalar{0} ? ScatteringLobe::reflection
                                                                : ScatteringLobe::transmission));
        expect_scalar_near(x_event.eta_scale_multiplier,
                           static_cast<ReferenceScalar>(y_event.eta_scale_multiplier));

        const auto x_value =
            x_rough->eval(outgoing, x_event.incoming_local, TransportMode::importance);
        const auto y_value =
            y_rough->eval(rotated_outgoing, expected_incoming, TransportMode::importance);
        const auto x_pdf =
            x_rough->pdf(outgoing, x_event.incoming_local, TransportMode::importance);
        const auto y_pdf =
            y_rough->pdf(rotated_outgoing, expected_incoming, TransportMode::importance);
        ASSERT_TRUE(x_value.has_value()) << x_value.error().message;
        ASSERT_TRUE(y_value.has_value()) << y_value.error().message;
        ASSERT_TRUE(x_pdf.has_value()) << x_pdf.error().message;
        ASSERT_TRUE(y_pdf.has_value()) << y_pdf.error().message;
        expect_spectrum_near(*x_value, *y_value);
        expect_scalar_near(x_pdf->value, static_cast<ReferenceScalar>(y_pdf->value),
                           static_cast<ReferenceScalar>(x_pdf->value));
    }
}

TEST(RoughDielectricTest, RotatesAnisotropicReflectionAndTransmissionCovariantly) {
    expect_axis_rotation_covariance<TransportScalar>();
    expect_axis_rotation_covariance<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_two_sided_support() {
    const auto dielectric = RoughDielectricT<Scalar>::create(test_coefficient<Scalar>(), Scalar{1},
                                                             Scalar{1.5}, Scalar{0.35});
    ASSERT_TRUE(dielectric.has_value()) << dielectric.error().message;
    const auto canonical = Point2T<Scalar>{.x = Scalar{0.08}, .y = Scalar{0.625}};
    const auto cosine = std::sqrt(Scalar{0.75});
    for (const auto outgoing : std::array{
             Vector3T<Scalar>{.x = Scalar{0.3}, .y = Scalar{0.4}, .z = cosine},
             Vector3T<Scalar>{.z = Scalar{-1}},
         }) {
        for (const auto event_sample : std::array{Scalar{0}, Scalar{0.5}}) {
            const auto sampled =
                dielectric->sample(outgoing, event_sample, canonical, TransportMode::importance);
            ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
            ASSERT_TRUE(sampled->has_value());
            const auto& event = **sampled;
            EXPECT_EQ(std::signbit(event.incoming_local.z), event_sample == Scalar{0}
                                                                ? std::signbit(outgoing.z)
                                                                : !std::signbit(outgoing.z));
            EXPECT_TRUE(is_valid_surface_scattering_event(event.lobes));
            const auto evaluated =
                dielectric->eval(outgoing, event.incoming_local, TransportMode::importance);
            const auto probability =
                dielectric->pdf(outgoing, event.incoming_local, TransportMode::importance);
            ASSERT_TRUE(evaluated.has_value()) << evaluated.error().message;
            ASSERT_TRUE(probability.has_value()) << probability.error().message;
            EXPECT_EQ(event.value, *evaluated);
            EXPECT_EQ(event.probability.value, probability->value);
        }
    }
}

TEST(RoughDielectricTest, SupportsBothSidesWithoutChangingTheLocalFrameConvention) {
    expect_two_sided_support<TransportScalar>();
    expect_two_sided_support<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_transport_mode_and_eta_scale() {
    const auto dielectric = RoughDielectricT<Scalar>::create(test_coefficient<Scalar>(), Scalar{1},
                                                             Scalar{1.5}, Scalar{0.4});
    ASSERT_TRUE(dielectric.has_value()) << dielectric.error().message;
    const auto canonical = Point2T<Scalar>{.x = Scalar{0.08}, .y = Scalar{0.25}};
    for (const auto outgoing :
         std::array{Vector3T<Scalar>{.z = Scalar{1}}, Vector3T<Scalar>{.z = Scalar{-1}}}) {
        const auto radiance =
            dielectric->sample(outgoing, Scalar{0.5}, canonical, TransportMode::radiance);
        const auto importance =
            dielectric->sample(outgoing, Scalar{0.5}, canonical, TransportMode::importance);
        ASSERT_TRUE(radiance.has_value()) << radiance.error().message;
        ASSERT_TRUE(importance.has_value()) << importance.error().message;
        ASSERT_TRUE(radiance->has_value());
        ASSERT_TRUE(importance->has_value());
        const auto& radiance_event = **radiance;
        const auto& importance_event = **importance;
        ASSERT_EQ(radiance_event.lobes, ScatteringLobe::glossy | ScatteringLobe::transmission);
        EXPECT_EQ(radiance_event.incoming_local, importance_event.incoming_local);
        EXPECT_EQ(radiance_event.probability.value, importance_event.probability.value);
        EXPECT_EQ(radiance_event.lobes, importance_event.lobes);

        const auto incident_eta = outgoing.z > Scalar{0} ? Scalar{1} : Scalar{1.5};
        const auto transmitted_eta = outgoing.z > Scalar{0} ? Scalar{1.5} : Scalar{1};
        const auto radiance_factor =
            (incident_eta / transmitted_eta) * (incident_eta / transmitted_eta);
        const auto eta_scale = (transmitted_eta / incident_eta) * (transmitted_eta / incident_eta);
        expect_scalar_near(radiance_event.eta_scale_multiplier,
                           static_cast<ReferenceScalar>(eta_scale));
        EXPECT_EQ(importance_event.eta_scale_multiplier, Scalar{1});
        for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
            expect_scalar_near(
                radiance_event.value[lane],
                static_cast<ReferenceScalar>(importance_event.value[lane] * radiance_factor),
                static_cast<ReferenceScalar>(importance_event.value[lane]));
            expect_scalar_near(radiance_event.value[lane] * radiance_event.eta_scale_multiplier,
                               static_cast<ReferenceScalar>(importance_event.value[lane]),
                               static_cast<ReferenceScalar>(importance_event.value[lane]));
        }
    }

    const auto reflected = dielectric->sample(Vector3T<Scalar>{.z = Scalar{1}}, Scalar{0},
                                              canonical, TransportMode::radiance);
    ASSERT_TRUE(reflected.has_value()) << reflected.error().message;
    ASSERT_TRUE(reflected->has_value());
    EXPECT_EQ((**reflected).lobes, ScatteringLobe::glossy | ScatteringLobe::reflection);
    EXPECT_EQ((**reflected).eta_scale_multiplier, Scalar{1});
}

TEST(RoughDielectricTest, SeparatesRadianceAdjointAndEtaScaleFromImportance) {
    expect_transport_mode_and_eta_scale<TransportScalar>();
    expect_transport_mode_and_eta_scale<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_total_internal_reflection_forces_reflection() {
    const auto dielectric = RoughDielectricT<Scalar>::create(
        constant_spectrum<Scalar>(Scalar{1}), Scalar{1}, Scalar{1.5}, Scalar{0.05}, Scalar{0.2});
    ASSERT_TRUE(dielectric.has_value()) << dielectric.error().message;
    const auto outgoing = Vector3T<Scalar>{.x = Scalar{0.8}, .z = Scalar{-0.6}};
    const auto event_sample = std::nextafter(Scalar{1}, Scalar{0});
    auto found_total_internal_reflection = false;
    for (auto radial = std::size_t{}; radial < 8 && !found_total_internal_reflection; ++radial) {
        for (auto azimuth = std::size_t{}; azimuth < 16 && !found_total_internal_reflection;
             ++azimuth) {
            const auto canonical = Point2T<Scalar>{
                .x = (static_cast<Scalar>(radial) + Scalar{0.5}) / Scalar{8},
                .y = (static_cast<Scalar>(azimuth) + Scalar{0.5}) / Scalar{16},
            };
            const auto sampled =
                dielectric->sample(outgoing, event_sample, canonical, TransportMode::radiance);
            ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
            if (!sampled->has_value()) {
                continue;
            }
            const auto& event = **sampled;
            if (event.lobes != (ScatteringLobe::glossy | ScatteringLobe::reflection)) {
                continue;
            }
            const auto microfacet = oriented_microfacet_normal(outgoing, event.incoming_local,
                                                               Scalar{1.5}, Scalar{1}, true);
            const auto fresnel = dielectric_fresnel(std::abs(vector_dot(outgoing, microfacet)),
                                                    Scalar{1.5}, Scalar{1});
            ASSERT_TRUE(fresnel.has_value()) << fresnel.error().message;
            if (*fresnel != Scalar{1}) {
                continue;
            }
            found_total_internal_reflection = true;
            EXPECT_EQ(event.eta_scale_multiplier, Scalar{1});
            EXPECT_EQ(event.probability.measure, ProbabilityMeasure::solid_angle);
            EXPECT_GT(event.probability.value, Scalar{0});
            const auto probability =
                dielectric->pdf(outgoing, event.incoming_local, TransportMode::radiance);
            ASSERT_TRUE(probability.has_value()) << probability.error().message;
            EXPECT_EQ(event.probability.value, probability->value);
        }
    }
    EXPECT_TRUE(found_total_internal_reflection)
        << "A transmission-selected microfacet under TIR must become glossy reflection.";
}

TEST(RoughDielectricTest, TotalInternalReflectionForcesThePhysicalGlossyReflectionEvent) {
    expect_total_internal_reflection_forces_reflection<TransportScalar>();
    expect_total_internal_reflection_forces_reflection<ReferenceScalar>();
}

template <SpectrumScalar Scalar>
[[nodiscard]] std::array<long double, TransportSpectrumSampleCount>
integrate_white_furnace(const DielectricFor<Scalar>& dielectric, const Vector3T<Scalar> outgoing) {
    constexpr auto cosine_steps = std::size_t{80};
    constexpr auto azimuth_steps = std::size_t{160};
    constexpr auto delta_cosine = 1.0L / static_cast<long double>(cosine_steps);
    constexpr auto delta_azimuth =
        2.0L * std::numbers::pi_v<long double> / static_cast<long double>(azimuth_steps);
    auto integral = std::array<long double, TransportSpectrumSampleCount>{};
    for (const auto side : std::array{-1.0L, 1.0L}) {
        for (auto cosine_index = std::size_t{}; cosine_index < cosine_steps; ++cosine_index) {
            const auto cosine = (static_cast<long double>(cosine_index) + 0.5L) * delta_cosine;
            const auto radial = std::sqrt((1.0L - cosine) * (1.0L + cosine));
            for (auto azimuth_index = std::size_t{}; azimuth_index < azimuth_steps;
                 ++azimuth_index) {
                const auto azimuth =
                    (static_cast<long double>(azimuth_index) + 0.5L) * delta_azimuth;
                const auto incoming = Vector3T<Scalar>{
                    .x = static_cast<Scalar>(radial * std::cos(azimuth)),
                    .y = static_cast<Scalar>(radial * std::sin(azimuth)),
                    .z = static_cast<Scalar>(side * cosine),
                };
                const auto evaluated =
                    dielectric.eval(outgoing, incoming, TransportMode::importance);
                if (!evaluated) {
                    ADD_FAILURE() << evaluated.error().message;
                    return integral;
                }
                for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
                    integral[lane] += static_cast<long double>((*evaluated)[lane]) * cosine *
                                      delta_cosine * delta_azimuth;
                }
            }
        }
    }
    return integral;
}

template <SpectrumScalar Scalar> void expect_white_furnace_energy_bound() {
    const auto outgoing_directions = std::array{
        Vector3T<Scalar>{.z = Scalar{1}},
        Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}},
        Vector3T<Scalar>{
            .x = Scalar{0.3}, .y = Scalar{0.4}, .z = -static_cast<Scalar>(std::sqrt(0.75L))},
    };
    for (const auto [alpha_x, alpha_y] :
         std::array{std::pair{Scalar{0.25}, Scalar{0.25}}, std::pair{Scalar{0.6}, Scalar{0.6}},
                    std::pair{Scalar{1}, Scalar{1}}, std::pair{Scalar{2}, Scalar{2}},
                    std::pair{Scalar{0.25}, Scalar{0.7}}, std::pair{Scalar{0.7}, Scalar{0.25}}}) {
        const auto dielectric = RoughDielectricT<Scalar>::create(
            constant_spectrum<Scalar>(Scalar{1}), Scalar{1}, Scalar{1.5}, alpha_x, alpha_y);
        ASSERT_TRUE(dielectric.has_value()) << dielectric.error().message;
        for (const auto outgoing : outgoing_directions) {
            const auto integral = integrate_white_furnace(*dielectric, outgoing);
            for (const auto lane : integral) {
                EXPECT_TRUE(std::isfinite(lane));
                EXPECT_GT(lane, 1.0e-4L);
                EXPECT_LE(lane, 1.005L);
            }
        }
    }
}

TEST(RoughDielectricTest, ObeysTheFullSphereWhiteFurnaceBoundAcrossWidthAndSides) {
    expect_white_furnace_energy_bound<TransportScalar>();
    expect_white_furnace_energy_bound<ReferenceScalar>();
}

static_assert(std::is_standard_layout_v<RoughDielectric>);
static_assert(std::is_trivially_copyable_v<RoughDielectric>);
static_assert(std::is_standard_layout_v<ReferenceRoughDielectric>);
static_assert(std::is_trivially_copyable_v<ReferenceRoughDielectric>);

} // namespace
} // namespace blackframe::renderer
