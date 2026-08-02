#include <Blackframe/Renderer/Fresnel.hpp>
#include <Blackframe/Renderer/GgxMicrofacet.hpp>
#include <Blackframe/Renderer/RoughConductorReflection.hpp>
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
using ReflectionFor =
    std::conditional_t<std::same_as<Scalar, TransportScalar>, RoughConductorReflection,
                       ReferenceRoughConductorReflection>;

template <SpectrumScalar Scalar>
using GgxFor = std::conditional_t<std::same_as<Scalar, TransportScalar>, GgxMicrofacet,
                                  ReferenceGgxMicrofacet>;

template <SpectrumScalar Scalar>
using SpectrumFor = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

template <SpectrumScalar Scalar>
inline constexpr auto AnalyticTolerance =
    std::same_as<Scalar, TransportScalar> ? ReferenceScalar{8.0e-6} : ReferenceScalar{2.0e-13};

template <typename Result> void expect_invalid(const Result& result) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(result.error().message.empty());
}

template <SpectrumScalar Scalar>
[[nodiscard]] SpectrumFor<Scalar> constant_spectrum(const Scalar value) {
    auto spectrum = SpectrumFor<Scalar>{};
    spectrum.values.fill(value);
    return spectrum;
}

template <SpectrumScalar Scalar> [[nodiscard]] SpectrumFor<Scalar> test_coefficient() {
    return {.values = {Scalar{0.25}, Scalar{0.5}, Scalar{0.75}, Scalar{1}}};
}

template <SpectrumScalar Scalar> [[nodiscard]] SpectrumFor<Scalar> test_relative_eta() {
    return {.values = {Scalar{0.2}, Scalar{0.5}, Scalar{1.5}, Scalar{2}}};
}

template <SpectrumScalar Scalar> [[nodiscard]] SpectrumFor<Scalar> test_relative_k() {
    return {.values = {Scalar{3}, Scalar{2}, Scalar{1}, Scalar{4}}};
}

template <SpectrumScalar Scalar>
void expect_spectrum_near(const SpectrumFor<Scalar>& actual, const SpectrumFor<Scalar>& expected) {
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(static_cast<ReferenceScalar>(actual[lane]),
                    static_cast<ReferenceScalar>(expected[lane]), AnalyticTolerance<Scalar>);
    }
}

template <SpectrumScalar Scalar> void expect_creation_contract() {
    const auto coefficient = test_coefficient<Scalar>();
    const auto relative_eta = test_relative_eta<Scalar>();
    const auto relative_k = test_relative_k<Scalar>();
    for (const auto alpha : std::array{Scalar{0.2}, Scalar{0.6}, Scalar{1}, Scalar{2}}) {
        const auto reflection =
            ReflectionFor<Scalar>::create(coefficient, relative_eta, relative_k, alpha);
        ASSERT_TRUE(reflection.has_value()) << reflection.error().message;
        EXPECT_EQ(reflection->coefficient(), coefficient);
        EXPECT_EQ(reflection->relative_eta(), relative_eta);
        EXPECT_EQ(reflection->relative_k(), relative_k);
        EXPECT_EQ(reflection->alpha(), alpha);
        EXPECT_EQ(reflection->alpha_x(), alpha);
        EXPECT_EQ(reflection->alpha_y(), alpha);
    }
    const auto anisotropic = ReflectionFor<Scalar>::create(coefficient, relative_eta, relative_k,
                                                           Scalar{0.25}, Scalar{0.75});
    ASSERT_TRUE(anisotropic.has_value()) << anisotropic.error().message;
    EXPECT_EQ(anisotropic->alpha_x(), Scalar{0.25});
    EXPECT_EQ(anisotropic->alpha_y(), Scalar{0.75});

    const auto infinity = std::numeric_limits<Scalar>::infinity();
    auto malformed_coefficient = coefficient;
    malformed_coefficient[0] = std::nextafter(Scalar{1}, infinity);
    expect_invalid(ReflectionFor<Scalar>::create(malformed_coefficient, relative_eta, relative_k,
                                                 Scalar{0.5}));
    auto malformed_eta = relative_eta;
    malformed_eta[1] = Scalar{0};
    expect_invalid(
        ReflectionFor<Scalar>::create(coefficient, malformed_eta, relative_k, Scalar{0.5}));
    auto malformed_k = relative_k;
    malformed_k[2] = -std::numeric_limits<Scalar>::denorm_min();
    expect_invalid(
        ReflectionFor<Scalar>::create(coefficient, relative_eta, malformed_k, Scalar{0.5}));
    for (const auto malformed :
         std::array{std::numeric_limits<Scalar>::quiet_NaN(), infinity, -infinity}) {
        malformed_coefficient[0] = malformed;
        expect_invalid(ReflectionFor<Scalar>::create(malformed_coefficient, relative_eta,
                                                     relative_k, Scalar{0.5}));
        malformed_eta = relative_eta;
        malformed_eta[1] = malformed;
        expect_invalid(
            ReflectionFor<Scalar>::create(coefficient, malformed_eta, relative_k, Scalar{0.5}));
        malformed_k = relative_k;
        malformed_k[2] = malformed;
        expect_invalid(
            ReflectionFor<Scalar>::create(coefficient, relative_eta, malformed_k, Scalar{0.5}));
        expect_invalid(
            ReflectionFor<Scalar>::create(coefficient, relative_eta, relative_k, malformed));
    }
    for (const auto invalid_alpha :
         std::array{Scalar{0}, Scalar{-0.0}, -std::numeric_limits<Scalar>::denorm_min()}) {
        expect_invalid(
            ReflectionFor<Scalar>::create(coefficient, relative_eta, relative_k, invalid_alpha));
    }
    for (const auto invalid_alpha :
         std::array{Scalar{0}, Scalar{-0.0}, -std::numeric_limits<Scalar>::denorm_min(),
                    std::numeric_limits<Scalar>::quiet_NaN(), infinity, -infinity}) {
        expect_invalid(ReflectionFor<Scalar>::create(coefficient, relative_eta, relative_k,
                                                     invalid_alpha, Scalar{0.5}));
        expect_invalid(ReflectionFor<Scalar>::create(coefficient, relative_eta, relative_k,
                                                     Scalar{0.5}, invalid_alpha));
    }

    auto signed_zero_k = relative_k;
    signed_zero_k[0] = Scalar{-0.0};
    const auto accepted =
        ReflectionFor<Scalar>::create(coefficient, relative_eta, signed_zero_k, Scalar{0.5});
    ASSERT_TRUE(accepted.has_value()) << accepted.error().message;
    EXPECT_EQ(accepted->relative_k()[0], Scalar{0});
}

TEST(RoughConductorReflectionTest, CreatesOnlyFinitePhysicalParametersAndPositiveWidth) {
    static_assert(std::same_as<decltype(RoughConductorReflection::create(
                                   TransportSpectrum{}, TransportSpectrum{}, TransportSpectrum{},
                                   TransportScalar{})),
                               core::Result<RoughConductorReflection>>);
    static_assert(std::same_as<decltype(ReferenceRoughConductorReflection::create(
                                   ReferenceSpectrum{}, ReferenceSpectrum{}, ReferenceSpectrum{},
                                   ReferenceScalar{})),
                               core::Result<ReferenceRoughConductorReflection>>);
    static_assert(std::same_as<decltype(RoughConductorReflection::create(
                                   TransportSpectrum{}, TransportSpectrum{}, TransportSpectrum{},
                                   TransportScalar{}, TransportScalar{})),
                               core::Result<RoughConductorReflection>>);
    static_assert(std::same_as<decltype(ReferenceRoughConductorReflection::create(
                                   ReferenceSpectrum{}, ReferenceSpectrum{}, ReferenceSpectrum{},
                                   ReferenceScalar{}, ReferenceScalar{})),
                               core::Result<ReferenceRoughConductorReflection>>);
    static_assert(!std::same_as<RoughConductorReflection, ReferenceRoughConductorReflection>);
    static_assert(!std::same_as<RoughConductorSample, ReferenceRoughConductorSample>);
    expect_creation_contract<TransportScalar>();
    expect_creation_contract<ReferenceScalar>();
}

template <SpectrumScalar Scalar>
void expect_reciprocal_analytic_value(const Scalar alpha_x, const Scalar alpha_y) {
    const auto coefficient = test_coefficient<Scalar>();
    const auto relative_eta = test_relative_eta<Scalar>();
    const auto relative_k = test_relative_k<Scalar>();
    const auto reflection =
        ReflectionFor<Scalar>::create(coefficient, relative_eta, relative_k, alpha_x, alpha_y);
    ASSERT_TRUE(reflection.has_value()) << reflection.error().message;

    const auto normal = Vector3T<Scalar>{.z = Scalar{1}};
    const auto normal_value = reflection->eval(normal, normal);
    const auto normal_fresnel = conductor_fresnel(Scalar{1}, relative_eta, relative_k);
    ASSERT_TRUE(normal_value.has_value()) << normal_value.error().message;
    ASSERT_TRUE(normal_fresnel.has_value()) << normal_fresnel.error().message;
    const auto normal_scale =
        Scalar{1} / (Scalar{4} * std::numbers::pi_v<Scalar> * alpha_x * alpha_y);
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_NEAR(static_cast<ReferenceScalar>((*normal_value)[lane]),
                    static_cast<ReferenceScalar>(coefficient[lane] * (*normal_fresnel)[lane] *
                                                 normal_scale),
                    AnalyticTolerance<Scalar>);
    }

    const auto first = Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}};
    const auto second = Vector3T<Scalar>{
        .x = Scalar{-0.3},
        .y = Scalar{0.4},
        .z = static_cast<Scalar>(std::sqrt(0.75L)),
    };
    for (const auto directions : std::array{std::pair{normal, first}, std::pair{first, second},
                                            std::pair{second, normal}}) {
        const auto forward = reflection->eval(directions.first, directions.second);
        const auto reverse = reflection->eval(directions.second, directions.first);
        ASSERT_TRUE(forward.has_value()) << forward.error().message;
        ASSERT_TRUE(reverse.has_value()) << reverse.error().message;
        expect_spectrum_near(*forward, *reverse);
        for (const auto value : forward->values) {
            EXPECT_TRUE(std::isfinite(value));
            EXPECT_GT(value, Scalar{0});
        }
    }
}

TEST(RoughConductorReflectionTest, EvaluatesTheAnalyticReciprocalGgxBrdf) {
    for (const auto alpha : std::array{TransportScalar{0.5}, TransportScalar{2}}) {
        expect_reciprocal_analytic_value<TransportScalar>(alpha, alpha);
    }
    expect_reciprocal_analytic_value<TransportScalar>(TransportScalar{0.25F},
                                                      TransportScalar{0.75F});
    for (const auto alpha : std::array{ReferenceScalar{0.5}, ReferenceScalar{2}}) {
        expect_reciprocal_analytic_value<ReferenceScalar>(alpha, alpha);
    }
    expect_reciprocal_analytic_value<ReferenceScalar>(ReferenceScalar{0.25}, ReferenceScalar{0.75});
}

template <SpectrumScalar Scalar>
void expect_vndf_jacobian_and_replay(const Scalar alpha_x, const Scalar alpha_y) {
    const auto reflection =
        ReflectionFor<Scalar>::create(test_coefficient<Scalar>(), test_relative_eta<Scalar>(),
                                      test_relative_k<Scalar>(), alpha_x, alpha_y);
    const auto distribution = GgxFor<Scalar>::create(alpha_x, alpha_y);
    ASSERT_TRUE(reflection.has_value()) << reflection.error().message;
    ASSERT_TRUE(distribution.has_value()) << distribution.error().message;

    const auto outgoing = Vector3T<Scalar>{.z = Scalar{1}};
    const auto canonical = Point2T<Scalar>{.x = Scalar{0.1}, .y = Scalar{0.375}};
    const auto sampled = reflection->sample(outgoing, canonical);
    const auto replay = reflection->sample(outgoing, canonical);
    ASSERT_TRUE(sampled.has_value()) << sampled.error().message;
    ASSERT_TRUE(sampled->has_value());
    ASSERT_TRUE(replay.has_value()) << replay.error().message;
    ASSERT_TRUE(replay->has_value());
    EXPECT_EQ((**sampled).incoming_local, (**replay).incoming_local);
    EXPECT_EQ((**sampled).value, (**replay).value);
    EXPECT_EQ((**sampled).probability.value, (**replay).probability.value);

    const auto evaluated = reflection->eval(outgoing, (**sampled).incoming_local);
    const auto probability = reflection->pdf(outgoing, (**sampled).incoming_local);
    ASSERT_TRUE(evaluated.has_value()) << evaluated.error().message;
    ASSERT_TRUE(probability.has_value()) << probability.error().message;
    EXPECT_EQ((**sampled).value, *evaluated);
    EXPECT_EQ((**sampled).probability.value, probability->value);
    EXPECT_EQ(probability->measure, ProbabilityMeasure::solid_angle);

    const auto half_x = outgoing.x + (**sampled).incoming_local.x;
    const auto half_y = outgoing.y + (**sampled).incoming_local.y;
    const auto half_z = outgoing.z + (**sampled).incoming_local.z;
    const auto inverse_half_length =
        Scalar{1} / std::sqrt(half_x * half_x + half_y * half_y + half_z * half_z);
    const auto microfacet_normal = Normal3T<Scalar>{
        .x = half_x * inverse_half_length,
        .y = half_y * inverse_half_length,
        .z = half_z * inverse_half_length,
    };
    const auto outgoing_dot_microfacet = outgoing.x * microfacet_normal.x +
                                         outgoing.y * microfacet_normal.y +
                                         outgoing.z * microfacet_normal.z;
    const auto microfacet_probability =
        distribution->visible_normal_pdf(outgoing, microfacet_normal);
    ASSERT_TRUE(microfacet_probability.has_value()) << microfacet_probability.error().message;
    const auto expected_probability =
        microfacet_probability->value / (Scalar{4} * std::abs(outgoing_dot_microfacet));
    EXPECT_NEAR(static_cast<ReferenceScalar>(probability->value),
                static_cast<ReferenceScalar>(expected_probability), AnalyticTolerance<Scalar>);
}

TEST(RoughConductorReflectionTest, SamplesTheVndfWithMatchingJacobianAndReplay) {
    for (const auto alpha : std::array{TransportScalar{0.45}, TransportScalar{2}}) {
        expect_vndf_jacobian_and_replay<TransportScalar>(alpha, alpha);
    }
    expect_vndf_jacobian_and_replay<TransportScalar>(TransportScalar{0.2F}, TransportScalar{0.7F});
    for (const auto alpha : std::array{ReferenceScalar{0.45}, ReferenceScalar{2}}) {
        expect_vndf_jacobian_and_replay<ReferenceScalar>(alpha, alpha);
    }
    expect_vndf_jacobian_and_replay<ReferenceScalar>(ReferenceScalar{0.2}, ReferenceScalar{0.7});
}

template <SpectrumScalar Scalar> void expect_axis_rotation_covariance() {
    const auto x_rough =
        ReflectionFor<Scalar>::create(test_coefficient<Scalar>(), test_relative_eta<Scalar>(),
                                      test_relative_k<Scalar>(), Scalar{0.2}, Scalar{0.7});
    const auto y_rough =
        ReflectionFor<Scalar>::create(test_coefficient<Scalar>(), test_relative_eta<Scalar>(),
                                      test_relative_k<Scalar>(), Scalar{0.7}, Scalar{0.2});
    ASSERT_TRUE(x_rough.has_value()) << x_rough.error().message;
    ASSERT_TRUE(y_rough.has_value()) << y_rough.error().message;
    const auto outgoing = Vector3T<Scalar>{
        .x = Scalar{0.3}, .y = Scalar{0.4}, .z = static_cast<Scalar>(std::sqrt(0.75L))};
    const auto incoming = Vector3T<Scalar>{.x = Scalar{-0.6}, .z = Scalar{0.8}};
    const auto rotate_quarter_turn = [](const Vector3T<Scalar> value) {
        return Vector3T<Scalar>{.x = -value.y, .y = value.x, .z = value.z};
    };
    const auto x_value = x_rough->eval(outgoing, incoming);
    const auto y_value =
        y_rough->eval(rotate_quarter_turn(outgoing), rotate_quarter_turn(incoming));
    const auto x_pdf = x_rough->pdf(outgoing, incoming);
    const auto y_pdf = y_rough->pdf(rotate_quarter_turn(outgoing), rotate_quarter_turn(incoming));
    ASSERT_TRUE(x_value.has_value()) << x_value.error().message;
    ASSERT_TRUE(y_value.has_value()) << y_value.error().message;
    ASSERT_TRUE(x_pdf.has_value()) << x_pdf.error().message;
    ASSERT_TRUE(y_pdf.has_value()) << y_pdf.error().message;
    expect_spectrum_near(*x_value, *y_value);
    EXPECT_NEAR(static_cast<ReferenceScalar>(x_pdf->value),
                static_cast<ReferenceScalar>(y_pdf->value), AnalyticTolerance<Scalar>);
}

TEST(RoughConductorReflectionTest, RotatesAnisotropicAxesCovariantly) {
    expect_axis_rotation_covariance<TransportScalar>();
    expect_axis_rotation_covariance<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_support_without_fallback() {
    const auto reflection =
        ReflectionFor<Scalar>::create(test_coefficient<Scalar>(), test_relative_eta<Scalar>(),
                                      test_relative_k<Scalar>(), Scalar{1});
    ASSERT_TRUE(reflection.has_value()) << reflection.error().message;
    const auto normal = Vector3T<Scalar>{.z = Scalar{1}};
    for (const auto incoming :
         std::array{Vector3T<Scalar>{.x = Scalar{1}}, Vector3T<Scalar>{.z = Scalar{-1}}}) {
        const auto evaluated = reflection->eval(normal, incoming);
        const auto probability = reflection->pdf(normal, incoming);
        ASSERT_TRUE(evaluated.has_value()) << evaluated.error().message;
        ASSERT_TRUE(probability.has_value()) << probability.error().message;
        EXPECT_EQ(*evaluated, SpectrumFor<Scalar>{});
        EXPECT_EQ(probability->value, Scalar{0});
        EXPECT_FALSE(std::signbit(probability->value));
        EXPECT_EQ(probability->measure, ProbabilityMeasure::solid_angle);
    }

    const auto malformed = Vector3T<Scalar>{
        .x = std::numeric_limits<Scalar>::quiet_NaN(),
        .z = Scalar{1},
    };
    expect_invalid(reflection->eval(normal, malformed));
    expect_invalid(reflection->pdf(malformed, normal));
    expect_invalid(reflection->sample(normal, Point2T<Scalar>{.x = Scalar{1}}));

    const auto rejected =
        reflection->sample(normal, Point2T<Scalar>{.x = Scalar{0.99}, .y = Scalar{0.25}});
    ASSERT_TRUE(rejected.has_value()) << rejected.error().message;
    EXPECT_FALSE(rejected->has_value());
}

TEST(RoughConductorReflectionTest, KeepsOneSidedSupportWithoutAHiddenFallback) {
    expect_support_without_fallback<TransportScalar>();
    expect_support_without_fallback<ReferenceScalar>();
}

template <SpectrumScalar Scalar>
[[nodiscard]] std::array<long double, TransportSpectrumSampleCount>
integrate_white_furnace(const ReflectionFor<Scalar>& reflection, const Vector3T<Scalar> outgoing) {
    constexpr auto cosine_steps = std::size_t{96};
    constexpr auto azimuth_steps = std::size_t{192};
    constexpr auto delta_cosine = 1.0L / static_cast<long double>(cosine_steps);
    constexpr auto delta_azimuth =
        2.0L * std::numbers::pi_v<long double> / static_cast<long double>(azimuth_steps);
    auto integral = std::array<long double, TransportSpectrumSampleCount>{};
    for (auto cosine_index = std::size_t{}; cosine_index < cosine_steps; ++cosine_index) {
        const auto cosine = (static_cast<long double>(cosine_index) + 0.5L) * delta_cosine;
        const auto radial = std::sqrt((1.0L - cosine) * (1.0L + cosine));
        for (auto azimuth_index = std::size_t{}; azimuth_index < azimuth_steps; ++azimuth_index) {
            const auto azimuth = (static_cast<long double>(azimuth_index) + 0.5L) * delta_azimuth;
            const auto incoming = Vector3T<Scalar>{
                .x = static_cast<Scalar>(radial * std::cos(azimuth)),
                .y = static_cast<Scalar>(radial * std::sin(azimuth)),
                .z = static_cast<Scalar>(cosine),
            };
            const auto evaluated = reflection.eval(outgoing, incoming);
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
    return integral;
}

template <SpectrumScalar Scalar> void expect_white_furnace_energy_bound() {
    const auto outgoing_directions = std::array{
        Vector3T<Scalar>{.z = Scalar{1}},
        Vector3T<Scalar>{.x = Scalar{0.6}, .z = Scalar{0.8}},
        Vector3T<Scalar>{.x = static_cast<Scalar>(std::sqrt(0.96L)), .z = Scalar{0.2}},
    };
    for (const auto [alpha_x, alpha_y] :
         std::array{std::pair{Scalar{0.25}, Scalar{0.25}}, std::pair{Scalar{0.6}, Scalar{0.6}},
                    std::pair{Scalar{1}, Scalar{1}}, std::pair{Scalar{2}, Scalar{2}},
                    std::pair{Scalar{0.25}, Scalar{0.7}}, std::pair{Scalar{0.7}, Scalar{0.25}}}) {
        const auto reflection = ReflectionFor<Scalar>::create(
            constant_spectrum<Scalar>(Scalar{1}), test_relative_eta<Scalar>(),
            test_relative_k<Scalar>(), alpha_x, alpha_y);
        ASSERT_TRUE(reflection.has_value()) << reflection.error().message;
        for (const auto outgoing : outgoing_directions) {
            const auto integral = integrate_white_furnace(*reflection, outgoing);
            for (const auto lane : integral) {
                EXPECT_TRUE(std::isfinite(lane));
                EXPECT_GT(lane, 1.0e-4L);
                EXPECT_LE(lane, 1.005L);
            }
        }
    }
}

TEST(RoughConductorReflectionTest, ObeysTheWhiteFurnaceEnergyBoundAcrossWidthAndViewAngle) {
    expect_white_furnace_energy_bound<TransportScalar>();
    expect_white_furnace_energy_bound<ReferenceScalar>();
}

static_assert(std::is_standard_layout_v<RoughConductorSample>);
static_assert(std::is_trivially_copyable_v<RoughConductorSample>);
static_assert(std::is_standard_layout_v<ReferenceRoughConductorSample>);
static_assert(std::is_trivially_copyable_v<ReferenceRoughConductorSample>);

} // namespace
} // namespace blackframe::renderer
