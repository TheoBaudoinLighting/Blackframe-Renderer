#include <Blackframe/Renderer/Fresnel.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {
namespace {

template <FresnelScalar Scalar>
inline constexpr auto FresnelTolerance =
    std::same_as<Scalar, TransportScalar> ? ReferenceScalar{2.0e-6} : ReferenceScalar{2.0e-13};

template <FresnelScalar Scalar>
void expect_near(const core::Result<Scalar>& result, const ReferenceScalar expected) {
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NEAR(static_cast<ReferenceScalar>(*result), expected, FresnelTolerance<Scalar>);
}

template <FresnelScalar Scalar> void expect_analytic_values() {
    static_assert(std::same_as<decltype(dielectric_fresnel(Scalar{}, Scalar{}, Scalar{})),
                               core::Result<Scalar>>);

    expect_near(dielectric_fresnel(Scalar{1}, Scalar{1}, Scalar{1.5}), ReferenceScalar{0.04});
    expect_near(dielectric_fresnel(Scalar{0.5}, Scalar{1}, Scalar{1.5}),
                ReferenceScalar{0.08918671280221276});
    // cos(theta_i) = 3/4 and eta_i / eta_t = 3/2 give cos(theta_t) = 1/8.
    // The squared parallel and perpendicular amplitudes are 0.36 and 0.64 exactly.
    expect_near(dielectric_fresnel(Scalar{0.75}, Scalar{1.5}, Scalar{1}), ReferenceScalar{0.5});

    const auto brewster_cosine = Scalar{1} / std::sqrt(Scalar{3.25});
    expect_near(dielectric_fresnel(brewster_cosine, Scalar{1}, Scalar{1.5}),
                ReferenceScalar{25} / ReferenceScalar{338});
}

TEST(FresnelTest, MatchesExactAnalyticDielectricValuesInBothPrecisions) {
    expect_analytic_values<TransportScalar>();
    expect_analytic_values<ReferenceScalar>();
}

template <FresnelScalar Scalar> void expect_reciprocity_and_scale_invariance() {
    constexpr auto incident_cosine = Scalar{0.8};
    const auto transmitted_cosine = std::sqrt(Scalar{0.84});
    const auto entering = dielectric_fresnel(incident_cosine, Scalar{1}, Scalar{1.5});
    const auto leaving = dielectric_fresnel(transmitted_cosine, Scalar{1.5}, Scalar{1});
    ASSERT_TRUE(entering.has_value()) << entering.error().message;
    ASSERT_TRUE(leaving.has_value()) << leaving.error().message;
    EXPECT_NEAR(static_cast<ReferenceScalar>(*entering), ReferenceScalar{0.04389473600344824},
                FresnelTolerance<Scalar>);
    EXPECT_NEAR(static_cast<ReferenceScalar>(*leaving), static_cast<ReferenceScalar>(*entering),
                FresnelTolerance<Scalar>);

    const auto scaled = dielectric_fresnel(Scalar{0.5}, Scalar{2}, Scalar{3});
    const auto unscaled = dielectric_fresnel(Scalar{0.5}, Scalar{1}, Scalar{1.5});
    ASSERT_TRUE(scaled.has_value()) << scaled.error().message;
    ASSERT_TRUE(unscaled.has_value()) << unscaled.error().message;
    EXPECT_NEAR(static_cast<ReferenceScalar>(*scaled), static_cast<ReferenceScalar>(*unscaled),
                FresnelTolerance<Scalar>);
}

TEST(FresnelTest, PreservesSnellReciprocityAndIndexScaleInBothPrecisions) {
    expect_reciprocity_and_scale_invariance<TransportScalar>();
    expect_reciprocity_and_scale_invariance<ReferenceScalar>();
}

template <FresnelScalar Scalar> void expect_total_internal_reflection_and_limits() {
    for (const auto cosine : std::array{Scalar{0.5}, Scalar{0.745}}) {
        const auto reflected = dielectric_fresnel(cosine, Scalar{1.5}, Scalar{1});
        ASSERT_TRUE(reflected.has_value()) << reflected.error().message;
        EXPECT_EQ(*reflected, Scalar{1});
    }

    const auto transmitted = dielectric_fresnel(Scalar{0.746}, Scalar{1.5}, Scalar{1});
    ASSERT_TRUE(transmitted.has_value()) << transmitted.error().message;
    EXPECT_GT(*transmitted, Scalar{0});
    EXPECT_LT(*transmitted, Scalar{1});

    // Construct the critical ratio in the same precision as the evaluator. This locks the
    // inclusive boundary without depending on a decimal approximation of the critical angle.
    constexpr auto constructed_cosine = Scalar{0.75};
    const auto critical_transmitted_eta =
        std::sqrt((Scalar{1} - constructed_cosine) * (Scalar{1} + constructed_cosine));
    for (const auto eta : std::array{critical_transmitted_eta,
                                     std::nextafter(critical_transmitted_eta, Scalar{0})}) {
        const auto critical = dielectric_fresnel(constructed_cosine, Scalar{1}, eta);
        ASSERT_TRUE(critical.has_value()) << critical.error().message;
        EXPECT_EQ(*critical, Scalar{1});
    }
    const auto above_critical_eta = std::nextafter(critical_transmitted_eta, Scalar{1});
    const auto below_critical_angle =
        dielectric_fresnel(constructed_cosine, Scalar{1}, above_critical_eta);
    ASSERT_TRUE(below_critical_angle.has_value()) << below_critical_angle.error().message;
    EXPECT_GT(*below_critical_angle, Scalar{0});
    EXPECT_LT(*below_critical_angle, Scalar{1});

    const auto grazing = dielectric_fresnel(Scalar{0}, Scalar{1}, Scalar{1.5});
    ASSERT_TRUE(grazing.has_value()) << grazing.error().message;
    EXPECT_EQ(*grazing, Scalar{1});

    for (const auto cosine : std::array{Scalar{0}, Scalar{0.5}, Scalar{1}}) {
        const auto homogeneous = dielectric_fresnel(cosine, Scalar{1.5}, Scalar{1.5});
        ASSERT_TRUE(homogeneous.has_value()) << homogeneous.error().message;
        EXPECT_EQ(*homogeneous, Scalar{0});
        EXPECT_FALSE(std::signbit(*homogeneous));
    }

    const auto maximum = std::numeric_limits<Scalar>::max();
    const auto minimum = std::numeric_limits<Scalar>::denorm_min();
    expect_near(dielectric_fresnel(Scalar{1}, maximum, maximum / Scalar{2}),
                ReferenceScalar{1} / ReferenceScalar{9});
    expect_near(dielectric_fresnel(Scalar{1}, maximum / Scalar{2}, maximum),
                ReferenceScalar{1} / ReferenceScalar{9});
    expect_near(dielectric_fresnel(Scalar{0.5}, (Scalar{2} / Scalar{3}) * maximum, maximum),
                ReferenceScalar{0.08918671280221276});
    expect_near(dielectric_fresnel(Scalar{0.5}, Scalar{2} * minimum, Scalar{3} * minimum),
                ReferenceScalar{0.08918671280221276});

    for (const auto reflected : std::array{dielectric_fresnel(Scalar{0.5}, minimum, maximum),
                                           dielectric_fresnel(Scalar{0.5}, maximum, minimum)}) {
        ASSERT_TRUE(reflected.has_value()) << reflected.error().message;
        EXPECT_EQ(*reflected, Scalar{1});
    }
}

TEST(FresnelTest, HandlesTotalInternalReflectionAndLimitsInBothPrecisions) {
    expect_total_internal_reflection_and_limits<TransportScalar>();
    expect_total_internal_reflection_and_limits<ReferenceScalar>();
}

template <typename Result> void expect_invalid(const Result& result) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(result.error().message.empty());
}

template <FresnelScalar Scalar> void expect_invalid_inputs() {
    const auto infinity = std::numeric_limits<Scalar>::infinity();
    const auto nan = std::numeric_limits<Scalar>::quiet_NaN();
    const auto negative_subnormal = -std::numeric_limits<Scalar>::denorm_min();
    const auto above_one = std::nextafter(Scalar{1}, infinity);
    for (const auto cosine :
         std::array{Scalar{-0.5}, negative_subnormal, above_one, nan, infinity, -infinity}) {
        expect_invalid(dielectric_fresnel(cosine, Scalar{1}, Scalar{1.5}));
    }

    for (const auto invalid_eta : std::array{Scalar{0}, Scalar{-0.0}, negative_subnormal,
                                             Scalar{-1}, nan, infinity, -infinity}) {
        expect_invalid(dielectric_fresnel(Scalar{0.5}, invalid_eta, Scalar{1.5}));
        expect_invalid(dielectric_fresnel(Scalar{0.5}, Scalar{1}, invalid_eta));
    }
}

TEST(FresnelTest, RejectsInvalidInputsWithoutClampSwapOrFallback) {
    expect_invalid_inputs<TransportScalar>();
    expect_invalid_inputs<ReferenceScalar>();
}

template <FresnelScalar Scalar>
using FresnelSpectrumFor = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

struct ConductorCurveSample final {
    ReferenceScalar incident_cosine{};
    std::array<ReferenceScalar, TransportSpectrumSampleCount> reflectance{};
};

inline constexpr auto SyntheticConductorCurve = std::array{
    ConductorCurveSample{
        .incident_cosine = 1.0,
        .reflectance = {0.1111111111111111, 0.2, 0.9233716475095783, 0.1111111111111111},
    },
    ConductorCurveSample{
        .incident_cosine = 0.9,
        .reflectance = {0.11208336739730956, 0.2032054795126786, 0.9231305868390667,
                        0.16456302436995565},
    },
    ConductorCurveSample{
        .incident_cosine = 0.75,
        .reflectance = {0.11890534281700538, 0.22216899152963052, 0.9218639379899276, 1.0},
    },
    ConductorCurveSample{
        .incident_cosine = 0.5,
        .reflectance = {0.16137659558805587, 0.3075653821238875, 0.9184110846593685, 1.0},
    },
    ConductorCurveSample{
        .incident_cosine = 0.25,
        .reflectance = {0.3184400826446281, 0.5128997581743571, 0.9261790871260233, 1.0},
    },
    ConductorCurveSample{
        .incident_cosine = 0.0,
        .reflectance = {1.0, 1.0, 1.0, 1.0},
    },
};

template <FresnelScalar Scalar> void expect_synthetic_conductor_curve() {
    const auto eta = FresnelSpectrumFor<Scalar>{
        .values = {Scalar{2}, Scalar{1}, Scalar{0.2}, Scalar{0.5}},
    };
    const auto k = FresnelSpectrumFor<Scalar>{
        .values = {Scalar{0}, Scalar{1}, Scalar{3}, Scalar{0}},
    };
    static_assert(std::same_as<decltype(conductor_fresnel(Scalar{}, eta, k)),
                               core::Result<FresnelSpectrumFor<Scalar>>>);

    for (const auto& sample : SyntheticConductorCurve) {
        const auto evaluated =
            conductor_fresnel(static_cast<Scalar>(sample.incident_cosine), eta, k);
        ASSERT_TRUE(evaluated.has_value()) << evaluated.error().message;
        for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
            EXPECT_NEAR(static_cast<ReferenceScalar>((*evaluated)[lane]), sample.reflectance[lane],
                        FresnelTolerance<Scalar>);
            EXPECT_TRUE(std::isfinite((*evaluated)[lane]));
            EXPECT_GE((*evaluated)[lane], Scalar{0});
            EXPECT_LE((*evaluated)[lane], Scalar{1});
        }
    }
}

TEST(FresnelTest, MatchesExactSyntheticConductorCurvesInBothPrecisions) {
    expect_synthetic_conductor_curve<TransportScalar>();
    expect_synthetic_conductor_curve<ReferenceScalar>();
}

template <FresnelScalar Scalar> void expect_conductor_dielectric_limit_and_lane_order() {
    const auto eta = FresnelSpectrumFor<Scalar>{
        .values = {Scalar{2}, Scalar{0.5}, Scalar{1.5}, Scalar{0.8}},
    };
    const auto zero_k = FresnelSpectrumFor<Scalar>{};
    for (const auto cosine : std::array{Scalar{0.9}, Scalar{0.75}, Scalar{0.5}}) {
        const auto conductor = conductor_fresnel(cosine, eta, zero_k);
        ASSERT_TRUE(conductor.has_value()) << conductor.error().message;
        for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
            const auto dielectric = dielectric_fresnel(cosine, Scalar{1}, eta[lane]);
            ASSERT_TRUE(dielectric.has_value()) << dielectric.error().message;
            EXPECT_NEAR(static_cast<ReferenceScalar>((*conductor)[lane]),
                        static_cast<ReferenceScalar>(*dielectric), FresnelTolerance<Scalar>);
        }
    }

    constexpr auto permutation = std::array<std::size_t, TransportSpectrumSampleCount>{2, 0, 3, 1};
    const auto source_eta = FresnelSpectrumFor<Scalar>{
        .values = {Scalar{2}, Scalar{1}, Scalar{0.2}, Scalar{0.5}},
    };
    const auto source_k = FresnelSpectrumFor<Scalar>{
        .values = {Scalar{0}, Scalar{1}, Scalar{3}, Scalar{0}},
    };
    auto permuted_eta = FresnelSpectrumFor<Scalar>{};
    auto permuted_k = FresnelSpectrumFor<Scalar>{};
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        permuted_eta[lane] = source_eta[permutation[lane]];
        permuted_k[lane] = source_k[permutation[lane]];
    }
    const auto source = conductor_fresnel(Scalar{0.5}, source_eta, source_k);
    const auto permuted = conductor_fresnel(Scalar{0.5}, permuted_eta, permuted_k);
    ASSERT_TRUE(source.has_value()) << source.error().message;
    ASSERT_TRUE(permuted.has_value()) << permuted.error().message;
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_EQ((*permuted)[lane], (*source)[permutation[lane]]);
    }
}

TEST(FresnelTest, PreservesDielectricLimitAndSpectralLaneOrderInBothPrecisions) {
    expect_conductor_dielectric_limit_and_lane_order<TransportScalar>();
    expect_conductor_dielectric_limit_and_lane_order<ReferenceScalar>();
}

template <FresnelScalar Scalar> void expect_conductor_limits_and_extremes() {
    const auto homogeneous_eta = FresnelSpectrumFor<Scalar>{
        .values = {Scalar{1}, Scalar{1}, Scalar{1}, Scalar{1}},
    };
    const auto zero_k = FresnelSpectrumFor<Scalar>{};
    for (const auto cosine : std::array{Scalar{0}, Scalar{0.5}, Scalar{1}}) {
        const auto homogeneous = conductor_fresnel(cosine, homogeneous_eta, zero_k);
        ASSERT_TRUE(homogeneous.has_value()) << homogeneous.error().message;
        EXPECT_EQ(*homogeneous, FresnelSpectrumFor<Scalar>{});
    }

    const auto grazing_k = FresnelSpectrumFor<Scalar>{
        .values = {Scalar{0}, Scalar{0}, Scalar{1}, std::numeric_limits<Scalar>::max()},
    };
    const auto grazing_eta = FresnelSpectrumFor<Scalar>{
        .values = {Scalar{1}, Scalar{2}, Scalar{1}, Scalar{1}},
    };
    const auto grazing = conductor_fresnel(Scalar{0}, grazing_eta, grazing_k);
    ASSERT_TRUE(grazing.has_value()) << grazing.error().message;
    EXPECT_EQ(grazing->values, (std::array<Scalar, TransportSpectrumSampleCount>{
                                   Scalar{0}, Scalar{1}, Scalar{1}, Scalar{1}}));

    const auto maximum = std::numeric_limits<Scalar>::max();
    const auto minimum = std::numeric_limits<Scalar>::denorm_min();
    const auto extreme_eta = FresnelSpectrumFor<Scalar>{
        .values = {minimum, maximum, Scalar{1}, maximum},
    };
    const auto extreme_k = FresnelSpectrumFor<Scalar>{
        .values = {Scalar{1}, Scalar{0}, maximum, maximum},
    };
    const auto extreme = conductor_fresnel(Scalar{0.5}, extreme_eta, extreme_k);
    ASSERT_TRUE(extreme.has_value()) << extreme.error().message;
    for (const auto value : extreme->values) {
        EXPECT_EQ(value, Scalar{1});
    }

    const auto negligible_k = FresnelSpectrumFor<Scalar>{
        .values = {minimum, minimum, minimum, minimum},
    };
    const auto eta_two = FresnelSpectrumFor<Scalar>{
        .values = {Scalar{2}, Scalar{2}, Scalar{2}, Scalar{2}},
    };
    const auto with_negligible_k = conductor_fresnel(Scalar{0.5}, eta_two, negligible_k);
    const auto without_k = conductor_fresnel(Scalar{0.5}, eta_two, zero_k);
    ASSERT_TRUE(with_negligible_k.has_value()) << with_negligible_k.error().message;
    ASSERT_TRUE(without_k.has_value()) << without_k.error().message;
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        EXPECT_EQ((*with_negligible_k)[lane], (*without_k)[lane]);
    }
}

TEST(FresnelTest, HandlesConductorLimitsAndFiniteExtremesInBothPrecisions) {
    expect_conductor_limits_and_extremes<TransportScalar>();
    expect_conductor_limits_and_extremes<ReferenceScalar>();
}

template <FresnelScalar Scalar> void expect_invalid_conductor_inputs() {
    const auto infinity = std::numeric_limits<Scalar>::infinity();
    const auto nan = std::numeric_limits<Scalar>::quiet_NaN();
    const auto homogeneous_eta = FresnelSpectrumFor<Scalar>{
        .values = {Scalar{1}, Scalar{1}, Scalar{1}, Scalar{1}},
    };
    const auto zero_k = FresnelSpectrumFor<Scalar>{};
    auto eta = FresnelSpectrumFor<Scalar>{
        .values = {Scalar{1}, Scalar{2}, Scalar{3}, Scalar{4}},
    };
    auto k = FresnelSpectrumFor<Scalar>{
        .values = {Scalar{0}, Scalar{1}, Scalar{2}, Scalar{3}},
    };

    for (const auto cosine :
         std::array{Scalar{-0.5}, std::nextafter(Scalar{1}, infinity), nan, infinity, -infinity}) {
        expect_invalid(conductor_fresnel(cosine, eta, k));
    }
    for (auto lane = std::size_t{}; lane < TransportSpectrumSampleCount; ++lane) {
        for (const auto invalid_eta :
             std::array{Scalar{0}, Scalar{-0.0}, -std::numeric_limits<Scalar>::denorm_min(),
                        Scalar{-1}, nan, infinity, -infinity}) {
            auto malformed = eta;
            malformed[lane] = invalid_eta;
            expect_invalid(conductor_fresnel(Scalar{0.5}, malformed, k));
        }
        for (const auto invalid_k : std::array{-std::numeric_limits<Scalar>::denorm_min(),
                                               Scalar{-1}, nan, infinity, -infinity}) {
            auto malformed = k;
            malformed[lane] = invalid_k;
            expect_invalid(conductor_fresnel(Scalar{0.5}, eta, malformed));
        }
    }

    auto negative_zero_k = k;
    negative_zero_k[0] = Scalar{-0.0};
    const auto signed_zero = conductor_fresnel(Scalar{0.5}, eta, negative_zero_k);
    const auto positive_zero = conductor_fresnel(Scalar{0.5}, eta, k);
    ASSERT_TRUE(signed_zero.has_value()) << signed_zero.error().message;
    ASSERT_TRUE(positive_zero.has_value()) << positive_zero.error().message;
    EXPECT_EQ(*signed_zero, *positive_zero);

    auto unrepresentable_k = zero_k;
    unrepresentable_k[0] = std::numeric_limits<Scalar>::denorm_min();
    expect_invalid(conductor_fresnel(Scalar{1}, homogeneous_eta, unrepresentable_k));
}

TEST(FresnelTest, RejectsInvalidOrUnrepresentableConductorPacketsWithoutFallback) {
    expect_invalid_conductor_inputs<TransportScalar>();
    expect_invalid_conductor_inputs<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
