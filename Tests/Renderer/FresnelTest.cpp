#include <Blackframe/Renderer/Fresnel.hpp>
#include <array>
#include <cmath>
#include <concepts>
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

} // namespace
} // namespace blackframe::renderer
