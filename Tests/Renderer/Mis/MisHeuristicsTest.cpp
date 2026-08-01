#include <Blackframe/Renderer/MisHeuristics.hpp>
#include <cmath>
#include <concepts>
#include <gtest/gtest.h>
#include <limits>

namespace blackframe::renderer {
namespace {

template <MisScalar Scalar>
[[nodiscard]] MisProbabilityDensityT<Scalar>
pdf(const Scalar value, const ProbabilityMeasure measure = ProbabilityMeasure::solid_angle) {
    return MisProbabilityDensityT<Scalar>{
        .value = value,
        .measure = measure,
    };
}

template <MisScalar Scalar> void expect_analytic_weights() {
    const auto balance = balance_heuristic<Scalar>(pdf(Scalar{0.25}), pdf(Scalar{0.75}));
    const auto power = power_heuristic<Scalar>(pdf(Scalar{0.25}), pdf(Scalar{0.75}));
    ASSERT_TRUE(balance.has_value()) << balance.error().message;
    ASSERT_TRUE(power.has_value()) << power.error().message;
    EXPECT_EQ(*balance, Scalar{0.25});
    constexpr auto power_tolerance =
        std::same_as<Scalar, TransportScalar> ? ReferenceScalar{1.0e-7} : ReferenceScalar{2.0e-16};
    EXPECT_NEAR(static_cast<ReferenceScalar>(*power), ReferenceScalar{0.1}, power_tolerance);

    const auto equal_balance = balance_heuristic<Scalar>(pdf(Scalar{4}), pdf(Scalar{4}));
    const auto equal_power = power_heuristic<Scalar>(pdf(Scalar{4}), pdf(Scalar{4}));
    ASSERT_TRUE(equal_balance.has_value()) << equal_balance.error().message;
    ASSERT_TRUE(equal_power.has_value()) << equal_power.error().message;
    EXPECT_EQ(*equal_balance, Scalar{0.5});
    EXPECT_EQ(*equal_power, Scalar{0.5});

    const auto unsupported =
        mis_weight<Scalar>(MisHeuristic::power, pdf(Scalar{2}), pdf(Scalar{0}));
    ASSERT_TRUE(unsupported.has_value()) << unsupported.error().message;
    EXPECT_EQ(*unsupported, Scalar{1});
}

TEST(MisHeuristicsTest, MatchesAnalyticBalanceAndPowerWeightsInBothPrecisions) {
    static_assert(std::same_as<decltype(balance_heuristic<TransportScalar>(ProbabilityDensity{},
                                                                           ProbabilityDensity{})),
                               core::Result<TransportScalar>>);
    static_assert(std::same_as<decltype(power_heuristic<ReferenceScalar>(
                                   ReferenceProbabilityDensity{}, ReferenceProbabilityDensity{})),
                               core::Result<ReferenceScalar>>);
    static_assert(
        std::same_as<decltype(balance_heuristic(ProbabilityDensity{}, ProbabilityDensity{})),
                     core::Result<TransportScalar>>);
    static_assert(std::same_as<decltype(power_heuristic(ReferenceProbabilityDensity{},
                                                        ReferenceProbabilityDensity{})),
                               core::Result<ReferenceScalar>>);
    expect_analytic_weights<TransportScalar>();
    expect_analytic_weights<ReferenceScalar>();
}

template <MisScalar Scalar> void expect_complementary_and_scale_invariant() {
    constexpr auto tolerance =
        std::same_as<Scalar, TransportScalar> ? ReferenceScalar{2.0e-7} : ReferenceScalar{2.0e-15};
    for (const auto heuristic : {MisHeuristic::balance, MisHeuristic::power}) {
        const auto left = mis_weight<Scalar>(heuristic, pdf(Scalar{0.2}), pdf(Scalar{0.7}));
        const auto right = mis_weight<Scalar>(heuristic, pdf(Scalar{0.7}), pdf(Scalar{0.2}));
        const auto scaled = mis_weight<Scalar>(heuristic, pdf(Scalar{20}), pdf(Scalar{70}));
        ASSERT_TRUE(left.has_value()) << left.error().message;
        ASSERT_TRUE(right.has_value()) << right.error().message;
        ASSERT_TRUE(scaled.has_value()) << scaled.error().message;
        EXPECT_NEAR(static_cast<ReferenceScalar>(*left + *right), 1.0, tolerance);
        EXPECT_NEAR(static_cast<ReferenceScalar>(*left), static_cast<ReferenceScalar>(*scaled),
                    tolerance);
    }
}

TEST(MisHeuristicsTest, IsComplementaryAndInvariantToCommonPdfScale) {
    expect_complementary_and_scale_invariant<TransportScalar>();
    expect_complementary_and_scale_invariant<ReferenceScalar>();
}

template <MisScalar Scalar> void expect_extreme_pdfs_do_not_overflow() {
    const auto maximum = std::numeric_limits<Scalar>::max();
    const auto minimum = std::numeric_limits<Scalar>::denorm_min();
    for (const auto heuristic : {MisHeuristic::balance, MisHeuristic::power}) {
        const auto equal_maximum = mis_weight<Scalar>(heuristic, pdf(maximum), pdf(maximum));
        const auto equal_minimum = mis_weight<Scalar>(heuristic, pdf(minimum), pdf(minimum));
        const auto dominant = mis_weight<Scalar>(heuristic, pdf(maximum), pdf(minimum));
        ASSERT_TRUE(equal_maximum.has_value()) << equal_maximum.error().message;
        ASSERT_TRUE(equal_minimum.has_value()) << equal_minimum.error().message;
        ASSERT_TRUE(dominant.has_value()) << dominant.error().message;
        EXPECT_EQ(*equal_maximum, Scalar{0.5});
        EXPECT_EQ(*equal_minimum, Scalar{0.5});
        EXPECT_EQ(*dominant, Scalar{1});
    }
}

TEST(MisHeuristicsTest, AvoidsOverflowForExtremeRepresentablePdfs) {
    expect_extreme_pdfs_do_not_overflow<TransportScalar>();
    expect_extreme_pdfs_do_not_overflow<ReferenceScalar>();
}

template <MisScalar Scalar> void expect_positive_underflow_is_an_error() {
    const auto maximum = std::numeric_limits<Scalar>::max();
    const auto minimum = std::numeric_limits<Scalar>::denorm_min();
    for (const auto heuristic : {MisHeuristic::balance, MisHeuristic::power}) {
        const auto result = mis_weight<Scalar>(heuristic, pdf(minimum), pdf(maximum));
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
        EXPECT_FALSE(result.error().message.empty());
    }
}

TEST(MisHeuristicsTest, ReportsPositiveWeightsThatCannotBeRepresented) {
    expect_positive_underflow_is_an_error<TransportScalar>();
    expect_positive_underflow_is_an_error<ReferenceScalar>();
}

template <MisScalar Scalar>
void expect_invalid(const MisProbabilityDensityT<Scalar> sampled,
                    const MisProbabilityDensityT<Scalar> competing) {
    for (const auto heuristic : {MisHeuristic::balance, MisHeuristic::power}) {
        const auto result = mis_weight<Scalar>(heuristic, sampled, competing);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
        EXPECT_FALSE(result.error().message.empty());
    }
}

TEST(MisHeuristicsTest, RejectsInvalidPdfValuesAndMismatchedMeasures) {
    expect_invalid<TransportScalar>(pdf(0.0F), pdf(1.0F));
    expect_invalid<TransportScalar>(pdf(-1.0F), pdf(1.0F));
    expect_invalid<TransportScalar>(pdf(std::numeric_limits<float>::infinity()), pdf(1.0F));
    expect_invalid<TransportScalar>(pdf(std::numeric_limits<float>::quiet_NaN()), pdf(1.0F));
    expect_invalid<TransportScalar>(pdf(1.0F), pdf(-1.0F));
    expect_invalid<TransportScalar>(pdf(1.0F), pdf(std::numeric_limits<float>::infinity()));
    expect_invalid<TransportScalar>(pdf(1.0F), pdf(std::numeric_limits<float>::quiet_NaN()));
    expect_invalid<TransportScalar>(pdf(1.0F, ProbabilityMeasure::solid_angle),
                                    pdf(1.0F, ProbabilityMeasure::area));
    expect_invalid<TransportScalar>(pdf(1.01F, ProbabilityMeasure::discrete),
                                    pdf(0.5F, ProbabilityMeasure::discrete));
    expect_invalid<ReferenceScalar>(pdf(0.0), pdf(1.0));
}

TEST(MisHeuristicsTest, RejectsUnknownHeuristicWithoutSelectingADefault) {
    const auto result =
        mis_weight<TransportScalar>(static_cast<MisHeuristic>(255), pdf(1.0F), pdf(1.0F));
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    EXPECT_FALSE(result.error().message.empty());
}

template <MisScalar Scalar> void expect_unknown_probability_measure_is_rejected() {
    const auto unknown = static_cast<ProbabilityMeasure>(255);
    for (const auto heuristic : {MisHeuristic::balance, MisHeuristic::power}) {
        const auto result =
            mis_weight<Scalar>(heuristic, pdf(Scalar{1}, unknown), pdf(Scalar{1}, unknown));
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
        EXPECT_FALSE(result.error().message.empty());
    }
}

TEST(MisHeuristicsTest, RejectsUnknownProbabilityMeasuresInBothPrecisions) {
    expect_unknown_probability_measure_is_rejected<TransportScalar>();
    expect_unknown_probability_measure_is_rejected<ReferenceScalar>();
}

TEST(MisHeuristicsTest, AcceptsEqualDiscreteAndContinuousMeasures) {
    const auto discrete = balance_heuristic<TransportScalar>(
        pdf(0.25F, ProbabilityMeasure::discrete), pdf(0.75F, ProbabilityMeasure::discrete));
    const auto area = power_heuristic<ReferenceScalar>(pdf(2.0, ProbabilityMeasure::area),
                                                       pdf(6.0, ProbabilityMeasure::area));
    ASSERT_TRUE(discrete.has_value()) << discrete.error().message;
    ASSERT_TRUE(area.has_value()) << area.error().message;
    EXPECT_EQ(*discrete, 0.25F);
    EXPECT_NEAR(*area, 0.1, 2.0e-16);
}

template <MisScalar Scalar> void expect_joint_light_pdf() {
    const auto joint = joint_light_pdf<Scalar>(pdf(Scalar{0.25}, ProbabilityMeasure::discrete),
                                               pdf(Scalar{2}, ProbabilityMeasure::solid_angle));
    ASSERT_TRUE(joint.has_value()) << joint.error().message;
    EXPECT_EQ(joint->measure, ProbabilityMeasure::solid_angle);
    EXPECT_EQ(joint->value, Scalar{0.5});

    const auto zero = joint_light_pdf<Scalar>(pdf(Scalar{1}, ProbabilityMeasure::discrete),
                                              pdf(Scalar{0}, ProbabilityMeasure::solid_angle));
    ASSERT_TRUE(zero.has_value()) << zero.error().message;
    EXPECT_EQ(zero->measure, ProbabilityMeasure::solid_angle);
    EXPECT_EQ(zero->value, Scalar{0});

    const auto maximum = joint_light_pdf<Scalar>(
        pdf(Scalar{1}, ProbabilityMeasure::discrete),
        pdf(std::numeric_limits<Scalar>::max(), ProbabilityMeasure::solid_angle));
    ASSERT_TRUE(maximum.has_value()) << maximum.error().message;
    EXPECT_EQ(maximum->value, std::numeric_limits<Scalar>::max());
}

TEST(MisHeuristicsTest, FormsTheJointLightPdfExactlyOnceInBothPrecisions) {
    expect_joint_light_pdf<TransportScalar>();
    expect_joint_light_pdf<ReferenceScalar>();
}

TEST(MisHeuristicsTest, RejectsInvalidAndUnrepresentableJointLightPdfs) {
    const auto wrong_selection_measure = joint_light_pdf<TransportScalar>(
        pdf(0.5F, ProbabilityMeasure::solid_angle), pdf(1.0F, ProbabilityMeasure::solid_angle));
    const auto wrong_conditional_measure = joint_light_pdf<TransportScalar>(
        pdf(0.5F, ProbabilityMeasure::discrete), pdf(1.0F, ProbabilityMeasure::area));
    const auto zero_selection = joint_light_pdf<TransportScalar>(
        pdf(0.0F, ProbabilityMeasure::discrete), pdf(1.0F, ProbabilityMeasure::solid_angle));
    const auto underflow = joint_light_pdf<TransportScalar>(
        pdf(std::numeric_limits<float>::denorm_min(), ProbabilityMeasure::discrete),
        pdf(std::numeric_limits<float>::denorm_min(), ProbabilityMeasure::solid_angle));

    for (const auto* const result :
         {&wrong_selection_measure, &wrong_conditional_measure, &zero_selection, &underflow}) {
        ASSERT_FALSE(result->has_value());
        EXPECT_EQ(result->error().code, core::StatusCode::invalid_argument);
        EXPECT_FALSE(result->error().message.empty());
    }
}

} // namespace
} // namespace blackframe::renderer
