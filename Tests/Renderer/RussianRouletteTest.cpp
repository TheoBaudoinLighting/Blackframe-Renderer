#include <Blackframe/Renderer/RussianRoulette.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar>
using SpectrumFor = SampledSpectrum<TransportSpectrumSampleCount, Scalar>;

template <SpectrumScalar Scalar> using PolicyFor = RussianRoulettePolicyT<Scalar>;

template <SpectrumScalar Scalar>
[[nodiscard]] SpectrumFor<Scalar>
spectrum(const std::array<Scalar, TransportSpectrumSampleCount> values) {
    return SpectrumFor<Scalar>{.values = values};
}

template <typename Result>
void expect_error(const Result& result, const core::StatusCode expected_code) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, expected_code);
}

template <SpectrumScalar Scalar>
void expect_probability(const RussianRouletteResultT<Scalar>& result, const Scalar value) {
    EXPECT_EQ(result.survival_probability.value, value);
    EXPECT_EQ(result.survival_probability.measure, ProbabilityMeasure::discrete);
}

template <SpectrumScalar Scalar> void expect_policy_contract() {
    const auto disabled = PolicyFor<Scalar>::disabled();
    EXPECT_EQ(disabled.mode(), RussianRouletteMode::disabled);
    EXPECT_FALSE(disabled.is_enabled());
    EXPECT_EQ(disabled.first_eligible_depth(), 0U);
    EXPECT_EQ(disabled.minimum_survival_probability(), Scalar{0});
    EXPECT_EQ(disabled.maximum_survival_probability(), Scalar{0});
    EXPECT_TRUE(validate_russian_roulette_policy(disabled).has_value());

    const auto enabled = PolicyFor<Scalar>::create_enabled(
        std::numeric_limits<std::uint32_t>::max(), Scalar{0.25}, Scalar{0.75});
    ASSERT_TRUE(enabled.has_value());
    EXPECT_EQ(enabled->mode(), RussianRouletteMode::enabled);
    EXPECT_TRUE(enabled->is_enabled());
    EXPECT_EQ(enabled->first_eligible_depth(), std::numeric_limits<std::uint32_t>::max());
    EXPECT_EQ(enabled->minimum_survival_probability(), Scalar{0.25});
    EXPECT_EQ(enabled->maximum_survival_probability(), Scalar{0.75});
    EXPECT_TRUE(validate_russian_roulette_policy(*enabled).has_value());

    const auto fixed_probability = PolicyFor<Scalar>::create_enabled(1, Scalar{0.25}, Scalar{0.25});
    ASSERT_TRUE(fixed_probability.has_value());
    EXPECT_EQ(fixed_probability->minimum_survival_probability(), Scalar{0.25});
    EXPECT_EQ(fixed_probability->maximum_survival_probability(), Scalar{0.25});

    expect_error(PolicyFor<Scalar>::create_enabled(0, Scalar{0.25}, Scalar{0.75}),
                 core::StatusCode::invalid_argument);

    const auto denormal = std::numeric_limits<Scalar>::denorm_min();
    const auto infinity = std::numeric_limits<Scalar>::infinity();
    const auto nan = std::numeric_limits<Scalar>::quiet_NaN();
    for (const auto minimum :
         std::array{Scalar{0}, -Scalar{0}, -denormal, Scalar{1}, nan, infinity, -infinity}) {
        expect_error(PolicyFor<Scalar>::create_enabled(1, minimum, Scalar{0.75}),
                     core::StatusCode::invalid_argument);
    }
    for (const auto maximum :
         std::array{Scalar{0.125}, std::nextafter(Scalar{1}, infinity), nan, infinity, -infinity}) {
        expect_error(PolicyFor<Scalar>::create_enabled(1, Scalar{0.25}, maximum),
                     core::StatusCode::invalid_argument);
    }

    const auto throughput = spectrum<Scalar>({Scalar{0.125}, Scalar{0.25}, Scalar{0.5}, Scalar{1}});
    const auto disabled_result = evaluate_russian_roulette(
        throughput, Scalar{1}, std::numeric_limits<std::uint32_t>::max(), Scalar{0.75}, disabled);
    ASSERT_TRUE(disabled_result.has_value());
    EXPECT_EQ(disabled_result->throughput, throughput);
    expect_probability(*disabled_result, Scalar{1});
    EXPECT_EQ(disabled_result->outcome, RussianRouletteOutcome::not_evaluated);
}

TEST(RussianRouletteTest, CreatesOnlyExplicitValidatedPolicies) {
    static_assert(!std::same_as<RussianRoulettePolicy, ReferenceRussianRoulettePolicy>);
    static_assert(!std::default_initializable<RussianRoulettePolicy>);
    static_assert(!std::default_initializable<ReferenceRussianRoulettePolicy>);
    static_assert(std::is_standard_layout_v<RussianRoulettePolicy>);
    static_assert(std::is_trivially_copyable_v<RussianRoulettePolicy>);
    static_assert(std::is_standard_layout_v<ReferenceRussianRoulettePolicy>);
    static_assert(std::is_trivially_copyable_v<ReferenceRussianRoulettePolicy>);
    static_assert(std::is_standard_layout_v<RussianRouletteResult>);
    static_assert(std::is_trivially_copyable_v<RussianRouletteResult>);
    static_assert(std::is_standard_layout_v<ReferenceRussianRouletteResult>);
    static_assert(std::is_trivially_copyable_v<ReferenceRussianRouletteResult>);

    expect_policy_contract<TransportScalar>();
    expect_policy_contract<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_depth_threshold_and_exact_decisions() {
    const auto policy = PolicyFor<Scalar>::create_enabled(3, Scalar{0.125}, Scalar{0.75});
    ASSERT_TRUE(policy.has_value());
    const auto throughput =
        spectrum<Scalar>({Scalar{0.125}, Scalar{0.25}, Scalar{0.5}, Scalar{0.0625}});

    const auto before = evaluate_russian_roulette(throughput, Scalar{1}, 2, Scalar{0.75}, *policy);
    ASSERT_TRUE(before.has_value());
    EXPECT_EQ(before->throughput, throughput);
    expect_probability(*before, Scalar{1});
    EXPECT_EQ(before->outcome, RussianRouletteOutcome::not_evaluated);

    const auto terminated_at_equality =
        evaluate_russian_roulette(throughput, Scalar{1}, 3, Scalar{0.5}, *policy);
    ASSERT_TRUE(terminated_at_equality.has_value());
    EXPECT_EQ(terminated_at_equality->throughput, throughput);
    expect_probability(*terminated_at_equality, Scalar{0.5});
    EXPECT_EQ(terminated_at_equality->outcome, RussianRouletteOutcome::terminated);

    const auto survived =
        evaluate_russian_roulette(throughput, Scalar{1}, 3, Scalar{0.25}, *policy);
    ASSERT_TRUE(survived.has_value());
    EXPECT_EQ(survived->throughput, throughput / Scalar{0.5});
    expect_probability(*survived, Scalar{0.5});
    EXPECT_EQ(survived->outcome, RussianRouletteOutcome::survived);
}

TEST(RussianRouletteTest, AppliesTheDepthThresholdAndStrictSurvivalDecisionExactly) {
    expect_depth_threshold_and_exact_decisions<TransportScalar>();
    expect_depth_threshold_and_exact_decisions<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_eta_scale_and_probability_clamps() {
    const auto policy = PolicyFor<Scalar>::create_enabled(1, Scalar{0.25}, Scalar{0.75});
    ASSERT_TRUE(policy.has_value());
    const auto throughput =
        spectrum<Scalar>({Scalar{0.0625}, Scalar{0.125}, Scalar{0.5}, Scalar{0.25}});

    struct ProbabilityCase final {
        Scalar eta_scale;
        Scalar expected_probability;
    };
    for (const auto test_case : std::array{
             ProbabilityCase{.eta_scale = Scalar{0.25}, .expected_probability = Scalar{0.25}},
             ProbabilityCase{.eta_scale = Scalar{1}, .expected_probability = Scalar{0.5}},
             ProbabilityCase{.eta_scale = Scalar{2}, .expected_probability = Scalar{0.75}},
         }) {
        const auto evaluated =
            evaluate_russian_roulette(throughput, test_case.eta_scale, 1, Scalar{0}, *policy);
        ASSERT_TRUE(evaluated.has_value());
        expect_probability(*evaluated, test_case.expected_probability);
        EXPECT_EQ(evaluated->outcome, RussianRouletteOutcome::survived);
        EXPECT_EQ(evaluated->throughput, throughput / test_case.expected_probability);
    }

    const auto unit_policy = PolicyFor<Scalar>::create_enabled(1, Scalar{0.25}, Scalar{1});
    ASSERT_TRUE(unit_policy.has_value());
    const auto maximum = std::numeric_limits<Scalar>::max();
    const auto maximum_throughput =
        spectrum<Scalar>({maximum, Scalar{0.5}, Scalar{0.25}, Scalar{0}});
    const auto clamped_without_overflow =
        evaluate_russian_roulette(maximum_throughput, maximum, 1, Scalar{0.999}, *unit_policy);
    ASSERT_TRUE(clamped_without_overflow.has_value());
    expect_probability(*clamped_without_overflow, Scalar{1});
    EXPECT_EQ(clamped_without_overflow->outcome, RussianRouletteOutcome::survived);
    EXPECT_EQ(clamped_without_overflow->throughput, maximum_throughput);
}

TEST(RussianRouletteTest, UsesEtaScaleAndBothExplicitProbabilityClamps) {
    expect_eta_scale_and_probability_clamps<TransportScalar>();
    expect_eta_scale_and_probability_clamps<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_exact_quarter_probability_quadrature() {
    const auto policy = PolicyFor<Scalar>::create_enabled(1, Scalar{0.125}, Scalar{1});
    ASSERT_TRUE(policy.has_value());
    const auto throughput =
        spectrum<Scalar>({Scalar{0.0625}, Scalar{0.125}, Scalar{0.25}, Scalar{0.03125}});
    constexpr auto sample_count = std::size_t{4};
    const auto samples = std::array{Scalar{0.125}, Scalar{0.375}, Scalar{0.625}, Scalar{0.875}};

    auto contribution_sum = SpectrumFor<Scalar>{};
    auto survived_count = std::size_t{0};
    auto terminated_count = std::size_t{0};
    for (const auto sample : samples) {
        const auto evaluated = evaluate_russian_roulette(throughput, Scalar{1}, 1, sample, *policy);
        ASSERT_TRUE(evaluated.has_value());
        expect_probability(*evaluated, Scalar{0.25});
        if (evaluated->outcome == RussianRouletteOutcome::survived) {
            ++survived_count;
            contribution_sum = contribution_sum + evaluated->throughput;
        } else {
            EXPECT_EQ(evaluated->outcome, RussianRouletteOutcome::terminated);
            EXPECT_EQ(evaluated->throughput, throughput);
            ++terminated_count;
        }
    }

    EXPECT_EQ(survived_count, 1U);
    EXPECT_EQ(terminated_count, 3U);
    EXPECT_EQ(contribution_sum / static_cast<Scalar>(sample_count), throughput);
}

TEST(RussianRouletteTest, PreservesTheMeanExactlyForQuarterProbabilityQuadrature) {
    expect_exact_quarter_probability_quadrature<TransportScalar>();
    expect_exact_quarter_probability_quadrature<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_invalid_evaluation_inputs_rejected() {
    const auto policy = PolicyFor<Scalar>::create_enabled(1, Scalar{0.25}, Scalar{0.75});
    ASSERT_TRUE(policy.has_value());
    const auto disabled = PolicyFor<Scalar>::disabled();
    const auto valid_throughput =
        spectrum<Scalar>({Scalar{0.125}, Scalar{0.25}, Scalar{0.5}, Scalar{0.0625}});
    const auto infinity = std::numeric_limits<Scalar>::infinity();
    const auto nan = std::numeric_limits<Scalar>::quiet_NaN();
    const auto denormal = std::numeric_limits<Scalar>::denorm_min();

    for (const auto eta_scale :
         std::array{Scalar{0}, -Scalar{0}, -denormal, nan, infinity, -infinity}) {
        expect_error(evaluate_russian_roulette(valid_throughput, eta_scale, 1, Scalar{0}, *policy),
                     core::StatusCode::invalid_argument);
    }
    for (const auto sample : std::array{-denormal, Scalar{1}, std::nextafter(Scalar{1}, infinity),
                                        nan, infinity, -infinity}) {
        expect_error(evaluate_russian_roulette(valid_throughput, Scalar{1}, 1, sample, *policy),
                     core::StatusCode::invalid_argument);
    }

    for (const auto invalid_lane : std::array{-denormal, nan, infinity, -infinity}) {
        for (auto lane = std::size_t{0}; lane < TransportSpectrumSampleCount; ++lane) {
            auto throughput = valid_throughput;
            throughput[lane] = invalid_lane;
            expect_error(evaluate_russian_roulette(throughput, Scalar{1}, 1, Scalar{0}, *policy),
                         core::StatusCode::invalid_argument);
        }
    }
    expect_error(evaluate_russian_roulette(SpectrumFor<Scalar>{}, Scalar{1}, 1, Scalar{0}, *policy),
                 core::StatusCode::invalid_argument);

    expect_error(evaluate_russian_roulette(valid_throughput, Scalar{0}, 0, Scalar{0}, disabled),
                 core::StatusCode::invalid_argument);
    expect_error(evaluate_russian_roulette(valid_throughput, Scalar{1}, 0, Scalar{1}, disabled),
                 core::StatusCode::invalid_argument);
    auto negative_throughput = valid_throughput;
    negative_throughput[0] = -denormal;
    expect_error(evaluate_russian_roulette(negative_throughput, Scalar{1}, 0, Scalar{0}, disabled),
                 core::StatusCode::invalid_argument);
}

TEST(RussianRouletteTest, RejectsInvalidInputsEvenWhenThePolicyIsDisabled) {
    expect_invalid_evaluation_inputs_rejected<TransportScalar>();
    expect_invalid_evaluation_inputs_rejected<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_unrepresentable_compensation_rejected() {
    const auto minimum = std::numeric_limits<Scalar>::denorm_min();
    const auto policy = PolicyFor<Scalar>::create_enabled(1, minimum, Scalar{1});
    ASSERT_TRUE(policy.has_value());
    const auto throughput =
        spectrum<Scalar>({std::numeric_limits<Scalar>::max(), Scalar{0}, Scalar{0}, Scalar{0}});

    const auto evaluated = evaluate_russian_roulette(throughput, minimum, 1, Scalar{0}, *policy);
    ASSERT_FALSE(evaluated.has_value());
    EXPECT_EQ(evaluated.error().code, core::StatusCode::resource_exhausted);
}

TEST(RussianRouletteTest, RejectsUnrepresentableSurvivalCompensationWithoutFallback) {
    expect_unrepresentable_compensation_rejected<TransportScalar>();
    expect_unrepresentable_compensation_rejected<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
