#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/SampleDimensionMap.hpp>
#include <Blackframe/Renderer/SampleStream.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar> using SamplerFor = LightSamplerT<Scalar>;
template <SpectrumScalar Scalar> using SpectrumFor = LightSpectrumT<Scalar>;

template <typename Value> [[nodiscard]] Value require_value(core::Result<Value> result) {
    if (!result) {
        ADD_FAILURE() << result.error().message;
        std::abort();
    }
    return std::move(*result);
}

template <typename Result>
void expect_error(const Result& result, const core::StatusCode expected_code) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, expected_code);
    EXPECT_FALSE(result.error().message.empty());
}

template <SpectrumScalar Scalar>
[[nodiscard]] SpectrumFor<Scalar> lanes(const Scalar lane0, const Scalar lane1, const Scalar lane2,
                                        const Scalar lane3) {
    return SpectrumFor<Scalar>{.values = {lane0, lane1, lane2, lane3}};
}

template <SpectrumScalar Scalar> [[nodiscard]] SpectrumFor<Scalar> filled(const Scalar value) {
    auto result = SpectrumFor<Scalar>{};
    result.values.fill(value);
    return result;
}

TEST(LightSamplerContractTest, ExposesDistinctTransportAndReferenceTypes) {
    static_assert(std::is_same_v<LightSampler, LightSamplerT<TransportScalar>>);
    static_assert(std::is_same_v<ReferenceLightSampler, LightSamplerT<ReferenceScalar>>);
    static_assert(!std::is_same_v<LightSampler, ReferenceLightSampler>);
    static_assert(std::is_same_v<decltype(std::declval<LightSelectionProbability>().value()),
                                 TransportScalar>);
    static_assert(
        std::is_same_v<decltype(std::declval<ReferenceLightSelectionProbability>().value()),
                       ReferenceScalar>);
    static_assert(
        std::is_same_v<decltype(std::declval<LightSelection>().light_index()), std::uint32_t>);
}

template <SpectrumScalar Scalar> void expect_exact_uniform_frequencies() {
    constexpr auto sample_count = std::uint32_t{4096};
    constexpr auto light_count = std::size_t{4};
    const auto sampler = require_value(SamplerFor<Scalar>::create_uniform(light_count));

    EXPECT_EQ(sampler.strategy(), LightSamplingStrategy::uniform);
    EXPECT_EQ(sampler.light_count(), light_count);
    auto counts = std::array<std::uint32_t, light_count>{};
    for (auto sample_index = std::uint32_t{}; sample_index < sample_count; ++sample_index) {
        const auto canonical =
            (static_cast<Scalar>(sample_index) + Scalar{0.5}) / static_cast<Scalar>(sample_count);
        const auto selection = sampler.sample(canonical);
        ASSERT_TRUE(selection.has_value()) << selection.error().message;
        ASSERT_LT(selection->light_index(), counts.size());
        ++counts[selection->light_index()];

        const auto queried = sampler.probability(selection->light_index());
        ASSERT_TRUE(queried.has_value()) << queried.error().message;
        EXPECT_EQ(selection->probability(), *queried);
        EXPECT_EQ(selection->probability().measure(), ProbabilityMeasure::discrete);
        EXPECT_EQ(selection->probability().probability_density().measure,
                  ProbabilityMeasure::discrete);
        EXPECT_EQ(selection->probability().value(), Scalar{0.25});
    }
    EXPECT_EQ(counts, (std::array<std::uint32_t, light_count>{1024, 1024, 1024, 1024}));
}

TEST(LightSamplerTest, UniformSelectionHasExactFrequencies) {
    expect_exact_uniform_frequencies<TransportScalar>();
    expect_exact_uniform_frequencies<ReferenceScalar>();
}

template <SpectrumScalar Scalar>
[[nodiscard]] std::array<SpectrumFor<Scalar>, 5> weighted_test_powers() {
    return {
        lanes<Scalar>(Scalar{4}, Scalar{0}, Scalar{0}, Scalar{0}),
        SpectrumFor<Scalar>{},
        lanes<Scalar>(Scalar{0}, Scalar{4}, Scalar{0}, Scalar{0}),
        lanes<Scalar>(Scalar{0}, Scalar{0}, Scalar{8}, Scalar{0}),
        lanes<Scalar>(Scalar{0}, Scalar{0}, Scalar{0}, Scalar{16}),
    };
}

template <SpectrumScalar Scalar> void expect_exact_power_frequencies() {
    constexpr auto sample_count = std::uint32_t{4096};
    constexpr auto expected_counts = std::array<std::uint32_t, 5>{512, 0, 512, 1024, 2048};
    constexpr auto expected_probabilities =
        std::array<Scalar, 5>{Scalar{0.125}, Scalar{0}, Scalar{0.125}, Scalar{0.25}, Scalar{0.5}};
    const auto powers = weighted_test_powers<Scalar>();
    const auto sampler = require_value(
        SamplerFor<Scalar>::create_power_weighted(std::span<const SpectrumFor<Scalar>>{powers}));

    EXPECT_EQ(sampler.strategy(), LightSamplingStrategy::power_weighted);
    EXPECT_EQ(sampler.light_count(), powers.size());
    auto counts = std::array<std::uint32_t, powers.size()>{};
    for (auto sample_index = std::uint32_t{}; sample_index < sample_count; ++sample_index) {
        const auto canonical =
            (static_cast<Scalar>(sample_index) + Scalar{0.5}) / static_cast<Scalar>(sample_count);
        const auto selection = sampler.sample(canonical);
        ASSERT_TRUE(selection.has_value()) << selection.error().message;
        ASSERT_LT(selection->light_index(), counts.size());
        ++counts[selection->light_index()];
        EXPECT_EQ(selection->probability().value(),
                  expected_probabilities[selection->light_index()]);
        const auto queried = sampler.probability(selection->light_index());
        ASSERT_TRUE(queried.has_value()) << queried.error().message;
        EXPECT_EQ(selection->probability(), *queried);
    }
    EXPECT_EQ(counts, expected_counts);

    auto probability_sum = Scalar{0};
    for (auto index = std::uint32_t{}; index < sampler.light_count(); ++index) {
        const auto probability = sampler.probability(index);
        ASSERT_TRUE(probability.has_value()) << probability.error().message;
        EXPECT_EQ(probability->value(), expected_probabilities[index]);
        EXPECT_EQ(probability->measure(), ProbabilityMeasure::discrete);
        probability_sum += probability->value();
    }
    EXPECT_EQ(probability_sum, Scalar{1});
}

TEST(LightSamplerTest, PowerSelectionHasExactFourLaneFrequencies) {
    expect_exact_power_frequencies<TransportScalar>();
    expect_exact_power_frequencies<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_boundary_classification() {
    const auto powers = weighted_test_powers<Scalar>();
    const auto sampler = require_value(
        SamplerFor<Scalar>::create_power_weighted(std::span<const SpectrumFor<Scalar>>{powers}));
    const auto previous_one = std::nextafter(Scalar{1}, Scalar{0});
    const auto previous_eighth = std::nextafter(Scalar{0.125}, Scalar{0});

    EXPECT_EQ(require_value(sampler.sample(Scalar{0})).light_index(), 0U);
    EXPECT_EQ(require_value(sampler.sample(previous_eighth)).light_index(), 0U);
    EXPECT_EQ(require_value(sampler.sample(Scalar{0.125})).light_index(), 2U);
    EXPECT_EQ(require_value(sampler.sample(Scalar{0.25})).light_index(), 3U);
    EXPECT_EQ(require_value(sampler.sample(Scalar{0.5})).light_index(), 4U);
    EXPECT_EQ(require_value(sampler.sample(previous_one)).light_index(), 4U);
    EXPECT_EQ(require_value(sampler.probability(1U)).value(), Scalar{0});
}

TEST(LightSamplerTest, ExactBoundariesSkipZeroPowerSlots) {
    expect_boundary_classification<TransportScalar>();
    expect_boundary_classification<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_single_support(const std::size_t supported_index) {
    auto powers = std::array<SpectrumFor<Scalar>, 3>{};
    powers[supported_index] = filled<Scalar>(Scalar{1});
    const auto sampler = require_value(
        SamplerFor<Scalar>::create_power_weighted(std::span<const SpectrumFor<Scalar>>{powers}));

    EXPECT_EQ(require_value(sampler.sample(Scalar{0})).light_index(), supported_index);
    EXPECT_EQ(require_value(sampler.sample(std::nextafter(Scalar{1}, Scalar{0}))).light_index(),
              supported_index);
    for (auto index = std::uint32_t{}; index < sampler.light_count(); ++index) {
        const auto probability = require_value(sampler.probability(index));
        EXPECT_EQ(probability.value(), index == supported_index ? Scalar{1} : Scalar{0});
    }
}

TEST(LightSamplerTest, LeadingAndTrailingZeroPowerRemainOutsideSupport) {
    for (auto index = std::size_t{}; index < 3U; ++index) {
        expect_single_support<TransportScalar>(index);
        expect_single_support<ReferenceScalar>(index);
    }
}

template <SpectrumScalar Scalar> void expect_indexed_stream_frequencies() {
    constexpr auto sample_count = std::uint32_t{65'536};
    constexpr auto expected_probabilities =
        std::array<ReferenceScalar, 5>{0.125, 0.0, 0.125, 0.25, 0.5};
    const auto dimensions = sample_dimensions_for_bounce(0U);
    ASSERT_TRUE(dimensions.has_value()) << dimensions.error().message;
    const auto powers = weighted_test_powers<Scalar>();
    const auto sampler = require_value(
        SamplerFor<Scalar>::create_power_weighted(std::span<const SpectrumFor<Scalar>>{powers}));
    auto counts = std::array<std::uint32_t, powers.size()>{};
    for (auto sample_index = std::uint32_t{}; sample_index < sample_count; ++sample_index) {
        const auto stream = SampleStreamT<Scalar>{SampleStreamIndex{
            .pixel_x = 37U,
            .pixel_y = 91U,
            .sample_index = sample_index,
            .seed = 0xD1B54A32D192ED03ULL,
        }};
        const auto selection = sampler.sample(stream.sample_1d(dimensions->light_selection));
        ASSERT_TRUE(selection.has_value()) << selection.error().message;
        ++counts[selection->light_index()];
    }

    for (auto index = std::size_t{}; index < counts.size(); ++index) {
        const auto probability = expected_probabilities[index];
        const auto expected = static_cast<ReferenceScalar>(sample_count) * probability;
        const auto sigma = std::sqrt(static_cast<ReferenceScalar>(sample_count) * probability *
                                     (1.0 - probability));
        const auto tolerance = 8.0 * sigma + 1.0;
        EXPECT_LE(std::abs(static_cast<ReferenceScalar>(counts[index]) - expected), tolerance);
    }
}

TEST(LightSamplerTest, IndexedSampleStreamFrequenciesMatchTheDiscretePdf) {
    expect_indexed_stream_frequencies<TransportScalar>();
    expect_indexed_stream_frequencies<ReferenceScalar>();
}

template <SpectrumScalar Scalar>
void expect_stratified_frequencies_match_stored_pdf(const SamplerFor<Scalar>& sampler) {
    constexpr auto sample_count = std::uint32_t{65'536};
    auto counts = std::vector<std::uint32_t>(sampler.light_count());
    for (auto sample_index = std::uint32_t{}; sample_index < sample_count; ++sample_index) {
        const auto canonical =
            (static_cast<Scalar>(sample_index) + Scalar{0.5}) / static_cast<Scalar>(sample_count);
        const auto selection = sampler.sample(canonical);
        ASSERT_TRUE(selection.has_value()) << selection.error().message;
        ++counts[selection->light_index()];
    }

    auto probability_sum = ReferenceScalar{0};
    for (auto index = std::uint32_t{}; index < sampler.light_count(); ++index) {
        const auto probability = sampler.probability(index);
        ASSERT_TRUE(probability.has_value()) << probability.error().message;
        probability_sum += static_cast<ReferenceScalar>(probability->value());
        const auto expected_count = static_cast<ReferenceScalar>(probability->value()) *
                                    static_cast<ReferenceScalar>(sample_count);
        EXPECT_LE(std::abs(static_cast<ReferenceScalar>(counts[index]) - expected_count), 1.0);
    }
    EXPECT_NEAR(probability_sum, 1.0,
                8.0 * static_cast<ReferenceScalar>(std::numeric_limits<Scalar>::epsilon()));
}

template <SpectrumScalar Scalar> void expect_non_dyadic_frequency_conformance() {
    const auto uniform = require_value(SamplerFor<Scalar>::create_uniform(7U));
    expect_stratified_frequencies_match_stored_pdf(uniform);

    const auto powers = std::array{
        filled<Scalar>(Scalar{1}), filled<Scalar>(Scalar{2}), filled<Scalar>(Scalar{3}),
        filled<Scalar>(Scalar{5}), filled<Scalar>(Scalar{7}),
    };
    const auto weighted = require_value(
        SamplerFor<Scalar>::create_power_weighted(std::span<const SpectrumFor<Scalar>>{powers}));
    expect_stratified_frequencies_match_stored_pdf(weighted);
}

TEST(LightSamplerTest, NonDyadicFrequenciesMatchTheAuthoritativePdf) {
    expect_non_dyadic_frequency_conformance<TransportScalar>();
    expect_non_dyadic_frequency_conformance<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_moved_from_state_is_explicit() {
    auto source = require_value(SamplerFor<Scalar>::create_uniform(4U));
    auto destination = std::move(source);

    EXPECT_EQ(destination.light_count(), 4U);
    EXPECT_EQ(require_value(destination.sample(Scalar{0.375})).light_index(), 1U);
    EXPECT_EQ(source.light_count(), 0U);
    expect_error(source.sample(Scalar{0.5}), core::StatusCode::internal_error);
    expect_error(source.probability(0U), core::StatusCode::invalid_argument);
}

TEST(LightSamplerTest, MovedFromSamplerFailsExplicitlyWithoutUndefinedAccess) {
    expect_moved_from_state_is_explicit<TransportScalar>();
    expect_moved_from_state_is_explicit<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_scale_and_range_robustness() {
    const auto base = std::array{
        filled<Scalar>(Scalar{1}),
        filled<Scalar>(Scalar{2}),
        filled<Scalar>(Scalar{4}),
    };
    const auto scaled = std::array{
        filled<Scalar>(Scalar{8}),
        filled<Scalar>(Scalar{16}),
        filled<Scalar>(Scalar{32}),
    };
    const auto base_sampler = require_value(
        SamplerFor<Scalar>::create_power_weighted(std::span<const SpectrumFor<Scalar>>{base}));
    const auto scaled_sampler = require_value(
        SamplerFor<Scalar>::create_power_weighted(std::span<const SpectrumFor<Scalar>>{scaled}));
    for (auto index = std::uint32_t{}; index < base_sampler.light_count(); ++index) {
        EXPECT_EQ(require_value(base_sampler.probability(index)),
                  require_value(scaled_sampler.probability(index)));
    }

    const auto maximum = std::numeric_limits<Scalar>::max();
    const auto maximum_powers = std::array{filled<Scalar>(maximum), filled<Scalar>(maximum)};
    const auto maximum_sampler = SamplerFor<Scalar>::create_power_weighted(
        std::span<const SpectrumFor<Scalar>>{maximum_powers});
    ASSERT_TRUE(maximum_sampler.has_value()) << maximum_sampler.error().message;
    EXPECT_EQ(require_value(maximum_sampler->probability(0U)).value(), Scalar{0.5});
    EXPECT_EQ(require_value(maximum_sampler->probability(1U)).value(), Scalar{0.5});

    const auto denormal = std::numeric_limits<Scalar>::denorm_min();
    const auto denormal_powers = std::array{filled<Scalar>(denormal), filled<Scalar>(denormal)};
    const auto denormal_sampler = SamplerFor<Scalar>::create_power_weighted(
        std::span<const SpectrumFor<Scalar>>{denormal_powers});
    ASSERT_TRUE(denormal_sampler.has_value()) << denormal_sampler.error().message;
    EXPECT_EQ(require_value(denormal_sampler->probability(0U)).value(), Scalar{0.5});
    EXPECT_EQ(require_value(denormal_sampler->probability(1U)).value(), Scalar{0.5});
}

TEST(LightSamplerTest, PowerNormalizationIsScaleInvariantAndAvoidsOverflow) {
    expect_scale_and_range_robustness<TransportScalar>();
    expect_scale_and_range_robustness<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_invalid_inputs_rejected() {
    expect_error(SamplerFor<Scalar>::create_uniform(0U), core::StatusCode::invalid_argument);
    expect_error(SamplerFor<Scalar>::create_power_weighted(std::span<const SpectrumFor<Scalar>>{}),
                 core::StatusCode::invalid_argument);

    const auto black = std::array<SpectrumFor<Scalar>, 2>{};
    expect_error(SamplerFor<Scalar>::create_power_weighted(black),
                 core::StatusCode::invalid_argument);

    for (const auto invalid : {Scalar{-1}, std::numeric_limits<Scalar>::quiet_NaN(),
                               std::numeric_limits<Scalar>::infinity()}) {
        auto powers = std::array{filled<Scalar>(Scalar{1}), filled<Scalar>(Scalar{2})};
        powers[1][2] = invalid;
        expect_error(SamplerFor<Scalar>::create_power_weighted(powers),
                     core::StatusCode::invalid_argument);
    }

    const auto uniform = require_value(SamplerFor<Scalar>::create_uniform(2U));
    for (const auto invalid : {Scalar{-1}, Scalar{1}, std::numeric_limits<Scalar>::quiet_NaN(),
                               std::numeric_limits<Scalar>::infinity()}) {
        expect_error(uniform.sample(invalid), core::StatusCode::invalid_argument);
    }
    expect_error(uniform.probability(2U), core::StatusCode::invalid_argument);

    const auto tiny =
        std::numeric_limits<Scalar>::epsilon() * std::numeric_limits<Scalar>::epsilon();
    const auto tiny_middle =
        std::array{filled<Scalar>(Scalar{1}), filled<Scalar>(tiny), filled<Scalar>(Scalar{1})};
    const auto tiny_terminal = std::array{filled<Scalar>(Scalar{1}), filled<Scalar>(tiny)};
    expect_error(SamplerFor<Scalar>::create_power_weighted(tiny_middle),
                 core::StatusCode::resource_exhausted);
    expect_error(SamplerFor<Scalar>::create_power_weighted(tiny_terminal),
                 core::StatusCode::resource_exhausted);

    const auto unscalable = std::array{
        lanes<Scalar>(std::numeric_limits<Scalar>::max(), Scalar{0}, Scalar{0}, Scalar{0}),
        lanes<Scalar>(std::numeric_limits<Scalar>::denorm_min(), Scalar{0}, Scalar{0}, Scalar{0}),
    };
    expect_error(SamplerFor<Scalar>::create_power_weighted(unscalable),
                 core::StatusCode::resource_exhausted);
}

TEST(LightSamplerValidationTest, RejectsInvalidOrUnrepresentableInputsWithoutFallback) {
    expect_invalid_inputs_rejected<TransportScalar>();
    expect_invalid_inputs_rejected<ReferenceScalar>();

    if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t)) {
        expect_error(LightSampler::create_uniform(
                         static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1U),
                     core::StatusCode::resource_exhausted);
    }
    expect_error(LightSampler::create_uniform((std::size_t{1} << 24U) + 1U),
                 core::StatusCode::resource_exhausted);
}

} // namespace
} // namespace blackframe::renderer
