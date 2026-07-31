#include <Blackframe/Renderer/LightSampler.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace blackframe::renderer {
namespace {

template <SpectrumScalar Scalar> using SamplerFor = LightSamplerT<Scalar>;
template <SpectrumScalar Scalar> using InputFor = LightTreeInputT<Scalar>;
template <SpectrumScalar Scalar> using SpectrumFor = LightSpectrumT<Scalar>;

template <typename Value> [[nodiscard]] Value require_tree_value(core::Result<Value> result) {
    if (!result) {
        ADD_FAILURE() << result.error().message;
        std::abort();
    }
    return std::move(*result);
}

template <typename Result>
void expect_tree_error(const Result& result, const core::StatusCode expected_code) {
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, expected_code);
    EXPECT_FALSE(result.error().message.empty());
}

template <SpectrumScalar Scalar> [[nodiscard]] SpectrumFor<Scalar> tree_power(const Scalar value) {
    return SpectrumFor<Scalar>{.values = {value, value, value, value}};
}

template <SpectrumScalar Scalar>
[[nodiscard]] Bounds3T<Scalar> tree_bounds(const Scalar minimum_x, const Scalar minimum_y,
                                           const Scalar minimum_z, const Scalar maximum_x,
                                           const Scalar maximum_y, const Scalar maximum_z) {
    return require_tree_value(Bounds3T<Scalar>::from_minimum_maximum(
        Point3T<Scalar>{.x = minimum_x, .y = minimum_y, .z = minimum_z},
        Point3T<Scalar>{.x = maximum_x, .y = maximum_y, .z = maximum_z}));
}

template <SpectrumScalar Scalar>
[[nodiscard]] Bounds3T<Scalar> point_tree_bounds(const Scalar x, const Scalar y, const Scalar z) {
    return tree_bounds(x, y, z, x, y, z);
}

template <SpectrumScalar Scalar>
[[nodiscard]] LightSampleContextT<Scalar> tree_context(const Scalar x, const Scalar y,
                                                       const Scalar z) {
    return require_tree_value(
        LightSampleContextT<Scalar>::create(Point3T<Scalar>{.x = x, .y = y, .z = z}, Scalar{0}));
}

template <SpectrumScalar Scalar> void expect_tree_contract_and_stable_slots() {
    auto inputs = std::array{
        InputFor<Scalar>{
            .bounds = point_tree_bounds<Scalar>(Scalar{-4}, Scalar{0}, Scalar{0}),
            .spectral_power = tree_power<Scalar>(Scalar{1}),
        },
        InputFor<Scalar>{
            .bounds = point_tree_bounds<Scalar>(Scalar{0}, Scalar{0}, Scalar{0}),
            .spectral_power = SpectrumFor<Scalar>{},
        },
        InputFor<Scalar>{
            .bounds = point_tree_bounds<Scalar>(Scalar{4}, Scalar{0}, Scalar{0}),
            .spectral_power = tree_power<Scalar>(Scalar{2}),
        },
        InputFor<Scalar>{
            .bounds = Bounds3T<Scalar>::unbounded(),
            .spectral_power = tree_power<Scalar>(Scalar{4}),
        },
        InputFor<Scalar>{
            .bounds = point_tree_bounds<Scalar>(Scalar{8}, Scalar{0}, Scalar{0}),
            .spectral_power = tree_power<Scalar>(Scalar{1}),
        },
    };
    const auto context = tree_context<Scalar>(Scalar{-4}, Scalar{0}, Scalar{0});
    const auto sampler = require_tree_value(
        SamplerFor<Scalar>::create_spatial_tree(std::span<const InputFor<Scalar>>{inputs}));
    const auto probability_before_mutation = require_tree_value(sampler.probability(context, 0U));
    inputs[0].spectral_power = tree_power<Scalar>(Scalar{1000});
    EXPECT_EQ(require_tree_value(sampler.probability(context, 0U)), probability_before_mutation);

    EXPECT_EQ(sampler.strategy(), LightSamplingStrategy::spatial_tree);
    EXPECT_EQ(sampler.light_count(), 5U);
    EXPECT_EQ(sampler.tree_node_count(), 7U);
    EXPECT_EQ(sampler.finite_light_count(), 4U);
    EXPECT_EQ(sampler.unbounded_light_count(), 1U);
    EXPECT_GE(sampler.maximum_tree_depth(), 3U);
    expect_tree_error(sampler.sample(Scalar{0.5}), core::StatusCode::unavailable);
    expect_tree_error(sampler.probability(0U), core::StatusCode::unavailable);

    EXPECT_EQ(require_tree_value(sampler.probability(context, 1U)).value(), Scalar{0});
    auto probability_sum = ReferenceScalar{0};
    for (auto index = std::uint32_t{0}; index < sampler.light_count(); ++index) {
        const auto probability = require_tree_value(sampler.probability(context, index));
        EXPECT_EQ(probability.measure(), ProbabilityMeasure::discrete);
        EXPECT_TRUE(std::isfinite(probability.value()));
        EXPECT_GE(probability.value(), Scalar{0});
        probability_sum += static_cast<ReferenceScalar>(probability.value());
    }
    EXPECT_NEAR(probability_sum, 1.0,
                64.0 * static_cast<ReferenceScalar>(std::numeric_limits<Scalar>::epsilon()));

    for (auto sample_index = std::uint32_t{0}; sample_index < 4096U; ++sample_index) {
        const auto canonical = (static_cast<Scalar>(sample_index) + Scalar{0.5}) / Scalar{4096};
        const auto selection = require_tree_value(sampler.sample(context, canonical));
        EXPECT_NE(selection.light_index(), 1U);
        EXPECT_EQ(selection.probability(),
                  require_tree_value(sampler.probability(context, selection.light_index())));
    }
}

TEST(LightTreeContractTest, IsImmutableAndPreservesStableRegistrySlots) {
    expect_tree_contract_and_stable_slots<TransportScalar>();
    expect_tree_contract_and_stable_slots<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_spatial_importance_and_scale_invariance() {
    const auto inputs = std::array{
        InputFor<Scalar>{
            .bounds = point_tree_bounds<Scalar>(Scalar{-10}, Scalar{0}, Scalar{0}),
            .spectral_power = tree_power<Scalar>(Scalar{1}),
        },
        InputFor<Scalar>{
            .bounds = point_tree_bounds<Scalar>(Scalar{10}, Scalar{0}, Scalar{0}),
            .spectral_power = tree_power<Scalar>(Scalar{1}),
        },
    };
    const auto sampler = require_tree_value(SamplerFor<Scalar>::create_spatial_tree(inputs));
    const auto left_context = tree_context<Scalar>(Scalar{-10}, Scalar{0}, Scalar{0});
    const auto right_context = tree_context<Scalar>(Scalar{10}, Scalar{0}, Scalar{0});
    const auto left_from_left = require_tree_value(sampler.probability(left_context, 0U)).value();
    const auto right_from_left = require_tree_value(sampler.probability(left_context, 1U)).value();
    const auto left_from_right = require_tree_value(sampler.probability(right_context, 0U)).value();
    const auto right_from_right =
        require_tree_value(sampler.probability(right_context, 1U)).value();
    EXPECT_NEAR(left_from_left, Scalar{2} / Scalar{3},
                Scalar{8} * std::numeric_limits<Scalar>::epsilon());
    EXPECT_NEAR(left_from_left, right_from_right,
                Scalar{2} * std::numeric_limits<Scalar>::epsilon());
    EXPECT_NEAR(right_from_left, left_from_right,
                Scalar{2} * std::numeric_limits<Scalar>::epsilon());
    EXPECT_GT(left_from_left, right_from_left);

    const auto scaled_inputs = std::array{
        InputFor<Scalar>{
            .bounds = point_tree_bounds<Scalar>(Scalar{-1000}, Scalar{20}, Scalar{30}),
            .spectral_power = tree_power<Scalar>(Scalar{1}),
        },
        InputFor<Scalar>{
            .bounds = point_tree_bounds<Scalar>(Scalar{1000}, Scalar{20}, Scalar{30}),
            .spectral_power = tree_power<Scalar>(Scalar{1}),
        },
    };
    const auto scaled = require_tree_value(SamplerFor<Scalar>::create_spatial_tree(scaled_inputs));
    const auto scaled_context = tree_context<Scalar>(Scalar{-1000}, Scalar{20}, Scalar{30});
    EXPECT_EQ(require_tree_value(scaled.probability(scaled_context, 0U)).value(), left_from_left);
}

TEST(LightTreeTest, SpatialImportanceIsSymmetricAndScaleInvariant) {
    expect_spatial_importance_and_scale_invariance<TransportScalar>();
    expect_spatial_importance_and_scale_invariance<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_degenerate_tree_matches_flat_power_sampling() {
    const auto powers = std::array{
        tree_power<Scalar>(Scalar{1}), SpectrumFor<Scalar>{},         tree_power<Scalar>(Scalar{1}),
        tree_power<Scalar>(Scalar{2}), tree_power<Scalar>(Scalar{4}),
    };
    auto inputs = std::vector<InputFor<Scalar>>{};
    inputs.reserve(powers.size());
    for (const auto power : powers) {
        inputs.push_back(InputFor<Scalar>{
            .bounds = point_tree_bounds<Scalar>(Scalar{3}, Scalar{-2}, Scalar{7}),
            .spectral_power = power,
        });
    }
    const auto flat = require_tree_value(SamplerFor<Scalar>::create_power_weighted(powers));
    const auto tree = require_tree_value(SamplerFor<Scalar>::create_spatial_tree(inputs));
    const auto context = tree_context<Scalar>(Scalar{100}, Scalar{-50}, Scalar{25});
    for (auto index = std::uint32_t{0}; index < powers.size(); ++index) {
        EXPECT_NEAR(require_tree_value(tree.probability(context, index)).value(),
                    require_tree_value(flat.probability(index)).value(),
                    Scalar{4} * std::numeric_limits<Scalar>::epsilon());
    }
    for (auto sample_index = std::uint32_t{0}; sample_index < 4096U; ++sample_index) {
        const auto canonical = (static_cast<Scalar>(sample_index) + Scalar{0.5}) / Scalar{4096};
        const auto flat_selection = require_tree_value(flat.sample(canonical));
        const auto tree_selection = require_tree_value(tree.sample(context, canonical));
        EXPECT_EQ(tree_selection.light_index(), flat_selection.light_index());
        EXPECT_NEAR(tree_selection.probability().value(), flat_selection.probability().value(),
                    Scalar{4} * std::numeric_limits<Scalar>::epsilon());
    }
}

TEST(LightTreeDistributionTest, DegenerateSpatialTreeMatchesPowerSamplerWithinTolerance) {
    expect_degenerate_tree_matches_flat_power_sampling<TransportScalar>();
    expect_degenerate_tree_matches_flat_power_sampling<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_extreme_spatial_ratios_remain_representable() {
    const auto equal_powers = std::array{
        InputFor<Scalar>{
            .bounds = point_tree_bounds<Scalar>(Scalar{0}, Scalar{0}, Scalar{0}),
            .spectral_power = tree_power<Scalar>(Scalar{1}),
        },
        InputFor<Scalar>{
            .bounds = point_tree_bounds<Scalar>(Scalar{1}, Scalar{0}, Scalar{0}),
            .spectral_power = tree_power<Scalar>(Scalar{1}),
        },
    };
    const auto far_tree = require_tree_value(SamplerFor<Scalar>::create_spatial_tree(equal_powers));
    const auto far_context =
        tree_context<Scalar>(std::numeric_limits<Scalar>::max(), Scalar{0}, Scalar{0});
    EXPECT_NEAR(require_tree_value(far_tree.probability(far_context, 0U)).value(), Scalar{0.5},
                Scalar{8} * std::numeric_limits<Scalar>::epsilon());
    EXPECT_NEAR(require_tree_value(far_tree.probability(far_context, 1U)).value(), Scalar{0.5},
                Scalar{8} * std::numeric_limits<Scalar>::epsilon());

    const auto maximum = std::numeric_limits<Scalar>::max();
    const auto wide_inputs = std::array{
        InputFor<Scalar>{
            .bounds = point_tree_bounds<Scalar>(-maximum, Scalar{0}, Scalar{0}),
            .spectral_power = tree_power<Scalar>(Scalar{1}),
        },
        InputFor<Scalar>{
            .bounds = point_tree_bounds<Scalar>(maximum, Scalar{0}, Scalar{0}),
            .spectral_power = tree_power<Scalar>(Scalar{1}),
        },
    };
    const auto wide_tree = require_tree_value(SamplerFor<Scalar>::create_spatial_tree(wide_inputs));
    const auto origin = tree_context<Scalar>(Scalar{0}, Scalar{0}, Scalar{0});
    EXPECT_EQ(require_tree_value(wide_tree.probability(origin, 0U)).value(), Scalar{0.5});
    EXPECT_EQ(require_tree_value(wide_tree.probability(origin, 1U)).value(), Scalar{0.5});
}

TEST(LightTreeNumericTest, ExtremeFiniteDistancesPreserveRelativeProbability) {
    expect_extreme_spatial_ratios_remain_representable<TransportScalar>();
    expect_extreme_spatial_ratios_remain_representable<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_upper_edge_remap_without_clamping() {
    constexpr auto small_probability = Scalar{0.00003418326};
    const auto remaining_half = (Scalar{1} - small_probability) / Scalar{2};
    const auto inputs = std::array{
        InputFor<Scalar>{
            .bounds = Bounds3T<Scalar>::unbounded(),
            .spectral_power = tree_power<Scalar>(small_probability),
        },
        InputFor<Scalar>{
            .bounds = Bounds3T<Scalar>::unbounded(),
            .spectral_power = tree_power<Scalar>(remaining_half),
        },
        InputFor<Scalar>{
            .bounds = Bounds3T<Scalar>::unbounded(),
            .spectral_power = tree_power<Scalar>(remaining_half),
        },
    };
    const auto tree = require_tree_value(SamplerFor<Scalar>::create_spatial_tree(inputs));
    const auto context = tree_context<Scalar>(Scalar{0}, Scalar{0}, Scalar{0});
    const auto selection =
        require_tree_value(tree.sample(context, std::nextafter(Scalar{1}, Scalar{0})));
    EXPECT_EQ(selection.light_index(), 2U);
    EXPECT_EQ(selection.probability(),
              require_tree_value(tree.probability(context, selection.light_index())));
}

TEST(LightTreeNumericTest, UpperEdgeRemapStaysInsideTheCanonicalDomain) {
    expect_upper_edge_remap_without_clamping<TransportScalar>();
    expect_upper_edge_remap_without_clamping<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_large_normalized_distribution() {
    constexpr auto light_count = std::size_t{1024};
    constexpr auto sample_count = std::uint32_t{65'536};
    auto inputs = std::vector<InputFor<Scalar>>{};
    inputs.reserve(light_count);
    for (auto index = std::size_t{0}; index < light_count; ++index) {
        inputs.push_back(InputFor<Scalar>{
            .bounds = Bounds3T<Scalar>::unbounded(),
            .spectral_power = tree_power<Scalar>(Scalar{1}),
        });
    }
    const auto sampler = require_tree_value(SamplerFor<Scalar>::create_spatial_tree(inputs));
    const auto context = tree_context<Scalar>(Scalar{17}, Scalar{-3}, Scalar{9});
    EXPECT_EQ(sampler.tree_node_count(), 2047U);
    EXPECT_EQ(sampler.maximum_tree_depth(), 11U);

    auto counts = std::array<std::uint32_t, light_count>{};
    auto probability_sum = Scalar{0};
    for (auto index = std::uint32_t{0}; index < light_count; ++index) {
        const auto probability = require_tree_value(sampler.probability(context, index));
        EXPECT_EQ(probability.value(), Scalar{1} / Scalar{1024});
        probability_sum += probability.value();
    }
    EXPECT_EQ(probability_sum, Scalar{1});
    for (auto sample_index = std::uint32_t{0}; sample_index < sample_count; ++sample_index) {
        const auto canonical =
            (static_cast<Scalar>(sample_index) + Scalar{0.5}) / static_cast<Scalar>(sample_count);
        const auto selection = require_tree_value(sampler.sample(context, canonical));
        ++counts[selection.light_index()];
        EXPECT_EQ(selection.probability().value(), Scalar{1} / Scalar{1024});
    }
    for (const auto count : counts) {
        EXPECT_EQ(count, 64U);
    }
}

TEST(LightTreeDistributionTest, LargeBalancedDistributionIsExactlyNormalized) {
    expect_large_normalized_distribution<TransportScalar>();
    expect_large_normalized_distribution<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_non_dyadic_normalization() {
    constexpr auto light_count = std::size_t{257};
    constexpr auto sample_count = std::uint32_t{65'536};
    auto inputs = std::vector<InputFor<Scalar>>{};
    inputs.reserve(light_count);
    for (auto index = std::size_t{0}; index < light_count; ++index) {
        const auto x = static_cast<Scalar>(index % 17U) - Scalar{8};
        const auto y = static_cast<Scalar>(index / 17U) - Scalar{7};
        const auto value = index % 19U == 0U ? Scalar{0} : static_cast<Scalar>(index % 7U + 1U);
        inputs.push_back(InputFor<Scalar>{
            .bounds = tree_bounds<Scalar>(x - Scalar{0.1}, y - Scalar{0.1}, Scalar{-0.1},
                                          x + Scalar{0.1}, y + Scalar{0.1}, Scalar{0.1}),
            .spectral_power = tree_power<Scalar>(value),
        });
    }
    const auto sampler = require_tree_value(SamplerFor<Scalar>::create_spatial_tree(inputs));
    const auto contexts = std::array{tree_context<Scalar>(Scalar{-8}, Scalar{-7}, Scalar{0}),
                                     tree_context<Scalar>(Scalar{0}, Scalar{0}, Scalar{3}),
                                     tree_context<Scalar>(Scalar{40}, Scalar{-25}, Scalar{11})};
    for (auto context_index = std::size_t{0}; context_index < contexts.size(); ++context_index) {
        const auto& context = contexts[context_index];
        auto probabilities = std::array<Scalar, light_count>{};
        auto counts = std::array<std::uint32_t, light_count>{};
        auto probability_sum = ReferenceScalar{0};
        for (auto index = std::uint32_t{0}; index < light_count; ++index) {
            const auto probability = require_tree_value(sampler.probability(context, index));
            EXPECT_TRUE(std::isfinite(probability.value()));
            EXPECT_EQ(probability.value() == Scalar{0}, index % 19U == 0U);
            probabilities[index] = probability.value();
            probability_sum += static_cast<ReferenceScalar>(probability.value());
        }
        EXPECT_NEAR(probability_sum, 1.0,
                    1024.0 * static_cast<ReferenceScalar>(std::numeric_limits<Scalar>::epsilon()));
        if (context_index != 0U) {
            continue;
        }
        for (auto sample_index = std::uint32_t{0}; sample_index < sample_count; ++sample_index) {
            const auto canonical = (static_cast<Scalar>(sample_index) + Scalar{0.5}) /
                                   static_cast<Scalar>(sample_count);
            const auto selection = require_tree_value(sampler.sample(context, canonical));
            ++counts[selection.light_index()];
            EXPECT_EQ(selection.probability().value(), probabilities[selection.light_index()]);
        }
        for (auto index = std::size_t{0}; index < light_count; ++index) {
            const auto expected = static_cast<ReferenceScalar>(probabilities[index]) *
                                  static_cast<ReferenceScalar>(sample_count);
            EXPECT_LE(std::abs(static_cast<ReferenceScalar>(counts[index]) - expected), 4.0);
        }
    }
}

TEST(LightTreeDistributionTest, NonDyadicSpatialDistributionsRemainNormalized) {
    expect_non_dyadic_normalization<TransportScalar>();
    expect_non_dyadic_normalization<ReferenceScalar>();
}

template <SpectrumScalar Scalar> void expect_tree_validation() {
    expect_tree_error(SamplerFor<Scalar>::create_spatial_tree(std::span<const InputFor<Scalar>>{}),
                      core::StatusCode::invalid_argument);

    const auto empty_input = std::array{InputFor<Scalar>{
        .bounds = Bounds3T<Scalar>::empty(),
        .spectral_power = tree_power<Scalar>(Scalar{1}),
    }};
    expect_tree_error(SamplerFor<Scalar>::create_spatial_tree(empty_input),
                      core::StatusCode::invalid_argument);

    const auto partial_bounds = require_tree_value(Bounds3T<Scalar>::from_minimum_maximum(
        Point3T<Scalar>{
            .x = -std::numeric_limits<Scalar>::infinity(), .y = Scalar{-1}, .z = Scalar{-1}},
        Point3T<Scalar>{
            .x = std::numeric_limits<Scalar>::infinity(), .y = Scalar{1}, .z = Scalar{1}}));
    const auto partial_input = std::array{InputFor<Scalar>{
        .bounds = partial_bounds,
        .spectral_power = tree_power<Scalar>(Scalar{1}),
    }};
    expect_tree_error(SamplerFor<Scalar>::create_spatial_tree(partial_input),
                      core::StatusCode::invalid_argument);

    const auto black_input = std::array{InputFor<Scalar>{
        .bounds = point_tree_bounds<Scalar>(Scalar{0}, Scalar{0}, Scalar{0}),
        .spectral_power = SpectrumFor<Scalar>{},
    }};
    expect_tree_error(SamplerFor<Scalar>::create_spatial_tree(black_input),
                      core::StatusCode::invalid_argument);

    auto negative_input = black_input;
    negative_input[0].spectral_power[2] = Scalar{-1};
    expect_tree_error(SamplerFor<Scalar>::create_spatial_tree(negative_input),
                      core::StatusCode::invalid_argument);

    const auto valid_input = std::array{
        InputFor<Scalar>{
            .bounds = point_tree_bounds<Scalar>(Scalar{-1}, Scalar{0}, Scalar{0}),
            .spectral_power = tree_power<Scalar>(Scalar{1}),
        },
        InputFor<Scalar>{
            .bounds = point_tree_bounds<Scalar>(Scalar{1}, Scalar{0}, Scalar{0}),
            .spectral_power = tree_power<Scalar>(Scalar{1}),
        },
    };
    auto sampler = require_tree_value(SamplerFor<Scalar>::create_spatial_tree(valid_input));
    const auto context = tree_context<Scalar>(Scalar{0}, Scalar{0}, Scalar{0});
    for (const auto invalid : {Scalar{-1}, Scalar{1}, std::numeric_limits<Scalar>::quiet_NaN(),
                               std::numeric_limits<Scalar>::infinity()}) {
        expect_tree_error(sampler.sample(context, invalid), core::StatusCode::invalid_argument);
    }
    expect_tree_error(sampler.probability(context, 2U), core::StatusCode::invalid_argument);

    auto moved = std::move(sampler);
    EXPECT_EQ(moved.light_count(), 2U);
    EXPECT_EQ(sampler.light_count(), 0U);
    expect_tree_error(sampler.sample(context, Scalar{0.5}), core::StatusCode::internal_error);
    expect_tree_error(sampler.probability(context, 0U), core::StatusCode::invalid_argument);
}

TEST(LightTreeValidationTest, RejectsInvalidInputsAndUnavailableQueriesWithoutFallback) {
    expect_tree_validation<TransportScalar>();
    expect_tree_validation<ReferenceScalar>();
}

} // namespace
} // namespace blackframe::renderer
