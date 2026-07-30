#include <Blackframe/Renderer/PathDepthLimits.hpp>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace blackframe::renderer {
namespace {

constexpr auto GenerousLimits = PathDepthLimits{
    .diffuse = 16,
    .glossy = 16,
    .specular = 16,
    .transmission = 16,
    .volume = 16,
};

[[nodiscard]] PathDepthEventResult expect_accepted(const PathDepthLimits& limits,
                                                   const PathDepthCounters& counters,
                                                   const ScatteringLobe event_lobes) {
    const auto result = evaluate_path_depth_event(limits, counters, event_lobes);
    EXPECT_TRUE(result.has_value()) << (result.has_value() ? "" : result.error().message);
    if (!result.has_value()) {
        return {};
    }
    EXPECT_TRUE(result->accepted());
    EXPECT_EQ(result->blocked_limits, ScatteringLobe::none);
    return *result;
}

TEST(PathDepthLimitsTest, ExactlyCountsEverySupportedCategory) {
    static_assert(std::is_standard_layout_v<PathDepthLimits>);
    static_assert(std::is_trivially_copyable_v<PathDepthLimits>);
    static_assert(std::is_standard_layout_v<PathDepthCounters>);
    static_assert(std::is_trivially_copyable_v<PathDepthCounters>);
    static_assert(std::is_standard_layout_v<PathDepthEventResult>);
    static_assert(std::is_trivially_copyable_v<PathDepthEventResult>);

    auto counters = PathDepthCounters{};
    EXPECT_EQ(*path_depth_total(counters), 0U);

    counters = expect_accepted(GenerousLimits, counters,
                               ScatteringLobe::diffuse | ScatteringLobe::reflection)
                   .counters;
    EXPECT_EQ(counters, (PathDepthCounters{.diffuse = 1}));
    EXPECT_EQ(*path_depth_total(counters), 1U);

    counters = expect_accepted(GenerousLimits, counters,
                               ScatteringLobe::glossy | ScatteringLobe::reflection)
                   .counters;
    EXPECT_EQ(counters, (PathDepthCounters{.diffuse = 1, .glossy = 1}));
    EXPECT_EQ(*path_depth_total(counters), 2U);

    counters = expect_accepted(GenerousLimits, counters,
                               ScatteringLobe::specular | ScatteringLobe::reflection)
                   .counters;
    EXPECT_EQ(counters, (PathDepthCounters{.diffuse = 1, .glossy = 1, .specular = 1}));
    EXPECT_EQ(*path_depth_total(counters), 3U);

    counters = expect_accepted(GenerousLimits, counters,
                               ScatteringLobe::diffuse | ScatteringLobe::transmission)
                   .counters;
    EXPECT_EQ(counters, (PathDepthCounters{
                            .diffuse = 2,
                            .glossy = 1,
                            .specular = 1,
                            .transmission = 1,
                        }));
    EXPECT_EQ(*path_depth_total(counters), 4U);

    counters = expect_accepted(GenerousLimits, counters,
                               ScatteringLobe::glossy | ScatteringLobe::transmission)
                   .counters;
    EXPECT_EQ(counters, (PathDepthCounters{
                            .diffuse = 2,
                            .glossy = 2,
                            .specular = 1,
                            .transmission = 2,
                        }));
    EXPECT_EQ(*path_depth_total(counters), 5U);

    counters = expect_accepted(GenerousLimits, counters,
                               ScatteringLobe::specular | ScatteringLobe::transmission)
                   .counters;
    EXPECT_EQ(counters, (PathDepthCounters{
                            .diffuse = 2,
                            .glossy = 2,
                            .specular = 2,
                            .transmission = 3,
                        }));
    EXPECT_EQ(*path_depth_total(counters), 6U);

    counters = expect_accepted(GenerousLimits, counters, ScatteringLobe::volume).counters;
    EXPECT_EQ(counters, (PathDepthCounters{
                            .diffuse = 2,
                            .glossy = 2,
                            .specular = 2,
                            .transmission = 3,
                            .volume = 1,
                        }));
    EXPECT_EQ(*path_depth_total(counters), 7U);
    EXPECT_TRUE(validate_path_depth_state(GenerousLimits, counters, 7).has_value());
}

TEST(PathDepthLimitsTest, AppliesZeroAndBoundaryLimitsAtomically) {
    const auto limits = PathDepthLimits{
        .diffuse = 2,
        .glossy = 0,
        .specular = 1,
        .transmission = 0,
        .volume = 0,
    };
    auto counters = PathDepthCounters{};

    for (auto accepted = 0U; accepted < 2U; ++accepted) {
        const auto result = evaluate_path_depth_event(
            limits, counters, ScatteringLobe::diffuse | ScatteringLobe::reflection);
        ASSERT_TRUE(result.has_value());
        ASSERT_TRUE(result->accepted());
        counters = result->counters;
        EXPECT_EQ(counters.diffuse, accepted + 1U);
    }

    const auto diffuse_blocked = evaluate_path_depth_event(
        limits, counters, ScatteringLobe::diffuse | ScatteringLobe::reflection);
    ASSERT_TRUE(diffuse_blocked.has_value());
    EXPECT_FALSE(diffuse_blocked->accepted());
    EXPECT_EQ(diffuse_blocked->blocked_limits, ScatteringLobe::diffuse);
    EXPECT_EQ(diffuse_blocked->counters, counters);

    const auto transmission_blocked = evaluate_path_depth_event(
        limits, counters, ScatteringLobe::specular | ScatteringLobe::transmission);
    ASSERT_TRUE(transmission_blocked.has_value());
    EXPECT_FALSE(transmission_blocked->accepted());
    EXPECT_EQ(transmission_blocked->blocked_limits, ScatteringLobe::transmission);
    EXPECT_EQ(transmission_blocked->counters, counters);

    const auto reflected = evaluate_path_depth_event(
        limits, counters, ScatteringLobe::specular | ScatteringLobe::reflection);
    ASSERT_TRUE(reflected.has_value());
    ASSERT_TRUE(reflected->accepted());
    EXPECT_EQ(reflected->counters.specular, 1U);
    EXPECT_EQ(reflected->counters.transmission, 0U);

    const auto both_blocked = evaluate_path_depth_event(
        limits, reflected->counters, ScatteringLobe::specular | ScatteringLobe::transmission);
    ASSERT_TRUE(both_blocked.has_value());
    EXPECT_FALSE(both_blocked->accepted());
    EXPECT_EQ(both_blocked->blocked_limits,
              ScatteringLobe::specular | ScatteringLobe::transmission);
    EXPECT_EQ(both_blocked->counters, reflected->counters);

    const auto volume_blocked =
        evaluate_path_depth_event(limits, reflected->counters, ScatteringLobe::volume);
    ASSERT_TRUE(volume_blocked.has_value());
    EXPECT_FALSE(volume_blocked->accepted());
    EXPECT_EQ(volume_blocked->blocked_limits, ScatteringLobe::volume);
    EXPECT_EQ(volume_blocked->counters, reflected->counters);

    struct IsolatedLimitCase final {
        PathDepthLimits limits;
        ScatteringLobe event_lobes;
        ScatteringLobe expected_blocked;
    };
    constexpr auto isolated_cases = std::array{
        IsolatedLimitCase{
            .limits = PathDepthLimits{},
            .event_lobes = ScatteringLobe::diffuse | ScatteringLobe::reflection,
            .expected_blocked = ScatteringLobe::diffuse,
        },
        IsolatedLimitCase{
            .limits = PathDepthLimits{},
            .event_lobes = ScatteringLobe::glossy | ScatteringLobe::reflection,
            .expected_blocked = ScatteringLobe::glossy,
        },
        IsolatedLimitCase{
            .limits = PathDepthLimits{},
            .event_lobes = ScatteringLobe::specular | ScatteringLobe::reflection,
            .expected_blocked = ScatteringLobe::specular,
        },
        IsolatedLimitCase{
            .limits = PathDepthLimits{.specular = 1},
            .event_lobes = ScatteringLobe::specular | ScatteringLobe::transmission,
            .expected_blocked = ScatteringLobe::transmission,
        },
        IsolatedLimitCase{
            .limits = PathDepthLimits{},
            .event_lobes = ScatteringLobe::volume,
            .expected_blocked = ScatteringLobe::volume,
        },
    };
    for (const auto& isolated : isolated_cases) {
        const auto result = evaluate_path_depth_event(isolated.limits, {}, isolated.event_lobes);
        ASSERT_TRUE(result.has_value());
        EXPECT_FALSE(result->accepted());
        EXPECT_EQ(result->blocked_limits, isolated.expected_blocked);
        EXPECT_EQ(result->counters, PathDepthCounters{});
    }
}

TEST(PathDepthLimitsTest, RejectsAmbiguousAndUnknownEventsWithoutClassificationFallback) {
    constexpr auto unknown = static_cast<ScatteringLobe>(1U << 31U);
    constexpr auto invalid_events = std::array{
        ScatteringLobe::none,
        unknown,
        ScatteringLobe::reflection,
        ScatteringLobe::diffuse,
        ScatteringLobe::diffuse | ScatteringLobe::glossy | ScatteringLobe::reflection,
        ScatteringLobe::diffuse | ScatteringLobe::reflection | ScatteringLobe::transmission,
        ScatteringLobe::volume | ScatteringLobe::diffuse,
        ScatteringLobe::volume | ScatteringLobe::transmission,
    };

    for (const auto event_lobes : invalid_events) {
        const auto result = evaluate_path_depth_event(GenerousLimits, {}, event_lobes);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
    }
}

TEST(PathDepthLimitsTest, RejectsInconsistentCountersAndTotalOverflowExactly) {
    constexpr auto maximum = std::numeric_limits<std::uint32_t>::max();
    constexpr auto maximum_limits = PathDepthLimits{
        .diffuse = maximum,
        .glossy = maximum,
        .specular = maximum,
        .transmission = maximum,
        .volume = maximum,
    };

    const auto exact_maximum = PathDepthCounters{
        .diffuse = maximum,
        .transmission = maximum,
    };
    const auto exact_total = path_depth_total(exact_maximum);
    ASSERT_TRUE(exact_total.has_value());
    EXPECT_EQ(*exact_total, maximum);
    EXPECT_TRUE(validate_path_depth_state(maximum_limits, exact_maximum, maximum).has_value());

    const auto at_limit = evaluate_path_depth_event(
        maximum_limits, exact_maximum, ScatteringLobe::diffuse | ScatteringLobe::reflection);
    ASSERT_TRUE(at_limit.has_value());
    EXPECT_FALSE(at_limit->accepted());
    EXPECT_EQ(at_limit->blocked_limits, ScatteringLobe::diffuse);
    EXPECT_EQ(at_limit->counters, exact_maximum);

    const auto overflowing_sum = PathDepthCounters{
        .diffuse = maximum,
        .glossy = 1,
    };
    const auto total = path_depth_total(overflowing_sum);
    ASSERT_FALSE(total.has_value());
    EXPECT_EQ(total.error().code, core::StatusCode::resource_exhausted);

    const auto evaluated_over_limit =
        evaluate_path_depth_event(PathDepthLimits{}, PathDepthCounters{.diffuse = 1},
                                  ScatteringLobe::diffuse | ScatteringLobe::reflection);
    ASSERT_FALSE(evaluated_over_limit.has_value());
    EXPECT_EQ(evaluated_over_limit.error().code, core::StatusCode::invalid_argument);

    const auto evaluated_overflow = evaluate_path_depth_event(
        maximum_limits, overflowing_sum, ScatteringLobe::diffuse | ScatteringLobe::reflection);
    ASSERT_FALSE(evaluated_overflow.has_value());
    EXPECT_EQ(evaluated_overflow.error().code, core::StatusCode::resource_exhausted);

    const auto overflowing_event = PathDepthCounters{
        .diffuse = maximum - 1U,
        .glossy = 1,
    };
    const auto event = evaluate_path_depth_event(
        maximum_limits, overflowing_event, ScatteringLobe::diffuse | ScatteringLobe::reflection);
    ASSERT_FALSE(event.has_value());
    EXPECT_EQ(event.error().code, core::StatusCode::resource_exhausted);

    const auto impossible_transmission = PathDepthCounters{
        .diffuse = 1,
        .transmission = 2,
    };
    const auto transmission_status =
        validate_path_depth_state(maximum_limits, impossible_transmission, 1);
    ASSERT_FALSE(transmission_status.has_value());
    EXPECT_EQ(transmission_status.error().code, core::StatusCode::invalid_argument);

    const auto smaller_limits = PathDepthLimits{.diffuse = 0};
    const auto over_limit =
        validate_path_depth_state(smaller_limits, PathDepthCounters{.diffuse = 1}, 1);
    ASSERT_FALSE(over_limit.has_value());
    EXPECT_EQ(over_limit.error().code, core::StatusCode::invalid_argument);

    const auto wrong_total =
        validate_path_depth_state(GenerousLimits, PathDepthCounters{.diffuse = 1}, 0);
    ASSERT_FALSE(wrong_total.has_value());
    EXPECT_EQ(wrong_total.error().code, core::StatusCode::invalid_argument);

    struct OverLimitCase final {
        PathDepthLimits limits;
        PathDepthCounters counters;
        std::uint32_t total;
    };
    constexpr auto over_limit_cases = std::array{
        OverLimitCase{
            .limits = PathDepthLimits{},
            .counters = PathDepthCounters{.diffuse = 1},
            .total = 1,
        },
        OverLimitCase{
            .limits = PathDepthLimits{},
            .counters = PathDepthCounters{.glossy = 1},
            .total = 1,
        },
        OverLimitCase{
            .limits = PathDepthLimits{},
            .counters = PathDepthCounters{.specular = 1},
            .total = 1,
        },
        OverLimitCase{
            .limits = PathDepthLimits{.specular = 1},
            .counters = PathDepthCounters{.specular = 1, .transmission = 1},
            .total = 1,
        },
        OverLimitCase{
            .limits = PathDepthLimits{},
            .counters = PathDepthCounters{.volume = 1},
            .total = 1,
        },
    };
    for (const auto& over_limit_case : over_limit_cases) {
        const auto status = validate_path_depth_state(
            over_limit_case.limits, over_limit_case.counters, over_limit_case.total);
        ASSERT_FALSE(status.has_value());
        EXPECT_EQ(status.error().code, core::StatusCode::invalid_argument);
    }
}

} // namespace
} // namespace blackframe::renderer
