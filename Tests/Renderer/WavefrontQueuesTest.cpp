#include <Blackframe/Renderer/WavefrontQueues.hpp>
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

namespace blackframe::renderer {
namespace {

template <typename First, typename... Rest>
inline constexpr bool DistinctFromRest = (!std::same_as<First, Rest> && ...);

template <typename... Types> struct AllTypesDistinct;

template <> struct AllTypesDistinct<> : std::true_type {};

template <typename First, typename... Rest>
struct AllTypesDistinct<First, Rest...>
    : std::bool_constant<DistinctFromRest<First, Rest...> && AllTypesDistinct<Rest...>::value> {};

template <typename Queue>
concept HasRvalueEntries = requires(Queue&& queue) { std::move(queue).entries(); };

template <typename Operation> void for_each_queue_type(Operation operation) {
    operation.template operator()<CameraQueue>();
    operation.template operator()<RayQueue>();
    operation.template operator()<HitQueue>();
    operation.template operator()<MissQueue>();
    operation.template operator()<ShadeQueue>();
    operation.template operator()<ShadowQueue>();
    operation.template operator()<ContinuationQueue>();
}

TEST(WavefrontQueuesTest, DeclaresSevenStronglyDistinctBoundedStages) {
    static_assert(AllTypesDistinct<CameraQueue, RayQueue, HitQueue, MissQueue, ShadeQueue,
                                   ShadowQueue, ContinuationQueue>::value);
    static_assert(CameraQueue::kind() == WavefrontQueueKind::camera);
    static_assert(RayQueue::kind() == WavefrontQueueKind::ray);
    static_assert(HitQueue::kind() == WavefrontQueueKind::hit);
    static_assert(MissQueue::kind() == WavefrontQueueKind::miss);
    static_assert(ShadeQueue::kind() == WavefrontQueueKind::shade);
    static_assert(ShadowQueue::kind() == WavefrontQueueKind::shadow);
    static_assert(ContinuationQueue::kind() == WavefrontQueueKind::continuation);
    static_assert(!HasRvalueEntries<CameraQueue>);
    static_assert(!HasRvalueEntries<const CameraQueue>);
    for_each_queue_type([]<typename Queue> {
        static_assert(!std::is_default_constructible_v<Queue>);
        static_assert(!std::is_copy_constructible_v<Queue>);
        static_assert(std::is_nothrow_move_constructible_v<Queue>);
        static_assert(!HasRvalueEntries<Queue>);
        static_assert(!HasRvalueEntries<const Queue>);
        static_assert(noexcept(std::declval<Queue&>().push(WavefrontPathSlot{})));
        static_assert(
            noexcept(std::declval<Queue&>().push_batch(std::span<const WavefrontPathSlot>{})));
        static_assert(noexcept(std::declval<Queue&>().clear()));
    });

    EXPECT_EQ(wavefront_queue_kind_name(WavefrontQueueKind::camera), "camera");
    EXPECT_EQ(wavefront_queue_kind_name(WavefrontQueueKind::ray), "ray");
    EXPECT_EQ(wavefront_queue_kind_name(WavefrontQueueKind::hit), "hit");
    EXPECT_EQ(wavefront_queue_kind_name(WavefrontQueueKind::miss), "miss");
    EXPECT_EQ(wavefront_queue_kind_name(WavefrontQueueKind::shade), "shade");
    EXPECT_EQ(wavefront_queue_kind_name(WavefrontQueueKind::shadow), "shadow");
    EXPECT_EQ(wavefront_queue_kind_name(WavefrontQueueKind::continuation), "continuation");
    auto known_kind_count = std::uint32_t{};
    for (auto value = std::uint32_t{}; value <= std::numeric_limits<std::uint8_t>::max(); ++value) {
        const auto kind = static_cast<WavefrontQueueKind>(value);
        const auto expected_known =
            value <= static_cast<std::uint32_t>(WavefrontQueueKind::continuation);
        EXPECT_EQ(is_known_wavefront_queue_kind(kind), expected_known);
        if (expected_known) {
            ++known_kind_count;
        } else {
            EXPECT_EQ(wavefront_queue_kind_name(kind), "unknown");
        }
    }
    EXPECT_EQ(known_kind_count, 7U);
}

TEST(WavefrontQueuesTest, PreservesExactCapacityInsertionOrderDuplicatesAndSlotBits) {
    for_each_queue_type([]<typename Queue> {
        auto created = Queue::create(4U);
        ASSERT_TRUE(created.has_value());
        auto queue = std::move(*created);
        EXPECT_EQ(queue.capacity(), 4U);
        EXPECT_EQ(queue.size(), 0U);
        EXPECT_EQ(queue.remaining_capacity(), 4U);
        EXPECT_TRUE(queue.empty());
        EXPECT_FALSE(queue.full());
        const auto* const storage = queue.entries().data();

        constexpr auto slots = std::array{
            WavefrontPathSlot{.value = 0U},
            WavefrontPathSlot{.value = std::numeric_limits<std::uint32_t>::max()},
            WavefrontPathSlot{.value = 17U},
            WavefrontPathSlot{.value = 17U},
        };
        for (const auto slot : slots) {
            EXPECT_EQ(queue.push(slot), WavefrontQueuePushStatus::pushed);
        }

        EXPECT_EQ(queue.capacity(), slots.size());
        EXPECT_EQ(queue.size(), slots.size());
        EXPECT_EQ(queue.remaining_capacity(), 0U);
        EXPECT_FALSE(queue.empty());
        EXPECT_TRUE(queue.full());
        EXPECT_EQ(queue.entries().data(), storage);
        EXPECT_TRUE(std::ranges::equal(queue.entries(), slots));
    });
}

TEST(WavefrontQueuesTest, RejectsOverflowWithoutGrowthEvictionOverwriteOrMutation) {
    for_each_queue_type([]<typename Queue> {
        auto created = Queue::create(2U);
        ASSERT_TRUE(created.has_value());
        auto queue = std::move(*created);
        ASSERT_EQ(queue.push(WavefrontPathSlot{.value = 7U}), WavefrontQueuePushStatus::pushed);
        ASSERT_EQ(queue.push(WavefrontPathSlot{.value = 9U}), WavefrontQueuePushStatus::pushed);
        const auto expected =
            std::array{WavefrontPathSlot{.value = 7U}, WavefrontPathSlot{.value = 9U}};
        const auto* const storage = queue.entries().data();

        EXPECT_EQ(queue.push(WavefrontPathSlot{.value = 11U}),
                  WavefrontQueuePushStatus::capacity_exhausted);
        EXPECT_EQ(queue.push(WavefrontPathSlot{.value = 13U}),
                  WavefrontQueuePushStatus::capacity_exhausted);
        EXPECT_EQ(queue.capacity(), 2U);
        EXPECT_EQ(queue.size(), 2U);
        EXPECT_EQ(queue.entries().data(), storage);
        EXPECT_TRUE(std::ranges::equal(queue.entries(), expected));
    });
}

TEST(WavefrontQueuesTest, RejectsBatchOverflowAtomicallyBeforeWritingAnyEntry) {
    for_each_queue_type([]<typename Queue> {
        auto created = Queue::create(4U);
        ASSERT_TRUE(created.has_value());
        auto queue = std::move(*created);
        ASSERT_EQ(queue.push(WavefrontPathSlot{.value = 1U}), WavefrontQueuePushStatus::pushed);
        const auto overflow = std::array{
            WavefrontPathSlot{.value = 2U},
            WavefrontPathSlot{.value = 3U},
            WavefrontPathSlot{.value = 4U},
            WavefrontPathSlot{.value = 5U},
        };
        const auto* const storage = queue.entries().data();

        EXPECT_EQ(queue.push_batch(overflow), WavefrontQueuePushStatus::capacity_exhausted);
        EXPECT_EQ(queue.push_batch(overflow), WavefrontQueuePushStatus::capacity_exhausted);
        EXPECT_EQ(queue.capacity(), 4U);
        EXPECT_EQ(queue.size(), 1U);
        EXPECT_EQ(queue.remaining_capacity(), 3U);
        EXPECT_FALSE(queue.empty());
        EXPECT_FALSE(queue.full());
        EXPECT_EQ(queue.entries().data(), storage);
        ASSERT_EQ(queue.entries().size(), 1U);
        EXPECT_EQ(queue.entries().front(), WavefrontPathSlot{.value = 1U});

        const auto fitting = std::span<const WavefrontPathSlot>{overflow}.first(3U);
        EXPECT_EQ(queue.push_batch(fitting), WavefrontQueuePushStatus::pushed);
        const auto expected = std::array{
            WavefrontPathSlot{.value = 1U},
            WavefrontPathSlot{.value = 2U},
            WavefrontPathSlot{.value = 3U},
            WavefrontPathSlot{.value = 4U},
        };
        EXPECT_TRUE(std::ranges::equal(queue.entries(), expected));
        EXPECT_EQ(queue.push_batch(std::span<const WavefrontPathSlot>{}),
                  WavefrontQueuePushStatus::pushed);
    });
}

TEST(WavefrontQueuesTest, CopiesAnAliasedActivePrefixIntoReservedStorage) {
    for_each_queue_type([]<typename Queue> {
        auto created = Queue::create(4U);
        ASSERT_TRUE(created.has_value());
        auto queue = std::move(*created);
        const auto initial =
            std::array{WavefrontPathSlot{.value = 6U}, WavefrontPathSlot{.value = 12U}};
        ASSERT_EQ(queue.push_batch(initial), WavefrontQueuePushStatus::pushed);
        const auto* const storage = queue.entries().data();

        EXPECT_EQ(queue.push_batch(queue.entries()), WavefrontQueuePushStatus::pushed);
        const auto expected = std::array{
            WavefrontPathSlot{.value = 6U},
            WavefrontPathSlot{.value = 12U},
            WavefrontPathSlot{.value = 6U},
            WavefrontPathSlot{.value = 12U},
        };
        EXPECT_EQ(queue.entries().data(), storage);
        EXPECT_TRUE(std::ranges::equal(queue.entries(), expected));
    });
}

TEST(WavefrontQueuesTest, RepresentsZeroCapacityAndReportsEveryNonEmptyPush) {
    for_each_queue_type([]<typename Queue> {
        auto created = Queue::create(0U);
        ASSERT_TRUE(created.has_value());
        auto queue = std::move(*created);
        EXPECT_EQ(queue.capacity(), 0U);
        EXPECT_EQ(queue.size(), 0U);
        EXPECT_EQ(queue.remaining_capacity(), 0U);
        EXPECT_TRUE(queue.empty());
        EXPECT_TRUE(queue.full());
        EXPECT_TRUE(queue.entries().empty());
        EXPECT_EQ(queue.push(WavefrontPathSlot{.value = 0U}),
                  WavefrontQueuePushStatus::capacity_exhausted);
        const auto one = std::array{WavefrontPathSlot{.value = 0U}};
        EXPECT_EQ(queue.push_batch(one), WavefrontQueuePushStatus::capacity_exhausted);
        EXPECT_EQ(queue.push_batch(std::span<const WavefrontPathSlot>{}),
                  WavefrontQueuePushStatus::pushed);
    });
}

TEST(WavefrontQueuesTest, ClearsAndReusesPreallocatedStorageAfterOverflow) {
    for_each_queue_type([]<typename Queue> {
        auto created = Queue::create(2U);
        ASSERT_TRUE(created.has_value());
        auto queue = std::move(*created);
        ASSERT_EQ(queue.push(WavefrontPathSlot{.value = 1U}), WavefrontQueuePushStatus::pushed);
        ASSERT_EQ(queue.push(WavefrontPathSlot{.value = 2U}), WavefrontQueuePushStatus::pushed);
        ASSERT_EQ(queue.push(WavefrontPathSlot{.value = 3U}),
                  WavefrontQueuePushStatus::capacity_exhausted);
        const auto* const storage = queue.entries().data();

        queue.clear();
        EXPECT_EQ(queue.capacity(), 2U);
        EXPECT_EQ(queue.size(), 0U);
        EXPECT_EQ(queue.entries().data(), storage);
        const auto replacement =
            std::array{WavefrontPathSlot{.value = 8U}, WavefrontPathSlot{.value = 5U}};
        EXPECT_EQ(queue.push_batch(replacement), WavefrontQueuePushStatus::pushed);
        EXPECT_EQ(queue.entries().data(), storage);
        EXPECT_TRUE(std::ranges::equal(queue.entries(), replacement));
    });
}

TEST(WavefrontQueuesTest, RejectsUnrepresentableCapacityBeforeAllocation) {
    for_each_queue_type([]<typename Queue> {
        const auto created = Queue::create(std::numeric_limits<std::size_t>::max());
        ASSERT_FALSE(created.has_value());
        EXPECT_EQ(created.error().code, core::StatusCode::resource_exhausted);
        EXPECT_NE(
            created.error().message.find(std::string{wavefront_queue_kind_name(Queue::kind())}),
            std::string::npos);
        EXPECT_NE(created.error().message.find("capacity"), std::string::npos);
    });
}

TEST(WavefrontQueuesTest, MoveConstructionPreservesDestinationAndEmptiesSource) {
    for_each_queue_type([]<typename Queue> {
        auto created = Queue::create(3U);
        ASSERT_TRUE(created.has_value());
        auto source = std::move(*created);
        ASSERT_EQ(source.push(WavefrontPathSlot{.value = 21U}), WavefrontQueuePushStatus::pushed);
        const auto* const storage = source.entries().data();
        auto destination = std::move(source);

        EXPECT_EQ(destination.capacity(), 3U);
        EXPECT_EQ(destination.size(), 1U);
        EXPECT_EQ(destination.entries().data(), storage);
        EXPECT_EQ(destination.entries().front(), WavefrontPathSlot{.value = 21U});
        EXPECT_EQ(source.capacity(), 0U);
        EXPECT_EQ(source.size(), 0U);
        EXPECT_EQ(source.remaining_capacity(), 0U);
        EXPECT_TRUE(source.empty());
        EXPECT_TRUE(source.full());
        EXPECT_TRUE(source.entries().empty());
        EXPECT_EQ(source.push(WavefrontPathSlot{.value = 0U}),
                  WavefrontQueuePushStatus::capacity_exhausted);
    });
}

} // namespace
} // namespace blackframe::renderer
