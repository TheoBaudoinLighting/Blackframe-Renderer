#include <Blackframe/Renderer/CpuWavefrontScheduler.hpp>
#include <Blackframe/Renderer/DisplayPsnr.hpp>
#include <Blackframe/Renderer/Film.hpp>
#include <Blackframe/Renderer/LinearMetrics.hpp>
#include <Blackframe/Renderer/SampleStream.hpp>
#include <Blackframe/Renderer/WavefrontQueues.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <gtest/gtest.h>
#include <latch>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
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

template <typename Buffers>
concept HasRvaluePendingReadEntries =
    requires(Buffers&& buffers) { std::move(buffers).pending_read_entries(); };

template <typename Buffers>
concept HasRvalueWriteEntries = requires(Buffers&& buffers) { std::move(buffers).write_entries(); };

template <typename Operation> void for_each_queue_type(Operation operation) {
    operation.template operator()<CameraQueue>();
    operation.template operator()<RayQueue>();
    operation.template operator()<HitQueue>();
    operation.template operator()<MissQueue>();
    operation.template operator()<ShadeQueue>();
    operation.template operator()<ShadowQueue>();
    operation.template operator()<ContinuationQueue>();
}

template <typename Buffers>
[[nodiscard]] std::span<const WavefrontPathSlot> required_pending_read(const Buffers& buffers) {
    return buffers.pending_read_entries().value();
}

template <typename Operation> void for_each_double_buffer_type(Operation operation) {
    {
        SCOPED_TRACE("camera");
        operation.template operator()<CameraQueueDoubleBuffer>();
    }
    {
        SCOPED_TRACE("ray");
        operation.template operator()<RayQueueDoubleBuffer>();
    }
    {
        SCOPED_TRACE("hit");
        operation.template operator()<HitQueueDoubleBuffer>();
    }
    {
        SCOPED_TRACE("miss");
        operation.template operator()<MissQueueDoubleBuffer>();
    }
    {
        SCOPED_TRACE("shade");
        operation.template operator()<ShadeQueueDoubleBuffer>();
    }
    {
        SCOPED_TRACE("shadow");
        operation.template operator()<ShadowQueueDoubleBuffer>();
    }
    {
        SCOPED_TRACE("continuation");
        operation.template operator()<ContinuationQueueDoubleBuffer>();
    }
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

TEST(WavefrontQueueDoubleBufferTest, DeclaresSevenStronglyDistinctBoundedStages) {
    static_assert(
        AllTypesDistinct<CameraQueueDoubleBuffer, RayQueueDoubleBuffer, HitQueueDoubleBuffer,
                         MissQueueDoubleBuffer, ShadeQueueDoubleBuffer, ShadowQueueDoubleBuffer,
                         ContinuationQueueDoubleBuffer>::value);
    static_assert(CameraQueueDoubleBuffer::kind() == WavefrontQueueKind::camera);
    static_assert(RayQueueDoubleBuffer::kind() == WavefrontQueueKind::ray);
    static_assert(HitQueueDoubleBuffer::kind() == WavefrontQueueKind::hit);
    static_assert(MissQueueDoubleBuffer::kind() == WavefrontQueueKind::miss);
    static_assert(ShadeQueueDoubleBuffer::kind() == WavefrontQueueKind::shade);
    static_assert(ShadowQueueDoubleBuffer::kind() == WavefrontQueueKind::shadow);
    static_assert(ContinuationQueueDoubleBuffer::kind() == WavefrontQueueKind::continuation);

    for_each_double_buffer_type([]<typename Buffers> {
        static_assert(!std::is_default_constructible_v<Buffers>);
        static_assert(!std::is_copy_constructible_v<Buffers>);
        static_assert(std::is_nothrow_move_constructible_v<Buffers>);
        static_assert(!std::is_move_assignable_v<Buffers>);
        static_assert(!HasRvaluePendingReadEntries<Buffers>);
        static_assert(!HasRvaluePendingReadEntries<const Buffers>);
        static_assert(!HasRvalueWriteEntries<Buffers>);
        static_assert(!HasRvalueWriteEntries<const Buffers>);
        static_assert(noexcept(std::declval<Buffers&>().push_write(WavefrontPathSlot{})));
        static_assert(noexcept(
            std::declval<Buffers&>().push_write_batch(std::span<const WavefrontPathSlot>{})));
        static_assert(noexcept(std::declval<Buffers&>().publish_write_buffer()));
        static_assert(noexcept(std::declval<Buffers&>().release_read_buffer()));
    });
}

TEST(WavefrontQueueDoubleBufferTest, CreatesTwoDistinctFixedCapacityBuffersPerStage) {
    for_each_double_buffer_type([]<typename Buffers> {
        auto created = Buffers::create(4U);
        ASSERT_TRUE(created.has_value());
        auto buffers = std::move(*created);

        EXPECT_EQ(buffers.capacity(), 4U);
        EXPECT_EQ(buffers.write_size(), 0U);
        EXPECT_EQ(buffers.write_remaining_capacity(), 4U);
        EXPECT_FALSE(buffers.has_pending_read());
        EXPECT_FALSE(buffers.pending_read_entries().has_value());
        EXPECT_TRUE(buffers.write_empty());
        EXPECT_FALSE(buffers.write_full());
        const auto* const initial_write_storage = buffers.write_entries().data();
        ASSERT_EQ(buffers.push_write(WavefrontPathSlot{.value = 1U}),
                  WavefrontQueuePushStatus::pushed);
        EXPECT_FALSE(buffers.write_empty());
        ASSERT_EQ(buffers.publish_write_buffer(), WavefrontQueuePublishStatus::published);
        EXPECT_NE(required_pending_read(buffers).data(), buffers.write_entries().data());
        EXPECT_EQ(required_pending_read(buffers).data(), initial_write_storage);
        EXPECT_EQ(buffers.capacity(), 4U);
    });
}

TEST(WavefrontQueueDoubleBufferTest, PublishesWriteBatchExactlyOnceInInsertionOrder) {
    for_each_double_buffer_type([]<typename Buffers> {
        auto created = Buffers::create(4U);
        ASSERT_TRUE(created.has_value());
        auto buffers = std::move(*created);
        const auto* const initial_write_storage = buffers.write_entries().data();
        constexpr auto batch = std::array{
            WavefrontPathSlot{.value = 0U},
            WavefrontPathSlot{.value = std::numeric_limits<std::uint32_t>::max()},
            WavefrontPathSlot{.value = 17U},
            WavefrontPathSlot{.value = 17U},
        };

        ASSERT_EQ(buffers.push_write_batch(batch), WavefrontQueuePushStatus::pushed);
        EXPECT_FALSE(buffers.has_pending_read());
        EXPECT_FALSE(buffers.pending_read_entries().has_value());
        EXPECT_EQ(buffers.write_entries().data(), initial_write_storage);
        EXPECT_FALSE(buffers.write_empty());
        EXPECT_TRUE(std::ranges::equal(buffers.write_entries(), batch));

        EXPECT_EQ(buffers.publish_write_buffer(), WavefrontQueuePublishStatus::published);
        EXPECT_TRUE(buffers.has_pending_read());
        EXPECT_TRUE(buffers.pending_read_entries().has_value());
        EXPECT_EQ(required_pending_read(buffers).data(), initial_write_storage);
        EXPECT_FALSE(required_pending_read(buffers).empty());
        EXPECT_TRUE(std::ranges::equal(required_pending_read(buffers), batch));
        EXPECT_NE(buffers.write_entries().data(), initial_write_storage);
        EXPECT_TRUE(buffers.write_empty());
        EXPECT_EQ(buffers.write_remaining_capacity(), batch.size());
    });
}

TEST(WavefrontQueueDoubleBufferTest, RejectsRepublishWithoutOverwritingEitherBuffer) {
    for_each_double_buffer_type([]<typename Buffers> {
        auto created = Buffers::create(3U);
        ASSERT_TRUE(created.has_value());
        auto buffers = std::move(*created);
        constexpr auto first =
            std::array{WavefrontPathSlot{.value = 1U}, WavefrontPathSlot{.value = 2U}};
        constexpr auto second =
            std::array{WavefrontPathSlot{.value = 3U}, WavefrontPathSlot{.value = 4U},
                       WavefrontPathSlot{.value = 5U}};
        ASSERT_EQ(buffers.push_write_batch(first), WavefrontQueuePushStatus::pushed);
        ASSERT_EQ(buffers.publish_write_buffer(), WavefrontQueuePublishStatus::published);
        ASSERT_EQ(buffers.push_write_batch(second), WavefrontQueuePushStatus::pushed);
        const auto* const read_storage = required_pending_read(buffers).data();
        const auto* const write_storage = buffers.write_entries().data();

        for (auto attempt = 0; attempt < 2; ++attempt) {
            EXPECT_EQ(buffers.publish_write_buffer(),
                      WavefrontQueuePublishStatus::read_buffer_pending);
            EXPECT_TRUE(buffers.has_pending_read());
            EXPECT_EQ(buffers.capacity(), 3U);
            EXPECT_EQ(required_pending_read(buffers).size(), first.size());
            EXPECT_EQ(buffers.write_size(), second.size());
            EXPECT_EQ(buffers.write_remaining_capacity(), 0U);
            EXPECT_TRUE(buffers.write_full());
            EXPECT_EQ(required_pending_read(buffers).data(), read_storage);
            EXPECT_EQ(buffers.write_entries().data(), write_storage);
            EXPECT_TRUE(std::ranges::equal(required_pending_read(buffers), first));
            EXPECT_TRUE(std::ranges::equal(buffers.write_entries(), second));
        }

        EXPECT_EQ(buffers.release_read_buffer(), WavefrontQueueReleaseStatus::released);
        EXPECT_FALSE(buffers.has_pending_read());
        EXPECT_FALSE(buffers.pending_read_entries().has_value());
        EXPECT_EQ(buffers.write_entries().data(), write_storage);
        EXPECT_TRUE(std::ranges::equal(buffers.write_entries(), second));
        EXPECT_EQ(buffers.publish_write_buffer(), WavefrontQueuePublishStatus::published);
        EXPECT_EQ(required_pending_read(buffers).data(), write_storage);
        EXPECT_TRUE(std::ranges::equal(required_pending_read(buffers), second));
        EXPECT_EQ(buffers.write_entries().data(), read_storage);
        EXPECT_TRUE(buffers.write_empty());
    });
}

TEST(WavefrontQueueDoubleBufferTest, RejectsReleaseWithoutPendingReadAndPreservesWrite) {
    for_each_double_buffer_type([]<typename Buffers> {
        auto created = Buffers::create(2U);
        ASSERT_TRUE(created.has_value());
        auto buffers = std::move(*created);
        constexpr auto staged =
            std::array{WavefrontPathSlot{.value = 8U}, WavefrontPathSlot{.value = 13U}};
        ASSERT_EQ(buffers.push_write_batch(staged), WavefrontQueuePushStatus::pushed);
        const auto* const write_storage = buffers.write_entries().data();

        EXPECT_EQ(buffers.release_read_buffer(),
                  WavefrontQueueReleaseStatus::no_read_buffer_pending);
        EXPECT_FALSE(buffers.has_pending_read());
        EXPECT_FALSE(buffers.pending_read_entries().has_value());
        EXPECT_EQ(buffers.write_entries().data(), write_storage);
        EXPECT_TRUE(std::ranges::equal(buffers.write_entries(), staged));

        ASSERT_EQ(buffers.publish_write_buffer(), WavefrontQueuePublishStatus::published);
        ASSERT_EQ(buffers.release_read_buffer(), WavefrontQueueReleaseStatus::released);
        EXPECT_EQ(buffers.release_read_buffer(),
                  WavefrontQueueReleaseStatus::no_read_buffer_pending);
    });
}

TEST(WavefrontQueueDoubleBufferTest, RejectsSingleWriteOverflowWithoutChangingRead) {
    for_each_double_buffer_type([]<typename Buffers> {
        auto created = Buffers::create(2U);
        ASSERT_TRUE(created.has_value());
        auto buffers = std::move(*created);
        constexpr auto published = std::array{WavefrontPathSlot{.value = 10U}};
        constexpr auto staged =
            std::array{WavefrontPathSlot{.value = 20U}, WavefrontPathSlot{.value = 21U}};
        ASSERT_EQ(buffers.push_write_batch(published), WavefrontQueuePushStatus::pushed);
        ASSERT_EQ(buffers.publish_write_buffer(), WavefrontQueuePublishStatus::published);
        ASSERT_EQ(buffers.push_write_batch(staged), WavefrontQueuePushStatus::pushed);
        const auto* const read_storage = required_pending_read(buffers).data();
        const auto* const write_storage = buffers.write_entries().data();

        for (auto attempt = 0; attempt < 2; ++attempt) {
            EXPECT_EQ(buffers.push_write(WavefrontPathSlot{.value = 22U}),
                      WavefrontQueuePushStatus::capacity_exhausted);
            EXPECT_TRUE(buffers.has_pending_read());
            EXPECT_EQ(required_pending_read(buffers).data(), read_storage);
            EXPECT_EQ(buffers.write_entries().data(), write_storage);
            EXPECT_TRUE(std::ranges::equal(required_pending_read(buffers), published));
            EXPECT_TRUE(std::ranges::equal(buffers.write_entries(), staged));
        }

        ASSERT_EQ(buffers.release_read_buffer(), WavefrontQueueReleaseStatus::released);
        ASSERT_EQ(buffers.publish_write_buffer(), WavefrontQueuePublishStatus::published);
        EXPECT_TRUE(std::ranges::equal(required_pending_read(buffers), staged));
    });
}

TEST(WavefrontQueueDoubleBufferTest, RejectsBatchOverflowAtomicallyWithoutChangingRead) {
    for_each_double_buffer_type([]<typename Buffers> {
        auto created = Buffers::create(3U);
        ASSERT_TRUE(created.has_value());
        auto buffers = std::move(*created);
        constexpr auto published = std::array{WavefrontPathSlot{.value = 1U}};
        ASSERT_EQ(buffers.push_write_batch(published), WavefrontQueuePushStatus::pushed);
        ASSERT_EQ(buffers.publish_write_buffer(), WavefrontQueuePublishStatus::published);
        ASSERT_EQ(buffers.push_write(WavefrontPathSlot{.value = 2U}),
                  WavefrontQueuePushStatus::pushed);
        constexpr auto overflow = std::array{
            WavefrontPathSlot{.value = 3U},
            WavefrontPathSlot{.value = 4U},
            WavefrontPathSlot{.value = 5U},
        };
        const auto* const read_storage = required_pending_read(buffers).data();
        const auto* const write_storage = buffers.write_entries().data();

        for (auto attempt = 0; attempt < 2; ++attempt) {
            EXPECT_EQ(buffers.push_write_batch(overflow),
                      WavefrontQueuePushStatus::capacity_exhausted);
            EXPECT_TRUE(buffers.has_pending_read());
            EXPECT_EQ(buffers.capacity(), 3U);
            EXPECT_EQ(required_pending_read(buffers).size(), published.size());
            EXPECT_EQ(buffers.write_size(), 1U);
            EXPECT_EQ(buffers.write_remaining_capacity(), 2U);
            EXPECT_FALSE(buffers.write_full());
            EXPECT_EQ(required_pending_read(buffers).data(), read_storage);
            EXPECT_EQ(buffers.write_entries().data(), write_storage);
            EXPECT_TRUE(std::ranges::equal(required_pending_read(buffers), published));
            ASSERT_EQ(buffers.write_entries().size(), 1U);
            EXPECT_EQ(buffers.write_entries().front(), WavefrontPathSlot{.value = 2U});
        }

        const auto fitting = std::span<const WavefrontPathSlot>{overflow}.first(2U);
        EXPECT_EQ(buffers.push_write_batch(fitting), WavefrontQueuePushStatus::pushed);
        EXPECT_EQ(buffers.push_write_batch(std::span<const WavefrontPathSlot>{}),
                  WavefrontQueuePushStatus::pushed);
        constexpr auto expected = std::array{
            WavefrontPathSlot{.value = 2U},
            WavefrontPathSlot{.value = 3U},
            WavefrontPathSlot{.value = 4U},
        };
        EXPECT_TRUE(std::ranges::equal(buffers.write_entries(), expected));
        EXPECT_TRUE(std::ranges::equal(required_pending_read(buffers), published));
        ASSERT_EQ(buffers.release_read_buffer(), WavefrontQueueReleaseStatus::released);
        ASSERT_EQ(buffers.publish_write_buffer(), WavefrontQueuePublishStatus::published);
        EXPECT_TRUE(std::ranges::equal(required_pending_read(buffers), expected));
    });
}

TEST(WavefrontQueueDoubleBufferTest, CopiesPublishedReadIntoIndependentWriteStorage) {
    for_each_double_buffer_type([]<typename Buffers> {
        auto created = Buffers::create(4U);
        ASSERT_TRUE(created.has_value());
        auto buffers = std::move(*created);
        constexpr auto batch =
            std::array{WavefrontPathSlot{.value = 6U}, WavefrontPathSlot{.value = 12U}};
        ASSERT_EQ(buffers.push_write_batch(batch), WavefrontQueuePushStatus::pushed);
        ASSERT_EQ(buffers.publish_write_buffer(), WavefrontQueuePublishStatus::published);
        const auto* const read_storage = required_pending_read(buffers).data();
        const auto* const write_storage = buffers.write_entries().data();

        ASSERT_EQ(buffers.push_write_batch(required_pending_read(buffers)),
                  WavefrontQueuePushStatus::pushed);
        EXPECT_NE(read_storage, write_storage);
        EXPECT_EQ(required_pending_read(buffers).data(), read_storage);
        EXPECT_EQ(buffers.write_entries().data(), write_storage);
        EXPECT_TRUE(std::ranges::equal(required_pending_read(buffers), batch));
        EXPECT_TRUE(std::ranges::equal(buffers.write_entries(), batch));
        EXPECT_EQ(buffers.publish_write_buffer(), WavefrontQueuePublishStatus::read_buffer_pending);
    });
}

TEST(WavefrontQueueDoubleBufferTest, TracksEmptyPublishedGenerationAtZeroCapacity) {
    for_each_double_buffer_type([]<typename Buffers> {
        auto created = Buffers::create(0U);
        ASSERT_TRUE(created.has_value());
        auto buffers = std::move(*created);
        EXPECT_EQ(buffers.capacity(), 0U);
        EXPECT_FALSE(buffers.has_pending_read());
        EXPECT_FALSE(buffers.pending_read_entries().has_value());
        EXPECT_TRUE(buffers.write_empty());
        EXPECT_TRUE(buffers.write_full());
        EXPECT_EQ(buffers.write_remaining_capacity(), 0U);
        EXPECT_EQ(buffers.push_write(WavefrontPathSlot{}),
                  WavefrontQueuePushStatus::capacity_exhausted);
        const auto one = std::array{WavefrontPathSlot{.value = 0U}};
        EXPECT_EQ(buffers.push_write_batch(one), WavefrontQueuePushStatus::capacity_exhausted);
        EXPECT_EQ(buffers.push_write_batch(std::span<const WavefrontPathSlot>{}),
                  WavefrontQueuePushStatus::pushed);

        EXPECT_EQ(buffers.publish_write_buffer(), WavefrontQueuePublishStatus::published);
        EXPECT_TRUE(buffers.has_pending_read());
        ASSERT_TRUE(buffers.pending_read_entries().has_value());
        EXPECT_TRUE(required_pending_read(buffers).empty());
        EXPECT_EQ(buffers.publish_write_buffer(), WavefrontQueuePublishStatus::read_buffer_pending);
        EXPECT_EQ(buffers.release_read_buffer(), WavefrontQueueReleaseStatus::released);
        EXPECT_FALSE(buffers.pending_read_entries().has_value());
        EXPECT_EQ(buffers.release_read_buffer(),
                  WavefrontQueueReleaseStatus::no_read_buffer_pending);
        EXPECT_EQ(buffers.publish_write_buffer(), WavefrontQueuePublishStatus::published);
    });
}

TEST(WavefrontQueueDoubleBufferTest, RejectsUnrepresentableCapacityBeforeAllocation) {
    for_each_double_buffer_type([]<typename Buffers> {
        const auto created = Buffers::create(std::numeric_limits<std::size_t>::max());
        ASSERT_FALSE(created.has_value());
        EXPECT_EQ(created.error().code, core::StatusCode::resource_exhausted);
        EXPECT_NE(
            created.error().message.find(std::string{wavefront_queue_kind_name(Buffers::kind())}),
            std::string::npos);
        EXPECT_NE(created.error().message.find("capacity"), std::string::npos);
    });
}

TEST(WavefrontQueueDoubleBufferTest, MovePreservesBothBuffersAndCanonicalizesSource) {
    for_each_double_buffer_type([]<typename Buffers> {
        auto created = Buffers::create(3U);
        ASSERT_TRUE(created.has_value());
        auto source = std::move(*created);
        constexpr auto published =
            std::array{WavefrontPathSlot{.value = 1U}, WavefrontPathSlot{.value = 2U}};
        constexpr auto staged = std::array{WavefrontPathSlot{.value = 3U}};
        ASSERT_EQ(source.push_write_batch(published), WavefrontQueuePushStatus::pushed);
        ASSERT_EQ(source.publish_write_buffer(), WavefrontQueuePublishStatus::published);
        ASSERT_EQ(source.push_write_batch(staged), WavefrontQueuePushStatus::pushed);
        const auto* const read_storage = required_pending_read(source).data();
        const auto* const write_storage = source.write_entries().data();

        auto destination = std::move(source);
        EXPECT_EQ(destination.capacity(), 3U);
        EXPECT_TRUE(destination.has_pending_read());
        EXPECT_EQ(required_pending_read(destination).data(), read_storage);
        EXPECT_EQ(destination.write_entries().data(), write_storage);
        EXPECT_TRUE(std::ranges::equal(required_pending_read(destination), published));
        EXPECT_TRUE(std::ranges::equal(destination.write_entries(), staged));
        EXPECT_EQ(destination.publish_write_buffer(),
                  WavefrontQueuePublishStatus::read_buffer_pending);
        ASSERT_EQ(destination.release_read_buffer(), WavefrontQueueReleaseStatus::released);
        ASSERT_EQ(destination.publish_write_buffer(), WavefrontQueuePublishStatus::published);
        EXPECT_TRUE(std::ranges::equal(required_pending_read(destination), staged));

        EXPECT_EQ(source.capacity(), 0U);
        EXPECT_FALSE(source.has_pending_read());
        EXPECT_FALSE(source.pending_read_entries().has_value());
        EXPECT_TRUE(source.write_empty());
        EXPECT_TRUE(source.write_full());
        EXPECT_EQ(source.push_write(WavefrontPathSlot{}),
                  WavefrontQueuePushStatus::capacity_exhausted);
        EXPECT_EQ(source.publish_write_buffer(), WavefrontQueuePublishStatus::published);
    });
}

TEST(WavefrontCompactionTest, ExposesOnlyExplicitLaneStatesAndOrderingPolicies) {
    static_assert(sizeof(WavefrontLaneState) == sizeof(std::uint8_t));
    static_assert(sizeof(WavefrontCompactionOrder) == sizeof(std::uint8_t));
    static_assert(std::is_standard_layout_v<WavefrontCompactionReport>);
    static_assert(std::is_trivially_copyable_v<WavefrontCompactionReport>);

    EXPECT_TRUE(is_known_wavefront_lane_state(WavefrontLaneState::active));
    EXPECT_TRUE(is_known_wavefront_lane_state(WavefrontLaneState::terminated));
    EXPECT_FALSE(is_known_wavefront_lane_state(static_cast<WavefrontLaneState>(255U)));
    EXPECT_TRUE(is_known_wavefront_compaction_order(WavefrontCompactionOrder::stable_input));
    EXPECT_TRUE(
        is_known_wavefront_compaction_order(WavefrontCompactionOrder::deterministic_path_slot));
    EXPECT_FALSE(is_known_wavefront_compaction_order(static_cast<WavefrontCompactionOrder>(255U)));
}

TEST(WavefrontCompactionTest, RemovesTerminatedLanesStablyAcrossEveryStageQueue) {
    for_each_queue_type([]<typename Queue> {
        auto created = Queue::create(8U);
        ASSERT_TRUE(created.has_value());
        auto queue = std::move(*created);
        constexpr auto slots = std::array{
            WavefrontPathSlot{.value = std::numeric_limits<std::uint32_t>::max()},
            WavefrontPathSlot{.value = 6U},
            WavefrontPathSlot{.value = 1U},
            WavefrontPathSlot{.value = 1U},
            WavefrontPathSlot{.value = 5U},
            WavefrontPathSlot{.value = 0U},
            WavefrontPathSlot{.value = 3U},
            WavefrontPathSlot{.value = 2U},
        };
        constexpr auto lane_states = std::array{
            WavefrontLaneState::active,     WavefrontLaneState::active,
            WavefrontLaneState::active,     WavefrontLaneState::active,
            WavefrontLaneState::terminated, WavefrontLaneState::active,
            WavefrontLaneState::terminated, WavefrontLaneState::active,
        };
        constexpr auto expected = std::array{
            WavefrontPathSlot{.value = std::numeric_limits<std::uint32_t>::max()},
            WavefrontPathSlot{.value = 6U},
            WavefrontPathSlot{.value = 1U},
            WavefrontPathSlot{.value = 1U},
            WavefrontPathSlot{.value = 0U},
            WavefrontPathSlot{.value = 2U},
        };
        ASSERT_EQ(queue.push_batch(slots), WavefrontQueuePushStatus::pushed);
        const auto* const storage = queue.entries().data();

        const auto report =
            queue.compact_terminated(lane_states, WavefrontCompactionOrder::stable_input);
        ASSERT_TRUE(report.has_value()) << report.error().message;
        EXPECT_EQ(*report, (WavefrontCompactionReport{
                               .input_lanes = 8U,
                               .active_lanes = 6U,
                               .terminated_lanes = 2U,
                               .order = WavefrontCompactionOrder::stable_input,
                           }));
        EXPECT_EQ(queue.entries().data(), storage);
        EXPECT_EQ(queue.capacity(), slots.size());
        EXPECT_EQ(queue.remaining_capacity(), 2U);
        EXPECT_TRUE(std::ranges::equal(queue.entries(), expected));
    });
}

TEST(WavefrontCompactionTest, OptionallyCanonicalizesSurvivorsByStablePathSlot) {
    for_each_queue_type([]<typename Queue> {
        constexpr auto first_slots = std::array{
            WavefrontPathSlot{.value = std::numeric_limits<std::uint32_t>::max()},
            WavefrontPathSlot{.value = 6U},
            WavefrontPathSlot{.value = 1U},
            WavefrontPathSlot{.value = 1U},
            WavefrontPathSlot{.value = 5U},
            WavefrontPathSlot{.value = 0U},
            WavefrontPathSlot{.value = 3U},
            WavefrontPathSlot{.value = 2U},
        };
        constexpr auto first_states = std::array{
            WavefrontLaneState::active,     WavefrontLaneState::active,
            WavefrontLaneState::active,     WavefrontLaneState::active,
            WavefrontLaneState::terminated, WavefrontLaneState::active,
            WavefrontLaneState::terminated, WavefrontLaneState::active,
        };
        constexpr auto second_slots = std::array{
            WavefrontPathSlot{.value = 2U},
            WavefrontPathSlot{.value = 3U},
            WavefrontPathSlot{.value = 0U},
            WavefrontPathSlot{.value = 5U},
            WavefrontPathSlot{.value = 1U},
            WavefrontPathSlot{.value = 1U},
            WavefrontPathSlot{.value = 6U},
            WavefrontPathSlot{.value = std::numeric_limits<std::uint32_t>::max()},
        };
        constexpr auto second_states = std::array{
            WavefrontLaneState::active, WavefrontLaneState::terminated,
            WavefrontLaneState::active, WavefrontLaneState::terminated,
            WavefrontLaneState::active, WavefrontLaneState::active,
            WavefrontLaneState::active, WavefrontLaneState::active,
        };
        constexpr auto expected = std::array{
            WavefrontPathSlot{.value = 0U},
            WavefrontPathSlot{.value = 1U},
            WavefrontPathSlot{.value = 1U},
            WavefrontPathSlot{.value = 2U},
            WavefrontPathSlot{.value = 6U},
            WavefrontPathSlot{.value = std::numeric_limits<std::uint32_t>::max()},
        };
        auto first_created = Queue::create(first_slots.size());
        auto second_created = Queue::create(second_slots.size());
        ASSERT_TRUE(first_created.has_value());
        ASSERT_TRUE(second_created.has_value());
        auto first = std::move(*first_created);
        auto second = std::move(*second_created);
        ASSERT_EQ(first.push_batch(first_slots), WavefrontQueuePushStatus::pushed);
        ASSERT_EQ(second.push_batch(second_slots), WavefrontQueuePushStatus::pushed);

        const auto first_report = first.compact_terminated(
            first_states, WavefrontCompactionOrder::deterministic_path_slot);
        const auto second_report = second.compact_terminated(
            second_states, WavefrontCompactionOrder::deterministic_path_slot);
        ASSERT_TRUE(first_report.has_value()) << first_report.error().message;
        ASSERT_TRUE(second_report.has_value()) << second_report.error().message;
        EXPECT_EQ(first_report->order, WavefrontCompactionOrder::deterministic_path_slot);
        EXPECT_EQ(second_report->order, WavefrontCompactionOrder::deterministic_path_slot);
        EXPECT_TRUE(std::ranges::equal(first.entries(), expected));
        EXPECT_TRUE(std::ranges::equal(second.entries(), expected));
    });
}

TEST(WavefrontCompactionTest, RejectsMalformedRequestsAtomicallyAndRecoversCapacity) {
    for_each_queue_type([]<typename Queue> {
        auto created = Queue::create(3U);
        ASSERT_TRUE(created.has_value());
        auto queue = std::move(*created);
        constexpr auto slots = std::array{
            WavefrontPathSlot{.value = 4U},
            WavefrontPathSlot{.value = 2U},
            WavefrontPathSlot{.value = 6U},
        };
        constexpr auto valid_states = std::array{
            WavefrontLaneState::active, WavefrontLaneState::active, WavefrontLaneState::active};
        constexpr auto short_states =
            std::array{WavefrontLaneState::active, WavefrontLaneState::terminated};
        constexpr auto long_states =
            std::array{WavefrontLaneState::active, WavefrontLaneState::active,
                       WavefrontLaneState::active, WavefrontLaneState::terminated};
        ASSERT_EQ(queue.push_batch(slots), WavefrontQueuePushStatus::pushed);
        const auto* const storage = queue.entries().data();

        const auto expect_rejected_without_mutation = [&](const auto& states,
                                                          const WavefrontCompactionOrder order) {
            const auto result = queue.compact_terminated(states, order);
            ASSERT_FALSE(result.has_value());
            EXPECT_EQ(result.error().code, core::StatusCode::invalid_argument);
            EXPECT_EQ(queue.entries().data(), storage);
            EXPECT_TRUE(std::ranges::equal(queue.entries(), slots));
        };
        expect_rejected_without_mutation(short_states, WavefrontCompactionOrder::stable_input);
        expect_rejected_without_mutation(long_states, WavefrontCompactionOrder::stable_input);
        auto unknown_states = valid_states;
        unknown_states[1U] = static_cast<WavefrontLaneState>(255U);
        expect_rejected_without_mutation(unknown_states, WavefrontCompactionOrder::stable_input);
        expect_rejected_without_mutation(valid_states, static_cast<WavefrontCompactionOrder>(255U));

        const auto all_live =
            queue.compact_terminated(valid_states, WavefrontCompactionOrder::stable_input);
        ASSERT_TRUE(all_live.has_value()) << all_live.error().message;
        EXPECT_TRUE(queue.full());
        EXPECT_TRUE(std::ranges::equal(queue.entries(), slots));
        constexpr auto all_terminated =
            std::array{WavefrontLaneState::terminated, WavefrontLaneState::terminated,
                       WavefrontLaneState::terminated};
        const auto terminated =
            queue.compact_terminated(all_terminated, WavefrontCompactionOrder::stable_input);
        ASSERT_TRUE(terminated.has_value()) << terminated.error().message;
        EXPECT_EQ(terminated->active_lanes, 0U);
        EXPECT_EQ(terminated->terminated_lanes, slots.size());
        EXPECT_TRUE(queue.empty());
        EXPECT_EQ(queue.remaining_capacity(), slots.size());
        EXPECT_EQ(queue.entries().data(), storage);
        EXPECT_EQ(queue.push_batch(slots), WavefrontQueuePushStatus::pushed);
        EXPECT_EQ(queue.push(WavefrontPathSlot{.value = 99U}),
                  WavefrontQueuePushStatus::capacity_exhausted);

        auto zero_created = Queue::create(0U);
        ASSERT_TRUE(zero_created.has_value());
        auto zero = std::move(*zero_created);
        const auto empty = zero.compact_terminated(std::span<const WavefrontLaneState>{},
                                                   WavefrontCompactionOrder::stable_input);
        ASSERT_TRUE(empty.has_value()) << empty.error().message;
        EXPECT_EQ(*empty, (WavefrontCompactionReport{
                              .input_lanes = 0U,
                              .active_lanes = 0U,
                              .terminated_lanes = 0U,
                              .order = WavefrontCompactionOrder::stable_input,
                          }));
    });
}

TEST(WavefrontCompactionTest, CompactsOnlyTheWriteGenerationOfEveryDoubleBuffer) {
    for_each_double_buffer_type([]<typename Buffers> {
        auto created = Buffers::create(4U);
        ASSERT_TRUE(created.has_value());
        auto buffers = std::move(*created);
        constexpr auto published =
            std::array{WavefrontPathSlot{.value = 91U}, WavefrontPathSlot{.value = 92U}};
        constexpr auto staged =
            std::array{WavefrontPathSlot{.value = 9U}, WavefrontPathSlot{.value = 1U},
                       WavefrontPathSlot{.value = 7U}, WavefrontPathSlot{.value = 3U}};
        constexpr auto lane_states =
            std::array{WavefrontLaneState::active, WavefrontLaneState::terminated,
                       WavefrontLaneState::active, WavefrontLaneState::terminated};
        constexpr auto expected =
            std::array{WavefrontPathSlot{.value = 9U}, WavefrontPathSlot{.value = 7U}};
        ASSERT_EQ(buffers.push_write_batch(published), WavefrontQueuePushStatus::pushed);
        ASSERT_EQ(buffers.publish_write_buffer(), WavefrontQueuePublishStatus::published);
        ASSERT_EQ(buffers.push_write_batch(staged), WavefrontQueuePushStatus::pushed);
        const auto* const read_storage = required_pending_read(buffers).data();
        const auto* const write_storage = buffers.write_entries().data();

        const auto expect_rejected_without_mutation =
            [&](const std::span<const WavefrontLaneState> states,
                const WavefrontCompactionOrder order) {
                const auto rejected = buffers.compact_write_terminated(states, order);
                ASSERT_FALSE(rejected.has_value());
                EXPECT_EQ(rejected.error().code, core::StatusCode::invalid_argument);
                EXPECT_TRUE(buffers.has_pending_read());
                EXPECT_EQ(required_pending_read(buffers).data(), read_storage);
                EXPECT_EQ(buffers.write_entries().data(), write_storage);
                EXPECT_EQ(required_pending_read(buffers).size(), published.size());
                EXPECT_EQ(buffers.write_size(), staged.size());
                EXPECT_EQ(buffers.capacity(), 4U);
                EXPECT_TRUE(std::ranges::equal(required_pending_read(buffers), published));
                EXPECT_TRUE(std::ranges::equal(buffers.write_entries(), staged));
            };
        expect_rejected_without_mutation(std::span<const WavefrontLaneState>{lane_states}.first(3U),
                                         WavefrontCompactionOrder::stable_input);
        auto unknown_states = lane_states;
        unknown_states[1U] = static_cast<WavefrontLaneState>(255U);
        expect_rejected_without_mutation(unknown_states, WavefrontCompactionOrder::stable_input);
        expect_rejected_without_mutation(lane_states, static_cast<WavefrontCompactionOrder>(255U));

        const auto report =
            buffers.compact_write_terminated(lane_states, WavefrontCompactionOrder::stable_input);
        ASSERT_TRUE(report.has_value()) << report.error().message;
        EXPECT_TRUE(buffers.has_pending_read());
        EXPECT_EQ(required_pending_read(buffers).data(), read_storage);
        EXPECT_EQ(buffers.write_entries().data(), write_storage);
        EXPECT_TRUE(std::ranges::equal(required_pending_read(buffers), published));
        EXPECT_TRUE(std::ranges::equal(buffers.write_entries(), expected));
        EXPECT_EQ(buffers.publish_write_buffer(), WavefrontQueuePublishStatus::read_buffer_pending);

        auto moved = std::move(buffers);
        EXPECT_TRUE(std::ranges::equal(required_pending_read(moved), published));
        EXPECT_TRUE(std::ranges::equal(moved.write_entries(), expected));
        ASSERT_EQ(moved.release_read_buffer(), WavefrontQueueReleaseStatus::released);
        ASSERT_EQ(moved.publish_write_buffer(), WavefrontQueuePublishStatus::published);
        EXPECT_TRUE(std::ranges::equal(required_pending_read(moved), expected));

        auto empty_created = Buffers::create(2U);
        ASSERT_TRUE(empty_created.has_value());
        auto empty_generation = std::move(*empty_created);
        constexpr auto terminated_slots =
            std::array{WavefrontPathSlot{.value = 1U}, WavefrontPathSlot{.value = 2U}};
        constexpr auto terminated_states =
            std::array{WavefrontLaneState::terminated, WavefrontLaneState::terminated};
        ASSERT_EQ(empty_generation.push_write_batch(terminated_slots),
                  WavefrontQueuePushStatus::pushed);
        ASSERT_TRUE(
            empty_generation
                .compact_write_terminated(terminated_states, WavefrontCompactionOrder::stable_input)
                .has_value());
        ASSERT_EQ(empty_generation.publish_write_buffer(), WavefrontQueuePublishStatus::published);
        ASSERT_TRUE(empty_generation.pending_read_entries().has_value());
        EXPECT_TRUE(required_pending_read(empty_generation).empty());
    });
}

TEST(CpuWavefrontSchedulerTest, RequiresAnExplicitBoundedWorkerCount) {
    static_assert(sizeof(CpuWavefrontSchedulerMode) == sizeof(std::uint8_t));
    static_assert(std::is_standard_layout_v<CpuWavefrontLane>);
    static_assert(std::is_trivially_copyable_v<CpuWavefrontLane>);
    static_assert(std::is_standard_layout_v<CpuWavefrontStageReport>);
    static_assert(std::is_trivially_copyable_v<CpuWavefrontStageReport>);
    static_assert(!std::is_default_constructible_v<CpuWavefrontScheduler>);
    static_assert(!std::is_copy_constructible_v<CpuWavefrontScheduler>);
    static_assert(std::is_nothrow_move_constructible_v<CpuWavefrontScheduler>);

    const auto zero = CpuWavefrontScheduler::create(0U);
    ASSERT_FALSE(zero.has_value());
    EXPECT_EQ(zero.error().code, core::StatusCode::invalid_argument);
    const auto excessive = CpuWavefrontScheduler::create(MaxCpuWavefrontWorkerCount + 1U);
    ASSERT_FALSE(excessive.has_value());
    EXPECT_EQ(excessive.error().code, core::StatusCode::invalid_argument);

    const auto one = CpuWavefrontScheduler::create(1U);
    ASSERT_TRUE(one.has_value()) << one.error().message;
    EXPECT_EQ(one->worker_count(), 1U);
    EXPECT_EQ(one->mode(), CpuWavefrontSchedulerMode::single_thread);
    const auto many = CpuWavefrontScheduler::create(4U);
    ASSERT_TRUE(many.has_value()) << many.error().message;
    EXPECT_EQ(many->worker_count(), 4U);
    EXPECT_EQ(many->mode(), CpuWavefrontSchedulerMode::fixed_thread_count);

    EXPECT_TRUE(is_known_cpu_wavefront_scheduler_mode(CpuWavefrontSchedulerMode::single_thread));
    EXPECT_TRUE(
        is_known_cpu_wavefront_scheduler_mode(CpuWavefrontSchedulerMode::fixed_thread_count));
    EXPECT_FALSE(
        is_known_cpu_wavefront_scheduler_mode(static_cast<CpuWavefrontSchedulerMode>(255U)));
}

TEST(CpuWavefrontSchedulerTest, RejectsEveryMalformedDispatchBeforeInvokingTheKernel) {
    const auto scheduler = CpuWavefrontScheduler::create(4U).value();
    auto invocation_count = std::atomic_uint32_t{};
    const auto kernel = CpuWavefrontStageKernel{[&](const CpuWavefrontLane) -> core::Status {
        invocation_count.fetch_add(1U, std::memory_order_relaxed);
        return {};
    }};
    constexpr auto valid_slots =
        std::array{WavefrontPathSlot{.value = 0U}, WavefrontPathSlot{.value = 1U}};

    const auto unknown_stage = scheduler.execute_stage(static_cast<WavefrontQueueKind>(255U),
                                                       valid_slots.size(), valid_slots, kernel);
    ASSERT_FALSE(unknown_stage.has_value());
    EXPECT_EQ(unknown_stage.error().code, core::StatusCode::invalid_argument);

    const auto out_of_domain =
        scheduler.execute_stage(WavefrontQueueKind::ray, 1U, valid_slots, kernel);
    ASSERT_FALSE(out_of_domain.has_value());
    EXPECT_EQ(out_of_domain.error().code, core::StatusCode::invalid_argument);

    constexpr auto maximum_slot =
        std::array{WavefrontPathSlot{.value = std::numeric_limits<std::uint32_t>::max()}};
    const auto maximum_out_of_domain = scheduler.execute_stage(
        WavefrontQueueKind::ray, std::numeric_limits<std::uint32_t>::max(), maximum_slot, kernel);
    ASSERT_FALSE(maximum_out_of_domain.has_value());
    EXPECT_EQ(maximum_out_of_domain.error().code, core::StatusCode::invalid_argument);

    const auto empty_kernel = CpuWavefrontStageKernel{};
    const auto missing_kernel = scheduler.execute_stage(
        WavefrontQueueKind::ray, 0U, std::span<const WavefrontPathSlot>{}, empty_kernel);
    ASSERT_FALSE(missing_kernel.has_value());
    EXPECT_EQ(missing_kernel.error().code, core::StatusCode::invalid_argument);

    if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t)) {
        constexpr auto excessive_domain =
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 2U;
        const auto unaddressable =
            scheduler.execute_stage(WavefrontQueueKind::ray, excessive_domain,
                                    std::span<const WavefrontPathSlot>{}, kernel);
        ASSERT_FALSE(unaddressable.has_value());
        EXPECT_EQ(unaddressable.error().code, core::StatusCode::invalid_argument);
    }
    EXPECT_EQ(invocation_count.load(std::memory_order_relaxed), 0U);

    if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t)) {
        constexpr auto complete_domain =
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1U;
        const auto maximum_valid =
            scheduler.execute_stage(WavefrontQueueKind::ray, complete_domain, maximum_slot, kernel);
        ASSERT_TRUE(maximum_valid.has_value()) << maximum_valid.error().message;
        EXPECT_EQ(maximum_valid->path_slot_domain_size, complete_domain);
        EXPECT_EQ(invocation_count.load(std::memory_order_relaxed), 1U);
    }
}

TEST(CpuWavefrontSchedulerTest, RunsTheSingleThreadModeInlineAndPreservesDuplicateLanes) {
    const auto scheduler = CpuWavefrontScheduler::create(1U).value();
    constexpr auto slots =
        std::array{WavefrontPathSlot{.value = 3U}, WavefrontPathSlot{.value = 1U},
                   WavefrontPathSlot{.value = 1U}, WavefrontPathSlot{.value = 7U}};
    auto invocations = std::array<CpuWavefrontLane, slots.size()>{};
    const auto calling_thread = std::this_thread::get_id();
    auto worker_thread = std::thread::id{};
    const auto kernel = CpuWavefrontStageKernel{[&](const CpuWavefrontLane lane) -> core::Status {
        worker_thread = std::this_thread::get_id();
        invocations[lane.lane_index] = lane;
        return {};
    }};

    const auto report = scheduler.execute_stage(WavefrontQueueKind::shade, 8U, slots, kernel);
    ASSERT_TRUE(report.has_value()) << report.error().message;
    EXPECT_EQ(*report, (CpuWavefrontStageReport{
                           .stage = WavefrontQueueKind::shade,
                           .mode = CpuWavefrontSchedulerMode::single_thread,
                           .input_lanes = slots.size(),
                           .path_slot_domain_size = 8U,
                           .configured_workers = 1U,
                           .workers_used = 1U,
                       }));
    EXPECT_EQ(worker_thread, calling_thread);
    for (auto lane_index = std::size_t{}; lane_index < slots.size(); ++lane_index) {
        EXPECT_EQ(invocations[lane_index], (CpuWavefrontLane{
                                               .stage = WavefrontQueueKind::shade,
                                               .lane_index = lane_index,
                                               .path_slot = slots[lane_index],
                                               .worker_index = 0U,
                                           }));
    }
    EXPECT_EQ(invocations[1U].path_slot, invocations[2U].path_slot);

    auto empty_invoked = false;
    const auto empty = scheduler.execute_stage(
        WavefrontQueueKind::shade, 0U, std::span<const WavefrontPathSlot>{},
        CpuWavefrontStageKernel{[&](const CpuWavefrontLane) -> core::Status {
            empty_invoked = true;
            return {};
        }});
    ASSERT_TRUE(empty.has_value()) << empty.error().message;
    EXPECT_EQ(empty->input_lanes, 0U);
    EXPECT_EQ(empty->workers_used, 0U);
    EXPECT_FALSE(empty_invoked);
}

TEST(CpuWavefrontSchedulerTest, CoversUnevenFixedThreadPartitionsExactlyOnceAndJoins) {
    constexpr auto lane_count = std::size_t{257};
    constexpr auto worker_count = std::uint32_t{4};
    auto slots = std::array<WavefrontPathSlot, lane_count>{};
    for (auto lane_index = std::size_t{}; lane_index < lane_count; ++lane_index) {
        slots[lane_index].value = static_cast<std::uint32_t>(lane_index % 31U);
    }
    auto visits = std::array<std::atomic_uint32_t, lane_count>{};
    auto observed_slots = std::array<std::atomic_uint32_t, lane_count>{};
    auto worker_for_lane = std::array<std::atomic_uint32_t, lane_count>{};
    for (auto& worker_index : worker_for_lane) {
        worker_index.store(std::numeric_limits<std::uint32_t>::max(), std::memory_order_relaxed);
    }
    auto first_worker_threads = std::array<std::thread::id, worker_count>{};
    auto first_callbacks_completed = std::atomic_size_t{};
    const auto calling_thread = std::this_thread::get_id();
    const auto scheduler = CpuWavefrontScheduler::create(worker_count).value();
    const auto kernel = CpuWavefrontStageKernel{[&](const CpuWavefrontLane lane) -> core::Status {
        visits[lane.lane_index].fetch_add(1U, std::memory_order_relaxed);
        observed_slots[lane.lane_index].store(lane.path_slot.value, std::memory_order_relaxed);
        auto unclaimed = std::numeric_limits<std::uint32_t>::max();
        worker_for_lane[lane.lane_index].compare_exchange_strong(unclaimed, lane.worker_index,
                                                                 std::memory_order_relaxed);
        first_worker_threads[lane.worker_index] = std::this_thread::get_id();
        first_callbacks_completed.fetch_add(1U, std::memory_order_release);
        return {};
    }};

    const auto report = scheduler.execute_stage(WavefrontQueueKind::hit, 31U, slots, kernel);
    ASSERT_TRUE(report.has_value()) << report.error().message;
    EXPECT_EQ(report->configured_workers, worker_count);
    EXPECT_EQ(report->workers_used, worker_count);
    EXPECT_EQ(report->input_lanes, lane_count);
    constexpr auto partition_boundaries =
        std::array<std::size_t, worker_count + 1U>{0U, 65U, 129U, 193U, 257U};
    for (auto worker_index = std::uint32_t{}; worker_index < worker_count; ++worker_index) {
        EXPECT_NE(first_worker_threads[worker_index], std::thread::id{});
        EXPECT_NE(first_worker_threads[worker_index], calling_thread);
        for (auto other = worker_index + 1U; other < worker_count; ++other) {
            EXPECT_NE(first_worker_threads[worker_index], first_worker_threads[other]);
        }
        for (auto lane_index = partition_boundaries[worker_index];
             lane_index < partition_boundaries[worker_index + 1U]; ++lane_index) {
            EXPECT_EQ(worker_for_lane[lane_index].load(std::memory_order_relaxed), worker_index);
        }
    }
    for (auto lane_index = std::size_t{}; lane_index < lane_count; ++lane_index) {
        EXPECT_EQ(visits[lane_index].load(std::memory_order_relaxed), 1U);
        EXPECT_EQ(observed_slots[lane_index].load(std::memory_order_relaxed),
                  slots[lane_index].value);
    }
    EXPECT_EQ(first_callbacks_completed.load(std::memory_order_acquire), lane_count);

    auto barrier_violation = std::atomic_bool{false};
    auto second_visits = std::array<std::atomic_uint32_t, lane_count>{};
    auto second_worker_threads = std::array<std::thread::id, worker_count>{};
    const auto second_report = scheduler.execute_stage(
        WavefrontQueueKind::continuation, 31U, slots,
        CpuWavefrontStageKernel{[&](const CpuWavefrontLane lane) -> core::Status {
            if (first_callbacks_completed.load(std::memory_order_acquire) != lane_count) {
                barrier_violation.store(true, std::memory_order_relaxed);
            }
            second_worker_threads[lane.worker_index] = std::this_thread::get_id();
            second_visits[lane.lane_index].fetch_add(1U, std::memory_order_release);
            return {};
        }});
    ASSERT_TRUE(second_report.has_value()) << second_report.error().message;
    EXPECT_FALSE(barrier_violation.load(std::memory_order_relaxed));
    for (auto worker_index = std::uint32_t{}; worker_index < worker_count; ++worker_index) {
        EXPECT_EQ(second_worker_threads[worker_index], first_worker_threads[worker_index]);
    }
    for (const auto& visit_count : second_visits) {
        EXPECT_EQ(visit_count.load(std::memory_order_acquire), 1U);
    }
}

TEST(CpuWavefrontSchedulerTest, ReturnsTheLowestInputLaneFailureAndTranslatesExceptions) {
    constexpr auto lane_count = std::size_t{33};
    auto slots = std::array<WavefrontPathSlot, lane_count>{};
    for (auto lane_index = std::size_t{}; lane_index < lane_count; ++lane_index) {
        slots[lane_index].value = static_cast<std::uint32_t>(lane_index);
    }
    const auto scheduler = CpuWavefrontScheduler::create(4U).value();
    auto visits = std::array<std::atomic_uint32_t, lane_count>{};
    auto later_failure_reached = std::latch{1};
    const auto deterministic_failure = scheduler.execute_stage(
        WavefrontQueueKind::shadow, lane_count, slots,
        CpuWavefrontStageKernel{[&](const CpuWavefrontLane lane) -> core::Status {
            visits[lane.lane_index].fetch_add(1U, std::memory_order_relaxed);
            if (lane.lane_index == 2U) {
                later_failure_reached.wait();
                return std::unexpected(core::Error{
                    .code = core::StatusCode::unavailable,
                    .message = "lowest input lane failure",
                });
            }
            if (lane.lane_index == 20U) {
                later_failure_reached.count_down();
                return std::unexpected(core::Error{
                    .code = core::StatusCode::internal_error,
                    .message = "later input lane failure",
                });
            }
            return {};
        }});
    ASSERT_FALSE(deterministic_failure.has_value());
    EXPECT_EQ(deterministic_failure.error().code, core::StatusCode::unavailable);
    EXPECT_EQ(deterministic_failure.error().message, "lowest input lane failure");
    for (const auto& visit_count : visits) {
        EXPECT_EQ(visit_count.load(std::memory_order_relaxed), 1U);
    }

    const auto allocation_exception = scheduler.execute_stage(
        WavefrontQueueKind::shadow, lane_count, slots,
        CpuWavefrontStageKernel{[](const CpuWavefrontLane lane) -> core::Status {
            if (lane.lane_index == 3U) {
                throw std::bad_alloc{};
            }
            return {};
        }});
    ASSERT_FALSE(allocation_exception.has_value());
    EXPECT_EQ(allocation_exception.error().code, core::StatusCode::resource_exhausted);

    const auto standard_exception = scheduler.execute_stage(
        WavefrontQueueKind::shadow, lane_count, slots,
        CpuWavefrontStageKernel{[](const CpuWavefrontLane lane) -> core::Status {
            if (lane.lane_index == 4U) {
                throw std::runtime_error{"synthetic worker failure"};
            }
            return {};
        }});
    ASSERT_FALSE(standard_exception.has_value());
    EXPECT_EQ(standard_exception.error().code, core::StatusCode::internal_error);

    const auto unknown_exception = scheduler.execute_stage(
        WavefrontQueueKind::shadow, lane_count, slots,
        CpuWavefrontStageKernel{[](const CpuWavefrontLane lane) -> core::Status {
            if (lane.lane_index == 5U) {
                throw 17;
            }
            return {};
        }});
    ASSERT_FALSE(unknown_exception.has_value());
    EXPECT_EQ(unknown_exception.error().code, core::StatusCode::internal_error);

    auto nested_status = std::atomic_uint32_t{};
    const auto reentrant = scheduler.execute_stage(
        WavefrontQueueKind::shadow, lane_count, slots,
        CpuWavefrontStageKernel{[&](const CpuWavefrontLane lane) -> core::Status {
            if (lane.lane_index == 0U) {
                const auto nested = scheduler.execute_stage(
                    WavefrontQueueKind::ray, 0U, std::span<const WavefrontPathSlot>{},
                    CpuWavefrontStageKernel{
                        [](const CpuWavefrontLane) -> core::Status { return {}; }});
                nested_status.store(static_cast<std::uint32_t>(nested.has_value()
                                                                   ? core::StatusCode::success
                                                                   : nested.error().code),
                                    std::memory_order_relaxed);
            }
            return {};
        }});
    ASSERT_TRUE(reentrant.has_value()) << reentrant.error().message;
    EXPECT_EQ(nested_status.load(std::memory_order_relaxed),
              static_cast<std::uint32_t>(core::StatusCode::unavailable));
}

TEST(CpuWavefrontSchedulerTest, MatchesIndexedStageResultsWithOneAndManyThreads) {
    constexpr auto width = std::uint32_t{16};
    constexpr auto height = std::uint32_t{16};
    constexpr auto lane_count = static_cast<std::size_t>(width) * height;
    constexpr auto queue_kinds = std::array{
        WavefrontQueueKind::camera,       WavefrontQueueKind::ray,   WavefrontQueueKind::hit,
        WavefrontQueueKind::miss,         WavefrontQueueKind::shade, WavefrontQueueKind::shadow,
        WavefrontQueueKind::continuation,
    };
    auto slots = std::array<WavefrontPathSlot, lane_count>{};
    for (auto lane_index = std::size_t{}; lane_index < lane_count; ++lane_index) {
        slots[lane_index].value = static_cast<std::uint32_t>((lane_index * 73U) % lane_count);
    }
    auto single_values = std::array<Film::Color, lane_count>{};
    auto parallel_values = std::array<Film::Color, lane_count>{};
    const auto single = CpuWavefrontScheduler::create(1U).value();
    const auto parallel = CpuWavefrontScheduler::create(4U).value();

    const auto execute_all_stages = [&](const CpuWavefrontScheduler& scheduler, auto& output) {
        for (auto stage_index = std::size_t{}; stage_index < queue_kinds.size(); ++stage_index) {
            const auto kernel = CpuWavefrontStageKernel{
                [&, stage_index](const CpuWavefrontLane lane) -> core::Status {
                    const auto pixel = lane.path_slot.value;
                    const auto stream = SampleStream{SampleStreamIndex{
                        .pixel_x = pixel % width,
                        .pixel_y = pixel / width,
                        .sample_index = stage_index,
                        .seed = 0xD1B54A32D192ED03ULL,
                    }};
                    constexpr auto stage_scale = TransportScalar{1.0F / 7.0F};
                    output[lane.lane_index].red += stream.sample_1d(stage_index * 3U) * stage_scale;
                    output[lane.lane_index].green +=
                        stream.sample_1d(stage_index * 3U + 1U) * stage_scale;
                    output[lane.lane_index].blue +=
                        stream.sample_1d(stage_index * 3U + 2U) * stage_scale;
                    return {};
                }};
            const auto report =
                scheduler.execute_stage(queue_kinds[stage_index], lane_count, slots, kernel);
            EXPECT_TRUE(report.has_value())
                << (report.has_value() ? std::string{} : report.error().message);
            if (report.has_value()) {
                EXPECT_EQ(report->stage, queue_kinds[stage_index]);
                EXPECT_EQ(report->input_lanes, lane_count);
            }
        }
    };
    execute_all_stages(single, single_values);
    execute_all_stages(parallel, parallel_values);
    EXPECT_EQ(single_values, parallel_values);

    auto single_film = Film::create(RenderExtent{.width = width, .height = height}).value();
    auto parallel_film = Film::create(RenderExtent{.width = width, .height = height}).value();
    for (auto lane_index = std::size_t{}; lane_index < lane_count; ++lane_index) {
        const auto x = static_cast<std::uint32_t>(lane_index) % width;
        const auto y = static_cast<std::uint32_t>(lane_index) / width;
        ASSERT_TRUE(single_film.add_sample(x, y, single_values[lane_index], 1.0F).has_value());
        ASSERT_TRUE(parallel_film.add_sample(x, y, parallel_values[lane_index], 1.0F).has_value());
    }

    const auto metrics = compute_linear_metrics(parallel_film, single_film);
    ASSERT_TRUE(metrics.has_value()) << metrics.error().message;
    EXPECT_LE(metrics->mse, 1.0e-12);
    EXPECT_LE(metrics->rmse, 1.0e-6);
    EXPECT_LE(std::abs(metrics->mean_bias), 1.0e-12);
    EXPECT_LE(metrics->maximum_absolute_error, 1.0e-6);
    const auto psnr = compute_display_psnr(parallel_film, single_film);
    ASSERT_TRUE(psnr.has_value()) << psnr.error().message;
    EXPECT_TRUE(std::isinf(psnr->psnr) || psnr->psnr >= 80.0);
    EXPECT_TRUE(std::ranges::all_of(psnr->difference_heatmap,
                                    [](const auto difference) { return difference == 0.0; }));
}

} // namespace
} // namespace blackframe::renderer
