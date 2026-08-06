#include <Blackframe/Backends/GPU/CUDA/WavefrontQueues.hpp>
#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <Blackframe/XPU/CUDA/WavefrontQueueCompaction.hpp>
#include <Blackframe/XPU/CUDA/WavefrontStageKernel.hpp>
#include <Blackframe/XPU/Shared/TransportAbi.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

namespace cuda = xpu::cuda;
namespace shared = xpu::shared;

constexpr auto QueueCount = static_cast<std::size_t>(cuda::CudaWavefrontQueueCount);
constexpr auto SentinelSlot = std::uint32_t{0xF17EC0DEU};

enum class SelectionPattern : std::uint8_t {
    none,
    all,
    alternating,
    sparse,
};

struct RawQueueState final {
    std::array<shared::QueueHeader, QueueCount> headers{};
    std::vector<std::uint32_t> slots;
};

struct CompactionObservation final {
    cuda::WavefrontQueueCompactionResult result{};
    std::size_t scratch_bytes{};
};

[[nodiscard]] testing::AssertionResult select_test_device() {
    auto device_count = int{};
    const auto count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess) {
        return testing::AssertionFailure()
               << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_status);
    }
    if (device_count <= 0) {
        return testing::AssertionFailure() << "No CUDA device is available.";
    }
    const auto select_status = cudaSetDevice(0);
    if (select_status != cudaSuccess) {
        return testing::AssertionFailure()
               << "cudaSetDevice failed: " << cudaGetErrorString(select_status);
    }
    return testing::AssertionSuccess();
}

[[nodiscard]] constexpr std::uint32_t route_value(const cuda::WavefrontStageRoute route) noexcept {
    return static_cast<std::uint32_t>(route);
}

[[nodiscard]] constexpr bool selected(const SelectionPattern pattern, const std::uint32_t index,
                                      const std::uint32_t count) noexcept {
    switch (pattern) {
    case SelectionPattern::none:
        return false;
    case SelectionPattern::all:
        return true;
    case SelectionPattern::alternating:
        return (index & 1U) == 0U;
    case SelectionPattern::sparse:
        return index % 17U == 0U || (count != 0U && index + 1U == count);
    }
    return false;
}

[[nodiscard]] const char* pattern_name(const SelectionPattern pattern) noexcept {
    switch (pattern) {
    case SelectionPattern::none:
        return "none";
    case SelectionPattern::all:
        return "all";
    case SelectionPattern::alternating:
        return "alternating";
    case SelectionPattern::sparse:
        return "sparse";
    }
    return "unknown";
}

[[nodiscard]] std::vector<cuda::WavefrontStageOutcome>
make_outcomes(const std::uint32_t count, const SelectionPattern pattern,
              const cuda::WavefrontStageRoute selected_route) {
    const auto rejected_route = selected_route == cuda::WavefrontStageRoute::miss
                                    ? cuda::WavefrontStageRoute::ray
                                    : cuda::WavefrontStageRoute::miss;
    auto outcomes = std::vector<cuda::WavefrontStageOutcome>{};
    outcomes.reserve(count);
    for (auto index = std::uint32_t{}; index < count; ++index) {
        outcomes.push_back(cuda::WavefrontStageOutcome{
            .status = static_cast<std::uint32_t>(cuda::WavefrontStageStatus::success),
            .route = route_value(selected(pattern, index, count) ? selected_route : rejected_route),
            .path_slot = count - index - 1U,
            .detail = 0U,
        });
    }
    return outcomes;
}

[[nodiscard]] std::vector<std::uint32_t>
selected_slots(const std::span<const cuda::WavefrontStageOutcome> outcomes,
               const cuda::WavefrontStageRoute route) {
    auto slots = std::vector<std::uint32_t>{};
    for (const auto& outcome : outcomes) {
        if (outcome.route == route_value(route)) {
            slots.push_back(outcome.path_slot);
        }
    }
    return slots;
}

void initialize_queue(CudaWavefrontQueues& queues, const cuda::WavefrontStageRoute route,
                      const std::span<const std::uint32_t> prefix) {
    const auto view = queues.device_view();
    ASSERT_EQ(view.queue_count, cuda::CudaWavefrontQueueCount);
    ASSERT_GE(view.slot_stride, prefix.size());
    ASSERT_NE(view.headers, nullptr);
    if (view.slot_stride != 0U) {
        ASSERT_NE(view.path_slots, nullptr);
    }

    auto host_slots =
        std::vector<shared::PathSlot>(static_cast<std::size_t>(view.queue_count) * view.slot_stride,
                                      shared::PathSlot{.value = SentinelSlot});
    const auto queue_kind = route_value(route);
    const auto queue_offset = static_cast<std::size_t>(queue_kind) * view.slot_stride;
    for (auto index = std::size_t{}; index < prefix.size(); ++index) {
        host_slots[queue_offset + index].value = prefix[index];
    }
    if (!host_slots.empty()) {
        ASSERT_EQ(cudaMemcpy(view.path_slots, host_slots.data(),
                             host_slots.size() * sizeof(shared::PathSlot), cudaMemcpyHostToDevice),
                  cudaSuccess);
    }

    auto headers = std::array<shared::QueueHeader, QueueCount>{};
    ASSERT_EQ(cudaMemcpy(headers.data(), view.headers, sizeof(headers), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_LT(queue_kind, headers.size());
    headers[queue_kind].size = static_cast<std::uint32_t>(prefix.size());
    ASSERT_EQ(cudaMemcpy(view.headers, headers.data(), sizeof(headers), cudaMemcpyHostToDevice),
              cudaSuccess);
}

[[nodiscard]] RawQueueState download_raw_state(CudaWavefrontQueues& queues) {
    const auto view = queues.device_view();
    auto state = RawQueueState{};
    EXPECT_EQ(cudaMemcpy(state.headers.data(), view.headers, sizeof(state.headers),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    const auto slot_count = static_cast<std::size_t>(view.queue_count) * view.slot_stride;
    auto host_slots = std::vector<shared::PathSlot>(slot_count);
    if (slot_count != 0U) {
        EXPECT_EQ(cudaMemcpy(host_slots.data(), view.path_slots,
                             host_slots.size() * sizeof(shared::PathSlot), cudaMemcpyDeviceToHost),
                  cudaSuccess);
    }
    state.slots.reserve(host_slots.size());
    for (const auto slot : host_slots) {
        state.slots.push_back(slot.value);
    }
    return state;
}

[[nodiscard]] std::optional<CompactionObservation>
run_compaction(CudaWavefrontQueues& queues,
               const std::span<const cuda::WavefrontStageOutcome> outcomes,
               const cuda::WavefrontStageRoute route,
               const std::optional<std::size_t> scratch_capacity = std::nullopt) {
    if (outcomes.size() > std::numeric_limits<std::uint32_t>::max()) {
        ADD_FAILURE() << "The test outcome count exceeds the device domain.";
        return std::nullopt;
    }
    const auto input_count = static_cast<std::uint32_t>(outcomes.size());
    auto minimum_scratch_bytes = std::size_t{};
    const auto query_status =
        cuda::query_wavefront_queue_compaction_scratch_bytes(input_count, &minimum_scratch_bytes);
    if (query_status != static_cast<int>(cudaSuccess)) {
        ADD_FAILURE() << "Scratch query failed: "
                      << cudaGetErrorString(static_cast<cudaError_t>(query_status));
        return std::nullopt;
    }
    const auto scratch_bytes = scratch_capacity.value_or(minimum_scratch_bytes);
    if (scratch_bytes < minimum_scratch_bytes) {
        ADD_FAILURE() << "The supplied scratch capacity is below the queried minimum.";
        return std::nullopt;
    }

    auto device_outcomes = std::optional<cuda::DeviceBuffer<cuda::WavefrontStageOutcome>>{};
    if (!outcomes.empty()) {
        auto allocated = cuda::DeviceBuffer<cuda::WavefrontStageOutcome>::allocate(outcomes.size());
        if (!allocated) {
            ADD_FAILURE() << allocated.error().message;
            return std::nullopt;
        }
        device_outcomes.emplace(std::move(*allocated));
        const auto copy_status = cudaMemcpy(device_outcomes->data(), outcomes.data(),
                                            outcomes.size_bytes(), cudaMemcpyHostToDevice);
        if (copy_status != cudaSuccess) {
            ADD_FAILURE() << "Outcome upload failed: " << cudaGetErrorString(copy_status);
            return std::nullopt;
        }
    }

    auto scratch = std::optional<cuda::DeviceAllocation>{};
    if (scratch_bytes != 0U) {
        auto allocated = cuda::DeviceAllocation::allocate_bytes(scratch_bytes);
        if (!allocated) {
            ADD_FAILURE() << allocated.error().message;
            return std::nullopt;
        }
        scratch.emplace(std::move(*allocated));
    }
    auto device_result_allocation =
        cuda::DeviceBuffer<cuda::WavefrontQueueCompactionResult>::allocate(1U);
    if (!device_result_allocation) {
        ADD_FAILURE() << device_result_allocation.error().message;
        return std::nullopt;
    }
    auto device_result = std::move(*device_result_allocation);

    const auto launch_status = cuda::launch_wavefront_queue_compaction(
        queues.device_view(), device_outcomes ? device_outcomes->data() : nullptr, input_count,
        route_value(route), scratch ? scratch->data() : nullptr, scratch_bytes,
        device_result.data());
    if (launch_status != static_cast<int>(cudaSuccess)) {
        ADD_FAILURE() << "Compaction launch failed: "
                      << cudaGetErrorString(static_cast<cudaError_t>(launch_status));
        return std::nullopt;
    }
    const auto sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        ADD_FAILURE() << "Compaction synchronization failed: " << cudaGetErrorString(sync_status);
        return std::nullopt;
    }

    auto observation = CompactionObservation{.scratch_bytes = scratch_bytes};
    const auto copy_status = cudaMemcpy(&observation.result, device_result.data(),
                                        sizeof(observation.result), cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess) {
        ADD_FAILURE() << "Compaction result download failed: " << cudaGetErrorString(copy_status);
        return std::nullopt;
    }
    return observation;
}

void expect_result(const cuda::WavefrontQueueCompactionResult& result,
                   const cuda::WavefrontQueueCompactionStatus status,
                   const cuda::WavefrontStageRoute route, const std::uint32_t input_count,
                   const std::uint32_t initial_size, const std::uint32_t selected_count,
                   const std::uint32_t published_count, const std::uint32_t rejected_count) {
    EXPECT_EQ(result.status, static_cast<std::uint32_t>(status));
    EXPECT_EQ(result.queue_kind, route_value(route));
    EXPECT_EQ(result.route, route_value(route));
    EXPECT_EQ(result.input_count, input_count);
    EXPECT_EQ(result.initial_size, initial_size);
    EXPECT_EQ(result.selected_count, selected_count);
    EXPECT_EQ(result.published_count, published_count);
    EXPECT_EQ(result.rejected_count, rejected_count);
    for (const auto reserved : result.reserved) {
        EXPECT_EQ(reserved, 0U);
    }
}

void expect_canonical_header(const shared::QueueHeader& header, const std::uint32_t queue_kind,
                             const std::uint32_t capacity, const std::uint32_t size,
                             const std::uint32_t overflow_count,
                             const std::uint32_t rejected_count) {
    EXPECT_EQ(header.abi_major, shared::HostDeviceTransportAbiMajor);
    EXPECT_EQ(header.abi_minor, shared::HostDeviceTransportAbiMinor);
    EXPECT_EQ(header.struct_size, sizeof(shared::QueueHeader));
    EXPECT_EQ(header.queue_kind, queue_kind);
    EXPECT_EQ(header.capacity, capacity);
    EXPECT_EQ(header.size, size);
    EXPECT_EQ(header.overflow_count, overflow_count);
    EXPECT_EQ(header.rejected_count, rejected_count);
    EXPECT_EQ(header.reserved, 0U);
    EXPECT_EQ(shared::validate_queue_header(header), shared::QueueHeaderValidationStatus::valid);
}

class CudaWavefrontQueueCompactionTest : public testing::Test {
  protected:
    void SetUp() override {
        ASSERT_TRUE(select_test_device());
    }
};

TEST_F(CudaWavefrontQueueCompactionTest, PrefixScanIsStableAcrossBoundarySizesAndPatterns) {
    constexpr auto sizes =
        std::array<std::uint32_t, 9U>{0U, 1U, 31U, 32U, 33U, 255U, 256U, 257U, 65'536U};
    constexpr auto patterns = std::array{SelectionPattern::none, SelectionPattern::all,
                                         SelectionPattern::alternating, SelectionPattern::sparse};
    constexpr auto route = cuda::WavefrontStageRoute::ray;
    for (const auto size : sizes) {
        for (const auto pattern : patterns) {
            SCOPED_TRACE(testing::Message{} << "size=" << size
                                            << ", pattern=" << pattern_name(pattern));
            const auto capacity = std::max(size, std::uint32_t{1U});
            auto created = CudaWavefrontQueues::create(capacity);
            ASSERT_TRUE(created.has_value()) << created.error().message;
            auto queues = std::move(*created);
            ASSERT_NO_FATAL_FAILURE(initialize_queue(queues, route, {}));
            const auto outcomes = make_outcomes(size, pattern, route);
            const auto expected_selected = selected_slots(outcomes, route);

            const auto observation = run_compaction(queues, outcomes, route);
            ASSERT_TRUE(observation.has_value());
            expect_result(observation->result, cuda::WavefrontQueueCompactionStatus::success, route,
                          size, 0U, static_cast<std::uint32_t>(expected_selected.size()),
                          static_cast<std::uint32_t>(expected_selected.size()), 0U);

            const auto state = download_raw_state(queues);
            const auto target = route_value(route);
            expect_canonical_header(state.headers[target], target, capacity,
                                    static_cast<std::uint32_t>(expected_selected.size()), 0U, 0U);
            auto expected_slots = std::vector<std::uint32_t>(
                static_cast<std::size_t>(cuda::CudaWavefrontQueueCount) * capacity, SentinelSlot);
            const auto target_offset = static_cast<std::size_t>(target) * capacity;
            std::copy(expected_selected.begin(), expected_selected.end(),
                      expected_slots.begin() + static_cast<std::ptrdiff_t>(target_offset));
            EXPECT_EQ(state.slots, expected_slots);
        }
    }
}

TEST_F(CudaWavefrontQueueCompactionTest, RoutesEveryPublishedQueueKindExactly) {
    constexpr auto input_count = std::uint32_t{33U};
    constexpr auto capacity = std::uint32_t{33U};
    for (auto route_number = std::uint32_t{1U}; route_number < cuda::CudaWavefrontQueueCount;
         ++route_number) {
        SCOPED_TRACE(testing::Message{} << "route=" << route_number);
        const auto route = static_cast<cuda::WavefrontStageRoute>(route_number);
        auto created = CudaWavefrontQueues::create(capacity);
        ASSERT_TRUE(created.has_value()) << created.error().message;
        auto queues = std::move(*created);
        ASSERT_NO_FATAL_FAILURE(initialize_queue(queues, route, {}));
        const auto outcomes = make_outcomes(input_count, SelectionPattern::alternating, route);
        const auto expected = selected_slots(outcomes, route);
        const auto observation = run_compaction(queues, outcomes, route);
        ASSERT_TRUE(observation.has_value());
        expect_result(observation->result, cuda::WavefrontQueueCompactionStatus::success, route,
                      input_count, 0U, static_cast<std::uint32_t>(expected.size()),
                      static_cast<std::uint32_t>(expected.size()), 0U);
        const auto state = download_raw_state(queues);
        const auto offset = static_cast<std::size_t>(route_number) * capacity;
        EXPECT_TRUE(std::equal(expected.begin(), expected.end(),
                               state.slots.begin() + static_cast<std::ptrdiff_t>(offset)));
    }
}

TEST_F(CudaWavefrontQueueCompactionTest, AppendsStablyAndReplaysBitExactly) {
    constexpr auto route = cuda::WavefrontStageRoute::continuation;
    constexpr auto input_count = std::uint32_t{257U};
    constexpr auto prefix = std::array<std::uint32_t, 3U>{4U, 2U, 9U};
    const auto outcomes = make_outcomes(input_count, SelectionPattern::sparse, route);
    const auto selected = selected_slots(outcomes, route);
    const auto capacity = std::max(
        input_count, static_cast<std::uint32_t>(prefix.size() + selected.size() + std::size_t{5U}));

    auto first_created = CudaWavefrontQueues::create(capacity);
    auto second_created = CudaWavefrontQueues::create(capacity);
    ASSERT_TRUE(first_created.has_value()) << first_created.error().message;
    ASSERT_TRUE(second_created.has_value()) << second_created.error().message;
    auto first = std::move(*first_created);
    auto second = std::move(*second_created);
    ASSERT_NO_FATAL_FAILURE(initialize_queue(first, route, prefix));
    ASSERT_NO_FATAL_FAILURE(initialize_queue(second, route, prefix));

    const auto first_observation = run_compaction(first, outcomes, route);
    const auto second_observation = run_compaction(second, outcomes, route);
    ASSERT_TRUE(first_observation.has_value());
    ASSERT_TRUE(second_observation.has_value());
    EXPECT_EQ(0, std::memcmp(&first_observation->result, &second_observation->result,
                             sizeof(first_observation->result)));
    EXPECT_EQ(first_observation->scratch_bytes, second_observation->scratch_bytes);

    const auto first_state = download_raw_state(first);
    const auto second_state = download_raw_state(second);
    EXPECT_EQ(0, std::memcmp(first_state.headers.data(), second_state.headers.data(),
                             sizeof(first_state.headers)));
    EXPECT_EQ(first_state.slots, second_state.slots);
    expect_result(first_observation->result, cuda::WavefrontQueueCompactionStatus::success, route,
                  input_count, static_cast<std::uint32_t>(prefix.size()),
                  static_cast<std::uint32_t>(selected.size()),
                  static_cast<std::uint32_t>(selected.size()), 0U);

    auto expected_prefix = std::vector<std::uint32_t>(prefix.begin(), prefix.end());
    expected_prefix.insert(expected_prefix.end(), selected.begin(), selected.end());
    const auto target = route_value(route);
    const auto target_offset = static_cast<std::size_t>(target) * capacity;
    EXPECT_TRUE(std::equal(expected_prefix.begin(), expected_prefix.end(),
                           first_state.slots.begin() + static_cast<std::ptrdiff_t>(target_offset)));
}

TEST_F(CudaWavefrontQueueCompactionTest, CapacitySizedScratchServesASmallerActivePrefix) {
    constexpr auto route = cuda::WavefrontStageRoute::hit;
    constexpr auto input_count = std::uint32_t{33U};
    constexpr auto maximum_input_count = std::uint32_t{257U};
    auto maximum_scratch_bytes = std::size_t{};
    ASSERT_EQ(cuda::query_wavefront_queue_compaction_scratch_bytes(maximum_input_count,
                                                                   &maximum_scratch_bytes),
              static_cast<int>(cudaSuccess));

    auto created = CudaWavefrontQueues::create(input_count);
    ASSERT_TRUE(created.has_value()) << created.error().message;
    auto queues = std::move(*created);
    ASSERT_NO_FATAL_FAILURE(initialize_queue(queues, route, {}));
    const auto outcomes = make_outcomes(input_count, SelectionPattern::alternating, route);
    const auto expected = selected_slots(outcomes, route);

    const auto observation = run_compaction(queues, outcomes, route, maximum_scratch_bytes);
    ASSERT_TRUE(observation.has_value());
    EXPECT_EQ(observation->scratch_bytes, maximum_scratch_bytes);
    expect_result(observation->result, cuda::WavefrontQueueCompactionStatus::success, route,
                  input_count, 0U, static_cast<std::uint32_t>(expected.size()),
                  static_cast<std::uint32_t>(expected.size()), 0U);
}

TEST_F(CudaWavefrontQueueCompactionTest,
       CapacityExhaustionRejectsTheWholeAppendWithoutChangingSlots) {
    constexpr auto route = cuda::WavefrontStageRoute::shade;
    constexpr auto capacity = std::uint32_t{8U};
    constexpr auto prefix = std::array<std::uint32_t, 6U>{0U, 1U, 2U, 3U, 4U, 5U};
    const auto outcomes = make_outcomes(3U, SelectionPattern::all, route);
    auto created = CudaWavefrontQueues::create(capacity);
    ASSERT_TRUE(created.has_value()) << created.error().message;
    auto queues = std::move(*created);
    ASSERT_NO_FATAL_FAILURE(initialize_queue(queues, route, prefix));
    const auto before = download_raw_state(queues);

    const auto observation = run_compaction(queues, outcomes, route);
    ASSERT_TRUE(observation.has_value());
    expect_result(observation->result, cuda::WavefrontQueueCompactionStatus::capacity_exhausted,
                  route, 3U, static_cast<std::uint32_t>(prefix.size()), 3U, 0U, 3U);
    const auto after = download_raw_state(queues);
    EXPECT_EQ(after.slots, before.slots);
    const auto target = route_value(route);
    expect_canonical_header(after.headers[target], target, capacity,
                            static_cast<std::uint32_t>(prefix.size()), 3U, 3U);
    for (auto queue_kind = std::uint32_t{}; queue_kind < cuda::CudaWavefrontQueueCount;
         ++queue_kind) {
        if (queue_kind != target) {
            EXPECT_EQ(0, std::memcmp(&after.headers[queue_kind], &before.headers[queue_kind],
                                     sizeof(shared::QueueHeader)));
        }
    }
}

TEST_F(CudaWavefrontQueueCompactionTest, SelectsOnlySuccessfulOutcomesForTheRequestedRoute) {
    constexpr auto route = cuda::WavefrontStageRoute::ray;
    constexpr auto capacity = std::uint32_t{4U};
    const auto outcomes = std::array{
        cuda::WavefrontStageOutcome{
            .status = static_cast<std::uint32_t>(cuda::WavefrontStageStatus::success),
            .route = route_value(route),
            .path_slot = 0U,
        },
        cuda::WavefrontStageOutcome{
            .status = static_cast<std::uint32_t>(cuda::WavefrontStageStatus::invalid_contract),
            .route = route_value(route),
            .path_slot = 1U,
        },
        cuda::WavefrontStageOutcome{
            .status = static_cast<std::uint32_t>(cuda::WavefrontStageStatus::success),
            .route = route_value(cuda::WavefrontStageRoute::miss),
            .path_slot = 2U,
        },
        cuda::WavefrontStageOutcome{
            .status = static_cast<std::uint32_t>(cuda::WavefrontStageStatus::success),
            .route = route_value(route),
            .path_slot = 3U,
        },
    };
    auto created = CudaWavefrontQueues::create(capacity);
    ASSERT_TRUE(created.has_value()) << created.error().message;
    auto queues = std::move(*created);
    ASSERT_NO_FATAL_FAILURE(initialize_queue(queues, route, {}));

    const auto observation = run_compaction(queues, outcomes, route);
    ASSERT_TRUE(observation.has_value());
    expect_result(observation->result, cuda::WavefrontQueueCompactionStatus::success, route,
                  capacity, 0U, 2U, 2U, 0U);
    const auto state = download_raw_state(queues);
    const auto target = route_value(route);
    expect_canonical_header(state.headers[target], target, capacity, 2U, 0U, 0U);
    const auto target_offset = static_cast<std::size_t>(target) * capacity;
    EXPECT_EQ(state.slots[target_offset], 0U);
    EXPECT_EQ(state.slots[target_offset + 1U], 3U);
}

TEST_F(CudaWavefrontQueueCompactionTest,
       InvalidSelectedPathSlotRejectsTheAppendWithoutChangingTheQueue) {
    constexpr auto route = cuda::WavefrontStageRoute::ray;
    constexpr auto capacity = std::uint32_t{4U};
    constexpr auto prefix = std::array<std::uint32_t, 1U>{1U};
    const auto outcomes = std::array{
        cuda::WavefrontStageOutcome{
            .status = static_cast<std::uint32_t>(cuda::WavefrontStageStatus::success),
            .route = route_value(route),
            .path_slot = 2U,
        },
        cuda::WavefrontStageOutcome{
            .status = static_cast<std::uint32_t>(cuda::WavefrontStageStatus::success),
            .route = route_value(route),
            .path_slot = capacity,
        },
    };
    auto created = CudaWavefrontQueues::create(capacity);
    ASSERT_TRUE(created.has_value()) << created.error().message;
    auto queues = std::move(*created);
    ASSERT_NO_FATAL_FAILURE(initialize_queue(queues, route, prefix));
    const auto before = download_raw_state(queues);

    const auto observation = run_compaction(queues, outcomes, route);
    ASSERT_TRUE(observation.has_value());
    expect_result(observation->result, cuda::WavefrontQueueCompactionStatus::invalid_contract,
                  route, static_cast<std::uint32_t>(outcomes.size()),
                  static_cast<std::uint32_t>(prefix.size()),
                  static_cast<std::uint32_t>(outcomes.size()), 0U, 0U);
    const auto after = download_raw_state(queues);
    EXPECT_EQ(after.slots, before.slots);
    EXPECT_EQ(0, std::memcmp(after.headers.data(), before.headers.data(), sizeof(before.headers)));
}

TEST_F(CudaWavefrontQueueCompactionTest,
       RejectsInsufficientScratchAndInvalidRoutesWithoutMutation) {
    constexpr auto route = cuda::WavefrontStageRoute::ray;
    constexpr auto input_count = std::uint32_t{257U};
    constexpr auto capacity = std::uint32_t{257U};
    constexpr auto prefix = std::array<std::uint32_t, 2U>{7U, 11U};
    const auto outcomes = make_outcomes(input_count, SelectionPattern::alternating, route);

    auto created = CudaWavefrontQueues::create(capacity);
    ASSERT_TRUE(created.has_value()) << created.error().message;
    auto queues = std::move(*created);
    ASSERT_NO_FATAL_FAILURE(initialize_queue(queues, route, prefix));
    const auto before = download_raw_state(queues);

    auto scratch_bytes = std::size_t{};
    ASSERT_EQ(cuda::query_wavefront_queue_compaction_scratch_bytes(input_count, &scratch_bytes),
              static_cast<int>(cudaSuccess));
    ASSERT_GT(scratch_bytes, 0U);
    auto scratch_result = cuda::DeviceAllocation::allocate_bytes(scratch_bytes);
    auto outcomes_result =
        cuda::DeviceBuffer<cuda::WavefrontStageOutcome>::allocate(outcomes.size());
    auto result_result = cuda::DeviceBuffer<cuda::WavefrontQueueCompactionResult>::allocate(1U);
    ASSERT_TRUE(scratch_result.has_value()) << scratch_result.error().message;
    ASSERT_TRUE(outcomes_result.has_value()) << outcomes_result.error().message;
    ASSERT_TRUE(result_result.has_value()) << result_result.error().message;
    auto scratch = std::move(*scratch_result);
    auto device_outcomes = std::move(*outcomes_result);
    auto device_result = std::move(*result_result);
    ASSERT_EQ(cudaMemcpy(device_outcomes.data(), outcomes.data(),
                         outcomes.size() * sizeof(outcomes[0]), cudaMemcpyHostToDevice),
              cudaSuccess);

    const auto short_scratch_status = cuda::launch_wavefront_queue_compaction(
        queues.device_view(), device_outcomes.data(), input_count, route_value(route),
        scratch.data(), scratch_bytes - 1U, device_result.data());
    EXPECT_EQ(short_scratch_status, static_cast<int>(cudaErrorInvalidValue));
    EXPECT_EQ(download_raw_state(queues).slots, before.slots);
    EXPECT_EQ(0, std::memcmp(download_raw_state(queues).headers.data(), before.headers.data(),
                             sizeof(before.headers)));

    for (const auto invalid_route :
         {std::uint32_t{0U}, route_value(cuda::WavefrontStageRoute::terminated)}) {
        SCOPED_TRACE(testing::Message{} << "invalid route=" << invalid_route);
        const auto status = cuda::launch_wavefront_queue_compaction(
            queues.device_view(), device_outcomes.data(), input_count, invalid_route,
            scratch.data(), scratch_bytes, device_result.data());
        EXPECT_EQ(status, static_cast<int>(cudaErrorInvalidValue));
        const auto after = download_raw_state(queues);
        EXPECT_EQ(after.slots, before.slots);
        EXPECT_EQ(0,
                  std::memcmp(after.headers.data(), before.headers.data(), sizeof(before.headers)));
    }
}

} // namespace
} // namespace blackframe::engine
