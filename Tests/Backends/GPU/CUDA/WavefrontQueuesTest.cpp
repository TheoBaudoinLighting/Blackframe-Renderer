#include <Blackframe/Backends/GPU/CUDA/WavefrontQueues.hpp>
#include <Blackframe/Core/Status.hpp>
#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <Blackframe/XPU/CUDA/WavefrontQueueKernel.hpp>
#include <Blackframe/XPU/Shared/TransportAbi.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

namespace cuda = xpu::cuda;
namespace shared = xpu::shared;

using cuda::WavefrontQueueDevicePush;
using cuda::WavefrontQueueDevicePushStatus;

constexpr auto QueueCount = static_cast<std::size_t>(cuda::CudaWavefrontQueueCount);

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

[[nodiscard]] constexpr std::uint32_t encoded_slot(const std::uint32_t queue_kind,
                                                   const std::uint32_t lane) noexcept {
    return (queue_kind << 28U) | lane;
}

[[nodiscard]] std::vector<WavefrontQueueDevicePush>
make_requests(const std::uint32_t queue_kind, const std::uint32_t request_count,
              const std::uint32_t first_lane = 0U) {
    auto requests = std::vector<WavefrontQueueDevicePush>{};
    requests.reserve(request_count);
    for (auto lane = std::uint32_t{}; lane < request_count; ++lane) {
        requests.push_back(WavefrontQueueDevicePush{
            .queue_kind = queue_kind,
            .slot = shared::PathSlot{.value = encoded_slot(queue_kind, first_lane + lane)},
        });
    }
    return requests;
}

[[nodiscard]] std::vector<WavefrontQueueDevicePush>
make_interleaved_requests(const std::uint32_t requests_per_queue,
                          const std::uint32_t first_lane = 0U) {
    auto requests = std::vector<WavefrontQueueDevicePush>{};
    requests.reserve(QueueCount * requests_per_queue);
    for (auto lane = std::uint32_t{}; lane < requests_per_queue; ++lane) {
        for (auto queue_kind = std::uint32_t{}; queue_kind < QueueCount; ++queue_kind) {
            requests.push_back(WavefrontQueueDevicePush{
                .queue_kind = queue_kind,
                .slot = shared::PathSlot{.value = encoded_slot(queue_kind, first_lane + lane)},
            });
        }
    }
    return requests;
}

void launch_pushes(const cuda::WavefrontQueueDeviceSoa queues,
                   const std::span<const WavefrontQueueDevicePush> requests,
                   std::vector<WavefrontQueueDevicePushStatus>& outcomes) {
    ASSERT_LE(requests.size(), std::numeric_limits<std::uint32_t>::max());
    outcomes.assign(requests.size(), WavefrontQueueDevicePushStatus::invalid_contract);
    if (requests.empty()) {
        ASSERT_EQ(cuda::launch_wavefront_queue_pushes(queues, nullptr, 0U, nullptr),
                  static_cast<int>(cudaSuccess));
        return;
    }

    auto device_requests_result =
        cuda::DeviceBuffer<WavefrontQueueDevicePush>::allocate(requests.size());
    ASSERT_TRUE(device_requests_result.has_value()) << device_requests_result.error().message;
    auto device_requests = std::move(*device_requests_result);
    auto device_outcomes_result =
        cuda::DeviceBuffer<WavefrontQueueDevicePushStatus>::allocate(outcomes.size());
    ASSERT_TRUE(device_outcomes_result.has_value()) << device_outcomes_result.error().message;
    auto device_outcomes = std::move(*device_outcomes_result);

    ASSERT_EQ(cudaMemcpy(device_requests.data(), requests.data(), requests.size_bytes(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    const auto launch_status = cuda::launch_wavefront_queue_pushes(
        queues, device_requests.data(), static_cast<std::uint32_t>(requests.size()),
        device_outcomes.data());
    ASSERT_EQ(launch_status, static_cast<int>(cudaSuccess))
        << cudaGetErrorString(static_cast<cudaError_t>(launch_status));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(outcomes.data(), device_outcomes.data(), device_outcomes.size_bytes(),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
}

void upload_slots(const cuda::WavefrontQueueDeviceSoa queues,
                  const std::span<const shared::PathSlot> slots) {
    const auto expected_size = static_cast<std::size_t>(queues.queue_count) * queues.slot_stride;
    ASSERT_EQ(slots.size(), expected_size);
    if (slots.empty()) {
        EXPECT_EQ(queues.path_slots, nullptr);
        return;
    }
    ASSERT_NE(queues.path_slots, nullptr);
    ASSERT_EQ(
        cudaMemcpy(queues.path_slots, slots.data(), slots.size_bytes(), cudaMemcpyHostToDevice),
        cudaSuccess);
}

[[nodiscard]] std::vector<shared::PathSlot>
download_slots(const cuda::WavefrontQueueDeviceSoa queues) {
    const auto slot_count = static_cast<std::size_t>(queues.queue_count) * queues.slot_stride;
    auto slots = std::vector<shared::PathSlot>(slot_count);
    if (slots.empty()) {
        return slots;
    }
    EXPECT_NE(queues.path_slots, nullptr);
    if (queues.path_slots != nullptr) {
        EXPECT_EQ(cudaMemcpy(slots.data(), queues.path_slots,
                             slots.size() * sizeof(shared::PathSlot), cudaMemcpyDeviceToHost),
                  cudaSuccess);
    }
    return slots;
}

[[nodiscard]] std::array<shared::QueueHeader, QueueCount>
download_headers(const cuda::WavefrontQueueDeviceSoa queues) {
    auto headers = std::array<shared::QueueHeader, QueueCount>{};
    EXPECT_NE(queues.headers, nullptr);
    if (queues.headers != nullptr) {
        EXPECT_EQ(
            cudaMemcpy(headers.data(), queues.headers, sizeof(headers), cudaMemcpyDeviceToHost),
            cudaSuccess);
    }
    return headers;
}

void expect_header(const shared::QueueHeader& header, const std::uint32_t queue_kind,
                   const std::uint32_t capacity, const std::uint32_t size,
                   const std::uint32_t overflow_count, const std::uint32_t rejected_count) {
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

[[nodiscard]] std::vector<std::uint32_t>
pushed_values(const std::span<const WavefrontQueueDevicePush> requests,
              const std::span<const WavefrontQueueDevicePushStatus> outcomes) {
    EXPECT_EQ(requests.size(), outcomes.size());
    auto values = std::vector<std::uint32_t>{};
    for (auto index = std::size_t{}; index < std::min(requests.size(), outcomes.size()); ++index) {
        if (outcomes[index] == WavefrontQueueDevicePushStatus::pushed) {
            values.push_back(requests[index].slot.value);
        }
    }
    std::sort(values.begin(), values.end());
    return values;
}

[[nodiscard]] std::vector<std::uint32_t>
column_values(const std::span<const shared::PathSlot> slots, const std::uint32_t queue_kind,
              const std::uint32_t capacity) {
    const auto first = static_cast<std::size_t>(queue_kind) * capacity;
    auto values = std::vector<std::uint32_t>{};
    values.reserve(capacity);
    for (auto index = std::uint32_t{}; index < capacity; ++index) {
        values.push_back(slots[first + index].value);
    }
    std::sort(values.begin(), values.end());
    return values;
}

[[nodiscard]] std::size_t
count_status(const std::span<const WavefrontQueueDevicePushStatus> outcomes,
             const WavefrontQueueDevicePushStatus expected) {
    return static_cast<std::size_t>(std::count(outcomes.begin(), outcomes.end(), expected));
}

TEST(CudaWavefrontQueuesTest, KeepsSevenSoaColumnsIndependentDuringForcedOverflow) {
    ASSERT_TRUE(select_test_device());
    constexpr auto capacity = std::uint32_t{37U};
    constexpr auto requests_per_queue = std::uint32_t{1'025U};
    constexpr auto sentinel_prefix = std::uint32_t{0xE0000000U};

    auto created = CudaWavefrontQueues::create(capacity);
    ASSERT_TRUE(created.has_value()) << created.error().message;
    auto queues = std::move(*created);
    const auto view = queues.device_view();
    ASSERT_EQ(view.queue_count, QueueCount);
    ASSERT_EQ(view.slot_stride, capacity);

    auto expected_slots = std::vector<shared::PathSlot>(QueueCount * capacity);
    for (auto index = std::size_t{}; index < expected_slots.size(); ++index) {
        expected_slots[index].value = sentinel_prefix | static_cast<std::uint32_t>(index);
    }
    upload_slots(view, expected_slots);

    for (auto queue_kind = std::uint32_t{}; queue_kind < QueueCount; ++queue_kind) {
        SCOPED_TRACE(queue_kind);
        const auto requests = make_requests(queue_kind, requests_per_queue);
        auto outcomes = std::vector<WavefrontQueueDevicePushStatus>{};
        launch_pushes(view, requests, outcomes);

        EXPECT_EQ(count_status(outcomes, WavefrontQueueDevicePushStatus::pushed), capacity);
        EXPECT_EQ(count_status(outcomes, WavefrontQueueDevicePushStatus::capacity_exhausted),
                  requests_per_queue - capacity);
        EXPECT_EQ(count_status(outcomes, WavefrontQueueDevicePushStatus::invalid_contract), 0U);

        const auto actual_slots = download_slots(view);
        ASSERT_EQ(actual_slots.size(), expected_slots.size());
        for (auto other_kind = std::uint32_t{}; other_kind < QueueCount; ++other_kind) {
            const auto first = static_cast<std::size_t>(other_kind) * capacity;
            if (other_kind == queue_kind) {
                EXPECT_EQ(column_values(actual_slots, other_kind, capacity),
                          pushed_values(requests, outcomes));
                for (auto index = std::uint32_t{}; index < capacity; ++index) {
                    expected_slots[first + index] = actual_slots[first + index];
                }
                continue;
            }
            for (auto index = std::uint32_t{}; index < capacity; ++index) {
                EXPECT_EQ(actual_slots[first + index].value, expected_slots[first + index].value)
                    << "queue column " << other_kind << ", slot " << index;
            }
        }

        const auto headers = download_headers(view);
        for (auto other_kind = std::uint32_t{}; other_kind < QueueCount; ++other_kind) {
            const auto populated = other_kind <= queue_kind;
            expect_header(headers[other_kind], other_kind, capacity, populated ? capacity : 0U,
                          populated ? requests_per_queue - capacity : 0U,
                          populated ? requests_per_queue - capacity : 0U);
        }
    }

    const auto snapshots = queues.download();
    ASSERT_TRUE(snapshots.has_value()) << snapshots.error().message;
    for (auto queue_kind = std::uint32_t{}; queue_kind < QueueCount; ++queue_kind) {
        const auto& snapshot = (*snapshots)[queue_kind];
        EXPECT_EQ(static_cast<std::uint32_t>(snapshot.counters.kind), queue_kind);
        EXPECT_EQ(snapshot.counters.capacity, capacity);
        EXPECT_EQ(snapshot.counters.size, capacity);
        EXPECT_EQ(snapshot.counters.overflow_count, requests_per_queue - capacity);
        EXPECT_EQ(snapshot.counters.rejected_count, requests_per_queue - capacity);
        auto values = std::vector<std::uint32_t>{};
        values.reserve(snapshot.entries.size());
        for (const auto entry : snapshot.entries) {
            values.push_back(entry.value);
        }
        std::sort(values.begin(), values.end());
        EXPECT_EQ(values, column_values(expected_slots, queue_kind, capacity));
    }
}

TEST(CudaWavefrontQueuesTest, RepeatedOverflowPreservesEveryStoredSlotAndSentinel) {
    ASSERT_TRUE(select_test_device());
    constexpr auto capacity = std::uint32_t{19U};
    constexpr auto queue_kind = std::uint32_t{5U};
    constexpr auto first_request_count = std::uint32_t{701U};
    constexpr auto second_request_count = std::uint32_t{513U};

    auto created = CudaWavefrontQueues::create(capacity);
    ASSERT_TRUE(created.has_value()) << created.error().message;
    auto queues = std::move(*created);
    const auto view = queues.device_view();
    auto sentinels = std::vector<shared::PathSlot>(QueueCount * capacity);
    for (auto index = std::size_t{}; index < sentinels.size(); ++index) {
        sentinels[index].value = 0xD0000000U | static_cast<std::uint32_t>(index);
    }
    upload_slots(view, sentinels);

    const auto first_requests = make_requests(queue_kind, first_request_count);
    auto first_outcomes = std::vector<WavefrontQueueDevicePushStatus>{};
    launch_pushes(view, first_requests, first_outcomes);
    EXPECT_EQ(count_status(first_outcomes, WavefrontQueueDevicePushStatus::pushed), capacity);
    EXPECT_EQ(count_status(first_outcomes, WavefrontQueueDevicePushStatus::capacity_exhausted),
              first_request_count - capacity);
    const auto before_repeated_overflow = download_slots(view);

    const auto second_requests =
        make_requests(queue_kind, second_request_count, first_request_count);
    auto second_outcomes = std::vector<WavefrontQueueDevicePushStatus>{};
    launch_pushes(view, second_requests, second_outcomes);
    EXPECT_EQ(count_status(second_outcomes, WavefrontQueueDevicePushStatus::pushed), 0U);
    EXPECT_EQ(count_status(second_outcomes, WavefrontQueueDevicePushStatus::capacity_exhausted),
              second_request_count);
    EXPECT_EQ(count_status(second_outcomes, WavefrontQueueDevicePushStatus::invalid_contract), 0U);

    const auto after_repeated_overflow = download_slots(view);
    ASSERT_EQ(after_repeated_overflow.size(), before_repeated_overflow.size());
    for (auto index = std::size_t{}; index < after_repeated_overflow.size(); ++index) {
        EXPECT_EQ(after_repeated_overflow[index].value, before_repeated_overflow[index].value)
            << "slot " << index;
    }
    const auto headers = download_headers(view);
    for (auto other_kind = std::uint32_t{}; other_kind < QueueCount; ++other_kind) {
        if (other_kind == queue_kind) {
            expect_header(headers[other_kind], other_kind, capacity, capacity,
                          first_request_count - capacity + second_request_count,
                          first_request_count - capacity + second_request_count);
        } else {
            expect_header(headers[other_kind], other_kind, capacity, 0U, 0U, 0U);
        }
    }
}

TEST(CudaWavefrontQueuesTest, ForcedOverflowPreservesOuterGuardsAndSaturatesCounters) {
    ASSERT_TRUE(select_test_device());
    constexpr auto capacity = std::uint32_t{257U};
    constexpr auto queue_kind = std::uint32_t{6U};
    constexpr auto request_count = std::uint32_t{4'099U};
    constexpr auto repeated_request_count = std::uint32_t{513U};
    constexpr auto slot_guard = std::uint32_t{0xA55AA55AU};
    constexpr auto outcome_guard =
        static_cast<WavefrontQueueDevicePushStatus>(std::uint32_t{0xC33CC33CU});

    auto host_headers = std::vector<shared::QueueHeader>(QueueCount + 2U);
    const auto header_guard = shared::QueueHeader{
        .abi_major = 0xA55AU,
        .abi_minor = 0x5AA5U,
        .struct_size = 0xA55AA55AU,
        .queue_kind = 0x5AA55AA5U,
        .capacity = 0xA5A5A5A5U,
        .size = 0x5A5A5A5AU,
        .overflow_count = 0xC3C3C3C3U,
        .rejected_count = 0x3C3C3C3CU,
        .reserved = 0xF00DF00DU,
    };
    host_headers.front() = header_guard;
    host_headers.back() = header_guard;
    for (auto kind = std::uint32_t{}; kind < QueueCount; ++kind) {
        host_headers[static_cast<std::size_t>(kind) + 1U] = shared::QueueHeader{
            .abi_major = shared::HostDeviceTransportAbiMajor,
            .abi_minor = shared::HostDeviceTransportAbiMinor,
            .struct_size = sizeof(shared::QueueHeader),
            .queue_kind = kind,
            .capacity = capacity,
            .size = 0U,
            .overflow_count = 0U,
            .rejected_count = 0U,
            .reserved = 0U,
        };
    }

    const auto active_slot_count = QueueCount * capacity;
    auto host_slots = std::vector<shared::PathSlot>(active_slot_count + 2U);
    for (auto index = std::size_t{}; index < host_slots.size(); ++index) {
        host_slots[index].value = slot_guard ^ static_cast<std::uint32_t>(index);
    }
    const auto slots_before = host_slots;

    auto device_headers_result =
        cuda::DeviceBuffer<shared::QueueHeader>::allocate(host_headers.size());
    ASSERT_TRUE(device_headers_result.has_value()) << device_headers_result.error().message;
    auto device_headers = std::move(*device_headers_result);
    auto device_slots_result = cuda::DeviceBuffer<shared::PathSlot>::allocate(host_slots.size());
    ASSERT_TRUE(device_slots_result.has_value()) << device_slots_result.error().message;
    auto device_slots = std::move(*device_slots_result);
    ASSERT_EQ(cudaMemcpy(device_headers.data(), host_headers.data(),
                         host_headers.size() * sizeof(shared::QueueHeader), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_slots.data(), host_slots.data(),
                         host_slots.size() * sizeof(shared::PathSlot), cudaMemcpyHostToDevice),
              cudaSuccess);

    const auto requests = make_requests(queue_kind, request_count);
    auto device_requests_result =
        cuda::DeviceBuffer<WavefrontQueueDevicePush>::allocate(requests.size());
    ASSERT_TRUE(device_requests_result.has_value()) << device_requests_result.error().message;
    auto device_requests = std::move(*device_requests_result);
    ASSERT_EQ(cudaMemcpy(device_requests.data(), requests.data(),
                         requests.size() * sizeof(WavefrontQueueDevicePush),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    auto host_outcomes =
        std::vector<WavefrontQueueDevicePushStatus>(request_count + 2U, outcome_guard);
    auto device_outcomes_result =
        cuda::DeviceBuffer<WavefrontQueueDevicePushStatus>::allocate(host_outcomes.size());
    ASSERT_TRUE(device_outcomes_result.has_value()) << device_outcomes_result.error().message;
    auto device_outcomes = std::move(*device_outcomes_result);
    ASSERT_EQ(cudaMemcpy(device_outcomes.data(), host_outcomes.data(),
                         host_outcomes.size() * sizeof(WavefrontQueueDevicePushStatus),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    const auto guarded_view = cuda::WavefrontQueueDeviceSoa{
        .headers = device_headers.data() + 1U,
        .path_slots = device_slots.data() + 1U,
        .queue_count = cuda::CudaWavefrontQueueCount,
        .slot_stride = capacity,
    };
    ASSERT_EQ(cuda::launch_wavefront_queue_pushes(guarded_view, device_requests.data(),
                                                  request_count, device_outcomes.data() + 1U),
              static_cast<int>(cudaSuccess));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_headers.data(), device_headers.data(),
                         host_headers.size() * sizeof(shared::QueueHeader), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_slots.data(), device_slots.data(),
                         host_slots.size() * sizeof(shared::PathSlot), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_outcomes.data(), device_outcomes.data(),
                         host_outcomes.size() * sizeof(WavefrontQueueDevicePushStatus),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    EXPECT_EQ(std::memcmp(&host_headers.front(), &header_guard, sizeof(header_guard)), 0);
    EXPECT_EQ(std::memcmp(&host_headers.back(), &header_guard, sizeof(header_guard)), 0);
    EXPECT_EQ(host_slots.front().value, slots_before.front().value);
    EXPECT_EQ(host_slots.back().value, slots_before.back().value);
    EXPECT_EQ(host_outcomes.front(), outcome_guard);
    EXPECT_EQ(host_outcomes.back(), outcome_guard);
    const auto active_outcomes = std::span{host_outcomes}.subspan(1U, request_count);
    EXPECT_EQ(count_status(active_outcomes, WavefrontQueueDevicePushStatus::pushed), capacity);
    EXPECT_EQ(count_status(active_outcomes, WavefrontQueueDevicePushStatus::capacity_exhausted),
              request_count - capacity);
    EXPECT_EQ(count_status(active_outcomes, WavefrontQueueDevicePushStatus::invalid_contract), 0U);
    for (auto kind = std::uint32_t{}; kind < QueueCount; ++kind) {
        if (kind == queue_kind) {
            expect_header(host_headers[static_cast<std::size_t>(kind) + 1U], kind, capacity,
                          capacity, request_count - capacity, request_count - capacity);
            continue;
        }
        expect_header(host_headers[static_cast<std::size_t>(kind) + 1U], kind, capacity, 0U, 0U,
                      0U);
        const auto first = 1U + static_cast<std::size_t>(kind) * capacity;
        for (auto index = std::uint32_t{}; index < capacity; ++index) {
            EXPECT_EQ(host_slots[first + index].value, slots_before[first + index].value);
        }
    }

    const auto slots_after_fill = host_slots;
    auto saturated_header = host_headers[static_cast<std::size_t>(queue_kind) + 1U];
    saturated_header.overflow_count = std::numeric_limits<std::uint32_t>::max() - 1U;
    saturated_header.rejected_count = std::numeric_limits<std::uint32_t>::max() - 1U;
    ASSERT_EQ(cudaMemcpy(device_headers.data() + 1U + queue_kind, &saturated_header,
                         sizeof(saturated_header), cudaMemcpyHostToDevice),
              cudaSuccess);
    std::fill(host_outcomes.begin(), host_outcomes.end(), outcome_guard);
    ASSERT_EQ(cudaMemcpy(device_outcomes.data(), host_outcomes.data(),
                         host_outcomes.size() * sizeof(WavefrontQueueDevicePushStatus),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cuda::launch_wavefront_queue_pushes(guarded_view, device_requests.data(),
                                                  repeated_request_count,
                                                  device_outcomes.data() + 1U),
              static_cast<int>(cudaSuccess));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_headers.data(), device_headers.data(),
                         host_headers.size() * sizeof(shared::QueueHeader), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_slots.data(), device_slots.data(),
                         host_slots.size() * sizeof(shared::PathSlot), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_outcomes.data(), device_outcomes.data(),
                         host_outcomes.size() * sizeof(WavefrontQueueDevicePushStatus),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    ASSERT_EQ(host_slots.size(), slots_after_fill.size());
    for (auto index = std::size_t{}; index < host_slots.size(); ++index) {
        EXPECT_EQ(host_slots[index].value, slots_after_fill[index].value) << "slot " << index;
    }
    EXPECT_EQ(std::memcmp(&host_headers.front(), &header_guard, sizeof(header_guard)), 0);
    EXPECT_EQ(std::memcmp(&host_headers.back(), &header_guard, sizeof(header_guard)), 0);
    expect_header(host_headers[static_cast<std::size_t>(queue_kind) + 1U], queue_kind, capacity,
                  capacity, std::numeric_limits<std::uint32_t>::max(),
                  std::numeric_limits<std::uint32_t>::max());
    EXPECT_EQ(host_outcomes.front(), outcome_guard);
    for (auto index = std::size_t{1U}; index <= repeated_request_count; ++index) {
        EXPECT_EQ(host_outcomes[index], WavefrontQueueDevicePushStatus::capacity_exhausted);
    }
    for (auto index = static_cast<std::size_t>(repeated_request_count) + 1U;
         index < host_outcomes.size(); ++index) {
        EXPECT_EQ(host_outcomes[index], outcome_guard);
    }
}

TEST(CudaWavefrontQueuesTest, CountsZeroCapacityRejectionsWithoutQueueStorage) {
    ASSERT_TRUE(select_test_device());
    constexpr auto requests_per_queue = std::uint32_t{513U};

    auto created = CudaWavefrontQueues::create(0U);
    ASSERT_TRUE(created.has_value()) << created.error().message;
    auto queues = std::move(*created);
    const auto view = queues.device_view();
    EXPECT_EQ(view.slot_stride, 0U);
    EXPECT_EQ(view.path_slots, nullptr);

    const auto requests = make_interleaved_requests(requests_per_queue);
    auto outcomes = std::vector<WavefrontQueueDevicePushStatus>{};
    launch_pushes(view, requests, outcomes);
    EXPECT_EQ(count_status(outcomes, WavefrontQueueDevicePushStatus::pushed), 0U);
    EXPECT_EQ(count_status(outcomes, WavefrontQueueDevicePushStatus::capacity_exhausted),
              requests.size());
    EXPECT_EQ(count_status(outcomes, WavefrontQueueDevicePushStatus::invalid_contract), 0U);
    EXPECT_TRUE(download_slots(view).empty());

    const auto headers = download_headers(view);
    for (auto queue_kind = std::uint32_t{}; queue_kind < QueueCount; ++queue_kind) {
        expect_header(headers[queue_kind], queue_kind, 0U, 0U, requests_per_queue,
                      requests_per_queue);
    }
    const auto refused_reset = queues.reset(CudaWavefrontQueueResetPolicy::require_no_overflow);
    ASSERT_FALSE(refused_reset.has_value());
    EXPECT_EQ(refused_reset.error().code, core::StatusCode::resource_exhausted);
    const auto preserved_headers = download_headers(view);
    for (auto queue_kind = std::uint32_t{}; queue_kind < QueueCount; ++queue_kind) {
        expect_header(preserved_headers[queue_kind], queue_kind, 0U, 0U, requests_per_queue,
                      requests_per_queue);
    }
    const auto reset_status = queues.reset(CudaWavefrontQueueResetPolicy::acknowledge_overflow);
    ASSERT_TRUE(reset_status.has_value()) << reset_status.error().message;
    const auto reset_headers = download_headers(view);
    for (auto queue_kind = std::uint32_t{}; queue_kind < QueueCount; ++queue_kind) {
        expect_header(reset_headers[queue_kind], queue_kind, 0U, 0U, 0U, 0U);
    }
}

TEST(CudaWavefrontQueuesTest, ResetClearsCountersAndReusesTheSameColumns) {
    ASSERT_TRUE(select_test_device());
    constexpr auto capacity = std::uint32_t{11U};
    constexpr auto initial_requests_per_queue = std::uint32_t{17U};
    constexpr auto reused_requests_per_queue = std::uint32_t{5U};

    auto created = CudaWavefrontQueues::create(capacity);
    ASSERT_TRUE(created.has_value()) << created.error().message;
    auto queues = std::move(*created);
    const auto initial_view = queues.device_view();

    const auto initial_requests = make_interleaved_requests(initial_requests_per_queue);
    auto initial_outcomes = std::vector<WavefrontQueueDevicePushStatus>{};
    launch_pushes(initial_view, initial_requests, initial_outcomes);
    EXPECT_EQ(count_status(initial_outcomes, WavefrontQueueDevicePushStatus::pushed),
              QueueCount * capacity);
    EXPECT_EQ(count_status(initial_outcomes, WavefrontQueueDevicePushStatus::capacity_exhausted),
              QueueCount * (initial_requests_per_queue - capacity));

    const auto refused_reset = queues.reset(CudaWavefrontQueueResetPolicy::require_no_overflow);
    ASSERT_FALSE(refused_reset.has_value());
    EXPECT_EQ(refused_reset.error().code, core::StatusCode::resource_exhausted);
    const auto preserved_headers = download_headers(initial_view);
    for (auto queue_kind = std::uint32_t{}; queue_kind < QueueCount; ++queue_kind) {
        expect_header(preserved_headers[queue_kind], queue_kind, capacity, capacity,
                      initial_requests_per_queue - capacity, initial_requests_per_queue - capacity);
    }
    const auto reset_status = queues.reset(CudaWavefrontQueueResetPolicy::acknowledge_overflow);
    ASSERT_TRUE(reset_status.has_value()) << reset_status.error().message;
    const auto reused_view = queues.device_view();
    EXPECT_EQ(reused_view.headers, initial_view.headers);
    EXPECT_EQ(reused_view.path_slots, initial_view.path_slots);
    EXPECT_EQ(reused_view.slot_stride, initial_view.slot_stride);
    const auto reset_snapshots = queues.download();
    ASSERT_TRUE(reset_snapshots.has_value()) << reset_snapshots.error().message;
    for (auto queue_kind = std::uint32_t{}; queue_kind < QueueCount; ++queue_kind) {
        const auto& snapshot = (*reset_snapshots)[queue_kind];
        EXPECT_EQ(snapshot.counters.size, 0U);
        EXPECT_EQ(snapshot.counters.overflow_count, 0U);
        EXPECT_EQ(snapshot.counters.rejected_count, 0U);
        EXPECT_TRUE(snapshot.entries.empty());
    }

    const auto reused_requests = make_interleaved_requests(reused_requests_per_queue, 100U);
    auto reused_outcomes = std::vector<WavefrontQueueDevicePushStatus>{};
    launch_pushes(reused_view, reused_requests, reused_outcomes);
    EXPECT_EQ(count_status(reused_outcomes, WavefrontQueueDevicePushStatus::pushed),
              reused_requests.size());
    EXPECT_EQ(count_status(reused_outcomes, WavefrontQueueDevicePushStatus::capacity_exhausted),
              0U);

    const auto snapshots = queues.download();
    ASSERT_TRUE(snapshots.has_value()) << snapshots.error().message;
    for (auto queue_kind = std::uint32_t{}; queue_kind < QueueCount; ++queue_kind) {
        const auto& snapshot = (*snapshots)[queue_kind];
        EXPECT_EQ(snapshot.counters.size, reused_requests_per_queue);
        EXPECT_EQ(snapshot.counters.overflow_count, 0U);
        EXPECT_EQ(snapshot.counters.rejected_count, 0U);
        auto actual = std::vector<std::uint32_t>{};
        actual.reserve(snapshot.entries.size());
        for (const auto entry : snapshot.entries) {
            actual.push_back(entry.value);
        }
        std::sort(actual.begin(), actual.end());
        auto expected = std::vector<std::uint32_t>{};
        for (auto lane = std::uint32_t{}; lane < reused_requests_per_queue; ++lane) {
            expected.push_back(encoded_slot(queue_kind, 100U + lane));
        }
        std::sort(expected.begin(), expected.end());
        EXPECT_EQ(actual, expected);
    }
}

TEST(CudaWavefrontQueuesTest, RejectsInvalidInputsAndBudgetsWithoutMutation) {
    ASSERT_TRUE(select_test_device());
    constexpr auto capacity = std::size_t{23U};
    constexpr auto required_bytes =
        QueueCount * sizeof(shared::QueueHeader) + QueueCount * capacity * sizeof(shared::PathSlot);

    const auto insufficient = CudaWavefrontQueues::create(
        capacity, CudaWavefrontQueueCreateOptions{
                      .device_memory_budget = {.maximum_bytes = required_bytes - 1U},
                  });
    ASSERT_FALSE(insufficient.has_value());
    EXPECT_EQ(insufficient.error().code, core::StatusCode::resource_exhausted);
    EXPECT_NE(insufficient.error().message.find("aggregate device-memory budget"),
              std::string::npos);

    auto exact = CudaWavefrontQueues::create(
        capacity, CudaWavefrontQueueCreateOptions{
                      .device_memory_budget = {.maximum_bytes = required_bytes},
                  });
    ASSERT_TRUE(exact.has_value()) << exact.error().message;
    auto queues = std::move(*exact);
    const auto view = queues.device_view();

    auto invalid_view = view;
    invalid_view.queue_count = cuda::CudaWavefrontQueueCount - 1U;
    EXPECT_EQ(cuda::launch_wavefront_queue_pushes(invalid_view, nullptr, 0U, nullptr),
              static_cast<int>(cudaErrorInvalidValue));
    invalid_view = view;
    invalid_view.headers = nullptr;
    EXPECT_EQ(cuda::launch_wavefront_queue_pushes(invalid_view, nullptr, 0U, nullptr),
              static_cast<int>(cudaErrorInvalidValue));
    invalid_view = view;
    invalid_view.path_slots = nullptr;
    EXPECT_EQ(cuda::launch_wavefront_queue_pushes(invalid_view, nullptr, 0U, nullptr),
              static_cast<int>(cudaErrorInvalidValue));
    EXPECT_EQ(cuda::launch_wavefront_queue_pushes(view, nullptr, 1U, nullptr),
              static_cast<int>(cudaErrorInvalidValue));
    EXPECT_EQ(cuda::launch_wavefront_queue_pushes(view, nullptr, 0U, nullptr),
              static_cast<int>(cudaSuccess));

    auto sentinels = std::vector<shared::PathSlot>(QueueCount * capacity);
    for (auto index = std::size_t{}; index < sentinels.size(); ++index) {
        sentinels[index].value = 0xC0000000U | static_cast<std::uint32_t>(index);
    }
    upload_slots(view, sentinels);
    const auto invalid_request = std::array{WavefrontQueueDevicePush{
        .queue_kind = cuda::CudaWavefrontQueueCount,
        .slot = shared::PathSlot{.value = 7U},
    }};
    auto invalid_outcome = std::vector<WavefrontQueueDevicePushStatus>{};
    launch_pushes(view, invalid_request, invalid_outcome);
    ASSERT_EQ(invalid_outcome.size(), 1U);
    EXPECT_EQ(invalid_outcome.front(), WavefrontQueueDevicePushStatus::invalid_contract);
    const auto unchanged_slots = download_slots(view);
    ASSERT_EQ(unchanged_slots.size(), sentinels.size());
    for (auto index = std::size_t{}; index < sentinels.size(); ++index) {
        EXPECT_EQ(unchanged_slots[index].value, sentinels[index].value);
    }
    const auto unchanged_headers = download_headers(view);
    for (auto queue_kind = std::uint32_t{}; queue_kind < QueueCount; ++queue_kind) {
        expect_header(unchanged_headers[queue_kind], queue_kind,
                      static_cast<std::uint32_t>(capacity), 0U, 0U, 0U);
    }

    if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t)) {
        const auto unrepresentable_capacity =
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1U;
        const auto rejected = CudaWavefrontQueues::create(unrepresentable_capacity);
        ASSERT_FALSE(rejected.has_value());
        EXPECT_EQ(rejected.error().code, core::StatusCode::resource_exhausted);
        EXPECT_NE(rejected.error().message.find("32-bit device contract"), std::string::npos);
    }

    const auto unknown_policy =
        queues.reset(static_cast<CudaWavefrontQueueResetPolicy>(std::uint8_t{0xFFU}));
    ASSERT_FALSE(unknown_policy.has_value());
    EXPECT_EQ(unknown_policy.error().code, core::StatusCode::invalid_argument);
    const auto headers_after_unknown_policy = download_headers(view);
    for (auto queue_kind = std::uint32_t{}; queue_kind < QueueCount; ++queue_kind) {
        expect_header(headers_after_unknown_policy[queue_kind], queue_kind,
                      static_cast<std::uint32_t>(capacity), 0U, 0U, 0U);
    }

    const auto close_status = queues.close();
    ASSERT_TRUE(close_status.has_value()) << close_status.error().message;
    EXPECT_FALSE(static_cast<bool>(queues));
    EXPECT_EQ(queues.device_view().headers, nullptr);
    const auto closed_download = queues.download();
    ASSERT_FALSE(closed_download.has_value());
    EXPECT_EQ(closed_download.error().code, core::StatusCode::invalid_argument);
    const auto closed_reset = queues.reset(CudaWavefrontQueueResetPolicy::require_no_overflow);
    ASSERT_FALSE(closed_reset.has_value());
    EXPECT_EQ(closed_reset.error().code, core::StatusCode::invalid_argument);
}

} // namespace
} // namespace blackframe::engine
