#include <Blackframe/Core/Status.hpp>
#include <Blackframe/XPU/CUDA/AsyncRuntime.hpp>
#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

namespace {

using blackframe::core::StatusCode;
using blackframe::xpu::cuda::DeviceBuffer;
using blackframe::xpu::cuda::Event;
using blackframe::xpu::cuda::PinnedHostAllocation;
using blackframe::xpu::cuda::PinnedHostBuffer;
using blackframe::xpu::cuda::PinnedHostMemoryBudget;
using blackframe::xpu::cuda::Stream;

struct alignas(8192) OverAlignedPinnedValue final {
    std::uint64_t value{};
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

[[nodiscard]] cudaStream_t native_stream(const Stream& stream) noexcept {
    return static_cast<cudaStream_t>(stream.native_handle());
}

TEST(CudaAsyncRuntime, StreamAndEventMoveOwnershipAndCloseIdempotently) {
    ASSERT_TRUE(select_test_device());

    auto created_stream = Stream::create();
    ASSERT_TRUE(created_stream) << created_stream.error().message;
    auto stream = std::move(*created_stream);
    EXPECT_TRUE(stream);
    EXPECT_NE(stream.native_handle(), nullptr);
    EXPECT_EQ(stream.device_ordinal(), 0);

    auto created_event = Event::create();
    ASSERT_TRUE(created_event) << created_event.error().message;
    auto event = std::move(*created_event);
    EXPECT_TRUE(event);
    EXPECT_NE(event.native_handle(), nullptr);
    EXPECT_EQ(event.device_ordinal(), 0);

    Stream moved_stream{std::move(stream)};
    Event moved_event{std::move(event)};
    EXPECT_FALSE(stream);
    EXPECT_FALSE(event);
    EXPECT_EQ(stream.device_ordinal(), -1);
    EXPECT_EQ(event.device_ordinal(), -1);

    auto status = moved_event.record(moved_stream);
    ASSERT_TRUE(status) << status.error().message;
    status = moved_stream.wait(moved_event);
    ASSERT_TRUE(status) << status.error().message;
    status = moved_event.synchronize();
    ASSERT_TRUE(status) << status.error().message;
    const auto complete = moved_event.is_complete();
    ASSERT_TRUE(complete) << complete.error().message;
    EXPECT_TRUE(*complete);

    status = moved_event.close();
    ASSERT_TRUE(status) << status.error().message;
    EXPECT_FALSE(moved_event);
    status = moved_event.close();
    ASSERT_TRUE(status) << status.error().message;

    status = moved_stream.close();
    ASSERT_TRUE(status) << status.error().message;
    EXPECT_FALSE(moved_stream);
    status = moved_stream.close();
    ASSERT_TRUE(status) << status.error().message;
}

TEST(CudaAsyncRuntime, ExplicitEventsOrderPinnedTransfersAcrossNonblockingStreams) {
    ASSERT_TRUE(select_test_device());
    constexpr auto element_count = std::size_t{4096};

    auto upload_storage = PinnedHostBuffer<std::uint32_t>::allocate(element_count);
    auto download_storage = PinnedHostBuffer<std::uint32_t>::allocate(element_count);
    auto device_storage = DeviceBuffer<std::uint32_t>::allocate(element_count);
    ASSERT_TRUE(upload_storage) << upload_storage.error().message;
    ASSERT_TRUE(download_storage) << download_storage.error().message;
    ASSERT_TRUE(device_storage) << device_storage.error().message;
    for (auto index = std::size_t{}; index < element_count; ++index) {
        (*upload_storage)[index] = static_cast<std::uint32_t>(index * 747796405U + 2891336453U);
        (*download_storage)[index] = 0U;
    }

    auto upload_stream = Stream::create();
    auto download_stream = Stream::create();
    auto upload_complete = Event::create();
    auto download_complete = Event::create();
    ASSERT_TRUE(upload_stream) << upload_stream.error().message;
    ASSERT_TRUE(download_stream) << download_stream.error().message;
    ASSERT_TRUE(upload_complete) << upload_complete.error().message;
    ASSERT_TRUE(download_complete) << download_complete.error().message;

    ASSERT_EQ(cudaMemcpyAsync(device_storage->data(), upload_storage->data(),
                              upload_storage->size_bytes(), cudaMemcpyHostToDevice,
                              native_stream(*upload_stream)),
              cudaSuccess);
    auto status = upload_complete->record(*upload_stream);
    ASSERT_TRUE(status) << status.error().message;
    status = download_stream->wait(*upload_complete);
    ASSERT_TRUE(status) << status.error().message;
    ASSERT_EQ(cudaMemcpyAsync(download_storage->data(), device_storage->data(),
                              download_storage->size_bytes(), cudaMemcpyDeviceToHost,
                              native_stream(*download_stream)),
              cudaSuccess);
    status = download_complete->record(*download_stream);
    ASSERT_TRUE(status) << status.error().message;
    status = download_complete->synchronize();
    ASSERT_TRUE(status) << status.error().message;

    for (auto index = std::size_t{}; index < element_count; ++index) {
        EXPECT_EQ((*download_storage)[index], (*upload_storage)[index]) << "lane " << index;
    }

    status = download_complete->close();
    ASSERT_TRUE(status) << status.error().message;
    status = upload_complete->close();
    ASSERT_TRUE(status) << status.error().message;
    status = download_stream->close();
    ASSERT_TRUE(status) << status.error().message;
    status = upload_stream->close();
    ASSERT_TRUE(status) << status.error().message;
    status = device_storage->close();
    ASSERT_TRUE(status) << status.error().message;
    status = download_storage->close();
    ASSERT_TRUE(status) << status.error().message;
    status = upload_storage->close();
    ASSERT_TRUE(status) << status.error().message;
}

TEST(CudaAsyncRuntime, StreamCloseCompletesQueuedPinnedTransfer) {
    ASSERT_TRUE(select_test_device());
    constexpr auto element_count = std::size_t{1024};

    auto host_storage = PinnedHostBuffer<std::uint32_t>::allocate(element_count);
    auto device_storage = DeviceBuffer<std::uint32_t>::allocate(element_count);
    auto stream = Stream::create();
    ASSERT_TRUE(host_storage) << host_storage.error().message;
    ASSERT_TRUE(device_storage) << device_storage.error().message;
    ASSERT_TRUE(stream) << stream.error().message;
    for (auto index = std::size_t{}; index < element_count; ++index) {
        (*host_storage)[index] = static_cast<std::uint32_t>(index + 17U);
    }

    ASSERT_EQ(cudaMemcpyAsync(device_storage->data(), host_storage->data(),
                              host_storage->size_bytes(), cudaMemcpyHostToDevice,
                              native_stream(*stream)),
              cudaSuccess);
    auto status = stream->close();
    ASSERT_TRUE(status) << status.error().message;

    for (auto index = std::size_t{}; index < element_count; ++index) {
        (*host_storage)[index] = 0U;
    }
    ASSERT_EQ(cudaMemcpy(host_storage->data(), device_storage->data(), host_storage->size_bytes(),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (auto index = std::size_t{}; index < element_count; ++index) {
        EXPECT_EQ((*host_storage)[index], static_cast<std::uint32_t>(index + 17U));
    }

    status = device_storage->close();
    ASSERT_TRUE(status) << status.error().message;
    status = host_storage->close();
    ASSERT_TRUE(status) << status.error().message;
}

TEST(CudaAsyncRuntime, ClosedHandlesAreRejectedExplicitly) {
    ASSERT_TRUE(select_test_device());

    const auto closed_stream = Stream{};
    const auto closed_event = Event{};
    auto status = closed_stream.synchronize();
    ASSERT_FALSE(status);
    EXPECT_EQ(status.error().code, StatusCode::invalid_argument);
    status = closed_stream.wait(closed_event);
    ASSERT_FALSE(status);
    EXPECT_EQ(status.error().code, StatusCode::invalid_argument);
    status = closed_event.record(closed_stream);
    ASSERT_FALSE(status);
    EXPECT_EQ(status.error().code, StatusCode::invalid_argument);
    status = closed_event.synchronize();
    ASSERT_FALSE(status);
    EXPECT_EQ(status.error().code, StatusCode::invalid_argument);
    const auto complete = closed_event.is_complete();
    ASSERT_FALSE(complete);
    EXPECT_EQ(complete.error().code, StatusCode::invalid_argument);
}

TEST(CudaAsyncRuntime, PinnedAllocationEnforcesBudgetAndTypedOverflow) {
    const auto over_budget =
        PinnedHostAllocation::allocate_bytes(65U, PinnedHostMemoryBudget{.maximum_bytes = 64U});
    ASSERT_FALSE(over_budget);
    EXPECT_EQ(over_budget.error().code, StatusCode::resource_exhausted);
    EXPECT_NE(over_budget.error().message.find("explicit memory budget"), std::string::npos);

    constexpr auto count =
        std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t) + std::size_t{1};
    const auto overflow = PinnedHostBuffer<std::uint64_t>::allocate(count);
    ASSERT_FALSE(overflow);
    EXPECT_EQ(overflow.error().code, StatusCode::resource_exhausted);
    EXPECT_NE(overflow.error().message.find("multiplication overflowed"), std::string::npos);

    const auto empty = PinnedHostAllocation::allocate_bytes(0U);
    ASSERT_TRUE(empty) << empty.error().message;
    EXPECT_FALSE(*empty);
    EXPECT_EQ(empty->size_bytes(), 0U);
    EXPECT_EQ(empty->device_ordinal(), -1);
}

TEST(CudaAsyncRuntime, PinnedBufferProvidesExactPayloadWithExplicitExtendedAlignment) {
    ASSERT_TRUE(select_test_device());
    constexpr auto element_count = std::size_t{3};
    constexpr auto payload_bytes = element_count * sizeof(OverAlignedPinnedValue);
    constexpr auto allocation_bytes = payload_bytes + alignof(OverAlignedPinnedValue) - 1U;

    const auto insufficient_budget = PinnedHostBuffer<OverAlignedPinnedValue>::allocate(
        element_count, PinnedHostMemoryBudget{.maximum_bytes = allocation_bytes - 1U});
    ASSERT_FALSE(insufficient_budget);
    EXPECT_EQ(insufficient_budget.error().code, StatusCode::resource_exhausted);

    auto buffer = PinnedHostBuffer<OverAlignedPinnedValue>::allocate(
        element_count, PinnedHostMemoryBudget{.maximum_bytes = allocation_bytes});
    ASSERT_TRUE(buffer) << buffer.error().message;
    ASSERT_NE(buffer->data(), nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buffer->data()) % alignof(OverAlignedPinnedValue),
              0U);
    EXPECT_EQ(buffer->size(), element_count);
    EXPECT_EQ(buffer->size_bytes(), payload_bytes);
    for (auto index = std::size_t{}; index < element_count; ++index) {
        (*buffer)[index].value = static_cast<std::uint64_t>(index + 41U);
    }
    for (auto index = std::size_t{}; index < element_count; ++index) {
        EXPECT_EQ((*buffer)[index].value, static_cast<std::uint64_t>(index + 41U));
    }

    const auto close_status = buffer->close();
    ASSERT_TRUE(close_status) << close_status.error().message;
    EXPECT_FALSE(*buffer);
    EXPECT_EQ(buffer->data(), nullptr);
    EXPECT_EQ(buffer->size_bytes(), 0U);
}

TEST(CudaAsyncRuntime, OperationsRequireTheOwningDeviceToBeActive) {
    ASSERT_TRUE(select_test_device());
    auto device_count = int{};
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count < 2) {
        GTEST_SKIP() << "A second CUDA device is required for the ownership check.";
    }

    auto stream = Stream::create();
    auto event = Event::create();
    auto pinned = PinnedHostAllocation::allocate_bytes(64U);
    ASSERT_TRUE(stream) << stream.error().message;
    ASSERT_TRUE(event) << event.error().message;
    ASSERT_TRUE(pinned) << pinned.error().message;
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);

    auto status = stream->synchronize();
    ASSERT_FALSE(status);
    EXPECT_EQ(status.error().code, StatusCode::invalid_argument);
    status = event->record(*stream);
    ASSERT_FALSE(status);
    EXPECT_EQ(status.error().code, StatusCode::invalid_argument);

    status = event->close();
    ASSERT_TRUE(status) << status.error().message;
    status = stream->close();
    ASSERT_TRUE(status) << status.error().message;
    status = pinned->close();
    ASSERT_TRUE(status) << status.error().message;
    auto active_device = int{-1};
    ASSERT_EQ(cudaGetDevice(&active_device), cudaSuccess);
    EXPECT_EQ(active_device, 1);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
}

static_assert(std::is_move_constructible_v<Stream>);
static_assert(!std::is_move_assignable_v<Stream>);
static_assert(!std::is_copy_constructible_v<Stream>);
static_assert(!std::is_copy_assignable_v<Stream>);
static_assert(std::is_move_constructible_v<Event>);
static_assert(!std::is_move_assignable_v<Event>);
static_assert(!std::is_copy_constructible_v<Event>);
static_assert(!std::is_copy_assignable_v<Event>);
static_assert(std::is_move_constructible_v<PinnedHostAllocation>);
static_assert(!std::is_move_assignable_v<PinnedHostAllocation>);
static_assert(!std::is_copy_constructible_v<PinnedHostAllocation>);
static_assert(!std::is_copy_assignable_v<PinnedHostAllocation>);
static_assert(std::is_move_constructible_v<PinnedHostBuffer<std::uint32_t>>);
static_assert(!std::is_move_assignable_v<PinnedHostBuffer<std::uint32_t>>);
static_assert(!std::is_copy_constructible_v<PinnedHostBuffer<std::uint32_t>>);
static_assert(!std::is_copy_assignable_v<PinnedHostBuffer<std::uint32_t>>);
static_assert(blackframe::xpu::cuda::PinnedHostBufferElement<OverAlignedPinnedValue>);

} // namespace
