#include <Blackframe/Core/Status.hpp>
#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <array>
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
using blackframe::xpu::cuda::cuda_memory_status_code;
using blackframe::xpu::cuda::DeviceAllocation;
using blackframe::xpu::cuda::DeviceBuffer;
using blackframe::xpu::cuda::DeviceMemoryBudget;
using blackframe::xpu::cuda::DeviceScratchBuffer;

[[nodiscard]] testing::AssertionResult select_test_device() {
    int device_count = 0;
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

TEST(CudaMemoryStatus, MapsOutOfMemoryExplicitly) {
    EXPECT_EQ(cuda_memory_status_code(static_cast<std::int32_t>(cudaSuccess)), StatusCode::success);
    EXPECT_EQ(cuda_memory_status_code(static_cast<std::int32_t>(cudaErrorMemoryAllocation)),
              StatusCode::resource_exhausted);
    EXPECT_EQ(cuda_memory_status_code(static_cast<std::int32_t>(cudaErrorNoDevice)),
              StatusCode::unavailable);
    EXPECT_EQ(cuda_memory_status_code(static_cast<std::int32_t>(cudaErrorInvalidValue)),
              StatusCode::platform_error);
}

TEST(CudaMemoryBudget, RejectsRequestsBeforeCallingCuda) {
    const auto allocation =
        DeviceAllocation::allocate_bytes(65, DeviceMemoryBudget{.maximum_bytes = 64});

    ASSERT_FALSE(allocation);
    EXPECT_EQ(allocation.error().code, StatusCode::resource_exhausted);
    EXPECT_NE(allocation.error().message.find("explicit device-memory budget"), std::string::npos);
}

TEST(CudaMemoryBudget, RejectsTypedSizeOverflow) {
    constexpr auto count = std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t) + 1;
    const auto buffer = DeviceBuffer<std::uint64_t>::allocate(count);

    ASSERT_FALSE(buffer);
    EXPECT_EQ(buffer.error().code, StatusCode::resource_exhausted);
    EXPECT_NE(buffer.error().message.find("multiplication overflowed"), std::string::npos);
}

TEST(CudaMemoryLifetime, OwnsMovesAndClosesTypedStorage) {
    ASSERT_TRUE(select_test_device());

    auto created = DeviceBuffer<std::uint32_t>::allocate(32);
    ASSERT_TRUE(created) << created.error().message;
    auto buffer = std::move(*created);
    ASSERT_TRUE(buffer);
    EXPECT_EQ(buffer.size(), 32);
    EXPECT_EQ(buffer.size_bytes(), 32 * sizeof(std::uint32_t));
    EXPECT_EQ(buffer.device_ordinal(), 0);

    ASSERT_EQ(cudaMemset(buffer.data(), 0x2A, buffer.size_bytes()), cudaSuccess);
    std::array<std::uint32_t, 32> host_values{};
    ASSERT_EQ(
        cudaMemcpy(host_values.data(), buffer.data(), buffer.size_bytes(), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (const auto value : host_values) {
        EXPECT_EQ(value, 0x2A2A2A2AU);
    }

    DeviceBuffer<std::uint32_t> moved{std::move(buffer)};
    EXPECT_FALSE(buffer);
    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.device_ordinal(), -1);
    EXPECT_TRUE(moved);

    const auto close_status = moved.close();
    ASSERT_TRUE(close_status) << close_status.error().message;
    EXPECT_FALSE(moved);
    EXPECT_TRUE(moved.empty());
    EXPECT_EQ(moved.size_bytes(), 0);
    const auto repeated_close = moved.close();
    ASSERT_TRUE(repeated_close) << repeated_close.error().message;
}

TEST(CudaMemoryLifetime, DestructorReleasesOwnedStorage) {
    ASSERT_TRUE(select_test_device());

    {
        auto created = DeviceAllocation::allocate_bytes(4096);
        ASSERT_TRUE(created) << created.error().message;
        EXPECT_NE(created->data(), nullptr);
        ASSERT_EQ(cudaMemset(created->data(), 0, created->size_bytes()), cudaSuccess);
    }

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(CudaMemoryLifetime, ScratchMoveTransfersOwnershipExactlyOnce) {
    ASSERT_TRUE(select_test_device());

    auto created = DeviceScratchBuffer::create(256, DeviceMemoryBudget{.maximum_bytes = 512});
    ASSERT_TRUE(created) << created.error().message;
    auto scratch = std::move(*created);
    const auto slice = scratch.allocate(64, 64);
    ASSERT_TRUE(slice) << slice.error().message;

    DeviceScratchBuffer moved{std::move(scratch)};
    EXPECT_TRUE(scratch.empty());
    EXPECT_EQ(scratch.data(), nullptr);
    EXPECT_EQ(scratch.used_bytes(), 0);
    EXPECT_EQ(moved.capacity_bytes(), 256);
    EXPECT_GT(moved.used_bytes(), 0);

    const auto close_status = moved.close();
    ASSERT_TRUE(close_status) << close_status.error().message;
    const auto repeated_close = moved.close();
    ASSERT_TRUE(repeated_close) << repeated_close.error().message;
}

TEST(CudaMemoryLifetime, ScratchGrowthKeepsItsOwningDevice) {
    ASSERT_TRUE(select_test_device());

    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count < 2) {
        GTEST_SKIP() << "A second CUDA device is required for the ownership transition check.";
    }

    auto created = DeviceScratchBuffer::create(128, DeviceMemoryBudget{.maximum_bytes = 256});
    ASSERT_TRUE(created) << created.error().message;
    auto scratch = std::move(*created);
    ASSERT_EQ(scratch.device_ordinal(), 0);

    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    const auto reserve_status = scratch.reserve(256);
    ASSERT_TRUE(reserve_status) << reserve_status.error().message;
    EXPECT_EQ(scratch.device_ordinal(), 0);

    int current_device = -1;
    ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    EXPECT_EQ(current_device, 1);

    const auto close_status = scratch.close();
    ASSERT_TRUE(close_status) << close_status.error().message;
    ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    EXPECT_EQ(current_device, 1);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
}

TEST(CudaScratchBuffer, AllocatesAlignedSlicesAndReusesStorage) {
    ASSERT_TRUE(select_test_device());

    auto created = DeviceScratchBuffer::create(256, DeviceMemoryBudget{.maximum_bytes = 512});
    ASSERT_TRUE(created) << created.error().message;
    auto scratch = std::move(*created);
    ASSERT_EQ(scratch.capacity_bytes(), 256);
    ASSERT_EQ(scratch.maximum_capacity_bytes(), 512);

    const auto first = scratch.allocate(17, 64);
    ASSERT_TRUE(first) << first.error().message;
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(first->data) % 64, 0);
    EXPECT_EQ(first->size_bytes, 17);
    ASSERT_EQ(cudaMemset(first->data, 0x11, first->size_bytes), cudaSuccess);

    const auto second = scratch.allocate(31, 32);
    ASSERT_TRUE(second) << second.error().message;
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second->data) % 32, 0);
    EXPECT_GE(second->offset_bytes, first->offset_bytes + first->size_bytes);
    ASSERT_EQ(cudaMemset(second->data, 0x22, second->size_bytes), cudaSuccess);

    const auto used_before_failure = scratch.used_bytes();
    const auto exhausted = scratch.allocate(256, 1);
    ASSERT_FALSE(exhausted);
    EXPECT_EQ(exhausted.error().code, StatusCode::resource_exhausted);
    EXPECT_EQ(scratch.used_bytes(), used_before_failure);

    const auto growth_while_live = scratch.reserve(384);
    ASSERT_FALSE(growth_while_live);
    EXPECT_EQ(growth_while_live.error().code, StatusCode::invalid_argument);
    EXPECT_EQ(scratch.capacity_bytes(), 256);

    scratch.reset();
    const auto growth_status = scratch.reserve(384);
    ASSERT_TRUE(growth_status) << growth_status.error().message;
    EXPECT_EQ(scratch.capacity_bytes(), 384);

    const auto reused = scratch.allocate(64, 128);
    ASSERT_TRUE(reused) << reused.error().message;
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(reused->data) % 128, 0);
    ASSERT_EQ(cudaMemset(reused->data, 0x33, reused->size_bytes), cudaSuccess);

    scratch.reset();
    const auto close_status = scratch.close();
    ASSERT_TRUE(close_status) << close_status.error().message;
    EXPECT_TRUE(scratch.empty());
}

TEST(CudaScratchBuffer, BudgetFailurePreservesExistingStorage) {
    ASSERT_TRUE(select_test_device());

    auto created = DeviceScratchBuffer::create(128, DeviceMemoryBudget{.maximum_bytes = 256});
    ASSERT_TRUE(created) << created.error().message;
    auto scratch = std::move(*created);
    const auto original_data = scratch.data();

    const auto reserve_status = scratch.reserve(257);
    ASSERT_FALSE(reserve_status);
    EXPECT_EQ(reserve_status.error().code, StatusCode::resource_exhausted);
    EXPECT_EQ(scratch.data(), original_data);
    EXPECT_EQ(scratch.capacity_bytes(), 128);
    EXPECT_EQ(scratch.used_bytes(), 0);

    const auto close_status = scratch.close();
    ASSERT_TRUE(close_status) << close_status.error().message;
}

TEST(CudaScratchBuffer, RejectsEmptyAndInvalidlyAlignedSlices) {
    ASSERT_TRUE(select_test_device());

    auto created = DeviceScratchBuffer::create(64, DeviceMemoryBudget{.maximum_bytes = 64});
    ASSERT_TRUE(created) << created.error().message;
    auto scratch = std::move(*created);

    const auto empty = scratch.allocate(0, 1);
    ASSERT_FALSE(empty);
    EXPECT_EQ(empty.error().code, StatusCode::invalid_argument);

    const auto zero_alignment = scratch.allocate(1, 0);
    ASSERT_FALSE(zero_alignment);
    EXPECT_EQ(zero_alignment.error().code, StatusCode::invalid_argument);

    const auto non_power_of_two = scratch.allocate(1, 3);
    ASSERT_FALSE(non_power_of_two);
    EXPECT_EQ(non_power_of_two.error().code, StatusCode::invalid_argument);

    EXPECT_EQ(scratch.used_bytes(), 0);
    const auto close_status = scratch.close();
    ASSERT_TRUE(close_status) << close_status.error().message;
}

static_assert(std::is_move_constructible_v<DeviceAllocation>);
static_assert(!std::is_move_assignable_v<DeviceAllocation>);
static_assert(!std::is_copy_constructible_v<DeviceAllocation>);
static_assert(!std::is_copy_assignable_v<DeviceAllocation>);
static_assert(std::is_move_constructible_v<DeviceBuffer<std::uint32_t>>);
static_assert(!std::is_move_assignable_v<DeviceBuffer<std::uint32_t>>);
static_assert(!std::is_copy_constructible_v<DeviceBuffer<std::uint32_t>>);
static_assert(std::is_move_constructible_v<DeviceScratchBuffer>);
static_assert(!std::is_move_assignable_v<DeviceScratchBuffer>);

} // namespace
