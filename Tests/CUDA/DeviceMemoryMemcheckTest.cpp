#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <utility>

extern "C" int blackframe_cuda_memory_memcheck_fill(void* destination, std::size_t byte_count,
                                                    std::uint8_t value) noexcept;

namespace {

using blackframe::xpu::cuda::DeviceBuffer;
using blackframe::xpu::cuda::DeviceMemoryBudget;
using blackframe::xpu::cuda::DeviceScratchBuffer;

TEST(CudaMemoryMemcheck, ExercisesRaiiAndScratchLifetimes) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    auto buffer_result = DeviceBuffer<std::uint8_t>::allocate(4096);
    ASSERT_TRUE(buffer_result) << buffer_result.error().message;
    auto buffer = std::move(*buffer_result);
    ASSERT_EQ(blackframe_cuda_memory_memcheck_fill(buffer.data(), buffer.size_bytes(), 0x5AU),
              static_cast<int>(cudaSuccess));

    std::array<std::uint8_t, 4096> host_values{};
    ASSERT_EQ(
        cudaMemcpy(host_values.data(), buffer.data(), buffer.size_bytes(), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (const auto value : host_values) {
        EXPECT_EQ(value, 0x5AU);
    }

    auto scratch_result =
        DeviceScratchBuffer::create(8192, DeviceMemoryBudget{.maximum_bytes = 8192});
    ASSERT_TRUE(scratch_result) << scratch_result.error().message;
    auto scratch = std::move(*scratch_result);
    const auto slice = scratch.allocate(1024, 256);
    ASSERT_TRUE(slice) << slice.error().message;
    ASSERT_EQ(blackframe_cuda_memory_memcheck_fill(slice->data, slice->size_bytes, 0xA5U),
              static_cast<int>(cudaSuccess));

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    // Both owners intentionally leave through their RAII destructors. Memcheck's
    // full leak pass makes a missing destructor release fail this test.
}

} // namespace
