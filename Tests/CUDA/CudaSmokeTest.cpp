#include <Blackframe/XPU/CUDA/SmokeKernel.hpp>
#include <Blackframe/XPU/CUDA/SmokeKernelPayload.hpp>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <type_traits>

namespace {

static_assert(std::is_standard_layout_v<blackframe::xpu::cuda::SmokeKernelPayload>);
static_assert(std::is_trivially_copyable_v<blackframe::xpu::cuda::SmokeKernelPayload>);

TEST(CudaSmoke, ExecutesKernelOnDevice) {
    constexpr auto input = std::uint32_t{0x12345678U};
    constexpr auto xor_mask = std::uint32_t{0xA5A5A5A5U};

    std::uint32_t output = 0U;
    int device_count = 0;
    const auto status = blackframe_cuda_run_smoke(input, xor_mask, &output, &device_count);

    ASSERT_EQ(status, static_cast<int>(cudaSuccess))
        << cudaGetErrorString(static_cast<cudaError_t>(status));
    EXPECT_GT(device_count, 0);
    EXPECT_EQ(output, input ^ xor_mask);

    int active_device = -1;
    ASSERT_EQ(cudaGetDevice(&active_device), cudaSuccess);
    EXPECT_EQ(active_device, 0);

    cudaDeviceProp device_properties{};
    ASSERT_EQ(cudaGetDeviceProperties(&device_properties, active_device), cudaSuccess);
    EXPECT_NE(device_properties.name[0], '\0');
}

TEST(CudaSharedHeaders, CompileAsCudaCxx20) {
    EXPECT_EQ(blackframe_cuda_shared_header_language_level(), 202002U);
}

} // namespace
