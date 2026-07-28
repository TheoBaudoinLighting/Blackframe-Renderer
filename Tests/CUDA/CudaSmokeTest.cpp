#include <Blackframe/XPU/CUDA/SmokeKernel.hpp>
#include <Blackframe/XPU/CUDA/SmokeKernelPayload.hpp>
#include <cstdint>
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

    ASSERT_EQ(status, 0);
    EXPECT_GT(device_count, 0);
    EXPECT_EQ(output, input ^ xor_mask);
}

TEST(CudaSharedHeaders, CompileAsCudaCxx20) {
    EXPECT_EQ(blackframe_cuda_shared_header_language_level(), 202002U);
}

} // namespace
