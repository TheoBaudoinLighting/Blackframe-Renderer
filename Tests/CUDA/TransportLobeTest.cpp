#include <Blackframe/XPU/CUDA/TransportLobeProbe.hpp>
#include <cmath>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

namespace {

namespace cuda = blackframe::xpu::cuda;

TEST(CudaTransportLobes, DeviceEvaluatesPdfAndSamplesEveryPortedFamily) {
    auto result = cuda::TransportLobeProbeResult{};
    int device_count = 0;
    const auto status = blackframe_cuda_run_transport_lobe_probe(&result, &device_count);
    ASSERT_EQ(status, static_cast<int>(cudaSuccess))
        << cudaGetErrorString(static_cast<cudaError_t>(status));
    ASSERT_GT(device_count, 0);
    EXPECT_EQ(result.device_cxx_standard, 202002U);
    EXPECT_EQ(result.passed_mask, cuda::TransportLobeProbeExpectedMask);
    for (const auto value : result.representative_values) {
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_GT(value, 0.0F);
    }
    for (const auto value : result.reserved) {
        EXPECT_EQ(value, 0U);
    }
}

TEST(CudaTransportLobes, ProbeRejectsInvalidOutputArguments) {
    auto result = cuda::TransportLobeProbeResult{};
    int device_count = 0;
    EXPECT_EQ(blackframe_cuda_run_transport_lobe_probe(nullptr, &device_count),
              static_cast<int>(cudaErrorInvalidValue));
    EXPECT_EQ(blackframe_cuda_run_transport_lobe_probe(&result, nullptr),
              static_cast<int>(cudaErrorInvalidValue));
}

} // namespace
