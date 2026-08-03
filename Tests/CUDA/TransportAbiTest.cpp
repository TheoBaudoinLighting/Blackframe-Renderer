#include <Blackframe/XPU/CUDA/TransportAbiProbe.hpp>
#include <Blackframe/XPU/Shared/TransportAbi.hpp>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

namespace {

namespace shared = blackframe::xpu::shared;

TEST(CudaTransportAbi, DeviceLayoutMatchesHostLayoutExactly) {
    const auto host = shared::host_layout_manifest(__cplusplus);
    auto device = shared::LayoutManifest{};
    int device_count = 0;

    const auto status = blackframe_cuda_query_transport_abi_layout(&device, &device_count);
    ASSERT_EQ(status, static_cast<int>(cudaSuccess))
        << cudaGetErrorString(static_cast<cudaError_t>(status));
    EXPECT_GT(device_count, 0);
    EXPECT_EQ(device.abi_major, host.abi_major);
    EXPECT_EQ(device.abi_minor, host.abi_minor);
    EXPECT_EQ(device.device_cxx_standard, 202002U);
    EXPECT_EQ(device.value_count, host.value_count);
    EXPECT_EQ(device.reserved_header, 0U);

    ASSERT_EQ(device.value_count, shared::HostDeviceLayoutValueCount);
    for (auto index = std::size_t{0}; index < shared::HostDeviceLayoutValueCount; ++index) {
        EXPECT_EQ(device.values[index], host.values[index]) << "layout value " << index;
    }
    for (const auto value : device.reserved_tail) {
        EXPECT_EQ(value, 0U);
    }
}

TEST(CudaTransportAbi, InvalidOutputArgumentsAreRejected) {
    auto manifest = shared::LayoutManifest{};
    int device_count = 0;

    EXPECT_EQ(blackframe_cuda_query_transport_abi_layout(nullptr, &device_count),
              static_cast<int>(cudaErrorInvalidValue));
    EXPECT_EQ(blackframe_cuda_query_transport_abi_layout(&manifest, nullptr),
              static_cast<int>(cudaErrorInvalidValue));
}

} // namespace
