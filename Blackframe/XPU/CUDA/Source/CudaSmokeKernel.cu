#include <Blackframe/XPU/CUDA/SmokeKernel.hpp>
#include <Blackframe/XPU/CUDA/SmokeKernelPayload.hpp>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace {

__global__ void smoke_kernel(blackframe::xpu::cuda::SmokeKernelPayload payload,
                             std::uint32_t* output) {
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
        *output = payload.input ^ payload.xor_mask;
    }
}

} // namespace

extern "C" int blackframe_cuda_run_smoke(std::uint32_t input, std::uint32_t xor_mask,
                                         std::uint32_t* output, int* device_count) noexcept {
    if (output == nullptr || device_count == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    *output = 0U;
    *device_count = 0;

    auto status = cudaGetDeviceCount(device_count);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    if (*device_count == 0) {
        return static_cast<int>(cudaErrorNoDevice);
    }

    status = cudaSetDevice(0);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    std::uint32_t* device_output = nullptr;
    status = cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(*device_output));
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    const blackframe::xpu::cuda::SmokeKernelPayload payload{
        .input = input,
        .xor_mask = xor_mask,
    };
    smoke_kernel<<<1, 1>>>(payload, device_output);

    status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(output, device_output, sizeof(*output), cudaMemcpyDeviceToHost);
    }

    const auto free_status = cudaFree(device_output);
    if (status == cudaSuccess) {
        status = free_status;
    }

    return static_cast<int>(status);
}
