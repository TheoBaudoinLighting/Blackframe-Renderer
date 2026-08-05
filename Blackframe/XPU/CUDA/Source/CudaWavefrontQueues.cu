#include <Blackframe/XPU/CUDA/WavefrontQueueDevice.cuh>
#include <Blackframe/XPU/CUDA/WavefrontQueueKernel.hpp>
#include <cstdint>
#include <cuda_runtime_api.h>

#if !defined(__CUDACC__)
#error "The CUDA wavefront queue kernel must be compiled by the CUDA compiler."
#endif

static_assert(__cplusplus == 202002L);

#if defined(__cpp_pack_indexing)
#error "C++26 features are forbidden in CUDA wavefront queue code."
#endif

namespace {

constexpr auto ThreadsPerBlockX = std::uint32_t{16U};
constexpr auto ThreadsPerBlockY = std::uint32_t{16U};
constexpr auto ThreadsPerBlock = ThreadsPerBlockX * ThreadsPerBlockY;

__global__ void push_wavefront_queues_kernel(
    const blackframe::xpu::cuda::WavefrontQueueDeviceSoa queues,
    const blackframe::xpu::cuda::WavefrontQueueDevicePush* const requests,
    const std::uint32_t request_count,
    blackframe::xpu::cuda::WavefrontQueueDevicePushStatus* const outcomes) {
    const auto thread_index = static_cast<std::uint32_t>(threadIdx.x) +
                              static_cast<std::uint32_t>(blockDim.x) * threadIdx.y;
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * ThreadsPerBlock + thread_index;
    if (index >= request_count) {
        return;
    }
    const auto request = requests[index];
    outcomes[index] = blackframe::xpu::cuda::try_push_wavefront_queue_warp(
        queues, request.queue_kind, request.slot);
}

} // namespace

extern "C" int blackframe_cuda_launch_wavefront_queue_pushes(
    blackframe::xpu::shared::QueueHeader* const headers,
    blackframe::xpu::shared::PathSlot* const path_slots, const std::uint32_t queue_count,
    const std::uint32_t slot_stride,
    const blackframe::xpu::cuda::WavefrontQueueDevicePush* const requests,
    const std::uint32_t request_count,
    blackframe::xpu::cuda::WavefrontQueueDevicePushStatus* const outcomes) noexcept {
    const auto queues = blackframe::xpu::cuda::WavefrontQueueDeviceSoa{
        .headers = headers,
        .path_slots = path_slots,
        .queue_count = queue_count,
        .slot_stride = slot_stride,
    };
    if (queues.headers == nullptr ||
        queues.queue_count != blackframe::xpu::cuda::CudaWavefrontQueueCount ||
        (queues.slot_stride != 0U && queues.path_slots == nullptr) ||
        (request_count != 0U && (requests == nullptr || outcomes == nullptr))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (request_count == 0U) {
        return static_cast<int>(cudaSuccess);
    }

    const auto block_count = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(request_count) + ThreadsPerBlock - 1U) / ThreadsPerBlock);
    push_wavefront_queues_kernel<<<block_count, dim3{ThreadsPerBlockX, ThreadsPerBlockY, 1U}>>>(
        queues, requests, request_count, outcomes);
    return static_cast<int>(cudaGetLastError());
}
