#pragma once

#include <Blackframe/XPU/CUDA/WavefrontQueueKernel.hpp>
#include <cstdint>
#include <cuda_runtime.h>

#if !defined(__CUDACC__)
#error "The CUDA wavefront queue device contract requires the CUDA compiler."
#endif

namespace blackframe::xpu::cuda {
namespace wavefront_queue_device_detail {

[[nodiscard]] __device__ inline std::uint32_t atomic_load(std::uint32_t* const value) noexcept {
    return atomicCAS(value, 0U, 0U);
}

__device__ inline void atomic_saturating_increment(std::uint32_t* const value) noexcept {
    auto observed = atomic_load(value);
    constexpr auto maximum = ~std::uint32_t{0};
    while (observed != maximum) {
        const auto previous = atomicCAS(value, observed, observed + 1U);
        if (previous == observed) {
            return;
        }
        observed = previous;
    }
}

[[nodiscard]] __device__ inline bool
immutable_header_contract_is_valid(const shared::QueueHeader& header,
                                   const std::uint32_t queue_kind,
                                   const std::uint32_t slot_stride) noexcept {
    return header.abi_major == shared::HostDeviceTransportAbiMajor &&
           header.abi_minor == shared::HostDeviceTransportAbiMinor &&
           header.struct_size == sizeof(shared::QueueHeader) && header.queue_kind == queue_kind &&
           header.capacity == slot_stride && header.reserved == 0U;
}

} // namespace wavefront_queue_device_detail

[[nodiscard]] __device__ inline WavefrontQueueDevicePushStatus
try_push_wavefront_queue(const WavefrontQueueDeviceSoa queues, const std::uint32_t queue_kind,
                         const shared::PathSlot slot) noexcept {
    if (queues.headers == nullptr || queues.queue_count != CudaWavefrontQueueCount ||
        queue_kind >= CudaWavefrontQueueCount ||
        (queues.slot_stride != 0U && queues.path_slots == nullptr)) {
        return WavefrontQueueDevicePushStatus::invalid_contract;
    }

    auto& header = queues.headers[queue_kind];
    if (!wavefront_queue_device_detail::immutable_header_contract_is_valid(header, queue_kind,
                                                                           queues.slot_stride)) {
        return WavefrontQueueDevicePushStatus::invalid_contract;
    }

    auto observed = wavefront_queue_device_detail::atomic_load(&header.size);
    while (observed < header.capacity) {
        const auto previous = atomicCAS(&header.size, observed, observed + 1U);
        if (previous == observed) {
            const auto destination = static_cast<std::uint64_t>(queue_kind) * queues.slot_stride +
                                     static_cast<std::uint64_t>(observed);
            queues.path_slots[destination] = slot;
            return WavefrontQueueDevicePushStatus::pushed;
        }
        observed = previous;
    }

    if (observed > header.capacity) {
        return WavefrontQueueDevicePushStatus::invalid_contract;
    }
    wavefront_queue_device_detail::atomic_saturating_increment(&header.overflow_count);
    wavefront_queue_device_detail::atomic_saturating_increment(&header.rejected_count);
    return WavefrontQueueDevicePushStatus::capacity_exhausted;
}

} // namespace blackframe::xpu::cuda
