#pragma once

#include <Blackframe/XPU/CUDA/WavefrontQueueKernel.hpp>
#include <cstdint>
#include <cuda_runtime.h>

#if !defined(__CUDACC__)
#error "The CUDA wavefront queue device contract requires the CUDA compiler."
#endif

namespace blackframe::xpu::cuda {
namespace wavefront_queue_device_detail {

[[nodiscard]] __device__ inline std::uint32_t warp_lane_index() noexcept {
    const auto linear_thread_index =
        threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    return linear_thread_index & 31U;
}

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

__device__ inline void atomic_saturating_add(std::uint32_t* const value,
                                             const std::uint32_t increment) noexcept {
    if (increment == 0U) {
        return;
    }

    auto observed = atomic_load(value);
    constexpr auto maximum = ~std::uint32_t{0};
    while (observed != maximum) {
        const auto remaining = maximum - observed;
        const auto desired = increment < remaining ? observed + increment : maximum;
        const auto previous = atomicCAS(value, observed, desired);
        if (previous == observed) {
            return;
        }
        observed = previous;
    }
}

[[nodiscard]] __device__ inline std::uint32_t
match_any_queue_kind(const std::uint32_t active_mask, const std::uint32_t queue_kind) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    return __match_any_sync(active_mask, queue_kind);
#else
    // Exact match-any equivalent for architectures predating the native intrinsic. Every active
    // lane executes every ballot so the synchronization mask remains valid throughout the loop.
    auto remaining = active_mask;
    auto matching = std::uint32_t{};
    while (remaining != 0U) {
        const auto leader = __ffs(static_cast<int>(remaining)) - 1;
        const auto candidate = __shfl_sync(active_mask, queue_kind, leader);
        const auto candidate_mask = __ballot_sync(active_mask, queue_kind == candidate);
        if ((candidate_mask & (1U << warp_lane_index())) != 0U) {
            matching = candidate_mask;
        }
        remaining &= ~candidate_mask;
    }
    return matching;
#endif
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

[[nodiscard]] __device__ inline WavefrontQueueDevicePushStatus
try_push_wavefront_queue_warp(const WavefrontQueueDeviceSoa queues, const std::uint32_t queue_kind,
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

    const auto active_mask = __activemask();
    const auto group_mask =
        wavefront_queue_device_detail::match_any_queue_kind(active_mask, queue_kind);
    const auto leader_lane = __ffs(static_cast<int>(group_mask)) - 1;
    const auto lane_index = wavefront_queue_device_detail::warp_lane_index();
    const auto lane_bit = 1U << lane_index;
    const auto lane_rank = __popc(group_mask & (lane_bit - 1U));
    const auto group_size = static_cast<std::uint32_t>(__popc(group_mask));

    auto reservation_base = std::uint32_t{};
    auto accepted_count = std::uint32_t{};
    auto reservation_valid = std::uint32_t{1U};
    if ((group_mask & lane_bit) != 0U && lane_index == leader_lane) {
        auto observed = wavefront_queue_device_detail::atomic_load(&header.size);
        for (;;) {
            if (observed > header.capacity) {
                reservation_valid = 0U;
                break;
            }

            const auto remaining = header.capacity - observed;
            accepted_count = remaining < group_size ? remaining : group_size;
            reservation_base = observed;
            if (accepted_count == 0U) {
                break;
            }

            const auto previous = atomicCAS(&header.size, observed, observed + accepted_count);
            if (previous == observed) {
                break;
            }
            observed = previous;
        }

        if (reservation_valid != 0U) {
            const auto rejected_count = group_size - accepted_count;
            wavefront_queue_device_detail::atomic_saturating_add(&header.overflow_count,
                                                                 rejected_count);
            wavefront_queue_device_detail::atomic_saturating_add(&header.rejected_count,
                                                                 rejected_count);
        }
    }

    reservation_base = __shfl_sync(active_mask, reservation_base, leader_lane);
    accepted_count = __shfl_sync(active_mask, accepted_count, leader_lane);
    reservation_valid = __shfl_sync(active_mask, reservation_valid, leader_lane);
    if (reservation_valid == 0U) {
        return WavefrontQueueDevicePushStatus::invalid_contract;
    }
    if (lane_rank >= accepted_count) {
        return WavefrontQueueDevicePushStatus::capacity_exhausted;
    }

    const auto destination = static_cast<std::uint64_t>(queue_kind) * queues.slot_stride +
                             static_cast<std::uint64_t>(reservation_base + lane_rank);
    queues.path_slots[destination] = slot;
    return WavefrontQueueDevicePushStatus::pushed;
}

} // namespace blackframe::xpu::cuda
