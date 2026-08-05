#pragma once

#include <Blackframe/XPU/Shared/TransportAbi.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace blackframe::xpu::cuda {

inline constexpr std::uint32_t CudaWavefrontQueueCount = 7U;

enum class WavefrontQueueDevicePushStatus : std::uint32_t {
    pushed = 0U,
    capacity_exhausted = 1U,
    invalid_contract = 2U,
};

// Backend-local launch view. The pointers name independent arrays on the
// active CUDA device: seven headers and seven queue-major PathSlot columns.
// A kernel boundary is the publication barrier between producers and
// consumers; simultaneous consumption is outside this contract.
struct WavefrontQueueDeviceSoa final {
    shared::QueueHeader* headers{};
    shared::PathSlot* path_slots{};
    std::uint32_t queue_count{};
    std::uint32_t slot_stride{};
};

struct WavefrontQueueDevicePush final {
    std::uint32_t queue_kind{};
    shared::PathSlot slot{};
};

static_assert(sizeof(WavefrontQueueDevicePushStatus) == sizeof(std::uint32_t));
static_assert(alignof(WavefrontQueueDevicePushStatus) == alignof(std::uint32_t));
static_assert(std::is_standard_layout_v<WavefrontQueueDeviceSoa>);
static_assert(std::is_trivially_copyable_v<WavefrontQueueDeviceSoa>);
static_assert(std::is_standard_layout_v<WavefrontQueueDevicePush>);
static_assert(std::is_trivially_copyable_v<WavefrontQueueDevicePush>);
static_assert(std::is_trivially_destructible_v<WavefrontQueueDevicePush>);
static_assert(sizeof(WavefrontQueueDevicePush) == 8U);
static_assert(alignof(WavefrontQueueDevicePush) == 4U);
static_assert(offsetof(WavefrontQueueDevicePush, queue_kind) == 0U);
static_assert(offsetof(WavefrontQueueDevicePush, slot) == 4U);

} // namespace blackframe::xpu::cuda

// All pointers name storage on the active CUDA device. requests and outcomes contain
// request_count elements; those arrays may not overlap each other or queue storage. Every queue
// uses the same slot_stride capacity. The launch uses the default stream and is asynchronous;
// reset and consumption must be ordered after it. Reservation order and the winning subset under
// overflow are intentionally unspecified across CUDA threads.
extern "C" int blackframe_cuda_launch_wavefront_queue_pushes(
    blackframe::xpu::shared::QueueHeader* headers, blackframe::xpu::shared::PathSlot* path_slots,
    std::uint32_t queue_count, std::uint32_t slot_stride,
    const blackframe::xpu::cuda::WavefrontQueueDevicePush* requests, std::uint32_t request_count,
    blackframe::xpu::cuda::WavefrontQueueDevicePushStatus* outcomes) noexcept;

namespace blackframe::xpu::cuda {

[[nodiscard]] inline int launch_wavefront_queue_pushes(
    const WavefrontQueueDeviceSoa queues, const WavefrontQueueDevicePush* const requests,
    const std::uint32_t request_count, WavefrontQueueDevicePushStatus* const outcomes) noexcept {
    return blackframe_cuda_launch_wavefront_queue_pushes(queues.headers, queues.path_slots,
                                                         queues.queue_count, queues.slot_stride,
                                                         requests, request_count, outcomes);
}

} // namespace blackframe::xpu::cuda
