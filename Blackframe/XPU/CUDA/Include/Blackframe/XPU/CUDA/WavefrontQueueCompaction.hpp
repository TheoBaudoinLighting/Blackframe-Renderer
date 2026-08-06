#pragma once

#include <Blackframe/XPU/CUDA/WavefrontStageKernel.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace blackframe::xpu::cuda {

inline constexpr std::size_t WavefrontQueueCompactionScratchAlignment = 256U;

enum class WavefrontQueueCompactionStatus : std::uint32_t {
    success = 0U,
    invalid_contract = 1U,
    capacity_exhausted = 2U,
};

// Device-written summary for one stable append. A capacity failure rejects the complete selected
// set: published_count is zero, the destination size is unchanged, and rejected_count reports the
// number of lanes reflected in the queue diagnostics.
struct alignas(16) WavefrontQueueCompactionResult final {
    std::uint32_t status{};
    std::uint32_t queue_kind{};
    std::uint32_t route{};
    std::uint32_t input_count{};
    std::uint32_t initial_size{};
    std::uint32_t selected_count{};
    std::uint32_t published_count{};
    std::uint32_t rejected_count{};
    std::uint32_t reserved[4U]{};
};

static_assert(sizeof(WavefrontQueueCompactionStatus) == sizeof(std::uint32_t));
static_assert(alignof(WavefrontQueueCompactionStatus) == alignof(std::uint32_t));
static_assert(std::is_standard_layout_v<WavefrontQueueCompactionResult>);
static_assert(std::is_trivially_copyable_v<WavefrontQueueCompactionResult>);
static_assert(std::is_trivially_destructible_v<WavefrontQueueCompactionResult>);
static_assert(sizeof(WavefrontQueueCompactionResult) == 48U);
static_assert(alignof(WavefrontQueueCompactionResult) == 16U);
static_assert(offsetof(WavefrontQueueCompactionResult, status) == 0U);
static_assert(offsetof(WavefrontQueueCompactionResult, queue_kind) == 4U);
static_assert(offsetof(WavefrontQueueCompactionResult, route) == 8U);
static_assert(offsetof(WavefrontQueueCompactionResult, input_count) == 12U);
static_assert(offsetof(WavefrontQueueCompactionResult, initial_size) == 16U);
static_assert(offsetof(WavefrontQueueCompactionResult, selected_count) == 20U);
static_assert(offsetof(WavefrontQueueCompactionResult, published_count) == 24U);
static_assert(offsetof(WavefrontQueueCompactionResult, rejected_count) == 28U);
static_assert(offsetof(WavefrontQueueCompactionResult, reserved) == 32U);

} // namespace blackframe::xpu::cuda

// Queries the minimum device scratch capacity for any input_count not greater than
// max_input_count. A zero maximum requires no scratch. The launch accepts a larger allocation so a
// workspace sized for its capacity can serve smaller active prefixes. Non-zero scratch must begin
// at WavefrontQueueCompactionScratchAlignment.
extern "C" int
blackframe_cuda_query_wavefront_queue_compaction_scratch_bytes(std::uint32_t max_input_count,
                                                               std::size_t* scratch_bytes) noexcept;

// Selects successful outcomes whose route equals route, computes a stable exclusive prefix scan,
// and appends their path slots to the queue whose numeric kind equals route. routes 1 through 6 are
// accepted. All pointers name non-overlapping storage on the active CUDA device. The default stream
// orders the scan, transactional capacity decision, and scatter. Invalid launch arguments are
// rejected synchronously; device-side contract failures are written to device_result.
extern "C" int blackframe_cuda_launch_wavefront_queue_compaction(
    blackframe::xpu::cuda::WavefrontQueueDeviceSoa queues,
    const blackframe::xpu::cuda::WavefrontStageOutcome* outcomes, std::uint32_t input_count,
    std::uint32_t route, void* scratch, std::size_t scratch_bytes,
    blackframe::xpu::cuda::WavefrontQueueCompactionResult* device_result) noexcept;

namespace blackframe::xpu::cuda {

[[nodiscard]] inline int
query_wavefront_queue_compaction_scratch_bytes(const std::uint32_t max_input_count,
                                               std::size_t* const scratch_bytes) noexcept {
    return blackframe_cuda_query_wavefront_queue_compaction_scratch_bytes(max_input_count,
                                                                          scratch_bytes);
}

[[nodiscard]] inline int launch_wavefront_queue_compaction(
    const WavefrontQueueDeviceSoa queues, const WavefrontStageOutcome* const outcomes,
    const std::uint32_t input_count, const std::uint32_t route, void* const scratch,
    const std::size_t scratch_bytes, WavefrontQueueCompactionResult* const device_result) noexcept {
    return blackframe_cuda_launch_wavefront_queue_compaction(queues, outcomes, input_count, route,
                                                             scratch, scratch_bytes, device_result);
}

} // namespace blackframe::xpu::cuda
