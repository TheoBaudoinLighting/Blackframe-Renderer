#include <Blackframe/XPU/CUDA/WavefrontQueueCompaction.hpp>
#include <Blackframe/XPU/CUDA/WavefrontQueueDevice.cuh>
#include <cstddef>
#include <cstdint>
#include <cub/device/device_scan.cuh>
#include <cuda/iterator>
#include <cuda_runtime_api.h>
#include <limits>

#if !defined(__CUDACC__)
#error "The CUDA wavefront queue compaction kernel must be compiled by the CUDA compiler."
#endif

static_assert(__cplusplus == 202002L);

#if defined(__cpp_pack_indexing)
#error "C++26 features are forbidden in CUDA wavefront queue compaction code."
#endif

namespace {

namespace bf_cuda = blackframe::xpu::cuda;
namespace shared = blackframe::xpu::shared;

using bf_cuda::WavefrontQueueCompactionResult;
using bf_cuda::WavefrontQueueCompactionStatus;
using bf_cuda::WavefrontQueueDeviceSoa;
using bf_cuda::WavefrontStageOutcome;
using bf_cuda::WavefrontStageStatus;

constexpr auto ThreadsPerBlock = std::uint32_t{256U};
constexpr auto FirstCompactableRoute = std::uint32_t{1U};
constexpr auto LastCompactableRoute = std::uint32_t{6U};

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
value(const WavefrontQueueCompactionStatus status) noexcept {
    return static_cast<std::uint32_t>(status);
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
value(const WavefrontStageStatus status) noexcept {
    return static_cast<std::uint32_t>(status);
}

[[nodiscard]] constexpr bool valid_route(const std::uint32_t route) noexcept {
    return route >= FirstCompactableRoute && route <= LastCompactableRoute;
}

[[nodiscard]] constexpr std::size_t align_up(const std::size_t size,
                                             const std::size_t alignment) noexcept {
    const auto remainder = size % alignment;
    return remainder == 0U ? size : size + alignment - remainder;
}

struct SelectSuccessfulRoute final {
    std::uint32_t route{};

    [[nodiscard]] __host__ __device__ std::uint32_t
    operator()(const WavefrontStageOutcome& outcome) const noexcept {
        return outcome.status == value(WavefrontStageStatus::success) && outcome.route == route
                   ? 1U
                   : 0U;
    }
};

using SelectionIterator =
    ::cuda::transform_iterator<SelectSuccessfulRoute, const WavefrontStageOutcome*>;

[[nodiscard]] cudaError_t query_scan_temp_bytes(const std::uint32_t input_count,
                                                std::size_t& temp_bytes) noexcept {
    temp_bytes = 0U;
    if (input_count == 0U) {
        return cudaSuccess;
    }
    const auto selection = SelectionIterator{static_cast<const WavefrontStageOutcome*>(nullptr),
                                             SelectSuccessfulRoute{.route = FirstCompactableRoute}};
    return cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, selection,
                                         static_cast<std::uint32_t*>(nullptr), input_count);
}

[[nodiscard]] cudaError_t minimum_scratch_bytes(const std::uint32_t input_count,
                                                std::size_t& scratch_bytes) noexcept {
    scratch_bytes = 0U;
    if (input_count == 0U) {
        return cudaSuccess;
    }
    constexpr auto alignment = bf_cuda::WavefrontQueueCompactionScratchAlignment;
    constexpr auto element_bytes = sizeof(std::uint32_t);
    if (static_cast<std::size_t>(input_count) >
        (std::numeric_limits<std::size_t>::max() - (alignment - 1U)) / element_bytes) {
        return cudaErrorInvalidValue;
    }
    const auto offset_bytes =
        align_up(static_cast<std::size_t>(input_count) * element_bytes, alignment);
    auto scan_temp_bytes = std::size_t{};
    const auto status = query_scan_temp_bytes(input_count, scan_temp_bytes);
    if (status != cudaSuccess) {
        return status;
    }
    if (scan_temp_bytes > std::numeric_limits<std::size_t>::max() - offset_bytes) {
        return cudaErrorInvalidValue;
    }
    scratch_bytes = offset_bytes + scan_temp_bytes;
    return cudaSuccess;
}

[[nodiscard]] bool host_queue_view_is_valid(const WavefrontQueueDeviceSoa queues) noexcept {
    return queues.headers != nullptr && queues.queue_count == bf_cuda::CudaWavefrontQueueCount &&
           (queues.slot_stride == 0U || queues.path_slots != nullptr);
}

__global__ void initialize_compaction_kernel(const WavefrontQueueDeviceSoa queues,
                                             const std::uint32_t route,
                                             const std::uint32_t input_count,
                                             WavefrontQueueCompactionResult* const result) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) {
        return;
    }

    auto initialized = WavefrontQueueCompactionResult{
        .status = value(WavefrontQueueCompactionStatus::success),
        .queue_kind = route,
        .route = route,
        .input_count = input_count,
    };
    const auto& header = queues.headers[route];
    initialized.initial_size = header.size;
    if (!bf_cuda::wavefront_queue_device_detail::immutable_header_contract_is_valid(
            header, route, queues.slot_stride) ||
        header.size > header.capacity || header.overflow_count != 0U ||
        header.rejected_count != 0U) {
        initialized.status = value(WavefrontQueueCompactionStatus::invalid_contract);
    }
    *result = initialized;
}

__global__ void validate_selected_slots_kernel(const WavefrontStageOutcome* const outcomes,
                                               const std::uint32_t input_count,
                                               const std::uint32_t route,
                                               const std::uint32_t path_capacity,
                                               WavefrontQueueCompactionResult* const result) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= input_count) {
        return;
    }
    const auto outcome = outcomes[index];
    if (outcome.status == value(WavefrontStageStatus::success) && outcome.route == route &&
        outcome.path_slot >= path_capacity) {
        atomicCAS(&result->status, value(WavefrontQueueCompactionStatus::success),
                  value(WavefrontQueueCompactionStatus::invalid_contract));
    }
}

__global__ void decide_compaction_kernel(const WavefrontQueueDeviceSoa queues,
                                         const WavefrontStageOutcome* const outcomes,
                                         const std::uint32_t* const offsets,
                                         const std::uint32_t input_count,
                                         WavefrontQueueCompactionResult* const result) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) {
        return;
    }

    auto selected_count = std::uint32_t{};
    if (input_count != 0U) {
        const auto last_index = input_count - 1U;
        const auto last_outcome = outcomes[last_index];
        const auto selected_last = last_outcome.status == value(WavefrontStageStatus::success) &&
                                           last_outcome.route == result->route
                                       ? 1U
                                       : 0U;
        selected_count = offsets[last_index] + selected_last;
    }
    result->selected_count = selected_count;
    if (result->status != value(WavefrontQueueCompactionStatus::success)) {
        return;
    }

    auto& header = queues.headers[result->queue_kind];
    if (!bf_cuda::wavefront_queue_device_detail::immutable_header_contract_is_valid(
            header, result->queue_kind, queues.slot_stride) ||
        header.size != result->initial_size || header.overflow_count != 0U ||
        header.rejected_count != 0U) {
        result->status = value(WavefrontQueueCompactionStatus::invalid_contract);
        return;
    }
    if (selected_count > header.capacity - header.size) {
        bf_cuda::wavefront_queue_device_detail::atomic_saturating_add(&header.overflow_count,
                                                                      selected_count);
        bf_cuda::wavefront_queue_device_detail::atomic_saturating_add(&header.rejected_count,
                                                                      selected_count);
        result->status = value(WavefrontQueueCompactionStatus::capacity_exhausted);
        result->rejected_count = selected_count;
    }
}

__global__ void scatter_compacted_slots_kernel(const WavefrontQueueDeviceSoa queues,
                                               const WavefrontStageOutcome* const outcomes,
                                               const std::uint32_t* const offsets,
                                               const std::uint32_t input_count,
                                               const WavefrontQueueCompactionResult* const result) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= input_count || result->status != value(WavefrontQueueCompactionStatus::success)) {
        return;
    }
    const auto outcome = outcomes[index];
    if (outcome.status != value(WavefrontStageStatus::success) || outcome.route != result->route) {
        return;
    }
    const auto destination = static_cast<std::uint64_t>(result->queue_kind) * queues.slot_stride +
                             result->initial_size + offsets[index];
    queues.path_slots[destination] = shared::PathSlot{.value = outcome.path_slot};
}

__global__ void publish_compaction_kernel(const WavefrontQueueDeviceSoa queues,
                                          WavefrontQueueCompactionResult* const result) {
    if (blockIdx.x != 0U || threadIdx.x != 0U ||
        result->status != value(WavefrontQueueCompactionStatus::success)) {
        return;
    }

    auto& header = queues.headers[result->queue_kind];
    if (!bf_cuda::wavefront_queue_device_detail::immutable_header_contract_is_valid(
            header, result->queue_kind, queues.slot_stride) ||
        header.size != result->initial_size || header.overflow_count != 0U ||
        header.rejected_count != 0U || result->selected_count > header.capacity - header.size) {
        result->status = value(WavefrontQueueCompactionStatus::invalid_contract);
        return;
    }
    header.size += result->selected_count;
    result->published_count = result->selected_count;
}

[[nodiscard]] constexpr std::uint32_t block_count(const std::uint32_t work_count) noexcept {
    return static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(work_count) + ThreadsPerBlock - 1U) / ThreadsPerBlock);
}

} // namespace

extern "C" int blackframe_cuda_query_wavefront_queue_compaction_scratch_bytes(
    const std::uint32_t max_input_count, std::size_t* const scratch_bytes) noexcept {
    if (scratch_bytes == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    auto required_bytes = std::size_t{};
    const auto status = minimum_scratch_bytes(max_input_count, required_bytes);
    if (status == cudaSuccess) {
        *scratch_bytes = required_bytes;
    }
    return static_cast<int>(status);
}

extern "C" int blackframe_cuda_launch_wavefront_queue_compaction(
    const WavefrontQueueDeviceSoa queues, const WavefrontStageOutcome* const outcomes,
    const std::uint32_t input_count, const std::uint32_t route, void* const scratch,
    const std::size_t scratch_bytes, WavefrontQueueCompactionResult* const device_result) noexcept {
    if (!host_queue_view_is_valid(queues) || !valid_route(route) ||
        input_count > queues.slot_stride || device_result == nullptr ||
        (input_count != 0U && outcomes == nullptr)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    auto required_bytes = std::size_t{};
    auto status = minimum_scratch_bytes(input_count, required_bytes);
    if (status != cudaSuccess || scratch_bytes < required_bytes ||
        (required_bytes != 0U &&
         (scratch == nullptr || reinterpret_cast<std::uintptr_t>(scratch) %
                                        bf_cuda::WavefrontQueueCompactionScratchAlignment !=
                                    0U))) {
        return static_cast<int>(status == cudaSuccess ? cudaErrorInvalidValue : status);
    }

    initialize_compaction_kernel<<<1U, 1U>>>(queues, route, input_count, device_result);
    status = cudaGetLastError();
    if (status != cudaSuccess || input_count == 0U) {
        return static_cast<int>(status);
    }

    validate_selected_slots_kernel<<<block_count(input_count), ThreadsPerBlock>>>(
        outcomes, input_count, route, queues.slot_stride, device_result);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    auto* const scratch_bytes_begin = static_cast<std::uint8_t*>(scratch);
    auto* const offsets = reinterpret_cast<std::uint32_t*>(scratch_bytes_begin);
    const auto temp_offset = align_up(static_cast<std::size_t>(input_count) * sizeof(*offsets),
                                      bf_cuda::WavefrontQueueCompactionScratchAlignment);
    auto* const scan_temp = scratch_bytes_begin + temp_offset;
    auto scan_temp_bytes = scratch_bytes - temp_offset;
    const auto selection = SelectionIterator{outcomes, SelectSuccessfulRoute{.route = route}};
    status =
        cub::DeviceScan::ExclusiveSum(scan_temp, scan_temp_bytes, selection, offsets, input_count);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    decide_compaction_kernel<<<1U, 1U>>>(queues, outcomes, offsets, input_count, device_result);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    scatter_compacted_slots_kernel<<<block_count(input_count), ThreadsPerBlock>>>(
        queues, outcomes, offsets, input_count, device_result);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    publish_compaction_kernel<<<1U, 1U>>>(queues, device_result);
    return static_cast<int>(cudaGetLastError());
}
