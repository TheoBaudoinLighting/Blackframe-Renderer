#include <Blackframe/Backends/GPU/CUDA/WavefrontQueues.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace blackframe::engine {
namespace {

using xpu::shared::PathSlot;
using xpu::shared::QueueHeader;

static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::camera) == 0U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::ray) == 1U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::hit) == 2U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::miss) == 3U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::shade) == 4U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::shadow) == 5U);
static_assert(static_cast<std::uint32_t>(renderer::WavefrontQueueKind::continuation) == 6U);

[[nodiscard]] core::Error queue_error(const core::StatusCode code, std::string message) {
    return core::Error{.code = code, .message = std::move(message)};
}

[[nodiscard]] core::Error cuda_queue_error(const cudaError_t status, const char* operation,
                                           const std::size_t byte_count) {
    return queue_error(xpu::cuda::cuda_memory_status_code(static_cast<std::int32_t>(status)),
                       std::string{"CUDA wavefront queue "} + operation + " failed for " +
                           std::to_string(byte_count) + " bytes: " + cudaGetErrorName(status) +
                           " (" + cudaGetErrorString(status) + ").");
}

[[nodiscard]] constexpr std::array<QueueHeader, xpu::cuda::CudaWavefrontQueueCount>
make_queue_headers(const std::uint32_t capacity) noexcept {
    auto headers = std::array<QueueHeader, xpu::cuda::CudaWavefrontQueueCount>{};
    for (auto queue_kind = std::uint32_t{0}; queue_kind < headers.size(); ++queue_kind) {
        headers[queue_kind] = QueueHeader{
            .abi_major = xpu::shared::HostDeviceTransportAbiMajor,
            .abi_minor = xpu::shared::HostDeviceTransportAbiMinor,
            .struct_size = sizeof(QueueHeader),
            .queue_kind = queue_kind,
            .capacity = capacity,
            .size = 0U,
            .overflow_count = 0U,
            .rejected_count = 0U,
            .reserved = 0U,
        };
    }
    return headers;
}

[[nodiscard]] core::Status require_active_device(const std::int32_t owner_device) {
    auto active_device = int{-1};
    const auto status = cudaGetDevice(&active_device);
    if (status != cudaSuccess) {
        return std::unexpected(cuda_queue_error(status, "device query", 0U));
    }
    if (owner_device < 0 || active_device != owner_device) {
        return std::unexpected(
            queue_error(core::StatusCode::invalid_argument,
                        "CUDA wavefront queue operation requires its owning device to be active."));
    }
    return {};
}

[[nodiscard]] core::Result<std::size_t>
checked_slot_count(const std::size_t capacity,
                   const xpu::cuda::DeviceMemoryBudget device_memory_budget) {
    if (capacity > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(
            queue_error(core::StatusCode::resource_exhausted,
                        "CUDA wavefront queue capacity exceeds its fixed 32-bit device contract."));
    }
    constexpr auto queue_count = static_cast<std::size_t>(xpu::cuda::CudaWavefrontQueueCount);
    if (capacity > std::numeric_limits<std::size_t>::max() / queue_count) {
        return std::unexpected(queue_error(core::StatusCode::resource_exhausted,
                                           "CUDA wavefront queue slot count overflowed."));
    }
    const auto slot_count = capacity * queue_count;
    if (slot_count > std::numeric_limits<std::size_t>::max() / sizeof(PathSlot)) {
        return std::unexpected(queue_error(core::StatusCode::resource_exhausted,
                                           "CUDA wavefront queue slot byte count overflowed."));
    }
    const auto slot_bytes = slot_count * sizeof(PathSlot);
    constexpr auto header_bytes = queue_count * sizeof(QueueHeader);
    if (slot_bytes > std::numeric_limits<std::size_t>::max() - header_bytes) {
        return std::unexpected(
            queue_error(core::StatusCode::resource_exhausted,
                        "CUDA wavefront queue aggregate byte count overflowed."));
    }
    const auto total_bytes = header_bytes + slot_bytes;
    if (total_bytes > device_memory_budget.maximum_bytes ||
        total_bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        return std::unexpected(queue_error(
            core::StatusCode::resource_exhausted,
            "CUDA wavefront queues exceed their explicit aggregate device-memory budget."));
    }
    return slot_count;
}

} // namespace

CudaWavefrontQueues::CudaWavefrontQueues(xpu::cuda::DeviceBuffer<QueueHeader> headers,
                                         xpu::cuda::DeviceBuffer<PathSlot> path_slots,
                                         const std::uint32_t capacity) noexcept
    : headers_(std::move(headers)), path_slots_(std::move(path_slots)), capacity_(capacity) {}

CudaWavefrontQueues::CudaWavefrontQueues(CudaWavefrontQueues&& other) noexcept
    : headers_(std::move(other.headers_)), path_slots_(std::move(other.path_slots_)),
      capacity_(std::exchange(other.capacity_, 0U)) {}

core::Result<CudaWavefrontQueues>
CudaWavefrontQueues::create(const std::size_t capacity,
                            const CudaWavefrontQueueCreateOptions options) {
    auto slot_count = checked_slot_count(capacity, options.device_memory_budget);
    if (!slot_count) {
        return std::unexpected(std::move(slot_count.error()));
    }

    auto headers = xpu::cuda::DeviceBuffer<QueueHeader>::allocate(
        xpu::cuda::CudaWavefrontQueueCount, options.device_memory_budget);
    if (!headers) {
        return std::unexpected(std::move(headers.error()));
    }
    auto path_slots =
        xpu::cuda::DeviceBuffer<PathSlot>::allocate(*slot_count, options.device_memory_budget);
    if (!path_slots) {
        return std::unexpected(std::move(path_slots.error()));
    }
    if (*slot_count != 0U && headers->device_ordinal() != path_slots->device_ordinal()) {
        return std::unexpected(queue_error(
            core::StatusCode::internal_error,
            "CUDA wavefront queue allocations unexpectedly landed on different devices."));
    }

    const auto device_capacity = static_cast<std::uint32_t>(capacity);
    const auto initial_headers = make_queue_headers(device_capacity);
    const auto copy_status = cudaMemcpy(headers->data(), initial_headers.data(),
                                        sizeof(initial_headers), cudaMemcpyHostToDevice);
    if (copy_status != cudaSuccess) {
        return std::unexpected(
            cuda_queue_error(copy_status, "initialization", sizeof(initial_headers)));
    }
    return CudaWavefrontQueues{std::move(*headers), std::move(*path_slots), device_capacity};
}

xpu::cuda::WavefrontQueueDeviceSoa CudaWavefrontQueues::device_view() noexcept {
    if (!*this) {
        return {};
    }
    return xpu::cuda::WavefrontQueueDeviceSoa{
        .headers = headers_.data(),
        .path_slots = path_slots_.data(),
        .queue_count = xpu::cuda::CudaWavefrontQueueCount,
        .slot_stride = capacity_,
    };
}

core::Result<CudaWavefrontQueueSnapshots> CudaWavefrontQueues::download() const try {
    if (!*this) {
        return std::unexpected(queue_error(core::StatusCode::invalid_argument,
                                           "Cannot download closed CUDA wavefront queues."));
    }
    auto active_status = require_active_device(device_ordinal());
    if (!active_status) {
        return std::unexpected(std::move(active_status.error()));
    }

    auto headers = std::array<QueueHeader, xpu::cuda::CudaWavefrontQueueCount>{};
    auto copy_status =
        cudaMemcpy(headers.data(), headers_.data(), sizeof(headers), cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess) {
        return std::unexpected(cuda_queue_error(copy_status, "counter download", sizeof(headers)));
    }

    auto snapshots = CudaWavefrontQueueSnapshots{};
    for (auto queue_kind = std::uint32_t{0}; queue_kind < headers.size(); ++queue_kind) {
        const auto& header = headers[queue_kind];
        if (xpu::shared::validate_queue_header(header) !=
                xpu::shared::QueueHeaderValidationStatus::valid ||
            header.queue_kind != queue_kind || header.capacity != capacity_ ||
            header.overflow_count != header.rejected_count) {
            return std::unexpected(queue_error(
                core::StatusCode::incompatible,
                "CUDA wavefront queue counters violate their fixed host/device contract."));
        }

        auto& snapshot = snapshots[queue_kind];
        snapshot.counters = CudaWavefrontQueueCounters{
            .kind = static_cast<renderer::WavefrontQueueKind>(queue_kind),
            .capacity = header.capacity,
            .size = header.size,
            .overflow_count = header.overflow_count,
            .rejected_count = header.rejected_count,
        };
        snapshot.entries.resize(header.size);
        if (header.size == 0U) {
            continue;
        }
        auto device_entries = std::vector<PathSlot>(header.size);
        const auto source_offset = static_cast<std::size_t>(queue_kind) * capacity_;
        copy_status = cudaMemcpy(device_entries.data(), path_slots_.data() + source_offset,
                                 static_cast<std::size_t>(header.size) * sizeof(PathSlot),
                                 cudaMemcpyDeviceToHost);
        if (copy_status != cudaSuccess) {
            return std::unexpected(
                cuda_queue_error(copy_status, "entry download",
                                 static_cast<std::size_t>(header.size) * sizeof(PathSlot)));
        }
        for (auto entry_index = std::size_t{0}; entry_index < device_entries.size();
             ++entry_index) {
            snapshot.entries[entry_index].value = device_entries[entry_index].value;
        }
    }
    return snapshots;
} catch (const std::bad_alloc&) {
    return std::unexpected(queue_error(core::StatusCode::resource_exhausted,
                                       "CUDA wavefront queue download exhausted host memory."));
} catch (const std::length_error&) {
    return std::unexpected(
        queue_error(core::StatusCode::resource_exhausted,
                    "CUDA wavefront queue download exceeded a host container length limit."));
}

core::Status CudaWavefrontQueues::reset(const CudaWavefrontQueueResetPolicy policy) {
    if (!*this) {
        return std::unexpected(queue_error(core::StatusCode::invalid_argument,
                                           "Cannot reset closed CUDA wavefront queues."));
    }
    switch (policy) {
    case CudaWavefrontQueueResetPolicy::require_no_overflow:
    case CudaWavefrontQueueResetPolicy::acknowledge_overflow:
        break;
    default:
        return std::unexpected(queue_error(core::StatusCode::invalid_argument,
                                           "Unknown CUDA wavefront queue reset policy."));
    }
    auto active_status = require_active_device(device_ordinal());
    if (!active_status) {
        return active_status;
    }

    auto current_headers = std::array<QueueHeader, xpu::cuda::CudaWavefrontQueueCount>{};
    auto copy_status = cudaMemcpy(current_headers.data(), headers_.data(), sizeof(current_headers),
                                  cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess) {
        return std::unexpected(
            cuda_queue_error(copy_status, "counter validation", sizeof(current_headers)));
    }
    for (auto queue_kind = std::uint32_t{0}; queue_kind < current_headers.size(); ++queue_kind) {
        const auto& header = current_headers[queue_kind];
        if (xpu::shared::validate_queue_header(header) !=
                xpu::shared::QueueHeaderValidationStatus::valid ||
            header.queue_kind != queue_kind || header.capacity != capacity_ ||
            header.overflow_count != header.rejected_count) {
            return std::unexpected(queue_error(
                core::StatusCode::incompatible,
                "CUDA wavefront queue reset found counters that violate the device contract."));
        }
        if (policy == CudaWavefrontQueueResetPolicy::require_no_overflow &&
            header.rejected_count != 0U) {
            return std::unexpected(queue_error(
                core::StatusCode::resource_exhausted,
                "CUDA wavefront queue reset refused to erase an unacknowledged overflow."));
        }
    }

    const auto initial_headers = make_queue_headers(capacity_);
    copy_status = cudaMemcpy(headers_.data(), initial_headers.data(), sizeof(initial_headers),
                             cudaMemcpyHostToDevice);
    if (copy_status != cudaSuccess) {
        return std::unexpected(cuda_queue_error(copy_status, "reset", sizeof(initial_headers)));
    }
    return {};
}

core::Status CudaWavefrontQueues::close() {
    auto first_error = core::Status{};
    auto slot_status = path_slots_.close();
    if (!slot_status) {
        first_error = std::unexpected(std::move(slot_status.error()));
    }
    auto header_status = headers_.close();
    if (!header_status && first_error) {
        first_error = std::unexpected(std::move(header_status.error()));
    }
    if (headers_.empty() && path_slots_.empty()) {
        capacity_ = 0U;
    }
    return first_error;
}

} // namespace blackframe::engine
