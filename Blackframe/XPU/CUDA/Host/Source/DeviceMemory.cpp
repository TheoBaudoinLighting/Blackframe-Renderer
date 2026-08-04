#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <limits>
#include <string>
#include <utility>

namespace blackframe::xpu::cuda {
namespace {

[[nodiscard]] core::Error memory_error(const core::StatusCode code, std::string message) {
    return core::Error{
        .code = code,
        .message = std::move(message),
    };
}

[[nodiscard]] core::Error cuda_memory_error(const cudaError_t status, const char* operation,
                                            const std::size_t byte_count) {
    auto message = std::string{operation} + " failed for " + std::to_string(byte_count) +
                   " bytes: " + cudaGetErrorName(status) + " (" + cudaGetErrorString(status) + ").";
    return memory_error(cuda_memory_status_code(static_cast<std::int32_t>(status)),
                        std::move(message));
}

[[nodiscard]] bool is_power_of_two(const std::size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

[[nodiscard]] core::Status validate_allocation_request(const std::size_t byte_count,
                                                       const DeviceMemoryBudget budget) {
    if (byte_count > budget.maximum_bytes) {
        return std::unexpected(
            memory_error(core::StatusCode::resource_exhausted,
                         "CUDA allocation exceeds its explicit device-memory budget."));
    }
    if (byte_count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        return std::unexpected(memory_error(core::StatusCode::resource_exhausted,
                                            "CUDA allocation exceeds the addressable range."));
    }
    return {};
}

} // namespace

core::StatusCode cuda_memory_status_code(const std::int32_t cuda_status) noexcept {
    const auto status = static_cast<cudaError_t>(cuda_status);
    if (status == cudaSuccess) {
        return core::StatusCode::success;
    }
    if (status == cudaErrorMemoryAllocation) {
        return core::StatusCode::resource_exhausted;
    }
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
        return core::StatusCode::unavailable;
    }
    return core::StatusCode::platform_error;
}

DeviceAllocation::DeviceAllocation(void* const data, const std::size_t size_bytes,
                                   const std::int32_t device_ordinal) noexcept
    : data_(data), size_bytes_(size_bytes), device_ordinal_(device_ordinal) {}

DeviceAllocation::~DeviceAllocation() noexcept {
    close_without_diagnostic();
}

DeviceAllocation::DeviceAllocation(DeviceAllocation&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_bytes_(std::exchange(other.size_bytes_, 0)),
      device_ordinal_(std::exchange(other.device_ordinal_, -1)) {}

core::Result<DeviceAllocation> DeviceAllocation::allocate_bytes(const std::size_t byte_count,
                                                                const DeviceMemoryBudget budget) {
    auto validation = validate_allocation_request(byte_count, budget);
    if (!validation) {
        return std::unexpected(std::move(validation.error()));
    }
    if (byte_count == 0) {
        return DeviceAllocation{};
    }

    int device_ordinal = -1;
    const auto status = cudaGetDevice(&device_ordinal);
    if (status != cudaSuccess) {
        return std::unexpected(cuda_memory_error(status, "cudaGetDevice", byte_count));
    }
    return allocate_bytes_on_device(byte_count, budget, device_ordinal);
}

core::Result<DeviceAllocation>
DeviceAllocation::allocate_bytes_on_device(const std::size_t byte_count,
                                           const DeviceMemoryBudget budget,
                                           const std::int32_t device_ordinal) {
    auto validation = validate_allocation_request(byte_count, budget);
    if (!validation) {
        return std::unexpected(std::move(validation.error()));
    }
    if (byte_count == 0) {
        return DeviceAllocation{};
    }

    int previous_device = -1;
    auto status = cudaGetDevice(&previous_device);
    if (status != cudaSuccess) {
        return std::unexpected(cuda_memory_error(status, "cudaGetDevice", byte_count));
    }

    const auto changed_device = previous_device != device_ordinal;
    if (changed_device) {
        status = cudaSetDevice(device_ordinal);
        if (status != cudaSuccess) {
            return std::unexpected(cuda_memory_error(status, "cudaSetDevice", byte_count));
        }
    }

    void* data = nullptr;
    const auto allocation_status = cudaMalloc(&data, byte_count);
    const auto restore_status =
        changed_device ? cudaSetDevice(previous_device) : static_cast<cudaError_t>(cudaSuccess);
    if (allocation_status != cudaSuccess) {
        if (restore_status != cudaSuccess) {
            return std::unexpected(
                cuda_memory_error(restore_status, "cudaSetDevice restore", byte_count));
        }
        return std::unexpected(cuda_memory_error(allocation_status, "cudaMalloc", byte_count));
    }
    if (restore_status != cudaSuccess) {
        DeviceAllocation cleanup_owner{data, byte_count, device_ordinal};
        auto cleanup_status = cleanup_owner.close();
        if (!cleanup_status) {
            return std::unexpected(std::move(cleanup_status.error()));
        }
        return std::unexpected(
            cuda_memory_error(restore_status, "cudaSetDevice restore", byte_count));
    }
    if (data == nullptr) {
        return std::unexpected(
            memory_error(core::StatusCode::internal_error,
                         "cudaMalloc reported success without returning device storage."));
    }

    return DeviceAllocation{data, byte_count, device_ordinal};
}

core::Status DeviceAllocation::close() {
    if (data_ == nullptr) {
        return {};
    }

    int previous_device = -1;
    auto status = cudaGetDevice(&previous_device);
    if (status != cudaSuccess) {
        return std::unexpected(cuda_memory_error(status, "cudaGetDevice", size_bytes_));
    }

    const auto changed_device = previous_device != device_ordinal_;
    if (changed_device) {
        status = cudaSetDevice(device_ordinal_);
        if (status != cudaSuccess) {
            return std::unexpected(cuda_memory_error(status, "cudaSetDevice", size_bytes_));
        }
    }

    const auto free_status = cudaFree(data_);
    const auto restore_status =
        changed_device ? cudaSetDevice(previous_device) : static_cast<cudaError_t>(cudaSuccess);
    if (free_status != cudaSuccess) {
        return std::unexpected(cuda_memory_error(free_status, "cudaFree", size_bytes_));
    }

    data_ = nullptr;
    size_bytes_ = 0;
    device_ordinal_ = -1;
    if (restore_status != cudaSuccess) {
        return std::unexpected(
            cuda_memory_error(restore_status, "cudaSetDevice restore", std::size_t{0}));
    }
    return {};
}

void DeviceAllocation::swap(DeviceAllocation& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_bytes_, other.size_bytes_);
    std::swap(device_ordinal_, other.device_ordinal_);
}

void DeviceAllocation::close_without_diagnostic() noexcept {
    if (data_ != nullptr) {
        int previous_device = -1;
        const auto get_status = cudaGetDevice(&previous_device);
        const auto changed_device = get_status == cudaSuccess && previous_device != device_ordinal_;
        const auto set_status =
            changed_device ? cudaSetDevice(device_ordinal_) : static_cast<cudaError_t>(cudaSuccess);
        if (set_status == cudaSuccess) {
            static_cast<void>(cudaFree(data_));
        }
        if (changed_device) {
            static_cast<void>(cudaSetDevice(previous_device));
        }
        data_ = nullptr;
        size_bytes_ = 0;
        device_ordinal_ = -1;
    }
}

DeviceScratchBuffer::DeviceScratchBuffer(DeviceAllocation allocation,
                                         const std::size_t capacity_bytes,
                                         const DeviceMemoryBudget budget) noexcept
    : allocation_(std::move(allocation)), capacity_bytes_(capacity_bytes), budget_(budget) {}

DeviceScratchBuffer::DeviceScratchBuffer(DeviceScratchBuffer&& other) noexcept
    : allocation_(std::move(other.allocation_)),
      capacity_bytes_(std::exchange(other.capacity_bytes_, 0)),
      used_bytes_(std::exchange(other.used_bytes_, 0)), budget_(other.budget_) {}

core::Result<DeviceScratchBuffer> DeviceScratchBuffer::create(const std::size_t capacity_bytes,
                                                              const DeviceMemoryBudget budget) {
    auto allocation = DeviceAllocation::allocate_bytes(capacity_bytes, budget);
    if (!allocation) {
        return std::unexpected(std::move(allocation.error()));
    }
    return DeviceScratchBuffer{std::move(*allocation), capacity_bytes, budget};
}

core::Status DeviceScratchBuffer::reserve(const std::size_t minimum_capacity_bytes) {
    if (minimum_capacity_bytes <= capacity_bytes_) {
        return {};
    }
    if (used_bytes_ != 0) {
        return std::unexpected(memory_error(
            core::StatusCode::invalid_argument,
            "CUDA scratch storage must be reset before a growth can invalidate its slices."));
    }

    auto replacement = allocation_.empty()
                           ? DeviceAllocation::allocate_bytes(minimum_capacity_bytes, budget_)
                           : DeviceAllocation::allocate_bytes_on_device(
                                 minimum_capacity_bytes, budget_, allocation_.device_ordinal());
    if (!replacement) {
        return std::unexpected(std::move(replacement.error()));
    }

    auto close_status = allocation_.close();
    if (!close_status) {
        if (allocation_.empty()) {
            capacity_bytes_ = 0;
            used_bytes_ = 0;
        }
        return std::unexpected(std::move(close_status.error()));
    }

    allocation_.swap(*replacement);
    capacity_bytes_ = minimum_capacity_bytes;
    return {};
}

core::Result<DeviceScratchSlice> DeviceScratchBuffer::allocate(const std::size_t byte_count,
                                                               const std::size_t alignment) {
    if (byte_count == 0) {
        return std::unexpected(memory_error(core::StatusCode::invalid_argument,
                                            "CUDA scratch slices cannot be empty."));
    }
    if (!is_power_of_two(alignment)) {
        return std::unexpected(
            memory_error(core::StatusCode::invalid_argument,
                         "CUDA scratch alignment must be a non-zero power of two."));
    }
    if (allocation_.empty()) {
        return std::unexpected(memory_error(core::StatusCode::resource_exhausted,
                                            "CUDA scratch storage has zero capacity."));
    }

    const auto base_address = reinterpret_cast<std::uintptr_t>(allocation_.data());
    if (used_bytes_ > std::numeric_limits<std::uintptr_t>::max() - base_address) {
        return std::unexpected(memory_error(core::StatusCode::resource_exhausted,
                                            "CUDA scratch address calculation overflowed."));
    }

    const auto current_address = base_address + used_bytes_;
    const auto remainder = current_address % alignment;
    const auto padding = remainder == 0 ? std::size_t{0} : alignment - remainder;
    if (padding > std::numeric_limits<std::uintptr_t>::max() - current_address) {
        return std::unexpected(memory_error(core::StatusCode::resource_exhausted,
                                            "CUDA scratch alignment overflowed."));
    }
    if (padding > remaining_bytes() || byte_count > remaining_bytes() - padding) {
        return std::unexpected(
            memory_error(core::StatusCode::resource_exhausted,
                         "CUDA scratch capacity is exhausted; explicit reserve is required."));
    }

    const auto offset = used_bytes_ + padding;
    const auto slice_address = current_address + padding;
    used_bytes_ = offset + byte_count;
    return DeviceScratchSlice{
        .data = reinterpret_cast<void*>(slice_address),
        .offset_bytes = offset,
        .size_bytes = byte_count,
    };
}

core::Status DeviceScratchBuffer::close() {
    auto status = allocation_.close();
    if (allocation_.empty()) {
        capacity_bytes_ = 0;
        used_bytes_ = 0;
    }
    return status;
}

} // namespace blackframe::xpu::cuda
