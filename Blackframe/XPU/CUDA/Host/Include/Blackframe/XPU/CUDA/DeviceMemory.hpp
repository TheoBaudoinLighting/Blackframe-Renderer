#pragma once

#include <Blackframe/Core/Status.hpp>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <type_traits>
#include <utility>

namespace blackframe::xpu::cuda {

class DeviceScratchBuffer;

struct DeviceMemoryBudget final {
    std::size_t maximum_bytes{static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())};
};

[[nodiscard]] core::StatusCode cuda_memory_status_code(std::int32_t cuda_status) noexcept;

class DeviceAllocation final {
  public:
    DeviceAllocation() noexcept = default;
    ~DeviceAllocation() noexcept;

    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    DeviceAllocation(DeviceAllocation&& other) noexcept;
    DeviceAllocation& operator=(DeviceAllocation&& other) = delete;

    [[nodiscard]] static core::Result<DeviceAllocation>
    allocate_bytes(std::size_t byte_count, DeviceMemoryBudget budget = {});

    // Explicit close reports CUDA teardown errors. The destructor remains the
    // no-throw ownership backstop for callers that do not need that diagnostic.
    [[nodiscard]] core::Status close();

    [[nodiscard]] void* data() noexcept {
        return data_;
    }
    [[nodiscard]] const void* data() const noexcept {
        return data_;
    }
    [[nodiscard]] std::size_t size_bytes() const noexcept {
        return size_bytes_;
    }
    [[nodiscard]] std::int32_t device_ordinal() const noexcept {
        return device_ordinal_;
    }
    [[nodiscard]] bool empty() const noexcept {
        return data_ == nullptr;
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return !empty();
    }

    void swap(DeviceAllocation& other) noexcept;

  private:
    friend class DeviceScratchBuffer;

    [[nodiscard]] static core::Result<DeviceAllocation>
    allocate_bytes_on_device(std::size_t byte_count, DeviceMemoryBudget budget,
                             std::int32_t device_ordinal);
    DeviceAllocation(void* data, std::size_t size_bytes, std::int32_t device_ordinal) noexcept;
    void close_without_diagnostic() noexcept;

    void* data_{nullptr};
    std::size_t size_bytes_{0};
    std::int32_t device_ordinal_{-1};
};

template <typename Element>
concept DeviceBufferElement = std::is_trivially_copyable_v<Element> && !std::is_const_v<Element> &&
                              !std::is_volatile_v<Element>;

template <DeviceBufferElement Element> class DeviceBuffer final {
  public:
    DeviceBuffer() noexcept = default;

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : allocation_(std::move(other.allocation_)), count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) = delete;

    [[nodiscard]] static core::Result<DeviceBuffer> allocate(const std::size_t count,
                                                             const DeviceMemoryBudget budget = {}) {
        if (count > static_cast<std::size_t>(-1) / sizeof(Element)) {
            return std::unexpected(core::Error{
                .code = core::StatusCode::resource_exhausted,
                .message = "CUDA device-buffer size multiplication overflowed.",
            });
        }

        auto allocation = DeviceAllocation::allocate_bytes(count * sizeof(Element), budget);
        if (!allocation) {
            return std::unexpected(std::move(allocation.error()));
        }

        return DeviceBuffer{std::move(*allocation), count};
    }

    [[nodiscard]] core::Status close() {
        auto status = allocation_.close();
        if (allocation_.empty()) {
            count_ = 0;
        }
        return status;
    }

    [[nodiscard]] Element* data() noexcept {
        return static_cast<Element*>(allocation_.data());
    }
    [[nodiscard]] const Element* data() const noexcept {
        return static_cast<const Element*>(allocation_.data());
    }
    [[nodiscard]] std::size_t size() const noexcept {
        return count_;
    }
    [[nodiscard]] std::size_t size_bytes() const noexcept {
        return allocation_.size_bytes();
    }
    [[nodiscard]] std::int32_t device_ordinal() const noexcept {
        return allocation_.device_ordinal();
    }
    [[nodiscard]] bool empty() const noexcept {
        return count_ == 0;
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return !allocation_.empty();
    }

  private:
    DeviceBuffer(DeviceAllocation allocation, const std::size_t count) noexcept
        : allocation_(std::move(allocation)), count_(count) {}

    DeviceAllocation allocation_{};
    std::size_t count_{0};
};

struct DeviceScratchSlice final {
    void* data{nullptr};
    std::size_t offset_bytes{0};
    std::size_t size_bytes{0};
};

class DeviceScratchBuffer final {
  public:
    DeviceScratchBuffer() noexcept = default;

    DeviceScratchBuffer(const DeviceScratchBuffer&) = delete;
    DeviceScratchBuffer& operator=(const DeviceScratchBuffer&) = delete;

    DeviceScratchBuffer(DeviceScratchBuffer&& other) noexcept;
    DeviceScratchBuffer& operator=(DeviceScratchBuffer&& other) = delete;

    [[nodiscard]] static core::Result<DeviceScratchBuffer> create(std::size_t capacity_bytes,
                                                                  DeviceMemoryBudget budget);

    // Growth is explicit and preserves the existing allocation if CUDA rejects
    // the request. The caller must complete all GPU work using outstanding
    // slices before reset, growth, move, or close invalidates them.
    [[nodiscard]] core::Status reserve(std::size_t minimum_capacity_bytes);
    [[nodiscard]] core::Result<DeviceScratchSlice> allocate(std::size_t byte_count,
                                                            std::size_t alignment);

    void reset() noexcept {
        used_bytes_ = 0;
    }
    [[nodiscard]] core::Status close();

    [[nodiscard]] void* data() noexcept {
        return allocation_.data();
    }
    [[nodiscard]] const void* data() const noexcept {
        return allocation_.data();
    }
    [[nodiscard]] std::int32_t device_ordinal() const noexcept {
        return allocation_.device_ordinal();
    }
    [[nodiscard]] std::size_t capacity_bytes() const noexcept {
        return capacity_bytes_;
    }
    [[nodiscard]] std::size_t maximum_capacity_bytes() const noexcept {
        return budget_.maximum_bytes;
    }
    [[nodiscard]] std::size_t used_bytes() const noexcept {
        return used_bytes_;
    }
    [[nodiscard]] std::size_t remaining_bytes() const noexcept {
        return capacity_bytes_ - used_bytes_;
    }
    [[nodiscard]] bool empty() const noexcept {
        return capacity_bytes_ == 0;
    }

  private:
    DeviceScratchBuffer(DeviceAllocation allocation, std::size_t capacity_bytes,
                        DeviceMemoryBudget budget) noexcept;

    DeviceAllocation allocation_{};
    std::size_t capacity_bytes_{0};
    std::size_t used_bytes_{0};
    DeviceMemoryBudget budget_{};
};

} // namespace blackframe::xpu::cuda
