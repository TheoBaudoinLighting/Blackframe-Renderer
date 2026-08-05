#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/WavefrontQueues.hpp>
#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <Blackframe/XPU/CUDA/WavefrontQueueKernel.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace blackframe::engine {

struct CudaWavefrontQueueCreateOptions final {
    xpu::cuda::DeviceMemoryBudget device_memory_budget{};
};

struct CudaWavefrontQueueCounters final {
    renderer::WavefrontQueueKind kind{};
    std::uint32_t capacity{};
    std::uint32_t size{};
    std::uint32_t overflow_count{};
    std::uint32_t rejected_count{};

    [[nodiscard]] constexpr bool
    operator==(const CudaWavefrontQueueCounters&) const noexcept = default;
};

struct CudaWavefrontQueueSnapshot final {
    CudaWavefrontQueueCounters counters{};
    std::vector<renderer::WavefrontPathSlot> entries;

    [[nodiscard]] bool operator==(const CudaWavefrontQueueSnapshot&) const = default;
};

using CudaWavefrontQueueSnapshots =
    std::array<CudaWavefrontQueueSnapshot, xpu::cuda::CudaWavefrontQueueCount>;

enum class CudaWavefrontQueueResetPolicy : std::uint8_t {
    require_no_overflow = 0U,
    acknowledge_overflow = 1U,
};

// Owns seven independent, queue-major PathSlot columns and their device counters. Producers may
// reserve slots concurrently, but consumers must wait for the producing kernel boundary. Queue
// capacity is fixed for the allocation lifetime and exhaustion is reported by device counters and
// per-request outcomes; entries beyond the active prefix are never published.
class CudaWavefrontQueues final {
  public:
    CudaWavefrontQueues(const CudaWavefrontQueues&) = delete;
    CudaWavefrontQueues& operator=(const CudaWavefrontQueues&) = delete;
    CudaWavefrontQueues(CudaWavefrontQueues&& other) noexcept;
    CudaWavefrontQueues& operator=(CudaWavefrontQueues&& other) = delete;
    ~CudaWavefrontQueues() noexcept = default;

    [[nodiscard]] static core::Result<CudaWavefrontQueues>
    create(std::size_t capacity, CudaWavefrontQueueCreateOptions options = {});

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return capacity_;
    }
    [[nodiscard]] std::int32_t device_ordinal() const noexcept {
        return headers_.device_ordinal();
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        const auto expected_slot_count =
            static_cast<std::size_t>(capacity_) * xpu::cuda::CudaWavefrontQueueCount;
        return static_cast<bool>(headers_) &&
               headers_.size() == xpu::cuda::CudaWavefrontQueueCount &&
               ((capacity_ == 0U && path_slots_.empty()) ||
                (capacity_ != 0U && static_cast<bool>(path_slots_) &&
                 path_slots_.size() == expected_slot_count &&
                 path_slots_.device_ordinal() == headers_.device_ordinal()));
    }

    [[nodiscard]] xpu::cuda::WavefrontQueueDeviceSoa device_view() noexcept;
    [[nodiscard]] core::Result<CudaWavefrontQueueSnapshots> download() const;
    // Reset always validates device counters first. The conservative policy refuses to erase an
    // overflow diagnostic; callers that already handled it must acknowledge that fact explicitly.
    [[nodiscard]] core::Status reset(CudaWavefrontQueueResetPolicy policy);
    [[nodiscard]] core::Status close();

  private:
    CudaWavefrontQueues(xpu::cuda::DeviceBuffer<xpu::shared::QueueHeader> headers,
                        xpu::cuda::DeviceBuffer<xpu::shared::PathSlot> path_slots,
                        std::uint32_t capacity) noexcept;

    xpu::cuda::DeviceBuffer<xpu::shared::QueueHeader> headers_{};
    xpu::cuda::DeviceBuffer<xpu::shared::PathSlot> path_slots_{};
    std::uint32_t capacity_{};
};

static_assert(std::is_standard_layout_v<CudaWavefrontQueueCreateOptions>);
static_assert(sizeof(CudaWavefrontQueueResetPolicy) == sizeof(std::uint8_t));
static_assert(!std::is_copy_constructible_v<CudaWavefrontQueues>);
static_assert(!std::is_copy_assignable_v<CudaWavefrontQueues>);
static_assert(std::is_nothrow_move_constructible_v<CudaWavefrontQueues>);
static_assert(!std::is_move_assignable_v<CudaWavefrontQueues>);
static_assert(std::is_nothrow_destructible_v<CudaWavefrontQueues>);

} // namespace blackframe::engine
