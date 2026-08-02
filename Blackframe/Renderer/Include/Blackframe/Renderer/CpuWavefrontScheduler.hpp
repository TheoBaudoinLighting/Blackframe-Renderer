#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Renderer/WavefrontQueues.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <type_traits>

namespace blackframe::renderer {

inline constexpr std::uint32_t MaxCpuWavefrontWorkerCount = 1024U;

enum class CpuWavefrontSchedulerMode : std::uint8_t {
    single_thread = 0U,
    fixed_thread_count = 1U,
};

[[nodiscard]] constexpr bool
is_known_cpu_wavefront_scheduler_mode(const CpuWavefrontSchedulerMode mode) noexcept {
    switch (mode) {
    case CpuWavefrontSchedulerMode::single_thread:
    case CpuWavefrontSchedulerMode::fixed_thread_count:
        return true;
    }
    return false;
}

// A lane index addresses caller-owned stage output, while path_slot addresses the immutable path
// domain. Duplicate path slots remain distinct lanes and are never deduplicated implicitly.
struct CpuWavefrontLane final {
    WavefrontQueueKind stage;
    std::size_t lane_index;
    WavefrontPathSlot path_slot;
    std::uint32_t worker_index;

    [[nodiscard]] constexpr bool operator==(const CpuWavefrontLane&) const noexcept = default;
};

using CpuWavefrontStageKernel = std::function<core::Status(CpuWavefrontLane)>;

struct CpuWavefrontStageReport final {
    WavefrontQueueKind stage;
    CpuWavefrontSchedulerMode mode;
    std::size_t input_lanes;
    std::size_t path_slot_domain_size;
    std::uint32_t configured_workers;
    std::uint32_t workers_used;

    [[nodiscard]] constexpr bool
    operator==(const CpuWavefrontStageReport&) const noexcept = default;
};

// Dispatch is synchronous. One configured worker executes inline on the calling thread. More than
// one worker uses a persistent pool with deterministic contiguous lane partitions and waits for
// every worker before return. The callback may write only lane-local output or worker-local
// scratch; source queues, films, and shared output queues remain outside the workers and may be
// merged after this barrier. Concurrent callback invocation must be supported when more than one
// worker is used.
class CpuWavefrontScheduler final {
  public:
    CpuWavefrontScheduler(const CpuWavefrontScheduler&) = delete;
    CpuWavefrontScheduler(CpuWavefrontScheduler&& other) noexcept;
    CpuWavefrontScheduler& operator=(const CpuWavefrontScheduler&) = delete;
    CpuWavefrontScheduler& operator=(CpuWavefrontScheduler&&) = delete;
    ~CpuWavefrontScheduler();

    [[nodiscard]] static core::Result<CpuWavefrontScheduler> create(std::uint32_t worker_count);

    [[nodiscard]] std::uint32_t worker_count() const noexcept {
        return worker_count_;
    }

    [[nodiscard]] CpuWavefrontSchedulerMode mode() const noexcept {
        return worker_count_ == 1U ? CpuWavefrontSchedulerMode::single_thread
                                   : CpuWavefrontSchedulerMode::fixed_thread_count;
    }

    // Every queue entry is validated against path_slot_domain_size before any callback runs.
    // Unknown stages, empty callbacks, and unaddressable slots fail atomically. Empty input is an
    // explicit successful dispatch with zero workers used. Once dispatch begins, every partition
    // reaches the barrier even if a kernel reports or throws an error; the error at the lowest
    // input lane is returned deterministically and the caller must discard that dispatch's outputs.
    [[nodiscard]] core::Result<CpuWavefrontStageReport>
    execute_stage(WavefrontQueueKind stage, std::size_t path_slot_domain_size,
                  std::span<const WavefrontPathSlot> lanes,
                  const CpuWavefrontStageKernel& kernel) const;

  private:
    class State;

    explicit CpuWavefrontScheduler(std::uint32_t worker_count,
                                   std::unique_ptr<State> state) noexcept;

    std::uint32_t worker_count_;
    std::unique_ptr<State> state_;
};

static_assert(sizeof(CpuWavefrontSchedulerMode) == sizeof(std::uint8_t));
static_assert(std::is_standard_layout_v<CpuWavefrontLane>);
static_assert(std::is_trivially_copyable_v<CpuWavefrontLane>);
static_assert(std::is_standard_layout_v<CpuWavefrontStageReport>);
static_assert(std::is_trivially_copyable_v<CpuWavefrontStageReport>);
static_assert(!std::is_default_constructible_v<CpuWavefrontScheduler>);
static_assert(!std::is_copy_constructible_v<CpuWavefrontScheduler>);
static_assert(std::is_nothrow_move_constructible_v<CpuWavefrontScheduler>);

} // namespace blackframe::renderer
