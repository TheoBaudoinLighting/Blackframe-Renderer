#include <Blackframe/Renderer/CpuWavefrontScheduler.hpp>
#include <algorithm>
#include <condition_variable>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace blackframe::renderer {
namespace {

enum class WorkerFailureKind : std::uint8_t {
    none,
    returned_error,
    allocation_exception,
    standard_exception,
    unknown_exception,
};

struct WorkerOutcome final {
    WorkerFailureKind failure{WorkerFailureKind::none};
    std::size_t first_failed_lane{};
    std::optional<core::Error> returned_error;
};

static_assert(std::is_nothrow_move_constructible_v<core::Error>);

[[nodiscard]] core::Error scheduler_error(const core::StatusCode code, std::string message) {
    return core::Error{
        .code = code,
        .message = std::move(message),
    };
}

[[nodiscard]] bool representable_path_slot_domain(const std::size_t domain_size) noexcept {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t)) {
        constexpr auto maximum_domain =
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1U;
        return domain_size <= maximum_domain;
    }
    return true;
}

[[nodiscard]] std::pair<std::size_t, std::size_t>
worker_partition(const std::size_t lane_count, const std::uint32_t worker_index,
                 const std::uint32_t worker_count) noexcept {
    const auto base_size = lane_count / worker_count;
    const auto extended_partition_count = lane_count % worker_count;
    const auto worker = static_cast<std::size_t>(worker_index);
    const auto begin = worker * base_size + std::min(worker, extended_partition_count);
    const auto count = base_size + (worker < extended_partition_count ? 1U : 0U);
    return {begin, begin + count};
}

void reset_outcome(WorkerOutcome& outcome) noexcept {
    outcome.failure = WorkerFailureKind::none;
    outcome.first_failed_lane = 0U;
    outcome.returned_error.reset();
}

void record_returned_error(WorkerOutcome& outcome, const std::size_t lane_index,
                           core::Error error) noexcept {
    if (outcome.failure != WorkerFailureKind::none) {
        return;
    }
    outcome.failure = WorkerFailureKind::returned_error;
    outcome.first_failed_lane = lane_index;
    outcome.returned_error.emplace(std::move(error));
}

void record_exception(WorkerOutcome& outcome, const std::size_t lane_index,
                      const WorkerFailureKind failure) noexcept {
    if (outcome.failure == WorkerFailureKind::none) {
        outcome.failure = failure;
        outcome.first_failed_lane = lane_index;
    }
}

void execute_partition(const WavefrontQueueKind stage,
                       const std::span<const WavefrontPathSlot> lanes,
                       const std::uint32_t worker_index, const std::uint32_t worker_count,
                       const CpuWavefrontStageKernel& kernel, WorkerOutcome& outcome) noexcept {
    const auto [begin, end] = worker_partition(lanes.size(), worker_index, worker_count);
    for (auto lane_index = begin; lane_index < end; ++lane_index) {
        try {
            auto status = kernel(CpuWavefrontLane{
                .stage = stage,
                .lane_index = lane_index,
                .path_slot = lanes[lane_index],
                .worker_index = worker_index,
            });
            if (!status.has_value()) {
                record_returned_error(outcome, lane_index, std::move(status.error()));
            }
        } catch (const std::bad_alloc&) {
            record_exception(outcome, lane_index, WorkerFailureKind::allocation_exception);
        } catch (const std::exception&) {
            record_exception(outcome, lane_index, WorkerFailureKind::standard_exception);
        } catch (...) {
            record_exception(outcome, lane_index, WorkerFailureKind::unknown_exception);
        }
    }
}

[[nodiscard]] core::Result<CpuWavefrontStageReport>
report_or_error(const WavefrontQueueKind stage, const CpuWavefrontSchedulerMode mode,
                const std::size_t lane_count, const std::size_t path_slot_domain_size,
                const std::uint32_t configured_workers, const std::uint32_t workers_used,
                std::span<WorkerOutcome> outcomes) {
    for (auto& outcome : outcomes) {
        switch (outcome.failure) {
        case WorkerFailureKind::none:
            break;
        case WorkerFailureKind::returned_error:
            return std::unexpected(std::move(*outcome.returned_error));
        case WorkerFailureKind::allocation_exception:
            return std::unexpected(
                scheduler_error(core::StatusCode::resource_exhausted,
                                "The " + std::string{wavefront_queue_kind_name(stage)} +
                                    " CPU stage kernel exhausted memory at input lane " +
                                    std::to_string(outcome.first_failed_lane) + "."));
        case WorkerFailureKind::standard_exception:
            return std::unexpected(
                scheduler_error(core::StatusCode::internal_error,
                                "The " + std::string{wavefront_queue_kind_name(stage)} +
                                    " CPU stage kernel threw a standard exception at input lane " +
                                    std::to_string(outcome.first_failed_lane) + "."));
        case WorkerFailureKind::unknown_exception:
            return std::unexpected(
                scheduler_error(core::StatusCode::internal_error,
                                "The " + std::string{wavefront_queue_kind_name(stage)} +
                                    " CPU stage kernel threw an unknown exception at input lane " +
                                    std::to_string(outcome.first_failed_lane) + "."));
        }
    }

    return CpuWavefrontStageReport{
        .stage = stage,
        .mode = mode,
        .input_lanes = lane_count,
        .path_slot_domain_size = path_slot_domain_size,
        .configured_workers = configured_workers,
        .workers_used = workers_used,
    };
}

} // namespace

class CpuWavefrontScheduler::State final {
  public:
    explicit State(const std::uint32_t worker_count)
        : worker_count_{worker_count}, outcomes_(worker_count) {
        if (worker_count_ > 1U) {
            workers_.reserve(worker_count_);
        }
    }

    State(const State&) = delete;
    State(State&&) = delete;
    State& operator=(const State&) = delete;
    State& operator=(State&&) = delete;

    ~State() {
        shutdown();
    }

    void launch_workers() {
        if (worker_count_ == 1U) {
            return;
        }
        for (auto worker_index = std::uint32_t{}; worker_index < worker_count_; ++worker_index) {
            workers_.emplace_back([this, worker_index] { worker_loop(worker_index); });
        }
    }

    [[nodiscard]] core::Result<CpuWavefrontStageReport>
    dispatch(const WavefrontQueueKind stage, const CpuWavefrontSchedulerMode mode,
             const std::size_t path_slot_domain_size,
             const std::span<const WavefrontPathSlot> lanes,
             const CpuWavefrontStageKernel& kernel) {
        auto dispatch_lock = std::unique_lock{dispatch_mutex_, std::try_to_lock};
        if (!dispatch_lock.owns_lock()) {
            return std::unexpected(scheduler_error(
                core::StatusCode::unavailable,
                "A CPU wavefront scheduler cannot execute concurrent or reentrant stages."));
        }

        if (lanes.empty()) {
            return CpuWavefrontStageReport{
                .stage = stage,
                .mode = mode,
                .input_lanes = 0U,
                .path_slot_domain_size = path_slot_domain_size,
                .configured_workers = worker_count_,
                .workers_used = 0U,
            };
        }

        const auto workers_used =
            static_cast<std::uint32_t>(std::min<std::size_t>(worker_count_, lanes.size()));
        if (worker_count_ == 1U) {
            reset_outcome(outcomes_[0U]);
            execute_partition(stage, lanes, 0U, 1U, kernel, outcomes_[0U]);
            return report_or_error(stage, mode, lanes.size(), path_slot_domain_size, worker_count_,
                                   1U, std::span<WorkerOutcome>{outcomes_}.first(1U));
        }

        auto work_lock = std::unique_lock{work_mutex_};
        if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
            return std::unexpected(scheduler_error(
                core::StatusCode::resource_exhausted,
                "The CPU wavefront scheduler exhausted its dispatch generation counter."));
        }
        for (auto& outcome : outcomes_) {
            reset_outcome(outcome);
        }
        stage_ = stage;
        lanes_ = lanes;
        kernel_ = &kernel;
        workers_used_ = workers_used;
        completed_workers_ = 0U;
        ++generation_;
        work_lock.unlock();
        work_available_.notify_all();

        work_lock.lock();
        work_complete_.wait(work_lock, [this] { return completed_workers_ == worker_count_; });
        kernel_ = nullptr;
        lanes_ = {};
        work_lock.unlock();
        return report_or_error(stage, mode, lanes.size(), path_slot_domain_size, worker_count_,
                               workers_used,
                               std::span<WorkerOutcome>{outcomes_}.first(workers_used));
    }

  private:
    void worker_loop(const std::uint32_t worker_index) {
        auto observed_generation = std::uint64_t{};
        auto work_lock = std::unique_lock{work_mutex_};
        for (;;) {
            work_available_.wait(work_lock, [this, observed_generation] {
                return stopping_ || generation_ != observed_generation;
            });
            if (stopping_) {
                return;
            }
            observed_generation = generation_;
            const auto stage = stage_;
            const auto lanes = lanes_;
            const auto workers_used = workers_used_;
            const auto* const kernel = kernel_;
            work_lock.unlock();
            if (worker_index < workers_used) {
                execute_partition(stage, lanes, worker_index, workers_used, *kernel,
                                  outcomes_[worker_index]);
            }
            work_lock.lock();
            ++completed_workers_;
            if (completed_workers_ == worker_count_) {
                work_complete_.notify_one();
            }
        }
    }

    void shutdown() noexcept {
        if (workers_.empty()) {
            return;
        }
        {
            const auto lock = std::lock_guard{work_mutex_};
            stopping_ = true;
        }
        work_available_.notify_all();
        workers_.clear();
    }

    std::uint32_t worker_count_;
    std::vector<WorkerOutcome> outcomes_;
    std::vector<std::jthread> workers_;
    std::mutex dispatch_mutex_;
    std::mutex work_mutex_;
    std::condition_variable work_available_;
    std::condition_variable work_complete_;
    bool stopping_{};
    std::uint64_t generation_{};
    std::uint32_t completed_workers_{};
    std::uint32_t workers_used_{};
    WavefrontQueueKind stage_{};
    std::span<const WavefrontPathSlot> lanes_;
    const CpuWavefrontStageKernel* kernel_{};
};

CpuWavefrontScheduler::CpuWavefrontScheduler(const std::uint32_t worker_count,
                                             std::unique_ptr<State> state) noexcept
    : worker_count_{worker_count}, state_{std::move(state)} {}

CpuWavefrontScheduler::CpuWavefrontScheduler(CpuWavefrontScheduler&& other) noexcept
    : worker_count_{std::exchange(other.worker_count_, 0U)}, state_{std::move(other.state_)} {}

CpuWavefrontScheduler::~CpuWavefrontScheduler() = default;

core::Result<CpuWavefrontScheduler>
CpuWavefrontScheduler::create(const std::uint32_t worker_count) {
    if (worker_count == 0U) {
        return std::unexpected(scheduler_error(
            core::StatusCode::invalid_argument,
            "A CPU wavefront scheduler requires an explicit non-zero worker count."));
    }
    if (worker_count > MaxCpuWavefrontWorkerCount) {
        return std::unexpected(scheduler_error(
            core::StatusCode::invalid_argument,
            "The requested CPU wavefront worker count exceeds the supported bound of " +
                std::to_string(MaxCpuWavefrontWorkerCount) + "."));
    }

    try {
        auto state = std::make_unique<State>(worker_count);
        state->launch_workers();
        return CpuWavefrontScheduler{worker_count, std::move(state)};
    } catch (const std::bad_alloc&) {
        return std::unexpected(scheduler_error(
            core::StatusCode::resource_exhausted,
            "CPU wavefront scheduler creation exhausted host memory; no fallback was selected."));
    } catch (const std::length_error&) {
        return std::unexpected(
            scheduler_error(core::StatusCode::resource_exhausted,
                            "The CPU wavefront worker pool is not representable on this host."));
    } catch (const std::system_error&) {
        return std::unexpected(scheduler_error(
            core::StatusCode::platform_error,
            "The host could not create every requested CPU wavefront worker; no fallback was "
            "selected."));
    } catch (const std::exception&) {
        return std::unexpected(scheduler_error(
            core::StatusCode::internal_error,
            "CPU wavefront scheduler creation failed unexpectedly; no fallback was selected."));
    } catch (...) {
        return std::unexpected(scheduler_error(
            core::StatusCode::internal_error,
            "CPU wavefront scheduler creation failed with an unknown error; no fallback was "
            "selected."));
    }
}

core::Result<CpuWavefrontStageReport> CpuWavefrontScheduler::execute_stage(
    const WavefrontQueueKind stage, const std::size_t path_slot_domain_size,
    const std::span<const WavefrontPathSlot> lanes, const CpuWavefrontStageKernel& kernel) const {
    if (!state_) {
        return std::unexpected(
            scheduler_error(core::StatusCode::unavailable,
                            "A moved-from CPU wavefront scheduler cannot execute a stage."));
    }
    if (!is_known_wavefront_queue_kind(stage)) {
        return std::unexpected(
            scheduler_error(core::StatusCode::invalid_argument,
                            "A CPU wavefront scheduler requires an explicitly known stage queue."));
    }
    if (!representable_path_slot_domain(path_slot_domain_size)) {
        return std::unexpected(scheduler_error(
            core::StatusCode::invalid_argument,
            "The CPU wavefront path-slot domain exceeds the 32-bit queue address space."));
    }
    if (!kernel) {
        return std::unexpected(
            scheduler_error(core::StatusCode::invalid_argument,
                            "A CPU wavefront stage requires an explicit non-empty kernel."));
    }
    for (auto lane_index = std::size_t{}; lane_index < lanes.size(); ++lane_index) {
        if (static_cast<std::size_t>(lanes[lane_index].value) >= path_slot_domain_size) {
            return std::unexpected(scheduler_error(
                core::StatusCode::invalid_argument,
                "The " + std::string{wavefront_queue_kind_name(stage)} +
                    " CPU stage path slot at input lane " + std::to_string(lane_index) +
                    " is outside the declared path domain."));
        }
    }
    return state_->dispatch(stage, mode(), path_slot_domain_size, lanes, kernel);
}

} // namespace blackframe::renderer
