#include <Blackframe/Backends/GPU/CUDA/WavefrontQueues.hpp>
#include <Blackframe/XPU/CUDA/DeviceMemory.hpp>
#include <Blackframe/XPU/CUDA/WavefrontQueueCompaction.hpp>
#include <Blackframe/XPU/CUDA/WavefrontStageKernel.hpp>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace blackframe::engine {
namespace {

using xpu::cuda::WavefrontQueueCompactionResult;
using xpu::cuda::WavefrontQueueCompactionStatus;
using xpu::cuda::WavefrontStageOutcome;
using xpu::cuda::WavefrontStageRoute;
using xpu::cuda::WavefrontStageStatus;

inline constexpr auto CompactionRoute = WavefrontStageRoute::ray;

[[nodiscard]] bool cuda_succeeded(benchmark::State& state, const cudaError_t status,
                                  const char* const operation) {
    if (status == cudaSuccess) {
        return true;
    }
    const auto message = std::string{"CUDA wavefront compaction "} + operation +
                         " failed: " + cudaGetErrorName(status) + " (" +
                         cudaGetErrorString(status) + ").";
    state.SkipWithError(message.c_str());
    return false;
}

[[nodiscard]] bool status_succeeded(benchmark::State& state, const core::Status& status,
                                    const char* const operation) {
    if (status) {
        return true;
    }
    const auto message = std::string{"CUDA wavefront compaction "} + operation +
                         " failed: " + status.error().message;
    state.SkipWithError(message.c_str());
    return false;
}

[[nodiscard]] bool selected_lane(const std::size_t lane,
                                 const std::uint32_t occupancy_percent) noexcept {
    const auto selected_quarters = occupancy_percent / 25U;
    return lane % 4U < selected_quarters;
}

[[nodiscard]] double
path_slot_checksum(const std::vector<renderer::WavefrontPathSlot>& entries) noexcept {
    auto checksum = double{};
    for (auto index = std::size_t{}; index < entries.size(); ++index) {
        const auto value = static_cast<double>(entries[index].value) + 1.0;
        const auto weight = static_cast<double>(index % 31U + 1U);
        checksum += value * weight;
    }
    return checksum;
}

[[nodiscard]] bool valid_result(const WavefrontQueueCompactionResult& result,
                                const std::uint32_t input_count,
                                const std::uint32_t selected_count) noexcept {
    if (result.status != static_cast<std::uint32_t>(WavefrontQueueCompactionStatus::success) ||
        result.queue_kind != static_cast<std::uint32_t>(renderer::WavefrontQueueKind::ray) ||
        result.route != static_cast<std::uint32_t>(CompactionRoute) ||
        result.input_count != input_count || result.initial_size != 0U ||
        result.selected_count != selected_count || result.published_count != selected_count ||
        result.rejected_count != 0U) {
        return false;
    }
    for (const auto reserved : result.reserved) {
        if (reserved != 0U) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool validate_queue(benchmark::State& state, const CudaWavefrontQueues& queues,
                                  const std::vector<renderer::WavefrontPathSlot>& expected) {
    const auto snapshots = queues.download();
    if (!snapshots) {
        const auto message =
            std::string{"Cannot download the compacted CUDA queue: "} + snapshots.error().message;
        state.SkipWithError(message.c_str());
        return false;
    }

    for (auto queue_index = std::size_t{}; queue_index < snapshots->size(); ++queue_index) {
        const auto& snapshot = (*snapshots)[queue_index];
        const auto is_destination =
            queue_index == static_cast<std::size_t>(renderer::WavefrontQueueKind::ray);
        const auto expected_size = is_destination ? expected.size() : std::size_t{};
        if (snapshot.counters.size != expected_size || snapshot.counters.overflow_count != 0U ||
            snapshot.counters.rejected_count != 0U || snapshot.entries.size() != expected_size) {
            state.SkipWithError("CUDA compaction returned non-canonical queue counters.");
            return false;
        }
        if (is_destination && snapshot.entries != expected) {
            state.SkipWithError("CUDA compaction did not preserve stable path-slot order.");
            return false;
        }
    }
    return true;
}

void cuda_wavefront_queue_compaction(benchmark::State& state) {
    const auto input_count = static_cast<std::uint32_t>(state.range(0));
    const auto occupancy_percent = static_cast<std::uint32_t>(state.range(1));
    if ((input_count != 4'096U && input_count != 65'536U) ||
        (occupancy_percent != 25U && occupancy_percent != 50U && occupancy_percent != 75U)) {
        state.SkipWithError("Unsupported CUDA compaction benchmark arguments.");
        return;
    }

    auto host_outcomes = std::vector<WavefrontStageOutcome>(input_count);
    auto expected_entries = std::vector<renderer::WavefrontPathSlot>{};
    expected_entries.reserve(input_count);
    for (auto lane = std::uint32_t{}; lane < input_count; ++lane) {
        const auto selected = selected_lane(lane, occupancy_percent);
        host_outcomes[lane] = WavefrontStageOutcome{
            .status = static_cast<std::uint32_t>(WavefrontStageStatus::success),
            .route = static_cast<std::uint32_t>(selected ? CompactionRoute
                                                         : WavefrontStageRoute::terminated),
            .path_slot = lane,
            .detail = 0U,
        };
        if (selected) {
            expected_entries.push_back(renderer::WavefrontPathSlot{.value = lane});
        }
    }
    if (expected_entries.size() > std::numeric_limits<std::uint32_t>::max()) {
        state.SkipWithError("Selected CUDA compaction lanes exceed the device count domain.");
        return;
    }
    const auto selected_count = static_cast<std::uint32_t>(expected_entries.size());
    const auto expected_checksum = path_slot_checksum(expected_entries);

    auto queues = CudaWavefrontQueues::create(input_count);
    if (!queues) {
        state.SkipWithError(queues.error().message.c_str());
        return;
    }
    auto queue_storage = std::move(*queues);

    auto device_outcomes = xpu::cuda::DeviceBuffer<WavefrontStageOutcome>::allocate(input_count);
    if (!device_outcomes) {
        state.SkipWithError(device_outcomes.error().message.c_str());
        return;
    }
    auto outcome_storage = std::move(*device_outcomes);
    auto device_result = xpu::cuda::DeviceBuffer<WavefrontQueueCompactionResult>::allocate(1U);
    if (!device_result) {
        state.SkipWithError(device_result.error().message.c_str());
        return;
    }
    auto result_storage = std::move(*device_result);

    auto scratch_bytes = std::size_t{};
    if (!cuda_succeeded(
            state,
            static_cast<cudaError_t>(xpu::cuda::query_wavefront_queue_compaction_scratch_bytes(
                input_count, &scratch_bytes)),
            "scratch query")) {
        return;
    }
    if (scratch_bytes == 0U) {
        state.SkipWithError("Non-empty CUDA compaction unexpectedly requires no scratch.");
        return;
    }
    auto device_scratch = xpu::cuda::DeviceAllocation::allocate_bytes(scratch_bytes);
    if (!device_scratch) {
        state.SkipWithError(device_scratch.error().message.c_str());
        return;
    }
    auto scratch_storage = std::move(*device_scratch);
    const auto scratch_address = reinterpret_cast<std::uintptr_t>(scratch_storage.data());
    if (scratch_address % xpu::cuda::WavefrontQueueCompactionScratchAlignment != 0U) {
        state.SkipWithError("CUDA compaction scratch does not satisfy its fixed alignment.");
        return;
    }

    const auto outcome_bytes = host_outcomes.size() * sizeof(WavefrontStageOutcome);
    if (!cuda_succeeded(state,
                        cudaMemcpy(outcome_storage.data(), host_outcomes.data(), outcome_bytes,
                                   cudaMemcpyHostToDevice),
                        "outcome upload")) {
        return;
    }

    const auto launch = [&]() {
        return static_cast<cudaError_t>(xpu::cuda::launch_wavefront_queue_compaction(
            queue_storage.device_view(), outcome_storage.data(), input_count,
            static_cast<std::uint32_t>(CompactionRoute), scratch_storage.data(), scratch_bytes,
            result_storage.data()));
    };
    const auto download_result = [&]() {
        auto result = WavefrontQueueCompactionResult{};
        const auto status =
            cudaMemcpy(&result, result_storage.data(), sizeof(result), cudaMemcpyDeviceToHost);
        return std::pair{status, result};
    };

    if (!cuda_succeeded(state, launch(), "warmup launch") ||
        !cuda_succeeded(state, cudaDeviceSynchronize(), "warmup synchronization")) {
        return;
    }
    auto [warmup_copy_status, warmup_result] = download_result();
    if (!cuda_succeeded(state, warmup_copy_status, "warmup result download")) {
        return;
    }
    if (!valid_result(warmup_result, input_count, selected_count)) {
        state.SkipWithError("CUDA compaction warmup returned a non-canonical result.");
        return;
    }
    if (!validate_queue(state, queue_storage, expected_entries)) {
        return;
    }

    auto latest_result = warmup_result;
    for (auto _ : state) {
        static_cast<void>(_);
        state.PauseTiming();
        const auto reset_status =
            queue_storage.reset(CudaWavefrontQueueResetPolicy::require_no_overflow);
        state.ResumeTiming();
        if (!status_succeeded(state, reset_status, "queue reset")) {
            return;
        }

        if (!cuda_succeeded(state, launch(), "launch") ||
            !cuda_succeeded(state, cudaDeviceSynchronize(), "synchronization")) {
            return;
        }

        state.PauseTiming();
        auto [copy_status, result] = download_result();
        latest_result = result;
        state.ResumeTiming();
        if (!cuda_succeeded(state, copy_status, "result download")) {
            return;
        }
        if (!valid_result(latest_result, input_count, selected_count)) {
            state.SkipWithError("CUDA compaction returned a non-canonical measured result.");
            return;
        }
        benchmark::DoNotOptimize(latest_result.published_count);
        benchmark::ClobberMemory();
    }

    if (!validate_queue(state, queue_storage, expected_entries)) {
        return;
    }
    const auto checksum = expected_checksum;
    if (!(checksum > 0.0)) {
        state.SkipWithError("CUDA compaction produced a non-positive stable checksum.");
        return;
    }

    state.counters["input_lanes"] = static_cast<double>(input_count);
    state.counters["selected_lanes"] = static_cast<double>(latest_result.selected_count);
    state.counters["published_lanes"] = static_cast<double>(latest_result.published_count);
    state.counters["rejected_lanes"] = static_cast<double>(latest_result.rejected_count);
    state.counters["scratch_bytes"] = static_cast<double>(scratch_bytes);
    state.counters["checksum"] = checksum;
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(input_count));
    state.SetBytesProcessed(
        state.iterations() *
        static_cast<std::int64_t>(outcome_bytes + expected_entries.size() * sizeof(std::uint32_t)));

    const auto result_closed = status_succeeded(state, result_storage.close(), "result close");
    const auto outcomes_closed = status_succeeded(state, outcome_storage.close(), "outcome close");
    const auto scratch_closed = status_succeeded(state, scratch_storage.close(), "scratch close");
    const auto queues_closed = status_succeeded(state, queue_storage.close(), "queue close");
    if (!result_closed || !outcomes_closed || !scratch_closed || !queues_closed) {
        return;
    }
}

BENCHMARK(cuda_wavefront_queue_compaction)
    ->Args({4'096, 25})
    ->Args({4'096, 50})
    ->Args({4'096, 75})
    ->Args({65'536, 25})
    ->Args({65'536, 50})
    ->Args({65'536, 75})
    ->UseRealTime();

} // namespace
} // namespace blackframe::engine
