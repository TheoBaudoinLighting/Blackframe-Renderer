#include <Blackframe/Renderer/CpuWavefrontScheduler.hpp>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace blackframe::renderer {
namespace {

[[nodiscard]] std::vector<WavefrontPathSlot>
scheduler_benchmark_slots(const std::size_t lane_count) {
    auto slots = std::vector<WavefrontPathSlot>(lane_count);
    for (auto lane = std::size_t{}; lane < lane_count; ++lane) {
        slots[lane].value = static_cast<std::uint32_t>(lane);
    }
    return slots;
}

void benchmark_scheduler(benchmark::State& state, const std::uint32_t worker_count) {
    const auto lane_count = static_cast<std::size_t>(state.range(0));
    const auto slots = scheduler_benchmark_slots(lane_count);
    auto output = std::vector<std::uint32_t>(lane_count);
    const auto scheduler = CpuWavefrontScheduler::create(worker_count);
    if (!scheduler) {
        state.SkipWithError(scheduler.error().message);
        return;
    }
    const auto kernel = CpuWavefrontStageKernel{[&](const CpuWavefrontLane lane) -> core::Status {
        output[lane.lane_index] =
            lane.path_slot.value * 747'796'405U + lane.worker_index + 2'891'336'453U;
        return {};
    }};

    for (auto _ : state) {
        const auto report =
            scheduler->execute_stage(WavefrontQueueKind::shade, lane_count, slots, kernel);
        if (!report) {
            state.SkipWithError(report.error().message);
            break;
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }
}

void schedule_cpu_one_thread(benchmark::State& state) {
    benchmark_scheduler(state, 1U);
}

void schedule_cpu_four_threads(benchmark::State& state) {
    benchmark_scheduler(state, 4U);
}

BENCHMARK(schedule_cpu_one_thread)->Arg(256)->Arg(4096)->Arg(65'536)->UseRealTime();
BENCHMARK(schedule_cpu_four_threads)->Arg(256)->Arg(4096)->Arg(65'536)->UseRealTime();

} // namespace
} // namespace blackframe::renderer
