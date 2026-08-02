#include <Blackframe/Renderer/WavefrontQueues.hpp>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace blackframe::renderer {
namespace {

[[nodiscard]] std::vector<WavefrontPathSlot> benchmark_slots(const std::size_t lane_count) {
    auto slots = std::vector<WavefrontPathSlot>(lane_count);
    for (auto lane = std::size_t{}; lane < lane_count; ++lane) {
        const auto index = static_cast<std::uint32_t>(lane);
        slots[lane].value = index * 747'796'405U + 2'891'336'453U;
    }
    return slots;
}

[[nodiscard]] std::vector<WavefrontLaneState> benchmark_lane_states(const std::size_t lane_count) {
    auto lane_states = std::vector<WavefrontLaneState>(lane_count, WavefrontLaneState::active);
    for (auto lane = std::size_t{}; lane < lane_count; ++lane) {
        if (lane % 3U == 0U) {
            lane_states[lane] = WavefrontLaneState::terminated;
        }
    }
    return lane_states;
}

void compact_wavefront_queue(benchmark::State& state, const WavefrontCompactionOrder order) {
    const auto lane_count = static_cast<std::size_t>(state.range(0));
    const auto slots = benchmark_slots(lane_count);
    const auto lane_states = benchmark_lane_states(lane_count);
    auto created = RayQueue::create(lane_count);
    if (!created) {
        state.SkipWithError(created.error().message);
        return;
    }
    auto queue = std::move(*created);

    for (auto _ : state) {
        state.PauseTiming();
        queue.clear();
        const auto pushed = queue.push_batch(std::span<const WavefrontPathSlot>{slots});
        state.ResumeTiming();
        if (pushed != WavefrontQueuePushStatus::pushed) {
            state.SkipWithError("Cannot prepare the fixed-capacity compaction queue.");
            break;
        }
        auto report =
            queue.compact_terminated(std::span<const WavefrontLaneState>{lane_states}, order);
        if (!report) {
            state.SkipWithError(report.error().message);
            break;
        }
        benchmark::DoNotOptimize(*report);
        benchmark::ClobberMemory();
    }
}

void compact_stable_input(benchmark::State& state) {
    compact_wavefront_queue(state, WavefrontCompactionOrder::stable_input);
}

void compact_deterministic_path_slot(benchmark::State& state) {
    compact_wavefront_queue(state, WavefrontCompactionOrder::deterministic_path_slot);
}

BENCHMARK(compact_stable_input)->Arg(256)->Arg(4096)->Arg(65'536);
BENCHMARK(compact_deterministic_path_slot)->Arg(256)->Arg(4096)->Arg(65'536);

} // namespace
} // namespace blackframe::renderer
