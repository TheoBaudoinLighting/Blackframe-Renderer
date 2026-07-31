#include <Blackframe/Renderer/LightSampler.hpp>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace blackframe::renderer {
namespace {

[[nodiscard]] std::vector<LightTreeInput> benchmark_lights(const std::size_t light_count) {
    auto lights = std::vector<LightTreeInput>{};
    lights.reserve(light_count);
    for (auto index = std::size_t{0}; index < light_count; ++index) {
        const auto x = static_cast<TransportScalar>(index % 256U);
        const auto y = static_cast<TransportScalar>((index / 256U) % 256U);
        const auto z = static_cast<TransportScalar>(index / (256U * 256U));
        const auto bounds =
            Bounds3::from_minimum_maximum(Point3{.x = x - 0.1F, .y = y - 0.1F, .z = z - 0.1F},
                                          Point3{.x = x + 0.1F, .y = y + 0.1F, .z = z + 0.1F});
        if (!bounds) {
            return {};
        }
        auto power = TransportSpectrum{};
        power[index % power.values.size()] = static_cast<TransportScalar>(index % 13U + 1U);
        lights.push_back(LightTreeInput{.bounds = *bounds, .spectral_power = power});
    }
    return lights;
}

[[nodiscard]] TransportScalar benchmark_canonical(const std::uint32_t index) noexcept {
    const auto bits = index * 747'796'405U + 2'891'336'453U;
    return static_cast<TransportScalar>(bits >> 8U) * (1.0F / 16'777'216.0F);
}

void build_light_tree(benchmark::State& state) {
    const auto light_count = static_cast<std::size_t>(state.range(0));
    const auto lights = benchmark_lights(light_count);
    if (lights.size() != light_count) {
        state.SkipWithError("Cannot prepare finite light-tree bounds.");
        return;
    }
    for (auto _ : state) {
        auto tree = LightSampler::create_spatial_tree(std::span<const LightTreeInput>{lights});
        if (!tree) {
            state.SkipWithError(tree.error().message);
            break;
        }
        benchmark::DoNotOptimize(*tree);
    }
}

void sample_light_tree(benchmark::State& state) {
    const auto light_count = static_cast<std::size_t>(state.range(0));
    const auto lights = benchmark_lights(light_count);
    const auto tree = LightSampler::create_spatial_tree(std::span<const LightTreeInput>{lights});
    const auto context =
        LightSampleContext::create(Point3{.x = 12.5F, .y = 17.25F, .z = 3.0F}, 0.0F);
    if (!tree || !context) {
        state.SkipWithError(!tree ? tree.error().message : context.error().message);
        return;
    }
    auto sample_index = std::uint32_t{0};
    for (auto _ : state) {
        const auto selection = tree->sample(*context, benchmark_canonical(sample_index++));
        if (!selection) {
            state.SkipWithError(selection.error().message);
            break;
        }
        auto selected = *selection;
        benchmark::DoNotOptimize(selected);
    }
}

void query_light_tree_probability(benchmark::State& state) {
    const auto light_count = static_cast<std::size_t>(state.range(0));
    const auto lights = benchmark_lights(light_count);
    const auto tree = LightSampler::create_spatial_tree(std::span<const LightTreeInput>{lights});
    const auto context =
        LightSampleContext::create(Point3{.x = 12.5F, .y = 17.25F, .z = 3.0F}, 0.0F);
    if (!tree || !context) {
        state.SkipWithError(!tree ? tree.error().message : context.error().message);
        return;
    }
    auto light_index = std::uint32_t{0};
    for (auto _ : state) {
        auto probability = tree->probability(*context, light_index);
        if (!probability) {
            state.SkipWithError(probability.error().message);
            break;
        }
        benchmark::DoNotOptimize(probability);
        light_index = (light_index + 1U) % static_cast<std::uint32_t>(light_count);
    }
}

void sample_power_sampler(benchmark::State& state) {
    const auto light_count = static_cast<std::size_t>(state.range(0));
    const auto lights = benchmark_lights(light_count);
    auto powers = std::vector<TransportSpectrum>{};
    powers.reserve(lights.size());
    for (const auto& light : lights) {
        powers.push_back(light.spectral_power);
    }
    const auto sampler =
        LightSampler::create_power_weighted(std::span<const TransportSpectrum>{powers});
    if (!sampler) {
        state.SkipWithError(sampler.error().message);
        return;
    }
    auto sample_index = std::uint32_t{0};
    for (auto _ : state) {
        auto selection = sampler->sample(benchmark_canonical(sample_index++));
        if (!selection) {
            state.SkipWithError(selection.error().message);
            break;
        }
        benchmark::DoNotOptimize(selection);
    }
}

BENCHMARK(build_light_tree)->Arg(16)->Arg(256)->Arg(4096)->Arg(65'536);
BENCHMARK(sample_light_tree)->Arg(16)->Arg(256)->Arg(4096)->Arg(65'536);
BENCHMARK(query_light_tree_probability)->Arg(16)->Arg(256)->Arg(4096)->Arg(65'536);
BENCHMARK(sample_power_sampler)->Arg(16)->Arg(256)->Arg(4096)->Arg(65'536);

} // namespace
} // namespace blackframe::renderer
