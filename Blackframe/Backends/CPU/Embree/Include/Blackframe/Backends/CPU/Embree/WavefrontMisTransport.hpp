#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Renderer/BsdfOnlyPathLoop.hpp>
#include <Blackframe/Renderer/CpuWavefrontScheduler.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/MisHeuristics.hpp>
#include <Blackframe/Renderer/PathDepthLimits.hpp>
#include <Blackframe/Renderer/PathState.hpp>
#include <Blackframe/Renderer/RussianRoulette.hpp>
#include <Blackframe/Renderer/SampleStream.hpp>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace blackframe::engine {

inline constexpr std::uint32_t CurrentCpuWavefrontMisReportSchemaVersion = 1U;

// One batch entry binds an already generated primary ray and initial spectral path state to the
// immutable indexed sampler address used by every later stage. Batch order internally defines
// unique stable path slots and preserves output order.
struct CpuWavefrontMisPathInput final {
    renderer::Ray primary_ray;
    renderer::PathState initial_state;
    renderer::SampleStreamIndex sample;
};

struct CpuWavefrontMisStageLaneCounts final {
    std::uint64_t camera{};
    std::uint64_t ray{};
    std::uint64_t hit{};
    std::uint64_t miss{};
    std::uint64_t shade{};
    std::uint64_t shadow{};
    std::uint64_t continuation{};

    [[nodiscard]] constexpr bool
    operator==(const CpuWavefrontMisStageLaneCounts&) const noexcept = default;
};

struct CpuWavefrontMisReport final {
    std::uint32_t schema_version{};
    std::uint32_t configured_workers{};
    std::size_t path_count{};
    CpuWavefrontMisStageLaneCounts stage_lanes{};
    std::uint64_t closure_samples{};
    std::uint64_t light_samples{};
    std::uint64_t shadow_queries{};

    [[nodiscard]] constexpr bool operator==(const CpuWavefrontMisReport&) const noexcept = default;
};

struct CpuWavefrontMisBatch final {
    std::vector<renderer::BsdfOnlyPathResult> paths;
    CpuWavefrontMisReport report;
};

// Executes the current four-wavelength FrameScene transport through the seven bounded wavefront
// queues and the supplied explicit CPU scheduler. The production path requires the selected
// Embree backend; the analytic implementation remains an external scalar oracle and is rejected
// here rather than substituted. Every input is validated before the first stage. A stage failure
// invalidates the complete batch and never retries with another worker count, backend, closure,
// light sampler, heuristic, or transport loop.
[[nodiscard]] core::Result<CpuWavefrontMisBatch> trace_cpu_wavefront_mis(
    std::span<const CpuWavefrontMisPathInput> inputs,
    const renderer::CpuWavefrontScheduler& scheduler, const AccelBackend& acceleration,
    const renderer::LightSampler& light_sampler, renderer::MisHeuristic heuristic,
    const renderer::PathDepthLimits& depth_limits,
    const renderer::RussianRoulettePolicy& roulette_policy);

static_assert(std::is_standard_layout_v<CpuWavefrontMisPathInput>);
static_assert(std::is_trivially_copyable_v<CpuWavefrontMisPathInput>);
static_assert(std::is_standard_layout_v<CpuWavefrontMisStageLaneCounts>);
static_assert(std::is_trivially_copyable_v<CpuWavefrontMisStageLaneCounts>);
static_assert(std::is_standard_layout_v<CpuWavefrontMisReport>);
static_assert(std::is_trivially_copyable_v<CpuWavefrontMisReport>);

} // namespace blackframe::engine
