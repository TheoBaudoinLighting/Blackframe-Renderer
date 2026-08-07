#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Renderer/BsdfOnlyPathLoop.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/MisHeuristics.hpp>
#include <Blackframe/Renderer/RayDifferential.hpp>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace blackframe::engine {

// Traces a primary bounded-closure path with one-sample next-event estimation and
// complementary weighting for non-delta emissive-surface hits. The immutable
// FrameScene owns the only emitter geometry and radiometry; the selected
// AccelBackend handles every primary, continuation, and shadow query.
[[nodiscard]] core::Result<renderer::BsdfOnlyPathResult>
trace_scene_mis(const renderer::Ray& initial_ray, const renderer::PathState& initial_state,
                const renderer::SampleStream& sample_stream, const AccelBackend& acceleration,
                const renderer::LightSampler& light_sampler, renderer::MisHeuristic heuristic,
                const renderer::PathDepthLimits& depth_limits,
                const renderer::RussianRoulettePolicy& roulette_policy);

enum class RayDifferentialLossReason : std::uint32_t {
    none = 0U,
    non_delta_scattering = 1U,
    specular_discontinuity = 2U,
};

struct SceneMisRayDifferentialResult final {
    renderer::BsdfOnlyPathResult path;
    std::optional<renderer::RayDifferential> terminal_differential;
    RayDifferentialLossReason loss_reason{RayDifferentialLossReason::none};
    std::uint32_t propagated_specular_bounces{};
};

// Scalar-reference-only entry point. It consumes exactly the same SampleStream dimensions and
// traces only the central Ray through AccelBackend. Auxiliary rays are propagated analytically
// across ideal reflection and transmission; continuous scattering and critical/TIR footprints
// report an explicit loss reason.
[[nodiscard]] core::Result<SceneMisRayDifferentialResult> trace_scene_mis_with_ray_differentials(
    const renderer::RayDifferential& initial_ray, const renderer::PathState& initial_state,
    const renderer::SampleStream& sample_stream, const AccelBackend& acceleration,
    const renderer::LightSampler& light_sampler, renderer::MisHeuristic heuristic,
    const renderer::PathDepthLimits& depth_limits,
    const renderer::RussianRoulettePolicy& roulette_policy);

static_assert(std::is_trivially_copyable_v<RayDifferentialLossReason>);

} // namespace blackframe::engine
