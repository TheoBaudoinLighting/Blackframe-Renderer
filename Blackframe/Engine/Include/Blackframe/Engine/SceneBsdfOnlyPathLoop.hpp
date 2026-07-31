#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Renderer/BsdfOnlyPathLoop.hpp>

namespace blackframe::engine {

// Traces the current float transport path against the exact FrameScene
// committed by the supplied acceleration backend. The scene must carry a
// complete spectral environment and material payload. Every bounce, including
// the primary ray, is queried through closest_hit(); this entry point has no
// triangle span and no alternate traversal path.
[[nodiscard]] core::Result<renderer::BsdfOnlyPathResult>
trace_scene_bsdf_only(const renderer::Ray& initial_ray, const renderer::PathState& initial_state,
                      const renderer::SampleStream& sample_stream, const AccelBackend& acceleration,
                      const renderer::PathDepthLimits& depth_limits,
                      const renderer::RussianRoulettePolicy& roulette_policy);

} // namespace blackframe::engine
