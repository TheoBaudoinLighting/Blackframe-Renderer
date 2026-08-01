#pragma once

#include <Blackframe/Core/Status.hpp>
#include <Blackframe/Engine/AccelBackend.hpp>
#include <Blackframe/Renderer/BsdfOnlyPathLoop.hpp>
#include <Blackframe/Renderer/LightSampler.hpp>
#include <Blackframe/Renderer/MisHeuristics.hpp>

namespace blackframe::engine {

// Traces a primary Lambertian path with one-sample next-event estimation and
// complementary weighting for non-delta emissive-surface hits. The immutable
// FrameScene owns the only emitter geometry and radiometry; the selected
// AccelBackend handles every primary, continuation, and shadow query.
[[nodiscard]] core::Result<renderer::BsdfOnlyPathResult>
trace_scene_mis(const renderer::Ray& initial_ray, const renderer::PathState& initial_state,
                const renderer::SampleStream& sample_stream, const AccelBackend& acceleration,
                const renderer::LightSampler& light_sampler, renderer::MisHeuristic heuristic,
                const renderer::PathDepthLimits& depth_limits,
                const renderer::RussianRoulettePolicy& roulette_policy);

} // namespace blackframe::engine
